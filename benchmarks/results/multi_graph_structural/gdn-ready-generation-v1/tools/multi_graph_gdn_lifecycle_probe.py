# SPDX-License-Identifier: Apache-2.0
"""Close the existing real-layer Q4 probe over dynamic state generations.

This is a fixture adapter, not a serving scheduler. Each timed trajectory pays
descriptor packing/upload, input staging, every complete real layer and join,
then one terminal accepted-prefix materialization. No state is reset between
steps. One-time dense-to-compact import is reported and charged separately.
"""

import hashlib
import statistics
import time
from pathlib import Path


def terminal_counts(schedule, capacity):
    """Return acceptance for each owner's last generated window, including pauses."""
    generations = [0] * capacity
    for entry in schedule:
        if not 0 <= entry["live"] <= capacity:
            raise ValueError("live count exceeds the declared owner capacity")
        for owner in range(entry["live"]):
            generations[owner] += 1
    return [1 + (owner + generation) % 4 for owner, generation in enumerate(generations)]


def lifecycle_samples(
    torch,
    dist,
    cpu_group,
    args,
    fixture,
    compact,
    backend,
    modes,
    schedule,
    prepare,
    restore,
    initial,
    graphs,
    graph_outputs,
    hidden,
    residual,
    active,
):
    capacity = len(compact.request_candidate_slots)
    with fixture.phases.context_for_name("full"):
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata[fixture.phases.gdn.prefix]
        causal = metadata.spec_decode_metadata.spec_causal_conv1d
        tensors = (
            metadata.spec_state_indices_tensor,
            causal.query_start_loc,
            causal.cache_indices,
            causal.num_accepted_tokens,
            metadata.spec_decode_metadata.actual_seq_lengths,
            active,
        )
    if metadata.num_accepted_tokens.data_ptr() != causal.num_accepted_tokens.data_ptr():
        raise ValueError("lifecycle fixture requires the qualified accepted-count alias")
    records = []
    for entry in schedule:
        prepare(entry)
        records.append(
            dict(
                descriptors=tuple(value.cpu().pin_memory() for value in tensors),
                inputs=(hidden.clone(), residual.clone()),
            )
        )
    staging = tuple(torch.empty_like(value, device="cpu", pin_memory=True) for value in tensors)
    # The terminal handoff resolves even paused owners' pending logs. Empty
    # immutable staging banks invalidate the log after its prefix is applied.
    final_counts = terminal_counts(schedule, capacity)
    terminal_sequence = torch.full((capacity + 1,), 4, dtype=torch.int32, device=hidden.device)
    terminal_sequence[0] = 0
    terminal_ids = torch.arange(1, capacity + 1, dtype=torch.int32, device=hidden.device).repeat_interleave(4)
    terminal_accepted = torch.empty(capacity, dtype=torch.int32, device=hidden.device)
    terminal_host = torch.tensor(final_counts, dtype=torch.int32, pin_memory=True)
    empty_logs, empty_valid = torch.zeros_like(compact.logs), torch.zeros_like(compact.valid)
    drain_stream, drain_graph = torch.npu.Stream(), torch.npu.NPUGraph()
    drain_stream.wait_stream(torch.npu.current_stream())
    terminal_accepted.copy_(terminal_host)
    with torch.npu.graph(drain_graph, stream=drain_stream, pool=torch.npu.graph_pool_handle()):
        backend.deferred_module.commit(
            compact.base,
            terminal_sequence,
            terminal_ids,
            terminal_accepted,
            compact.logs,
            compact.valid,
            empty_logs,
            empty_valid,
        )
    torch.npu.synchronize()
    restore(initial)

    def upload(record, stream):
        for target, host, packed in zip(tensors, record["descriptors"], staging, strict=True):
            packed.copy_(host)
            target.copy_(packed, non_blocking=True)
        # Staging is reused only after the step completion below. Frozen layer
        # inputs are device resident; their explicit D2D copies are charged.
        hidden.copy_(record["inputs"][0])
        residual.copy_(record["inputs"][1])

    samples, references = [], {}
    checks = []
    for round_id in range(args.rounds):
        for mode in modes if round_id % 2 == 0 else reversed(modes):
            restore(initial)
            graph, stream = graphs[mode]
            stream.wait_stream(torch.npu.current_stream())
            dist.barrier(group=cpu_group)
            started = time.perf_counter()
            device_us = 0.0
            # Synchronization here defines each dynamic step's completion and
            # prevents overwriting pinned staging during an outstanding DMA.
            for step, record in enumerate(records):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record(stream)
                    upload(record, stream)
                    graph.replay()
                    end.record(stream)
                end.synchronize()
                device_us += begin.elapsed_time(end) * 1000
            terminal_begin, terminal_end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            with torch.npu.stream(stream):
                terminal_begin.record(stream)
                if mode != "official_padding":
                    terminal_accepted.copy_(terminal_host, non_blocking=True)
                    drain_graph.replay()
                terminal_end.record(stream)
            terminal_end.synchronize()
            terminal_us = terminal_begin.elapsed_time(terminal_end) * 1000
            elapsed = (time.perf_counter() - started) * 1e6
            samples.append(
                dict(
                    mode=mode,
                    round=round_id,
                    steps=len(records),
                    device_us=device_us + terminal_us,
                    wall_us=elapsed,
                    terminal_device_us=terminal_us,
                )
            )
            torch.npu.current_stream().wait_stream(stream)
            outputs = tuple(value.clone() for value in graph_outputs[mode])
            if mode == "official_padding":
                slots = [compact.request_candidate_slots[owner][count - 1] for owner, count in enumerate(final_counts)]
                selected = compact.dense_state[torch.tensor(slots, device=hidden.device)].clone()
            else:
                selected = compact.base[1:].clone()
                if torch.count_nonzero(compact.valid).item() != 0:
                    raise AssertionError("terminal handoff left a pending compact log")
            torch.npu.synchronize()
            for value in (*outputs, selected):
                if not torch.isfinite(value).all().item():
                    raise AssertionError("nonfinite lifecycle output or terminal state")
            if mode not in references:
                references[mode] = (*outputs, selected)
            else:
                for value, expected in zip((*outputs, selected), references[mode], strict=True):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            if mode not in ("official_padding", "compact_padding"):
                for value, expected in zip((*outputs, selected), references["compact_padding"], strict=True):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            checks.append(
                dict(
                    mode=mode,
                    round=round_id,
                    continuous_steps=len(records),
                    terminal_finite=True,
                    terminal_logs_empty=mode != "official_padding",
                )
            )
    official, candidate = references["official_padding"][-1], references["compact_padding"][-1]
    difference = candidate.float() - official.float()
    nrmse = (
        difference.square().flatten(1).sum(1) / official.float().square().flatten(1).sum(1).clamp_min(1e-30)
    ).sqrt()
    if nrmse.max().item() > args.max_nrmse:
        raise AssertionError("terminal accepted state exceeds cross-algorithm NRMSE gate")
    return dict(
        samples=samples,
        checks=checks,
        terminal_counts=final_counts,
        terminal_worst_request_nrmse=nrmse.max().item(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        boundary="descriptor pack/H2D + device input staging + continuous real layers/joins + terminal handoff",
        excludes="scheduler, draft model, upstream layers, graph capture, synthetic fixture generation",
        terminal="resolve last accepted prefix for every owner, invalidate logs, retain canonical FP32 base",
        descriptor_bytes_per_step=sum(value.numel() * value.element_size() for value in tensors),
        extra_terminal_buffer_bytes=sum(
            value.numel() * value.element_size()
            for value in (terminal_sequence, terminal_ids, terminal_accepted, empty_logs, empty_valid)
        ),
    )


def summarize_lifecycle(ranks, import_cost):
    local = [rank["lifecycle"] for rank in ranks]
    samples = []
    for index, row in enumerate(local[0]["samples"]):
        if any(
            (item["samples"][index]["mode"], item["samples"][index]["round"]) != (row["mode"], row["round"])
            for item in local
        ):
            raise ValueError("rank lifecycle sample order differs")
        samples.append(
            dict(
                row,
                **{
                    field: max(item["samples"][index][field] for item in local)
                    for field in ("device_us", "wall_us", "terminal_device_us")
                },
            )
        )
    modes = sorted({row["mode"] for row in samples})
    summary = {}
    for mode in modes:
        rows = [row for row in samples if row["mode"] == mode]
        summary[mode] = dict(
            round_device_us=[row["device_us"] for row in rows],
            round_wall_us=[row["wall_us"] for row in rows],
            median_step_device_us=statistics.median(row["device_us"] / row["steps"] for row in rows),
            median_step_wall_us=statistics.median(row["wall_us"] / row["steps"] for row in rows),
            one_import_plus_trajectory_wall_us=[
                row["wall_us"] + (import_cost if mode != "official_padding" else 0) for row in rows
            ],
        )
    return dict(samples=samples, summary=summary, one_time_compact_import_wall_us=import_cost)
