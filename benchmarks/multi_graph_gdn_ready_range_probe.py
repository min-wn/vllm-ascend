# SPDX-License-Identifier: Apache-2.0
"""Real-activation Q4 load-range qualification and matched ready ablation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from multi_graph_gdn_ready_range_protocol import protocol

MODES = ("official", "compact_delayed", "compact_ready", "ring_delayed", "ring_ready")


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    import torch
    import torch.distributed as dist
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_log_backend import Q4LogBackend, Q4LogState
    from multi_graph_gdn_q4_log_layer_probe import installed_commit_pre_hook, restore_active_outputs
    from multi_graph_gdn_ready_activations import ActivationSource
    from multi_graph_replayssm_backend import RingBackend, RingState, installed_ring
    from multi_graph_tp_gdn import qualify_gdn
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context

    rank, group = dist.get_rank(), get_world_group().cpu_group
    torch.npu.matmul.allow_hf32 = False
    frozen = protocol()
    source = ActivationSource(torch, model, runner.vllm_config)
    pref_conv, pref_dense = source.initial_target()
    hidden_size = runner.vllm_config.model_config.hf_text_config.hidden_size
    hidden, residual = (torch.zeros(400, hidden_size, dtype=torch.bfloat16, device="npu") for _ in range(2))
    fixture = qualify_gdn(args, workload, runner, model, (hidden, residual), group, setup_only=True)
    conv, dense = fixture.states
    candidates = [[1 + owner * 4 + token for token in range(4)] for owner in range(100)]
    for owner, slots in enumerate(candidates, 1):
        for slot in slots:
            dense[slot].copy_(pref_dense[owner])
            conv[slot].copy_(pref_conv[owner])
    dense[0].zero_()
    conv[0].zero_()
    torch.npu.synchronize()
    start = time.perf_counter()
    ring = RingState(torch, dense, candidates)
    torch.npu.synchronize()
    ring_import_us = (time.perf_counter() - start) * 1e6
    start = time.perf_counter()
    compact = Q4LogState(torch, dense, candidates)
    torch.npu.synchronize()
    compact_import_us = (time.perf_counter() - start) * 1e6
    ring_backend = RingBackend(ring, load_extension("gdn_replayssm", args.ring_extension))
    backend = Q4LogBackend(
        compact,
        load_extension("gdn_q4_log", args.extension),
        load_extension("gdn_q4_products", args.products_extension),
        mode="deferred_tile_ready",
        deferred_module=load_extension("gdn_q4_log_deferred", args.deferred_extension),
        tile_module=load_extension("gdn_q4_tile_pipeline", args.tile_extension),
    )
    backend.reserve_exclusive_capacity(100)
    staging_state = (backend._exclusive_buffers["new_logs"], backend._exclusive_buffers["new_valid"])
    active = torch.zeros(400, 1, dtype=torch.bool, device="npu")
    fixture.refresh(100)
    with fixture.phases.context_for_name("full"):
        meta = get_forward_context().attn_metadata[fixture.phases.gdn.prefix]
    causal = meta.spec_decode_metadata.spec_causal_conv1d
    assert causal.cache_indices.data_ptr() == meta.spec_state_indices_tensor.data_ptr()
    descriptors = (
        meta.spec_state_indices_tensor,
        causal.query_start_loc,
        causal.num_accepted_tokens,
        meta.spec_decode_metadata.actual_seq_lengths,
        active,
    )
    packed = tuple(torch.empty_like(tensor, device="cpu", pin_memory=True) for tensor in descriptors)

    def snapshot():
        return dict(
            dense=tuple(tensor.clone() for tensor in fixture.states),
            ring=ring.snapshot(),
            compact=tuple(tensor.clone() for tensor in compact.mutable_tensors),
            staging=tuple(tensor.clone() for tensor in staging_state),
        )

    def restore(saved):
        for tensors, key in (
            (fixture.states, "dense"),
            (ring.mutable, "ring"),
            (compact.mutable_tensors, "compact"),
            (staging_state, "staging"),
        ):
            for target, value in zip(tensors, saved[key], strict=True):
                target.copy_(value)
        torch.npu.synchronize()

    def exact_snapshot(actual, expected):
        for key in actual:
            for left, right in zip(actual[key], expected[key], strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def metadata_record(entry, pair):
        live = entry["live"]
        blocks = torch.zeros_like(packed[0])
        for row, owner in enumerate(entry["request_ids"]):
            blocks[row] = torch.tensor(candidates[owner], dtype=torch.int32)
        boundaries = torch.tensor([min(row, live) * 4 for row in range(101)], dtype=torch.int32)
        counts = torch.tensor([*entry["accepted"], *([1] * (100 - live))], dtype=torch.int32)
        lengths = torch.tensor([0, *([4] * live), *([0] * (100 - live))], dtype=torch.int32)
        mask = torch.arange(400).view(-1, 1) < live * 4
        return dict(host=tuple(t.pin_memory() for t in (blocks, boundaries, counts, lengths, mask)), inputs=pair)

    def upload(record):
        for target, scratch, host in zip(descriptors, packed, record["host"], strict=True):
            scratch.copy_(host)
            target.copy_(scratch, non_blocking=True)
        hidden.copy_(record["inputs"][0])
        residual.copy_(record["inputs"][1])

    def operation(mode):
        if mode.startswith("ring_"):
            ring_backend.streaming = mode == "ring_ready"
            with installed_ring(ring_backend):
                result = fixture.run(hidden, residual)
        elif mode.startswith("compact_"):
            backend.mode = "deferred_tile_ready" if mode == "compact_ready" else "deferred_tile_delayed"
            with installed_ring(backend):
                result = fixture.run(hidden, residual)
                backend.finish_pending_commit()
        else:
            result = fixture.run(hidden, residual)
        return restore_active_outputs(torch, result, active)

    terminal_sequence = torch.full((101,), 4, dtype=torch.int32, device="npu")
    terminal_sequence[0] = 0
    terminal_ids = torch.arange(1, 101, dtype=torch.int32, device="npu").repeat_interleave(4)
    terminal_accepted = torch.ones(100, dtype=torch.int32, device="npu")
    empty_logs, empty_valid = torch.zeros_like(compact.logs), torch.zeros_like(compact.valid)

    def drain(mode):
        if mode.startswith("ring_"):
            ring_backend.drain(terminal_sequence, terminal_ids, terminal_accepted)
        elif mode.startswith("compact_"):
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

    initial = snapshot()
    graphs, outputs, streams, drains = {}, {}, {}, {}
    local_profiles = []
    with installed_commit_pre_hook(fixture.phases.gdn.out_proj, backend):
        for profile_index, profile in enumerate(frozen["profiles"]):
            activations, activation_manifest = source.records(profile, hidden_size)
            restore(initial)
            records = [metadata_record(entry, pair) for entry, pair in zip(profile["trace"], activations, strict=True)]
            if profile_index == 0:
                upload(records[0])
                for mode in MODES:
                    for _ in range(2):
                        restore(initial)
                        operation(mode)
                        torch.npu.synchronize()
                    restore(initial)
                    stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
                    stream.wait_stream(torch.npu.current_stream())
                    dist.barrier(group=group)
                    with torch.npu.graph(graph, stream=stream, pool=torch.npu.graph_pool_handle()):
                        outputs[mode] = operation(mode)
                    torch.npu.synchronize()
                    dgraph = torch.npu.NPUGraph()
                    with torch.npu.graph(dgraph, stream=stream, pool=torch.npu.graph_pool_handle()):
                        drain(mode)
                    torch.npu.synchronize()
                    streams[mode], graphs[mode], drains[mode] = stream, graph, dgraph
            histories = {mode: initial for mode in ("official", "compact_delayed", "ring_delayed")}
            errors, checks = [], []

            def metric(actual, expected, label, step, profile_name=profile["name"], error_rows=errors):
                dims = tuple(range(1, actual.ndim))
                a, b = actual.float(), expected.float()
                ratios = ((a - b).square().sum(dims) / b.square().sum(dims).clamp_min(1e-30)).sqrt()
                worst = ratios.max().item()
                if not torch.isfinite(ratios).all().item() or worst > 0.005:
                    raise AssertionError(f"{profile_name} {label} step={step} worst NRMSE={worst}")
                error_rows.append(dict(step=step, label=label, worst_request_nrmse=worst))

            for entry, record in zip(profile["trace"], records, strict=True):
                expected_pairs, next_histories, official_output = {}, {}, None
                for mode in MODES:
                    family = (
                        "compact_delayed"
                        if mode.startswith("compact_")
                        else "ring_delayed"
                        if mode.startswith("ring_")
                        else "official"
                    )
                    before = histories[family]
                    restore(before)
                    upload(record)
                    stream = streams[mode]
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        graphs[mode].replay()
                    torch.npu.synchronize()
                    observed = tuple(t.clone() for t in outputs[mode])
                    saved = snapshot()
                    for tensor in (*observed, *(tensor for bank in saved.values() for tensor in bank)):
                        if not torch.isfinite(tensor).all().item():
                            raise AssertionError(f"nonfinite {mode}")
                    if mode == "official":
                        official_output = observed
                    else:
                        for label, a, b in zip(("hidden", "residual"), observed, official_output, strict=True):
                            metric(
                                a[: entry["live"] * 4].view(entry["live"], 4, -1),
                                b[: entry["live"] * 4].view(entry["live"], 4, -1),
                                mode + "/" + label,
                                entry["step"],
                            )
                    paused = [0, *(owner + 1 for owner in range(100) if owner not in entry["request_ids"])]
                    bank = "ring" if mode.startswith("ring_") else "compact" if mode.startswith("compact_") else None
                    if bank:
                        for tensor, prior in zip(saved[bank], before[bank], strict=True):
                            torch.testing.assert_close(tensor[paused], prior[paused], rtol=0, atol=0)
                        torch.testing.assert_close(saved["dense"][1], before["dense"][1], rtol=0, atol=0)
                    inactive_slots = [
                        0,
                        *(
                            slot
                            for owner in range(100)
                            if owner not in entry["request_ids"]
                            for slot in candidates[owner]
                        ),
                    ]
                    for tensor, prior in zip(saved["dense"], before["dense"], strict=True):
                        torch.testing.assert_close(tensor[inactive_slots], prior[inactive_slots], rtol=0, atol=0)
                    if mode.endswith("_ready"):
                        prior_output, prior_state = expected_pairs[family]
                        exact_snapshot(saved, prior_state)
                        for a, b in zip(observed, prior_output, strict=True):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    else:
                        expected_pairs[family] = (observed, saved)
                        next_histories[family] = saved
                    checks.append(
                        dict(
                            step=entry["step"],
                            mode=mode,
                            paused_exact=True,
                            same_algorithm_exact=mode.endswith("_ready"),
                        )
                    )
                histories = next_histories
                print(f"rank={rank} profile={profile['name']} qualified={entry['step']}", flush=True)
            terminal_host = torch.tensor(profile["terminal_counts"], dtype=torch.int32).pin_memory()
            terminal_accepted.copy_(terminal_host)
            closed = {}
            terminal_reference = None
            for mode in MODES:
                family = (
                    "compact_delayed"
                    if mode.startswith("compact_")
                    else "ring_delayed"
                    if mode.startswith("ring_")
                    else "official"
                )
                restore(histories[family])
                drain(mode)
                torch.npu.synchronize()
                if mode == "official":
                    selected = torch.tensor(
                        [slots[n - 1] for slots, n in zip(candidates, profile["terminal_counts"], strict=True)],
                        device="npu",
                    )
                    terminal_reference = dense.index_select(0, selected).clone()
                else:
                    state = ring.base if mode.startswith("ring_") else compact.base
                    metric(state[1:], terminal_reference, mode + "/terminal", 40)
                    assert not (ring.cursors if mode.startswith("ring_") else compact.valid).count_nonzero().item()
                closed[mode] = snapshot()
            samples = []
            for round_id in range(3):
                for mode in MODES if round_id % 2 == 0 else reversed(MODES):
                    restore(initial)
                    stream = streams[mode]
                    stream.wait_stream(torch.npu.current_stream())
                    dist.barrier(group=group)
                    start = time.perf_counter()
                    step_times = []
                    for record in records:
                        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        with torch.npu.stream(stream):
                            begin.record(stream)
                            upload(record)
                            graphs[mode].replay()
                            end.record(stream)
                        end.synchronize()
                        step_times.append(begin.elapsed_time(end) * 1000)
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    with torch.npu.stream(stream):
                        begin.record(stream)
                        terminal_accepted.copy_(terminal_host, non_blocking=True)
                        drains[mode].replay()
                        end.record(stream)
                    end.synchronize()
                    wall = (time.perf_counter() - start) * 1e6
                    terminal_us = begin.elapsed_time(end) * 1000
                    exact_snapshot(snapshot(), closed[mode])
                    imported = (
                        ring_import_us
                        if mode.startswith("ring_")
                        else compact_import_us
                        if mode.startswith("compact_")
                        else 0
                    )
                    samples.append(
                        dict(
                            round=round_id,
                            mode=mode,
                            wall_us=wall,
                            closed_wall_us=wall + imported,
                            device_us=sum(step_times) + terminal_us,
                            terminal_us=terminal_us,
                            import_us=imported,
                            step_device_us=step_times,
                            terminal_exact=True,
                        )
                    )
                    print(
                        f"rank={rank} profile={profile['name']} round={round_id} "
                        f"mode={mode} closed={wall + imported:.1f}",
                        flush=True,
                    )
            local_profiles.append(
                dict(
                    name=profile["name"],
                    activation_manifest=activation_manifest,
                    checks=checks,
                    errors=errors,
                    samples=samples,
                )
            )
            del records, activations, histories, closed, expected_pairs, next_histories
    local = dict(rank=rank, profiles=local_profiles, ring_import_us=ring_import_us, compact_import_us=compact_import_us)
    gathered = [None] * 2
    dist.all_gather_object(gathered, local, group=group)
    if rank == 0:
        report = dict(
            status="qualified_real_activation_single_layer_ranges",
            protocol=frozen,
            ranks=gathered,
            token_sha256=source.token_sha256,
            scope="real text prefill and official layers 0..3 activations; layer4 only timed",
            precision=dict(activation="BF16", compact_log="FP32", ring_dk="FP32", state="FP32", hf32=False),
            boundary=(
                "descriptor packing/H2D, input D2D, full layer, state updates/flush, "
                "per-step join, final accepted drain and one import"
            ),
            excludes="text/activation generation, common prefill, graph capture, diagnostics and model loading",
            sources={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("multi_graph_gdn_ready*.py")
            },
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status=report["status"], protocol_sha256=frozen["sha256"])), flush=True)


def main():
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    for name in ("extension", "products-extension", "deferred-extension", "tile-extension", "ring-extension"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("new output and TP2 required")
    args.query, args.dtype, args.batch_invariant, args.compiled = 4, "bfloat16", False, False
    args.gdn_probe, args.gdn_pipeline, args.kv_length, args.pair = True, False, 512, (200, 200)
    args.selection = select_workload(args)
    run_qwen35(args, ReplayWorkload(98, 100, 4, 512, (50, 48)), experiment=experiment)


if __name__ == "__main__":
    main()
