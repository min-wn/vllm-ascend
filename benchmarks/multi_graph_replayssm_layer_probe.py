# SPDX-License-Identifier: Apache-2.0
"""Matched ReplaySSM delayed/ready and previous compact-ready real-layer lifecycle."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from multi_graph_replayssm_protocol import COMMUNITY_COMMIT, coverage, frozen_trace


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    import torch
    import torch.distributed as dist
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_log_backend import Q4LogBackend, Q4LogState
    from multi_graph_gdn_q4_log_layer_probe import installed_commit_pre_hook, restore_active_outputs
    from multi_graph_replayssm_backend import RingBackend, RingState, installed_ring
    from multi_graph_tp_gdn import qualify_gdn
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context

    rank, cpu_group = dist.get_rank(), get_world_group().cpu_group
    capacity, rows = workload.capacity, workload.capacity * 4
    if capacity != 100:
        raise ValueError("frozen qualification requires 400-token capture capacity")
    hidden_size = runner.vllm_config.model_config.hf_text_config.hidden_size
    hidden, residual = (torch.zeros(rows, hidden_size, dtype=torch.bfloat16, device="npu") for _ in range(2))
    active = torch.zeros(rows, 1, dtype=torch.bool, device="npu")
    fixture = qualify_gdn(args, workload, runner, model, (hidden, residual), cpu_group, setup_only=True)
    dense = fixture.states[1]
    candidates = [[1 + owner * 4 + token for token in range(4)] for owner in range(capacity)]
    for quartet in candidates:
        for slot in quartet[1:]:
            dense[slot].copy_(dense[quartet[0]])
    torch.npu.synchronize()
    start = time.perf_counter()
    ring = RingState(torch, dense, candidates)
    torch.npu.synchronize()
    ring_import_us = (time.perf_counter() - start) * 1e6
    start = time.perf_counter()
    compact = Q4LogState(torch, dense, candidates)
    torch.npu.synchronize()
    compact_import_us = (time.perf_counter() - start) * 1e6
    ring_module = load_extension("gdn_replayssm", args.ring_extension)
    ring_backend = RingBackend(ring, ring_module)
    compact_backend = Q4LogBackend(
        compact,
        load_extension("gdn_q4_log", args.extension),
        load_extension("gdn_q4_products", args.products_extension),
        mode="deferred_tile_ready",
        deferred_module=load_extension("gdn_q4_log_deferred", args.deferred_extension),
        tile_module=load_extension("gdn_q4_tile_pipeline", args.tile_extension),
    )
    modes = ("official", "compact_ready", "ring_delayed", "ring_ready")
    trace, terminal_counts = frozen_trace()
    with fixture.phases.context_for_name("full"):
        metadata = get_forward_context().attn_metadata[fixture.phases.gdn.prefix]
        causal = metadata.spec_decode_metadata.spec_causal_conv1d
        descriptors = (
            metadata.spec_state_indices_tensor,
            causal.query_start_loc,
            causal.cache_indices,
            causal.num_accepted_tokens,
            metadata.spec_decode_metadata.actual_seq_lengths,
            active,
        )
    assert metadata.num_accepted_tokens.data_ptr() == causal.num_accepted_tokens.data_ptr()

    def snapshot():
        return dict(
            dense=tuple(tensor.clone() for tensor in fixture.states), ring=ring.snapshot(), compact=compact.snapshot()
        )

    def restore(saved):
        fixture.reset(saved["dense"])
        ring.restore(saved["ring"])
        compact.restore(saved["compact"])
        torch.npu.synchronize()

    records = []
    for entry in trace:
        fixture.refresh(entry["live"], accepted=1)
        causal.num_accepted_tokens.copy_(torch.tensor(entry["accepted"], dtype=torch.int32, device="npu"))
        torch.manual_seed(entry["seed"])
        for tensor in (hidden, residual):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
            tensor[entry["live"] * 4 :].zero_()
        active.zero_()
        active[: entry["live"] * 4] = True
        records.append(
            dict(
                descriptors=tuple(tensor.cpu().pin_memory() for tensor in descriptors),
                inputs=(hidden.clone(), residual.clone()),
            )
        )
    staging = tuple(torch.empty_like(tensor, device="cpu", pin_memory=True) for tensor in descriptors)

    def upload(record):
        for target, host, packed in zip(descriptors, record["descriptors"], staging, strict=True):
            packed.copy_(host)
            target.copy_(packed, non_blocking=True)
        hidden.copy_(record["inputs"][0])
        residual.copy_(record["inputs"][1])

    def operation(mode):
        if mode.startswith("ring_"):
            ring_backend.streaming = mode == "ring_ready"
            with installed_ring(ring_backend):
                result = fixture.run(hidden, residual)
        elif mode == "compact_ready":
            with installed_ring(compact_backend):
                result = fixture.run(hidden, residual)
                compact_backend.finish_pending_commit()
        else:
            result = fixture.run(hidden, residual)
        return restore_active_outputs(torch, result, active)

    initial = snapshot()
    graphs, outputs, streams = {}, {}, {}
    checks, errors = [], []

    def exact(actual, wanted):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def metric(actual, wanted, label, step):
        actual, wanted = actual.float(), wanted.float()
        dims = tuple(range(1, actual.ndim))
        nrmse = ((actual - wanted).square().sum(dims) / wanted.square().sum(dims).clamp_min(1e-30)).sqrt()
        worst = nrmse.max().item()
        if not torch.isfinite(nrmse).all().item() or (worst > args.max_nrmse and not args.diagnostic_steps):
            raise AssertionError(f"{label} step={step} worst request NRMSE={worst}")
        errors.append(dict(step=step, label=label, worst_request_nrmse=worst))

    terminal_sequence = torch.full((capacity + 1,), 4, dtype=torch.int32, device="npu")
    terminal_sequence[0] = 0
    terminal_ids = torch.arange(1, capacity + 1, dtype=torch.int32, device="npu").repeat_interleave(4)
    terminal_accepted = torch.tensor(terminal_counts, dtype=torch.int32, device="npu")
    terminal_host = terminal_accepted.cpu().pin_memory()
    empty_logs, empty_valid = torch.zeros_like(compact.logs), torch.zeros_like(compact.valid)

    def drain(mode):
        if mode.startswith("ring_"):
            ring_backend.drain(terminal_sequence, terminal_ids, terminal_accepted)
        elif mode == "compact_ready":
            compact_backend.deferred_module.commit(
                compact.base,
                terminal_sequence,
                terminal_ids,
                terminal_accepted,
                compact.logs,
                compact.valid,
                empty_logs,
                empty_valid,
            )

    drains = {}
    with installed_commit_pre_hook(fixture.phases.gdn.out_proj, compact_backend):
        upload(records[0])
        for mode in modes:
            for _ in range(2):
                restore(initial)
                operation(mode)
                torch.npu.synchronize()
            restore(initial)
            stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
            stream.wait_stream(torch.npu.current_stream())
            dist.barrier(group=cpu_group)
            with torch.npu.graph(graph, stream=stream, pool=torch.npu.graph_pool_handle()):
                outputs[mode] = operation(mode)
            torch.npu.synchronize()
            graphs[mode], streams[mode] = graph, stream
            dgraph = torch.npu.NPUGraph()
            with torch.npu.graph(dgraph, stream=stream, pool=torch.npu.graph_pool_handle()):
                drain(mode)
            torch.npu.synchronize()
            drains[mode] = dgraph
        restore(initial)
        histories = {"official": initial, "compact_ready": initial, "ring_delayed": initial}
        for entry, record in zip(trace, records, strict=True):
            ring_before = histories["ring_delayed"]
            expected_ring, expected_ring_output, official_output = None, None, None
            for mode in modes:
                before = histories["ring_delayed" if mode.startswith("ring_") else mode]
                restore(before)
                upload(record)
                stream = streams[mode]
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    graphs[mode].replay()
                torch.npu.synchronize()
                observed_output = tuple(tensor.clone() for tensor in outputs[mode])
                saved = snapshot()
                for tensor in (*observed_output, *saved["ring"], *saved["compact"].tensors):
                    if not torch.isfinite(tensor).all().item():
                        raise AssertionError(f"nonfinite {mode} step={entry['step']}")
                if mode == "official":
                    official_output = observed_output
                else:
                    for index, label in enumerate(("hidden", "residual")):
                        metric(
                            observed_output[index][: entry["live"] * 4].view(entry["live"], 4, -1),
                            official_output[index][: entry["live"] * 4].view(entry["live"], 4, -1),
                            f"{mode}/{label}",
                            entry["step"],
                        )
                if mode.startswith("ring_"):
                    for tensor, previous in zip(saved["ring"], before["ring"], strict=True):
                        exact(tensor[0], previous[0])
                        exact(tensor[entry["live"] + 1 :], previous[entry["live"] + 1 :])
                    exact(saved["dense"][1], before["dense"][1])
                    if mode == "ring_delayed":
                        expected_ring, expected_ring_output = saved, observed_output
                    else:
                        for current, wanted in zip(saved["ring"], expected_ring["ring"], strict=True):
                            exact(current, wanted)
                        for current, wanted in zip(observed_output, expected_ring_output, strict=True):
                            exact(current, wanted)
                        for current, wanted in zip(saved["dense"], expected_ring["dense"], strict=True):
                            exact(current, wanted)
                if mode in ("official", "compact_ready"):
                    histories[mode] = saved
                checks.append(
                    dict(step=entry["step"], mode=mode, continuous=True, same_algorithm_exact=mode == "ring_ready")
                )
            histories["ring_delayed"] = expected_ring
            print(f"rank={rank} qualified step={entry['step']}", flush=True)
            if args.diagnostic_steps and entry["step"] + 1 == args.diagnostic_steps:
                from multi_graph_replayssm_probe import oracle_step
                from multi_graph_replayssm_protocol import Cursor

                restore(ring_before)
                upload(record)
                eager = operation("ring_ready")
                torch.npu.synchronize()
                query, key, value, beta, gate, _, _, _ = ring_backend.inputs
                # Resolve the largest output error request and reconstruct only
                # that real recurrence input against the FP16-ring oracle.
                actual, wanted = eager[0].float(), official_output[0].float()
                ratios = (
                    (actual - wanted).view(capacity, -1).square().sum(1)
                    / wanted.view(capacity, -1).square().sum(1).clamp_min(1e-30)
                ).sqrt()
                owner = int(ratios.argmax().item())
                row = slice(owner * 4, (owner + 1) * 4)
                cache = [torch.stack((tensor[0], tensor[owner + 1])) for tensor in ring_before["ring"][:3]]
                cursor_values = ring_before["ring"][3][owner + 1, :, 0].cpu().tolist()
                cursor = Cursor(cursor_values[0], cursor_values[1], bool(cursor_values[2]), bool(cursor_values[3]))
                reference = oracle_step(
                    torch,
                    cache,
                    [cursor],
                    query[row],
                    key[row],
                    value[row],
                    beta[row],
                    gate[row],
                    1,
                    [entry["accepted"][owner]],
                )
                native = ring_backend.buffers[next(reversed(ring_backend.buffers))]["output"][row]
                error = (
                    (
                        (native.float() - reference.float()).square().sum()
                        / reference.float().square().sum().clamp_min(1e-30)
                    )
                    .sqrt()
                    .item()
                )
                diagnostic = dict(
                    rank=rank,
                    owner=owner,
                    real_recurrence_oracle_nrmse=error,
                    dense_errors=errors,
                    precision_gate_passed=False,
                )
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, diagnostic, group=cpu_group)
                if rank == 0:
                    args.output.write_text(json.dumps(dict(status="diagnostic_only", ranks=gathered), indent=2) + "\n")
                    print(json.dumps(dict(status="diagnostic_only", ranks=gathered)), flush=True)
                return
        terminal_reference = None
        closed_references = {}
        for mode in modes:
            restore(histories["ring_delayed" if mode.startswith("ring_") else mode])
            drain(mode)
            torch.npu.synchronize()
            if mode == "official":
                selected = torch.tensor(
                    [slots[count - 1] for slots, count in zip(candidates, terminal_counts, strict=True)], device="npu"
                )
                terminal_reference = dense.index_select(0, selected).clone()
            else:
                state = ring.base if mode.startswith("ring_") else compact.base
                metric(state[1:], terminal_reference, f"{mode}/terminal_state", 40)
                if mode.startswith("ring_"):
                    assert not ring.cursors.count_nonzero().item()
                else:
                    assert not compact.valid.count_nonzero().item()
            closed_references[mode] = snapshot()

        samples = []
        for round_id in range(args.rounds):
            for mode in modes if round_id % 2 == 0 else reversed(modes):
                restore(initial)
                stream = streams[mode]
                stream.wait_stream(torch.npu.current_stream())
                dist.barrier(group=cpu_group)
                start = time.perf_counter()
                device_us, step_times = 0.0, []
                for record in records:
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    with torch.npu.stream(stream):
                        begin.record(stream)
                        upload(record)
                        graphs[mode].replay()
                        end.record(stream)
                    end.synchronize()
                    elapsed = begin.elapsed_time(end) * 1000
                    step_times.append(elapsed)
                    device_us += elapsed
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record(stream)
                    terminal_accepted.copy_(terminal_host, non_blocking=True)
                    drains[mode].replay()
                    end.record(stream)
                end.synchronize()
                terminal_us = begin.elapsed_time(end) * 1000
                wall_us = (time.perf_counter() - start) * 1e6
                closed = closed_references[mode]
                for actual, wanted in zip(fixture.states, closed["dense"], strict=True):
                    exact(actual, wanted)
                for actual, wanted in zip(ring.mutable, closed["ring"], strict=True):
                    exact(actual, wanted)
                for actual, wanted in zip(compact.mutable_tensors, closed["compact"].tensors, strict=True):
                    exact(actual, wanted)
                imported = (
                    ring_import_us if mode.startswith("ring_") else compact_import_us if mode == "compact_ready" else 0
                )
                samples.append(
                    dict(
                        round=round_id,
                        mode=mode,
                        wall_us=wall_us,
                        device_us=device_us + terminal_us,
                        terminal_us=terminal_us,
                        import_us=imported,
                        closed_wall_us=wall_us + imported,
                        step_device_us=step_times,
                        terminal_exact=True,
                    )
                )
                print(f"rank={rank} round={round_id} mode={mode} closed_us={wall_us + imported:.1f}", flush=True)
    result = dict(
        rank=rank,
        ring_import_us=ring_import_us,
        compact_import_us=compact_import_us,
        samples=samples,
        checks=checks,
        errors=errors,
        persistent_bytes=dict(
            ring=sum(t.numel() * t.element_size() for t in ring.mutable),
            compact=sum(t.numel() * t.element_size() for t in compact.mutable_tensors),
        ),
    )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, result, group=cpu_group)
    if rank == 0:
        paired = []
        for round_id in range(args.rounds):
            row = dict(round=round_id)
            for mode in modes:
                per_rank = [
                    next(
                        sample for sample in entry["samples"] if sample["round"] == round_id and sample["mode"] == mode
                    )
                    for entry in ranks
                ]
                row[mode] = {
                    name: max(sample[name] for sample in per_rank)
                    for name in ("wall_us", "closed_wall_us", "device_us", "terminal_us")
                }
            row["ready_net_percent"] = 100 * (
                1 - row["ring_ready"]["closed_wall_us"] / row["ring_delayed"]["closed_wall_us"]
            )
            row["ready_vs_compact_percent"] = 100 * (
                1 - row["ring_ready"]["closed_wall_us"] / row["compact_ready"]["closed_wall_us"]
            )
            paired.append(row)
        report = dict(
            status="qualified_40_step_real_layer_closed_lifecycle",
            community_commit=COMMUNITY_COMMIT,
            scope="Qwen3.5-27B TP2 layer 4; not full model or serving",
            trace=trace,
            terminal_counts=terminal_counts,
            coverage=coverage(trace),
            ranks=ranks,
            paired=paired,
            precision=dict(
                activation="BF16", checkpoint="FP32", ring_dk="FP32", ring_g="FP32", compact_log="FP32", hf32=False
            ),
            boundary=(
                "CPU descriptor packing/H2D, input D2D, complete layer, cursor update, every flush, "
                "per-step join, final accepted-state drain and one-time import"
            ),
            excludes=(
                "fixture/model loading, input synthesis, diagnostic copies, capture/compilation; "
                "import is one observation per process"
            ),
            port_differences=(
                "A2 keys duplicated per V head; padded g/cursors; separate pre-output AIV flush; "
                "sequential FP32 replay; not GPU-kernel performance parity"
            ),
            source_sha256={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("multi_graph_replayssm*.py")
            },
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                dict(
                    status=report["status"],
                    paired=paired,
                    mean_ready_percent=statistics.mean(row["ready_net_percent"] for row in paired),
                )
            ),
            flush=True,
        )


def main():
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    for name in ("extension", "products-extension", "deferred-extension", "tile-extension", "ring-extension"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, type=int, default=(200, 200))
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-nrmse", type=float, default=0.005)
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.rounds < 2 or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("new output path, >=2 rounds and torchrun TP2 required")
    args.query, args.dtype, args.batch_invariant, args.compiled = 4, "bfloat16", False, False
    args.gdn_probe, args.gdn_pipeline = True, False
    args.selection = select_workload(args)
    args.pair = args.selection["pair_tokens"]
    batch, capacity = args.selection["batch"], args.selection["capacity_requests"]
    workload = ReplayWorkload(batch, capacity, args.query, args.kv_length, (50, batch - 50))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_qwen35(args, workload, experiment=experiment)


if __name__ == "__main__":
    main()
