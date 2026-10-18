# SPDX-License-Identifier: Apache-2.0
"""Qualify compact Q4 recurrence inside one complete real TP2 GDN decoder layer.

The layer includes official convolution, gating, Q/K normalization, input/output
projections, TP reductions, residual paths and MLP. Only the recurrence backend
changes. Synthetic hidden/residual inputs and metadata are prepared outside the
timer. This is a connected real-layer causal experiment, not full-model replay.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


def qualification_schedule(batch, capacity):
    """Fix eight dynamic steps; acceptance belongs to a stable request generation."""
    if not 3 <= batch <= capacity:
        raise ValueError("three or more live requests within capacity required")
    generations = [0] * capacity
    result = []
    for step, live in enumerate((batch, batch - 1, min(capacity, batch + 1), batch) * 2):
        counts = [1 + (request + generations[request]) % 4 for request in range(capacity)]
        result.append(dict(step=step, live=live, accepted=counts))
        for request in range(live):
            generations[request] += 1
    return result


def mode_backend_name(mode):
    if mode == "official_padding":
        return None
    modes = dict(
        compact_padding="padding",
        compact_serial="serial",
        compact_pipeline="pipeline",
        compact_deferred_serial="deferred_serial",
        compact_deferred_overlap="deferred_overlap",
        compact_deferred_pipeline="deferred_pipeline",
        compact_deferred_tile_delayed="deferred_tile_delayed",
        compact_deferred_tile_ready="deferred_tile_ready",
        compact_deferred_tile_separate="deferred_tile_separate",
    )
    if mode not in modes:
        raise ValueError("unknown real-layer recurrence control")
    return modes[mode]


def layer_modes(deferred_extension, tile_extension=None):
    if tile_extension is not None and not deferred_extension:
        raise ValueError("request-tile controls require the deferred extension")
    modes = ("official_padding", "compact_padding", "compact_serial", "compact_pipeline")
    modes += (
        ("compact_deferred_serial", "compact_deferred_overlap", "compact_deferred_pipeline")
        if deferred_extension
        else ()
    )
    return modes + (
        ("compact_deferred_tile_delayed", "compact_deferred_tile_ready", "compact_deferred_tile_separate")
        if tile_extension is not None
        else ()
    )


def deferred_pipeline_comparison(round_medians):
    """Attribute only request-range overlap beyond the common deferred commit."""
    baseline, candidate = "compact_deferred_overlap", "compact_deferred_pipeline"
    if candidate not in round_medians:
        return None
    baseline_times, candidate_times = round_medians[baseline], round_medians[candidate]
    return dict(
        baseline=baseline,
        candidate=candidate,
        common_schedule="both use deferred accepted-state commit overlapping output GEMM/TP/MLP and a timed final join",
        incremental_change="fixed-half request products/solve pipeline before the same deferred commit schedule",
        baseline_round_medians_us=baseline_times,
        candidate_round_medians_us=candidate_times,
        paired_round_latency_reduction_percent=[
            100 * (1 - after / before) for before, after in zip(baseline_times, candidate_times, strict=True)
        ],
        median_of_round_medians_reduction_percent=(
            100 * (1 - statistics.median(candidate_times) / statistics.median(baseline_times))
        ),
        scope="one complete real GDN layer; does not establish full-model replay improvement",
    )


def tile_pipeline_comparison(round_medians):
    """Keep public kernel gains separate from request-tile readiness overlap."""
    candidate = "compact_deferred_tile_ready"
    if candidate not in round_medians:
        return None
    public_controls = (
        "compact_deferred_overlap",
        "compact_deferred_tile_separate",
        "compact_deferred_tile_delayed",
    )
    baseline = min(public_controls, key=lambda mode: statistics.median(round_medians[mode]))

    def comparison(control):
        before, after = round_medians[control], round_medians[candidate]
        return dict(
            baseline=control,
            candidate=candidate,
            baseline_round_medians_us=before,
            candidate_round_medians_us=after,
            paired_round_latency_reduction_percent=[
                100 * (1 - current / previous) for previous, current in zip(before, after, strict=True)
            ],
            median_of_round_medians_reduction_percent=(
                100 * (1 - statistics.median(after) / statistics.median(before))
            ),
        )

    return dict(
        scope="one complete real GDN layer; does not establish full-model replay improvement",
        common_schedule="all controls retain deferred accepted-state commit and the timed complete-layer join",
        baseline_selection="lowest median of round medians among all three public padding controls",
        eligible_public_padding_controls=list(public_controls),
        versus_best_public_padding=comparison(baseline),
        versus_each_public_padding={control: comparison(control) for control in public_controls},
        same_kernel_readiness_increment=comparison("compact_deferred_tile_delayed"),
        versus_previous_request_pipeline=comparison("compact_deferred_pipeline"),
        attribution=(
            "delayed and ready use the same tile producer/consumer kernels; delayed waits for each producer wave, "
            "while ready consumes request tiles as they become available within that wave; "
            "separate and delayed expose public kernel gains to padding before assigning incremental split benefit"
        ),
    )


@contextmanager
def installed_commit_pre_hook(projection, backend):
    """Start pending commit after gated Norm, before the real output GEMM."""

    def launch(module, inputs):
        backend.launch_pending_commit()
        # Forward pre-hooks must not replace the projection's arguments.

    handle = projection.register_forward_pre_hook(launch)
    try:
        yield
    finally:
        handle.remove()


def restore_active_outputs(torch, values, active):
    """Define inactive output lanes without propagating unspecified NaNs."""
    return tuple(torch.where(active, value, 0) for value in values)


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    import torch
    import torch.distributed as dist
    from multi_graph_gdn_q4_log_backend import Q4LogState, installed_log_backend
    from multi_graph_tp_gdn import qualify_gdn
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context

    rank, world = dist.get_rank(), dist.get_world_size()
    cpu_group = get_world_group().cpu_group
    if world != 2 or args.tensor_parallel_size != 2 or workload.query != 4:
        raise ValueError("native compact kernels currently require TP2 and four-token requests")
    rows = workload.capacity * 4
    hidden_size = runner.vllm_config.model_config.hf_text_config.hidden_size
    torch.manual_seed(20719)
    hidden, residual = (torch.randn(rows, hidden_size, dtype=torch.bfloat16, device="npu") * 0.1 for _ in range(2))
    output_active = torch.zeros((rows, 1), dtype=torch.bool, device="npu")
    fixture = qualify_gdn(args, workload, runner, model, (hidden, residual), cpu_group, setup_only=True)
    dense_state = fixture.states[1]
    if dense_state.dtype != torch.float32 or tuple(dense_state.shape[1:]) != (24, 128, 128):
        raise ValueError("real model did not expose the required FP32 TP2 SSM state")
    candidate_slots = [[1 + request * 4 + token for token in range(4)] for request in range(workload.capacity)]
    # All accepted counts select the same first-use SSM state. The official
    # convolution cache keeps its original independent speculative semantics.
    for quartet in candidate_slots:
        seed = dense_state[quartet[0]].clone()
        for slot in quartet:
            dense_state[slot].copy_(seed)
    compact_state = Q4LogState(torch, dense_state, candidate_slots, [1] * workload.capacity)
    prefix = fixture.phases.gdn.prefix
    modes = layer_modes(args.deferred_extension, args.tile_extension)
    schedule = qualification_schedule(workload.batch, workload.capacity)
    graphs, graph_outputs, checks, diagnostics, samples = {}, {}, [], [], []

    def snapshot():
        compact = compact_state.snapshot()
        return dict(
            dense=tuple(tensor.clone() for tensor in fixture.states),
            compact=compact,
            compact_values=compact.tensors,
        )

    def restore(saved):
        fixture.reset(saved["dense"])
        compact_state.restore(saved["compact"])
        torch.npu.synchronize()

    def exact(actual, expected, label):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda detail: f"{label}: {detail}")

    def exact_state(saved, label):
        for index, (actual, expected) in enumerate(zip(fixture.states, saved["dense"], strict=True)):
            exact(actual, expected, f"{label}/dense{index}")
        for index, (actual, expected) in enumerate(
            zip(compact_state.mutable_tensors, saved["compact_values"], strict=True)
        ):
            exact(actual, expected, f"{label}/compact{index}")

    def exact_outputs(actual, expected, label):
        for name, observed, wanted in zip(("hidden", "residual"), actual, expected, strict=True):
            exact(observed, wanted, f"{label}/{name}")

    def prepare(entry):
        fixture.refresh(entry["live"], accepted=1)
        with fixture.phases.context_for_name("full"):
            metadata = get_forward_context().attn_metadata[prefix]
            causal = metadata.spec_decode_metadata.spec_causal_conv1d
            counts = torch.tensor(entry["accepted"], dtype=torch.int32, device=dense_state.device)
            causal.num_accepted_tokens.copy_(counts)
            # Current builder aliases these buffers; keep both fields coherent
            # explicitly so a future builder change cannot silently split them.
            if getattr(metadata, "num_accepted_tokens", None) is not None:
                metadata.num_accepted_tokens.copy_(counts)
        torch.manual_seed(20719 + entry["step"])
        for tensor in (hidden, residual):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
            tensor[entry["live"] * 4 :].zero_()
        output_active.zero_()
        output_active[: entry["live"] * 4].fill_(True)
        torch.npu.synchronize()

    def finite(values, label):
        for value in values:
            if not torch.isfinite(value).all().item():
                raise AssertionError(f"{label}: nonfinite output or state")

    def cross_metrics(actual, expected, label, step):
        actual, expected = actual.float(), expected.float()
        delta = actual - expected
        dimensions = tuple(range(1, actual.ndim))
        relative = (delta.square().sum(dimensions) / expected.square().sum(dimensions).clamp_min(1e-30)).sqrt()
        metric = dict(
            rank=rank,
            step=step,
            field=label,
            max_abs=delta.abs().max().item(),
            worst_request_nrmse=relative.max().item(),
        )
        if not torch.isfinite(relative).all().item() or metric["worst_request_nrmse"] > args.max_nrmse:
            raise AssertionError(f"cross-algorithm screening gate failed: {metric}")
        diagnostics.append(metric)

    def unchanged_paused(saved, live, mode):
        dense_slots = [0] + [slot for quartet in candidate_slots[live:] for slot in quartet]
        slot_index = torch.tensor(dense_slots, device=dense_state.device)
        for index, (actual, old) in enumerate(zip(fixture.states, saved["dense"], strict=True)):
            exact(actual[slot_index], old[slot_index], f"{mode}/null_inactive_dense{index}")
        owners = torch.arange(live + 1, workload.capacity + 1, device=dense_state.device)
        owners = torch.cat((torch.zeros(1, dtype=torch.int64, device=dense_state.device), owners))
        for index, (actual, old) in enumerate(zip(compact_state.mutable_tensors, saved["compact_values"], strict=True)):
            exact(actual[owners], old[owners], f"{mode}/null_inactive_compact{index}")

    def clone_outputs(values):
        return tuple(value.clone() for value in values)

    prepare(schedule[0])
    initial = snapshot()
    with (
        installed_log_backend(
            compact_state,
            args.extension,
            args.products_extension,
            mode="padding",
            deferred_path=args.deferred_extension,
            tile_path=args.tile_extension,
        ) as backend,
        installed_commit_pre_hook(fixture.phases.gdn.out_proj, backend),
    ):

        def operation(mode):
            backend_mode = mode_backend_name(mode)
            if backend_mode is None:
                with backend.suspended():
                    result = fixture.run(hidden, residual)
            else:
                backend.mode = backend_mode
                result = fixture.run(hidden, residual)
            # Commit can overlap output projection, both TP reductions and MLP.
            # Its final join remains inside every deferred layer endpoint.
            backend.finish_pending_commit()
            # The official recurrence leaves inactive output lanes unspecified.
            # Restore the same explicit zero contract in every timed control.
            return restore_active_outputs(torch, result, output_active)

        # Each capture starts from the same coherent dense and compact seed.
        for mode in modes:
            stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
            for _ in range(2):
                restore(initial)
                operation(mode)
                torch.npu.synchronize()
            restore(initial)
            stream.wait_stream(torch.npu.current_stream())
            dist.barrier(group=cpu_group)
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                graph_outputs[mode] = operation(mode)
            torch.npu.synchronize()
            graphs[mode] = (graph, stream)
        restore(initial)
        histories = dict(official_padding=initial, compact_padding=initial)
        for entry in schedule:
            prepare(entry)
            canonical, canonical_outputs, official_outputs, official_values = None, None, None, None
            for mode in modes:
                before = histories["official_padding" if mode == "official_padding" else "compact_padding"]
                restore(before)
                expected_output = clone_outputs(operation(mode))
                torch.npu.synchronize()
                expected = snapshot()
                restore(before)
                repeated = operation(mode)
                torch.npu.synchronize()
                exact_outputs(repeated, expected_output, f"{mode}/repeat_eager/step{entry['step']}")
                exact_state(expected, f"{mode}/repeat_eager/step{entry['step']}")
                restore(before)
                graph, stream = graphs[mode]
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    graph.replay()
                torch.npu.synchronize()
                exact_outputs(graph_outputs[mode], expected_output, f"{mode}/replay/step{entry['step']}")
                exact_state(expected, f"{mode}/replay/step{entry['step']}")
                unchanged_paused(before, entry["live"], mode)
                finite((*fixture.states, *compact_state.mutable_tensors, *expected_output), mode)
                if mode == "official_padding":
                    histories[mode] = expected
                    official_outputs, official_values = expected_output, expected["dense"]
                elif mode == "compact_padding":
                    canonical, canonical_outputs = expected, expected_output
                    # Compact owns SSM; its dense tensor is only a construction
                    # source/control and must never be silently updated.
                    exact(fixture.states[1], before["dense"][1], "compact/dense_source_untouched")
                    exported = compact_state.export(backend.log_module)
                    torch.npu.synchronize()
                    finite((exported,), "all logical compact candidates")
                    live = entry["live"]
                    exact(fixture.states[0], official_values[0], "official_conv_unchanged_algorithm")
                    for index, name in enumerate(("hidden", "residual")):
                        cross_metrics(
                            expected_output[index][: live * 4].view(live, 4, -1),
                            official_outputs[index][: live * 4].view(live, 4, -1),
                            name,
                            entry["step"],
                        )
                    for token in range(4):
                        physical = torch.tensor([quartet[token] for quartet in candidate_slots[:live]], device="npu")
                        cross_metrics(
                            exported[1 : live + 1, token],
                            official_values[1][physical],
                            f"logical_candidate{token + 1}",
                            entry["step"],
                        )
                else:
                    exact_outputs(expected_output, canonical_outputs, f"{mode}/full_compact_output")
                    exact_state(canonical, f"{mode}/full_compact_state")
                checks.append(
                    dict(
                        mode=mode,
                        step=entry["step"],
                        live=entry["live"],
                        eager_repeat_exact=True,
                        eager_replay_exact=True,
                        inactive_null_exact=True,
                        full_compact_exact=mode not in ("official_padding", "compact_padding"),
                    )
                )
            # All partition trials consume the same generation. Publish only
            # canonical compact once to the next step's persistent history.
            histories["compact_padding"] = canonical
        timing_entry = dict(step=101, live=workload.batch, accepted=schedule[-1]["accepted"])
        prepare(timing_entry)
        torch.npu.reset_peak_memory_stats()
        for round_id in range(args.rounds):
            for mode in modes if round_id % 2 == 0 else reversed(modes):
                before = histories["official_padding" if mode == "official_padding" else "compact_padding"]
                restore(before)
                graph, stream = graphs[mode]
                stream.wait_stream(torch.npu.current_stream())
                dist.barrier(group=cpu_group)
                for sample in range(-3, args.repeats):
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    started = time.perf_counter()
                    with torch.npu.stream(stream):
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                    end.synchronize()
                    if sample >= 0:
                        samples.append(
                            dict(
                                mode=mode,
                                round=round_id,
                                sample=sample,
                                device_us=begin.elapsed_time(end) * 1000,
                                wall_us=(time.perf_counter() - started) * 1e6,
                            )
                        )
                finite((*fixture.states, *compact_state.mutable_tensors, *graph_outputs[mode]), f"timed/{mode}")
        local = dict(
            rank=rank,
            checks=checks,
            diagnostics=diagnostics,
            samples=samples,
            memory=dict(
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                peak_reserved_bytes=torch.npu.max_memory_reserved(),
                compact_state_bytes=sum(
                    tensor.numel() * tensor.element_size() for tensor in compact_state.mutable_tensors
                ),
                retained_dense_state_bytes=sum(tensor.numel() * tensor.element_size() for tensor in fixture.states),
                scope="whole weight-loaded process, all control graphs and correctness histories",
            ),
        )
    ranks = [None] * world
    dist.all_gather_object(ranks, local, group=cpu_group)
    if rank == 0:
        joined_samples = [
            dict(
                mode=row["mode"],
                round=row["round"],
                sample=row["sample"],
                device_us=max(item["samples"][index]["device_us"] for item in ranks),
                wall_us=max(item["samples"][index]["wall_us"] for item in ranks),
            )
            for index, row in enumerate(ranks[0]["samples"])
        ]
        round_medians = {
            mode: [
                statistics.median(
                    row["device_us"] for row in joined_samples if row["mode"] == mode and row["round"] == round_id
                )
                for round_id in range(args.rounds)
            ]
            for mode in modes
        }
        report = dict(
            status="qualified_complete_gdn_layer_not_full_model",
            scope="one_real_gdn_decoder_layer_including_mlp_and_tp_reductions",
            model=args.model,
            selection=args.selection,
            loader=args.loader_metadata,
            layer_index=fixture.layer_index,
            qualification_schedule=schedule,
            ranks=ranks,
            critical_rank_samples=joined_samples,
            round_medians_us=round_medians,
            deferred_pipeline_increment=deferred_pipeline_comparison(round_medians),
            request_tile_increment=tile_pipeline_comparison(round_medians),
            includes=(
                "whole real GDN layer, official conv/gates/L2norm, projections, MLP, "
                "TP reductions, state drains and joins"
            ),
            excludes=(
                "full model, preceding FIA, serving scheduler, draft execution, synthetic input/metadata preparation"
            ),
            preparation=(
                "one-time compact import and all snapshot restores outside timing; "
                "required replay maps/casts/clear inside"
            ),
            inactive_output="dynamic device mask restores zero hidden/residual in every timed control",
            deferred_commit=dict(
                enabled=bool(args.deferred_extension),
                launch="real GDN out_proj forward pre-hook, after gated Norm and before output GEMM",
                finish="after the complete GDN layer and MLP, before output masking and endpoint return",
                control="deferred_serial commits inline; deferred_overlap uses the same staged state representation",
                final_join_timed=True,
            ),
            request_partition=dict(
                boundary=workload.capacity // 2,
                capacity=workload.capacity,
                full_bucket_and_global_buffers=True,
                new_capture_sizes=False,
            ),
            numerical_gate=(
                f"cross algorithm worst-request NRMSE <= {args.max_nrmse}; same-schedule and partition controls exact"
            ),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            libraries={
                name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for name, path in (
                    (("log", args.extension), ("products", args.products_extension))
                    + ((("deferred", args.deferred_extension),) if args.deferred_extension else ())
                    + ((("tile", args.tile_extension),) if args.tile_extension else ())
                )
            },
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            json.dumps(dict(status=report["status"], round_medians_us=report["round_medians_us"]), indent=2), flush=True
        )


def main():
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--deferred-extension", type=Path, help="explicit deferred commit/solve library")
    parser.add_argument("--tile-extension", type=Path, help="request-ready tile library; requires --deferred-extension")
    parser.add_argument("--pair", nargs=2, type=int, default=(200, 200))
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-nrmse", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.rounds < 2 or args.repeats < 3 or not 0 < args.max_nrmse <= 0.005:
        parser.error("new output, rounds >= 2, repeats >= 3 and 0 < max-nrmse <= 0.005 required")
    if args.tile_extension is not None and args.deferred_extension is None:
        parser.error("--tile-extension requires --deferred-extension")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or args.tensor_parallel_size != 2:
        parser.error("launch through torchrun with exactly two ranks")
    args.query, args.dtype, args.batch_invariant, args.compiled = 4, "bfloat16", False, False
    args.gdn_probe, args.gdn_pipeline = True, False
    args.selection = select_workload(args)
    args.pair = args.selection["pair_tokens"]
    batch, capacity = args.selection["batch"], args.selection["capacity_requests"]
    left = min(batch - 1, args.selection["pair_requests"][0])
    workload = ReplayWorkload(batch, capacity, args.query, args.kv_length, (left, batch - left))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_qwen35(args, workload, experiment=experiment)
    except Exception as error:
        rank = int(os.environ.get("RANK", "0"))
        report = dict(
            status="failed",
            scope="complete_real_gdn_layer_not_full_model",
            rank=rank,
            error_type=type(error).__name__,
            error=str(error),
            selection=args.selection,
            loader=getattr(args, "loader_metadata", None),
        )
        text = json.dumps(report, indent=2)
        args.output.with_suffix(f".rank{rank}.failure.json").write_text(text, encoding="utf-8")
        if rank == 0:
            args.output.write_text(text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
