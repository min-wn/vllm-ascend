# SPDX-License-Identifier: Apache-2.0
"""Closed previous-MLP/next-GDN endpoint with a join before next O or core.

Synthetic segment inputs; real TP2 layer weights and state. Optionally prepend
the previous official FIA attention, including per-replay H2D and host updates.
All controls use the public ready-tile recurrence.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

MODES = ("padding", "split_barrier", "split_early")


def experiment_modes(fused):
    return MODES + tuple("fused_" + mode for mode in MODES) if fused else MODES


def validate_fused_lanes(pair, catalog):
    if len(pair) != 2 or any(type(n) is not int or n <= 0 or n % 8 or n not in catalog for n in pair):
        raise ValueError("each fused lane must be a default bucket with whole Q4 requests on both ranks")
    if sum(pair) not in catalog:
        raise ValueError("full fused capacity must belong to the default catalog")


def join_projected_phases(phases, *, cat=None):
    """Reassemble contiguous request lanes, including initialized core output.

    All six copies execute inside the measured graph. No stateful work has run
    in either lane; the returned phase uses full metadata and one transaction.
    """
    from multi_graph_tp_gdn_phases import GDNPhase

    if len(phases) != 2 or tuple(p.name for p in phases) != ("left", "right"):
        raise ValueError("require ordered left/right projected request lanes")
    owner = phases[0].owner
    fields = ("mixed_qkv", "z", "b", "a", "core_output", "residual")
    for phase in phases:
        if phase.owner is not owner or phase.completed or phase.normalized_output is not None:
            raise ValueError("join requires unconsumed projections from one owner")
        rows = phase.core_output.shape[0]
        if any(getattr(phase, field).shape[0] != rows for field in fields):
            raise ValueError("projected tensors disagree on request extent")
    if cat is None:
        import torch

        cat = torch.cat
    merged = GDNPhase(owner, "full", *(cat([getattr(p, field) for p in phases]) for field in fields))
    for phase in phases:
        phase.completed = True
    return merged


def pair_samples(ranks):
    if len(ranks) != 2 or len(ranks[0]) != len(ranks[1]):
        raise ValueError("two equally sized rank sample lists required")
    paired = []
    for first, second in zip(*ranks, strict=True):
        if any(first[key] != second[key] for key in ("mode", "round", "sample")):
            raise ValueError("rank sample identities disagree")
        paired.append(
            dict(
                mode=first["mode"],
                round=first["round"],
                sample=first["sample"],
                wall_us=max(first["wall_us"], second["wall_us"]),
                device_us=max(first["device_us"], second["device_us"]),
            )
        )
    return paired


@contextmanager
def installed_lane_backends(backends):
    import vllm_ascend.ops.gdn as gdn

    original = gdn.recurrent_gated_delta_rule
    selected = ["full"]

    def dispatch(*args, **kwargs):
        return backends[selected[0]](*args, **kwargs)

    gdn.recurrent_gated_delta_rule = dispatch
    try:
        yield selected
        if any(backend._pending_commit is not None for backend in backends.values()):
            raise RuntimeError("endpoint returned with pending state publication")
    finally:
        try:
            for backend in backends.values():
                backend._cleanup_pending_commit()
        finally:
            gdn.recurrent_gated_delta_rule = original


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_log_backend import Q4LogBackend, Q4LogState
    from multi_graph_tp_gdn import qualify_gdn
    from vllm.distributed.parallel_state import get_tp_group, get_world_group

    modes = experiment_modes(getattr(args, "fused_down", False))
    rank = dist.get_rank()
    group, cpu_group = get_tp_group(), get_world_group().cpu_group
    rows, boundary = workload.capacity * 4, args.pair[0]
    previous = model.get_submodule(f"language_model.model.layers.{args.loader_metadata['layer_index']}")
    width = runner.vllm_config.model_config.hf_text_config.hidden_size
    source = [torch.zeros((rows, width), dtype=torch.bfloat16, device="npu") for _ in range(2)]
    data = [torch.empty_like(value) for value in source]
    active = torch.zeros((rows, 1), dtype=torch.bool, device="npu")
    fixture = qualify_gdn(args, workload, runner, model, tuple(data), cpu_group, setup_only=True)
    layer, gdn = fixture.phases.layer, fixture.phases.gdn
    candidate_slots = [[1 + request * 4 + token for token in range(4)] for request in range(workload.capacity)]
    state = Q4LogState(torch, fixture.states[1], candidate_slots, [1] * workload.capacity)
    libraries = {
        "log": load_extension("gdn_q4_log", args.extension),
        "products": load_extension("gdn_q4_products", args.products_extension),
        "deferred": load_extension("gdn_q4_log_deferred", args.deferred_extension),
        "tile": load_extension("gdn_q4_tile_pipeline", args.tile_extension),
    }
    # Each lane has its own scratch/transaction; only disjoint live owners share state.
    backends = {
        name: Q4LogBackend(
            state,
            libraries["log"],
            libraries["products"],
            "deferred_tile_ready",
            deferred_module=libraries["deferred"],
            tile_module=libraries["tile"],
        )
        for name in ("full", "left", "right")
    }
    converted = []
    torch.npu.config.allow_internal_format = True
    for owner in (previous, layer):
        for name, child in owner.named_modules():
            if hasattr(child, "weight") and child.weight is not None and child.weight.ndim == 2:
                child.weight.data = torch_npu.npu_format_cast(child.weight.data, 29)
                converted.append((owner.layer_type, name, torch_npu.get_npu_format(child.weight)))
    if any(record[2] != 29 for record in converted):
        raise ValueError("all participating dense weights must retain NZ")
    if previous.layer_scale or previous.use_attn_reduce_scatter_for_moe:
        raise ValueError("requires an ordinary dense predecessor")

    upstream = None
    if args.include_fia:
        from aclgraph_bounded_lead_fia import FIAPrefix

        upstream = FIAPrefix(workload, runner, model, previous)

    root, comm = torch.npu.Stream(), torch.npu.Stream()
    update_stream = torch.npu.Stream()
    staged, updated = torch.npu.Event(), torch.npu.Event()
    ready = [torch.npu.Event() for _ in range(2)]
    done = [torch.npu.Event() for _ in range(2)]
    graphs, outputs, checks, samples, diagnostics = {}, {}, [], [], []
    report = dict(
        status="running",
        selection=args.selection,
        loader=args.loader_metadata,
        scope="real layer-3 complete MLP followed by real layer-4 complete GDN; synthetic reduced-O inputs",
        exclusions=["preceding FIA and its updates", "full model", "serving", "draft model", "metadata construction"],
        includes=[
            "input refresh copy",
            "all endpoint projections and TP reductions",
            "state commits",
            "all stream joins",
        ],
        modes=modes,
        packed_join=args.packed_join,
        fused_down=args.fused_down,
        join_before=args.join_before,
        include_fia=args.include_fia,
        converted_weights=converted,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend_scratch="private per lane; shared canonical state only for disjoint request owners",
        checks=checks,
        samples=samples,
        numerical_diagnostics=diagnostics,
    )
    if upstream is not None:
        report["scope"] = "complete real FIA layer 3 -> complete real GDN layer 4; synthetic segment inputs/prefix"
        report["exclusions"] = [
            "full model",
            "serving",
            "draft model",
            "CPU metadata construction",
            "state snapshot restore",
        ]
        report["includes"] += [
            "official preceding FIA attention",
            "per-replay true host-length update",
            "FIA metadata H2D",
        ]
        report["fia_update_tasks"] = {}

    def progress(stage):
        if rank == 0:
            print(json.dumps(dict(stage=stage)), flush=True)

    def snapshot():
        torch.npu.synchronize()
        return (
            tuple(value.clone() for value in fixture.states),
            state.snapshot(),
            tuple(value.clone() for value in upstream.caches) if upstream is not None else (),
        )

    def restore(saved):
        fixture.reset(saved[0])
        state.restore(saved[1])
        if upstream is not None:
            for value, source in zip(upstream.caches, saved[2], strict=True):
                value.copy_(source)
        torch.npu.synchronize()

    def tensors(saved):
        return (*saved[0], *saved[1].tensors, *saved[2])

    def assert_values(actual, expected, label):
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=f"{label}/{i}")

    def check_inactive(before, after, live):
        slots = torch.tensor([0] + [s for quartet in candidate_slots[live:] for s in quartet], device="npu")
        owners = torch.tensor([0] + list(range(live + 1, workload.capacity + 1)), device="npu")
        assert_values(tuple(v[slots] for v in after[0]), tuple(v[slots] for v in before[0]), "inactive/dense")
        assert_values(
            tuple(v[owners] for v in after[1].tensors), tuple(v[owners] for v in before[1].tensors), "inactive/compact"
        )
        torch.testing.assert_close(after[0][1], before[0][1], rtol=0, atol=0)
        if upstream is not None:
            upstream.check_unwritten(before[2], after[2])

    def prepare(step, live):
        fixture.refresh(live, accepted=1 + step % 3)
        if upstream is not None:
            upstream.prepare(live, step)
            upstream.stage()
        torch.manual_seed(75100 + step)
        for value in source:
            value.copy_(torch.randn_like(value) * 0.1)
            value[live * 4 :].zero_()
        active.zero_()
        active[: live * 4].fill_(True)
        torch.npu.synchronize()

    hcom = None
    if args.fused_down:
        hcom = group.device_group._get_backend(torch.device("npu")).get_hccl_comm_name(group.rank_in_group)

    def fused_partial(value):
        return torch_npu.npu_mm_reduce_scatter_base(value, previous.mlp.down_proj.weight.t(), hcom, 2, comm_mode="aiv")

    def gather_rows(value):
        result = torch.empty((value.shape[0] * 2, value.shape[1]), dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(result, value, group=group.device_group)
        return result

    def submit(value, lane, fused=False):
        ready[lane].record(torch.npu.current_stream())
        with torch.npu.stream(comm):
            comm.wait_event(ready[lane])
            result = gather_rows(value) if fused else group.all_reduce(value)
            done[lane].record(comm)
        value.record_stream(comm)
        return result

    with installed_lane_backends(backends) as selected:

        def operation(mode, capturing=False):
            fused = mode.startswith("fused_")
            schedule = mode.removeprefix("fused_")
            for destination, origin in zip(data, source, strict=True):
                destination.copy_(origin)
            prior_input = upstream.forward(*data, capturing=capturing) if upstream is not None else data
            normalized, prior = previous.post_attention_layernorm(*prior_input)
            packed, _ = previous.mlp.gate_up_proj(normalized)
            intermediate = previous.mlp.act_fn(packed)
            if schedule == "padding":
                if fused:
                    down = gather_rows(fused_partial(intermediate))
                else:
                    down, _ = previous.mlp.down_proj(intermediate)
                selected[0] = "full"
                phase = fixture.phases.project(down, prior)
                fixture.phases.core(phase)
                backends["full"].launch_pending_commit()
                joined, residual = phase.normalized_output, phase.residual
            else:
                partials = []
                for lane, (start, stop) in enumerate(((0, boundary), (boundary, rows))):
                    value = (
                        fused_partial(intermediate[start:stop])
                        if fused
                        else functional.linear(intermediate[start:stop], previous.mlp.down_proj.weight)
                    )
                    partials.append(submit(value, lane, fused=fused))
                # Both Down GEMMs precede any next-layer GEMM: no Cube/Cube race.
                if schedule == "split_barrier":
                    torch.npu.current_stream().wait_event(done[1])
                phases = []
                for lane, (name, start, stop) in enumerate((("left", 0, boundary), ("right", boundary, rows))):
                    torch.npu.current_stream().wait_event(done[lane])
                    selected[0] = name
                    if args.packed_join:
                        from aclgraph_full_core_join import project_packed

                        phase = project_packed(fixture.phases, partials[lane], prior[start:stop], name)
                    else:
                        phase = fixture.phases.project(partials[lane], prior[start:stop], name)
                    if args.join_before == "o":
                        fixture.phases.core(phase)
                        backends[name].launch_pending_commit()
                    phases.append(phase)
                if args.join_before == "core":
                    if args.packed_join:
                        from aclgraph_full_core_join import join_packed

                        phase = join_packed(phases)
                    else:
                        phase = join_projected_phases(phases)
                    selected[0] = "full"
                    fixture.phases.core(phase)
                    backends["full"].launch_pending_commit()
                    joined, residual = phase.normalized_output, phase.residual
                else:
                    joined = torch.cat([phase.normalized_output for phase in phases])
                    residual = torch.cat([phase.residual for phase in phases])
            # All later projections and collectives retain full M.
            with fixture.phases.context_for_name("full"):
                output, _ = gdn.out_proj(joined)
                output, residual = layer.post_attention_layernorm(output, residual)
                output = layer.mlp(output)
            for backend in backends.values():
                backend.finish_pending_commit()
            return tuple(torch.where(active, value, 0) for value in (output, residual))

        def replay(mode):
            graph = graphs[mode]
            if upstream is None:
                graph.replay()
                return
            tasks = upstream.resolve(graph)
            upstream.stage()
            staged.record(root)
            update_stream.wait_event(staged)
            graph.replay()
            graph.update(update_stream, tasks)
            updated.record(update_stream)
            root.wait_event(updated)

        prepare(0, workload.batch)
        initial = snapshot()
        if upstream is not None:
            # Independent complete official layer calls guard the decomposition.
            with upstream.context():
                prior_output = previous(
                    hidden_states=source[0].clone(), residual=source[1].clone(), positions=upstream.positions
                )
            selected[0] = "full"
            official = fixture.run(*prior_output)
            backends["full"].launch_pending_commit()
            backends["full"].finish_pending_commit()
            official = tuple(torch.where(active, value, 0).clone() for value in official)
            official_state = snapshot()
            restore(initial)
            observed = operation("padding")
            assert_values(observed, official, "two-official-layers/output")
            assert_values(tensors(snapshot()), tensors(official_state), "two-official-layers/state")
            report["official_two_layer_exact"] = True
            restore(initial)
        for mode in modes:
            progress(f"warmup/{mode}")
            for _ in range(2):
                restore(initial)
                with torch.npu.stream(root):
                    root.wait_stream(torch.npu.default_stream())
                    operation(mode)
                torch.npu.synchronize()
            restore(initial)
            root.wait_stream(torch.npu.current_stream())
            if upstream is not None:
                from vllm_ascend.compilation.updatable_graph import UpdatableGraph

                graph = UpdatableGraph()
            else:
                graph = torch.npu.NPUGraph()
            dist.barrier(group=cpu_group)
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=root):
                outputs[mode] = operation(mode, capturing=True)
            torch.npu.synchronize()
            graphs[mode] = graph
            if upstream is not None:
                report["fia_update_tasks"][mode] = len(upstream.resolve(graph))

        histories = {mode: initial for mode in modes}
        for step, live in enumerate((workload.batch, workload.batch - 1, workload.batch + 1)):
            prepare(step, live)
            step_results = {}
            for mode in modes:
                before = histories[mode]
                restore(before)
                with torch.npu.stream(root):
                    root.wait_stream(torch.npu.default_stream())
                    expected = tuple(v.clone() for v in operation(mode))
                wanted = snapshot()
                restore(before)
                root.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(root):
                    replay(mode)
                actual = snapshot()
                assert_values(outputs[mode], expected, f"eager/replay/{mode}/{step}")
                assert_values(tensors(actual), tensors(wanted), f"eager/replay-state/{mode}/{step}")
                check_inactive(before, actual, live)
                if any(not torch.isfinite(value).all().item() for value in (*outputs[mode], *tensors(actual))):
                    raise AssertionError("nonfinite output/state")
                histories[mode] = actual
                step_results[mode] = (tuple(v.clone() for v in outputs[mode]), actual)
                checks.append(dict(mode=mode, step=step, live=live, eager_replay_exact=True, inactive_null_exact=True))
            for prefix in ("", "fused_") if args.fused_down else ("",):
                early, barrier = step_results[prefix + "split_early"], step_results[prefix + "split_barrier"]
                assert_values(early[0], barrier[0], f"{prefix}schedule/output")
                assert_values(tensors(early[1]), tensors(barrier[1]), f"{prefix}schedule/state")
                if prefix:
                    for mode in ("fused_padding", "fused_split_barrier", "fused_split_early"):
                        for field, actual_value, reference in zip(
                            ("hidden", "residual"), step_results[mode][0], step_results["padding"][0], strict=True
                        ):
                            delta = actual_value.float() - reference.float()
                            diagnostics.append(
                                dict(
                                    step=step,
                                    mode=mode,
                                    field=field,
                                    max_abs=delta.abs().max().item(),
                                    nrmse=(delta.square().mean() / reference.float().square().mean().clamp_min(1e-30))
                                    .sqrt()
                                    .item(),
                                    tolerance_failures=(delta.abs() > 0.02 + 0.02 * reference.float().abs())
                                    .sum()
                                    .item(),
                                )
                            )
            for field, observed, reference in zip(
                ("hidden", "residual", "conv", "dense_source", "base", "logs", "valid")
                + (("fia_key", "fia_value") if upstream is not None else ()),
                (*step_results["split_early"][0], *tensors(step_results["split_early"][1])),
                (*step_results["padding"][0], *tensors(step_results["padding"][1])),
                strict=True,
            ):
                delta = observed.float() - reference.float()
                diagnostics.append(
                    dict(
                        step=step,
                        field=field,
                        max_abs=delta.abs().max().item(),
                        nrmse=(delta.square().mean() / reference.float().square().mean().clamp_min(1e-30))
                        .sqrt()
                        .item(),
                        tolerance_failures=(delta.abs() > 0.02 + 0.02 * reference.float().abs()).sum().item(),
                    )
                )
            progress(f"qualified/step{step}")

        prepare(10, workload.batch)
        for trial in range(args.rounds):
            for mode in modes if trial % 2 == 0 else reversed(modes):
                for sample in range(-2, args.repeats):
                    # Identical coherent initial state for every timed repetition.
                    restore(initial)
                    root.wait_stream(torch.npu.current_stream())
                    dist.barrier(group=cpu_group)
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    started = time.perf_counter_ns()
                    with torch.npu.stream(root):
                        begin.record(root)
                        replay(mode)
                        end.record(root)
                    end.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1000
                    if sample >= 0:
                        samples.append(
                            dict(
                                mode=mode,
                                round=trial,
                                sample=sample,
                                wall_us=elapsed,
                                device_us=begin.elapsed_time(end) * 1000,
                            )
                        )
            progress(f"timed/round{trial}")

        report["profiles"] = []
        for mode in modes:
            restore(initial)
            root.wait_stream(torch.npu.current_stream())
            dist.barrier(group=cpu_group)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
            ) as prof:
                # Profiler setup can differ by milliseconds between ranks.
                # Align after setup, before either rank submits a collective.
                dist.barrier(group=cpu_group)
                with torch.npu.stream(root):
                    replay(mode)
                torch.npu.synchronize()
            path = args.output.with_name(f"{args.output.stem}.{mode}.rank{rank}.trace.json")
            prof.export_chrome_trace(str(path))
            report["profiles"].append(
                dict(mode=mode, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            )

    rank_samples = [None, None]
    dist.all_gather_object(rank_samples, samples, group=cpu_group)
    paired = pair_samples(rank_samples)
    report["critical_rank_samples"] = paired
    report["round_medians"] = {
        mode: {
            metric: [
                statistics.median(s[metric] for s in paired if s["mode"] == mode and s["round"] == trial)
                for trial in range(args.rounds)
            ]
            for metric in ("wall_us", "device_us")
        }
        for mode in modes
    }
    report["status"] = "completed_closed_boundary_with_cross_shape_numerical_diagnostics"
    args.output.with_suffix(f".rank{rank}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["round_medians"], indent=2), flush=True)


def main():
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    for name in ("extension", "products-extension", "deferred-extension", "tile-extension"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--join-before", choices=("o", "core"), default="o")
    parser.add_argument("--include-fia", action="store_true")
    parser.add_argument("--packed-join", action="store_true")
    parser.add_argument("--fused-down", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.rounds < 2 or args.repeats < 3:
        parser.error("new output, at least two rounds and three repeats required")
    if args.packed_join and args.join_before != "core":
        parser.error("packed join requires rejoining before core")
    args.tensor_parallel_size, args.pair, args.query = 2, (200, 200), 4
    args.dtype, args.batch_invariant, args.compiled = "bfloat16", False, False
    args.gdn_probe, args.gdn_pipeline = True, False
    args.selection = select_workload(args)
    args.pair = args.selection["pair_tokens"]
    if args.fused_down:
        validate_fused_lanes(args.pair, args.selection["capture_sizes"])
    batch, capacity = args.selection["batch"], args.selection["capacity_requests"]
    if batch + 1 > capacity:
        raise ValueError("qualification growth must stay in the default bucket")
    left = min(batch - 1, args.selection["pair_requests"][0])
    workload = ReplayWorkload(batch, capacity, 4, 512, (left, batch - left))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_qwen35(args, workload, experiment=experiment)
    except Exception as error:
        args.output.with_suffix(f".rank{os.environ.get('RANK', '0')}.failure.json").write_text(
            json.dumps(dict(status="failed", error_type=type(error).__name__, error=str(error)), indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
