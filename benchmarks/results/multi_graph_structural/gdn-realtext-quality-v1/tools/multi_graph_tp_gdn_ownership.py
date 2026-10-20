"""Qualify request-owned GDN shapes and boundary costs, not decoder latency.

The dense probes use actual checkpoint weights and frozen synthetic inputs at
each projection boundary. Recurrence is a separately qualified endpoint. Their
latencies must not be added and described as a measured three-layer speedup.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def merge_component_shards(shards, component_sizes, dim=0):
    """Reconstruct packed components rather than concatenating packed ranks."""
    import torch

    shards, sizes = tuple(shards), tuple(component_sizes)
    if not shards or dim not in (0, 1) or not sizes or any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError("nonempty shards and positive component sizes on dimension 0/1 required")
    shape = shards[0].shape
    if len(shape) != 2 or sum(sizes) != shape[dim] or any(value.shape != shape for value in shards):
        raise ValueError("shard shapes and packed component sizes disagree")
    components = [value.split(sizes, dim=dim) for value in shards]
    return torch.cat(
        [torch.cat([parts[index] for parts in components], dim=dim) for index in range(len(sizes))], dim=dim
    )


def shard_component_rows(weight, component_sizes, rank, world):
    """Select one TP shard independently within each full packed component."""
    import torch

    sizes = tuple(component_sizes)
    if (
        not isinstance(world, int)
        or world < 1
        or not isinstance(rank, int)
        or not 0 <= rank < world
        or not sizes
        or weight.ndim != 2
        or any(not isinstance(n, int) or n <= 0 or n % world for n in sizes)
        or sum(sizes) != weight.shape[0]
    ):
        raise ValueError("full packed rows must divide across valid TP ranks")
    return torch.cat(
        [part[rank * (size // world) : (rank + 1) * (size // world)] for part, size in zip(weight.split(sizes), sizes)],
        dim=0,
    )


def owner_token_rows(capacity, query, rank, world):
    """Stable ownership follows global request identity, not batch position."""
    if (
        any(not isinstance(n, int) or isinstance(n, bool) for n in (capacity, query, rank, world))
        or min(capacity, query, world) < 1
        or capacity % world
        or not 0 <= rank < world
    ):
        raise ValueError("positive divisible capacity/query and a valid rank required")
    return tuple(request * query + token for request in range(rank, capacity, world) for token in range(query))


def canonical_gather_indices(capacity, query, world):
    """Invert rank-major owner output into canonical request/token order."""
    owner_token_rows(capacity, query, 0, world)
    gathered = [token for rank in range(world) for token in owner_token_rows(capacity, query, rank, world)]
    inverse = [0] * len(gathered)
    for index, token in enumerate(gathered):
        inverse[token] = index
    return tuple(inverse)


def load_checkpoint_weights(model_path, layer_index):
    import torch
    from safetensors import safe_open

    root = Path(model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_index}."

    def read(suffix):
        name = prefix + suffix
        with safe_open(root / index[name], framework="pt", device="cpu") as handle:
            result = handle.get_tensor(name)
        if result.dtype != torch.bfloat16:
            raise ValueError(f"expected an unquantized BF16 projection: {name}")
        return result

    qkv, z = read("linear_attn.in_proj_qkv.weight"), read("linear_attn.in_proj_z.weight")
    b, a = read("linear_attn.in_proj_b.weight"), read("linear_attn.in_proj_a.weight")
    gate, up = read("mlp.gate_proj.weight"), read("mlp.up_proj.weight")
    if qkv.shape[0] <= z.shape[0] or (qkv.shape[0] - z.shape[0]) % 2:
        raise ValueError("unsupported Qwen3.5 Q/K/V/Z widths")
    key = (qkv.shape[0] - z.shape[0]) // 2
    return {
        "qkvz": (torch.cat((qkv, z)), (key, key, z.shape[0], z.shape[0])),
        "ba": (torch.cat((b, a)), (b.shape[0], a.shape[0])),
        "output": (read("linear_attn.out_proj.weight"), None),
        "gateup": (torch.cat((gate, up)), (gate.shape[0], up.shape[0])),
        "down": (read("mlp.down_proj.weight"), None),
    }


def capture_entry(operation, root):
    import torch

    with torch.npu.stream(root):
        for _ in range(3):
            operation()
        expected = tuple(value.clone() for value in operation())
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=root):
        outputs = operation()
    with torch.npu.stream(root):
        graph.replay()
    torch.npu.synchronize()
    for actual, wanted in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    return dict(graph=graph, operation=operation, outputs=outputs, root=root)


def time_entries(entries, cpu_group, rounds, repeats):
    import torch
    import torch.distributed as dist

    samples = []
    for round_id in range(rounds):
        names = tuple(entries) if round_id % 2 == 0 else tuple(reversed(entries))
        for name in names:
            entry = entries[name]
            if "prepare_timing" in entry:
                entry["prepare_timing"]()
            with torch.npu.stream(entry["root"]):
                for _ in range(3):
                    entry["graph"].replay()
            torch.npu.synchronize()
            for sample in range(repeats):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                dist.barrier(group=cpu_group)
                started = time.perf_counter()
                with torch.npu.stream(entry["root"]):
                    begin.record()
                    entry["graph"].replay()
                    end.record()
                end.synchronize()
                samples.append(
                    dict(
                        name=name,
                        round=round_id,
                        sample=sample,
                        wall_us=(time.perf_counter() - started) * 1e6,
                        device_us=begin.elapsed_time(end) * 1000,
                    )
                )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples, group=cpu_group)
    worst = []
    for per_rank in zip(*gathered, strict=True):
        keys = {(item["name"], item["round"], item["sample"]) for item in per_rank}
        if len(keys) != 1:
            raise ValueError("rank sample ordering disagrees")
        worst.append(
            {**per_rank[0], **{field: max(item[field] for item in per_rank) for field in ("wall_us", "device_us")}}
        )
    summary = {}
    for name in entries:
        rows = [item for item in worst if item["name"] == name]
        summary[name] = {
            field: statistics.median(
                statistics.median(item[field] for item in rows if item["round"] == round_id)
                for round_id in range(rounds)
            )
            for field in ("wall_us", "device_us")
        }
    return dict(summary=summary, slowest_rank_samples=worst, per_rank_samples=gathered)


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module):
    # All stage boundaries are frozen. No synthetic connection between QKVZ
    # and O is represented as a complete GDN forward.
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from multi_graph_numerical_assessment import error_metrics
    from vllm.distributed.parallel_state import get_tp_group, get_world_group

    group, cpu_group = get_tp_group(), get_world_group().cpu_group
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2 or workload.query != 4 or workload.capacity * workload.query != 400:
        raise ValueError("qualifier currently requires TP2, Q4 and the default 400-token bucket")
    layer_index = args.loader_metadata["layer_index"] + 1
    layer = model.get_submodule(f"language_model.model.layers.{layer_index}")
    if layer.linear_attn.gqa_interleaved_layout or layer.input_layernorm.weight.numel() != 5120:
        raise ValueError("only the qualified noninterleaved, hidden-size-5120 Qwen3.5 layout is supported")
    if args.compute_overlap_only:
        from multi_graph_tp_gdn_compute_overlap import measure_compute_overlap

        overlap = measure_compute_overlap(args, workload, runner, model, group, cpu_group)
        report = dict(
            status=overlap["status"],
            compute_overlap=overlap,
            selection=args.selection,
            model=args.model,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(overlap["timing"]["summary"], indent=2), flush=True)
        return
    if args.mlp_only:
        from multi_graph_tp_gdn_owner_mlp import measure_owner_mlp

        mlp_report = measure_owner_mlp(args, runner, model, group, cpu_group, args.selection)
        report = dict(
            status=mlp_report["status"],
            mlp=mlp_report,
            selection=args.selection,
            model=args.model,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(mlp_report["timing"]["summary"], indent=2), flush=True)
        return
    if args.boundary_only:
        from multi_graph_tp_gdn_owner_boundary import measure_owner_boundary

        boundary_report = measure_owner_boundary(args, group, cpu_group, workload.capacity, workload.query)
        report = dict(
            status=boundary_report["status"],
            boundary=boundary_report,
            selection=args.selection,
            model=args.model,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(boundary_report["timing"]["summary"], indent=2), flush=True)
        return
    if args.core_only:
        from multi_graph_tp_gdn import qualify_gdn
        from multi_graph_tp_gdn_owner_core import measure_owner_core

        torch.manual_seed(731)
        source = tuple(torch.randn((400, 5120), device="npu", dtype=torch.bfloat16) * 0.05 for _ in range(2))
        adjacent = qualify_gdn(args, workload, runner, model, source, cpu_group, setup_only=True)
        phase = adjacent.phases.project(*source, "full")
        core = measure_owner_core(args, workload, runner, model, phase, cpu_group)
        report = dict(
            status=core["status"],
            scope="actual_core_layout_qualification_only",
            core=core,
            selection=args.selection,
            model=args.model,
            loader=args.loader_metadata,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return
    checkpoint = load_checkpoint_weights(args.model, layer_index)
    main, root = torch.npu.current_stream(), torch.npu.Stream()
    root.wait_stream(main)
    torch.npu.config.allow_internal_format = True
    owner_rows = torch.tensor(owner_token_rows(workload.capacity, 4, rank, world), device="npu")
    inverse = torch.tensor(canonical_gather_indices(workload.capacity, 4, world), device="npu")
    weights, buffers, descriptions = {}, {}, {}
    for name, (full_cpu, components) in checkpoint.items():
        tp_cpu = (
            shard_component_rows(full_cpu, components, rank, world)
            if components
            else full_cpu.chunk(world, dim=1)[rank].contiguous()
        )
        width = full_cpu.shape[1]
        full_input = torch.empty((400, width), device="npu", dtype=torch.bfloat16)
        tp_input = torch.empty((400, tp_cpu.shape[1]), device="npu", dtype=torch.bfloat16)
        owner_input = torch.empty((200, width), device="npu", dtype=torch.bfloat16)
        buffers[name] = dict(full=full_input, tp=tp_input, owner=owner_input, components=components)
        descriptions[name] = dict(
            full_weight=list(full_cpu.shape),
            tp_weight=list(tp_cpu.shape),
            components=components,
            tp_input=list(tp_input.shape),
            owner_input=list(owner_input.shape),
        )
        for label, tensor in (("tp", tp_cpu), ("owner", full_cpu)):
            dense_weight = tensor.to("npu")
            weights[name, label, "ND"] = torch_npu.npu_format_cast(dense_weight, 2)
            weights[name, label, "NZ"] = torch_npu.npu_format_cast(dense_weight, 29)
            if torch_npu.get_npu_format(weights[name, label, "NZ"]) != 29:
                raise ValueError("NZ conversion did not take effect")

    hidden = torch.empty((400, 5120), device="npu", dtype=torch.bfloat16)
    residual = torch.empty_like(hidden)
    hidden_local, residual_local = torch.empty_like(hidden[:200]), torch.empty_like(hidden[:200])
    # HCCL's qualified group reduces in place. Zero payloads remain stable on
    # repeated graph replay without charging artificial reset copies to AR.
    reduce_inputs = tuple(torch.zeros_like(hidden) for _ in range(6))

    def refresh(live, variant):
        for index, (_, item) in enumerate(buffers.items()):
            torch.manual_seed(1900 + variant * 17 + index)
            item["full"].copy_(torch.randn_like(item["full"]) * 0.05)
            item["full"][live * 4 :].zero_()
            value = item["full"]
            item["tp"].copy_(value if item["components"] else value.chunk(world, dim=1)[rank])
            item["owner"].copy_(value.index_select(0, owner_rows))
        # Integer-exact sentinels give communication an independent oracle.
        rows = torch.arange(400, device="npu", dtype=torch.int32).unsqueeze(1)
        hidden.copy_((rows % 31).expand_as(hidden))
        residual.copy_((-(rows % 29) - 3).expand_as(residual))
        hidden[live * 4 :].zero_()
        residual[live * 4 :].zero_()
        hidden_local.copy_(hidden.index_select(0, owner_rows))
        residual_local.copy_(residual.index_select(0, owner_rows))
        torch.npu.synchronize()

    refresh(98, 0)
    entries, checks = {}, []
    for fmt in args.formats:
        for label in ("tp", "owner"):
            for name in (*checkpoint, "bundle"):
                selected = tuple(checkpoint) if name == "bundle" else (name,)

                def operation(selected=selected, label=label, fmt=fmt):
                    return tuple(
                        functional.linear(buffers[item][label], weights[item, label, fmt]) for item in selected
                    )

                key = f"dense_{name}_{fmt}_{label}"
                entries[key] = capture_entry(operation, root)
                checks.append(f"{key}/eager_replay")

    def owner_exit():
        gathered = (group.all_gather(hidden_local, dim=0), group.all_gather(residual_local, dim=0))
        return tuple(value.index_select(0, inverse) for value in gathered)

    def owner_entry():
        return hidden.index_select(0, owner_rows), residual.index_select(0, owner_rows)

    def tp_reductions():
        return tuple(group.all_reduce(value) for value in reduce_inputs)

    for name, operation in (
        ("owner_entry_select", owner_entry),
        ("owner_exit_gather_reorder", owner_exit),
        ("tp_six_ar", tp_reductions),
    ):
        entries[name] = capture_entry(operation, root)
        checks.append(f"{name}/eager_replay")
    expected_hidden, expected_residual = owner_exit()
    torch.testing.assert_close(expected_hidden, hidden, rtol=0, atol=0)
    torch.testing.assert_close(expected_residual, residual, rtol=0, atol=0)
    torch.testing.assert_close(group.all_reduce(hidden + rank), hidden * 2 + 1, rtol=0, atol=0)
    checks.extend(("owner_exit/independent_hidden", "owner_exit/independent_residual", "tp_ar/independent_sum"))

    diagnostics = []
    for fmt in args.formats:
        for name, (_, components) in checkpoint.items():
            tp_value = entries[f"dense_{name}_{fmt}_tp"]["outputs"][0]
            if components:
                combined = group.all_gather(tp_value, dim=-1)
                shards = combined.chunk(world, dim=-1)
                reference = merge_component_shards(shards, [n // world for n in components], dim=1)
            else:
                reference = group.all_reduce(tp_value)
            owner_value = entries[f"dense_{name}_{fmt}_owner"]["outputs"][0]
            actual = group.all_gather(owner_value, dim=0).index_select(0, inverse)
            metrics = error_metrics(actual, reference)
            if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise ValueError(f"nonfinite dense output: {name}/{fmt}")
            diagnostics.append(dict(stage=name, format=fmt, **metrics))

    for variant, live in enumerate((98, 97, 99, 98), start=1):
        refresh(live, variant)
        for name, entry in entries.items():
            with torch.npu.stream(root):
                expected = tuple(value.clone() for value in entry["operation"]())
                entry["graph"].replay()
            torch.npu.synchronize()
            for actual, wanted in zip(entry["outputs"], expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            checks.append(f"{name}/dynamic_{live}_{variant}")
        torch.testing.assert_close(entries["owner_exit_gather_reorder"]["outputs"][0], hidden, rtol=0, atol=0)
        torch.testing.assert_close(entries["owner_exit_gather_reorder"]["outputs"][1], residual, rtol=0, atol=0)
    refresh(98, 0)
    timing = time_entries(entries, cpu_group, args.rounds, args.repeats)
    # A mixed-format bundle uses measured per-stage choices on both sides.
    # It is timed again as one graph; sums of isolated kernels are not latency.
    mixed_formats, mixed_entries = {}, {}
    for label in ("tp", "owner"):
        mixed_formats[label] = {
            name: min(args.formats, key=lambda fmt: timing["summary"][f"dense_{name}_{fmt}_{label}"]["device_us"])
            for name in checkpoint
        }

        def mixed_operation(label=label):
            return tuple(
                functional.linear(buffers[name][label], weights[name, label, mixed_formats[label][name]])
                for name in checkpoint
            )

        mixed_entries[f"dense_bundle_mixed_{label}"] = capture_entry(mixed_operation, root)
    for variant, live in enumerate((98, 97, 99, 98), start=1):
        refresh(live, variant)
        for name, entry in mixed_entries.items():
            with torch.npu.stream(root):
                expected = tuple(value.clone() for value in entry["operation"]())
                entry["graph"].replay()
            torch.npu.synchronize()
            for actual, wanted in zip(entry["outputs"], expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            checks.append(f"{name}/dynamic_{live}_{variant}")
    refresh(98, 0)
    mixed_timing = time_entries(mixed_entries, cpu_group, args.rounds, args.repeats)
    core = None
    if not args.skip_core:
        from multi_graph_tp_gdn import qualify_gdn
        from multi_graph_tp_gdn_owner_core import measure_owner_core

        torch.manual_seed(731)
        source = tuple(torch.randn_like(hidden) * 0.05 for _ in range(2))
        adjacent = qualify_gdn(args, workload, runner, model, source, cpu_group, setup_only=True)
        phase = adjacent.phases.project(*source, "full")
        core = measure_owner_core(args, workload, runner, model, phase, cpu_group)
    local = dict(rank=rank, checks=checks, diagnostics=diagnostics)
    ranks = [None] * world
    dist.all_gather_object(ranks, local, group=cpu_group)
    report = dict(
        status="qualified_frozen_stage_costs_not_three_layer_latency",
        scope=__doc__,
        model=args.model,
        selection=args.selection,
        layer_index=layer_index,
        formats=args.formats,
        loader=args.loader_metadata,
        inputs="frozen synthetic projection-boundary inputs; checkpoint weights; no connected GDN dense bundle",
        excluded=(
            "inter-stage recurrence, activation/Norm and weight variation across three layers; see separate core"
        ),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shapes=descriptions,
        rank_checks=ranks,
        timing=timing,
        measured_mixed_format_bundle=dict(formats=mixed_formats, **mixed_timing),
        core=core,
        ownership="request_id modulo TP size; canonical output permutation and both residual endpoints",
        isolated_ar_payload="six zero tensors for stable in-place replay; nonzero sentinel oracle outside timing",
        steady_state_excludes=(
            "weight loading/NZ conversion; frozen-stage input selection; separately timed owner_entry_select"
        ),
    )
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(dict(summary=timing["summary"], core=core), indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--query", type=int, default=4)
    parser.add_argument("--pair", type=int, nargs=2, default=(200, 200))
    parser.add_argument("--formats", nargs="+", choices=("ND", "NZ"), default=("ND", "NZ"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--compute-overlap-only", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or min(args.rounds, args.repeats) < 1
        or len(set(args.formats)) != len(args.formats)
        or (args.skip_core and args.core_only)
        or (args.boundary_only and (args.skip_core or args.core_only))
        or (args.mlp_only and (args.skip_core or args.core_only or args.boundary_only))
        or (args.compute_overlap_only and (args.skip_core or args.core_only or args.boundary_only or args.mlp_only))
    ):
        parser.error("new output path, unique formats and positive counts required")
    args.gdn_pipeline, args.dtype, args.batch_invariant, args.kv_length = True, "bfloat16", False, 512
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    args.selection = select_workload(args)
    capacity, batch = args.selection["capacity_requests"], args.selection["batch"]
    workload = ReplayWorkload(
        batch, capacity, args.query, 512, (min(batch, capacity // 2), max(0, batch - capacity // 2))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_qwen35(args, workload, experiment=experiment)
    except Exception as error:
        failure = dict(
            status="failed",
            rank=int(os.environ.get("RANK", 0)),
            error_type=type(error).__name__,
            error=str(error),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        args.output.with_suffix(f".rank{failure['rank']}.failure.json").write_text(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
