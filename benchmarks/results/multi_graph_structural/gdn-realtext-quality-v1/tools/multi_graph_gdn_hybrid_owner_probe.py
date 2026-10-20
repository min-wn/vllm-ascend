# SPDX-License-Identifier: Apache-2.0
"""Screen GDN request ownership with efficient full-row tensor-parallel MLPs.

The GDN recurrence is replaced by a frozen, independently supplied core output.
This is a dense/communication budget gate, never a connected layer/model result.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def audit_owner_capacities(trace, catalog):
    """Account for fixed-ID ownership imbalance using only existing buckets."""
    sizes = sorted({value for value in catalog if isinstance(value, int) and value > 0 and value % 4 == 0})
    if not sizes:
        raise ValueError("a nonempty default Q4 capture catalog is required")
    rows = []
    for step in trace["steps"]:
        requests = step["request_ids"]
        if len(set(requests)) != len(requests) or len(requests) != step["batch"] or not requests:
            raise ValueError("step request IDs must be unique and match a positive batch")
        if any(isinstance(request, bool) or not isinstance(request, int) or request < 0 for request in requests):
            raise ValueError("canonical nonnegative request IDs required")
        counts = [sum(request % 2 == rank for request in requests) for rank in (0, 1)]

        def bucket(tokens):
            return next((value for value in sizes if value >= max(4, tokens)), None)

        padding = bucket(4 * step["batch"])
        individual = [bucket(4 * count) for count in counts]
        if padding is None or None in individual:
            raise ValueError("default capture catalog does not cover this owner workload")
        equal = max(individual)
        if equal * 2 not in sizes:
            raise ValueError("equal-owner reconstructed MLP rows must also be a legal default bucket")
        rows.append(
            dict(
                step=step["step"],
                batch=step["batch"],
                owner_live_requests=counts,
                padding_tokens=padding,
                individually_rounded_owner_tokens=individual,
                equal_owner_tokens=equal,
                hybrid_mlp_tokens=2 * equal,
                work_capacity_ratio=2 * equal / padding,
                same_total_unequal_pair_exists=any(
                    left >= 4 * counts[0] and padding - left in sizes and padding - left >= 4 * counts[1]
                    for left in sizes
                ),
            )
        )
    if not rows:
        raise ValueError("nonempty immutable trajectory required")
    padding_total = sum(row["padding_tokens"] for row in rows)
    hybrid_total = sum(row["hybrid_mlp_tokens"] for row in rows)
    return dict(
        ownership="global_request_id modulo two; no state migration on row compaction",
        protocol="equal-count AllGather/ReduceScatter; step selects the larger legal owner bucket on both ranks",
        total_padding_capacity_tokens=padding_total,
        total_hybrid_capacity_tokens=hybrid_total,
        total_capacity_increase_percent=100 * (hybrid_total / padding_total - 1),
        fixed_maximum_capacity_increase_percent=100
        * (len(rows) * max(row["hybrid_mlp_tokens"] for row in rows) / padding_total - 1),
        required_owner_tokens=sorted({row["equal_owner_tokens"] for row in rows}),
        worst_capacity_step=max(rows, key=lambda row: row["work_capacity_ratio"]),
        same_total_unequal_pair_steps=sum(row["same_total_unequal_pair_exists"] for row in rows),
        steps=rows,
    )


def owner_row_plan(requests, padding_tokens, owner_tokens, rank):
    """Describe rank-major owner storage without changing logical row order."""
    if rank not in (0, 1) or owner_tokens < 4 or owner_tokens % 4 or padding_tokens < len(requests) * 4:
        raise ValueError("valid TP2 rank and sufficient whole-request capacities required")
    if len(set(requests)) != len(requests):
        raise ValueError("duplicate logical owners are not allowed")
    counts = [sum(request % 2 == owner for request in requests) for owner in (0, 1)]
    if max(counts) * 4 > owner_tokens:
        raise ValueError("equal owner capacity cannot hold this request assignment")
    return dict(
        local_token_indices=tuple(
            index * 4 + token for index, request in enumerate(requests) if request % 2 == rank for token in range(4)
        ),
        gather_active_token_indices=tuple(
            owner * owner_tokens + offset for owner in (0, 1) for offset in range(counts[owner] * 4)
        ),
        canonical_active_token_indices=tuple(
            index * 4 + token
            for owner in (0, 1)
            for index, request in enumerate(requests)
            if request % 2 == owner
            for token in range(4)
        ),
    )


def load_audit_inputs(args):
    from multi_graph_dynamic_trace import trace_digest

    manifest = json.loads(args.trace_manifest.read_text(encoding="utf-8"))
    digest = trace_digest(manifest["trace"])
    expected = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
    if digest != expected or manifest["trace_sha256"] != digest:
        raise ValueError("refuse a changed structural dynamic trajectory")
    catalog = json.loads(args.catalog_report.read_text(encoding="utf-8"))["selection"]["capture_sizes"]
    return (
        manifest["trace"],
        catalog,
        dict(
            trace_sha256=digest,
            catalog_report=str(args.catalog_report),
            catalog_report_sha256=hashlib.sha256(args.catalog_report.read_bytes()).hexdigest(),
            capture_sizes=catalog,
        ),
    )


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from multi_graph_numerical_assessment import error_metrics
    from multi_graph_tp_gdn_ownership import (
        capture_entry,
        load_checkpoint_weights,
        merge_component_shards,
        shard_component_rows,
        time_entries,
    )
    from vllm.distributed.parallel_state import get_tp_group, get_world_group

    group, cpu_group = get_tp_group(), get_world_group().cpu_group
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2 or workload.query != 4 or args.selection["capture_sizes"] != args.audit_sources["capture_sizes"]:
        raise ValueError("TP2/Q4 and unchanged installed default capture catalog required")
    layer_index = args.loader_metadata["layer_index"] + 1
    layer = model.get_submodule(f"language_model.model.layers.{layer_index}")
    if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe or layer.linear_attn.gqa_interleaved_layout:
        raise ValueError("ordinary unscaled dense/noninterleaved Qwen3.5 GDN required")
    checkpoint = load_checkpoint_weights(args.model, layer_index)
    weights = {}
    torch.npu.config.allow_internal_format = True
    for name, (full_cpu, components) in checkpoint.items():
        tp_cpu = (
            shard_component_rows(full_cpu, components, rank, world)
            if components
            else full_cpu.chunk(world, dim=1)[rank].contiguous()
        )
        weights[name, "tp"] = torch_npu.npu_format_cast(tp_cpu.to("npu"), 29)
        if name in ("qkvz", "ba", "output"):
            weights[name, "owner"] = torch_npu.npu_format_cast(full_cpu.to("npu"), 29)
    if any(torch_npu.get_npu_format(value) != 29 for value in weights.values()):
        raise ValueError("all timed TP/owner weights must retain NZ format")
    hidden_size, head_width = checkpoint["output"][0].shape
    if hidden_size != 5120 or head_width != 6144:
        raise ValueError("the budget gate requires the actual 27B GDN dimensions")
    root = torch.npu.Stream()
    root.wait_stream(torch.npu.current_stream())

    def measure_profile(selected_step):
        step = args.trace["steps"][selected_step]
        capacity = args.capacity_audit["steps"][selected_step]
        total_rows, owner_rows = capacity["padding_tokens"], capacity["equal_owner_tokens"]
        live_rows = 4 * step["batch"]
        plan = owner_row_plan(step["request_ids"], total_rows, owner_rows, rank)
        local_indices = plan["local_token_indices"]
        local_index = torch.tensor(local_indices, dtype=torch.int64, device="npu")
        packed_to_canonical = torch.tensor(plan["canonical_active_token_indices"], dtype=torch.int64, device="npu")
        gather_rows = torch.tensor(
            plan["gather_active_token_indices"],
            dtype=torch.int64,
            device="npu",
        )
        source = torch.empty((total_rows, hidden_size), dtype=torch.bfloat16, device="npu")
        prior = torch.empty_like(source)
        frozen_core = torch.empty((total_rows, head_width), dtype=source.dtype, device=source.device)
        local_source = torch.empty((owner_rows, hidden_size), dtype=source.dtype, device=source.device)
        local_prior = torch.empty_like(local_source)
        local_core = torch.empty((owner_rows, head_width), dtype=source.dtype, device=source.device)
        tp_core = frozen_core[:, rank * (head_width // 2) : (rank + 1) * (head_width // 2)].contiguous()

        def refresh(variant):
            generator = torch.Generator(device="cpu").manual_seed(step["input_seed"] + variant)
            for value in (source, prior, frozen_core):
                value.copy_(torch.randn(value.shape, generator=generator, dtype=torch.float32).to(value.dtype) * 0.05)
                value[live_rows:].zero_()
            tp_core.copy_(frozen_core[:, rank * (head_width // 2) : (rank + 1) * (head_width // 2)])
            for local, full in ((local_source, source), (local_prior, prior), (local_core, frozen_core)):
                local.zero_()
                local[: len(local_indices)].copy_(full.index_select(0, local_index))
            torch.npu.synchronize()

        def dense_bundle(label, *, complete, reduce_scatter=True):
            value, residual = layer.input_layernorm(
                source if label == "tp" else local_source, prior if label == "tp" else local_prior
            )
            qkvz = functional.linear(value, weights["qkvz", label])
            ba = functional.linear(value, weights["ba", label])
            # The actual recurrence is intentionally a frozen cut, not skipped
            # from a purported forward. Both sides time every dense projection.
            projected = functional.linear(tp_core if label == "tp" else local_core, weights["output", label])
            if label == "tp":
                projected = group.all_reduce(projected)
            value, residual = layer.post_attention_layernorm(projected, residual)
            if label == "owner":
                value = group.all_gather(value, dim=0)
            activated = layer.mlp.act_fn(functional.linear(value, weights["gateup", "tp"]))
            down = functional.linear(activated, weights["down", "tp"])
            if label == "tp":
                hidden = group.all_reduce(down)
                if not complete:
                    selected_hidden, selected_residual = torch.zeros_like(local_source), torch.zeros_like(local_prior)
                    selected_hidden[: len(local_indices)].copy_(hidden.index_select(0, local_index))
                    selected_residual[: len(local_indices)].copy_(residual.index_select(0, local_index))
                    hidden, residual = selected_hidden, selected_residual
            else:
                hidden = (
                    group.reduce_scatter(down, dim=0)
                    if reduce_scatter
                    else group.all_reduce(down)[rank * owner_rows : (rank + 1) * owner_rows]
                )
                if complete:
                    packed = group.all_gather(torch.cat((hidden, residual), dim=-1), dim=0)
                    active = packed.index_select(0, gather_rows)
                    canonical = torch.zeros((total_rows, hidden_size * 2), dtype=packed.dtype, device=packed.device)
                    canonical.index_copy_(0, packed_to_canonical, active)
                    hidden, residual = canonical[:, :hidden_size].contiguous(), canonical[:, hidden_size:].contiguous()
            return hidden, residual, qkvz, ba

        def ownership_roundtrip():
            selected = []
            for full in (source, prior):
                owned = torch.zeros_like(local_source)
                owned[: len(local_indices)].copy_(full.index_select(0, local_index))
                selected.append(owned)
            packed = group.all_gather(torch.cat(selected, dim=-1), dim=0)
            canonical = torch.zeros((total_rows, hidden_size * 2), dtype=packed.dtype, device=packed.device)
            canonical.index_copy_(0, packed_to_canonical, packed.index_select(0, gather_rows))
            return canonical[:, :hidden_size].contiguous(), canonical[:, hidden_size:].contiguous()

        operations = {
            "tp_complete": lambda: dense_bundle("tp", complete=True),
            "tp_owner_endpoint": lambda: dense_bundle("tp", complete=False),
            "hybrid_owner_endpoint": lambda: dense_bundle("owner", complete=False),
            "hybrid_owner_ar_slice": lambda: dense_bundle("owner", complete=False, reduce_scatter=False),
            "hybrid_complete": lambda: dense_bundle("owner", complete=True),
            "ownership_roundtrip": ownership_roundtrip,
        }
        refresh(0)
        entries = {}
        for name, operation in operations.items():
            print(f"rank={rank} hybrid dense gate capture step={selected_step} mode={name}", flush=True)
            entries[name] = capture_entry(operation, root)
        checks, numerical = [], []
        args.partial_evidence = dict(step=selected_step, capacity=capacity, checks=checks, numerical=numerical)
        for variant in (1, 2):
            refresh(variant)
            for name, entry in entries.items():
                with torch.npu.stream(root):
                    eager = tuple(value.clone() for value in entry["operation"]())
                    entry["graph"].replay()
                torch.npu.synchronize()
                for actual, expected in zip(entry["outputs"], eager, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if not torch.isfinite(actual).all().item():
                        raise ValueError(f"nonfinite dense/communication output: {name}")
                active_rows = (
                    live_rows
                    if name in ("tp_complete", "hybrid_complete", "ownership_roundtrip")
                    else len(local_indices)
                )
                for actual in entry["outputs"][:2]:
                    torch.testing.assert_close(
                        actual[active_rows:], torch.zeros_like(actual[active_rows:]), rtol=0, atol=0
                    )
                checks.append(f"{name}/variant{variant}/strict_eager_replay_finite")
            for actual, expected in zip(entries["ownership_roundtrip"]["outputs"], (source, prior), strict=True):
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
            checks.append(f"ownership_roundtrip/variant{variant}/independent_original_row_oracle")
            for candidate, baseline in (
                ("hybrid_complete", "tp_complete"),
                ("hybrid_owner_endpoint", "tp_owner_endpoint"),
                ("hybrid_owner_ar_slice", "hybrid_owner_endpoint"),
            ):
                for field, actual, expected in zip(
                    ("hidden", "residual"), entries[candidate]["outputs"][:2], entries[baseline]["outputs"][:2]
                ):
                    numerical.append(
                        dict(
                            candidate=candidate,
                            baseline=baseline,
                            field=field,
                            variant=variant,
                            **error_metrics(actual, expected),
                        )
                    )
                    exact = candidate == "hybrid_owner_ar_slice"
                    torch.testing.assert_close(actual, expected, rtol=0 if exact else 0.02, atol=0 if exact else 0.02)
                    checks.append(f"{candidate}/{field}/variant{variant}/cross_control_gate")
            # Packed TP outputs require reconstruction inside each component.
            # Blind rank concatenation would put Q/K/V/Z in the wrong order.
            for offset, projection in ((2, "qkvz"), (3, "ba")):
                tp_output = entries["tp_complete"]["outputs"][offset]
                full = group.all_gather(tp_output, dim=-1)
                shards = full.chunk(world, dim=-1)
                components = tuple(size // world for size in checkpoint[projection][1])
                canonical_projection = merge_component_shards(shards, components, dim=1)
                oracle = canonical_projection.index_select(0, local_index)
                observed = entries["hybrid_owner_endpoint"]["outputs"][offset][: len(local_indices)]
                numerical.append(dict(projection=projection, variant=variant, **error_metrics(observed, oracle)))
                torch.testing.assert_close(observed, oracle, rtol=0.02, atol=0.02)
                checks.append(f"{projection}/variant{variant}/component_aware_owner_oracle")
        refresh(0)
        timing = time_entries(entries, cpu_group, args.rounds, args.repeats)
        ranks = [None] * world
        dist.all_gather_object(ranks, dict(rank=rank, checks=checks, numerical=numerical), group=cpu_group)
        summary = timing["summary"]
        print(f"rank={rank} hybrid dense/communication gate step={selected_step} completed", flush=True)
        return dict(
            capacity=capacity,
            timing=timing,
            ranks=ranks,
            interior_bundle_saving_us=summary["tp_owner_endpoint"]["device_us"]
            - summary["hybrid_owner_endpoint"]["device_us"],
            gross_interior_budget_us=summary["tp_complete"]["device_us"]
            - summary["hybrid_owner_endpoint"]["device_us"],
            complete_bundle_saving_us=summary["tp_complete"]["device_us"] - summary["hybrid_complete"]["device_us"],
            three_layer_screen_only_us=3
            * (summary["tp_complete"]["device_us"] - summary["hybrid_owner_endpoint"]["device_us"])
            - summary["ownership_roundtrip"]["device_us"],
            shapes=dict(
                tp_gateup=[total_rows, 5120, 17408],
                hybrid_gateup=[2 * owner_rows, 5120, 17408],
                tp_down=[total_rows, 8704, 5120],
                hybrid_down=[2 * owner_rows, 8704, 5120],
            ),
        )

    profiles = [measure_profile(selected_step) for selected_step in args.steps]
    report = dict(
        status="dense_communication_screen_not_connected_gdn",
        scope="Actual inputNorm/QKVZ/BA and frozen core-output/O/postNorm/TP-MLP/communication; recurrence absent",
        limitation=(
            "No split-parallel performance qualification. Frozen core breaks the QKV-to-O dependency; "
            "current public request-tile kernel is TP2 24-head only and needs equally optimized owner qualification."
        ),
        capacity_audit=args.capacity_audit,
        sources=args.audit_sources,
        loader=args.loader_metadata,
        layer_index=layer_index,
        profiles=profiles,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(
            Path(__file__).with_name("multi_graph_tp_gdn_ownership.py").read_bytes()
        ).hexdigest(),
    )
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / "results" / "multi_graph_structural"
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--trace-manifest", type=Path, default=root / "structural-dynamic-trace-v1.json")
    parser.add_argument("--catalog-report", type=Path, default=root / "gdn-q4-full-allocation-v1.rank0.json")
    parser.add_argument("--capacity-audit-only", action="store_true")
    parser.add_argument("--steps", type=int, nargs="+", default=(0, 53))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.repeats) < 1 or len(set(args.steps)) != len(args.steps):
        parser.error("new output path, unique steps and positive timing counts required")
    args.trace, catalog, args.audit_sources = load_audit_inputs(args)
    if any(not 0 <= step < len(args.trace["steps"]) for step in args.steps):
        parser.error("selected step must belong to the fixed trace")
    args.capacity_audit = audit_owner_capacities(args.trace, catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.capacity_audit_only:
        args.output.write_text(json.dumps(dict(sources=args.audit_sources, audit=args.capacity_audit), indent=2) + "\n")
        return
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    args.tensor_parallel_size, args.query, args.pair = 2, 4, (200, 200)
    args.gdn_pipeline, args.dtype, args.batch_invariant, args.kv_length = True, "bfloat16", False, 512
    args.selection = select_workload(args)
    capacity, batch = args.selection["capacity_requests"], args.selection["batch"]
    workload = ReplayWorkload(batch, capacity, 4, 512, (min(batch, capacity // 2), max(0, batch - capacity // 2)))
    try:
        run_qwen35(args, workload, experiment=experiment)
    except Exception as error:
        failure = dict(
            status="failed",
            rank=int(os.environ.get("RANK", 0)),
            error_type=type(error).__name__,
            error=str(error),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            partial_evidence=getattr(args, "partial_evidence", None),
        )
        args.output.with_suffix(f".rank{failure['rank']}.failure.json").write_text(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
