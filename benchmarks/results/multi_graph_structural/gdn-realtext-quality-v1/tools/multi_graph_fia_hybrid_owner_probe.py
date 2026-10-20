# SPDX-License-Identifier: Apache-2.0
"""Screen permanent FIA request ownership without running an attention kernel.

The O input is an independent frozen gated-attention output. Real QKV/O and
TP MLP weights are timed with complete communication, not as a layer forward.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from multi_graph_gdn_hybrid_owner_probe import audit_owner_capacities, load_audit_inputs, owner_row_plan


def pack_fia_weights(q_gate, key, value, output, gate, up, down, *, heads, kv_heads, head_dim):
    """Preserve whole per-head Q/gate pairs inside the first packed component."""
    import torch

    hidden = q_gate.shape[1] if q_gate.ndim == 2 else 0
    expected = ((2 * heads * head_dim, hidden), (kv_heads * head_dim, hidden), (kv_heads * head_dim, hidden))
    if (
        min(heads, kv_heads, head_dim, hidden) < 1
        or heads % kv_heads
        or any(tuple(tensor.shape) != shape for tensor, shape in zip((q_gate, key, value), expected, strict=True))
        or tuple(output.shape) != (hidden, heads * head_dim)
        or gate.ndim != 2
        or gate.shape != up.shape
        or gate.shape[1] != hidden
        or tuple(down.shape) != (hidden, gate.shape[0])
        or any(tensor.dtype != torch.bfloat16 for tensor in (q_gate, key, value, output, gate, up, down))
    ):
        raise ValueError("BF16 FIA Q/gate/K/V/O and matching MLP shapes required")
    return {
        "qkv": (torch.cat((q_gate, key, value)), tuple(tensor.shape[0] for tensor in (q_gate, key, value))),
        "output": (output, None),
        "gateup": (torch.cat((gate, up)), (gate.shape[0], up.shape[0])),
        "down": (down, None),
    }


def load_fia_checkpoint_weights(model_path, layer_index, attention):
    from safetensors import safe_open

    root = Path(model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_index}."

    def read(name):
        name = prefix + name
        with safe_open(root / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    return pack_fia_weights(
        *(read(f"self_attn.{name}_proj.weight") for name in ("q", "k", "v", "o")),
        *(read(f"mlp.{name}_proj.weight") for name in ("gate", "up", "down")),
        heads=attention.total_num_heads,
        kv_heads=attention.total_num_kv_heads,
        head_dim=attention.head_dim,
    )


class FIADenseCase:
    def __init__(self, torch, functional, layer, group, weights, rank, step, capacity):
        self.torch, self.functional, self.layer, self.group = torch, functional, layer, group
        self.weights, self.rank, self.step, self.capacity = weights, rank, step, capacity
        self.rows, self.owner_rows = capacity["padding_tokens"], capacity["equal_owner_tokens"]
        plan = owner_row_plan(step["request_ids"], self.rows, self.owner_rows, rank)
        self.indices = torch.tensor(plan["local_token_indices"], dtype=torch.int64, device="npu")
        self.active = len(plan["local_token_indices"])
        self.gather = torch.tensor(plan["gather_active_token_indices"], dtype=torch.int64, device="npu")
        self.canonical = torch.tensor(plan["canonical_active_token_indices"], dtype=torch.int64, device="npu")
        self.source = torch.empty((self.rows, 5120), dtype=torch.bfloat16, device="npu")
        self.prior = torch.empty_like(self.source)
        self.core = torch.empty((self.rows, 6144), dtype=self.source.dtype, device=self.source.device)
        self.local_source = torch.empty((self.owner_rows, 5120), dtype=self.source.dtype, device=self.source.device)
        self.local_prior = torch.empty_like(self.local_source)
        self.local_core = torch.empty((self.owner_rows, 6144), dtype=self.source.dtype, device=self.source.device)
        self.tp_core = torch.empty((self.rows, 3072), dtype=self.source.dtype, device=self.source.device)

    def refresh(self, variant):
        generator = self.torch.Generator(device="cpu").manual_seed(self.step["input_seed"] + variant)
        for tensor in (self.source, self.prior, self.core):
            tensor.copy_(
                self.torch.randn(tensor.shape, generator=generator, dtype=self.torch.float32).to(tensor.dtype) * 0.05
            )
            tensor[self.step["batch"] * 4 :].zero_()
        self.tp_core.copy_(self.core[:, self.rank * 3072 : (self.rank + 1) * 3072])
        for local, full in (
            (self.local_source, self.source),
            (self.local_prior, self.prior),
            (self.local_core, self.core),
        ):
            local.zero_()
            local[: self.active].copy_(full.index_select(0, self.indices))
        self.torch.npu.synchronize()

    def identity(self):
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.indices,
                self.gather,
                self.canonical,
                self.source,
                self.prior,
                self.core,
                self.local_source,
                self.local_prior,
                self.local_core,
                self.tp_core,
            )
        )

    def select(self, value):
        result = self.torch.zeros_like(self.local_source)
        result[: self.active].copy_(value.index_select(0, self.indices))
        return result

    def reconstruct(self, hidden, residual):
        packed = self.group.all_gather(self.torch.cat((hidden, residual), dim=-1), dim=0)
        canonical = self.torch.zeros((self.rows, 10240), dtype=packed.dtype, device=packed.device)
        canonical.index_copy_(0, self.canonical, packed.index_select(0, self.gather))
        return canonical[:, :5120].contiguous(), canonical[:, 5120:].contiguous()

    def roundtrip(self):
        return self.reconstruct(self.select(self.source), self.select(self.prior))

    def dense(self, label, *, complete, reduce_scatter=True):
        value, residual = self.layer.input_layernorm(
            self.source if label == "tp" else self.local_source,
            self.prior if label == "tp" else self.local_prior,
        )
        qkv = self.functional.linear(value, self.weights["qkv", label])
        projected = self.functional.linear(
            self.tp_core if label == "tp" else self.local_core, self.weights["output", label]
        )
        if label == "tp":
            projected = self.group.all_reduce(projected)
        value, residual = self.layer.post_attention_layernorm(projected, residual)
        if label == "owner":
            value = self.group.all_gather(value, dim=0)
        activated = self.layer.mlp.act_fn(self.functional.linear(value, self.weights["gateup", "tp"]))
        down = self.functional.linear(activated, self.weights["down", "tp"])
        if label == "tp":
            hidden = self.group.all_reduce(down)
            if not complete:
                hidden, residual = self.select(hidden), self.select(residual)
        else:
            hidden = (
                self.group.reduce_scatter(down, dim=0)
                if reduce_scatter
                else self.group.all_reduce(down)[self.rank * self.owner_rows : (self.rank + 1) * self.owner_rows]
            )
            if complete:
                hidden, residual = self.reconstruct(hidden, residual)
        return hidden, residual, qkv


def qualify_case(args, case, entries, root, checkpoint, cpu_group):
    import torch.distributed as dist
    from multi_graph_numerical_assessment import error_metrics
    from multi_graph_tp_gdn_ownership import merge_component_shards

    torch, checks, numerical = case.torch, [], []
    identity = case.identity()
    output_addresses = {name: tuple(value.data_ptr() for value in entry["outputs"]) for name, entry in entries.items()}
    args.partial_evidence = dict(step=case.step["step"], checks=checks, numerical=numerical)
    for variant in (1, 2):
        case.refresh(variant)
        if case.identity() != identity:
            raise ValueError("fixed FIA dense input addresses changed")
        for name, entry in entries.items():
            with torch.npu.stream(root):
                expected = tuple(tensor.clone() for tensor in entry["operation"]())
                entry["graph"].replay()
            torch.npu.synchronize()
            if tuple(value.data_ptr() for value in entry["outputs"]) != output_addresses[name]:
                raise ValueError(f"fixed FIA dense capture output addresses changed: {name}")
            for actual, eager in zip(entry["outputs"], expected, strict=True):
                torch.testing.assert_close(actual, eager, rtol=0, atol=0)
                if not torch.isfinite(actual).all().item():
                    raise ValueError(f"nonfinite FIA dense screen: {name}")
            active = (
                case.step["batch"] * 4
                if name in ("tp_complete", "hybrid_complete", "ownership_roundtrip")
                else case.active
            )
            for actual in entry["outputs"][:2]:
                torch.testing.assert_close(actual[active:], torch.zeros_like(actual[active:]), rtol=0, atol=0)
            checks.append(f"{name}/variant{variant}/strict_eager_replay_finite_inactive")
        for actual, expected in zip(entries["ownership_roundtrip"]["outputs"], (case.source, case.prior), strict=True):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        checks.append(f"roundtrip/variant{variant}/original_canonical_oracle")
        for candidate, baseline in (
            ("hybrid_complete", "tp_complete"),
            ("hybrid_owner_endpoint", "tp_owner_endpoint"),
            ("hybrid_owner_ar_slice", "hybrid_owner_endpoint"),
        ):
            for field, actual, expected in zip(
                ("hidden", "residual"), entries[candidate]["outputs"][:2], entries[baseline]["outputs"][:2], strict=True
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
                checks.append(f"{candidate}/{field}/variant{variant}/cross_control")
        gathered = case.group.all_gather(entries["tp_complete"]["outputs"][2], dim=-1)
        components = tuple(size // 2 for size in checkpoint["qkv"][1])
        canonical = merge_component_shards(gathered.chunk(2, dim=-1), components, dim=1)
        expected = canonical.index_select(0, case.indices)
        actual = entries["hybrid_owner_endpoint"]["outputs"][2][: case.active]
        numerical.append(dict(field="Q_gate_K_V", variant=variant, **error_metrics(actual, expected)))
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        checks.append(f"qkv/variant{variant}/per_head_gate_component_oracle")
    ranks = [None] * 2
    dist.all_gather_object(ranks, dict(rank=case.rank, checks=checks, numerical=numerical), group=cpu_group)
    return ranks


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from multi_graph_tp_gdn_ownership import capture_entry, shard_component_rows, time_entries
    from vllm.distributed.parallel_state import get_tp_group, get_world_group

    group, cpu_group = get_tp_group(), get_world_group().cpu_group
    if dist.get_world_size() != 2 or args.selection["capture_sizes"] != args.audit_sources["capture_sizes"]:
        raise ValueError("TP2 and unchanged default capture sizes required")
    layer_index = args.loader_metadata["layer_index"]
    layer = model.get_submodule(f"language_model.model.layers.{layer_index}")
    if layer.layer_type != "full_attention" or layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
        raise ValueError("unscaled dense full-attention layer required")
    attention = layer.self_attn
    if (attention.total_num_heads, attention.total_num_kv_heads, attention.head_dim) != (24, 4, 256):
        raise ValueError("actual Qwen3.5-27B full-attention dimensions required")
    checkpoint = load_fia_checkpoint_weights(args.model, layer_index, attention)
    expected_shapes = dict(qkv=(14336, 5120), output=(5120, 6144), gateup=(34816, 5120), down=(5120, 17408))
    if any(tuple(checkpoint[name][0].shape) != shape for name, shape in expected_shapes.items()):
        raise ValueError("actual Qwen3.5-27B FIA and MLP checkpoint shapes required")
    weights, rank = {}, dist.get_rank()
    torch.npu.config.allow_internal_format = True
    for name, (full, components) in checkpoint.items():
        shard = (
            shard_component_rows(full, components, rank, 2) if components else full.chunk(2, dim=1)[rank].contiguous()
        )
        weights[name, "tp"] = torch_npu.npu_format_cast(shard.to("npu"), 29)
        if name in ("qkv", "output"):
            weights[name, "owner"] = torch_npu.npu_format_cast(full.to("npu"), 29)
    if any(torch_npu.get_npu_format(tensor) != 29 for tensor in weights.values()):
        raise ValueError("NZ weights required for every timed control")
    root, profiles = torch.npu.Stream(), []
    root.wait_stream(torch.npu.current_stream())
    for step_index in args.steps:
        case = FIADenseCase(
            torch,
            functional,
            layer,
            group,
            weights,
            rank,
            args.trace["steps"][step_index],
            args.capacity_audit["steps"][step_index],
        )
        operations = dict(
            tp_complete=lambda case=case: case.dense("tp", complete=True),
            tp_owner_endpoint=lambda case=case: case.dense("tp", complete=False),
            hybrid_owner_endpoint=lambda case=case: case.dense("owner", complete=False),
            hybrid_owner_ar_slice=lambda case=case: case.dense("owner", complete=False, reduce_scatter=False),
            hybrid_complete=lambda case=case: case.dense("owner", complete=True),
            ownership_roundtrip=case.roundtrip,
        )
        case.refresh(0)
        entries = {}
        for name, operation in operations.items():
            print(f"rank={rank} FIA dense gate capture step={step_index} mode={name}", flush=True)
            entries[name] = capture_entry(operation, root)
        ranks = qualify_case(args, case, entries, root, checkpoint, cpu_group)
        case.refresh(0)
        timing = time_entries(entries, cpu_group, args.rounds, args.repeats)
        summary = timing["summary"]
        profiles.append(
            dict(
                capacity=case.capacity,
                timing=timing,
                ranks=ranks,
                matrix_shapes=dict(
                    tp_qkv=[case.rows, 5120, 7168],
                    owner_qkv=[case.owner_rows, 5120, 14336],
                    tp_o=[case.rows, 3072, 5120],
                    owner_o=[case.owner_rows, 6144, 5120],
                    tp_gateup=[case.rows, 5120, 17408],
                    owner_mlp_gateup=[case.owner_rows * 2, 5120, 17408],
                    tp_down=[case.rows, 8704, 5120],
                    owner_mlp_down=[case.owner_rows * 2, 8704, 5120],
                ),
                gross_interior_budget_us=summary["tp_complete"]["device_us"]
                - summary["hybrid_owner_endpoint"]["device_us"],
                complete_bundle_saving_us=summary["tp_complete"]["device_us"] - summary["hybrid_complete"]["device_us"],
            )
        )
        print(f"rank={rank} FIA dense gate completed step={step_index}", flush=True)
    report = dict(
        status="fia_dense_communication_screen_not_attention_execution",
        scope="actual inputNorm/Q-gate-K-V/O/postNorm/TP-MLP/AG/RS; independent frozen gated-attention O input",
        excluded=["FIA kernel", "QK norm/RoPE", "KV read/write", "real host FIA update", "continuous cache state"],
        limitation="Gross budget has different output ownership; no actual FIA or full-model performance result.",
        loader=args.loader_metadata,
        layer_index=layer_index,
        profiles=profiles,
        capacity_audit=args.capacity_audit,
        audit_sources=args.audit_sources,
        sources={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "multi_graph_gdn_hybrid_owner_probe.py",
                "multi_graph_tp_gdn_ownership.py",
                "multi_graph_tp_boundary_loader.py",
            )
        },
    )
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts = Path(__file__).resolve().parent / "results" / "multi_graph_structural"
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--trace-manifest", type=Path, default=artifacts / "structural-dynamic-trace-v1.json")
    parser.add_argument("--catalog-report", type=Path, default=artifacts / "gdn-q4-full-allocation-v1.rank0.json")
    parser.add_argument("--steps", type=int, nargs="+", default=(0, 53))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.repeats) < 1 or len(set(args.steps)) != len(args.steps):
        parser.error("new output, unique steps and positive timing counts required")
    args.trace, catalog, args.audit_sources = load_audit_inputs(args)
    if any(not 0 <= step < len(args.trace["steps"]) for step in args.steps):
        parser.error("step outside immutable trace")
    args.capacity_audit = audit_owner_capacities(args.trace, catalog)
    args.tensor_parallel_size, args.query, args.pair = 2, 4, (200, 200)
    args.gdn_pipeline, args.dtype, args.batch_invariant, args.kv_length = True, "bfloat16", False, 512
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    args.selection = select_workload(args)
    batch, capacity = args.selection["batch"], args.selection["capacity_requests"]
    workload = ReplayWorkload(batch, capacity, 4, 512, (min(batch, capacity // 2), max(0, batch - capacity // 2)))
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
            partial_evidence=getattr(args, "partial_evidence", None),
        )
        args.output.with_suffix(f".rank{failure['rank']}.failure.json").write_text(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
