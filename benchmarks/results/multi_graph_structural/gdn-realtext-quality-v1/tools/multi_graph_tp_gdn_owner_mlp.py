"""Test channel-partitioned real GDN MLPs under TP and request ownership."""

import hashlib
from pathlib import Path


def partition_gate_up(weight, parts):
    """Partition matching gate/up channels; each returned tensor is packed."""
    import torch

    if (
        weight.ndim != 2
        or min(weight.shape) <= 0
        or weight.shape[0] % 2
        or not isinstance(parts, int)
        or isinstance(parts, bool)
        or parts not in (1, 2, 4)
        or weight.shape[0] // 2 % parts
    ):
        raise ValueError("GateUp requires even packed rows and 1/2/4 equal channel groups")
    gate, up = weight.chunk(2, dim=0)
    return tuple(
        torch.cat((left, right), dim=0).contiguous()
        for left, right in zip(gate.chunk(parts, dim=0), up.chunk(parts, dim=0), strict=True)
    )


def measure_owner_mlp(args, runner, model, group, cpu_group, selection):
    """Measure connected Norm/MLP math; reductions are separate diagnostics."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from multi_graph_numerical_assessment import error_metrics
    from multi_graph_tp_gdn_ownership import (
        canonical_gather_indices,
        capture_entry,
        load_checkpoint_weights,
        owner_token_rows,
        shard_component_rows,
        time_entries,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    capacity, batch = selection["capacity_requests"], selection["batch"]
    if world != 2 or capacity != 100:
        raise ValueError("owner MLP probe requires TP2 and the default 400-token Q4 capacity")
    layer_index = args.loader_metadata["layer_index"] + 1
    layer = model.get_submodule(f"language_model.model.layers.{layer_index}")
    if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
        raise ValueError("owner MLP probe requires an ordinary unscaled dense layer")
    checkpoint = load_checkpoint_weights(args.model, layer_index)
    gate_cpu, components = checkpoint["gateup"]
    down_cpu, _ = checkpoint["down"]
    hidden_size = gate_cpu.shape[1]
    if tuple(down_cpu.shape) != (hidden_size, gate_cpu.shape[0] // 2):
        raise ValueError("checkpoint GateUp and Down intermediate channels disagree")
    torch.npu.config.allow_internal_format = True
    owner_rows = torch.tensor(owner_token_rows(capacity, 4, rank, world), device="npu", dtype=torch.int64)
    inverse = torch.tensor(canonical_gather_indices(capacity, 4, world), device="npu", dtype=torch.int64)
    source = torch.empty((capacity * 4, hidden_size), dtype=torch.bfloat16, device="npu")
    prior = torch.empty_like(source)
    local_source = torch.empty((capacity * 4 // world, hidden_size), dtype=source.dtype, device=source.device)
    local_prior = torch.empty_like(local_source)
    weights, descriptions = {}, {}
    for label in ("tp", "owner"):
        packed = shard_component_rows(gate_cpu, components, rank, world) if label == "tp" else gate_cpu
        down = down_cpu.chunk(world, dim=1)[rank].contiguous() if label == "tp" else down_cpu
        down_nz = torch_npu.npu_format_cast(down.to("npu"), 29)
        for parts in (1, 2, 4):
            grouped = tuple(
                torch_npu.npu_format_cast(weight.to("npu"), 29) for weight in partition_gate_up(packed, parts)
            )
            if any(torch_npu.get_npu_format(weight) != 29 for weight in (*grouped, down_nz)):
                raise ValueError("owner MLP NZ weight conversion did not take effect")
            weights[label, parts] = (grouped, down_nz)
            descriptions[f"{label}_parts{parts}"] = dict(
                gateup_shapes=[list(weight.shape) for weight in grouped],
                down_shape=list(down_nz.shape),
                tokens=400 if label == "tp" else 200,
                weight_format="NZ",
            )

    def refresh(live, variant):
        torch.manual_seed(2201 + variant * 19)
        source.copy_(torch.randn_like(source) * 0.05)
        prior.copy_(torch.randn_like(prior) * 0.05)
        source[live * 4 :].zero_()
        prior[live * 4 :].zero_()
        local_source.copy_(source.index_select(0, owner_rows))
        local_prior.copy_(prior.index_select(0, owner_rows))
        torch.npu.synchronize()

    def operation(label, parts):
        value, residual = layer.post_attention_layernorm(
            source if label == "tp" else local_source,
            prior if label == "tp" else local_prior,
        )
        gate_weights, down = weights[label, parts]
        activated = tuple(layer.mlp.act_fn(functional.linear(value, weight)) for weight in gate_weights)
        intermediate = activated[0] if parts == 1 else torch.cat(activated, dim=-1)
        return functional.linear(intermediate, down), residual

    def bare_gate_up(label, parts):
        value = source if label == "tp" else local_source
        gate_weights, _ = weights[label, parts]
        packed = tuple(functional.linear(value, weight) for weight in gate_weights)
        if parts == 1:
            return (packed[0],)
        # Restore [all gate channels, all up channels], not [gate0,up0,...].
        components = tuple(value.chunk(2, dim=-1) for value in packed)
        return (torch.cat([torch.cat([item[index] for item in components], dim=-1) for index in (0, 1)], dim=-1),)

    refresh(batch, 0)
    root = torch.npu.Stream()
    root.wait_stream(torch.npu.current_stream())
    entries = {}
    for scope, function in (("mlp", operation), ("gateup", bare_gate_up)):
        for label in ("tp", "owner"):
            for parts in (1, 2, 4):
                key = f"{scope}_{label}_parts{parts}"
                entries[key] = capture_entry(
                    lambda function=function, label=label, parts=parts: function(label, parts), root
                )

    checks, diagnostics = [], []
    for variant, offset in enumerate((0, -1, 1, 0), start=1):
        live = min(capacity, max(1, batch + offset))
        refresh(live, variant)
        for name, entry in entries.items():
            with torch.npu.stream(root):
                expected = tuple(value.clone() for value in entry["operation"]())
                entry["graph"].replay()
            torch.npu.synchronize()
            for actual, wanted in zip(entry["outputs"], expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                if not torch.isfinite(actual).all().item():
                    raise ValueError(f"nonfinite owner MLP output: {name}")
            checks.append(f"{name}/dynamic_{live}_{variant}/strict_eager_replay")
        for scope in ("mlp", "gateup"):
            for label in ("tp", "owner"):
                baseline = entries[f"{scope}_{label}_parts1"]["outputs"][0]
                count = (
                    live * 4 if label == "tp" else sum(request < live for request in range(rank, capacity, world)) * 4
                )
                for parts in (2, 4):
                    observed = entries[f"{scope}_{label}_parts{parts}"]["outputs"][0]
                    diagnostics.append(
                        dict(
                            scope=scope,
                            layout=label,
                            parts=parts,
                            live=live,
                            variant=variant,
                            against="same_layout_parts1",
                            **error_metrics(observed[:count], baseline[:count]),
                        )
                    )
        for parts in (1, 2, 4):
            tp_output, tp_residual = entries[f"mlp_tp_parts{parts}"]["outputs"]
            owner_output, owner_residual = entries[f"mlp_owner_parts{parts}"]["outputs"]
            # Copy: the TP all-reduce is in-place and must not corrupt capture buffers.
            reference = group.all_reduce(tp_output.clone())
            observed = group.all_gather(owner_output, dim=0).index_select(0, inverse)
            gathered_residual = group.all_gather(owner_residual, dim=0).index_select(0, inverse)
            torch.testing.assert_close(gathered_residual, tp_residual, rtol=0, atol=0)
            diagnostics.append(
                dict(
                    scope="mlp",
                    parts=parts,
                    live=live,
                    variant=variant,
                    against="canonical_owner_vs_tp_sum",
                    **error_metrics(observed[: live * 4], reference[: live * 4]),
                )
            )
            checks.append(f"parts{parts}/dynamic_{live}_{variant}/canonical_residual")
    refresh(batch, 0)
    timing = time_entries(entries, cpu_group, args.rounds, args.repeats)
    ranks = [None] * world
    dist.all_gather_object(ranks, dict(rank=rank, checks=checks, diagnostics=diagnostics), group=cpu_group)
    return dict(
        status="completed_with_numerical_diagnostics",
        scope="actual_checkpoint_post_norm_gateup_activation_concat_down_local_math",
        gateup_scope="actual_checkpoint_gateup_and_canonical_packed_concat_only",
        excluded=["TP_output_all_reduce", "owner_input_selection", "owner_output_gather", "weight_NZ_conversion"],
        timing_contract="MLP and bare GateUp are independent endpoints; their times must not be added",
        owner_assignment="global_request_id_modulo_2",
        capacity_tokens=400,
        batch=batch,
        layer_index=layer_index,
        weights=descriptions,
        ranks=ranks,
        timing=timing,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
