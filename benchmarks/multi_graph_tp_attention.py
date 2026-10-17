"""Isolated Qwen3.5 TP communication-to-attention graph prototype.

The selected checkpoint layer supplies real Norm, QKV, QK-Norm, RoPE, FIA and O
operations. Rank-local Down outputs, residuals and prefix hidden states are
synthetic. Every replay retains FIA host-parameter updates. --include-mlp adds
the selected layer's post-Norm, MLP and final TP reduction, returning both
hidden states and residual. Neither scope runs the complete hybrid model.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from multi_graph_tp_boundary import select_workload
from multi_graph_tp_boundary_loader import run_qwen35

from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload


def experiment_modes(include_mlp, split_output_overlap):
    if include_mlp:
        return ("joined", "split_output_overlap", "split_mlp_overlap")
    modes = ("joined", "split_serial", "split_overlap")
    return modes + ("split_output_overlap",) if split_output_overlap else modes


def experiment_scope(include_mlp):
    if include_mlp:
        return "selected_decoder_layer_and_input_communication_not_full_model"
    return "selected_layer_communication_norm_qkv_rope_fia_gate_o_not_full_decoder"


def execution_layout(mode, rows, pair, include_mlp, shared_qkv=False):
    if mode == "split_attention_parallel":
        if not include_mlp or not shared_qkv:
            raise ValueError("parallel FIA requires shared QKV and full MLP scope")
        layout = execution_layout("joined", rows, pair, include_mlp, shared_qkv)
        layout.update(attention_tokens=list(pair), attention_streams=["root", "comm"])
        return layout
    if mode not in experiment_modes(include_mlp, True):
        raise ValueError("mode is not supported in the requested timing scope")
    split_output = mode in ("split_output_overlap", "split_mlp_overlap")
    split_mlp = mode == "split_mlp_overlap"
    collectives = (
        ["input_A", "input_B", "output_A", "output_B"]
        if split_output
        else ["input", "output"]
        if mode == "joined"
        else ["input_A", "input_B", "output"]
    )
    if include_mlp:
        collectives += ["down_A", "down_B"] if split_mlp else ["down"]
    if shared_qkv and mode != "joined":
        collectives = ["input"] + collectives[2:]
    return dict(
        input_and_qkv_tokens=[rows] if shared_qkv or mode == "joined" else list(pair),
        shared_input_norm_qkv=shared_qkv,
        projection_tokens=list(pair) if split_output else [rows],
        reduction_tokens=list(pair) if split_output else [rows],
        reduction_stream="comm" if split_output else "root",
        shared_full_bucket=not split_output,
        join_copy_timed=split_output and not split_mlp,
        collective_order=collectives,
        mlp=(
            dict(
                projection_tokens=list(pair) if split_mlp else [rows],
                reduction_tokens=list(pair) if split_mlp else [rows],
                reduction_stream="comm" if split_mlp else "root",
                shared_full_bucket=not split_mlp,
                real_down_all_reduces_per_replay=2 if split_mlp else 1,
                hidden_and_residual_join_timed=split_mlp,
            )
            if include_mlp
            else None
        ),
    )


def partition_gate_up(weight, partitions):
    """Prepack matching gate/up channels without changing intermediate order."""
    import torch

    if weight.ndim != 2 or partitions < 1 or weight.shape[0] % (2 * partitions):
        raise ValueError("GateUp rows must contain equally divisible gate and up channels")
    if partitions == 1:
        return (weight,)
    gate, up = weight.chunk(2, dim=0)
    return tuple(
        torch.cat((gate_part, up_part), dim=0)
        for gate_part, up_part in zip(gate.chunk(partitions), up.chunk(partitions), strict=True)
    )


def project_mlp(value, residual, post_norm, act_fn, gate_up_weight, down_weight):
    """Return local Down and the real post-Norm residual, before any TP sum."""
    import torch.nn.functional as functional

    normalized, next_residual = post_norm(value, residual)
    if isinstance(gate_up_weight, tuple):
        import torch

        intermediate = torch.cat([act_fn(functional.linear(normalized, weight)) for weight in gate_up_weight], dim=-1)
    else:
        intermediate = act_fn(functional.linear(normalized, gate_up_weight))
    return functional.linear(intermediate, down_weight), next_residual


class BranchSource:
    def __init__(self, graph, input_data):
        self.graph = graph
        self.input_data = input_data

    def get(self, provider):
        return tuple(
            provider.resolve(self.input_data[lane].mapping)
            for task, lane in zip(self.graph.tasks, self.graph.task_lanes, strict=True)
            if task.provider == provider
        )


def request_values(workload, start, stop, batch, variant):
    """Keep capacity fixed while changing live requests and host KV lengths."""
    if not 0 <= start < stop <= workload.capacity or not 1 <= batch <= workload.capacity:
        raise ValueError("request interval or live batch exceeds captured capacity")
    if workload.kv_length < workload.query + 64:
        raise ValueError("the dynamic-length qualification requires 64 prefix tokens")
    lengths = [
        workload.kv_length - ((index * 7 + variant * 13) % 64) if index < batch else 0 for index in range(start, stop)
    ]
    width = workload.pages_per_request
    positions = [
        position
        for length in lengths
        for position in (range(length - workload.query, length) if length else [0] * workload.query)
    ]
    slots = [
        request * width * workload.block_size + position if length else -1
        for request, length in zip(range(start, stop), lengths, strict=True)
        for position in (range(length - workload.query, length) if length else [0] * workload.query)
    ]
    return dict(
        lengths=lengths,
        boundaries=[index * workload.query for index in range(stop - start + 1)],
        pages=[
            list(range(request * width, (request + 1) * width)) if length else [0] * width
            for request, length in zip(range(start, stop), lengths, strict=True)
        ],
        positions=positions,
        slots=slots,
    )


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module):
    # The weight-only loader's callbacks intentionally carry no attention state.
    # This experiment owns and qualifies only the selected full-attention layer.
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from vllm.config import CUDAGraphMode
    from vllm.distributed.parallel_state import get_tp_group, get_world_group
    from vllm.forward_context import BatchDescriptor, get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, AscendMetadata
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph
    from vllm_ascend.ops.linear import (
        AscendMergedColumnParallelLinear,
        AscendRowParallelLinear,
        AscendUnquantizedLinearMethod,
    )
    from vllm_ascend.patch.worker.patch_qwen3_5 import _uses_multimodal_rope

    group, cpu_group = get_tp_group(), get_world_group().cpu_group
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != args.tensor_parallel_size or group.world_size != world:
        raise ValueError("loaded TP world disagrees with requested parallelism")
    attention = model.get_submodule(args.boundary_module_path.rsplit(".", 1)[0])
    layer = model.get_submodule(args.boundary_module_path.rsplit(".", 2)[0])
    backend, impl = attention.attn, attention.attn.impl
    if (
        type(impl) is not AscendAttentionBackendImpl
        or impl.sinks is not None
        or impl.sliding_window is not None
        or impl.use_bnsd_kv_cache
        or impl.pcp_enabled
        or impl.kv_sharing_target_layer_name is not None
        or impl.enable_c8_quant
    ):
        raise ValueError("this experiment requires standard dense BF16 TND FIA")
    if not attention.attn_output_gate or module is not attention.qkv_proj:
        raise ValueError("expected the loaded Qwen3.5 gated-Q full-attention projection")
    rows, boundary = workload.capacity * workload.query, args.pair[0]
    split_request = boundary // workload.query
    intervals = ((0, workload.capacity), (0, split_request), (split_request, workload.capacity))
    main = torch.npu.current_stream()
    checks, entries = [], {}
    probe_inputs = {}
    torch.npu.config.allow_internal_format = True
    weight_nd = torch_npu.npu_format_cast(module.weight, 2)
    weight_nz = torch_npu.npu_format_cast(weight_nd, 29)
    if torch_npu.get_npu_format(weight_nz) != 29:
        raise ValueError("NZ projection conversion did not take effect")
    output_module = attention.o_proj
    if (
        type(output_module) is not AscendRowParallelLinear
        or type(output_module.quant_method) is not AscendUnquantizedLinearMethod
        or output_module.custom_op is not None
        or output_module.bias is not None
        or not output_module.input_is_parallel
        or not output_module.reduce_results
        or output_module.weight.dtype != torch.bfloat16
    ):
        raise ValueError("requires a local BF16 O projection with the ordinary TP all-reduce")
    output_weight_nd = torch_npu.npu_format_cast(output_module.weight, 2)
    output_weight_nz = torch_npu.npu_format_cast(output_weight_nd, 29)
    if torch_npu.get_npu_format(output_weight_nz) != 29:
        raise ValueError("NZ O projection conversion did not take effect")
    if args.include_mlp:
        if layer.layer_scale or layer.layer_type != "full_attention" or layer.use_attn_reduce_scatter_for_moe:
            raise ValueError("the MLP scope requires an unscaled dense full-attention decoder layer")
        mlp = layer.mlp
        if getattr(mlp, "expert_gate", None) is not None:
            raise ValueError("the MLP scope does not support expert-specific gating")
        gate_up_module, down_module = mlp.gate_up_proj, mlp.down_proj
        if (
            type(gate_up_module) is not AscendMergedColumnParallelLinear
            or type(down_module) is not AscendRowParallelLinear
            or gate_up_module.gather_output
            or not down_module.input_is_parallel
            or not down_module.reduce_results
            or any(
                type(projection.quant_method) is not AscendUnquantizedLinearMethod
                or projection.custom_op is not None
                or projection.bias is not None
                or not projection.return_bias
                or projection.weight.dtype != torch.bfloat16
                or projection.tp_size != world
                or projection.tp_rank != group.rank_in_group
                for projection in (gate_up_module, down_module)
            )
        ):
            raise ValueError("MLP requires local BF16 GateUp and Down shards with the ordinary TP all-reduce")
        if (
            tuple(gate_up_module.weight.shape) != (2 * down_module.weight.shape[1], weight_nd.shape[1])
            or down_module.weight.shape[0] != weight_nd.shape[1]
        ):
            raise ValueError("loaded MLP shards disagree with the gated intermediate and hidden dimensions")
        gate_up_weight_nd = torch_npu.npu_format_cast(gate_up_module.weight, 2)
        down_weight_nd = torch_npu.npu_format_cast(down_module.weight, 2)
        gate_up_weight_nz = torch_npu.npu_format_cast(gate_up_weight_nd, 29)
        down_weight_nz = torch_npu.npu_format_cast(down_weight_nd, 29)
        if any(torch_npu.get_npu_format(weight) != 29 for weight in (gate_up_weight_nz, down_weight_nz)):
            raise ValueError("NZ MLP projection conversion did not take effect")
        gate_up_weight_timed = gate_up_weight_nd if args.gate_up_format == "ND" else gate_up_weight_nz
        if args.gate_up_partitions > 1:
            # Prepack before capture. Each activation consumes matching gate/up
            # channels; the runtime intermediate join remains inside the timer.
            weight_format = 2 if args.gate_up_format == "ND" else 29
            gate_up_weight_timed = tuple(
                torch_npu.npu_format_cast(weight, weight_format)
                for weight in partition_gate_up(gate_up_weight_nd, args.gate_up_partitions)
            )
    down = torch.empty((rows, weight_nd.shape[1]), dtype=torch.bfloat16, device="npu")
    residual = torch.empty_like(down)
    cache_shape = (workload.kv_blocks, workload.block_size, attention.num_kv_heads, attention.head_dim)
    caches = tuple(
        torch_npu.npu_format_cast(torch.zeros(cache_shape, dtype=down.dtype, device=down.device), 2) for _ in range(2)
    )
    if not all(value.is_contiguous() and torch_npu.get_npu_format(value) == 2 for value in caches):
        raise ValueError("the old-driver qualification requires contiguous ND paged KV")
    backend.kv_cache = caches
    impl.key_cache, impl.value_cache = caches
    mask = torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu_(1)
    layer_name = backend.layer_name

    class InputData:
        def __init__(self, start, stop):
            self.start, self.stop = start, stop
            values = request_values(workload, start, stop, workload.batch, 0)
            lengths = torch.tensor(values["lengths"], dtype=torch.int32)
            boundaries = torch.tensor(values["boundaries"], dtype=torch.int32)
            self.positions = torch.tensor(values["positions"], dtype=torch.int64, device="npu")
            self.metadata = AscendMetadata(
                attn_mask=mask,
                attn_state=(
                    AscendAttentionState.DecodeOnly if workload.query == 1 else AscendAttentionState.ChunkedPrefill
                ),
                # Captured extents include dummy rows; zero KV lengths and -1
                # slots exclude their attention reads/writes at each replay.
                num_actual_tokens=(stop - start) * workload.query,
                num_decode_tokens=(stop - start) * workload.query,
                num_decodes=stop - start,
                seq_lens=lengths,
                seq_lens_cpu=lengths,
                seq_lens_list=values["lengths"],
                seq_lens_gpu=lengths.npu(),
                query_start_loc=boundaries,
                query_start_loc_gpu=boundaries.npu(),
                actual_seq_lengths_q=values["boundaries"][1:],
                max_query_len=workload.query,
                block_tables=torch.tensor(values["pages"], dtype=torch.int32, device="npu"),
                slot_mapping=torch.tensor(values["slots"], dtype=torch.int64, device="npu"),
                model_runner_type="generate",
            )
            self.mapping = {layer_name: self.metadata}

        def refresh(self, batch, variant):
            values = request_values(workload, self.start, self.stop, batch, variant)
            self.metadata.seq_lens_list[:] = values["lengths"]
            self.metadata.seq_lens.copy_(torch.tensor(values["lengths"], dtype=torch.int32))
            self.metadata.seq_lens_gpu.copy_(self.metadata.seq_lens)
            self.metadata.block_tables.copy_(torch.tensor(values["pages"], dtype=torch.int32))
            self.metadata.slot_mapping.copy_(torch.tensor(values["slots"], dtype=torch.int64))
            self.positions.copy_(torch.tensor(values["positions"], dtype=torch.int64))

        def forward_context(self):
            count = (self.stop - self.start) * workload.query
            return set_ascend_forward_context(
                self.mapping,
                runner.vllm_config,
                num_tokens=count,
                num_actual_tokens=count,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=count, num_reqs=self.stop - self.start, uniform=True),
                model_instance=model,
                skip_compiled=True,
            )

        def identity(self):
            return tuple(
                value.data_ptr()
                for value in (
                    self.positions,
                    self.metadata.block_tables,
                    self.metadata.slot_mapping,
                    self.metadata.seq_lens_gpu,
                    self.metadata.query_start_loc_gpu,
                )
            )

    input_data = [InputData(start, stop) for start, stop in intervals]
    metadata_identities = tuple(data.identity() for data in input_data)
    cache_addresses = tuple(value.data_ptr() for value in caches)

    def transform_qkv(qkv, positions, cos_sin=None):
        if _uses_multimodal_rope(attention):
            if cos_sin is None:
                planes = positions.unsqueeze(0).expand(3, -1)
                cos_sin = attention.rotary_emb.cos_sin_cache[planes].to(device=qkv.device, dtype=qkv.dtype)
            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + attention.q_norm.weight,
                k_weight=1.0 + attention.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=attention.num_heads,
                num_kv_heads=attention.num_kv_heads,
                head_size=attention.head_dim,
                eps=attention.config.rms_norm_eps,
                mrope_section=attention.rotary_emb.mrope_section,
                is_interleaved=attention.rotary_emb.mrope_interleaved,
                rope_dim=attention.rotary_emb.rotary_dim,
                has_gate=True,
            )
        else:
            q_gate, k, v = qkv.split([attention.q_size * 2, attention.kv_size, attention.kv_size], dim=-1)
            q, gate = q_gate.view(-1, attention.num_heads, attention.head_dim * 2).chunk(2, dim=-1)
            q = attention.q_norm(q).reshape(-1, attention.q_size)
            k = attention.k_norm(k.reshape(-1, attention.num_kv_heads, attention.head_dim)).reshape(
                -1, attention.kv_size
            )
            q, k = attention.rotary_emb(positions, q, k)
        # Perform all layout materialization before the FIA update task group.
        return (
            q.reshape(-1, attention.num_heads, attention.head_dim).contiguous(),
            k.reshape(-1, attention.num_kv_heads, attention.head_dim).contiguous(),
            v.reshape(-1, attention.num_kv_heads, attention.head_dim).contiguous(),
            gate.reshape(-1, attention.q_size),
        )

    def projected(normalized, positions, weight, cos_sin=None):
        return transform_qkv(functional.linear(normalized, weight), positions, cos_sin)

    def attend_projected(projection, data):
        q, k, v, gate = projection
        output = torch.empty_like(q)
        impl.forward(backend, q, k, v, caches, data.metadata, output=output)
        return output.reshape(-1, attention.q_size) * torch.sigmoid(gate)

    def attend(value, residual_value, data, weight, cos_sin=None):
        normalized, next_residual = layer.input_layernorm(value, residual_value)
        attended = attend_projected(projected(normalized, data.positions, weight, cos_sin), data)
        return (attended, next_residual) if args.include_mlp else attended

    def shared_output(value, weight):
        return group.all_reduce(functional.linear(value, weight))

    def local_mlp(value, residual_value, gate_up_weight, down_weight):
        return project_mlp(
            value, residual_value, layer.post_attention_layernorm, mlp.act_fn, gate_up_weight, down_weight
        )

    def shared_mlp(value, residual_value, gate_up_weight, down_weight):
        projected_down, next_residual = local_mlp(value, residual_value, gate_up_weight, down_weight)
        return group.all_reduce(projected_down), next_residual

    def assert_close(actual, expected, label, live_rows=None):
        actual, expected = actual.float().cpu(), expected.float().cpu()
        if live_rows is not None:
            actual, expected = actual[:live_rows], expected[:live_rows]
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02, msg=lambda msg: f"{label}: {msg}")
        checks.append(dict(label=label, max_abs=(actual - expected).abs().max().item(), elements=actual.numel()))

    def assert_result(actual, expected, label, live_rows):
        if args.include_mlp:
            for field, value, wanted in zip(("hidden", "residual"), actual, expected, strict=True):
                assert_close(value, wanted, f"{label}/{field}", live_rows)
        else:
            assert_close(actual, expected, label, live_rows)

    if args.shared_rope:
        if not _uses_multimodal_rope(attention):
            raise ValueError("shared lookup requires the selected MRoPE kernel")
        positions = input_data[0].positions
        table = attention.rotary_emb.cos_sin_cache
        gathered = table.index_select(0, positions).to(dtype=down.dtype)
        expanded = gathered.unsqueeze(0).expand(3, -1, -1)
        varied_positions = torch.stack((positions, positions + 1, positions + 2))
        varied = table[varied_positions].to(dtype=down.dtype)
        torch.manual_seed(900)
        normalized = torch.randn(down.shape, dtype=down.dtype, device=down.device) * 0.1
        for label, full_cos in (("expanded", expanded), ("different_planes", varied)):
            for start, stop in ((0, boundary), (boundary, rows)):
                view = full_cos[:, start:stop]
                observed = projected(normalized[start:stop], positions[start:stop], weight_nd, view)
                expected = projected(normalized[start:stop], positions[start:stop], weight_nd, view.contiguous())
                for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                    checks.append(dict(label=f"rope_view/{label}/{start}/{index}", bitwise_equal=True))

    # A coherent prefix for this layer's synthetic hidden states, not a model
    # prefill. The same projected prefix is copied into each request's pages.
    torch.manual_seed(901)
    prefix_hidden = torch.randn((workload.kv_length, down.shape[1]), dtype=down.dtype, device=down.device) * 0.1
    with input_data[0].forward_context():
        normalized = layer.input_layernorm(prefix_hidden)
        _, prefix_k, prefix_v, _ = projected(normalized, torch.arange(workload.kv_length, device="npu"), weight_nd)
    page_rows = workload.pages_per_request * workload.block_size
    for cache, prefix in zip(caches, (prefix_k, prefix_v), strict=True):
        view = cache.view(workload.capacity, page_rows, attention.num_kv_heads, attention.head_dim)
        view[:, : workload.kv_length].copy_(prefix.unsqueeze(0).expand(workload.capacity, -1, -1, -1))
    initial_caches = tuple(value.clone() for value in caches)
    adjacent = None
    adjacent_expected_states = None
    if args.gdn_pipeline:
        from multi_graph_tp_gdn import qualify_gdn

        adjacent = qualify_gdn(args, workload, runner, model, (down, residual), cpu_group, setup_only=True)

    def restore_cache():
        torch.npu.synchronize()
        if adjacent is not None:
            adjacent.reset()
        for value, initial in zip(caches, initial_caches, strict=True):
            value.copy_(initial)
        torch.npu.synchronize()

    def prepare(batch, variant):
        torch.manual_seed(42 + variant)
        down.copy_(torch.randn(down.shape, dtype=down.dtype, device=down.device) * (0.1 * (rank + 1)))
        residual.copy_(torch.randn(residual.shape, dtype=residual.dtype, device=residual.device) * 0.1)
        down[batch * workload.query :].zero_()
        residual[batch * workload.query :].zero_()
        for data in input_data:
            data.refresh(batch, variant)
        if adjacent is not None:
            adjacent.refresh(batch)
        if tuple(data.identity() for data in input_data) != metadata_identities:
            raise ValueError("captured metadata tensor addresses changed")
        if tuple(value.data_ptr() for value in caches) != cache_addresses:
            raise ValueError("selected-layer KV addresses changed")
        torch.npu.synchronize()

    def reference(batch, label):
        nonlocal adjacent_expected_states
        independent_sum = down.float().cpu()
        dist.all_reduce(independent_sum, group=cpu_group)
        reduced = group.all_reduce(down.clone())
        assert_close(reduced, independent_sum, f"{label}/collective_vs_cpu")
        restore_cache()
        with input_data[0].forward_context():
            attended = attend(reduced.clone(), residual.clone(), input_data[0], weight_nd)
            if args.include_mlp:
                attended, attention_residual = attended
            projected_result_nd = shared_output(attended, output_weight_nd).clone()
            if args.include_mlp:
                result = tuple(
                    value.clone()
                    for value in shared_mlp(
                        projected_result_nd.clone(), attention_residual.clone(), gate_up_weight_nd, down_weight_nd
                    )
                )
            else:
                result = projected_result_nd
        torch.npu.synchronize()
        if label == "initial":
            # Qualify the gated-Q decomposition against the official complete
            # attention module, including its real O projection and collective.
            with input_data[0].forward_context():
                projected_result, _ = attention.o_proj(attended)
            assert_close(projected_result_nd, projected_result, "shared_o_vs_official", batch * workload.query)
            restore_cache()
            with input_data[0].forward_context():
                normalized, _ = layer.input_layernorm(reduced.clone(), residual.clone())
                official = attention(positions=input_data[0].positions, hidden_states=normalized)
            assert_close(projected_result, official, "decomposed_vs_official_attention", batch * workload.query)
            if args.include_mlp:
                with input_data[0].forward_context():
                    normalized, official_residual = layer.post_attention_layernorm(
                        projected_result_nd.clone(), attention_residual.clone()
                    )
                    official_mlp = mlp(normalized)
                assert_result(
                    result, (official_mlp, official_residual), "decomposed_vs_official_mlp", batch * workload.query
                )
        if args.include_mlp:
            restore_cache()
            with input_data[0].forward_context():
                official_layer = layer(
                    hidden_states=reduced.clone(), residual=residual.clone(), positions=input_data[0].positions
                )
            assert_result(result, official_layer, f"{label}/decomposed_vs_official_layer", batch * workload.query)
            if args.contention_probe:
                with input_data[0].forward_context():
                    probe_inputs["normalized"], _ = layer.post_attention_layernorm(
                        projected_result_nd.clone(), attention_residual.clone()
                    )
                    probe_inputs["local_down"], _ = local_mlp(
                        projected_result_nd.clone(), attention_residual.clone(), gate_up_weight_nd, down_weight_nd
                    )
        if adjacent is not None:
            adjacent.reset()
            result = tuple(value.clone() for value in adjacent.run(*result))
            adjacent_expected_states = tuple(value.clone() for value in adjacent.states)
            torch.npu.synchronize()
        return result

    def check_adjacent(label):
        if adjacent is not None:
            for name, actual, expected in zip(("conv", "ssm"), adjacent.states, adjacent_expected_states, strict=True):
                assert_close(actual, expected, f"{label}/gdn_{name}")

    class BranchGraph(UpdatableGraph):
        def __init__(self):
            super().__init__()
            self.lane = 0
            self.task_lanes = []

        def get_capture_resource(self, key, factory, use_max_workspace=False):
            return super().get_capture_resource((self.lane, key), factory, use_max_workspace)

        def register_task(self, operation, kwargs, provider):
            super().register_task(operation, kwargs, provider)
            self.task_lanes.append(self.lane)

    class CapturedAttention:
        def __init__(self, mode):
            self.gdn_overlap = mode != "split_gdn_barrier"
            mode = "split_mlp_overlap" if mode == "split_gdn_barrier" else mode
            self.mode = mode
            self.root, self.comm, self.update_stream = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
            self.events = [torch.npu.Event() for _ in range(3)]
            self.prepared, self.updated = torch.npu.Event(), torch.npu.Event()
            self.graph, self.pool = BranchGraph(), torch.npu.graph_pool_handle()
            self.working, self.residual_working = torch.empty_like(down), torch.empty_like(residual)
            self.attention_output = torch.empty((rows, attention.q_size), dtype=down.dtype, device=down.device)
            self.output_partitions = ()
            self.mlp_partitions = ()
            self.shared_qkv_output = None
            if args.include_mlp:
                self.attention_residual = torch.empty_like(residual)
            if mode in ("split_output_overlap", "split_mlp_overlap"):
                self.output_events = [torch.npu.Event() for _ in range(4)]
                self.joined_output = torch.empty(
                    (rows, output_weight_nz.shape[0]), dtype=down.dtype, device=down.device
                )
            if mode == "split_mlp_overlap":
                self.mlp_events = [torch.npu.Event() for _ in range(4)]
                self.joined_residual = torch.empty_like(residual)
                if adjacent is not None:
                    self.gdn_events = [torch.npu.Event() for _ in range(8)]
            self.root.wait_stream(main)
            restore_cache()
            with torch.npu.stream(self.root), input_data[0].forward_context():
                for _ in range(3):
                    self.operation()
            torch.npu.synchronize()
            dist.barrier(group=cpu_group)
            with input_data[0].forward_context():
                get_forward_context().capturing = True
                with torch.npu.graph(self.graph, pool=self.pool, stream=self.root):
                    self.output = self.operation()
            torch.npu.synchronize()
            expected = 1 if mode == "joined" else 2
            if len(self.graph.tasks) != expected or any(
                task.operation != torch_npu.npu_fused_infer_attention_score.out for task in self.graph.tasks
            ):
                raise ValueError(f"{mode}: expected {expected} real FIA update tasks")
            self.workspaces = tuple(value.data_ptr() for value in self.graph.capture_resources.values())
            if len(set(self.workspaces)) != expected:
                raise ValueError(f"{mode}: each FIA branch must own its workspace")
            self.source = BranchSource(self.graph, input_data)

        def branch(self, lane, value):
            data = input_data[lane]
            start, stop = data.start * workload.query, data.stop * workload.query
            capturing = get_forward_context().capturing
            self.graph.lane = lane
            with data.forward_context():
                get_forward_context().capturing = capturing
                cos_sin = self.cos_sin[:, start:stop] if args.shared_rope else None
                if args.shared_qkv:
                    result = attend_projected(
                        transform_qkv(self.shared_qkv_output[start:stop], data.positions, cos_sin), data
                    )
                else:
                    result = attend(value, self.residual_working[start:stop], data, weight_nz, cos_sin)
                    if args.include_mlp:
                        result, next_residual = result
                        # Norm may return new storage instead of mutating its input.
                        self.attention_residual[start:stop].copy_(next_residual)
                self.attention_output[start:stop].copy_(result)

        def output_lane(self, lane):
            start, stop = (0, boundary) if lane == 0 else (boundary, rows)
            projected_output = functional.linear(self.attention_output[start:stop], output_weight_nz)
            ready, done = self.output_events[lane * 2 : lane * 2 + 2]
            ready.record(self.root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(ready)
                reduced_output = group.all_reduce(projected_output)
                done.record(self.comm)
            return projected_output, reduced_output

        def mlp_lane(self, lane, value):
            start, stop = (0, boundary) if lane == 0 else (boundary, rows)
            projected_down, next_residual = local_mlp(
                value, self.attention_residual[start:stop], gate_up_weight_timed, down_weight_nz
            )
            ready, done = self.mlp_events[lane * 2 : lane * 2 + 2]
            ready.record(self.root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(ready)
                reduced_down = group.all_reduce(projected_down)
                done.record(self.comm)
            return projected_down, reduced_down, next_residual

        def finish_split_mlp(self, first_output, second_output):
            # Both O reductions have been submitted before either Down. All
            # ranks keep output_A, output_B, down_A, down_B after the input sum.
            self.root.wait_event(self.output_events[1])
            first_mlp = self.mlp_lane(0, first_output[1])
            self.root.wait_event(self.output_events[3])
            second_mlp = self.mlp_lane(1, second_output[1])
            self.root.wait_event(self.mlp_events[1])
            if adjacent is not None and not args.gdn_shared:
                return self.finish_adjacent(first_mlp, second_mlp)
            self.root.wait_event(self.mlp_events[3])
            self.mlp_partitions = (first_mlp, second_mlp)
            self.joined_output[:boundary].copy_(first_mlp[1])
            self.joined_output[boundary:].copy_(second_mlp[1])
            self.joined_residual[:boundary].copy_(first_mlp[2])
            self.joined_residual[boundary:].copy_(second_mlp[2])
            result = self.joined_output, self.joined_residual
            return adjacent.run(*result) if adjacent is not None else result

        def gdn_reduce(self, value, index):
            ready, done = self.gdn_events[index * 2 : index * 2 + 2]
            ready.record(self.root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(ready)
                reduced = group.all_reduce(value)
                done.record(self.comm)
            return reduced

        def finish_adjacent(self, first_mlp, second_mlp):
            # The first GDN local prefix can cover layer 3 Down_B. Wait before
            # submitting new collectives so every rank has identical HCCL order.
            if not self.gdn_overlap:
                self.root.wait_event(self.mlp_events[3])
            first, first_residual = adjacent.run(first_mlp[1], first_mlp[2], "left", local=True)
            self.root.wait_event(self.mlp_events[3])
            first_reduced = self.gdn_reduce(first, 0)
            second, second_residual = adjacent.run(second_mlp[1], second_mlp[2], "right", local=True)
            second_reduced = self.gdn_reduce(second, 1)
            self.root.wait_event(self.gdn_events[1])
            first_down, first_final_residual = adjacent.local_mlp(first_reduced, first_residual)
            first_final = self.gdn_reduce(first_down, 2)
            self.root.wait_event(self.gdn_events[3])
            second_down, second_final_residual = adjacent.local_mlp(second_reduced, second_residual)
            second_final = self.gdn_reduce(second_down, 3)
            self.root.wait_event(self.gdn_events[5])
            self.root.wait_event(self.gdn_events[7])
            self.joined_output[:boundary].copy_(first_final)
            self.joined_output[boundary:].copy_(second_final)
            self.joined_residual[:boundary].copy_(first_final_residual)
            self.joined_residual[boundary:].copy_(second_final_residual)
            self.mlp_partitions = (first_mlp, second_mlp)
            self.gdn_partitions = (
                first,
                second,
                first_reduced,
                second_reduced,
                first_down,
                second_down,
                first_final,
                second_final,
                first_final_residual,
                second_final_residual,
            )
            return self.joined_output, self.joined_residual

        def finish_shared_mlp(self, value):
            if not args.include_mlp:
                return value
            result = shared_mlp(value, self.attention_residual, gate_up_weight_timed, down_weight_nz)
            return adjacent.run(*result) if adjacent is not None else result

        def finish_output_lanes(self, first_output, second_output):
            self.output_partitions = (first_output, second_output)
            if self.mode == "split_mlp_overlap":
                return self.finish_split_mlp(first_output, second_output)
            self.root.wait_event(self.output_events[1])
            self.root.wait_event(self.output_events[3])
            # Retain and copy actual collective returns, which may use new storage.
            self.joined_output[:boundary].copy_(first_output[1])
            self.joined_output[boundary:].copy_(second_output[1])
            return self.finish_shared_mlp(self.joined_output)

        def operation(self):
            # The communicator reduces in place. Every schedule pays the same
            # source/restore copies; a real producer would supply these buffers.
            self.working.copy_(down)
            self.residual_working.copy_(residual)
            if args.shared_rope:
                # Text positions share all three MRoPE planes. Gather once at
                # full capacity; each branch consumes a stride-aware view.
                gathered = attention.rotary_emb.cos_sin_cache.index_select(0, input_data[0].positions)
                self.cos_sin = gathered.to(dtype=down.dtype).unsqueeze(0).expand(3, -1, -1)
            if args.shared_qkv:
                # Keep QKV GEMM's row shape identical to padding. Only the
                # downstream attention and output/MLP communication are split.
                reduced = group.all_reduce(self.working)
                normalized, next_residual = layer.input_layernorm(reduced, self.residual_working)
                self.shared_qkv_output = functional.linear(normalized, weight_nz)
                self.attention_residual.copy_(next_residual)
                if self.mode == "joined":
                    self.branch(0, reduced)
                elif self.mode == "split_attention_parallel":
                    fork, _, finished = self.events
                    fork.record(self.root)
                    self.branch(1, reduced[:boundary])
                    with torch.npu.stream(self.comm):
                        self.comm.wait_event(fork)
                        self.branch(2, reduced[boundary:])
                        finished.record(self.comm)
                    self.root.wait_event(finished)
                else:
                    self.branch(1, reduced[:boundary])
                    first_output = self.output_lane(0)
                    self.branch(2, reduced[boundary:])
                    return self.finish_output_lanes(first_output, self.output_lane(1))
            elif self.mode == "joined":
                self.branch(0, group.all_reduce(self.working))
            elif self.mode == "split_serial":
                left, right = self.working[:boundary], self.working[boundary:]
                first, second = group.all_reduce(left), group.all_reduce(right)
                self.branch(1, first)
                self.branch(2, second)
            else:
                left, right = self.working[:boundary], self.working[boundary:]
                fork, first_ready, second_ready = self.events
                fork.record(self.root)
                self.comm.wait_event(fork)
                with torch.npu.stream(self.comm):
                    first = group.all_reduce(left)
                    first_ready.record(self.comm)
                    second = group.all_reduce(right)
                    second_ready.record(self.comm)
                self.root.wait_event(first_ready)
                self.branch(1, first)
                if self.mode in ("split_output_overlap", "split_mlp_overlap"):
                    first_output = self.output_lane(0)
                self.root.wait_event(second_ready)
                self.branch(2, second)
                if self.mode in ("split_output_overlap", "split_mlp_overlap"):
                    second_output = self.output_lane(1)
                    return self.finish_output_lanes(first_output, second_output)
            # Both lanes finish before the same full-bucket O and TP reduction.
            return self.finish_shared_mlp(shared_output(self.attention_output, output_weight_nz))

        def replay(self):
            tasks = self.graph.resolve_tasks(self.source)
            self.prepared.record(main)
            self.root.wait_event(self.prepared)
            self.update_stream.wait_event(self.prepared)
            with torch.npu.stream(self.root):
                self.graph.replay()
            self.graph.update(self.update_stream, tasks)
            self.updated.record(self.update_stream)
            self.root.wait_event(self.updated)

        def identity(self):
            return (
                id(self.graph),
                tuple(value.data_ptr() for value in self.output) if args.include_mlp else self.output.data_ptr(),
                self.attention_output.data_ptr(),
                self.workspaces,
                tuple(value.data_ptr() for partition in self.output_partitions for value in partition),
                tuple(value.data_ptr() for partition in self.mlp_partitions for value in partition),
                self.attention_residual.data_ptr() if args.include_mlp else None,
                self.shared_qkv_output.data_ptr() if args.shared_qkv else None,
            )

    prepare(workload.batch, 0)
    expected = reference(workload.batch, "initial")
    modes = experiment_modes(args.include_mlp, args.split_output_overlap)
    if adjacent is not None:
        modes = (
            ("joined", "split_attention_parallel")
            if args.gdn_shared_dense
            else ("joined", "split_output_overlap")
            if args.gdn_shared_mlp
            else ("joined", "split_mlp_overlap")
            if args.gdn_shared
            else ("joined", "split_gdn_barrier", "split_mlp_overlap")
        )
    for mode in modes:
        print(f"rank={rank} capture {mode}", flush=True)
        entry = CapturedAttention(mode)
        restore_cache()
        entry.replay()
        torch.npu.synchronize()
        if adjacent is not None and mode != "joined" and not args.gdn_shared:
            # Isolate GDN shape effects from differences already present at
            # layer 3; both ranks execute the same official eager collectives.
            captured_states = tuple(value.clone() for value in adjacent.states)
            candidate_input = tuple(torch.cat([part[i] for part in entry.mlp_partitions]) for i in (1, 2))
            adjacent.reset()
            same_input = adjacent.run(*candidate_input)
            torch.npu.synchronize()
            args.pipeline_diagnostics = []
            for field, actual, wanted in zip(("hidden", "residual"), entry.output, same_input, strict=True):
                try:
                    assert_close(
                        actual[: workload.batch * workload.query],
                        wanted[: workload.batch * workload.query],
                        f"{mode}/same_layer3_input/{field}",
                    )
                    args.pipeline_diagnostics.append(dict(field=field, same_layer3_input_passed=True))
                except AssertionError as error:
                    args.pipeline_diagnostics.append(
                        dict(field=field, same_layer3_input_passed=False, error=str(error))
                    )
                    print(f"rank={rank} GDN shape diagnostic: {error}", flush=True)
            adjacent.reset(captured_states)
        assert_result(entry.output, expected, f"{mode}/initial", workload.batch * workload.query)
        check_adjacent(f"{mode}/initial")
        entries[mode] = entry
    identities = {mode: entry.identity() for mode, entry in entries.items()}
    for batch, variant in ((workload.batch - 1, 1), (workload.capacity - 1, 2), (workload.batch, 3)):
        prepare(batch, variant)
        expected = reference(batch, f"b{batch}-v{variant}")
        for mode, entry in entries.items():
            restore_cache()
            entry.replay()
            torch.npu.synchronize()
            assert_result(entry.output, expected, f"{mode}/b{batch}-v{variant}", batch * workload.query)
            check_adjacent(f"{mode}/b{batch}-v{variant}")
            if entry.identity() != identities[mode]:
                raise ValueError("graph, output or workspace address changed during metadata refresh")

    if args.gdn_probe:
        from multi_graph_tp_gdn import qualify_gdn

        report = qualify_gdn(args, workload, runner, model, expected, cpu_group)
        report.update(
            selection=args.selection,
            loader=args.loader_metadata,
            decoder_checks_rank0=checks,
            attention_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return

    if args.contention_probe:
        from multi_graph_tp_contention import measure_contention

        report = measure_contention(
            probe_inputs["normalized"],
            gate_up_weight_nz,
            probe_inputs["local_down"],
            boundary,
            group,
            cpu_group,
            trace_path=args.output.with_suffix(f".rank{rank}.trace.json") if args.trace else None,
        )
        report.update(
            selection=args.selection,
            loader=args.loader_metadata,
            decoder_checks_rank0=checks,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            probe_source_sha256=hashlib.sha256(
                Path(__file__).with_name("multi_graph_tp_contention.py").read_bytes()
            ).hexdigest(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return

    samples = []
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for round_id in range(args.rounds):
        for mode in modes if round_id % 2 == 0 else reversed(modes):
            entry = entries[mode]
            restore_cache()
            for _ in range(3):
                entry.replay()
                torch.npu.synchronize()
            wall, device = [], []
            for _ in range(args.repeats):
                if adjacent is not None:
                    restore_cache()
                dist.barrier(group=cpu_group)
                begin.record(entry.root)
                started = time.perf_counter_ns()
                entry.replay()
                end.record(entry.root)
                end.synchronize()
                wall.append((time.perf_counter_ns() - started) / 1000)
                device.append(begin.elapsed_time(end) * 1000)
            assert_result(entry.output, expected, f"{mode}/round{round_id}", workload.batch * workload.query)
            check_adjacent(f"{mode}/round{round_id}")
            samples.append(dict(mode=mode, round=round_id, wall_us=wall, device_us=device))
            print(f"rank={rank} {mode} round={round_id}: wall={statistics.median(wall):.3f} us", flush=True)
    if args.trace:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for mode in modes:
                for _ in range(2):
                    dist.barrier(group=cpu_group)
                    with torch.profiler.record_function(f"tp_attention.{mode}"):
                        entries[mode].replay()
                        torch.npu.synchronize()
        profiler.export_chrome_trace(str(args.output.with_suffix(f".rank{rank}.trace.json")))
    local = dict(rank=rank, checks=checks, samples=samples)
    results = [None] * world
    dist.all_gather_object(results, local, group=cpu_group)
    if rank == 0:
        summaries = {}
        for mode in modes:
            summary = {}
            for metric in ("wall_us", "device_us"):
                rounds = []
                for round_id in range(args.rounds):
                    rank_rows = [
                        next(row for row in result["samples"] if row["mode"] == mode and row["round"] == round_id)
                        for result in results
                    ]
                    rounds.append(
                        statistics.median(max(row[metric][i] for row in rank_rows) for i in range(args.repeats))
                    )
                summary[metric], summary[f"rounds_{metric}"] = statistics.median(rounds), rounds
            summaries[mode] = summary
        report = dict(
            status="passed",
            scope="two_adjacent_decoder_layers_not_full_model"
            if adjacent is not None
            else experiment_scope(args.include_mlp),
            model=args.model,
            tensor_parallel_size=world,
            selection=args.selection,
            loader=args.loader_metadata,
            layer_name=layer_name,
            query=workload.query,
            kv_length_maximum=workload.kv_length,
            local_qkv_shape=list(weight_nd.shape),
            local_o_shape=list(output_weight_nd.shape),
            local_gate_up_shape=list(gate_up_weight_nd.shape) if args.include_mlp else None,
            local_down_shape=list(down_weight_nd.shape) if args.include_mlp else None,
            o_and_all_reduce_timed=True,
            output_layouts={
                mode: execution_layout(
                    "split_mlp_overlap" if mode == "split_gdn_barrier" else mode,
                    rows,
                    args.pair,
                    args.include_mlp,
                    args.shared_qkv,
                )
                for mode in modes
            },
            adjacent_gdn_pipeline=adjacent is not None,
            adjacent_gdn_source_sha256=hashlib.sha256(
                Path(__file__).with_name("multi_graph_tp_gdn.py").read_bytes()
            ).hexdigest()
            if adjacent is not None
            else None,
            adjacent_gdn_shared=args.gdn_shared,
            first_layer_shared_mlp=args.gdn_shared_mlp,
            first_layer_shared_dense=args.gdn_shared_dense,
            adjacent_gdn_collectives_per_replay={
                mode: 2 if mode == "joined" or args.gdn_shared else 4 for mode in modes
            }
            if adjacent is not None
            else None,
            shared_input_norm_qkv=bool(args.shared_qkv),
            gate_up_weight_format=args.gate_up_format if args.include_mlp else None,
            gate_up_channel_partitions=args.gate_up_partitions if args.include_mlp else None,
            split_output_overlap=bool(args.split_output_overlap),
            split_output_overlap_requested=args.split_output_overlap_requested,
            include_mlp=bool(args.include_mlp),
            post_norm_mlp_and_down_all_reduce_timed=bool(args.include_mlp),
            hidden_and_residual_qualified=bool(args.include_mlp),
            real_down_all_reduces_per_replay={
                mode: (2 if mode in ("split_mlp_overlap", "split_gdn_barrier") else 1) if args.include_mlp else 0
                for mode in modes
            },
            shared_rope_lookup=bool(args.shared_rope),
            rope_source_sha256=hashlib.sha256(
                (
                    Path(__file__).parents[1] / "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py"
                ).read_bytes()
            ).hexdigest(),
            kv_cache_shape=list(cache_shape),
            cache_layout="benchmark_owned_contiguous_ND",
            real_fia_updates_per_replay={mode: len(entry.graph.tasks) for mode, entry in entries.items()},
            summaries=summaries,
            ranks=results,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=(
                "Synthetic rank-local Down outputs/residuals and a projected synthetic single-layer prefix. "
                "Official attention is an untimed eager oracle. Timed graphs include O and all output reductions; "
                "output_layouts identifies shared output work or split output work with timed result-join copies. "
                + (
                    "This scope also times post-Norm, real MLP weights/activation and final Down reductions. "
                    "Both returned hidden and residual are checked against the official selected decoder layer. "
                    if args.include_mlp
                    else "Post-Norm, MLP and Down reduction are outside this attention-only scope. "
                )
                + "No complete hybrid model forward, GDN state, draft execution, serving or real scheduler. "
                "Both schedules pay identical source restoration; true producer cost is absent. "
                "Single replay times include all FIA updates and host submission gaps."
            ),
        )
        if adjacent is not None:
            report["output_layouts_scope"] = "layer_3_only"
            report["two_layer_collective_order"] = {
                mode: layout["collective_order"]
                + (
                    ["gdn_output", "gdn_down"]
                    if mode == "joined" or args.gdn_shared
                    else ["gdn_output_A", "gdn_output_B", "gdn_down_A", "gdn_down_B"]
                )
                for mode, layout in report["output_layouts"].items()
            }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--query", type=int, choices=(1, 4), default=4)
    parser.add_argument("--pair", type=int, nargs=2, help="default Q4 pair is 272 + 128; Q1 uses the balanced pair")
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--gdn-probe", action="store_true", help="qualify actual adjacent GDN layer and state capture")
    parser.add_argument("--gdn-pipeline", action="store_true", help="time complete layer 3 to layer 4 GDN pipeline")
    parser.add_argument(
        "--gdn-shared", action="store_true", help="with --gdn-pipeline keep the adjacent GDN layer whole"
    )
    parser.add_argument("--gdn-shared-mlp", action="store_true", help="with --gdn-shared keep layer 3 MLP whole too")
    parser.add_argument(
        "--gdn-shared-dense", action="store_true", help="with --gdn-shared-mlp split only FIA across streams"
    )
    parser.add_argument(
        "--contention-probe", action="store_true", help="qualify decoder, then run only causal GateUp/HCCL probes"
    )
    parser.add_argument("--shared-rope", action="store_true", help="one full-bucket text-position lookup for all lanes")
    parser.add_argument(
        "--shared-qkv", action="store_true", help="with --include-mlp, share full-bucket input reduction, Norm and QKV"
    )
    parser.add_argument(
        "--gate-up-format",
        type=str.upper,
        choices=("NZ", "ND"),
        default="NZ",
        help="GateUp weight format used equally by padding and both split schedules; conversion is outside capture",
    )
    parser.add_argument(
        "--gate-up-partitions",
        type=int,
        choices=(1, 2),
        default=1,
        help="prepack paired gate/up channel blocks equally for padding and split; runtime joins are timed",
    )
    parser.add_argument(
        "--split-output-overlap",
        action="store_true",
        help="also time split O projections with output reduction A overlapping attention branch B",
    )
    parser.add_argument(
        "--include-mlp",
        action="store_true",
        help="time the selected decoder layer; enable shared-MLP and split-MLP output pipeline controls",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.split_output_overlap_requested = args.split_output_overlap
    if args.gdn_shared and not args.gdn_pipeline:
        parser.error("--gdn-shared requires --gdn-pipeline")
    if args.gdn_shared_mlp and not args.gdn_shared:
        parser.error("--gdn-shared-mlp requires --gdn-shared")
    if args.gdn_shared_dense and not args.gdn_shared_mlp:
        parser.error("--gdn-shared-dense requires --gdn-shared-mlp")
    if args.gdn_pipeline and (
        not args.include_mlp or not args.shared_qkv or args.query != 4 or args.gdn_probe or args.contention_probe
    ):
        parser.error("--gdn-pipeline requires Q4 --include-mlp --shared-qkv without other probes")
    if args.gdn_probe and (not args.include_mlp or not args.shared_qkv or args.query != 4 or args.contention_probe):
        parser.error("--gdn-probe requires Q4 --include-mlp --shared-qkv without --contention-probe")
    if args.include_mlp:
        args.split_output_overlap = True
    if args.shared_qkv and not args.include_mlp:
        parser.error("--shared-qkv requires the three same-endpoint --include-mlp controls")
    if (args.gate_up_format != "NZ" or args.gate_up_partitions != 1) and not args.include_mlp:
        parser.error("GateUp variants require --include-mlp")
    if args.contention_probe and (
        not args.include_mlp or not args.shared_qkv or args.gate_up_format != "NZ" or args.gate_up_partitions != 1
    ):
        parser.error("--contention-probe requires --include-mlp --shared-qkv and default GateUp controls")
    if args.output.exists() or min(args.rounds, args.repeats) < 1:
        parser.error("positive sample counts and a new output path are required")
    if int(os.environ.get("WORLD_SIZE", "1")) != args.tensor_parallel_size or args.tensor_parallel_size < 2:
        parser.error("launch with torchrun and matching multi-rank tensor parallelism")
    if not args.query + 64 <= args.kv_length <= 2048:
        parser.error("KV length must leave 64 prefix tokens and not exceed the causal mask")
    args.dtype, args.batch_invariant, args.compiled = "bfloat16", False, False
    if args.pair is None and args.query == 4:
        args.pair = (272, 128)
    args.selection = select_workload(args)
    args.pair = args.selection["pair_tokens"]
    capacity, batch = args.selection["capacity_requests"], args.selection["batch"]
    left = min(batch - 1, args.selection["pair_requests"][0])
    workload = ReplayWorkload(batch, capacity, args.query, args.kv_length, (left, batch - left))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_qwen35(args, workload, experiment=experiment)
    except Exception as error:
        rank = int(os.environ.get("RANK", "0"))
        report = dict(
            status="failed",
            rank=rank,
            scope="two_adjacent_decoder_layers_not_full_model"
            if args.gdn_pipeline
            else experiment_scope(args.include_mlp),
            include_mlp=bool(args.include_mlp),
            shared_input_norm_qkv=bool(args.shared_qkv),
            gate_up_weight_format=args.gate_up_format if args.include_mlp else None,
            gate_up_channel_partitions=args.gate_up_partitions if args.include_mlp else None,
            split_output_overlap=bool(args.split_output_overlap),
            selection=args.selection,
            model=args.model,
            loader=getattr(args, "loader_metadata", None),
            error_type=type(error).__name__,
            error=str(error),
            requested_probe="gdn_pipeline"
            if args.gdn_pipeline
            else "gdn"
            if args.gdn_probe
            else "contention"
            if args.contention_probe
            else None,
            pipeline_diagnostics=getattr(args, "pipeline_diagnostics", None),
            adjacent_gdn_shared=args.gdn_shared,
            first_layer_shared_mlp=args.gdn_shared_mlp,
            first_layer_shared_dense=args.gdn_shared_dense,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        content = json.dumps(report, indent=2, allow_nan=False)
        args.output.with_suffix(f".rank{rank}.failure.json").write_text(content, encoding="utf-8")
        if rank == 0:
            args.output.write_text(content, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
