"""Qualify real Q4 GDN kernels under TP and stable request ownership.

Projection and redistribution are deliberately outside this core endpoint.
Both layouts retain all four dense speculative state snapshots.
"""

import copy
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace


def owner_request_ids(capacity, rank, world_size=2):
    if (
        not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity <= 0
        or world_size != 2
        or capacity % world_size
        or rank not in (0, 1)
    ):
        raise ValueError("owner core requires an even capacity and TP2 rank")
    return tuple(range(rank, capacity, world_size))


def state_row_ids(request_ids, query=4):
    requests = tuple(request_ids)
    if (
        query != 4
        or len(requests) != len(set(requests))
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in requests)
    ):
        raise ValueError("state rows require unique nonnegative request IDs and Q4")
    return (0, *(1 + query * request + token for request in requests for token in range(query)))


def merge_qkv_shards(shards, local_widths, *, dim=-1):
    """Merge head shards without interleaving the Q, K, and V sections."""
    import torch

    shards, widths = tuple(shards), tuple(local_widths)
    if (
        len(shards) < 2
        or len(widths) != 3
        or any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in widths)
        or not isinstance(dim, int)
        or not shards[0].ndim
        or not -shards[0].ndim <= dim < shards[0].ndim
    ):
        raise ValueError("QKV merge requires at least two shards, three positive widths and a valid axis")
    axis = dim % shards[0].ndim
    shape = shards[0].shape
    if any(
        shard.ndim != len(shape)
        or shard.shape[axis] != sum(widths)
        or any(size != shape[index] for index, size in enumerate(shard.shape) if index != axis)
        for shard in shards
    ):
        raise ValueError("QKV shards disagree with the packed local dimensions")
    parts = [torch.split(shard, widths, dim=axis) for shard in shards]
    return torch.cat([torch.cat([part[index] for part in parts], dim=axis) for index in range(3)], dim=axis)


def build_gdn_owner_core(args, workload, runner, model, phase, cpu_group):
    """Construct two independent real-kernel layouts without changing TP globals."""
    import torch
    import torch.distributed as dist
    from fla_npu.ops.ascendc import causal_conv1d_update, recurrent_gated_delta_rule
    from vllm.config import CUDAGraphMode
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.gdn import _get_packed_conv_weights
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    rank, world = dist.get_rank(), dist.get_world_size()
    requests = owner_request_ids(workload.capacity, rank, world)
    capacity = workload.capacity
    layer_index = args.loader_metadata["layer_index"] + 1
    gdn = model.get_submodule(f"language_model.model.layers.{layer_index}").linear_attn
    if gdn.tp_size != 2 or gdn.num_spec != 3 or gdn.conv1d.bias is not None:
        raise ValueError("owner core requires bias-free TP2 Q4 GDN")
    if gdn.num_k_heads % world or gdn.num_v_heads % world:
        raise ValueError("owner core requires evenly sharded key/value heads")
    device = phase.mixed_qkv.device
    local_widths = (gdn.key_dim // world, gdn.key_dim // world, gdn.value_dim // world)
    ids = torch.tensor(requests, dtype=torch.int64, device=device)
    token_ids = (ids[:, None] * 4 + torch.arange(4, device=device)).reshape(-1)
    rows = torch.tensor(state_row_ids(requests), dtype=torch.int64, device=device)

    def gather(value):
        outputs = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(outputs, value.contiguous())
        return outputs

    def merge_inputs():
        mixed = merge_qkv_shards(gather(phase.mixed_qkv), local_widths).index_select(0, token_ids)
        return dict(
            mixed_qkv=mixed,
            b=torch.cat(gather(phase.b), dim=-1).index_select(0, token_ids),
            a=torch.cat(gather(phase.a), dim=-1).index_select(0, token_ids),
            z=torch.cat(gather(phase.z), dim=-2).index_select(0, token_ids),
        )

    owner_inputs = merge_inputs()
    owner_conv = merge_qkv_shards(gather(_get_packed_conv_weights(gdn)), local_widths)
    owner_a_log = torch.cat(gather(gdn.A_log), dim=0)
    owner_dt_bias = torch.cat(gather(gdn.dt_bias), dim=0)
    # Both ranks select the same owner interval before gathering head shards.
    owner_initial = None
    for owner in range(world):
        selected = torch.tensor(state_row_ids(owner_request_ids(capacity, owner)), dtype=torch.int64, device=device)
        conv = merge_qkv_shards(gather(gdn.kv_cache[0].index_select(0, selected)), local_widths)
        ssm = torch.cat(gather(gdn.kv_cache[1].index_select(0, selected)), dim=1)
        if rank == owner:
            owner_initial = (conv, ssm)
    assert owner_initial is not None
    initial = dict(tp=tuple(value.clone() for value in gdn.kv_cache), owner=owner_initial)
    states = {name: tuple(value.clone() for value in source) for name, source in initial.items()}
    inputs = dict(
        tp={field: getattr(phase, field).clone() for field in ("mixed_qkv", "b", "a", "z")},
        owner=owner_inputs,
    )
    original_inputs = {name: {key: value.clone() for key, value in fields.items()} for name, fields in inputs.items()}
    configs, builders, metadata, pointers = {}, {}, {}, {}
    spec = gdn.get_kv_cache_spec(runner.vllm_config)
    for name, count in (("tp", capacity), ("owner", len(requests))):
        config = copy.copy(runner.vllm_config)
        config.compilation_config = copy.copy(config.compilation_config)
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        config.compilation_config.max_cudagraph_capture_size = count * 4
        configs[name] = config
        builders[name] = AscendGDNAttentionMetadataBuilder(spec, [gdn.prefix], config, device)

    def refresh(live, accepted=1, variant=0):
        if not 0 <= live <= capacity or accepted not in (1, 2, 3, 4):
            raise ValueError("invalid Q4 live batch or accepted-token count")
        for name, fields in inputs.items():
            for field, value in fields.items():
                value.copy_(original_inputs[name][field] + variant * 0.0001)
            logical_ids = tuple(range(capacity)) if name == "tp" else requests
            lengths = [4 if request < live else 0 for request in logical_ids]
            boundaries = [0]
            for length in lengths:
                boundaries.append(boundaries[-1] + length)
            count = len(logical_ids)
            boundaries_cpu = torch.tensor(boundaries, dtype=torch.int32)
            lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
            seq_cpu = torch.tensor([128 if length else 0 for length in lengths], dtype=torch.int32)
            block_ids = [
                [1 + index * 4 + token for token in range(4)] if length else [0] * 4
                for index, length in enumerate(lengths)
            ]
            common = AscendCommonAttentionMetadata(
                query_start_loc=boundaries_cpu.to(device),
                query_start_loc_cpu=boundaries_cpu,
                seq_lens=seq_cpu.to(device),
                _seq_lens_cpu=seq_cpu,
                seq_lens_cpu=seq_cpu,
                seq_lens_cpu_upper_bound=seq_cpu,
                _num_computed_tokens_cpu=seq_cpu - lengths_cpu,
                num_computed_tokens_cpu=seq_cpu - lengths_cpu,
                num_reqs=count,
                num_actual_tokens=sum(lengths),
                max_query_len=4,
                max_seq_len=128,
                block_table_tensor=torch.tensor(block_ids, dtype=torch.int32, device=device),
                slot_mapping=torch.arange(count * 4, dtype=torch.int64, device=device),
                causal=True,
                is_prefilling=torch.zeros(count, dtype=torch.bool),
            )
            data = builders[name].build(
                0,
                common,
                num_accepted_tokens=torch.full((count,), accepted, dtype=torch.int32, device=device),
                num_decode_draft_tokens_cpu=torch.tensor(
                    [3 if length else -1 for length in lengths], dtype=torch.int32
                ),
                num_actual_reqs=sum(bool(length) for length in lengths),
            )
            if data.num_prefills or data.num_decodes or data.spec_decode_metadata is None:
                raise ValueError("owner qualification unexpectedly entered a non-Q4 path")
            data.num_actual_tokens = count * 4
            causal = data.spec_decode_metadata.spec_causal_conv1d
            addresses = tuple(
                value.data_ptr()
                for value in (
                    data.spec_state_indices_tensor,
                    causal.query_start_loc,
                    causal.cache_indices,
                    causal.num_accepted_tokens,
                    data.spec_decode_metadata.actual_seq_lengths,
                )
            )
            if name in pointers and pointers[name] != addresses:
                raise ValueError("owner core metadata storage changed")
            pointers[name] = addresses
            metadata[name] = data

    def reset(name, source=None):
        for state, value in zip(states[name], initial[name] if source is None else source, strict=True):
            state.copy_(value)

    def run(name):
        data = metadata[name]
        fields = inputs[name]
        causal = data.spec_decode_metadata.spec_causal_conv1d
        conv_state, ssm = states[name]
        width_factor = world if name == "owner" else 1
        weights = owner_conv if name == "owner" else _get_packed_conv_weights(gdn)
        convolved = torch.empty_like(fields["mixed_qkv"])
        causal_conv1d_update(
            fields["mixed_qkv"],
            conv_state,
            weights,
            bias=None,
            activation="silu" if gdn.activation else None,
            conv_state_indices=causal.cache_indices[:, 0].contiguous(),
            num_accepted_tokens=causal.num_accepted_tokens,
            query_start_loc=causal.query_start_loc,
            max_query_len=4,
            null_block_id=0,
            out=convolved,
        )
        # Match the official rearrangement: one packed copy yields contiguous Q/K/V.
        widths = tuple(width * width_factor for width in local_widths)
        chunks = torch.split(convolved, widths, dim=-1)
        packed = torch.cat([value.reshape(-1) for value in chunks])
        count = convolved.shape[0]
        sizes = tuple(count * width for width in widths)
        query, key, value = (
            tensor.view(count, -1, dimension)
            for tensor, dimension in zip(
                torch.split(packed, sizes), (gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim), strict=True
            )
        )
        gating, beta = DeviceOperator.fused_gdn_gating(
            owner_a_log if name == "owner" else gdn.A_log,
            fields["a"],
            fields["b"],
            owner_dt_bias if name == "owner" else gdn.dt_bias,
        )
        output = recurrent_gated_delta_rule(
            l2norm_fwd(query),
            l2norm_fwd(key),
            value,
            ssm,
            g=gating.squeeze(0),
            beta=beta.squeeze(0),
            scale=gdn.head_k_dim**-0.5,
            actual_seq_lengths=data.spec_decode_metadata.actual_seq_lengths,
            ssm_state_indices=data.spec_state_indices_tensor.flatten(),
            num_accepted_tokens=causal.num_accepted_tokens.to(torch.int32),
        )
        return gdn.norm(output.reshape(-1, gdn.head_v_dim), fields["z"].reshape(-1, gdn.head_v_dim)).reshape(
            count, -1, gdn.head_v_dim
        )

    def tp_as_owner(tp_output):
        output = torch.cat(gather(tp_output), dim=1).index_select(0, token_ids)
        conv = merge_qkv_shards(gather(states["tp"][0]), local_widths).index_select(0, rows)
        ssm = torch.cat(gather(states["tp"][1]), dim=1).index_select(0, rows)
        return output, (conv, ssm)

    def check_official_tp():
        """Check the hand-separated TP path against the registered official core."""
        reset("tp")
        observed = run("tp").clone()
        expected_states = tuple(value.clone() for value in states["tp"])
        saved = tuple(value.clone() for value in gdn.kv_cache)
        try:
            for state, source in zip(gdn.kv_cache, initial["tp"], strict=True):
                state.copy_(source)
            output = torch.zeros_like(observed)
            with set_ascend_forward_context(
                {gdn.prefix: metadata["tp"]},
                runner.vllm_config,
                num_tokens=capacity * 4,
                num_actual_tokens=capacity * 4,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=model,
                skip_compiled=True,
            ):
                torch.ops.vllm.qwen_gdn_attention_core(
                    inputs["tp"]["mixed_qkv"],
                    inputs["tp"]["b"],
                    inputs["tp"]["a"],
                    output,
                    gdn.prefix,
                    False,
                )
            expected = gdn.norm(
                output.reshape(-1, gdn.head_v_dim), inputs["tp"]["z"].reshape(-1, gdn.head_v_dim)
            ).reshape_as(observed)
            torch.npu.synchronize()
            torch.testing.assert_close(observed[: workload.batch * 4], expected[: workload.batch * 4], rtol=0, atol=0)
            for actual, expected_state in zip(gdn.kv_cache, expected_states, strict=True):
                torch.testing.assert_close(actual, expected_state, rtol=0, atol=0)
        finally:
            for state, source in zip(gdn.kv_cache, saved, strict=True):
                state.copy_(source)
        return dict(status="passed", check="strict_registered_official_tp_core_output_and_full_state")

    refresh(workload.batch)
    return SimpleNamespace(
        refresh=refresh,
        reset=reset,
        run=run,
        tp_as_owner=tp_as_owner,
        check_official_tp=check_official_tp,
        states=states,
        initial=initial,
        requests=requests,
        rank=rank,
        device=device,
        description=dict(
            scope="actual_gdn_conv_gating_recurrence_norm_layout_qualification_not_full_model_speedup",
            excluded=[
                "dense_input_projection",
                "activation_redistribution",
                "dense_output_projection",
                "output_collective",
            ],
            owner_assignment="global_request_id_modulo_2",
            owner_request_ids=list(requests),
            tp_tokens=capacity * 4,
            owner_tokens=len(requests) * 4,
            tp_value_heads=gdn.num_v_heads // world,
            owner_value_heads=gdn.num_v_heads,
            state_contract="four_dense_candidate_states_plus_conv_state_and_null_row",
            tp_state_shapes=[list(value.shape) for value in states["tp"]],
            owner_state_shapes=[list(value.shape) for value in states["owner"]],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )


def measure_owner_core(args, workload, runner, model, phase, cpu_group):
    """Capture, qualify dynamic/continued states, then time the actual kernels."""
    import statistics

    import torch
    import torch.distributed as dist
    from multi_graph_numerical_assessment import error_metrics

    adapter = build_gdn_owner_core(args, workload, runner, model, phase, cpu_group)
    oracle = adapter.check_official_tp()
    entries, checks, diagnostics = {}, [], []
    for name in ("tp", "owner"):
        for _ in range(2):
            adapter.reset(name)
            adapter.run(name)
        torch.npu.synchronize()
        adapter.reset(name)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            output = adapter.run(name)
        entries[name] = (graph, stream, output)

    trajectories = {name: tuple(value.clone() for value in adapter.initial[name]) for name in entries}
    for step, (offset, accepted) in enumerate(zip((0, -1, 1, 0, 0, -1, 1, 0), (1, 2, 1, 3, 1, 2, 1, 3), strict=True)):
        live = min(workload.capacity, max(1, workload.batch + offset))
        adapter.refresh(live, accepted=accepted, variant=step + 1)
        for name, (graph, stream, output) in entries.items():
            previous = trajectories[name]
            adapter.reset(name, previous)
            eager = adapter.run(name).clone()
            expected_states = tuple(value.clone() for value in adapter.states[name])
            torch.npu.synchronize()
            adapter.reset(name, previous)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.synchronize()
            logical_ids = tuple(range(workload.capacity)) if name == "tp" else adapter.requests
            active_tokens = sum(request < live for request in logical_ids) * 4
            torch.testing.assert_close(output[:active_tokens], eager[:active_tokens], rtol=0, atol=0)
            for actual, expected, before in zip(adapter.states[name], expected_states, previous, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual[0], before[0], rtol=0, atol=0)
                inactive = tuple(index for index, request in enumerate(logical_ids) if request >= live)
                if inactive:
                    inactive_rows = torch.tensor(state_row_ids(inactive)[1:], dtype=torch.int64, device=actual.device)
                    torch.testing.assert_close(
                        actual.index_select(0, inactive_rows), before.index_select(0, inactive_rows), rtol=0, atol=0
                    )
            if not torch.isfinite(output[:active_tokens]).all().item():
                raise ValueError("nonfinite owner-core output")
            trajectories[name] = tuple(value.clone() for value in adapter.states[name])
            checks.append(
                dict(
                    mode=name,
                    step=step,
                    live=live,
                    accepted=accepted,
                    check="strict_eager_replay_full_state_and_inactive_rows",
                )
            )
        canonical_output, canonical_states = adapter.tp_as_owner(entries["tp"][2])
        active = sum(request < live for request in adapter.requests) * 4
        # These paths consume identical frozen projections. Request/head
        # ownership changes scheduling, not within-head arithmetic.
        torch.testing.assert_close(entries["owner"][2][:active], canonical_output[:active], rtol=0, atol=0)
        diagnostics.append(
            dict(
                step=step,
                live=live,
                field="core_output",
                metrics=error_metrics(entries["owner"][2][:active], canonical_output[:active]),
            )
        )
        for field, actual, expected in zip(("conv", "ssm"), adapter.states["owner"], canonical_states, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            diagnostics.append(dict(step=step, live=live, field=field, metrics=error_metrics(actual, expected)))

    adapter.refresh(workload.batch)
    timings = {name: [] for name in entries}
    for round_index in range(args.rounds):
        names = tuple(entries) if round_index % 2 == 0 else tuple(reversed(entries))
        for name in names:
            graph, stream, _ = entries[name]
            adapter.reset(name)
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                for _ in range(3):
                    graph.replay()
            torch.npu.synchronize()
            for _ in range(args.repeats):
                dist.barrier(group=cpu_group)
                start = torch.npu.Event(enable_timing=True)
                stop = torch.npu.Event(enable_timing=True)
                wall_start = time.perf_counter_ns()
                with torch.npu.stream(stream):
                    start.record(stream)
                    graph.replay()
                    stop.record(stream)
                stop.synchronize()
                local = dict(
                    wall_us=(time.perf_counter_ns() - wall_start) / 1000, device_us=start.elapsed_time(stop) * 1000
                )
                rank_samples = [None] * dist.get_world_size()
                dist.all_gather_object(rank_samples, local, group=cpu_group)
                timings[name].append(
                    dict(
                        round=round_index,
                        wall_us=max(item["wall_us"] for item in rank_samples),
                        device_us=max(item["device_us"] for item in rank_samples),
                    )
                )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, dict(rank=adapter.rank, checks=checks, diagnostics=diagnostics), group=cpu_group)
    return dict(
        status="completed_with_numerical_diagnostics",
        **adapter.description,
        official_tp_oracle=oracle,
        ranks=ranks,
        timing={
            name: dict(
                samples=samples,
                wall_us=statistics.median(item["wall_us"] for item in samples),
                device_us=statistics.median(item["device_us"] for item in samples),
            )
            for name, samples in timings.items()
        },
    )
