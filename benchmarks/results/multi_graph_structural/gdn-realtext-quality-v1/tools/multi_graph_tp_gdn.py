"""Actual adjacent GDN qualification before cross-layer scheduling experiments."""

import copy
import hashlib
from pathlib import Path


def request_state_values(start, stop, live, query=4):
    if not 0 <= start < stop or live < 0 or query != 4:
        raise ValueError("expected a nonempty request interval and Q4")
    lengths = [query if request < live else 0 for request in range(start, stop)]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return dict(
        boundaries=boundaries,
        blocks=[
            [1 + query * request + token for token in range(query)] if request < live else [0] * query
            for request in range(start, stop)
        ],
        drafts=[query - 1 if length else -1 for length in lengths],
        lengths=lengths,
    )


def qualify_gdn(args, workload, runner, model, previous_output, cpu_group, *, setup_only=False):
    """Qualify real stateful eager/capture paths before cross-layer scheduling."""
    import torch
    import torch.distributed as dist
    from vllm.config import CUDAGraphMode

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    index = args.loader_metadata["layer_index"] + 1
    layer = model.get_submodule(f"language_model.model.layers.{index}")
    assert layer.layer_type == "linear_attention"
    gdn = layer.linear_attn
    assert gdn.num_spec == 3
    config = runner.vllm_config
    spec = gdn.get_kv_cache_spec(config)
    count = workload.capacity
    device = previous_output[0].device
    raw = torch.zeros((1 + count * 4, 1, 1, spec.page_size_bytes), dtype=torch.int8, device=device)
    gdn.bind_kv_cache(raw)
    torch.manual_seed(731)
    for state in gdn.kv_cache:
        state.copy_(torch.randn_like(state) * 0.01)
    initial = [state.clone() for state in gdn.kv_cache]
    hidden, residual = (value.clone() for value in previous_output)
    positions = torch.arange(count * 4, device=device)
    builder_config = copy.copy(config)
    builder_config.compilation_config = copy.copy(config.compilation_config)
    builder_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    builder_config.compilation_config.max_cudagraph_capture_size = count * 4
    boundary = args.pair[0] // 4
    intervals = {"full": (0, count), "left": (0, boundary), "right": (boundary, count)}
    builders = {
        name: AscendGDNAttentionMetadataBuilder(spec, [gdn.prefix], builder_config, device) for name in intervals
    }

    def metadata(live, name="full", accepted=1):
        start, stop = intervals[name]
        size = stop - start
        local_live = max(0, min(stop, live) - start)
        values = request_state_values(start, stop, live)
        boundaries = torch.tensor(values["boundaries"], dtype=torch.int32)
        seq_cpu = torch.tensor([128 if length else 0 for length in values["lengths"]], dtype=torch.int32)
        computed = seq_cpu - torch.tensor(values["lengths"], dtype=torch.int32)
        common = AscendCommonAttentionMetadata(
            query_start_loc=boundaries.to(device),
            query_start_loc_cpu=boundaries,
            seq_lens=seq_cpu.to(device),
            _seq_lens_cpu=seq_cpu,
            seq_lens_cpu=seq_cpu,
            seq_lens_cpu_upper_bound=seq_cpu,
            _num_computed_tokens_cpu=computed,
            num_computed_tokens_cpu=computed,
            num_reqs=size,
            num_actual_tokens=local_live * 4,
            max_query_len=4,
            max_seq_len=128,
            block_table_tensor=torch.tensor(values["blocks"], dtype=torch.int32, device=device),
            slot_mapping=torch.arange(size * 4, dtype=torch.int64, device=device),
            causal=True,
            is_prefilling=torch.zeros(size, dtype=torch.bool),
        )
        result = builders[name].build(
            0,
            common,
            num_accepted_tokens=torch.full((size,), accepted, dtype=torch.int32, device=device),
            num_decode_draft_tokens_cpu=torch.tensor(values["drafts"], dtype=torch.int32),
            num_actual_reqs=local_live,
        )
        assert result.num_prefills == result.num_decodes == 0 and result.spec_decode_metadata is not None
        return result

    def identity(data):
        causal = data.spec_decode_metadata.spec_causal_conv1d
        return tuple(
            tensor.data_ptr()
            for tensor in (
                data.spec_state_indices_tensor,
                causal.query_start_loc,
                causal.cache_indices,
                causal.num_accepted_tokens,
                data.spec_decode_metadata.actual_seq_lengths,
            )
        )

    def reset(sources=initial):
        for state, source in zip(gdn.kv_cache, sources, strict=True):
            state.copy_(source)
        torch.npu.synchronize()

    def forward(data, name="full"):
        start, stop = (request * 4 for request in intervals[name])
        with set_ascend_forward_context(
            {gdn.prefix: data},
            config,
            num_tokens=stop - start,
            num_actual_tokens=stop - start,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=model,
            skip_compiled=True,
        ):
            return layer(
                hidden_states=hidden[start:stop], residual=residual[start:stop], positions=positions[start:stop]
            )

    if setup_only:
        from types import SimpleNamespace

        import torch.nn.functional as functional
        from multi_graph_tp_gdn_phases import make_gdn_phases

        if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
            raise ValueError("adjacent pipeline requires an unscaled dense GDN layer")
        for projection in (gdn.out_proj, layer.mlp.down_proj):
            if projection.custom_op is not None or projection.bias is not None or not projection.reduce_results:
                raise ValueError("adjacent pipeline requires ordinary bias-free TP reductions")
        data_by_name = {name: metadata(count, name) for name in intervals}
        addresses = {name: identity(data) for name, data in data_by_name.items()}

        def refresh(live, accepted=1):
            for name in intervals:
                data_by_name[name] = metadata(live, name, accepted)
                # Python slicing is captured once. Keep the full lane extent;
                # zero query intervals and null state IDs mask inactive rows.
                start, stop = intervals[name]
                data_by_name[name].num_actual_tokens = (stop - start) * 4
                if identity(data_by_name[name]) != addresses[name]:
                    raise ValueError("GDN metadata buffer address changed")

        def run(value, prior, name="full", local=False, projection_done=None):
            start, stop = (request * 4 for request in intervals[name])
            with set_ascend_forward_context(
                {gdn.prefix: data_by_name[name]},
                config,
                num_tokens=stop - start,
                num_actual_tokens=stop - start,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=model,
                skip_compiled=True,
            ):
                if not local:
                    return layer(hidden_states=value, residual=prior, positions=positions[start:stop])
                normalized, next_residual = layer.input_layernorm(value, prior)
                gdn.out_proj.reduce_results = False
                projection_signalled = False
                projection_hook = None
                if projection_done is not None:
                    # Record outside the opaque state-core custom op. The
                    # last input projection is Z for the unpacked path, BA
                    # otherwise; following reshapes/layout work is vector.
                    last_projection = gdn.in_proj_z if hasattr(gdn, "in_proj_qkv") else gdn.in_proj_ba

                    def signal_projection(module, inputs, output):
                        nonlocal projection_signalled
                        projection_signalled = True
                        projection_done.record(torch.npu.current_stream())

                    projection_hook = last_projection.register_forward_hook(signal_projection)
                try:
                    projected = gdn(hidden_states=normalized)
                    if projection_done is not None and not projection_signalled:
                        raise ValueError("the GDN input projection did not publish its completion event")
                finally:
                    gdn.out_proj.reduce_results = True
                    if projection_hook is not None:
                        projection_hook.remove()
                return projected, next_residual

        def prepare_mlp(value, prior):
            normalized, next_residual = layer.post_attention_layernorm(value, prior)
            packed, _ = layer.mlp.gate_up_proj(normalized)
            return layer.mlp.act_fn(packed), next_residual

        def project_down(intermediate):
            return functional.linear(intermediate, layer.mlp.down_proj.weight)

        def local_mlp(value, prior):
            intermediate, next_residual = prepare_mlp(value, prior)
            return project_down(intermediate), next_residual

        def phase_context(name):
            start, stop = (request * 4 for request in intervals[name])
            return set_ascend_forward_context(
                {gdn.prefix: data_by_name[name]},
                config,
                num_tokens=stop - start,
                num_actual_tokens=stop - start,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=model,
                skip_compiled=True,
            )

        return SimpleNamespace(
            refresh=refresh,
            reset=reset,
            run=run,
            local_mlp=local_mlp,
            prepare_mlp=prepare_mlp,
            project_down=project_down,
            states=gdn.kv_cache,
            initial=initial,
            layer_index=index,
            phases=make_gdn_phases(layer, phase_context),
        )

    checks = []
    entries = {}
    for name in intervals:
        data = metadata(count, name)
        for _ in range(2):
            reset()
            forward(data, name)
        reset()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        print(f"rank={dist.get_rank()} capture actual GDN layer {index}/{name}", flush=True)
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            output = forward(data, name)
        entries[name] = (graph, stream, output, data)
    for step, live in enumerate((98, 97, 99, 98)):
        hidden.copy_(previous_output[0] + (step + 1) * 0.001)
        residual.copy_(previous_output[1] - (step + 1) * 0.001)
        data = metadata(live)
        reset()
        eager = tuple(value.clone() for value in forward(data))
        state_expected = [state.clone() for state in gdn.kv_cache]
        torch.npu.synchronize()
        for mode, names in (("full", ("full",)), ("split", ("left", "right"))):
            for name in names:
                updated = metadata(live, name)
                assert identity(updated) == identity(entries[name][3])
            reset()
            outputs = []
            for name in names:
                graph, stream, output, _ = entries[name]
                with torch.npu.stream(stream):
                    graph.replay()
                torch.npu.synchronize()
                outputs.append(output)
            joined = tuple(torch.cat([out[i] for out in outputs]) for i in (0, 1))
            for name, actual, expected in zip(("hidden", "residual"), joined, eager, strict=True):
                torch.testing.assert_close(
                    actual[: live * 4],
                    expected[: live * 4],
                    rtol=0.02,
                    atol=0.02,
                    msg=lambda message, label=f"{mode}/{name}/step{step}": f"{label}: {message}",
                )
                checks.append(
                    dict(
                        mode=mode,
                        live=live,
                        field=name,
                        max_abs=(actual[: live * 4] - expected[: live * 4]).abs().max().item(),
                    )
                )
            for name, state, expected, source in zip(
                ("conv", "ssm"), gdn.kv_cache, state_expected, initial, strict=True
            ):
                torch.testing.assert_close(
                    state,
                    expected,
                    rtol=0.02,
                    atol=0.02,
                    msg=lambda message, label=f"{mode}/{name}/step{step}": f"{label}: {message}",
                )
                torch.testing.assert_close(state[0], source[0], rtol=0, atol=0)
                torch.testing.assert_close(state[1 + live * 4 :], source[1 + live * 4 :], rtol=0, atol=0)
                checks.append(dict(mode=mode, live=live, field=name, max_abs=(state - expected).abs().max().item()))
    # Continue independent eager/full/split state trajectories with accepted-token
    # progression. Restoring each trajectory does not reset it to initial state.
    trajectories = {mode: [state.clone() for state in initial] for mode in ("eager", "full", "split")}
    for step, accepted in enumerate((1, 2, 1)):
        live = 98
        hidden.copy_(previous_output[0] + (step + 1) * 0.003)
        residual.copy_(previous_output[1] - (step + 1) * 0.002)
        expected_output = None
        for mode in ("eager", "full", "split"):
            names = ("left", "right") if mode == "split" else ("full",)
            for name in names:
                data = metadata(live, name, accepted)
                assert identity(data) == identity(entries[name][3])
            reset(trajectories[mode])
            if mode == "eager":
                expected_output = tuple(value.clone() for value in forward(data))
                observed = expected_output
            else:
                outputs = []
                for name in names:
                    graph, stream, output, _ = entries[name]
                    with torch.npu.stream(stream):
                        graph.replay()
                    torch.npu.synchronize()
                    outputs.append(output)
                observed = tuple(torch.cat([out[i] for out in outputs]) for i in (0, 1))
            torch.npu.synchronize()
            for actual, expected in zip(observed, expected_output, strict=True):
                torch.testing.assert_close(actual[: live * 4], expected[: live * 4], rtol=0.02, atol=0.02)
            trajectories[mode] = [state.clone() for state in gdn.kv_cache]
            for actual, expected, source in zip(trajectories[mode], trajectories["eager"], initial, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1 + live * 4 :], source[1 + live * 4 :], rtol=0, atol=0)
            checks.append(dict(mode=mode, step=step, accepted=accepted, field="continued_output_and_state"))
    assert all(not torch.equal(actual, source) for actual, source in zip(trajectories["eager"], initial, strict=True))
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, dict(rank=dist.get_rank(), checks=checks), group=cpu_group)
    return dict(
        status="passed",
        scope="adjacent_gdn_capture_qualification_not_pipeline_performance",
        layer_index=index,
        state_shapes=[list(state.shape) for state in gdn.kv_cache],
        ranks=ranks,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
