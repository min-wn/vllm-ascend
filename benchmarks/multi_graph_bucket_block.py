"""One real decoder block with fixed default-bucket internal request parallelism.

Includes embedding, metadata preparation, required FIA updates and final sync.
This is a controlled block experiment, not a complete decoder or serving runner.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import random
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


def bucket_topology(batch, sizes, requested=None):
    sizes = sorted(set(sizes))
    if batch < 2 or not sizes or min(sizes) < 1:
        raise ValueError("positive default sizes and batch >= 2 required")
    capacity = next((size for size in sizes if size >= batch), None)
    if requested is not None:
        if len(requested) != 2 or any(size not in sizes for size in requested) or sum(requested) != capacity:
            raise ValueError("requested branches must be default sizes summing to the selected bucket")
        return capacity, tuple(requested)
    pairs = [(a, b) for a in sizes for b in sizes if a >= b and a + b == capacity]
    if not pairs:
        raise ValueError("no fixed topology covering the default bucket; refuse custom sizes")
    return capacity, min(pairs, key=lambda pair: abs(pair[0] - pair[1]))


def fixed_lane_counts(batch, pair):
    if len(pair) != 2 or min(pair) < 1 or not 2 <= batch <= sum(pair):
        raise ValueError("batch must fit the fixed positive lane capacities")
    return min(batch, pair[0]), max(0, batch - pair[0])


def reuse_cases(batch, sizes):
    capacity, _ = bucket_topology(batch, sizes)
    lower = max((size for size in sizes if size < capacity), default=1)
    return [(batch, 0), (max(lower + 1, batch - 1), 1), (max(lower + 1, capacity - 1), 0), (batch, 0)]


def lane_core_limits(common, lanes):
    if common < 0 or (lanes is not None and (common or len(lanes) != 2 or min(lanes) < 0)):
        raise ValueError("specify either a nonnegative common core limit or two nonnegative lane limits")
    return tuple(lanes) if lanes is not None else (common, common)


def task_submission_order(pair, largest_first):
    return tuple(sorted(range(len(pair)), key=lambda lane: -pair[lane])) if largest_first else tuple(range(len(pair)))


def request_lengths(capacity, maximum, short, seed):
    if capacity < 1 or not 2 <= short <= maximum:
        raise ValueError("requires positive capacity and 2 <= short <= maximum")
    lengths = [short] * (capacity // 2) + [maximum] * (capacity - capacity // 2)
    random.Random(seed).shuffle(lengths)
    return lengths


def ordered_request_metadata(workload, lengths, batch, variant, order):
    if order not in ("original", "length_asc", "length_desc"):
        raise ValueError("unknown request order")
    if len(lengths) != workload.capacity or not 1 <= batch <= workload.capacity:
        raise ValueError("request lengths must match capacity and batch must fit")
    live_lengths = [length - variant for length in lengths[:batch]]
    if min(live_lengths) < 1 or max(live_lengths) > workload.kv_length:
        raise ValueError("runtime lengths must fit allocated KV")
    permutation = list(range(batch))
    if order != "original":
        permutation.sort(key=lambda index: live_lengths[index], reverse=order == "length_desc")
    inverse = [0] * workload.capacity
    for position, request in enumerate(permutation):
        inverse[request] = position
    inverse[batch:] = range(batch, workload.capacity)
    values = workload.metadata_values(0, batch)
    dummy = workload.capacity - batch
    return dict(
        ids=[100 + variant + request for request in permutation] + [0] * dummy,
        positions=[live_lengths[request] - 1 for request in permutation] + [0] * dummy,
        lengths=[live_lengths[request] for request in permutation] + [0] * dummy,
        pages=[values["pages"][request] for request in permutation]
        + [[0] * workload.pages_per_request for _ in range(dummy)],
        slots=[values["slots"][request] - workload.kv_length + live_lengths[request] for request in permutation]
        + [-1] * dummy,
        restore=inverse,
    )


class PackedStepBuffers:
    """One runtime-filled host/device allocation with fixed aligned integer views."""

    def __init__(self, targets):
        import numpy as np
        import torch

        if not targets:
            raise ValueError("nonempty integer buffer schema required")
        device = next(iter(targets.values())).device
        self.schema = {}
        offset = 0
        alignment = 64
        for name, tensor in targets.items():
            if tensor.device != device or tensor.dtype not in (torch.int32, torch.int64) or tensor.numel() == 0:
                raise ValueError("requires nonempty int32/int64 tensors on one device")
            offset = (offset + alignment - 1) // alignment * alignment
            size = tensor.numel() * tensor.element_size()
            self.schema[name] = dict(offset=offset, bytes=size, shape=list(tensor.shape))
            offset += size
        self.host_buffer = torch.zeros(offset, dtype=torch.uint8, device="cpu", pin_memory=device.type != "cpu")
        self.device_buffer = torch.empty(offset, dtype=torch.uint8, device=device)
        self.host, self.device = {}, {}
        for name, tensor in targets.items():
            item = self.schema[name]
            for buffer, views in ((self.host_buffer, self.host), (self.device_buffer, self.device)):
                views[name] = buffer.narrow(0, item["offset"], item["bytes"]).view(tensor.dtype).view(tensor.shape)
        self.arrays = {name: view.numpy() for name, view in self.host.items()}
        self.np = np

    def write(self, name, values):
        target = self.arrays[name]
        source = self.np.asarray(values, dtype=target.dtype)
        if source.shape != target.shape:
            raise ValueError(f"{name}: runtime shape changed from {target.shape} to {source.shape}")
        self.np.copyto(target, source)

    def transfer(self):
        self.device_buffer.copy_(self.host_buffer, non_blocking=True)


@contextmanager
def phase_timer(name, samples):
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        samples[name] = (time.perf_counter_ns() - started) / 1000


def experiment(args, workload, runner, model, inputs, context, reset_kv):
    # One captured block per strategy, independent of live batch and KV lengths.
    import torch
    import torch_npu
    from vllm.forward_context import get_forward_context

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
    from vllm_ascend.worker.multi_graph.benchmark import summarize

    native, native_source_sha256 = None, None
    if args.native_update:
        from multi_graph_native_update import load_native_update

        native, native_source_sha256 = load_native_update()
    layer = model.model.layers[0]
    attention = layer.self_attn
    backend = attention.attn
    if getattr(attention, "dual_chunk_attention_config", None) or getattr(attention, "qk_norm", False):
        raise ValueError("requires standard Qwen2/Qwen3 attention")
    if args.shared_cache and (
        type(backend.impl).__name__ != "AscendAttentionBackendImpl"
        or backend.impl.pcp_enabled
        or backend.impl.sinks is not None
        or backend.impl.kv_sharing_target_layer_name is not None
        or backend.impl.vllm_config.kv_transfer_config is not None
        or backend._k_scale_float != 1.0
        or backend._v_scale_float != 1.0
    ):
        raise ValueError("shared-cache experiment requires ordinary unquantized local FIA")
    main = torch.npu.current_stream()
    if any(args.lane_limits):
        limits = torch.npu.get_device_limit(torch.npu.current_device())
        if max(args.lane_limits) > limits["cube_core_num"] or 2 * max(args.lane_limits) > limits["vector_core_num"]:
            raise ValueError("attention resource limit exceeds device capacity")

    class BranchGraph(UpdatableGraph):
        def __init__(self):
            super().__init__()
            self.lane = None
            self.task_lanes = []

        def get_capture_resource(self, key, factory, use_max_workspace=False):
            return super().get_capture_resource((self.lane, key), factory, use_max_workspace)

        def register_task(self, operation, kwargs, provider):
            super().register_task(operation, kwargs, provider)
            self.task_lanes.append(self.lane)

    class BranchSource:
        def __init__(self, entry):
            self.entry = entry

        def get(self, provider):
            graph = self.entry["graph"]
            return tuple(
                provider.resolve(self.entry["branches"][lane][2])
                for task, lane in zip(graph.tasks, graph.task_lanes, strict=True)
                if task.provider == provider
            )

    entries = {}
    intervals = ((0, args.pair[0]), (args.pair[0], workload.capacity))
    modes = ("padding", "split_serial", "split_parallel") if args.serial_control else ("padding", "split_parallel")
    for mode in modes:
        data = inputs(0, workload.capacity)
        metadata = next(iter(data[2].values()))
        entry = dict(data=data, branches=[], streams=[], fork=torch.npu.Event(), joins=[])
        if args.request_order != "original":
            entry["restore"] = torch.arange(workload.capacity, dtype=torch.int64, device=data[0].device)
        if args.packed_metadata:
            packed = PackedStepBuffers(
                dict(
                    ids=data[0],
                    positions=data[1],
                    lengths=metadata.seq_lens_gpu,
                    pages=metadata.block_tables,
                    slots=metadata.slot_mapping,
                    **({"restore": entry["restore"]} if "restore" in entry else {}),
                )
            )
            metadata = dataclasses.replace(
                metadata,
                seq_lens=packed.host["lengths"],
                seq_lens_cpu=packed.host["lengths"],
                seq_lens_gpu=packed.device["lengths"],
                block_tables=packed.device["pages"],
                slot_mapping=packed.device["slots"],
            )
            data = (packed.device["ids"], packed.device["positions"], {name: metadata for name in data[2]})
            entry.update(data=data, packed=packed)
            if "restore" in entry:
                entry["restore"] = packed.device["restore"]
        if mode != "padding":
            for start, stop in intervals:
                count = stop - start
                boundaries = torch.arange(count + 1, dtype=torch.int32)
                # Tensor views share the one full-bucket H2D update, with fixed offsets.
                branch_metadata = dataclasses.replace(
                    metadata,
                    num_actual_tokens=count,
                    num_decode_tokens=count,
                    num_decodes=count,
                    seq_lens=metadata.seq_lens_cpu[start:stop],
                    seq_lens_cpu=metadata.seq_lens_cpu[start:stop],
                    seq_lens_gpu=metadata.seq_lens_gpu[start:stop],
                    seq_lens_list=metadata.seq_lens_list[start:stop],
                    actual_seq_lengths_q=list(range(1, count + 1)),
                    query_start_loc=boundaries,
                    query_start_loc_gpu=boundaries.npu(),
                    block_tables=metadata.block_tables[start:stop],
                    slot_mapping=metadata.slot_mapping[start:stop],
                )
                entry["branches"].append(
                    (data[0][start:stop], data[1][start:stop], {name: branch_metadata for name in data[2]})
                )
                stream = torch.npu.Stream()
                cores = args.lane_limits[len(entry["streams"])]
                if mode == "split_parallel" and cores:
                    torch.npu.set_stream_limit(stream, cube_num=cores, vector_num=2 * cores)
                entry["streams"].append(stream)
                entry["joins"].append(torch.npu.Event())
        targets = dict(
            ids=data[0],
            positions=data[1],
            lengths=metadata.seq_lens_gpu,
            pages=metadata.block_tables,
            slots=metadata.slot_mapping,
            **({"restore": entry["restore"]} if "restore" in entry else {}),
        )
        entry["host"] = (
            entry["packed"].host
            if args.packed_metadata
            else {name: torch.empty_like(t, device="cpu", pin_memory=True) for name, t in targets.items()}
        )
        entries[mode] = entry

    def prepare(entry, batch, variant):
        data = entry["data"]
        metadata = next(iter(data[2].values()))
        count = workload.capacity - batch
        values = workload.metadata_values(0, batch)
        lengths = [workload.kv_length - variant] * batch + [0] * count

        def copy(name, target, values):
            if args.packed_metadata:
                entry["packed"].write(name, values)
                return
            staging = entry["host"][name]
            staging.copy_(torch.tensor(values, dtype=target.dtype))
            target.copy_(staging, non_blocking=True)

        payload = None
        if args.short_kv_length is not None:
            payload = ordered_request_metadata(
                workload, args.request_lengths, batch, variant, "original" if reference_phase else args.request_order
            )
            lengths = payload["lengths"]
        copy(
            "ids",
            data[0],
            payload["ids"] if payload else list(range(100 + variant, 100 + variant + batch)) + [0] * count,
        )
        copy(
            "positions",
            data[1],
            payload["positions"] if payload else [workload.kv_length - variant - 1] * batch + [0] * count,
        )
        copy("lengths", metadata.seq_lens_gpu, lengths)
        if not args.packed_metadata:
            metadata.seq_lens_cpu.copy_(entry["host"]["lengths"])
        metadata.seq_lens_list = lengths
        copy(
            "pages",
            metadata.block_tables,
            payload["pages"] if payload else values["pages"] + [[0] * workload.pages_per_request for _ in range(count)],
        )
        copy(
            "slots",
            metadata.slot_mapping,
            payload["slots"] if payload else [slot - variant for slot in values["slots"]] + [-1] * count,
        )
        if "restore" in entry:
            copy("restore", entry["restore"], payload["restore"])
        if args.packed_metadata:
            entry["packed"].transfer()
        for branch, (start, stop) in zip(entry["branches"], intervals):
            next(iter(branch[2].values())).seq_lens_list = lengths[start:stop]
        if "graph" in entry:
            if args.static_task_plan:
                sources = entry["branches"] or [entry["data"]]
                kv_lengths = [next(iter(data[2].values())).seq_lens_list for data in sources]
                for group in entry["update_groups"]:
                    group["native"].bind_kv_lengths([kv_lengths[i] for i in group["indices"]])
            else:
                entry["tasks"] = entry["graph"].resolve_tasks(entry["source"])
                if "native" in entry:
                    for group in entry["update_groups"]:
                        group["native"].bind(tuple(entry["tasks"][i] for i in group["indices"]))

    def forward_impl(mode, parallel=True, partition=True):
        entry = entries[mode]
        data = entry["data"]
        hidden = model.model.embed_tokens(data[0])
        optimized = args.shared_cache and not reference_phase
        if mode == "padding" and not optimized:
            return layer(data[1], hidden, None)
        partition = partition and mode != "padding"
        parallel = parallel and mode == "split_parallel"
        normalized = layer.input_layernorm(hidden)
        qkv, _ = attention.qkv_proj(normalized)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        if hasattr(attention, "q_norm"):
            q = attention.q_norm(q.view(-1, attention.num_heads, attention.head_dim)).view(q.shape)
            k = attention.k_norm(k.view(-1, attention.num_kv_heads, attention.head_dim)).view(k.shape)
        q, k = attention.rotary_emb(data[1], q, k)
        output = torch.empty_like(q)
        pipeline = args.pipeline_output_projection and mode != "padding" and not reference_phase
        pipeline_ffn = args.pipeline_ffn and mode != "padding" and not reference_phase
        projected = torch.empty_like(hidden) if pipeline else None
        final_hidden = torch.empty_like(hidden) if pipeline_ffn else None
        final_residual = torch.empty_like(hidden) if pipeline_ffn else None
        root = torch.npu.current_stream()
        capturing = get_forward_context().capturing
        if optimized:
            # Write the full bucket's K/V once before forking; all lanes consume
            # the same physical cache through their original request page tables.
            backend.impl._layer_name = backend.layer_name
            backend.impl.reshape_and_cache(
                q.view(workload.capacity, attention.num_heads, attention.head_dim),
                k.view(workload.capacity, attention.num_kv_heads, attention.head_dim),
                v.view(workload.capacity, attention.num_kv_heads, attention.head_dim),
                backend.kv_cache,
                next(iter(data[2].values())),
                output.view(workload.capacity, attention.num_heads, attention.head_dim),
            )
        active = zip(entry["branches"], intervals) if partition else [(data, (0, workload.capacity))]
        if parallel:
            entry["fork"].record(root)
        for lane, (branch_data, (start, stop)) in enumerate(active):
            if "graph" in entry:
                entry["graph"].lane = lane
            stream = entry["streams"][lane] if parallel else root
            if parallel:
                stream.wait_event(entry["fork"])
            with torch.npu.stream(stream), context(branch_data):
                get_forward_context().capturing = capturing
                query = q[start:stop].view(stop - start, attention.num_heads, attention.head_dim)
                key = k[start:stop].view(stop - start, attention.num_kv_heads, attention.head_dim)
                value = v[start:stop].view(stop - start, attention.num_kv_heads, attention.head_dim)
                target = output[start:stop].view(stop - start, attention.num_heads, attention.head_dim)
                metadata = next(iter(branch_data[2].values()))
                if optimized:
                    # FIA's out variant already writes target; retain its task
                    # registration/update, avoiding wrapper self-copy kernels.
                    backend.impl.full_graph_fia(query, key, value, metadata, target)
                else:
                    backend.impl.forward(backend, query, key, value, backend.kv_cache, metadata, output=target)
                if pipeline:
                    # A finished request lane can project while the other FIA
                    # is still running. Keep QKV and both large FFN GEMMs shared.
                    lane_projected, _ = attention.o_proj(output[start:stop])
                    projected[start:stop].copy_(lane_projected)
                if pipeline_ffn:
                    lane_projected, _ = attention.o_proj(output[start:stop])
                    lane_norm, lane_residual = layer.post_attention_layernorm(lane_projected, hidden[start:stop])
                    final_hidden[start:stop].copy_(layer.mlp(lane_norm))
                    final_residual[start:stop].copy_(lane_residual)
                if parallel:
                    entry["joins"][lane].record(stream)
        if parallel:
            for join in entry["joins"]:
                root.wait_event(join)
        if pipeline_ffn:
            return final_hidden, final_residual
        if not pipeline:
            projected, _ = attention.o_proj(output)
        normalized, residual = layer.post_attention_layernorm(projected, hidden)
        return layer.mlp(normalized), residual

    def forward(mode, parallel=True, partition=True):
        result = forward_impl(mode, parallel, partition)
        if "restore" in entries[mode]:
            # Restore canonical request order inside the graph on both strategies.
            result = tuple(tensor.index_select(0, entries[mode]["restore"]) for tensor in result)
        return result

    checks = []

    def cpu_pair(pair, batch):
        return tuple(tensor[:batch].float().cpu().clone() for tensor in pair)

    def check(pair, expected, batch, name):
        observed = cpu_pair(pair, batch)
        for value, reference in zip(observed, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0.02, atol=0.02, msg=lambda detail: f"{name}: {detail}")
        checks.append(
            dict(
                name=name,
                batch=batch,
                passed=True,
                max_abs_errors=[(a - b).abs().max().item() for a, b in zip(observed, expected)],
            )
        )

    def eager(mode, batch, variant, partition=True):
        prepare(entries[mode], batch, variant)
        reset_kv()
        with context(entries[mode]["data"]):
            return cpu_pair(forward(mode, parallel=False, partition=partition), batch)

    # References use original ND weights and no stream limits, for both outputs.
    references = {}
    cross_structure = []
    reference_phase = True
    for batch, variant in dict.fromkeys([*args.reuse_cases, *args.dynamic_cases]):
        for mode in entries:
            references[mode, batch, variant] = eager(mode, batch, variant)
        check(
            eager("split_parallel", batch, variant, partition=False),
            references["padding", batch, variant],
            batch,
            "direct_backend_vs_official",
        )
        left, right = references["padding", batch, variant], references["split_parallel", batch, variant]
        cross_structure.append(
            dict(batch=batch, variant=variant, max_abs_errors=[(a - b).abs().max().item() for a, b in zip(left, right)])
        )
    reference_phase = False
    layout_changes = []
    if args.nz:
        for module in (attention.qkv_proj, attention.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj):
            before = torch_npu.get_npu_format(module.weight)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, ACL_FORMAT_FRACTAL_NZ)
            layout_changes.append(dict(before=before, after=torch_npu.get_npu_format(module.weight)))
        torch.npu.synchronize()

    for mode, entry in entries.items():
        prepare(entry, workload.batch, 0)
        reset_kv()
        stream, update = torch.npu.Stream(), torch.npu.Stream()
        if mode == "split_parallel" and args.lane_limits[0] and args.lane_limits[0] == args.lane_limits[1]:
            torch.npu.set_stream_limit(update, cube_num=args.lane_limits[0], vector_num=2 * args.lane_limits[0])
        graph = UpdatableGraph() if mode == "padding" else BranchGraph()
        stream.wait_stream(main)
        with torch.npu.stream(stream), context(entry["data"]):
            for _ in range(3):
                forward(mode)
        stream.synchronize()
        entry["graph"] = graph
        pool = torch.npu.graph_pool_handle()
        with context(entry["data"]):
            get_forward_context().capturing = True
            with torch.npu.graph(graph, pool=pool, stream=stream):
                output = forward(mode)
        stream.synchronize()
        if len(graph.tasks) != (1 if mode == "padding" else 2):
            raise ValueError("unexpected FIA task count")
        pointers = [value.data_ptr() for value in graph.capture_resources.values()]
        if mode != "padding" and len(set(pointers)) != 2:
            raise ValueError("attention lanes must own distinct workspaces")
        source = ContextSource(entry["data"][2]) if mode == "padding" else BranchSource(entry)
        entry.update(
            source=source,
            pool=pool,
            output=output,
            stream=stream,
            update=update,
            done=torch.npu.Event(),
            updated=torch.npu.Event(),
            workspaces=len(set(pointers)),
        )
        separate = mode == "split_parallel" and (args.independent_updates or args.lane_limits[0] != args.lane_limits[1])
        entry["update_groups"] = []
        for lane in range(2 if separate else 1):
            update_stream = torch.npu.Stream() if separate else update
            if separate and args.lane_limits[lane]:
                torch.npu.set_stream_limit(
                    update_stream, cube_num=args.lane_limits[lane], vector_num=2 * args.lane_limits[lane]
                )
            entry["update_groups"].append(
                dict(
                    stream=update_stream,
                    indices=(lane,)
                    if separate
                    else task_submission_order(
                        (workload.capacity,) if mode == "padding" else args.pair, args.large_lane_first
                    ),
                    updated=torch.npu.Event(),
                )
            )
        if separate and args.large_lane_first:
            entry["update_groups"].sort(key=lambda group: -args.pair[group["indices"][0]])
        if native is not None:
            tasks = graph.resolve_tasks(source)
            if any(task.operation != torch_npu.npu_fused_infer_attention_score.out for task in tasks):
                raise ValueError("native submission requires exclusively FIA v1 out tasks")
            for task in tasks:
                if any(
                    set(values) - {"actual_seq_lengths", "actual_seq_lengths_kv", "block_table"}
                    for values in source.get(task.provider)
                ):
                    raise ValueError("unsupported dynamic FIA arguments for native submission")
            entry["native"] = True
            for group in entry["update_groups"]:
                group["native"] = native.NativeLane(tuple(tasks[i] for i in group["indices"]), group["stream"])
            if args.native_replay:
                entry["timing_events"] = (
                    (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                    if args.phase_diagnostics
                    else ()
                )
                entry["native_replay"] = native.NativeReplay(
                    graph, main, stream, [group["native"] for group in entry["update_groups"]], entry["timing_events"]
                )
                if args.parallel_host_update and mode == "split_parallel":
                    entry["native_replay"].enable_parallel_updates()
        print(f"captured {mode}: Dense={workload.capacity}, tasks={len(graph.tasks)}", flush=True)

    def identities():
        result = []
        for entry in entries.values():
            metadata = next(iter(entry["data"][2].values()))
            result.append(
                (
                    id(entry["graph"]),
                    *(t.data_ptr() for t in entry["data"][:2]),
                    metadata.block_tables.data_ptr(),
                    metadata.slot_mapping.data_ptr(),
                    metadata.seq_lens_cpu.data_ptr(),
                    metadata.seq_lens_gpu.data_ptr(),
                    *((entry["restore"].data_ptr(),) if "restore" in entry else ()),
                    *(value.data_ptr() for value in entry["graph"].capture_resources.values()),
                    *(
                        tensor.data_ptr()
                        for branch in entry["branches"]
                        for tensor in (
                            branch[0],
                            branch[1],
                            next(iter(branch[2].values())).seq_lens_gpu,
                            next(iter(branch[2].values())).block_tables,
                            next(iter(branch[2].values())).slot_mapping,
                        )
                    ),
                    *(t.data_ptr() for t in entry["output"]),
                )
            )
        return result

    captured_identities = identities()
    ready = torch.npu.Event()

    def replay(mode, batch, variant, profile=False, diagnostics=None, graph_events=None):
        entry = entries[mode]
        region = (
            (lambda name: phase_timer(name, diagnostics))
            if diagnostics is not None
            else (torch.profiler.record_function if profile else nullcontext)
        )
        with region("host.prepare_h2d_resolve_bind"):
            prepare(entry, batch, variant)
        if args.native_replay:
            with region("host.native_replay_update_join"):
                entry["native_replay"].replay(diagnostics is not None)
            return entry["output"]
        with region("host.replay_submit"):
            ready.record(main)
            entry["stream"].wait_event(ready)
            if graph_events is not None:
                graph_events[0].record(entry["stream"])
            with torch.npu.stream(entry["stream"]):
                entry["graph"].replay()
        with region("host.update_submit"):
            for group in entry["update_groups"]:
                group["stream"].wait_event(ready)
            if native is None:
                for group in entry["update_groups"]:
                    entry["graph"].update(group["stream"], tuple(entry["tasks"][i] for i in group["indices"]))
            else:
                native.submit([group["native"] for group in entry["update_groups"]], 1)
        with region("host.join_submit"):
            for group in entry["update_groups"]:
                group["updated"].record(group["stream"])
                entry["stream"].wait_event(group["updated"])
            entry["done"].record(entry["stream"])
            if graph_events is not None:
                graph_events[1].record(entry["stream"])
            main.wait_event(entry["done"])
        return entry["output"]

    qualification_cases = args.reuse_cases + [
        case for case in dict.fromkeys(args.dynamic_cases) if case not in args.reuse_cases
    ]
    for index, (batch, variant) in enumerate(qualification_cases):
        for mode in entries:
            reset_kv()
            check(replay(mode, batch, variant), references[mode, batch, variant], batch, f"reuse/{mode}/{index}")
    if captured_identities != identities():
        raise ValueError("dynamic reuse changed graph or buffer identity")
    measurements = []
    for mode in entries:
        for _ in range(5):
            replay(mode, workload.batch, 0)
            torch.npu.synchronize()
    for schedule, cases in (("fixed", [(workload.batch, 0)]), ("dynamic", args.dynamic_cases)):
        for round_id in range(args.rounds):
            for mode in tuple(entries) if round_id % 2 == 0 else tuple(reversed(entries)):
                reset_kv()
                samples = []
                for repeat in range(args.repeats):
                    batch, variant = cases[repeat % len(cases)]
                    started = time.perf_counter_ns()
                    output = replay(mode, batch, variant)
                    torch.npu.synchronize()
                    samples.append((time.perf_counter_ns() - started) / 1000)
                check(output, references[mode, batch, variant], batch, f"timed/{schedule}/{mode}/{round_id}")
                measurements.append(
                    dict(
                        schedule=schedule,
                        mode=mode,
                        round=round_id,
                        wall_us=samples,
                        summary_us=summarize(samples),
                        total_us=sum(samples),
                    )
                )
                print(f"{schedule}/{mode}/{round_id}: {summarize(samples)['p50']:.2f} us", flush=True)
    if captured_identities != identities():
        raise ValueError("timed replay changed graph, workspace or buffer identity")
    phase_diagnostics = []
    if args.phase_diagnostics:
        # Separate, unprofiled diagnostics. Event span includes graph waits and
        # update readiness; it is not pure kernel time or subtracted from wall.
        for mode in entries:
            events = entry_events = entries[mode].get("timing_events")
            if not entry_events:
                events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            for repeat in range(32):
                batch, variant = args.dynamic_cases[repeat % len(args.dynamic_cases)]
                row = dict(mode=mode, batch=batch)
                started = time.perf_counter_ns()
                output = replay(mode, batch, variant, diagnostics=row, graph_events=events)
                with phase_timer("host.final_sync", row):
                    torch.npu.synchronize()
                row["wall_us"] = (time.perf_counter_ns() - started) / 1000
                row["graph_plus_update_ready_span_us"] = events[0].elapsed_time(events[1]) * 1000
                phase_diagnostics.append(row)
            check(output, references[mode, batch, variant], batch, f"diagnostic/{mode}")
    trace_rows = []
    if args.trace:
        from vllm_ascend.worker.multi_graph.replay_benchmark import inspect_shared_dense_trace

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for mode in entries:
                marker = "layer.single_padded" if mode == "padding" else "layer." + mode
                for _ in range(3):
                    with torch.profiler.record_function(marker):
                        replay(mode, workload.batch, 0, profile=True)
                        with torch.profiler.record_function("host.final_sync"):
                            torch.npu.synchronize()
        trace_path = args.output.with_suffix(".trace.json")
        profiler.export_chrome_trace(str(trace_path))
        trace_rows = inspect_shared_dense_trace(json.loads(trace_path.read_text()))
        if len(trace_rows) != 3 * len(entries) or any(
            row["matmul_count"]
            != (
                (7 if args.pipeline_ffn else 5)
                if (args.pipeline_ffn or args.pipeline_output_projection) and "single_" not in row["marker"]
                else 4
            )
            for row in trace_rows
        ):
            raise ValueError("unexpected trace marker or MatMul count")
    timings = {
        schedule: {
            mode: statistics.median(
                row["summary_us"]["p50"] for row in measurements if row["mode"] == mode and row["schedule"] == schedule
            )
            / 1000
            for mode in entries
        }
        for schedule in ("fixed", "dynamic")
    }
    report = dict(
        status="passed",
        scope="first_decoder_block_with_embedding_not_full_decoder_or_serving",
        selection=args.selection,
        pair=args.pair,
        dense_capacity=workload.capacity,
        fixed_boundary=args.pair[0],
        reuse_cases=args.reuse_cases,
        qualification_cases=qualification_cases,
        timed_dynamic_cases=args.dynamic_cases,
        timed_kv_length=workload.kv_length,
        request_order=args.request_order,
        request_lengths=args.request_lengths,
        canonical_original_order_reference=True,
        snapshot_first_block_only=args.snapshot_first_block_only,
        output_restore_gathers=2 if args.request_order != "original" else 0,
        timed_request_metadata=[
            dict(
                batch=batch,
                variant=variant,
                **ordered_request_metadata(workload, args.request_lengths, batch, variant, args.request_order),
            )
            for batch, variant in dict.fromkeys(args.dynamic_cases)
        ]
        if args.short_kv_length is not None
        else [],
        real_lane_counts=[fixed_lane_counts(batch, args.pair) for batch, _ in args.reuse_cases],
        graph_count=len(entries),
        serial_control=args.serial_control,
        shared_cache=args.shared_cache,
        padding_backend="direct_fia_shared_cache" if args.shared_cache else "official_layer",
        lane_core_limits=args.lane_limits,
        independent_updates=args.independent_updates,
        graphs_per_strategy_per_bucket=1,
        combination_graph_count=0,
        reuse_graph_and_buffer_identity_unchanged=True,
        explicit_output_merge_copies=4 if args.pipeline_ffn else (2 if args.pipeline_output_projection else 0),
        attention_cores=args.attention_cores,
        native_update=args.native_update,
        native_replay=args.native_replay,
        parallel_host_update=args.parallel_host_update,
        packed_metadata=args.packed_metadata,
        h2d_submissions_per_step=1 if args.packed_metadata else (6 if args.request_order != "original" else 5),
        packed_buffer_schema=entries["padding"]["packed"].schema if args.packed_metadata else None,
        phase_diagnostics=phase_diagnostics,
        static_task_plan=args.static_task_plan,
        pipeline_output_projection=args.pipeline_output_projection,
        pipeline_ffn=args.pipeline_ffn,
        large_lane_first=args.large_lane_first,
        update_task_order=[
            dict(mode=mode, groups=[g["indices"] for g in e["update_groups"]]) for mode, e in entries.items()
        ],
        final_assembly_copies_per_split_replay=4 if args.pipeline_ffn else 0,
        projection_assembly_copies_per_split_replay=2 if args.pipeline_output_projection else 0,
        native_source_sha256=native_source_sha256,
        nz=args.nz,
        layout_changes=layout_changes,
        original_nd_eager_reference=True,
        cross_shape_qualified=False,
        cross_structure=cross_structure,
        graphs=[
            dict(
                mode=mode,
                capacity=workload.capacity,
                update_tasks=len(e["graph"].tasks),
                update_streams=len(e["update_groups"]),
                workspaces=e["workspaces"],
                query_layouts=[
                    dict(shape=list(t.kwargs["query"].shape), stride=list(t.kwargs["query"].stride()))
                    for t in e["graph"].tasks
                ],
            )
            for mode, e in entries.items()
        ],
        metric="prepare_h2d_resolve_update_embedding_block_dependencies_sync",
        checks=checks,
        measurements=measurements,
        summary_ms=timings,
        trace_analysis=trace_rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loader_source_sha256=hashlib.sha256(Path(__file__).with_name("multi_graph_replay.py").read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(timings), flush=True)


def main():
    from multi_graph_default_capture import select_case
    from multi_graph_replay import run

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--batch-min", type=int, default=64)
    parser.add_argument("--batch-max", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--short-kv-length", type=int, help="seeded mixed short/long requests; maximum is --kv-length")
    parser.add_argument("--snapshot-first-block-only", action="store_true", help="snapshot only the measured block KV")
    parser.add_argument("--request-order", choices=("original", "length_asc", "length_desc"), default="original")
    parser.add_argument("--attention-cores", type=int, default=0)
    parser.add_argument("--lane-cores", type=int, nargs=2)
    parser.add_argument("--independent-updates", action="store_true")
    parser.add_argument("--native-update", action="store_true")
    parser.add_argument("--native-replay", action="store_true", help="native replay/update/event orchestration")
    parser.add_argument(
        "--parallel-host-update", action="store_true", help="persistent C++ workers submit lane updates"
    )
    parser.add_argument("--static-task-plan", action="store_true", help="bind only KV lengths in a fixed native plan")
    parser.add_argument(
        "--packed-metadata", action="store_true", help="fill fixed pinned views and upload once per step"
    )
    parser.add_argument("--phase-diagnostics", action="store_true", help="separate unprofiled host and event timings")
    parser.add_argument("--serial-control", action="store_true", help="also capture identical split attention serially")
    parser.add_argument("--branch-capacities", type=int, nargs=2, help="ordered default capacities summing to bucket")
    parser.add_argument("--pipeline-ffn", action="store_true", help="pipeline complete lane O/Norm/FFN after FIA")
    parser.add_argument("--large-lane-first", action="store_true", help="submit larger FIA branch update first")
    parser.add_argument("--shared-cache", action="store_true", help="symmetric shared KV write and direct FIA out path")
    parser.add_argument(
        "--pipeline-output-projection", action="store_true", help="pipeline lane O projections with FIA"
    )
    parser.add_argument("--nz", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.batch_min <= args.batch_max <= args.max_num_seqs:
        parser.error("invalid batch range")
    if args.kv_length < 2 or min(args.rounds, args.repeats) < 1 or args.attention_cores < 0:
        parser.error("invalid length, sample count or core limit")
    if args.repeats % 4:
        parser.error("repeats must be a multiple of four to cover equal dynamic cycles")
    if args.static_task_plan and not args.native_update:
        parser.error("static task plans require native update")
    if args.native_replay and not args.native_update:
        parser.error("native replay requires native update")
    if args.parallel_host_update and (not args.native_replay or not args.independent_updates):
        parser.error("parallel host updates require native replay and independent update streams")
    if args.pipeline_output_projection and not args.shared_cache:
        parser.error("projection pipeline requires the qualified shared-cache path")
    if args.pipeline_ffn and (not args.shared_cache or args.pipeline_output_projection):
        parser.error("FFN pipeline requires shared cache and excludes the O-only pipeline")
    if args.request_order != "original" and args.short_kv_length is None:
        parser.error("request sorting requires mixed lengths")
    if args.short_kv_length is not None and not 2 <= args.short_kv_length <= args.kv_length:
        parser.error("short KV length must be between 2 and the maximum KV length")
    try:
        args.lane_limits = lane_core_limits(args.attention_cores, args.lane_cores)
    except ValueError as error:
        parser.error(str(error))
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        parser.error("requires ordinary FIA")
    args.dtype, args.batch_invariant = "bfloat16", False
    args.selection = select_case(args)
    capacity, args.pair = bucket_topology(
        args.selection["batch"], args.selection["capture_sizes"], args.branch_capacities
    )
    if capacity != args.selection["selected_requests"]:
        raise ValueError("bucket does not match the default dispatcher")
    args.selection["split_shapes"] = "fixed_default_bucket_internal_attention_branches"
    args.selection["scope"] = "one_block_default_bucket_reuse_not_compiled_serving"
    args.reuse_cases = reuse_cases(args.selection["batch"], args.selection["capture_sizes"])
    # Shortening a request overwrites prefill KV that a later longer request reads.
    # Qualify changed lengths with restored prefixes; time batch changes at fixed KV.
    args.dynamic_cases = [(batch, 0) for batch, _ in args.reuse_cases]
    workload = ReplayWorkload(
        args.selection["batch"], capacity, 1, args.kv_length, fixed_lane_counts(args.selection["batch"], args.pair)
    )
    args.request_lengths = request_lengths(capacity, args.kv_length, args.short_kv_length or args.kv_length, args.seed)
    if args.output.exists():
        parser.error("output exists; use a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="running")), encoding="utf-8")
    try:
        run(args, workload, experiment=experiment, snapshot_first_block=args.snapshot_first_block_only)
    except Exception as error:
        args.output.write_text(
            json.dumps(
                dict(
                    status="failed",
                    error=str(error),
                    selection=args.selection,
                    pair=args.pair,
                    attention_cores=args.attention_cores,
                )
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
