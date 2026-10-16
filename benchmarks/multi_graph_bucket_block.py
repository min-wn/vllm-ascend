"""One real decoder block with fixed default-bucket internal request parallelism.

Includes embedding, metadata preparation, required FIA updates and final sync.
This is a controlled block experiment, not a complete decoder or serving runner.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def bucket_topology(batch, sizes):
    sizes = sorted(set(sizes))
    if batch < 2 or not sizes or min(sizes) < 1:
        raise ValueError("positive default sizes and batch >= 2 required")
    capacity = next((size for size in sizes if size >= batch), None)
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
    main = torch.npu.current_stream()
    if args.attention_cores:
        limits = torch.npu.get_device_limit(torch.npu.current_device())
        if args.attention_cores > limits["cube_core_num"] or 2 * args.attention_cores > limits["vector_core_num"]:
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
    for mode in ("padding", "split_parallel"):
        data = inputs(0, workload.capacity)
        metadata = next(iter(data[2].values()))
        entry = dict(data=data, branches=[], streams=[], fork=torch.npu.Event(), joins=[])
        if mode == "split_parallel":
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
                if args.attention_cores:
                    torch.npu.set_stream_limit(
                        stream, cube_num=args.attention_cores, vector_num=2 * args.attention_cores
                    )
                entry["streams"].append(stream)
                entry["joins"].append(torch.npu.Event())
        targets = dict(
            ids=data[0],
            positions=data[1],
            lengths=metadata.seq_lens_gpu,
            pages=metadata.block_tables,
            slots=metadata.slot_mapping,
        )
        entry["host"] = {name: torch.empty_like(t, device="cpu", pin_memory=True) for name, t in targets.items()}
        entries[mode] = entry

    def prepare(entry, batch, variant):
        data = entry["data"]
        metadata = next(iter(data[2].values()))
        count = workload.capacity - batch
        values = workload.metadata_values(0, batch)
        lengths = [workload.kv_length - variant] * batch + [0] * count

        def copy(name, target, values):
            staging = entry["host"][name]
            staging.copy_(torch.tensor(values, dtype=target.dtype))
            target.copy_(staging, non_blocking=True)

        copy("ids", data[0], list(range(100 + variant, 100 + variant + batch)) + [0] * count)
        copy("positions", data[1], [workload.kv_length - variant - 1] * batch + [0] * count)
        copy("lengths", metadata.seq_lens_gpu, lengths)
        metadata.seq_lens_cpu.copy_(entry["host"]["lengths"])
        metadata.seq_lens_list = lengths
        copy("pages", metadata.block_tables, values["pages"] + [[0] * workload.pages_per_request for _ in range(count)])
        copy("slots", metadata.slot_mapping, [slot - variant for slot in values["slots"]] + [-1] * count)
        for branch, (start, stop) in zip(entry["branches"], intervals):
            next(iter(branch[2].values())).seq_lens_list = lengths[start:stop]
        if "graph" in entry:
            entry["tasks"] = entry["graph"].resolve_tasks(entry["source"])
            if "native" in entry:
                entry["native"].bind(entry["tasks"])

    def forward(mode, parallel=True, partition=True):
        entry = entries[mode]
        data = entry["data"]
        hidden = model.model.embed_tokens(data[0])
        if mode == "padding":
            return layer(data[1], hidden, None)
        normalized = layer.input_layernorm(hidden)
        qkv, _ = attention.qkv_proj(normalized)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        if hasattr(attention, "q_norm"):
            q = attention.q_norm(q.view(-1, attention.num_heads, attention.head_dim)).view(q.shape)
            k = attention.k_norm(k.view(-1, attention.num_kv_heads, attention.head_dim)).view(k.shape)
        q, k = attention.rotary_emb(data[1], q, k)
        output = torch.empty_like(q)
        root = torch.npu.current_stream()
        capturing = get_forward_context().capturing
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
                backend.impl.forward(
                    backend,
                    q[start:stop].view(stop - start, attention.num_heads, attention.head_dim),
                    k[start:stop].view(stop - start, attention.num_kv_heads, attention.head_dim),
                    v[start:stop].view(stop - start, attention.num_kv_heads, attention.head_dim),
                    backend.kv_cache,
                    next(iter(branch_data[2].values())),
                    output=output[start:stop].view(stop - start, attention.num_heads, attention.head_dim),
                )
                if parallel:
                    entry["joins"][lane].record(stream)
        if parallel:
            for join in entry["joins"]:
                root.wait_event(join)
        projected, _ = attention.o_proj(output)
        normalized, residual = layer.post_attention_layernorm(projected, hidden)
        return layer.mlp(normalized), residual

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
        if mode == "split_parallel" and args.attention_cores:
            torch.npu.set_stream_limit(update, cube_num=args.attention_cores, vector_num=2 * args.attention_cores)
        graph = BranchGraph() if mode == "split_parallel" else UpdatableGraph()
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
        if len(graph.tasks) != (2 if mode == "split_parallel" else 1):
            raise ValueError("unexpected FIA task count")
        pointers = [value.data_ptr() for value in graph.capture_resources.values()]
        if mode == "split_parallel" and len(set(pointers)) != 2:
            raise ValueError("attention lanes must own distinct workspaces")
        source = BranchSource(entry) if mode == "split_parallel" else ContextSource(entry["data"][2])
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
            entry["native"] = native.NativeLane(tasks, update)
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
                    *(t.data_ptr() for t in entry["output"]),
                )
            )
        return result

    captured_identities = identities()
    ready = torch.npu.Event()

    def replay(mode, batch, variant):
        entry = entries[mode]
        prepare(entry, batch, variant)
        ready.record(main)
        entry["stream"].wait_event(ready)
        with torch.npu.stream(entry["stream"]):
            entry["graph"].replay()
        entry["update"].wait_event(ready)
        if native is None:
            entry["graph"].update(entry["update"], entry["tasks"])
        else:
            native.submit([entry["native"]], 1)
        entry["updated"].record(entry["update"])
        entry["stream"].wait_event(entry["updated"])
        entry["done"].record(entry["stream"])
        main.wait_event(entry["done"])
        return entry["output"]

    for index, (batch, variant) in enumerate(args.reuse_cases):
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
    trace_rows = []
    if args.trace:
        from vllm_ascend.worker.multi_graph.replay_benchmark import inspect_shared_dense_trace

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for mode in entries:
                marker = "layer.single_padded" if mode == "padding" else "layer.joined_parallel"
                for _ in range(3):
                    with torch.profiler.record_function(marker):
                        replay(mode, workload.batch, 0)
                        torch.npu.synchronize()
        trace_path = args.output.with_suffix(".trace.json")
        profiler.export_chrome_trace(str(trace_path))
        trace_rows = inspect_shared_dense_trace(json.loads(trace_path.read_text()))
        if len(trace_rows) != 6 or any(row["matmul_count"] != 4 for row in trace_rows):
            raise ValueError("expected six trace markers with four MatMuls per block")
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
        timed_dynamic_cases=args.dynamic_cases,
        timed_kv_length=workload.kv_length,
        real_lane_counts=[fixed_lane_counts(batch, args.pair) for batch, _ in args.reuse_cases],
        graph_count=2,
        graphs_per_strategy_per_bucket=1,
        combination_graph_count=0,
        reuse_graph_and_buffer_identity_unchanged=True,
        explicit_output_merge_copies=0,
        attention_cores=args.attention_cores,
        native_update=args.native_update,
        native_source_sha256=native_source_sha256,
        nz=args.nz,
        layout_changes=layout_changes,
        original_nd_eager_reference=True,
        cross_shape_qualified=False,
        cross_structure=cross_structure,
        graphs=[
            dict(mode=mode, capacity=workload.capacity, update_tasks=len(e["graph"].tasks), workspaces=e["workspaces"])
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
    parser.add_argument("--attention-cores", type=int, default=0)
    parser.add_argument("--native-update", action="store_true")
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
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        parser.error("requires ordinary FIA")
    args.dtype, args.batch_invariant = "bfloat16", False
    args.selection = select_case(args)
    capacity, args.pair = bucket_topology(args.selection["batch"], args.selection["capture_sizes"])
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
    if args.output.exists():
        parser.error("output exists; use a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="running")), encoding="utf-8")
    try:
        run(args, workload, experiment=experiment)
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
