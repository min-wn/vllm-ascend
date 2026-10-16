"""Two complete decoder graphs restricted to default capture buckets.

Measures host preparation, buffer copies, FIA updates, parallel replay and merge.
This controlled decoder benchmark does not reproduce the compiled serving engine.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path


def bucket_pair(batch, sizes, strategy):
    sizes = sorted(set(sizes))
    capacity = next((size for size in sizes if size >= batch), None)
    if capacity is None or batch < 2:
        raise ValueError("batch must fit the default capture range and have at least two requests")
    candidates = [(a, b) for a in sizes for b in sizes if a >= b and batch <= a + b <= capacity]
    if strategy == "exact":
        candidates = [pair for pair in candidates if sum(pair) == batch]
    if not candidates:
        raise ValueError("no legal pair; do not introduce a custom capture size")
    return capacity, min(candidates, key=lambda pair: (abs(pair[0] - pair[1]), sum(pair)))


def lane_counts(batch, pair):
    if not 2 <= batch <= sum(pair):
        raise ValueError("batch must fit both nonempty lanes")
    left = min(pair[0], max((batch + 1) // 2, batch - pair[1]))
    return left, batch - left


def update_groups(task_counts, chunk_size):
    """Preserve each graph's update order while releasing both lanes early."""
    if chunk_size < 0 or not task_counts or min(task_counts) < 1:
        raise ValueError("nonnegative chunk size and positive task counts required")
    width = chunk_size or max(task_counts)
    return [
        (lane, start, min(start + width, count))
        for start in range(0, max(task_counts), width)
        for lane, count in enumerate(task_counts)
        if start < count
    ]


def experiment(args, workload, runner, model, inputs, context, reset_kv):
    # Capture fixed buffers once; only their contents and host parameters change.
    import torch
    from vllm.forward_context import get_forward_context

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
    from vllm_ascend.worker.multi_graph.benchmark import summarize

    main = torch.npu.current_stream()
    layer_count = len(model.model.layers)
    entries = []
    checks = []
    profiling = False

    def phase(name):
        return torch.profiler.record_function(f"bucket_phase.{name}") if profiling else nullcontext()

    limits = torch.npu.get_device_limit(torch.npu.current_device()) if any(args.lane_cores) else None
    if limits and any(
        cores > limits["cube_core_num"] or 2 * cores > limits["vector_core_num"] for cores in args.lane_cores
    ):
        raise ValueError("lane core limit exceeds device capacity")

    def forward(data, capturing=False):
        with context(data):
            get_forward_context().capturing = capturing
            return model(input_ids=data[0], positions=data[1])

    def prepare(entry, start, count, variant=0):
        capacity = entry["capacity"]
        data = entry["data"]
        metadata = next(iter(data[2].values()))
        padding = capacity - count
        values = workload.metadata_values(start, start + count)
        # Change real lengths as well as IDs in reuse checks to exercise FIA update.
        lengths = [workload.kv_length - variant] * count + [0] * padding
        positions = [workload.kv_length - variant - 1] * count + [0] * padding
        slots = [slot - variant for slot in values["slots"]] + [-1] * padding
        pages = values["pages"] + [[0] * workload.pages_per_request for _ in range(padding)]

        def copy(name, target, values):
            source = torch.tensor(values, dtype=target.dtype)
            if args.pinned_inputs:
                staging = entry["host"][name]
                staging.copy_(source)
                target.copy_(staging, non_blocking=True)
            else:
                target.copy_(source)

        copy("ids", data[0], list(range(100 + start + variant, 100 + start + count + variant)) + [0] * padding)
        copy("positions", data[1], positions)
        metadata.seq_lens_list = lengths
        metadata.seq_lens_cpu.copy_(torch.tensor(lengths, dtype=torch.int32))
        if args.pinned_inputs:
            entry["host"]["lengths"].copy_(metadata.seq_lens_cpu)
            metadata.seq_lens_gpu.copy_(entry["host"]["lengths"], non_blocking=True)
        else:
            metadata.seq_lens_gpu.copy_(metadata.seq_lens_cpu)
        copy("pages", metadata.block_tables, pages)
        copy("slots", metadata.slot_mapping, slots)
        # Query extents remain at captured capacity; dummy KV lengths are zero.
        if "graph" in entry:
            entry["tasks"] = entry["graph"].resolve_tasks(ContextSource(data[2]))

    capacities = (workload.capacity, *args.pair)
    first_counts = lane_counts(workload.batch, args.pair)
    for index, capacity in enumerate(capacities):
        entry = dict(capacity=capacity, data=inputs(0, capacity))
        if args.pinned_inputs:
            metadata = next(iter(entry["data"][2].values()))
            targets = dict(
                ids=entry["data"][0],
                positions=entry["data"][1],
                lengths=metadata.seq_lens_gpu,
                pages=metadata.block_tables,
                slots=metadata.slot_mapping,
            )
            # Storage stays alive until replay's final sync, before the next refill.
            entry["host"] = {name: torch.empty_like(t, device="cpu", pin_memory=True) for name, t in targets.items()}
        start = 0 if index < 2 else first_counts[0]
        count = workload.batch if index == 0 else first_counts[index - 1]
        prepare(entry, start, count)
        reset_kv()
        stream, update = torch.npu.Stream(), torch.npu.Stream()
        cores = 0 if index == 0 else args.lane_cores[index - 1]
        if cores:
            # Capture and FIA update must use identical tiling resource limits.
            for limited in (stream, update):
                torch.npu.set_stream_limit(limited, cube_num=cores, vector_num=2 * cores)
        stream.wait_stream(main)
        with torch.npu.stream(stream):
            for _ in range(3):
                forward(entry["data"])
        stream.synchronize()
        graph, pool = UpdatableGraph(), torch.npu.graph_pool_handle()
        with context(entry["data"]):
            get_forward_context().capturing = True
            with torch.npu.graph(graph, pool=pool, stream=stream):
                output = model(input_ids=entry["data"][0], positions=entry["data"][1])
        stream.synchronize()
        if len(graph.tasks) != layer_count:
            raise ValueError("expected one FIA update per layer in each complete decoder graph")
        entry.update(
            graph=graph,
            pool=pool,
            output=output,
            stream=stream,
            update=update,
            done=torch.npu.Event(),
            updated=torch.npu.Event(),
        )
        entries.append(entry)
        print(f"captured complete decoder capacity={capacity}", flush=True)

    merged = torch.empty(
        (workload.batch, entries[0]["output"].shape[-1]), device="npu", dtype=entries[0]["output"].dtype
    )
    ready = torch.npu.Event()
    identities = [(id(e["graph"]), e["data"][0].data_ptr(), e["output"].data_ptr()) for e in entries]

    def prepare_mode(mode, batch, variant=0):
        if mode == "padding":
            prepare(entries[0], 0, batch, variant)
            return [entries[0]], (batch,)
        counts = lane_counts(batch, args.pair)
        if mode.startswith("lane"):
            lane = int(mode[-1])
            entry = entries[lane + 1]
            prepare(entry, 0 if lane == 0 else counts[0], counts[lane], variant)
            return [entry], (counts[lane],)
        prepare(entries[1], 0, counts[0], variant)
        prepare(entries[2], counts[0], counts[1], variant)
        return entries[1:], counts

    def replay(mode, batch, variant=0):
        with phase("prepare"):
            active, counts = prepare_mode(mode, batch, variant)
        ready.record(main)
        # Submit both graphs before their updates; captured external events gate FIA.
        with phase("submit"):
            for entry in active:
                entry["stream"].wait_event(ready)
                with torch.npu.stream(entry["stream"]):
                    entry["graph"].replay()
        for entry in active:
            entry["update"].wait_event(ready)
        for lane, start, stop in update_groups([len(e["tasks"]) for e in active], args.update_chunk_size):
            entry = active[lane]
            with phase(f"update_lane{lane}"):
                entry["graph"].update(entry["update"], entry["tasks"][start:stop])
        for entry in active:
            entry["updated"].record(entry["update"])
            entry["stream"].wait_event(entry["updated"])
            entry["done"].record(entry["stream"])
            main.wait_event(entry["done"])
        if len(active) == 1:
            return active[0]["output"][: counts[0]]
        with phase("merge"):
            torch.cat([entry["output"][:count] for entry, count in zip(active, counts)], out=merged[:batch])
        return merged[:batch]

    references = {}
    for batch, variant in ((workload.batch, 0), (workload.batch - 1, 1), (workload.batch, 0)):
        for mode in ("padding", "split_parallel"):
            active, counts = prepare_mode(mode, batch, variant)
            reset_kv()
            expected = torch.cat([forward(e["data"])[:count].float().cpu() for e, count in zip(active, counts)])
            reset_kv()
            actual = replay(mode, batch, variant).float().cpu()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            checks.append(
                dict(
                    mode=mode,
                    batch=batch,
                    variant=variant,
                    max_abs_error=(actual - expected).abs().max().item(),
                    passed=True,
                )
            )
            references[mode, batch, variant] = expected
    if identities != [(id(e["graph"]), e["data"][0].data_ptr(), e["output"].data_ptr()) for e in entries]:
        raise ValueError("reuse check changed graph or buffer identity")

    measurements = []
    for mode in ("padding", "split_parallel"):
        for _ in range(5):
            replay(mode, workload.batch)
            torch.npu.synchronize()
    for round_id in range(args.rounds):
        for mode in ("padding", "split_parallel") if round_id % 2 == 0 else ("split_parallel", "padding"):
            reset_kv()
            wall = []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                result = replay(mode, workload.batch)
                torch.npu.synchronize()
                wall.append((time.perf_counter_ns() - started) / 1000)
            torch.testing.assert_close(result.float().cpu(), references[mode, workload.batch, 0], rtol=0.02, atol=0.02)
            measurements.append(dict(mode=mode, round=round_id, wall_us=wall, wall_summary_us=summarize(wall)))
            print(f"{mode} round={round_id}: {summarize(wall)['p50']:.2f} us", flush=True)

    timing = {
        mode: statistics.median(r["wall_summary_us"]["p50"] for r in measurements if r["mode"] == mode) / 1000
        for mode in ("padding", "split_parallel")
    }
    lane_diagnostics = []
    if args.diagnose_lanes:
        # Isolated lanes diagnose contention; they are not alternative benchmark modes.
        for lane in range(2):
            mode = f"lane{lane}"
            reset_kv()
            for _ in range(5):
                replay(mode, workload.batch)
                torch.npu.synchronize()
            wall = []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                result = replay(mode, workload.batch)
                torch.npu.synchronize()
                wall.append((time.perf_counter_ns() - started) / 1000)
            offset = 0 if lane == 0 else first_counts[0]
            expected = references["split_parallel", workload.batch, 0][offset : offset + first_counts[lane]]
            torch.testing.assert_close(result.float().cpu(), expected, rtol=0.02, atol=0.02)
            lane_diagnostics.append(dict(lane=lane, wall_summary_us=summarize(wall)))
    if args.trace:
        import torch_npu

        profiling = True
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        ) as profiler:
            for mode in (
                ("padding", "split_parallel", "lane0", "lane1")
                if args.diagnose_lanes
                else ("padding", "split_parallel")
            ):
                for _ in range(3):
                    with torch.profiler.record_function(f"buckets.{mode}"):
                        replay(mode, workload.batch)
                        with phase("synchronize"):
                            torch.npu.synchronize()
        profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        profiling = False
    report = dict(
        status="passed",
        selection=args.selection,
        pair=args.pair,
        real_lane_counts=first_counts,
        metric="host_prepare_copy_resolve_update_full_decoder_replay_merge_sync",
        compiled_serving_qualified=False,
        cross_shape_qualified=False,
        graphs=[dict(capacity=e["capacity"], update_tasks=len(e["graph"].tasks)) for e in entries],
        graph_count=3,
        combination_graph_count=0,
        reuse_graph_and_buffer_identity_unchanged=True,
        checks=checks,
        measurements=measurements,
        summary_ms=timing,
        gain_percent=100 * (1 - timing["split_parallel"] / timing["padding"]),
        lane_cores=args.lane_cores,
        update_chunk_size=args.update_chunk_size,
        pinned_inputs=args.pinned_inputs,
        lane_diagnostics=lane_diagnostics,
        source_sha256=args.source_sha256,
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary_ms"]), flush=True)


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
    parser.add_argument("--strategy", choices=("exact", "balanced"), default="exact")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--lane-cores", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--update-chunk-size", type=int, default=0)
    parser.add_argument("--pinned-inputs", action="store_true")
    parser.add_argument("--diagnose-lanes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.batch_min <= args.batch_max <= args.max_num_seqs:
        parser.error("require 3 <= batch-min <= batch-max <= max-num-seqs")
    if min(args.rounds, args.repeats) < 1 or args.kv_length < 2:
        parser.error("positive counts and KV length >= 2 required")
    if min(args.lane_cores) < 0 or args.update_chunk_size < 0:
        parser.error("core limits and update chunk size must be nonnegative")
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        parser.error("requires ordinary FIA")
    args.dtype, args.batch_invariant = "bfloat16", False
    args.selection = select_case(args)
    capacity, args.pair = bucket_pair(args.selection["batch"], args.selection["capture_sizes"], args.strategy)
    assert capacity == args.selection["selected_requests"]
    args.selection["split_shapes"] = "complete_decoder_default_bucket_pair"
    args.selection["scope"] = "default_bucket_reuse_controlled_decoder_not_compiled_serving"
    args.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    workload = ReplayWorkload(
        args.selection["batch"], capacity, 1, args.kv_length, lane_counts(args.selection["batch"], args.pair)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output exists; choose a new path")
    args.output.write_text(
        json.dumps(dict(status="running", selection=args.selection, pair=args.pair)), encoding="utf-8"
    )
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
                    source_sha256=args.source_sha256,
                    lane_cores=args.lane_cores,
                    update_chunk_size=args.update_chunk_size,
                    pinned_inputs=args.pinned_inputs,
                )
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
