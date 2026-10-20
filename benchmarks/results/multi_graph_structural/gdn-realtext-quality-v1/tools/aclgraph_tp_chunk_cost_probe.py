# SPDX-License-Identifier: Apache-2.0
"""Measure added TP2 all-reduce cost at existing model graph capacities.

This transport-only screen has no GEMM producer or tile-readiness protocol.
All variants move the same payload, refresh inputs and join before timing ends.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def request_chunks(capacity, chunks):
    if capacity <= 0 or capacity % 4 or not 1 <= chunks <= capacity // 4:
        raise ValueError("chunks must partition whole Q4 requests")
    count, remainder = divmod(capacity // 4, chunks)
    sizes = [4 * (count + (i < remainder)) for i in range(chunks)]
    return sizes


def run(args):
    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise ValueError("TP2 required")
    cpu_group = dist.new_group(backend="gloo")
    report = dict(
        status="running",
        rank=rank,
        full_model=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        samples=[],
        checks=[],
        capacities=args.capacities,
        limitation="transport-only cost; no ready producer, no prediction of end-to-end gain",
    )
    entries = {}
    for capacity in args.capacities:
        source = torch.full((capacity, 5120), rank + 1, dtype=torch.bfloat16, device="npu")
        value = torch.empty_like(source)
        for chunks in (1, 2, 4, 8):
            sizes = request_chunks(capacity, chunks)
            pieces = value.split(sizes, dim=0)

            def execute(value=value, source=source, pieces=pieces):
                value.copy_(source)
                for piece in pieces:
                    dist.all_reduce(piece)

            for _ in range(3):
                execute()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            with torch.npu.graph(graph, stream=stream):
                execute()
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(value, torch.full_like(value, 3), rtol=0, atol=0)
            entries[capacity, chunks] = (graph, value, source, stream)
            report["checks"].append(dict(capacity=capacity, chunks=chunks, rows=sizes, passed=True))
    for trial in range(4):
        order = list(entries) if trial % 2 == 0 else list(reversed(entries))
        for capacity, chunks in order:
            graph, value, source, stream = entries[capacity, chunks]
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
            for repetition in range(10):
                dist.barrier(group=cpu_group)
                start = time.perf_counter_ns()
                graph.replay()
                torch.npu.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
                critical = torch.tensor(elapsed, dtype=torch.float64)
                dist.all_reduce(critical, op=dist.ReduceOp.MAX, group=cpu_group)
                report["samples"].append(
                    dict(
                        trial=trial,
                        repetition=repetition,
                        capacity=capacity,
                        chunks=chunks,
                        local_us=elapsed,
                        critical_rank_us=critical.item(),
                    )
                )
            torch.testing.assert_close(value, torch.full_like(value, 3), rtol=0, atol=0)
    report["medians"] = {
        f"{cap}/{n}": statistics.median(
            s["critical_rank_us"] for s in report["samples"] if s["capacity"] == cap and s["chunks"] == n
        )
        for cap, n in entries
    }
    report["profiles"] = []
    for chunks in (1, 2, 4, 8):
        graph, *_ = entries[args.capacities[0], chunks]
        dist.barrier(group=cpu_group)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as prof:
            graph.replay()
            torch.npu.synchronize()
        path = args.output.with_name(f"{args.output.stem}.chunks{chunks}.rank{rank}.trace.json")
        prof.export_chrome_trace(str(path))
        report["profiles"].append(
            dict(chunks=chunks, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    report["status"] = "completed_transport_cost_only"
    args.output.with_suffix(f".rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", nargs="+", type=int, default=[400, 304, 496])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for capacity in args.capacities:
        request_chunks(capacity, 8)
    if args.output.with_suffix(f".rank{os.environ.get('LOCAL_RANK', '0')}.json").exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
