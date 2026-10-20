"""Attribute synchronized bucket replays without counting overlapping kernels twice."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def union_us(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def intervals(events):
    return [(event["ts"], event["ts"] + event["dur"]) for event in events]


def analyze(document):
    events = document if isinstance(document, list) else document["traceEvents"]
    hardware = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name" and e.get("args", {}).get("name") == "Ascend Hardware"
    }
    complete = [
        {**e, "ts": float(e["ts"]), "dur": float(e["dur"])}
        for e in events
        if e.get("ph") == "X" and float(e.get("dur", 0)) > 0
    ]
    rows = []
    for marker in complete:
        if marker["name"] not in ("buckets.padding", "buckets.split_parallel", "buckets.lane0", "buckets.lane1"):
            continue
        start, stop = marker["ts"], marker["ts"] + marker["dur"]
        chosen = [
            {**e, "ts": max(start, e["ts"]), "dur": min(stop, e["ts"] + e["dur"]) - max(start, e["ts"])}
            for e in complete
            if e["ts"] < stop and e["ts"] + e["dur"] > start
        ]
        kernels = [
            e for e in chosen if e["pid"] in hardware and e.get("args", {}).get("Task Type", "").startswith("KERNEL")
        ]
        phases = defaultdict(float)
        groups = defaultdict(list)
        streams = defaultdict(list)
        for e in chosen:
            if e["name"].startswith("bucket_phase."):
                phases[e["name"].removeprefix("bucket_phase.")] += e["dur"]
        for e in kernels:
            name = e["name"].lower()
            key = "FIA" if "fusedinferattention" in name else "MatMul" if "matmul" in name else "other"
            groups[key].append(e)
            streams[str(e["tid"])].append(e)
        lanes = {}
        for tid, values in streams.items():
            fia = [e for e in values if "fusedinferattention" in e["name"].lower()]
            if not fia:
                continue
            first, last = min(e["ts"] for e in values), max(e["ts"] + e["dur"] for e in values)
            lanes[tid] = dict(
                first_kernel_us=first - start,
                last_kernel_us=last - start,
                first_fia_us=min(e["ts"] for e in fia) - start,
                busy_us=union_us(intervals(values)),
                internal_gap_us=last - first - union_us(intervals(values)),
                fia_count=len(fia),
            )
        rows.append(
            dict(
                mode=marker["name"],
                wall_us=marker["dur"],
                host_phases_us=dict(phases),
                kernel_union_us=union_us(intervals(kernels)),
                kernel_overlap_us=sum(union_us(intervals(v)) for v in streams.values()) - union_us(intervals(kernels)),
                operators={
                    name: dict(
                        count=len(values),
                        sum_us=sum(e["dur"] for e in values),
                        union_us=union_us(intervals(values)),
                        overlap_us=max(0.0, sum(e["dur"] for e in values) - union_us(intervals(values))),
                    )
                    for name, values in groups.items()
                },
                lanes=lanes,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path")
    result = {str(path): analyze(json.loads(path.read_text(encoding="utf-8"))) for path in args.traces}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
