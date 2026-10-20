"""Qwen3 decoder MatMul attribution validated by the per-layer MM/FIA sequence.

Only the four-projection, one-FIA-per-layer TP1 decoder is supported. Overlap
is correlation, not proof that one kernel caused another kernel's slowdown.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from multi_graph_bucket_trace import intervals, union_us


def classify_lane(kernels, layer_count):
    sequence = []
    for event in sorted(kernels, key=lambda e: e["ts"]):
        name = event["name"].lower()
        if "matmul" in name or "fusedinferattention" in name:
            sequence.append(("fia" if "fusedinferattention" in name else "mm", event))
    if [kind for kind, _ in sequence] != ["mm", "fia", "mm", "mm", "mm"] * layer_count:
        raise ValueError("trace does not match the validated Qwen3 projection/FIA sequence")
    return [
        dict(event, projection=("qkv", "fia", "o", "gate_up", "down")[index % 5], layer=index // 5)
        for index, (_, event) in enumerate(sequence)
    ]


def overlap_us(event, others):
    left, right = event["ts"], event["ts"] + event["dur"]
    return union_us(
        [
            (max(left, other["ts"]), min(right, other["ts"] + other["dur"]))
            for other in others
            if other["ts"] < right and other["ts"] + other["dur"] > left
        ]
    )


def analyze(document, layer_count=36):
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
        streams = defaultdict(list)
        for event in complete:
            if (
                event["pid"] in hardware
                and event.get("args", {}).get("Task Type", "").startswith("KERNEL")
                and start <= event["ts"] < stop
            ):
                streams[event["tid"]].append(event)
        lanes = {
            tid: classify_lane(values, layer_count)
            for tid, values in streams.items()
            if any("fusedinferattention" in e["name"].lower() for e in values)
        }
        expected = 2 if marker["name"] == "buckets.split_parallel" else 1
        if len(lanes) != expected:
            raise ValueError("unexpected decoder stream count")
        measurements = []
        other_operations = []
        for tid, values in lanes.items():
            other = [e for stream, ops in lanes.items() if stream != tid for e in ops]
            for event in values:
                measurements.append(
                    dict(
                        stream=tid,
                        layer=event["layer"],
                        projection=event["projection"],
                        duration_us=event["dur"],
                        start_us=event["ts"] - start,
                        other_fia_overlap_us=overlap_us(event, [e for e in other if e["projection"] == "fia"]),
                        other_mm_overlap_us=overlap_us(event, [e for e in other if e["projection"] != "fia"]),
                    )
                )
            for event in streams[tid]:
                if "matmul" in event["name"].lower() or "fusedinferattention" in event["name"].lower():
                    continue
                other_operations.append(
                    dict(
                        stream=tid,
                        name=event["name"],
                        duration_us=event["dur"],
                        other_fia_overlap_us=overlap_us(event, [e for e in other if e["projection"] == "fia"]),
                        other_mm_overlap_us=overlap_us(event, [e for e in other if e["projection"] != "fia"]),
                    )
                )
        rows.append(
            dict(
                mode=marker["name"],
                wall_us=marker["dur"],
                kernel_union_us=union_us(intervals([e for values in streams.values() for e in values])),
                measurements=measurements,
                other_operations=other_operations,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.layers < 1:
        parser.error("new output path and positive layer count required")
    args.output.write_text(
        json.dumps({str(p): analyze(json.loads(p.read_text()), args.layers) for p in args.traces}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
