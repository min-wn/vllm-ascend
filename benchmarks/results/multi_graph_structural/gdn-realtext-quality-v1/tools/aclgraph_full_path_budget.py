# SPDX-License-Identifier: Apache-2.0
"""Inventory full Qwen3.5 TP2 timelines; never infer tile readiness from spans."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def interval(event):
    start = Decimal(str(event["ts"]))
    return start, start + Decimal(str(event["dur"]))


def union(intervals):
    merged = []
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("reversed interval")
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def length(intervals):
    return sum((right - left for left, right in union(intervals)), Decimal(0))


def intersection(left, right):
    first, second = union(left), union(right)
    pairs, i, j = [], 0, 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        stop = min(first[i][1], second[j][1])
        if start < stop:
            pairs.append((start, stop))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return length(pairs)


def analyze_trace(path):
    document = json.loads(path.read_bytes(), parse_float=Decimal)
    events = document if isinstance(document, list) else document["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("ph") == "X" and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL_")),
        key=lambda e: interval(e)[0],
    )
    ai = [e for e in kernels if e["args"]["Task Type"] != "KERNEL_AICPU"]
    comm = sorted(
        (e for e in events if e.get("ph") == "X" and e["name"].startswith("hcom_allReduce")),
        key=lambda e: interval(e)[0],
    )
    mm = [e for e in kernels if "MatMul" in e["name"]]
    if len(mm) != 304 or len(comm) != 129:
        raise ValueError(f"expected one 64-layer TP2 replay: {len(mm)} matmuls, {len(comm)} reductions")
    rows, layers, cursor = [], [], 0
    for layer in range(64):
        fia = layer % 4 == 3
        names = ("QKV", "O", "GateUp", "Down") if fia else ("QKVZ", "BA", "O", "GateUp", "Down")
        assigned = dict(zip(names, mm[cursor : cursor + len(names)], strict=True))
        cursor += len(names)
        attn_name = "FusedInferAttentionScore" if fia else "q4_tile_solve_1"
        attention = [
            e
            for e in kernels
            if e["name"] == attn_name and interval(assigned[names[0]])[0] < interval(e)[0] < interval(assigned["O"])[0]
        ]
        if len(attention) != 1:
            raise ValueError(f"layer {layer}: ambiguous attention/GDN assignment")
        layers.append(
            dict(
                layer=layer,
                kind="FIA" if fia else "GDN",
                matrix_us={name: float(e["dur"]) for name, e in assigned.items()},
                attention_us=float(attention[0]["dur"]),
            )
        )
        for offset, label in enumerate(("O", "Down")):
            producer = assigned[label]
            collective = comm[1 + layer * 2 + offset]
            following = assigned["GateUp"] if label == "O" else (mm[cursor] if cursor < len(mm) else None)
            # Device trace clocks are allowed one microsecond of rounding skew.
            if interval(collective)[0] + 1 < interval(producer)[1]:
                raise ValueError("collective starts before its declared producer completes")
            if following is not None and interval(collective)[1] > interval(following)[0] + 1:
                raise ValueError("collective crosses the next dependent matrix operation")
            exposed = length([interval(collective)]) - intersection([interval(collective)], map(interval, ai))
            rows.append(
                dict(
                    layer=layer,
                    kind="FIA" if fia else "GDN",
                    boundary=label,
                    producer_us=float(producer["dur"]),
                    collective_us=float(collective["dur"]),
                    exposed_without_ai_us=float(exposed),
                    ideal_overlap_ceiling_us=float(min(Decimal(str(producer["dur"])), exposed)),
                )
            )
    work = defaultdict(list)
    for e in kernels:
        work[e["name"]].append(e)
    ai_intervals, comm_intervals = list(map(interval, ai)), list(map(interval, comm))
    origin = min(start for start, stop in ai_intervals + comm_intervals)
    end = max(stop for start, stop in ai_intervals + comm_intervals)
    ai_us, comm_us = length(ai_intervals), length(comm_intervals)
    both = intersection(ai_intervals, comm_intervals)
    commits = [e for e in kernels if e["name"] == "q4_log_deferred_0"]
    commit_intervals = list(map(interval, commits))
    other_ai = [interval(e) for e in ai if e["name"] != "q4_log_deferred_0"]
    commit_overlap_comm = intersection(commit_intervals, comm_intervals)
    commit_overlap_mm = intersection(commit_intervals, map(interval, mm))
    commit_exclusive = length(commit_intervals) - intersection(commit_intervals, other_ai + comm_intervals)
    markers = {e["name"]: float(e["dur"]) for e in events if e.get("ph") == "X" and e["name"].startswith("full_path/")}
    return dict(
        source=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        span_us=float(end - origin),
        ai_union_us=float(ai_us),
        communication_union_us=float(comm_us),
        communication_without_ai_us=float(comm_us - both),
        compute_communication_overlap_us=float(both),
        neither_ai_nor_communication_us=float(end - origin - ai_us - comm_us + both),
        host_markers_us=markers,
        work_inventory={
            name: dict(count=len(es), sum_us=float(sum(Decimal(str(e["dur"])) for e in es)))
            for name, es in sorted(work.items())
        },
        boundaries=rows,
        layers=layers,
        deferred_commit=dict(
            count=len(commits),
            union_us=float(length(commit_intervals)),
            overlap_communication_us=float(commit_overlap_comm),
            overlap_matmul_us=float(commit_overlap_mm),
            without_other_ai_or_communication_us=float(commit_exclusive),
        ),
        ideal_equal_tiles_saving_us={
            str(n): sum(r["ideal_overlap_ceiling_us"] for r in rows) * (1 - 1 / n) for n in (2, 4, 8)
        },
        limitations=[
            "Kernel sums are work, not critical-path elapsed time; interval unions avoid double counting.",
            "Communication without AI is exposure, not proof that arbitrary work can legally overlap it.",
            "Equal-tile savings assume proportional GEMM/collective durations, immediate tile readiness, "
            "no interference or added cost.",
            "Whole-kernel traces do not reveal per-row readiness. Rank budgets cannot be added.",
            "Profile durations are diagnostic only and must not be mixed with unprofiled endpoint timing.",
        ],
    )


def analyze_chunk_cost(root):
    ranks = [json.loads((root / f"chunk.rank{rank}.json").read_bytes()) for rank in (0, 1)]
    if any(r["status"] != "completed_transport_cost_only" or len(r["samples"]) != 480 for r in ranks):
        raise ValueError("require both completed transport screens")
    if ranks[0]["medians"] != ranks[1]["medians"]:
        raise ValueError("critical-rank aggregates disagree")
    for left, right in zip(ranks[0]["samples"], ranks[1]["samples"], strict=True):
        key = ("trial", "repetition", "capacity", "chunks")
        if any(left[k] != right[k] for k in key):
            raise ValueError("unpaired transport samples")
        if left["critical_rank_us"] != right["critical_rank_us"] or left["critical_rank_us"] != max(
            left["local_us"], right["local_us"]
        ):
            raise ValueError("invalid critical-rank reduction")
    rows = []
    for capacity in (400, 304, 496):
        medians = {
            n: statistics.median(
                s["critical_rank_us"] for s in ranks[0]["samples"] if s["capacity"] == capacity and s["chunks"] == n
            )
            for n in (1, 2, 4, 8)
        }
        for n in (1, 2, 4, 8):
            rounds = []
            for trial in range(4):
                times = {
                    k: statistics.median(
                        s["critical_rank_us"]
                        for s in ranks[0]["samples"]
                        if s["capacity"] == capacity and s["chunks"] == k and s["trial"] == trial
                    )
                    for k in (1, n)
                }
                rounds.append(times[n] - times[1])
            rows.append(
                dict(
                    capacity=capacity,
                    chunks=n,
                    median_us=medians[n],
                    added_us=medians[n] - medians[1],
                    round_added_us=rounds,
                )
            )
    return dict(
        samples=480,
        rank_reports=2,
        rows=rows,
        limitation="Isolated serial transport cost with common input copy. No concurrent GEMM; not model savings.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("diagnostic.step*.rank*.trace.json"))
    if len(paths) != 6:
        raise ValueError("require all three steps on both ranks")
    result = dict(
        status="diagnostic_inventory_and_ideal_ceiling_not_proven_speedup", traces=[analyze_trace(p) for p in paths]
    )
    if args.chunk_root is not None:
        result["transport_cost"] = analyze_chunk_cost(args.chunk_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
