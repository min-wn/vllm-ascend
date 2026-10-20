# SPDX-License-Identifier: Apache-2.0
"""Decompose existing TP2 replay collectives without launching NPU experiments.

SDMA is a task classification, not proof of link bytes or memory bandwidth.
Only per-rank intervals are compared; the two device clocks are not aligned.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from aclgraph_full_path_budget import intersection, interval, length

SUBTASK_TYPES = ("SDMA", "Write Value", "Notify Wait")
EXPECTED_COUNTS = {"SDMA": 6, "Write Value": 4, "Notify Wait": 4}


def decompose(collective, subtasks):
    start, stop = interval(collective)
    selected = [e for e in subtasks if start <= interval(e)[0] and interval(e)[1] <= stop]
    counts = Counter(e["args"]["Task Type"] for e in selected)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected captured HCCL decomposition: {counts}")
    identities = {(e["pid"], e["args"]["Physic Stream Id"], e["args"]["Task Id"]) for e in selected}
    if len(identities) != 1:
        raise ValueError("subtasks from different HCCL device tasks")
    indices = [e["args"]["Subtask Id"] for e in selected]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate HCCL subtask")
    groups = {name: [interval(e) for e in selected if e["args"]["Task Type"] == name] for name in SUBTASK_TYPES}
    for i, name in enumerate(SUBTASK_TYPES):
        if any(intersection(groups[name], groups[other]) for other in SUBTASK_TYPES[i + 1 :]):
            raise ValueError("overlapping task classes cannot form an additive breakdown")
    covered = length([interval(e) for e in selected])
    return dict(
        collective_us=float(stop - start),
        sdma_us=float(length(groups["SDMA"])),
        notify_wait_us=float(length(groups["Notify Wait"])),
        write_value_us=float(length(groups["Write Value"])),
        unclassified_gap_us=float(stop - start - covered),
        counts=dict(counts),
        subtask_pattern=[
            dict(
                index=e["args"]["Subtask Id"],
                kind=e["args"]["Task Type"],
                start_us=float(interval(e)[0] - start),
                duration_us=float(e["dur"]),
            )
            for e in sorted(selected, key=lambda e: interval(e)[0])
        ],
    )


def summarize(rows):
    if not rows:
        raise ValueError("empty boundary inventory")
    fields = (
        "collective_us",
        "sdma_us",
        "notify_wait_us",
        "write_value_us",
        "unclassified_gap_us",
        "producer_us",
        "producer_to_collective_us",
        "collective_to_next_ai_us",
    )
    totals = {k: sum(row[k] for row in rows) for k in fields}
    return dict(
        count=len(rows),
        total_us=totals,
        median_us={k: statistics.median(row[k] for row in rows) for k in fields},
        sdma_share=totals["sdma_us"] / totals["collective_us"],
        notify_wait_share=totals["notify_wait_us"] / totals["collective_us"],
    )


def analyze(path):
    document = json.loads(path.read_bytes(), parse_float=Decimal)
    events = document if isinstance(document, list) else document["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X"]
    collectives = sorted((e for e in complete if e["name"].startswith("hcom_allReduce")), key=lambda e: interval(e)[0])
    kernels = sorted(
        (e for e in complete if e.get("args", {}).get("Task Type", "").startswith("KERNEL_")),
        key=lambda e: interval(e)[0],
    )
    matrices = [e for e in kernels if "MatMul" in e["name"]]
    subtasks = [e for e in complete if e.get("args", {}).get("Task Type") in SUBTASK_TYPES]
    if len(collectives) != 129 or len(matrices) != 304 or len(subtasks) != 129 * 14:
        raise ValueError("require a single complete model trace with the observed HCCL task pattern")
    rows, cursor = [], 0
    for layer in range(64):
        names = ("QKV", "O", "GateUp", "Down") if layer % 4 == 3 else ("QKVZ", "BA", "O", "GateUp", "Down")
        projections = dict(zip(names, matrices[cursor : cursor + len(names)], strict=True))
        cursor += len(names)
        for offset, name in enumerate(("O", "Down")):
            producer, collective = projections[name], collectives[1 + 2 * layer + offset]
            start, stop = interval(collective)
            if start < interval(producer)[1]:
                raise ValueError("collective precedes producer completion")
            following = [
                e
                for e in kernels
                if e["pid"] == producer["pid"] and e["tid"] == producer["tid"] and interval(e)[0] >= stop
            ]
            if not following:
                raise ValueError("missing downstream consumer")
            next_kernel = following[0]
            row = dict(
                layer=layer,
                boundary=name,
                kind="FIA" if layer % 4 == 3 else "GDN",
                elements=collective["args"]["count"],
                producer_us=float(producer["dur"]),
                producer_to_collective_us=float(start - interval(producer)[1]),
                collective_to_next_ai_us=float(interval(next_kernel)[0] - stop),
                next_ai=next_kernel["name"],
                **decompose(collective, subtasks),
            )
            rows.append(row)
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["boundary"]].append(row)
        by_kind[row["kind"] + "/" + row["boundary"]].append(row)
    return dict(
        source=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        embedding=decompose(collectives[0], subtasks),
        summaries={k: summarize(v) for k, v in sorted(by_kind.items())},
        boundaries=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("diagnostic.step*.rank*.trace.json"))
    if len(paths) != 6:
        raise ValueError("require all six preserved full-model profiles")
    result = dict(
        status="existing_profile_analysis_no_new_device_runs",
        traces=[analyze(p) for p in paths],
        limitations=[
            "SDMA durations include the recorded DMA tasks, not a measurement of pure interconnect transfer.",
            "No task byte counts, source/destination addresses or phase labels exist in these exported traces.",
            "Notify Wait measures explicit waits only; other transport tasks may contain hidden stalls.",
            "Unclassified gaps are not all CPU launch overhead; no direct causal attribution is made.",
            "Different rank timestamps are not aligned. Rank durations must not be summed.",
            "The first output tile readiness is not observable from these whole-kernel profiles.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
