# SPDX-License-Identifier: Apache-2.0
"""Task-level critical-path accounting; never infer intrakernel overlap from spans."""

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def merged(intervals):
    output = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative profile interval")
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(end, output[-1][1]))
        elif end > start:
            output.append((start, end))
    return output


def length(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(left, right):
    left, right = merged(left), merged(right)
    first, second, total = 0, 0, 0.0
    while first < len(left) and second < len(right):
        total += max(0, min(left[first][1], right[second][1]) - max(left[first][0], right[second][0]))
        if left[first][1] <= right[second][1]:
            first += 1
        else:
            second += 1
    return total


def synchronization_counts(capacity, cores=20):
    if type(capacity) is not int or capacity % 4 or capacity < 4 or type(cores) is not int or cores < 1:
        raise ValueError("positive Q4 capacity and producer core count required")
    groups = capacity  # four groups per Q4 request, including null padding groups
    return dict(
        groups_including_padding=groups,
        delayed=dict(ready_set=cores, ready_wait=2 * cores, consumed_set=0, consumed_wait=0),
        ready=dict(ready_set=groups, ready_wait=2 * groups, consumed_set=2 * groups, consumed_wait=groups),
        scope="static instruction counts from unchanged mixed-kernel loops, not elapsed synchronization time",
    )


def analyze_trace(path, capacity):
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw if isinstance(raw, list) else raw["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X" and float(e.get("dur", 0)) > 0]
    origin = min(Decimal(str(e["ts"])) for e in complete)

    def interval(event):
        start = float(Decimal(str(event["ts"])) - origin)
        return start, start + float(event["dur"])

    kernels = [e for e in complete if e.get("args", {}).get("Task Type", "").startswith("KERNEL_")]
    communication = [e for e in complete if e.get("name", "").startswith("hcom_")]
    mixed = [e for e in kernels if "q4_tile_solve" in e["name"]]
    commits = [e for e in kernels if "q4_log_deferred" in e["name"]]
    matmuls = [e for e in kernels if "matmul" in e["name"].lower()]
    head = sorted(matmuls, key=lambda e: Decimal(str(e["ts"])))[-1:]
    head_id = {id(e) for e in head}
    dense = [e for e in matmuls if id(e) not in head_id]
    fia = [e for e in kernels if "fusedinferattention" in e["name"].lower()]
    sdma = [e for e in complete if e.get("args", {}).get("Task Type") in ("SDMA", "SDMA_SQE", "PCIE_DMA_SQE")]
    known = {id(e) for e in [*mixed, *commits, *matmuls, *fia]}
    other = [e for e in kernels if id(e) not in known]
    if len(mixed) != 48 or len(commits) != 48 or len(matmuls) != 305:
        raise ValueError(
            f"full chain task counts not identified: mixed={len(mixed)}, commit={len(commits)}, mm={len(matmuls)}"
        )
    groups = dict(
        gdn_mixed=mixed,
        state_commit=commits,
        dense_without_head=dense,
        head_projection=head,
        fia=fia,
        other_kernels=other,
        tp_communication=communication,
        dma=sdma,
    )
    summary = {
        name: dict(
            count=len(items), sum_us=sum(float(e["dur"]) for e in items), union_us=length([interval(e) for e in items])
        )
        for name, items in groups.items()
    }
    device = [interval(e) for e in [*kernels, *communication, *sdma]]
    markers = {e["name"]: float(e["dur"]) for e in complete if e.get("name", "").startswith("ready_chain/")}
    waits = [e for e in complete if e.get("args", {}).get("Task Type") in ("EVENT_WAIT", "Notify Wait", "NOTIFY_WAIT")]
    return dict(
        path=str(path),
        tasks=summary,
        device_span_us=max(end for _, end in device) - min(start for start, _ in device),
        device_busy_union_us=length(device),
        state_commit_dense_overlap_us=intersection([interval(e) for e in commits], [interval(e) for e in dense]),
        tp_kernel_overlap_us=intersection([interval(e) for e in communication], [interval(e) for e in kernels]),
        runtime_wait_tasks=dict(count=len(waits), sum_us=sum(float(e["dur"]) for e in waits)),
        host_markers_us=markers,
        synchronization=synchronization_counts(capacity),
        allgather=[
            dict(name=e["name"], duration_us=float(e["dur"]), args=e.get("args", {}))
            for e in communication
            if "allgather" in e["name"].lower()
        ],
        kernel_names=Counter(e["name"] for e in kernels),
        intrakernel_aic_aiv_overlap_us=None,
        cross_core_flag_wait_us=None,
        limitations=[
            "kernel task spans include work and internal waiting; not per-core execution intervals",
            "host final_wait includes outstanding device work; not an exclusive extra cost",
            "runtime event waits are distinct from intrakernel CrossCoreWaitFlag",
            "mixed-kernel task span does not expose Cube/AIV intersection or separate flag latency",
            "profile durations are not unprofiled timing samples",
        ],
    )


def metric_tables(directory):
    records = []
    for path in sorted(directory.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if not any(
                any(word in column.lower() for word in ("cube", "vec", "aicore", "aiv_time")) for column in columns
            ):
                continue
            matching = [row for row in reader if any("q4_tile_solve" in value for value in row.values() if value)]
        if matching:
            records.append(dict(path=str(path), columns=columns, matching_mixed_rows=matching))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--load", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for mode in ("delayed", "ready"):
        directory = args.directory / f"{args.load}.pairprofile.{mode}"
        for rank in (0, 1):
            report = json.loads((directory / f"result.rank{rank}.json").read_text(encoding="utf-8"))
            for profile in report["profiles"]:
                path = directory / Path(profile["path"]).name
                results.append(
                    dict(
                        mode=mode,
                        rank=rank,
                        step=profile["step"],
                        capacity=profile["capacity"],
                        **analyze_trace(path, profile["capacity"]),
                    )
                )
    output = dict(
        scope="isolated profiled fixed-trajectory steps, not latency samples",
        rows=results,
        raw_metric_tables=metric_tables(args.directory),
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: row[k] for k in ("mode", "rank", "step", "device_span_us", "tasks")} for row in results]))


if __name__ == "__main__":
    main()
