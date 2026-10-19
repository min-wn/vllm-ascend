# SPDX-License-Identifier: Apache-2.0
"""Audit the bounded request-lead experiment without rerunning NPU work."""

import argparse
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

from aclgraph_bounded_lead_probe import MODES, pair_samples
from aclgraph_full_path_budget import intersection, interval


def analyze_trace(path, mode):
    events = json.loads(path.read_bytes(), parse_float=Decimal)
    if isinstance(events, dict):
        events = events["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("ph") == "X" and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL_")),
        key=lambda e: interval(e)[0],
    )
    comm = sorted(
        (e for e in events if e.get("ph") == "X" and e["name"].startswith("hcom_allReduce")),
        key=lambda e: interval(e)[0],
    )
    mm = [e for e in kernels if "MatMul" in e["name"]]
    solves = [e for e in kernels if e["name"] == "q4_tile_solve_1"]
    commits = [e for e in kernels if e["name"] == "q4_log_deferred_0"]
    split = mode != "padding"
    expected = (10, 4, 2, 2) if split else (7, 3, 1, 1)
    observed = tuple(map(len, (mm, comm, solves, commits)))
    if mode not in MODES or observed != expected:
        raise ValueError(f"incomplete or unexpected endpoint inventory: {mode}, {observed}")
    down, inputs, out = (mm[1:3], mm[3:7], mm[7]) if split else (mm[1:2], mm[2:4], mm[4])
    root = mm[0]["tid"]
    begin, end = interval(down[-1])[1], interval(out)[0]
    # All main-stream work between the last previous Down and next O, including
    # input norms, projection, layout, recurrence, gated norm and explicit joins.
    body = [e for e in kernels if e["tid"] == root and begin <= interval(e)[0] < end]
    mats = [e for e in body if "MatMul" in e["name"]]
    other = [e for e in body if "MatMul" not in e["name"] and e["name"] != "q4_tile_solve_1"]

    def total(items):
        return float(sum((Decimal(str(e["dur"])) for e in items), Decimal(0)))

    row = dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        counts=dict(zip(("matmul", "all_reduce", "solve", "commit"), observed, strict=True)),
        previous_down_matmul_us=total(down),
        input_projection_us=total(mats),
        solve_us=total(solves),
        other_main_stream_body_us=total(other),
        commit_work_us=total(commits),
        first_collective_us=float(comm[0]["dur"]),
        input_gemm_before_previous_down_end_us=float(max(Decimal(0), begin - interval(inputs[0])[0])),
        body_inventory={
            name: dict(count=sum(e["name"] == name for e in body), sum_us=total([e for e in body if e["name"] == name]))
            for name in sorted({e["name"] for e in body})
        },
    )
    if row["input_gemm_before_previous_down_end_us"] > 1:
        raise ValueError("next input GEMM starts before previous Down completes")
    if split:
        window = [interval(comm[1])]
        row.update(
            last_down_collective_us=float(comm[1]["dur"]),
            input_a_overlap_last_collective_us=float(intersection(window, map(interval, inputs[:2]))),
            solve_a_overlap_last_collective_us=float(intersection(window, [interval(solves[0])])),
            body_overlap_last_collective_us=float(intersection(window, map(interval, body))),
            solve_a_start_after_last_collective_us=float(interval(solves[0])[0] - interval(comm[1])[1]),
            join_copy_us=total(
                [e for e in body if "ConcatD" in e["name"] and interval(e)[0] > interval(solves[-1])[1]]
            ),
        )
        if mode == "split_barrier" and interval(inputs[0])[0] + 1 < interval(comm[1])[1]:
            raise ValueError("barrier control starts input projection before final reduction")
    return row


def analyze(root):
    reports = [json.loads((root / "results" / f"probe.rank{rank}.json").read_bytes()) for rank in (0, 1)]
    paired = pair_samples([r["samples"] for r in reports])
    for report in reports:
        if report["status"] != "completed_closed_boundary_with_cross_shape_numerical_diagnostics":
            raise ValueError("incomplete experiment")
        if report["critical_rank_samples"] != paired:
            raise ValueError("stored critical-rank aggregation disagrees with raw samples")
        if len(report["checks"]) != 9 or not all(
            c["eager_replay_exact"] and c["inactive_null_exact"] for c in report["checks"]
        ):
            raise ValueError("qualification missing or failed")
    manifest = json.loads((root / "launch.json").read_bytes())
    for remote, sha in manifest["sources"].items():
        if hashlib.sha256((root / "sources" / Path(remote).name).read_bytes()).hexdigest() != sha:
            raise ValueError("staged source hash mismatch")
    rounds = sorted({s["round"] for s in paired})
    rows = []
    for trial in rounds:
        for metric in ("device_us", "wall_us"):
            medians = {
                mode: statistics.median(s[metric] for s in paired if s["round"] == trial and s["mode"] == mode)
                for mode in MODES
            }
            rows.append(
                dict(
                    round=trial,
                    metric=metric,
                    medians=medians,
                    early_saves_vs_barrier_us=medians["split_barrier"] - medians["split_early"],
                    early_added_vs_padding_us=medians["split_early"] - medians["padding"],
                    early_slowdown_vs_padding_pct=100 * (medians["split_early"] / medians["padding"] - 1),
                )
            )
    traces = []
    for rank, report in enumerate(reports):
        for profile in report["profiles"]:
            path = root / "results" / profile["file"]
            trace = analyze_trace(path, profile["mode"])
            if trace["sha256"] != profile["sha256"]:
                raise ValueError("trace hash mismatch")
            traces.append(dict(rank=rank, mode=profile["mode"], **trace))
    return dict(
        status="audited_closed_boundary_only",
        selection=reports[0]["selection"],
        per_round=rows,
        traces=traces,
        numerical_diagnostics=[dict(rank=i, values=r["numerical_diagnostics"]) for i, r in enumerate(reports)],
        limitations=[
            "Unprofiled per-sample slowest-rank measurements determine performance, not profile spans.",
            "Profiling has large first-collective rank arrival skew; do not use it to estimate transport cost.",
            "Kernel duration sums diagnose work but are not additive critical-path savings.",
            "Exact eager/replay and schedule agreement do not certify cross-shape model quality.",
            "Synthetic reduced-O input; no preceding FIA/update or full-model/serving result.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["per_round"], indent=2))


if __name__ == "__main__":
    main()
