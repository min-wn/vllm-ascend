# SPDX-License-Identifier: Apache-2.0
"""Audit equal-fusion padding/split controls at the complete two-layer endpoint."""

import argparse
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

from aclgraph_bounded_lead_analysis import analyze_trace
from aclgraph_bounded_lead_probe import experiment_modes, pair_samples, validate_fused_lanes
from aclgraph_full_path_budget import intersection, interval


def summarize(samples):
    modes = experiment_modes(True)
    identities = {(s["round"], s["mode"], s["sample"]) for s in samples}
    rounds, repeats = sorted({s["round"] for s in samples}), sorted({s["sample"] for s in samples})
    expected = {(r, m, s) for r in rounds for m in modes for s in repeats}
    if len(rounds) < 2 or len(repeats) < 3 or len(samples) != len(expected) or identities != expected:
        raise ValueError("incomplete paired six-control grid")
    if any(not 0 < s[k] < float("inf") for s in samples for k in ("device_us", "wall_us")):
        raise ValueError("invalid endpoint latency")
    rows = []
    for trial in rounds:
        for metric in ("device_us", "wall_us"):
            values = {
                m: statistics.median(s[metric] for s in samples if s["round"] == trial and s["mode"] == m)
                for m in modes
            }
            rows.append(
                dict(
                    round=trial,
                    metric=metric,
                    medians=values,
                    public_fusion_pct=100 * (1 - values["fused_padding"] / values["padding"]),
                    ordinary_split_pct=100 * (1 - values["split_early"] / values["padding"]),
                    equal_fusion_split_pct=100 * (1 - values["fused_split_early"] / values["fused_padding"]),
                    fused_early_saves_vs_barrier_us=values["fused_split_barrier"] - values["fused_split_early"],
                )
            )
    return rows


def fused_trace(path, mode):
    doc = json.loads(path.read_bytes(), parse_float=Decimal)
    events = doc if isinstance(doc, list) else doc["traceEvents"]
    kernels = sorted(
        [e for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type", "").startswith("KERNEL_")],
        key=lambda e: interval(e)[0],
    )
    comm = sorted(
        [e for e in events if e.get("ph") == "X" and e["name"].startswith("hcom_")], key=lambda e: interval(e)[0]
    )
    mm = [e for e in kernels if "MatMul" in e["name"]]
    fused = [e for e in kernels if e["name"] == "MatmulReduceScatterV2"]
    ar = [e for e in comm if e["name"].startswith("hcom_allReduce")]
    ag = [e for e in comm if e["name"].startswith("hcom_allGather")]
    core = [e for e in kernels if e["name"] == "q4_tile_solve_1"]
    split = mode != "fused_padding"
    observed = (
        len(mm),
        len(fused),
        len(ar),
        len(ag),
        len(core),
        sum(e["name"] == "q4_log_deferred_0" for e in kernels),
        sum(e["name"] == "FusedInferAttentionScore" for e in kernels),
    )
    if observed != ((10, 2, 3, 2, 1, 1, 1) if split else (8, 1, 3, 1, 1, 1, 1)):
        raise ValueError(f"unexpected fused two-layer inventory: {observed}")
    inputs = mm[3:7] if split else mm[3:5]
    out = mm[7] if split else mm[5]
    if interval(inputs[0])[0] + 1 < interval(fused[-1])[1] or interval(core[0])[0] + 1 < interval(inputs[-1])[1]:
        raise ValueError("projection/core dependency violated")
    if mode in ("fused_padding", "fused_split_barrier") and interval(inputs[0])[0] + 1 < interval(ag[-1])[1]:
        raise ValueError("barrier control reads unfinished output")
    if split and (interval(inputs[0])[0] + 1 < interval(ag[0])[1] or interval(inputs[2])[0] + 1 < interval(ag[1])[1]):
        raise ValueError("lane projection reads before its gathered rows are complete")
    return dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        counts=observed,
        down_fused_work_us=float(sum(Decimal(str(e["dur"])) for e in fused)),
        input_projection_us=float(sum(Decimal(str(e["dur"])) for e in inputs)),
        all_gather_work_us=float(sum(Decimal(str(e["dur"])) for e in ag)),
        input_a_overlap_last_gather_us=float(intersection([interval(ag[-1])], map(interval, inputs[:2])))
        if split
        else 0,
        down_b_overlap_first_gather_us=float(intersection([interval(ag[0])], [interval(fused[-1])])) if split else 0,
        boundary_span_us=float(interval(out)[0] - interval(fused[0])[0]),
    )


def analyze(root):
    reports = [json.loads((root / "results" / f"probe.rank{rank}.json").read_bytes()) for rank in (0, 1)]
    modes = experiment_modes(True)
    manifest = json.loads((root / "launch.json").read_bytes())
    for source, digest in manifest["sources"].items():
        if hashlib.sha256((root / "sources" / Path(source).name).read_bytes()).hexdigest() != digest:
            raise ValueError("source hash mismatch")
    paired = pair_samples([r["samples"] for r in reports])
    traces = []
    if reports[0]["selection"] != reports[1]["selection"]:
        raise ValueError("rank selection mismatch")
    for rank, report in enumerate(reports):
        if report["status"] != "completed_closed_boundary_with_cross_shape_numerical_diagnostics":
            raise ValueError("incomplete experiment")
        if not report["official_two_layer_exact"] or report["fia_update_tasks"] != dict.fromkeys(modes, 1):
            raise ValueError("missing official-layer oracle or true FIA updates")
        if not report["fused_down"] or not report["packed_join"] or report["join_before"] != "core":
            raise ValueError("wrong fusion/join configuration")
        validate_fused_lanes(report["selection"]["pair_tokens"], report["selection"]["capture_sizes"])
        wanted = {(m, s) for m in modes for s in range(3)}
        if len(report["checks"]) != len(wanted) or {(c["mode"], c["step"]) for c in report["checks"]} != wanted:
            raise ValueError("missing continuous qualification")
        if not all(c["eager_replay_exact"] and c["inactive_null_exact"] for c in report["checks"]):
            raise ValueError("qualification failed")
        if report["critical_rank_samples"] != paired:
            raise ValueError("raw rank pairing mismatch")
        if len(report["profiles"]) != 6 or {p["mode"] for p in report["profiles"]} != set(modes):
            raise ValueError("missing profile controls")
        for profile in report["profiles"]:
            path = root / "results" / profile["file"]
            mode = profile["mode"]
            row = fused_trace(path, mode) if mode.startswith("fused_") else analyze_trace(path, mode, "core", True)
            if row["sha256"] != profile["sha256"]:
                raise ValueError("trace hash mismatch")
            traces.append(dict(rank=rank, mode=mode, **row))
    return dict(
        scope="two complete real layers with synthetic segment inputs/prefix; not full model or serving",
        selection=reports[0]["selection"],
        per_round=summarize(paired),
        traces=traces,
        numerical_diagnostics=[r["numerical_diagnostics"] for r in reports],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["per_round"], indent=2))
