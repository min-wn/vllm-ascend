# SPDX-License-Identifier: Apache-2.0
"""Pair full-model rank timings and attribute the four modified layer boundaries."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from aclgraph_full_path_budget import intersection, interval


def pair_samples(reports):
    if len(reports) != 2 or not reports[0] or len(reports[0]) != len(reports[1]):
        raise ValueError("two nonempty equally sized rank samples required")
    result = []
    identities = set()
    for a, b in zip(*reports, strict=True):
        keys = ("trial", "mode", "step", "batch", "capacity")
        if any(a[k] != b[k] for k in keys):
            raise ValueError("rank sample identities disagree")
        identity = tuple(a[k] for k in keys)
        if identity in identities:
            raise ValueError("duplicate sample")
        identities.add(identity)
        if a["critical_rank_us"] != b["critical_rank_us"] or a["critical_rank_us"] != max(
            a["elapsed_us"], b["elapsed_us"]
        ):
            raise ValueError("invalid critical rank wall time")
        row = {k: a[k] for k in keys}
        row.update(wall_us=a["critical_rank_us"], device_us=max(a["device_us"], b["device_us"]))
        if any(not 0 < row[k] < float("inf") for k in ("wall_us", "device_us")):
            raise ValueError("invalid latency")
        result.append(row)
    return result


def analyze_trace(path, mode, boundaries):
    doc = json.loads(path.read_bytes(), parse_float=Decimal)
    events = doc if isinstance(doc, list) else doc["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type", "").startswith("KERNEL_")),
        key=lambda e: interval(e)[0],
    )
    comm = sorted(
        (e for e in events if e.get("ph") == "X" and e["name"].startswith("hcom_allReduce")),
        key=lambda e: interval(e)[0],
    )
    mm = [e for e in kernels if "MatMul" in e["name"]]
    # A large captured logical stream can span multiple physical stream IDs.
    main_stream_ids = {e["tid"] for e in mm}
    names = Counter(e["name"] for e in kernels)
    changed = set() if mode == "padding" else set(boundaries)
    if (
        len(mm),
        len(comm),
        names["q4_tile_solve_1"],
        names["q4_log_deferred_0"],
        names["FusedInferAttentionScore"],
    ) != (304 + 3 * len(changed), 129 + len(changed), 48, 48, 16):
        raise ValueError("incomplete full-model kernel inventory")
    matrices, collectives, mi, ci = {}, {}, 0, 1
    for layer in range(64):
        fields = ["QKV", "O", "GateUp", "Down"] if layer % 4 == 3 else ["QKVZ", "BA", "O", "GateUp", "Down"]
        if layer in changed:
            fields[-1:] = ["DownA", "DownB"]
        if layer - 1 in changed:
            fields[:2] = ["QKVZA", "BAA", "QKVZB", "BAB"]
        matrices[layer] = dict(zip(fields, mm[mi : mi + len(fields)], strict=True))
        mi += len(fields)
        fields = ["O", "DownA", "DownB"] if layer in changed else ["O", "Down"]
        collectives[layer] = dict(zip(fields, comm[ci : ci + len(fields)], strict=True))
        ci += len(fields)
        projection = matrices[layer]
        attention_name = "FusedInferAttentionScore" if layer % 4 == 3 else "q4_tile_solve_1"
        first, last = interval(next(iter(projection.values())))[0], interval(projection["O"])[0]
        attention = [e for e in kernels if e["name"] == attention_name and first < interval(e)[0] < last]
        if len(attention) != 1:
            raise ValueError("missing or repeated attention core within a decoder")
        if layer - 1 in changed and interval(attention[0])[0] + 1 < interval(projection["BAB"])[1]:
            raise ValueError("joined core begins before both projections finish")
    rows = []

    def total(items):
        return float(sum((Decimal(str(e["dur"])) for e in items), Decimal(0)))

    for boundary in boundaries:
        before, after = matrices[boundary], matrices[boundary + 1]
        split = boundary in changed
        downs = [before[k] for k in (("DownA", "DownB") if split else ("Down",))]
        projections = [after[k] for k in (("QKVZA", "BAA", "QKVZB", "BAB") if split else ("QKVZ", "BA"))]
        last_ar = collectives[boundary]["DownB" if split else "Down"]
        begin, end = interval(downs[-1])[1], interval(after["O"])[0]
        body = [e for e in kernels if e["tid"] in main_stream_ids and begin <= interval(e)[0] < end]
        if sum("MatMul" in e["name"] for e in body) != len(projections):
            raise ValueError("boundary body omitted a projection across physical stream segments")
        if interval(projections[0])[0] + 1 < begin:
            raise ValueError("next projection races the preceding Down GEMM")
        if mode == "split_barrier" and interval(projections[0])[0] + 1 < interval(last_ar)[1]:
            raise ValueError("barrier control permits early execution")
        publication = [
            e
            for e in kernels
            if split
            and e["tid"] not in main_stream_ids
            and "InplaceCopy" in e["name"]
            and interval(downs[0])[1] <= interval(e)[0] < interval(projections[-1])[1]
        ]
        if split and len(publication) != 2:
            raise ValueError("both asynchronous row outputs must be published into the full tensor")
        rows.append(
            dict(
                boundary=boundary,
                down_gemm_us=total(downs),
                input_projection_us=total(projections),
                last_down_ar_us=float(last_ar["dur"]),
                input_overlap_last_ar_us=float(intersection([interval(last_ar)], map(interval, projections[:2]))),
                concat_us=total([e for e in body if "ConcatD" in e["name"]]),
                body_kernel_us=total(body),
                down_output_copy_us=total(publication),
                closed_boundary_profile_span_us=float(interval(after["O"])[0] - interval(downs[0])[0]),
                down_b_overlap_first_ar_us=(
                    float(intersection([interval(collectives[boundary]["DownA"])], [interval(downs[1])]))
                    if split
                    else 0.0
                ),
                inventory={
                    name: dict(
                        count=sum(e["name"] == name for e in body), sum_us=total([e for e in body if e["name"] == name])
                    )
                    for name in sorted({e["name"] for e in body})
                },
            )
        )
    return dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        counts=dict(matmul=len(mm), all_reduce=len(comm), solve=48, commit=48, fia=16),
        boundaries=rows,
    )


def analyze(root):
    reports = [json.loads((root / "results" / f"diagnostic.rank{rank}.json").read_bytes()) for rank in (0, 1)]
    if any(r["status"] != "split_full_model_diagnostic_complete_not_serving" for r in reports):
        raise ValueError("incomplete full-model experiment")
    if (
        reports[0]["split_schedule"] != reports[1]["split_schedule"]
        or reports[0]["selection"] != reports[1]["selection"]
    ):
        raise ValueError("rank configuration mismatch")
    manifest = json.loads((root / "launch.json").read_bytes())
    for source, wanted in manifest["sources"].items():
        if hashlib.sha256((root / "sources" / Path(source).name).read_bytes()).hexdigest() != wanted:
            raise ValueError("source hash mismatch")
    for report in reports:
        if report["prefix"]["completed_requests"] != 128 or report["capture"]["fia_tasks_per_graph"] != 16:
            raise ValueError("missing complete prefix or real FIA tasks")
        checks = [c for c in report["checks"] if c.get("label") == "complete_same_schedule_eager_replay_exact"]
        wanted = {(mode, step) for mode in report["split_schedule"]["modes"] for step in range(2)}
        if {(c["mode"], c["step"]) for c in checks} != wanted or len(checks) != len(wanted):
            raise ValueError("missing continued eager/replay state qualification")
        for check in checks:
            if check["replay_audit"]["completed_steps"] != check["step"] + 1:
                raise ValueError("state audit continuity mismatch")
    samples = pair_samples([r["samples"] for r in reports])
    modes = reports[0]["split_schedule"]["modes"]
    trials = sorted({s["trial"] for s in samples})
    wanted_samples = {(trial, mode, step) for trial in trials for mode in modes for step in range(2)}
    if len(trials) < 2 or {(s["trial"], s["mode"], s["step"]) for s in samples} != wanted_samples:
        raise ValueError("incomplete balanced timing matrix")
    rows = []
    for trial in sorted({s["trial"] for s in samples}):
        for metric in ("device_us", "wall_us"):
            means = {
                mode: statistics.mean(s[metric] for s in samples if s["trial"] == trial and s["mode"] == mode)
                for mode in modes
            }
            rows.append(
                dict(
                    trial=trial,
                    metric=metric,
                    means=means,
                    early_reduction_percent=100 * (1 - means["split_early"] / means["padding"]),
                    early_saves_vs_barrier_us=means["split_barrier"] - means["split_early"],
                )
            )
    per_step = []
    for step in sorted({s["step"] for s in samples}):
        for metric in ("device_us", "wall_us"):
            medians = {
                mode: statistics.median(s[metric] for s in samples if s["step"] == step and s["mode"] == mode)
                for mode in modes
            }
            per_step.append(
                dict(
                    step=step,
                    metric=metric,
                    medians=medians,
                    early_reduction_percent=100 * (1 - medians["split_early"] / medians["padding"]),
                )
            )
    traces = []
    for rank, report in enumerate(reports):
        for profile in report["profiles"]:
            path = root / "results" / Path(profile["path"]).name
            result = analyze_trace(path, profile["mode"], report["split_schedule"]["boundaries"])
            if result["sha256"] != profile["sha256"]:
                raise ValueError("trace hash mismatch")
            traces.append(dict(rank=rank, mode=profile["mode"], **result))
    return dict(
        scope="64 decoders and final norm; two trajectory steps; no logits/sampler/serving",
        per_trial=rows,
        per_step=per_step,
        traces=traces,
        samples=samples,
        cross_padding_diagnostics=[r["cross_padding_diagnostics"] for r in reports],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["per_step"], indent=2))
