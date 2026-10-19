# SPDX-License-Identifier: Apache-2.0
"""Audit the fused Down feasibility records without extrapolating to a model."""

import argparse
import hashlib
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

from aclgraph_fused_down_probe import MODES, pair_samples, validate_capacity


def validate_final_join(events, producer, collective):
    def interval(event):
        start = Decimal(str(event["ts"]))
        return start, start + Decimal(str(event["dur"]))

    producer_end, collective_end = interval(producer)[1], interval(collective)[1]
    # Exported event and HCCL wrappers have small timestamp-boundary differences.
    tolerance = Decimal(2)
    waits = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("tid") == producer["tid"] and e.get("args", {}).get("Task Type") == "EVENT_WAIT"
    ]
    if not any(
        producer_end - tolerance <= interval(e)[0] <= collective_end + tolerance
        and interval(e)[1] >= collective_end - tolerance
        for e in waits
    ):
        raise ValueError("missing compute-stream wait for the final collective")


def summarize_samples(samples):
    expected = {(trial, mode, sample) for trial in range(5) for mode in MODES for sample in range(6)}
    if len(samples) != len(expected) or {(s["trial"], s["mode"], s["sample"]) for s in samples} != expected:
        raise ValueError("require the complete five-round, three-mode paired grid")
    if any(not 0 < s[metric] < float("inf") for s in samples for metric in ("device_us", "wall_us")):
        raise ValueError("invalid latency")
    medians = {
        mode: {
            metric: statistics.median(s[metric] for s in samples if s["mode"] == mode)
            for metric in ("device_us", "wall_us")
        }
        for mode in MODES
    }
    per_trial = []
    for trial in range(5):
        for metric in ("device_us", "wall_us"):
            times = {
                mode: statistics.median(s[metric] for s in samples if s["mode"] == mode and s["trial"] == trial)
                for mode in MODES
            }
            per_trial.append(
                dict(
                    trial=trial,
                    metric=metric,
                    medians=times,
                    fused_reduction_percent=100 * (1 - times["fused_rs_ag"] / times["mm_ar"]),
                )
            )
    return dict(
        medians=medians,
        per_trial=per_trial,
        fused_reduction_percent={
            k: 100 * (1 - medians["fused_rs_ag"][k] / medians["mm_ar"][k]) for k in ("device_us", "wall_us")
        },
    )


def analyze(root):
    launch = json.loads((root / "launch-replay.json").read_bytes())
    source = root / "sources/aclgraph_fused_down_probe.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != launch["source_sha256"]:
        raise ValueError("source hash mismatch")
    reports = [json.loads((root / "results" / f"replay.rank{rank}.json").read_bytes()) for rank in (0, 1)]
    raw_present = ["rank_samples" in report for report in reports]
    if any(raw_present) and not all(raw_present):
        raise ValueError("partial raw rank samples")
    if all(raw_present) and pair_samples([r["rank_samples"] for r in reports]) != reports[0]["samples"]:
        raise ValueError("raw rank pairing mismatch")
    for field in ("capacity", "selection", "catalog_sha256", "samples"):
        if reports[0][field] != reports[1][field]:
            raise ValueError(f"rank {field} mismatch")
    traces = []
    for rank, report in enumerate(reports):
        if report["rank"] != rank or report["status"] != "qualified_synthetic_down_endpoint_not_full_model":
            raise ValueError("incomplete endpoint qualification")
        if report["source_sha256"] != digest or report["weight"] != dict(shape=[5120, 8704], format=29):
            raise ValueError("wrong source or weight layout")
        validate_capacity(report["capacity"], report["selection"]["capture_sizes"])
        checks = {c["label"]: c for c in report["checks"]}
        wanted = {f"{phase}/{step}/{mode}" for phase in ("eager", "replay") for step in range(2) for mode in MODES}
        if len(report["checks"]) != len(wanted) or set(checks) != wanted:
            raise ValueError("missing changed-input qualification")
        if any(
            not checks[name]["exact"] or checks[name]["max_abs"] != 0 for name in wanted if name.startswith("replay")
        ):
            raise ValueError("replay differs from same-mode eager")
        if len(report["profiles"]) != 3 or {p["mode"] for p in report["profiles"]} != set(MODES):
            raise ValueError("missing profile control")
        for profile in report["profiles"]:
            path = root / "results" / profile["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != profile["sha256"]:
                raise ValueError("profile hash mismatch")
            data = json.loads(path.read_bytes())
            events = data if isinstance(data, list) else data["traceEvents"]
            tasks = [
                e
                for e in events
                if e.get("ph") == "X"
                and (e.get("args", {}).get("Task Type", "").startswith("KERNEL_") or e["name"].startswith("hcom_"))
            ]
            names = [e["name"] for e in tasks]
            pattern = (
                sum("MatMul" in n for n in names),
                sum(n == "MatmulReduceScatterV2" for n in names),
                sum(n.startswith("hcom_allReduce") for n in names),
                sum(n.startswith("hcom_reduceScatter") for n in names),
                sum(n.startswith("hcom_allGather") for n in names),
            )
            expected = dict(mm_ar=(1, 0, 1, 0, 0), mm_rs_ag=(1, 0, 0, 1, 1), fused_rs_ag=(0, 1, 0, 0, 1))
            if pattern != expected[profile["mode"]]:
                raise ValueError("unexpected closed-endpoint kernel/collective inventory")
            producer = next(e for e in tasks if "MatMul" in e["name"] or e["name"] == "MatmulReduceScatterV2")
            collective_name = "hcom_allReduce" if profile["mode"] == "mm_ar" else "hcom_allGather"
            collective = next(e for e in tasks if e["name"].startswith(collective_name))
            validate_final_join(events, producer, collective)
            traces.append(
                dict(
                    rank=rank,
                    mode=profile["mode"],
                    file=path.name,
                    sha256=profile["sha256"],
                    final_compute_stream_join=True,
                    tasks=[dict(name=e["name"], duration_us=e["dur"], stream=e["tid"]) for e in tasks],
                )
            )
    summary = summarize_samples(reports[0]["samples"])
    for report in reports:
        for mode in MODES:
            for metric in ("device_us", "wall_us"):
                if not math.isclose(summary["medians"][mode][metric], report["medians"][mode][metric], abs_tol=1e-9):
                    raise ValueError("reported median mismatch")
    return dict(
        scope=reports[0]["scope"],
        capacity=reports[0]["capacity"],
        checkpoint_weights=[r.get("checkpoint_weight") for r in reports],
        checks=[r["checks"] for r in reports],
        traces=traces,
        limitations=[
            "no full-model or split-specific speedup proof",
            "profiler perturbs these short operators; durations are not unprofiled timings",
            "raw rank pairing verified"
            if all(raw_present)
            else "saved samples are already slowest-rank paired values; individual rank timings were not saved",
        ],
        **summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(capacity=result["capacity"], medians=result["medians"], reduction=result["fused_reduction_percent"]),
            indent=2,
        )
    )
