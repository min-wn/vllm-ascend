# SPDX-License-Identifier: Apache-2.0
"""Recompute small-screen effects and scope gates from preserved raw reports."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(path):
    value = json.loads(path.read_bytes(), parse_float=Decimal)
    return value if isinstance(value, list) else value["traceEvents"]


def kernel_summary(path):
    tasks = [
        r
        for r in events(path)
        if r.get("ph") == "X" and str(r.get("args", {}).get("Task Type", "")).startswith("KERNEL_")
    ]
    first = min(Decimal(str(r["ts"])) for r in tasks)
    rows = [
        dict(
            name=r["name"],
            type=r["args"]["Task Type"],
            stream=r["args"]["Physic Stream Id"],
            start_us=float(Decimal(str(r["ts"])) - first),
            duration_us=float(r["dur"]),
        )
        for r in tasks
    ]
    return rows


def analyze(root):
    raw = root / "raw"
    manifest = json.loads((raw / "manifest.json").read_bytes())
    for name, identity in manifest.items():
        assert sha(raw / name) == identity["sha256"]
        assert (raw / name).stat().st_size == identity["bytes"]
    reports = [json.loads(p.read_bytes()) for p in sorted((raw / "page-screen").glob("cell-*/consumer-probe.json"))]
    assert len(reports) == 4
    builds = {v: json.loads((raw / f"build-{v}/page-reuse-build.json").read_bytes()) for v in ("control", "candidate")}
    binary = "FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673.o"
    assert (
        builds["control"]["artifacts"][binary]["sha256"]
        == "0ef799f0ec513f2fb2c9b0e642379f3123911fdaf9b656f17ab3976c139d1743"
    )
    outputs, summaries, fp32_count = {}, [], 0
    for index, report in enumerate(reports):
        assert report["state"] == "completed" and report["returncode"] == 0
        assert len(report["samples"]) == 20 and len(report["profiles"]) == 2
        for field in ("input_identity", "frame_query_identity", "mask_identity", "selected_records"):
            assert report[field] == reports[0][field]
        for field in ("input_identity", "frame_query_identity", "mask_identity"):
            assert report[field] == report["post_" + field]
        variant = ("control", "candidate", "candidate", "control")[index]
        assert (
            report["opp_artifacts"][
                "op_impl/ai_core/tbe/kernel/ascend910b/fused_infer_attention_score_v2_sink/" + binary
            ]
            == builds[variant]["artifacts"][binary]["sha256"]
        )
        for phase in ("pre_checks", "post_checks"):
            for check in report[phase]:
                cap = check["capacity"]
                assert (
                    check["all_outputs_finite"] and check["stable_addresses"] and check["exact_descriptors_read_only"]
                )
                if cap in outputs:
                    assert outputs[cap] == check["output_hashes"]
                outputs[cap] = check["output_hashes"]
                fp32_count += len(check["fp32_sampled_rows"])
        for cap in (400, 512):
            values = [Decimal(str(s["wall_us"])) for s in report["samples"] if s["capacity"] == cap]
            assert len(values) == 10
            summaries.append(
                dict(
                    cell=index,
                    capacity=cap,
                    mean_us=float(sum(values) / len(values)),
                    median_us=float(statistics.median(values)),
                )
            )
    effects = []
    for direction, control, candidate in (("forward", 0, 1), ("reverse", 3, 2)):
        for cap in (400, 512):
            a = next(s for s in summaries if s["cell"] == control and s["capacity"] == cap)
            b = next(s for s in summaries if s["cell"] == candidate and s["capacity"] == cap)
            effects.append(
                dict(
                    direction=direction,
                    capacity=cap,
                    mean_gain_percent=100 * (a["mean_us"] - b["mean_us"]) / a["mean_us"],
                    median_gain_percent=100 * (a["median_us"] - b["median_us"]) / a["median_us"],
                )
            )
    overlap = []
    for folder, tools in (("overlap-forward", "tools-v1"), ("overlap-reverse", "tools-v2")):
        report = json.loads((raw / folder / "overlap.json").read_bytes())
        assert report["state"] == "completed" and report["returncode"] == 0
        assert report["source_sha256"] == sha(raw / tools / "aclgraph_fia_sink_overlap_probe.py")
        assert len(report["samples"]) == 160 and all(c["passed"] for c in report["checks"])
        assert not report["full_model"] and not report["split_vs_padding"]
        medians = {
            mode: statistics.median(s["us"] for s in report["samples"] if s["mode"] == mode)
            for mode in report["median_us"]
        }
        assert medians == report["median_us"]
        profiles = {mode: kernel_summary(raw / folder / (mode + ".json")) for mode in medians}
        for rows in profiles.values():
            assert len(rows) == 6
            assert Counter(r["type"] for r in rows) == {
                "KERNEL_AIVEC": 2,
                "KERNEL_AICPU": 1,
                "KERNEL_MIX_AIC": 1,
                "KERNEL_AICORE": 2,
            }
        rounds = []
        for iteration in range(4):
            m = {
                mode: statistics.median(
                    s["us"] for s in report["samples"] if s["mode"] == mode and s["round"] == iteration
                )
                for mode in medians
            }
            rounds.append(
                dict(
                    round=iteration,
                    medians=m,
                    full_gain_percent=100 * (m["full_serial"] - m["full_parallel"]) / m["full_serial"],
                )
            )
        overlap.append(
            dict(
                run=folder,
                medians=medians,
                rounds=rounds,
                checks=len(report["checks"]),
                full_gain_percent=100 * (medians["full_serial"] - medians["full_parallel"]) / medians["full_serial"],
                profiles=profiles,
            )
        )
    planner_raw = root / "planner-control/raw"
    planner_manifest = json.loads((planner_raw / "manifest.json").read_bytes())
    for name, identity in planner_manifest.items():
        assert sha(planner_raw / name) == identity["sha256"]
        assert (planner_raw / name).stat().st_size == identity["bytes"]
    planner = json.loads((planner_raw / "planner-controls/overlap.json").read_bytes())
    assert planner["state"] == "completed" and planner["returncode"] == 0
    assert planner["source_sha256"] == sha(planner_raw / "tools-v3/aclgraph_fia_sink_overlap_probe.py")
    assert len(planner["samples"]) == 160 and len(planner["checks"]) == 12
    assert all(check["passed"] for check in planner["checks"])
    planner_medians = {
        mode: statistics.median(s["us"] for s in planner["samples"] if s["mode"] == mode)
        for mode in planner["median_us"]
    }
    assert planner_medians == planner["median_us"]
    planner_profiles = {
        mode: kernel_summary(planner_raw / "planner-controls" / (mode + ".json")) for mode in planner_medians
    }
    for mode, rows in planner_profiles.items():
        assert len(rows) == 6
        attention = next(r for r in rows if r["type"] == "KERNEL_MIX_AIC")
        metadata = next(r for r in rows if r["type"] == "KERNEL_AICPU")
        dense = [r for r in rows if "MatmulWeightNz" in r["name"] or r["name"] == "SwiGlu"]
        assert len(dense) == 3
        if mode == "plan_only":
            assert max(r["start_us"] + r["duration_us"] for r in dense) <= attention["start_us"] + 0.01
        if mode == "plan_first_parallel":
            assert (
                metadata["start_us"] + metadata["duration_us"] <= min(r["start_us"] for r in [attention, *dense]) + 0.01
            )
    return dict(
        state="passed",
        raw_files=len(manifest),
        page_summaries=summaries,
        page_effects=effects,
        page_fp32_rows=fp32_count,
        page_complete_outputs_equal=True,
        page_adopt=False,
        overlap=overlap,
        overlap_full_model_or_split_gain=False,
        planner_controls=dict(
            medians=planner_medians,
            gain_percent={
                mode: 100 * (planner_medians["full_serial"] - value) / planner_medians["full_serial"]
                for mode, value in planner_medians.items()
            },
            profiles=planner_profiles,
            required_plan_remains_timed=True,
            one_process_only=True,
            raw_files=len(planner_manifest),
        ),
        limitation=(
            "CPU address model and small device experiments only; no full-model implementation or split advantage"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite analysis")
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("overlap", "page_summaries", "planner_controls")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
