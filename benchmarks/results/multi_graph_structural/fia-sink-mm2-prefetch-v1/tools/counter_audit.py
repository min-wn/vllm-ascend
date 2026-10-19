# SPDX-License-Identifier: Apache-2.0
"""Audit exact counter files and recompute aggregate pipeline activity.

Pipeline activity overlaps; percentages are not additive time shares. These
two diagnostic processes do not establish causality or stable performance.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a counter audit")
    target = Path(__file__).resolve().parents[1]
    raw = target / "counter-diagnostic/raw"
    manifest = json.loads((raw / "manifest.json").read_bytes())
    index = {}
    for item in manifest["files"]:
        name = item["alias"]
        assert Path(name).name == name and item["source"] not in index
        data = (raw / name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
        index[item["source"]] = raw / name

    def find(suffix):
        matches = [p for name, p in index.items() if name.endswith(suffix)]
        (path,) = matches
        return path

    driver = json.loads(find("/counter-diagnostic.json").read_bytes())
    assert driver["state"] == "completed" and driver["returncode"] == 0
    assert driver["timing_samples_are_diagnostic"] and not driver["performance_gain_measured"]
    assert driver["metric"] == "PipeUtilization" and driver["profile_level"] == 1
    assert [e["variant"] for e in driver["runs"]] == ["control", "candidate"]
    main_raw = target / "raw"
    for name, digest in driver["source_sha256"].items():
        if name in ("aclgraph_fia_sink_consumer_probe.py", "mm2_prefetch_counter_driver.py"):
            path = find("/" + name)
        else:
            path = main_raw / "tools" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    before = find("/probe-before.py.txt").read_bytes()
    assert hashlib.sha256(before).hexdigest() == "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9"
    after = find("/aclgraph_fia_sink_consumer_probe.py").read_text(encoding="utf-8")
    anchor = (
        "            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]\n"
    )
    replacement = (
        anchor.rstrip() + ",\n"
        "            experimental_config=torch_npu.profiler._ExperimentalConfig(\n"
        "                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,\n"
        "                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),\n"
        '            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target.parent / "counters")),\n'
    )
    assert after == before.decode().replace(anchor, replacement)
    reference = json.loads((main_raw / "screen/cell-00-control-whole/consumer-probe.json").read_bytes())
    for number, variant in enumerate(("control", "candidate")):
        report = json.loads(find(f"/cell-{number}-{variant}/consumer-probe.json").read_bytes())
        build = json.loads((main_raw / f"builds/{variant}/mm2-prefetch-build.json").read_bytes())
        assert report["state"] == "completed" and report["returncode"] == 0
        for key in ("input_identity", "selected_records", "canonical_digest", "frame_query_identity", "mask_identity"):
            assert report[key] == reference[key]
        for name, identity in build["artifacts"].items():
            assert (
                report["opp_artifacts"][
                    "op_impl/ai_core/tbe/kernel/ascend910b/fused_infer_attention_score_v2_sink/" + name
                ]
                == identity["sha256"]
            )
        for phase in ("pre_checks", "post_checks"):
            for row in report[phase]:
                (old,) = [e for e in reference[phase] if e["capacity"] == row["capacity"]]
                assert row["output_hashes"] == old["output_hashes"]
                assert row["plan"]["words"] == old["plan"]["words"]
                assert row["all_outputs_finite"] and row["exact_zero_padding"] and row["stable_addresses"]
        for descriptor in report["prepared_descriptors"].values():
            n = descriptor["prepared_steps"]
            assert descriptor["completed_steps"] == n
            assert descriptor["step_copies"] == dict(q_lengths=0, kv_lengths=n, block_table=n)
    metrics = (
        "Duration(us)",
        "aicore_time(us)",
        "aic_mac_time(us)",
        "aic_scalar_time(us)",
        "aic_mte1_time(us)",
        "aic_mte2_time(us)",
        "aic_fixpipe_time(us)",
    )
    summaries = []
    expected_types = {"FusedInferAttentionScoreV2Sink": 16, "FusedInferAttentionScoreV2SinkMetadata": 1, "ZerosLike": 1}
    for variant, number in (("control", 0), ("candidate", 1)):
        for cap in (400, 512):
            prefix = f"/cell-{number}-{variant}/profiles/cap{cap}/"
            matches = [p for name, p in index.items() if prefix in name and name.endswith("/kernel_details.csv")]
            (path,) = matches
            rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
            assert dict(collections.Counter(row["Type"] for row in rows)) == expected_types
            rows = [row for row in rows if row["Type"] == "FusedInferAttentionScoreV2Sink"]
            for row in rows:
                assert row["Accelerator Core"] == "MIX_AIC" and row["Block Num"] == "20"
                assert row["Mix Block Num"] == "40"
                for metric in metrics:
                    assert math.isfinite(float(row[metric])) and float(row[metric]) > 0
            means = {metric: statistics.mean(float(row[metric]) for row in rows) for metric in metrics}
            activity = {
                pipe: 100 * means[f"aic_{pipe}_time(us)"] / means["aicore_time(us)"]
                for pipe in ("mac", "scalar", "mte1", "mte2", "fixpipe")
            }
            summaries.append(
                dict(
                    variant=variant,
                    capacity=cap,
                    attention_calls=16,
                    mean_us=means,
                    activity_percent=activity,
                    csv_alias=path.name,
                )
            )
    deltas = []
    for cap in (400, 512):
        (old,) = [s for s in summaries if s["capacity"] == cap and s["variant"] == "control"]
        (new,) = [s for s in summaries if s["capacity"] == cap and s["variant"] == "candidate"]
        deltas.append(
            dict(capacity=cap, candidate_minus_control_us={k: new["mean_us"][k] - old["mean_us"][k] for k in metrics})
        )
    result = dict(
        state="passed",
        retained_files_verified=len(index),
        diagnostic_processes=2,
        profiled_attention_calls=64,
        source_delta_is_profiling_only=True,
        performance_samples_used=False,
        baseline_outputs_and_plans_match=True,
        summaries=summaries,
        deltas=deltas,
        stable_root_cause_established=False,
        limitation="Whole-attention aggregate, one process per variant; overlapping activity is not additive.",
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
