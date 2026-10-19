# SPDX-License-Identifier: Apache-2.0
"""Re-audit the saved MM2 screen, explicitly accounting for four omitted binaries.

Run from the repository root with --output pointing to a fresh result file.
All measured source snapshots and reports are read without modification.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise ValueError("refusing to overwrite an audit")
    target = Path(__file__).resolve().parents[1]
    ROOT = target / "raw"
    REPO = target.parents[3]
    sys.path.insert(0, str(ROOT / "tools"))
    import aclgraph_fia_sink_consumer_analyze as analyzer
    import aclgraph_fia_sink_mm2_double_buffer_patch as patch
    import aclgraph_fia_sink_mm2_transfer_patch as first

    omitted = json.loads((target / "omitted-binaries.json").read_bytes())
    binary = "FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673.o"
    expected = {"screen/cpu-double-buffer-contract"} | {
        "builds/" + v + "/binary/" + binary for v in ("original", "control", "wide")
    }
    if set(omitted) != expected:
        raise ValueError("unexpected raw omission")
    # Audit full contracts before deriving paired statistics; profile time is diagnostic.
    checks = 0

    def require(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)

    def sha(path):
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        # Omitted binaries were hashed at collection; no local byte verification is claimed.
        return omitted[relative]["sha256"]

    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    for relative, identity in manifest["files"].items():
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT.resolve()), "manifest path confinement")
        if relative in omitted:
            require(
                not path.exists() and identity == omitted[relative],
                "explicit omitted binary identity",
            )
            continue
        require(
            path.is_file() and path.stat().st_size == identity["bytes"],
            "manifest file size",
        )
        require(sha(path) == identity["sha256"], "manifest file hash")
    screen = ROOT / "screen"
    bundle = json.loads((screen / "mm2-double-buffer-screen.json").read_bytes())
    require(bundle["state"] == "completed" and bundle["returncode"] == 0, "complete suite")
    require(
        bundle["design"]
        == dict(
            cells=["original", "control", "wide", "wide", "control", "original"],
            policy="whole",
            trial=0,
            seed=1703,
            warmups=5,
            repeats=10,
            profile_repeats=1,
            capacities=[400, 512],
            occurrences=16,
        ),
        "fixed ABC/CBA design",
    )
    require(
        bundle["cpu_contract"]["passed"] and bundle["cpu_contract"]["ordered_k_cases"] == 1026,
        "native CPU contracts",
    )
    require(len(bundle["runs"]) == 6, "exact six fresh processes")
    for label in (
        "before",
        "cell0",
        "cell1",
        "cell2",
        "cell3",
        "cell4",
        "cell5",
        "after",
    ):
        require(
            "No running processes found in NPU 0" in (screen / ("device-" + label + ".txt")).read_text(),
            "idle device gate",
        )
    for name, digest in bundle["source_sha256"].items():
        require(sha(ROOT / "tools" / name) == digest, "deployed source hash")
    require(
        sha(ROOT / "tools/aclgraph_fia_sink_consumer_probe.py")
        == "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9",
        "unchanged runtime probe",
    )
    before = (screen / "cube-before.h.txt").read_bytes()
    require(
        hashlib.sha256(before).hexdigest() == patch.EXPECTED_SHA256,
        "qualified initial cube",
    )
    after = patch.transformed(before)
    require(
        (screen / "cube-candidate.h.txt").read_bytes() == after,
        "exact candidate transform",
    )
    native = json.loads((screen / "cpu-double-buffer-contract.json").read_bytes())
    require(native == bundle["cpu_contract"], "saved native contract identity")
    require(sha(screen / "cpu-double-buffer-contract.cpp") == native["source_sha256"], "compiled CPU source identity")
    for name, digest in native["native_helpers"].items():
        require(sha(screen / (name.replace("/", "_") + ".txt")) == digest, "native helper identity")
    require(native["runtime_layout_cases"] == 16 and native["static_fallback_cases"] == 7, "all layout cases")
    require(native["q_nz_tail_cases"] == 128 and native["q_slots"] == 2, "both Q slots and all tails")
    require(
        "1026 ordered K contracts passed" in (screen / "cpu-double-buffer-contract.log").read_text(encoding="utf-8"),
        "actual native executable success",
    )
    build_reports = []
    for variant, enabled in (("original", False), ("control", False), ("wide", True)):
        path = ROOT / "builds" / variant
        tag = "mm2-transfer" if variant == "original" else "mm2-double-buffer"
        build = json.loads((path / (tag + "-build.json")).read_bytes())
        require(
            build["state"] == "completed" and build["wide_v"] is enabled,
            "build macro state",
        )
        require(
            build["npu_work_executed"] is False and build["system_install_performed"] is False,
            "isolated build",
        )
        require(
            build["binary_qualification"]["actual_tiling_keys"] == ["104000000010222223", "104000000010322223"],
            "restricted existing keys",
        )
        require(
            (path / "fia_block_cube_nonquant_gqa_sink.h.txt").read_bytes()
            == (first.transformed(before) if variant == "original" else after),
            "expected patched cube source",
        )
        wrapper = (path / "FusedInferAttentionScoreV2Sink.py.txt").read_text()
        macro = first.MACRO if variant == "original" else patch.MACRO
        require(
            wrapper.count("-D" + macro + "=" + str(int(enabled))) == 1,
            "exact compile definition",
        )
        require(
            wrapper.count("-DTHESIS_FIA_MM2_DOUBLE_V_LAYOUT=1") == int(variant != "original"),
            "layout definition",
        )
        require("THESIS_FIA_RUNNING_O_UB" not in wrapper, "running-O patch excluded")
        require(len(build["changed_source_files"]) == 2, "only cube and wrapper changed")
        for name, identity in build["artifacts"].items():
            require(
                sha(path / "binary" / name) == identity["sha256"],
                "retained native binary",
            )
        build_reports.append(build)
    objects = [build["artifacts"][binary]["sha256"] for build in build_reports]
    require(len(set(objects)) == 3, "three distinct native variants")
    require(
        sha(ROOT / "builds/original/mm2-transfer-build.json") == bundle["baseline_build_report_sha256"],
        "reused original build identity",
    )
    reference = json.loads(
        (
            REPO / "benchmarks/results/multi_graph_structural/fia-sink-running-o-v1/"
            "cell-03-control-whole/consumer-probe.json"
        ).read_bytes()
    )
    reports = []
    summaries = []
    profiles = []
    fp32_rows = 0
    maximum_error = 0
    plan_count = 0
    preflight_count = 0
    pids = set()
    previous_end = bundle["started_unix"]
    identities = (
        "input_identity",
        "frame_query_identity",
        "mask_identity",
        "selected_records",
        "canonical_digest",
        "trace_contract",
        "core_limit",
        "extension_sha256",
    )
    for index, (variant, entry) in enumerate(zip(bundle["design"]["cells"], bundle["runs"], strict=True)):
        path = screen / entry["report_path"]
        report = json.loads(path.read_bytes())
        require(
            entry["variant"] == variant and entry["process_order_index"] == index,
            "fixed process order",
        )
        require(sha(path) == entry["report_sha256"], "retained process report")
        require(
            entry["returncode"] == 0 and report["state"] == "completed" and report["returncode"] == 0,
            "successful process",
        )
        require(
            previous_end <= entry["launched_unix"] < entry["finished_unix"] <= bundle["finished_unix"],
            "process intervals do not overlap",
        )
        previous_end = entry["finished_unix"]
        require(report["child_pid"] not in pids, "fresh native process")
        pids.add(report["child_pid"])
        require(
            report["policy"] == "whole" and report["trial"] == 0 and report["occurrences"] == 16,
            "cell identity",
        )
        for flag in (
            "full_trace_measured",
            "full_model_qualified",
            "end_to_end_gain_established",
        ):
            require(report[flag] is False, "bounded result scope")
        built = build_reports[("original", "control", "wide").index(variant)]
        for name, identity in built["artifacts"].items():
            key = "op_impl/ai_core/tbe/kernel/ascend910b/fused_infer_attention_score_v2_sink/" + name
            require(
                report["opp_artifacts"][key] == identity["sha256"],
                "runtime staged binary identity",
            )
        require(
            report["policy_provenance"]["library_sha256"] == reference["policy_provenance"]["library_sha256"],
            "same real metadata producer",
        )
        for key in identities:
            require(report[key] == reference[key], "same frozen identity: " + key)
        require(
            report["input_identity"] == report["post_input_identity"],
            "resident inputs unmodified",
        )
        require(
            report["frame_query_identity"] == report["post_frame_query_identity"],
            "frame bytes unmodified",
        )
        require(report["mask_identity"] == report["post_mask_identity"], "mask unmodified")
        require(len(report["metadata_preflight"]) == 2, "complete metadata preflight")
        for item in report["metadata_preflight"]:
            (original,) = [r for r in reference["metadata_preflight"] if r["capacity"] == item["capacity"]]
            require(
                item["current_exact_base_verified"] is True and item["verified_before_attention_capture"] is True,
                "preflight exact base before capture",
            )
            require(
                len(item["words"]) == 1024 and item["words"] == original["words"],
                "preflight complete plan",
            )
            preflight_count += 1
        for preflight in report["metadata_preflight"]:
            require(
                preflight["words"][577] == 512 and preflight["words"][582] == 4,
                "active runtime size gates",
            )
        require(len(report["samples"]) == 20, "retain all twenty cell samples")
        wanted = [(repeat, cap) for repeat in range(10) for cap in ([400, 512] if repeat % 2 == 0 else [512, 400])]
        require(
            [(s["repeat"], s["capacity"]) for s in report["samples"]] == wanted,
            "fixed sample order",
        )
        for sample in report["samples"]:
            for metric in (
                "wall_us",
                "preparation_enqueue_us",
                "submit_join_cpu_us",
                "queue_span_us",
            ):
                require(
                    math.isfinite(sample[metric]) and sample[metric] >= 0,
                    "finite timing metric",
                )
            require(sample["wall_us"] > 0, "positive full endpoint")
            require(
                sample["preparation_enqueue_us"] + sample["submit_join_cpu_us"] <= sample["wall_us"] + 1e-6,
                "same-clock subdivisions",
            )
        for phase in ("pre_checks", "post_checks"):
            require(len(report[phase]) == 2, "complete separate qualification")
            for row in report[phase]:
                cap = row["capacity"]
                (original,) = [r for r in reference[phase] if r["capacity"] == cap]
                for key in (
                    "all_outputs_finite",
                    "exact_zero_padding",
                    "stable_addresses",
                    "exact_descriptors_read_only",
                ):
                    require(row[key] is True, "qualification: " + key)
                require(
                    row["plan"]["words"] == original["plan"]["words"],
                    "complete unchanged 1024-word plan",
                )
                require(len(row["output_hashes"]) == 16, "all private output identities")
                require(len(row["fp32_sampled_rows"]) == 9, "all nine oracle rows")
                for oracle in row["fp32_sampled_rows"]:
                    error = oracle["max_abs_error"]
                    require(
                        math.isfinite(error) and 0 <= error < 0.005,
                        "finite FP32 error within recorded qualification",
                    )
                    maximum_error = max(maximum_error, error)
                    fp32_rows += 1
                plan_count += 1
        for cap in (400, 512):
            (pre,) = [r for r in report["pre_checks"] if r["capacity"] == cap]
            (post,) = [r for r in report["post_checks"] if r["capacity"] == cap]
            require(
                pre["output_hashes"] == post["output_hashes"],
                "pre/post complete output consistency",
            )
            descriptor = report["prepared_descriptors"][str(cap)]
            require(
                descriptor["prepared_steps"] == descriptor["completed_steps"] == 18,
                "necessary preparation retained",
            )
            require(
                descriptor["step_copies"] == dict(q_lengths=0, kv_lengths=18, block_table=18),
                "real descriptor H2D",
            )
            samples = [s["wall_us"] for s in report["samples"] if s["capacity"] == cap]
            summaries.append(
                dict(
                    cell=index,
                    variant=variant,
                    capacity=cap,
                    samples=10,
                    mean_us=statistics.mean(samples),
                    median_us=statistics.median(samples),
                    min_us=min(samples),
                    max_us=max(samples),
                )
            )
        require(len(report["profiles"]) == 2, "two independent diagnostic profiles")
        for item in report["profiles"]:
            trace_path = path.parent / item["path"]
            require(
                sha(trace_path) == item["sha256"] and item["outside_timing"] is True,
                "profile identity/scope",
            )
            document = json.loads(trace_path.read_bytes())
            events = document["traceEvents"] if isinstance(document, dict) else document
            labels = [e for e in events if e.get("ph") == "X" and str(e.get("name", "")).startswith("consumer/")]
            require(
                len(labels) == 1 and labels[0]["name"] == f"consumer/whole/cap{item['capacity']}/repeat0",
                "one actual profile label",
            )
            analysis = analyzer.profile_replay(events, labels[0], "whole")
            profiles.append(dict(cell=index, variant=variant, capacity=item["capacity"], **analysis))
        reports.append(report)
    require(
        fp32_rows == 216 and plan_count == 24 and preflight_count == 12,
        "complete oracle and plan totals",
    )
    equality = []
    for cap in (400, 512):
        output_hashes = [
            [r for r in report["pre_checks"] if r["capacity"] == cap][0]["output_hashes"] for report in reports
        ]
        equality.append(
            dict(
                capacity=cap,
                all_six_cells_complete_output_hashes_equal=all(value == output_hashes[0] for value in output_hashes),
            )
        )
    require(
        all(r["all_six_cells_complete_output_hashes_equal"] for r in equality),
        "cross-variant complete outputs",
    )
    pairs = []
    contrasts = (
        ("layout_only", ((0, 1), (5, 4))),
        ("transfer_same_layout", ((1, 2), (4, 3))),
        ("candidate_vs_original", ((0, 2), (5, 3))),
    )
    for contrast, directional in contrasts:
        for pair_index, (control, wide) in enumerate(directional):
            for cap in (400, 512):
                (old,) = [r for r in summaries if r["cell"] == control and r["capacity"] == cap]
                (new,) = [r for r in summaries if r["cell"] == wide and r["capacity"] == cap]
                (pold,) = [r for r in profiles if r["cell"] == control and r["capacity"] == cap]
                (pnew,) = [r for r in profiles if r["cell"] == wide and r["capacity"] == cap]
                pairs.append(
                    dict(
                        contrast=contrast,
                        pair=pair_index,
                        order="forward" if pair_index == 0 else "reverse",
                        capacity=cap,
                        control_cell=control,
                        candidate_cell=wide,
                        control_mean_us=old["mean_us"],
                        candidate_mean_us=new["mean_us"],
                        mean_reduction_percent=100 * (old["mean_us"] - new["mean_us"]) / old["mean_us"],
                        median_reduction_percent=100 * (old["median_us"] - new["median_us"]) / old["median_us"],
                        profile_attention_saved_us=pold["attention_duration_sum_us"]
                        - pnew["attention_duration_sum_us"],
                    )
                )
    result = dict(
        state="passed",
        assertions=checks,
        raw_files=len(manifest["files"]),
        timings=120,
        fp32_rows=fp32_rows,
        maximum_fp32_absolute_error=maximum_error,
        complete_plan_checks=plan_count + preflight_count,
        collected_raw_files=len(manifest["files"]),
        retained_raw_files=len(manifest["files"]) - len(omitted),
        local_binary_rehash_available=False,
        omitted_binaries=list(omitted),
        summaries=summaries,
        directional_pairs=pairs,
        output_equality=equality,
        profiles=profiles,
        scope=(
            "Six-process screen, two fixed records, synthetic attention chain; "
            "no full-trace/model/split-exclusive claim"
        ),
        note=(
            "Pair results remain separate; profiler observations are diagnostic. "
            "Four omitted binaries are checked against collection identities, not rehashed locally."
        ),
        auditor_sha256=sha(Path(__file__)),
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: v for k, v in result.items() if k != "profiles"}, indent=2))


if __name__ == "__main__":
    main()
