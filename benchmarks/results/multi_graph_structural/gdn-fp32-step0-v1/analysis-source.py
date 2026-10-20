# SPDX-License-Identifier: Apache-2.0
"""Validate the bounded attribution, keeping numerical qualification closed."""

import argparse
import hashlib
import json
from pathlib import Path

TRACE_SHA256 = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
SCREEN_NRMSE = 0.005


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(report, original):
    require(report["status"] == "complete_bounded_fp32_step0_attribution", "incomplete attribution")
    require(report["trace_sha256"] == original["trace_sha256"] == TRACE_SHA256, "changed trajectory")
    require(report["selection"] == original["selection"], "changed default graph catalog or dispatch")
    require(report["loader"] == original["loader"], "changed model, precision or loader")
    require(report["prefix"]["token_sha256"] == original["prefix"]["token_sha256"], "changed prefix inputs")
    require(report["prefix"]["completed_requests"] == 128, "partial prefill")
    contract, audit = report["step0_contract"], report["step0_start_audit"]
    require(contract["screen_nrmse"] == SCREEN_NRMSE, "changed numerical threshold")
    require(contract["batch"] == 98 and contract["capacity_tokens"] == 400, "changed batch or bucket")
    require(contract["performance_rounds"] == 0 and not contract["continued_qualification"], "expanded round")
    require(audit["passed"] and len(audit["gdn"]) == 48 and len(audit["fia"]) == 32, "incomplete start audit")
    require(not report["full_trajectory_numerical_qualification"], "step zero cannot qualify a trajectory")
    rows = report["step0_errors"]
    first = next(row for row in rows if row["differing_elements"])
    require(first["label"] == "layer0/recurrence/output", "unexpected earlier divergence")
    recurrence = [r for r in rows if r["label"].endswith("/recurrence/output")]
    decoders = [r for r in rows if "/decoder/" in r["label"]]
    require(len(recurrence) == 48 and len(decoders) == 128, "incomplete layer observations")
    local = report["step0_local"]
    require([item["layer"] for item in local] == [0, 1], "local pair expanded or changed")
    for item in local:
        require(item["identical_decoder_inputs"] and item["identical_initial_state"], "unpaired local inputs")
        inputs = [row for row in item["errors"] if "/input/" in row["label"] or "/recurrence/input/" in row["label"]]
        require(inputs and all(row["bitwise_identical"] for row in inputs), "local input mismatch")
    substitution = report["step0_single_substitution"]
    require(substitution["layer"] == 0 and substitution["replacements"] == 1, "intervention budget expanded")
    for row in report["step0_endpoint"]:
        prior = next(r for r in original["numeric_errors"] if r["step"] == 0 and r["label"] == row["label"])
        require(abs(row["worst_request_nrmse"] - prior["worst_request_nrmse"]) <= 2e-8, "failure not reproduced")
        require(not row["screen_passed"], "unexpected pass requires separate review")
    by_label = {row["label"]: row for row in rows}
    boundaries = (
        "layer0/recurrence/output",
        "layer0/gdn/norm",
        "layer0/gdn/out_proj/0",
        "layer0/post_attention_norm/0",
        "layer0/mlp/gate_up_proj/0",
        "layer0/mlp/act_fn",
        "layer0/mlp/down_proj/0",
        "layer1/recurrence/output",
        "layer1/gdn/norm",
        "layer1/gdn/out_proj/0",
        "layer1/mlp/down_proj/0",
        "final_norm/0",
        "final_norm/1",
    )
    return dict(
        rank=report["rank"],
        start_audit_passed=True,
        endpoint=report["step0_endpoint"],
        earliest_difference=first,
        first_failed_decoder=report["step0_sites"]["first_failed_decoder"],
        boundaries=[by_label[label] for label in boundaries],
        local=[
            dict(
                layer=item["layer"],
                boundaries=[
                    r
                    for r in item["errors"]
                    if any(
                        r["label"].endswith(suffix)
                        for suffix in (
                            "/recurrence/output",
                            "/gdn/norm",
                            "/gdn/out_proj/0",
                            "/mlp/down_proj/0",
                            "/decoder/1",
                        )
                    )
                ],
            )
            for item in local
        ],
        single_substitution=substitution,
        observed_recurrences=len(recurrence),
        observed_decoder_outputs=len(decoders),
        observation_records=len(rows),
    )


def analyze(directory, original_directory):
    manifest = json.loads((directory / "evidence-manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest.items():
        data = (directory / name).read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], name)
    require(json.loads((directory / "exit.json").read_text())["returncode"] == 0, "failed remote process")
    ranks = []
    for rank in (0, 1):
        report = json.loads((directory / f"step0.rank{rank}.json").read_text(encoding="utf-8"))
        original = json.loads((original_directory / "numeric" / f"numeric.rank{rank}.json").read_text(encoding="utf-8"))
        require(report["rank"] == rank, "changed rank identity")
        ranks.append(summarize(report, original))
    arithmetic = json.loads((directory / "arithmetic.json").read_text(encoding="utf-8"))
    snapshots = json.loads((directory / "snapshot-manifest.json").read_text(encoding="utf-8"))
    require(arithmetic["status"] == "complete_saved_step0_arithmetic_audit", "incomplete arithmetic audit")
    require(len(arithmetic["observations"]) == 4, "incomplete or expanded arithmetic audit")
    for row in arithmetic["observations"]:
        require(row["snapshot_sha256"] == snapshots[row["snapshot"]]["sha256"], "arithmetic snapshot differs")
        require(row["metrics"]["compact/torch.float64"]["worst_request_nrmse"] < 1e-13, "algebra mismatch")
    return dict(
        status="complete_bounded_fp32_diagnosis_candidate_stays_closed",
        trace_sha256=TRACE_SHA256,
        screen_nrmse=SCREEN_NRMSE,
        ranks=ranks,
        arithmetic=arithmetic,
        actual_repair_performed=False,
        local_index_state_or_precision_configuration_bug_found=False,
        exact_native_suboperation_attribution_complete=False,
        remaining_uncertainty=(
            "relative contributions of native Cube reduction, vector reduction, exp and scale reassociation "
            "not individually isolated; no claim of identifying one faulty native instruction"
        ),
        numerical_qualification_passed=False,
        proceed_to_full_numeric_requalification=False,
        retained_ready_scheduling_reduction_percent=4.801135,
        retained_gain_is_same_algorithm_only=True,
        additional_models=0,
        performance_rounds=0,
        fp16_reopened=False,
        candidate_search_continued=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory, args.original)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
