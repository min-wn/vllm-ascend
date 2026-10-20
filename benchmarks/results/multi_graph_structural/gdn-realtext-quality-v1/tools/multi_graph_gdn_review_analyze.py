# SPDX-License-Identifier: Apache-2.0
"""Recompute the bounded GDN review without attributing common math to novelty."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def reductions(before, after):
    return [100 * (1 - current / previous) for previous, current in zip(before, after, strict=True)]


def analyze(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["status"] != "qualified_dynamic_real_gdn_layer_lifecycle_not_model":
        raise ValueError("qualified continuous real-layer lifecycle required")
    lifecycle = document["lifecycle"]
    summary = lifecycle["summary"]
    controls = [mode for mode in summary if mode != "compact_deferred_tile_ready"]
    best = min(controls, key=lambda mode: statistics.mean(summary[mode]["round_wall_us"]))
    candidate = summary["compact_deferred_tile_ready"]
    comparisons = {}
    for control in controls:
        prior = summary[control]
        comparisons[control] = {
            field: reductions(prior[field], candidate[field])
            for field in ("round_device_us", "round_wall_us", "one_import_plus_trajectory_wall_us")
        }
    optimistic_control = [
        min(values) for values in zip(*(summary[mode]["round_wall_us"] for mode in controls), strict=True)
    ]
    samples = lifecycle["samples"]
    return dict(
        source_file=path.name,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope="eight continuous steps of one real GDN layer per round; no serving/model result",
        measured_steps=len(document["qualification_schedule"]),
        best_nonready_control_by_mean_wall=best,
        comparisons=comparisons,
        versus_best_per_round_wall_percent=reductions(optimistic_control, candidate["round_wall_us"]),
        one_time_compact_import_wall_us=lifecycle["one_time_compact_import_wall_us"],
        terminal_device_us={
            mode: [row["terminal_device_us"] for row in samples if row["mode"] == mode] for mode in summary
        },
        worst_terminal_request_nrmse=max(
            rank["lifecycle"]["terminal_worst_request_nrmse"] for rank in document["ranks"]
        ),
        qualifications=dict(
            rank_step_mode_checks=sum(len(rank["checks"]) for rank in document["ranks"]),
            rank_round_mode_lifecycle_checks=sum(len(rank["lifecycle"]["checks"]) for rank in document["ranks"]),
            post_terminal_pending_logs=False,
        ),
        limitations=[
            "No matched A2 RecoverSSM or periodic-flush ReplaySSM implementation is measured.",
            "One-time import is one observation per process, not a stable initialization distribution.",
            "Owner pause/resume is covered; arbitrary owner eviction/reuse and production KV integration are not.",
            "Per-step completion and terminal drain are charged; graph setup and fixture generation are excluded.",
            "Eight-step correctness and no output validation of a serving workload; no tail claim.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = dict(
        decision="retain_as_conditional_mainline_candidate_not_established_independent_algorithm",
        rounds=[analyze(path) for path in sorted(args.directory.glob("lifecycle-v*.json"))],
        decisive_missing_evidence=(
            "same-A2 matched compact-state community control versus ready schedule over the same closed lifecycle, "
            "with sustained correctness and repeatable net benefit"
        ),
    )
    (args.directory / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for item in result["rounds"]:
        print(
            item["source_file"], item["best_nonready_control_by_mean_wall"], item["versus_best_per_round_wall_percent"]
        )


if __name__ == "__main__":
    main()
