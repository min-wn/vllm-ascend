# SPDX-License-Identifier: Apache-2.0
"""Independently replay quality accounting from saved O(tokens) observations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from multi_graph_gdn_q4_full_metadata import load_fixed_trace
from multi_graph_gdn_quality import score_layout, summarize
from multi_graph_gdn_quality_probe import load_inputs, task_summary


def same_statistics(actual, expected):
    """Allow cross-platform libm rounding; decisions and discrete data stay exact."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(same_statistics(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            same_statistics(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float):
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return actual == expected


def analyze(root):
    trace = load_fixed_trace(root / "trace.json")
    frozen = load_inputs(root / "manifest.json", root / "protocol.md", trace)
    payload = frozen["payload"]
    report = json.loads((root / "screen.rank0.json").read_bytes())
    peer = json.loads((root / "screen.rank1.json").read_bytes())
    for rank, document in enumerate((report, peer)):
        if (
            document["status"] != "complete_real_text_quality_audit"
            or document["rank"] != rank
            or document["completed_steps"] != 128
            or not document["terminal_closed"]
            or document["payload_sha256"] != frozen["payload_sha256"]
            or document["manifest_sha256"] != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        ):
            raise ValueError("incomplete paired-rank trajectory or frozen input mismatch")
        if len(document["hidden_diagnostics"]) != 128:
            raise ValueError("hidden diagnostic coverage incomplete")
    if report["prefill"] != peer["prefill"] or report["selection"] != peer["selection"]:
        raise ValueError("TP rank prefill or dispatcher disagrees")
    rows = report["rows"]
    expected = [(step, item) for step, items in enumerate(score_layout(trace)) for item in items]
    if len(rows) != len(expected):
        raise ValueError("missing or extra scored token")
    for row, (step, item) in zip(rows, expected, strict=True):
        source = payload["screen"][item["owner"]]
        if any(
            (
                row["step"] != step,
                row["owner"] != item["owner"],
                row["position"] != item["position"],
                row["target"] != source["tokens"][item["position"]],
                row["article"] != source["article"],
            )
        ):
            raise ValueError("next-token identity/owner/article misalignment")
        if abs(row["delta_nll"] - (row["compact_nll"] - row["official_nll"])) > 1e-6:
            raise ValueError("inconsistent saved paired NLL")
    recomputed = summarize(rows, expected_tokens=payload["screen_expected_tokens"])
    if not same_statistics(recomputed, report["summary"]):
        raise ValueError("stored quality gate or bootstrap does not reproduce")
    segments = []
    for start in range(0, 128, 32):
        part = [row for row in rows if start <= row["step"] < start + 32]
        segments.append(
            dict(
                start_step=start,
                stop_step=start + 31,
                tokens=len(part),
                means={
                    key: sum(r[key] for r in part) / len(part)
                    for key in ("official_nll", "compact_nll", "delta_nll", "kl", "tv", "top1_disagreement")
                },
            )
        )
    mismatches = [row for row in rows if row["top1_disagreement"]]
    result = dict(
        status="audited_real_text_quality",
        summary=recomputed,
        segments=segments,
        disagreeing_tokens=len(mismatches),
        disagreement_official_margin=dict(
            mean=sum(r["official_margin"] for r in mismatches) / max(1, len(mismatches)),
            maximum=max((r["official_margin"] for r in mismatches), default=0),
        ),
        hidden_diagnostic_max=max(
            r["worst_request_nrmse"] for doc in (report, peer) for r in doc["hidden_diagnostics"]
        ),
        hidden_diagnostic_screen_failed_steps=sum(
            any(not doc["hidden_diagnostics"][step]["screen_passed"] for doc in (report, peer)) for step in range(128)
        ),
        task_run=False,
        derived_statistics_audit_tolerance=dict(absolute=1e-12, relative=1e-12, decisions="exact"),
    )
    result["largest_kl_observations"] = sorted(rows, key=lambda row: row["kl"], reverse=True)[:20]
    result["largest_nll_losses"] = sorted(rows, key=lambda row: row["delta_nll"], reverse=True)[:20]
    result["largest_nll_gains"] = sorted(rows, key=lambda row: row["delta_nll"])[:20]
    result["paired_target_accuracy"] = dict(
        official_correct_compact_wrong=sum(
            bool(r["official_correct"]) and not bool(r["compact_correct"]) for r in rows
        ),
        official_wrong_compact_correct=sum(
            bool(r["compact_correct"]) and not bool(r["official_correct"]) for r in rows
        ),
        both_correct=sum(bool(r["official_correct"]) and bool(r["compact_correct"]) for r in rows),
        both_wrong=sum(not bool(r["official_correct"]) and not bool(r["compact_correct"]) for r in rows),
    )
    owner_rows = []
    for owner in sorted({row["owner"] for row in rows}):
        part = [row for row in rows if row["owner"] == owner]
        owner_rows.append(
            dict(
                owner=owner,
                article=part[0]["article"],
                tokens=len(part),
                means={
                    key: sum(r[key] for r in part) / len(part)
                    for key in ("official_nll", "compact_nll", "delta_nll", "kl", "tv", "top1_disagreement")
                },
            )
        )
    result["owners"] = owner_rows
    if (root / "task.rank0.json").exists():
        task = json.loads((root / "task.rank0.json").read_bytes())
        if not recomputed["screen_passed"]:
            raise ValueError("task ran despite failed predeclared screen")
        if task["status"] != "complete_real_text_quality_audit" or not task["terminal_closed"]:
            raise ValueError("incomplete limited task audit")
        task_rows = task["rows"]
        task_expected = [(item, j) for item in payload["task"] for j in range(len(item["target_tokens"]))]
        if len(task_rows) != len(task_expected):
            raise ValueError("incomplete target word")
        for row, (item, j) in zip(task_rows, task_expected, strict=True):
            if (
                row["owner"] != item["owner"]
                or row["position"] != item["prefix"] + 1 + j
                or row["target"] != item["target_tokens"][j]
            ):
                raise ValueError("task word/token alignment mismatch")
        summary = task_summary(task_rows)
        if not same_statistics(summary, task["summary"]):
            raise ValueError("task paired accounting does not reproduce")
        result.update(task_run=True, task=summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.root), indent=2) + "\n")


if __name__ == "__main__":
    main()
