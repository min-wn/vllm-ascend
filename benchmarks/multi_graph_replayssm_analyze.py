# SPDX-License-Identifier: Apache-2.0
"""Audit the frozen A2 ReplaySSM lifecycle and isolate its ready contribution."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

from multi_graph_replayssm_protocol import COMMUNITY_COMMIT, Cursor, coverage, frozen_trace

MODES = ("official", "compact_ready", "ring_delayed", "ring_ready")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(document):
    trace, terminal = frozen_trace()
    if (
        document["status"] != "qualified_40_step_real_layer_closed_lifecycle"
        or document["community_commit"] != COMMUNITY_COMMIT
        or document["trace"] != trace
        or document["terminal_counts"] != terminal
        or document["coverage"] != coverage(trace)
        or document["precision"]["ring_dk"] != "FP32"
        or {rank["rank"] for rank in document["ranks"]} != {0, 1}
    ):
        raise ValueError("the frozen qualified FP32 A2 lifecycle is required")
    for rank in document["ranks"]:
        wanted = {(step, mode) for step in range(40) for mode in MODES}
        if len(rank["checks"]) != len(wanted) or {(row["step"], row["mode"]) for row in rank["checks"]} != wanted:
            raise ValueError("incomplete continuous qualification")
        if not all(
            row["continuous"] and (row["mode"] != "ring_ready" or row["same_algorithm_exact"]) for row in rank["checks"]
        ):
            raise ValueError("continuous exact delayed/ready qualification is required")
        for mode in MODES[1:]:
            labels = {f"{mode}/hidden", f"{mode}/residual", f"{mode}/terminal_state"}
            if {row["label"] for row in rank["errors"] if row["label"].startswith(mode + "/")} != labels:
                raise ValueError("missing output/closed-state qualification")
        if not all(
            math.isfinite(row["worst_request_nrmse"]) and row["worst_request_nrmse"] <= 0.005 for row in rank["errors"]
        ):
            raise ValueError("frozen precision threshold failed")
        wanted = {(round_id, mode) for round_id in range(3) for mode in MODES}
        if len(rank["samples"]) != len(wanted) or {(row["round"], row["mode"]) for row in rank["samples"]} != wanted:
            raise ValueError("three complete paired rounds required")
        for sample in rank["samples"]:
            if not sample["terminal_exact"] or len(sample["step_device_us"]) != 40:
                raise ValueError("timed trajectory did not close to the qualified state")
            if not math.isclose(sample["closed_wall_us"], sample["wall_us"] + sample["import_us"]):
                raise ValueError("import must be charged")
    rows = []
    for round_id in range(3):
        row = dict(round=round_id)
        for mode in MODES:
            samples = [
                next(sample for sample in rank["samples"] if sample["round"] == round_id and sample["mode"] == mode)
                for rank in document["ranks"]
            ]
            row[mode] = {
                name: max(sample[name] for sample in samples)
                for name in ("wall_us", "closed_wall_us", "device_us", "terminal_us")
            }
        for control, name in (
            ("ring_delayed", "ready_net_percent"),
            ("compact_ready", "ready_vs_compact_percent"),
            ("official", "ready_vs_official_percent"),
        ):
            row[name] = 100 * (1 - row["ring_ready"]["closed_wall_us"] / row[control]["closed_wall_us"])
        for key, value in document["paired"][round_id].items():
            if value != row[key]:
                raise ValueError(f"stored paired result differs: {key}")
        rows.append(row)
    return rows


def analyze(directory):
    documents = [read(directory / f"lifecycle-v{index}.json") for index in (1, 2)]
    if documents[0]["source_sha256"] != documents[1]["source_sha256"]:
        raise ValueError("independent processes must measure the same sources")
    manifest = read(directory / "evidence-manifest.json")
    for name, entry in manifest.items():
        blob = (directory / name).read_bytes()
        if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise ValueError(f"raw evidence checksum mismatch: {name}")
    sources = read(directory / "measured-source-manifest.json")
    for name, digest in documents[0]["source_sha256"].items():
        if sources["tools/" + name] != digest:
            raise ValueError(f"measured source mismatch: {name}")
    with tarfile.open(directory / "fp32-measured-sources.tar.gz") as archive:
        for name, digest in sources.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError(f"archived source mismatch: {name}")
    diagnostic = read(directory / "diagnostic.json")
    if diagnostic["status"] != "diagnostic_only" or any(rank["precision_gate_passed"] for rank in diagnostic["ranks"]):
        raise ValueError("the failed original FP16 configuration must remain a diagnostic")
    fp16_error = max(
        row["worst_request_nrmse"]
        for rank in diagnostic["ranks"]
        for row in rank["dense_errors"]
        if row["label"].startswith("ring_")
    )
    if fp16_error <= 0.005:
        raise ValueError("expected the archived FP16 precision failure")
    rows = [dict(process=process, **row) for process, document in enumerate(documents, 1) for row in audit(document)]
    errors = [row for document in documents for rank in document["ranks"] for row in rank["errors"]]
    trace = documents[0]["trace"]
    cursors = [Cursor() for _ in range(100)]
    flush_by_step = []
    for entry in trace:
        flushed = 0
        for owner in range(entry["live"]):
            cursors[owner] = cursors[owner].advance(entry["accepted"][owner])
            flushed += int(cursors[owner].is_flush)
        flush_by_step.append(flushed)
    summary = {
        mode: {
            field: statistics.mean(row[mode][field] for row in rows)
            for field in ("wall_us", "closed_wall_us", "device_us", "terminal_us")
        }
        for mode in MODES
    }
    return dict(
        decision="retain_ready_as_conditional_A2_execution_mainline_not_new_state_algorithm",
        original_fp16_gate=dict(status="failed_no_timing_claim", max_output_nrmse=fp16_error, threshold=0.005),
        scope=documents[0]["scope"],
        precision=documents[0]["precision"],
        paired=rows,
        means=summary,
        ready_net_percent=[row["ready_net_percent"] for row in rows],
        ready_net_mean_percent=statistics.mean(row["ready_net_percent"] for row in rows),
        ready_net_pooled_percent=100
        * (1 - summary["ring_ready"]["closed_wall_us"] / summary["ring_delayed"]["closed_wall_us"]),
        ready_vs_compact_percent=[row["ready_vs_compact_percent"] for row in rows],
        ready_vs_official_percent=[row["ready_vs_official_percent"] for row in rows],
        max_output_nrmse={
            mode: max(
                row["worst_request_nrmse"] for row in errors if row["label"] in (mode + "/hidden", mode + "/residual")
            )
            for mode in MODES[1:]
        },
        max_terminal_state_nrmse={
            mode: max(row["worst_request_nrmse"] for row in errors if row["label"] == mode + "/terminal_state")
            for mode in MODES[1:]
        },
        imports_per_rank_us=[
            {key: rank[key] for key in ("ring_import_us", "compact_import_us")}
            for document in documents
            for rank in document["ranks"]
        ],
        persistent_bytes_per_rank=documents[0]["ranks"][0]["persistent_bytes"],
        coverage=coverage(trace),
        flush_requests_per_step=flush_by_step,
        qualification_checks=sum(len(rank["checks"]) for document in documents for rank in document["ranks"]),
        timed_terminal_exact_checks=sum(len(rank["samples"]) for document in documents for rank in document["ranks"]),
        boundary=documents[0]["boundary"],
        limitations=[
            "FP32 d/k is an A2 precision variant, not the community BF16-activation FP16-log configuration.",
            "A2 port uses separate pre-output sequential flush, duplicated keys and padded scalars; "
            "no GPU performance parity.",
            "Real layer weights with synthetic hidden/residual and continuous state; "
            "no model, service, quality or tail claim.",
            "One import observation per process/rank, not a stable initialization distribution.",
            "Frozen existing owners only; paged cache, eviction, reuse and prefill transitions not implemented.",
            "Event span includes descriptor/input upload; grouping flush/nonflush steps does not isolate flush cost.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "decision",
                    "ready_net_percent",
                    "means",
                    "max_output_nrmse",
                    "max_terminal_state_nrmse",
                    "persistent_bytes_per_rank",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
