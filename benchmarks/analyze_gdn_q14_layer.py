# SPDX-License-Identifier: Apache-2.0
"""Aggregate the frozen complete-layer TP2 reports without new measurements."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from multi_graph_gdn_q14_layer_protocol import CONFIGS, PROCESSES, REPEATS, summarize

RANKS = (0, 1)
STEPS = tuple(range(54, 70))
COST_FIELDS = ("initialization_ms", "execution_ms", "terminal_ms", "lifecycle_ms")
EVENT_FIELDS = ("prepare_ms", "layer_ms", "total_ms")
EVENT_TOLERANCE = dict(rel_tol=1e-4, abs_tol=1e-4)


def _finite_values(record, fields):
    try:
        values = [record[field] for field in fields]
    except KeyError as error:
        raise ValueError(f"missing measurement field {error.args[0]}") from error
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("measurements must be finite nonnegative numbers")
    return values


def _validate_samples(report, process):
    expected = {(config, sample) for config in CONFIGS for sample in range(REPEATS)}
    values = report.get("samples", [])
    if len(values) != len(expected):
        raise ValueError("each process/rank must contain exactly 16 samples")
    result = {}
    for sample in values:
        if type(sample.get("process")) is not int or sample["process"] != process:
            raise ValueError("sample process differs from its report")
        if type(sample.get("sample")) is not int:
            raise ValueError("sample identity must be an integer")
        key = sample.get("config"), sample["sample"]
        if key not in expected or key in result:
            raise ValueError("sample identity missing, duplicated or outside the frozen comparison")
        initialization, execution, terminal, lifecycle = _finite_values(sample, COST_FIELDS)
        if lifecycle <= 0 or not math.isclose(
            initialization + execution + terminal, lifecycle, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("sample lifecycle is not closed")
        result[key] = sample
    if set(result) != expected:
        raise ValueError("sample identity is incomplete")
    return result


def _validate_events(report, process):
    expected = {(config, step) for config in CONFIGS for step in STEPS}
    values = report.get("segments", [])
    if len(values) != len(expected):
        raise ValueError("each process/rank must contain 32 event endpoints")
    result = {}
    for event in values:
        if type(event.get("process")) is not int or event["process"] != process:
            raise ValueError("event process differs from its report")
        if type(event.get("step")) is not int:
            raise ValueError("event step must be an integer")
        key = event.get("config"), event["step"]
        if key not in expected or key in result:
            raise ValueError("event identity missing, duplicated or outside the frozen comparison")
        if event.get("segment") not in ("q1_only", "mixed"):
            raise ValueError("only the frozen Q1-only and mixed segments are present")
        prepare, layer, total = _finite_values(event, EVENT_FIELDS)
        if total <= 0 or not math.isclose(prepare + layer, total, **EVENT_TOLERANCE):
            raise ValueError("event endpoint total differs from prepare plus layer")
        result[key] = event
    for config in CONFIGS:
        counts = Counter(event["segment"] for (mode, _), event in result.items() if mode == config)
        if counts != {"mixed": 6, "q1_only": 10}:
            raise ValueError("event segment counts differ from the frozen 16-step workload")
    return result


def _costs(samples):
    return dict(
        samples=len(samples),
        mean_ms={field: statistics.mean(sample[field] for sample in samples) for field in COST_FIELDS},
        median_ms={field: statistics.median(sample[field] for sample in samples) for field in COST_FIELDS},
        min_lifecycle_ms=min(sample["lifecycle_ms"] for sample in samples),
        max_lifecycle_ms=max(sample["lifecycle_ms"] for sample in samples),
    )


def _segment_costs(events):
    result = {}
    for config in CONFIGS:
        result[config] = {}
        for kind in ("q1_only", "mixed"):
            process_sums = []
            for process in range(PROCESSES):
                selected = [
                    event
                    for event in events
                    if event["config"] == config and event["process"] == process and event["segment"] == kind
                ]
                process_sums.append(
                    dict(
                        process=process,
                        steps=len(selected),
                        **{field: sum(event[field] for event in selected) for field in EVENT_FIELDS},
                    )
                )
            result[config][kind] = dict(
                steps_per_process=process_sums[0]["steps"],
                process_sums_ms=process_sums,
                mean_process_sum_ms={
                    field: statistics.mean(item[field] for item in process_sums) for field in EVENT_FIELDS
                },
            )
    return result


def analyze(input_dir):
    """Select one complete rank record per sample and per diagnostic step.

    A lifecycle sample uses the rank with the larger whole lifecycle, retaining
    that rank's three components. Event endpoints independently use the larger
    total for the same process/config/step. Event diagnostics are never added to
    lifecycle samples and overlapping endpoint fields are never summed together.
    """
    input_dir = Path(input_dir)
    reports, sources = {}, []
    for process in range(PROCESSES):
        for rank in RANKS:
            path = input_dir / f"performance{process}.rank{rank}.json"
            blob = path.read_bytes()
            report = json.loads(blob)
            if report.get("status") != "completed_performance_process":
                raise ValueError(f"{path.name}: performance process did not complete")
            if type(report.get("rank")) is not int or report["rank"] != rank:
                raise ValueError(f"{path.name}: rank differs from its filename")
            for name in ("fingerprint", "source_sha256", "libraries", "input_file_sha256"):
                if name not in report:
                    raise ValueError(f"{path.name}: missing qualified identity {name}")
            reports[process, rank] = report
            sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(blob).hexdigest()))
    reference = reports[0, 0]
    for report in reports.values():
        for name in ("fingerprint", "source_sha256", "libraries", "settings"):
            if report.get(name) != reference.get(name):
                raise ValueError(f"qualified identity differs between reports: {name}")
    for rank in RANKS:
        if reports[0, rank]["input_file_sha256"] != reports[1, rank]["input_file_sha256"]:
            raise ValueError("rank input changed between independent processes")

    indexed_samples, indexed_events = {}, {}
    raw_samples, raw_events = [], []
    for (process, rank), report in reports.items():
        indexed_samples[process, rank] = _validate_samples(report, process)
        indexed_events[process, rank] = _validate_events(report, process)
        raw_samples.extend(dict(item, rank=rank) for item in report["samples"])
        raw_events.extend(dict(item, rank=rank) for item in report["segments"])

    selected_samples, selected_events = [], []
    for process in range(PROCESSES):
        for config in CONFIGS:
            for sample in range(REPEATS):
                rank = max(RANKS, key=lambda r: indexed_samples[process, r][config, sample]["lifecycle_ms"])
                selected_samples.append(dict(indexed_samples[process, rank][config, sample], selected_rank=rank))
            for step in STEPS:
                pair = [indexed_events[process, rank][config, step] for rank in RANKS]
                if pair[0]["segment"] != pair[1]["segment"]:
                    raise ValueError("paired ranks disagree on event segment")
                if pair[0]["segment"] != indexed_events[0, 0][config, step]["segment"]:
                    raise ValueError("independent processes disagree on event segment")
                rank = max(RANKS, key=lambda r: pair[r]["total_ms"])
                selected_events.append(dict(pair[rank], selected_rank=rank))
    summary = summarize(selected_samples)
    costs = {
        config: _costs([sample for sample in selected_samples if sample["config"] == config]) for config in CONFIGS
    }
    processes = []
    for process in range(PROCESSES):
        processes.append(
            dict(
                summary["processes"][process],
                costs_by_config={
                    config: _costs(
                        [
                            sample
                            for sample in selected_samples
                            if sample["process"] == process and sample["config"] == config
                        ]
                    )
                    for config in CONFIGS
                },
            )
        )
    return dict(
        schema="gdn-q14-complete-layer-tp2-analysis-v1",
        input_reports=sources,
        rank_aggregation="per sample: maximum whole lifecycle rank; all components retained from that same rank",
        component_aggregation="component means are additive; component medians are independent and must not be added",
        event_aggregation=(
            "separate diagnostic run; maximum whole endpoint rank per matching step; "
            "prepare and layer partition total; never add these three overlapping fields together"
        ),
        event_endpoint_closure_tolerance=EVENT_TOLERANCE,
        summary=summary,
        costs_by_config=costs,
        processes=processes,
        selected_samples=selected_samples,
        raw_rank_samples=raw_samples,
        selected_event_endpoints=selected_events,
        raw_rank_event_endpoints=raw_events,
        event_segments=_segment_costs(selected_events),
        qualified_identity={name: reference[name] for name in ("fingerprint", "source_sha256", "libraries")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing analysis")
    report = analyze(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
