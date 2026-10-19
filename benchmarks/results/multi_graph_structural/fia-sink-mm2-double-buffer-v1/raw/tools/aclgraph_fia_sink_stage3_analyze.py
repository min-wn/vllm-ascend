# SPDX-License-Identifier: Apache-2.0
"""Analyze isolated stage-three timing evidence using only the standard library.

The archived trace and installed-default selection independently define all
expected step/batch/capacity triples. Missing samples cannot reduce that contract.
Queue spans include host submission gaps; no result is a pure kernel duration or
full-model speedup. This tool never imports the native runner, Torch, or vLLM.
"""

import argparse
import bisect
import hashlib
import json
import math
import statistics
from pathlib import Path

CANONICAL_TRACE_SHA256 = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
MODES = ("per_layer", "shared", "attention_only", "planner", "legacy")
METRICS = ("wall_us", "queue_span_us", "preparation_enqueue_us", "submit_join_cpu_us")
QUERY_LENGTH = 4
STEP_COUNT = 128
DEFAULT_TRIALS = 3
ATTENTION_OCCURRENCES = 16
FULL_MODEL_REFERENCE_US = 143474.0
TARGET_FRACTION = 0.05
RESULTS_DIRECTORY = Path(__file__).parent / "results" / "multi_graph_structural"


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def statistics_us(values):
    """Linear-interpolated percentiles over finite observations, not percentiles of means."""
    if not values:
        return dict(count=0, mean_us=None, p50_us=None, p95_us=None, p99_us=None)
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return dict(
        count=len(values),
        mean_us=statistics.mean(values),
        p50_us=percentile(0.5),
        p95_us=percentile(0.95),
        p99_us=percentile(0.99),
    )


def expected_schedule(trace_path=None, selection_path=None):
    """Rebuild expected dispatch from independently pinned repository artifacts."""
    trace_path = Path(trace_path or RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json")
    selection_path = Path(selection_path or RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json")
    trace_artifact, selection_artifact = load_json(trace_path), load_json(selection_path)
    trace = trace_artifact["trace"]
    normalized = dict(trace)
    prefixes = trace["initial_prefix_by_request"]
    if not isinstance(prefixes, dict) or any(not str(key).isdecimal() or str(int(key)) != str(key) for key in prefixes):
        raise ValueError("canonical trace has invalid initial owner keys")
    normalized["initial_prefix_by_request"] = {int(key): value for key, value in prefixes.items()}
    if not (
        digest(normalized) == trace_artifact.get("trace_sha256") == CANONICAL_TRACE_SHA256
        and trace.get("query") == QUERY_LENGTH
        and isinstance(trace.get("steps"), list)
        and len(trace["steps"]) == STEP_COUNT
        and selection_artifact.get("trace_sha256") == CANONICAL_TRACE_SHA256
    ):
        raise ValueError("canonical trace or capture evidence does not match the pinned 128-step contract")
    selection = selection_artifact["selection"]
    catalog, dispatched = selection["capture_sizes"], selection["dispatched_tokens"]
    if not (
        isinstance(catalog, list)
        and catalog
        and all(integer(value, 1) for value in catalog)
        and catalog == sorted(set(catalog))
        and isinstance(dispatched, list)
        and len(dispatched) == STEP_COUNT
        and selection.get("query") == QUERY_LENGTH
        and selection.get("config_source") == "installed EngineArgs defaults; no capture sizes supplied"
    ):
        raise ValueError("invalid archived installed-default capture contract")
    steps = []
    for index, step in enumerate(trace["steps"]):
        batch = step.get("batch")
        if step.get("step") != index or not integer(batch, 1):
            raise ValueError("invalid canonical step index or batch")
        offset = bisect.bisect_left(catalog, batch * QUERY_LENGTH)
        if offset == len(catalog) or catalog[offset] % QUERY_LENGTH or dispatched[index] != catalog[offset]:
            raise ValueError("archived dispatch is not the smallest legal default capture capacity")
        steps.append(dict(step=index, batch=batch, capacity=catalog[offset]))
    return dict(
        canonical_digest=CANONICAL_TRACE_SHA256,
        records=steps,
        default_capture_sizes=catalog,
        trace_file_sha256=sha256(trace_path),
        selection_sha256=sha256(selection_path),
    )


def measurement_contract(report, expected):
    configuration = object_or_empty(report.get("configuration"))
    trials = report.get("trials", configuration.get("trials"))
    requested_steps = report.get("screen_steps", configuration.get("screen_steps"))
    issues = []
    if not integer(trials, 1):
        issues.append("missing or invalid explicit trials; sample counts cannot define the expected trial count")
        trials = 0
    if not integer(requested_steps, 1) or requested_steps > STEP_COUNT:
        issues.append("missing or invalid explicit screen_steps in 1..128")
        requested_steps = 0
    modes = report.get("modes", configuration.get("modes"))
    if (
        not isinstance(modes, list)
        or len(modes) != len(MODES)
        or any(not isinstance(mode, str) for mode in modes)
        or set(modes) != set(MODES)
    ):
        issues.append("all five explicitly declared timing modes are required")
    schedule = object_or_empty(report.get("schedule"))
    canonical = schedule.get("canonical_digest", report.get("canonical_digest", report.get("trace_sha256")))
    if canonical != CANONICAL_TRACE_SHA256:
        issues.append("reported schedule canonical digest does not match the pinned trace")
    catalog = schedule.get("default_capture_sizes")
    if catalog != expected["default_capture_sizes"]:
        issues.append("reported capture catalog differs from the archived public default catalog")
    records = schedule.get("records", schedule.get("steps"))
    if not isinstance(records, list) or len(records) != STEP_COUNT:
        issues.append("reported schedule must retain all 128 canonical steps, including in a screen")
    else:
        for index, (actual, wanted) in enumerate(zip(records, expected["records"], strict=True)):
            actual = object_or_empty(actual)
            capacity = actual.get("capacity", actual.get("physical_tokens", actual.get("default_capacity")))
            if (
                type(actual.get("step")) is not int
                or type(actual.get("batch")) is not int
                or type(capacity) is not int
                or actual["step"] != wanted["step"]
                or actual["batch"] != wanted["batch"]
                or capacity != wanted["capacity"]
            ):
                issues.append(f"reported schedule step/batch/capacity mismatch at index {index}")
                break
    for name in ("trace_file_sha256", "selection_sha256"):
        if schedule.get(name) != expected[name]:
            issues.append(f"reported schedule {name} differs from the independent input artifact")
    return trials, requested_steps, issues


def collect_samples(report, expected, trials, requested_steps):
    samples = report.get("samples")
    if not isinstance(samples, list):
        return {}, dict(complete=False, issues=["samples must be a list"])
    wanted = {(trial, mode, step) for trial in range(trials) for mode in MODES for step in range(requested_steps)}
    indexed, duplicates, unexpected, invalid = {}, [], [], []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            invalid.append(dict(index=index, reason="sample is not an object"))
            continue
        trial, mode, step = (sample.get(name) for name in ("trial", "mode", "step"))
        if not integer(trial) or not isinstance(mode, str) or not integer(step):
            invalid.append(dict(index=index, reason="invalid trial/mode/step fields"))
            continue
        key = (trial, mode, step)
        if key not in wanted:
            unexpected.append(dict(index=index, trial=trial, mode=mode, step=step))
            continue
        if key in indexed:
            duplicates.append(dict(index=index, trial=trial, mode=mode, step=step))
            continue
        schedule = expected["records"][step]
        if (
            type(sample.get("batch")) is not int
            or sample["batch"] != schedule["batch"]
            or type(sample.get("capacity")) is not int
            or sample["capacity"] != schedule["capacity"]
        ):
            invalid.append(dict(index=index, reason="batch/capacity differs from canonical step"))
            continue
        if any(not finite(sample.get(metric)) for metric in METRICS) or sample["wall_us"] <= 0:
            invalid.append(
                dict(index=index, reason="timing fields must be finite nonnegative numbers; wall must be positive")
            )
            continue
        indexed[key] = sample
    missing = [dict(trial=trial, mode=mode, step=step) for trial, mode, step in sorted(wanted - indexed.keys())]
    return indexed, dict(
        complete=bool(wanted) and not (duplicates or unexpected or invalid or missing),
        expected_sample_count=len(wanted),
        raw_sample_count=len(samples),
        distinct_valid_sample_count=len(indexed),
        missing=missing,
        duplicates=duplicates,
        unexpected=unexpected,
        invalid=invalid,
        issues=[],
    )


def paired_comparison(indexed, left, right, *, diagnostic, meaning):
    rows = []
    for (trial, mode, step), first in sorted(indexed.items()):
        if mode != left or (trial, right, step) not in indexed:
            continue
        second = indexed[trial, right, step]
        rows.append(
            dict(
                trial=trial,
                step=step,
                batch=first["batch"],
                capacity=first["capacity"],
                baseline_wall_us=first["wall_us"],
                comparison_wall_us=second["wall_us"],
                wall_difference_us=first["wall_us"] - second["wall_us"],
                queue_span_difference_us=first["queue_span_us"] - second["queue_span_us"],
                reduction_percent=100 * (1 - second["wall_us"] / first["wall_us"]),
            )
        )
    mean_left = statistics.mean(row["baseline_wall_us"] for row in rows) if rows else None
    mean_delta = statistics.mean(row["wall_difference_us"] for row in rows) if rows else None
    per_trial = []
    for trial in sorted({row["trial"] for row in rows}):
        selected = [row for row in rows if row["trial"] == trial]
        per_trial.append(
            dict(
                trial=trial,
                paired_step_count=len(selected),
                wall_difference=statistics_us([row["wall_difference_us"] for row in selected]),
                trajectory_difference_us=sum(row["wall_difference_us"] for row in selected),
            )
        )
    per_capacity = []
    for capacity in sorted({row["capacity"] for row in rows}):
        selected = [row for row in rows if row["capacity"] == capacity]
        baseline = statistics.mean(row["baseline_wall_us"] for row in selected)
        comparison = statistics.mean(row["comparison_wall_us"] for row in selected)
        per_capacity.append(
            dict(
                capacity=capacity,
                paired_count=len(selected),
                distinct_step_count=len({row["step"] for row in selected}),
                baseline_mean_wall_us=baseline,
                comparison_mean_wall_us=comparison,
                wall_difference=statistics_us([row["wall_difference_us"] for row in selected]),
                queue_span_difference=statistics_us([row["queue_span_difference_us"] for row in selected]),
                aggregate_wall_reduction_percent=100 * (1 - comparison / baseline),
                comparison_faster_count=sum(row["wall_difference_us"] > 0 for row in selected),
                comparison_slower_count=sum(row["wall_difference_us"] < 0 for row in selected),
                equal_count=sum(row["wall_difference_us"] == 0 for row in selected),
            )
        )
    return dict(
        baseline=left,
        comparison=right,
        diagnostic_only=diagnostic,
        interpretation=meaning,
        paired_count=len(rows),
        wall_difference=statistics_us([row["wall_difference_us"] for row in rows]),
        queue_span_difference=statistics_us([row["queue_span_difference_us"] for row in rows]),
        aggregate_wall_reduction_percent=100 * mean_delta / mean_left if rows else None,
        comparison_faster_count=sum(row["wall_difference_us"] > 0 for row in rows),
        comparison_slower_count=sum(row["wall_difference_us"] < 0 for row in rows),
        equal_count=sum(row["wall_difference_us"] == 0 for row in rows),
        per_trial=per_trial,
        per_capacity=per_capacity,
        pairs=rows,
    )


def execution_qualification(report, expected, trials, requested_steps, coverage_valid):
    """Separate timing-table coverage from the actual stage-three protocol."""
    state = report.get("state", report.get("status"))
    catalog = object_or_empty(report.get("installed_catalog"))
    selection = object_or_empty(catalog.get("selection"))
    schedule = object_or_empty(report.get("schedule"))
    checks = dict(
        numeric_coverage_complete=coverage_valid,
        completed_execution_state=state == "completed",
        successful_returncode=type(report.get("returncode")) is int and report["returncode"] == 0,
        npu_execution_reported=report.get("npu_work_executed") is True,
        performance_measurement_reported=report.get("performance_measured") is True,
        all_canonical_steps=requested_steps == STEP_COUNT,
        at_least_three_trials=trials >= DEFAULT_TRIALS,
        sixteen_attention_occurrences=type(report.get("occurrences")) is int
        and report["occurrences"] == ATTENTION_OCCURRENCES,
        installed_catalog_revalidation_reported=schedule.get("installed_catalog_revalidated") is True
        and catalog.get("installed_catalog_revalidated") is True,
        installed_catalog_matches_expected=selection.get("capture_sizes") == expected["default_capture_sizes"]
        and selection.get("dispatched_tokens") == [record["capacity"] for record in expected["records"]],
    )
    return dict(
        complete=all(checks.values()),
        checks=checks,
        unqualified_checks=[name for name, passed in checks.items() if not passed],
        correctness_scope=report.get("qualification_scope"),
        limitation="Timing protocol qualification does not establish all-step correctness or full-model accuracy.",
    )


def analyze(report, *, trace_path=None, selection_path=None):
    """Return descriptive statistics, strict coverage and bounded attribution."""
    if not isinstance(report, dict):
        raise ValueError("timing report must be a JSON object")
    expected = expected_schedule(trace_path, selection_path)
    trials, requested_steps, issues = measurement_contract(report, expected)
    indexed, coverage = collect_samples(report, expected, trials, requested_steps)
    summaries = {
        mode: {
            metric: statistics_us([row[metric] for (_, current, _), row in indexed.items() if current == mode])
            for metric in METRICS
        }
        for mode in MODES
    }
    trial_summaries = [
        dict(
            trial=trial,
            modes={
                mode: {
                    metric: statistics_us(
                        [
                            row[metric]
                            for (current_trial, current_mode, _), row in indexed.items()
                            if current_trial == trial and current_mode == mode
                        ]
                    )
                    for metric in METRICS
                }
                for mode in MODES
            },
        )
        for trial in range(trials)
    ]
    comparisons = dict(
        per_layer_minus_shared=paired_comparison(
            indexed,
            "per_layer",
            "shared",
            diagnostic=False,
            meaning="Observed attention-chain path difference from removing fifteen repeated metadata builds; "
            "sharing alone is not this thesis's independent algorithm.",
        ),
        shared_minus_attention_only=paired_comparison(
            indexed,
            "shared",
            "attention_only",
            diagnostic=True,
            meaning="Graph-outside real-metadata oracle diagnostic. Excluded planning is required dynamic work; "
            "the difference is not a pure planner duration or a valid dynamic execution speedup.",
        ),
        legacy_minus_shared=paired_comparison(
            indexed,
            "legacy",
            "shared",
            diagnostic=False,
            meaning="Isolated attention-chain difference includes legacy host updates and sink kernel replacement. "
            "It does not separate planner savings from attention arithmetic or prove full-model improvement.",
        ),
    )
    valid = not issues and coverage["complete"]
    qualification = execution_qualification(report, expected, trials, requested_steps, valid)
    measured = all(
        qualification["checks"][name]
        for name in (
            "completed_execution_state",
            "successful_returncode",
            "npu_execution_reported",
            "performance_measurement_reported",
        )
    )
    full_trace = valid and measured and requested_steps == STEP_COUNT
    planner_mean = summaries["planner"]["wall_us"]["mean_us"]
    shared_diagnostic = comparisons["shared_minus_attention_only"]["wall_difference"]["mean_us"]
    required = FULL_MODEL_REFERENCE_US * TARGET_FRACTION
    return dict(
        schema_version=1,
        state="complete_trace" if full_trace else "complete_screen" if valid and measured else "partial_or_invalid",
        analysis_valid=valid,
        issues=issues,
        coverage=coverage,
        canonical_digest=CANONICAL_TRACE_SHA256,
        requested_trials=trials,
        default_trials=DEFAULT_TRIALS,
        requested_steps=requested_steps,
        canonical_step_count=STEP_COUNT,
        full_128_step_trace_measured=full_trace,
        three_trial_trace_protocol_complete=qualification["complete"],
        qualification=qualification,
        screen_only=requested_steps != STEP_COUNT,
        npu_execution_reported=report.get("npu_work_executed") is True,
        performance_measurement_reported=report.get("performance_measured") is True,
        installed_catalog_revalidated=object_or_empty(report.get("schedule")).get("installed_catalog_revalidated")
        is True,
        summaries=summaries,
        trial_summaries=trial_summaries,
        comparisons=comparisons,
        planning_budget=dict(
            single_planner_wall=summaries["planner"]["wall_us"],
            single_planner_queue_span=summaries["planner"]["queue_span_us"],
            single_planner_preparation_enqueue=summaries["planner"]["preparation_enqueue_us"],
            single_planner_after_preparation_host_elapsed=statistics_us(
                [
                    row["wall_us"] - row["preparation_enqueue_us"]
                    for (_, mode, _), row in indexed.items()
                    if mode == "planner"
                ]
            ),
            shared_minus_attention_only_mean_us=shared_diagnostic,
            full_model_reference_us=FULL_MODEL_REFERENCE_US,
            full_model_target_fraction=TARGET_FRACTION,
            full_model_required_net_saving_us=required,
            planner_mean_fraction_of_required_saving=planner_mean / required if planner_mean is not None else None,
            planner_saving_multiplier_against_shared_baseline=1,
            planner_endpoint_limitation="Planner wall includes common host descriptor preparation, H2D enqueue, "
            "submission and synchronization. It is not the planner kernel cost. Subtracting host preparation "
            "still leaves host/device dependency and synchronization costs, rather than pure device computation.",
            interpretation="The shared baseline has one planner. Incremental scheduling can save only its exposed cost "
            "plus measured kernel reductions, minus every added maintenance/copy/synchronization cost. "
            "Neither the planner-only wall measurement nor an oracle difference proves achievable exposed savings.",
        ),
        reference_scale_comparisons={
            name: dict(
                isolated_chain_mean_difference_us=comparison["wall_difference"]["mean_us"],
                full_model_reference_us=FULL_MODEL_REFERENCE_US,
                percent_of_full_model_reference=100 * comparison["wall_difference"]["mean_us"] / FULL_MODEL_REFERENCE_US
                if comparison["wall_difference"]["mean_us"] is not None
                else None,
                end_to_end_prediction=False,
                interpretation="Numerical scale comparison across different measurement scopes only; "
                "not an end-to-end speedup prediction or bound.",
            )
            for name, comparison in comparisons.items()
        },
        timing_boundaries=dict(
            wall_us="Runner-defined host preparation through final completion; verify the raw runner scope.",
            queue_span_us="External timing events include host launch gaps and dependency waits; not pure kernel time.",
            preparation_enqueue_us="Host enqueue interval; asynchronous device copies need not have completed.",
            submit_join_cpu_us="Runner-defined submission/join interval; do not add it to overlapping queue spans.",
            aggregation="P50/P95/P99 cover step/trial observations. Differences match the same trial and step.",
        ),
        expected_schedule_sources={key: value for key, value in expected.items() if key != "records"},
        full_model_measured=False,
        end_to_end_speedup_proven=False,
        independent_algorithm_speedup_proven=False,
        scope="Isolated attention chain; model, serving, TP communication and incremental planner are unmeasured.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    protected = [
        source,
        Path(args.trace or RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json").resolve(),
        Path(args.selection or RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json").resolve(),
    ]
    if output in protected:
        parser.error("--output cannot overwrite the timing report or independent schedule evidence")
    raw_report = load_json(source)
    report = analyze(raw_report, trace_path=args.trace, selection_path=args.selection)
    report["input"] = dict(path=str(source), sha256=sha256(source))
    projection = object_or_empty(raw_report.get("evidence_projection"))
    if projection:
        report["input"]["evidence_projection"] = projection
        report["input"]["original_report_sha256"] = projection.get("original_report_sha256")
    report["analyzer_sha256"] = sha256(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(dict(output=str(output), state=report["state"], analysis_valid=report["analysis_valid"])))
    return 0 if report["analysis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
