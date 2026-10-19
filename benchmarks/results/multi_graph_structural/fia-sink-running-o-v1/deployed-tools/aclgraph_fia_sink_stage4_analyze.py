# SPDX-License-Identifier: Apache-2.0
"""Audit the four preparation controls without importing Torch or a device runner.

Schedule expectations come from the independently pinned stage-three artifacts.
All results concern one isolated attention chain. Queue spans contain launch
gaps and dependencies; preparation enqueue times are not completed-copy times.
"""

import argparse
import json
from pathlib import Path

import aclgraph_fia_sink_stage3_analyze as stage3

CASES = ("sink_fresh", "sink_reuse", "legacy_fresh", "legacy_reuse")
METRICS = stage3.METRICS
EXTRA_DIFFERENCES = ("preparation_enqueue_us", "submit_join_cpu_us")
ORDERING_STRATEGIES = ("step_rotation_reverse_odd_trial_v1", "reverse_odd_trial_v1")


def measurement_contract(report, expected):
    configuration = stage3.object_or_empty(report.get("configuration"))
    trials = report.get("trials", configuration.get("trials"))
    steps = report.get("screen_steps", configuration.get("screen_steps", configuration.get("steps")))
    cases = report.get("cases", configuration.get("cases"))
    issues = []
    if not stage3.integer(trials, 1):
        issues.append("explicit positive integer trials required; sample counts cannot define the trial contract")
        trials = 0
    if not stage3.integer(steps, 1) or steps > stage3.STEP_COUNT:
        issues.append("explicit screen_steps/configuration.steps in 1..128 required")
        steps = 0
    if (
        not isinstance(cases, list)
        or len(cases) != len(CASES)
        or any(not isinstance(case, str) for case in cases)
        or set(cases) != set(CASES)
    ):
        issues.append("exactly the four fresh/reuse sink/legacy preparation controls must be declared")
    schedule = stage3.object_or_empty(report.get("schedule"))
    for key in ("canonical_digest", "default_capture_sizes", "trace_file_sha256", "selection_sha256"):
        if schedule.get(key) != expected[key]:
            issues.append(f"schedule.{key} differs from independent pinned evidence")
    records = schedule.get("records", schedule.get("steps"))
    if not isinstance(records, list) or len(records) != stage3.STEP_COUNT:
        issues.append("retain the complete 128-step schedule, including for a short screen")
    else:
        for index, (actual, wanted) in enumerate(zip(records, expected["records"], strict=True)):
            actual = stage3.object_or_empty(actual)
            capacity = actual.get("capacity", actual.get("physical_tokens", actual.get("default_capacity")))
            if (
                type(actual.get("step")) is not int
                or type(actual.get("batch")) is not int
                or type(capacity) is not int
                or actual["step"] != wanted["step"]
                or actual["batch"] != wanted["batch"]
                or capacity != wanted["capacity"]
            ):
                issues.append(f"schedule step/batch/capacity mismatch at index {index}")
                break
    return trials, steps, issues


def collect_samples(report, expected, trials, steps):
    samples = report.get("samples")
    if not isinstance(samples, list):
        return {}, dict(complete=False, issues=["samples must be a list"])
    wanted = {(trial, case, step) for trial in range(trials) for case in CASES for step in range(steps)}
    indexed, seen, duplicates, invalid, unexpected = {}, set(), [], [], []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            invalid.append(dict(index=index, reason="sample is not an object"))
            continue
        trial, case, step = (sample.get(name) for name in ("trial", "case", "step"))
        if not stage3.integer(trial) or not isinstance(case, str) or not stage3.integer(step):
            invalid.append(dict(index=index, reason="invalid trial/case/step fields"))
            continue
        key = (trial, case, step)
        identity = dict(index=index, trial=trial, case=case, step=step)
        if key not in wanted:
            unexpected.append(identity)
            continue
        if key in seen:
            duplicates.append(identity)
            continue
        seen.add(key)
        record = expected["records"][step]
        if any(type(sample.get(name)) is not int or sample[name] != record[name] for name in ("batch", "capacity")):
            invalid.append(dict(**identity, reason="sample batch/capacity differs from canonical step"))
            continue
        if any(not stage3.finite(sample.get(metric)) for metric in METRICS) or sample["wall_us"] <= 0:
            invalid.append(dict(**identity, reason="timings must be finite nonnegative numbers; wall must be positive"))
            continue
        indexed[key] = sample
    missing = [dict(trial=trial, case=case, step=step) for trial, case, step in sorted(wanted - indexed.keys())]
    return indexed, dict(
        complete=bool(wanted) and not (duplicates or invalid or unexpected or missing),
        expected_sample_count=len(wanted),
        raw_sample_count=len(samples),
        distinct_valid_sample_count=len(indexed),
        missing=missing,
        duplicates=duplicates,
        invalid=invalid,
        unexpected=unexpected,
        issues=[],
    )


def recorded_case_order(report, indexed, trials, steps):
    strategy = report.get("ordering_strategy")
    if strategy not in ORDERING_STRATEGIES:
        return False
    order = {(trial, step): [] for trial in range(trials) for step in range(steps)}
    sequences = {(trial, case): [] for trial in range(trials) for case in CASES}
    observed = []
    samples = report.get("samples")
    for sample in samples if isinstance(samples, list) else []:
        if not isinstance(sample, dict):
            continue
        key = (sample.get("trial"), sample.get("case"), sample.get("step"))
        if not all(isinstance(value, (int, str)) for value in key) or key not in indexed:
            continue
        trial, case, step = key
        order[trial, step].append(case)
        sequences[trial, case].append(step)
        observed.append(key)
    if strategy == "step_rotation_reverse_odd_trial_v1":
        expected = []
        for trial in range(trials):
            base = CASES if trial % 2 == 0 else tuple(reversed(CASES))
            for step in range(steps):
                offset = (step + trial) % len(CASES)
                expected.extend((trial, case, step) for case in base[offset:] + base[:offset])
        return bool(expected) and observed == expected
    return (
        bool(order)
        and all(cases == list(CASES if trial % 2 == 0 else reversed(CASES)) for (trial, _), cases in order.items())
        and all(sequence == list(range(steps)) for sequence in sequences.values())
    )


def paired_comparison(indexed, left, right, meaning):
    result = stage3.paired_comparison(indexed, left, right, diagnostic=False, meaning=meaning)
    for row in result["pairs"]:
        first, second = indexed[row["trial"], left, row["step"]], indexed[row["trial"], right, row["step"]]
        for metric in EXTRA_DIFFERENCES:
            row[metric.removesuffix("_us") + "_difference_us"] = first[metric] - second[metric]
    groups = [(result, result["pairs"])]
    groups.extend(
        (group, [row for row in result["pairs"] if row["trial"] == group["trial"]]) for group in result["per_trial"]
    )
    groups.extend(
        (group, [row for row in result["pairs"] if row["capacity"] == group["capacity"]])
        for group in result["per_capacity"]
    )
    for group, rows in groups:
        for metric in EXTRA_DIFFERENCES:
            name = metric.removesuffix("_us") + "_difference"
            group[name] = stage3.statistics_us([row[name + "_us"] for row in rows])
    return result


def analyze(report, *, trace_path=None, selection_path=None):
    if not isinstance(report, dict):
        raise ValueError("preparation timing report must be an object")
    expected = stage3.expected_schedule(trace_path, selection_path)
    trials, steps, issues = measurement_contract(report, expected)
    indexed, coverage = collect_samples(report, expected, trials, steps)
    valid = not issues and coverage["complete"]
    qualification = stage3.execution_qualification(report, expected, trials, steps, valid)
    qualification["checks"]["supported_ordering_strategy"] = report.get("ordering_strategy") in ORDERING_STRATEGIES
    qualification["checks"]["alternating_recorded_case_order"] = recorded_case_order(report, indexed, trials, steps)
    qualification["complete"] = all(qualification["checks"].values())
    qualification["unqualified_checks"] = [name for name, passed in qualification["checks"].items() if not passed]
    execution_success = all(
        qualification["checks"][name]
        for name in (
            "completed_execution_state",
            "successful_returncode",
            "npu_execution_reported",
            "performance_measurement_reported",
        )
    )
    full_trace = valid and execution_success and steps == stage3.STEP_COUNT
    summaries = {
        case: {
            metric: stage3.statistics_us([row[metric] for (_, current, _), row in indexed.items() if current == case])
            for metric in METRICS
        }
        for case in CASES
    }
    per_trial = [
        dict(
            trial=trial,
            cases={
                case: {
                    metric: stage3.statistics_us(
                        [
                            row[metric]
                            for (current_trial, current, _), row in indexed.items()
                            if current_trial == trial and current == case
                        ]
                    )
                    for metric in METRICS
                }
                for case in CASES
            },
        )
        for trial in range(trials)
    ]
    comparisons = dict(
        sink_fresh_minus_reuse=paired_comparison(
            indexed,
            "sink_fresh",
            "sink_reuse",
            "Within the same shared-sink attention graph, measured preparation-path reduction. "
            "Pinned-buffer reuse and invariant Q4 handling are combined; their individual effects are not isolated.",
        ),
        legacy_fresh_minus_reuse=paired_comparison(
            indexed,
            "legacy_fresh",
            "legacy_reuse",
            "Within the same legacy graph and true host updates, the equivalent reusable-table preparation control.",
        ),
        reused_legacy_minus_sink=paired_comparison(
            indexed,
            "legacy_reuse",
            "sink_reuse",
            "Comparison after optimizing preparation on both paths. It still mixes host updates, planning, "
            "different attention kernels and length transport; no end-to-end or independent-novelty attribution.",
        ),
    )
    return dict(
        schema_version=1,
        state="complete_trace"
        if full_trace
        else "complete_screen"
        if valid and execution_success
        else "partial_or_invalid",
        analysis_valid=valid,
        issues=issues,
        coverage=coverage,
        qualification=qualification,
        full_128_step_trace_measured=full_trace,
        three_trial_trace_protocol_complete=qualification["complete"],
        screen_only=steps != stage3.STEP_COUNT,
        canonical_digest=stage3.CANONICAL_TRACE_SHA256,
        requested_trials=trials,
        requested_steps=steps,
        cases=list(CASES),
        ordering_strategy=report.get("ordering_strategy"),
        summaries=summaries,
        trial_summaries=per_trial,
        comparisons=comparisons,
        expected_schedule_sources={name: value for name, value in expected.items() if name != "records"},
        runner_scope=report.get("scope"),
        runner_endpoint=report.get("endpoint"),
        timing_boundaries=dict(
            wall_us="Host preparation, required transfers, planning/host updates, replay and final completion.",
            preparation_enqueue_us="Host construction and enqueue interval; not completed H2D copy duration.",
            queue_span_us="External-event queue span includes submission gaps and waits; not pure kernel duration.",
            submit_join_cpu_us="Host submission interval overlaps queue work; do not add it to queue duration.",
            case_order="Declared case ordering is checked against recorded samples. The rotation strategy shifts "
            "odd-trial-reversed case order by (step + trial) modulo four on every step.",
        ),
        assumptions=[
            "Fresh/reuse cases for one backend use the same captured graph and private sixteen-occurrence Q/K/V.",
            "Reuse refreshes current KV lengths and owner-page mappings inside the timed endpoint.",
            "Q4 query endpoints are invariant for each default bucket; no dynamic KV length is frozen.",
            "Pinned-buffer lifetime extends through the final completion fence before host storage is overwritten.",
            "Legacy cases retain all real host-length graph updates and final dependency joins.",
        ],
        full_model_measured=False,
        end_to_end_speedup_proven=False,
        independent_algorithm_speedup_proven=False,
        novelty_claim_established=False,
        limitation="Ordinary preparation reuse and invariant-data handling are engineering controls. "
        "This experiment cannot establish a new scheduling algorithm or a full-model performance gain.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.resolve()
    protected = [
        source,
        Path(args.trace or stage3.RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json").resolve(),
        Path(args.selection or stage3.RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json").resolve(),
    ]
    if output in protected:
        parser.error("--output cannot overwrite timing or canonical schedule evidence")
    raw = stage3.load_json(source)
    result = analyze(raw, trace_path=args.trace, selection_path=args.selection)
    result["input"] = dict(path=str(source), sha256=stage3.sha256(source))
    projection = stage3.object_or_empty(raw.get("evidence_projection"))
    if projection:
        result["input"]["evidence_projection"] = projection
        result["input"]["original_report_sha256"] = projection.get("original_report_sha256")
    result["analyzer_sha256"] = stage3.sha256(__file__)
    result["stage3_helper_sha256"] = stage3.sha256(stage3.__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(dict(output=str(output), state=result["state"], analysis_valid=result["analysis_valid"])))
    return 0 if result["analysis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
