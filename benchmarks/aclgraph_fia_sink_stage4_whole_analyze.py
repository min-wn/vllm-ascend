# SPDX-License-Identifier: Apache-2.0
"""Analyze the bounded whole-plan attention diagnostic without Torch or vLLM.

The analyzer treats the canonical trace, archived default catalog, selected
records, and the four-dimensional sample key as independent contracts. Oracle
controls exclude plan construction and upload; ``whole_cpu`` includes both in
the measured wall interval. Neither comparison proves a full-model or serving
speedup.
"""

import argparse
import bisect
import hashlib
import json
import math
import statistics
from pathlib import Path

CANONICAL_TRACE_SHA256 = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
CASES = ("native_shared", "native_oracle", "whole_cpu", "whole_oracle")
METRICS = ("wall_us", "queue_span_us", "preparation_enqueue_us", "submit_join_cpu_us")
QUERY_LENGTH = 4
STEP_COUNT = 128
OCCURRENCES = 16
FORMAL_CAPACITIES = (256, 400, 496, 512)
RESULTS_DIRECTORY = Path(__file__).parent / "results" / "multi_graph_structural"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def stats(values):
    if not values:
        return dict(count=0, mean_us=None, p50_us=None, p95_us=None, p99_us=None)
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return dict(
        count=len(ordered),
        mean_us=statistics.mean(ordered),
        p50_us=percentile(0.50),
        p95_us=percentile(0.95),
        p99_us=percentile(0.99),
    )


def expected_schedule(trace_path=None, selection_path=None):
    trace_path = Path(trace_path or RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json")
    selection_path = Path(selection_path or RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json")
    trace_artifact, selection_artifact = load_json(trace_path), load_json(selection_path)
    trace = trace_artifact["trace"]
    normalized = dict(trace)
    prefixes = trace["initial_prefix_by_request"]
    if not isinstance(prefixes, dict) or any(not str(key).isdecimal() or str(int(key)) != str(key) for key in prefixes):
        raise ValueError("canonical trace has invalid owner keys")
    normalized["initial_prefix_by_request"] = {int(key): value for key, value in prefixes.items()}
    if (
        digest(normalized) != trace_artifact.get("trace_sha256")
        or trace_artifact.get("trace_sha256") != CANONICAL_TRACE_SHA256
        or trace.get("query") != QUERY_LENGTH
        or len(trace.get("steps", ())) != STEP_COUNT
        or selection_artifact.get("trace_sha256") != CANONICAL_TRACE_SHA256
    ):
        raise ValueError("canonical trace or selection artifact changed")
    selection = selection_artifact["selection"]
    catalog, dispatched = selection.get("capture_sizes"), selection.get("dispatched_tokens")
    if (
        not isinstance(catalog, list)
        or catalog != sorted(set(catalog))
        or not catalog
        or any(type(value) is not int or value < 1 for value in catalog)
        or not isinstance(dispatched, list)
        or len(dispatched) != STEP_COUNT
        or selection.get("query") != QUERY_LENGTH
        or selection.get("config_source") != "installed EngineArgs defaults; no capture sizes supplied"
    ):
        raise ValueError("invalid archived default catalog")
    records = []
    for index, item in enumerate(trace["steps"]):
        batch = item.get("batch")
        offset = bisect.bisect_left(catalog, batch * QUERY_LENGTH)
        if item.get("step") != index or type(batch) is not int or batch < 1:
            raise ValueError("invalid canonical step")
        if offset == len(catalog) or catalog[offset] % QUERY_LENGTH or dispatched[index] != catalog[offset]:
            raise ValueError("archived dispatch is not the smallest legal default bucket")
        records.append(dict(step=index, batch=batch, capacity=catalog[offset]))
    metadata_path = RESULTS_DIRECTORY.parent.parent / "multi_graph_gdn_q4_full_metadata.py"
    resolver_path = RESULTS_DIRECTORY.parent.parent / "multi_graph_gdn_q4_full_loader.py"
    return dict(
        canonical_digest=CANONICAL_TRACE_SHA256,
        records=records,
        default_capture_sizes=catalog,
        dispatched_tokens=dispatched,
        trace_file_sha256=sha256(trace_path),
        selection_sha256=sha256(selection_path),
        metadata_source_sha256=sha256(metadata_path),
        catalog_resolver_source_sha256=sha256(resolver_path),
    )


def selected_contract(report, expected):
    screen = report.get("screen") is True
    wanted_capacities = (400,) if screen else FORMAL_CAPACITIES
    wanted = {
        capacity: [record for record in expected["records"] if record["capacity"] == capacity]
        for capacity in wanted_capacities
    }
    selected = report.get("selected_records")
    issues = []
    if not isinstance(selected, list) or len(selected) != len(wanted_capacities):
        issues.append("selected_records must contain exactly the bounded representative records")
        selected = []
    seen = set()
    for item in selected:
        if not isinstance(item, dict):
            issues.append("selected record is not an object")
            continue
        capacity = item.get("physical_tokens", item.get("capacity"))
        key = (item.get("step"), item.get("batch"), capacity)
        matches = wanted.get(capacity, ())
        if not any(key == (candidate["step"], candidate["batch"], capacity) for candidate in matches):
            issues.append("selected record differs from the canonical trace/default bucket")
        if capacity in seen:
            issues.append("selected records contain a duplicate capacity")
        seen.add(capacity)
    if seen != set(wanted_capacities):
        issues.append("selected records omit a required representative capacity")
    selected_records = []
    for item in selected:
        capacity = item.get("physical_tokens", item.get("capacity"))
        selected_records.append(dict(step=item.get("step"), batch=item.get("batch"), capacity=capacity))
    return selected_records, issues


def schedule_contract(report, expected):
    schedule = object_or_empty(report.get("schedule"))
    issues = []
    if schedule.get("canonical_digest", schedule.get("trace_sha256")) != expected["canonical_digest"]:
        issues.append("schedule canonical digest mismatch")
    if schedule.get("default_capture_sizes") != expected["default_capture_sizes"]:
        issues.append("schedule capture catalog mismatch")
    records = schedule.get("records")
    if not isinstance(records, list) or len(records) != STEP_COUNT:
        issues.append("schedule must retain all 128 canonical records")
    else:
        for index, (actual, wanted) in enumerate(zip(records, expected["records"], strict=True)):
            if not isinstance(actual, dict) or (
                actual.get("step"),
                actual.get("batch"),
                actual.get("physical_tokens", actual.get("capacity")),
            ) != (wanted["step"], wanted["batch"], wanted["capacity"]):
                issues.append(f"schedule record mismatch at index {index}")
                break
    for name in (
        "trace_file_sha256",
        "selection_sha256",
        "metadata_source_sha256",
        "catalog_resolver_source_sha256",
    ):
        if schedule.get(name) != expected[name]:
            issues.append(f"schedule {name} mismatch")
    return issues


def collect_samples(report, expected, selected, trials, repeats):
    samples = report.get("samples")
    if not isinstance(samples, list):
        return {}, dict(
            complete=False,
            expected_sample_count=0,
            raw_sample_count=0,
            distinct_valid_sample_count=0,
            missing=[],
            duplicates=[],
            unexpected=[],
            invalid=["samples must be a list"],
        )
    wanted = {
        (trial, repeat, case, step)
        for trial in range(trials)
        for repeat in range(repeats)
        for case in CASES
        for step in (item["step"] for item in selected)
    }
    indexed, duplicates, unexpected, invalid = {}, [], [], []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            invalid.append(dict(index=index, reason="sample is not an object"))
            continue
        key = tuple(sample.get(name) for name in ("trial", "repeat", "case", "step"))
        if key not in wanted:
            unexpected.append(dict(index=index, key=list(key)))
            continue
        if key in indexed:
            duplicates.append(dict(index=index, key=list(key)))
            continue
        step, capacity = sample.get("step"), sample.get("capacity")
        wanted_record = next((item for item in selected if item["step"] == step), None)
        if (
            not isinstance(sample.get("case"), str)
            or sample["case"] not in CASES
            or wanted_record is None
            or sample.get("batch") != wanted_record["batch"]
            or capacity != wanted_record["capacity"]
            or any(not finite(sample.get(metric)) for metric in METRICS)
            or sample["wall_us"] <= 0
        ):
            invalid.append(dict(index=index, reason="sample identity, bucket or timing field is invalid"))
            continue
        indexed[key] = sample
    missing = [dict(trial=t, repeat=r, case=c, step=s) for t, r, c, s in sorted(wanted - indexed.keys())]
    return indexed, dict(
        complete=bool(wanted) and not (duplicates or unexpected or invalid or missing),
        expected_sample_count=len(wanted),
        raw_sample_count=len(samples),
        distinct_valid_sample_count=len(indexed),
        missing=missing,
        duplicates=duplicates,
        unexpected=unexpected,
        invalid=invalid,
    )


def numeric_qualification(report, selected):
    checks = report.get("checks")
    qualified = True
    issues = []
    if not isinstance(checks, list) or len(checks) != len(selected):
        return False, ["checks must cover every selected representative"], []
    for item in checks:
        if (
            not isinstance(item, dict)
            or not all(
                item.get(name) is True
                for name in (
                    "controls_close",
                    "zero_padding_exact",
                    "resident_queries_unchanged",
                    "exact_descriptors_unchanged",
                )
            )
            or item.get("private_occurrences") != OCCURRENCES
        ):
            qualified = False
            issues.append("numeric control qualification failed")
            continue
        rows = item.get("fp32_sampled_rows")
        if (
            not isinstance(rows, list)
            or {row.get("case") for row in rows} != set(CASES)
            or any(not finite(row.get("max_abs_error")) for row in rows)
        ):
            qualified = False
            issues.append("FP32 sampled rows do not cover all controls")
    post = report.get("post_timing_checks")
    if isinstance(post, list) and len(post) == 1:
        post = post[0]
    if not isinstance(post, dict) or not all(
        post.get(name) is True
        for name in (
            "controls_close",
            "zero_padding_exact",
            "resident_queries_unchanged",
            "exact_descriptors_unchanged",
        )
    ):
        qualified = False
        issues.append("post-timing qualification failed")
    return qualified, issues, checks


def summarize(indexed, selected):
    by_case = {
        case: {metric: stats([row[metric] for key, row in indexed.items() if key[2] == case]) for metric in METRICS}
        for case in CASES
    }
    by_capacity = {
        str(item["capacity"]): {
            case: {
                metric: stats(
                    [row[metric] for key, row in indexed.items() if key[2] == case and key[3] == item["step"]]
                )
                for metric in METRICS
            }
            for case in CASES
        }
        for item in selected
    }
    return dict(by_case=by_case, by_capacity=by_capacity)


def comparison(indexed, selected, baseline, candidate, meaning):
    rows = []
    for key, left in sorted(indexed.items()):
        trial, repeat, case, step = key
        if case != baseline or (trial, repeat, candidate, step) not in indexed:
            continue
        right = indexed[trial, repeat, candidate, step]
        rows.append(
            dict(
                trial=trial,
                repeat=repeat,
                step=step,
                capacity=left["capacity"],
                values={metric: right[metric] - left[metric] for metric in METRICS},
            )
        )
    per_capacity = []
    for item in selected:
        subset = [row for row in rows if row["capacity"] == item["capacity"]]
        per_capacity.append(
            dict(
                capacity=item["capacity"],
                paired_count=len(subset),
                metrics={metric: stats([row["values"][metric] for row in subset]) for metric in METRICS},
            )
        )
    return dict(
        baseline=baseline,
        candidate=candidate,
        interpretation=meaning,
        paired_count=len(rows),
        metrics={metric: stats([row["values"][metric] for row in rows]) for metric in METRICS},
        per_capacity=per_capacity,
        candidate_faster_count=sum(row["values"]["wall_us"] < 0 for row in rows),
        candidate_slower_count=sum(row["values"]["wall_us"] > 0 for row in rows),
        equal_count=sum(row["values"]["wall_us"] == 0 for row in rows),
        pairs=rows,
    )


def analyze(report, *, trace_path=None, selection_path=None):
    if not isinstance(report, dict):
        raise ValueError("whole-plan report must be a JSON object")
    expected = expected_schedule(trace_path, selection_path)
    selected, selected_issues = selected_contract(report, expected)
    issues = schedule_contract(report, expected) + selected_issues
    trials, repeats = report.get("trials"), report.get("repeats")
    if type(trials) is not int or trials < 1 or type(repeats) is not int or repeats < 1:
        issues.append("trials and repeats must be positive integers")
        trials = repeats = 0
    indexed, coverage = collect_samples(report, expected, selected, trials, repeats)
    coverage_valid = not issues and coverage["complete"]
    numeric_ok, numeric_issues, checks = numeric_qualification(report, selected)
    if not numeric_ok:
        issues.extend(numeric_issues)
    installed = object_or_empty(report.get("installed_catalog"))
    selection = object_or_empty(installed.get("selection"))
    schedule = object_or_empty(report.get("schedule"))
    protocol_checks = dict(
        coverage_complete=coverage["complete"] and not schedule_contract(report, expected) and not selected_issues,
        completed_state=report.get("state") == "completed",
        successful_returncode=report.get("returncode") == 0,
        npu_execution=report.get("npu_work_executed") is True,
        performance_measured=report.get("performance_measured") is True,
        expected_case_set=tuple(report.get("cases", ())) == CASES,
        occurrences=report.get("occurrences") == OCCURRENCES,
        numeric_controls=numeric_ok,
        installed_catalog_revalidated=schedule.get("installed_catalog_revalidated") is True
        and installed.get("installed_catalog_revalidated") is True,
        installed_catalog_matches=selection.get("capture_sizes") == expected["default_capture_sizes"]
        and selection.get("dispatched_tokens") == expected["dispatched_tokens"],
        expected_sample_count=coverage["expected_sample_count"] == len(selected) * trials * repeats * len(CASES),
        formal_representatives=report.get("screen") is not True
        and {item["capacity"] for item in selected} == set(FORMAL_CAPACITIES),
    )
    protocol_qualified = all(protocol_checks.values())
    measured = (
        protocol_checks["completed_state"]
        and protocol_checks["successful_returncode"]
        and protocol_checks["npu_execution"]
        and protocol_checks["performance_measured"]
    )
    screen = report.get("screen") is True
    return dict(
        schema_version=1,
        state="complete_trace"
        if coverage_valid and measured and not screen
        else "complete_screen"
        if coverage_valid
        else "partial_or_invalid",
        analysis_valid=coverage_valid,
        protocol_qualified=protocol_qualified,
        issues=issues,
        qualification=dict(
            complete=protocol_qualified,
            checks=protocol_checks,
            unqualified_checks=[key for key, value in protocol_checks.items() if not value],
        ),
        coverage=coverage,
        canonical_digest=expected["canonical_digest"],
        selected_records=selected,
        selected_capacities=[item["capacity"] for item in selected],
        requested_trials=trials,
        requested_repeats=repeats,
        expected_sample_count=coverage["expected_sample_count"],
        full_trace_measured=coverage_valid and measured and not screen,
        screen_only=screen,
        summaries=summarize(indexed, selected),
        comparisons={
            "whole_oracle_minus_native_oracle": comparison(
                indexed,
                selected,
                "native_oracle",
                "whole_oracle",
                "Oracle comparison excludes whole-plan build and upload from both controls; "
                "it is a kernel-path diagnostic.",
            ),
            "whole_cpu_minus_native_shared": comparison(
                indexed,
                selected,
                "native_shared",
                "whole_cpu",
                "whole_cpu includes exact whole-plan build, pinned host fill, metadata copy, replay and final join.",
            ),
        },
        interpretation=dict(
            whole_cpu="Measured dynamic whole-plan endpoint including plan construction and copy.",
            whole_oracle="Plan construction and copy excluded; diagnostic only.",
            limitation="Isolated attention chain only; no full-model, serving or end-to-end speedup claim.",
        ),
        full_model_measured=False,
        end_to_end_speedup_proven=False,
        source_schedule={key: expected[key] for key in expected if key != "records"},
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
    report = analyze(load_json(source), trace_path=args.trace, selection_path=args.selection)
    report["input"] = dict(path=str(source), sha256=sha256(source))
    report["analyzer_sha256"] = sha256(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(dict(output=str(output), state=report["state"], protocol_qualified=report["protocol_qualified"])))
    return 0 if report["analysis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
