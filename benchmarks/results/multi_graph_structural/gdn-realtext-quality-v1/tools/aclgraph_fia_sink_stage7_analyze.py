# SPDX-License-Identifier: Apache-2.0
"""Audit complete dynamic-trace native attention-chain measurements on CPU.

Fresh-process blocks compare current descriptor preparation, native planning
and sixteen sequential attention consumers. Correctness passes are separate
from timing, and neither endpoint is full-model or split-versus-padding E2E.
"""

import argparse
import contextlib
import json
import math
from collections import Counter
from pathlib import Path

import aclgraph_fia_sink_stage6_analyze as controls
from aclgraph_fia_sink_stage3_trace import load_trace_schedule

POLICIES = controls.POLICIES
METRICS = controls.METRICS
ROUNDS = 4
REPEATS = 3
STEP_COUNT = 128
OCCURRENCES = 16
evidence = controls.evidence


def expected_contract(trace_path=None, selection_path=None):
    """Reconstruct every temporal request/page/length field independently."""
    independent = evidence.expected_schedule(trace_path, selection_path)
    schedule = controls.normalized(load_trace_schedule(trace_path, selection_path))
    records = [dict(record, page_count=schedule["owner_pages"]["page_count"]) for record in schedule["records"]]
    capacities = schedule["required_capture_sizes"]
    first = {}
    for record in records:
        first.setdefault(record["physical_tokens"], record)
    captures = list(first.values())
    contract = {key: value for key, value in schedule.items() if key != "records"}
    contract["installed_catalog_revalidated"] = True
    scripts = set(controls.RUNTIME_SCRIPTS) | {
        "aclgraph_fia_sink_stage7_probe.py",
        "aclgraph_fia_sink_stage7_suite.py",
        "aclgraph_fia_sink_stage6_suite.py",
    }
    return dict(
        schedule=schedule,
        records=records,
        captures=captures,
        capacities=capacities,
        trace_contract=contract,
        independent=independent,
        scripts={name: evidence.sha256(Path(__file__).with_name(name)) for name in sorted(scripts)},
    )


def transition(record, previous):
    changed = record["physical_tokens"] != previous["physical_tokens"]
    return dict(
        previous_capacity=previous["physical_tokens"],
        capacity_transition=changed,
        transition_class="pass_start" if record["step"] == 0 else "capacity_transition" if changed else "same_capacity",
        membership_changed=record["request_ids"] != previous["request_ids"],
        kv_lengths_changed=record["kv_lengths"] != previous["kv_lengths"],
        page_routes_changed=record["block_table"] != previous["block_table"],
    )


def design_issues(bundle, expected):
    wanted = dict(
        policies=list(POLICIES),
        capacities=expected["capacities"],
        occurrences=OCCURRENCES,
        rounds=ROUNDS,
        repeats=REPEATS,
        step_count=STEP_COUNT,
        process_orders=[list(POLICIES if trial % 2 == 0 else reversed(POLICIES)) for trial in range(ROUNDS)],
        screen=False,
        seed=1703,
        step_order="unchanged chronological canonical trace",
        qualification_passes=2,
        reference_rows_per_step=3,
        reference_occurrence="step % occurrences",
        warmup_passes_per_repeat=1,
    )
    design = bundle.get("design")
    issues = []
    if not isinstance(design, dict) or any(not controls.exact(design.get(key), value) for key, value in wanted.items()):
        issues.append("design differs from the complete four-round/three-pass chronological protocol")
    if bundle.get("state") != "completed" or type(bundle.get("returncode")) is not int or bundle["returncode"] != 0:
        issues.append("suite did not complete successfully")
    if bundle.get("full_trace_measured") is not True or any(
        bundle.get(key) is not False for key in ("full_model_qualified", "end_to_end_gain_established")
    ):
        issues.append("suite must report full attention trace and exclude model/E2E qualification")
    return issues


def tensor_identities(value, count, shape, dtype):
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, dict)
            and controls.exact(item.get("shape"), shape)
            and item.get("dtype") == dtype
            and controls.sha(item.get("sha256"))
            for item in value
        )
    )


def frame_issues(report, expected):
    before, after = report.get("frame_query_identity"), report.get("post_frame_query_identity")
    capacities = expected["capacities"]
    if (
        not isinstance(before, dict)
        or set(before) != {str(capacity) for capacity in capacities}
        or not controls.exact(before, after)
    ):
        return ["all captured query-frame byte identities must be retained unchanged"]
    issues = []
    for capacity in capacities:
        if not tensor_identities(before[str(capacity)], OCCURRENCES, [capacity, 12, 256], "torch.bfloat16"):
            issues.append(f"capacity {capacity}: all private frame query identities required")
    addresses = report.get("captured_addresses")
    if not isinstance(addresses, dict) or set(addresses) != set(before):
        issues.append("capture-origin descriptors must cover every required default bucket")
        return issues
    for capacity in capacities:
        shapes = dict(
            q_lengths=[capacity // 4],
            kv_lengths=[capacity // 4],
            block_table=[capacity // 4, expected["schedule"]["owner_pages"]["width"]],
            atten_mask=[2048, 2048],
            metadata=[1024],
        )
        shapes.update(
            {f"{name}{index}": [capacity, 12, 256] for name in ("query", "output") for index in range(OCCURRENCES)}
        )
        captured = addresses[str(capacity)]
        if not isinstance(captured, dict) or set(captured) != set(shapes):
            issues.append(f"capacity {capacity}: incomplete exact descriptor coverage")
            continue
        pointers = []
        for name, shape in shapes.items():
            item = captured[name]
            strides = [math.prod(shape[index + 1 :]) for index in range(len(shape))]
            if (
                not isinstance(item, dict)
                or not controls.integer(item.get("pointer"), 1)
                or not controls.exact(item.get("shape"), shape)
                or not controls.exact(item.get("stride"), strides)
            ):
                issues.append(f"capacity {capacity}: invalid captured {name} descriptor")
            else:
                pointers.append(item["pointer"])
        if len(set(pointers)) != len(pointers):
            issues.append(f"capacity {capacity}: captured tensors alias")
    return issues


def preparation_issues(report, expected):
    summaries = report.get("prepared_descriptors")
    if not isinstance(summaries, dict) or set(summaries) != {str(c) for c in expected["capacities"]}:
        return ["prepared sources must cover all seventeen default buckets"]
    frequencies = Counter(record["physical_tokens"] for record in expected["records"])
    issues = []
    for capacity, frequency in frequencies.items():
        count = (2 + 2 * REPEATS) * frequency
        wanted = dict(
            mode="sink",
            physical_tokens=capacity,
            physical_requests=capacity // 4,
            host_pinned=dict.fromkeys(("q_lengths", "kv_lengths", "block_table"), True),
            stable_host_allocations=True,
            stable_device_allocations=True,
            host_allocation_count=3,
            device_allocation_count=0,
            setup_copies=dict(q_lengths=1, kv_lengths=0, block_table=0),
            step_copies=dict(q_lengths=0, kv_lengths=count, block_table=count),
            prepared_steps=count,
            completed_steps=count,
            inflight=False,
            q4_invariant=dict(query_length=4, cumulative_q=list(range(4, capacity + 1, 4))),
        )
        item = summaries[str(capacity)]
        if not isinstance(item, dict) or any(not controls.exact(item.get(key), value) for key, value in wanted.items()):
            issues.append(f"capacity {capacity}: dynamic preparation/copy/lifetime contract failed")
    return issues


def oracle_issues(check, record):
    rows = check.get("fp32_sampled_rows")
    occurrence = record["step"] % OCCURRENCES
    wanted = {0, record["batch"] // 2, record["batch"] - 1}
    if (
        type(check.get("oracle_occurrence")) is not int
        or check["oracle_occurrence"] != occurrence
        or not isinstance(rows, list)
        or len(rows) != len(wanted)
        or any(not isinstance(item, dict) or type(item.get("row")) is not int for item in rows)
        or {item["row"] for item in rows} != wanted
    ):
        return ["each step requires the rotating occurrence and three exact FP32 row identities"]
    issues = []
    for item in rows:
        row, kv = item["row"], record["kv_lengths"][item["row"]]
        pages = record["block_table"][row][: (kv + 127) // 128]
        if (
            type(item.get("occurrence")) is not int
            or item["occurrence"] != occurrence
            or type(item.get("kv_length")) is not int
            or item["kv_length"] != kv
            or not controls.exact(item.get("original_page_ids"), pages)
            or not controls.exact(item.get("local_page_ids"), list(range(len(pages))))
            or not evidence.finite(item.get("max_abs_error"))
        ):
            issues.append("FP32 oracle occurrence/current KV/ordered page remapping evidence is invalid")
    return issues


def qualification_issues(report, expected, policy):
    """Keep chronological correctness passes separate from timed observations."""
    issues = []
    preflight = report.get("metadata_preflight")
    if not isinstance(preflight, list) or len(preflight) != len(expected["captures"]):
        issues.append("metadata preflight must cover each initial bucket frame")
    else:
        for item, record in zip(preflight, expected["captures"], strict=True):
            if (
                not isinstance(item, dict)
                or not controls.exact(item.get("record"), record)
                or type(item.get("capacity")) is not int
                or item["capacity"] != record["physical_tokens"]
                or item.get("verified_before_attention_capture") is not True
            ):
                issues.append("metadata preflight differs from the first chronological frame for its bucket")
                continue
            issues.extend(f"preflight step{record['step']}: {x}" for x in controls.plan_issues(item, record, policy))
    for label in ("pre_checks", "post_checks"):
        checks = report.get(label)
        if not isinstance(checks, list) or len(checks) != STEP_COUNT:
            issues.append(f"{label} must contain a separate complete chronological qualification pass")
            continue
        for item, record in zip(checks, expected["records"], strict=True):
            if not isinstance(item, dict):
                issues.append(f"{label}: qualification record must be an object")
                continue
            wanted = dict(
                step=record["step"],
                batch=record["batch"],
                capacity=record["physical_tokens"],
                private_occurrences=OCCURRENCES,
            )
            flags = (
                "finite",
                "all_outputs_finite",
                "exact_zero_padding",
                "stable_addresses",
                "exact_descriptors_read_only",
            )
            if any(not controls.exact(item.get(key), value) for key, value in wanted.items()) or any(
                item.get(key) is not True for key in flags
            ):
                issues.append(f"{label}: step/occurrence/finite/read-only/zero-padding qualification failed")
            issues.extend(f"{label} step{record['step']}: {x}" for x in oracle_issues(item, record))
            issues.extend(
                f"{label} step{record['step']}: {x}" for x in controls.plan_issues(item.get("plan"), record, policy)
            )
            if not isinstance(item.get("plan"), dict) or item["plan"].get("current_exact_base_verified") is not True:
                issues.append(f"{label}: exact current native base verification is required")
            if not tensor_identities(
                item.get("output_hashes"), OCCURRENCES, [record["physical_tokens"], 12, 256], "torch.bfloat16"
            ):
                issues.append(f"{label}: retain all private output hashes as repeatability diagnostics")
    return issues


def report_issues(report, entry, expected):
    policy, trial = entry.get("policy"), entry.get("trial")
    if policy not in POLICIES or not controls.integer(trial) or trial >= ROUNDS:
        return ["invalid native policy or round identity"]
    issues = []
    wanted = dict(
        state="completed",
        stage="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        full_trace_measured=True,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        policy=policy,
        trial=trial,
        repeats=REPEATS,
        occurrences=OCCURRENCES,
        canonical_digest=evidence.CANONICAL_TRACE_SHA256,
        selected_records=expected["records"],
        capture_records=expected["captures"],
        trace_contract=expected["trace_contract"],
        core_limit=dict(cube_core_num=20, vector_core_num=40),
        independent_graph_pool_per_bucket=True,
        frame_queries_match_base_prefixes=True,
        separate_qualification=True,
        timed_samples_individually_qualified=False,
    )
    for key, value in wanted.items():
        if not controls.exact(report.get(key), value):
            issues.append(f"report {key} differs from the full dynamic attention-trace contract")
    installed = report.get("installed_catalog")
    selection = installed.get("selection") if isinstance(installed, dict) else None
    if (
        not isinstance(installed, dict)
        or installed.get("installed_catalog_revalidated") is not True
        or not isinstance(selection, dict)
        or not controls.exact(selection.get("capture_sizes"), expected["schedule"]["default_capture_sizes"])
        or not controls.exact(selection.get("dispatched_tokens"), expected["independent"]["dispatched_tokens"])
        or selection.get("config_source") != "installed EngineArgs defaults; no capture sizes supplied"
    ):
        issues.append("installed default capture catalog and chronological dispatch are not qualified")
    warmup = report.get("warmup_passes")
    wanted_warmup = [dict(repeat=repeat, steps=list(range(STEP_COUNT)), completed=True) for repeat in range(REPEATS)]
    if not controls.exact(warmup, wanted_warmup):
        issues.append("each timed pass requires its separate complete chronological warmup pass")
    issues.extend(controls.input_issues(report, expected))
    issues.extend(controls.provenance_issues(report, policy, expected))
    issues.extend(frame_issues(report, expected))
    issues.extend(preparation_issues(report, expected))
    issues.extend(qualification_issues(report, expected, policy))
    return issues


def collect_samples(runs, expected):
    """Require every round/policy/pass/step exactly once and in trace order."""
    indexed, issues, raw_count = {}, [], 0
    records = expected["records"]
    for entry, report in runs:
        samples = report.get("samples")
        if not isinstance(samples, list):
            issues.append("samples must be a list")
            continue
        raw_count += len(samples)
        sequence = []
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or any(
                    type(sample.get(key)) is not int
                    for key in ("trial", "repeat", "step", "batch", "capacity", "previous_capacity")
                )
                or not isinstance(sample.get("policy"), str)
            ):
                issues.append("sample identity fields require exact integers and a policy")
                continue
            trial, repeat, step, policy = (sample[key] for key in ("trial", "repeat", "step", "policy"))
            key = trial, policy, repeat, step
            if (
                trial != entry.get("trial")
                or policy != entry.get("policy")
                or not 0 <= trial < ROUNDS
                or not 0 <= repeat < REPEATS
                or not 0 <= step < STEP_COUNT
            ):
                issues.append("sample round/policy/pass/step differs from its process")
                continue
            record, previous = records[step], records[step - 1]
            timing = all(evidence.finite(sample.get(metric)) for metric in METRICS) and sample.get("wall_us", 0) > 0
            temporal = transition(record, previous)
            if (
                not timing
                or sample["batch"] != record["batch"]
                or sample["capacity"] != record["physical_tokens"]
                or any(
                    not controls.exact(sample.get(name), temporal[name])
                    for name in ("previous_capacity", "capacity_transition", "transition_class")
                )
            ):
                issues.append("sample exact temporal shape/transition or finite timing contract failed")
                continue
            if sample["preparation_enqueue_us"] + sample["submit_join_cpu_us"] > sample["wall_us"] + 1e-6:
                issues.append("same-clock preparation and submission exceed the measured wall endpoint")
                continue
            if key in indexed:
                issues.append("duplicate round/policy/pass/step sample")
                continue
            indexed[key] = sample
            sequence.append((repeat, step))
        if sequence != [(repeat, step) for repeat in range(REPEATS) for step in range(STEP_COUNT)]:
            issues.append("timed samples do not follow each complete unchanged chronological pass")
    wanted = {
        (trial, policy, repeat, step)
        for trial in range(ROUNDS)
        for policy in POLICIES
        for repeat in range(REPEATS)
        for step in range(STEP_COUNT)
    }
    missing = sorted(wanted - indexed.keys())
    if missing:
        issues.append("full dynamic timing matrix is incomplete")
    return indexed, dict(
        complete=not issues,
        expected_sample_count=len(wanted),
        raw_sample_count=raw_count,
        valid_sample_count=len(indexed),
        missing=[list(key) for key in missing],
        issues=issues,
    )


def group_statistics(indexed, pairs):
    result = dict(paired_count=len(pairs), metrics={})
    for metric in METRICS:
        left = [indexed[row["trial"], "upstream", row["repeat"], row["step"]][metric] for row in pairs]
        right = [indexed[row["trial"], "whole", row["repeat"], row["step"]][metric] for row in pairs]
        deltas = [a - b for a, b in zip(left, right, strict=True)]
        result["metrics"][metric] = dict(
            upstream=evidence.stats(left),
            whole=evidence.stats(right),
            upstream_minus_whole=evidence.stats(deltas),
            upstream_reduction_percent=100 * sum(deltas) / sum(left) if sum(left) else None,
            whole_faster_count=sum(value > 0 for value in deltas),
            whole_slower_count=sum(value < 0 for value in deltas),
            equal_count=sum(value == 0 for value in deltas),
        )
    return result


def comparison(indexed, expected):
    pairs = []
    for trial in range(ROUNDS):
        for repeat in range(REPEATS):
            for step, record in enumerate(expected["records"]):
                left, right = (
                    indexed.get((trial, "upstream", repeat, step)),
                    indexed.get((trial, "whole", repeat, step)),
                )
                if left is None or right is None:
                    continue
                pairs.append(
                    dict(
                        trial=trial,
                        repeat=repeat,
                        step=step,
                        batch=record["batch"],
                        capacity=record["physical_tokens"],
                        **transition(record, expected["records"][step - 1]),
                        upstream_minus_whole_us={metric: left[metric] - right[metric] for metric in METRICS},
                        upstream_reduction_percent={
                            metric: 100 * (left[metric] - right[metric]) / left[metric] if left[metric] else None
                            for metric in METRICS
                        },
                    )
                )
    groups = {}
    for name, field, values in (
        ("per_capacity", "capacity", expected["capacities"]),
        ("per_round", "trial", range(ROUNDS)),
        ("per_pass", "repeat", range(REPEATS)),
        ("per_step", "step", range(STEP_COUNT)),
        ("per_transition", "transition_class", ("pass_start", "capacity_transition", "same_capacity")),
    ):
        groups[name] = [
            dict(**{field: value}, **group_statistics(indexed, [row for row in pairs if row[field] == value]))
            for value in values
        ]
    return dict(
        overall=group_statistics(indexed, pairs),
        **groups,
        pairs=pairs,
        slower_counterexamples=sorted(
            [row for row in pairs if row["upstream_minus_whole_us"]["wall_us"] < 0],
            key=lambda row: row["upstream_minus_whole_us"]["wall_us"],
        ),
    )


def numerical_summary(runs):
    errors, plans, repeatability, metadata_repeatability, preflight_count = [], 0, [], [], 0
    for entry, report in runs:
        preflight_count += len(controls.array(report.get("metadata_preflight")))
        for label in ("pre_checks", "post_checks"):
            for check in controls.array(report.get(label)):
                if not isinstance(check, dict):
                    continue
                plans += 1
                errors.extend(
                    row["max_abs_error"]
                    for row in controls.array(check.get("fp32_sampled_rows"))
                    if isinstance(row, dict) and evidence.finite(row.get("max_abs_error"))
                )
        for left, right in zip(controls.array(report.get("pre_checks")), controls.array(report.get("post_checks"))):
            if isinstance(left, dict) and isinstance(right, dict):
                repeatability.append(
                    dict(
                        trial=entry.get("trial"),
                        policy=entry.get("policy"),
                        step=left.get("step"),
                        identical=controls.exact(left.get("output_hashes"), right.get("output_hashes")),
                    )
                )
                first, second = left.get("plan"), right.get("plan")
                metadata_repeatability.append(
                    dict(
                        trial=entry.get("trial"),
                        policy=entry.get("policy"),
                        step=left.get("step"),
                        identical=isinstance(first, dict)
                        and isinstance(second, dict)
                        and first.get("sha256") == second.get("sha256"),
                    )
                )
    return dict(
        separate_qualification=True,
        timed_samples_individually_qualified=False,
        expected_step_checks=ROUNDS * len(POLICIES) * 2 * STEP_COUNT,
        observed_step_checks=plans,
        expected_fp32_rows=ROUNDS * len(POLICIES) * 2 * STEP_COUNT * 3,
        observed_fp32_rows=len(errors),
        maximum_sampled_fp32_abs_error=max(errors) if errors else None,
        oracle_rotation="occurrence step % 16; three rows per step; each occurrence covers eight steps per pass",
        output_bitwise_repeatability=repeatability,
        metadata_same_step_repeatability=metadata_repeatability,
        observed_pre_capture_payloads=preflight_count,
        observed_total_payloads=plans + preflight_count,
        metadata_repeatability_semantics="same chronological step before/after timing; diagnostic only; "
        "different steps are not required to produce different payload hashes",
    )


def analyze(bundle, *, base_dir, trace_path=None, selection_path=None):
    """Qualify temporal controls before interpreting unfiltered block statistics."""
    if not isinstance(bundle, dict):
        raise ValueError("stage-seven suite bundle must be an object")
    expected = expected_contract(trace_path, selection_path)
    issues = design_issues(bundle, expected)
    runs, paths = controls.load_runs(bundle, Path(base_dir).resolve(strict=True), issues)
    issues.extend(controls.process_issues(bundle, runs))
    issues.extend(controls.common_control_issues(runs))
    first = runs[0][1] if runs else {}
    for index, (entry, report) in enumerate(runs):
        issues.extend(f"run {index}: {issue}" for issue in report_issues(report, entry, expected))
        if not controls.exact(report.get("frame_query_identity"), first.get("frame_query_identity")):
            issues.append(f"run {index}: captured query-frame bytes differ between processes")
    indexed, coverage = collect_samples(runs, expected)
    issues.extend(coverage["issues"])
    qualified = not issues
    frequencies = Counter(record["physical_tokens"] for record in expected["records"])
    return dict(
        schema_version=1,
        state="complete_dynamic_attention_trace" if qualified else "partial_or_invalid",
        analysis_valid=qualified,
        protocol_qualified=qualified,
        issues=issues,
        coverage=coverage,
        full_trace_measured=qualified,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        split_parallel_vs_padding_measured=False,
        absolute_causal_attribution_established=False,
        design=bundle.get("design"),
        canonical_digest=evidence.CANONICAL_TRACE_SHA256,
        capacity_frequency=[
            dict(
                capacity=capacity, steps=frequencies[capacity], expected_pairs=frequencies[capacity] * ROUNDS * REPEATS
            )
            for capacity in expected["capacities"]
        ],
        raw_reports=[dict(path=str(path), sha256=evidence.sha256(path)) for path in paths],
        runtime_script_hashes=expected["scripts"],
        comparison=comparison(indexed, expected),
        numerical_qualification=numerical_summary(runs),
        samples=[
            sample for _, report in runs for sample in controls.array(report.get("samples")) if isinstance(sample, dict)
        ],
        queue_clock_diagnostics=[
            dict(
                trial=key[0],
                policy=key[1],
                repeat=key[2],
                step=key[3],
                queue_span_us=sample["queue_span_us"],
                wall_us=sample["wall_us"],
            )
            for key, sample in sorted(indexed.items())
            if sample["queue_span_us"] > sample["wall_us"]
        ],
        statistics_contract=dict(
            filtering="none; retain every timing observation and every slower counterexample",
            delta="upstream minus whole; positive means the whole-row endpoint is lower",
            reduction="100 * sum(upstream-whole) / sum(upstream), weighted by actual trace frequency",
            percentile="linear interpolation at (n-1)*p; empirical P95/P99 are not tail-stability proofs",
            pairing="same round, pass and chronological trace step, in different fresh processes",
            transition="step0 is pass_start; capacity_transition compares the preceding trace step, including 127 to0",
            dependence="passes and trace steps are nested within four process blocks, not independent process trials",
        ),
        endpoint="Pinned current-KV/page preparation, required H2D, one native shared producer, "
        "sixteen sequential attention consumers and final completion.",
        correctness_scope="Two separate chronological qualification passes per process; every step validates all "
        "sixteen outputs structurally and checks three FP32 rows for one rotating occurrence. "
        "Timed samples are not individually FP32-qualified.",
        limitation="Fresh-process AB/BA blocks retain possible process/time drift. The combined row assignment "
        "and FD-removal comparison does not isolate FD merge cost, and excludes dense layers, model scheduling, "
        "sampler and serving. No full-model, split-parallel-versus-padding or end-to-end gain is established.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.resolve()
    bundle = evidence.load_json(source)
    protected = {
        source,
        Path(args.trace or evidence.RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json").resolve(),
        Path(args.selection or evidence.RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json").resolve(),
    }
    if isinstance(bundle, dict):
        for entry in controls.array(bundle.get("runs")):
            if isinstance(entry, dict):
                with contextlib.suppress(ValueError, OSError):
                    protected.add(controls.resolve_report(entry, source.parent))
    if output in protected:
        parser.error("--output cannot overwrite suite, raw process reports or canonical evidence")
    result = analyze(bundle, base_dir=source.parent, trace_path=args.trace, selection_path=args.selection)
    result["input"] = dict(path=str(source), sha256=evidence.sha256(source))
    result["analyzer_sha256"] = evidence.sha256(__file__)
    result["stage6_control_helper_sha256"] = evidence.sha256(controls.__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(dict(output=str(output), state=result["state"], protocol_qualified=result["protocol_qualified"])))
    return 0 if result["protocol_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
