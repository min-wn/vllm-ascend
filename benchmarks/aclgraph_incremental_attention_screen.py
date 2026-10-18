"""CPU-only structural screening; no device latency or speedup is predicted."""

import argparse
import bisect
import dataclasses
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from aclgraph_incremental_attention_plan import (
    Geometry,
    IncrementalAttentionPlanner,
    PlanPolicy,
    RequestKey,
    RequestRow,
    full_rebuild_plan,
)
from multi_graph_dynamic_trace import trace_digest

ORIGINAL_TRACE_SHA256 = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def assert_semantic_domain(plan):
    """Check each exact KV interval independently of planner decomposition."""
    intervals = {}
    for tile in plan.tiles:
        source = plan.rows[tile.row]
        if source.key is None or tile.key != source.key:
            raise AssertionError("work tile uses a stale request identity")
        if not 0 <= tile.kv_head < plan.geometry.kv_heads:
            raise AssertionError("invalid KV head")
        if not 0 <= tile.query_start < tile.query_end <= source.query_length:
            raise AssertionError("invalid query interval")
        if not 0 <= tile.kv_start < tile.kv_end <= source.kv_length:
            raise AssertionError("invalid KV interval")
        for query in range(tile.query_start, tile.query_end):
            end = source.kv_length
            if plan.geometry.causal:
                end = max(0, end - source.query_length + query + 1)
            if tile.kv_start < min(tile.kv_end, end):
                intervals.setdefault((tile.row, tile.kv_head, query), []).append((tile.kv_start, min(tile.kv_end, end)))
    for index, source in enumerate(plan.rows):
        if source.key is None:
            continue
        for head in range(plan.geometry.kv_heads):
            for query in range(source.query_length):
                expected = source.kv_length
                if plan.geometry.causal:
                    expected = max(0, expected - source.query_length + query + 1)
                cursor = 0
                for begin, end in sorted(intervals.pop((index, head, query), ())):
                    if begin != cursor:
                        raise AssertionError("KV gap or duplicate coverage")
                    cursor = end
                if cursor != expected:
                    raise AssertionError("incomplete exact KV coverage")
    if intervals:
        raise AssertionError("unexpected attention work")
    assignment = Counter(index for part in plan.partitions for index in range(part.start, part.end))
    if assignment != Counter({index: 1 for index in range(len(plan.tiles))}):
        raise AssertionError("work tile assigned zero or multiple times")


def pad_rows(rows, capture_sizes, query=4):
    rows = tuple(rows)
    tokens = len(rows) * query
    offset = bisect.bisect_left(capture_sizes, max(query, tokens))
    if offset == len(capture_sizes) or capture_sizes[offset] % query:
        raise ValueError("workload does not fit the immutable default graph catalog")
    capacity = capture_sizes[offset]
    padding = (RequestRow(None, query, 0),) * (capacity // query - len(rows))
    return rows + padding, capacity


def original_workload(trace, capture_sizes, dispatched):
    result = []
    for index, step in enumerate(trace["steps"]):
        rows = [
            RequestRow(RequestKey(request, 0), trace["query"], length)
            for request, length in zip(step["request_ids"], step["seq_lengths"], strict=True)
        ]
        rows, capacity = pad_rows(rows, capture_sizes, trace["query"])
        if capacity != dispatched[index]:
            raise AssertionError("default graph selection changed from captured evidence")
        result.append(dict(step=index, rows=rows, retired=(), default_capacity=capacity))
    return result


def adversarial_workloads(capture_sizes):
    workloads = {}
    lengths = (511, 512, 513, 1023, 1024, 1025, 1026, 1027)
    boundary = []
    churn = []
    lifecycle = []
    for step, length in enumerate(lengths):
        order = range(24) if step % 2 == 0 else reversed(range(24))
        rows, capacity = pad_rows(
            [RequestRow(RequestKey(request, 0), 4, length + request % 3) for request in order], capture_sizes
        )
        boundary.append(dict(step=step, rows=rows, retired=(), default_capacity=capacity))
        requests = range((step % 2) * 24, (step % 2 + 1) * 24)
        rows, capacity = pad_rows(
            [RequestRow(RequestKey(request, 0), 4, 64 if request % 7 else 8192) for request in requests], capture_sizes
        )
        churn.append(dict(step=step, rows=rows, retired=(), default_capacity=capacity))
    lifetime_rows = (
        ((0, 0, 15, 4), (1, 0, 7, 4), (2, 0, 0, 0)),
        ((2, 0, 0, 0), (0, 0, 16, 4)),
        (),
        ((1, 0, 8, 4), (0, 0, 17, 4)),
        ((0, 1, 4, 4), (1, 0, 9, 4)),
        ((0, 1, 5, 4), (2, 0, 3, 0)),
    )
    for step, definitions in enumerate(lifetime_rows):
        rows = [
            RequestRow(RequestKey(request, generation), query, length)
            for request, generation, length, query in definitions
        ]
        rows, capacity = pad_rows(rows, capture_sizes)
        retired = (RequestKey(0, 0),) if step == 4 else ()
        lifecycle.append(dict(step=step, rows=rows, retired=retired, default_capacity=capacity))
    workloads["adversarial_chunk_boundaries"] = boundary
    workloads["adversarial_membership_churn_and_long_tail"] = churn
    workloads["adversarial_pause_resume_recycle_zero"] = lifecycle
    return workloads


def structural_record(plan, elapsed_us, step, rules):
    assert_semantic_domain(plan)
    if not 0 <= plan.scratch_used_bytes <= rules.scratch_limit_bytes:
        raise AssertionError("scratch allocation exceeds declared bound")
    loads = [part.cost_units for part in plan.partitions]
    return dict(
        step=step["step"],
        default_capacity=step["default_capacity"],
        cpu_planner_us=elapsed_us,
        live_requests=sum(row.key is not None and row.query_length > 0 for row in plan.rows),
        tiles=len(plan.tiles),
        reductions=len(plan.reductions),
        scratch_used_bytes=plan.scratch_used_bytes,
        estimated_work_units=sum(tile.cost_units for tile in plan.tiles),
        estimated_max_core_units=max(loads, default=0),
        estimated_core_units=loads,
        diagnostics=plain(plan.diagnostics),
    )


def run_variant(workload, geometry, rules, *, incremental, repeats):
    rounds = []
    for _ in range(repeats):
        planner = IncrementalAttentionPlanner(geometry, rules)
        samples = []
        for step in workload:
            started = time.perf_counter_ns()
            if incremental:
                plan = planner.plan(step["rows"], retired=step["retired"])
            else:
                plan = full_rebuild_plan(step["rows"], geometry, rules)
            elapsed_us = (time.perf_counter_ns() - started) / 1000
            samples.append(structural_record(plan, elapsed_us, step, rules))
        rounds.append(samples)
    timings = [sum(sample["cpu_planner_us"] for sample in samples) for samples in rounds]
    return dict(
        cpu_total_us_by_round=timings,
        cpu_median_total_us=statistics.median(timings),
        cpu_timing_scope=(
            "Python CPU planner call only; semantic audits/reporting excluded; not NPU submission or replay"
        ),
        rounds=rounds,
    )


def independent_field_totals(samples):
    """Aggregate reported counters without asserting their performance value."""
    totals = Counter()
    labels = {}
    for sample in samples:
        for name, value in sample["diagnostics"].items():
            if isinstance(value, bool):
                totals[name] += int(value)
            elif isinstance(value, (int, float)):
                totals[name] += value
            elif isinstance(value, str) or value is None:
                labels.setdefault(name, Counter())[str(value)] += 1
    return dict(numeric_sums=dict(totals), categorical_counts={name: dict(count) for name, count in labels.items()})


def compare_variants(variants):
    comparisons = {}
    for variant in ("incremental_no_split", "incremental_selective_split"):
        reference = "full_rebuild_" + variant.removeprefix("incremental_")
        baseline = variants[reference]["rounds"][0]
        candidate = variants[variant]["rounds"][0]
        if len(baseline) != len(candidate):
            raise AssertionError("variant workload lengths differ")
        comparisons[variant] = dict(
            same_exact_work=all(
                left["estimated_work_units"] == right["estimated_work_units"]
                for left, right in zip(baseline, candidate, strict=True)
            ),
            full_rebuild_sum_estimated_max_core_units=sum(item["estimated_max_core_units"] for item in baseline),
            incremental_sum_estimated_max_core_units=sum(item["estimated_max_core_units"] for item in candidate),
            scratch_peak_bytes=max((item["scratch_used_bytes"] for item in candidate), default=0),
            diagnostic_totals_first_round=independent_field_totals(candidate),
            interpretation="Cost-unit sums describe this scheduling proxy, never measured device latency",
        )
    return comparisons


def merge_cost_sensitivity(workload, geometry, rules):
    cases = {}
    for charge in (0, 4, 16):
        selected = dataclasses.replace(rules, split_merge_cost_units=charge)
        results = {}
        for incremental in (False, True):
            planner = IncrementalAttentionPlanner(geometry, selected)
            records = []
            for step in workload:
                if incremental:
                    plan = planner.plan(step["rows"], retired=step["retired"])
                else:
                    plan = full_rebuild_plan(step["rows"], geometry, selected)
                assert_semantic_domain(plan)
                records.append(
                    dict(
                        step=step["step"],
                        estimated_critical_path_units=plan.diagnostics.estimated_critical_path_units,
                        estimated_max_core_units=plan.diagnostics.max_load_units,
                        split_groups=len(plan.reductions),
                        scratch_used_bytes=plan.scratch_used_bytes,
                        fallback_reason=plan.diagnostics.fallback_reason,
                    )
                )
            results["incremental" if incremental else "full_rebuild"] = dict(
                sum_estimated_critical_path_units=sum(item["estimated_critical_path_units"] for item in records),
                total_split_groups=sum(item["split_groups"] for item in records),
                scratch_peak_bytes=max((item["scratch_used_bytes"] for item in records), default=0),
                fallback_counts=dict(Counter(str(item["fallback_reason"]) for item in records)),
                steps=records,
            )
        cases[str(charge)] = results
    return dict(
        fixed_charges=(0, 4, 16),
        units="Uncalibrated abstract cost units per merge; no device time conversion or best-case selection",
        zero_charge="Optimistic zero-merge-cost ceiling only, not a deployable performance claim",
        cases=cases,
    )


def locality_summary(workload, tile):
    previous_active = set()
    remembered = {}
    flips, crossings, observations, repeated = [], 0, 0, 0
    for step in workload:
        current = {row.key for row in step["rows"] if row.key is not None and row.query_length > 0}
        if step["step"]:
            flips.append(len(previous_active ^ current))
        previous_active = current
        for row in step["rows"]:
            if row.key is None or row.query_length == 0:
                continue
            chunks = (row.kv_length + tile - 1) // tile
            if row.key in remembered:
                repeated += 1
                crossings += chunks != remembered[row.key]
            remembered[row.key] = chunks
            observations += 1
    return dict(
        active_observations=observations,
        repeated_request_observations=repeated,
        mean_active_membership_flips=statistics.mean(flips) if flips else 0,
        requests_crossing_chunk_count=crossings,
        chunk_crossing_fraction=crossings / observations if observations else 0,
        chunk_crossing_fraction_of_repeated=crossings / repeated if repeated else 0,
        interpretation=(
            "Active-set changes and chunk topology changes are separate; exact lengths still refresh every step"
        ),
    )


def screen(trace_path, selection_path, repeats):
    trace_path, selection_path = Path(trace_path), Path(selection_path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = payload["trace"]
    if trace_digest(trace) != payload["trace_sha256"]:
        raise ValueError("immutable trace digest mismatch")
    if payload["trace_sha256"] != ORIGINAL_TRACE_SHA256 or len(trace["steps"]) != 128 or trace["query"] != 4:
        raise ValueError("original workload must match the pinned 128-step Q4 trace")
    evidence = json.loads(selection_path.read_text(encoding="utf-8"))
    if evidence["trace_sha256"] != payload["trace_sha256"]:
        raise ValueError("capture evidence belongs to another trace")
    selection = evidence["selection"]
    geometry = Geometry(num_cores=20)
    rules = PlanPolicy()
    workloads = {
        "original_immutable_128_steps": original_workload(
            trace, selection["capture_sizes"], selection["dispatched_tokens"]
        )
    }
    workloads.update(adversarial_workloads(selection["capture_sizes"]))
    reports = {}
    for name, workload in workloads.items():
        variants = {}
        for label, split_count in (("no_split", 1), ("selective_split", rules.max_splits_per_group)):
            selected_rules = dataclasses.replace(rules, max_splits_per_group=split_count)
            for incremental in (False, True):
                variant = f"{'incremental' if incremental else 'full_rebuild'}_{label}"
                variants[variant] = run_variant(
                    workload, geometry, selected_rules, incremental=incremental, repeats=repeats
                )
        reports[name] = dict(
            steps=len(workload),
            workload_sha256=hashlib.sha256(
                json.dumps(plain(workload), sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            locality=locality_summary(workload, geometry.kv_tile),
            comparisons=compare_variants(variants),
            merge_cost_sensitivity=merge_cost_sensitivity(workload, geometry, rules),
            variants=variants,
        )
    sources = [
        trace_path,
        selection_path,
        Path(__file__),
        Path(__file__).with_name("aclgraph_incremental_attention_plan.py"),
        Path(__file__).with_name("aclgraph_incremental_attention_plan_types.py"),
        Path(__file__).with_name("multi_graph_dynamic_trace.py"),
    ]
    return dict(
        scope=(
            "CPU algorithm/proof screening only; "
            "no native codec, NPU kernel, execution qualification or performance acceptance"
        ),
        device_measurements=None,
        predicted_npu_speedup=None,
        cost_model=(
            "WorkTile cost_units are arithmetic scheduling proxies; they are not microseconds or hardware counters"
        ),
        trace_sha256=payload["trace_sha256"],
        geometry=plain(geometry),
        policy=plain(rules),
        default_capture_sizes=selection["capture_sizes"],
        source_sha256={str(path.resolve()): sha256(path) for path in sources},
        workloads=reports,
    )


def main():
    root = Path(__file__).resolve().parent / "results" / "multi_graph_structural"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=root / "structural-dynamic-trace-v1.json")
    parser.add_argument("--selection", type=Path, default=root / "q4-tile-full-replay-v8-queue0.rank0.json")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("positive repeats and a new output path are required")
    report = screen(args.trace, args.selection, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(
        json.dumps(dict(output=str(args.output.resolve()), scope=report["scope"], workloads=list(report["workloads"])))
    )


if __name__ == "__main__":
    main()
