# SPDX-License-Identifier: Apache-2.0
"""Prove bounded search preserves exact minimax and canonical restoration."""

import argparse
import bisect
import hashlib
import itertools
import json
from pathlib import Path

from aclgraph_descriptor_audit import measured_costs
from aclgraph_fia_sink_stage4_whole_plan import _cuts, _head_cost


def prefix_bound(costs, cores):
    """Return a feasible <=P-chunk certificate, never a consumer partition."""
    if not costs or any(type(c) is not int or c < 0 for c in costs) or type(cores) is not int or cores < 1:
        raise ValueError("nonempty nonnegative integer costs and positive integer cores required")
    used = max(1, min(cores, sum(c > 0 for c in costs)))
    prefix = [0, *itertools.accumulate(costs)]
    total = prefix[-1]
    ends = [bisect.bisect_right(prefix, j * total // used) - 1 for j in range(1, used)] + [len(costs)]
    loads = [prefix[b] - prefix[a] for a, b in zip([0, *ends[:-1]], ends)]
    return dict(
        lower=max(max(costs), (total + used - 1) // used),
        upper=max(loads),
        total=total,
        used=used,
        prefix_ends=ends,
        chunk_loads=loads,
    )


def canonical_at_limit(costs, cores, limit):
    """Original parallel greedy jumps plus exact positive-atom reservation."""
    positive = [i for i, value in enumerate(costs) if value]
    if not positive:
        return [len(costs)]
    used = min(cores, len(positive))
    prefix = [0, *itertools.accumulate(costs)]
    jumps = [bisect.bisect_right(prefix, value + limit) - 1 for value in prefix]

    def apply(count):
        position, power = 0, jumps
        while count:
            if count & 1:
                position = power[position]
            power = [power[j] for j in power]
            count >>= 1
        return position

    return [min(apply(j), positive[len(positive) - used + j]) for j in range(1, used)] + [len(costs)]


def exact_partition(costs, cores, *, tightened):
    bound = prefix_bound(costs, cores)
    low, high = bound["lower"], bound["upper"] if tightened else bound["total"]
    iterations = 0
    while low < high:
        iterations += 1
        middle = (low + high) // 2
        groups, current = 1, 0
        for cost in costs:
            if current + cost > middle:
                groups, current = groups + 1, 0
            current += cost
        if groups <= bound["used"]:
            high = middle
        else:
            low = middle + 1
    return dict(limit=low, iterations=iterations, cuts=canonical_at_limit(costs, cores, low), **bound)


def check(costs, cores):
    bounded = exact_partition(costs, cores, tightened=True)
    original = exact_partition(costs, cores, tightened=False)
    assert bounded["lower"] <= bounded["limit"] <= bounded["upper"] <= bounded["total"]
    assert bounded["limit"] == original["limit"]
    assert bounded["cuts"] == original["cuts"] == _cuts(costs, cores)
    # Even a repeated quantile endpoint is a legal <=P certificate: its zero
    # chunk is ignored. Canonical refinement produces exactly P useful groups.
    feasible = canonical_at_limit(costs, cores, bounded["upper"])
    loads = [sum(costs[a:b]) for a, b in zip([0, *feasible[:-1]], feasible)]
    assert len(loads) == bounded["used"] and max(loads) <= bounded["upper"]
    assert all(a < b for a, b in zip([0, *feasible[:-1]], feasible))
    assert not any(costs) or all(loads)
    return bounded | dict(original_iterations=original["iterations"])


def costs_for_lengths(lengths):
    inner = min(max(lengths) or 1024, 512)
    return [cost for length in lengths for cost in [_head_cost(length, inner)] * 2]


def build():
    root = Path(__file__).resolve().parents[1]
    results = root / "benchmarks/results/multi_graph_structural"
    tested = 0
    for count in range(1, 8):
        for costs in itertools.product(range(3), repeat=count):
            for cores in range(1, min(5, count) + 1):
                check(costs, cores)
                tested += 1
    trace_path = results / "structural-dynamic-trace-v1.json"
    trace = json.loads(trace_path.read_bytes())
    catalog = json.loads((results / "two-layer-fused-v1/analysis.json").read_bytes())["selection"]["capture_sizes"]
    rows = []
    for step in trace["trace"]["steps"]:
        capacity = next(c for c in catalog if c >= step["batch"] * 4)
        lengths = step["seq_lengths"] + [0] * (capacity // 4 - step["batch"])
        record = check(costs_for_lengths(lengths), 20)
        rows.append(dict(step=step["step"], physical_tokens=capacity, **record))
    cases_path = results / "dynamic-descriptor-v1/plan-cases.json"
    cases = json.loads(cases_path.read_bytes())
    qualified = [dict(name=c["name"], **check(costs_for_lengths(c["kv_lengths"]), 20)) for c in cases["cases"]]
    return dict(
        cpu_exhaustive_cases=tested,
        original_trace_steps=rows,
        device_case_oracles=qualified,
        cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        stability_gate="All four paired-round medians improve device AND wall by at least 3% versus AICPU.",
        approximate_consumer_partition=False,
        bound_upload="No bound upload: constructed in the captured kernel. Current KV H2D remains timed.",
    )


def paired_gate(costs):
    gains = [
        dict(
            trial=row["trial"],
            **{
                endpoint: 100 * (1 - row[endpoint]["aiv_bounded"] / row[endpoint]["aicpu"])
                for endpoint in ("device_us", "wall_us")
            },
        )
        for row in costs["rounds"]
    ]
    return gains, all(row[endpoint] >= 3 for row in gains for endpoint in ("device_us", "wall_us"))


def analyse_producer(report, audit, root):
    """Apply the declared net-benefit gate, never the old-AIV-only comparison."""
    assert report["status"] == "complete_bounded_exact_producer_probe"
    assert report["post_timing_all_1024_words_checked"]
    assert report["cases_sha256"] == audit["cases_sha256"]
    for key, filename in (
        ("source_sha256", "aclgraph_descriptor_bound_probe.py"),
        ("kernel_sha256", "aclgraph_descriptor_bound_kernel.py"),
        ("unbounded_kernel_sha256", "aclgraph_descriptor_plan_kernel.py"),
    ):
        assert report[key] == hashlib.sha256((root / "benchmarks" / filename).read_bytes()).hexdigest()
    qualified = json.loads(
        (root / "benchmarks/results/multi_graph_structural/dynamic-descriptor-v1/plan-result-v5.json").read_bytes()
    )
    for key in ("extension_sha256", "opapi_sha256", "aicpu_sha256", "unbounded_kernel_sha256"):
        assert report[key] == qualified["kernel_sha256" if key == "unbounded_kernel_sha256" else key]
    names = [c["name"] for c in audit["device_case_oracles"]]
    assert [c["name"] for c in report["checks"]] == names
    assert all(c["all_1024_words_equal"] and c["descriptor_read_only"] for c in report["checks"])
    assert [c["name"] for c in report["bound_diagnostics"]] == names
    for actual, expected in zip(report["bound_diagnostics"], audit["device_case_oracles"], strict=True):
        assert actual["words"] == [expected[k] for k in ("lower", "upper", "limit", "iterations", "used", "total")] + [
            0,
            0,
        ]
    costs = measured_costs(report, ("aicpu", "aiv_unbounded", "aiv_bounded"))
    for mode, value in report["median_device_us"].items():
        assert costs["device_us"][mode] == value
    gains, passed = paired_gate(costs)
    history = json.loads(
        (root / "benchmarks/results/multi_graph_structural/fia-sink-consumer-analysis.json").read_bytes()
    )
    budget = [
        dict(
            capacity=p["capacity"],
            producer_us=p["metadata_duration_sum_us"],
            sixteen_attention_us=p["attention_duration_sum_us"],
            zero_cost_producer_inventory_fraction_percent=100
            * p["metadata_duration_sum_us"]
            / (p["metadata_duration_sum_us"] + p["attention_duration_sum_us"]),
        )
        for p in history["profile"]["replays"]
        if p["mode"] == "whole"
    ]
    folder = root / "benchmarks/results/multi_graph_structural/dynamic-descriptor-bound-v1"
    diagnostic = json.loads((folder / "producer-profile-result.json").read_bytes())
    assert diagnostic["status"] == "complete_metadata_only_diagnostic" and not diagnostic["samples"]
    for field in (
        "source_sha256",
        "kernel_sha256",
        "unbounded_kernel_sha256",
        "cases_sha256",
        "extension_sha256",
        "opapi_sha256",
        "aicpu_sha256",
    ):
        assert diagnostic[field] == report[field]
    profiles = {}
    for mode in ("aicpu", "aiv_unbounded", "aiv_bounded"):
        path = folder / f"profile/{mode}.json"
        events = json.loads(path.read_bytes())
        kernels = [
            e for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type", "").startswith("KERNEL_")
        ]
        expected = {
            "aicpu": ["aclnnInplaceZero_ZerosLikeAiCore_ZerosLike", "FusedInferAttentionScoreV2SinkMetadata"],
            "aiv_unbounded": ["descriptor_plan"],
            "aiv_bounded": ["descriptor_plan_bound"],
        }[mode]
        assert [e["name"] for e in kernels] == expected
        profiles[mode] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            kernels=[dict(name=e["name"], duration_us=e["dur"]) for e in kernels],
            dma_us=[
                e["dur"] for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type") == "PCIE_DMA_SQE"
            ],
        )
    return dict(
        scope="Exact-planner producer endpoint; bound construction and current-length H2D included.",
        costs=costs,
        improvement_vs_aicpu_by_round=gains,
        gate=audit["stability_gate"],
        stable_net_benefit=passed,
        consumer_chain_permitted=passed,
        disposition="qualified_for_existing_small_consumer_chain"
        if passed
        else "closed_current_shared_planner_candidate",
        device_bound_counters_match_cpu=True,
        complete_metadata_unchanged=True,
        diagnostic_profiles=profiles,
        profile_caveat=(
            "One profiled replay per mode, not a performance endpoint or causal breakdown; waits not added twice."
        ),
        historical_once_per_step_budget=budget,
        budget_caveat=(
            "Optimistic historical task-inventory fraction only; "
            "not a measured gain, end-to-end claim or universal bound."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path)
    args = parser.parse_args()
    result = build()
    target = (
        Path(__file__).resolve().parent / "results/multi_graph_structural/dynamic-descriptor-bound-v1/cpu-audit.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.producer is not None:
        analysis = analyse_producer(json.loads(args.producer.read_bytes()), result, Path(__file__).resolve().parents[1])
        target.with_name("producer-audit.json").write_text(
            json.dumps(analysis, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(analysis, indent=2))
    print(
        json.dumps(
            dict(
                exhaustive_cases=result["cpu_exhaustive_cases"],
                device_cases=[
                    {k: r[k] for k in ("name", "lower", "upper", "limit", "iterations", "original_iterations")}
                    for r in result["device_case_oracles"]
                ],
            ),
            indent=2,
        )
    )
