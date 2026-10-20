# SPDX-License-Identifier: Apache-2.0
"""Recompute candidate costs, the CPU mechanism, and bounded probe evidence."""

import bisect
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks/results/multi_graph_structural/dynamic-descriptor-v1"
sys.path.insert(0, str(ROOT / "benchmarks"))

from aclgraph_fia_sink_stage4_whole_plan import _cuts, whole_plan  # noqa: E402


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_costs(report, modes, samples_per_round=10):
    """Reject partial/duplicate samples; keep device and wall endpoints apart."""
    expected = set(itertools.product(range(4), modes, range(samples_per_round)))
    samples = report["samples"]
    actual = [(r["trial"], r["mode"], r["sample"]) for r in samples]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicate paired sample identities")
    if any(r[field] <= 0 for r in samples for field in ("device_us", "wall_us")):
        raise ValueError("nonpositive endpoint")
    return {
        endpoint: {mode: statistics.median(r[endpoint] for r in samples if r["mode"] == mode) for mode in modes}
        for endpoint in ("device_us", "wall_us")
    } | {
        "rounds": [
            dict(
                trial=trial,
                **{
                    endpoint: {
                        mode: statistics.median(
                            r[endpoint] for r in samples if r["mode"] == mode and r["trial"] == trial
                        )
                        for mode in modes
                    }
                    for endpoint in ("device_us", "wall_us")
                },
            )
            for trial in range(4)
        ]
    }


def profile_inventory(events, mode):
    """Inspect kernels, without adding dependent NOTIFY_WAIT a second time."""
    events = events.get("traceEvents", []) if isinstance(events, dict) else events
    work = [e for e in events if e.get("ph") == "X" and "Task Type" in e.get("args", {})]
    kernels = [e for e in work if e["args"]["Task Type"].startswith("KERNEL_")]
    expected = (
        [
            ("aclnnInplaceZero_ZerosLikeAiCore_ZerosLike", "KERNEL_AIVEC"),
            ("FusedInferAttentionScoreV2SinkMetadata", "KERNEL_AICPU"),
        ]
        if mode == "aicpu"
        else [("descriptor_plan", "KERNEL_AIVEC")]
    )
    if [(e["name"], e["args"]["Task Type"]) for e in kernels] != expected:
        raise ValueError("unexpected or incomplete producer kernel inventory")
    return dict(
        kernels=[dict(name=e["name"], task=e["args"]["Task Type"], duration_us=e["dur"]) for e in kernels],
        dma_us=[e["dur"] for e in work if e["args"]["Task Type"] == "PCIE_DMA_SQE"],
        note=(
            "Single diagnostic replay with profiler perturbation; "
            "not an unprofiled endpoint or instruction attribution."
        ),
    )


def plan_evidence(filename, source, kernel):
    report = read(HERE / filename)
    assert report["status"] == "complete_metadata_only_probe"
    assert report["source_sha256"] == sha(source)
    assert report["kernel_sha256"] == sha(kernel)
    assert report["cases_sha256"] == sha(HERE / "plan-cases.json")
    assert report["extension_sha256"] == "25a41e47473a3bc9c34a50c053766181c7a764ea496b93d4d3b70180b0ecf76b"
    assert report["opapi_sha256"] == "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40"
    assert report["aicpu_sha256"] == "1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90"
    assert [c["name"] for c in report["checks"]] == [c["name"] for c in read(HERE / "plan-cases.json")["cases"]]
    assert all(c["all_1024_words_equal"] and c["descriptor_read_only"] for c in report["checks"])
    costs = measured_costs(report, ("aicpu", "aiv"))
    for mode in ("aicpu", "aiv"):
        assert report["median_device_us"][mode] == costs["device_us"][mode]
    return dict(
        source=source.relative_to(ROOT).as_posix(),
        costs=costs,
        all_seven_generations_equal=True,
        post_timing_all_words_checked=report.get("post_timing_all_1024_words_checked", False),
        device_slowdown_percent=100 * (costs["device_us"]["aiv"] / costs["device_us"]["aicpu"] - 1),
        wall_slowdown_percent=100 * (costs["wall_us"]["aiv"] / costs["wall_us"]["aicpu"] - 1),
    )


def jump_cuts(costs, cores):
    """CPU model of prefix lookup and binary lifting, not timed device code."""
    positive = [i for i, c in enumerate(costs) if c]
    if not positive:
        return [len(costs)]
    used = min(cores, len(positive))
    prefix = [0, *itertools.accumulate(costs)]

    def successors(limit):
        return [bisect.bisect_right(prefix, value + limit) - 1 for value in prefix]

    def apply(jumps, count):
        position = 0
        while count:
            if count & 1:
                position = jumps[position]
            jumps = [jumps[j] for j in jumps]
            count >>= 1
        return position

    low, high = max(max(costs), (sum(costs) + used - 1) // used), sum(costs)
    while low < high:
        middle = (low + high) // 2
        if apply(successors(middle), used) >= len(costs):
            high = middle
        else:
            low = middle + 1
    jumps = successors(low)
    return [min(apply(jumps, k), positive[len(positive) - used + k]) for k in range(1, used)] + [len(costs)]


def golden_case(name, lengths):
    batch = len(lengths)
    maximum = max(lengths) or 1024
    words = [0] * 1024
    words[576:586] = [512, min(maximum, 512), 8, 0, 0, 0, 4, maximum, batch, batch]
    record = dict(physical_tokens=batch * 4, actual_seq_qlen=list(range(4, batch * 4 + 1, 4)), kv_lengths=lengths)
    return dict(name=name, kv_lengths=lengths, expected_metadata=whole_plan(record, words))


def build():
    results = HERE.parent
    trajectory = read(results / "structural-dynamic-trace-v1.json")
    trace = dict(trajectory["trace"])
    trace["initial_prefix_by_request"] = {int(k): v for k, v in trace["initial_prefix_by_request"].items()}
    digest = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    assert digest == trajectory["trace_sha256"] == "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
    selection = read(results / "two-layer-fused-v1/analysis.json")["selection"]
    catalog = selection["capture_sizes"]
    rows = []
    for step in trace["steps"]:
        live = step["batch"] * 4
        capacity = next(m for m in catalog if m >= live)
        rows.append(dict(step=step["step"], live=live, capacity=capacity))
    padded, live = sum(r["capacity"] for r in rows), sum(r["live"] for r in rows)
    tiles = {
        str(tile): dict(
            capacity_blocks=sum((r["capacity"] + tile - 1) // tile for r in rows),
            live_blocks=sum((r["live"] + tile - 1) // tile for r in rows),
        )
        for tile in (16, 32, 64, 128)
    }
    assert all(v["capacity_blocks"] == v["live_blocks"] for v in tiles.values())
    tested = 0
    for count in range(1, 8):
        for costs in itertools.product(range(3), repeat=count):
            for cores in range(1, min(5, count) + 1):
                assert jump_cuts(costs, cores) == _cuts(costs, cores)
                tested += 1
    same_capacity = [s for s, r in zip(trace["steps"], rows, strict=True) if r["capacity"] == 400]
    cases = [
        golden_case(f"original_step{s['step']}", s["seq_lengths"] + [0] * (100 - s["batch"])) for s in same_capacity[:2]
    ]
    assert len(cases) == 2
    cases += [
        golden_case("zero_kv", [0] * 100),
        golden_case("positive_singleton", [4097] + [0] * 99),
        golden_case("boundary_holes", ([0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 513, 4097] * 9)[:100]),
        golden_case("all_positive", [4097] * 100),
        cases[0] | {"name": "recover_original"},
    ]
    for case in cases:
        inner = case["expected_metadata"][577]
        costs = []
        for length in case["kv_lengths"]:
            full, tail = divmod(length, inner)
            cost = full * (12 + 10 * ((inner + 63) // 64)) + (12 + 10 * ((tail + 63) // 64) if tail else 0)
            costs.extend((cost, cost))
        assert jump_cuts(costs, 20) == _cuts(costs, 20)
    # Recompute semantics, but never rewrite the byte-hashed device input.
    assert read(HERE / "plan-cases.json") == dict(physical_batch=100, cases=cases)

    consumer = read(results / "fia-sink-consumer-analysis.json")
    planner = [
        dict(
            mode=p["mode"],
            capacity=p["capacity"],
            producer_us=p["metadata_duration_sum_us"],
            sixteen_attention_us=p["attention_duration_sum_us"],
        )
        for p in consumer["profile"]["replays"]
        if p["mode"] == "whole"
    ]
    fusion = read(results / "two-layer-fused-v1/analysis.json")
    fused = [
        dict(
            rank=r["rank"],
            down_rs_us=r["down_fused_work_us"],
            all_gather_us=r["all_gather_work_us"],
            projections_us=r["input_projection_us"],
        )
        for r in fusion["traces"]
        if r["mode"] == "fused_padding"
    ]
    probe = read(HERE / "dense-result.json")
    assert probe["status"] == "complete_bounded_community_gmm_probe"
    assert probe["source_sha256"] == sha(HERE / "dense-probe-success.py")
    assert (HERE / "dense-probe-success.py").read_bytes().replace(b"\r\n", b"\n") == (
        ROOT / "benchmarks/aclgraph_descriptor_dense_probe.py"
    ).read_bytes().replace(b"\r\n", b"\n")
    assert len(probe["checks"]) == 7 and probe["post_timing_full_output_checked"]
    medians = measured_costs(probe, ("matmul_nz", "gmm_full", "gmm_dynamic"))["device_us"]
    plans = {
        "v4": plan_evidence("plan-result-v4.json", HERE / "plan-probe-v4.py", HERE / "plan-kernel-v2.py"),
        "v5": plan_evidence(
            "plan-result-v5.json",
            ROOT / "benchmarks/aclgraph_descriptor_plan_probe.py",
            ROOT / "benchmarks/aclgraph_descriptor_plan_kernel.py",
        ),
    }
    assert plans["v5"]["post_timing_all_words_checked"]
    diagnostic = read(HERE / "plan-profile-result.json")
    assert diagnostic["status"] == "complete_metadata_only_diagnostic" and not diagnostic["samples"]
    assert diagnostic["source_sha256"] == sha(HERE / "plan-probe-profile.py")
    assert diagnostic["kernel_sha256"] == sha(HERE / "plan-kernel-v2.py")
    profiles = {mode: profile_inventory(read(HERE / f"profile/{mode}.json"), mode) for mode in ("aicpu", "aiv")}
    return dict(
        scope="Candidate cost and metadata mechanism audit; no model, router or multigraph prerequisite.",
        trace_digest=digest,
        input_files={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in (
                results / "structural-dynamic-trace-v1.json",
                results / "two-layer-fused-v1/analysis.json",
                results / "fia-sink-consumer-analysis.json",
                ROOT / "benchmarks/aclgraph_fia_sink_stage4_whole_plan.py",
                HERE / "dense-result.json",
                HERE / "plan-result-v4.json",
                HERE / "plan-result-v5.json",
                HERE / "plan-profile-result.json",
                HERE / "plan-cases.json",
                HERE / "profile/aicpu.json",
                HERE / "profile/aiv.json",
            )
        },
        dense=dict(
            steps=len(rows),
            live_rows=live,
            capacity_rows=padded,
            unused_rows=padded - live,
            unused_row_fraction_percent=100 * (1 - live / padded),
            tile_count_models=tiles,
            note="Counts only; neither FLOP fraction nor missing tile count proves a latency bound.",
            community_gmm_median_device_us=medians,
            dynamic_vs_community_full_slowdown_pct=100 * (medians["gmm_dynamic"] / medians["gmm_full"] - 1),
        ),
        attention=dict(
            current_shared_whole_profile_records=planner,
            shared_producers_per_step=1,
            cpu_parallel_partition_cases=tested,
            golden_cases=len(cases),
            mechanism="Prefix costs, parallel successors, binary lifting, canonical reservation; unchanged ABI.",
            producer_endpoints=plans,
            diagnostic_profiles=profiles,
            decision="AIV producer is slower; stop this implementation before consumer/model expansion.",
        ),
        communication_fusion=dict(
            current_fused_inventory=fused,
            note="Existing AG is necessary output reconstruction, not automatically removable cost.",
        ),
        selected="Shared attention metadata: replace AICPU by AIV, retaining canonical whole-head policy.",
        pending=[],
        not_measured=["Consumer-inclusive net benefit", "Full model performance"],
        next_boundary=(
            "Change plan-generation mechanism before another bounded producer gate; no router or model expansion."
        ),
    )


if __name__ == "__main__":
    result = build()
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                cpu_partition_cases=result["attention"]["cpu_parallel_partition_cases"],
                dense=result["dense"],
                selected=result["selected"],
                attention_producer=result["attention"]["producer_endpoints"],
            ),
            indent=2,
        )
    )
