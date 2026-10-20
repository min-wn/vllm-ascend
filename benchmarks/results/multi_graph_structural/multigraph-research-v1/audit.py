"""Offline research audit; no NPU execution and no fitted latency prediction.

Run from the repository root with Python. Recomputes catalog work counts and
historical timing summaries; source anchors are evidence locations, not an
automatic proof of the historical execution topology.
"""

import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_paired_summary(report, summary, mode, profile=None):
    rank_samples = [
        [
            r
            for r in rank["samples"]
            if r.get("mode", r.get("case")) == mode and (profile is None or r["profile"] == profile)
        ]
        for rank in report["ranks"]
    ]
    assert len(rank_samples) == 2
    rounds = sorted(r["round"] for r in rank_samples[0])
    assert rounds == list(range(len(rounds))) and len(rounds) >= 3
    assert all(sorted(r["round"] for r in rows) == rounds for rows in rank_samples)
    for metric in ("device_us", "wall_us"):
        medians = []
        for trial in rounds:
            pair = [next(r[metric] for r in rows if r["round"] == trial) for rows in rank_samples]
            assert len(pair[0]) == len(pair[1]) >= 3
            assert all(math.isfinite(v) and v > 0 for row in pair for v in row)
            medians.append(statistics.median(max(values) for values in zip(*pair, strict=True)))
        assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(medians, summary["rounds_" + metric], strict=True))
        assert math.isclose(statistics.median(medians), summary[metric], abs_tol=1e-6)


selection_path = HERE.parent / "two-layer-fused-v1/analysis.json"
selection = read(selection_path)["selection"]
catalog = selection["capture_sizes"]
assert catalog == sorted(set(catalog)) and selection["query"] == 4
sizes = [size for size in catalog if size % 4 == 0]
capacity_rows = []
ordered_pairs = set()
for batch in range(64, 129):
    live = 4 * batch
    padding = next(size for size in sizes if size >= live)
    pairs = [(a, b) for a in sizes for b in sizes if live <= a + b <= padding]
    assert pairs
    best = min(pairs, key=lambda pair: (sum(pair), abs(pair[0] - pair[1]), -pair[0]))
    # Independent check of the minimal legal sum; no selected timing involved.
    legal_sums = {a + b for a in sizes for b in sizes}
    assert sum(best) == min(total for total in legal_sums if live <= total <= padding)
    ordered_pairs.update(pairs)
    strict = [pair for pair in pairs if sum(pair) in catalog and all(n % 8 == 0 for n in pair)]
    assert strict and min(map(sum, strict)) == padding
    capacity_rows.append(
        dict(
            batch=batch,
            live=live,
            padding=padding,
            smallest_pair=best,
            sum=sum(best),
            strict_fused_sum=min(map(sum, strict)),
        )
    )

history = []
for path in sorted((HERE / "history").glob("*.json")):
    report = read(path)
    row = dict(file=path.name, sha256=digest(path), status=report["status"], source_sha256=report["source_sha256"])
    if "measurements" in report:
        row["graph_count"] = report["graph_count"]
        row["combination_graph_count"] = report["combination_graph_count"]
        row["median_round_wall_us"] = {
            mode: statistics.median(
                statistics.median(r["wall_us"]) for r in report["measurements"] if r["mode"] == mode
            )
            for mode in ("padding", "split_parallel")
        }
    if "summaries" in report:
        for mode, summary in report["summaries"].items():
            verify_paired_summary(report, summary, mode)
        row["reported_timing_summaries"] = report["summaries"]
        row["raw_rank_pairing_verified"] = True
    if "profiles" in report:
        for name, profile in report["profiles"].items():
            for mode, summary in profile["summaries"].items():
                verify_paired_summary(report, summary, mode, name)
        row["profiles_keys"] = {name: list(profile) for name, profile in report["profiles"].items()}
        row["profile_timing_summaries"] = {name: profile["summaries"] for name, profile in report["profiles"].items()}
        row["raw_rank_pairing_verified"] = True
    history.append(row)

sources = []
for name, fragment, classification in (
    (
        "benchmarks/multi_graph_bucket_replay.py",
        "for index, capacity in enumerate(capacities)",
        "separate full decoder graphs",
    ),
    ("benchmarks/multi_graph_bucket_block.py", "with torch.npu.graph(graph,", "one composite graph per mode"),
    ("benchmarks/multi_graph_tp_attention.py", "with torch.npu.graph(self.graph,", "one composite graph per mode"),
    ("benchmarks/aclgraph_bounded_lead_probe.py", "with torch.npu.graph(graph,", "one composite graph per mode"),
    (
        "vllm_ascend/worker/multi_graph/distributed.py",
        'proposal.execution != "serial"',
        "parallel TP proposal rejected",
    ),
    (
        "vllm_ascend/worker/multi_graph/npu_backend.py",
        "def update_interleaved",
        "interleaved updates already implemented",
    ),
):
    path = ROOT / name
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [i for i, line in enumerate(lines, 1) if fragment in line]
    assert matches, (name, fragment)
    sources.append(dict(path=name, sha256=digest(path), lines=matches, classification=classification))

matched_source = HERE / "history/source-34a0d25d8485.py"
assert digest(matched_source) == read(HERE / "history/cached-python-control.json")["source_sha256"]
padding_total = sum(row["padding"] for row in capacity_rows)
pair_total = sum(row["sum"] for row in capacity_rows)
full_model_path = HERE.parent / "full-core-join-v3/analysis.json"
full_model = read(full_model_path)
routing_screen = []
for metric in ("device_us", "wall_us"):
    steps = [row for row in full_model["per_step"] if row["metric"] == metric]
    padding = sum(row["medians"]["padding"] for row in steps)
    split = sum(row["medians"]["split_early"] for row in steps)
    hindsight = sum(min(row["medians"]["padding"], row["medians"]["split_early"]) for row in steps)
    routing_screen.append(
        dict(
            metric=metric,
            per_step=steps,
            padding_sum_of_medians=padding,
            split_sum_of_medians=split,
            hindsight_sum_of_medians=hindsight,
            incremental_pct_points_vs_best_fixed=100 * (min(padding, split) - hindsight) / padding,
            note="hindsight aggregation of two step medians, not an online result or bound for other schedules",
        )
    )
result = dict(
    scope="offline source/history audit and exhaustive Q4 capacity enumeration; not a speedup measurement",
    selection_sha256=digest(selection_path),
    catalog=catalog,
    catalog_work=dict(
        batches=list(range(64, 129)),
        padding_tokens=padding_total,
        pair_tokens=pair_total,
        capacity_reduction_pct=100 * (1 - pair_total / padding_total),
        rows=capacity_rows,
        ordered_pairs=len(ordered_pairs),
        unordered_pairs=len({tuple(sorted(pair)) for pair in ordered_pairs}),
        distinct_lane_sizes=len({n for pair in ordered_pairs for n in pair}),
        same_total_catalog_capacity_reduction_pct=100
        * (1 - sum(row["strict_fused_sum"] for row in capacity_rows) / padding_total),
        note="counts are not graph byte costs, stream costs, or a latency bound",
    ),
    history=history,
    exact_historical_source=dict(file=matched_source.name, sha256=digest(matched_source)),
    latest_composite_routing_screen=dict(source_sha256=digest(full_model_path), rows=routing_screen),
    source_anchors=sources,
    limitation="Current source anchors alone do not prove old source topology when exact snapshots are missing.",
)
(HERE / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result["catalog_work"].items() if key != "rows"}, indent=2))
print(json.dumps([{k: v for k, v in row.items() if k != "profiles_keys"} for row in history], indent=2))
