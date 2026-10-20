# SPDX-License-Identifier: Apache-2.0
"""Recompute dynamic catalog and current mixed-kernel evidence, without NPU use.

Capacity sums are row-count proxies, never graph-memory or latency estimates.
Trace intervals describe the observed schedule, not a counterfactual schedule.
"""

import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_audit():
    path = RESULTS / "structural-dynamic-trace-v1.json"
    document = read(path)
    trace = dict(document["trace"])
    trace["initial_prefix_by_request"] = {int(k): v for k, v in trace["initial_prefix_by_request"].items()}
    normalized = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("ascii")
    assert hashlib.sha256(normalized).hexdigest() == document["trace_sha256"]
    assert document["trace_sha256"] == "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
    source = RESULTS / "two-layer-fused-v1/analysis.json"
    catalog = read(source)["selection"]["capture_sizes"]
    legal = [n for n in catalog if n % trace["query"] == 0]
    rows, fixed_roles, balanced_roles = [], set(), set()
    for expected, step in enumerate(trace["steps"]):
        assert step["step"] == expected and len(set(step["request_ids"])) == step["batch"]
        live = trace["query"] * step["batch"]
        padding = next(n for n in legal if n >= live)
        pairs = [(a, b) for a in legal for b in legal if live <= a + b <= padding]
        best = min(pairs, key=lambda pair: (sum(pair), abs(pair[0] - pair[1]), pair))
        assert padding // 2 in legal and padding % 16 == 0
        balanced = [padding // 2, padding // 2]
        balanced_roles.update(enumerate(balanced))
        fixed = [256, padding - 256] if padding > 256 else [256]
        assert all(n in legal for n in fixed) and sum(fixed) == padding
        fixed_roles.update(enumerate(fixed))
        rows.append(dict(step=expected, batch=step["batch"], live=live, padding=padding, best_pair=list(best)))
    capacities = sorted({r["padding"] for r in rows})
    padding_total = sum(r["padding"] for r in rows)
    best_total = sum(sum(r["best_pair"]) for r in rows)
    return dict(
        trace_sha256=document["trace_sha256"],
        manifest_sha256=digest(path),
        selection_sha256=digest(source),
        steps=len(rows),
        observed_batches=len({r["batch"] for r in rows}),
        observed_capacities=capacities,
        histogram=dict(sorted(Counter(r["padding"] for r in rows).items())),
        padding_token_sum=padding_total,
        minimum_pair_token_sum=best_total,
        minimum_pair_capacity_reduction_pct=100 * (1 - best_total / padding_total),
        strict_fused_capacity_reduction_pct=0,
        resource_proxies=dict(
            padding=dict(
                handles=len(capacities), private_capacity_sum=sum(capacities), maximum_endpoint_rows=max(capacities)
            ),
            balanced_two_lanes=dict(
                handles=len(balanced_roles),
                private_capacity_sum=sum(n for _, n in balanced_roles),
                maximum_endpoint_rows=max(capacities),
            ),
            fixed_256_with_tail=dict(
                handles=len(fixed_roles),
                private_capacity_sum=sum(n for _, n in fixed_roles),
                maximum_endpoint_rows=max(capacities),
                role_capacities=[sorted(n for role, n in fixed_roles if role == i) for i in (0, 1)],
            ),
            note=(
                "Role-bound handles for these recipes only; "
                "endpoint rows and private capacity sums are not allocator bytes."
            ),
        ),
        rows=rows,
    )


def trace_audit(path, expected_hash):
    assert digest(path) == expected_hash
    document = json.loads(path.read_bytes(), parse_float=Decimal)
    events = document if isinstance(document, list) else document["traceEvents"]
    kernels = [
        e for e in events if e.get("ph") == "X" and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL_")
    ]
    ai = [e for e in kernels if e["args"]["Task Type"] != "KERNEL_AICPU"]
    comm = [e for e in events if e.get("ph") == "X" and e["name"].startswith(("hcom_allReduce", "hcom_allGather"))]
    changes = defaultdict(lambda: [0, 0])
    for category, selected in enumerate((ai, comm)):
        for e in selected:
            start, duration = Decimal(str(e["ts"])), Decimal(str(e["dur"]))
            assert duration > 0
            changes[start][category] += 1
            changes[start + duration][category] -= 1
    # An endpoint sweep counts each observed interval once, even with overlap.
    totals, active = defaultdict(Decimal), [0, 0]
    previous = min(changes)
    for point in sorted(changes):
        totals[tuple(n > 0 for n in active)] += point - previous
        active = [n + delta for n, delta in zip(active, changes[point], strict=True)]
        assert min(active) >= 0
        previous = point
    assert active == [0, 0] and sum(totals.values()) == max(changes) - min(changes)
    core = [e for e in kernels if e["name"] == "q4_tile_solve_1"]
    fia = [e for e in kernels if e["name"] == "FusedInferAttentionScore"]
    assert len(core) == len(fia) == sum(e["name"] == "q4_log_deferred_0" for e in kernels) == 1
    # Independently merge observed work and locate gaps. A gap is not proof of
    # an idle device: host supply, update waits and other task types may exist.
    merged = []
    for e in sorted(ai + comm, key=lambda event: Decimal(str(event["ts"]))):
        start = Decimal(str(e["ts"]))
        stop = start + Decimal(str(e["dur"]))
        if merged and start <= merged[-1][1]:
            if stop > merged[-1][1]:
                merged[-1][1], merged[-1][3] = stop, e["name"]
        else:
            merged.append([start, stop, e["name"], e["name"]])
    gaps = [
        dict(duration_us=float(b[0] - a[1]), before=a[3], after=b[2]) for a, b in zip(merged, merged[1:]) if b[0] > a[1]
    ]
    gap_sum = sum((b[0] - a[1] for a, b in zip(merged, merged[1:])), Decimal(0))
    assert gap_sum == totals[False, False]
    before_fia = sum(
        (b[0] - a[1] for a, b in zip(merged, merged[1:]) if b[2] == "FusedInferAttentionScore"), Decimal(0)
    )
    return dict(
        file=path.name,
        sha256=expected_hash,
        span_us=float(max(changes) - min(changes)),
        ai_union_us=float(totals[True, False] + totals[True, True]),
        communication_without_ai_us=float(totals[False, True]),
        ai_communication_overlap_us=float(totals[True, True]),
        neither_us=float(totals[False, False]),
        gap_immediately_before_fia_us=float(before_fia),
        other_neither_us=float(totals[False, False] - before_fia),
        largest_gaps=sorted(gaps, key=lambda g: g["duration_us"], reverse=True)[:3],
        core_task_type=core[0]["args"]["Task Type"],
        core_us=float(core[0]["dur"]),
        fia_us=float(fia[0]["dur"]),
        task_type_counts=dict(sorted(Counter(e["args"]["Task Type"] for e in kernels).items())),
        note="Mixed task labels establish neither exclusive core use nor spare Cube/Vector capacity.",
    )


def source_anchor(name, fragments):
    path = ROOT / name
    lines = path.read_text(encoding="utf-8").splitlines()
    found = {fragment: [i for i, line in enumerate(lines, 1) if fragment in line] for fragment in fragments}
    assert all(found.values())
    return dict(path=name, sha256=digest(path), lines=found)


def main():
    analysis = read(RESULTS / "two-layer-fused-v1/analysis.json")
    traces = []
    for row in analysis["traces"]:
        if row["mode"].startswith("fused_"):
            result = trace_audit(RESULTS / "two-layer-fused-v1/results" / row["file"], row["sha256"])
            result.update(rank=row["rank"], mode=row["mode"])
            traces.append(result)
    assert len(traces) == 6 and all(row["core_task_type"] == "KERNEL_MIX_AIC" for row in traces)
    result = dict(
        scope="offline mechanism screening; no new NPU execution, graph-memory measurement or latency prediction",
        catalog=catalog_audit(),
        current_fused_traces=traces,
        source_anchors=[
            source_anchor(
                "benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp",
                ["class Producer", "mm_.IterateAll", "MatmulImpl<AType, BType, CType>", "CrossCoreWaitFlag"],
            ),
            source_anchor("vllm_ascend/compilation/acl_graph.py", ["current_platform.get_global_graph_pool()"]),
        ],
        limitations=[
            "These recipes do not enumerate all useful specializations, schedules or modular designs.",
            "Shared graph-pool replay safety and allocator savings require runtime qualification.",
            "No PP speedup is inferred from old TP or request-owner measurements.",
            "Observed communication-only intervals need not be legally usable by another graph.",
        ],
    )
    (HERE / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result["catalog"].items() if k != "rows"}, indent=2))
    print(json.dumps(traces, indent=2))


if __name__ == "__main__":
    main()
