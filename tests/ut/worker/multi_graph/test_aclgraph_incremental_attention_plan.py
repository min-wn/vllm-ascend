"""Independent CPU semantics for incremental attention work planning."""

import importlib.util
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch


@pytest.fixture
def planner_module(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    name = "aclgraph_incremental_attention_plan_cpu"
    spec = importlib.util.spec_from_file_location(name, root / "aclgraph_incremental_attention_plan.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def geometry(module, **changes):
    return module.Geometry(
        kv_heads=2, query_heads=4, head_dim=8, query_tile=2, kv_tile=4, page_size=4, num_cores=4, **changes
    )


def policy(module, **changes):
    values = dict(
        max_rows=8,
        max_requests=16,
        max_cached_requests=16,
        max_kv_length=64,
        max_query_length=4,
        max_splits_per_group=2,
        scratch_limit_bytes=4096,
    )
    return module.PlanPolicy(**(values | changes))


def row(module, request, kv, query=4, generation=0):
    return module.RequestRow(module.RequestKey(request, generation), query, kv)


def assert_exact_coverage(plan, rows, shape):
    """Enumerate the semantic attention domain without the planner's grouping."""
    actual = Counter()
    for tile in plan.tiles:
        source = rows[tile.row]
        assert source.key is not None and tile.key == source.key
        assert 0 <= tile.kv_head < shape.kv_heads
        assert 0 <= tile.query_start < tile.query_end <= source.query_length
        assert 0 <= tile.kv_start < tile.kv_end <= source.kv_length
        for query in range(tile.query_start, tile.query_end):
            causal_end = source.kv_length - source.query_length + query + 1 if shape.causal else source.kv_length
            for key in range(tile.kv_start, min(tile.kv_end, max(0, causal_end))):
                actual[(source.key, tile.kv_head, query, key)] += 1
    expected = Counter()
    for source in rows:
        if source.key is None:
            continue
        for head in range(shape.kv_heads):
            for query in range(source.query_length):
                end = source.kv_length - source.query_length + query + 1 if shape.causal else source.kv_length
                for key in range(max(0, end)):
                    expected[(source.key, head, query, key)] = 1
    assert actual == expected
    assigned = Counter(index for part in plan.partitions for index in range(part.start, part.end))
    assert assigned == Counter({index: 1 for index in range(len(plan.tiles))})
    for part in plan.partitions:
        assert part.cost_units == sum(tile.cost_units for tile in plan.tiles[part.start : part.end])
    assert_reductions_describe_partitions(plan)


def assert_reductions_describe_partitions(plan):
    expected = {}
    for part in plan.partitions:
        for index in range(part.start, part.end):
            tile = plan.tiles[index]
            key = (tile.row, tile.key, tile.kv_head, tile.query_start, tile.query_end)
            core_intervals = expected.setdefault(key, {})
            interval = core_intervals.setdefault(part.core, [index, index + 1, 0])
            interval[1] = index + 1
            interval[2] += tile.cost_units
    expected = {key: pieces for key, pieces in expected.items() if len(pieces) > 1}
    actual = {}
    for group in plan.reductions:
        key = (group.row, group.key, group.kv_head, group.query_start, group.query_end)
        assert key not in actual
        actual[key] = {part.core: [part.start, part.end, part.cost_units] for part in group.pieces}
        assert len(actual[key]) == len(group.pieces) > 1
        assert group.scratch_bytes > 0
    assert actual == expected
    assert plan.scratch_used_bytes == sum(group.scratch_bytes for group in plan.reductions)


def split_softmax(plan, rows, shape, queries, keys, values):
    """Execute independent FP32 split softmax using core partitions as splits."""
    outputs = {}
    ratio = shape.query_heads // shape.kv_heads
    for index, source in enumerate(rows):
        if source.key is None:
            continue
        result = torch.zeros((source.query_length, shape.query_heads, shape.head_dim), dtype=torch.float32)
        for query in range(source.query_length):
            limit = source.kv_length - source.query_length + query + 1 if shape.causal else source.kv_length
            for head in range(shape.query_heads):
                partials = []
                for part in plan.partitions:
                    positions = []
                    for tile in plan.tiles[part.start : part.end]:
                        if (
                            tile.row == index
                            and tile.kv_head == head // ratio
                            and tile.query_start <= query < tile.query_end
                        ):
                            positions.extend(range(tile.kv_start, min(tile.kv_end, max(0, limit))))
                    if positions:
                        scores = keys[source.key][positions, head // ratio] @ queries[source.key][query, head]
                        scores = scores / shape.head_dim**0.5
                        maximum = scores.max()
                        weights = (scores - maximum).exp()
                        partials.append(
                            (maximum, weights.sum(), weights @ values[source.key][positions, head // ratio])
                        )
                if partials:
                    maximum = torch.stack([part[0] for part in partials]).max()
                    denominator = sum(part[1] * (part[0] - maximum).exp() for part in partials)
                    numerator = sum(part[2] * (part[0] - maximum).exp() for part in partials)
                    result[query, head] = numerator / denominator
        outputs[source.key] = result
    return outputs


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("score_scale", [1, 200])
def test_cpu_plan_disjoint_domain_and_split_softmax(planner_module, causal, score_scale):
    module = planner_module
    shape = geometry(module, causal=causal)
    rows = (row(module, 0, 13), row(module, 1, 3, query=2), row(module, 2, 0, query=0), row(module, 3, 0))
    rows += (module.RequestRow(None, 4, 0),)
    plan = module.IncrementalAttentionPlanner(shape, policy(module)).plan(rows)
    assert_exact_coverage(plan, rows, shape)
    generator = torch.Generator().manual_seed(919)
    queries = {
        r.key: torch.randn((r.query_length, shape.query_heads, shape.head_dim), generator=generator) * score_scale
        for r in rows
        if r.key
    }
    keys = {
        r.key: torch.randn((r.kv_length, shape.kv_heads, shape.head_dim), generator=generator) for r in rows if r.key
    }
    values = {
        r.key: torch.randn((r.kv_length, shape.kv_heads, shape.head_dim), generator=generator) for r in rows if r.key
    }
    actual = split_softmax(plan, rows, shape, queries, keys, values)
    for source in rows:
        if source.key is None:
            continue
        expected = torch.zeros_like(actual[source.key])
        for query in range(source.query_length):
            end = source.kv_length - source.query_length + query + 1 if causal else source.kv_length
            for head in range(shape.query_heads):
                if end > 0:
                    scores = keys[source.key][:end, head // 2] @ queries[source.key][query, head] / shape.head_dim**0.5
                    expected[query, head] = scores.softmax(0) @ values[source.key][:end, head // 2]
        assert torch.isfinite(actual[source.key]).all()
        torch.testing.assert_close(actual[source.key], expected, rtol=2e-6, atol=2e-6)


def test_cpu_reorder_pause_resume_and_recycle(planner_module):
    module = planner_module
    shape, rules = geometry(module), policy(module)
    planner = module.IncrementalAttentionPlanner(shape, rules)
    first = (row(module, 0, 7), row(module, 1, 12), row(module, 2, 4))
    sequences = [first, (first[2], row(module, 0, 8)), (), (row(module, 1, 13), first[0])]
    for rows in sequences:
        plan = planner.plan(rows)
        assert_exact_coverage(plan, rows, shape)
        reference = module.full_rebuild_plan(rows, shape, rules)
        assert plan.tiles == reference.tiles
    fresh = row(module, 0, 5, generation=1)
    plan = planner.plan((fresh,), retired=(first[0].key,))
    assert_exact_coverage(plan, (fresh,), shape)
    assert all(tile.key.generation == 1 for tile in plan.tiles)


def test_cpu_force_rebuild_preserves_exact_work(planner_module):
    module = planner_module
    shape, rules = geometry(module), policy(module)
    planner = module.IncrementalAttentionPlanner(shape, rules)
    planner.plan((row(module, 0, 7),))
    rows = (row(module, 0, 8), row(module, 1, 17))
    ordinary = planner.plan(rows)
    forced = planner.plan(rows, force_rebuild=True)
    assert ordinary.tiles == forced.tiles == module.full_rebuild_plan(rows, shape, rules).tiles
    assert_exact_coverage(forced, rows, shape)


@pytest.mark.parametrize(
    "bad_rows",
    [
        lambda m: (row(m, 0, -1),),
        lambda m: (row(m, 0, 65),),
        lambda m: (row(m, 0, 3, query=5),),
        lambda m: (row(m, 0, 4), row(m, 0, 4)),
        lambda m: (m.RequestRow(None, 4, 1),),
    ],
)
def test_cpu_invalid_plan_is_transactional(planner_module, bad_rows):
    module = planner_module
    planner = module.IncrementalAttentionPlanner(geometry(module), policy(module))
    planner.plan((row(module, 1, 7),))
    before = planner.snapshot()
    with pytest.raises(ValueError):
        planner.plan(bad_rows(module))
    assert planner.snapshot() == before


def test_cpu_no_split_and_scratch_bound_preserve_coverage(planner_module):
    module = planner_module
    shape = geometry(module)
    rows = (row(module, 0, 63), row(module, 1, 4))
    for rules in (policy(module, max_splits_per_group=1), policy(module, scratch_limit_bytes=0)):
        plan = module.IncrementalAttentionPlanner(shape, rules).plan(rows)
        assert_exact_coverage(plan, rows, shape)
        assert plan.scratch_used_bytes <= rules.scratch_limit_bytes


def test_cpu_dominant_group_exposes_parallelism_even_at_unsplit_bound(planner_module):
    module = planner_module
    shape = replace(geometry(module), kv_heads=1, query_heads=2, query_tile=4)
    rows = (row(module, 0, 64),)
    unsplit = module.full_rebuild_plan(rows, shape, policy(module, max_splits_per_group=1))
    split = module.full_rebuild_plan(rows, shape, policy(module, max_splits_per_group=2))
    assert_exact_coverage(split, rows, shape)
    assert split.reductions
    assert max(part.cost_units for part in split.partitions) < max(part.cost_units for part in unsplit.partitions)
    assert len(split.reductions[0].pieces) == 2
    charged = module.full_rebuild_plan(rows, shape, policy(module, split_merge_cost_units=10**9))
    assert_exact_coverage(charged, rows, shape)
    assert charged.reductions == ()


def test_cpu_max_tiles_fails_before_state_change(planner_module):
    module = planner_module
    planner = module.IncrementalAttentionPlanner(geometry(module), policy(module, max_tiles=8))
    planner.plan((row(module, 0, 4),))
    before = planner.snapshot()
    with pytest.raises(ValueError):
        planner.plan((row(module, 1, 64),))
    assert planner.snapshot() == before


def test_cpu_oracle_rejects_missing_work_and_missing_reduction(planner_module):
    module = planner_module
    shape = replace(geometry(module), kv_heads=1, query_heads=2, query_tile=4)
    rows = (row(module, 0, 64),)
    plan = module.full_rebuild_plan(rows, shape, policy(module))
    assert plan.reductions
    with pytest.raises(AssertionError):
        assert_exact_coverage(replace(plan, tiles=plan.tiles[1:]), rows, shape)
    with pytest.raises(AssertionError):
        assert_exact_coverage(replace(plan, reductions=()), rows, shape)


def test_cpu_failed_retirement_and_geometry_leave_state_unchanged(planner_module):
    module = planner_module
    shape = geometry(module)
    planner = module.IncrementalAttentionPlanner(shape, policy(module))
    old = row(module, 0, 12)
    planner.plan((old,))
    before = planner.snapshot()
    with pytest.raises(ValueError):
        planner.plan((row(module, 2, 65),), retired=(old.key,))
    assert planner.snapshot() == before
    with pytest.raises(ValueError):
        planner.plan((old,), geometry=replace(shape, query_heads=3))
    assert planner.snapshot() == before


def test_cpu_repair_merge_charge_has_fresh_no_split_floor(planner_module):
    module = planner_module
    shape = replace(geometry(module), kv_heads=1, query_heads=2)
    rules = policy(module, split_merge_cost_units=16)
    planner = module.IncrementalAttentionPlanner(shape, rules)
    planner.plan((row(module, 0, 7), row(module, 1, 12)), force_rebuild=True)
    rows = (row(module, 0, 8), row(module, 1, 17))
    repaired = planner.plan(rows)
    rebuilt = module.full_rebuild_plan(rows, shape, rules)
    assert repaired.diagnostics.estimated_critical_path_units <= rebuilt.diagnostics.estimated_critical_path_units
    assert repaired.diagnostics.fallback_reason == "imbalance_or_split_budget"
    assert_exact_coverage(repaired, rows, shape)


def test_cpu_no_split_floor_uses_all_groups_when_cores_exceed_groups(planner_module):
    module = planner_module
    shape = replace(geometry(module), kv_heads=1, query_heads=2, query_tile=4, num_cores=20)
    rows = (row(module, 0, 8), row(module, 1, 16), row(module, 2, 12))
    reference = module.full_rebuild_plan(rows, shape, policy(module, max_splits_per_group=1))
    parts = module._no_split_partition(reference.tiles, module._groups(reference.tiles), shape)
    floor = replace(reference, partitions=parts)
    assert len(parts) == 3
    assert [(part.start, part.end) for part in parts] == [(0, 2), (2, 6), (6, 9)]
    assert_exact_coverage(floor, rows, shape)


def test_cpu_geometry_change_is_validated_and_exact(planner_module):
    module = planner_module
    shape, rules = geometry(module), policy(module)
    planner = module.IncrementalAttentionPlanner(shape, rules)
    rows = (row(module, 0, 13),)
    planner.plan(rows)
    changed = replace(shape, kv_tile=8)
    result = planner.plan(rows, geometry=changed)
    assert_exact_coverage(result, rows, changed)
