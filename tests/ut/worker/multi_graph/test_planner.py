from dataclasses import replace
from itertools import product

import pytest

from vllm_ascend.worker.multi_graph.config import MultiGraphConfig
from vllm_ascend.worker.multi_graph.planner import CostKey, ExactDecodePlanner, MeasuredCost
from vllm_ascend.worker.multi_graph.types import (
    BatchSummary,
    ExecutionPlan,
    GraphKey,
    NoSplit,
    NoSplitReason,
    SlicePlan,
)


def batch_for(count, kv_length=64):
    return BatchSummary((1,) * count, (kv_length,) * count, full_decode=True)


def test_416_uses_384_and_32_without_padding(config, profile, signature, entries):
    plan = ExactDecodePlanner(config).plan(batch_for(416), profile, signature, entries)
    assert isinstance(plan, ExecutionPlan)
    assert [part.actual_tokens for part in plan.slices] == [384, 32]
    assert plan.num_requests == 416
    assert plan.total_padding == 0


def test_equal_sizes_use_different_entry_keys(config, profile, signature, entries):
    plan = ExactDecodePlanner(config).plan(batch_for(512), profile, signature, entries)
    assert isinstance(plan, ExecutionPlan)
    assert plan.slices[0].graph_key != plan.slices[1].graph_key
    assert [part.actual_tokens for part in plan.slices] == [256, 256]


def test_exact_mode_does_not_use_256_capacity_twice_for_416(config, profile, signature):
    entries = frozenset(GraphKey(lane, 256, 256, signature) for lane in (0, 1))
    assert (
        ExactDecodePlanner(config).plan(batch_for(416), profile, signature, entries).reason == NoSplitReason.GRAPH_MISS
    )


@pytest.mark.parametrize("num_requests", range(10))
def test_exact_plans_cover_every_request_once(config, profile, signature, entries, num_requests):
    result = ExactDecodePlanner(config).plan(batch_for(num_requests), profile, signature, entries)
    if num_requests < 2:
        assert result.reason == NoSplitReason.TOO_SMALL
    elif num_requests <= 8:
        assert isinstance(result, ExecutionPlan)
        indices = [index for part in result.slices for index in range(part.request_start, part.request_stop)]
        assert indices == list(range(num_requests))
        assert all(part.actual_reqs == part.actual_tokens == part.graph_key.graph_tokens for part in result.slices)
    else:
        assert result.reason == NoSplitReason.GRAPH_MISS


def test_entry_subset_property_against_exhaustive_oracle(profile, signature):
    lane_sizes = (1, 2, 3)
    config = MultiGraphConfig(
        enabled=True,
        selection_policy="force",
        capture_sizes_per_lane=(lane_sizes, lane_sizes),
        graph_memory_budget_bytes=1024,
    )
    possible = tuple(GraphKey(lane, size, size, signature) for lane in (0, 1) for size in lane_sizes)
    planner = ExactDecodePlanner(config)
    for membership in product((False, True), repeat=len(possible)):
        entries = frozenset(key for key, present in zip(possible, membership) if present)
        for count in range(2, 8):
            oracle = {
                (left.graph_tokens, right.graph_tokens)
                for left in entries
                for right in entries
                if left.lane_id == 0 and right.lane_id == 1 and left.graph_tokens + right.graph_tokens == count
            }
            result = planner.plan(batch_for(count), profile, signature, entries)
            if oracle:
                assert isinstance(result, ExecutionPlan)
                assert tuple(part.actual_tokens for part in result.slices) in oracle
            else:
                assert result.reason == NoSplitReason.GRAPH_MISS


def test_force_choice_is_deterministic_and_handles_odd_batches(config, profile, signature, entries):
    planner = ExactDecodePlanner(config)
    first = planner.plan(batch_for(5), profile, signature, entries)
    assert [part.actual_tokens for part in first.slices] == [3, 2]
    assert first == planner.plan(batch_for(5), profile, signature, set(reversed(tuple(entries))))


def test_unconfigured_entries_cannot_expand_capture_budget(config, profile, signature):
    entries = frozenset((GraphKey(0, 100, 100, signature), GraphKey(1, 100, 100, signature)))
    assert (
        ExactDecodePlanner(config).plan(batch_for(200), profile, signature, entries).reason == NoSplitReason.GRAPH_MISS
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"buffer_generation": 1},
        {"model_config_hash": "different-model"},
        {"device_identity": "test-npu-1"},
        {"dtype": "float16"},
        {"kv_layout": "different-layout"},
        {"block_size": 64},
    ],
)
def test_graph_key_rejects_stale_or_incompatible_entries(config, profile, signature, entries, changes):
    changed = replace(signature, **changes)
    assert ExactDecodePlanner(config).plan(batch_for(416), profile, changed, entries).reason == NoSplitReason.GRAPH_MISS


def test_missing_lane_is_not_replaced_by_same_lane_entry(config, profile, signature, entries):
    entries = frozenset(key for key in entries if key.lane_id == 0)
    assert (
        ExactDecodePlanner(config).plan(batch_for(512), profile, signature, entries).reason == NoSplitReason.GRAPH_MISS
    )


def test_disabled_short_circuits_without_runtime_admission(profile, signature):
    result = ExactDecodePlanner(MultiGraphConfig()).plan(
        batch_for(416), replace(profile, runner_version="v2"), signature, frozenset()
    )
    assert result == NoSplit(NoSplitReason.DISABLED)


def test_prefill_or_nonfull_batch_falls_back(config, profile, signature, entries):
    planner = ExactDecodePlanner(config)
    for batch in (BatchSummary((2, 1), (64, 64), True), replace(batch_for(512), full_decode=False)):
        assert planner.plan(batch, profile, signature, entries).reason == NoSplitReason.UNSUPPORTED_BATCH


def test_force_does_not_bypass_support_checks(config, profile, signature, entries):
    result = ExactDecodePlanner(config).plan(batch_for(512), replace(profile, model_verified=False), signature, entries)
    assert result.reason == NoSplitReason.UNSUPPORTED_RUNTIME


def test_measured_default_requires_actual_pair_measurement(config, profile, signature, entries):
    planner = ExactDecodePlanner(replace(config, selection_policy="measured"))
    assert planner.plan(batch_for(512), profile, signature, entries).reason == NoSplitReason.MISSING_MEASUREMENT


@pytest.mark.parametrize("split_us,should_split", [(70, True), (100, False), (110, False)])
def test_measured_policy_compares_end_to_end_cost(config, profile, signature, entries, split_us, should_split):
    batch = batch_for(512)
    keys = (GraphKey(0, 256, 256, signature), GraphKey(1, 256, 256, signature))
    table = {CostKey(keys, "parallel", batch.kv_lengths): MeasuredCost(100, split_us)}
    planner = ExactDecodePlanner(replace(config, selection_policy="measured"))
    result = planner.plan(batch, profile, signature, entries, table)
    if should_split:
        assert isinstance(result, ExecutionPlan)
        assert result.cost_source == "measured"
    else:
        assert result.reason == NoSplitReason.NO_GAIN


def test_measurement_scope_includes_mode_kv_lengths_and_generation(config, profile, signature, entries):
    batch = batch_for(512)
    keys = (GraphKey(0, 256, 256, signature), GraphKey(1, 256, 256, signature))
    table = {CostKey(keys, "parallel", batch.kv_lengths): MeasuredCost(100, 50)}
    planner = ExactDecodePlanner(replace(config, selection_policy="measured"))
    assert (
        planner.plan(batch_for(512, 128), profile, signature, entries, table).reason
        == NoSplitReason.MISSING_MEASUREMENT
    )
    serial = ExactDecodePlanner(replace(config, selection_policy="measured", execution="serial"))
    assert serial.plan(batch, profile, signature, entries, table).reason == NoSplitReason.MISSING_MEASUREMENT
    stale_keys = tuple(replace(key, signature=replace(signature, buffer_generation=1)) for key in keys)
    table = {CostKey(stale_keys, "parallel", batch.kv_lengths): MeasuredCost(100, 50)}
    assert planner.plan(batch, profile, signature, entries, table).reason == NoSplitReason.MISSING_MEASUREMENT


def test_fastest_measured_pair_can_override_balanced_force_choice(config, profile, signature, entries):
    batch = batch_for(6)
    table = {
        CostKey(
            (GraphKey(0, 3, 3, signature), GraphKey(1, 3, 3, signature)), "parallel", batch.kv_lengths
        ): MeasuredCost(100, 90),
        CostKey(
            (GraphKey(0, 4, 4, signature), GraphKey(1, 2, 2, signature)), "parallel", batch.kv_lengths
        ): MeasuredCost(100, 70),
    }
    planner = ExactDecodePlanner(replace(config, selection_policy="measured"))
    result = planner.plan(batch, profile, signature, entries, table)
    assert [part.actual_tokens for part in result.slices] == [4, 2]


@pytest.mark.parametrize("time", [0, -1, float("nan"), float("inf")])
def test_invalid_measurements_are_rejected(time):
    with pytest.raises(ValueError):
        MeasuredCost(100, time)
    with pytest.raises(ValueError):
        MeasuredCost(time, 50)


def test_malformed_cpu_summary_is_rejected():
    for query, kv in (((1,), ()), ((0,), (64,)), ((2,), (1,))):
        with pytest.raises(ValueError):
            BatchSummary(query, kv, True)


def test_malformed_plan_is_rejected(signature):
    first_key = GraphKey(0, 2, 2, signature)
    second_key = GraphKey(1, 2, 2, signature)
    with pytest.raises(ValueError, match="capacity"):
        SlicePlan(0, 3, 0, 3, first_key)
    with pytest.raises(ValueError, match="contiguous"):
        ExecutionPlan((SlicePlan(0, 2, 0, 2, first_key), SlicePlan(3, 5, 3, 5, second_key)), "parallel", "force")
    with pytest.raises(ValueError, match="signatures"):
        ExecutionPlan(
            (
                SlicePlan(0, 2, 0, 2, first_key),
                SlicePlan(2, 4, 2, 4, replace(second_key, signature=replace(signature, buffer_generation=1))),
            ),
            "parallel",
            "force",
        )
