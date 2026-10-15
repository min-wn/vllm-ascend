from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.worker.multi_graph.test_manager import execution as make_execution
from tests.ut.worker.multi_graph.test_metadata import DecodeMetadata
from vllm_ascend.worker.multi_graph.async_executor import AsyncGraphExecutor
from vllm_ascend.worker.multi_graph.buckets import BucketCostTable, KVBucketPolicy
from vllm_ascend.worker.multi_graph.calibration import CalibrationPolicy, OnlineCalibrator
from vllm_ascend.worker.multi_graph.costs import load_cost_table
from vllm_ascend.worker.multi_graph.distributed import DistributedPlanCoordinator, RankProposal, agree, request_digest
from vllm_ascend.worker.multi_graph.mixed_metadata import slice_mixed_metadata
from vllm_ascend.worker.multi_graph.planner import CostKey, MeasuredCost
from vllm_ascend.worker.multi_graph.request_plan import (
    RequestBatch,
    merge_capacity_outputs,
    mixed_request_slices,
    plan_capacity_decode,
    prepare_padded_decode,
    slice_query_boundaries,
)
from vllm_ascend.worker.multi_graph.runner_v2 import MultiGraphV2Bridge, normalize_v2_batch
from vllm_ascend.worker.multi_graph.types import GraphKey


def cost_key(signature, lengths=(2, 3, 4, 5)):
    return CostKey(tuple(GraphKey(lane, 2, 2, signature) for lane in (0, 1)), "parallel", lengths)


def test_bucket_conservative_estimate_retains_order_and_exact_priority(signature):
    first = cost_key(signature)
    second = cost_key(signature, (3, 4, 5, 6))
    table = BucketCostTable({first: MeasuredCost(100, 60), second: MeasuredCost(90, 80)}, KVBucketPolicy((8, 16)))
    assert table[first] == MeasuredCost(100, 60)
    assert table[cost_key(signature, (4, 5, 6, 7))] == MeasuredCost(90, 80)
    assert table.get(cost_key(signature, (4, 5, 6, 17))) is None
    assert table.get(cost_key(signature, (4, 5, 6, 9))) is None
    assert table.get(replace(first, execution="serial", kv_lengths=(4, 5, 6, 7))) is None
    sparse = BucketCostTable({first: MeasuredCost(100, 60)}, KVBucketPolicy((8,)))
    assert sparse.get(second) is None


@pytest.mark.parametrize("bounds", [(), (8, 8), (16, 8), (0, 8), (True, 8)])
def test_invalid_bucket_policy(bounds):
    with pytest.raises(ValueError):
        KVBucketPolicy(bounds)


def test_online_calibration_executes_once_per_step_and_roundtrips(signature, tmp_path):
    signature = replace(signature, runtime_identity="test-runtime")
    calibrator = OnlineCalibrator(signature, CalibrationPolicy(2, 2, 1))
    calls = []
    ticks = iter(range(0, 100000, 1000))
    for _ in range(4):
        calibrator.dispatch(
            cost_key(signature),
            lambda: calls.append("single"),
            lambda: calls.append("split"),
            lambda: None,
            lambda: next(ticks),
        )
    assert calls == ["single", "split", "single", "split"]
    path = tmp_path / "costs.json"
    calibrator.save(path)
    assert load_cost_table(str(path), signature, 2)[cost_key(signature)] == MeasuredCost(1, 1)
    calibrator.dispatch(
        cost_key(signature), lambda: calls.append("single"), lambda: calls.append("split"), lambda: None
    )
    assert calls[-1] == "single"
    assert len(calibrator.document().records[0].split_us) == 2


def test_online_calibration_failure_never_retries_other_path(signature):
    calibrator = OnlineCalibrator(signature, CalibrationPolicy(1, 1, 1))
    calls = []

    def fail():
        calls.append("single")
        raise RuntimeError("submitted")

    with pytest.raises(RuntimeError, match="submitted"):
        calibrator.dispatch(cost_key(signature), fail, lambda: calls.append("split"), lambda: None)
    assert calls == ["single"]
    with pytest.raises(RuntimeError, match="failed"):
        calibrator.dispatch(cost_key(signature), fail, fail, lambda: None)


@pytest.mark.parametrize("count", range(2, 10))
def test_padding_plans_cover_all_requests_and_trim_outputs(signature, count):
    entries = {GraphKey(lane, 4, 4, signature) for lane in (0, 1)}
    parts = plan_capacity_decode(count, entries, 6)
    if count > 8:
        assert parts is None
        return
    assert [index for part in parts for index in range(part.request_start, part.request_stop)] == list(range(count))
    outputs = tuple(torch.full((4, 2), float(part.graph_key.lane_id)) for part in parts)
    merged = merge_capacity_outputs(parts, outputs)
    assert merged.shape == (count, 2)
    assert merged[:, 0].tolist() == [0] * parts[0].count + [1] * parts[1].count


def test_padding_uses_reserved_scratch_and_leaves_sources_unchanged(signature):
    entries = {GraphKey(lane, 4, 4, signature) for lane in (0, 1)}
    part = plan_capacity_decode(5, entries, 3)[0]
    ids = torch.arange(5)
    blocks = torch.arange(10).reshape(5, 2)
    result = prepare_padded_decode(ids, ids, blocks, ids, (2,) * 5, part, (20,), frozenset({20}), frozenset(range(10)))
    assert result.input_ids.tolist() == [0, 1, 2, 0]
    assert result.block_tables[-1].tolist() == [20, 0]
    assert result.slot_mapping.tolist() == [0, 1, 2, -1]
    assert result.kv_lengths == (2, 2, 2, 1)
    assert result.valid_rows.tolist() == [True, True, True, False]
    assert blocks.tolist() == torch.arange(10).reshape(5, 2).tolist()
    with pytest.raises(ValueError, match="disjoint"):
        prepare_padded_decode(ids, ids, blocks, ids, (2,) * 5, part, (2,), frozenset({2}), frozenset(range(10)))


def test_mixed_slices_preserve_request_boundaries_and_prefill_flags():
    batch = RequestBatch((4, 1, 1, 1, 3), (8, 1, 8, 9, 10), (True, True, False, False, False))
    parts = mixed_request_slices(batch)
    assert [part.workload for part in parts] == ["prefill", "decode", "multi_token_decode"]
    assert [slice_query_boundaries(batch, part) for part in parts] == [(0, 4, 5), (0, 1, 2), (0, 3)]
    assert [index for part in parts for index in range(part.token_start, part.token_stop)] == list(range(10))


def test_async_slots_do_not_reuse_pending_resources(signature):
    first = make_execution.__wrapped__(signature)
    second = make_execution.__wrapped__(signature)
    first[1].query = lambda _: False
    second[1].query = lambda _: False
    executor = AsyncGraphExecutor((first[0], second[0]))
    _, _, plan, inputs, metadata = first
    output = executor.try_run(plan, inputs, inputs, metadata)
    executor.try_run(plan, inputs + 10, inputs, metadata)
    assert executor.try_run(plan, inputs + 20, inputs, metadata) is None
    assert first[0].pending_resources is not None
    first[1].query = lambda _: True
    executor.try_run(plan, inputs + 20, inputs, metadata)
    assert [slot.generation for slot in executor.slots] == [2, 1]
    assert output[:, 0].tolist() == [0, 3, 6, 9]
    assert not any(name == "wait" for name, _ in first[1].calls)


def test_async_slots_reject_shared_graph_ownership(execution):
    with pytest.raises(ValueError, match="independent"):
        AsyncGraphExecutor((execution[0], execution[0]))


def test_rank_agreement_fallback_and_fatal_divergence(execution):
    proposal = RankProposal(0, "tp2", request_digest(("a", "b"), (1, 1), (4, 5)), (2, 2), True)
    assert agree((proposal, proposal), 2) == (2, 2)
    assert agree((proposal, replace(proposal, ready=False)), 2) is None
    assert agree((proposal, replace(proposal, lane_sizes=(1, 3))), 2) is None
    assert agree((proposal, replace(proposal, execution="parallel")), 2) is None
    with pytest.raises(RuntimeError, match="mismatch"):
        agree((proposal, replace(proposal, step=1)), 2)
    coordinator = DistributedPlanCoordinator(2, lambda local: (local, local))
    plan = replace(execution[2], execution="serial")
    assert coordinator.coordinate(proposal, plan) == plan
    with pytest.raises(RuntimeError, match="out of order"):
        coordinator.coordinate(proposal, plan)


def v2_batch(inputs):
    return SimpleNamespace(
        num_reqs=4,
        req_ids=["a", "b", "c", "d"],
        query_start_loc_np=np.arange(5),
        num_tokens=4,
        seq_lens_np=np.array([2, 3, 4, 5]),
        is_prefilling_np=np.zeros(4, dtype=bool),
        num_draft_tokens=0,
        input_ids=inputs,
        positions=inputs,
    )


def test_v2_bridge_executes_actual_tokens_with_existing_core(execution):
    manager, _, plan, inputs, metadata = execution
    batch = v2_batch(inputs)
    output = MultiGraphV2Bridge(manager).run(batch, metadata, plan)
    assert output[:, 0].tolist() == [0, 3, 6, 9]
    assert normalize_v2_batch(batch).summary.query_lengths == (1,) * 4


@pytest.mark.parametrize("field,value", [("num_draft_tokens", 1), ("num_tokens", 8), ("num_reqs", 3)])
def test_v2_bridge_rejects_unproven_layout(execution, field, value):
    batch = v2_batch(execution[3])
    setattr(batch, field, value)
    with pytest.raises(ValueError):
        normalize_v2_batch(batch)


def test_mixed_metadata_rebases_queries_and_preserves_physical_slots():
    batch = RequestBatch((3, 1, 1), (8, 9, 10), (True, False, False))
    source = DecodeMetadata(
        seq_lens_list=[8, 9, 10],
        slot_mapping=torch.tensor([40, 41, 42, 80, 90]),
        block_tables=torch.tensor([[4], [8], [9]]),
        query_start_loc=torch.tensor([0, 3, 4, 5]),
    )
    part = mixed_request_slices(batch)[1]
    result = slice_mixed_metadata(source, batch, part, "DecodeOnly")
    assert result.query_start_loc.tolist() == [0, 1, 2]
    assert result.slot_mapping.tolist() == [80, 90]
    assert result.num_prefills == 0 and result.num_decode_tokens == 2
    result.slot_mapping.zero_()
    assert source.slot_mapping.tolist() == [40, 41, 42, 80, 90]


def test_v2_fractional_boundaries_cannot_be_silently_truncated(execution):
    batch = v2_batch(execution[3])
    batch.query_start_loc_np = np.array([0, 1.5, 2, 3, 4])
    with pytest.raises(ValueError, match="integral"):
        normalize_v2_batch(batch)


def test_async_query_failure_is_terminal(signature):
    first = make_execution.__wrapped__(signature)
    second = make_execution.__wrapped__(signature)
    executor = AsyncGraphExecutor((first[0], second[0]))
    _, _, plan, inputs, metadata = first
    executor.try_run(plan, inputs, inputs, metadata)
    executor.try_run(plan, inputs, inputs, metadata)

    def fail_query(_):
        raise RuntimeError("device query failed")

    first[1].query = fail_query
    with pytest.raises(RuntimeError, match="query failed"):
        executor.try_run(plan, inputs, inputs, metadata)
    assert executor.failed
    assert first[0].pending_resources is not None
    with pytest.raises(RuntimeError, match="failed"):
        executor.try_run(plan, inputs, inputs, metadata)


def test_calibration_bounds_sample_memory_and_ignores_stale_identity(signature):
    signature = replace(signature, runtime_identity="test-runtime")
    calibrator = OnlineCalibrator(signature, CalibrationPolicy(1, 1, 1))
    calls = []
    calibrator.dispatch(
        cost_key(signature), lambda: calls.append("baseline"), lambda: calls.append("split"), lambda: None
    )
    calibrator.dispatch(
        cost_key(signature, (8,) * 4), lambda: calls.append("baseline"), lambda: calls.append("split"), lambda: None
    )
    assert len(calibrator.samples) == 1 and calls == ["baseline", "baseline"]
    with pytest.raises(ValueError, match="signature"):
        calibrator.dispatch(cost_key(replace(signature, buffer_generation=1)), lambda: None, lambda: None, lambda: None)


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_bucket_upper_bounds": (32, 16)},
        {"minimum_bucket_vectors": 1},
        {"online_calibration": True},
        {"calibration_sample_interval": 0},
    ],
)
def test_invalid_extension_config_is_rejected(changes):
    from pydantic import ValidationError

    from vllm_ascend.worker.multi_graph.config import MultiGraphConfig

    with pytest.raises(ValidationError):
        MultiGraphConfig(**changes)
