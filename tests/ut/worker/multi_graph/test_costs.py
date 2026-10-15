from dataclasses import replace

import pytest
from pydantic import ValidationError

from vllm_ascend.worker.multi_graph.costs import (
    DecodeCostDocument,
    DecodeCostRecord,
    DecodeForwardSample,
    build_cost_document,
    load_cost_table,
)
from vllm_ascend.worker.multi_graph.planner import ExactDecodePlanner
from vllm_ascend.worker.multi_graph.types import BatchSummary, ExecutionPlan, GraphKey, NoSplitReason


@pytest.fixture
def document(signature):
    return DecodeCostDocument(
        metric="synchronized_decode_forward_us",
        signature=replace(signature, runtime_identity="test-software-1"),
        records=(
            DecodeCostRecord(
                lane_sizes=(2, 2),
                execution="parallel",
                kv_lengths=(32, 64, 128, 256),
                single_graph_us=(100, 100, 100, 100, 1000),
                split_us=(70, 70, 70, 70, 700),
            ),
        ),
    )


def test_loaded_costs_enable_only_the_measured_batch(document, config, profile, tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(document.model_dump_json(), encoding="utf-8")
    costs = load_cost_table(str(path), document.signature, 5)
    assert next(iter(costs.values())).single_graph_us == 100
    planner = ExactDecodePlanner(replace(config, selection_policy="measured"))
    entries = {GraphKey(lane, 2, 2, document.signature) for lane in (0, 1)}
    batch = BatchSummary((1,) * 4, document.records[0].kv_lengths, True)
    assert isinstance(planner.plan(batch, profile, document.signature, entries, costs), ExecutionPlan)
    assert planner.plan(replace(batch, kv_lengths=(32,) * 4), profile, document.signature, entries, costs).reason == (
        NoSplitReason.MISSING_MEASUREMENT
    )
    assert not document.cost_table(document.signature, 6)


@pytest.mark.parametrize(
    "change",
    [
        {"runtime_identity": "different-runtime"},
        {"device_identity": "different-device"},
        {"model_config_hash": "different-weights"},
        {"buffer_generation": 1},
    ],
)
def test_stale_costs_fail_explicitly(document, change):
    with pytest.raises(ValueError, match="signature"):
        document.cost_table(replace(document.signature, **change), 5)


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_lengths": [32]},
        {"kv_lengths": [32, 64, 128, True]},
        {"lane_sizes": [2.0, 2]},
        {"split_us": []},
        {"single_graph_us": [100]},
        {"split_us": [float("nan")] * 5},
        {"split_us": [float("inf")] * 5},
        {"split_us": [0] * 5},
        {"unknown": True},
    ],
)
def test_invalid_record_rejected(document, changes):
    values = document.records[0].model_dump()
    values.update(changes)
    with pytest.raises(ValidationError):
        DecodeCostRecord.model_validate(values)


@pytest.mark.parametrize("metric", ["generation_ms", "kernel_us", "profiler_us"])
def test_other_metrics_cannot_be_imported(document, metric):
    with pytest.raises(ValidationError):
        DecodeCostDocument.model_validate({**document.model_dump(), "metric": metric})


def test_duplicate_records_and_missing_identity_rejected(document):
    with pytest.raises(ValidationError, match="duplicate"):
        DecodeCostDocument(**{**document.model_dump(), "records": document.records * 2})
    with pytest.raises(ValidationError, match="runtime identity"):
        DecodeCostDocument(**{**document.model_dump(), "signature": replace(document.signature, runtime_identity="")})


def test_cost_file_size_limit(document, tmp_path, monkeypatch):
    import vllm_ascend.worker.multi_graph.costs as costs

    monkeypatch.setattr(costs, "MAX_COST_FILE_BYTES", 8)
    path = tmp_path / "too_large.json"
    path.write_text(document.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="size limit"):
        load_cost_table(str(path), document.signature, 5)


def test_builder_aggregates_only_matching_decode_steps(document):
    baseline = tuple(DecodeForwardSample(kv_lengths=(2, 3, 4, 5), elapsed_us=100) for _ in range(5))
    split = tuple(DecodeForwardSample(kv_lengths=(2, 3, 4, 5), elapsed_us=70, lane_sizes=(2, 2)) for _ in range(5))
    result = build_cost_document(document.signature, "parallel", baseline, split)
    assert len(result.records) == 1
    assert len(result.records[0].split_us) == 5
    assert result.cost_table(document.signature, 5)
    with pytest.raises(ValueError, match="sample counts"):
        build_cost_document(document.signature, "parallel", baseline, split[:1])
    with pytest.raises(ValueError, match="same ordered"):
        build_cost_document(document.signature, "parallel", baseline, baseline)
    changed = tuple(sample.model_copy(update={"kv_lengths": (3, 2, 4, 5)}) for sample in split)
    with pytest.raises(ValueError, match="same ordered"):
        build_cost_document(document.signature, "parallel", baseline, changed)


def test_minimum_gain_threshold(document, config, profile):
    config = replace(config, selection_policy="measured", minimum_gain_fraction=0.4)
    signature = document.signature
    entries = {GraphKey(lane, 2, 2, signature) for lane in (0, 1)}
    batch = BatchSummary((1,) * 4, document.records[0].kv_lengths, True)
    result = ExactDecodePlanner(config).plan(batch, profile, signature, entries, document.cost_table(signature, 5))
    assert result.reason == NoSplitReason.NO_GAIN
