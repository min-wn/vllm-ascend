# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[4] / "benchmarks/aclgraph_fused_down_probe.py"
    spec = importlib.util.spec_from_file_location("fused_down_probe", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("capacity", [True, 400.0, 392, 4, 0])
def test_catalog_and_q4_ownership_gate(capacity):
    with pytest.raises(ValueError, match="whole Q4"):
        module().validate_capacity(capacity, [4, 304, 400])


def test_existing_default_capacities():
    for capacity in (304, 400):
        module().validate_capacity(capacity, [304, 400])


def test_slowest_rank_pairing_and_rejection():
    pair = module().pair_samples
    a = dict(trial=0, mode="mm_ar", sample=0, device_us=11, wall_us=14)
    b = dict(a, device_us=12, wall_us=13)
    assert pair([[a], [b]]) == [dict(a, device_us=12, wall_us=14)]
    for ranks in ([[a], [dict(b, sample=1)]], [[a, a], [b, b]], [[], []], [[a], []]):
        with pytest.raises(ValueError):
            pair(ranks)
    with pytest.raises(ValueError, match="nonfinite"):
        pair([[a], [dict(b, device_us=float("nan"))]])


def test_real_down_tp_ownership_is_column_sharded(tmp_path):
    import json

    index = tmp_path / "model.safetensors.index.json"
    key = "model.language_model.layers.3.mlp.down_proj.weight"
    index.write_text(json.dumps(dict(weight_map={key: "weights.safetensors"})))
    for rank, columns in ((0, (0, 8704)), (1, (8704, 17408))):
        assert module().down_weight_spec(index, rank) == (tmp_path / "weights.safetensors", key, columns)
    for rank in (True, 2, -1):
        with pytest.raises(ValueError, match="ownership"):
            module().down_weight_spec(index, rank)


def test_summary_requires_all_causal_controls(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_fused_down_analysis import summarize_samples

    samples = [
        dict(trial=t, mode=m, sample=s, device_us=d, wall_us=d + 1)
        for t in range(5)
        for m, d in (("mm_ar", 10), ("mm_rs_ag", 11), ("fused_rs_ag", 8))
        for s in range(6)
    ]
    result = summarize_samples(samples)
    assert result["fused_reduction_percent"]["device_us"] == pytest.approx(20)
    assert len(result["per_trial"]) == 10
    with pytest.raises(ValueError, match="complete"):
        summarize_samples(samples[:-1])
    with pytest.raises(ValueError, match="complete"):
        summarize_samples(samples[:-1] + [samples[0]])


def test_endpoint_must_join_final_collective(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_fused_down_analysis import validate_final_join

    producer = dict(ts=0, dur=100, tid=3)
    collective = dict(ts=101, dur=100, tid=7)
    wait = dict(ph="X", ts=100, dur=102, tid=3, args={"Task Type": "EVENT_WAIT"})
    validate_final_join([wait], producer, collective)
    for events in ([], [dict(wait, dur=20)], [dict(wait, tid=7)]):
        with pytest.raises(ValueError, match="final collective"):
            validate_final_join(events, producer, collective)
