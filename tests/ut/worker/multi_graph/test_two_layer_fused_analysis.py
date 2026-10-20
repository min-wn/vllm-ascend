# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


def test_equal_fusion_attribution_uses_stronger_padding(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_two_layer_fused_analysis import summarize

    times = dict(
        padding=100, split_barrier=110, split_early=98, fused_padding=90, fused_split_barrier=105, fused_split_early=95
    )
    samples = [
        dict(round=r, sample=s, mode=m, device_us=t, wall_us=t + 1)
        for r in range(3)
        for m, t in times.items()
        for s in range(3)
    ]
    row = summarize(samples)[0]
    assert row["public_fusion_pct"] == pytest.approx(10)
    assert row["ordinary_split_pct"] == pytest.approx(2)
    assert row["equal_fusion_split_pct"] == pytest.approx(-100 * 5 / 90)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(samples[:-1])


def test_fused_trace_rejects_early_barrier_and_missing_core(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_two_layer_fused_analysis import fused_trace

    events, clock = [], 0

    def emit(name):
        nonlocal clock
        event = dict(name=name, ts=clock, dur=10, ph="X", tid=1, args={"Task Type": "KERNEL_AICORE"})
        events.append(event)
        clock += 10
        return event

    emit("MatMul")
    emit("FusedInferAttentionScore")
    emit("MatMul")
    emit("hcom_allReduce")
    emit("MatMul")
    emit("MatmulReduceScatterV2")
    emit("hcom_allGather")
    emit("MatmulReduceScatterV2")
    last_ag = emit("hcom_allGather")
    for _ in range(4):
        emit("MatMul")
    emit("q4_tile_solve_1")
    emit("q4_log_deferred_0")
    emit("MatMul")
    emit("hcom_allReduce")
    emit("MatMul")
    emit("MatMul")
    emit("hcom_allReduce")
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(events))
    assert fused_trace(path, "fused_split_barrier")["input_a_overlap_last_gather_us"] == 0
    last_ag["dur"] += 20
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="unfinished"):
        fused_trace(path, "fused_split_barrier")
    assert fused_trace(path, "fused_split_early")["input_a_overlap_last_gather_us"] == 20
    last_ag["dur"] += 10
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="gathered rows"):
        fused_trace(path, "fused_split_early")
    path.write_text(json.dumps([e for e in events if e["name"] != "q4_tile_solve_1"]))
    with pytest.raises(ValueError, match="inventory"):
        fused_trace(path, "fused_split_early")
