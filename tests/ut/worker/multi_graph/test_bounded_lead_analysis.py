# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from decimal import Decimal
from pathlib import Path

import pytest


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    return importlib.import_module("aclgraph_bounded_lead_analysis")


def trace():
    origin = Decimal("1792419813172707.961")

    def event(name, start, duration, tid=1, kernel=True):
        return dict(
            name=name,
            ts=str(origin + Decimal(str(start))),
            dur=duration,
            tid=tid,
            ph="X",
            args={"Task Type": "KERNEL_AICORE"} if kernel else {},
        )

    events = [event("MatMul", start, 1) for start in (0, 2, 4, 7, 8.1, 13, 14.1, 20, 25, 27)]
    events += [
        event("hcom_allReduce", start, duration, 0, False) for start, duration in ((3, 3), (6, 4), (21, 3), (28, 3))
    ]
    events += [event("q4_tile_solve_1", start, 2) for start in (10, 16)]
    events += [event("q4_log_deferred_0", start, 2, 2) for start in (12, 18)]
    return events


def test_overlap_is_local_and_preserves_decimal_precision(tmp_path, monkeypatch):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace()))
    result = load(monkeypatch).analyze_trace(path, "split_early")
    assert result["input_a_overlap_last_collective_us"] == 2
    assert result["solve_a_overlap_last_collective_us"] == 0
    assert result["input_gemm_before_previous_down_end_us"] == 0
    assert result["counts"] == dict(matmul=10, all_reduce=4, solve=2, commit=2)


def test_barrier_control_cannot_execute_early(tmp_path, monkeypatch):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace()))
    with pytest.raises(ValueError, match="barrier control"):
        load(monkeypatch).analyze_trace(path, "split_barrier")


def test_incomplete_profile_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace()[:-1]))
    with pytest.raises(ValueError, match="inventory"):
        load(monkeypatch).analyze_trace(path, "split_early")
