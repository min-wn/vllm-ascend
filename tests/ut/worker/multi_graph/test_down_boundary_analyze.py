# SPDX-License-Identifier: Apache-2.0
"""Reject ambiguous communication attribution without importing NPU packages."""

import importlib
from decimal import Decimal
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    return importlib.import_module("aclgraph_down_boundary_analyze")


def fixture():
    start = Decimal("1792419813123428.756")
    kinds = ["SDMA"] * 6 + ["Write Value"] * 4 + ["Notify Wait"] * 4
    tasks = [
        dict(
            ts=str(start + 2 * i),
            dur=Decimal("1"),
            pid=1,
            args={
                "Task Type": kind,
                "Physic Stream Id": 42,
                "Task Id": 9,
                "Subtask Id": i,
            },
        )
        for i, kind in enumerate(kinds)
    ]
    return dict(ts=str(start), dur=Decimal("30")), tasks


def test_additive_breakdown_preserves_fractional_timestamps(analyzer):
    collective, tasks = fixture()
    result = analyzer.decompose(collective, tasks)
    assert result["sdma_us"] == 6
    assert result["notify_wait_us"] == 4
    assert result["write_value_us"] == 4
    assert result["unclassified_gap_us"] == 16
    assert result["subtask_pattern"][1]["start_us"] == 2


def test_missing_subtask_is_not_silently_counted_as_launch_cost(analyzer):
    collective, tasks = fixture()
    with pytest.raises(ValueError, match="decomposition"):
        analyzer.decompose(collective, tasks[:-1])


def test_temporal_containment_does_not_prove_task_ownership(analyzer):
    collective, tasks = fixture()
    tasks[0]["args"]["Task Id"] = 10
    with pytest.raises(ValueError, match="different HCCL"):
        analyzer.decompose(collective, tasks)


def test_overlapping_task_classes_cannot_be_added(analyzer):
    collective, tasks = fixture()
    tasks[6]["ts"] = tasks[0]["ts"]
    with pytest.raises(ValueError, match="overlapping"):
        analyzer.decompose(collective, tasks)
