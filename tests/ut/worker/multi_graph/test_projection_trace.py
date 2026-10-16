import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    folder = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("projection_trace_test", folder / "multi_graph_projection_trace.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def event(name, start, duration=1, tid=7):
    return dict(name=name, ts=start, dur=duration, tid=tid, pid=2, ph="X", args={"Task Type": "KERNEL_AICORE"})


def test_validated_projection_order(analyzer):
    events = [
        event(name, index)
        for index, name in enumerate(("MatMul", "FusedInferAttentionScore", "MatMul", "MatMul", "MatMul"))
    ]
    result = analyzer.classify_lane(list(reversed(events)), 1)
    assert [row["projection"] for row in result] == ["qkv", "fia", "o", "gate_up", "down"]
    assert all(row["layer"] == 0 for row in result)
    with pytest.raises(ValueError, match="sequence"):
        analyzer.classify_lane(events[:-1], 1)
    with pytest.raises(ValueError, match="sequence"):
        analyzer.classify_lane(events, 2)


def test_overlap_uses_union_and_clips(analyzer):
    assert (
        analyzer.overlap_us(
            event("MatMul", 10, 10), [event("MatMul", 8, 7), event("MatMul", 12, 5), event("MatMul", 19, 5)]
        )
        == 8
    )


def test_marker_requires_complete_decoder_lanes(analyzer):
    document = [
        dict(name="process_name", pid=2, args={"name": "Ascend Hardware"}),
        dict(name="buckets.split_parallel", ts="0", dur=10, ph="X", pid=1),
    ]
    for tid in (7, 8):
        document.extend(
            event(name, index, tid=tid)
            for index, name in enumerate(("MatMul", "FusedInferAttentionScore", "MatMul", "MatMul", "MatMul"))
        )
    rows = analyzer.analyze(document, 1)
    assert len(rows[0]["measurements"]) == 10
    assert rows[0]["kernel_union_us"] == 5
    assert all(row["other_mm_overlap_us"] == 1 for row in rows[0]["measurements"] if row["projection"] != "fia")
    with pytest.raises(ValueError, match="stream count"):
        analyzer.analyze(document[:-5], 1)
