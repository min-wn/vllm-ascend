# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_union_and_intersection_do_not_double_count_nested_streams():
    m = load("aclgraph_full_path_budget")
    a, b = [(0, 5), (2, 3), (4, 7)], [(1, 2), (1, 4), (6, 9)]
    assert m.length(a) == 7
    assert m.intersection(a, b) == 4
    assert m.length([]) == 0
    with pytest.raises(ValueError, match="reversed"):
        m.union([(3, 1)])


def test_timestamp_keeps_sub_microsecond_resolution():
    m = load("aclgraph_full_path_budget")
    a, b = m.interval(dict(ts="1792419813172707.961", dur=Decimal("0.001")))
    assert b - a == Decimal("0.001")


def full_trace():
    events, clock = [], -10

    def add(name, duration, kernel=True):
        nonlocal clock
        events.append(
            dict(name=name, ts=str(clock), dur=duration, ph="X", args={"Task Type": "KERNEL_AICORE"} if kernel else {})
        )
        clock += duration

    add("hcom_allReduce_embed", 6, False)
    for layer in range(64):
        add("MatMul", 4)
        if layer % 4 != 3:
            add("MatMul", 2)
        add("FusedInferAttentionScore" if layer % 4 == 3 else "q4_tile_solve_1", 5)
        add("MatMul", 4)
        add("hcom_allReduce_o", 6, False)
        add("MatMul", 7)
        add("MatMul", 4)
        add("hcom_allReduce_down", 6, False)
    return events


def test_budget_counts_exposed_union_and_fails_incomplete_model(tmp_path):
    m = load("aclgraph_full_path_budget")
    events = full_trace()
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(events))
    result = m.analyze_trace(path)
    assert result["communication_without_ai_us"] == 129 * 6
    assert result["ideal_equal_tiles_saving_us"]["4"] == 128 * 4 * 0.75
    # A separate AI stream covers communication. Exposure must vanish rather
    # than remaining the sum of collective durations.
    events.append(dict(name="background", ts="-10", dur=10000, ph="X", args={"Task Type": "KERNEL_AIVEC"}))
    path.write_text(json.dumps(events))
    result = m.analyze_trace(path)
    assert result["communication_without_ai_us"] == 0
    assert result["ideal_equal_tiles_saving_us"]["4"] == 0
    path.write_text(json.dumps(events[1:]))
    with pytest.raises(ValueError, match="64-layer"):
        m.analyze_trace(path)


@pytest.mark.parametrize("capacity", [304, 400, 496])
@pytest.mark.parametrize("chunks", [1, 2, 4, 8])
def test_chunks_cover_payload_once_without_splitting_q4(capacity, chunks):
    sizes = load("aclgraph_tp_chunk_cost_probe").request_chunks(capacity, chunks)
    assert len(sizes) == chunks and sum(sizes) == capacity
    assert all(s > 0 and s % 4 == 0 for s in sizes)
    assert max(sizes) - min(sizes) <= 4


@pytest.mark.parametrize("capacity,chunks", [(0, 1), (399, 2), (4, 2), (400, 0)])
def test_chunks_reject_invalid_partition(capacity, chunks):
    with pytest.raises(ValueError):
        load("aclgraph_tp_chunk_cost_probe").request_chunks(capacity, chunks)
