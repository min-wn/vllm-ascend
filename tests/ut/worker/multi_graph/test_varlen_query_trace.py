# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the frozen variable-query workload, not NPU emulation."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("varlen_trace", ROOT / "benchmarks/aclgraph_varlen_query_trace.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def frozen():
    return json.loads(
        (ROOT / "benchmarks/results/multi_graph_structural/varlen-query-v1/input-trace.json").read_bytes()
    )


def test_frozen_trace_reproduces_original_and_semantics():
    original = json.loads(
        (ROOT / "benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json").read_bytes()
    )
    expected = frozen()
    assert TRACE.make_trace(original) == TRACE.validate_trace(expected)
    assert len(expected["trace"]["steps"]) == 128
    for source, step in zip(original["trace"]["steps"], expected["trace"]["steps"], strict=True):
        assert step["request_ids"] == source["request_ids"]
        assert step["q_endpoints"][-1] == step["live_tokens"]
        assert set(step["q_lengths"][: step["batch"]]) <= {1, 2, 3, 4}
        for old_kv, q, kv in zip(source["seq_lengths"], step["q_lengths"], step["kv_lengths"]):
            assert old_kv - 4 == kv - q
    assert {q for step in expected["trace"]["steps"] for q in step["q_lengths"] if q} == {1, 2, 3, 4}


def test_tampered_trace_rejected():
    record = frozen()
    record["trace"]["steps"][0]["q_lengths"][0] += 1
    with pytest.raises(ValueError, match="identity"):
        TRACE.validate_trace(record)


@pytest.mark.parametrize("field", ["q_endpoints", "kv_lengths", "q_lengths"])
def test_resigned_invalid_contract_rejected(field):
    record = copy.deepcopy(frozen())
    record["trace"]["steps"][0][field][0] = -1
    record["trace_sha256"] = TRACE.digest(record["trace"])
    with pytest.raises(ValueError):
        TRACE.validate_trace(record)


def test_probe_scope_and_budget_frozen_before_measurement():
    trace = frozen()["trace"]
    assert trace["probe_steps"] == list(range(8))
    assert trace["profile_steps"] == [0, 7]
    assert trace["predeclared_gate"]["required_us_per_attention"] == 10
    assert trace["predeclared_gate"]["required_fraction"] == 0.05


def test_profile_inventory_does_not_double_count_join_or_summary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    analyze = importlib.import_module("aclgraph_varlen_query_analyze")
    events = [
        dict(ph="X", name="attention", dur=200, args={"Task Type": "KERNEL_MIX_AIC"}),
        dict(ph="X", name="NOTIFY_WAIT", dur=210, args={"Task Type": "NOTIFY_WAIT"}),
        dict(ph="X", name="Computing", dur=200, args={}),
        dict(ph="X", name="MEMCPY_ASYNC", dur=3, args={"Task Type": "PCIE_DMA_SQE"}),
        dict(ph="s", name="attention", args={"Task Type": "KERNEL_MIX_AIC"}),
    ]
    result = analyze.inventory({"traceEvents": events})
    assert result["kernels"] == {"attention": dict(count=1, duration_us=[200], median_us=200)}
    assert result["dma"] == dict(count=1, total_us=3)
    assert not result["waits"]["counted_as_extra_work"]


def test_measured_sources_and_profile_evidence_are_still_reproducible(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    analyze = importlib.import_module("aclgraph_varlen_query_analyze")
    result = analyze.analyze(ROOT / "benchmarks/results/multi_graph_structural/varlen-query-v1", ROOT)
    assert result["checked_q4_steps"] == 128
    assert result["source_hashes_verified"]
    assert result["decision"]["fused_output_initialization_cost_us"] is None
