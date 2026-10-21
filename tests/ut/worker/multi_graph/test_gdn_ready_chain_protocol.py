# SPDX-License-Identifier: Apache-2.0
"""Frozen workloads and additive TP2 endpoints for independent controls."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def protocol(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    return importlib.import_module("multi_graph_gdn_ready_chain_protocol")


@pytest.fixture
def trace_path():
    return (
        Path(__file__).resolve().parents[4]
        / "benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json"
    )


def test_original_trace_and_terminal_contract_unchanged(protocol, trace_path):
    from multi_graph_gdn_q4_full_metadata import FIXED_TRACE_SHA256, load_fixed_trace
    from multi_graph_gdn_ready_full_lifecycle import terminal_counts

    actual, terminal, _ = protocol.workload(trace_path, "original")
    original = load_fixed_trace(trace_path)
    assert actual == original and actual.digest == FIXED_TRACE_SHA256
    assert terminal == terminal_counts(128)


@pytest.mark.parametrize("name", ["low", "medium"])
def test_range_preserves_existing_identity_acceptance_and_length_history(protocol, trace_path, name):
    from multi_graph_gdn_q4_full_metadata import decode_values, reserve_owner_pages
    from multi_graph_gdn_ready_range_protocol import frozen_profiles

    actual, terminal, contract = protocol.workload(trace_path, name)
    previous = next(row for row in frozen_profiles() if row["name"] == name)
    assert len(actual.steps) == 40 and len(actual.initial_prefix_by_request) == len(terminal) == 128
    assert set(actual.initial_prefix_by_request) == {16}
    assert tuple(terminal[:100]) == tuple(previous["terminal_counts"])
    assert contract["active_pool"] == previous["pool"]
    pages, committed, seen, history = reserve_owner_pages(actual), [0] * 128, set(), []
    resumed, reordered = False, False
    for row, old in zip(actual.steps, previous["trace"], strict=True):
        assert tuple(old["request_ids"]) == row["request_ids"]
        assert tuple(old["seq_lengths"]) == row["seq_lengths"]
        assert tuple(old["accepted"]) == row["accepted_from_previous"]
        decode_values(row, 4 * row["batch"], pages)
        for owner, count, length in zip(
            row["request_ids"], row["accepted_from_previous"], row["seq_lengths"], strict=True
        ):
            if owner in seen:
                committed[owner] += count
                resumed |= owner not in history[-1]
            assert length == 16 + committed[owner] + 4
        reordered |= list(row["request_ids"]) != sorted(row["request_ids"])
        seen.update(row["request_ids"])
        history.append(set(row["request_ids"]))
    assert resumed and reordered
    assert protocol.workload(trace_path, name)[0].digest == actual.digest
    with pytest.raises(TypeError):
        actual.steps[0]["request_ids"][0] = 0


def test_only_predeclared_loads_and_alternating_independent_process_pairs(protocol, trace_path):
    runs = protocol.run_plan()
    assert len(runs) == 12
    for name in protocol.LOADS:
        subset = [r for r in runs if r["load"] == name]
        assert [(r["pair"], r["order"], r["mode"]) for r in subset] == [
            (0, 0, "delayed"),
            (0, 1, "ready"),
            (1, 0, "ready"),
            (1, 1, "delayed"),
        ]
    with pytest.raises(ValueError):
        protocol.workload(trace_path, "high")


def costs():
    return [
        dict(rank=0, execution_us=100.0, initialization_us=20.0, terminal_us=5.0, total_us=125.0),
        dict(rank=1, execution_us=110.0, initialization_us=10.0, terminal_us=6.0, total_us=126.0),
    ]


def test_critical_endpoint_uses_all_columns_from_same_rank(protocol):
    assert protocol.critical_cost(costs()) == costs()[1]
    assert sum(protocol.critical_cost(costs())[k] for k in ("execution_us", "initialization_us", "terminal_us")) == 126


@pytest.mark.parametrize("bad", ["rank", "missing", "nan", "negative", "not_additive"])
def test_invalid_or_unclosed_rank_evidence_rejected(protocol, bad):
    rows = copy.deepcopy(costs())
    if bad == "rank":
        rows[1]["rank"] = 0
    elif bad == "missing":
        rows.pop()
    elif bad == "nan":
        rows[1]["execution_us"] = float("nan")
    elif bad == "negative":
        rows[1]["terminal_us"] = -1
    else:
        rows[1]["total_us"] += 1
    with pytest.raises(ValueError):
        protocol.critical_cost(rows)
