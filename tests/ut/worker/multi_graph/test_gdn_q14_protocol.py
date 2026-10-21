# SPDX-License-Identifier: Apache-2.0
"""CPU ownership and deferred-prefix checks independent of device arithmetic."""

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "benchmarks/multi_graph_gdn_q14_protocol.py"
SPEC = importlib.util.spec_from_file_location("gdn_q14_protocol", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def site(width, seed):
    return dict(
        query=[0.2 * (seed + i + 1) for i in range(width)],
        key=[(-1) ** i * (0.1 + 0.03 * (seed + i)) for i in range(width)],
        value=[0.4 - 0.07 * (seed + i) for i in range(width)],
        beta=[0.15 + 0.02 * i for i in range(width)],
        gate=[-0.05 * (i + 1) for i in range(width)],
    )


def dense_oracle(initial, inputs):
    """Expanded dense recurrence, with every prefix materialized independently."""
    state, states, outputs, logs = initial, [], [], []
    for q, k, v, beta, gate in zip(*inputs.values(), strict=True):
        decay = math.exp(gate)
        before = state
        state = decay * (1 - beta * k * k) * before + beta * k * v
        states.append(state)
        outputs.append(q * state)
        logs.append((decay, k, beta * (v - decay * k * before)))
    return states, outputs, logs


def assert_logs(actual, expected):
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed == pytest.approx(wanted, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("schedule", ["delayed", "ready"])
@pytest.mark.parametrize(
    ("previous_width", "current_width", "accepted"),
    [(1, 1, 1), (1, 4, 1)] + [(4, q, accepted) for q in (1, 4) for accepted in range(1, 5)],
)
def test_width_switch_uses_previous_prefix_independent_of_current_width(
    schedule, previous_width, current_width, accepted
):
    previous_inputs, current_inputs = site(previous_width, 1), site(current_width, 7)
    initial = 0.625
    old_states, old_outputs, old_logs = dense_oracle(initial, previous_inputs)
    compact = MODULE.ScalarCompact(initial)
    # Segment entry has no pending log, even when metadata says accepted=4.
    assert compact.step(**previous_inputs, accepted=4, schedule=schedule) == pytest.approx(old_outputs)
    assert compact.base == initial
    assert_logs(compact.logs, old_logs)
    wanted_base = old_states[accepted - 1]
    assert compact.accepted_state(accepted) == pytest.approx(wanted_base)
    states, outputs, logs = dense_oracle(wanted_base, current_inputs)
    assert compact.step(**current_inputs, accepted=accepted, schedule=schedule) == pytest.approx(outputs)
    assert compact.base == pytest.approx(wanted_base)
    assert_logs(compact.logs, logs)
    for prefix, wanted in enumerate(states, 1):
        assert compact.accepted_state(prefix) == pytest.approx(wanted)
    terminal_prefix = min(2, current_width)
    assert compact.terminal(terminal_prefix) == pytest.approx(states[terminal_prefix - 1])
    assert compact.base == pytest.approx(states[terminal_prefix - 1])
    assert compact.logs == []
    assert compact.accepted_state(4) == compact.base


def test_mixed_batches_follow_owner_through_reorder_pause_and_resume():
    initial = {9: 0.75, 7: -0.4, 3: 1.25}
    compact = {owner: MODULE.ScalarCompact(value) for owner, value in initial.items()}
    prefixes = {}
    previous_widths = {}
    batches = [
        ([9, 7, 3], [4, 1, 4], [1, 4, 3]),
        ([3, 9], [1, 4], [4, 2]),
        ([7, 9], [4, 1], [1, 4]),
        ([9, 3, 7], [1, 4, 4], [1, 1, 3]),
    ]
    for index, (owners, widths, accepted) in enumerate(batches):
        MODULE.validate_contract(owners, widths, accepted, previous_widths)
        paused = {owner: copy.deepcopy(state) for owner, state in compact.items() if owner not in owners}
        for owner, width, count in zip(owners, widths, accepted, strict=True):
            base = prefixes[owner][count - 1] if owner in prefixes else initial[owner]
            inputs = site(width, owner + index)
            states, outputs, logs = dense_oracle(base, inputs)
            schedule = "ready" if index % 2 else "delayed"
            assert compact[owner].step(**inputs, accepted=count, schedule=schedule) == pytest.approx(outputs)
            assert compact[owner].base == pytest.approx(base)
            assert_logs(compact[owner].logs, logs)
            prefixes[owner], previous_widths[owner] = states, width
        assert all(compact[owner] == state for owner, state in paused.items())
    for owner, count in {9: 1, 3: 4, 7: 2}.items():
        assert compact[owner].terminal(count) == pytest.approx(prefixes[owner][count - 1])
        assert compact[owner].logs == []


def test_packed_metadata_owner_is_first_current_row_and_padding_has_no_owner():
    result = MODULE.metadata([9, 2, 7], [1, 4, 1], [4, 1, 3], capacity_requests=4)
    assert result["sequence"] == [0, 1, 4, 1, 0]
    assert result["ids"] == [10, 3, 3, 3, 3, 8] + [0] * 10
    assert result["accepted"] == [4, 1, 3, 1]
    assert result["actual_query_tokens"] == 6
    assert result["buffer_rows"] == 16
    assert result["inactive_buffer_rows"] == 10
    assert result["forced_q4_filler_tokens"] == 6
    assert result["candidate_filler_tokens"] == 0
    assert result["useful_producer_columns_per_request"] == [2, 8, 2]
    assert result["producer_columns_per_key_head"] == 8
    begins = [0, 1, 5]
    assert [result["ids"][begin] - 1 for begin in begins] == [9, 2, 7]
    assert result["ids"][begins[0] + result["accepted"][0] - 1] != 10


@pytest.mark.parametrize("width", [2, 3])
def test_intermediate_query_widths_rejected_at_every_input_boundary(width):
    with pytest.raises(ValueError, match="Q1/Q4"):
        MODULE.query_lengths(dict(batch=1, request_ids=[0], proposals=[[1] * (width - 1)]))
    with pytest.raises(ValueError, match="Q1 or Q4"):
        MODULE.metadata([0], [width], [1])
    with pytest.raises(ValueError, match="Q1/Q4"):
        MODULE.ScalarCompact(1.0).step(**site(width, 0), accepted=1)


@pytest.mark.parametrize("accepted", [0, 5, -1, True, 1.5])
def test_invalid_acceptance_rejected_even_without_a_pending_log(accepted):
    with pytest.raises(ValueError, match="1..4"):
        MODULE.validate_contract([0], [1], [accepted])
    with pytest.raises(ValueError, match="1..4"):
        MODULE.ScalarCompact(1.0).accepted_state(accepted)


@pytest.mark.parametrize("accepted", [2, 3, 4])
def test_previous_q1_log_cannot_accept_unpublished_rows(accepted):
    with pytest.raises(ValueError, match="previous published log length"):
        MODULE.validate_contract([7], [4], [accepted], {7: 1})
    state = MODULE.ScalarCompact(1.0)
    state.step(**site(1, 0), accepted=1)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="previous published log length"):
        state.step(**site(4, 1), accepted=accepted)
    assert state == before


@pytest.mark.parametrize(
    ("owners", "widths", "accepted"),
    [([1, 1], [1, 4], [1, 1]), ([1], [1, 4], [1]), ([1], [1], []), ([True], [1], [1]), ([-1], [1], [1])],
)
def test_invalid_owner_rows_rejected(owners, widths, accepted):
    with pytest.raises(ValueError):
        MODULE.metadata(owners, widths, accepted)


@pytest.mark.parametrize("capacity", [0, 129, True, 1.5])
def test_invalid_backing_capacity_rejected(capacity):
    with pytest.raises(ValueError, match="capacity"):
        MODULE.metadata([1], [1], [1], capacity_requests=capacity)


@pytest.fixture(scope="module")
def ledger():
    path = ROOT / "benchmarks/results/multi_graph_structural/gdn-ready-generation-v1/pair0.ready/result.rank0.json"
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_freeze_selects_first_continuous_eligible_window_without_timing(ledger):
    source, checksum = ledger
    assert checksum == "4db07f4e6ebcd704ad462231105052e91e126a43602c7143d4cbf5edd8162588"
    original = copy.deepcopy(source)
    first = None
    for start in range(len(source["generation"]["steps"]) - 15):
        rows = source["generation"]["steps"][start : start + 16]
        previous, transitions, mixed = {}, set(), False
        for row in rows:
            widths = [len(tokens) + 1 for tokens in row["proposals"]]
            mixed |= len(set(widths)) > 1
            for owner, width in zip(row["request_ids"], widths, strict=True):
                if owner in previous:
                    transitions.add((previous[owner], width))
                previous[owner] = width
        if mixed and transitions == {(1, 1), (1, 4), (4, 1), (4, 4)}:
            first = rows[0]["step"]
            break
    document = MODULE.freeze_ledger(source, checksum)
    payload = document["payload"]
    assert payload["first_step"] == first == 54
    assert payload["last_step"] == 69
    assert [row["step"] for row in payload["steps"]] == list(range(54, 70))
    assert payload["capacity_requests"] == 56
    assert payload["stable_owners"] == list(range(56))
    assert payload["summary"] == dict(
        steps=16,
        request_steps=896,
        query_length_counts={1: 812, 4: 84},
        previous_acceptance_counts={1: 896},
        transitions=dict(q1_to_q1=700, q1_to_q4=70, q4_to_q1=56, q4_to_q4=14),
        step_modes=dict(q1=10, mixed=6),
        actual_query_tokens=1148,
        forced_q4_logical_tokens=3584,
        forced_q4_filler_tokens=2436,
        original_graph_capacity_rows=3584,
        original_graph_padding_rows=0,
        no_latency_inferred_from_token_counts=True,
    )
    assert payload["initial_previous_log_lengths"] == dict.fromkeys(range(56), 0)
    terminal = {}
    for row in payload["steps"]:
        for owner, decision in zip(row["request_ids"], row["decisions"], strict=True):
            terminal[str(owner)] = decision["accepted"]
    assert payload["terminal_accepted"] == terminal
    assert source == original
    payload["steps"][0]["request_ids"][0] = -1
    assert source == original


@pytest.mark.parametrize("bad", ["checksum", "workload"])
def test_freeze_rejects_changed_source(ledger, bad):
    source, checksum = copy.deepcopy(ledger)
    if bad == "checksum":
        checksum = "0" * 64
    else:
        source["workload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unchanged saved short-text ledger"):
        MODULE.freeze_ledger(source, checksum)


@pytest.mark.parametrize("bad", [None, "digest", "settings", "counts"])
def test_manifest_round_trip_and_integrity_rejection(ledger, tmp_path, bad):
    document = MODULE.freeze_ledger(*ledger)
    if bad == "digest":
        document["payload"]["first_step"] = 55
    elif bad == "settings":
        document["payload"]["settings"]["query_lengths"] = [1, 2, 4]
    elif bad == "counts":
        document["payload"]["summary"]["actual_query_tokens"] += 1
    if bad in ("settings", "counts"):
        document["payload_sha256"] = MODULE.digest(document["payload"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    if bad is None:
        assert MODULE.load_manifest(path) == json.loads(json.dumps(document))
    else:
        with pytest.raises(ValueError):
            MODULE.load_manifest(path)


@pytest.mark.parametrize(
    ("candidate", "baseline", "stable"),
    [
        ([90, 90, 90], [100, 100, 100], True),
        ([99, 99, 99], [100, 100, 100], False),
        ([99.1, 99.1, 99.1], [100, 100, 100], False),
        ([100, 100, 100], [100, 100, 100], False),
        ([87.75, 90, 92.25], [100, 100, 100], False),
        ([89.55, 90, 90.45], [100, 100, 100], True),
        ([95, 95, 95], [97, 100, 103], False),
        ([90, 90, 100.01], [100, 100, 100], False),
    ],
)
def test_stability_requires_three_positive_gains_over_one_percent_and_twice_repeat_span(candidate, baseline, stable):
    observed = MODULE.stability_screen(candidate, baseline)
    expected = [100 * (b - c) / b for c, b in zip(candidate, baseline, strict=True)]
    assert observed["paired_improvement_percent"] == pytest.approx(expected, abs=1e-12)
    assert observed["pooled_improvement_percent"] == pytest.approx(
        100 * (sum(baseline) - sum(candidate)) / sum(baseline), abs=1e-12
    )
    assert observed["stable"] is stable


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan])
@pytest.mark.parametrize("side", ["candidate", "baseline"])
def test_stability_rejects_nonpositive_or_nonfinite_lifecycle_endpoints(bad, side):
    endpoints = dict(candidate=[90, 90, 90], baseline=[100, 100, 100])
    endpoints[side][1] = bad
    with pytest.raises(ValueError, match="finite positive"):
        MODULE.stability_screen(**endpoints)


@pytest.mark.parametrize("candidate,baseline", [([90, 90], [100] * 3), ([90] * 3, [100] * 4)])
def test_stability_requires_exactly_three_paired_rounds(candidate, baseline):
    with pytest.raises(ValueError, match="exactly three"):
        MODULE.stability_screen(candidate, baseline)
