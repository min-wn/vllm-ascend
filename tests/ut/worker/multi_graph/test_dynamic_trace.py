"""CPU checks of the fixed dynamic workload's request and state history."""

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dynamic_trace(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_dynamic_trace.py"
    spec = importlib.util.spec_from_file_location("multi_graph_dynamic_trace_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_trace_and_digest_are_deterministic_without_global_rng_side_effects(dynamic_trace):
    previous_rng = random.getstate()
    first = dynamic_trace.make_trace()
    second = dynamic_trace.make_trace()
    assert random.getstate() == previous_rng
    assert first == second
    assert dynamic_trace.trace_digest(first) == dynamic_trace.trace_digest(second)
    assert len(dynamic_trace.trace_digest(first)) == 64
    assert len(first["steps"]) == 128 and first["query"] == 4
    assert min(step["batch"] for step in first["steps"]) >= 64
    assert max(step["batch"] for step in first["steps"]) <= 128
    assert all(512 <= value <= 1536 for value in first["initial_prefix_by_request"].values())
    assert len({step["batch"] for step in first["steps"]}) > 1


@pytest.mark.parametrize("change", [dict(seed=20261019), dict(steps=127), dict(min_prefix=513)])
def test_changed_workload_parameters_change_digest(dynamic_trace, change):
    baseline = dynamic_trace.make_trace()
    assert dynamic_trace.trace_digest(baseline) != dynamic_trace.trace_digest(dynamic_trace.make_trace(**change))


def test_digest_ignores_mapping_order_but_detects_content_changes(dynamic_trace):
    trace = dynamic_trace.make_trace(steps=4, min_batch=2, max_batch=4)
    reordered = {key: value for key, value in reversed(tuple(trace.items()))}
    reordered["parameters"] = dict(reversed(tuple(reordered["parameters"].items())))
    assert dynamic_trace.trace_digest(trace) == dynamic_trace.trace_digest(reordered)
    changed = copy.deepcopy(trace)
    changed["steps"][0]["accepted_after_step"][0] = 5
    assert dynamic_trace.trace_digest(changed) != dynamic_trace.trace_digest(trace)


def test_saved_manifest_preserves_original_digest_after_json_round_trip(dynamic_trace):
    path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json"
    )
    document = json.loads(path.read_text())
    generated = dynamic_trace.make_trace()
    assert dynamic_trace.trace_digest(document["trace"]) == document["trace_sha256"]
    assert dynamic_trace.trace_digest(generated) == document["trace_sha256"]
    assert dynamic_trace.trace_digest(json.loads(json.dumps(generated))) == document["trace_sha256"]


@pytest.mark.parametrize("request_id", ["01", "-1", True, "x", "1.0"])
def test_digest_rejects_ambiguous_request_identity(dynamic_trace, request_id):
    trace = dynamic_trace.make_trace(steps=1, min_batch=1, max_batch=1)
    trace["initial_prefix_by_request"] = {request_id: 10}
    with pytest.raises(ValueError):
        dynamic_trace.trace_digest(trace)


def test_identity_preserves_pending_prefix_across_reordering_and_inactive_gaps(dynamic_trace):
    trace = dynamic_trace.make_trace()
    initial = trace["initial_prefix_by_request"]
    history, accepted_values, previous_steps, positions = {}, set(), {}, {}
    resumed, moved = 0, 0
    for number, step in enumerate(trace["steps"]):
        requests = step["request_ids"]
        assert step["step"] == number
        assert len(requests) == step["batch"] == len(set(requests))
        assert set(requests) <= set(initial)
        assert 0 <= step["input_seed"] < 2**31
        records = zip(
            requests,
            step["first_use"],
            step["seq_lengths"],
            step["accepted_from_previous"],
            step["accepted_after_step"],
            strict=True,
        )
        for position, (request, first_use, sequence, accepted, after) in enumerate(records):
            assert accepted in (1, 2, 3, 4) and after in (1, 2, 3, 4)
            accepted_values.add(after)
            assert first_use == (request not in history)
            if first_use:
                assert sequence == initial[request] + 4
                assert accepted == 1
            else:
                old_sequence, pending = history[request]
                # Paused steps contribute nothing: only the last emitted
                # accepted count advances this request at its next appearance.
                assert accepted == pending
                assert sequence == old_sequence + pending
                assert sequence > old_sequence
                resumed += int(number - previous_steps[request] > 1)
                moved += int(position != positions[request])
            history[request] = (sequence, after)
            previous_steps[request], positions[request] = number, position
    assert accepted_values == {1, 2, 3, 4}
    assert set(history) == set(initial)
    assert resumed > 0 and moved > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seed=True),
        dict(seed=1.5),
        dict(steps=0),
        dict(steps=-1),
        dict(steps=True),
        dict(min_batch=0),
        dict(min_batch=5, max_batch=4),
        dict(max_batch=False),
        dict(min_prefix=-1),
        dict(min_prefix=5, max_prefix=4),
        dict(max_prefix=2.5),
    ],
)
def test_trace_rejects_invalid_workload_contract(dynamic_trace, kwargs):
    with pytest.raises(ValueError):
        dynamic_trace.make_trace(**kwargs)


def test_single_identity_reuses_one_prefix_without_batch_position_dependence(dynamic_trace):
    trace = dynamic_trace.make_trace(seed=1, steps=12, min_batch=1, max_batch=1, min_prefix=0, max_prefix=0)
    expected_sequence, pending = 4, 1
    for index, step in enumerate(trace["steps"]):
        assert step["request_ids"] == [0]
        assert step["first_use"] == [index == 0]
        if index:
            expected_sequence += pending
        assert step["seq_lengths"] == [expected_sequence]
        assert step["accepted_from_previous"] == [pending]
        pending = step["accepted_after_step"][0]
