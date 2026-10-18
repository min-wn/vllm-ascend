# SPDX-License-Identifier: Apache-2.0
"""CPU dependency/mode contracts; native phase equivalence needs the device."""

import importlib.util
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def deferred_probe(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "q4_log_deferred_probe_cpu", directory / "multi_graph_gdn_q4_log_deferred_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def fake_npu(calls):
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.index))

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is False
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            assert stream is npu.active
            calls.append(("record", stream.name, self.index))

    @contextmanager
    def use_stream(stream):
        previous, npu.active = npu.active, stream
        try:
            yield
        finally:
            npu.active = previous

    npu = SimpleNamespace(Stream=lambda: Stream("commit"), Event=Event, active=Stream("setup"), stream=use_stream)
    npu.current_stream = lambda: npu.active
    return npu, Stream, events


def test_all_controls_include_identical_preparation_and_projection(deferred_probe):
    calls = []
    npu, Stream, events = fake_npu(calls)
    result = object()

    def operation(name):
        def launch():
            calls.append((name, npu.active.name))
            return result

        return launch

    operations = deferred_probe.deferred_operations(
        SimpleNamespace(npu=npu), *(operation(name) for name in ("prepare", "fused", "solve", "commit", "project"))
    )
    assert calls == [] and len(events) == 2
    for mode, sequence in (
        ("fused_projection", ("prepare", "fused", "project")),
        ("deferred_serial", ("prepare", "solve", "commit", "project")),
    ):
        calls.clear()
        npu.active = Stream(mode)
        assert operations[mode]() is result
        assert calls == [(name, mode) for name in sequence]


def test_deferred_overlap_waits_for_solve_and_joins_commit_before_return(deferred_probe):
    calls = []
    npu, Stream, events = fake_npu(calls)
    result = object()

    def launch(name):
        def call():
            calls.append((name, npu.active.name))
            return result

        return call

    operations = deferred_probe.deferred_operations(
        SimpleNamespace(npu=npu), *(launch(name) for name in ("prepare", "fused", "solve", "commit", "project"))
    )
    for root in ("capture_one", "capture_two"):
        calls.clear()
        npu.active = Stream(root)
        assert operations["deferred_overlap"]() is result
        assert calls == [
            ("prepare", root),
            ("solve", root),
            ("record", root, 0),
            ("wait", "commit", 0),
            ("commit", "commit"),
            ("record", "commit", 1),
            ("project", root),
            ("wait", root, 1),
        ]
        assert npu.active.name == root and len(events) == 2


def test_mock_state_transaction_and_projection_are_identical_across_controls(deferred_probe):
    calls = []
    npu, _, _ = fake_npu(calls)
    state = {}

    def reset():
        state.clear()
        state.update(base=5, old=3, new=-1, output=99)

    def prepare():
        state["output"] = 0
        state["products"] = 11

    def solve():
        assert state["base"] == 5 and state["old"] == 3
        state["output"] = state["products"] + state["old"]
        state["new"] = 9

    def commit():
        state["base"] += state["old"]
        state["old"] = state["new"]

    def fused():
        solve()
        commit()

    def project():
        return state["output"] * 7 + 2

    operations = deferred_probe.deferred_operations(SimpleNamespace(npu=npu), prepare, fused, solve, commit, project)
    expected = None
    for operation in operations.values():
        reset()
        result = operation()
        if expected is None:
            expected = result, dict(state)
        assert (result, state) == expected
    assert state["base"] == 8 and state["old"] == 9


def test_request_trajectory_is_deterministic_unique_and_does_not_touch_global_rng(deferred_probe):
    before = random.getstate()
    first = [deferred_probe.request_record(100, live, step) for step, live in enumerate((98, 97, 99, 98) * 2)]
    second = [deferred_probe.request_record(100, live, step) for step, live in enumerate((98, 97, 99, 98) * 2)]
    assert first == second and random.getstate() == before
    for requests, counts in first:
        assert len(set(requests)) == len(requests) == len(counts)
        assert all(0 <= request < 100 for request in requests)
        assert all(1 <= count <= 4 for count in counts)
    assert set(request for requests, _ in first for request in requests) == set(range(100))
    assert len({tuple(requests) for requests, _ in first}) == 8


@pytest.mark.parametrize("batch,live,step", [(0, 0, 0), (129, 98, 0), (4, 5, 0), (4, 0, 0), (4, 4, -1)])
def test_bad_local_trajectory_contract_is_rejected(deferred_probe, batch, live, step):
    with pytest.raises(ValueError):
        deferred_probe.request_record(batch, live, step)
