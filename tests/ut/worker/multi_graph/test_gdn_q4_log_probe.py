# SPDX-License-Identifier: Apache-2.0
"""Stable request addressing for dense candidates versus compact base slots."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "q4_log_probe_cpu_test", directory / "multi_graph_gdn_q4_log_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_request_identity_and_each_accepted_prefix_select_the_same_base(probe):
    requests = [7, 0, 4, 2]
    dense = probe.candidate_slot_ids(requests)
    compact = probe.base_slot_ids(requests)
    assert dense == [29, 30, 31, 32, 1, 2, 3, 4, 17, 18, 19, 20, 9, 10, 11, 12]
    for row, request_id in enumerate(requests):
        for accepted in (1, 2, 3, 4):
            assert (dense[row * 4 + accepted - 1] - 1) // 4 == request_id
            assert compact[row * 4 + accepted - 1] == request_id + 1
    assert len(set(compact)) == len(requests)
    assert 0 not in dense and 0 not in compact


def test_null_capacity_is_not_assigned_to_absent_requests(probe):
    assert probe.candidate_slot_ids([]) == []
    assert probe.base_slot_ids([]) == []
    assert probe.base_slot_ids([2]) == [3, 3, 3, 3]


@pytest.mark.parametrize("batch", [7, 100])
def test_request_pipeline_preserves_global_operands_and_explicit_capture_dependencies(probe, batch):
    calls, events = [], []

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
            assert npu.active is stream
            calls.append(("record", stream.name, self.index))

    @contextmanager
    def use_stream(stream):
        previous, npu.active = npu.active, stream
        try:
            yield
        finally:
            npu.active = previous

    side = Stream("side")
    npu = SimpleNamespace(Stream=lambda: side, Event=Event, stream=use_stream, active=Stream("setup"))
    npu.current_stream = lambda: npu.active
    q_float, k_float, output = object(), object(), object()

    def prepare():
        calls.append(("prepare", npu.active.name))
        return q_float, k_float

    def products(q, k, start, stop):
        assert q is q_float and k is k_float
        calls.append(("products", npu.active.name, start, stop))

    def step(start, stop):
        calls.append(("step", npu.active.name, start, stop))

    operations = probe.request_pipeline_operations(SimpleNamespace(npu=npu), batch, prepare, products, step, output)
    assert calls == [] and len(events) == 3
    boundary = batch // 2
    npu.active = Stream("serial_capture")
    assert operations["compact_serial"]() is output
    assert calls == [
        ("prepare", "serial_capture"),
        ("products", "serial_capture", 0, boundary),
        ("step", "serial_capture", 0, boundary),
        ("products", "serial_capture", boundary, batch),
        ("step", "serial_capture", boundary, batch),
    ]
    for root_name in ("capture_one", "capture_two"):
        calls.clear()
        npu.active = Stream(root_name)
        assert operations["compact_pipeline"]() is output
        assert calls == [
            ("prepare", root_name),
            ("products", root_name, 0, boundary),
            ("record", root_name, 0),
            ("wait", "side", 0),
            ("step", "side", 0, boundary),
            ("products", root_name, boundary, batch),
            ("record", root_name, 1),
            ("wait", "side", 1),
            ("step", "side", boundary, batch),
            ("record", "side", 2),
            ("wait", root_name, 2),
        ]
        assert npu.active.name == root_name
        assert len(events) == 3
