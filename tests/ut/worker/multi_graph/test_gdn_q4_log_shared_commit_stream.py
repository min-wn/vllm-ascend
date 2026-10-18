# SPDX-License-Identifier: Apache-2.0
"""Bound full-model commit streams while retaining per-lane dependencies."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def contracts(monkeypatch):
    path = Path(__file__).with_name("test_gdn_q4_log_backend.py")
    spec = importlib.util.spec_from_file_location("q4_shared_commit_fixtures", path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers, helpers.log_backend.__wrapped__(monkeypatch)


class TrackingNPU:
    def __init__(self):
        self.timeline, self.streams, self.events = [], [], []
        self.current = SimpleNamespace(name="root", wait_event=self._root_wait)

    def _root_wait(self, event):
        self.timeline.append(("wait", "root", event.index))

    def Stream(self):
        name = f"side{len(self.streams)}"
        stream = SimpleNamespace(name=name, wait_event=lambda event: self.timeline.append(("wait", name, event.index)))
        self.streams.append(stream)
        return stream

    def Event(self, **kwargs):
        index = len(self.events)
        event = SimpleNamespace(
            index=index,
            record=lambda stream: self.timeline.append(("record", stream.name, index)),
        )
        self.events.append(event)
        return event

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def synchronize(self):
        self.timeline.append(("synchronize",))


def invocation(fixture, capacity):
    _, _, _, _, args, kwargs, _ = fixture
    # Each call owns fresh metadata, as independent captured capacity lanes do.
    options = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
    if capacity == 1:
        args = (*[value[:4] for value in args[:3]], args[3])
        options.update(
            actual_seq_lengths=options["actual_seq_lengths"][:2],
            ssm_state_indices=options["ssm_state_indices"][:4],
            num_accepted_tokens=options["num_accepted_tokens"][:1],
            g=options["g"][:4],
            beta=options["beta"][:4],
        )
    return args, options


@pytest.mark.parametrize("shared", [False, True])
def test_exclusive_stream_override_avoids_all_per_lane_stream_allocations(contracts, shared):
    helpers, module = contracts
    fixture = helpers.fixture_backend(module, "deferred_overlap", deferred=True)
    backend, state, _, _, _, _, api = fixture
    npu = api.npu = TrackingNPU()
    override = npu.Stream() if shared else None
    assert backend.reserve_exclusive_capacity(2, commit_stream=override) is backend
    before = state.snapshot()
    for capacity in (2, 1, 2):
        state.restore(before)
        args, options = invocation(fixture, capacity)
        backend(*args, **options)
        backend.launch_pending_commit()
        backend.finish_pending_commit()
    lanes = tuple(backend.buffers.values())
    assert len(lanes) == 3 and len(npu.events) == 6
    if shared:
        assert len(npu.streams) == 1 and all(lane["commit_stream"] is override for lane in lanes)
    else:
        assert len(npu.streams) == 3
        assert len({id(lane["commit_stream"]) for lane in lanes}) == 3
    assert len({id(lane["solve_ready"]) for lane in lanes}) == 3
    assert len({id(lane["commit_done"]) for lane in lanes}) == 3
    assert backend.describe()["exclusive_commit_stream_shared"] is shared


def test_shared_commit_stream_keeps_two_layer_multi_capacity_state_and_dependency_order(contracts, monkeypatch):
    helpers, module = contracts
    fixtures = [helpers.fixture_backend(module, "deferred_overlap", deferred=True, direct=True) for _ in range(2)]
    api = fixtures[0][6]
    npu = api.npu = TrackingNPU()
    shared = npu.Stream()
    registry = module.CompactSSMRegistry()
    projections = [torch.nn.Identity(), torch.nn.Identity()]
    for index, (fixture, projection) in enumerate(zip(fixtures, projections, strict=True)):
        backend, state = fixture[:2]
        state.torch = api
        backend.reserve_exclusive_capacity(2, commit_stream=shared)
        registry.register(state.handle, backend, projection, name=f"layer{index}")
    helpers.install_fixture_modules(module, monkeypatch, fixtures[0][0])
    initial = [fixture[1].snapshot() for fixture in fixtures]
    with module.installed_log_registry(registry):
        for capacity in (2, 1, 2):
            for fixture, saved in zip(fixtures, initial, strict=True):
                fixture[1].restore(saved)
            npu.timeline.clear()
            lanes = []
            for fixture, projection, saved in zip(fixtures, projections, initial, strict=True):
                backend, state, _, _, _, _, _ = fixture
                args, options = invocation(fixture, capacity)
                output = registry(*args, **options)
                # Solving attention does not publish accepted state.
                assert all(
                    torch.equal(value, old) for value, old in zip(state.mutable_tensors, saved.tensors, strict=True)
                )
                projection(output)
                lanes.append(next(reversed(backend.buffers.values())))
                npu.timeline.append(("next_layer",))
            assert registry.finish_pending_commits() == 2
            first, second = lanes
            assert npu.timeline == [
                ("record", "root", first["solve_ready"].index),
                ("wait", "side0", first["solve_ready"].index),
                ("record", "side0", first["commit_done"].index),
                ("next_layer",),
                ("record", "root", second["solve_ready"].index),
                ("wait", "side0", second["solve_ready"].index),
                ("record", "side0", second["commit_done"].index),
                ("next_layer",),
                ("wait", "root", first["commit_done"].index),
                ("wait", "root", second["commit_done"].index),
            ]
            for fixture, saved in zip(fixtures, initial, strict=True):
                backend, state = fixture[:2]
                assert backend._pending_commit is None
                for owner in range(1, 3):
                    live = owner <= capacity
                    assert torch.equal(state.base[owner], saved.tensors[0][owner] + int(live))
                    assert torch.equal(state.logs[owner], saved.tensors[1][owner] + 2 * int(live))
                    assert state.valid[owner].eq(int(live)).all()
                assert torch.equal(state.base[0], saved.tensors[0][0])
                assert not state.logs[0].count_nonzero() and not state.valid[0].count_nonzero()
    assert len(npu.streams) == 1 and len(npu.events) == 12
    assert len({id(event) for event in npu.events}) == 12
    assert all(lane["commit_stream"] is shared for fixture in fixtures for lane in fixture[0].buffers.values())


@pytest.mark.parametrize("late", ["reserved", "executed", "failed"])
def test_commit_stream_override_cannot_be_added_after_exclusive_scope_initialization(contracts, late):
    helpers, module = contracts
    fixture = helpers.fixture_backend(module, "deferred_overlap", deferred=True)
    backend, _, _, _, args, options, api = fixture
    npu = api.npu = TrackingNPU()
    first, replacement = npu.Stream(), npu.Stream()
    if late == "reserved":
        backend.reserve_exclusive_capacity(2, commit_stream=first)
    elif late == "executed":
        backend(*args, **options)
        backend.launch_pending_commit()
        backend.finish_pending_commit()
    else:
        backend._failed = True
    with pytest.raises(RuntimeError):
        backend.reserve_exclusive_capacity(2, commit_stream=replacement)
    assert backend._exclusive_commit_stream is (first if late == "reserved" else None)
