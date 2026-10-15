from dataclasses import replace

import pytest
import torch

from tests.ut.worker.multi_graph.test_metadata import DecodeMetadata
from vllm_ascend.worker.multi_graph.buffers import LaneBuffers
from vllm_ascend.worker.multi_graph.manager import ManagerState, MultiGraphEntry, MultiGraphManager
from vllm_ascend.worker.multi_graph.types import ExecutionPlan, GraphKey, SlicePlan


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.fail_at = None

    def call(self, name, lane=None):
        self.calls.append((name, lane))
        if self.fail_at == (name, lane):
            raise RuntimeError("injected failure")

    def resolve(self, entry):
        self.call("resolve", entry.key.lane_id)
        return entry.key

    def preparation_complete(self):
        self.call("prepare")
        return "prepared"

    def replay(self, entry, prepared):
        assert prepared == "prepared"
        self.call("replay", entry.key.lane_id)
        entry.output[:, 0].copy_(entry.buffers.input_ids * 2 + entry.buffers.positions)

    def update(self, entry, tasks, prepared):
        assert tasks == entry.key
        self.call("update", entry.key.lane_id)

    def finish(self, entry):
        self.call("finish", entry.key.lane_id)
        return entry.key.lane_id

    def wait(self, event):
        self.call("wait", event)

    def join(self, event):
        self.call("join", event)

    def output_consumed(self):
        self.call("consumed")
        return "consumed"


@pytest.fixture
def execution(signature):
    backend = FakeBackend()
    manager = MultiGraphManager(backend, 2)
    inputs = torch.arange(4)
    source = DecodeMetadata(
        seq_lens_list=[2, 3, 4, 5],
        actual_seq_lengths_q=[1, 2, 3, 4],
        slot_mapping=torch.arange(4),
        block_tables=torch.arange(8).reshape(4, 2),
    )
    metadata = {"layer": source}
    parts = []
    for lane in (0, 1):
        key = GraphKey(lane, 2, 2, signature)
        part = SlicePlan(lane * 2, lane * 2 + 2, lane * 2, lane * 2 + 2, key)
        entry = MultiGraphEntry(
            key,
            LaneBuffers.create(inputs, inputs, metadata, part),
            object(),
            object(),
            object(),
            object(),
            torch.empty((2, 1)),
        )
        manager.add_entry(entry)
        parts.append(part)
    manager.ready()
    return manager, backend, ExecutionPlan(tuple(parts), "parallel", "force"), inputs, metadata


def test_preflight_and_resolve_all_entries_before_replay(execution):
    manager, backend, plan, inputs, metadata = execution
    output = manager.run(plan, inputs, inputs + 10, metadata)
    assert output[:, 0].tolist() == [10, 13, 16, 19]
    assert backend.calls[:3] == [("resolve", 0), ("resolve", 1), ("prepare", None)]
    assert backend.calls.index(("replay", 0)) < backend.calls.index(("update", 0))
    assert manager.state == ManagerState.IN_FLIGHT


def test_resolve_failure_never_launches_any_graph(execution):
    manager, backend, plan, inputs, metadata = execution
    backend.fail_at = ("resolve", 1)
    with pytest.raises(RuntimeError, match="injected"):
        manager.run(plan, inputs, inputs, metadata)
    assert not any(name == "replay" for name, _ in backend.calls)
    assert manager.state == ManagerState.READY


@pytest.mark.parametrize("failure", [("replay", 0), ("update", 0), ("replay", 1), ("update", 1), ("join", 1)])
def test_submitted_failure_poisoned_without_eager_retry(execution, failure):
    manager, backend, plan, inputs, metadata = execution
    backend.fail_at = failure
    with pytest.raises(RuntimeError, match="injected"):
        manager.run(plan, inputs, inputs, metadata)
    assert manager.state == ManagerState.FAILED
    calls = list(backend.calls)
    with pytest.raises(RuntimeError, match="not usable"):
        manager.run(plan, inputs, inputs, metadata)
    assert backend.calls == calls
    with pytest.raises(RuntimeError, match="worker teardown"):
        manager.close()


def test_cross_step_fence_and_output_ownership(execution):
    manager, backend, plan, inputs, metadata = execution
    first = manager.run(plan, inputs, inputs, metadata)
    backend.calls.clear()
    second = manager.run(plan, inputs + 100, inputs, metadata)
    assert backend.calls[0] == ("wait", "consumed")
    assert first[:, 0].tolist() == [0, 3, 6, 9]
    assert second[:, 0].tolist() == [200, 203, 206, 209]


def test_serial_mode_waits_after_update_before_second_replay(execution):
    manager, backend, plan, inputs, metadata = execution
    manager.run(replace(plan, execution="serial"), inputs, inputs, metadata)
    assert backend.calls.index(("update", 0)) < backend.calls.index(("wait", 0))
    assert backend.calls.index(("wait", 0)) < backend.calls.index(("replay", 1))


def test_missing_second_entry_causes_no_buffer_mutation_or_launch(execution):
    manager, backend, plan, inputs, metadata = execution
    manager.entries.pop(plan.slices[1].graph_key)
    first = manager.entries[plan.slices[0].graph_key]
    with pytest.raises(KeyError):
        manager.run(plan, inputs + 100, inputs, metadata)
    assert first.buffers.input_ids.tolist() == [0, 1]
    assert backend.calls == []


def test_invalidation_drains_then_rejects_stale_execution(execution):
    manager, backend, plan, inputs, metadata = execution
    manager.run(plan, inputs, inputs, metadata)
    manager.invalidate()
    assert manager.state == ManagerState.INVALID
    assert not manager.entries
    with pytest.raises(RuntimeError, match="invalid"):
        manager.run(plan, inputs, inputs, metadata)


@pytest.mark.parametrize("shared", ["graph", "graph_pool", "compute_stream", "update_stream", "output"])
def test_registration_rejects_shared_resources(execution, shared):
    original, backend, _, _, _ = execution
    first, second = original.entries.values()
    manager = MultiGraphManager(backend, 2)
    manager.add_entry(first)
    with pytest.raises(ValueError):
        manager.add_entry(replace(second, **{shared: getattr(first, shared)}))


def test_startup_failure_is_terminal_without_waiting_for_events():
    backend = FakeBackend()
    manager = MultiGraphManager(backend, 2)
    manager.startup_failed()
    assert manager.state == ManagerState.FAILED
    with pytest.raises(RuntimeError, match="not usable"):
        manager.drain()
    assert backend.calls == []


@pytest.mark.parametrize("failure", [("finish", 0), ("consumed", None), ("wait", "consumed")])
def test_completion_failures_retain_resources_for_teardown(execution, failure):
    manager, backend, plan, inputs, metadata = execution
    if failure == ("wait", "consumed"):
        manager.run(plan, inputs, inputs, metadata)
    backend.fail_at = failure
    with pytest.raises(RuntimeError, match="injected"):
        manager.run(plan, inputs, inputs, metadata)
    assert manager.state == ManagerState.FAILED
    assert manager.pending_resources is not None
