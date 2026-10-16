from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch

from vllm_ascend.worker.multi_graph.buffers import LaneBuffers, assert_disjoint
from vllm_ascend.worker.multi_graph.types import ExecutionPlan, GraphKey


class ManagerState(StrEnum):
    NEW = "new"
    READY = "ready"
    IN_FLIGHT = "in_flight"
    FAILED = "failed"
    INVALID = "invalid"
    CLOSED = "closed"


class ExecutionBackend(Protocol):
    def resolve(self, entry: MultiGraphEntry) -> Any: ...
    def preparation_complete(self) -> Any: ...
    def replay(self, entry: MultiGraphEntry, prepared: Any) -> None: ...
    def update(self, entry: MultiGraphEntry, tasks: Any, prepared: Any) -> None: ...
    def update_interleaved(self, entries: Any, tasks: Any, prepared: Any) -> None: ...
    def finish(self, entry: MultiGraphEntry) -> Any: ...
    def wait(self, event: Any) -> None: ...
    def join(self, event: Any) -> None: ...
    def output_consumed(self) -> Any: ...
    def query(self, event: Any) -> bool: ...


@dataclass
class MultiGraphEntry:
    key: GraphKey
    buffers: LaneBuffers
    graph: Any
    graph_pool: Any
    compute_stream: Any
    update_stream: Any
    output: torch.Tensor

    def writable_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.buffers.writable_tensors(), self.output)


class MultiGraphManager:
    """Own entries, preflight all lanes, and poison the executor after submission failures."""

    def __init__(self, backend: ExecutionBackend, max_entries: int):
        if max_entries < 2:
            raise ValueError("multi-graph requires an entry budget of at least two")
        self.backend = backend
        self.max_entries = max_entries
        self.entries: dict[GraphKey, MultiGraphEntry] = {}
        self.state = ManagerState.NEW
        self.completion = None
        self.pending_resources = None
        self.submission_order = "lane"

    def add_entry(self, entry: MultiGraphEntry) -> None:
        if self.state != ManagerState.NEW:
            raise RuntimeError("entries can only be registered during startup")
        if entry.key in self.entries or len(self.entries) >= self.max_entries:
            raise ValueError("duplicate entry or graph entry budget exceeded")
        if entry.output.shape[0] != entry.key.graph_tokens:
            raise ValueError("captured output shape does not match graph key")
        for existing in self.entries.values():
            if existing.graph is entry.graph or existing.graph_pool == entry.graph_pool:
                raise ValueError("each entry requires an independent graph and pool")
            if existing.update_stream is entry.update_stream and existing.key.lane_id != entry.key.lane_id:
                raise ValueError("lanes require independent update streams")
            if existing.compute_stream is entry.compute_stream and existing.key.lane_id != entry.key.lane_id:
                raise ValueError("lanes require independent compute streams")
            assert_disjoint(existing.writable_tensors(), entry.writable_tensors())
        self.entries[entry.key] = entry

    def ready(self) -> None:
        if self.state != ManagerState.NEW or {key.lane_id for key in self.entries} != {0, 1}:
            raise RuntimeError("startup must capture both lanes before entering READY")
        self.state = ManagerState.READY

    def startup_failed(self) -> None:
        if self.state != ManagerState.NEW:
            raise RuntimeError("startup failure can only be recorded during capture")
        self.state = ManagerState.FAILED

    def drain(self) -> None:
        if self.state == ManagerState.IN_FLIGHT:
            try:
                self.backend.wait(self.completion)
            except Exception:
                self.state = ManagerState.FAILED
                raise
            self.completion = None
            self.pending_resources = None
            self.state = ManagerState.READY
        if self.state != ManagerState.READY:
            raise RuntimeError(f"multi-graph executor is not usable: {self.state}")

    def poll(self) -> bool:
        """Release in-flight resources only after the output-consumption event."""
        if self.state == ManagerState.IN_FLIGHT:
            try:
                if not self.backend.query(self.completion):
                    return False
            except Exception:
                self.state = ManagerState.FAILED
                raise
            self.completion = None
            self.pending_resources = None
            self.state = ManagerState.READY
        if self.state != ManagerState.READY:
            raise RuntimeError(f"multi-graph executor is not usable: {self.state}")
        return True

    def run(self, plan: ExecutionPlan, input_ids: torch.Tensor, positions: torch.Tensor, metadata: dict[str, Any]):
        if self.submission_order not in ("lane", "replay_first", "interleaved"):
            raise ValueError("unknown graph submission order")
        if plan.execution != "parallel" and self.submission_order != "lane":
            raise ValueError("experimental submission order requires parallel execution")
        self.drain()
        entries = tuple(self.entries[part.graph_key] for part in plan.slices)
        updates = tuple(
            entry.buffers.preflight(input_ids, positions, metadata, part) for entry, part in zip(entries, plan.slices)
        )
        for entry, part, values in zip(entries, plan.slices, updates):
            entry.buffers.refresh(input_ids, positions, part, values)
        tasks = tuple(self.backend.resolve(entry) for entry in entries)
        prepared = self.backend.preparation_complete()
        events = []
        self.pending_resources = (updates, tasks, prepared, events)
        self.state = ManagerState.IN_FLIGHT
        try:
            if self.submission_order != "lane":
                for entry in entries:
                    self.backend.replay(entry, prepared)
                if self.submission_order == "interleaved":
                    self.backend.update_interleaved(entries, tasks, prepared)
            for entry, resolved in zip(entries, tasks):
                if self.submission_order == "lane":
                    self.backend.replay(entry, prepared)
                if self.submission_order != "interleaved":
                    self.backend.update(entry, resolved, prepared)
                event = self.backend.finish(entry)
                events.append(event)
                if plan.execution == "serial":
                    self.backend.wait(event)
                self.backend.join(event)
            output = torch.cat([entry.output for entry in entries], dim=0)
            self.completion = self.backend.output_consumed()
            return output
        except Exception:
            self.state = ManagerState.FAILED
            raise

    def invalidate(self) -> None:
        self.drain()
        self.state = ManagerState.INVALID
        self.entries.clear()

    def close(self) -> None:
        if self.state == ManagerState.IN_FLIGHT:
            self.drain()
        if self.state == ManagerState.FAILED:
            raise RuntimeError("failed graphs require worker teardown; do not wait on unsignalled external events")
        self.entries.clear()
        self.state = ManagerState.CLOSED
