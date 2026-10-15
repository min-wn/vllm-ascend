from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.worker.multi_graph.buffers import assert_disjoint
from vllm_ascend.worker.multi_graph.manager import ManagerState, MultiGraphManager
from vllm_ascend.worker.multi_graph.types import ExecutionPlan


@dataclass
class AsyncSlot:
    manager: MultiGraphManager
    generation: int = 0


class AsyncGraphExecutor:
    """Bounded independent capture sets with event-based nonblocking reuse.

    Return None only before submission when every set remains occupied.
    Completion events include the output merge, protecting buffers and host
    task resources. Caller forward/consumer operations must use one parent
    stream; arbitrary cross-stream consumers need their own dependency events.
    """

    def __init__(self, managers: tuple[MultiGraphManager, ...]):
        if len(managers) < 2 or len({id(manager) for manager in managers}) != len(managers):
            raise ValueError("async execution requires at least two independent capture sets")
        for manager in managers:
            if manager.state != ManagerState.READY:
                raise ValueError("async capture sets must be READY")
        expected = set(managers[0].entries)
        for index, manager in enumerate(managers):
            if set(manager.entries) != expected:
                raise ValueError("capture sets must contain the same keys")
            for previous in managers[:index]:
                for entry in manager.entries.values():
                    for existing in previous.entries.values():
                        if entry.graph is existing.graph or entry.graph_pool == existing.graph_pool:
                            raise ValueError("async capture sets require independent graphs and pools")
                        if (
                            entry.compute_stream is existing.compute_stream
                            or entry.update_stream is existing.update_stream
                        ):
                            raise ValueError("async capture sets require independent streams")
                        assert_disjoint(entry.writable_tensors(), existing.writable_tensors())
        self.slots = tuple(AsyncSlot(manager) for manager in managers)
        self.cursor = 0
        self.failed = False

    def try_run(
        self, plan: ExecutionPlan, input_ids: torch.Tensor, positions: torch.Tensor, metadata: dict[str, Any]
    ) -> torch.Tensor | None:
        if self.failed:
            raise RuntimeError("async executor is failed")
        for offset in range(len(self.slots)):
            index = (self.cursor + offset) % len(self.slots)
            slot = self.slots[index]
            try:
                if not slot.manager.poll():
                    continue
                result = slot.manager.run(plan, input_ids, positions, metadata)
            except Exception:
                self.failed = True
                raise
            slot.generation += 1
            self.cursor = (index + 1) % len(self.slots)
            return result
        return None

    def close(self) -> None:
        if self.failed:
            raise RuntimeError("failed async executor requires worker teardown")
        for slot in self.slots:
            slot.manager.close()
