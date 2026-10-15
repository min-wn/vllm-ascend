from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.compilation.updatable_graph import ContextSource
from vllm_ascend.worker.multi_graph.manager import MultiGraphEntry


@dataclass(frozen=True)
class LaneCompletion:
    update_done: Any
    compute_done: Any


class NPUExecutionBackend:
    def resolve(self, entry: MultiGraphEntry):
        return entry.graph.resolve_tasks(ContextSource(entry.buffers.metadata))

    def preparation_complete(self):
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        return event

    def replay(self, entry: MultiGraphEntry, prepared: Any) -> None:
        entry.compute_stream.wait_event(prepared)
        with torch.npu.stream(entry.compute_stream):
            entry.graph.replay()

    def update(self, entry: MultiGraphEntry, tasks: Any, prepared: Any) -> None:
        entry.update_stream.wait_event(prepared)
        entry.graph.update(entry.update_stream, tasks)

    def finish(self, entry: MultiGraphEntry):
        update_done = torch.npu.Event()
        update_done.record(entry.update_stream)
        entry.compute_stream.wait_event(update_done)
        event = torch.npu.Event()
        event.record(entry.compute_stream)
        return LaneCompletion(update_done, event)

    def wait(self, event: Any) -> None:
        (event.compute_done if isinstance(event, LaneCompletion) else event).synchronize()

    def join(self, event: Any) -> None:
        torch.npu.current_stream().wait_event(event.compute_done if isinstance(event, LaneCompletion) else event)

    def output_consumed(self):
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        return event

    def query(self, event: Any) -> bool:
        return (event.compute_done if isinstance(event, LaneCompletion) else event).query()
