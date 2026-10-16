import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


def test_stream_dependencies_and_event_ownership(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append((self.name, "wait", event))

    class Event:
        def record(self, stream):
            calls.append((stream.name, "record", self))

        def synchronize(self):
            calls.append(("host", "wait", self))

        def query(self):
            return True

    parent, compute, update = (Stream(name) for name in ("parent", "compute", "update"))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=Event, current_stream=lambda: parent, stream=lambda _: nullcontext()),
        raising=False,
    )
    module = types.ModuleType("vllm_ascend.compilation.updatable_graph")
    module.ContextSource = lambda metadata: metadata
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/multi_graph/npu_backend.py"
    spec = importlib.util.spec_from_file_location("npu_backend_under_test", path)
    backend_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, backend_module)
    spec.loader.exec_module(backend_module)
    graph = SimpleNamespace(
        replay=lambda: calls.append(("graph", "replay", None)),
        update=lambda stream, tasks: calls.append((stream.name, "update", tasks)),
        resolve_tasks=lambda metadata: metadata,
    )
    entry = SimpleNamespace(
        graph=graph, compute_stream=compute, update_stream=update, buffers=SimpleNamespace(metadata={"layer": 1})
    )
    backend = backend_module.NPUExecutionBackend()
    prepared = backend.preparation_complete()
    tasks = backend.resolve(entry)
    backend.replay(entry, prepared)
    backend.update(entry, tasks, prepared)
    completion = backend.finish(entry)
    backend.join(completion)
    backend.wait(completion)
    consumed = backend.output_consumed()
    backend.wait(consumed)
    assert backend.query(completion)
    assert backend.query(consumed)
    assert calls == [
        ("parent", "record", prepared),
        ("compute", "wait", prepared),
        ("graph", "replay", None),
        ("update", "wait", prepared),
        ("update", "update", tasks),
        ("update", "record", completion.update_done),
        ("compute", "wait", completion.update_done),
        ("compute", "record", completion.compute_done),
        ("parent", "wait", completion.compute_done),
        ("host", "wait", completion.compute_done),
        ("parent", "record", consumed),
        ("host", "wait", consumed),
    ]
    calls.clear()
    second = SimpleNamespace(graph=graph, update_stream=Stream("update1"))
    backend.update_interleaved((entry, second), (("a0", "a1"), ("b0",)), prepared)
    assert [call for call in calls if call[1] == "update"] == [
        ("update", "update", ("a0",)),
        ("update1", "update", ("b0",)),
        ("update", "update", ("a1",)),
    ]
