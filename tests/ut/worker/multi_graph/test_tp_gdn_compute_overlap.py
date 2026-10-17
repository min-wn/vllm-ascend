"""CPU checks of real Cube/Vector pair orchestration and event dependencies."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional


@pytest.fixture
def compute_overlap(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_tp_gdn_compute_overlap.py"
    spec = importlib.util.spec_from_file_location("gdn_compute_overlap_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_execution():
    current, events = ["root"], []

    class Event:
        def __init__(self, name):
            self.name, self.producer = name, None

        def record(self, stream):
            assert current[0] == stream.name
            self.producer = stream.name
            events.append(("record", self.name, stream.name))

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert current[0] == self.name
            assert event.producer is not None, f"wait before producer: {event.name}"
            events.append(("wait", event.name, self.name))

    @contextmanager
    def stream(selected):
        previous, current[0] = current[0], selected.name
        events.append(("enter", selected.name))
        try:
            yield
        finally:
            events.append(("exit", selected.name))
            current[0] = previous

    return SimpleNamespace(
        root=Stream("root"),
        compute=Stream("compute"),
        ready=Event("ready"),
        done=Event("done"),
        torch_module=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        current=current,
        events=events,
    )


@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_compute_pair_preserves_real_math_state_and_activation_dependencies(compute_overlap, mode):
    runtime = make_execution()
    torch.manual_seed(551)
    rows, hidden, width = 6, 5, 4
    value, prior = (torch.randn(rows, hidden, dtype=torch.float64) for _ in range(2))
    residual = value + prior
    normalized = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + 1e-6)
    weight = torch.randn(width * 2, hidden, dtype=torch.float64)
    core_source = torch.randn(rows, 3, dtype=torch.float64)
    state = torch.randn_like(core_source)
    initial_state = state.clone()
    outputs, callbacks = {}, []

    def core():
        expected_stream = "compute" if mode == "overlap" else "root"
        assert runtime.current[0] == expected_stream
        if mode == "overlap":
            assert ("wait", "ready", "compute") in runtime.events
        state.add_(core_source)
        result = state.square()
        callbacks.append("core")
        outputs["core"] = result
        return result

    def gateup():
        assert runtime.current[0] == "root"
        if mode == "overlap":
            assert ("record", "done", "compute") in runtime.events
            assert ("wait", "done", "root") not in runtime.events
        result = functional.linear(normalized, weight)
        callbacks.append("gateup")
        outputs["gateup"] = result
        return result

    def activation(packed):
        assert runtime.current[0] == "root"
        assert packed is outputs["gateup"]
        if mode == "overlap":
            assert ("wait", "done", "root") in runtime.events
        gate, up = packed.chunk(2, dim=-1)
        result = functional.silu(gate) * up
        callbacks.append("activation")
        outputs["activation"] = result
        return result

    actual = compute_overlap.run_compute_pair(
        mode,
        run_core=core,
        run_gateup=gateup,
        run_activation=activation,
        root=runtime.root,
        compute=runtime.compute,
        ready=runtime.ready,
        done=runtime.done,
        torch_module=runtime.torch_module,
    )
    expected_packed = functional.linear(normalized, weight)
    gate, up = expected_packed.chunk(2, dim=-1)
    assert callbacks == ["core", "gateup", "activation"]
    assert all(actual[index] is outputs[name] for index, name in enumerate(("gateup", "activation", "core")))
    torch.testing.assert_close(actual[0], expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], functional.silu(gate) * up, rtol=0, atol=0)
    torch.testing.assert_close(actual[2], (initial_state + core_source).square(), rtol=0, atol=0)
    torch.testing.assert_close(state, initial_state + core_source, rtol=0, atol=0)
    torch.testing.assert_close(residual, value + prior, rtol=0, atol=0)
    assert runtime.current[0] == "root"
    if mode == "serial":
        assert runtime.events == []
    else:
        assert runtime.events == [
            ("record", "ready", "root"),
            ("enter", "compute"),
            ("wait", "ready", "compute"),
            ("record", "done", "compute"),
            ("exit", "compute"),
            ("wait", "done", "root"),
        ]


@pytest.mark.parametrize("mode", ["serial", "overlap"])
@pytest.mark.parametrize("failure", ["core", "gateup", "activation"])
def test_compute_pair_propagates_callback_failure_and_restores_stream(compute_overlap, mode, failure):
    runtime, calls = make_execution(), []

    def callback(name):
        def execute(*args):
            calls.append(name)
            if name == failure:
                raise RuntimeError(f"failed {name}")
            return torch.ones(2)

        return execute

    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        compute_overlap.run_compute_pair(
            mode,
            run_core=callback("core"),
            run_gateup=callback("gateup"),
            run_activation=callback("activation"),
            root=runtime.root,
            compute=runtime.compute,
            ready=runtime.ready,
            done=runtime.done,
            torch_module=runtime.torch_module,
        )
    order = ["core", "gateup", "activation"]
    assert calls == order[: order.index(failure) + 1]
    assert runtime.current[0] == "root"
    if failure == "core":
        assert runtime.done.producer is None


@pytest.mark.parametrize("mode", ["", "activation_early", "unknown", None])
def test_compute_pair_rejects_unknown_mode_before_any_work(compute_overlap, mode):
    runtime = make_execution()

    def unexpected(*args):
        raise AssertionError("invalid mode started device work")

    with pytest.raises(ValueError):
        compute_overlap.run_compute_pair(
            mode,
            run_core=unexpected,
            run_gateup=unexpected,
            run_activation=unexpected,
            root=runtime.root,
            compute=runtime.compute,
            ready=runtime.ready,
            done=runtime.done,
            torch_module=runtime.torch_module,
        )
    assert runtime.events == []
