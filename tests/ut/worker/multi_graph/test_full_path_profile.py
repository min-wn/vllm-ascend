# SPDX-License-Identifier: Apache-2.0
"""Diagnostic replay keeps intervening steps and preparation inside profiles."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[4] / "benchmarks/aclgraph_full_path_profile.py"
    spec = importlib.util.spec_from_file_location("full_path_profile", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("count", [0, 1, True, 2.5])
def test_invalid_profile_count(count):
    with pytest.raises(ValueError):
        module().profile_indices(count)


def test_two_steps_do_not_profile_the_last_step_twice():
    assert module().profile_indices(2) == [0, 1]


def test_profile_replays_preserves_continuity_and_real_update(monkeypatch, tmp_path):
    events = []

    @contextmanager
    def record(name):
        events.append(("enter", name))
        yield
        events.append(("exit", name))

    @contextmanager
    def profile(**kwargs):
        events.append("profile_start")
        yield SimpleNamespace(export_chrome_trace=lambda path: Path(path).write_text("{}"))
        events.append("profile_end")

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            profiler=SimpleNamespace(profile=profile, ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"))
        ),
    )
    torch = SimpleNamespace(
        profiler=SimpleNamespace(record_function=record), npu=SimpleNamespace(synchronize=lambda: events.append("sync"))
    )
    dist = SimpleNamespace(get_rank=lambda: 0, barrier=lambda **kwargs: events.append("barrier"))
    registry = SimpleNamespace(
        prepare_step=lambda *args: events.append(("prepare", args[0])),
        finish_step=lambda step: events.append(("finish", step)),
    )
    steps = [dict(step=i, request_ids=[i], accepted_from_previous=[1], batch=1) for i in range(6)]
    records = module().profile_replays(
        SimpleNamespace(output=tmp_path / "result.json"),
        torch,
        dist,
        registry,
        steps,
        {"dispatched_tokens": [4] * 6},
        {("deferred_tile_ready", 4): "graph"},
        lambda step, cap: step["step"],
        lambda entry, frame: events.append(("replay", frame)),
        restore=lambda: events.append("restore"),
        cpu_group="gloo",
    )
    assert events[0] == "restore" and events.count("restore") == 1
    assert [e for e in events if isinstance(e, tuple) and e[0] == "replay"] == [("replay", i) for i in range(6)]
    assert [r["step"] for r in records] == [0, 1, 5]
    assert events.count("sync") == 6
    for i in range(6):
        assert events.index(("prepare", i)) < events.index(("replay", i)) < events.index(("finish", i))
    assert events.index("profile_start") < events.index(("prepare", 0))
    assert events.index(("finish", 0)) < events.index("profile_end")
