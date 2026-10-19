# SPDX-License-Identifier: Apache-2.0
"""Chronological-trace suite isolation, scope and failure regressions."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def suite(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("stage7_suite_cpu", root / "aclgraph_fia_sink_stage7_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    for name in ("upstream", "whole"):
        (root / name).mkdir()
    probe = root / "probe.py"
    probe.write_text("# frozen probe\n")
    evidence = Path(__file__).resolve().parents[4] / "benchmarks/results/multi_graph_structural"
    return SimpleNamespace(
        root=root,
        output_dir=root / "formal",
        probe=probe,
        upstream_vendor=root / "upstream",
        whole_vendor=root / "whole",
        trace=evidence / "structural-dynamic-trace-v1.json",
        selection=evidence / "q4-tile-full-replay-v8-queue0.rank0.json",
        rounds=4,
        repeats=3,
        screen=False,
        timeout=10,
        seed=1703,
        model="config-only",
    )


def fake_runner(suite, *, fail=False, invalid_trial=False):
    commands, active = [], []

    class Process:
        def __init__(self, command, **kwargs):
            assert not active, "policy processes must never overlap"
            assert kwargs["start_new_session"] is True
            self.command, self.pid, self.done = command, 2000 + len(commands), False
            commands.append(command)
            active.append(self)

        def poll(self):
            return (2 if fail else 0) if self.done else None

        def wait(self, timeout):
            assert timeout == 70
            active.remove(self)
            self.done = True
            words = self.command[2:-1] if self.command[-1] == "--screen" else self.command[2:]
            options = dict(zip(words[::2], words[1::2], strict=True))
            output = Path(options["--output-dir"])
            output.mkdir()
            data = dict(
                state="failed" if fail else "completed",
                returncode=2 if fail else 0,
                policy=options["--policy"],
                trial=False if invalid_trial else int(options["--trial"]),
            )
            (output / suite.PROBE_REPORT_NAME).write_text(json.dumps(data))
            return data["returncode"]

    return Process, commands


def test_all_serial_blocks_require_complete_chronological_design(suite, args):
    runner, commands = fake_runner(suite)
    status = suite.run_suite(args, runner=runner)
    assert status["state"] == "completed" and status["full_trace_measured"] is True
    assert status["design"]["step_count"] == 128
    assert len(status["design"]["capacities"]) == 17
    assert status["design"]["repeats"] == 3
    assert status["design"]["step_order"] == "unchanged chronological canonical trace"
    assert len(commands) == 8
    assert [row["policy"] for row in status["runs"]] == [
        "upstream",
        "whole",
        "whole",
        "upstream",
        "upstream",
        "whole",
        "whole",
        "upstream",
    ]
    for index, row in enumerate(status["runs"]):
        data = (args.output_dir / row["report_path"]).read_bytes()
        assert row["report_sha256"] == suite.hashlib.sha256(data).hexdigest()
        if index:
            assert status["runs"][index - 1]["finished_unix"] <= row["launched_unix"]
    assert status["full_model_qualified"] is False and status["end_to_end_gain_established"] is False


def test_prefix_screen_never_claims_full_trace(suite, args):
    args.screen = True
    runner, commands = fake_runner(suite)
    status = suite.run_suite(args, runner=runner)
    assert len(commands) == 2
    assert status["design"]["step_count"] == 8
    assert status["design"]["occurrences"] == 1
    assert status["full_trace_measured"] is False


@pytest.mark.parametrize("option", ["fail", "invalid_trial"])
def test_failed_or_false_trial_report_stops_after_first_probe(suite, args, option):
    runner, commands = fake_runner(suite, **{option: True})
    with pytest.raises(RuntimeError):
        suite.run_suite(args, runner=runner)
    report = json.loads((args.output_dir / suite.REPORT_NAME).read_text())
    assert report["state"] == "failed" and len(commands) == 1
    assert report["full_trace_measured"] is False
    assert "report_sha256" in report["runs"][0]


def test_timeout_terminates_and_joins_owned_launcher(suite, tmp_path, monkeypatch):
    actions = []

    class Process:
        pid = 123
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout):
            actions.append(("wait", timeout))
            if not self.done:
                raise subprocess.TimeoutExpired("probe", timeout)
            return 0

    process = Process()

    def kill(pid, number):
        actions.append(("kill", pid, number))
        process.done = True

    monkeypatch.setattr(suite.os, "killpg", kill, raising=False)
    with pytest.raises(RuntimeError, match="deadline"):
        suite.run_one([], tmp_path / "probe.log", 10, {}, lambda: None, runner=lambda *a, **k: process)
    assert actions == [("wait", 70), ("kill", 123, suite.signal.SIGTERM), ("wait", 15)]


def test_existing_evidence_or_shared_vendor_is_rejected(suite, args):
    args.whole_vendor = args.upstream_vendor
    with pytest.raises(ValueError, match="distinct"):
        suite.run_suite(args)
    args.whole_vendor = args.root / "whole"
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        suite.run_suite(args)
