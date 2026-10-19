# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for serial policy isolation and preserved raw reports."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def suite():
    path = Path(__file__).resolve().parents[4] / "benchmarks/aclgraph_fia_sink_stage6_suite.py"
    spec = importlib.util.spec_from_file_location("fia_stage6_suite_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    for policy in ("upstream", "whole"):
        (root / policy).mkdir()
    probe = root / "probe.py"
    probe.write_text("# frozen probe\n")
    return SimpleNamespace(
        root=root,
        output_dir=root / "formal",
        probe=probe,
        upstream_vendor=root / "upstream",
        whole_vendor=root / "whole",
        trace=root / "trace.json",
        selection=root / "selection.json",
        rounds=4,
        repeats=5,
        screen=False,
        timeout=10,
        seed=1703,
        model="config-only-model",
    )


def fake_runner(suite, *, fail_at=None, wrong_identity=False):
    active = []
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            assert not active, "policy processes may never overlap"
            assert kwargs["start_new_session"] is True
            self.command, self.pid = command, 1000 + len(commands)
            commands.append(command)
            active.append(self)

        def wait(self, timeout):
            assert timeout == 40
            active.remove(self)
            options = dict(zip(self.command[2::2], self.command[3::2], strict=True))
            output = Path(options["--output-dir"])
            output.mkdir()
            code = 2 if fail_at == len(commands) - 1 else 0
            report = dict(
                state="completed" if not code else "failed",
                returncode=code,
                policy="wrong" if wrong_identity else options["--policy"],
                trial=int(options["--trial"]),
            )
            (output / suite.PROBE_REPORT_NAME).write_text(json.dumps(report))
            return code

    return Process, commands


def test_alternating_serial_runs_preserve_exact_raw_file_hashes(suite, args):
    runner, commands = fake_runner(suite)
    status = suite.run_suite(args, runner=runner)
    assert status["state"] == "completed"
    assert len(commands) == 8
    assert [item["policy"] for item in status["runs"]] == [
        "upstream",
        "whole",
        "whole",
        "upstream",
        "upstream",
        "whole",
        "whole",
        "upstream",
    ]
    for index, item in enumerate(status["runs"]):
        path = args.output_dir / item["report_path"]
        assert path.name == "stage6-producer-probe.json"
        assert suite.hashlib.sha256(path.read_bytes()).hexdigest() == item["report_sha256"]
        assert item["process_order_index"] == index
        if index:
            assert status["runs"][index - 1]["finished_unix"] <= item["launched_unix"]
    assert status["full_trace_measured"] is False
    assert status["end_to_end_gain_established"] is False


def test_first_failed_probe_stops_suite_and_retains_evidence(suite, args):
    runner, commands = fake_runner(suite, fail_at=0)
    with pytest.raises(RuntimeError, match="probe failed"):
        suite.run_suite(args, runner=runner)
    report = json.loads((args.output_dir / suite.REPORT_NAME).read_text())
    assert report["state"] == "failed" and len(commands) == 1
    assert report["runs"][0]["returncode"] == 2
    assert "report_sha256" in report["runs"][0]


def test_successful_exit_with_wrong_policy_cannot_continue(suite, args):
    runner, commands = fake_runner(suite, wrong_identity=True)
    with pytest.raises(RuntimeError, match="policy/trial"):
        suite.run_suite(args, runner=runner)
    assert len(commands) == 1


def test_same_vendor_and_existing_evidence_are_rejected(suite, args):
    args.whole_vendor = args.upstream_vendor
    with pytest.raises(ValueError, match="distinct"):
        suite.run_suite(args)
    args.whole_vendor = args.root / "whole"
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        suite.run_suite(args)


@pytest.mark.parametrize("rounds", [True, 0, 9, 2.5])
def test_invalid_rounds_are_rejected(suite, rounds):
    with pytest.raises(ValueError, match="exact integer"):
        suite.process_orders(rounds)
