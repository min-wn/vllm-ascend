# SPDX-License-Identifier: Apache-2.0
"""Bounded consumer suite isolation, exact protocol and failure contracts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def suite(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("consumer_suite_cpu", root / "aclgraph_fia_sink_consumer_suite.py")
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
    probe.write_text("# frozen\n")
    evidence = Path(__file__).resolve().parents[4] / "benchmarks/results/multi_graph_structural"
    return SimpleNamespace(
        root=root,
        output_dir=root / "bounded",
        probe=probe,
        upstream_vendor=root / "upstream",
        whole_vendor=root / "whole",
        trace=evidence / "structural-dynamic-trace-v1.json",
        selection=evidence / "q4-tile-full-replay-v8-queue0.rank0.json",
        timeout=10,
        model="config-only",
    )


def fake_runner(suite, *, mutation=None):
    commands, active = [], []

    class Process:
        def __init__(self, command, **kwargs):
            assert not active
            assert kwargs["start_new_session"] is True
            self.command, self.pid, self.done = command, 3000 + len(commands), False
            commands.append(command)
            active.append(self)

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout):
            assert timeout == 70
            self.done = True
            active.remove(self)
            options = dict(zip(self.command[2::2], self.command[3::2], strict=True))
            directory = Path(options["--output-dir"])
            directory.mkdir()
            data = dict(state="completed", returncode=0, mode=options["--mode"], trial=int(options["--trial"]))
            if mutation == "failure":
                data.update(state="failed", returncode=2)
            elif mutation == "bool_trial":
                data["trial"] = False
            elif mutation == "wrong_mode":
                data["mode"] = "other"
            (directory / suite.PROBE_REPORT_NAME).write_text(json.dumps(data))
            return data["returncode"]

    return Process, commands


def test_exact_bounded_orders_copy_contract_and_scope(suite, args):
    runner, commands = fake_runner(suite)
    result = suite.run_suite(args, runner=runner)
    assert result["state"] == "completed" and result["performance_measured"] is True
    assert result["design"] == dict(
        modes=["upstream", "whole", "legacy"],
        capacities=[400, 512],
        steps=[0, 29],
        occurrences=16,
        rounds=2,
        repeats=10,
        warmups=5,
        profile_repeats=3,
        process_orders=[["upstream", "whole", "legacy"], ["legacy", "whole", "upstream"]],
        seed=1703,
        legacy_update_audit=(
            "pre/post qualification only; uninstrumented timing/profile; independent profile task counts"
        ),
        profile_label_scope="replay and final join; preparation completed before each label",
    )
    assert all(
        result[key] is False for key in ("full_trace_measured", "full_model_qualified", "end_to_end_gain_established")
    )
    assert len(commands) == 6
    for index, entry in enumerate(result["runs"]):
        options = dict(zip(commands[index][2::2], commands[index][3::2], strict=True))
        expected = args.whole_vendor if entry["mode"] == "whole" else args.upstream_vendor
        assert options["--opp-root"] == str(expected)
        assert options["--repeats"] == "10" and options["--profile-repeats"] == "3"
        path = args.output_dir / entry["report_path"]
        assert entry["report_sha256"] == suite.hashlib.sha256(path.read_bytes()).hexdigest()
        if index:
            assert result["runs"][index - 1]["finished_unix"] <= entry["launched_unix"]


@pytest.mark.parametrize("mutation", ["failure", "bool_trial", "wrong_mode"])
def test_incomplete_or_misattributed_probe_stops_before_next_process(suite, args, mutation):
    runner, commands = fake_runner(suite, mutation=mutation)
    with pytest.raises(RuntimeError):
        suite.run_suite(args, runner=runner)
    result = json.loads((args.output_dir / suite.REPORT_NAME).read_text())
    assert result["state"] == "failed" and result["performance_measured"] is False
    assert len(commands) == 1 and "report_sha256" in result["runs"][0]


def test_shared_vendor_and_existing_evidence_are_rejected(suite, args):
    args.whole_vendor = args.upstream_vendor
    with pytest.raises(ValueError, match="distinct"):
        suite.run_suite(args)
    args.whole_vendor = args.root / "whole"
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        suite.run_suite(args)


def test_original_first_records_cannot_be_replaced(suite):
    with pytest.raises(ValueError, match="original"):
        suite.design_for(dict(records=[dict(physical_tokens=400, step=1), dict(physical_tokens=512, step=29)]))
