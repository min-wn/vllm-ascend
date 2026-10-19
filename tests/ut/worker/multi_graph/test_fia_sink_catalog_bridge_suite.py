# SPDX-License-Identifier: Apache-2.0
"""Catalog-context suite order, identity, ownership and failure contracts."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def suite(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "catalog_bridge_suite_cpu", root / "aclgraph_fia_sink_catalog_bridge_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    for name in ("upstream", "whole"):
        (root / name).mkdir()
    probe = root / "bridge.py"
    probe.write_text("# frozen bridge\n")
    evidence = Path(__file__).resolve().parents[4] / "benchmarks/results/multi_graph_structural"
    return SimpleNamespace(
        root=root,
        output_dir=root / "bridge-output",
        probe=probe,
        upstream_vendor=root / "upstream",
        whole_vendor=root / "whole",
        trace=evidence / "structural-dynamic-trace-v1.json",
        selection=evidence / "q4-tile-full-replay-v8-queue0.rank0.json",
        timeout=10,
        model="configuration-only",
    )


def runner_for(suite, *, mutation=None, missing=False, exitcode=0):
    commands, active = [], []

    class Process:
        def __init__(self, command, **kwargs):
            assert not active, "fresh-process cells must never overlap"
            assert kwargs["start_new_session"] is True
            self.command, self.pid, self.done = command, 4500 + len(commands), False
            commands.append(command)
            active.append(self)

        def poll(self):
            return exitcode if self.done else None

        def wait(self, timeout):
            assert timeout == 70
            assert not self.done
            self.done = True
            active.remove(self)
            options = dict(zip(self.command[2::2], self.command[3::2], strict=True))
            directory = Path(options["--output-dir"])
            directory.mkdir()
            data = dict(
                state="completed",
                returncode=0,
                mode=options["--mode"],
                catalog_count=int(options["--catalog-count"]),
                trial=int(options["--trial"]),
            )
            if mutation:
                data.update(mutation)
            if not missing:
                (directory / suite.PROBE_REPORT_NAME).write_text(json.dumps(data))
            return exitcode

    return Process, commands


def test_four_context_cells_keep_measurement_schedule_identical(suite, args):
    runner, commands = runner_for(suite)
    result = suite.run_suite(args, runner=runner)
    assert result["state"] == "completed" and result["performance_measured"] is True
    assert result["context_screen"] is True
    assert all(
        result[key] is False
        for key in (
            "stable_performance_gain_established",
            "catalog_memory_causality_established",
            "full_trace_measured",
            "full_model_qualified",
            "end_to_end_gain_established",
        )
    )
    assert len(commands) == 4
    assert [(entry["mode"], entry["catalog_count"]) for entry in result["runs"]] == list(suite.CELLS)
    assert result["design"]["timed_step_order"] == [[0, 29], [29, 0]] * 5
    assert result["design"]["capture_orders"] == {
        "2": [400, 512],
        "17": [400, 304, 288, 352, 272, 496, 464, 480, 320, 448, 384, 368, 432, 256, 512, 336, 416],
    }
    assert result["design"]["capture_steps"]["17"] == [0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 18, 21, 24, 29, 37, 38]
    assert result["design"]["profile_repeats"] == 0
    assert result["design"]["process_replicates_per_cell"] == 1
    for index, entry in enumerate(result["runs"]):
        options = dict(zip(commands[index][2::2], commands[index][3::2], strict=True))
        assert options["--trial"] == "0" and options["--repeats"] == "10"
        assert options["--warmups"] == "5" and options["--seed"] == "1703"
        assert options["--catalog-count"] == str(entry["catalog_count"])
        assert "--profile-repeats" not in options
        assert options["--opp-root"] == str(getattr(args, entry["mode"] + "_vendor"))
        assert entry["process_order_index"] == index and entry["trial"] == 0
        assert "\\" not in entry["report_path"]
        path = args.output_dir / entry["report_path"]
        assert entry["report_sha256"] == suite.hashlib.sha256(path.read_bytes()).hexdigest()
        if index:
            assert result["runs"][index - 1]["finished_unix"] <= entry["launched_unix"]
    assert result["probe_sha256"] == suite.hashlib.sha256(args.probe.read_bytes()).hexdigest()
    assert json.loads((args.output_dir / suite.REPORT_NAME).read_text()) == result


@pytest.mark.parametrize(
    "mutation",
    [
        dict(state="failed"),
        dict(returncode=False),
        dict(returncode=1),
        dict(trial=False),
        dict(trial=1),
        dict(mode="legacy"),
        dict(catalog_count=17),
        dict(catalog_count=True),
        dict(catalog_count="2"),
    ],
)
def test_wrong_or_incomplete_cell_stops_before_next_process(suite, args, mutation):
    runner, commands = runner_for(suite, mutation=mutation)
    with pytest.raises(RuntimeError, match="identity"):
        suite.run_suite(args, runner=runner)
    result = json.loads((args.output_dir / suite.REPORT_NAME).read_text())
    assert result["state"] == "failed" and result["performance_measured"] is False
    assert len(commands) == 1 and "report_sha256" in result["runs"][0]


def test_nonzero_exit_preserves_raw_report_and_stops(suite, args):
    runner, commands = runner_for(suite, mutation=dict(state="failed", returncode=7), exitcode=7)
    with pytest.raises(RuntimeError, match="returncode=7"):
        suite.run_suite(args, runner=runner)
    result = json.loads((args.output_dir / suite.REPORT_NAME).read_text())
    assert result["state"] == "failed" and len(commands) == 1
    assert result["runs"][0]["returncode"] == 7 and "report_sha256" in result["runs"][0]


def test_success_exit_without_raw_report_is_not_completion(suite, args):
    runner, commands = runner_for(suite, missing=True)
    with pytest.raises(FileNotFoundError):
        suite.run_suite(args, runner=runner)
    result = json.loads((args.output_dir / suite.REPORT_NAME).read_text())
    assert result["state"] == "failed" and len(commands) == 1
    assert result["performance_measured"] is False
    assert "report_sha256" not in result["runs"][0]


@pytest.mark.parametrize("path_name", ["output_dir", "probe", "upstream_vendor", "whole_vendor"])
def test_paths_cannot_escape_the_experiment_root(suite, args, tmp_path, path_name):
    outside = tmp_path / "outside"
    if path_name == "probe":
        outside.write_text("# not owned\n")
    elif path_name != "output_dir":
        outside.mkdir()
    setattr(args, path_name, outside)
    with pytest.raises(ValueError, match="inside"):
        suite.run_suite(args)


def test_shared_vendor_existing_evidence_and_invalid_timeout_rejected(suite, args):
    args.whole_vendor = args.upstream_vendor
    with pytest.raises(ValueError, match="distinct"):
        suite.run_suite(args)
    args.whole_vendor = args.root / "whole"
    args.timeout = False
    with pytest.raises(ValueError, match="positive integer"):
        suite.run_suite(args)
    args.timeout = 10
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        suite.run_suite(args)


def test_catalog_uses_first_occurrence_order_not_sorted_capacity(suite):
    schedule = suite.load_trace_schedule()
    before = copy.deepcopy(schedule)
    design = suite.design_for(schedule)
    assert schedule == before
    assert design["capture_orders"]["17"] != sorted(design["capture_orders"]["17"])
    schedule["records"][1]["physical_tokens"] = 400
    with pytest.raises(ValueError, match="first-use"):
        suite.design_for(schedule)


def test_incomplete_trace_cannot_define_the_resident_catalog(suite):
    schedule = suite.load_trace_schedule()
    schedule["records"] = schedule["records"][:40]
    with pytest.raises(ValueError, match="128-step"):
        suite.design_for(schedule)
