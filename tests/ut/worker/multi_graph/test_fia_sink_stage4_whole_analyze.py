"""CPU contract tests for the bounded whole-plan report analyzer."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def module(monkeypatch):
    path = ROOT / "benchmarks/aclgraph_fia_sink_stage4_whole_analyze.py"
    spec = importlib.util.spec_from_file_location("stage4_whole_analyzer", path)
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def report():
    path = ROOT / "benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-plan.json"
    return json.loads(path.read_text())


def test_formal_report_is_complete(module, report):
    result = module.analyze(report)
    assert result["analysis_valid"]
    assert result["protocol_qualified"]
    assert result["state"] == "complete_probe"
    assert result["full_trace_measured"] is False
    assert result["bounded_probe_measured"] is True
    assert result["expected_sample_count"] == 240
    assert result["comparisons"]["whole_oracle_minus_native_oracle"]["paired_count"] == 60


@pytest.mark.parametrize("field", ["state", "returncode", "performance_measured", "npu_work_executed"])
def test_failed_execution_cannot_qualify(module, report, field):
    broken = copy.deepcopy(report)
    broken[field] = "failed" if field == "state" else (1 if field == "returncode" else False)
    result = module.analyze(broken)
    assert not result["protocol_qualified"]


def test_missing_sample_is_rejected(module, report):
    broken = copy.deepcopy(report)
    broken["samples"].pop()
    result = module.analyze(broken)
    assert not result["analysis_valid"]
    assert result["coverage"]["missing"]


def test_duplicate_sample_is_rejected(module, report):
    broken = copy.deepcopy(report)
    broken["samples"].append(copy.deepcopy(broken["samples"][0]))
    result = module.analyze(broken)
    assert not result["analysis_valid"]
    assert result["coverage"]["duplicates"]


def test_nonfinite_sample_is_rejected(module, report):
    broken = copy.deepcopy(report)
    broken["samples"][0]["wall_us"] = float("nan")
    result = module.analyze(broken)
    assert not result["analysis_valid"]
    assert result["coverage"]["invalid"]


def test_screen_does_not_qualify_formal_representatives(module, report):
    broken = copy.deepcopy(report)
    broken["screen"] = True
    result = module.analyze(broken)
    assert not result["analysis_valid"]
    assert not result["protocol_qualified"]


def test_selected_capacity_mutation_is_rejected(module, report):
    broken = copy.deepcopy(report)
    broken["selected_records"][0]["physical_tokens"] = 400
    result = module.analyze(broken)
    assert not result["analysis_valid"]


def test_all_comparisons_preserve_scope_limits(module, report):
    result = module.analyze(report)
    assert result["full_model_measured"] is False
    assert result["end_to_end_speedup_proven"] is False
    assert result["comparisons"]["whole_cpu_minus_native_shared"]["paired_count"] == 60


def test_one_trial_cannot_qualify_three_trial_protocol(module, report):
    report["trials"] = 1
    report["samples"] = [sample for sample in report["samples"] if sample["trial"] == 0]
    result = module.analyze(report)
    assert result["analysis_valid"]
    assert not result["protocol_qualified"]
    assert not result["qualification"]["checks"]["three_trials_five_repeats"]


def test_bool_success_code_is_not_numeric_success(module, report):
    report["returncode"] = False
    result = module.analyze(report)
    assert not result["protocol_qualified"]
    assert result["state"] == "partial_or_invalid"


@pytest.mark.parametrize("field,value", [("trial", False), ("repeat", []), ("step", []), ("case", [])])
def test_malformed_indices_do_not_crash_or_qualify(module, report, field, value):
    report["samples"][0][field] = value
    result = module.analyze(report)
    assert not result["analysis_valid"]
    assert not result["protocol_qualified"]


def test_duplicated_numeric_record_does_not_cover_all_capacities(module, report):
    report["checks"] = [copy.deepcopy(report["checks"][0]) for _ in report["checks"]]
    assert not module.analyze(report)["protocol_qualified"]


def test_post_timing_requires_fp32_evidence(module, report):
    report["post_timing_checks"][0].pop("fp32_sampled_rows")
    assert not module.analyze(report)["protocol_qualified"]
