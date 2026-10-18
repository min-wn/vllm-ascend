# SPDX-License-Identifier: Apache-2.0
"""CPU evidence contracts for matched preparation experiments."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
CASES = ("sink_fresh", "sink_reuse", "legacy_fresh", "legacy_reuse")
TIMINGS = dict(sink_fresh=600, sink_reuse=450, legacy_fresh=800, legacy_reuse=650)


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("aclgraph_fia_sink_stage4_analyze")


@pytest.fixture
def report(analyzer):
    expected = analyzer.stage3.expected_schedule()
    return dict(
        state="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        trials=3,
        screen_steps=128,
        occurrences=16,
        cases=list(CASES),
        ordering_strategy="step_rotation_reverse_odd_trial_v1",
        schedule={**copy.deepcopy(expected), "installed_catalog_revalidated": True},
        installed_catalog=dict(
            selection=dict(
                capture_sizes=expected["default_capture_sizes"],
                dispatched_tokens=[row["capacity"] for row in expected["records"]],
            ),
            installed_catalog_revalidated=True,
        ),
        samples=[
            dict(
                trial=trial,
                case=case,
                **row,
                wall_us=TIMINGS[case] + row["step"] + trial,
                queue_span_us=200 + trial,
                preparation_enqueue_us=100 if case.endswith("fresh") else 25,
                submit_join_cpu_us=50,
            )
            for trial in range(3)
            for row in expected["records"]
            for order in [CASES if trial % 2 == 0 else tuple(reversed(CASES))]
            for offset in [(row["step"] + trial) % len(CASES)]
            for case in order[offset:] + order[:offset]
        ],
    )


def test_complete_contract_has_matched_statistics_and_limited_claims(analyzer, report):
    result = analyzer.analyze(report)
    assert result["state"] == "complete_trace"
    assert result["analysis_valid"] is True
    assert result["coverage"]["expected_sample_count"] == 1536
    assert result["coverage"]["distinct_valid_sample_count"] == 1536
    assert result["three_trial_trace_protocol_complete"] is True
    assert [sample["case"] for sample in report["samples"][:8]] == [
        "sink_fresh",
        "sink_reuse",
        "legacy_fresh",
        "legacy_reuse",
        "sink_reuse",
        "legacy_fresh",
        "legacy_reuse",
        "sink_fresh",
    ]
    assert [sample["case"] for sample in report["samples"][512:516]] == [
        "legacy_fresh",
        "sink_reuse",
        "sink_fresh",
        "legacy_reuse",
    ]
    assert result["full_128_step_trace_measured"] is True
    assert result["summaries"]["sink_reuse"]["wall_us"]["mean_us"] == 514.5
    pair = result["comparisons"]["sink_fresh_minus_reuse"]
    assert pair["paired_count"] == 384
    assert pair["comparison_faster_count"] == 384
    assert pair["wall_difference"]["mean_us"] == 150
    assert pair["preparation_enqueue_difference"]["mean_us"] == 75
    assert pair["queue_span_difference"]["mean_us"] == 0
    assert len(pair["per_capacity"]) == 17
    assert pair["per_trial"][0]["wall_difference"]["mean_us"] == 150
    assert sum(row["paired_count"] for row in pair["per_capacity"]) == 384
    assert pair["per_capacity"][-1]["capacity"] == 512
    assert pair["per_capacity"][-1]["preparation_enqueue_difference"]["p95_us"] == 75
    assert result["comparisons"]["legacy_fresh_minus_reuse"]["wall_difference"]["mean_us"] == 150
    assert result["comparisons"]["reused_legacy_minus_sink"]["wall_difference"]["mean_us"] == 200
    for field in (
        "full_model_measured",
        "end_to_end_speedup_proven",
        "independent_algorithm_speedup_proven",
        "novelty_claim_established",
    ):
        assert result[field] is False


@pytest.mark.parametrize(
    ("mutation", "missing_count"),
    [
        ("trial", 512),
        ("step", 12),
        ("case", 384),
        ("one", 1),
    ],
)
def test_missing_samples_do_not_shrink_contract(analyzer, report, mutation, missing_count):
    if mutation == "one":
        report["samples"].pop()
    else:
        omitted = {"trial": 2, "step": 127, "case": "sink_reuse"}[mutation]
        report["samples"] = [sample for sample in report["samples"] if sample[mutation] != omitted]
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is False
    assert result["coverage"]["expected_sample_count"] == 1536
    assert len(result["coverage"]["missing"]) == missing_count
    assert result["three_trial_trace_protocol_complete"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_case", "wrong_capacity", "wrong_batch", "bool_trial"])
def test_misaligned_and_duplicate_evidence_rejected(analyzer, report, mutation):
    if mutation == "duplicate":
        report["samples"].append(copy.deepcopy(report["samples"][0]))
    else:
        changes = {
            "wrong_case": dict(case="shared"),
            "wrong_capacity": dict(capacity=392),
            "wrong_batch": dict(batch=99),
            "bool_trial": dict(trial=False),
        }
        report["samples"][0].update(changes[mutation])
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is False
    assert result["coverage"]["complete"] is False
    assert result["three_trial_trace_protocol_complete"] is False


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, -1, 0])
def test_nonfinite_and_invalid_wall_rejected(analyzer, report, value):
    report["samples"][0]["wall_us"] = value
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is False
    assert len(result["coverage"]["invalid"]) == 1


@pytest.mark.parametrize("field", ["queue_span_us", "preparation_enqueue_us", "submit_join_cpu_us"])
def test_all_timing_metrics_require_finite_numbers(analyzer, report, field):
    report["samples"][0][field] = float("inf")
    assert analyzer.analyze(report)["analysis_valid"] is False


@pytest.mark.parametrize("mutation", ["truncated", "digest", "catalog", "source_hash", "changed_dispatch"])
def test_schedule_independently_pinned(analyzer, report, mutation):
    if mutation == "truncated":
        report["schedule"]["records"].pop()
    elif mutation == "digest":
        report["schedule"]["canonical_digest"] = "0" * 64
    elif mutation == "catalog":
        report["schedule"]["default_capture_sizes"] = [392]
    elif mutation == "source_hash":
        report["schedule"]["selection_sha256"] = "0" * 64
    else:
        report["schedule"]["records"][0]["capacity"] = 392
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is False
    assert result["issues"]


@pytest.mark.parametrize("field", ["trials", "screen_steps", "cases"])
def test_sample_count_does_not_replace_explicit_contract(analyzer, report, field):
    report.pop(field)
    assert analyzer.analyze(report)["analysis_valid"] is False


@pytest.mark.parametrize("mutation", ["rc", "state", "catalog", "occurrences", "order"])
def test_coverage_does_not_replace_execution_qualification(analyzer, report, mutation):
    if mutation == "rc":
        report["returncode"] = 1
    elif mutation == "state":
        report.update(state="failed", status="completed")
    elif mutation == "catalog":
        report["installed_catalog"]["installed_catalog_revalidated"] = False
    elif mutation == "occurrences":
        report["occurrences"] = 1
    else:
        # Keep all matched observations but remove the declared odd-trial reversal.
        report["samples"].sort(key=lambda row: (row["trial"], CASES.index(row["case"]), row["step"]))
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is True
    assert result["coverage"]["complete"] is True
    assert result["three_trial_trace_protocol_complete"] is False
    if mutation in ("rc", "state"):
        assert result["full_128_step_trace_measured"] is False


def test_short_screen_never_claims_full_trace_protocol(analyzer, report):
    report.update(trials=1, screen_steps=1)
    report["samples"] = [sample for sample in report["samples"] if sample["trial"] == sample["step"] == 0]
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is True
    assert result["coverage"]["expected_sample_count"] == 4
    assert result["state"] == "complete_screen"
    assert result["full_128_step_trace_measured"] is False
    assert result["three_trial_trace_protocol_complete"] is False


def test_configuration_schema_aliases(analyzer, report):
    report["configuration"] = dict(
        trials=report.pop("trials"), steps=report.pop("screen_steps"), cases=report.pop("cases")
    )
    assert analyzer.analyze(report)["three_trial_trace_protocol_complete"] is True


def test_fixed_reverse_order_requires_its_own_declared_strategy(analyzer, report):
    report["samples"].sort(
        key=lambda row: (
            row["trial"],
            CASES.index(row["case"]) if row["trial"] % 2 == 0 else len(CASES) - CASES.index(row["case"]),
            row["step"],
        )
    )
    assert analyzer.analyze(report)["three_trial_trace_protocol_complete"] is False
    report["ordering_strategy"] = "reverse_odd_trial_v1"
    assert analyzer.analyze(report)["three_trial_trace_protocol_complete"] is True


@pytest.mark.parametrize("strategy", [None, "arbitrary", "step_rotation_reverse_odd_trial_v2"])
def test_undeclared_or_unknown_order_strategy_not_qualified(analyzer, report, strategy):
    report["ordering_strategy"] = strategy
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is True
    assert result["three_trial_trace_protocol_complete"] is False


def test_rotation_is_bound_to_both_step_and_trial(analyzer, report):
    # All observations still exist, but the first odd-trial block uses offset0
    # instead of the required (step0 + trial1) modulo4.
    report["samples"][512:516] = report["samples"][515:516] + report["samples"][512:515]
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is True
    assert result["three_trial_trace_protocol_complete"] is False


@pytest.mark.parametrize("samples", [None, {}, [None]])
def test_malformed_sample_container_remains_explicit(analyzer, report, samples):
    report["samples"] = samples
    result = analyzer.analyze(report)
    assert result["analysis_valid"] is False
    assert result["three_trial_trace_protocol_complete"] is False


def test_high_capacity_regression_is_visible(analyzer, report):
    for sample in report["samples"]:
        if sample["case"] == "sink_reuse" and sample["capacity"] == 512:
            sample["wall_us"] += 400
    result = analyzer.analyze(report)
    high = result["comparisons"]["reused_legacy_minus_sink"]["per_capacity"][-1]
    assert high["capacity"] == 512
    assert high["wall_difference"]["mean_us"] == -200
    assert high["comparison_faster_count"] == 0
    assert high["comparison_slower_count"] == high["paired_count"]


def test_analyzer_import_does_not_load_native_stack():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import aclgraph_fia_sink_stage4_analyze; "
        "assert not any(name == 'torch' or name.startswith(('torch.', 'torch_npu', 'vllm')) for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(BENCHMARKS)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cli_preserves_projection_and_does_not_overwrite_input(analyzer, report, tmp_path):
    report["evidence_projection"] = dict(original_report_sha256="1" * 64, timing_samples_preserved_exactly=True)
    source, output = tmp_path / "raw.json", tmp_path / "analysis.json"
    source.write_text(json.dumps(report), encoding="ascii")
    script = BENCHMARKS / "aclgraph_fia_sink_stage4_analyze.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["input"]["original_report_sha256"] == "1" * 64
    assert result["input"]["evidence_projection"] == report["evidence_projection"]
    assert len(result["stage3_helper_sha256"]) == 64
    unchanged = source.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(source)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert source.read_bytes() == unchanged
