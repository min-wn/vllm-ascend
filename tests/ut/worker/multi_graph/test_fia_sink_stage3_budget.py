# SPDX-License-Identifier: Apache-2.0
"""Independent CPU contract and completeness tests for stage-three evidence."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
CANONICAL_DIGEST = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
MODES = ("per_layer", "shared", "attention_only", "planner", "legacy")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return (
        importlib.import_module("aclgraph_fia_sink_stage3_trace"),
        importlib.import_module("aclgraph_fia_sink_stage3_analyze"),
    )


@pytest.fixture
def schedule(modules):
    return modules[0].load_trace_schedule()


@pytest.fixture
def report(schedule):
    """Keep the real runner's JSON keys without embedding raw device evidence."""
    frames = schedule["records"]
    base_times = dict(per_layer=2000, shared=1000, attention_only=900, planner=150, legacy=3000)
    return dict(
        state="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        trials=3,
        screen_steps=128,
        modes=list(MODES),
        occurrences=16,
        schedule=dict(
            canonical_digest=schedule["canonical_digest"],
            default_capture_sizes=list(schedule["default_capture_sizes"]),
            trace_file_sha256=schedule["trace_file_sha256"],
            selection_sha256=schedule["selection_sha256"],
            installed_catalog_revalidated=True,
            records=[{name: frame[name] for name in ("step", "batch", "physical_tokens")} for frame in frames],
        ),
        installed_catalog=dict(
            selection=dict(
                capture_sizes=list(schedule["default_capture_sizes"]),
                dispatched_tokens=[frame["physical_tokens"] for frame in frames],
            ),
            model_config_only=True,
            weights_loaded=False,
            installed_catalog_revalidated=True,
        ),
        samples=[
            dict(
                trial=trial,
                mode=mode,
                step=frame["step"],
                batch=frame["batch"],
                capacity=frame["physical_tokens"],
                wall_us=base_times[mode] + trial + frame["step"],
                queue_span_us=100 + trial,
                preparation_enqueue_us=20,
                submit_join_cpu_us=50,
            )
            for trial in range(3)
            for mode in MODES
            for frame in frames
        ],
    )


def test_pinned_trace_keeps_default_q4_physical_padding(schedule):
    assert schedule["canonical_digest"] == CANONICAL_DIGEST
    assert len(schedule["records"]) == 128
    assert schedule["installed_catalog_revalidated"] is False
    first = schedule["records"][0]
    assert (first["batch"], first["live_tokens"], first["physical_tokens"]) == (98, 392, 400)
    assert sum(frame["batch"] for frame in schedule["records"]) == 12493
    for frame in schedule["records"]:
        live, physical = frame["batch"], frame["physical_batch"]
        assert frame["physical_tokens"] == min(
            size for size in schedule["default_capture_sizes"] if size >= frame["live_tokens"]
        )
        assert frame["q_lengths"] == (4,) * physical
        assert frame["actual_seq_qlen"] == tuple(range(4, frame["physical_tokens"] + 1, 4))
        assert frame["kv_lengths"][live:] == (0,) * (physical - live)
        assert frame["owner_keys"][live:] == (None,) * (physical - live)
        assert all(all(page == 0 for page in row) for row in frame["block_table"][live:])
        assert frame["slot_mapping"][4 * live :] == (-1,) * (4 * (physical - live))
        assert not any(frame["active_tokens"][4 * live :])


def test_owner_history_and_pages_survive_pause_and_row_permutation(schedule):
    trace_path = BENCHMARKS / "results/multi_graph_structural/structural-dynamic-trace-v1.json"
    initial = json.loads(trace_path.read_text(encoding="utf-8"))["trace"]["initial_prefix_by_request"]
    lengths, pending, last_row, previous, seen = {}, {}, {}, set(), set()
    admissions = resumes = departures = moved = 0
    owners = schedule["owner_pages"]["pages"]
    allocated = [page for owner in owners for page in owner]
    assert len(allocated) == len(set(allocated))
    assert 0 not in allocated
    for frame in schedule["records"]:
        current = set(frame["request_ids"])
        assert set(frame["admitted"]) == current - seen
        assert set(frame["resumed"]) == current & seen - previous
        assert set(frame["departed"]) == previous - current
        assert frame["retired"] == ()
        assert frame["owner_active"] == tuple(request in current for request in range(128))
        for row, request in enumerate(frame["request_ids"]):
            first = request not in seen
            expected = int(initial[str(request)]) + 4 if first else lengths[request] + pending[request]
            assert frame["first_use"][row] is first
            assert frame["accepted_from_previous"][row] == pending.get(request, 1)
            assert frame["kv_lengths"][row] == expected
            assert frame["owner_keys"][row] == (request, 0)
            assert frame["block_table"][row][: len(owners[request])] == owners[request]
            positions = tuple(range(expected - 4, expected))
            slots = tuple(owners[request][position // 128] * 128 + position % 128 for position in positions)
            assert frame["positions"][4 * row : 4 * row + 4] == positions
            assert frame["slot_mapping"][4 * row : 4 * row + 4] == slots
            moved += request in last_row and last_row[request] != row
            lengths[request], pending[request], last_row[request] = (
                expected,
                frame["accepted_after_step"][row],
                row,
            )
        admissions += len(frame["admitted"])
        resumes += len(frame["resumed"])
        departures += len(frame["departed"])
        previous = current
        seen.update(current)
    assert (admissions, resumes, departures) == (128, 2876, 2927)
    assert moved > 0


@pytest.mark.parametrize("mutation", ["digest", "dispatch", "custom_catalog"])
def test_trace_rejects_changed_capture_contract(modules, tmp_path, mutation):
    path = BENCHMARKS / "results/multi_graph_structural/q4-tile-full-replay-v8-queue0.rank0.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "digest":
        selection["trace_sha256"] = "different-trace"
    elif mutation == "dispatch":
        selection["selection"]["dispatched_tokens"][0] = 392
    else:
        selection["selection"]["capture_sizes"].append(514)
        selection["selection"]["config_source"] = "custom sizes"
    changed = tmp_path / "selection.json"
    changed.write_text(json.dumps(selection), encoding="ascii")
    with pytest.raises(ValueError):
        modules[0].load_trace_schedule(selection_path=changed)


def test_helpers_do_not_import_native_stack():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import aclgraph_fia_sink_stage3_trace, aclgraph_fia_sink_stage3_analyze; "
        "assert not any(name == 'torch' or name.startswith(('torch.', 'torch_npu', 'vllm')) "
        "for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(BENCHMARKS)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_complete_runner_schema_preserves_pairing_and_claim_boundaries(modules, report):
    result = modules[1].analyze(report)
    assert result["state"] == "complete_trace"
    assert result["analysis_valid"] is True
    assert result["coverage"]["expected_sample_count"] == 1920
    assert result["coverage"]["distinct_valid_sample_count"] == 1920
    assert result["full_128_step_trace_measured"] is True
    assert result["three_trial_trace_protocol_complete"] is True
    assert result["summaries"]["shared"]["wall_us"]["mean_us"] == 1064.5
    comparisons = result["comparisons"]
    assert comparisons["per_layer_minus_shared"]["paired_count"] == 384
    assert comparisons["per_layer_minus_shared"]["wall_difference"]["mean_us"] == 1000
    assert comparisons["shared_minus_attention_only"]["wall_difference"]["mean_us"] == 100
    assert comparisons["shared_minus_attention_only"]["diagnostic_only"] is True
    assert comparisons["legacy_minus_shared"]["wall_difference"]["mean_us"] == 2000
    assert result["planning_budget"]["planner_saving_multiplier_against_shared_baseline"] == 1
    assert result["planning_budget"]["full_model_required_net_saving_us"] == pytest.approx(7173.7)
    assert result["full_model_measured"] is False
    assert result["end_to_end_speedup_proven"] is False
    assert result["independent_algorithm_speedup_proven"] is False


@pytest.mark.parametrize(
    ("mutation", "missing"), [("last_trial", 640), ("last_step", 15), ("whole_mode", 384), ("one", 1)]
)
def test_missing_observations_cannot_shrink_expected_contract(modules, report, mutation, missing):
    if mutation == "last_trial":
        report["samples"] = [sample for sample in report["samples"] if sample["trial"] != 2]
    elif mutation == "last_step":
        report["samples"] = [sample for sample in report["samples"] if sample["step"] != 127]
    elif mutation == "whole_mode":
        report["samples"] = [sample for sample in report["samples"] if sample["mode"] != "shared"]
    else:
        report["samples"].pop()
    result = modules[1].analyze(report)
    assert result["analysis_valid"] is False
    assert result["coverage"]["expected_sample_count"] == 1920
    assert len(result["coverage"]["missing"]) == missing
    assert result["three_trial_trace_protocol_complete"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected", "capacity", "batch"])
def test_duplicate_and_misaligned_samples_invalidate_coverage(modules, report, mutation):
    if mutation == "duplicate":
        report["samples"].append(copy.deepcopy(report["samples"][0]))
    elif mutation == "unexpected":
        report["samples"][0]["trial"] = 3
    elif mutation == "capacity":
        report["samples"][0]["capacity"] = 392
    else:
        report["samples"][0]["batch"] = 99
    result = modules[1].analyze(report)
    assert result["analysis_valid"] is False
    assert result["coverage"]["complete"] is False
    assert result["three_trial_trace_protocol_complete"] is False
    key = "duplicates" if mutation == "duplicate" else "unexpected" if mutation == "unexpected" else "invalid"
    assert len(result["coverage"][key]) == 1


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, 0])
def test_invalid_wall_samples_cannot_produce_a_complete_measurement(modules, report, value):
    report["samples"][0]["wall_us"] = value
    result = modules[1].analyze(report)
    assert result["analysis_valid"] is False
    assert len(result["coverage"]["invalid"]) == 1
    assert result["full_128_step_trace_measured"] is False


@pytest.mark.parametrize("field", ["trials", "screen_steps", "modes"])
def test_sample_counts_cannot_replace_explicit_measurement_contract(modules, report, field):
    del report[field]
    result = modules[1].analyze(report)
    assert result["analysis_valid"] is False
    assert result["issues"]
    assert result["three_trial_trace_protocol_complete"] is False


@pytest.mark.parametrize("mutation", ["truncated_schedule", "changed_catalog", "changed_trace_sha"])
def test_reported_schedule_is_checked_against_independent_artifacts(modules, report, mutation):
    if mutation == "truncated_schedule":
        report["schedule"]["records"].pop()
    elif mutation == "changed_catalog":
        report["schedule"]["default_capture_sizes"].append(514)
    else:
        report["schedule"]["trace_file_sha256"] = "modified"
    result = modules[1].analyze(report)
    assert result["analysis_valid"] is False
    assert result["coverage"]["complete"] is True
    assert result["issues"]


def test_short_screen_retains_canonical_schedule_without_claiming_full_trace(modules, report):
    report["screen_steps"] = 4
    report["samples"] = [sample for sample in report["samples"] if sample["step"] < 4]
    result = modules[1].analyze(report)
    assert result["state"] == "complete_screen"
    assert result["analysis_valid"] is True
    assert result["coverage"]["expected_sample_count"] == 60
    assert result["screen_only"] is True
    assert result["full_128_step_trace_measured"] is False
    assert result["three_trial_trace_protocol_complete"] is False
    report["schedule"]["records"] = report["schedule"]["records"][:4]
    assert modules[1].analyze(report)["analysis_valid"] is False


@pytest.mark.parametrize(
    "mutation", ["returncode", "missing_returncode", "occurrences", "catalog_flag", "installed_dispatch"]
)
def test_complete_samples_do_not_override_failed_or_unqualified_protocol(modules, report, mutation):
    if mutation == "returncode":
        report["returncode"] = 1
    elif mutation == "missing_returncode":
        del report["returncode"]
    elif mutation == "occurrences":
        report["occurrences"] = 1
    elif mutation == "catalog_flag":
        report["schedule"]["installed_catalog_revalidated"] = False
    else:
        report["installed_catalog"]["selection"]["dispatched_tokens"][0] = 392
    result = modules[1].analyze(report)
    assert result["coverage"]["complete"] is True
    assert result["three_trial_trace_protocol_complete"] is False
    assert result["qualification"]["complete"] is False
    assert result["qualification"]["unqualified_checks"]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_reader_rejects_nonfinite_constants(modules, tmp_path, constant):
    path = tmp_path / "nonfinite.json"
    path.write_text('{"wall_us":' + constant + "}", encoding="ascii")
    with pytest.raises(ValueError, match="nonfinite JSON constant"):
        modules[1].load_json(path)
