import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.multi_graph.benchmark import (
    resolve_workload,
    summarize,
    verify_decode_steps,
    verify_outputs,
)


def workload(**overrides):
    fields = dict(
        batch_size=7,
        single_capacity=None,
        lane_sizes=None,
        warmups=3,
        repeats=20,
        output_tokens=64,
        prompt_tokens=32,
        prompt_spread=0,
        prompt_pattern="uniform",
        max_model_len=512,
        max_batched_tokens=None,
        kv_blocks=None,
        graph_budget=1 << 30,
        gpu_memory_utilization=0.8,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_padding_and_exact_lane_capacities_are_independent():
    args = workload(batch_size=416, single_capacity=512, lane_sizes=[384, 32])
    resolve_workload(args)
    assert args.single_capacity == 512 and args.lane_sizes == [384, 32]
    assert args.prompt_lengths == [32] * 416
    assert args.kv_blocks == 2048
    assert args.max_batched_tokens == 416 * 32


@pytest.mark.parametrize(
    "fields",
    [
        {"batch_size": 1},
        {"single_capacity": 6},
        {"single_capacity": 0},
        {"lane_sizes": [4, 4]},
        {"lane_sizes": [7, 0]},
        {"warmups": 0},
        {"repeats": 0},
        {"prompt_spread": -1},
        {"max_model_len": 64},
        {"kv_blocks": 1},
        {"kv_blocks": 0},
        {"graph_budget": 0},
        {"gpu_memory_utilization": float("nan")},
        {"max_batched_tokens": 16},
    ],
)
def test_reject_invalid_or_preempting_workloads(fields):
    with pytest.raises(ValueError):
        resolve_workload(workload(**fields))


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("uniform", [32] * 3),
        ("staggered", [32, 48, 64]),
        ("bimodal", [32, 64, 32]),
    ],
)
def test_exact_prompt_lengths(pattern, expected):
    args = workload(batch_size=3, prompt_spread=32, prompt_pattern=pattern)
    resolve_workload(args)
    assert args.prompt_lengths == expected


def test_statistics_include_interpolated_percentiles_and_variance():
    result = summarize([4.0, 1.0, 3.0, 2.0])
    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["count"] == 4 and result["stdev"] > 0
    assert summarize([1.0])["stdev"] == 0


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_reject_invalid_measurements(values):
    with pytest.raises(ValueError):
        summarize(values)


def results():
    request = {"tokens": [1, 2], "selected_logprobs": [-0.5, -0.3]}
    return {
        mode: {"generation_outputs": [[copy.deepcopy(request)], [copy.deepcopy(request)]]}
        for mode in ("eager", "single", "serial", "parallel")
    }


def test_correctness_checks_every_repeat_and_rejects_nonfinite_values():
    data = results()
    assert verify_outputs(data, "bfloat16")["passed"]
    data["parallel"]["generation_outputs"][0][0]["tokens"][0] = 9
    with pytest.raises(ValueError, match="tokens differ"):
        verify_outputs(data, "bfloat16")
    data = results()
    data["serial"]["generation_outputs"][0][0]["selected_logprobs"][1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        verify_outputs(data, "bfloat16")


def test_reject_logprob_regression_and_missing_repeats():
    data = results()
    data["parallel"]["generation_outputs"][0][0]["selected_logprobs"][0] = -1.0
    with pytest.raises(ValueError, match="tolerance"):
        verify_outputs(data, "float16")
    data = results()
    data["parallel"]["generation_outputs"].pop()
    with pytest.raises(ValueError, match="repeat counts"):
        verify_outputs(data, "float16")
    data = {mode: {"generation_outputs": []} for mode in results()}
    with pytest.raises(ValueError, match="empty"):
        verify_outputs(data, "float16")


def step(**overrides):
    fields = dict(actual_requests=7, lane_sizes=None, single_replays=[8], fallback=[])
    fields.update(overrides)
    return fields


def test_verify_observed_replay_instead_of_inferred_dispatch():
    verify_decode_steps([step()], "single", 7, 8, [4, 3])
    verify_decode_steps([step(lane_sizes=[4, 3], single_replays=[])], "parallel", 7, 8, [4, 3])
    verify_decode_steps([step(single_replays=[])], "eager", 7, 8, [4, 3])
    for invalid in ([], [step(single_replays=[])], [step(actual_requests=6)], [step(single_replays=[16])]):
        with pytest.raises(ValueError):
            verify_decode_steps(invalid, "single", 7, 8, [4, 3])
    with pytest.raises(ValueError, match="missed"):
        verify_decode_steps([step(fallback=["miss"])], "parallel", 7, 8, [4, 3])
    with pytest.raises(ValueError, match="coverage"):
        verify_decode_steps([step()], "single", 7, 8, [4, 3], expected_steps=2)


def test_parent_separates_generation_and_profiler_processes(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_split.py"
    spec = importlib.util.spec_from_file_location("multi_graph_benchmark_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        measurement = command[command.index("--measurement") + 1]
        mode = command[command.index("--mode") + 1]
        destination = Path(command[command.index("--result") + 1])
        assert kwargs["env"]["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        assert command[command.index("--kv-blocks") + 1] == "256"
        assert command[command.index("--prefix-cache") + 1] == "off"
        payload = results()[mode]
        if measurement == "trace":
            payload["generation_outputs"] = payload["generation_outputs"][:1]
        payload.update(
            resources={"kv_blocks": 256},
            generation_latency_ms={"p50": 10.0},
            output_tokens_per_second=1400.0,
            decode_forward_us=None,
        )
        destination.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--trace",
            "--repeats",
            "2",
            "--output-dir",
            str(tmp_path),
            "--mode-order",
            "parallel",
            "serial",
            "single",
            "eager",
        ],
    )
    module.main()
    assert len(calls) == 8
    assert [call[call.index("--mode") + 1] for call in calls[:4]] == ["parallel", "serial", "single", "eager"]
    assert [call[call.index("--measurement") + 1] for call in calls] == ["generation"] * 4 + ["trace"] * 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["measurements"]["generation"]["correctness"]["all_repeats_checked"]
    assert summary["measurements"]["trace"]["results"]["parallel"]["generation_speedup_vs_single"] is None
