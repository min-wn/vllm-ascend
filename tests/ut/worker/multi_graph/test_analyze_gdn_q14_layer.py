# SPDX-License-Identifier: Apache-2.0
"""CPU checks for complete-layer TP2 aggregation without new measurements."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "benchmarks/analyze_gdn_q14_layer.py"


def module():
    assert SOURCE.exists(), "complete-layer analyzer is not implemented"
    sys.path.insert(0, str(SOURCE.parent))
    try:
        spec = importlib.util.spec_from_file_location("analyze_q14_layer_tested", SOURCE)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def reports():
    result = {}
    for process in range(2):
        for rank in range(2):
            samples, segments = [], []
            for config in ("forced_q4", "q14_ready"):
                cost = (100.0 if config == "forced_q4" else 99.0) + process * 0.01 + rank * 0.001
                for sample in range(8):
                    samples.append(
                        dict(
                            process=process,
                            config=config,
                            sample=sample,
                            initialization_ms=1.0,
                            execution_ms=cost - 1.5,
                            terminal_ms=0.5,
                            lifecycle_ms=cost,
                        )
                    )
                for step in range(54, 70):
                    prepare = 0.5 + rank * 0.1
                    layer = 2.0 if config == "forced_q4" else 1.9
                    segments.append(
                        dict(
                            process=process,
                            config=config,
                            step=step,
                            segment="mixed" if step < 60 else "q1_only",
                            prepare_ms=prepare,
                            layer_ms=layer,
                            total_ms=prepare + layer,
                        )
                    )
            result[process, rank] = dict(
                status="completed_performance_process",
                rank=rank,
                samples=samples,
                segments=segments,
                input_file_sha256=f"input.rank{rank}",
                fingerprint={"files": {"frozen": "sha"}},
                source_sha256={"probe": "sha"},
                libraries={"new": "sha"},
            )
    return result


def write_reports(path, data):
    for (process, rank), report in data.items():
        (path / f"performance{process}.rank{rank}.json").write_text(json.dumps(report))


def test_selects_one_whole_rank_record_for_lifecycle_and_event_endpoint(tmp_path):
    analyzer = module()
    data = reports()
    # Rank 0 has a larger initialization, rank 1 a larger execution and total.
    for rank, values in ((0, (9.0, 90.0, 1.0)), (1, (1.0, 99.0, 1.0))):
        sample = data[0, rank]["samples"][0]
        sample.update(
            initialization_ms=values[0], execution_ms=values[1], terminal_ms=values[2], lifecycle_ms=sum(values)
        )
    data[0, 0]["segments"][0].update(prepare_ms=1.9, layer_ms=0.6, total_ms=2.5)
    data[0, 1]["segments"][0].update(prepare_ms=0.1, layer_ms=2.5, total_ms=2.6)
    write_reports(tmp_path, data)
    result = analyzer.analyze(tmp_path)
    selected = result["selected_samples"][0]
    assert selected["selected_rank"] == 1
    assert selected["initialization_ms"] == 1.0
    assert selected["execution_ms"] == 99.0
    assert selected["terminal_ms"] == 1.0
    assert selected["lifecycle_ms"] == 101.0
    event = result["selected_event_endpoints"][0]
    assert event["selected_rank"] == 1
    assert event["prepare_ms"] == 0.1 and event["layer_ms"] == 2.5 and event["total_ms"] == 2.6
    assert len(result["raw_rank_event_endpoints"]) == 128
    assert result["summary"]["sample_count"] == 32
    assert result["costs_by_config"]["forced_q4"]["samples"] == 16
    assert result["event_segments"]["forced_q4"]["mixed"]["steps_per_process"] == 6
    assert result["event_segments"]["forced_q4"]["q1_only"]["steps_per_process"] == 10


@pytest.mark.parametrize(
    "failure",
    [
        "missing_file",
        "missing_sample",
        "duplicate_sample",
        "rank",
        "identity",
        "lifecycle",
        "event_sum",
        "missing_event",
        "duplicate_event",
        "segment",
    ],
)
def test_rejects_incomplete_or_inconsistent_input(tmp_path, failure):
    analyzer = module()
    data = reports()
    if failure == "missing_file":
        del data[1, 1]
    elif failure == "missing_sample":
        data[1, 1]["samples"].pop()
    elif failure == "duplicate_sample":
        data[1, 1]["samples"][-1] = copy.deepcopy(data[1, 1]["samples"][-2])
    elif failure == "rank":
        data[1, 1]["rank"] = 0
    elif failure == "identity":
        data[1, 1]["fingerprint"]["files"]["frozen"] = "changed"
    elif failure == "lifecycle":
        data[1, 1]["samples"][0]["lifecycle_ms"] += 1.0
    elif failure == "event_sum":
        data[1, 1]["segments"][0]["total_ms"] += 1.0
    elif failure == "missing_event":
        data[1, 1]["segments"].pop()
    elif failure == "duplicate_event":
        data[1, 1]["segments"][-1] = copy.deepcopy(data[1, 1]["segments"][-2])
    else:
        data[1, 1]["segments"][0]["segment"] = "q4_only"
    write_reports(tmp_path, data)
    with pytest.raises((ValueError, FileNotFoundError)):
        analyzer.analyze(tmp_path)


def test_cli_saves_json_and_refuses_to_overwrite(tmp_path, monkeypatch):
    analyzer = module()
    write_reports(tmp_path, reports())
    target = tmp_path / "analysis.json"
    monkeypatch.setattr(sys, "argv", ["analyze", "--input-dir", str(tmp_path), "--output", str(target)])
    analyzer.main()
    assert json.loads(target.read_text())["summary"]["stable"]
    with pytest.raises(FileExistsError):
        analyzer.main()
