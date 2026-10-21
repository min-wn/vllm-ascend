# SPDX-License-Identifier: Apache-2.0
"""Task intervals do not identify arithmetic-pipe or cross-core wait overlap."""

import csv
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    return importlib.import_module("multi_graph_gdn_ready_chain_profile")


def test_interval_accounting_merges_nested_and_touching_tasks(profile):
    assert profile.length([(0, 10), (1, 3), (10, 15), (20, 30)]) == 25
    assert profile.intersection([(0, 10), (5, 15)], [(7, 8), (8, 20)]) == 8
    with pytest.raises(ValueError):
        profile.length([(2, 1)])


def test_static_flags_include_padding_and_idle_core_final_notifications(profile):
    counts = profile.synchronization_counts(112)
    assert counts["delayed"] == dict(ready_set=20, ready_wait=40, consumed_set=0, consumed_wait=0)
    assert counts["ready"] == dict(ready_set=112, ready_wait=224, consumed_set=224, consumed_wait=112)
    assert profile.synchronization_counts(4)["delayed"]["ready_set"] == 20


@pytest.mark.parametrize("cann91", [False, True])
def test_task_parser_preserves_small_deltas_and_leaves_unobserved_core_wait_unknown(profile, tmp_path, cann91):
    def task(name, index, kind):
        return dict(name=name, ts=f"1792436705758187.{index:03}", dur=0.001, ph="X", args={"Task Type": kind})

    events = (
        [task("q4_tile_solve_1", i, "KERNEL_MIX_AIC") for i in range(48)]
        + [task("q4_log_deferred_0", 48 + i, "KERNEL_MIX_AIV") for i in range(48)]
        + [task("aclnnMatmulWeightNz_MatMulCommon_MatMulV2", 96 + i, "KERNEL_AICORE") for i in range(305)]
        + [task("hcom_allGather__1", 401, "HCCL")]
    )
    if cann91:
        kinds = {"KERNEL_MIX_AIC": "MIX_AIC", "KERNEL_MIX_AIV": "AI_VECTOR_CORE", "KERNEL_AICORE": "AI_CORE"}
        for event in events:
            event["args"]["Task Type"] = kinds.get(event["args"]["Task Type"], event["args"]["Task Type"])
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(events))
    result = profile.analyze_trace(path, 400)
    assert result["tasks"]["gdn_mixed"]["count"] == 48
    assert result["tasks"]["dense_without_head"]["count"] == 304
    assert result["device_span_us"] == pytest.approx(0.402)
    assert result["intrakernel_aic_aiv_overlap_us"] is None
    assert result["cross_core_flag_wait_us"] is None
    assert len(result["allgather"]) == 1


def test_pmu_uses_one_table_and_rejects_changed_core_resources(profile, tmp_path):
    fields = dict.fromkeys(
        (
            "aicore_time(us)",
            "aic_mac_time(us)",
            "aic_scalar_time(us)",
            "aic_mte1_time(us)",
            "aic_mte2_time(us)",
            "aic_fixpipe_time(us)",
            "aiv_time(us)",
            "aiv_vec_time(us)",
            "aiv_scalar_time(us)",
            "aiv_mte2_time(us)",
            "aiv_mte3_time(us)",
        ),
        "10",
    )
    row = dict(Name="q4_tile_solve_1", **{"Block Num": "20", "Mix Block Num": "40", "Duration(us)": "20"}, **fields)
    rows = [dict(row) for _ in range(48)]

    def write(name):
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

    write("kernel_details.csv")
    write("op_summary_duplicate.csv")
    result = profile.mixed_metrics(tmp_path)
    assert result["count"] == 48
    assert result["mean_task_us"] == 20
    assert result["mean_reported_per_core_us"]["aiv_time(us)"] == 10
    assert result["flag_wait_us"] is None and result["cube_vector_overlap_us"] is None
    rows[0]["Mix Block Num"] = "38"
    write("kernel_details.csv")
    with pytest.raises(ValueError, match="preserve 20 AIC / 40 AIV"):
        profile.mixed_metrics(tmp_path)
