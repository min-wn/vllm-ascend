"""Fixed-query target replay qualification; no drafter or acceptance sampling."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("query", [1, 2, 4, 8])
@pytest.mark.parametrize("merge", ["cat", "captured"])
def test_fixed_query_replay(tmp_path, query, merge, request):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "replay.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_replay.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch-invariant",
            "--merge",
            merge,
            "--batch",
            "7",
            "--capacity",
            "8",
            "--query",
            str(query),
            "--kv-length",
            "64",
            "--rounds",
            "1",
            "--protocols",
            "update",
            "frozen",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert result["batch_invariant"]
    assert result["merge"] == merge
    assert all(case["passed"] for case in result["controls"])
    assert result["draft_invocations"] == 0
    assert len(result["correctness"]) == 6
    assert all(case["passed"] for case in result["correctness"])
    assert len(result["measurements"]) == 6
    for graph in result["graphs"]:
        assert (graph["update_tasks"] > 0) == (graph["protocol"] == "update")
        assert (graph["frozen_tasks"] > 0) == (graph["protocol"] == "frozen")


def test_native_same_shape_replay_does_not_claim_cross_shape_accuracy(tmp_path, request):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "native.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_replay.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch",
            "33",
            "--capacity",
            "64",
            "--query",
            "4",
            "--kv-length",
            "512",
            "--qualification",
            "same-shape",
            "--submission",
            "replay-first",
            "--merge",
            "captured",
            "--logit-check",
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert not result["batch_invariant"]
    assert not result["cross_shape_qualified"]
    assert result["qualification"] == "same-shape"
    padding_control = next(row for row in result["controls"] if row["name"] == "eager_padded")
    assert not padding_control["required"]
    assert padding_control["passed"] is None
    assert len(result["correctness"]) == 3
    for row in result["correctness"]:
        assert row["passed"]
        assert "relative_rms" in row["cross_shape"]
        assert row["last_query_logit_diagnostics"]["tokens"] == 33
    assert all(row["update_tasks"] > 0 for row in result["graphs"])


@pytest.mark.parametrize("query,lanes,merge", [(1, (4, 3), "cat"), (8, (6, 1), "captured")])
def test_shared_dense_attention_branches(tmp_path, request, query, lanes, merge):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "shared_dense.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_shared_dense.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch",
            "7",
            "--capacity",
            "8",
            "--query",
            str(query),
            "--kv-length",
            "128",
            "--lanes",
            *map(str, lanes),
            "--merge",
            merge,
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert result["full_split_merge"] == merge
    assert not result["batch_invariant"]
    assert not result["cross_architecture_qualified"]
    assert all(row["passed"] for row in result["checks"])
    assert len([row for row in result["checks"] if row["name"].endswith("/changed_input")]) == 8
    assert len(result["measurements"]) == 8
    for row in result["measurements"]:
        assert len(row["wall_us"]) == 3
        assert min(row["wall_us"]) > 0
    graphs = {row["name"]: row for row in result["graphs"]}
    for name in ("shared_dense_pre", "shared_dense_post"):
        assert graphs[name]["update_tasks"] == 0
    for name in ("attention_lane_0", "attention_lane_1"):
        assert graphs[name]["update_tasks"] == 1
    for name in ("joined_shared_serial", "joined_shared_parallel"):
        assert graphs[name]["update_tasks"] == 2
        assert graphs[name]["distinct_workspaces"] == 2


@pytest.mark.parametrize(
    "schedule,kv_write,cores",
    [("joined", "branches", 0), ("post-overlap", "branches", 0), ("joined", "shared", 0), ("joined", "branches", 10)],
)
def test_full_decoder_padding_vs_split_parallel(tmp_path, request, schedule, kv_write, cores):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "model_scaling.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_model_scaling.py"),
            "--split-schedule",
            schedule,
            "--kv-write",
            kv_write,
            "--attention-cores",
            str(cores),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch",
            "7",
            "--capacity",
            "8",
            "--query",
            "8",
            "--kv-length",
            "128",
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=300,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert result["update_included"]
    assert result["split_schedule"] == schedule
    assert result["kv_write"] == kv_write
    assert result["attention_cores"] == cores
    assert not result["cross_architecture_qualified"]
    assert {row["mode"] for row in result["measurements"]} == {"padding", "split_parallel"}
    assert len(result["measurements"]) == 2
    assert result["tensor_parallel_size"] == 1
    for rank in result["ranks"]:
        assert all(check["passed"] for check in rank["checks"])
        assert len([check for check in rank["checks"] if check["name"].endswith("/changed_input")]) == 2
    graphs = {graph["mode"]: graph for graph in result["graphs"]}
    assert graphs["padding"]["update_tasks"] == result["layers"]
    assert graphs["split_parallel"]["update_tasks"] == 2 * result["layers"]
    assert graphs["split_parallel"]["workspaces"] == 2
