"""Fixed-query target replay qualification; no drafter or acceptance sampling."""

import json
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
