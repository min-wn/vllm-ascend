import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def boundary(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("tp_boundary", directory / "multi_graph_tp_boundary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_failure_preserves_each_rank_without_overwriting_other_rank(boundary, monkeypatch, tmp_path, rank):
    monkeypatch.setenv("RANK", str(rank))
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        output=output,
        model="test-model",
        tensor_parallel_size=4,
        selection={"capacity_tokens": 400, "pair_tokens": [272, 128]},
        loader="qwen35-weights",
        numerical_diagnostics=[{"rank": rank, "mismatches": 7}],
    )
    boundary.write_failure(args, AssertionError("split_serial/initial: projection mismatch"))
    result = json.loads(output.with_suffix(f".rank{rank}.failure.json").read_text(encoding="utf-8"))
    assert result["rank"] == rank
    assert result["status"] == "failed"
    assert result["error_type"] == "AssertionError"
    assert result["error"] == "split_serial/initial: projection mismatch"
    assert result["numerical_diagnostics"] == args.numerical_diagnostics
    assert result["selection"] == args.selection
    assert len(result["source_sha256"]) == 64
    if rank == 0:
        assert json.loads(output.read_text(encoding="utf-8")) == result
    else:
        assert not output.exists()
