import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("batch_size", [6, 7, 8])
@pytest.mark.parametrize("kv_cache_layout", ["default", "contiguous"])
def test_multi_graph_tokens_and_selected_logprobs(tmp_path, dtype, batch_size, kv_cache_layout):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU and the supported vLLM/torch-npu environment")
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks" / "multi_graph_split.py"),
            "--dtype",
            dtype,
            "--batch-size",
            str(batch_size),
            "--repeats",
            "2",
            "--warmups",
            "1",
            "--output-tokens",
            "16",
            "--kv-cache-layout",
            kv_cache_layout,
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        timeout=2400,
    )
    results = {
        mode: json.loads((tmp_path / f"{mode}.json").read_text(encoding="utf-8"))
        for mode in ("eager", "single", "serial", "parallel")
    }
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["measurements"]["generation"]["correctness"]["all_repeats_checked"]
    assert len({result["resources"]["kv_blocks"] for result in results.values()}) == 1
    assert all(result["prefix_cache"] == "off" for result in results.values())
    baseline = results["single"]["requests"]
    eager = results["eager"]["requests"]
    for mode, result in results.items():
        assert len(result["requests"]) == batch_size
        if mode in ("serial", "parallel"):
            assert result["split_steps"] > 0
        for reference, actual, eager_reference in zip(baseline, result["requests"], eager):
            assert actual["tokens"] == reference["tokens"]
            baseline_error = max(
                abs(first - second)
                for first, second in zip(reference["selected_logprobs"], eager_reference["selected_logprobs"])
            )
            torch.testing.assert_close(
                torch.tensor(actual["selected_logprobs"]),
                torch.tensor(reference["selected_logprobs"]),
                rtol=0,
                atol=baseline_error + torch.finfo(getattr(torch, dtype)).eps,
            )
