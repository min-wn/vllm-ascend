# SPDX-License-Identifier: Apache-2.0
"""Reject confounded, unclosed and numerically different scheduling pairs."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    module = importlib.import_module("multi_graph_gdn_ready_chain_analyze")
    protocol = importlib.import_module("multi_graph_gdn_ready_chain_protocol")
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            dict(plan=protocol.run_plan(), workloads={name: dict(sha256=name, steps=1) for name in protocol.LOADS})
        )
    )
    for item in protocol.run_plan():
        directory = tmp_path / f"{item['load']}.pair{item['pair']}.{item['mode']}"
        directory.mkdir()
        wall = 100.0 if item["mode"] == "delayed" else 90.0
        costs = [
            dict(
                rank=rank, execution_us=wall + rank, initialization_us=10.0, terminal_us=1.0, total_us=wall + rank + 11
            )
            for rank in (0, 1)
        ]
        for rank in (0, 1):
            row = dict(
                rank=rank,
                mode=item["mode"],
                load=item["load"],
                status=module.COMPLETE,
                timed_final_state_exact=True,
                head=dict(weight_dtype="torch.bfloat16", output_dtype="torch.bfloat16", actual_projection_audited=True),
                head_precision_changes=False,
                trace_sha256=item["load"],
                correctness=dict(output_hashes=[dict(step=0, fields=["h", "r", "l"])], closed_state="state"),
                critical_cost=protocol.critical_cost(costs),
                cost_ranks=costs,
                prefix=dict(token_sha256="same", elapsed_seconds=1 + rank + (item["mode"] == "ready")),
                selection=dict(required_tokens=[4], dispatched_tokens=[4]),
                load_contract="same",
                terminal_contract="same",
                initial_state_digest="same",
                dense_weights="same",
                loader="same",
                capture=dict(required_default_buckets=[4]),
            )
            (directory / f"result.rank{rank}.json").write_text(json.dumps(row))
    return module, tmp_path


def test_complete_same_algorithm_pairs_keep_quality_claim_separate(evidence):
    module, path = evidence
    result = module.analyze(path)
    assert result["exact_same_algorithm_pair_audit"] and not result["quality_claim"]
    assert len(result["pairs"]) == 6
    assert all(row["both_pairs_benefit"] for row in result["loads"])
    assert result["loads"][0]["closed_percent"] == pytest.approx(100 * (1 - 102 / 112))


@pytest.mark.parametrize("bad", ["state", "outputs", "head", "selection", "terminal", "total", "length", "status"])
def test_rejects_invalid_pairs_including_unmeasured_terminal_cost(evidence, bad):
    module, path = evidence
    target = path / "original.pair0.ready/result.rank1.json"
    row = json.loads(target.read_text())
    if bad == "state":
        row["correctness"]["closed_state"] = "different"
    elif bad == "outputs":
        row["correctness"]["output_hashes"][0]["fields"][2] = "different"
    elif bad == "head":
        row["head"]["output_dtype"] = "torch.float32"
    elif bad == "selection":
        row["selection"]["required_tokens"] = [8]
    elif bad == "terminal":
        row["terminal_contract"] = "different"
    elif bad == "total":
        row["cost_ranks"][0]["total_us"] -= 1
    elif bad == "length":
        row["correctness"]["output_hashes"] = []
    else:
        row["status"] = "unfinished"
    target.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        module.analyze(path)
