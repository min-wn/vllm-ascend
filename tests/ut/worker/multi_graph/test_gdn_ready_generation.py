# SPDX-License-Identifier: Apache-2.0
"""CPU checks for frozen inputs, speculative capacity and paired-work rejection."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
spec = importlib.util.spec_from_file_location(
    "ready_generation_protocol", BENCHMARKS / "multi_graph_gdn_ready_generation.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 2, *messages[0]["content"].encode()]


def test_frozen_text_and_speculative_tail_fit(tmp_path):
    document = protocol.freeze(Tokenizer(), {})
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert protocol.load_manifest(path) == document
    trace = protocol.capacity_trace(document)
    assert [step["batch"] for step in trace.steps] == list(range(1, 57))
    assert len(set(r["text"] for r in document["payload"]["inputs"])) == 4
    for row, length in zip(document["payload"]["inputs"], trace.steps[-1]["seq_lengths"], strict=True):
        assert length == len(row["prompt_tokens"]) + 256 + 4 <= 2048


def test_frozen_inputs_cannot_be_silently_changed(tmp_path):
    document = protocol.freeze(Tokenizer(), {})
    document["payload"]["inputs"][0]["prompt_tokens"][0] += 1
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol changed"):
        protocol.load_manifest(path)


def test_three_pairs_alternate_order():
    assert [(r["pair"], r["mode"]) for r in protocol.run_plan()] == [
        (0, "delayed"),
        (0, "ready"),
        (1, "ready"),
        (1, "delayed"),
        (2, "delayed"),
        (2, "ready"),
    ]


def reports(mode):
    ranks = [
        dict(rank=i, prefill_us=10.0, state_import_us=2.0, generation_us=100.0, terminal_us=3.0, batch_us=115.0)
        for i in (0, 1)
    ]
    shared = dict(
        status=protocol.COMPLETE,
        mode=mode,
        timed_matches_validation=True,
        fallback_calls=0,
        cost_ranks=ranks,
        critical_cost=protocol.critical_cost(ranks),
        workload_sha256="frozen",
        selection={"default": True},
        loader={"TP": 2},
        dense_weights=[29],
        head="BF16",
        initial_state="same",
        validation={"closed_state": "same"},
        generation={"tokens": [1, 2]},
        terminal_contract={"accepted": [1]},
        kernel_identities={"kernel": "same"},
    )
    return [dict(copy.deepcopy(shared), rank=rank) for rank in (0, 1)]


def test_identical_generation_is_accepted():
    assert protocol.check_pair(reports("delayed"), reports("ready"))


@pytest.mark.parametrize("key", ["generation", "validation", "initial_state", "head", "selection", "terminal_contract"])
def test_work_or_state_changes_block_timing_comparison(key):
    delayed, ready = reports("delayed"), reports("ready")
    ready[0][key] = "changed"
    with pytest.raises(ValueError, match="identical work and state"):
        protocol.check_pair(delayed, ready)


def test_missing_lifecycle_cost_is_rejected():
    rows = reports("ready")[0]["cost_ranks"]
    rows[0]["batch_us"] -= rows[0]["terminal_us"]
    with pytest.raises(ValueError, match="all four"):
        protocol.critical_cost(rows)


def analysis_fixture(root):
    (root / "protocol.json").write_text(json.dumps(dict(plan=protocol.run_plan())), encoding="utf-8")
    generation = dict(
        output_tokens=2,
        loop_output_tokens=1,
        requests=[dict(tokens=[9, 248046], finish_reason="eos")],
        steps=[dict(proposals=[[]], decisions=[dict(accepted=1, matched_drafts=0)])],
    )
    for item in protocol.run_plan():
        path = root / f"pair{item['pair']}.{item['mode']}"
        path.mkdir()
        rows = reports(item["mode"])
        for row in rows:
            row["generation"] = copy.deepcopy(generation)
            if item["mode"] == "ready":
                for rank in row["cost_ranks"]:
                    rank.update(generation_us=90.0, batch_us=105.0)
                row["critical_cost"] = protocol.critical_cost(row["cost_ranks"])
            (path / f"result.rank{row['rank']}.json").write_text(json.dumps(row), encoding="utf-8")


def test_generation_throughput_excludes_prefill_token_and_q4_candidates(tmp_path):
    from multi_graph_gdn_ready_generation_analyze import analyze

    analysis_fixture(tmp_path)
    result = analyze(tmp_path)
    assert result["stable_generation_benefit"]
    assert result["generation_gain_percent"] == pytest.approx(10.0)
    assert result["pairs"][0]["loop_output_tokens_per_second"]["ready"] == pytest.approx(1e6 / 90)
    assert result["pairs"][0]["batch_output_tokens_per_second"]["ready"] == pytest.approx(2e6 / 105)


def test_repetitions_with_different_emitted_work_cannot_be_averaged(tmp_path):
    from multi_graph_gdn_ready_generation_analyze import analyze

    analysis_fixture(tmp_path)
    for mode in protocol.MODES:
        for rank in (0, 1):
            path = tmp_path / f"pair1.{mode}" / f"result.rank{rank}.json"
            row = json.loads(path.read_bytes())
            row["generation"]["requests"][0]["tokens"][0] += 1
            path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="work changed between"):
        analyze(tmp_path)
