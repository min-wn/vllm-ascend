# SPDX-License-Identifier: Apache-2.0
"""Real greedy Q4 acceptance, state lengths and complete paired accounting."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from multi_graph_gdn_aime26 import (  # noqa: E402
    SCHEMA,
    GenerationRequest,
    digest,
    generation_config,
    load_manifest,
    paired_summary,
    terminal_metadata,
    verify_greedy,
)
from multi_graph_gdn_aime26_audit import HistoryNgrams  # noqa: E402
from multi_graph_gdn_aime26_score import score_all  # noqa: E402


@pytest.mark.parametrize(
    "drafts,predictions,expected",
    [
        ([], [7, 8, 9, 10], [7]),
        ([2, 3, 4], [8, 9, 10, 11], [8]),
        ([2, 3, 4], [2, 9, 10, 11], [2, 9]),
        ([2, 3, 4], [2, 3, 10, 11], [2, 3, 10]),
        ([2, 3, 4], [2, 3, 4, 11], [2, 3, 4, 11]),
        ([2], [2, 9, 10, 11], [2, 9]),
    ],
)
def test_verifier(drafts, predictions, expected):
    result = verify_greedy(drafts, predictions, remaining=10, eos=(99,))
    assert result["tokens"] == expected
    assert result["accepted"] == len(expected)


@pytest.mark.parametrize("stop,reason", [(3, "eos"), (99, "length")])
def test_stop_inside_accepted_draft(stop, reason):
    result = verify_greedy([2, 3, 4], [2, 3, 4, 8], remaining=2, eos=(stop,))
    assert result["tokens"] == [2, 3]
    assert result["accepted"] == 2
    assert result["finish_reason"] == reason


def test_lossless_greedy_continuity_and_rollback():
    # A small deterministic autoregressive model: next=(last+1)%19. Drafts
    # exercise every accepted length, a short proposal and no proposal.
    request = GenerationRequest(0, (5, 6), max_new_tokens=23, eos=(99,))
    request.begin(7)
    patterns = [0, 1, 2, 3, -1, -2]
    for step in range(30):
        if request.finish_reason:
            break
        last = request.tokens[-1]
        correct = [(last + j + 1) % 19 for j in range(4)]
        kind = patterns[step % len(patterns)]
        drafts = [] if kind == -1 else correct[:1] if kind == -2 else correct[:3]
        if kind in (0, 1, 2):
            drafts[kind] = 98
        supplied = request.input_tokens(drafts)
        predictions = [(token + 1) % 19 for token in supplied]
        request.advance(drafts, predictions)
        assert request.tokens == [(7 + j) % 19 for j in range(len(request.tokens))]
        request.require_consistent()
    assert request.finish_reason == "length"
    assert len(request.tokens) == 23
    assert request.processed == 24
    with pytest.raises(ValueError):
        request.advance([], [1, 2, 3, 4])


def test_initial_eos_and_exact_accounting():
    request = GenerationRequest(0, (5, 6), eos=(99,))
    request.begin(99)
    assert request.q4_steps == 0 and request.finish_reason == "eos"
    with pytest.raises(ValueError):
        request.begin(4)
    rows = [
        dict(
            index=i,
            **{
                v: dict(correct=(i % 4 in truth), finish_reason="eos", prediction_valid=True)
                for v, truth in [("official", (0, 1)), ("compact", (0, 2))]
            },
        )
        for i in range(30)
    ]
    summary = paired_summary(rows)
    assert [summary[k] for k in ("both_correct", "both_wrong", "official_only", "compact_only")] == [8, 7, 8, 7]
    with pytest.raises(ValueError):
        paired_summary(rows[:-1])


def test_terminal_native_lengths_are_not_cumulative_boundaries():
    result = terminal_metadata([1, 3, 4])
    assert result == dict(sequence=[0, 4, 4, 4], ids=[1] * 4 + [2] * 4 + [3] * 4, accepted=[1, 3, 4])
    for invalid in ([], [0], [5], [True], [2.0]):
        with pytest.raises(ValueError):
            terminal_metadata(invalid)


def test_manifest_rejects_answers_and_policy_changes(tmp_path):
    payload = dict(
        schema=SCHEMA,
        generation=generation_config(),
        inputs=[
            dict(index=i, id=str(i), question="Diagnostic", prompt="Diagnostic", prompt_tokens=[1, 2])
            for i in range(30)
        ],
    )
    path = tmp_path / "manifest.json"

    def write(value):
        path.write_text(json.dumps(dict(payload=value, payload_sha256=digest(value))), encoding="utf-8")

    write(payload)
    assert len(load_manifest(path)["payload"]["inputs"]) == 30
    changed = copy.deepcopy(payload)
    changed["inputs"][0]["target"] = "7"
    write(changed)
    with pytest.raises(ValueError, match="targets"):
        load_manifest(path)
    changed = copy.deepcopy(payload)
    changed["generation"]["temperature"] = 1
    write(changed)
    with pytest.raises(ValueError, match="configuration"):
        load_manifest(path)
    path.write_text(json.dumps(dict(payload=payload, payload_sha256="bad")), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(path)


def test_ngram_oracle_keeps_earliest_longest_match():
    history = HistoryNgrams([1, 2, 3, 4, 5, 1, 2])
    assert history.propose() == [3, 4, 5]
    history.append([3, 4])
    assert history.propose() == [5, 1, 2]
    empty = HistoryNgrams([7, 8])
    assert empty.propose() == []


def test_scoring_retains_missing_runtime_and_parser_failures(tmp_path, monkeypatch):
    manifest = dict(payload_sha256="frozen", payload=dict(inputs=[dict(index=i, id=str(i)) for i in range(30)]))
    targets = [dict(index=i, id=str(i), target="7") for i in range(30)]

    class ParserError(Exception):
        pass

    def rule_score(rules, text, target):
        if text == "parser_failure":
            raise ParserError("test unavailable scoring")
        return dict(matched=text == target, extracted=text, prediction_valid=True, reference_valid=True, reason="test")

    monkeypatch.setattr("multi_graph_gdn_aime26_score.score_output", rule_score)
    for index in range(30):
        for variant in ("official", "compact"):
            if index == 2 and variant == "compact":
                continue
            text = "wrong" if (index, variant) in ((0, "compact"), (1, "official")) else "7"
            if index == 4 and variant == "compact":
                text = "parser_failure"
            record = dict(
                manifest_sha256="frozen",
                index=index,
                scorer_text=text,
                finish_reason="runtime_error" if (index, variant) == (3, "official") else "eos",
                generated_token_count=10,
                q4_steps=5,
                fallback_calls=0,
            )
            (tmp_path / f"{variant}.question{index:02}.json").write_text(json.dumps(record), encoding="utf-8")
    result = score_all(manifest, targets, tmp_path, {"MathEvaluationError": ParserError})
    assert len(result["rows"]) == 30
    assert not result["actual_generation_complete_all"]
    assert result["summary"]["official"]["correct"] == 28
    assert result["summary"]["compact"]["correct"] == 27
    assert result["summary"]["official_only"] == 3
    assert result["summary"]["compact_only"] == 2
    assert result["summary"]["both_correct"] == 25
    assert result["summary"]["compact"]["parse_failures"] == 2
    assert result["rows"][4]["compact"]["reason"] == "scoring_error"
