# SPDX-License-Identifier: Apache-2.0
"""Quality evaluation must not hide rejected-token duplication or tail loss."""

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def quality(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    return importlib.import_module("multi_graph_gdn_quality")


def test_next_token_alignment_handles_reorder_pause_and_terminal_acceptance(quality):
    trace = SimpleNamespace(
        initial_prefix_by_request=(3, 7),
        steps=(
            dict(step=0, request_ids=(1, 0), seq_lengths=(11, 7), accepted_after_step=(2, 3)),
            dict(step=1, request_ids=(0,), seq_lengths=(10,), accepted_after_step=(4,)),
            dict(step=2, request_ids=(1,), seq_lengths=(13,), accepted_after_step=(4,)),
        ),
    )
    layout = quality.score_layout(trace)
    assert [(r["owner"], r["position"], r["flat"]) for r in layout[0]] == [
        (1, 8, 0),
        (1, 9, 1),
        (0, 4, 4),
        (0, 5, 5),
        (0, 6, 6),
    ]
    assert [(r["owner"], r["position"]) for r in layout[1]] == [(0, 7)]
    assert [(r["owner"], r["position"]) for r in layout[2]] == [(1, 10), (1, 11)]
    trace.steps[2]["seq_lengths"] = (12,)
    with pytest.raises(ValueError, match="duplicate or missing"):
        quality.score_layout(trace)


def test_full_vocab_distributions_and_target_shift(quality):
    p = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float32)
    q = torch.tensor([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6]], dtype=torch.float32)
    result = quality.distribution_metrics(torch, p.log(), q.log(), torch.tensor([1, 2]))
    assert result["official_nll"] == pytest.approx([-math.log(0.3), -math.log(0.7)])
    assert result["compact_nll"] == pytest.approx([-math.log(0.4), -math.log(0.6)])
    assert result["tv"] == pytest.approx([0.1, 0.1], abs=1e-6)
    assert result["kl"][0] == pytest.approx(0.6 * math.log(1.2) + 0.3 * math.log(0.75), abs=1e-6)
    assert result["official_correct"] == [0, 1]
    assert result["top1_disagreement"] == [0, 0]
    with pytest.raises(ValueError, match="full-vocabulary"):
        quality.distribution_metrics(torch, p, q[:1], torch.tensor([1, 2]))
    p[0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        quality.distribution_metrics(torch, p, q, torch.tensor([1, 2]))


def observations(delta=0):
    rows = []
    for owner in range(128):
        for position in range(1 + owner % 3):
            rows.append(
                dict(
                    owner=owner,
                    position=position,
                    article=owner // 4,
                    official_nll=1 + owner / 128,
                    compact_nll=1 + owner / 128 + delta,
                    delta_nll=delta,
                    kl=0,
                    js=0,
                    tv=0,
                    top1_disagreement=0,
                    official_correct=1,
                    compact_correct=1,
                )
            )
    return rows


def test_ppl_uses_token_weighting_and_article_paired_uncertainty(quality):
    rows = observations(0.001)
    result = quality.summarize(rows, expected_tokens=len(rows))
    assert result["screen_passed"]
    assert result["ppl_ratio"] == pytest.approx(math.exp(0.001))
    assert result["ppl_ratio_upper95"] == pytest.approx(math.exp(0.001))
    assert result["source_articles"] == 32
    assert result["official_ppl"] == pytest.approx(math.exp(sum(r["official_nll"] for r in rows) / len(rows)))


def test_quality_gate_rejects_degradation_and_missing_or_duplicate_tokens(quality):
    rows = observations(0.02)
    assert not quality.summarize(rows, expected_tokens=len(rows))["gates"]["ppl_noninferiority"]
    with pytest.raises(ValueError, match="incomplete"):
        quality.summarize(rows, expected_tokens=len(rows) + 1)
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        quality.summarize(rows, expected_tokens=len(rows))


def test_quality_gate_cannot_hide_one_bad_owner_in_global_mean(quality):
    rows = observations()
    for row in rows:
        if row["owner"] == 9:
            row["delta_nll"] = 0.06
            row["compact_nll"] += 0.06
    result = quality.summarize(rows, expected_tokens=len(rows))
    assert result["gates"]["ppl_noninferiority"]
    assert not result["gates"]["owner_tail"]
    assert not result["screen_passed"]


def test_input_windows_never_cross_or_reuse_articles(quality):
    prep = importlib.import_module("multi_graph_gdn_quality_inputs")
    assert prep.articles_from_rows([" = A = \n", "body\n", " = = Section = = \n", "more\n", " = B = \n", "end\n"]) == [
        " = A = \nbody\n = = Section = = \nmore\n",
        " = B = \nend\n",
    ]
    windows = prep.allocate_windows(["0123456789", "abcdefghij"], [4, 4, 5], lambda x: list(x))
    assert [w["tokens"] for w in windows] == [list("0123"), list("4567"), list("abcde")]
    assert [w["article"] for w in windows] == [0, 0, 1]
    with pytest.raises(ValueError, match="not enough"):
        prep.allocate_windows(["abc"], [2, 2], list)


def test_task_requires_every_word_token_and_reports_paired_new_failures(quality):
    probe = importlib.import_module("multi_graph_gdn_quality_probe")
    rows = observations()
    result = probe.task_summary(rows)
    assert result["passed"] and result["compact_correct"] == 128
    for owner in range(4):
        next(r for r in rows if r["owner"] == owner)["compact_correct"] = False
    result = probe.task_summary(rows)
    assert result["new_failures"] == 4
    assert not result["passed"]


def test_manifest_rejects_changed_protocol_limits_and_owner_order(quality, tmp_path):
    probe = importlib.import_module("multi_graph_gdn_quality_probe")
    import hashlib
    import json

    protocol = tmp_path / "protocol.md"
    protocol.write_text("predeclared")
    manifest = tmp_path / "manifest.json"
    data = dict(
        schema=quality.SCHEMA,
        trace_sha256="trace",
        limits=quality.LIMITS,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        screen=[dict(owner=i) for i in range(128)],
        task=[dict(owner=i) for i in range(128)],
    )
    manifest.write_text(json.dumps(dict(payload=data, payload_sha256=quality.digest(data))))
    probe.load_inputs(manifest, protocol, SimpleNamespace(digest="trace"))
    protocol.write_text("changed after results")
    with pytest.raises(ValueError, match="protocol changed"):
        probe.load_inputs(manifest, protocol, SimpleNamespace(digest="trace"))


@pytest.fixture
def saved_quality_audit(quality, monkeypatch, tmp_path):
    import hashlib
    import json

    audit = importlib.import_module("multi_graph_gdn_quality_analyze")
    monkeypatch.setattr(quality, "BOOTSTRAP_DRAWS", 20)
    trace = SimpleNamespace(
        initial_prefix_by_request=(2,),
        steps=tuple(dict(step=i, request_ids=(0,), seq_lengths=(6 + i,), accepted_after_step=(1,)) for i in range(128)),
    )
    source = dict(tokens=list(range(200)), article=0)
    frozen = dict(payload=dict(screen=[source], screen_expected_tokens=128), payload_sha256="frozen")
    monkeypatch.setattr(audit, "load_fixed_trace", lambda _: trace)
    monkeypatch.setattr(audit, "load_inputs", lambda *args: frozen)
    (tmp_path / "manifest.json").write_text("{}")
    rows = []
    for step, group in enumerate(quality.score_layout(trace)):
        for item in group:
            rows.append(
                dict(
                    step=step,
                    owner=0,
                    position=item["position"],
                    article=0,
                    target=source["tokens"][item["position"]],
                    official_nll=1.0,
                    compact_nll=1.0,
                    delta_nll=0.0,
                    kl=0.0,
                    js=0.0,
                    tv=0.0,
                    top1_disagreement=0.0,
                    official_correct=1.0,
                    compact_correct=1.0,
                    official_margin=0.1,
                )
            )
    for rank in (0, 1):
        report = dict(
            status="complete_real_text_quality_audit",
            rank=rank,
            completed_steps=128,
            terminal_closed=True,
            payload_sha256="frozen",
            manifest_sha256=hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest(),
            prefill={},
            selection={},
            rows=rows if rank == 0 else [],
            hidden_diagnostics=[dict(worst_request_nrmse=0, screen_passed=True) for _ in range(128)],
        )
        if rank == 0:
            report["summary"] = quality.summarize(rows, expected_tokens=128)
        (tmp_path / f"screen.rank{rank}.json").write_text(json.dumps(report))
    return audit, tmp_path


def test_independent_audit_rejects_wrong_next_token_and_unclosed_peer(saved_quality_audit):
    import json

    audit, root = saved_quality_audit
    result = audit.analyze(root)
    assert result["status"] == "audited_real_text_quality"
    assert not result["summary"]["screen_passed"]  # This miniature fixture has only one owner/article.
    assert result["paired_target_accuracy"]["both_correct"] == 128
    assert result["paired_target_accuracy"]["official_correct_compact_wrong"] == 0
    assert len(result["owners"]) == 1
    path = root / "screen.rank0.json"
    data = json.loads(path.read_bytes())
    data["rows"][0]["target"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="misalignment"):
        audit.analyze(root)
    data["rows"][0]["target"] -= 1
    path.write_text(json.dumps(data))
    peer = root / "screen.rank1.json"
    data = json.loads(peer.read_bytes())
    data["terminal_closed"] = False
    peer.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete paired-rank"):
        audit.analyze(root)


def test_independent_audit_rejects_modified_acceptance_result(saved_quality_audit):
    import json

    audit, root = saved_quality_audit
    path = root / "screen.rank0.json"
    data = json.loads(path.read_bytes())
    data["summary"]["screen_passed"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not reproduce"):
        audit.analyze(root)


def test_cross_platform_derived_rounding_never_changes_gates(quality):
    audit = importlib.import_module("multi_graph_gdn_quality_analyze")
    assert audit.same_statistics(dict(value=1.0 + 1e-14, passed=False), dict(value=1.0, passed=False))
    assert not audit.same_statistics(dict(value=1.0 + 1e-5, passed=False), dict(value=1.0, passed=False))
    assert not audit.same_statistics(dict(value=1.0, passed=True), dict(value=1.0, passed=False))
    assert not audit.same_statistics(True, 1)
