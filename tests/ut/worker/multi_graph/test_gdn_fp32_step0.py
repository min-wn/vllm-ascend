# SPDX-License-Identifier: Apache-2.0
"""Attribution must preserve owner order, residual snapshots and round bounds."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("fp32_step0", directory / "multi_graph_gdn_fp32_step0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_head_flattening_preserves_nonsequential_owner_order_and_excludes_padding(probe):
    reference = torch.ones(12 * 3, 7)
    actual = reference.clone().reshape(12, 3, 7)
    actual[4:8] *= 1.01
    actual[8:] = 1000
    step = dict(batch=2, request_ids=[91, 12])
    row = probe.token_error(torch, actual.reshape_as(reference), reference, label="gated_norm", capacity=12, step=step)
    assert row["worst_owner"] == 12
    assert row["worst_request_nrmse"] == pytest.approx(0.01)
    assert row["differing_elements"] == 4 * 3 * 7
    assert not row["screen_passed"]
    with pytest.raises(ValueError, match="token axis"):
        probe.token_error(torch, actual[0], reference, label="bad", capacity=12, step=step)


def test_observer_takes_immutable_copies_before_residual_is_modified(probe):
    observer = probe.StepObserver(torch, dict(batch=1, request_ids=[55]), 4)
    residual = torch.ones(4, 3)
    observer.observe("residual", residual)
    residual.add_(10)
    observer.phase = "compact"
    observer.observe("residual", torch.ones(4, 3))
    assert observer.errors[0]["bitwise_identical"]
    observer.observe("residual", residual)
    assert not observer.errors[1]["screen_passed"]
    with pytest.raises(RuntimeError, match="unpaired"):
        observer.observe("wrong_boundary", residual)


def test_site_selection_is_bounded_and_uses_the_preceding_gdn_for_a_fia_block(probe):
    def row(label, different=1, passed=True):
        return dict(label=label, differing_elements=different, screen_passed=passed)

    rows = [row("layer0/recurrence/output"), row("layer3/decoder/0", passed=False)]
    sites = probe.select_sites(rows, [0, 1, 2, 4, 5, 6])
    assert sites["local_layer_indices"] == [0, 2]
    assert sites["first_failed_decoder"]["label"] == "layer3/decoder/0"
    assert probe.select_sites([row("layer0/recurrence/output", different=0)], [0])["local_layer_indices"] == []


def test_step_zero_cli_rejects_partial_prefix_and_changed_weight_format(probe, monkeypatch, tmp_path):
    import multi_graph_gdn_q4_full_probe as full

    base = [
        "probe",
        "--phase",
        "fp32-step0",
        "--extension",
        "log.so",
        "--products-extension",
        "products.so",
        "--deferred-extension",
        "deferred.so",
        "--tile-extension",
        "tile.so",
        "--output",
        str(tmp_path / "out.json"),
    ]
    for flags in (["--prefix-limit", "1", "--dense-nz"], []):
        monkeypatch.setattr("sys.argv", base + flags)
        with pytest.raises(SystemExit) as error:
            full.main()
        assert error.value.code == 2


def test_offline_dense_and_compact_share_real_arithmetic_without_a_new_precision_mode(probe):
    from multi_graph_gdn_fp32_arithmetic import compact_recurrence, dense_recurrence

    generator = torch.Generator().manual_seed(721)
    state = torch.randn((3, 2, 8, 8), generator=generator, dtype=torch.float64)
    q, k, v = (torch.randn((3, 4, 2, 8), generator=generator, dtype=torch.float64) / 4 for _ in range(3))
    g = -torch.rand((3, 4, 2), generator=generator, dtype=torch.float64)
    beta = torch.rand((3, 4, 2), generator=generator, dtype=torch.float64)
    original = state.clone()
    reference = dense_recurrence(torch, state, q, k, v, g, beta, scale=8**-0.5)
    candidate = compact_recurrence(torch, state, q, k, v, g, beta, scale=8**-0.5)
    torch.testing.assert_close(candidate, reference, atol=1e-14, rtol=1e-13)
    assert torch.equal(state, original)


def test_bf16_ulp_distance_orders_negative_values_and_both_signed_zeros(probe):
    from multi_graph_gdn_fp32_arithmetic import ulp_distance

    low = torch.tensor([-1.0, -0.0, 0.0, 1.0], dtype=torch.bfloat16)
    high = torch.nextafter(low, torch.full_like(low, float("inf")))
    assert torch.equal(ulp_distance(torch, low, high), torch.ones(4, dtype=torch.int32))
    assert ulp_distance(torch, torch.tensor([-0.0], dtype=torch.bfloat16), low[2:3]).item() == 0
    with pytest.raises(ValueError, match="BF16"):
        ulp_distance(torch, low.float(), high.float())


def test_saved_evidence_keeps_candidate_closed_and_rejects_relaxed_threshold(probe):
    import copy
    import json

    from multi_graph_gdn_fp32_analyze import analyze, summarize

    directory = Path(__file__).resolve().parents[4] / "benchmarks/results/multi_graph_structural"
    evidence, original = directory / "gdn-fp32-step0-v1", directory / "gdn-ready-full-v1"
    summary = analyze(evidence, original)
    assert not summary["numerical_qualification_passed"]
    assert not summary["proceed_to_full_numeric_requalification"]
    assert not summary["actual_repair_performed"]
    assert summary["performance_rounds"] == 0
    assert all(rank["observation_records"] == 1408 for rank in summary["ranks"])
    report = json.loads((evidence / "step0.rank0.json").read_text(encoding="utf-8"))
    reference = json.loads((original / "numeric/numeric.rank0.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(report)
    bad["step0_contract"]["screen_nrmse"] = 0.02
    with pytest.raises(ValueError, match="threshold"):
        summarize(bad, reference)
    bad = copy.deepcopy(report)
    bad["step0_single_substitution"]["replacements"] = 2
    with pytest.raises(ValueError, match="budget"):
        summarize(bad, reference)
    bad = copy.deepcopy(report)
    row = next(r for r in bad["step0_local"][1]["errors"] if r["label"].endswith("/recurrence/input/query"))
    row["bitwise_identical"] = False
    with pytest.raises(ValueError, match="input mismatch"):
        summarize(bad, reference)
