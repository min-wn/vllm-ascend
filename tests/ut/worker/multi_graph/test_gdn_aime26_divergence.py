# SPDX-License-Identifier: Apache-2.0
"""Check emission-to-Q4 alignment without a model or device."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from multi_graph_gdn_aime26 import GenerationRequest  # noqa: E402
from multi_graph_gdn_aime26_divergence import common_prefix, reconstruct  # noqa: E402


@pytest.mark.parametrize("left,right,expected", [([], [], 0), ([1], [1, 2], 1), ([1, 2], [1, 3], 1), ([2], [1], 0)])
def test_longest_common_prefix(left, right, expected):
    assert common_prefix(left, right) == expected


def fixture():
    request = GenerationRequest(27, (10, 11))
    request.begin(12)
    decision = request.advance([13, 14, 15], [13, 14, 99, 100])
    step = dict(
        step=0,
        batch=1,
        request_ids=[0],
        question_indices=[27],
        seq_lengths=[6],
        accepted_from_previous=[1],
        proposals=[[13, 14, 15]],
        predictions=[13, 14, 99, 100],
        decisions=[decision],
    )
    return [dict(index=27, prompt_tokens=[10, 11])], {27: dict(generated_tokens=[12, -999])}, [step]


def test_verification_offset_is_not_step_or_next_input_position():
    rows, generations, journal = fixture()
    replay, site = reconstruct(rows, generations, journal, 4, target_offset=3)
    assert replay[0]["input_ids"] == [12, 13, 14, 15]
    assert site["verification_offset"] == 2
    assert site["predictor_position"] == 4
    assert site["generated_token_position"] == 5
    assert site["accepted_this_step"] == 3
    # The poison future token was never read to reconstruct the prefix.
    assert -999 not in replay[0]["input_ids"]


@pytest.mark.parametrize("field,value", [("seq_lengths", [7]), ("accepted_from_previous", [2])])
def test_reject_history_that_cannot_produce_logged_metadata(field, value):
    rows, generations, journal = fixture()
    changed = copy.deepcopy(journal)
    changed[0][field] = value
    with pytest.raises(AssertionError):
        reconstruct(rows, generations, changed, 4, target_offset=3)


def test_reject_uncovered_emission():
    rows, generations, journal = fixture()
    with pytest.raises(ValueError, match="not covered"):
        reconstruct(rows, generations, journal, 4, target_offset=4)


def test_fp64_state_oracle_and_native_log_layout():
    import torch
    from multi_graph_gdn_aime26_divergence_analyze import dense_prefixes, represented_prefixes
    from multi_graph_gdn_fp32_arithmetic import dense_recurrence

    generator = torch.Generator().manual_seed(28)
    state = torch.randn(1, 2, 3, 5, generator=generator, dtype=torch.float64)
    q, k = [torch.randn(1, 4, 2, 5, generator=generator, dtype=torch.float64) for _ in range(2)]
    value = torch.randn(1, 4, 2, 3, generator=generator, dtype=torch.float64)
    gate = -torch.rand(1, 4, 2, generator=generator, dtype=torch.float64)
    beta = torch.rand(1, 4, 2, generator=generator, dtype=torch.float64)
    output, prefixes = dense_prefixes(torch, state, q, k, value, gate, beta, 0.5)
    assert torch.equal(output, dense_recurrence(torch, state, q, k, value, gate, beta, scale=0.5))
    assert torch.equal(output[:, -1], (prefixes[:, -1] * (q[:, -1] * 0.5)[..., None, :]).sum(-1))

    # A single nonzero (v,k) cell makes offsets, decay ordering and the valid
    # guard independently observable, rather than comparing two implementations.
    base = torch.zeros(2, 1, 128, 128)
    base[:, 0, 7, 9] = 4
    logs = torch.zeros(2, 1, 1056)
    for token in range(4):
        logs[:, 0, token * 128 + 9] = 1
        logs[:, 0, 512 + token * 128 + 7] = 1
        logs[:, 0, 1024 + token * 8 : 1032 + token * 8] = 0.5
    valid = torch.zeros(2, 1, 8, dtype=torch.int32)
    valid[0] = 1
    represented = represented_prefixes(torch, (base, logs, valid), torch.float64)
    expected = torch.tensor([[3, 2.5, 2.25, 2.125], [4, 4, 4, 4]], dtype=torch.float64)
    assert torch.equal(represented[:, :, 0, 7, 9], expected)
    assert represented.count_nonzero() == 8


def test_pooled_error_includes_a_rank_with_zero_error():
    import torch
    from multi_graph_gdn_aime26_divergence_analyze import error, pool_errors

    reference = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    actual = reference.clone()
    actual[-1] += 0.5
    pooled = pool_errors([error(torch, actual[:2], reference[:2]), error(torch, actual[2:], reference[2:])])
    expected = error(torch, actual, reference)
    assert pooled["nrmse"] == pytest.approx(expected["nrmse"], rel=1e-14)
    assert pooled["rmse"] == pytest.approx(expected["rmse"], rel=1e-14)
