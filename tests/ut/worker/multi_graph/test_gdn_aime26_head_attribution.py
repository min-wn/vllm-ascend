# SPDX-License-Identifier: Apache-2.0
"""Reject a cast-only or incomplete head control before making quality claims."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from multi_graph_gdn_aime26_head_attribution import answer_observation, audit_head  # noqa: E402


def head_record(dtype):
    event = dict(
        op="torch.nn.functional.linear",
        inputs=[dict(dtype=dtype), dict(dtype=dtype)],
        output=dict(dtype=dtype),
    )
    return dict(
        calls=4,
        fallback_calls=0,
        input_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        configured_head_dtype=dtype,
        output_dtype=dtype,
        allow_hf32=False,
        float32_matmul_precision="highest",
        audited_calls=[dict(rows=1, operations=[event]), dict(rows=4, operations=[event])],
    )


@pytest.mark.parametrize("dtype", ["torch.float32", "torch.bfloat16"])
def test_projection_control_has_real_operands_and_complete_lifetime(dtype):
    head = head_record(dtype)
    audit_head(head, dtype, 3)
    with pytest.raises(AssertionError):
        audit_head(head, dtype, 4)
    head["audited_calls"][0]["operations"][0]["inputs"][0]["dtype"] = "torch.float64"
    with pytest.raises(AssertionError):
        audit_head(head, dtype, 3)


def test_both_correct_does_not_claim_head_restored_quality():
    assert answer_observation(True, True) == "both_correct_no_observed_B1_answer_accuracy_gain"
    assert answer_observation(True, False) == "controlled_answer_improvement_on_this_one_question"
