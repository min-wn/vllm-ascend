# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or cast-only evidence before starting actual generation."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from multi_graph_gdn_aime26_fp32_head import (  # noqa: E402
    generation_gate,
    projection_audit,
    real_fp32_projection,
)
from multi_graph_gdn_aime26_fp32_head_analyze import emission_site  # noqa: E402


def evidence():
    sample = dict(
        bfloat16=dict(saved_logits_exact=True),
        float32=dict(
            fp32_operands_and_output=True,
            dtype="torch.float32",
            vocabulary=248320,
            argmax=866,
            finite=True,
            top1_top2_gap=0.03,
        ),
    )
    return [
        dict(
            rank=rank,
            weight_roundtrip_exact=True,
            weight_dtype="torch.bfloat16",
            samples={name: copy.deepcopy(sample) for name in ("official", "compact_ready")},
        )
        for rank in (0, 1)
    ]


def test_gated_complete_vocabulary_unique_winner():
    reports = evidence()
    assert generation_gate(reports)
    assert not generation_gate(reports[:1])
    reports[1]["rank"] = 0
    assert not generation_gate(reports)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fp32_operands_and_output", False),
        ("dtype", "torch.bfloat16"),
        ("vocabulary", 3),
        ("argmax", 660),
        ("finite", False),
        ("top1_top2_gap", 0),
    ],
)
def test_fail_closed_on_partial_scene_evidence(field, value):
    reports = evidence()
    reports[1]["samples"]["compact_ready"]["float32"][field] = value
    assert not generation_gate(reports)


def test_original_scene_and_weight_values_required():
    reports = evidence()
    reports[0]["samples"]["official"]["bfloat16"]["saved_logits_exact"] = False
    assert not generation_gate(reports)
    reports = evidence()
    reports[0]["weight_roundtrip_exact"] = False
    assert not generation_gate(reports)


def test_post_projection_cast_is_not_fp32_projection():
    fp32_tensor = dict(dtype="torch.float32")
    event = dict(op="aten.mm.default", inputs=[fp32_tensor, fp32_tensor], output=fp32_tensor)
    assert real_fp32_projection([event])
    event["inputs"] = [dict(dtype="torch.bfloat16")] * 2
    assert not real_fp32_projection([event])
    event["op"] = "aten._to_copy.default"
    assert not real_fp32_projection([event])


def test_operand_audit_delegates_without_caching_or_changing_results():
    import torch
    import torch.nn.functional as functional

    weight = torch.tensor([[1.25, 2.5], [3.75, -4.0]], dtype=torch.bfloat16)
    hidden = torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16)
    expected = functional.linear(hidden.float(), weight.float())
    original_to, original_linear = torch.Tensor.to, functional.linear
    with projection_audit(torch, weight) as audit:
        first = functional.linear(hidden.to(torch.float32), weight.to(torch.float32))
        second = functional.linear(hidden.to(torch.float32), weight.to(torch.float32))
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert torch.Tensor.to is original_to and functional.linear is original_linear
    assert real_fp32_projection(audit.operations)
    conversions = [op for op in audit.operations if op["op"] == "lm_head.weight.to"]
    assert len(conversions) == 2
    assert all(op["allocated_new_storage"] for op in conversions)


def test_emission_position_uses_actual_verified_prefix():
    journal = [
        dict(
            step=0,
            batch=1,
            question_indices=[27],
            request_ids=[0],
            seq_lengths=[173],
            accepted_from_previous=[1],
            proposals=[[20, 21, 22]],
            predictions=[20, 21, 99, 0],
            decisions=[dict(tokens=[20, 21, 99], accepted=3)],
        )
    ]
    site = emission_site(iter(journal), output_offset=3, prompt_length=169)
    assert site["query_offset"] == 2
    assert site["predictor_position"] == 171
    assert site["output_position"] == 172
    assert emission_site([], 0, 169)["predictor_position"] == 168
    with pytest.raises(ValueError, match="not covered"):
        emission_site(iter(journal), 4, 169)
