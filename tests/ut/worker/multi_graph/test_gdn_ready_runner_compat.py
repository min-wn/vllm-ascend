# SPDX-License-Identifier: Apache-2.0
"""Reject workload and provenance confounders in the runner compatibility audit."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from multi_graph_gdn_ready_runner_compat import conversion_scenario, summarize_ledger, summarize_metadata


def step(index, owners, drafts, accepted, capacity):
    return dict(
        step=index,
        request_ids=owners,
        batch=len(owners),
        proposals=[[10] * count for count in drafts],
        accepted_from_previous=accepted,
        capacity_tokens=capacity,
    )


@pytest.fixture
def ledger():
    return dict(
        workload_sha256="frozen",
        generation=dict(steps=[step(0, [7, 2], [0, 3], [4, 1], 16), step(1, [2], [0], [4], 4)]),
    )


@pytest.fixture
def metadata():
    return dict(
        model_executed=False,
        native_generation_observation=False,
        physical_block_ids_observed=False,
        workload_sha256="frozen",
        effective=dict(capture_sizes=[4, 8, 16]),
        native_allocation_budget=dict(cpu_bookkeeping_only=True),
        steps=[
            dict(
                step=0,
                owners=[7, 2],
                draft_counts=[0, 3],
                query_lengths=[1, 4],
                head_rows=5,
                capacity=8,
                graph_mode="PIECEWISE",
                gdn=dict(
                    num_spec=1,
                    num_spec_tokens=4,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_prefills=1,
                    num_prefill_tokens=1,
                ),
            ),
            dict(
                step=1,
                owners=[2],
                draft_counts=[0],
                query_lengths=[1],
                head_rows=1,
                capacity=4,
                graph_mode="PIECEWISE",
                gdn=dict(
                    num_spec=0,
                    num_spec_tokens=0,
                    num_decodes=1,
                    num_decode_tokens=1,
                    num_prefills=0,
                    num_prefill_tokens=0,
                ),
            ),
        ],
    )


def test_candidates_filler_and_graph_padding_have_distinct_counts(ledger):
    counts = summarize_ledger(ledger)
    assert counts["real_proposed_candidates"] == 3
    assert counts["q4_filler"] == 6
    assert counts["logical_q4_rows"] == 12
    assert counts["graph_padding_rows"] == 8
    assert counts["executed_q4_graph_rows"] == 20
    assert counts["native_unpadded_rows"] == 6
    assert counts["native_query_lengths"] == {1: 2, 4: 1}
    assert counts["native_generation_observation"] is False


def test_previous_acceptance_is_not_current_query_length(ledger, metadata):
    alternate = copy.deepcopy(ledger)
    for row in alternate["generation"]["steps"]:
        row["accepted_from_previous"] = [1] * row["batch"]
    assert summarize_ledger(alternate)["native_query_lengths"] == summarize_ledger(ledger)["native_query_lengths"]
    assert summarize_metadata(metadata, ledger)["total_head_rows"] == 6
    assert summarize_metadata(metadata, alternate)["total_head_rows"] == 6


def test_owner_identity_survives_pause_and_row_reordering():
    rows = [
        step(0, [9], [0], [1], 4),
        step(1, [3], [3], [1], 4),
        step(2, [3, 9], [0, 3], [4, 1], 8),
    ]
    counts = summarize_ledger(dict(generation=dict(steps=rows)))
    assert counts["request_mode_transitions"] == {"q1_to_spec": 1, "spec_to_q1": 1}
    owners = {row["owner"]: row for row in counts["per_request"]}
    assert owners[9]["transitions"] == {"q1_to_spec": 1}
    assert owners[3]["transitions"] == {"spec_to_q1": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_ids", [7, 7]),
        ("proposals", [[], [1, 2, 3, 4]]),
        ("capacity_tokens", 4),
        ("capacity_tokens", 9),
        ("accepted_from_previous", [0, 1]),
    ],
)
def test_invalid_frozen_q4_contract_is_rejected(ledger, field, value):
    ledger["generation"]["steps"][0][field] = value
    with pytest.raises(ValueError):
        summarize_ledger(ledger)


def test_installed_metadata_counts_each_query_once(ledger, metadata):
    result = summarize_metadata(metadata, ledger)
    assert result["total_capacity_rows"] == 12
    assert result["total_head_rows"] == 6
    assert result["graph_padding_rows"] == 6
    assert result["gdn_work"]["num_spec_tokens"] == 4
    assert result["gdn_work"]["num_prefill_tokens"] == 1
    assert result["gdn_work"]["num_decode_tokens"] == 1


@pytest.mark.parametrize("flag", ["model_executed", "native_generation_observation", "physical_block_ids_observed"])
def test_cpu_replay_cannot_be_relabelled_as_native_generation(ledger, metadata, flag):
    metadata[flag] = True
    with pytest.raises(ValueError, match="relabeled"):
        summarize_metadata(metadata, ledger)


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", 1),
        ("owners", [2, 7]),
        ("draft_counts", [3, 0]),
        ("query_lengths", [4, 1]),
        ("head_rows", 8),
        ("capacity", 3),
        ("capacity", 12),
    ],
)
def test_replay_cannot_change_work_or_default_capacity(ledger, metadata, field, value):
    metadata["steps"][0][field] = value
    with pytest.raises(ValueError):
        summarize_metadata(metadata, ledger)


def test_missing_step_or_unprocessed_query_is_rejected(ledger, metadata):
    partial = copy.deepcopy(metadata)
    partial["steps"].pop()
    with pytest.raises(ValueError, match="complete"):
        summarize_metadata(partial, ledger)
    metadata["steps"][0]["gdn"]["num_spec_tokens"] = 3
    with pytest.raises(ValueError, match="each actual query"):
        summarize_metadata(metadata, ledger)


def test_different_workload_is_rejected(ledger, metadata):
    metadata["workload_sha256"] = "another"
    with pytest.raises(ValueError, match="same frozen workload"):
        summarize_metadata(metadata, ledger)


def test_copy_budget_is_explicitly_a_scenario_not_measured_latency(ledger):
    result = conversion_scenario(summarize_ledger(ledger), 0.01)
    assert result["implemented"] is False
    assert result["fp32_state_bytes_per_owner_per_rank"] == 72 * 1024**2
    assert result["owner_switches"] == 1
    assert result["copy_read_write_bytes_per_rank"] == 144 * 1024**2
    assert result["break_even_seconds_per_owner_switch"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="positive"):
        conversion_scenario(summarize_ledger(ledger), 0)
