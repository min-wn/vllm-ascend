# SPDX-License-Identifier: Apache-2.0
"""Closed trajectory accounting must charge setup and both TP ranks."""

import copy
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_ready_full_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("ready_lifecycle", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def samples():
    return [
        dict(
            rank=rank,
            lifecycle_samples=[
                dict(
                    trial=trial,
                    mode=mode,
                    wall_us=wall,
                    transition_us=10.0,
                    closed_wall_us=wall + 10.0,
                    terminal_wall_us=2.0,
                    terminal_exact=True,
                    steps=128,
                )
                for trial in range(2)
                for mode, wall in zip(MODULE.MODES, ((100.0, 90.0) if rank == 0 else (120.0, 100.0)), strict=True)
            ],
        )
        for rank in range(2)
    ]


def test_terminal_acceptance_covers_all_owners_and_prefixes():
    counts = MODULE.terminal_counts(128)
    assert len(counts) == 128 and set(counts) == {1, 2, 3, 4}
    with pytest.raises(ValueError, match="128"):
        MODULE.terminal_counts(100)


def test_critical_rank_and_transition_are_in_the_denominator():
    result = MODULE.paired_lifecycles(samples(), 2)
    assert result[0]["closed_wall_us"] == dict(zip(MODULE.MODES, (130.0, 110.0), strict=True))
    assert result[0]["ready_percent"] == pytest.approx(100 * (1 - 110 / 130))


@pytest.mark.parametrize("bad", ["duplicate", "short", "pending", "missing_cost", "nonfinite"])
def test_accounting_rejects_incomplete_or_invalid_evidence(bad):
    ranks = copy.deepcopy(samples())
    row = ranks[1]["lifecycle_samples"][0]
    if bad == "duplicate":
        ranks[1]["lifecycle_samples"][-1] = copy.copy(row)
    elif bad == "short":
        row["steps"] = 127
    elif bad == "pending":
        row["terminal_exact"] = False
    elif bad == "missing_cost":
        row["closed_wall_us"] = row["wall_us"]
    else:
        row["wall_us"] = float("nan")
    with pytest.raises(ValueError):
        MODULE.paired_lifecycles(ranks, 2)
