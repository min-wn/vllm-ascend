# SPDX-License-Identifier: Apache-2.0
"""Check that endpoint accounting includes paused owners and terminal costs."""

import importlib.util
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_gdn_lifecycle_probe.py"
SPEC = importlib.util.spec_from_file_location("gdn_lifecycle_probe", PROBE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_terminal_acceptance_tracks_owner_generations_across_pause():
    schedule = [dict(live=3), dict(live=1), dict(live=2)]
    assert MODULE.terminal_counts(schedule, 4) == [4, 4, 4, 4]
    with pytest.raises(ValueError, match="capacity"):
        MODULE.terminal_counts([dict(live=5)], 4)


def test_critical_rank_accounting_and_one_time_import():
    ranks = [
        dict(
            lifecycle=dict(
                samples=[
                    dict(
                        mode=mode,
                        round=0,
                        steps=8,
                        device_us=offset + cost,
                        wall_us=offset + cost + 20,
                        terminal_device_us=10 + offset,
                    )
                    for mode, cost in (("official_padding", 1000), ("compact_padding", 800))
                ]
            )
        )
        for offset in (0, 100)
    ]
    result = MODULE.summarize_lifecycle(ranks, 200)
    official, compact = (result["summary"][mode] for mode in ("official_padding", "compact_padding"))
    assert official["median_step_device_us"] == 1100 / 8
    assert compact["median_step_device_us"] == 900 / 8
    assert official["one_import_plus_trajectory_wall_us"] == [1120]
    assert compact["one_import_plus_trajectory_wall_us"] == [1120]
    assert result["samples"][1]["terminal_device_us"] == 110
    ranks[1]["lifecycle"]["samples"].reverse()
    with pytest.raises(ValueError, match="order"):
        MODULE.summarize_lifecycle(ranks, 200)
