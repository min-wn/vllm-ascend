# SPDX-License-Identifier: Apache-2.0
"""Input contracts for the device request-tile protocol qualifier."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def probe():
    source = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_gdn_q4_tile_transport_probe.py"
    spec = importlib.util.spec_from_file_location("q4_tile_transport_probe_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_inputs_change_live_membership_and_acceptance_without_changing_capacity(probe):
    records = [probe.metadata_values(100, step) for step in range(8)]
    assert [record["live"] for record in records[:4]] == [100, 99, 98, 100]
    assert records[0]["owners"] != records[3]["owners"]
    for record in records:
        live = record["live"]
        assert len(record["sequence"]) == 101
        assert len(record["accepted"]) == 100
        assert len(record["ids"]) == 400
        assert record["sequence"] == [0, *([4] * live), *([0] * (100 - live))]
        assert len(set(record["owners"])) == live
        for row, owner in enumerate(record["owners"]):
            assert record["ids"][4 * row : 4 * row + 4] == [owner + 1] * 4
            assert 1 <= record["accepted"][row] <= 4
        assert record["ids"][4 * live :] == [0] * (4 * (100 - live))
    assert {count for record in records for count in record["accepted"]} == {1, 2, 3, 4}


@pytest.mark.parametrize("batch,step", [(2, 0), (129, 0), (100, -1)])
def test_invalid_transport_extent_is_rejected(probe, batch, step):
    with pytest.raises(ValueError):
        probe.metadata_values(batch, step)
