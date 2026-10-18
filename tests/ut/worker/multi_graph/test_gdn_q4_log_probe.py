# SPDX-License-Identifier: Apache-2.0
"""Stable request addressing for dense candidates versus compact base slots."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "q4_log_probe_cpu_test", directory / "multi_graph_gdn_q4_log_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_request_identity_and_each_accepted_prefix_select_the_same_base(probe):
    requests = [7, 0, 4, 2]
    dense = probe.candidate_slot_ids(requests)
    compact = probe.base_slot_ids(requests)
    assert dense == [29, 30, 31, 32, 1, 2, 3, 4, 17, 18, 19, 20, 9, 10, 11, 12]
    for row, request_id in enumerate(requests):
        for accepted in (1, 2, 3, 4):
            assert (dense[row * 4 + accepted - 1] - 1) // 4 == request_id
            assert compact[row * 4 + accepted - 1] == request_id + 1
    assert len(set(compact)) == len(requests)
    assert 0 not in dense and 0 not in compact


def test_null_capacity_is_not_assigned_to_absent_requests(probe):
    assert probe.candidate_slot_ids([]) == []
    assert probe.base_slot_ids([]) == []
    assert probe.base_slot_ids([2]) == [3, 3, 3, 3]
