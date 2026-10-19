# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    return importlib.import_module("aclgraph_bounded_lead_fia")


def test_replay_requires_current_lengths_and_one_actual_fia(monkeypatch):
    validate = load(monkeypatch).validate_fia_tasks
    operation = object()
    task = SimpleNamespace(operation=operation, kwargs={"actual_seq_lengths_kv": [128, 0]})
    validate([task], operation, [128, 0])
    for tasks, op, lengths in (
        ([], operation, [128, 0]),
        ([task, task], operation, [128, 0]),
        ([task], object(), [128, 0]),
        ([task], operation, [127, 0]),
    ):
        with pytest.raises(ValueError):
            validate(tasks, op, lengths)


def test_only_live_kv_slots_may_change(monkeypatch):
    fixture = object.__new__(load(monkeypatch).FIAPrefix)
    fixture.torch = torch
    fixture.workload = SimpleNamespace(kv_blocks=2, block_size=4)
    fixture.cpu_inputs = {"slots": torch.tensor([2, 3, -1, -1])}
    before = tuple(torch.zeros(2, 4, 1, 2) for _ in range(2))
    after = tuple(t.clone() for t in before)
    after[0][0, 2:4] = 1
    fixture.check_unwritten(before, after)
    after[1][1, 3] = 1
    with pytest.raises(AssertionError):
        fixture.check_unwritten(before, after)
