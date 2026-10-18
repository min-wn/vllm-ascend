# SPDX-License-Identifier: Apache-2.0
"""Exclusive bucket workspaces preserve state while avoiding per-lane copies."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tests.ut.worker.multi_graph.test_gdn_q4_log_backend import (
    fixture_backend,
    fixture_registry,
)


@pytest.fixture
def log_backend(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "q4_exclusive_backend_contract", directory / "multi_graph_gdn_q4_log_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def padded_inputs(args, kwargs, batch):
    """Keep two live owners; larger capacities add null/inactive requests."""
    query = torch.ones(batch * 4, 8, 128, dtype=torch.bfloat16)
    value = torch.ones(batch * 4, 24, 128, dtype=torch.bfloat16)
    options = dict(
        g=torch.ones(batch * 4, 24),
        beta=torch.ones(batch * 4, 24, dtype=torch.bfloat16),
        scale=128**-0.5,
        actual_seq_lengths=torch.tensor([0, 4, 4] + [0] * (batch - 2), dtype=torch.int32),
        ssm_state_indices=torch.cat((kwargs["ssm_state_indices"], torch.zeros((batch - 2) * 4, dtype=torch.int32))),
        num_accepted_tokens=torch.tensor([2, 4] + [1] * (batch - 2), dtype=torch.int32),
    )
    return (query, query.clone(), value, args[3]), options


@pytest.mark.parametrize("capacity", [False, True, -1, 0, 129, 2.5, "128", None])
def test_exclusive_reservation_rejects_invalid_capacity(log_backend, capacity):
    backend, *_ = fixture_backend(log_backend)
    with pytest.raises(ValueError, match="integer in 1..128"):
        backend.reserve_exclusive_capacity(capacity)
    assert backend._exclusive_buffers is None and not backend.buffers


@pytest.mark.parametrize("used", ["reserved", "recurrence", "failed"])
def test_exclusive_reservation_cannot_replace_live_addresses(log_backend, used):
    backend, _, _, _, args, kwargs, _ = fixture_backend(log_backend)
    if used == "reserved":
        assert backend.reserve_exclusive_capacity(2) is backend
    elif used == "recurrence":
        backend(*args, **kwargs)
    else:
        backend._failed = True
    with pytest.raises(RuntimeError):
        backend.reserve_exclusive_capacity(3)


def test_exclusive_capacity_is_checked_before_submission(log_backend):
    backend, _, _, calls, args, kwargs, _ = fixture_backend(log_backend)
    backend.reserve_exclusive_capacity(1)
    with pytest.raises(ValueError, match="reserved exclusive compact capacity"):
        backend(*args, **kwargs)
    assert not calls and not backend.buffers and backend.calls == 0


@pytest.mark.parametrize("mode", ["padding", "deferred_serial", "deferred_overlap", "deferred_pipeline"])
def test_exclusive_bucket_views_match_independent_state_and_keep_lane_events(log_backend, mode):
    reserved = fixture_backend(log_backend, mode, deferred=True, direct=True)
    independent = fixture_backend(log_backend, mode, deferred=True, direct=True)
    backend, state, _, _, args, kwargs, _ = reserved
    backend.reserve_exclusive_capacity(3)
    initial_null = tuple(tensor[0].clone() for tensor in state.mutable_tensors)
    snapshots = []
    for batch in (2, 3, 2):
        current_args, current_kwargs = padded_inputs(args, kwargs, batch)
        snapshots.append((current_args, current_kwargs))  # Retain metadata identities for all lanes.
        expected = independent[0](*current_args[:3], independent[1].handle, **current_kwargs).clone()
        actual = backend(*current_args, **current_kwargs).clone()
        if mode in ("deferred_overlap", "deferred_pipeline"):
            for active in (independent[0], backend):
                assert active.launch_pending_commit() and active.finish_pending_commit()
        assert torch.equal(actual, expected)
        for actual_state, expected_state in zip(state.mutable_tensors, independent[1].mutable_tensors, strict=True):
            assert torch.equal(actual_state, expected_state)
        if batch > 2:
            assert not actual[8:].count_nonzero()
    assert len(backend.buffers) == 3
    workspace = backend._exclusive_buffers
    for identity, lane in backend.buffers.items():
        batch = identity[-1]
        assert lane["products"].shape == (batch, 24, 128, 8)
        assert lane["output"].shape == (batch * 4, 24, 128)
        assert lane["ids"].shape == (batch * 4,)
        for name in ("products", "output", "ids"):
            assert lane[name].data_ptr() == workspace[name].data_ptr()
            assert lane[name].is_contiguous()
        if mode.startswith("deferred_"):
            assert lane["new_logs"] is workspace["new_logs"]
            assert lane["new_valid"] is workspace["new_valid"]
    if mode.startswith("deferred_"):
        for name in ("commit_stream", "solve_ready", "commit_done"):
            assert len({id(lane[name]) for lane in backend.buffers.values()}) == 3
    for actual, initial in zip(state.mutable_tensors, initial_null, strict=True):
        assert torch.equal(actual[0], initial)
    report = backend.describe()
    assert report["exclusive_workspace_capacity"] == 3
    assert "never overlap" in report["exclusive_workspace_contract"]
    assert report["scratch_bytes"] == sum(t.numel() * t.element_size() for t in workspace.values())
    assert report["persistent_lanes"] == 3


def test_default_bucket_workspaces_still_have_independent_storage(log_backend):
    backend, _, _, _, args, kwargs, _ = fixture_backend(log_backend, "deferred_serial", deferred=True)
    retained = [padded_inputs(args, kwargs, batch) for batch in (2, 3)]
    for current_args, current_kwargs in retained:
        backend(*current_args, **current_kwargs)
    lanes = list(backend.buffers.values())
    for name in ("products", "output", "ids", "new_logs", "new_valid"):
        assert lanes[0][name].data_ptr() != lanes[1][name].data_ptr()
    assert backend.describe()["exclusive_workspace_capacity"] is None


@pytest.mark.parametrize("snapshot_before_use", [True, False])
def test_registry_snapshots_one_shared_staging_bank_across_new_bucket_lanes(log_backend, snapshot_before_use):
    registry, fixtures, _ = fixture_registry(log_backend, mode="deferred_serial")
    for backend, _, _, _, args, kwargs, _ in fixtures:
        backend.reserve_exclusive_capacity(3)
        if not snapshot_before_use:
            backend(*args, **kwargs)
    saved = registry.snapshot(destination="cpu")
    assert all(len(banks) == 1 and banks[0][0] is None for banks in saved.staging)
    retained = []
    for backend, _, _, _, args, kwargs, _ in fixtures:
        current_args, current_kwargs = padded_inputs(args, kwargs, 3)
        retained.append((current_args, current_kwargs))
        backend(*current_args, **current_kwargs)
        backend._exclusive_buffers["new_logs"].fill_(31)
        backend._exclusive_buffers["new_valid"].fill_(7)
    registry.restore(saved)
    for fixture, state_snapshot, banks in zip(fixtures, saved.states, saved.staging, strict=True):
        backend, state = fixture[:2]
        for current, expected in zip(state.mutable_tensors, state_snapshot.tensors, strict=True):
            assert torch.equal(current, expected)
        _, expected_logs, expected_valid = banks[0]
        for lane in backend.buffers.values():
            assert torch.equal(lane["new_logs"], expected_logs)
            assert torch.equal(lane["new_valid"], expected_valid)
