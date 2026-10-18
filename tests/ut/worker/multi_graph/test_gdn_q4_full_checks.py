# SPDX-License-Identifier: Apache-2.0
"""Small CPU regressions for complete-model state and ownership audits."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_checks():
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_q4_full_checks.py"
    spec = importlib.util.spec_from_file_location("full_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture():
    def tensor(rows, dtype=torch.float32):
        result = torch.arange(rows * 6, dtype=dtype).reshape(rows, 2, 3)
        result[0].zero_()
        return result

    candidates = ((9, 10, 11, 12), (1, 2, 3, 4), (5, 6, 7, 8))
    states = tuple(
        SimpleNamespace(
            tensors=(tensor(4), tensor(4), tensor(4, torch.int32)),
            state_identity=(f"state{layer}", candidates),
            generations=(2, 3, 1),
            initialized=(True, True, True),
            last_step=9,
            pending=None,
        )
        for layer in range(2)
    )
    snapshot = dict(
        compact=SimpleNamespace(
            states=states,
            staging=tuple(((f"lane{layer}", tensor(4), tensor(4, torch.int32)),) for layer in range(2)),
            registry_identity=("registered0", "registered1"),
        ),
        conv=tuple(tensor(13, torch.bfloat16) for _ in range(2)),
        fia=tuple((tensor(7, torch.bfloat16), tensor(7, torch.bfloat16)) for _ in range(2)),
    )
    pages = ((1, 2), (3,), (4, 5))
    return load_checks(), snapshot, copy.deepcopy(snapshot), pages


def test_complete_exact_accepts_copies_and_rejects_staging_value(fixture):
    checks, before, after, _ = fixture
    checks.compare_complete_snapshots(torch, after, before, "exact")
    after["compact"].staging[1][0][1][3, 1, 2] += 1
    with pytest.raises(AssertionError, match="gdn1/staginglane1/logs"):
        checks.compare_complete_snapshots(torch, after, before)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_identity", ("other", ())),
        ("generations", (2, 4, 1)),
        ("initialized", (True, False, True)),
        ("last_step", 10),
        ("pending", (10, (1,), (2,), (3,))),
    ],
)
def test_complete_exact_rejects_host_contract_changes(fixture, field, value):
    checks, before, after, _ = fixture
    setattr(after["compact"].states[0], field, value)
    with pytest.raises(AssertionError, match=field):
        checks.compare_complete_snapshots(torch, after, before)


@pytest.mark.parametrize("component", ["registry", "lane", "missing_layer", "missing_cache"])
def test_complete_exact_rejects_identity_and_structure_changes(fixture, component):
    checks, before, after, _ = fixture
    if component == "registry":
        after["compact"].registry_identity = ("foreign", "registered1")
    elif component == "lane":
        _, logs, valid = after["compact"].staging[0][0]
        after["compact"].staging = ((("foreign", logs, valid),), *after["compact"].staging[1:])
    elif component == "missing_layer":
        after["fia"] = after["fia"][:1]
    else:
        del after["conv"]
    with pytest.raises(AssertionError):
        checks.compare_complete_snapshots(torch, after, before)


@pytest.mark.parametrize("location", ["base", "staging", "conv", "fia"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected_even_if_identical_or_inactive(fixture, location, value):
    checks, before, _, _ = fixture
    targets = dict(
        base=before["compact"].states[1].tensors[0],
        staging=before["compact"].staging[1][0][1],
        conv=before["conv"][1],
        fia=before["fia"][1][1],
    )
    targets[location][-1, -1, -1] = value
    with pytest.raises(AssertionError, match="nonfinite"):
        checks.compare_complete_snapshots(torch, before, before)


def test_paused_audit_accepts_active_changes_and_uses_registered_conv_slots(fixture):
    checks, before, after, pages = fixture
    for layer, state in enumerate(after["compact"].states):
        for tensor in state.tensors:
            tensor[1].add_(1)
        for tensor in after["compact"].staging[layer][0][1:]:
            tensor[1].add_(1)
        state.generations = (3, 3, 1)
        state.last_step = 10
        for slot in (9, 10, 11, 12):
            after["conv"][layer][slot].add_(1)
    for pair in after["fia"]:
        for tensor in pair:
            tensor[1:3].add_(1)
    checks.verify_untouched_owners(torch, before, after, [0], SimpleNamespace(pages=pages))


@pytest.mark.parametrize(
    "component", ["base", "logs", "valid", "staging_logs", "staging_valid", "conv", "fia", "unused_page"]
)
def test_paused_audit_rejects_every_inactive_state_kind(fixture, component):
    checks, before, after, pages = fixture
    state = after["compact"].states[0]
    targets = dict(
        base=state.tensors[0][2],
        logs=state.tensors[1][2],
        valid=state.tensors[2][2],
        staging_logs=after["compact"].staging[0][0][1][2],
        staging_valid=after["compact"].staging[0][0][2][2],
        conv=after["conv"][0][1],
        fia=after["fia"][0][0][3],
        unused_page=after["fia"][0][1][6],
    )
    targets[component].add_(1)
    with pytest.raises(AssertionError, match="tensor values differ"):
        checks.verify_untouched_owners(torch, before, after, [0], pages)


@pytest.mark.parametrize("component", ["base", "staging", "conv", "fia"])
def test_null_must_be_zero_even_if_unchanged(fixture, component):
    checks, before, _, pages = fixture
    targets = dict(
        base=before["compact"].states[0].tensors[0],
        staging=before["compact"].staging[0][0][1],
        conv=before["conv"][0],
        fia=before["fia"][0][0],
    )
    targets[component][0].add_(1)
    with pytest.raises(AssertionError, match="null state must stay zero"):
        checks.verify_untouched_owners(torch, before, copy.deepcopy(before), [0, 1, 2], pages)


@pytest.mark.parametrize("requests", [[0, 0], [-1], [3], [True]])
def test_paused_audit_rejects_invalid_schedule(fixture, requests):
    checks, before, after, pages = fixture
    with pytest.raises(AssertionError, match="active request"):
        checks.verify_untouched_owners(torch, before, after, requests, pages)


def test_paused_lifecycle_changes_rejected(fixture):
    checks, before, after, pages = fixture
    after["compact"].states[1].generations = (2, 4, 1)
    with pytest.raises(AssertionError, match="paused lifecycle"):
        checks.verify_untouched_owners(torch, before, after, [0], pages)


def test_chunked_scan_handles_noncontiguous_and_small_scan_limit(fixture):
    checks, before, after, _ = fixture
    checks.SCAN_ELEMENTS = 3
    for snapshot in (before, after):
        snapshot["conv"] = tuple(tensor.transpose(1, 2) for tensor in snapshot["conv"])
    checks.compare_complete_snapshots(torch, after, before)
    after["conv"][1][-1, -1, -1] = float("inf")
    with pytest.raises(AssertionError, match="nonfinite"):
        checks.assert_complete_finite(torch, after)
