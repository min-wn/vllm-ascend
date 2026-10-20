# SPDX-License-Identifier: Apache-2.0
"""Bounded exact trajectory checks, without an accelerator runtime."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@dataclass(frozen=True)
class StateSnapshot:
    tensors: tuple
    state_identity: tuple
    generations: tuple
    initialized: tuple
    last_step: int
    pending: tuple | None


@dataclass(frozen=True)
class RegistrySnapshot:
    states: tuple
    staging: tuple
    registry_identity: tuple


@pytest.fixture(params=(False, True), ids=("torch_scan", "array_scan"))
def fixture(monkeypatch, request):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    path = directory / "multi_graph_gdn_q4_full_stream_audit.py"
    spec = importlib.util.spec_from_file_location("full_stream_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_checkpoint = module.StreamingCheckpoint
    module.StreamingCheckpoint = lambda *args, **kwargs: original_checkpoint(*args, cpu_array=request.param, **kwargs)

    def tensor(rows, dtype=torch.float32):
        result = torch.arange(rows * 6, dtype=dtype).reshape(rows, 2, 3)
        result[0].zero_()
        return result

    candidates = ((9, 10, 11, 12), (1, 2, 3, 4), (5, 6, 7, 8))
    pages = ((1, 2), (3,), (4, 5))
    gdn, registered, backends = [], [], {}
    for layer in range(2):
        handle = object()
        state = SimpleNamespace(
            handle=handle,
            _snapshot_identity=(handle, candidates),
            mutable_tensors=(tensor(4), tensor(4), tensor(4, torch.int32)),
            generations=[0, 0, 0],
            initialized=[True, True, True],
            last_step=-1,
            pending=None,
        )
        staging = {None: dict(new_logs=tensor(4), new_valid=tensor(4, torch.int32))}
        backend = SimpleNamespace(state=state, _staging_buffers=lambda staging=staging: staging)
        conv = tensor(13, torch.bfloat16)
        gdn.append(SimpleNamespace(state=state, conv=conv, gdn=SimpleNamespace(kv_cache=(conv, handle))))
        registered.append((handle, backend, object(), f"layer{layer}"))
        backends[handle] = backend
    fia = []
    for _ in range(2):
        key, value = tensor(7, torch.bfloat16), tensor(7, torch.bfloat16)
        fia.append(
            SimpleNamespace(
                key=key,
                value=value,
                backend=SimpleNamespace(kv_cache=(key, value)),
                impl=SimpleNamespace(key_cache=key, value_cache=value),
            )
        )
    bundle = SimpleNamespace(gdn_entries=gdn, fia_entries=fia, pages=SimpleNamespace(pages=pages))
    registry = SimpleNamespace(entries=registered, backends=backends, _require_joined=lambda: None)

    def snapshot():
        states, staging = [], []
        for _, backend, _, _ in registered:
            state = backend.state
            states.append(
                StateSnapshot(
                    tuple(t.clone() for t in state.mutable_tensors),
                    state._snapshot_identity,
                    tuple(state.generations),
                    tuple(state.initialized),
                    state.last_step,
                    state.pending,
                )
            )
            staging.append(
                tuple(
                    (identity, lane["new_logs"].clone(), lane["new_valid"].clone())
                    for identity, lane in backend._staging_buffers().items()
                )
            )
        return dict(
            compact=RegistrySnapshot(tuple(states), tuple(staging), tuple(backends)),
            conv=tuple(entry.conv.clone() for entry in gdn),
            fia=tuple((entry.key.clone(), entry.value.clone()) for entry in fia),
        )

    def advance(step, active):
        for entry, (_, backend, _, _) in zip(gdn, registered, strict=True):
            state = entry.state
            for owner in active:
                for item in state.mutable_tensors:
                    item[owner + 1].add_(1)
                for lane in backend._staging_buffers().values():
                    lane["new_logs"][owner + 1].add_(1)
                    lane["new_valid"][owner + 1].add_(1)
                for slot in candidates[owner]:
                    entry.conv[slot].add_(1)
                state.generations[owner] += 1
            state.last_step = step
        for entry in fia:
            for owner in active:
                for page in pages[owner]:
                    entry.key[page].add_(1)
                    entry.value[page].add_(1)

    return SimpleNamespace(module=module, bundle=bundle, registry=registry, snapshot=snapshot, advance=advance)


def test_streaming_checkpoint_restores_and_compares_all_continued_state(fixture):
    from multi_graph_gdn_q4_full_checks import compare_complete_snapshots

    initial = fixture.snapshot()
    audit = fixture.module.StreamingCheckpoint(torch, initial, scan_elements=5)
    for step, owners in enumerate(((2, 0), (1,), (2, 1))):
        before = fixture.snapshot()
        fixture.advance(step, owners)
        expected = fixture.snapshot()
        transferred = audit.capture_eager_and_restore(fixture.bundle, fixture.registry, owners)
        assert transferred["scratch_elements"] == 5
        assert transferred["copied_to_cpu_bytes"] == transferred["restored_to_device_bytes"] > 0
        compare_complete_snapshots(torch, fixture.snapshot(), before)
        compare_complete_snapshots(torch, audit.saved, expected)
        fixture.advance(step, owners)
        result = audit.compare_replay(fixture.bundle, fixture.registry)
        assert result["completed_steps"] == step + 1
    assert initial["compact"].states[0].last_step == -1
    assert torch.equal(initial["compact"].states[0].tensors[0][1], torch.arange(6, 12).reshape(2, 3))


@pytest.mark.parametrize(
    "component", ("base", "logs", "valid", "stage_logs", "stage_valid", "conv", "fia", "unused_fia")
)
def test_paused_and_unused_corruption_is_rejected(fixture, component):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot(), scan_elements=5)
    fixture.advance(0, (0,))
    state = fixture.bundle.gdn_entries[0].state
    lane = fixture.registry.entries[0][1]._staging_buffers()[None]
    targets = dict(
        base=state.mutable_tensors[0][2],
        logs=state.mutable_tensors[1][2],
        valid=state.mutable_tensors[2][2],
        stage_logs=lane["new_logs"][2],
        stage_valid=lane["new_valid"][2],
        conv=fixture.bundle.gdn_entries[0].conv[1],
        fia=fixture.bundle.fia_entries[0].key[3],
        unused_fia=fixture.bundle.fia_entries[0].value[6],
    )
    targets[component].add_(1)
    with pytest.raises(AssertionError, match="inactive state changed"):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))
    with pytest.raises(AssertionError, match="not ready"):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))


@pytest.mark.parametrize("component", ("base", "staging", "conv", "fia"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_values_anywhere_are_rejected(fixture, component, value):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    fixture.advance(0, (0,))
    state = fixture.bundle.gdn_entries[0].state
    targets = dict(
        base=state.mutable_tensors[0][1],
        staging=fixture.registry.entries[0][1]._staging_buffers()[None]["new_logs"][1],
        conv=fixture.bundle.gdn_entries[0].conv[9],
        fia=fixture.bundle.fia_entries[0].value[1],
    )
    targets[component][-1, -1] = value
    with pytest.raises(AssertionError, match="nonfinite"):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))


@pytest.mark.parametrize("component", ("canonical", "staging", "conv", "fia", "host"))
def test_replay_difference_is_rejected(fixture, component):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    fixture.advance(0, (0,))
    audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))
    fixture.advance(0, (0,))
    if component == "host":
        fixture.bundle.gdn_entries[0].state.generations[0] += 1
        message = "host contract differs"
    else:
        targets = dict(
            canonical=fixture.bundle.gdn_entries[0].state.mutable_tensors[0],
            staging=fixture.registry.entries[0][1]._staging_buffers()[None]["new_logs"],
            conv=fixture.bundle.gdn_entries[0].conv,
            fia=fixture.bundle.fia_entries[0].key,
        )
        targets[component][1].add_(1)
        message = "replay tensor values differ"
    with pytest.raises(AssertionError, match=message):
        audit.compare_replay(fixture.bundle, fixture.registry)


def test_replay_signed_zero_difference_is_rejected(fixture):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    fixture.advance(0, (0,))
    audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))
    fixture.advance(0, (0,))
    fixture.bundle.fia_entries[0].key[0, 0, 0] = -0.0
    with pytest.raises(AssertionError, match="replay tensor values differ"):
        audit.compare_replay(fixture.bundle, fixture.registry)


@pytest.mark.parametrize("mutation", ("identity", "staging", "cache", "generation", "pending", "last_step"))
def test_contract_and_lifecycle_changes_are_rejected(fixture, mutation):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    fixture.advance(0, (0,))
    state = fixture.bundle.gdn_entries[0].state
    if mutation == "identity":
        state._snapshot_identity = (object(), state._snapshot_identity[1])
    elif mutation == "staging":
        lane = fixture.registry.entries[0][1]._staging_buffers().pop(None)
        fixture.registry.entries[0][1]._staging_buffers()["foreign"] = lane
    elif mutation == "cache":
        fixture.bundle.fia_entries[0].impl.key_cache = torch.empty(1)
    elif mutation == "generation":
        state.generations[1] += 1
    elif mutation == "pending":
        state.pending = (0, (0,), (1,), (0,))
    else:
        state.last_step = -1
    with pytest.raises(AssertionError):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))


def test_calls_must_follow_eager_then_replay(fixture):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    with pytest.raises(AssertionError, match="not pending"):
        audit.compare_replay(fixture.bundle, fixture.registry)
    fixture.advance(0, (0,))
    audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))
    with pytest.raises(AssertionError, match="not ready"):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,))


@pytest.mark.parametrize("owners", ((0, 0), (-1,), (3,), (True,)))
def test_invalid_active_schedule_is_rejected(fixture, owners):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    fixture.advance(0, (0,))
    with pytest.raises(AssertionError, match="active request"):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, owners)


@pytest.mark.parametrize("pages", (((1, 2), (2,), (4, 5)), ((1, 2), (0,), (4, 5)), ((1, 2), (3,), (4, 9))))
def test_invalid_or_out_of_range_fia_pages_are_rejected(fixture, pages):
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot())
    fixture.advance(0, (0,))
    with pytest.raises(AssertionError, match="FIA"):
        audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (0,), pages=pages)


def test_chunks_cover_every_element_once_and_are_bounded(fixture):
    live = torch.arange(42).reshape(2, 3, 7).transpose(1, 2)
    saved = live.clone()
    chunks = list(fixture.module._paired_chunks(live, saved, 4))
    assert all(left.numel() <= 4 for left, _, _ in chunks)
    expected_offset = 0
    for left, right, offset in chunks:
        assert offset == expected_offset
        assert torch.equal(left, right)
        expected_offset += left.numel()
    assert expected_offset == live.numel()
    assert torch.equal(torch.cat([left for left, _, _ in chunks]), live.reshape(-1))


@pytest.mark.parametrize("limit", (0, -1, True, 1.5))
def test_chunk_limit_must_be_positive_integer(fixture, limit):
    with pytest.raises(ValueError, match="positive integer"):
        fixture.module.StreamingCheckpoint(torch, fixture.snapshot(), scan_elements=limit)


def test_noncontiguous_storage_uses_bounded_logical_chunks(fixture):
    for entry in fixture.bundle.gdn_entries:
        entry.conv = entry.conv.transpose(1, 2)
        entry.gdn.kv_cache = (entry.conv, entry.state.handle)
    audit = fixture.module.StreamingCheckpoint(torch, fixture.snapshot(), scan_elements=2)
    fixture.advance(0, (1,))
    audit.capture_eager_and_restore(fixture.bundle, fixture.registry, (1,))
    fixture.advance(0, (1,))
    assert audit.compare_replay(fixture.bundle, fixture.registry)["completed_steps"] == 1
