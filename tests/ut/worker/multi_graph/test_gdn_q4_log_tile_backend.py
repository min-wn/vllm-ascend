# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch/ownership controls for optional request-tile recurrence."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def contracts(monkeypatch):
    source = Path(__file__).with_name("test_gdn_q4_log_backend.py")
    spec = importlib.util.spec_from_file_location("q4_tile_backend_fixtures", source)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers, helpers.log_backend.__wrapped__(monkeypatch)


def tile_fixture(contracts, *, direct=False):
    helpers, module = contracts
    fixture = helpers.fixture_backend(module, deferred=True, direct=direct)
    backend, state, _, calls, _, _, _ = fixture
    original_solve = backend.deferred_module.solve
    invocations = []

    def consume(*operands):
        invocations.append(("consume", operands))
        assert len(operands) == 16 and operands[-1] == 40
        assert operands[6] is state.base
        original_solve(*operands)

    def solve(*operands):
        invocations.append(("tile_solve", operands))
        assert len(operands) == 19 and operands[-2] == 20 and type(operands[-1]) is bool
        (
            base,
            query_float,
            key_float,
            query,
            key,
            value,
            beta,
            gate,
            sequence,
            ids,
            accepted,
            products,
            output,
            old_logs,
            old_valid,
            new_logs,
            new_valid,
            _,
            _,
        ) = operands
        assert base is state.base and query_float.dtype == key_float.dtype == torch.float32
        assert torch.equal(query_float, query.float()) and torch.equal(key_float, key.float())
        products.fill_(1)
        original_solve(
            products,
            query,
            key,
            value,
            beta,
            gate,
            base,
            sequence,
            ids,
            accepted,
            output,
            old_logs,
            old_valid,
            new_logs,
            new_valid,
        )

    backend.tile_module = SimpleNamespace(solve=solve, consume=consume)
    return fixture, invocations


@pytest.mark.parametrize("mode", ["deferred_tile_delayed", "deferred_tile_ready", "deferred_tile_separate"])
def test_tile_controls_match_public_deferred_and_keep_identical_commit_join(contracts, mode):
    fixture, invocations = tile_fixture(contracts)
    backend, state, dense, calls, args, kwargs, api = fixture
    before, dense_before = state.snapshot(), dense.clone()
    backend.mode = "deferred_overlap"
    expected = backend(*args, **kwargs).clone()
    backend.launch_pending_commit()
    backend.finish_pending_commit()
    expected_state = state.snapshot()
    state.restore(before)
    calls.clear()
    api.npu.events.clear()

    backend.mode = mode
    actual = backend(*args, **kwargs)
    for tensor, old in zip(state.mutable_tensors, before.tensors, strict=True):
        assert torch.equal(tensor, old)
    assert api.npu.events == [] and backend._pending_commit is not None
    assert [call[0] for call in calls] == (["products", "solve"] if mode.endswith("separate") else ["solve"])
    assert len(invocations) == 1
    operation, operands = invocations[0]
    assert operation == ("consume" if mode.endswith("separate") else "tile_solve")
    if operation == "tile_solve":
        assert operands[-1] is mode.endswith("ready")
        assert operands[9].tolist() == [1] * 4 + [2] * 4
    assert backend.launch_pending_commit() and backend.finish_pending_commit()
    assert api.npu.events == ["record", "wait", "record", "wait"]
    assert calls[-1][0] == "commit" and torch.equal(actual, expected)
    assert torch.equal(dense, dense_before)
    for tensor, wanted in zip(state.mutable_tensors, expected_state.tensors, strict=True):
        assert torch.equal(tensor, wanted)
    assert backend.describe()["tile_extension_loaded"] and not backend.describe()["deferred_commit_pending"]


@pytest.mark.parametrize("mode", ["deferred_tile_delayed", "deferred_tile_ready", "deferred_tile_separate"])
@pytest.mark.parametrize("missing", ["deferred", "tile"])
def test_tile_modes_require_both_explicit_extensions_without_changing_mode(contracts, mode, missing):
    fixture, _ = tile_fixture(contracts)
    backend = fixture[0]
    setattr(backend, f"{missing}_module", None)
    with pytest.raises(ValueError, match=f"explicit {missing} extension"):
        backend.mode = mode
    assert backend.mode == "padding" and not backend.buffers and backend.calls == 0


@pytest.mark.parametrize("mode", ["deferred_tile_delayed", "deferred_tile_ready", "deferred_tile_separate"])
def test_tile_stale_staging_never_changes_paused_or_null_owner(contracts, mode):
    fixture, _ = tile_fixture(contracts)
    backend, state, _, _, args, kwargs, _ = fixture
    backend.mode = mode
    kwargs["actual_seq_lengths"][2] = 0
    kwargs["ssm_state_indices"][4:].zero_()
    kwargs["num_accepted_tokens"][1] = 0
    initial = state.snapshot()
    for iteration in range(2):
        state.restore(initial)
        output = backend(*args, **kwargs)
        backend.launch_pending_commit()
        backend.finish_pending_commit()
        for tensor, old in zip(state.mutable_tensors, initial.tensors, strict=True):
            assert torch.equal(tensor[0], old[0]) and torch.equal(tensor[2], old[2])
        assert not output[4:].count_nonzero()
        lane = next(iter(backend.buffers.values()))
        lane["new_logs"][2].fill_(13 + iteration)
        lane["new_valid"][2].fill_(1)


def test_tile_variants_reuse_exclusive_workspace_with_current_operands(contracts):
    fixture, invocations = tile_fixture(contracts, direct=True)
    backend, state, _, _, args, kwargs, _ = fixture
    backend.reserve_exclusive_capacity(4)
    reserved = backend._exclusive_buffers
    before = state.snapshot()
    pointers = []
    for index, mode in enumerate(("deferred_tile_ready", "deferred_tile_separate", "deferred_tile_delayed")):
        state.restore(before)
        backend.mode = mode
        args[0].fill_(index + 2)
        options = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
        backend(*args, **options)
        lane = next(reversed(backend.buffers.values()))
        pointers.append(tuple(lane[name].data_ptr() for name in ("products", "output", "ids", "new_logs", "new_valid")))
        assert lane["new_logs"] is reserved["new_logs"] and lane["new_valid"] is reserved["new_valid"]
        backend.launch_pending_commit()
        backend.finish_pending_commit()
    assert pointers[0] == pointers[1] == pointers[2]
    assert len(backend.buffers) == 3
    assert invocations[0][1][1].eq(2).all() and invocations[2][1][1].eq(4).all()


@pytest.mark.parametrize("mode", ["deferred_tile_delayed", "deferred_tile_ready", "deferred_tile_separate"])
def test_tile_transaction_requires_projection_hook_and_endpoint_join(contracts, mode):
    fixture, invocations = tile_fixture(contracts)
    backend, _, _, _, args, kwargs, _ = fixture
    backend.mode = mode
    backend(*args, **kwargs)
    with pytest.raises(RuntimeError, match="before another recurrence"):
        backend(*args, **kwargs)
    with pytest.raises(RuntimeError, match="before the pending"):
        backend.mode = "deferred_overlap"
    with pytest.raises(RuntimeError, match="did not launch"):
        backend.finish_pending_commit()
    assert len(invocations) == 1
    assert backend.launch_pending_commit()
    with pytest.raises(RuntimeError, match="already launched"):
        backend.launch_pending_commit()
    assert backend.finish_pending_commit() and not backend.finish_pending_commit()


def test_tile_loader_records_library_and_uses_same_undrained_cleanup(contracts, monkeypatch, tmp_path):
    helpers, module = contracts
    fixture, _ = tile_fixture(contracts)
    backend, state, _, calls, args, kwargs, api = fixture
    gdn, original = helpers.install_fixture_modules(module, monkeypatch, backend)
    libraries = dict(
        gdn_q4_log=backend.log_module,
        gdn_q4_products=backend.product_module,
        gdn_q4_log_deferred=backend.deferred_module,
        gdn_q4_tile_pipeline=backend.tile_module,
    )
    loaded = []

    def load(name, path):
        loaded.append(name)
        return libraries[name]

    monkeypatch.setattr(module, "load_extension", load)
    library = tmp_path / "tile.so"
    library.write_bytes(b"tile-provenance")
    with (
        pytest.raises(RuntimeError, match="without joining"),
        module.installed_log_backend(
            state, library, library, mode="deferred_tile_ready", deferred_path=library, tile_path=library
        ) as active,
    ):
        assert active.sources["tile"] == dict(path=str(library), sha256=hashlib.sha256(b"tile-provenance").hexdigest())
        active(*args, **kwargs)
    assert loaded == ["gdn_q4_log", "gdn_q4_products", "gdn_q4_log_deferred", "gdn_q4_tile_pipeline"]
    assert gdn.recurrent_gated_delta_rule is original
    assert [call[0] for call in calls] == ["solve", "commit"]
    assert api.npu.events == ["record", "wait", "record", "wait", "synchronize"]
    assert active._failed and active._pending_commit is None


@pytest.mark.parametrize("mode", ["deferred_tile_delayed", "deferred_tile_ready", "deferred_tile_separate"])
def test_tile_registry_checks_every_layer_then_hooks_and_joins_independent_commits(contracts, monkeypatch, mode):
    helpers, module = contracts
    registry, fixtures, projections = helpers.fixture_registry(module, mode="deferred_overlap")
    tile_fixtures = [tile_fixture(contracts, direct=True)[0] for _ in range(2)]
    for fixture, replacement in zip(fixtures, tile_fixtures, strict=True):
        fixture[0].tile_module = replacement[0].tile_module
    # A missing second layer must fail atomically, before the first mode changes.
    registry.entries[1][1].tile_module = None
    with pytest.raises(ValueError, match="every compact layer needs the tile extension"):
        registry.mode = mode
    assert registry.mode == "deferred_overlap"
    # Use matching tile owners after the all-layer guard has been exercised.
    registry = module.CompactSSMRegistry()
    for index, (fixture, projection) in enumerate(zip(tile_fixtures, projections, strict=True)):
        registry.register(fixture[1].handle, fixture[0], projection, name=f"tile_{index}")
    registry.mode = mode
    gdn, original = helpers.install_fixture_modules(module, monkeypatch, tile_fixtures[0][0])
    with module.installed_log_registry(registry):
        helpers.execute_registry_step(registry, tile_fixtures, projections)
        assert registry.finish_pending_commits() == 2
        for fixture in tile_fixtures:
            assert fixture[6].npu.events == ["record", "wait", "record", "wait"]
            assert fixture[3][-1][0] == "commit" and fixture[0]._pending_commit is None
    assert gdn.recurrent_gated_delta_rule is original
    assert all(not projection._forward_pre_hooks for projection in projections)
