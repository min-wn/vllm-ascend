# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/lifecycle contracts; native math has separate device gates."""

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def log_backend(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "q4_log_backend_contract", directory / "multi_graph_gdn_q4_log_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class FakeNPU:
    def __init__(self):
        self.events = []
        self.current = self.Stream()

    def Stream(self):
        owner = self
        return SimpleNamespace(wait_event=lambda event: owner.events.append("wait"))

    def Event(self, **kwargs):
        return SimpleNamespace(record=lambda stream: self.events.append("record"))

    def current_stream(self):
        return self.current

    def synchronize(self):
        self.events.append("synchronize")

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class TorchWithNPU:
    def __init__(self):
        self.npu = FakeNPU()

    def __getattr__(self, name):
        return getattr(torch, name)


def fixture_state(module, *, accepted=(2, 4), direct=False):
    # Deliberately shuffled, noncontiguous quartets; division by four is wrong.
    slots = [[7, 2, 9, 4], [1, 8, 5, 10]]
    dense = torch.arange(11, dtype=torch.float32).view(11, 1, 1, 1).expand(11, 24, 128, 128).clone()
    dense.add_(0.0001)
    api = TorchWithNPU()
    if direct:
        selected = [0, *(quartet[count - 1] for quartet, count in zip(slots, accepted, strict=True))]
        state = module.Q4LogState.from_base(api, dense[selected], slots)
    else:
        state = module.Q4LogState(api, dense, slots, initial_accepted=list(accepted))
    return state, dense, slots, api


def fixture_backend(module, mode="padding", deferred=False, direct=False):
    state, dense, slots, api = fixture_state(module, direct=direct)
    calls = []

    def products(base, query, key, sequence, ids, accepted, output, cores, start, stop):
        assert base is state.base and query.dtype == key.dtype == torch.float32
        calls.append(("products", start, stop, ids.clone(), query.clone(), key.clone()))
        output[start:stop].fill_(1)

    def step(
        product, query, key, value, beta, gate, base, sequence, ids, accepted, output, logs, valid, cores, start, stop
    ):
        assert query.dtype == key.dtype == torch.bfloat16 and base is state.base
        calls.append(("step", start, stop, ids.clone(), query, key))
        for request in range(start, stop):
            if sequence[request + 1] == 0:
                continue
            owner = int(ids[request * 4])
            if owner <= 0:
                continue
            assert not output[request * 4 : (request + 1) * 4].count_nonzero()
            base[owner].add_(1)
            logs[owner].add_(2)
            valid[owner].fill_(1)
            output[request * 4 : (request + 1) * 4].fill_(7)

    def export(base, logs, valid, output):
        calls.append(("export",))
        output.copy_(base[:, None].expand_as(output))

    def solve(
        product,
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
        cores=40,
        start=0,
        stop=None,
    ):
        stop = accepted.numel() if stop is None else stop
        calls.append(("solve", start, stop, ids.clone(), query, key))
        for request in range(start, stop):
            if sequence[request + 1] == 0:
                continue
            owner = int(ids[request * 4])
            if owner <= 0:
                continue
            assert not output[request * 4 : (request + 1) * 4].count_nonzero()
            new_logs[owner].copy_(old_logs[owner] + 2)
            new_valid[owner].fill_(1)
            output[request * 4 : (request + 1) * 4].fill_(7)

    def commit(base, sequence, ids, accepted, old_logs, old_valid, new_logs, new_valid):
        calls.append(("commit",))
        for request in range(accepted.numel()):
            if sequence[request + 1] == 0:
                continue
            owner = int(ids[request * 4])
            if owner <= 0:
                continue
            base[owner].add_(1)
            old_logs[owner].copy_(new_logs[owner])
            old_valid[owner].copy_(new_valid[owner])

    log_module = SimpleNamespace(step=step, export_candidates=export)
    backend = module.Q4LogBackend(
        state,
        log_module,
        SimpleNamespace(products=products),
        mode=mode,
        deferred_module=SimpleNamespace(solve=solve, commit=commit) if deferred else None,
    )
    query = torch.ones(8, 8, 128, dtype=torch.bfloat16)
    key = query.clone()
    value = torch.ones(8, 24, 128, dtype=torch.bfloat16)
    kwargs = dict(
        g=torch.ones(8, 24),
        beta=torch.ones(8, 24, dtype=torch.bfloat16),
        scale=128**-0.5,
        actual_seq_lengths=torch.tensor([0, 4, 4], dtype=torch.int32),
        ssm_state_indices=torch.tensor(slots, dtype=torch.int32).flatten(),
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32),
    )
    return backend, state, dense, calls, (query, key, value, state.handle if direct else dense), kwargs, api


def test_initial_import_uses_exact_selected_quartet_and_preserves_null(log_backend):
    state, dense, _, _ = fixture_state(log_backend)
    assert torch.equal(state.base[0], dense[0])
    assert torch.equal(state.base[1], dense[2])
    assert torch.equal(state.base[2], dense[10])
    assert state.base.dtype == torch.float32 and state.initialized == [True, True]
    assert not state.logs.count_nonzero() and not state.valid.count_nonzero()
    assert state.device_owner_lookup.tolist() == [0, 2, 1, -1, 1, 2, -1, 1, 2, 1, 2]
    assert state.mutable_tensors == (state.base, state.logs, state.valid)


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "stride", "zero", "overlap", "unknown", "bool", "empty", "accept_count", "accept_value"]
)
def test_initial_import_rejects_ambiguous_owner_or_precision(log_backend, bad):
    state, dense, slots, _ = fixture_state(log_backend)
    accepted = [1, 1]
    if bad == "dtype":
        dense = dense.bfloat16()
    elif bad == "shape":
        dense = dense[:, :12]
    elif bad == "stride":
        dense = dense.transpose(-1, -2)
    elif bad == "zero":
        slots[0][0] = 0
    elif bad == "overlap":
        slots[1][0] = slots[0][0]
    elif bad == "unknown":
        slots[0][0] = 11
    elif bad == "bool":
        slots[0][0] = True
    elif bad == "empty":
        slots = []
    elif bad == "accept_count":
        accepted = [1]
    else:
        accepted = [0, 4]
    with pytest.raises(ValueError):
        log_backend.Q4LogState(state.torch, dense, slots, initial_accepted=accepted)


def test_snapshot_restore_keeps_all_addresses_and_lifecycle(log_backend):
    state, _, _, _ = fixture_state(log_backend)
    state.prepare_step(3, [1], [4], generations=[0])
    before = state.snapshot()
    addresses = [tensor.data_ptr() for tensor in state.mutable_tensors]
    for tensor in state.mutable_tensors:
        tensor.add_(9)
    state.finish_step(3)
    state.restore(before)
    assert state.pending == before.pending and state.generations == [0, 0] and state.last_step == -1
    assert addresses == [tensor.data_ptr() for tensor in state.mutable_tensors]
    for actual, expected in zip(state.mutable_tensors, before.tensors, strict=True):
        assert torch.equal(actual, expected)
    other, _, _, _ = fixture_state(log_backend)
    with pytest.raises(ValueError, match="different compact cache"):
        other.restore(before)


def test_host_lifecycle_pause_resume_reorder_and_single_commit(log_backend):
    state, _, _, _ = fixture_state(log_backend)
    state.prepare_step(0, [1, 0], [4, 2], generations=[0, 0])
    with pytest.raises(RuntimeError, match="in flight"):
        state.prepare_step(1, [0], [1])
    with pytest.raises(RuntimeError, match="matching"):
        state.finish_step(1)
    state.finish_step(0)
    state.prepare_step(1, [0], [3], generations=[1])
    state.finish_step(1)
    assert state.generations == [2, 1]
    state.prepare_step(2, [1], [1], generations=[1])
    state.finish_step(2)
    assert state.generations == [2, 2]
    with pytest.raises(RuntimeError):
        state.finish_step(2)
    with pytest.raises(ValueError, match="newer"):
        state.prepare_step(2, [1], [1])


@pytest.mark.parametrize(
    "requests,accepted,generations",
    [
        ([0, 0], [1, 2], None),
        ([2], [1], None),
        ([-1], [1], None),
        ([True], [1], None),
        ([0], [0], None),
        ([0], [True], None),
        ([0], [], None),
        ([0], [1], [1]),
    ],
)
def test_host_lifecycle_rejects_bad_declared_schedule(log_backend, requests, accepted, generations):
    state, _, _, _ = fixture_state(log_backend)
    with pytest.raises(ValueError):
        state.prepare_step(0, requests, accepted, generations=generations)
    assert state.pending is None and state.generations == [0, 0]


@pytest.mark.parametrize("mode", ["padding", "serial", "pipeline"])
def test_compact_backend_only_maps_metadata_casts_qk_and_preserves_dense_null(log_backend, mode):
    backend, state, dense, calls, args, kwargs, api = fixture_backend(log_backend, mode)
    before = dense.clone()
    null = tuple(tensor[0].clone() for tensor in state.mutable_tensors)
    first = backend(*args, **kwargs)
    second = backend(*args, **kwargs)
    assert first is second and backend.calls == 2 and len(backend.buffers) == 1
    assert torch.equal(dense, before)
    assert torch.equal(first, torch.full_like(first, 7))
    for tensor, expected in zip(state.mutable_tensors, null, strict=True):
        assert torch.equal(tensor[0], expected)
    products = [call for call in calls if call[0] == "products"]
    assert all(call[3].tolist() == [1] * 4 + [2] * 4 for call in products)
    assert products[0][4].dtype == torch.float32
    assert all(call[4] is args[0] and call[5] is args[1] for call in calls if call[0] == "step")
    if mode == "pipeline":
        assert len(api.npu.events) == 12  # three event records and three waits per invocation
    report = backend.describe()
    assert report["dense_initialization_source_becomes_stale"] and not report["device_metadata_value_validation"]


def test_eager_calls_use_current_qk_and_mode_changes_keep_state_snapshot_portable(log_backend):
    backend, state, _, calls, args, kwargs, _ = fixture_backend(log_backend, "serial")
    before = state.snapshot()
    backend(*args, **kwargs)
    args[0].fill_(2)
    args[1].fill_(4)
    state.restore(before)
    backend.mode = "pipeline"
    backend(*args, **kwargs)
    products = [call for call in calls if call[0] == "products"]
    assert torch.all(products[-1][4] == 2) and torch.all(products[-1][5] == 4)
    assert len(backend.buffers) == 1


@pytest.mark.parametrize("metadata", ["actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens"])
def test_distinct_metadata_lanes_have_separate_products_output_and_mapping(log_backend, metadata):
    backend, _, _, _, args, kwargs, _ = fixture_backend(log_backend)
    first = backend(*args, **kwargs)
    kwargs[metadata] = kwargs[metadata].clone()
    second = backend(*args, **kwargs)
    assert first.data_ptr() != second.data_ptr()
    lanes = list(backend.buffers.values())
    for name in ("products", "output", "ids"):
        assert lanes[0][name].data_ptr() != lanes[1][name].data_ptr()


def test_inactive_accepted_zero_does_not_touch_paused_owner(log_backend):
    backend, state, _, _, args, kwargs, _ = fixture_backend(log_backend)
    kwargs["actual_seq_lengths"][2] = 0
    kwargs["ssm_state_indices"][4:].zero_()
    kwargs["num_accepted_tokens"][1] = 0
    before = tuple(tensor[2].clone() for tensor in state.mutable_tensors)
    output = backend(*args, **kwargs)
    assert not output[4:].count_nonzero()
    for tensor, expected in zip(state.mutable_tensors, before, strict=True):
        assert torch.equal(tensor[2], expected)


@pytest.mark.parametrize(
    "bad",
    [
        "non_spec",
        "option",
        "state",
        "query_dtype",
        "metadata_dtype",
        "metadata_rank",
        "gate",
        "scale",
        "contiguous",
        "mode",
    ],
)
def test_backend_refuses_unqualified_contracts_without_fallback(log_backend, bad):
    backend, _, _, calls, args, kwargs, _ = fixture_backend(log_backend)
    args = list(args)
    if bad == "non_spec":
        kwargs.pop("num_accepted_tokens")
    elif bad == "option":
        kwargs["unknown"] = True
    elif bad == "state":
        args[3] = args[3].clone()
    elif bad == "query_dtype":
        args[0] = args[0].float()
    elif bad == "metadata_dtype":
        kwargs["ssm_state_indices"] = kwargs["ssm_state_indices"].long()
    elif bad == "metadata_rank":
        kwargs["num_accepted_tokens"] = kwargs["num_accepted_tokens"].view(1, 2)
    elif bad == "gate":
        kwargs["g"] = None
    elif bad == "scale":
        kwargs["scale"] = 1
    elif bad == "contiguous":
        args[0] = torch.zeros(8, 8, 256, dtype=torch.bfloat16)[:, :, ::2]
    else:
        with pytest.raises(ValueError):
            backend.mode = "unknown"
        return
    with pytest.raises(ValueError):
        backend(*args, **kwargs)
    assert not calls


def test_all_candidate_export_is_explicit_and_uses_separate_scratch(log_backend):
    backend, state, _, calls, args, kwargs, _ = fixture_backend(log_backend)
    backend(*args, **kwargs)
    assert all(call[0] != "export" for call in calls)
    result = state.export(backend.log_module)
    assert result.shape == (3, 4, 24, 128, 128)
    assert state.export(backend.log_module) is result
    assert result.data_ptr() != state.base.data_ptr()


@pytest.mark.parametrize("mode", ["deferred_serial", "deferred_overlap", "deferred_pipeline"])
def test_deferred_requires_explicit_extension(log_backend, mode):
    with pytest.raises(ValueError, match="explicit deferred extension"):
        fixture_backend(log_backend, mode)


@pytest.mark.parametrize("mode", ["deferred_serial", "deferred_overlap", "deferred_pipeline"])
def test_deferred_publishes_active_staging_into_identical_canonical_bank(log_backend, mode):
    backend, state, dense, calls, args, kwargs, api = fixture_backend(log_backend, deferred=True)
    initial = state.snapshot()
    expected = backend(*args, **kwargs).clone()
    expected_state = state.snapshot()
    state.restore(initial)
    calls.clear()
    backend.mode = mode
    dense_before = dense.clone()
    actual = backend(*args, **kwargs)
    if mode in ("deferred_overlap", "deferred_pipeline"):
        # Returning attention must leave the old canonical bank untouched.
        for tensor, old in zip(state.mutable_tensors, initial.tensors, strict=True):
            assert torch.equal(tensor, old)
        expected_calls = ["products", "solve"] * (2 if mode == "deferred_pipeline" else 1)
        assert [call[0] for call in calls] == expected_calls
        solved_events = ["record", "wait", "record", "wait", "record", "wait"] if mode == "deferred_pipeline" else []
        assert api.npu.events == solved_events
        assert backend.launch_pending_commit()
        assert api.npu.events == solved_events + ["record", "wait", "record"]
        assert backend.finish_pending_commit()
        assert api.npu.events == solved_events + ["record", "wait", "record", "wait"]
    else:
        assert not backend.launch_pending_commit() and not backend.finish_pending_commit()
    assert torch.equal(actual, expected) and torch.equal(dense, dense_before)
    for tensor, wanted in zip(state.mutable_tensors, expected_state.tensors, strict=True):
        assert torch.equal(tensor, wanted)
    lane = next(iter(backend.buffers.values()))
    assert lane["new_logs"].data_ptr() != state.logs.data_ptr()
    assert lane["new_valid"].data_ptr() != state.valid.data_ptr()
    assert not lane["new_logs"][0].count_nonzero() and not lane["new_valid"][0].count_nonzero()
    assert len(state.mutable_tensors) == 3
    report = backend.describe()
    assert report["deferred_extension_loaded"] and not report["deferred_commit_pending"]
    assert report["scratch_bytes"] >= sum(t.numel() * t.element_size() for t in (lane["new_logs"], lane["new_valid"]))


def test_deferred_hook_order_reentry_mode_and_official_switch_are_guarded(log_backend):
    backend, _, _, calls, args, kwargs, _ = fixture_backend(log_backend, "deferred_overlap", deferred=True)
    backend(*args, **kwargs)
    with pytest.raises(RuntimeError, match="before another recurrence"):
        backend(*args, **kwargs)
    with pytest.raises(RuntimeError, match="before the pending"):
        backend.mode = "padding"
    with pytest.raises(RuntimeError, match="before joining"), backend.suspended():
        pass
    with pytest.raises(RuntimeError, match="did not launch"):
        backend.finish_pending_commit()
    assert [call[0] for call in calls] == ["products", "solve"]
    assert backend.launch_pending_commit()
    with pytest.raises(RuntimeError, match="already launched"):
        backend.launch_pending_commit()
    assert backend.finish_pending_commit()
    assert not backend.finish_pending_commit()
    backend.mode = "padding"


def test_deferred_paused_owner_and_canonical_snapshot_exclude_staging(log_backend):
    backend, state, _, _, args, kwargs, _ = fixture_backend(log_backend, "deferred_serial", deferred=True)
    kwargs["actual_seq_lengths"][2] = 0
    kwargs["ssm_state_indices"][4:].zero_()
    kwargs["num_accepted_tokens"][1] = 0
    before = state.snapshot()
    backend(*args, **kwargs)
    lane = next(iter(backend.buffers.values()))
    for tensor, old in zip(state.mutable_tensors, before.tensors, strict=True):
        assert torch.equal(tensor[0], old[0]) and torch.equal(tensor[2], old[2])
    assert not lane["new_logs"][2].count_nonzero() and not lane["new_valid"][2].count_nonzero()
    state.restore(before)
    # Scratch can retain the previous generation: solve overwrites active
    # staging and commit never consumes paused staging from another generation.
    lane["new_logs"][2].fill_(11)
    lane["new_valid"][2].fill_(1)
    backend(*args, **kwargs)
    for tensor, old in zip(state.mutable_tensors, before.tensors, strict=True):
        assert torch.equal(tensor[2], old[2])


def test_deferred_pipeline_has_separate_dispatch_and_matches_full_overlap_and_fused(log_backend):
    backend, state, _, calls, args, kwargs, _ = fixture_backend(log_backend, deferred=True)
    initial = state.snapshot()
    reference = None
    for mode in ("padding", "pipeline", "deferred_overlap", "deferred_pipeline", "pipeline", "deferred_pipeline"):
        state.restore(initial)
        calls.clear()
        backend.mode = mode
        output = backend(*args, **kwargs).clone()
        if mode.startswith("deferred_"):
            backend.launch_pending_commit()
            backend.finish_pending_commit()
        actual = tuple(tensor.clone() for tensor in state.mutable_tensors)
        if reference is None:
            reference = output, actual
        assert torch.equal(output, reference[0])
        assert all(torch.equal(tensor, wanted) for tensor, wanted in zip(actual, reference[1], strict=True))
        if mode == "pipeline":
            assert [call[0] for call in calls] == ["products", "step", "products", "step"]
        if mode == "deferred_pipeline":
            assert [call[0] for call in calls] == ["products", "solve", "products", "solve", "commit"]
    lane = next(iter(backend.buffers.values()))
    assert lane["range_dispatch"] is not lane["deferred_range_dispatch"]
    assert lane["range_operations"] is not lane["deferred_range_operations"]


def test_deferred_pipeline_full_prepare_once_solve_join_then_projection_hook_and_commit_join(log_backend):
    backend, state, _, _, args, kwargs, api = fixture_backend(log_backend, "deferred_pipeline", deferred=True)
    timeline, events, streams = [], [], []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            timeline.append(("wait", self.name, event.index))

    class Event:
        def __init__(self, **kwargs):
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            assert npu.active is stream
            timeline.append(("record", stream.name, self.index))

    def new_stream():
        stream = Stream(f"side{len(streams)}")
        streams.append(stream)
        return stream

    @contextmanager
    def stream_context(stream):
        previous, npu.active = npu.active, stream
        try:
            yield
        finally:
            npu.active = previous

    npu = SimpleNamespace(active=Stream("capture"), Stream=new_stream, Event=Event, stream=stream_context)
    npu.current_stream = lambda: npu.active
    api.npu = npu
    product, solve, commit = (
        backend.product_module.products,
        backend.deferred_module.solve,
        backend.deferred_module.commit,
    )

    def traced_products(*operands):
        timeline.append(("products", npu.active.name, *operands[-2:]))
        product(*operands)

    def traced_solve(*operands):
        timeline.append(("solve", npu.active.name, *operands[-2:]))
        solve(*operands)

    def traced_commit(*operands):
        timeline.append(("commit", npu.active.name))
        commit(*operands)

    def mapped(*operands, **options):
        timeline.append(("prepare_map", npu.active.name))
        return torch.index_select(*operands, **options)

    api.index_select = mapped
    backend.product_module.products = traced_products
    backend.deferred_module.solve, backend.deferred_module.commit = traced_solve, traced_commit
    before = state.snapshot()
    backend(*args, **kwargs)
    assert all(torch.equal(tensor, old) for tensor, old in zip(state.mutable_tensors, before.tensors, strict=True))
    timeline.append(("official_gated_norm", npu.active.name))
    backend.launch_pending_commit()
    timeline.append(("official_projection_mlp", npu.active.name))
    backend.finish_pending_commit()
    assert timeline == [
        ("prepare_map", "capture"),
        ("products", "capture", 0, 1),
        ("record", "capture", 2),
        ("wait", "side1", 2),
        ("solve", "side1", 0, 1),
        ("products", "capture", 1, 2),
        ("record", "capture", 3),
        ("wait", "side1", 3),
        ("solve", "side1", 1, 2),
        ("record", "side1", 4),
        ("wait", "capture", 4),
        ("official_gated_norm", "capture"),
        ("record", "capture", 0),
        ("wait", "side0", 0),
        ("commit", "side0"),
        ("record", "side0", 1),
        ("official_projection_mlp", "capture"),
        ("wait", "capture", 1),
    ]


def test_deferred_pipeline_requires_two_nonempty_request_ranges(log_backend):
    backend, _, _, calls, args, kwargs, _ = fixture_backend(log_backend, "deferred_pipeline", deferred=True)
    args = [tensor[:4] for tensor in args[:3]] + [args[3]]
    kwargs["num_accepted_tokens"] = kwargs["num_accepted_tokens"][:1]
    kwargs["actual_seq_lengths"] = kwargs["actual_seq_lengths"][:2]
    kwargs["ssm_state_indices"] = kwargs["ssm_state_indices"][:4]
    kwargs["g"], kwargs["beta"] = kwargs["g"][:4], kwargs["beta"][:4]
    with pytest.raises(ValueError, match="two captured requests"):
        backend(*args, **kwargs)
    assert not calls


def install_fixture_modules(log_backend, monkeypatch, backend):
    gdn = ModuleType("vllm_ascend.ops.gdn")
    original = lambda: "official"
    gdn.recurrent_gated_delta_rule = original
    ops = ModuleType("vllm_ascend.ops")
    ops.gdn = gdn
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.gdn", gdn)
    modules = dict(
        gdn_q4_log=backend.log_module,
        gdn_q4_products=backend.product_module,
        gdn_q4_log_deferred=backend.deferred_module,
    )
    monkeypatch.setattr(log_backend, "load_extension", lambda name, path: modules[name])
    return gdn, original


@pytest.mark.parametrize("launched", [False, True])
def test_undrained_context_rejects_result_and_closes_commit_before_restoring_official(
    log_backend, monkeypatch, tmp_path, launched
):
    backend, state, _, calls, args, kwargs, api = fixture_backend(log_backend, deferred=True)
    gdn, original = install_fixture_modules(log_backend, monkeypatch, backend)
    library = tmp_path / "deferred.so"
    library.write_bytes(b"deferred-provenance")
    with (
        pytest.raises(RuntimeError, match="without joining"),
        log_backend.installed_log_backend(
            state, library, library, mode="deferred_overlap", deferred_path=library
        ) as active,
    ):
        active(*args, **kwargs)
        if launched:
            active.launch_pending_commit()
    assert gdn.recurrent_gated_delta_rule is original
    assert [call[0] for call in calls] == ["products", "solve", "commit"]
    assert api.npu.events == ["record", "wait", "record", "wait", "synchronize"]
    assert active._pending_commit is None and active._failed
    with pytest.raises(RuntimeError, match="cannot be reused"):
        active(*args, **kwargs)


def test_deferred_hooked_context_completes_and_reports_provenance(log_backend, monkeypatch, tmp_path):
    backend, state, _, _, args, kwargs, _ = fixture_backend(log_backend, deferred=True)
    gdn, original = install_fixture_modules(log_backend, monkeypatch, backend)
    library = tmp_path / "deferred.so"
    library.write_bytes(b"deferred-provenance")
    with log_backend.installed_log_backend(
        state, library, library, mode="deferred_overlap", deferred_path=library
    ) as active:
        active(*args, **kwargs)
        active.launch_pending_commit()
        active.finish_pending_commit()
        assert active.sources["deferred"]["path"] == str(library)
        with active.suspended():
            assert gdn.recurrent_gated_delta_rule is original
            assert not active.launch_pending_commit() and not active.finish_pending_commit()
    assert gdn.recurrent_gated_delta_rule is original


def test_installed_backend_suspends_official_and_restores_on_failure(log_backend, monkeypatch, tmp_path):
    backend, state, _, _, _, _, _ = fixture_backend(log_backend)
    gdn = ModuleType("vllm_ascend.ops.gdn")
    original = lambda: "official"
    gdn.recurrent_gated_delta_rule = original
    ops = ModuleType("vllm_ascend.ops")
    ops.gdn = gdn
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.gdn", gdn)
    monkeypatch.setattr(
        log_backend,
        "load_extension",
        lambda name, path: backend.log_module if name == "gdn_q4_log" else backend.product_module,
    )
    library = tmp_path / "extension.so"
    library.write_bytes(b"test-provenance")
    with (
        pytest.raises(RuntimeError, match="fixture failed"),
        log_backend.installed_log_backend(state, library, library) as active,
    ):
        assert gdn.recurrent_gated_delta_rule is active
        with active.suspended() as official:
            assert official is original and gdn.recurrent_gated_delta_rule() == "official"
        assert gdn.recurrent_gated_delta_rule is active
        raise RuntimeError("fixture failed")
    assert gdn.recurrent_gated_delta_rule is original
    with pytest.raises(RuntimeError, match="never called"), log_backend.installed_log_backend(state, library, library):
        pass
    assert gdn.recurrent_gated_delta_rule is original


def test_direct_base_adopts_exact_fp32_storage_and_keeps_handle_opaque(log_backend):
    storage = torch.empty((3, 24 * 128 * 128 + 128), dtype=torch.float32)
    base = storage[:, : 24 * 128 * 128].view(3, 24, 128, 128)
    base.copy_(torch.arange(3).view(3, 1, 1, 1))
    slots = [[7, 2, 9, 4], [1, 8, 5, 10]]
    state = log_backend.Q4LogState.from_base(torch, base, slots)
    assert state.base is base and state.base.data_ptr() == storage.data_ptr()
    assert state.dense_state is None and state.dense_identity is None
    assert state.initial_accepted == (1, 1)
    assert state.device_owner_lookup.tolist() == [0, 2, 1, -1, 1, 2, -1, 1, 2, 1, 2]
    assert not state.logs.count_nonzero() and not state.valid.count_nonzero()
    assert type(state.handle) is log_backend.CompactSSMHandle
    for attribute in ("shape", "dtype", "device", "data_ptr", "clone"):
        with pytest.raises(AttributeError):
            getattr(state.handle, attribute)
    with pytest.raises(TypeError):
        state.handle[0]


@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "empty", "quartet", "zero", "negative", "bool", "alias"])
def test_direct_base_rejects_ambiguous_layout_or_candidate_ownership(log_backend, bad):
    base = torch.empty((3, 24, 128, 128))
    slots = [[7, 2, 9, 4], [1, 8, 5, 10]]
    if bad == "dtype":
        base = base.bfloat16()
    elif bad == "shape":
        base = base[:2]
    elif bad == "stride":
        base = base.transpose(-1, -2)
    elif bad == "empty":
        slots = []
    elif bad == "quartet":
        slots[0].pop()
    elif bad in ("zero", "negative", "bool"):
        slots[0][0] = {"zero": 0, "negative": -1, "bool": True}[bad]
    else:
        slots[1][0] = slots[0][0]
    with pytest.raises(ValueError):
        log_backend.Q4LogState.from_base(torch, base, slots)


@pytest.mark.parametrize("foreign", ["dense", "base", "handle"])
def test_direct_backend_requires_exact_handle_without_dense_fallback(log_backend, foreign):
    backend, state, dense, calls, args, kwargs, _ = fixture_backend(log_backend, direct=True)
    wrong = {"dense": dense, "base": state.base, "handle": log_backend.CompactSSMHandle()}[foreign]
    with pytest.raises(ValueError, match="exact registered compact SSM handle"):
        backend(*args[:3], wrong, **kwargs)
    assert not calls
    backend(*args, **kwargs)
    report = backend.describe()
    assert report["opaque_compact_handle"]
    assert not report["dense_initialization_source_retained"]
    assert not report["dense_initialization_source_becomes_stale"]


def test_direct_backend_detects_replaced_base_storage(log_backend):
    backend, state, _, calls, args, kwargs, _ = fixture_backend(log_backend, direct=True)
    state.base = state.base.clone()
    with pytest.raises(ValueError, match="base storage"):
        backend(*args, **kwargs)
    assert not calls


def fixture_registry(module, *, mode="deferred_overlap"):
    fixtures = [fixture_backend(module, mode, deferred=True, direct=True) for _ in range(2)]
    registry = module.CompactSSMRegistry()
    projections = [torch.nn.Identity(), torch.nn.Identity()]
    for index, (fixture, projection) in enumerate(zip(fixtures, projections, strict=True)):
        backend, state = fixture[:2]
        registry.register(state.handle, backend, projection, name=f"layer_{index}")
    return registry, fixtures, projections


def execute_registry_step(registry, fixtures, projections):
    outputs = []
    for fixture, projection in zip(fixtures, projections, strict=True):
        _, _, _, _, args, kwargs, _ = fixture
        outputs.append(registry(*args, **kwargs))
        projection(outputs[-1])
    return outputs


def test_multilayer_registry_routes_exact_handles_and_hooks_each_projection(log_backend, monkeypatch):
    registry, fixtures, projections = fixture_registry(log_backend)
    gdn, original = install_fixture_modules(log_backend, monkeypatch, fixtures[0][0])
    before = [fixture[1].snapshot() for fixture in fixtures]
    with log_backend.installed_log_registry(registry) as active:
        assert active is registry and gdn.recurrent_gated_delta_rule is registry
        registry.prepare_step(4, [1, 0], [4, 2], generations=[0, 0])
        outputs = execute_registry_step(registry, fixtures, projections)
        assert outputs[0].data_ptr() != outputs[1].data_ptr()
        assert registry.finish_pending_commits() == 2
        registry.finish_step(4)
        for fixture, saved in zip(fixtures, before, strict=True):
            backend, state, _, calls, _, _, api = fixture
            assert [call[0] for call in calls] == ["products", "solve", "commit"]
            assert api.npu.events == ["record", "wait", "record", "wait"]
            assert backend.calls == 1 and state.generations == [1, 1]
            assert torch.equal(state.base[1:], saved.tensors[0][1:] + 1)
        assert registry.calls == 2 and registry.describe()["layers"] == 2
        with pytest.raises(RuntimeError, match="installed compact registry"):
            registry.register(fixtures[0][1].handle, fixtures[0][0], projections[0])
    assert gdn.recurrent_gated_delta_rule is original and not registry._installed
    assert all(not projection._forward_pre_hooks for projection in projections)


@pytest.mark.parametrize("bad", ["unknown", "dense", "non_spec"])
def test_multilayer_registry_rejects_foreign_or_nonspec_calls(log_backend, bad):
    registry, fixtures, _ = fixture_registry(log_backend)
    _, _, dense, calls, args, kwargs, _ = fixtures[0]
    args = list(args)
    if bad == "unknown":
        args[3] = log_backend.CompactSSMHandle()
    elif bad == "dense":
        args[3] = dense
    else:
        kwargs.pop("num_accepted_tokens")
    with pytest.raises(ValueError):
        registry(*args, **kwargs)
    assert not calls and registry.calls == 0


@pytest.mark.parametrize("duplicate", ["handle", "projection", "name", "storage", "dense", "foreign"])
def test_registry_registration_enforces_independent_layer_ownership(log_backend, duplicate):
    registry, fixtures, projections = fixture_registry(log_backend)
    backend, state, _, _, _, _, _ = fixture_backend(log_backend, direct=duplicate != "dense", deferred=True)
    handle, projection, name = state.handle, torch.nn.Identity(), "extra"
    if duplicate == "handle":
        backend, state = fixtures[0][:2]
        handle = state.handle
    elif duplicate == "projection":
        projection = projections[0]
    elif duplicate == "name":
        name = "layer_0"
    elif duplicate == "storage":
        state.base = fixtures[0][1].base
        state.mutable_tensors = (state.base, state.logs, state.valid)
    elif duplicate == "foreign":
        handle = log_backend.CompactSSMHandle()
    with pytest.raises(ValueError):
        registry.register(handle, backend, projection, name=name)
    assert len(registry.entries) == 2


def test_registry_cpu_snapshot_restores_canonical_staging_and_host_generations(log_backend):
    registry, fixtures, _ = fixture_registry(log_backend, mode="deferred_serial")
    for fixture in fixtures:
        backend, _, _, _, args, kwargs, _ = fixture
        backend(*args, **kwargs)
    registry.prepare_step(7, [1], [4], generations=[0])
    saved = registry.snapshot(destination="cpu")
    addresses = [tuple(tensor.data_ptr() for tensor in fixture[1].mutable_tensors) for fixture in fixtures]
    for fixture in fixtures:
        backend, state = fixture[:2]
        for tensor in state.mutable_tensors:
            tensor.add_(13)
        for lane in backend.buffers.values():
            lane["new_logs"].fill_(31)
            lane["new_valid"].fill_(1)
    registry.finish_step(7)
    registry.restore(saved)
    for index, fixture in enumerate(fixtures):
        backend, state = fixture[:2]
        assert state.pending == saved.states[index].pending
        assert state.generations == [0, 0] and state.last_step == -1
        assert addresses[index] == tuple(tensor.data_ptr() for tensor in state.mutable_tensors)
        for target, source in zip(state.mutable_tensors, saved.states[index].tensors, strict=True):
            assert source.device.type == "cpu" and target.data_ptr() != source.data_ptr()
            assert torch.equal(target, source)
        for identity, logs, valid in saved.staging[index]:
            lane = backend.buffers[identity]
            assert torch.equal(lane["new_logs"], logs) and torch.equal(lane["new_valid"], valid)


def test_registry_precapture_cpu_snapshot_zeroes_new_staging_lanes_on_restore(log_backend):
    registry, fixtures, _ = fixture_registry(log_backend, mode="deferred_serial")
    saved = registry.snapshot(destination="cpu")
    for fixture in fixtures:
        backend, _, _, _, args, kwargs, _ = fixture
        backend(*args, **kwargs)
    assert all(backend.buffers for _, backend, _, _ in registry.entries)
    registry.restore(saved)
    for fixture, initial in zip(fixtures, saved.states, strict=True):
        backend, state = fixture[:2]
        assert all(
            torch.equal(target, source) for target, source in zip(state.mutable_tensors, initial.tensors, strict=True)
        )
        for lane in backend.buffers.values():
            assert not lane["new_logs"].count_nonzero() and not lane["new_valid"].count_nonzero()


@pytest.mark.parametrize("bad", ["registry", "state", "staging"])
def test_registry_restore_prevalidates_all_layers_before_mutating_any(log_backend, bad):
    registry, fixtures, _ = fixture_registry(log_backend, mode="deferred_serial")
    for fixture in fixtures:
        backend, _, _, _, args, kwargs, _ = fixture
        backend(*args, **kwargs)
    saved = registry.snapshot(destination="cpu")
    if bad == "registry":
        saved = replace(saved, registry_identity=(log_backend.CompactSSMHandle(),))
    elif bad == "state":
        bad_state = replace(saved.states[1], tensors=saved.states[1].tensors[:2])
        saved = replace(saved, states=(saved.states[0], bad_state))
    else:
        identity, logs, valid = saved.staging[1][0]
        saved = replace(saved, staging=(saved.staging[0], ((identity, logs[:1], valid),)))
    for fixture in fixtures:
        for tensor in fixture[1].mutable_tensors:
            tensor.add_(8)
    before = [[tensor.clone() for tensor in fixture[1].mutable_tensors] for fixture in fixtures]
    with pytest.raises(ValueError):
        registry.restore(saved)
    for fixture, expected in zip(fixtures, before, strict=True):
        assert all(
            torch.equal(target, source) for target, source in zip(fixture[1].mutable_tensors, expected, strict=True)
        )


def test_registry_lifecycle_prepare_failure_rolls_back_prior_layers(log_backend):
    registry, fixtures, _ = fixture_registry(log_backend)
    fixtures[1][1].generations[0] = 3
    with pytest.raises(ValueError, match="stale or future"):
        registry.prepare_step(0, [0], [1], generations=[0])
    assert all(fixture[1].pending is None for fixture in fixtures)
    fixtures[1][1].generations[0] = 0
    registry.prepare_step(0, [0], [1])
    fixtures[1][1].pending = None
    with pytest.raises(RuntimeError, match="every compact layer"):
        registry.finish_step(0)
    assert fixtures[0][1].generations == [0, 0]


def test_registry_missing_projection_prevents_partial_joins_and_drains_on_exit(log_backend, monkeypatch):
    registry, fixtures, projections = fixture_registry(log_backend)
    gdn, original = install_fixture_modules(log_backend, monkeypatch, fixtures[0][0])
    with pytest.raises(RuntimeError, match="did not launch"), log_backend.installed_log_registry(registry):
        for fixture in fixtures:
            registry(*fixture[4], **fixture[5])
        projections[0](fixtures[0][0].buffers[next(iter(fixtures[0][0].buffers))]["output"])
        with pytest.raises(RuntimeError, match="joined"):
            registry.snapshot(destination="cpu")
        with pytest.raises(RuntimeError, match="joined"):
            registry.prepare_step(0, [0], [1])
        with pytest.raises(RuntimeError, match="join all"):
            registry.mode = "padding"
        registry.finish_pending_commits()
    assert gdn.recurrent_gated_delta_rule is original and not registry._installed
    assert all(fixture[0]._pending_commit is None and fixture[0]._failed for fixture in fixtures)
    assert all([call[0] for call in fixture[3]] == ["products", "solve", "commit"] for fixture in fixtures)
    assert all(not projection._forward_pre_hooks for projection in projections)


def test_registry_requires_every_registered_layer_to_execute(log_backend, monkeypatch):
    registry, fixtures, projections = fixture_registry(log_backend, mode="padding")
    gdn, original = install_fixture_modules(log_backend, monkeypatch, fixtures[0][0])
    with pytest.raises(RuntimeError, match="never called"), log_backend.installed_log_registry(registry):
        registry(*fixtures[0][4], **fixtures[0][5])
    assert gdn.recurrent_gated_delta_rule is original
    assert all(not projection._forward_pre_hooks for projection in projections)


def test_registry_removes_partial_hook_installation_on_setup_failure(log_backend, monkeypatch):
    registry, fixtures, projections = fixture_registry(log_backend)
    gdn, original = install_fixture_modules(log_backend, monkeypatch, fixtures[0][0])

    def fail_hook(hook):
        raise RuntimeError("projection setup failed")

    monkeypatch.setattr(projections[1], "register_forward_pre_hook", fail_hook)
    with pytest.raises(RuntimeError, match="projection setup failed"), log_backend.installed_log_registry(registry):
        pass
    assert gdn.recurrent_gated_delta_rule is original and not registry._installed
    assert not projections[0]._forward_pre_hooks


def test_registry_cleanup_continues_after_one_layer_reports_failure(log_backend, monkeypatch):
    registry, fixtures, projections = fixture_registry(log_backend)
    gdn, original = install_fixture_modules(log_backend, monkeypatch, fixtures[0][0])
    cleanup = fixtures[0][0]._cleanup_pending_commit

    def fail_after_cleanup():
        cleanup()
        raise RuntimeError("first cleanup failed")

    monkeypatch.setattr(fixtures[0][0], "_cleanup_pending_commit", fail_after_cleanup)
    with pytest.raises(RuntimeError, match="first cleanup failed"), log_backend.installed_log_registry(registry):
        execute_registry_step(registry, fixtures, projections)
        raise RuntimeError("model endpoint failed")
    assert gdn.recurrent_gated_delta_rule is original and not registry._installed
    assert all(fixture[0]._pending_commit is None and fixture[0]._failed for fixture in fixtures)
    assert all(not projection._forward_pre_hooks for projection in projections)
