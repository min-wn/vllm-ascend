# SPDX-License-Identifier: Apache-2.0
"""Full-model launch and traversal gates without importing accelerator runtime."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_benchmark(name):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_resolves_defaults_before_using_eager_options():
    module = load_benchmark("multi_graph_gdn_q4_full_loader")
    defaults = module.engine_options("checkpoint", 2, eager=False)
    eager = module.engine_options("checkpoint", 2, eager=True)
    assert "compilation_config" not in defaults
    assert defaults["speculative_config"] == {"method": "ngram", "num_speculative_tokens": 3}
    assert defaults["max_model_len"] >= 1795
    assert defaults["max_num_batched_tokens"] >= 1536
    assert eager["compilation_config"]["cudagraph_capture_sizes"] == []
    assert eager["enforce_eager"] and not defaults["enforce_eager"]
    with pytest.raises(ValueError, match="TP2"):
        module.engine_options("checkpoint", 4, eager=False)


def test_token_generation_is_rank_independent_and_preserves_global_rng():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    torch.manual_seed(37)
    before = torch.get_rng_state()
    first = module.deterministic_token_ids(torch, 128, 1000, 42)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(99)
    second = module.deterministic_token_ids(torch, 128, 1000, 42)
    assert torch.equal(first, second)
    assert not torch.equal(first, module.deterministic_token_ids(torch, 128, 1000, 43))
    assert first.min() >= 0 and first.max() < 1000
    assert first.dtype == torch.int64


@pytest.mark.parametrize("visits", [[1] * 63, [1] * 65, [0] + [1] * 63, [2] + [1] * 63])
def test_traversal_gate_rejects_incomplete_or_repeated_layers(visits):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    with pytest.raises(RuntimeError, match="exactly once"):
        module.verify_layer_visits(visits)


def test_model_observer_records_real_norm_pair_and_removes_hooks():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):
            return hidden + 1, residual + hidden

    class Norm(torch.nn.Module):
        def forward(self, hidden, residual):
            return hidden + residual, residual

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(64)])
            self.norm = Norm()

        def forward(self, hidden):
            residual = torch.zeros_like(hidden)
            for layer in self.layers:
                hidden, residual = layer(hidden, residual)
            return self.norm(hidden, residual)[0]

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.model = Inner()
    with module.model_observer(model) as observation:
        output = model.language_model.model(torch.ones(2))
        module.verify_layer_visits(observation["visits"])
        assert observation["endpoint"][0] is output
        assert observation["endpoint"][1].shape == output.shape
    assert not model.language_model.model.norm._forward_hooks
    assert all(not layer._forward_hooks for layer in model.language_model.model.layers)


def fia_graph(layer_names):
    return SimpleNamespace(tasks=[SimpleNamespace(provider=SimpleNamespace(layer_name=name)) for name in layer_names])


def test_fia_task_gate_accepts_all_sixteen_layers_in_any_capture_order():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    names = [f"model.layers.{index}.self_attn.attn" for index in range(3, 64, 4)]
    assert len(names) == 16
    module.verify_fia_tasks(fia_graph(names[::-1]), names)


@pytest.mark.parametrize("bad", ["missing", "repeated", "foreign", "extra", "unnamed"])
def test_fia_task_gate_rejects_missing_repeated_or_foreign_provider(bad):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    names = [f"fia_{index}" for index in range(16)]
    actual = names.copy()
    if bad == "missing":
        actual.pop()
    elif bad == "repeated":
        actual[-1] = actual[0]
    elif bad == "foreign":
        actual[-1] = "unregistered_layer"
    elif bad == "extra":
        actual.append(actual[0])
    else:
        actual[-1] = None
    with pytest.raises(RuntimeError, match="FIA"):
        module.verify_fia_tasks(fia_graph(actual), names)


@pytest.mark.parametrize("bad", ["missing_declaration", "repeated_declaration"])
def test_fia_task_gate_does_not_accept_a_self_consistent_incomplete_layer_catalog(bad):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    names = [f"fia_{index}" for index in range(16)]
    if bad == "missing_declaration":
        names.pop()
    else:
        names[-1] = names[0]
    with pytest.raises(RuntimeError, match="16|sixteen|FIA"):
        module.verify_fia_tasks(fia_graph(sorted(set(names))), names)


def fixture_cache_bindings():
    conv, base, key, value = (torch.zeros(2, 3) for _ in range(4))
    handle = object()
    return SimpleNamespace(
        gdn_entries=[
            SimpleNamespace(
                index=0,
                conv=conv,
                base=base,
                state=SimpleNamespace(handle=handle),
                gdn=SimpleNamespace(kv_cache=(conv, handle)),
            )
        ],
        fia_entries=[
            SimpleNamespace(
                index=3,
                key=key,
                value=value,
                backend=SimpleNamespace(kv_cache=(key, value)),
                impl=SimpleNamespace(key_cache=key, value_cache=value),
            )
        ],
    )


def test_cache_binding_identity_retains_addresses_and_opaque_handle_identity():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    bundle = fixture_cache_bindings()
    gdn, fia = bundle.gdn_entries[0], bundle.fia_entries[0]
    expected = (
        ((gdn.conv.data_ptr(), gdn.base.data_ptr(), id(gdn.state.handle)),),
        ((fia.key.data_ptr(), fia.value.data_ptr()),),
    )
    assert module.cache_binding_identity(bundle) == expected
    for tensor in (gdn.conv, gdn.base, fia.key, fia.value):
        tensor.add_(7)
    assert module.cache_binding_identity(bundle) == expected


@pytest.mark.parametrize("field", ["conv", "handle"])
def test_cache_binding_identity_rejects_gdn_declared_owner_replacement(field):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    bundle = fixture_cache_bindings()
    item = bundle.gdn_entries[0]
    if field == "conv":
        item.gdn.kv_cache = (item.conv.clone(), item.state.handle)
    else:
        item.gdn.kv_cache = (item.conv, object())
    with pytest.raises(RuntimeError, match="GDN layer 0.*declared conv/compact handle"):
        module.cache_binding_identity(bundle)


@pytest.mark.parametrize("target,field", [("backend", "key"), ("backend", "value"), ("impl", "key"), ("impl", "value")])
def test_cache_binding_identity_rejects_independently_rebound_fia_consumers(target, field):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    bundle = fixture_cache_bindings()
    item = bundle.fia_entries[0]
    replacement = getattr(item, field).clone()
    if target == "backend":
        item.backend.kv_cache = (replacement, item.value) if field == "key" else (item.key, replacement)
    else:
        setattr(item.impl, f"{field}_cache", replacement)
    with pytest.raises(RuntimeError, match="FIA layer 3 cache binding changed"):
        module.cache_binding_identity(bundle)


@pytest.mark.parametrize("owner,field", [("gdn", "conv"), ("gdn", "base"), ("fia", "key"), ("fia", "value")])
def test_cache_binding_identity_changes_if_same_bound_tensor_gets_new_storage(owner, field):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    bundle = fixture_cache_bindings()
    before = module.cache_binding_identity(bundle)
    item = bundle.gdn_entries[0] if owner == "gdn" else bundle.fia_entries[0]
    tensor = getattr(item, field)
    # Tensor identity alone cannot protect graph pointers after storage rebinding.
    replacement = torch.zeros_like(tensor)
    assert replacement.data_ptr() != tensor.data_ptr()
    tensor.set_(replacement)
    after = module.cache_binding_identity(bundle)
    assert after != before
    if owner == "gdn":
        assert item.gdn.kv_cache[0] is item.conv and item.gdn.kv_cache[1] is item.state.handle
    else:
        assert item.backend.kv_cache[0] is item.key and item.impl.key_cache is item.key
        assert item.backend.kv_cache[1] is item.value and item.impl.value_cache is item.value


def test_cache_binding_identity_changes_after_coherent_compact_handle_replacement():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    bundle = fixture_cache_bindings()
    before = module.cache_binding_identity(bundle)
    item = bundle.gdn_entries[0]
    item.state.handle = object()
    item.gdn.kv_cache = (item.conv, item.state.handle)
    assert module.cache_binding_identity(bundle) != before


def test_latency_statistics_reports_exact_order_statistics_without_mutating_samples():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    samples = list(range(100, -1, -1))
    original = samples.copy()
    assert module.latency_statistics(samples) == dict(mean_us=50, p50_us=50, p95_us=95, p99_us=99)
    assert samples == original


@pytest.mark.parametrize(
    "samples,expected",
    [
        ([200, 0, 100], dict(mean_us=100, p50_us=100, p95_us=190, p99_us=198)),
        ([17], dict(mean_us=17, p50_us=17, p95_us=17, p99_us=17)),
    ],
)
def test_latency_statistics_interpolates_tails_and_accepts_single_sample(samples, expected):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    assert module.latency_statistics(samples) == expected


def test_latency_statistics_rejects_missing_samples():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    with pytest.raises(ValueError, match="nonempty"):
        module.latency_statistics([])


class SnapshotDeviceTensor:
    """Small CPU storage that enforces the accelerator snapshot transfer API."""

    def __init__(self, name, value, timeline):
        self.name, self.value, self.timeline = name, value, timeline

    def clone(self):
        raise AssertionError("complete snapshots must not clone state on the accelerator")

    def to(self, *, device, copy):
        assert torch.device(device).type == "cpu" and copy
        self.timeline.append(("to_cpu", self.name))
        return self.value.to(device="cpu", copy=True)

    def cpu(self):
        return self.to(device="cpu", copy=True)

    def copy_(self, saved):
        assert isinstance(saved, torch.Tensor) and saved.device.type == "cpu"
        self.timeline.append(("restore", self.name))
        self.value.copy_(saved)
        return self

    def data_ptr(self):
        return self.value.data_ptr()


class SnapshotRegistry:
    def __init__(self, tensors, timeline):
        self.tensors, self.timeline = tensors, timeline

    def finish_pending_commits(self):
        self.timeline.append(("join",))

    def snapshot(self, destination):
        assert destination == "cpu"
        self.timeline.append(("registry_snapshot",))
        return tuple(tensor.to(device=destination, copy=True) for tensor in self.tensors)

    def restore(self, saved):
        self.timeline.append(("registry_restore",))
        for tensor, value in zip(self.tensors, saved, strict=True):
            tensor.copy_(value)


def fixture_complete_snapshot():
    timeline, tensors = [], []

    def tensor(name, shape, dtype=torch.float32):
        value = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).view(shape)
        wrapped = SnapshotDeviceTensor(name, value, timeline)
        tensors.append(wrapped)
        return wrapped

    canonical = [
        tensor(f"gdn_{index}_{field}", (2, 3), torch.int32 if field == "valid" else torch.float32)
        for index in range(2)
        for field in ("base", "logs", "valid", "staging_logs", "staging_valid")
    ]
    bundle = SimpleNamespace(
        gdn_entries=[SimpleNamespace(conv=tensor(f"conv_{index}", (3, 2, 4))) for index in range(2)],
        fia_entries=[
            SimpleNamespace(
                key=tensor(f"fia_{index}_key", (3, 2, 5), torch.bfloat16),
                value=tensor(f"fia_{index}_value", (3, 2, 5), torch.bfloat16),
            )
            for index in range(2)
        ],
    )
    registry = SnapshotRegistry(canonical, timeline)
    api = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: timeline.append(("synchronize",))))
    return api, bundle, registry, tensors, timeline


def test_complete_cpu_snapshot_joins_then_transfers_each_cache_without_device_clone():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    api, bundle, registry, tensors, timeline = fixture_complete_snapshot()
    saved = module.complete_snapshot(api, bundle, registry)
    assert timeline[:3] == [("join",), ("synchronize",), ("registry_snapshot",)]
    copied = [name for event, *rest in timeline if event == "to_cpu" for name in rest]
    assert copied == [tensor.name for tensor in tensors]
    assert len(saved["compact"]) == 10 and len(saved["conv"]) == len(saved["fia"]) == 2
    flat = [*saved["compact"], *saved["conv"], *(value for pair in saved["fia"] for value in pair)]
    for live, cpu in zip(tensors, flat, strict=True):
        assert cpu.device.type == "cpu" and cpu.data_ptr() != live.data_ptr()
        assert torch.equal(cpu, live.value)
        live.value.add_(7)
        assert not torch.equal(cpu, live.value)


def test_complete_restore_restores_canonical_conv_and_fia_in_place_then_synchronizes():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    api, bundle, registry, tensors, timeline = fixture_complete_snapshot()
    addresses = [tensor.data_ptr() for tensor in tensors]
    saved = module.complete_snapshot(api, bundle, registry)
    before = [tensor.value.to(device="cpu", copy=True) for tensor in tensors]
    for tensor in tensors:
        tensor.value.add_(19)
    timeline.clear()
    module.restore_complete(api, bundle, registry, saved)
    assert timeline[0] == ("registry_restore",) and timeline[-1] == ("synchronize",)
    assert [name for event, *rest in timeline if event == "restore" for name in rest] == [
        tensor.name for tensor in tensors
    ]
    assert addresses == [tensor.data_ptr() for tensor in tensors]
    assert all(torch.equal(tensor.value, original) for tensor, original in zip(tensors, before, strict=True))


def test_replay_attribution_keeps_large_ready_kernel_gain_in_public_padding_baseline():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    means = dict(
        deferred_overlap=120,
        deferred_tile_separate=110,
        deferred_tile_delayed=100,
        deferred_tile_ready=80,
    )
    original = means.copy()
    result = module.replay_attribution(means, tile_enabled=True)
    assert result["scope"] == "public_full_bucket_kernel_pipeline"
    assert result["measured_variant"] == "deferred_tile_ready"
    assert result["algorithm_control"] == "deferred_tile_delayed"
    assert result["algorithm_reduction_percent"] == pytest.approx(20)
    assert result["public_padding_modes"] == sorted(means)
    assert result["fastest_public_padding"] == "deferred_tile_ready"
    assert result["split_parallel_candidate"] is None
    assert result["split_incremental_reduction_percent"] is None
    assert result["split_attribution_proven"] is False
    assert "padding can use it unchanged" in result["limitation"]
    assert means == original


@pytest.mark.parametrize("fastest", ["deferred_overlap", "deferred_tile_separate", "deferred_tile_delayed"])
def test_replay_attribution_selects_fastest_public_padding_even_when_ready_is_slower(fastest):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    means = dict(
        deferred_tile_ready=80,
        deferred_tile_delayed=100,
        deferred_tile_separate=110,
        deferred_overlap=120,
    )
    means[fastest] = 75
    result = module.replay_attribution(means, tile_enabled=True)
    assert result["fastest_public_padding"] == fastest
    assert result["algorithm_control"] == fastest
    assert result["algorithm_reduction_percent"] == pytest.approx(100 * (1 - 80 / 75))
    assert set(result["public_padding_modes"]) == set(means)
    assert "deferred_tile_ready" in result["public_padding_modes"]
    assert result["split_incremental_reduction_percent"] is None
    assert result["split_attribution_proven"] is False


def test_replay_attribution_without_tile_control_cannot_prove_large_range_pipeline_gain():
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    result = module.replay_attribution(dict(deferred_overlap=100, deferred_pipeline=70), tile_enabled=False)
    assert result["scope"] == "request_range_pipeline_without_strongest_tile_padding_control"
    assert result["measured_variant"] == "deferred_pipeline"
    assert result["algorithm_control"] == "deferred_overlap"
    assert result["algorithm_reduction_percent"] == pytest.approx(30)
    assert result["public_padding_modes"] == ["deferred_overlap"]
    assert result["fastest_public_padding"] == "deferred_overlap"
    assert result["split_parallel_candidate"] is None
    assert result["split_incremental_reduction_percent"] is None
    assert result["split_attribution_proven"] is False
    assert "does not include the public ready-tile padding kernel" in result["limitation"]


@pytest.mark.parametrize("tile_enabled", [False, True])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_replay_attribution_rejects_incomplete_or_undeclared_controls(tile_enabled, change):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    means = (
        dict(deferred_overlap=120, deferred_tile_separate=110, deferred_tile_delayed=100, deferred_tile_ready=80)
        if tile_enabled
        else dict(deferred_overlap=100, deferred_pipeline=80)
    )
    if change == "missing":
        means.pop("deferred_overlap")
    else:
        means["undeclared_faster_algorithm"] = 1
    with pytest.raises(ValueError, match="exactly the declared controls"):
        module.replay_attribution(means, tile_enabled=tile_enabled)


@pytest.mark.parametrize(
    "tile_enabled,invalid_mode",
    [
        (False, "deferred_overlap"),
        (False, "deferred_pipeline"),
        (True, "deferred_overlap"),
        (True, "deferred_tile_separate"),
        (True, "deferred_tile_delayed"),
        (True, "deferred_tile_ready"),
    ],
)
@pytest.mark.parametrize("invalid_latency", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_replay_attribution_rejects_invalid_latency_in_any_control(tile_enabled, invalid_mode, invalid_latency):
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    means = (
        dict(deferred_overlap=120, deferred_tile_separate=110, deferred_tile_delayed=100, deferred_tile_ready=80)
        if tile_enabled
        else dict(deferred_overlap=100, deferred_pipeline=80)
    )
    means[invalid_mode] = invalid_latency
    with pytest.raises(ValueError, match="finite positive latencies"):
        module.replay_attribution(means, tile_enabled=tile_enabled)


def test_launcher_keeps_crash_handler_without_starting_unsafe_frame_watchdog(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    module = load_benchmark("multi_graph_gdn_q4_full_probe")
    import multi_graph_gdn_q4_full_loader as loader

    calls = []
    monkeypatch.setattr(module.faulthandler, "enable", lambda: calls.append("natural_crashes"))

    def forbidden(*args, **kwargs):
        raise AssertionError("periodic unsynchronized frame reader must not be started")

    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", forbidden)
    monkeypatch.setattr(loader, "run_full_model", lambda args, trace, **kwargs: calls.append("run"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--extension",
            "log.so",
            "--products-extension",
            "products.so",
            "--deferred-extension",
            "deferred.so",
            "--output",
            str(tmp_path / "new.json"),
        ],
    )
    module.main()
    assert calls == ["natural_crashes", "run"]
