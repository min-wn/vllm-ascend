# SPDX-License-Identifier: Apache-2.0
"""Full-model launch and traversal gates without importing accelerator runtime."""

import importlib.util
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
