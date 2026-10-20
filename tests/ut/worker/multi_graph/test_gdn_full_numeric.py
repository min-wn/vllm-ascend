# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for independent numerical and exact audit interpretation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    result = []
    for name in ("multi_graph_gdn_cpu_audit", "multi_graph_gdn_full_numeric"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float64, torch.int32))
def test_array_scan_preserves_signed_zero_and_nonfinite_semantics(modules, dtype):
    scan, _ = modules
    value = torch.zeros((2, 8), dtype=dtype).t()
    assert scan.finite(torch, value) and scan.zero(torch, value)
    other = value.clone()
    if dtype != torch.int32:
        other[0, 0] = -0.0
        assert scan.zero(torch, other)
        assert not scan.identical(torch, value, other)
        for special in (float("nan"), float("inf"), -float("inf")):
            other[0, 0] = special
            assert not scan.finite(torch, other)
    other[0, 0] = 1
    assert not scan.zero(torch, other)
    assert not scan.identical(torch, value, other)


def test_numeric_metrics_keep_owner_axis_and_do_not_convert_error_to_exactness(modules):
    _, numeric = modules
    reference = torch.ones(2, 4, 3)
    actual = reference.clone()
    actual[1] *= 1.01
    row = numeric.error_record(torch, actual, reference, label="hidden", step=7, owners=[91, 12])
    assert row["worst_owner"] == 12
    assert row["worst_request_nrmse"] == pytest.approx(0.01)
    assert not row["screen_passed"]
    assert numeric.error_record(torch, reference, reference, label="hidden", step=7, owners=[91, 12])["screen_passed"]
    actual[0, 0, 0] = float("nan")
    with pytest.raises(AssertionError, match="nonfinite"):
        numeric.error_record(torch, actual, reference, label="hidden", step=7, owners=[91, 12])


@pytest.fixture
def official_harness(modules):
    _, numeric = modules
    entries = []

    class Layer(torch.nn.Module):
        def __init__(self, entry):
            super().__init__()
            self.entry, self.owner = entry, 11

        def forward(self, value):
            conv, dense = self.entry.gdn.kv_cache
            slot = 1 + 4 * self.owner
            dense[slot : slot + 4].add_(1)
            conv[slot : slot + 4].add_(1)
            return value

    layers = []
    for index in range(2):
        base = torch.arange(129, dtype=torch.float32).view(129, 1, 1, 1) + index * 100
        base[0].zero_()
        conv = torch.zeros(513, 1)
        gdn = SimpleNamespace(kv_cache=(conv, object()))
        item = SimpleNamespace(index=index, base=base, conv=conv, gdn=gdn)
        entries.append(item)
        layers.append(Layer(item))
    key, value = torch.zeros(3, 2), torch.ones(3, 2)
    fia = SimpleNamespace(
        index=2,
        key=key,
        value=value,
        backend=SimpleNamespace(kv_cache=(key, value)),
        impl=SimpleNamespace(key_cache=key, value_cache=value),
    )
    bundle = SimpleNamespace(gdn_entries=entries, fia_entries=[fia], device=torch.device("cpu"))
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    return numeric.OfficialBanks(torch, bundle), bundle, model, layers


def test_official_histories_are_independent_and_shared_scratch_does_not_alias_layers(official_harness):
    banks, bundle, model, layers = official_harness
    original = [item.gdn.kv_cache for item in bundle.gdn_entries]
    fia = bundle.fia_entries[0]
    with banks.installed(model, [11]):
        for layer in layers:
            layer(torch.tensor(1))
        fia.impl.key_cache.add_(3)
    for index, item in enumerate(bundle.gdn_entries):
        assert item.gdn.kv_cache is original[index]
        assert banks.dense[index][45].item() == 13 + index * 100
        assert banks.dense[index][1].item() == 1 + index * 100
        assert item.conv.count_nonzero().item() == 0
        assert banks.conv[index][45].item() == 1
        item.base[12].add_(999)
        assert banks.dense[index][45].item() == 13 + index * 100
    assert fia.backend.kv_cache[0] is fia.key and fia.impl.key_cache is fia.key
    assert fia.key.count_nonzero().item() == 0 and torch.equal(banks.fia[2][0], torch.full_like(fia.key, 3))
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in layers)


def test_official_paused_state_violation_restores_every_cache_and_removes_hooks(official_harness):
    banks, bundle, model, layers = official_harness
    original = [item.gdn.kv_cache for item in bundle.gdn_entries]
    layers[0].owner = 3
    with pytest.raises(AssertionError, match="inactive state changed"), banks.installed(model, [11]):
        layers[0](torch.tensor(1))
    assert all(item.gdn.kv_cache is cache for item, cache in zip(bundle.gdn_entries, original, strict=True))
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in layers)
