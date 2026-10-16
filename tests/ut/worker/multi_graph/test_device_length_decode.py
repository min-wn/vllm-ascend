import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def prototype(monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args: calls.append((grid, args))

    triton = types.ModuleType("triton")
    triton.jit = lambda function: Kernel()
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/device_length_decode.py"
    spec = importlib.util.spec_from_file_location("device_length_decode_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.device_length_decode, calls


def arguments():
    return [
        torch.empty((2, 4, 64), dtype=torch.float16),
        torch.empty((8, 128, 2, 64), dtype=torch.float16),
        torch.empty((8, 128, 2, 64), dtype=torch.float16),
        torch.empty(2, dtype=torch.int32),
        torch.empty((2, 4), dtype=torch.int32),
        torch.empty((2, 4, 64), dtype=torch.float16),
    ]


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (0, torch.empty((2, 3, 64), dtype=torch.float16)),
        (1, torch.empty((8, 64, 2, 64), dtype=torch.float16)),
        (2, torch.empty((8, 128, 2, 32), dtype=torch.float16)),
        (3, torch.empty(2, dtype=torch.int64)),
        (3, torch.empty(3, dtype=torch.int32)),
        (4, torch.empty((2, 0), dtype=torch.int32)),
        (5, torch.empty((2, 4, 64), dtype=torch.float32)),
        (5, torch.empty((2, 4, 128), dtype=torch.float16)[:, :, ::2]),
    ],
)
def test_invalid_descriptors_never_launch(prototype, index, replacement):
    launch, calls = prototype
    values = arguments()
    values[index] = replacement
    with pytest.raises(ValueError):
        launch(*values)
    assert calls == []


def test_launcher_does_not_read_device_metadata_values(prototype, monkeypatch):
    launch, calls = prototype

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata value readback")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    values = arguments()
    launch(*values)
    assert calls[0][0] == (2, 4, 1)
    assert calls[0][1][3] is values[3]
    assert calls[0][1][4] is values[4]


def test_split_workspace_preallocated_and_validated(prototype):
    launch, calls = prototype
    with pytest.raises(ValueError, match="preallocated"):
        launch(*arguments(), splits=4)
    assert calls == []
    workspace = torch.empty((2, 4, 4, 66), dtype=torch.float32)
    launch(*arguments(), splits=4, workspace=workspace)
    assert [grid for grid, _ in calls] == [(2, 4, 4), (2, 4)]
