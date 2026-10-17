# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU contracts for the real fused MRoPE wrapper; kernel math runs on NPU."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def operation(monkeypatch):
    # Load the production wrapper without registering or compiling an NPU op.
    monkeypatch.setitem(
        sys.modules,
        "vllm.triton_utils",
        SimpleNamespace(tl=SimpleNamespace(constexpr=object), triton=SimpleNamespace(jit=lambda **_: lambda fn: fn)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.torch_utils",
        SimpleNamespace(direct_register_custom_op=lambda **_: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.triton.triton_utils",
        SimpleNamespace(extract_slice=None, insert_slice=None, get_vectorcore_num=lambda: 8),
    )
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py"
    spec = importlib.util.spec_from_file_location("mrope_wrapper_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = MagicMock()
    monkeypatch.setattr(module, "split_qkv_rmsnorm_mrope_kernel", kernel)
    return module.triton_split_qkv_rmsnorm_mrope, kernel.__getitem__.return_value


def invoke(operation, cos_sin, rope_dim=64):
    function, _ = operation
    return function(
        qkv=torch.empty(4, 6 * 128, dtype=torch.bfloat16),
        q_weight=torch.ones(128, dtype=torch.bfloat16),
        k_weight=torch.ones(128, dtype=torch.bfloat16),
        cos_sin=cos_sin,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=128,
        eps=1e-6,
        mrope_section=[11, 11, 10],
        is_interleaved=True,
        rope_dim=rope_dim,
        has_gate=True,
    )


@pytest.mark.parametrize("layout", ["contiguous", "bucket_slice", "broadcast_slice"])
def test_passes_original_view_and_actual_plane_stride(operation, layout):
    if layout == "contiguous":
        cos_sin = torch.randn(3, 4, 64)
        expected_stride = 4 * 64
    elif layout == "bucket_slice":
        cos_sin = torch.randn(3, 10, 64)[:, 3:7]
        expected_stride = 10 * 64
    else:
        cos_sin = torch.randn(10, 64).unsqueeze(0).expand(3, -1, -1)[:, 3:7]
        expected_stride = 0
    outputs = invoke(operation, cos_sin)
    _, launch = operation
    launch.assert_called_once()
    arguments = launch.call_args.args
    assert arguments[5] is cos_sin
    assert arguments[6] == expected_stride
    assert [output.shape for output in outputs] == [(4, 256), (4, 128), (4, 128), (4, 256)]


@pytest.mark.parametrize(
    "shape",
    [(4, 64), (2, 4, 64), (3, 3, 64), (3, 4, 63), (3, 4, 64, 1)],
)
def test_rejects_invalid_extent_before_kernel_launch(operation, shape):
    with pytest.raises(ValueError, match="must have shape"):
        invoke(operation, torch.empty(shape))
    operation[1].assert_not_called()


@pytest.mark.parametrize("layout", ["stepped_tokens", "stepped_features", "broadcast_tokens"])
def test_rejects_unsupported_inner_strides(operation, layout):
    if layout == "stepped_tokens":
        cos_sin = torch.empty(3, 8, 64)[:, ::2]
    elif layout == "stepped_features":
        cos_sin = torch.empty(3, 4, 128)[:, :, ::2]
    else:
        cos_sin = torch.empty(3, 1, 64).expand(3, 4, 64)
    with pytest.raises(ValueError, match="dense token/feature strides"):
        invoke(operation, cos_sin)
    operation[1].assert_not_called()


@pytest.mark.parametrize("rope_dim", [0, -2, 63, 130])
def test_rejects_invalid_rotary_extent_before_kernel_launch(operation, rope_dim):
    with pytest.raises(ValueError, match="rope_dim must be"):
        invoke(operation, torch.empty(3, 4, 64), rope_dim=rope_dim)
    operation[1].assert_not_called()


def test_rejects_cross_device_cos_sin_before_kernel_launch(operation):
    with pytest.raises(ValueError, match="same device"):
        invoke(operation, torch.empty(3, 4, 64, device="meta"))
    operation[1].assert_not_called()
