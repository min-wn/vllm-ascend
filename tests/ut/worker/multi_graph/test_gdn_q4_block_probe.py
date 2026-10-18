# SPDX-License-Identifier: Apache-2.0
"""CPU checks for probe precision, selected-state mapping and page canaries."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def block_probe():
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_q4_block_probe.py"
    spec = importlib.util.spec_from_file_location("gdn_q4_block_probe_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padding", [0, 32])
def test_page_storage_preserves_precision_and_gap_canaries(block_probe, dtype, padding):
    initial = torch.linspace(-0.13, 0.31, 5 * 3 * 8 * 8).view(5, 3, 8, 8).to(dtype)
    pages, state = block_probe.allocate_cache_pages(initial, padding)
    assert state.dtype == dtype
    assert state.stride() == (3 * 8 * 8 + padding, 8 * 8, 8, 1)
    torch.testing.assert_close(state, initial, rtol=0, atol=0)
    state[2].add_(1)
    torch.testing.assert_close(pages[2, : 3 * 8 * 8], (initial[2] + 1).flatten(), rtol=0, atol=0)
    if padding:
        assert torch.equal(pages[:, 3 * 8 * 8 :], torch.full((5, padding), 7, dtype=dtype))
    torch.testing.assert_close(state[1], initial[1], rtol=0, atol=0)
    if dtype == torch.float32:
        assert (state != state.bfloat16().float()).any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fold_gqa", [False, True])
def test_product_pack_maps_mixed_accepted_slots_without_precision_loss(block_probe, dtype, fold_gqa):
    torch.manual_seed(73)
    batch, heads, key_heads, width = 3, 6, 2, 8
    initial = torch.randn(13, heads, width, width).to(dtype)
    _, state = block_probe.allocate_cache_pages(initial, 32)
    q, k = (torch.randn(batch * 4, key_heads, width).bfloat16() for _ in range(2))
    ids = torch.tensor([[9, 10, 11, 12], [1, 2, 3, 4], [0, 0, 0, 0]], dtype=torch.int32).flatten()
    accepted = torch.tensor([2, 4, 1], dtype=torch.int32)
    base, rhs = block_probe.product_operands(state, q, k, ids, accepted, fold_gqa=fold_gqa)
    assert base.dtype == rhs.dtype == dtype
    assert base.is_contiguous() and rhs.is_contiguous()
    actual = torch.bmm(base.float(), rhs.float()).view(batch, heads, width, 8)
    expected = torch.empty_like(actual)
    for request, slot in enumerate((10, 4, 0)):
        for head in range(heads):
            key_head = head // (heads // key_heads)
            raw = torch.stack(
                [k[request * 4 + token, key_head] for token in range(4)]
                + [q[request * 4 + token, key_head] for token in range(4)],
                dim=1,
            ).float()
            expected[request, head] = initial[slot, head].float() @ raw
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_direct_products_converts_current_raw_qk_inside_each_call(block_probe, dtype):
    state = torch.randn(5, 3, 8, 8).to(dtype)
    q, k = (torch.randn(4, 1, 8).bfloat16() for _ in range(2))
    sequence, ids, accepted = torch.tensor([0, 4]), torch.arange(1, 5), torch.tensor([1])
    target = torch.empty(1, 3, 8, 8)
    calls = []

    def record(actual_state, actual_q, actual_k, *metadata):
        assert actual_state is state
        assert all(actual is expected for actual, expected in zip(metadata, (sequence, ids, accepted, target)))
        assert actual_q.dtype == actual_k.dtype == dtype
        calls.append((actual_q.clone(), actual_k.clone()))

    module = SimpleNamespace(products=record)
    assert block_probe.direct_products(module, state, q, k, sequence, ids, accepted, target) is target
    q.add_(1)
    k.mul_(2)
    block_probe.direct_products(module, state, q, k, sequence, ids, accepted, target)
    torch.testing.assert_close(calls[1][0], q.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(calls[1][1], k.to(dtype), rtol=0, atol=0)
    assert not torch.equal(calls[0][0], calls[1][0])
    assert q.dtype == k.dtype == torch.bfloat16
