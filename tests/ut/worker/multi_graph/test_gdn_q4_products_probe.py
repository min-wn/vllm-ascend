# SPDX-License-Identifier: Apache-2.0
"""CPU checks for device product packing and accepted-candidate addressing."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def products():
    source = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_q4_products_probe.py"
    spec = importlib.util.spec_from_file_location("q4_products_probe_cpu_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("columns", [8, 16])
def test_raw_query_and_key_order_grouped_heads_and_padding(products, columns):
    # Distinct head/token lanes detect accidental [K,Q] or GQA permutation.
    query = torch.arange(2 * 4 * 2 * 8).reshape(2, 4, 2, 8).bfloat16()
    key = (-query - 1).bfloat16()
    actual = products.pack_rhs(query, key, 6, columns)
    assert actual.shape == (2, 6, 8, columns)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    for request in range(2):
        for value_head in range(6):
            for token in range(4):
                key_head = value_head // 3
                torch.testing.assert_close(actual[request, value_head, :, token], key[request, token, key_head])
                torch.testing.assert_close(actual[request, value_head, :, token + 4], query[request, token, key_head])
    assert not actual[..., 8:].count_nonzero()


def test_selected_cache_slots_follow_stable_ids_and_each_accepted_prefix(products):
    request_ids = torch.tensor([3, -1, 0, 2, 1, 3, 3, 3])
    accepted = torch.tensor([1, 4, 4, 2, 3, 2, 3, 4])
    observed = products.selected_indices(request_ids, accepted)
    assert observed.tolist() == [13, 0, 4, 10, 7, 14, 15, 16]
    cache = torch.arange(17 * 2 * 3 * 3).reshape(17, 2, 3, 3).bfloat16()
    gathered = cache.index_select(0, observed)
    for row, slot in enumerate([13, 0, 4, 10, 7, 14, 15, 16]):
        torch.testing.assert_close(gathered[row], cache[slot], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["columns", "tokens", "heads", "empty", "dtype", "different"])
def test_packing_rejects_unsupported_contract(products, invalid):
    query = torch.ones(2, 4, 2, 8, dtype=torch.bfloat16)
    key, heads, columns = query.clone(), 6, 8
    if invalid == "columns":
        columns = 12
    elif invalid == "tokens":
        query, key = query[:, :3], key[:, :3]
    elif invalid == "heads":
        heads = 5
    elif invalid == "empty":
        query, key = query[:, :, :0], key[:, :, :0]
    elif invalid == "dtype":
        query = query.float()
    else:
        key = key[..., :7]
    with pytest.raises(ValueError):
        products.pack_rhs(query, key, heads, columns)


def test_product_addressing_and_packing_match_independent_token_products(products):
    generator = torch.Generator().manual_seed(31)
    cache = torch.randn(13, 6, 8, 8, generator=generator).bfloat16()
    query, key = tuple(torch.randn(3, 4, 2, 8, generator=generator).bfloat16() for _ in range(2))
    ids, accepted = torch.tensor([2, 0, -1]), torch.tensor([3, 4, 2])
    base = cache.index_select(0, products.selected_indices(ids, accepted)).float()
    actual = torch.matmul(base, products.pack_rhs(query, key, 6, 8).float())
    for row, cache_slot in enumerate([11, 4, 0]):
        for head in range(6):
            for token in range(4):
                matrix = cache[cache_slot, head].float()
                torch.testing.assert_close(actual[row, head, :, token], matrix @ key[row, token, head // 3].float())
                torch.testing.assert_close(
                    actual[row, head, :, token + 4], matrix @ query[row, token, head // 3].float()
                )
