import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def balance():
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_tp_kv_balance.py"
    spec = importlib.util.spec_from_file_location("tp_kv_balance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_and_clustered_have_identical_histogram(balance):
    profiles = balance.length_profiles(100, 98, 4, 512)
    assert sorted(profiles["random"]) == sorted(profiles["clustered"])
    assert profiles["random"][98:] == [0, 0]
    assert profiles["clustered"][:98] == sorted(profiles["random"][:98], reverse=True)
    assert profiles == balance.length_profiles(100, 98, 4, 512)


@pytest.mark.parametrize("capacities", [(50, 50), (68, 32)])
def test_balancing_preserves_capacity_and_request_identity(balance, capacities):
    lengths = balance.length_profiles(100, 98, 4, 512)["clustered"]
    contiguous = balance.assign_requests(lengths, capacities, False)
    assigned = balance.assign_requests(lengths, capacities, True)
    assert tuple(map(len, assigned)) == capacities
    assert sorted(sum((list(ids) for ids in assigned), [])) == list(range(100))
    cost = lambda lanes: max(sum(lengths[index] for index in ids) for ids in lanes)
    assert cost(assigned) < cost(contiguous)


def test_gpu_permutation_mapping_keeps_tokens_and_inverse_order(balance):
    requests = (3, 0, 2, 1)
    indices = balance.token_indices(requests, 4)
    original = list(range(16))
    permuted = [original[index] for index in indices]
    restored = [None] * 16
    for index, value in zip(indices, permuted, strict=True):
        restored[index] = value
    assert restored == original
    assert indices[:4] == [12, 13, 14, 15]


def test_metadata_moves_request_identity_without_moving_kv_history(balance):
    workload = SimpleNamespace(capacity=4, query=4, kv_length=512, pages_per_request=4, block_size=128)
    lengths = [128, 512, 256, 0]
    whole = balance.mapped_metadata(workload, lengths, range(4))
    mapped = balance.mapped_metadata(workload, lengths, (2, 0, 3, 1))
    for local, global_id in enumerate((2, 0, 3, 1)):
        assert mapped["pages"][local] == whole["pages"][global_id]
        assert (
            mapped["positions"][local * 4 : (local + 1) * 4] == whole["positions"][global_id * 4 : (global_id + 1) * 4]
        )
        assert mapped["slots"][local * 4 : (local + 1) * 4] == whole["slots"][global_id * 4 : (global_id + 1) * 4]
    assert mapped["pages"][2] == [0] * 4
    assert mapped["slots"][8:12] == [-1] * 4
    assert mapped["boundaries"] == [0, 4, 8, 12, 16]


@pytest.mark.parametrize("capacities", [(0, 4), (2, 1), (1, 1, 2)])
def test_invalid_assignment_quotas_rejected(balance, capacities):
    with pytest.raises(ValueError, match="capacities"):
        balance.assign_requests([4, 8, 12, 16], capacities, True)


def test_bad_or_duplicate_request_metadata_rejected(balance):
    workload = SimpleNamespace(capacity=2, query=4, kv_length=512, pages_per_request=4, block_size=128)
    with pytest.raises(ValueError, match="unique"):
        balance.mapped_metadata(workload, [128, 512], (1, 1))
    with pytest.raises(ValueError, match="identity"):
        balance.mapped_metadata(workload, [128, 512], (2,))
    with pytest.raises(ValueError, match="lengths"):
        balance.mapped_metadata(workload, [3, 512], (0, 1))
