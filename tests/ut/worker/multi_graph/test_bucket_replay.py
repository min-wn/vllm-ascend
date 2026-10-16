import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_bucket_replay.py"
    spec = importlib.util.spec_from_file_location("bucket_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_sample_uses_only_default_bucket_shapes(benchmark):
    sizes = [1, 2, 4, *range(8, 257, 8)]
    assert benchmark.bucket_pair(98, sizes, "exact") == (104, (96, 2))
    assert benchmark.bucket_pair(98, sizes, "balanced") == (104, (56, 48))
    assert benchmark.lane_counts(98, (96, 2)) == (96, 2)
    assert benchmark.lane_counts(98, (56, 48)) == (50, 48)


def test_reuse_never_exceeds_either_captured_lane(benchmark):
    for pair in ((96, 2), (56, 48), (4, 4)):
        for batch in range(2, sum(pair) + 1):
            left, right = benchmark.lane_counts(batch, pair)
            assert 0 < left <= pair[0]
            assert 0 < right <= pair[1]
            assert left + right == batch


def test_missing_pair_does_not_create_custom_graph(benchmark):
    with pytest.raises(ValueError, match="no legal pair"):
        benchmark.bucket_pair(7, [1, 2, 4, 8], "exact")
    with pytest.raises(ValueError, match="capture range"):
        benchmark.bucket_pair(9, [1, 2, 4, 8], "balanced")
    with pytest.raises(ValueError, match="nonempty"):
        benchmark.lane_counts(9, (4, 4))
