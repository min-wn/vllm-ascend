import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("chunk", [0, 1, 4, 100])
def test_update_groups_preserve_every_task_once_in_lane_order(benchmark, chunk):
    groups = benchmark.update_groups([36, 31], chunk)
    for lane, count in enumerate((36, 31)):
        indices = [index for selected, start, stop in groups if selected == lane for index in range(start, stop)]
        assert indices == list(range(count))
    if chunk and chunk < 31:
        assert groups[:2] == [(0, 0, chunk), (1, 0, chunk)]
    if chunk == 0:
        assert groups == [(0, 0, 36), (1, 0, 31)]


@pytest.mark.parametrize("counts,chunk", [([], 0), ([0], 1), ([36], -1)])
def test_update_groups_reject_invalid_counts(benchmark, counts, chunk):
    with pytest.raises(ValueError):
        benchmark.update_groups(counts, chunk)


def test_projection_selection_preserves_shared_weights_and_other_layers(benchmark):
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(qkv_proj=object(), o_proj=object()),
            mlp=SimpleNamespace(gate_up_proj=object(), down_proj=object()),
        )
        for _ in range(2)
    ]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    selected = benchmark.projection_modules(model, ["qkv", "down"])
    assert selected == [
        (i, name, module)
        for i, layer in enumerate(layers)
        for name, module in (("qkv", layer.self_attn.qkv_proj), ("down", layer.mlp.down_proj))
    ]
    assert benchmark.projection_modules(model, []) == []


@pytest.mark.parametrize("names", [["unknown"], ["qkv", "qkv"]])
def test_projection_selection_rejects_invalid_names_before_model_access(benchmark, names):
    with pytest.raises(ValueError, match="unique supported"):
        benchmark.projection_modules(None, names)
