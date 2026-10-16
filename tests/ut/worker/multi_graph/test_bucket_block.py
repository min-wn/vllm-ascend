import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_bucket_block.py"
    spec = importlib.util.spec_from_file_location("bucket_block", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topology_depends_on_bucket_not_live_batch(benchmark):
    sizes = [1, 2, 4, *range(8, 257, 8)]
    for batch in range(97, 105):
        assert benchmark.bucket_topology(batch, sizes) == (104, (56, 48))
    assert benchmark.bucket_topology(105, sizes) == (112, (56, 56))


def test_fixed_slots_preserve_request_offsets(benchmark):
    assert benchmark.fixed_lane_counts(98, (56, 48)) == (56, 42)
    assert benchmark.fixed_lane_counts(97, (56, 48)) == (56, 41)
    assert benchmark.fixed_lane_counts(103, (56, 48)) == (56, 47)
    assert benchmark.fixed_lane_counts(104, (56, 48)) == (56, 48)


def test_dynamic_sequence_stays_in_default_bucket(benchmark):
    sizes = [1, 2, 4, *range(8, 257, 8)]
    assert benchmark.reuse_cases(98, sizes) == [(98, 0), (97, 1), (103, 0), (98, 0)]
    for batch in range(2, 257):
        capacity, pair = benchmark.bucket_topology(batch, sizes)
        for live, variant in benchmark.reuse_cases(batch, sizes):
            assert benchmark.bucket_topology(live, sizes) == (capacity, pair)
            assert sum(benchmark.fixed_lane_counts(live, pair)) == live
            assert variant in (0, 1)


@pytest.mark.parametrize("batch,sizes", [(1, [1, 2]), (105, [48, 56, 104]), (98, [48, 104]), (2, [])])
def test_invalid_topology_does_not_create_special_shapes(benchmark, batch, sizes):
    with pytest.raises(ValueError):
        benchmark.bucket_topology(batch, sizes)


@pytest.mark.parametrize("batch,pair", [(105, (56, 48)), (0, (56, 48)), (98, (104,)), (98, (104, 0))])
def test_invalid_fixed_slots_are_rejected(benchmark, batch, pair):
    with pytest.raises(ValueError):
        benchmark.fixed_lane_counts(batch, pair)


@pytest.mark.parametrize("common,lanes,expected", [(0, None, (0, 0)), (16, None, (16, 16)), (0, [16, 12], (16, 12))])
def test_lane_core_limits_are_explicit(benchmark, common, lanes, expected):
    assert benchmark.lane_core_limits(common, lanes) == expected


@pytest.mark.parametrize("common,lanes", [(-1, None), (10, [16, 12]), (0, [16]), (0, [-1, 12])])
def test_invalid_lane_limits_are_rejected(benchmark, common, lanes):
    with pytest.raises(ValueError):
        benchmark.lane_core_limits(common, lanes)


def test_packed_runtime_values_use_disjoint_fixed_typed_views(benchmark):
    import torch

    targets = {
        "ids": torch.empty(3, dtype=torch.int64),
        "lengths": torch.empty(3, dtype=torch.int32),
        "pages": torch.empty((3, 2), dtype=torch.int32),
        "slots": torch.empty(3, dtype=torch.int64),
    }
    buffers = benchmark.PackedStepBuffers(targets)
    pointers = {name: tensor.data_ptr() for name, tensor in buffers.device.items()}
    previous_end = 0
    for name, item in buffers.schema.items():
        assert item["offset"] % 64 == 0 and item["offset"] >= previous_end
        previous_end = item["offset"] + item["bytes"]
        assert buffers.device[name].untyped_storage().data_ptr() == buffers.device_buffer.data_ptr()
    for token in (41, 76, 99):
        values = dict(
            ids=[token, token + 1, 0], lengths=[512, 511, 0], pages=[[1, 2], [3, 4], [0, 0]], slots=[127, 383, -1]
        )
        for name, value in values.items():
            buffers.write(name, value)
        buffers.transfer()
        for name, value in values.items():
            assert torch.equal(buffers.device[name], torch.tensor(value, dtype=targets[name].dtype))
            assert buffers.device[name].data_ptr() == pointers[name]


def test_packed_runtime_shapes_cannot_broadcast_or_change(benchmark):
    import torch

    buffers = benchmark.PackedStepBuffers({"ids": torch.empty(3, dtype=torch.int64)})
    for values in ([1], [[1, 2, 3]], [1, 2, 3, 4]):
        with pytest.raises(ValueError, match="runtime shape changed"):
            buffers.write("ids", values)
    with pytest.raises(KeyError):
        buffers.write("unknown", [1, 2, 3])


@pytest.mark.parametrize("kind", ["empty", "float", "zero"])
def test_packed_buffers_reject_invalid_schema(benchmark, kind):
    import torch

    targets = (
        {} if kind == "empty" else {"ids": torch.empty(0, dtype=torch.int64) if kind == "zero" else torch.empty(3)}
    )
    with pytest.raises(ValueError):
        benchmark.PackedStepBuffers(targets)
