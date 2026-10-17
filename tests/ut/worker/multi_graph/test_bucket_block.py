import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload


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


@pytest.mark.parametrize("order", ["original", "length_asc", "length_desc"])
def test_mixed_request_mapping_preserves_physical_kv_and_output_order(benchmark, order):
    workload = ReplayWorkload(7, 8, 1, 512, (4, 3))
    lengths = benchmark.request_lengths(8, 512, 128, 42)
    for batch, variant in ((7, 0), (6, 1), (7, 0)):
        payload = benchmark.ordered_request_metadata(workload, lengths, batch, variant, order)
        assert sum(payload["lengths"]) == sum(lengths[:batch]) - variant * batch
        assert sorted(payload["restore"]) == list(range(8))
        for request in range(batch):
            row = payload["restore"][request]
            position = lengths[request] - variant - 1
            assert payload["ids"][row] == 100 + variant + request
            assert payload["positions"][row] == position
            assert payload["pages"][row] == workload.metadata_values(request, request + 1)["pages"][0]
            page = payload["pages"][row][position // workload.block_size]
            assert payload["slots"][row] == page * workload.block_size + position % workload.block_size
        assert payload["slots"][batch:] == [-1] * (8 - batch)
        assert payload["lengths"][batch:] == [0] * (8 - batch)
        if order != "original":
            assert payload["lengths"][:batch] == sorted(payload["lengths"][:batch], reverse=order == "length_desc")


def test_mixed_lengths_reject_invalid_bounds_and_order(benchmark):
    workload = ReplayWorkload(7, 8, 1, 512, (4, 3))
    for short in (0, 1, 513):
        with pytest.raises(ValueError):
            benchmark.request_lengths(8, 512, short, 42)
    assert benchmark.request_lengths(8, 512, 128, 42) == benchmark.request_lengths(8, 512, 128, 42)
    with pytest.raises(ValueError):
        benchmark.ordered_request_metadata(workload, [128] * 8, 7, 0, "invalid")
    with pytest.raises(ValueError):
        benchmark.ordered_request_metadata(workload, [128] * 8, 7, 128, "original")


@pytest.mark.parametrize("compiled,experiment", [(False, None), (True, lambda: None)])
def test_scoped_snapshot_rejects_full_or_compiled_model(compiled, experiment):
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_replay.py"
    spec = importlib.util.spec_from_file_location("replay_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="explicit uncompiled block"):
        module.run(SimpleNamespace(compiled=compiled), None, experiment=experiment, snapshot_first_block=True)


@pytest.mark.parametrize("pair", [(8, 96), (24, 80), (56, 48)])
def test_explicit_topology_uses_default_sizes_and_fixed_boundary(benchmark, pair):
    sizes = [1, 2, 4, *range(8, 257, 8)]
    for batch in (97, 98, 103, 104):
        assert benchmark.bucket_topology(batch, sizes, pair) == (104, pair)
        assert benchmark.fixed_lane_counts(batch, pair) == (pair[0], batch - pair[0])


@pytest.mark.parametrize("pair", [(49, 55), (8, 88), (0, 104), (104,), (8, 48, 48)])
def test_explicit_topology_rejects_special_or_wrong_capacity(benchmark, pair):
    with pytest.raises(ValueError, match="default sizes"):
        benchmark.bucket_topology(98, [1, 2, 4, *range(8, 257, 8)], pair)


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


@pytest.mark.parametrize(
    "pair,enabled,order",
    [((8, 96), True, (1, 0)), ((8, 96), False, (0, 1)), ((56, 48), True, (0, 1)), ((104,), True, (0,))],
)
def test_submission_priority_preserves_lane_identity(benchmark, pair, enabled, order):
    assert benchmark.task_submission_order(pair, enabled) == order


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
