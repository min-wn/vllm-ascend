import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload


@dataclass(frozen=True)
class MetadataProvider:
    layer_name: str

    def resolve(self, mapping):
        return mapping[self.layer_name]


@pytest.fixture
def benchmark(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("tp_attention", directory / "multi_graph_tp_attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("query", [1, 4])
@pytest.mark.parametrize("batch,variant", [(98, 0), (97, 1), (99, 2), (98, 3)])
def test_split_metadata_preserves_request_identity_and_physical_kv(benchmark, query, batch, variant):
    workload = ReplayWorkload(98, 100, query, 512, (68, 30))
    full = benchmark.request_values(workload, 0, 100, batch, variant)
    left = benchmark.request_values(workload, 0, 68, batch, variant)
    right = benchmark.request_values(workload, 68, 100, batch, variant)
    for field in ("lengths", "pages", "positions", "slots"):
        assert left[field] + right[field] == full[field]
    assert left["boundaries"] == [index * query for index in range(69)]
    assert right["boundaries"] == [index * query for index in range(33)]
    assert full["lengths"][batch:] == [0] * (100 - batch)
    assert full["slots"][batch * query :] == [-1] * ((100 - batch) * query)
    assert len(set(full["slots"][: batch * query])) == batch * query
    for request in range(batch):
        length = full["lengths"][request]
        for token in range(query):
            index = request * query + token
            position = length - query + token
            page = full["pages"][request][position // workload.block_size]
            assert full["positions"][index] == position
            assert full["slots"][index] == page * workload.block_size + position % workload.block_size


def test_changed_lengths_preserve_capacity_and_leave_prefix(benchmark):
    workload = ReplayWorkload(98, 100, 4, 512, (68, 30))
    initial = benchmark.request_values(workload, 68, 100, 98, 0)
    changed = benchmark.request_values(workload, 68, 100, 99, 1)
    assert initial["lengths"] != changed["lengths"]
    assert initial["positions"] != changed["positions"]
    assert all(length >= workload.query + 1 for length in changed["lengths"] if length)
    assert all(len(initial[field]) == len(changed[field]) for field in initial)
    assert initial["boundaries"] == changed["boundaries"]


@pytest.mark.parametrize("start,stop,batch", [(-1, 100, 98), (0, 101, 98), (68, 68, 98), (0, 100, 101), (0, 100, 0)])
def test_invalid_live_batch_or_slice_is_rejected(benchmark, start, stop, batch):
    workload = ReplayWorkload(98, 100, 4, 512, (68, 30))
    with pytest.raises(ValueError, match="captured capacity"):
        benchmark.request_values(workload, start, stop, batch, 0)


def test_short_prefix_is_rejected(benchmark):
    workload = ReplayWorkload(98, 100, 4, 64, (68, 30))
    with pytest.raises(ValueError, match="64 prefix tokens"):
        benchmark.request_values(workload, 0, 100, 98, 0)


def test_branch_source_filters_provider_and_preserves_capture_order(benchmark):
    attention = MetadataProvider("attention")
    other = MetadataProvider("other")
    full, left, right = object(), object(), object()
    input_data = [
        SimpleNamespace(mapping={"other": full}),
        SimpleNamespace(mapping={"attention": left}),
        SimpleNamespace(mapping={"attention": right}),
    ]
    graph = SimpleNamespace(
        tasks=[
            SimpleNamespace(provider=attention),
            SimpleNamespace(provider=other),
            SimpleNamespace(provider=MetadataProvider("attention")),
        ],
        task_lanes=[2, 0, 1],
    )
    source = benchmark.BranchSource(graph, input_data)
    assert source.get(attention) == (right, left)
    assert source.get(other) == (full,)
    assert source.get(MetadataProvider("missing")) == ()


def test_branch_source_binds_right_lane_local_query_lengths_and_current_kv(benchmark):
    workload = ReplayWorkload(98, 100, 4, 512, (68, 30))
    provider = MetadataProvider("attention")
    intervals = ((0, 100), (0, 68), (68, 100))
    input_data = [SimpleNamespace(mapping={}) for _ in intervals]
    graph = SimpleNamespace(tasks=[SimpleNamespace(provider=provider)] * 2, task_lanes=[1, 2])
    source = benchmark.BranchSource(graph, input_data)

    for batch, variant in ((98, 0), (99, 1)):
        for data, (start, stop) in zip(input_data, intervals, strict=True):
            values = benchmark.request_values(workload, start, stop, batch, variant)
            data.mapping["attention"] = {
                "actual_seq_lengths": values["boundaries"][1:],
                "actual_seq_lengths_kv": values["lengths"],
                "block_table": values["pages"],
            }
        left, right = source.get(provider)
        full = input_data[0].mapping["attention"]
        assert left is input_data[1].mapping["attention"]
        assert right is input_data[2].mapping["attention"]
        assert right["actual_seq_lengths"] == list(range(4, 129, 4))
        assert right["actual_seq_lengths_kv"] == full["actual_seq_lengths_kv"][68:]
        assert right["block_table"] == full["block_table"][68:]
        assert right["actual_seq_lengths_kv"][batch - 68 :] == [0] * (100 - batch)
        assert right["block_table"][0] == list(range(272, 276))
