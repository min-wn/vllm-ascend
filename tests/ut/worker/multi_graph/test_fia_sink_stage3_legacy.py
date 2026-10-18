# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for exact legacy updates and sequential scratch reuse."""

import importlib.util
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

OCCURRENCES = 16
FAKE_WORKSPACE_BYTES = 4096


@pytest.fixture
def legacy(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "aclgraph_fia_sink_stage3_legacy.py"
    name = "fia_sink_stage3_legacy_cpu"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FakeDevice:
    type: str = "npu"
    index: int = 0


class FakeTensor:
    def __init__(self, shape, dtype="bf16", device=None, *, contiguous=True, storage_owner=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = FakeDevice() if device is None else device
        self.ndim, self.contiguous = len(shape), contiguous
        self.storage_owner = self if storage_owner is None else storage_owner

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return FakeTensor(shape, self.dtype, self.device, storage_owner=self.storage_owner)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return {"bf16": 2, "u8": 1, "i32": 4, "bool": 1}[self.dtype]


def fake_layers(pages=8):
    return tuple(
        dict(
            query=FakeTensor((8, 12, 256)),
            key=FakeTensor((pages, 128, 2, 256)),
            value=FakeTensor((pages, 128, 2, 256)),
        )
        for _ in range(OCCURRENCES)
    )


def fake_inputs(legacy, layers=None):
    instance = legacy.LegacyGraph.__new__(legacy.LegacyGraph)
    instance.layers = fake_layers() if layers is None else layers
    instance.capacity, instance.maximum_kv_length = 2, 256
    instance.query_lengths, instance.kv_lengths = [4, 8], [127, 0]
    instance.block_table = FakeTensor((2, 2), dtype="i32")
    instance.atten_mask = FakeTensor((2048, 2048), dtype="bool")
    instance.torch = SimpleNamespace(bfloat16="bf16", bool="bool")
    return instance


def test_exact_q4_endpoints_and_true_zero_kv_are_preserved(legacy):
    query_lengths, kv_lengths = [4, 8], [128, 0]
    actual_q, actual_kv = legacy.validate_lengths(query_lengths, kv_lengths, 2, 256)
    assert actual_q == query_lengths and actual_kv == [128, 0]
    assert actual_q is not query_lengths and actual_kv is not kv_lengths


@pytest.mark.parametrize(
    "query_lengths,kv_lengths",
    [
        ([4, 7], [128, 0]),
        ([4, 8], [257, 0]),
        ([4, 8], [128, -1]),
        ([4, 8], [128, False]),
        ([4.0, 8], [128, 0]),
        ([4], [128]),
    ],
)
def test_invalid_length_descriptors_are_rejected(legacy, query_lengths, kv_lengths):
    with pytest.raises(ValueError):
        legacy.validate_lengths(query_lengths, kv_lengths, 2, 256)


class FakeEvent:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record(self, stream):
        self.log.append(("record", self.name, stream.name))


class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_event(self, event):
        self.log.append(("wait", self.name, event.name))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


@pytest.mark.parametrize("external_prepared", [False, True])
def test_every_replay_keeps_sixteen_exact_updates_and_final_join(legacy, external_prepared):
    log = []
    instance = fake_inputs(legacy)
    root, update, main = (FakeStream(name, log) for name in ("root", "update", "main"))

    class FakeGraph:
        def resolve_tasks(self, context):
            log.append(("resolve", tuple(context["kv_lengths"])))
            assert context["query_lengths"] == [4, 8]
            assert context["block_table"] is instance.block_table
            return tuple(legacy._LengthProvider(index).resolve(context) for index in range(OCCURRENCES))

        def replay(self):
            log.append(("replay",))

        def update(self, stream, tasks):
            assert stream is update and len(tasks) == OCCURRENCES
            for index, task in enumerate(tasks):
                assert task["actual_seq_lengths"] == [4, 8]
                assert task["actual_seq_lengths_kv"] == [127, 0]
                assert task["block_table"] is instance.block_table
                log.append(("apply_update", index))

    instance.torch.npu = SimpleNamespace(current_stream=lambda: main, stream=lambda _: nullcontext())
    instance.graph = FakeGraph()
    instance.context_source = lambda context: context
    instance.stream, instance.update_stream = root, update
    instance.prepared, instance.updated, instance.completion = (
        FakeEvent(name, log) for name in ("prepared", "updated", "completion")
    )
    instance.outputs = [object() for _ in range(OCCURRENCES)]
    event = FakeEvent("external", log) if external_prepared else None
    prepared_name = "external" if external_prepared else "prepared"

    assert instance.replay([4, 8], [127, 0], prepared=event) is instance.outputs
    expected = [("resolve", (127, 0))]
    if not external_prepared:
        expected.append(("record", "prepared", "main"))
    expected.extend([("wait", "root", prepared_name), ("wait", "update", prepared_name), ("replay",)])
    expected.extend(("apply_update", index) for index in range(OCCURRENCES))
    expected.extend([("record", "updated", "update"), ("wait", "root", "updated"), ("record", "completion", "root")])
    assert log == expected
    instance.synchronize()
    assert log[-1] == ("synchronize", "root")


def test_sequential_occurrences_share_one_workspace_but_keep_private_outputs(legacy):
    instance = fake_inputs(legacy)
    instance._validate_inputs()
    calls = []
    workspace = FakeTensor((FAKE_WORKSPACE_BYTES,), dtype="u8")

    def maximum_workspace(**kwargs):
        calls.append(kwargs)
        assert kwargs["actual_seq_lengths_kv"] == [256, 256]
        return workspace

    instance.torch.empty_like = lambda query: FakeTensor(query.shape)
    instance.torch.empty = lambda length, **kwargs: FakeTensor((length,), **kwargs)
    instance.torch_npu = SimpleNamespace(_npu_fused_infer_attention_score_get_max_workspace=maximum_workspace)
    instance.outputs, instance.softmax_lse, instance.workspaces, instance.kwargs = [], [], [], []
    instance._allocate_operations()

    assert len(calls) == 1
    assert instance.workspaces == [workspace]
    assert instance.workspace_bytes() == [FAKE_WORKSPACE_BYTES]
    assert len(instance.kwargs) == len(instance.outputs) == len(instance.softmax_lse) == OCCURRENCES
    assert len({id(output) for output in instance.outputs}) == OCCURRENCES
    assert len({id(output) for output in instance.softmax_lse}) == OCCURRENCES
    for layer, kwargs in zip(instance.layers, instance.kwargs, strict=True):
        assert kwargs["workspace"] is workspace
        assert kwargs["query"] is layer["query"]
        assert kwargs["key"].storage_owner is layer["key"]
        assert kwargs["value"].storage_owner is layer["value"]
        assert kwargs["actual_seq_lengths_kv"] == [127, 0]


def test_shared_workspace_rejects_inhomogeneous_page_geometry(legacy):
    layers = list(fake_layers())
    layers[-1] = {**layers[-1], "key": FakeTensor((9, 128, 2, 256)), "value": FakeTensor((9, 128, 2, 256))}
    instance = fake_inputs(legacy, layers)
    with pytest.raises(ValueError, match="same paged geometry"):
        instance._validate_inputs()
