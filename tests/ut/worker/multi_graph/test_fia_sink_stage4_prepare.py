# SPDX-License-Identifier: Apache-2.0
"""CPU ownership and descriptor-copy contracts without importing native Torch."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"


class FakeStorage:
    def __init__(self, size):
        self.size = size

    def nbytes(self):
        return self.size


class FakeTensor:
    def __init__(self, shape, dtype, *, device="npu:0", pinned=False):
        self.array = np.zeros(shape, dtype=dtype)
        self.shape, self.dtype, self.device = self.array.shape, self.array.dtype, device
        self.pinned, self.contiguous, self.offset = pinned, True, 0
        self.storage_bytes = self.array.nbytes
        self.copies = []
        self.fail_copy = False

    def data_ptr(self):
        return self.array.ctypes.data

    def stride(self):
        return tuple(stride // self.array.itemsize for stride in self.array.strides)

    def untyped_storage(self):
        return FakeStorage(self.storage_bytes)

    def storage_offset(self):
        return self.offset

    def numel(self):
        return self.array.size

    def element_size(self):
        return self.array.itemsize

    def is_contiguous(self):
        return self.contiguous

    def is_pinned(self):
        return self.pinned

    def numpy(self):
        return self.array

    def copy_(self, source, *, non_blocking=False):
        if self.fail_copy:
            raise RuntimeError("injected copy failure")
        self.copies.append(dict(source=source, non_blocking=non_blocking))
        np.copyto(self.array, source.array)
        return self


class FakeTorch:
    int32 = np.dtype("int32")
    int64 = np.dtype("int64")

    def __init__(self):
        self.allocations = []

    def empty(self, shape, *, dtype, device, pin_memory):
        tensor = FakeTensor(shape, dtype, device=device, pinned=pin_memory)
        self.allocations.append(tensor)
        return tensor


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("aclgraph_fia_sink_stage4_prepare")


@pytest.fixture
def inputs():
    torch = FakeTorch()
    tensors = dict(
        q_lengths=FakeTensor((3,), torch.int64),
        kv_lengths=FakeTensor((3,), torch.int64),
        block_table=FakeTensor((3, 2), torch.int32),
    )
    record = dict(
        physical_tokens=12,
        q_lengths=(4, 4, 4),
        actual_seq_qlen=(4, 8, 12),
        kv_lengths=(129, 0, 0),
        block_table=((1, 2), (3, 0), (0, 0)),
    )
    return torch, tensors, record


def test_sink_reuses_three_pinned_sources_and_existing_independent_device_storage(module, inputs):
    torch, tensors, record = inputs
    pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    prepared = module.PreparedDescriptors(torch, tensors, record)
    assert prepared.inflight
    assert len(torch.allocations) == 3
    assert all(tensor.is_pinned() for tensor in torch.allocations)
    assert tensors["q_lengths"].array.tolist() == [4, 8, 12]
    assert len(tensors["q_lengths"].copies) == 1
    assert all(tensor.device == "cpu" for tensor in torch.allocations)
    prepared.complete()
    original = prepared.summary()
    q, kv = prepared.prepare(record)
    assert (q, kv) == ([4, 8, 12], [129, 0, 0])
    assert q is not record["actual_seq_qlen"] and kv is not record["kv_lengths"]
    assert tensors["kv_lengths"].array.tolist() == [129, 0, 0]
    assert tensors["block_table"].array.tolist() == [[1, 2], [3, 0], [0, 0]]
    assert prepared.inflight
    prepared.complete()
    changed = copy.deepcopy(record)
    changed.update(kv_lengths=(0, 256, 0), block_table=((0, 0), (4, 5), (0, 0)))
    prepared.prepare(changed)
    prepared.complete()
    assert len(torch.allocations) == 3
    assert tensors["kv_lengths"].array.tolist() == [0, 256, 0]
    assert tensors["block_table"].array.tolist() == [[0, 0], [4, 5], [0, 0]]
    assert {name: tensor.data_ptr() for name, tensor in tensors.items()} == pointers
    summary = prepared.summary()
    assert summary["host"] == original["host"]
    assert summary["device"] == original["device"]
    assert summary["stable_host_allocations"] and summary["stable_device_allocations"]
    assert summary["host_allocation_count"] == 3 and summary["device_allocation_count"] == 0
    assert summary["setup_copies"] == dict(q_lengths=1, kv_lengths=0, block_table=0)
    assert summary["step_copies"] == dict(q_lengths=0, kv_lengths=2, block_table=2)
    assert summary["prepared_steps"] == summary["completed_steps"] == 2
    assert all(copy["non_blocking"] for tensor in tensors.values() for copy in tensor.copies)
    json.dumps(summary, allow_nan=False)


def test_legacy_allocates_only_table_source_and_returns_current_real_host_lengths(module, inputs):
    torch, tensors, record = inputs
    prepared = module.PreparedDescriptors(torch, tensors, record, mode="legacy")
    assert not prepared.inflight
    assert len(torch.allocations) == 1
    assert prepared.summary()["setup_copies"] == dict.fromkeys(tensors, 0)
    q, kv = prepared.prepare(record)
    assert (q, kv) == ([4, 8, 12], [129, 0, 0])
    assert not tensors["q_lengths"].copies and not tensors["kv_lengths"].copies
    assert len(tensors["block_table"].copies) == 1
    prepared.complete()
    changed = dict(record, kv_lengths=(127, 128, 0))
    assert prepared.prepare(changed)[1] == [127, 128, 0]
    prepared.complete()
    summary = prepared.summary()
    assert summary["step_copies"] == dict(q_lengths=0, kv_lengths=0, block_table=2)
    assert summary["host_pinned"] == dict(block_table=True)
    assert summary["host_allocation_count"] == 1


def test_inflight_guard_prevents_source_overwrite_until_explicit_completion(module, inputs):
    torch, tensors, record = inputs
    prepared = module.PreparedDescriptors(torch, tensors, record)
    with pytest.raises(RuntimeError, match="remain in use"):
        prepared.prepare(record)
    prepared.complete()
    prepared.prepare(record)
    source = tensors["kv_lengths"].copies[-1]["source"]
    original = source.array.copy()
    with pytest.raises(RuntimeError, match="remain in use"):
        prepared.prepare(dict(record, kv_lengths=(0, 0, 0)))
    assert np.array_equal(source.array, original)
    prepared.complete()
    prepared.prepare(dict(record, kv_lengths=(0, 0, 0)))
    assert not np.any(source.array)
    prepared.complete()
    prepared.complete()
    assert not prepared.inflight


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_tokens", 16),
        ("q_lengths", (4, 4, 0)),
        ("actual_seq_qlen", (4, 8, 8)),
        ("kv_lengths", (129, 0)),
        ("block_table", ((1, 2), (0, 0))),
    ],
)
def test_changed_bucket_or_q4_invariant_is_rejected_before_any_upload(module, inputs, field, value):
    torch, tensors, record = inputs
    prepared = module.PreparedDescriptors(torch, tensors, record)
    prepared.complete()
    changed = dict(record, **{field: value})
    with pytest.raises(ValueError):
        prepared.prepare(changed)
    assert not prepared.inflight
    assert not tensors["kv_lengths"].copies and not tensors["block_table"].copies
    assert prepared.summary()["prepared_steps"] == 0
    prepared.prepare(record)
    prepared.complete()


@pytest.mark.parametrize("width", [1, 3])
def test_wrong_table_width_cannot_broadcast_into_fixed_bucket_without_upload(module, inputs, width):
    torch, tensors, record = inputs
    prepared = module.PreparedDescriptors(torch, tensors, record)
    prepared.complete()
    with pytest.raises(ValueError):
        prepared.prepare(dict(record, block_table=((1,) * width, (0,) * width, (0,) * width)))
    assert not prepared.inflight
    assert not tensors["kv_lengths"].copies and not tensors["block_table"].copies


@pytest.mark.parametrize("mutation", ["shape", "dtype", "noncontiguous", "backing", "offset"])
def test_device_descriptors_must_have_exact_independent_storage(module, inputs, mutation):
    torch, tensors, record = inputs
    if mutation == "shape":
        tensors["kv_lengths"] = FakeTensor((4,), torch.int64)
    elif mutation == "dtype":
        tensors["kv_lengths"] = FakeTensor((3,), torch.int32)
    elif mutation == "noncontiguous":
        tensors["block_table"].contiguous = False
    elif mutation == "backing":
        tensors["q_lengths"].storage_bytes *= 2
    else:
        tensors["q_lengths"].offset = 1
    with pytest.raises(ValueError):
        module.PreparedDescriptors(torch, tensors, record)
    assert not torch.allocations
    assert all(not tensor.copies for tensor in tensors.values())


def test_failed_device_copy_keeps_sources_owned_until_caller_joins(module, inputs):
    torch, tensors, record = inputs
    prepared = module.PreparedDescriptors(torch, tensors, record)
    prepared.complete()
    tensors["block_table"].fail_copy = True
    with pytest.raises(RuntimeError, match="injected"):
        prepared.prepare(record)
    assert prepared.inflight
    assert len(tensors["kv_lengths"].copies) == 1
    with pytest.raises(RuntimeError, match="remain in use"):
        prepared.prepare(record)
    prepared.complete()
    tensors["block_table"].fail_copy = False
    prepared.prepare(record)
    prepared.complete()


def test_unrecognized_mode_is_rejected_without_allocating(module, inputs):
    torch, tensors, record = inputs
    with pytest.raises(ValueError, match="mode"):
        module.PreparedDescriptors(torch, tensors, record, mode="unknown")
    assert not torch.allocations
