# SPDX-License-Identifier: Apache-2.0
"""Persistent pinned descriptors for the isolated fixed-bucket Q4 experiment.

The caller supplies Torch and existing independent device tensors. No device
storage is allocated or repacked here. ``complete()`` is an explicit assertion
by the caller that its final device synchronization has completed; it does not
add another synchronization to the measured path.
"""

import numpy as np

QUERY_LENGTH = 4
DESCRIPTOR_NAMES = ("q_lengths", "kv_lengths", "block_table")


def _storage_bytes(tensor):
    return tensor.untyped_storage().nbytes()


def _describe(tensor):
    return dict(
        pointer=tensor.data_ptr(),
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        storage_bytes=_storage_bytes(tensor),
    )


class PreparedDescriptors:
    """Own one bucket's pinned sources until the consumer has completed.

    Construct once per bucket and mode. Sink construction uploads invariant
    cumulative Q once, so synchronize and call ``complete()`` before first use.
    Legacy construction only allocates its table source and needs no upload.
    Each ``prepare(record)`` fills current metadata inside the measured endpoint
    and returns fresh real Q/KV host lists for the existing update interface.
    After the caller's final join, ``complete()`` releases the source buffers for
    the next fill. A second preparation while sources may be in use is rejected.

    Records come from the independently checked stage-three trace helper. This
    class checks capacity and Q4 invariance each step; it does not repeat owner
    lifetime or per-page semantic validation in the measured preparation path.
    """

    def __init__(self, torch, tensors, record, *, mode="sink"):
        if mode not in ("sink", "legacy"):
            raise ValueError("descriptor mode must be sink or legacy")
        capacity = record["physical_tokens"]
        if type(capacity) is not int or capacity < QUERY_LENGTH or capacity % QUERY_LENGTH:
            raise ValueError("a positive physical capacity divisible by four is required")
        self.torch, self.mode, self.capacity = torch, mode, capacity
        self.batch = capacity // QUERY_LENGTH
        self._cumulative_q = tuple(range(QUERY_LENGTH, capacity + 1, QUERY_LENGTH))
        self._device = {name: tensors[name] for name in DESCRIPTOR_NAMES}
        self._validate_device()
        self._validate_record(record)
        self._inflight = False
        self._prepared_steps = 0
        self._completed_steps = 0
        self._setup_copies = dict.fromkeys(DESCRIPTOR_NAMES, 0)
        self._step_copies = dict.fromkeys(DESCRIPTOR_NAMES, 0)
        host_names = DESCRIPTOR_NAMES if mode == "sink" else ("block_table",)
        self._host = {
            name: torch.empty(
                tuple(self._device[name].shape),
                dtype=self._device[name].dtype,
                device="cpu",
                pin_memory=True,
            )
            for name in host_names
        }
        self._views = {name: tensor.numpy() for name, tensor in self._host.items()}
        if any(not tensor.is_pinned() for tensor in self._host.values()):
            raise RuntimeError("descriptor sources must be page-locked CPU allocations")
        self._initial_host = {name: _describe(tensor) for name, tensor in self._host.items()}
        self._initial_device = {name: _describe(tensor) for name, tensor in self._device.items()}
        if mode == "sink":
            self._views["q_lengths"][:] = self._cumulative_q
            self._inflight = True
            self._device["q_lengths"].copy_(self._host["q_lengths"], non_blocking=True)
            self._setup_copies["q_lengths"] += 1

    def _validate_device(self):
        for name in DESCRIPTOR_NAMES:
            tensor = self._device[name]
            expected_dtype = self.torch.int32 if name == "block_table" else self.torch.int64
            if name == "block_table" and (len(tensor.shape) != 2 or tensor.shape[1] < 1):
                raise ValueError("block_table must be a nonempty two-dimensional bucket table")
            expected_shape = (self.batch,) if name != "block_table" else (self.batch, tensor.shape[1])
            if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype or not tensor.is_contiguous():
                raise ValueError(f"{name} must retain its exact independent bucket shape and dtype")
            if tensor.storage_offset() or _storage_bytes(tensor) != tensor.numel() * tensor.element_size():
                raise ValueError(f"{name} must own its exact storage extent; a larger backing/slab is unsupported")

    def _validate_record(self, record):
        if type(record["physical_tokens"]) is not int or record["physical_tokens"] != self.capacity:
            raise ValueError("a prepared descriptor cannot be reused for a different default bucket")
        if tuple(record["q_lengths"]) != (QUERY_LENGTH,) * self.batch:
            raise ValueError("every physical request, including padding, must retain Q4")
        if tuple(record["actual_seq_qlen"]) != self._cumulative_q:
            raise ValueError("current native cumulative Q differs from the bucket invariant")
        if len(record["kv_lengths"]) != self.batch or len(record["block_table"]) != self.batch:
            raise ValueError("current KV lengths and page table must cover the physical request capacity")

    @property
    def inflight(self):
        return self._inflight

    def prepare(self, record):
        if self._inflight:
            raise RuntimeError("descriptor sources remain in use; synchronize the consumer and call complete()")
        self._validate_record(record)
        q, kv = list(record["actual_seq_qlen"]), list(record["kv_lengths"])
        table = np.asarray(record["block_table"], dtype=np.int32)
        if table.shape != self._views["block_table"].shape:
            raise ValueError("current page table must match the exact fixed bucket shape without broadcasting")
        # Fill only this step's real routing/lengths. NumPy views alias the
        # persistent pinned allocations and create no temporary Torch tensors.
        if self.mode == "sink":
            self._views["kv_lengths"][:] = kv
        self._views["block_table"][:] = table
        self._inflight = True
        if self.mode == "sink":
            self._device["kv_lengths"].copy_(self._host["kv_lengths"], non_blocking=True)
            self._step_copies["kv_lengths"] += 1
        self._device["block_table"].copy_(self._host["block_table"], non_blocking=True)
        self._step_copies["block_table"] += 1
        self._prepared_steps += 1
        return q, kv

    def complete(self):
        """Caller asserts that all preceding copies and consumers have joined."""
        if self._inflight:
            self._completed_steps = self._prepared_steps
        self._inflight = False

    def summary(self):
        """Return JSON-safe provenance outside the measured preparation path."""
        host = {name: _describe(tensor) for name, tensor in self._host.items()}
        device = {name: _describe(tensor) for name, tensor in self._device.items()}
        return dict(
            mode=self.mode,
            physical_tokens=self.capacity,
            physical_requests=self.batch,
            host_pinned={name: tensor.is_pinned() for name, tensor in self._host.items()},
            host=host,
            device=device,
            stable_host_allocations=host == self._initial_host,
            stable_device_allocations=device == self._initial_device,
            host_allocation_count=len(self._host),
            device_allocation_count=0,
            setup_copies=dict(self._setup_copies),
            step_copies=dict(self._step_copies),
            prepared_steps=self._prepared_steps,
            completed_steps=self._completed_steps,
            inflight=self._inflight,
            q4_invariant=dict(query_length=QUERY_LENGTH, cumulative_q=list(self._cumulative_q)),
            completion_contract="Caller performs the final device join before complete(); no extra join is inserted.",
        )
