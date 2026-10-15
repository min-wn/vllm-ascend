from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from vllm_ascend.worker.multi_graph.metadata import clone_metadata, refresh_metadata, validate_metadata_refresh
from vllm_ascend.worker.multi_graph.types import SlicePlan


@dataclass
class LaneBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    metadata: dict[str, Any]

    @classmethod
    def create(cls, input_ids: torch.Tensor, positions: torch.Tensor, metadata: dict[str, Any], part: SlicePlan):
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("multi-graph requires one-dimensional token IDs and positions")
        cache: dict[int, Any] = {}
        cloned: dict[str, Any] = {}
        for name, source in metadata.items():
            if id(source) not in cache:
                cache[id(source)] = clone_metadata(source, part)
            cloned[name] = cache[id(source)]
        return cls(
            input_ids[part.token_start : part.token_stop].clone(),
            positions[part.token_start : part.token_stop].clone(),
            cloned,
        )

    def preflight(self, input_ids: torch.Tensor, positions: torch.Tensor, metadata: dict[str, Any], part: SlicePlan):
        if self.metadata.keys() != metadata.keys():
            raise ValueError("attention layer set changed")
        for captured, source in ((self.input_ids, input_ids), (self.positions, positions)):
            if source.ndim != 1 or source.shape[0] < part.token_stop:
                raise ValueError("token inputs do not cover the slice")
            if captured.shape != (part.actual_tokens,) or (captured.dtype, captured.device) != (
                source.dtype,
                source.device,
            ):
                raise ValueError("captured token input layout changed")
        updates = {}
        for name, captured in self.metadata.items():
            updates[name] = validate_metadata_refresh(captured, metadata[name], part)
        return updates

    def refresh(self, input_ids: torch.Tensor, positions: torch.Tensor, part: SlicePlan, updates) -> None:
        self.input_ids.copy_(input_ids[part.token_start : part.token_stop], non_blocking=True)
        self.positions.copy_(positions[part.token_start : part.token_stop], non_blocking=True)
        for name, values in updates.items():
            refresh_metadata(self.metadata[name], values)

    def writable_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [self.input_ids, self.positions]
        for metadata in self.metadata.values():
            tensors.extend(
                value for field in fields(metadata) if isinstance(value := getattr(metadata, field.name), torch.Tensor)
            )
        return tuple(tensors)


def storage_regions(tensors: tuple[torch.Tensor, ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        for tensor in tensors
        if tensor.numel()
    )


def assert_disjoint(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> None:
    for first_device, first_address, first_bytes in storage_regions(first):
        for second_device, second_address, second_bytes in storage_regions(second):
            if first_device == second_device and max(first_address, second_address) < min(
                first_address + first_bytes, second_address + second_bytes
            ):
                raise ValueError("multi-graph entries share writable storage")
