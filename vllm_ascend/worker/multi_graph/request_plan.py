from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

import torch

from vllm_ascend.worker.multi_graph.types import GraphKey


@dataclass(frozen=True)
class RequestBatch:
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    is_prefill: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not (len(self.query_lengths) == len(self.kv_lengths) == len(self.is_prefill)):
            raise ValueError("request fields must have equal lengths")
        if any(
            type(query) is not int or query <= 0 or type(kv) is not int or kv < query
            for query, kv in zip(self.query_lengths, self.kv_lengths)
        ):
            raise ValueError("request lengths must be positive and include query tokens")
        if any(type(prefill) is not bool for prefill in self.is_prefill):
            raise ValueError("prefill flags must be boolean")

    @property
    def boundaries(self) -> tuple[int, ...]:
        result = [0]
        for length in self.query_lengths:
            result.append(result[-1] + length)
        return tuple(result)


@dataclass(frozen=True)
class RequestSlice:
    start: int
    stop: int
    token_start: int
    token_stop: int
    workload: str


def mixed_request_slices(batch: RequestBatch) -> tuple[RequestSlice, ...]:
    """Split contiguous homogeneous runs without reordering or splitting requests."""
    if not batch.query_lengths:
        return ()
    kinds = tuple(
        "prefill" if prefill else "decode" if query == 1 else "multi_token_decode"
        for prefill, query in zip(batch.is_prefill, batch.query_lengths)
    )
    boundaries = batch.boundaries
    result = []
    start = 0
    for stop in range(1, len(kinds) + 1):
        if stop == len(kinds) or kinds[stop] != kinds[start]:
            result.append(RequestSlice(start, stop, boundaries[start], boundaries[stop], kinds[start]))
            start = stop
    return tuple(result)


def slice_query_boundaries(batch: RequestBatch, part: RequestSlice) -> tuple[int, ...]:
    expected = batch.boundaries
    if not 0 <= part.start < part.stop <= len(batch.query_lengths) or (part.token_start, part.token_stop) != (
        expected[part.start],
        expected[part.stop],
    ):
        raise ValueError("slice must follow request/token boundaries")
    return tuple(boundary - part.token_start for boundary in expected[part.start : part.stop + 1])


@dataclass(frozen=True)
class CapacitySlice:
    request_start: int
    request_stop: int
    graph_key: GraphKey

    @property
    def count(self) -> int:
        return self.request_stop - self.request_start


def plan_capacity_decode(count: int, entries: Set[GraphKey], max_padding: int) -> tuple[CapacitySlice, ...] | None:
    if type(count) is not int or type(max_padding) is not int or count < 2 or max_padding < 0:
        return None
    candidates = []
    for first in entries:
        for second in entries:
            if first.lane_id != 0 or second.lane_id != 1 or first.signature != second.signature:
                continue
            padding = first.graph_tokens + second.graph_tokens - count
            if not 0 <= padding <= max_padding:
                continue
            for boundary in range(max(1, count - second.graph_tokens), min(count - 1, first.graph_tokens) + 1):
                candidates.append((padding, abs(2 * boundary - count), -boundary, first, second))
    if not candidates:
        return None
    _, _, negative_boundary, first, second = min(
        candidates, key=lambda value: (*value[:3], value[3].graph_tokens, value[4].graph_tokens)
    )
    boundary = -negative_boundary
    return (CapacitySlice(0, boundary, first), CapacitySlice(boundary, count, second))


@dataclass
class PaddedDecodeInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    kv_lengths: tuple[int, ...]
    valid_rows: torch.Tensor
    actual_count: int


def prepare_padded_decode(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_lengths: tuple[int, ...],
    part: CapacitySlice,
    scratch_block_ids: tuple[int, ...],
    reserved_blocks: frozenset[int],
    live_blocks: frozenset[int],
) -> PaddedDecodeInputs:
    """Use allocator-reserved scratch blocks; never assume block zero is scratch.

    The allocator supplies all live physical blocks, including prefix reads.
    Scratch blocks require zero-initialized KV tensors before device capture.
    This prepares tensors only; it does not assert FIA padding compatibility.
    """
    start, stop = part.request_start, part.request_stop
    capacity = part.graph_key.graph_tokens
    if not 0 <= start < stop <= len(kv_lengths) or part.count > capacity:
        raise ValueError("invalid capacity slice")
    if input_ids.ndim != 1 or positions.ndim != 1 or block_tables.ndim != 2 or slot_mapping.ndim != 1:
        raise ValueError("padding requires one-dimensional decode inputs and a two-dimensional block table")
    if any(tensor.shape[0] < stop for tensor in (input_ids, positions, block_tables, slot_mapping)):
        raise ValueError("input tensors do not cover the requested slice")
    if block_tables.shape[1] == 0 or any(type(length) is not int or length <= 0 for length in kv_lengths):
        raise ValueError("invalid KV layout")
    dummy_count = capacity - part.count
    if len(scratch_block_ids) != dummy_count or len(set(scratch_block_ids)) != dummy_count:
        raise ValueError("each dummy row requires a distinct scratch block")
    if (
        any(type(block) is not int or block < 0 for block in scratch_block_ids)
        or not set(scratch_block_ids).issubset(reserved_blocks)
        or set(scratch_block_ids) & live_blocks
    ):
        raise ValueError("scratch blocks must be allocator-reserved and disjoint from all live KV")
    ids = input_ids.new_zeros(capacity)
    pos = positions.new_zeros(capacity)
    blocks = block_tables.new_zeros((capacity, block_tables.shape[1]))
    slots = slot_mapping.new_full((capacity,), -1)
    ids[: part.count].copy_(input_ids[start:stop])
    pos[: part.count].copy_(positions[start:stop])
    blocks[: part.count].copy_(block_tables[start:stop])
    slots[: part.count].copy_(slot_mapping[start:stop])
    if dummy_count:
        blocks[part.count :, 0] = block_tables.new_tensor(scratch_block_ids)
    valid = torch.arange(capacity, device=input_ids.device) < part.count
    return PaddedDecodeInputs(ids, pos, blocks, slots, kv_lengths[start:stop] + (1,) * dummy_count, valid, part.count)


def merge_capacity_outputs(parts: tuple[CapacitySlice, ...], outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(parts) != len(outputs) or not parts:
        raise ValueError("one output is required per capacity slice")
    boundary = 0
    for part, output in zip(parts, outputs):
        if part.request_start != boundary or not 0 < part.count <= part.graph_key.graph_tokens:
            raise ValueError("capacity slices must cover requests contiguously")
        if output.ndim != 2 or output.shape[0] != part.graph_key.graph_tokens:
            raise ValueError("captured output does not match capacity")
        boundary = part.request_stop
    return torch.cat([output[: part.count] for part, output in zip(parts, outputs)], dim=0)
