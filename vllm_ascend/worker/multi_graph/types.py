from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class NoSplitReason(StrEnum):
    DISABLED = "disabled"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    UNSUPPORTED_BATCH = "unsupported_batch"
    TOO_SMALL = "too_small"
    GRAPH_MISS = "graph_miss"
    MISSING_MEASUREMENT = "missing_measurement"
    NO_GAIN = "no_gain"


@dataclass(frozen=True)
class CaptureSignature:
    """Identity shared by entries captured for one model/device generation."""

    buffer_generation: int
    model_config_hash: str
    device_identity: str
    dtype: str
    attention_backend: str
    attention_variant: str
    kv_layout: str
    block_size: int
    graph_mode: str = "FULL"
    attention_state: str = "DecodeOnly"
    query_len: int = 1
    runtime_identity: str = ""

    def __post_init__(self) -> None:
        if self.buffer_generation < 0 or self.block_size <= 0:
            raise ValueError("invalid capture generation or block size")
        if not all((self.model_config_hash, self.device_identity, self.kv_layout)):
            raise ValueError("capture signature requires model, device and KV layout identities")


@dataclass(frozen=True)
class GraphKey:
    lane_id: int
    graph_tokens: int
    graph_reqs: int
    signature: CaptureSignature

    def __post_init__(self) -> None:
        if self.lane_id not in (0, 1) or self.graph_tokens <= 0 or self.graph_reqs != self.graph_tokens:
            raise ValueError("exact decode graph keys require lane 0/1 and equal positive request/token counts")


@dataclass(frozen=True)
class SlicePlan:
    request_start: int
    request_stop: int
    token_start: int
    token_stop: int
    graph_key: GraphKey

    def __post_init__(self) -> None:
        if self.request_start < 0 or self.request_stop <= self.request_start:
            raise ValueError("slices must be nonempty")
        if (self.request_start, self.request_stop) != (self.token_start, self.token_stop):
            raise ValueError("uniform decode requires one token per request")
        if self.actual_tokens != self.graph_key.graph_tokens:
            raise ValueError("exact decode slices must match captured capacity")

    @property
    def actual_tokens(self) -> int:
        return self.token_stop - self.token_start

    @property
    def actual_reqs(self) -> int:
        return self.request_stop - self.request_start

    @property
    def lane_id(self) -> int:
        return self.graph_key.lane_id


@dataclass(frozen=True)
class ExecutionPlan:
    slices: tuple[SlicePlan, SlicePlan]
    execution: Literal["serial", "parallel"]
    cost_source: Literal["force", "measured"]

    def __post_init__(self) -> None:
        first, second = self.slices
        if first.request_start != 0 or first.request_stop != second.request_start:
            raise ValueError("execution plan must cover contiguous requests in their original order")
        if (first.lane_id, second.lane_id) != (0, 1):
            raise ValueError("execution plan must use two distinct lanes in order")
        if first.graph_key.signature != second.graph_key.signature:
            raise ValueError("execution plan cannot mix capture signatures")

    @property
    def num_requests(self) -> int:
        return self.slices[-1].request_stop

    @property
    def total_padding(self) -> int:
        return 0


@dataclass(frozen=True)
class NoSplit:
    reason: NoSplitReason
    detail: str = ""


@dataclass(frozen=True)
class BatchSummary:
    """CPU-side request summary; never extract this from device tensors."""

    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    full_decode: bool

    def __post_init__(self) -> None:
        if len(self.query_lengths) != len(self.kv_lengths):
            raise ValueError("query and KV length summaries must describe the same requests")
        if any(length <= 0 for length in self.query_lengths):
            raise ValueError("query lengths must be positive")
        if any(kv_len < query_len for kv_len, query_len in zip(self.kv_lengths, self.query_lengths)):
            raise ValueError("KV lengths must include the current query")
