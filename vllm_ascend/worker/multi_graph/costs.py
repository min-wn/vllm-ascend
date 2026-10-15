from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vllm_ascend.worker.multi_graph.planner import CostKey, MeasuredCost
from vllm_ascend.worker.multi_graph.types import CaptureSignature, GraphKey

PositiveCount = Annotated[int, Field(strict=True, gt=0)]
PositiveTime = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
MAX_COST_FILE_BYTES = 16 << 20


class DecodeCostRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_sizes: tuple[PositiveCount, PositiveCount]
    execution: Literal["serial", "parallel"]
    kv_lengths: tuple[PositiveCount, ...]
    single_graph_us: tuple[PositiveTime, ...]
    split_us: tuple[PositiveTime, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> DecodeCostRecord:
        if len(self.kv_lengths) != sum(self.lane_sizes):
            raise ValueError("KV lengths must describe every request in both lanes")
        if not self.single_graph_us or len(self.single_graph_us) != len(self.split_us):
            raise ValueError("paired baseline/split measurements must have equal nonzero sample counts")
        return self


class DecodeCostDocument(BaseModel):
    """Synchronized model-forward wall time, including preparation and output merge.

    Exact ordered KV vectors prevent unmeasured interpolation. Generation time,
    kernel-only time and profiler runs are deliberately not accepted metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    metric: Literal["synchronized_decode_forward_us", "synchronized_decode_dispatch_us"]
    signature: CaptureSignature
    records: tuple[DecodeCostRecord, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> DecodeCostDocument:
        if not self.signature.runtime_identity:
            raise ValueError("persistent costs require a software/device runtime identity")
        identities = [(record.lane_sizes, record.execution, record.kv_lengths) for record in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate cost record")
        return self

    def cost_table(self, signature: CaptureSignature, minimum_samples: int) -> dict[CostKey, MeasuredCost]:
        if self.signature != signature:
            raise ValueError("cost table capture signature does not match this worker")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        result = {}
        for record in self.records:
            if len(record.split_us) < minimum_samples:
                continue
            keys = tuple(GraphKey(lane, size, size, signature) for lane, size in enumerate(record.lane_sizes))
            result[CostKey(keys, record.execution, record.kv_lengths)] = MeasuredCost(
                statistics.median(record.single_graph_us), statistics.median(record.split_us)
            )
        return result


def load_cost_table(path: str, signature: CaptureSignature, minimum_samples: int) -> dict[CostKey, MeasuredCost]:
    with Path(path).open("rb") as source:
        payload = source.read(MAX_COST_FILE_BYTES + 1)
    if len(payload) > MAX_COST_FILE_BYTES:
        raise ValueError("cost table exceeds the file size limit")
    document = DecodeCostDocument.model_validate_json(payload)
    return document.cost_table(signature, minimum_samples)


class DecodeForwardSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kv_lengths: tuple[PositiveCount, ...]
    elapsed_us: PositiveTime
    lane_sizes: tuple[PositiveCount, PositiveCount] | None = None


def build_cost_document(
    signature: CaptureSignature,
    execution: Literal["serial", "parallel"],
    baseline: tuple[DecodeForwardSample, ...],
    split: tuple[DecodeForwardSample, ...],
) -> DecodeCostDocument:
    if len(baseline) != len(split) or not baseline:
        raise ValueError("decode traces must contain equal nonzero sample counts")
    groups = defaultdict(lambda: ([], []))
    for single, pair in zip(baseline, split):
        if single.lane_sizes is not None or pair.lane_sizes is None or single.kv_lengths != pair.kv_lengths:
            raise ValueError("decode traces must pair the same ordered KV lengths and an executed split")
        single_times, split_times = groups[(pair.lane_sizes, pair.kv_lengths)]
        single_times.append(single.elapsed_us)
        split_times.append(pair.elapsed_us)
    records = tuple(
        DecodeCostRecord(
            lane_sizes=sizes,
            execution=execution,
            kv_lengths=lengths,
            single_graph_us=single_times,
            split_us=split_times,
        )
        for (sizes, lengths), (single_times, split_times) in sorted(groups.items())
    )
    return DecodeCostDocument(metric="synchronized_decode_forward_us", signature=signature, records=records)
