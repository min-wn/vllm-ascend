from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass

from vllm_ascend.worker.multi_graph.config import MultiGraphConfig
from vllm_ascend.worker.multi_graph.support import RuntimeProfile, rejection_reasons
from vllm_ascend.worker.multi_graph.types import (
    BatchSummary,
    CaptureSignature,
    ExecutionPlan,
    GraphKey,
    NoSplit,
    NoSplitReason,
    SlicePlan,
)


@dataclass(frozen=True)
class CostKey:
    """Measurements are valid only for the same batch, signature and mode."""

    graph_keys: tuple[GraphKey, GraphKey]
    execution: str
    kv_lengths: tuple[int, ...]


@dataclass(frozen=True)
class MeasuredCost:
    """End-to-end split time includes prepare, updates and output collection."""

    single_graph_us: float
    split_us: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(cost) or cost <= 0 for cost in (self.single_graph_us, self.split_us)):
            raise ValueError("measured times must be finite and positive")


class ExactDecodePlanner:
    """Select two already-captured entries, preserving request and token order.

    Force mode is for correctness/benchmark runs. Measured mode requires an
    explicit end-to-end measurement and returns NoSplit when it is unavailable.
    This class neither captures graphs nor touches device tensors.
    """

    def __init__(self, config: MultiGraphConfig):
        self.config = config

    def plan(
        self,
        batch: BatchSummary,
        profile: RuntimeProfile,
        signature: CaptureSignature,
        available_entries: Set[GraphKey],
        cost_table: Mapping[CostKey, MeasuredCost] | None = None,
    ) -> ExecutionPlan | NoSplit:
        if not self.config.enabled:
            return NoSplit(NoSplitReason.DISABLED)
        reasons = rejection_reasons(profile, signature)
        if reasons:
            return NoSplit(NoSplitReason.UNSUPPORTED_RUNTIME, "; ".join(reasons))
        if not batch.full_decode or any(length != 1 for length in batch.query_lengths):
            return NoSplit(NoSplitReason.UNSUPPORTED_BATCH, "requires FULL decode with one token per request")
        num_requests = len(batch.query_lengths)
        if num_requests < self.config.num_lanes:
            return NoSplit(NoSplitReason.TOO_SMALL)

        candidates: list[tuple[GraphKey, GraphKey]] = []
        left_sizes, right_sizes = self.config.capture_sizes_per_lane
        right_sizes_set = frozenset(right_sizes)
        for left_size in left_sizes:
            right_size = num_requests - left_size
            if right_size not in right_sizes_set:
                continue
            keys = (GraphKey(0, left_size, left_size, signature), GraphKey(1, right_size, right_size, signature))
            if all(key in available_entries for key in keys):
                candidates.append(keys)
        if not candidates:
            return NoSplit(NoSplitReason.GRAPH_MISS, "no exact pair of ready entries for the current signature")

        candidates.sort(key=lambda keys: (abs(keys[0].graph_tokens - keys[1].graph_tokens), -keys[0].graph_tokens))
        if self.config.selection_policy == "measured":
            measured = [
                (keys, cost)
                for keys in candidates
                if cost_table is not None
                and (cost := cost_table.get(CostKey(keys, self.config.execution, batch.kv_lengths))) is not None
            ]
            if not measured:
                return NoSplit(NoSplitReason.MISSING_MEASUREMENT)
            beneficial = [
                (keys, cost)
                for keys, cost in measured
                if cost.split_us < cost.single_graph_us * (1 - self.config.minimum_gain_fraction)
            ]
            if not beneficial:
                return NoSplit(NoSplitReason.NO_GAIN)
            chosen, _ = min(beneficial, key=lambda candidate: candidate[1].split_us)
        else:
            chosen = candidates[0]

        boundary = chosen[0].graph_tokens
        return ExecutionPlan(
            slices=(
                SlicePlan(0, boundary, 0, boundary, chosen[0]),
                SlicePlan(boundary, num_requests, boundary, num_requests, chosen[1]),
            ),
            execution=self.config.execution,
            cost_source=self.config.selection_policy,
        )
