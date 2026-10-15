from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vllm_ascend.worker.multi_graph.planner import CostKey, MeasuredCost


@dataclass(frozen=True)
class KVBucketPolicy:
    upper_bounds: tuple[int, ...]
    minimum_vectors: int = 2

    def __post_init__(self) -> None:
        if not self.upper_bounds or tuple(sorted(set(self.upper_bounds))) != self.upper_bounds:
            raise ValueError("bucket upper bounds must be strictly increasing")
        if any(type(bound) is not int or bound <= 0 for bound in self.upper_bounds):
            raise ValueError("bucket upper bounds must be positive integers")
        if type(self.minimum_vectors) is not int or self.minimum_vectors < 2:
            raise ValueError("bucket interpolation requires at least two distinct measured vectors")

    def vector(self, lengths: tuple[int, ...]) -> tuple[int, ...] | None:
        result = []
        for length in lengths:
            if type(length) is not int or length <= 0:
                raise ValueError("KV lengths must be positive integers")
            bound = next((bound for bound in self.upper_bounds if length <= bound), None)
            if bound is None:
                return None
            result.append(bound)
        return tuple(result)


class BucketCostTable(Mapping[CostKey, MeasuredCost]):
    """Exact observations take precedence; interpolation uses conservative extrema.

    Ordered per-request buckets retain lane load imbalance. Every contributing
    exact record has already passed the minimum sample threshold. The slowest
    split and fastest baseline bound an interpolated estimate conservatively.
    """

    def __init__(self, exact: Mapping[CostKey, MeasuredCost], policy: KVBucketPolicy):
        self.exact = dict(exact)
        self.policy = policy
        grouped: dict[CostKey, list[MeasuredCost]] = {}
        for key, cost in self.exact.items():
            vector = policy.vector(key.kv_lengths)
            if vector is not None:
                grouped.setdefault(CostKey(key.graph_keys, key.execution, vector), []).append(cost)
        self.buckets = {
            key: MeasuredCost(min(cost.single_graph_us for cost in values), max(cost.split_us for cost in values))
            for key, values in grouped.items()
            if len(values) >= policy.minimum_vectors
        }

    def __getitem__(self, key: CostKey) -> MeasuredCost:
        if key in self.exact:
            return self.exact[key]
        vector = self.policy.vector(key.kv_lengths)
        if vector is None:
            raise KeyError(key)
        return self.buckets[CostKey(key.graph_keys, key.execution, vector)]

    def __iter__(self):
        return iter(self.exact)

    def __len__(self) -> int:
        return len(self.exact)
