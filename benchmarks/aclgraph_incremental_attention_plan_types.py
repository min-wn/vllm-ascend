# SPDX-License-Identifier: Apache-2.0
"""Immutable CPU specification; none of these objects is a native sink ABI."""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class RequestKey:
    request_id: int
    generation: int


@dataclass(frozen=True)
class RequestRow:
    key: RequestKey | None
    query_length: int
    kv_length: int


@dataclass(frozen=True)
class Geometry:
    kv_heads: int = 2
    query_heads: int = 12
    head_dim: int = 256
    query_tile: int = 4
    kv_tile: int = 512
    causal: bool = True
    page_size: int = 128
    dtype: str = "bfloat16"
    layout: str = "TND/BnBsH"
    num_cores: int = 20


@dataclass(frozen=True)
class AbstractCostModel:
    """Uncalibrated rounded-tile work units; coefficients do not model latency."""

    query_coefficient: int = 6
    kv_coefficient: int = 10
    query_alignment: int = 16
    kv_alignment: int = 64


@dataclass(frozen=True)
class PlanPolicy:
    max_rows: int = 128
    max_requests: int = 256
    max_cached_requests: int = 256
    max_kv_length: int = 65536
    max_query_length: int = 4
    max_tiles: int = 65536
    max_splits_per_group: int = 2
    scratch_limit_bytes: int = 16777216
    imbalance_tolerance: float = 0.15
    boundary_repair_radius: int = 8
    split_merge_cost_units: int = 0
    minimum_split_gain_units: int = 1
    cost_model: AbstractCostModel = AbstractCostModel()


@dataclass(frozen=True)
class WorkTile:
    row: int
    key: RequestKey
    kv_head: int
    query_start: int
    query_end: int
    kv_start: int
    kv_end: int
    cost_units: int


@dataclass(frozen=True)
class TileTemplate:
    kv_head: int
    query_start: int
    query_end: int
    kv_start: int
    kv_end: int
    cost_units: int


@dataclass(frozen=True)
class TopologySummary:
    shape: tuple
    templates: tuple[TileTemplate, ...]
    prefix_costs: tuple[int, ...]
    groups: tuple[tuple[int, int], ...]
    maximum_full_tile_cost: int


@dataclass(frozen=True)
class RowWork:
    row: int
    request: RequestRow
    summary: TopologySummary
    corrections: tuple[tuple[int, int, int], ...]
    correction_prefix: tuple[int, ...]

    def prefix_cost(self, count):
        changed = bisect_left(self.corrections, (count,))
        return self.summary.prefix_costs[count] + self.correction_prefix[changed]


@dataclass(frozen=True)
class TileTable(Sequence):
    """Lazy exact views over cached templates; planning need not emit all tiles."""

    work: tuple[RowWork, ...]
    row_ends: tuple[int, ...]
    cost_ends: tuple[int, ...]

    def __len__(self):
        return self.row_ends[-1] if self.row_ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row_index = bisect_right(self.row_ends, index)
        prior = self.row_ends[row_index - 1] if row_index else 0
        item = self.work[row_index]
        local = index - prior
        template = item.summary.templates[local]
        correction = bisect_left(item.corrections, (local,))
        end, cost = template.kv_end, template.cost_units
        if correction < len(item.corrections) and item.corrections[correction][0] == local:
            _, end, delta = item.corrections[correction]
            cost += delta
        return WorkTile(
            item.row,
            item.request.key,
            template.kv_head,
            template.query_start,
            template.query_end,
            template.kv_start,
            end,
            cost,
        )

    def prefix_cost(self, count):
        if not 0 <= count <= len(self):
            raise IndexError(count)
        if not count:
            return 0
        row_index = bisect_left(self.row_ends, count)
        prior = self.row_ends[row_index - 1] if row_index else 0
        prior_cost = self.cost_ends[row_index - 1] if row_index else 0
        return prior_cost + self.work[row_index].prefix_cost(count - prior)


@dataclass(frozen=True)
class CorePartition:
    core: int
    start: int
    end: int
    cost_units: int


@dataclass(frozen=True)
class ReductionGroup:
    row: int
    key: RequestKey
    kv_head: int
    query_start: int
    query_end: int
    pieces: tuple[CorePartition, ...]
    scratch_bytes: int
    merge_cost_units: int


@dataclass(frozen=True)
class PlanDiagnostics:
    """Summary/boundary counters; prefix queries count binary-search probes only.

    Full rebuilds count boundary repartition, including fallback that retains
    cached request summaries. Zero merge charge is an optimistic ceiling.
    """

    mode: str
    fallback_reason: str | None
    active_requests: int
    admitted_requests: int
    removed_requests: int
    reordered_requests: int
    topology_summaries_built: int
    topology_summaries_reused: int
    topology_tiles_constructed: int
    full_tile_costs_reused: int
    exact_tail_evaluations: int
    boundary_repairs: int
    boundary_prefix_queries: int
    full_rebuilds: int
    split_groups: int
    split_pieces: int
    max_load_units: int
    total_cost_units: int
    cache_entries: int
    cache_evictions: int
    shape_invalidated: bool
    merge_cost_units: int
    estimated_critical_path_units: int


@dataclass(frozen=True)
class AttentionPlan:
    rows: tuple[RequestRow, ...]
    geometry: Geometry
    tiles: TileTable
    partitions: tuple[CorePartition, ...]
    reductions: tuple[ReductionGroup, ...]
    scratch_used_bytes: int
    diagnostics: PlanDiagnostics


@dataclass(frozen=True)
class PlannerSnapshot:
    geometry: Geometry
    cache: tuple
    lifetimes: tuple
    previous: AttentionPlan | None
    revision: int
