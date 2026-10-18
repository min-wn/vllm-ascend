# SPDX-License-Identifier: Apache-2.0
"""CPU reference for temporal attention-plan maintenance, not NPU integration.

Cost units are an abstract rounded-tile model, never predicted latency. Exact
lengths stay in every row; cached scheduling summaries do not replace masks.
Native sink metadata export and device execution remain unqualified.
"""

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import asdict
from itertools import accumulate
from math import isfinite

from aclgraph_incremental_attention_plan_types import (
    AbstractCostModel,
    AttentionPlan,
    CorePartition,
    Geometry,
    PlanDiagnostics,
    PlannerSnapshot,
    PlanPolicy,
    ReductionGroup,
    RequestKey,
    RequestRow,
    RowWork,
    TileTable,
    TileTemplate,
    TopologySummary,
)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _validate_config(geometry, policy):
    if not isinstance(geometry, Geometry) or not isinstance(policy, PlanPolicy):
        raise ValueError("Geometry and PlanPolicy required")
    numeric = (value for name, value in asdict(geometry).items() if name not in ("causal", "dtype", "layout"))
    if not all(_integer(value, 1) for value in numeric):
        raise ValueError("geometry dimensions and core count must be positive integers")
    if geometry.query_heads % geometry.kv_heads or type(geometry.causal) is not bool:
        raise ValueError("integer GQA ratio and boolean causal mode required")
    if geometry.dtype not in ("bfloat16", "float16") or geometry.layout != "TND/BnBsH":
        raise ValueError("CPU reference specifies BF16/FP16 TND and paged BnBsH only")
    if geometry.kv_tile % geometry.page_size:
        raise ValueError("KV scheduling tile must contain whole pages")
    noncapacities = (
        "imbalance_tolerance",
        "boundary_repair_radius",
        "scratch_limit_bytes",
        "cost_model",
        "split_merge_cost_units",
        "minimum_split_gain_units",
    )
    if any(not _integer(value, 1) for name, value in asdict(policy).items() if name not in noncapacities):
        raise ValueError("policy capacities must be positive integers")
    if any(
        not _integer(value)
        for value in (
            policy.boundary_repair_radius,
            policy.scratch_limit_bytes,
            policy.split_merge_cost_units,
            policy.minimum_split_gain_units,
        )
    ):
        raise ValueError("nonnegative repair radius and scratch capacity required")
    if not isinstance(policy.cost_model, AbstractCostModel) or not all(
        _integer(value, 1) for value in asdict(policy.cost_model).values()
    ):
        raise ValueError("positive integer abstract cost coefficients/alignments required")
    if (
        type(policy.imbalance_tolerance) not in (int, float)
        or not isfinite(policy.imbalance_tolerance)
        or policy.imbalance_tolerance < 0
    ):
        raise ValueError("finite nonnegative imbalance tolerance required")


def _validate_rows(rows, policy):
    if len(rows) > policy.max_rows or any(not isinstance(row, RequestRow) for row in rows):
        raise ValueError("row capacity exceeded or invalid RequestRow")
    owners = set()
    for row in rows:
        if (
            not _integer(row.query_length)
            or not _integer(row.kv_length)
            or row.query_length > policy.max_query_length
            or row.kv_length > policy.max_kv_length
        ):
            raise ValueError("query/KV length outside declared capacity")
        if row.kv_length and row.query_length > row.kv_length:
            raise ValueError("live KV length must contain current queries")
        if row.key is None:
            if row.kv_length:
                raise ValueError("padding row must have zero KV length")
            continue
        key = row.key
        if (
            not isinstance(key, RequestKey)
            or not _integer(key.request_id)
            or not _integer(key.generation)
            or key.request_id >= policy.max_requests
            or key.request_id in owners
        ):
            raise ValueError("invalid, duplicate or out-of-range request identity")
        owners.add(key.request_id)


def _cost(query_rows, kv_rows, ratio, model):
    return model.query_coefficient * (
        (query_rows * ratio + model.query_alignment - 1) // model.query_alignment
    ) + model.kv_coefficient * ((kv_rows + model.kv_alignment - 1) // model.kv_alignment)


def _shape(row, geometry):
    ends = tuple(
        min(row.kv_length, row.kv_length - row.query_length + end) if geometry.causal else row.kv_length
        for end in range(geometry.query_tile, row.query_length, geometry.query_tile)
    )
    if row.query_length:
        ends += (row.kv_length,)
    return row.query_length, tuple((max(end, 0) + geometry.kv_tile - 1) // geometry.kv_tile for end in ends)


def _summary(shape, geometry, model):
    query_length, counts = shape
    templates, groups = [], []
    ratio = geometry.query_heads // geometry.kv_heads
    for head in range(geometry.kv_heads):
        for query_index, count in enumerate(counts):
            start = len(templates)
            qstart = query_index * geometry.query_tile
            qend = min(query_length, qstart + geometry.query_tile)
            for tile in range(count):
                templates.append(
                    TileTemplate(
                        head,
                        qstart,
                        qend,
                        tile * geometry.kv_tile,
                        (tile + 1) * geometry.kv_tile,
                        _cost(qend - qstart, geometry.kv_tile, ratio, model),
                    )
                )
            if len(templates) > start:
                groups.append((start, len(templates)))
    full_cost = max((templates[start].cost_units for start, end in groups if end - start > 1), default=0)
    return TopologySummary(
        shape, tuple(templates), (0, *accumulate(tile.cost_units for tile in templates)), tuple(groups), full_cost
    )


def _row_work(index, row, summary, geometry, model):
    corrections = []
    for _, end in summary.groups:
        tail = summary.templates[end - 1]
        limit = row.kv_length - row.query_length + tail.query_end if geometry.causal else row.kv_length
        limit = min(row.kv_length, max(0, limit))
        exact_cost = _cost(
            tail.query_end - tail.query_start, limit - tail.kv_start, geometry.query_heads // geometry.kv_heads, model
        )
        corrections.append((end - 1, limit, exact_cost - tail.cost_units))
    return RowWork(index, row, summary, tuple(corrections), (0, *accumulate(item[2] for item in corrections)))


def _groups(table):
    result, offset = [], 0
    for work in table.work:
        for start, end in work.summary.groups:
            result.append((offset + start, offset + end))
        offset += len(work.summary.templates)
    return tuple(result)


def _partitions(cuts, table):
    return tuple(
        CorePartition(core, start, end, table.prefix_cost(end) - table.prefix_cost(start))
        for core, (start, end) in enumerate(zip(cuts, cuts[1:]))
        if end > start
    )


def _reductions(parts, table, groups, geometry, policy):
    result = []
    for start, end in groups:
        pieces = tuple(
            CorePartition(
                part.core,
                max(part.start, start),
                min(part.end, end),
                table.prefix_cost(min(part.end, end)) - table.prefix_cost(max(part.start, start)),
            )
            for part in parts
            if part.start < end and part.end > start
        )
        if len(pieces) > 1:
            tile = table[start]
            scratch = len(pieces) * (tile.query_end - tile.query_start) * (geometry.query_heads // geometry.kv_heads)
            scratch *= (geometry.head_dim + 2) * 4
            result.append(
                ReductionGroup(
                    tile.row,
                    tile.key,
                    tile.kv_head,
                    tile.query_start,
                    tile.query_end,
                    pieces,
                    scratch,
                    (len(pieces) - 1) * policy.split_merge_cost_units,
                )
            )
    return tuple(result)


def _legal(cuts, table, groups, geometry, policy):
    splits = {}
    for cut in cuts[1:-1]:
        index = bisect_right(groups, (cut, len(table) + 1)) - 1
        if index >= 0 and groups[index][0] < cut < groups[index][1]:
            splits[index] = splits.get(index, 1) + 1
    if any(pieces > policy.max_splits_per_group for pieces in splits.values()):
        return False
    scratch = 0
    for index, pieces in splits.items():
        tile = table[groups[index][0]]
        scratch += pieces * (tile.query_end - tile.query_start) * (geometry.query_heads // geometry.kv_heads)
    # Each group can have a different query width, but shares head dimension.
    return scratch * (geometry.head_dim + 2) * 4 <= policy.scratch_limit_bytes


def _objective(parts, table, groups, geometry, policy):
    merge = sum(group.merge_cost_units for group in _reductions(parts, table, groups, geometry, policy))
    return max((part.cost_units for part in parts), default=0) + merge


def _nearest_cut(target, low, high, table, queries):
    while low < high:
        middle = (low + high) // 2
        queries[0] += 1
        if table.prefix_cost(middle) < target:
            low = middle + 1
        else:
            high = middle
    return low


def _full_partition(table, groups, geometry, policy, queries):
    if not len(table):
        return ()
    total = table.prefix_cost(len(table))
    group_ends = (0, *(end for _, end in groups))
    unsplit = [0]
    split = [0]
    for index in range(1, geometry.num_cores):
        target = total * index / geometry.num_cores
        pivot = _nearest_cut(target, 1, len(table), table, queries)
        for cuts, candidates in (
            (unsplit, group_ends[max(0, bisect_left(group_ends, pivot) - 1) : bisect_left(group_ends, pivot) + 1]),
            (split, range(max(1, pivot - 1), min(len(table), pivot + 1) + 1)),
        ):
            legal = [
                cut
                for cut in candidates
                if cuts[-1] < cut < len(table) and _legal((*cuts, cut, len(table)), table, groups, geometry, policy)
            ]
            if legal:
                cuts.append(min(legal, key=lambda cut: (abs(table.prefix_cost(cut) - target), cut)))
    candidates = [_partitions((*cuts, len(table)), table) for cuts in (unsplit, split)]
    unsplit, split = candidates
    baseline = _objective(unsplit, table, groups, geometry, policy)
    candidate = _objective(split, table, groups, geometry, policy)
    return split if baseline - candidate >= policy.minimum_split_gain_units else unsplit


def _no_split_partition(table, groups, geometry):
    """Build a fresh group-boundary partition used as the repair quality floor."""
    if not len(table):
        return ()
    boundaries = tuple(end for _, end in groups[:-1])
    partition_count = min(geometry.num_cores, len(groups))
    total = table.prefix_cost(len(table))
    cuts = [0]
    # A no-split partition may use fewer physical partitions than cores when
    # there are fewer attention groups; every cut remains a group boundary.
    for core in range(1, partition_count):
        low = bisect_right(boundaries, cuts[-1])
        remaining_cuts = partition_count - core - 1
        high = len(boundaries) - 1 - remaining_cuts
        target = total * core / partition_count
        candidate = min(
            boundaries[low : high + 1],
            key=lambda cut: (abs(table.prefix_cost(cut) - target), cut),
        )
        cuts.append(candidate)
    return _partitions((*cuts, len(table)), table)


def _repair(previous, table, groups, geometry, policy, queries):
    if not previous.partitions or not len(table):
        return () if not len(table) else None
    cuts = [0]
    total = table.prefix_cost(len(table))
    for old in previous.partitions[:-1]:
        guess = round(old.end * len(table) / len(previous.tiles))
        low, high = (
            max(cuts[-1] + 1, guess - policy.boundary_repair_radius),
            min(len(table) - 1, guess + policy.boundary_repair_radius),
        )
        if low > high:
            continue
        target = total * (old.core + 1) / len(previous.partitions)
        pivot = _nearest_cut(target, low, high, table, queries)
        candidates = sorted(range(low, high + 1), key=lambda cut: (abs(cut - pivot), cut))
        cut = next(
            (cut for cut in candidates if _legal((*cuts, cut, len(table)), table, groups, geometry, policy)), None
        )
        if cut is not None:
            cuts.append(cut)
    parts = _partitions((*cuts, len(table)), table)
    largest = max(
        max(
            item.summary.maximum_full_tile_cost,
            max((item.summary.templates[index].cost_units + delta for index, _, delta in item.corrections), default=0),
        )
        for item in table.work
    )
    lower_bound = max(total / geometry.num_cores, largest)
    # Compare against a fresh, valid no-split plan.  The previous implementation
    # removed only the current repair cuts, which could leave a very poor
    # boundary-only baseline and make expensive merge plans look beneficial.
    unsplit = _no_split_partition(table, groups, geometry)
    candidate_objective = _objective(parts, table, groups, geometry, policy)
    baseline_objective = _objective(unsplit, table, groups, geometry, policy)
    if baseline_objective - candidate_objective < policy.minimum_split_gain_units:
        return None
    return parts if max(part.cost_units for part in parts) <= (1 + policy.imbalance_tolerance) * lower_bound else None


class IncrementalAttentionPlanner:
    def __init__(self, geometry=None, policy=None):
        geometry = Geometry() if geometry is None else geometry
        policy = PlanPolicy() if policy is None else policy
        _validate_config(geometry, policy)
        self.geometry, self.policy = geometry, policy
        self._cache, self._lifetimes = OrderedDict(), {}
        self._previous, self._revision = None, 0

    def snapshot(self):
        return PlannerSnapshot(
            self.geometry,
            tuple(self._cache.items()),
            tuple(sorted(self._lifetimes.items())),
            self._previous,
            self._revision,
        )

    def plan(self, rows, *, geometry=None, force_rebuild=False, retired=()):
        rows, retired = tuple(rows), tuple(retired)
        geometry = self.geometry if geometry is None else geometry
        _validate_config(geometry, self.policy)
        _validate_rows(rows, self.policy)
        if type(force_rebuild) is not bool or len(set(retired)) != len(retired):
            raise ValueError("boolean force_rebuild and unique retired identities required")
        lifetimes = dict(self._lifetimes)
        for key in retired:
            if not isinstance(key, RequestKey) or lifetimes.get(key.request_id) != (key.generation, False):
                raise ValueError("retirement must match the current live generation")
            lifetimes[key.request_id] = key.generation, True
        for row in rows:
            if row.key is None:
                continue
            key = row.key
            prior = lifetimes.get(key.request_id)
            if prior is not None and prior != (key.generation, False):
                if prior != (key.generation - 1, True):
                    raise ValueError("stale generation or identity recycled without retirement")
            lifetimes[key.request_id] = key.generation, False
        shape_changed = geometry != self.geometry
        full = force_rebuild or shape_changed or self._previous is None
        cache = OrderedDict() if shape_changed else self._cache.copy()
        for key in retired:
            cache.pop(key, None)
        built = reused = constructed = reused_costs = tails = 0
        work = []
        shapes = tuple(_shape(row, geometry) if row.key else None for row in rows)
        tile_count = sum(sum(shape[1]) * geometry.kv_heads for shape in shapes if shape is not None)
        if tile_count > self.policy.max_tiles:
            raise ValueError("tile capacity exceeded before template allocation")
        for index, row in enumerate(rows):
            if row.key is None:
                continue
            shape = shapes[index]
            summary = cache.get(row.key)
            if full or summary is None or summary.shape != shape:
                summary = _summary(shape, geometry, self.policy.cost_model)
                built += 1
                constructed += len(summary.templates)
            else:
                reused += 1
                reused_costs += len(summary.templates) - len(summary.groups)
            item = _row_work(index, row, summary, geometry, self.policy.cost_model)
            work.append(item)
            tails += len(summary.groups)
            cache[row.key] = summary
            cache.move_to_end(row.key)
        table = TileTable(
            tuple(work),
            tuple(accumulate(len(item.summary.templates) for item in work)),
            tuple(accumulate(item.prefix_cost(len(item.summary.templates)) for item in work)),
        )
        evictions = 0
        cache_tiles = sum(len(summary.templates) for summary in cache.values())
        while len(cache) > self.policy.max_cached_requests or cache_tiles > self.policy.max_tiles:
            _, summary = cache.popitem(last=False)
            cache_tiles -= len(summary.templates)
            evictions += 1
        groups, queries = _groups(table), [0]
        reason = "geometry_changed" if shape_changed else "forced" if force_rebuild else "initial" if full else None
        parts = None if full else _repair(self._previous, table, groups, geometry, self.policy, queries)
        if parts is None:
            reason = reason or "imbalance_or_split_budget"
            parts = _full_partition(table, groups, geometry, self.policy, queries)
        reductions = _reductions(parts, table, groups, geometry, self.policy)
        old_rows = (
            {} if self._previous is None else {row.key: i for i, row in enumerate(self._previous.rows) if row.key}
        )
        current = {row.key: i for i, row in enumerate(rows) if row.key}
        diagnostics = PlanDiagnostics(
            "full_rebuild" if full else "partition_fallback" if reason else "incremental",
            reason,
            len(current),
            len(current.keys() - old_rows.keys()),
            len(old_rows.keys() - current.keys()),
            sum(old_rows[key] != current[key] for key in old_rows.keys() & current.keys()),
            built,
            reused,
            constructed,
            reused_costs,
            tails,
            0 if reason else max(0, len(parts) - 1),
            queries[0],
            int(reason is not None),
            len(reductions),
            sum(len(group.pieces) for group in reductions),
            max((part.cost_units for part in parts), default=0),
            table.prefix_cost(len(table)),
            len(cache),
            evictions,
            shape_changed,
            sum(group.merge_cost_units for group in reductions),
            _objective(parts, table, groups, geometry, self.policy),
        )
        result = AttentionPlan(
            rows, geometry, table, parts, reductions, sum(group.scratch_bytes for group in reductions), diagnostics
        )
        self.geometry, self._cache, self._lifetimes = geometry, cache, lifetimes
        self._previous, self._revision = result, self._revision + 1
        return result


def full_rebuild_plan(rows, geometry=None, policy=None):
    """Stateless full-summary and full-boundary construction control."""
    return IncrementalAttentionPlanner(geometry, policy).plan(rows, force_rebuild=True)
