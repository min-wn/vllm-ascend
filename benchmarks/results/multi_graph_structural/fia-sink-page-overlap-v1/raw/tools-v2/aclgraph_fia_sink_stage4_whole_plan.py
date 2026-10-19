# SPDX-License-Identifier: Apache-2.0
"""Emit whole-head no-FD plans for the qualified Q4 GQA sink consumer.

This CPU producer retains the qualified inner tiling. Its plan construction and
1024-word H2D copy belong inside any executable algorithm's wall endpoint. A
plan prepared outside timing is only an attention-budget diagnostic.
"""

from dataclasses import dataclass

METADATA_WORDS = 1024
BASE_OFFSET = 576
BASE_FIELDS = 10
AIC_STRIDE = 10
AIV_OFFSET = 360
AIV_STRIDE = 3
MAX_CORES = 20
QUERY_LENGTH = 4
QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 256
GROUP_SIZE = QUERY_HEADS // KV_HEADS
M_BASE = 512
MAX_S_INNER = 512
FD_M_BASE = 8
PAGE_SIZE = 128
MAX_PHYSICAL_TOKENS = 512
MAX_KV_LENGTH = 4097
COST_MODEL = "abstract_6_10_upper_bound"


@dataclass(frozen=True, slots=True)
class WholePlanPreflight:
    """Strict geometry/template facts reusable by one stable record source."""

    record: object
    template_metadata: object
    cores: int
    batch: int
    kv_lengths: tuple[int, ...]
    base: tuple[int, ...]


def _integers(values, name):
    result = list(values)
    if any(type(value) is not int for value in result):
        raise ValueError(f"{name} must contain exact Python integers")
    return result


def _geometry(record):
    """Validate exact shapes; page-ID upper bounds require trace provenance."""
    for field, expected in (
        ("query_heads", QUERY_HEADS),
        ("kv_heads", KV_HEADS),
        ("head_dim", HEAD_DIM),
        ("page_size", PAGE_SIZE),
        ("sparse_mode", 3),
    ):
        if field in record and record[field] != expected:
            raise ValueError("record geometry differs from the qualified Q12/KV2/D256/page128 target")
    physical = record["physical_tokens"]
    if type(physical) is not int or not 0 < physical <= MAX_PHYSICAL_TOKENS or physical % QUERY_LENGTH:
        raise ValueError("qualified physical token capacity must be a positive multiple of four, at most 512")
    batch = physical // QUERY_LENGTH
    query = _integers(record["actual_seq_qlen"], "query endpoints")
    kv = _integers(record["kv_lengths"], "KV lengths")
    if query != list(range(QUERY_LENGTH, physical + 1, QUERY_LENGTH)) or len(kv) != batch:
        raise ValueError("one fixed Q4 segment and exact KV length per physical request required")
    if "q_lengths" in record and _integers(record["q_lengths"], "query lengths") != [QUERY_LENGTH] * batch:
        raise ValueError("only full-width physical Q4 segments are supported")
    if "physical_batch" in record and record["physical_batch"] != batch:
        raise ValueError("physical request capacity disagrees with the TND extent")
    if any(value < 0 or value > MAX_KV_LENGTH for value in kv):
        raise ValueError("KV lengths must be within the qualified exact range 0..4097")
    if "block_table" in record:
        table = record["block_table"]
        if not isinstance(table, (list, tuple)) or len(table) != batch:
            raise ValueError("block_table must be a fixed-width list/tuple with one row per physical request")
        if not table or any(not isinstance(row, (list, tuple)) for row in table):
            raise ValueError("each block_table row must be a list/tuple")
        width = len(table[0])
        if width < 1 or any(len(row) != width for row in table):
            raise ValueError("block_table rows must have one shared positive page width")
        page_count = record.get("page_count")
        if page_count is None and isinstance(record.get("owner_pages"), dict):
            page_count = record["owner_pages"].get("page_count")
        if page_count is None and record.get("owner_pages") is not None:
            page_count = getattr(record["owner_pages"], "page_count", None)
        if page_count is not None and (type(page_count) is not int or page_count < 1):
            raise ValueError("optional page_count must be a positive exact integer")
        for row in table:
            for page in row:
                if type(page) is not int or page < 0 or (page_count is not None and page >= page_count):
                    raise ValueError("block_table page IDs must be nonnegative exact integers within page_count")
        if any(len(row) * PAGE_SIZE < value for row, value in zip(table, kv, strict=True)):
            raise ValueError("page-table extent must cover every exact KV length")
    return batch, kv


def _base(template_metadata, batch, kv):
    template = _integers(template_metadata, "metadata template")
    if len(template) != METADATA_WORDS or any(not 0 <= value <= 2**31 - 1 for value in template):
        raise ValueError("expected a qualified nonnegative int32[1024] native metadata template")
    base = template[BASE_OFFSET : BASE_OFFSET + BASE_FIELDS]
    maximum_kv = max(kv)
    effective_maximum_kv = maximum_kv or 1024
    expected_s_inner = min(MAX_S_INNER, effective_maximum_kv)
    if base[:3] != [M_BASE, expected_s_inner, FD_M_BASE] or base[6] != QUERY_LENGTH:
        raise ValueError("template must retain the qualified fresh Q4 GQA M/S2/FD inner tiling")
    if base[7] != effective_maximum_kv or base[8] != batch or base[9] != batch:
        raise ValueError("template max KV and exact-length dimensions must match this physical record")
    base[3:6] = [0, 0, 0]
    base[6:] = [QUERY_LENGTH, effective_maximum_kv, batch, batch]
    return base


def _head_cost(kv_length, s_inner):
    """Return the existing abstract 6/10 aligned upper-bound cost proxy."""
    if not kv_length:
        return 0
    complete, tail = divmod(kv_length, s_inner)
    aligned_query = (QUERY_LENGTH * GROUP_SIZE + 15) // 16
    full_cost = 6 * aligned_query + 10 * ((s_inner + 63) // 64)
    tail_cost = 6 * aligned_query + 10 * ((tail + 63) // 64) if tail else 0
    return complete * full_cost + tail_cost


def _cuts(costs, cores):
    positive = [(index, cost) for index, cost in enumerate(costs) if cost]
    if not positive:
        return [len(costs)]
    used = min(cores, len(positive))
    weights = [cost for _, cost in positive]

    def groups_needed(limit):
        groups, current = 1, 0
        for cost in weights:
            if current + cost > limit:
                groups, current = groups + 1, 0
            current += cost
        return groups

    lower, upper = max(max(weights), (sum(weights) + used - 1) // used), sum(weights)
    while lower < upper:
        middle = (lower + upper) // 2
        if groups_needed(middle) <= used:
            upper = middle
        else:
            lower = middle + 1

    ends, current = [], 0
    for position, (index, cost) in enumerate(positive):
        groups_left = used - len(ends)
        atoms_left = len(positive) - position
        if current and (current + cost > lower or atoms_left == groups_left - 1):
            ends.append(index)
            current = 0
        current += cost
    ends.append(len(costs))
    if len(ends) != used:
        raise AssertionError("whole-head partition did not create the declared nonempty core prefix")
    return ends


def validate_whole_plan(record, metadata, *, cores=MAX_CORES):
    """Independently check canonical coverage and the no-FD consumer contract."""
    if type(cores) is not int or not 1 <= cores <= MAX_CORES:
        raise ValueError("qualified no-FD core count must be an integer in 1..20")
    batch, kv = _geometry(record)
    words = _integers(metadata, "whole plan")
    if len(words) != METADATA_WORDS:
        raise ValueError("whole plan must occupy exactly 1024 int32 words")
    base = words[BASE_OFFSET : BASE_OFFSET + BASE_FIELDS]
    used = base[4]
    if not 1 <= used <= cores <= MAX_CORES or base[3] != 0 or base[5] != 0:
        raise ValueError("no-FD requires a valid prefix of attention cores and no reduction cores/groups")
    if base[:3] != [M_BASE, min(MAX_S_INNER, max(kv) or 1024), FD_M_BASE]:
        raise ValueError("whole plan inner tiling differs from the qualified target")
    if base[6:] != [QUERY_LENGTH, max(kv) or 1024, batch, batch]:
        raise ValueError("whole plan exact lengths and descriptor dimensions are stale")
    expected = [0] * METADATA_WORDS
    expected[BASE_OFFSET : BASE_OFFSET + BASE_FIELDS] = base
    start, owners, loads = 0, [None] * (batch * KV_HEADS), []
    for core in range(used):
        slot = words[core * AIC_STRIDE : (core + 1) * AIC_STRIDE]
        end = slot[1]
        if slot[0] != 1 or slot[2:] != [0] * (AIC_STRIDE - 2) or not start < end <= len(owners):
            raise ValueError("each core requires a strictly advancing whole-head endpoint and zero FD fields")
        load = 0
        for head in range(start, end):
            if owners[head] is not None:
                raise ValueError("whole-head plan duplicates work")
            owners[head] = core
            load += _head_cost(kv[head // KV_HEADS], base[1])
        loads.append(load)
        expected[core * AIC_STRIDE : (core + 1) * AIC_STRIDE] = slot
        start = end
    if start != len(owners) or any(owner is None for owner in owners):
        raise ValueError("whole-head plan must cover the final physical padding sentinel")
    if any(kv) and any(load == 0 for load in loads):
        raise ValueError("no empty active cores may be inserted among useful assignments")
    for vector in range(2 * used):
        expected[AIV_OFFSET + vector * AIV_STRIDE] = 1
    if words != expected:
        raise ValueError("unused metadata, reduction endpoints and padding words must be zero")
    # Query-group/S2 coordinates are enumerated independently of the partition
    # algorithm; every semantic tile belongs to exactly its complete head.
    tiles = sum((length + base[1] - 1) // base[1] * KV_HEADS for length in kv)
    return dict(
        used_cores=used,
        physical_heads=len(owners),
        attention_tiles=tiles,
        core_costs=loads,
        cost_model=COST_MODEL,
    )


def preflight_whole_plan(record, template_metadata, cores=MAX_CORES):
    """Run strict record/page/template checks once for a stable plan source."""
    if type(cores) is not int or not 1 <= cores <= MAX_CORES:
        raise ValueError("qualified no-FD core count must be an integer in 1..20")
    batch, kv = _geometry(record)
    base = _base(template_metadata, batch, kv)
    return WholePlanPreflight(record, template_metadata, cores, batch, tuple(kv), tuple(base))


def build_whole_plan(record, template_metadata, cores=MAX_CORES, *, validate=False, preflight=None):
    """Build a contiguous no-FD plan, optionally running full qualification.

    ``validate=False`` is an execution-path optimization only. Callers must
    first run ``whole_plan(..., validate=True)`` or ``validate_whole_plan`` on
    the same record/template geometry. It skips the second O(B) owner/load
    audit; it does not broaden the supported geometry. Supplying a
    ``WholePlanPreflight`` also skips repeating the strict O(B) page-table and
    template checks; identity guards prevent applying it to another source.

    Geometry is fixed by this producer to Q12/KV2/D256, with one query-group
    tile per physical Q4 head. The template must come from the qualified sink
    variant; it conveys inner tiling rather than FD scheduling arrays.
    """
    if type(cores) is not int or not 1 <= cores <= MAX_CORES:
        raise ValueError("qualified no-FD core count must be an integer in 1..20")
    if not validate and preflight is None:
        raise ValueError("fast whole-plan construction requires strict preflight")
    if preflight is None:
        batch, kv = _geometry(record)
        base = _base(template_metadata, batch, kv)
    else:
        if (
            preflight.record is not record
            or preflight.template_metadata is not template_metadata
            or preflight.cores != cores
        ):
            raise ValueError("whole-plan preflight belongs to a different stable record/template source")
        batch, kv = preflight.batch, list(preflight.kv_lengths)
        base = list(preflight.base)
    costs = [_head_cost(length, base[1]) for length in kv for _ in range(KV_HEADS)]
    ends = _cuts(costs, cores)
    words = [0] * METADATA_WORDS
    for core, end in enumerate(ends):
        words[core * AIC_STRIDE] = 1
        words[core * AIC_STRIDE + 1] = end
    for vector in range(2 * len(ends)):
        words[AIV_OFFSET + vector * AIV_STRIDE] = 1
    base[4] = len(ends)
    words[BASE_OFFSET : BASE_OFFSET + BASE_FIELDS] = base
    if validate:
        validate_whole_plan(record, words, cores=cores)
    return words


def whole_plan(record, template_metadata, cores=MAX_CORES):
    """Return a qualified legal no-FD plan, including full coverage checks."""
    return build_whole_plan(record, template_metadata, cores, validate=True)


def whole_plan_fast(record, template_metadata, cores=MAX_CORES, *, preflight=None):
    """Build after strict preflight, skipping duplicate full-plan validation."""
    return build_whole_plan(record, template_metadata, cores, validate=False, preflight=preflight)
