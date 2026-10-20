# SPDX-License-Identifier: Apache-2.0
"""Exact whole-head planning with an inexpensive feasible prefix upper bound.

Only the initial search interval changes. The minimax objective, canonical
reconstruction and complete output ABI are identical to the first prototype.
"""

import triton
import triton.language as tl
from aclgraph_descriptor_plan_kernel import gather_integer, upper_bound


@triton.jit
def descriptor_plan_bound(KV, OUT, STATS, BATCH: tl.constexpr, BLOCK: tl.constexpr, DIAGNOSTIC: tl.constexpr):
    index = tl.arange(0, BLOCK)
    heads: tl.constexpr = BATCH * 2
    # Prefix entry i contains the cost of complete heads [0, i).
    length = tl.load(KV + tl.maximum(index - 1, 0) // 2, (index > 0) & (index <= heads), 0).to(tl.int32)
    maximum_kv = tl.max(length, 0)
    effective_max = tl.where(maximum_kv > 0, maximum_kv, 1024)
    inner = tl.minimum(effective_max, 512)
    complete = length // inner
    tail = length % inner
    cost = complete * (12 + 10 * ((inner + 63) // 64)) + tl.where(tail > 0, 12 + 10 * ((tail + 63) // 64), 0)
    prefix = tl.cumsum(cost, 0)
    positive_prefix = tl.cumsum((cost > 0).to(tl.int32), 0)
    total = tl.sum(cost, 0)
    positive = tl.sum((cost > 0).to(tl.int32), 0)
    used = tl.maximum(1, tl.minimum(20, positive))
    lower = tl.maximum(tl.max(cost, 0), (total + used - 1) // used)
    # Quantile endpoints are nondecreasing; repeated/zero-cost chunks can be
    # merged, leaving at most `used` positive chunks. Splitting positive atoms
    # refines that feasible partition to exactly `used` without raising its
    # largest load. No approximate partition is exported to the consumer.
    ordinal = index + 1
    proposed = upper_bound(prefix, ordinal * total // used, heads, BLOCK)
    proposed = tl.where(ordinal < used, proposed, heads)
    boundary_cost = gather_integer(prefix, proposed)
    previous_cost = gather_integer(boundary_cost, tl.maximum(index - 1, 0))
    previous_cost = tl.where(index == 0, 0, previous_cost)
    chunk_cost = tl.where(index < used, boundary_cost - previous_cost, 0)
    higher = tl.max(chunk_cost, 0)
    initial_upper = higher
    initial_lower = lower
    iterations = 0
    while lower < higher:
        if DIAGNOSTIC:
            iterations += 1
        limit = (lower + higher) // 2
        jump = tl.minimum(upper_bound(prefix, prefix + limit, heads, BLOCK), heads)
        # Binary lifting evaluates up to twenty greedy segments without a
        # serial loop through every request. All search starts run in parallel.
        position = tl.full((1,), 0, tl.int32)
        power = jump
        for bit in tl.static_range(5):
            position = tl.where((used & (1 << bit)) != 0, gather_integer(power, position), position)
            power = gather_integer(power, power)
        fits = tl.sum(position, 0) >= heads
        higher = tl.where(fits, limit, higher)
        lower = tl.where(fits, lower, limit + 1)

    jump = tl.minimum(upper_bound(prefix, prefix + lower, heads, BLOCK), heads)
    ordinal = index + 1
    ends = tl.full((BLOCK,), 0, tl.int32)
    power = jump
    for bit in tl.static_range(5):
        ends = tl.where((ordinal & (1 << bit)) != 0, gather_integer(power, ends), ends)
        power = gather_integer(power, power)
    # Match the existing canonical reconstruction, reserving one positive atom
    # per remaining core, even with zero-cost holes and padding sentinels.
    reserve = upper_bound(positive_prefix, positive - used + ordinal, heads, BLOCK)
    ends = tl.where(ordinal < used, tl.minimum(ends, reserve), heads)

    word = tl.arange(0, 1024)
    core = word // 10
    end = gather_integer(ends, tl.minimum(core, BLOCK - 1))
    result = tl.full((1024,), 0, tl.int32)
    result = tl.where((word < 200) & (core < used) & (word % 10 == 0), 1, result)
    result = tl.where((word < 200) & (core < used) & (word % 10 == 1), end, result)
    result = tl.where((word >= 360) & (word < 360 + 6 * used) & ((word - 360) % 3 == 0), 1, result)
    result = tl.where(word == 576, 512, result)
    result = tl.where(word == 577, inner, result)
    result = tl.where(word == 578, 8, result)
    result = tl.where(word == 580, used, result)
    result = tl.where(word == 582, 4, result)
    result = tl.where(word == 583, effective_max, result)
    result = tl.where((word == 584) | (word == 585), BATCH, result)
    tl.store(OUT + word, result)
    if DIAGNOSTIC:
        field = tl.arange(0, 8)
        stats = tl.where(field == 0, initial_lower, 0)
        stats = tl.where(field == 1, initial_upper, stats)
        stats = tl.where(field == 2, lower, stats)
        stats = tl.where(field == 3, iterations, stats)
        stats = tl.where(field == 4, used, stats)
        stats = tl.where(field == 5, total, stats)
        tl.store(STATS + field, stats)


def launch(kv_lengths, output, *, diagnostics=None):
    """Build the bound inside the captured kernel, never as a host input."""
    batch = kv_lengths.numel()
    if not 1 <= batch <= 128 or tuple(output.shape) != (1024,):
        raise ValueError("bounded Q4 plan requires 1..128 physical requests and 1024 metadata words")
    if diagnostics is not None and tuple(diagnostics.shape) != (8,):
        raise ValueError("optional untimed diagnostic requires eight int32 words")
    descriptor_plan_bound[(1,)](
        kv_lengths,
        output,
        diagnostics if diagnostics is not None else output,
        batch,
        triton.next_power_of_2(2 * batch + 1),
        diagnostics is not None,
    )
