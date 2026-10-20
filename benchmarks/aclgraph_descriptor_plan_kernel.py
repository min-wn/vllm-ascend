# SPDX-License-Identifier: Apache-2.0
"""Experimental bounded AIV producer for the existing whole-head metadata ABI.

This is a research kernel, not a registered model path. Its cost proxy and
consumer contract are inherited from the previously qualified Q4 policy.
"""

import triton
import triton.language as tl


@triton.jit
def gather_integer(values, indices):
    # This installed Ascend backend does not implement int32 gather. Every
    # value here is an integer <= 2 * 256 * 758 < 2**24, so numeric FP32
    # conversion is exact. Costs, prefix addition and comparisons stay integer.
    return tl.gather(values.to(tl.float32), indices, 0).to(tl.int32)


@triton.jit
def upper_bound(values, targets, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), HEADS + 1, tl.int32)
    for _ in tl.static_range(9):
        active = low < high
        middle = (low + high) // 2
        sample = gather_integer(values, tl.minimum(middle, BLOCK - 1))
        take = (middle <= HEADS) & (sample <= targets)
        low = tl.where(active & take, middle + 1, low)
        high = tl.where(active & ~take, middle, high)
    return low - 1


@triton.jit
def descriptor_plan(KV, OUT, BATCH: tl.constexpr, BLOCK: tl.constexpr):
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
    higher = total
    while lower < higher:
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


def launch(kv_lengths, output):
    """Launch one fixed-grid plan kernel; never read a device value on host."""
    batch = kv_lengths.numel()
    if not 1 <= batch <= 128 or tuple(output.shape) != (1024,):
        raise ValueError("bounded Q4 plan requires 1..128 physical requests and 1024 metadata words")
    descriptor_plan[(1,)](kv_lengths, output, batch, triton.next_power_of_2(2 * batch + 1))
