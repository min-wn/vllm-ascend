"""Experimental vector decode reference for fixed-address device metadata.

This kernel is a qualification prototype, not an attention backend replacement.
It supports dense paged GQA, one query per request and no sliding window.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(
    query,
    key,
    value,
    lengths,
    table,
    output,
    workspace,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // (HEADS // KV_HEADS)
    dimensions = tl.arange(0, DIM)
    positions = tl.arange(0, TILE)
    q = tl.load(query + request * HEADS * DIM + head * DIM + dimensions).to(tl.float32)
    length = tl.load(lengths + request)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    accumulator = tl.full((DIM,), 0.0, tl.float32)
    sections = triton.cdiv(TABLE_WIDTH * BLOCK_SIZE, TILE * SPLITS)
    for section in range(sections):
        start = (split * sections + section) * TILE
        if start < length:
            tokens = start + positions
            valid = tokens < length
            pages = tl.load(table + request * TABLE_WIDTH + tokens // BLOCK_SIZE, valid, other=0)
            offsets = (
                (pages[:, None] * BLOCK_SIZE + tokens[:, None] % BLOCK_SIZE) * KV_HEADS + kv_head
            ) * DIM + dimensions[None, :]
            k = tl.load(key + offsets, valid[:, None], other=0).to(tl.float32)
            scores = tl.sum(k * q[None, :], 1) * SCALE
            scores = tl.where(valid, scores, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 0))
            adjustment = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum)
            v = tl.load(value + offsets, valid[:, None], other=0).to(tl.float32)
            accumulator = accumulator * adjustment + tl.sum(v * probabilities[:, None], 0)
            denominator = denominator * adjustment + tl.sum(probabilities, 0)
            maximum = next_maximum
    if SPLITS == 1:
        result = accumulator / tl.maximum(denominator, 1.0e-20)
        tl.store(output + request * HEADS * DIM + head * DIM + dimensions, result)
    else:
        base = ((request * HEADS + head) * SPLITS + split) * (DIM + 2)
        tl.store(workspace + base + dimensions, accumulator)
        tl.store(workspace + base + DIM, maximum)
        tl.store(workspace + base + DIM + 1, denominator)


@triton.jit
def _reduce(workspace, output, HEADS: tl.constexpr, DIM: tl.constexpr, SPLITS: tl.constexpr):
    request = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dimensions = tl.arange(0, DIM)
    base = ((request * HEADS + head) * SPLITS + splits) * (DIM + 2)
    maximum = tl.load(workspace + base + DIM)
    denominator = tl.load(workspace + base + DIM + 1)
    global_maximum = tl.max(maximum, 0)
    global_maximum = tl.where(global_maximum == float("-inf"), 0.0, global_maximum)
    weights = tl.where(denominator > 0, tl.exp(maximum - global_maximum), 0.0)
    accumulator = tl.load(workspace + base[:, None] + dimensions[None, :])
    numerator = tl.sum(accumulator * weights[:, None], 0)
    normalizer = tl.sum(denominator * weights, 0)
    tl.store(output + (request * HEADS + head) * DIM + dimensions, numerator / tl.maximum(normalizer, 1.0e-20))


def device_length_decode(query, key, value, lengths, table, output, scale=None, *, splits=1, workspace=None):
    """Launch with static descriptors; callers must guarantee valid device values.

    Lengths must be in [0, table width * block size]; live page IDs must be in
    range. No value validation/readback belongs inside this captured launcher.
    """
    if query.ndim != 3 or key.ndim != 4 or value.shape != key.shape or output.shape != query.shape:
        raise ValueError("expected query [B,H,D], KV [P,S,Hkv,D] and matching output")
    batch, heads, dimension = query.shape
    pages, block_size, kv_heads, kv_dimension = key.shape
    if min(batch, heads, pages, kv_heads) <= 0:
        raise ValueError("prototype requires nonempty batch, heads and KV storage")
    if dimension not in (64, 128) or kv_dimension != dimension or heads % kv_heads:
        raise ValueError("prototype requires head dimension 64/128 and integral GQA")
    if lengths.shape != (batch,) or table.ndim != 2 or table.shape[0] != batch or table.shape[1] == 0:
        raise ValueError("invalid length or page-table descriptors")
    if lengths.dtype != torch.int32 or table.dtype != torch.int32:
        raise ValueError("lengths and page table must be int32")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != query.dtype for tensor in (key, value, output)
    ):
        raise ValueError("prototype requires matching FP16/BF16 tensors")
    if any(
        tensor.device != query.device or not tensor.is_contiguous() for tensor in (key, value, output, lengths, table)
    ):
        raise ValueError("prototype requires contiguous tensors on the query device")
    if not query.is_contiguous() or block_size != 128:
        raise ValueError("prototype requires contiguous query and block size 128")
    if splits not in (1, 4):
        raise ValueError("prototype supports one or four KV splits")
    if splits > 1 and (
        workspace is None
        or workspace.shape != (batch, heads, splits, dimension + 2)
        or workspace.dtype != torch.float32
        or workspace.device != query.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("split decode requires a preallocated FP32 workspace")
    _decode[(batch, heads, splits)](
        query,
        key,
        value,
        lengths,
        table,
        output,
        workspace,
        heads,
        kv_heads,
        dimension,
        block_size,
        table.shape[1],
        dimension**-0.5 if scale is None else scale,
        128,
        splits,
    )
    if splits > 1:
        _reduce[(batch, heads)](workspace, output, heads, dimension, splits)
