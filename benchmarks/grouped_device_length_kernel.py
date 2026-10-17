"""Experimental vector decode reference for fixed-address device metadata.

This kernel is a qualification prototype, not an attention backend replacement.
It supports dense paged GQA, one query per request and no sliding window.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def grouped_decode(
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
    head = tl.program_id(1) * 4 + tl.arange(0, 4)
    split = tl.program_id(2)
    kv_head = tl.program_id(1)
    dimensions = tl.arange(0, DIM)
    positions = tl.arange(0, TILE)
    q = tl.load(query + request * HEADS * DIM + head[:, None] * DIM + dimensions[None, :]).to(tl.float32)
    length = tl.load(lengths + request)
    maximum = tl.full((4,), float("-inf"), tl.float32)
    denominator = tl.full((4,), 0.0, tl.float32)
    accumulator = tl.full((4, DIM), 0.0, tl.float32)
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
            scores = tl.sum(k[None, :, :] * q[:, None, :], 2) * SCALE
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            adjustment = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum[:, None])
            v = tl.load(value + offsets, valid[:, None], other=0).to(tl.float32)
            accumulator = accumulator * adjustment[:, None] + tl.sum(v[None, :, :] * probabilities[:, :, None], 1)
            denominator = denominator * adjustment + tl.sum(probabilities, 1)
            maximum = next_maximum
    if SPLITS == 1:
        result = accumulator / tl.maximum(denominator[:, None], 1.0e-20)
        tl.store(output + request * HEADS * DIM + head[:, None] * DIM + dimensions[None, :], result)


def launch(query, key, value, lengths, table, output, *, splits=1, workspace=None):
    if splits != 1:
        raise ValueError("Grouped probe only supports one KV partition")
    grouped_decode[(query.shape[0], 8, 1)](
        query, key, value, lengths, table, output, None, 32, 8, 128, 128, table.shape[1], 128**-0.5, 32, 1
    )
