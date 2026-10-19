# SPDX-License-Identifier: Apache-2.0
"""CPU Q4 block formula; a mathematical oracle, not a device implementation.

Reassociating decay, reductions and outer products changes FP32 rounding.
Candidate storage and the next-step load preserve the input BF16 or FP32
state precision. FP64 mode checks the real-arithmetic identity; it is not a
proposed device precision.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Q4BlockResult:
    states: torch.Tensor
    unrounded_states: torch.Tensor
    outputs: torch.Tensor
    updates: torch.Tensor
    lower: torch.Tensor
    initial_products: torch.Tensor


def _validate(base, query, key, value, gamma, beta, accumulation_dtype):
    if (
        not isinstance(base, torch.Tensor)
        or base.ndim != 4
        or min(base.shape) < 1
        or base.dtype not in (torch.bfloat16, torch.float32)
        or accumulation_dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("expected a nonempty BF16/FP32 base and FP32/FP64 CPU accumulation")
    batch, heads, width_v, width_k = base.shape
    if not isinstance(query, torch.Tensor) or query.ndim != 4 or min(query.shape) < 1:
        raise ValueError("query requires [requests, four tokens, key heads, key width]")
    key_heads = query.shape[2]
    shapes = (
        (query, (batch, 4, key_heads, width_k)),
        (key, (batch, 4, key_heads, width_k)),
        (value, (batch, 4, heads, width_v)),
        (gamma, (batch, 4, heads)),
        (beta, (batch, 4, heads)),
    )
    if heads % key_heads or any(
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype not in (torch.bfloat16, torch.float32)
        for tensor, shape in shapes
    ):
        raise ValueError("Q4 dimensions, input precision or grouped-head ratio disagree")
    if gamma.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("gamma and beta must retain FP32 operands")
    if (
        any(
            tensor.device.type != "cpu" or not torch.isfinite(tensor).all()
            for tensor in (base, query, key, value, gamma, beta)
        )
        or (gamma < 0).any()
    ):
        raise ValueError("finite CPU inputs and nonnegative decay required")
    return heads // key_heads


@torch.no_grad()
def q4_block_step(base, query, key, value, gamma, beta, *, scale=None, accumulation_dtype=torch.float32):
    """Solve four rank-one updates using initial-state projections and a 4x4 system.

    base is [B,Hv,V,K], Q/K are [B,4,Hk,K], V is [B,4,Hv,V]. The
    four candidates retain base.dtype and must be selected/loaded at the
    next step. BF16 rounding is required only for a BF16 cache; an FP32 cache
    must never be rounded to BF16 at the step boundary.
    """
    ratio = _validate(base, query, key, value, gamma, beta, accumulation_dtype)
    width_k = base.shape[-1]
    scale = width_k**-0.5 if scale is None else scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not torch.isfinite(torch.tensor(scale)):
        raise ValueError("query scale must be a finite scalar")
    q = query.to(accumulation_dtype).repeat_interleave(ratio, dim=2) * scale
    k = key.to(accumulation_dtype).repeat_interleave(ratio, dim=2)
    v, g, b = (tensor.to(accumulation_dtype) for tensor in (value, gamma, beta))
    h0 = base.to(accumulation_dtype)
    # Eight skinny right-hand sides share the initial matrix. No transition
    # divides by cumulative decay, so zero/underflowed gamma is well-defined.
    rhs = torch.cat((k, q), dim=1).permute(0, 2, 3, 1)
    products = torch.matmul(h0, rhs)
    h0k, h0q = (tensor.permute(0, 3, 1, 2) for tensor in products.split(4, dim=-1))
    kk = torch.einsum("bihk,bjhk->bhij", k, k)
    kq = torch.einsum("bjhk,bihk->bhij", k, q)
    alpha, decay_rows = [], []
    for token in range(4):
        alpha.append(g[:, token] if token == 0 else alpha[-1] * g[:, token])
        row = []
        for previous in range(4):
            product = torch.ones_like(g[:, token])
            for offset in range(previous + 1, token + 1):
                product = product * g[:, offset]
            row.append(product if previous <= token else torch.zeros_like(product))
        decay_rows.append(torch.stack(row, dim=-1))
    alpha = torch.stack(alpha, dim=1)
    decay = torch.stack(decay_rows, dim=-2)
    lower = torch.eye(4, dtype=accumulation_dtype)[None, None] + torch.tril(
        b.permute(0, 2, 1)[..., :, None] * decay * kk, diagonal=-1
    )
    target = b[..., None] * (v - alpha[..., None] * h0k)
    updates = []
    for token in range(4):
        update = target[:, token]
        for previous in range(token):
            update = update - lower[:, :, token, previous, None] * updates[previous]
        updates.append(update)
    states, outputs = [], []
    for token in range(4):
        state = alpha[:, token, :, None, None] * h0
        output = alpha[:, token, :, None] * h0q[:, token]
        for previous in range(token + 1):
            coefficient = decay[:, :, token, previous]
            state = state + coefficient[:, :, None, None] * updates[previous][..., None] * k[:, previous, :, None, :]
            output = output + coefficient[..., None] * updates[previous] * kq[:, :, token, previous, None]
        states.append(state)
        outputs.append(output)
    unrounded = torch.stack(states, dim=1)
    return Q4BlockResult(
        states=unrounded.to(base.dtype),
        unrounded_states=unrounded,
        outputs=torch.stack(outputs, dim=1),
        updates=torch.stack(updates, dim=1),
        lower=lower,
        initial_products=products,
    )
