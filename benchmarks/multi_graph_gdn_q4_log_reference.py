# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for FP32 Q4 state logs, not a device implementation.

One dense base and four (decay, key, update) logs represent all four candidate
states. The next accepted candidate is reconstructed once, in token order,
for the new base. Its eight new projections instead use the old base product
and small log corrections. This reassociates FP32 arithmetic: it is not an
exact replacement for dense FP32 reduction. FP64 caches test the algebraic
identity separately, without proposing a model precision change.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Q4LogCache:
    base: torch.Tensor
    key: torch.Tensor
    gamma: torch.Tensor
    updates: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class Q4LogResult:
    cache: Q4LogCache
    outputs: torch.Tensor
    initial_products: torch.Tensor
    selected_previous_states: torch.Tensor
    updates: torch.Tensor
    lower: torch.Tensor


def _finite_cpu(tensors):
    return all(
        isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and torch.isfinite(tensor).all()
        for tensor in tensors
    )


def _validate_cache(cache):
    if (
        not isinstance(cache, Q4LogCache)
        or cache.base.ndim != 4
        or min(cache.base.shape) < 1
        or cache.base.dtype not in (torch.float32, torch.float64)
        or cache.key.ndim != 4
        or min(cache.key.shape) < 1
    ):
        raise ValueError("nonempty FP32 cache required; FP64 is reserved for CPU identity checks")
    slots, heads, width_v, width_k = cache.base.shape
    key_heads = cache.key.shape[2]
    shapes = (
        (cache.key, (slots, 4, key_heads, width_k)),
        (cache.gamma, (slots, 4, heads)),
        (cache.updates, (slots, 4, heads, width_v)),
    )
    if (
        heads % key_heads
        or any(tensor.shape != shape or tensor.dtype != cache.base.dtype for tensor, shape in shapes)
        or cache.valid.shape != (slots,)
        or cache.valid.dtype != torch.bool
        or not _finite_cpu((cache.base, cache.key, cache.gamma, cache.updates, cache.valid))
        or (cache.gamma < 0).any()
    ):
        raise ValueError("cache dimensions, state/log precision, finite values or decay disagree")
    return slots, heads, key_heads, width_v, width_k


def initialize_q4_log_cache(base, key_heads):
    """Create first-use state; no pending logs means accepted count is ignored."""
    if (
        not isinstance(base, torch.Tensor)
        or base.ndim != 4
        or min(base.shape) < 1
        or base.dtype not in (torch.float32, torch.float64)
        or not _finite_cpu((base,))
        or isinstance(key_heads, bool)
        or not isinstance(key_heads, int)
        or key_heads < 1
        or base.shape[1] % key_heads
    ):
        raise ValueError("finite FP32/FP64 CPU base and a dividing positive key-head count required")
    slots, heads, width_v, width_k = base.shape
    return Q4LogCache(
        base=base.clone(),
        key=base.new_zeros(slots, 4, key_heads, width_k),
        gamma=base.new_ones(slots, 4, heads),
        updates=base.new_zeros(slots, 4, heads, width_v),
        valid=torch.zeros(slots, dtype=torch.bool),
    )


def _request_ids(request_ids, batch, slots):
    if (
        not isinstance(request_ids, torch.Tensor)
        or request_ids.device.type != "cpu"
        or request_ids.dtype not in (torch.int32, torch.int64)
        or request_ids.shape != (batch,)
        or ((request_ids < -1) | (request_ids >= slots)).any()
    ):
        raise ValueError("request IDs require a CPU integer vector in [-1, slots); -1 is null")
    active = request_ids >= 0
    indices = request_ids[active].long()
    if len(indices.unique()) != len(indices):
        raise ValueError("active request IDs must be unique within a step")
    return active, indices


def _ordered_selected(cache, indices, accepted):
    state = cache.base.index_select(0, indices).clone()
    ratio = state.shape[1] // cache.key.shape[2]
    keys = cache.key.index_select(0, indices).repeat_interleave(ratio, dim=2)
    gamma, updates = cache.gamma[indices], cache.updates[indices]
    valid = cache.valid[indices]
    for token in range(4):
        candidate = state * gamma[:, token, :, None, None]
        candidate = candidate + updates[:, token, :, :, None] * keys[:, token, :, None, :]
        state = torch.where((valid & (accepted > token))[:, None, None, None], candidate, state)
    return state


@torch.no_grad()
def reconstruct_q4_log_candidates(cache, request_ids=None):
    """Materialize all logical prefixes for diagnostics, outside compact storage."""
    slots, heads, _, width_v, width_k = _validate_cache(cache)
    request_ids = torch.arange(slots) if request_ids is None else request_ids
    active, indices = _request_ids(request_ids, len(request_ids), slots)
    candidates = cache.base.new_zeros(len(request_ids), 4, heads, width_v, width_k)
    for prefix in range(4):
        accepted = torch.full((len(indices),), prefix + 1, dtype=torch.int64)
        candidates[active, prefix] = _ordered_selected(cache, indices, accepted)
    return candidates


def _solve_from_products(products, query, key, value, gamma, beta):
    """Four-step triangular solve using initial projections rather than dense H."""
    h0k, h0q = (tensor.permute(0, 3, 1, 2) for tensor in products.split(4, dim=-1))
    kk = torch.einsum("bihk,bjhk->bhij", key, key)
    kq = torch.einsum("bjhk,bihk->bhij", key, query)
    alpha, decay_rows = [], []
    for token in range(4):
        alpha.append(gamma[:, token] if token == 0 else alpha[-1] * gamma[:, token])
        row = []
        for previous in range(4):
            decay = torch.ones_like(gamma[:, token])
            for offset in range(previous + 1, token + 1):
                decay = decay * gamma[:, offset]
            row.append(decay if previous <= token else torch.zeros_like(decay))
        decay_rows.append(torch.stack(row, dim=-1))
    alpha = torch.stack(alpha, dim=1)
    decay = torch.stack(decay_rows, dim=-2)
    lower = torch.eye(4, dtype=products.dtype)[None, None] + torch.tril(
        beta.permute(0, 2, 1)[..., :, None] * decay * kk, diagonal=-1
    )
    target = beta[..., None] * (value - alpha[..., None] * h0k)
    updates, outputs = [], []
    for token in range(4):
        update = target[:, token]
        for previous in range(token):
            update = update - lower[:, :, token, previous, None] * updates[previous]
        updates.append(update)
        output = alpha[:, token, :, None] * h0q[:, token]
        for previous in range(token + 1):
            output = output + decay[:, :, token, previous, None] * updates[previous] * kq[:, :, token, previous, None]
        outputs.append(output)
    return torch.stack(updates, dim=1), torch.stack(outputs, dim=1), lower


@torch.no_grad()
def q4_log_step(cache, query, key, value, gamma, beta, accepted, request_ids, *, scale=None):
    """Commit one previous accepted state and retain the current four small logs.

    Input rows map to stable request IDs; omitted IDs pause, -1 rows are null.
    The FP32 production oracle never converts state or updates to BF16. Using
    an explicitly FP64 cache instead selects the CPU real-arithmetic check.
    """
    slots, heads, key_heads, width_v, width_k = _validate_cache(cache)
    if not isinstance(query, torch.Tensor) or query.ndim != 4 or min(query.shape) < 1:
        raise ValueError("query requires [batch, four tokens, key heads, key width]")
    batch = len(query)
    shapes = (
        (query, (batch, 4, key_heads, width_k)),
        (key, (batch, 4, key_heads, width_k)),
        (value, (batch, 4, heads, width_v)),
        (gamma, (batch, 4, heads)),
        (beta, (batch, 4, heads)),
    )
    allowed = (torch.bfloat16, torch.float32) + ((torch.float64,) if cache.base.dtype == torch.float64 else ())
    if (
        any(
            not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype not in allowed
            for tensor, shape in shapes
        )
        or not _finite_cpu((query, key, value, gamma, beta))
        or gamma.dtype not in (torch.float32, cache.base.dtype)
        or beta.dtype not in (torch.float32, cache.base.dtype)
        or (gamma < 0).any()
    ):
        raise ValueError("finite Q4 inputs, grouped-head dimensions and FP32 gamma/beta required")
    if (
        not isinstance(accepted, torch.Tensor)
        or accepted.device.type != "cpu"
        or accepted.dtype not in (torch.int32, torch.int64)
        or accepted.shape != (batch,)
        or ((accepted < 1) | (accepted > 4)).any()
    ):
        raise ValueError("accepted counts require a CPU integer vector with values 1 through 4")
    active, indices = _request_ids(request_ids, batch, slots)
    scale = width_k**-0.5 if scale is None else scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not torch.isfinite(torch.tensor(scale)):
        raise ValueError("query scale must be finite")
    dtype, ratio = cache.base.dtype, heads // key_heads
    q = query[active].to(dtype).repeat_interleave(ratio, dim=2) * scale
    k = key[active].to(dtype).repeat_interleave(ratio, dim=2)
    v, g, b = (tensor[active].to(dtype) for tensor in (value, gamma, beta))
    counts = accepted[active]
    selected = _ordered_selected(cache, indices, counts)
    rhs = torch.cat((k, q), dim=1).permute(0, 2, 3, 1)
    products = cache.base[indices] @ rhs
    previous_keys = cache.key[indices].repeat_interleave(ratio, dim=2)
    # Projection corrections avoid another dense matmul on the reconstructed
    # state. They are algebraically equal, but their FP32 reductions differ.
    for token in range(4):
        key_products = torch.einsum("bhk,bhkc->bhc", previous_keys[:, token], rhs)
        corrected = products * cache.gamma[indices, token, :, None, None]
        corrected = corrected + cache.updates[indices, token, :, :, None] * key_products[:, :, None, :]
        mask = cache.valid[indices] & (counts > token)
        products = torch.where(mask[:, None, None, None], corrected, products)
    updates, outputs, lower = _solve_from_products(products, q, k, v, g, b)
    if not _finite_cpu((selected, products, updates, outputs, lower)):
        raise ValueError("nonfinite Q4 log intermediate or output")
    next_cache = Q4LogCache(
        *(tensor.clone() for tensor in (cache.base, cache.key, cache.gamma, cache.updates, cache.valid))
    )
    next_cache.base[indices] = selected
    next_cache.key[indices] = key[active].to(dtype)
    next_cache.gamma[indices] = g
    next_cache.updates[indices] = updates
    next_cache.valid[indices] = True
    result_outputs = cache.base.new_zeros(batch, 4, heads, width_v)
    result_products = cache.base.new_zeros(batch, heads, width_v, 8)
    result_selected = cache.base.new_zeros(batch, heads, width_v, width_k)
    result_updates = cache.base.new_zeros(batch, 4, heads, width_v)
    result_lower = cache.base.new_zeros(batch, heads, 4, 4)
    for target, source in (
        (result_outputs, outputs),
        (result_products, products),
        (result_selected, selected),
        (result_updates, updates),
        (result_lower, lower),
    ):
        target[active] = source
    return Q4LogResult(next_cache, result_outputs, result_products, result_selected, result_updates, result_lower)
