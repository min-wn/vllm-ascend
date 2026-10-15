from __future__ import annotations

import math

import torch


def contiguous_dense_kv_views(raw: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition an unused dense allocation into contiguous K and V storage."""
    if len(shape) != 5 or shape[0] != 2 or any(size <= 0 for size in shape):
        raise ValueError("contiguous KV requires a nonempty standard dense KV shape")
    if not raw.is_contiguous() or raw.numel() != math.prod(shape):
        raise ValueError("contiguous KV requires an exact unpadded contiguous allocation")
    cache = raw.view(shape)
    return cache[0], cache[1]
