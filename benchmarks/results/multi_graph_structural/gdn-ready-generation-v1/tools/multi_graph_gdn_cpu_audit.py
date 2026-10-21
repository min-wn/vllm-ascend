# SPDX-License-Identifier: Apache-2.0
"""CPU array scans with the same finite and bitwise state-audit contracts."""

import numpy as np


def identical(torch, left, right):
    if left.device.type != "cpu" or right.device.type != "cpu":
        raise ValueError("array audit only accepts CPU tensors")
    return np.array_equal(left.contiguous().view(torch.uint8).numpy(), right.contiguous().view(torch.uint8).numpy())


def finite(torch, tensor):
    if tensor.device.type != "cpu":
        raise ValueError("array audit only accepts CPU tensors")
    if tensor.dtype == torch.bfloat16:
        bits = tensor.contiguous().view(torch.uint16).numpy()
        return bool(np.all((bits & 0x7F80) != 0x7F80))
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(np.isfinite(tensor.numpy()).all())
    return True


def zero(torch, tensor):
    if tensor.device.type != "cpu":
        raise ValueError("array audit only accepts CPU tensors")
    if tensor.dtype == torch.bfloat16:
        bits = tensor.contiguous().view(torch.uint16).numpy()
        return not np.any(bits & 0x7FFF)
    return not np.any(tensor.numpy())
