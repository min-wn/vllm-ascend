# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for an ordered Q4 GDN update log with BF16 step boundaries.

This is a representation/correctness oracle, not a performance implementation.
CPU exp/reduction and the explicit FP32 FMA simulation do not certify equality
with the device kernel's ExpMasked, AddFold reduction, or MulAddDst behavior.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompactState:
    base: torch.Tensor
    gamma: torch.Tensor
    keys: torch.Tensor
    updates: torch.Tensor
    valid: torch.Tensor


def _tensor(value, shape, dtype, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(f"{name} has an incompatible shape or dtype")
    if value.device.type != "cpu" or not torch.isfinite(value).all():
        raise ValueError(f"{name} requires finite CPU values")


def _base_shape(base):
    if not isinstance(base, torch.Tensor) or base.ndim != 4 or min(base.shape) <= 0:
        raise ValueError("base requires nonempty [requests, heads, value, key] dimensions")
    _tensor(base, base.shape, torch.bfloat16, "base")
    return tuple(base.shape)


def _step_shapes(base, query, key, value, gamma, beta):
    batch, heads, value_width, key_width = _base_shape(base)
    for name, tensor, shape in (
        ("query", query, (batch, 4, heads, key_width)),
        ("key", key, (batch, 4, heads, key_width)),
        ("value", value, (batch, 4, heads, value_width)),
        ("gamma", gamma, (batch, 4, heads)),
        ("beta", beta, (batch, 4, heads)),
    ):
        _tensor(tensor, shape, torch.float32, name)
    if (gamma < 0).any():
        raise ValueError("gamma must be a nonnegative decay")


def _state_shapes(state):
    if not isinstance(state, CompactState):
        raise ValueError("expected a CompactState")
    batch, heads, value_width, key_width = _base_shape(state.base)
    for name, tensor, shape, dtype in (
        ("gamma", state.gamma, (batch, 4, heads), torch.float32),
        ("keys", state.keys, (batch, 4, heads, key_width), torch.float32),
        ("updates", state.updates, (batch, 4, heads, value_width), torch.float32),
        ("valid", state.valid, (batch,), torch.bool),
    ):
        _tensor(tensor, shape, dtype, name)
    if (state.gamma < 0).any():
        raise ValueError("gamma must be a nonnegative decay")
    return batch


def _fma32(destination, left, right):
    # FP64 retains FP32 products before one final FP32 conversion. This is an
    # explicit CPU reference choice, not a universal device FMA equivalence.
    return (destination.double() + left.double() * right.double()).float()


def _apply_update(state, gamma, key, update):
    decayed = state * gamma[:, :, None, None]
    return _fma32(decayed, update[:, :, :, None], key[:, :, None, :])


@torch.no_grad()
def initial_compact_state(base):
    batch, heads, value_width, key_width = _base_shape(base)
    return CompactState(
        base.clone(),
        torch.ones(batch, 4, heads, dtype=torch.float32),
        torch.zeros(batch, 4, heads, key_width, dtype=torch.float32),
        torch.zeros(batch, 4, heads, value_width, dtype=torch.float32),
        torch.zeros(batch, dtype=torch.bool),
    )


@torch.no_grad()
def dense_step(base, query, key, value, gamma, beta):
    """Return four BF16 candidates and FP32 outputs, retaining FP32 within Q4."""
    _step_shapes(base, query, key, value, gamma, beta)
    state = base.float()
    candidates, outputs = [], []
    for token in range(4):
        decayed = state * gamma[:, token, :, None, None]
        prediction = (decayed * key[:, token, :, None, :]).sum(-1)
        update = (value[:, token] - prediction) * beta[:, token, :, None]
        state = _fma32(decayed, update[:, :, :, None], key[:, token, :, None, :])
        candidates.append(state.to(torch.bfloat16))
        outputs.append((state * query[:, token, :, None, :]).sum(-1))
    return torch.stack(candidates, dim=1), torch.stack(outputs, dim=1)


@torch.no_grad()
def emit_compact_step(base, query, key, value, gamma, beta):
    """Emit exact CPU update operands without materializing dense candidates."""
    _step_shapes(base, query, key, value, gamma, beta)
    state = base.float()
    updates, outputs = [], []
    for token in range(4):
        decayed = state * gamma[:, token, :, None, None]
        prediction = (decayed * key[:, token, :, None, :]).sum(-1)
        update = (value[:, token] - prediction) * beta[:, token, :, None]
        state = _fma32(decayed, update[:, :, :, None], key[:, token, :, None, :])
        updates.append(update)
        outputs.append((state * query[:, token, :, None, :]).sum(-1))
    log = CompactState(
        base.clone(),
        gamma.clone(),
        key.clone(),
        torch.stack(updates, dim=1),
        torch.ones(base.shape[0], dtype=torch.bool),
    )
    return log, torch.stack(outputs, dim=1)


@torch.no_grad()
def restore_accepted(state, accepted):
    """Restore per-request accepted prefix, rounding at the next BF16 load."""
    batch = _state_shapes(state)
    if isinstance(accepted, int) and not isinstance(accepted, bool):
        accepted = torch.full((batch,), accepted, dtype=torch.int64)
    if (
        not isinstance(accepted, torch.Tensor)
        or accepted.device.type != "cpu"
        or accepted.dtype not in (torch.int32, torch.int64)
        or tuple(accepted.shape) != (batch,)
        or (accepted < 1).any()
        or (accepted > 4).any()
    ):
        raise ValueError("accepted must contain one integer prefix in [1, 4] per request")
    current = state.base.float()
    for token in range(4):
        updated = _apply_update(current, state.gamma[:, token], state.keys[:, token], state.updates[:, token])
        choose = state.valid & (accepted > token)
        current = torch.where(choose[:, None, None, None], updated, current)
    return current.to(torch.bfloat16)


@torch.no_grad()
def export_logical_states(state):
    """Materialize every logical BF16 candidate; this export is never free."""
    _state_shapes(state)
    current, candidates = state.base.float(), []
    for token in range(4):
        updated = _apply_update(current, state.gamma[:, token], state.keys[:, token], state.updates[:, token])
        current = torch.where(state.valid[:, None, None, None], updated, current)
        candidates.append(current.to(torch.bfloat16))
    return torch.stack(candidates, dim=1)


@torch.no_grad()
def qualify_closed_loop(seed=41, *, steps=8, requests=7, heads=2, key_width=128, value_width=128):
    """Reproduce ordered-log equality through mixed prefixes and request churn."""
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (steps, requests, heads, key_width, value_width)
    ):
        raise ValueError("qualification dimensions and step count must be positive integers")
    generator = torch.Generator().manual_seed(seed)

    def random(*shape):
        return torch.randn(*shape, generator=generator)

    base = (random(requests, heads, value_width, key_width) * 0.02).to(torch.bfloat16)
    state = initial_compact_state(base)
    dense = base[:, None].expand(-1, 4, -1, -1, -1).clone()
    records, candidate_elements, output_elements = [], 0, 0
    for step in range(steps):
        active = [request for request in range(requests) if request != 0 and (request + step) % 5 != 0]
        if step % 2:
            active.reverse()
        indices = torch.tensor(active, dtype=torch.int64)
        if not active:
            continue
        accepted = torch.tensor([(1, 2, 4)[(request + step) % 3] for request in active])
        selected = CompactState(
            *(
                tensor.index_select(0, indices)
                for tensor in (state.base, state.gamma, state.keys, state.updates, state.valid)
            )
        )
        before = tuple(tensor.clone() for tensor in (state.base, state.gamma, state.keys, state.updates, state.valid))
        before_dense = dense.clone()
        dense_base = dense[indices, accepted - 1]
        compact_base = restore_accepted(selected, accepted)
        if not torch.equal(compact_base, dense_base):
            raise AssertionError("accepted-prefix BF16 state differs")
        q, k = (
            torch.nn.functional.normalize(random(len(active), 4, heads, key_width), dim=-1).to(torch.bfloat16).float()
            for _ in range(2)
        )
        v = (random(len(active), 4, heads, value_width) * 0.1).to(torch.bfloat16).float()
        gamma = torch.exp(-torch.rand(len(active), 4, heads, generator=generator) * 0.4)
        beta = torch.sigmoid(random(len(active), 4, heads)).to(torch.bfloat16).float()
        expected_states, expected_outputs = dense_step(dense_base, q, k, v, gamma, beta)
        updated, actual_outputs = emit_compact_step(compact_base, q, k, v, gamma, beta)
        if not torch.equal(expected_outputs, actual_outputs):
            raise AssertionError("current Q4 output differs")
        for destination, source in zip(
            (state.base, state.gamma, state.keys, state.updates, state.valid),
            (updated.base, updated.gamma, updated.keys, updated.updates, updated.valid),
            strict=True,
        ):
            destination.index_copy_(0, indices, source)
        dense.index_copy_(0, indices, expected_states)
        exported = export_logical_states(state)
        if not torch.equal(exported, dense):
            raise AssertionError("one of four materialized logical candidates differs")
        inactive = [request for request in range(requests) if request not in active]
        for actual, previous in zip(
            (state.base, state.gamma, state.keys, state.updates, state.valid), before, strict=True
        ):
            if not torch.equal(actual[inactive], previous[inactive]):
                raise AssertionError("inactive/null compact state changed")
        if not torch.equal(dense[inactive], before_dense[inactive]):
            raise AssertionError("inactive/null dense state changed")
        candidate_elements += exported.numel()
        output_elements += actual_outputs.numel()
        records.append(
            dict(step=step, active_request_ids=active, accepted=accepted.tolist(), inactive_request_ids=inactive)
        )
    return dict(
        status="passed_under_identical_CPU_FP32_arithmetic",
        seed=seed,
        steps=steps,
        candidate_elements_checked=candidate_elements,
        output_elements_checked=output_elements,
        records=records,
        scope="CPU representation oracle; not NPU precision, kernel latency, or full-model quality qualification",
    )
