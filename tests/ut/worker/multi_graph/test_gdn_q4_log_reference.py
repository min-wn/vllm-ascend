# SPDX-License-Identifier: Apache-2.0
"""Independent dense-history checks for the compact FP32 Q4 state contract."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def oracle(monkeypatch):
    source = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_gdn_q4_log_reference.py"
    spec = importlib.util.spec_from_file_location("q4_log_reference_cpu_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def inputs(batch, *, seed=11, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    q, k = (torch.randn(batch, 4, 2, 8, generator=generator).to(dtype) for _ in range(2))
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(batch, 4, 6, 8, generator=generator).to(dtype)
    gamma = torch.rand(batch, 4, 6, generator=generator).to(dtype) * 0.3 + 0.65
    beta = torch.rand(batch, 4, 6, generator=generator).to(dtype) * 0.4
    return q, k, v, gamma, beta


def dense_step(base, q, k, v, gamma, beta):
    state = base.clone()
    q, k = (tensor.repeat_interleave(3, dim=2) for tensor in (q, k))
    states, outputs = [], []
    for token in range(4):
        decayed = state * gamma[:, token, :, None, None]
        prediction = (decayed * k[:, token, :, None, :]).sum(dim=-1)
        update = beta[:, token, :, None] * (v[:, token] - prediction)
        state = decayed + update[..., None] * k[:, token, :, None, :]
        states.append(state)
        outputs.append((state * q[:, token, :, None, :]).sum(dim=-1) * 8**-0.5)
    return torch.stack(states, dim=1), torch.stack(outputs, dim=1)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 2e-14), (torch.float32, 2e-6)])
def test_all_four_candidates_and_attention_match_independent_dense_history(oracle, dtype, tolerance):
    torch.manual_seed(47)
    base = torch.randn(4, 6, 8, 8, dtype=dtype) * 0.01
    cache = oracle.initialize_q4_log_cache(base, 2)
    dense = base[:, None].expand(-1, 4, -1, -1, -1).clone()
    for step in range(12):
        current = inputs(4, seed=step + 53, dtype=dtype)
        accepted = torch.tensor([(step + i) % 4 + 1 for i in range(4)])
        selected = dense[torch.arange(4), accepted - 1]
        wanted_states, wanted_output = dense_step(selected, *current)
        result = oracle.q4_log_step(cache, *current, accepted, torch.arange(4))
        cache = result.cache
        actual_states = oracle.reconstruct_q4_log_candidates(cache)
        torch.testing.assert_close(result.outputs, wanted_output, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual_states, wanted_states, rtol=tolerance, atol=tolerance)
        assert cache.base.dtype == cache.updates.dtype == dtype
        assert torch.isfinite(actual_states).all()
        # Advance the independent dense implementation, not the compact output.
        dense = wanted_states


def test_stable_request_ids_pause_resume_null_and_first_use(oracle):
    torch.manual_seed(7)
    base = torch.randn(7, 6, 8, 8) * 0.01
    cache = oracle.initialize_q4_log_cache(base, 2)
    history = base[:, None].expand(-1, 4, -1, -1, -1).clone()
    schedules = ([5, 1, -1, 3], [3, 6, 0], [1, -1, 5], [-1, -1], [6, 3, 1, 0, 5])
    for step, requests in enumerate(schedules):
        ids = torch.tensor(requests)
        accepted = (torch.arange(len(ids)) + step) % 4 + 1
        current = inputs(len(ids), seed=step + 81)
        old_cache = cache
        result = oracle.q4_log_step(cache, *current, accepted, ids)
        cache = result.cache
        active = ids >= 0
        selected = history[ids[active], accepted[active] - 1]
        wanted_states, wanted_outputs = dense_step(selected, *(tensor[active] for tensor in current))
        torch.testing.assert_close(result.outputs[active], wanted_outputs, rtol=2e-6, atol=2e-6)
        assert torch.equal(result.outputs[~active], torch.zeros_like(result.outputs[~active]))
        history[ids[active]] = wanted_states
        exported = oracle.reconstruct_q4_log_candidates(cache, ids)
        torch.testing.assert_close(exported[active], wanted_states, rtol=2e-6, atol=2e-6)
        assert not exported[~active].count_nonzero()
        paused = torch.tensor([request for request in range(7) if request not in requests])
        for name in ("base", "key", "gamma", "updates", "valid"):
            torch.testing.assert_close(getattr(cache, name)[paused], getattr(old_cache, name)[paused], rtol=0, atol=0)
        if active.any():
            first = ~old_cache.valid[ids[active]]
            torch.testing.assert_close(
                result.selected_previous_states[active][first], base[ids[active]][first], rtol=0, atol=0
            )


@pytest.mark.parametrize("accepted_count", [1, 2, 3, 4])
def test_committed_base_is_exact_ordered_export_of_previous_prefix(oracle, accepted_count):
    initial = oracle.initialize_q4_log_cache(torch.zeros(2, 6, 8, 8), 2)
    first = oracle.q4_log_step(initial, *inputs(2), torch.ones(2, dtype=torch.int64), torch.arange(2))
    candidates = oracle.reconstruct_q4_log_candidates(first.cache)
    second = oracle.q4_log_step(first.cache, *inputs(2, seed=12), torch.full((2,), accepted_count), torch.arange(2))
    torch.testing.assert_close(second.cache.base, candidates[:, accepted_count - 1], rtol=0, atol=0)
    # A call returns a new generation and does not mutate a previous snapshot.
    torch.testing.assert_close(initial.base, torch.zeros_like(initial.base), rtol=0, atol=0)


def test_zero_and_underflow_decay_are_finite_without_inverse_products(oracle):
    base = torch.randn(2, 6, 8, 8)
    cache = oracle.initialize_q4_log_cache(base, 2)
    q, k, v, gamma, beta = inputs(2)
    gamma[:, :2].zero_()
    gamma[:, 2:].fill_(1e-30)
    accepted, ids = torch.tensor([1, 4]), torch.arange(2)
    for _ in range(3):
        result = oracle.q4_log_step(cache, q, k, v, gamma, beta, accepted, ids)
        cache = result.cache
        for tensor in (result.outputs, result.initial_products, oracle.reconstruct_q4_log_candidates(cache)):
            assert torch.isfinite(tensor).all()


def test_finite_logs_and_outputs_do_not_imply_finite_logical_candidates(oracle):
    cache = oracle.initialize_q4_log_cache(torch.full((1, 1, 1, 1), 2e38), 1)
    q = k = v = torch.zeros(1, 4, 1, 1)
    gamma, beta = torch.full((1, 4, 1), 2.0), torch.zeros(1, 4, 1)
    result = oracle.q4_log_step(cache, q, k, v, gamma, beta, torch.tensor([1]), torch.tensor([0]))
    assert torch.isfinite(result.outputs).all()
    with pytest.raises(ValueError, match="nonfinite logical candidate"):
        oracle.reconstruct_q4_log_candidates(result.cache)


@pytest.mark.parametrize("invalid", ["duplicate", "oob", "accepted", "nonfinite", "state_precision", "gamma"])
def test_invalid_log_contract_is_rejected(oracle, invalid):
    base = torch.zeros(3, 6, 8, 8)
    if invalid == "state_precision":
        with pytest.raises(ValueError):
            oracle.initialize_q4_log_cache(base.bfloat16(), 2)
        return
    cache = oracle.initialize_q4_log_cache(base, 2)
    q, k, v, gamma, beta = inputs(2)
    ids, accepted = torch.tensor([0, 1]), torch.tensor([1, 4])
    if invalid == "duplicate":
        ids[:] = 1
    elif invalid == "oob":
        ids[0] = 3
    elif invalid == "accepted":
        accepted[0] = 0
    elif invalid == "nonfinite":
        q[0, 0, 0, 0] = float("nan")
    else:
        gamma[0, 0, 0] = -1
    with pytest.raises(ValueError):
        oracle.q4_log_step(cache, q, k, v, gamma, beta, accepted, ids)
