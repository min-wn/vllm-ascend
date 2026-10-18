# SPDX-License-Identifier: Apache-2.0
"""CPU algebra and rounding checks for the specialized four-token GDN formula."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def block_reference(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "gdn_q4_block_cpu_test", directory / "multi_graph_gdn_q4_block_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def inputs(seed=31, *, batch=3, key_heads=1, value_heads=3, width_k=16, width_v=12, state_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)

    def random(*shape):
        return torch.randn(*shape, generator=generator)

    base = (random(batch, value_heads, width_v, width_k) * 0.02).to(state_dtype)
    q, k = (torch.nn.functional.normalize(random(batch, 4, key_heads, width_k), dim=-1).bfloat16() for _ in range(2))
    value = (random(batch, 4, value_heads, width_v) * 0.1).bfloat16()
    gamma = torch.exp(-torch.rand(batch, 4, value_heads, generator=generator) * 0.4)
    beta = torch.sigmoid(random(batch, 4, value_heads)).bfloat16().float()
    return base, q, k, value, gamma, beta


def sequential(base, q, k, value, gamma, beta, *, dtype, scale):
    ratio = base.shape[1] // q.shape[2]
    q, k = (tensor.to(dtype).repeat_interleave(ratio, dim=2) for tensor in (q, k))
    state = base.to(dtype)
    states, outputs, updates = [], [], []
    for token in range(4):
        state = state * gamma[:, token].to(dtype)[:, :, None, None]
        prediction = (state * k[:, token, :, None, :]).sum(-1)
        update = (value[:, token].to(dtype) - prediction) * beta[:, token].to(dtype)[..., None]
        state = state + update[..., None] * k[:, token, :, None, :]
        outputs.append((state * (q[:, token] * scale)[:, :, None, :]).sum(-1))
        states.append(state.clone())
        updates.append(update)
    return torch.stack(states, dim=1), torch.stack(outputs, dim=1), torch.stack(updates, dim=1)


@pytest.mark.parametrize("key_heads,value_heads", [(1, 1), (1, 3), (2, 6)])
@pytest.mark.parametrize("scale", [1.0, 0.25])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_q4_formula_matches_fp64_recurrence_and_all_prefixes(
    block_reference, key_heads, value_heads, scale, state_dtype
):
    data = inputs(key_heads=key_heads, value_heads=value_heads, state_dtype=state_dtype)
    result = block_reference.q4_block_step(*data, scale=scale, accumulation_dtype=torch.float64)
    states, output, updates = sequential(*data, dtype=torch.float64, scale=scale)
    torch.testing.assert_close(result.unrounded_states, states, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.outputs, output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.updates, updates, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.states, states.to(state_dtype), rtol=0, atol=0)
    assert torch.equal(torch.diagonal(result.lower, dim1=-2, dim2=-1), torch.ones(3, value_heads, 4))
    assert not torch.triu(result.lower, diagonal=1).count_nonzero()
    # Check the actual unit-triangular equations independently of substitution.
    _, _, _, value, gamma, beta = data
    alpha = gamma.double().cumprod(1)
    h0k = result.initial_products[..., :4].permute(0, 3, 1, 2)
    target = beta.double()[..., None] * (value.double() - alpha[..., None] * h0k)
    observed = torch.matmul(result.lower, result.updates.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    torch.testing.assert_close(observed, target, rtol=1e-12, atol=1e-12)


def test_zero_and_underflowed_decay_need_no_inverse(block_reference):
    data = list(inputs())
    data[4][:, 0] = 0
    data[4][:, 1] = torch.finfo(torch.float32).tiny
    result = block_reference.q4_block_step(*data, accumulation_dtype=torch.float64)
    states, output, _ = sequential(*data, dtype=torch.float64, scale=16**-0.5)
    assert torch.isfinite(result.unrounded_states).all() and torch.isfinite(result.outputs).all()
    torch.testing.assert_close(result.unrounded_states, states, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.outputs, output, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed,width", [(31, 16), (53, 128), (992, 16)])
def test_fp32_closed_loop_keeps_bf16_boundaries_and_small_recorded_drift(block_reference, seed, width):
    from multi_graph_gdn_compact_reference import dense_step

    data = list(inputs(seed, width_k=width, width_v=width))
    reference_base, block_base = data[0].clone(), data[0].clone()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for step in range(8):
            data = list(inputs(seed + step, width_k=width, width_v=width))
            data[0] = block_base
            observed = block_reference.q4_block_step(*data)
            _, q, k, value, gamma, beta = data
            q = q.float().repeat_interleave(3, dim=2) * width**-0.5
            k = k.float().repeat_interleave(3, dim=2)
            states, output = dense_step(reference_base, q, k, value.float(), gamma, beta)
            for actual, wanted in ((observed.states.float(), states.float()), (observed.outputs, output)):
                relative = (actual.double() - wanted.double()).norm() / wanted.double().norm().clamp_min(1e-30)
                assert relative < 1e-3
                assert torch.isfinite(actual).all()
            accepted = torch.tensor([(1, 2, 4)[(step + request) % 3] for request in range(3)])
            block_base = observed.states[torch.arange(3), accepted - 1].clone()
            reference_base = states[torch.arange(3), accepted - 1].clone()
            assert block_base.dtype == reference_base.dtype == torch.bfloat16
    finally:
        torch.set_num_threads(previous_threads)


def test_reference_does_not_mutate_or_alias_inputs(block_reference):
    data = inputs()
    before = tuple(tensor.clone() for tensor in data)
    result = block_reference.q4_block_step(*data)
    for actual, wanted in zip(data, before, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    input_storage = {tensor.untyped_storage().data_ptr() for tensor in data}
    for tensor in (
        result.states,
        result.unrounded_states,
        result.outputs,
        result.updates,
        result.lower,
        result.initial_products,
    ):
        assert tensor.untyped_storage().data_ptr() not in input_storage
    expected = result.states.clone()
    data[0].fill_(9)
    torch.testing.assert_close(result.states, expected, rtol=0, atol=0)


def test_fp32_closed_loop_preserves_candidate_precision(block_reference):
    data = list(inputs(state_dtype=torch.float32))
    reference_base, block_base = data[0].clone(), data[0].clone()
    for step in range(8):
        data = list(inputs(31 + step, state_dtype=torch.float32))
        data[0] = block_base
        observed = block_reference.q4_block_step(*data)
        states, output, _ = sequential(reference_base, *data[1:], dtype=torch.float32, scale=data[1].shape[-1] ** -0.5)
        assert observed.states.dtype == torch.float32
        assert (observed.states != observed.states.bfloat16().float()).any()
        torch.testing.assert_close(observed.states, states, rtol=1e-5, atol=3e-8)
        torch.testing.assert_close(observed.outputs, output, rtol=1e-5, atol=3e-8)
        accepted = (torch.arange(3) + step) % 4
        block_base = observed.states[torch.arange(3), accepted].clone()
        reference_base = states[torch.arange(3), accepted].clone()


@pytest.mark.parametrize(
    "invalid",
    [
        "base_dtype",
        "query_tokens",
        "key_heads",
        "value_width",
        "gamma_dtype",
        "negative_gamma",
        "nan_beta",
        "accumulation",
        "scale",
    ],
)
def test_block_rejects_invalid_contract(block_reference, invalid):
    data, kwargs = list(inputs()), {}
    if invalid == "base_dtype":
        data[0] = data[0].half()
    elif invalid == "query_tokens":
        data[1] = data[1][:, :3]
    elif invalid == "key_heads":
        data[1] = data[1].repeat(1, 1, 2, 1)
        data[2] = data[2].repeat(1, 1, 2, 1)
    elif invalid == "value_width":
        data[3] = data[3][..., :-1]
    elif invalid == "gamma_dtype":
        data[4] = data[4].bfloat16()
    elif invalid == "negative_gamma":
        data[4][0, 0, 0] = -1
    elif invalid == "nan_beta":
        data[5][0, 0, 0] = float("nan")
    elif invalid == "accumulation":
        kwargs["accumulation_dtype"] = torch.bfloat16
    else:
        kwargs["scale"] = float("inf")
    with pytest.raises(ValueError):
        block_reference.q4_block_step(*data, **kwargs)
