# SPDX-License-Identifier: Apache-2.0
"""CPU representation checks of ordered GDN logs and BF16 step boundaries."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


@pytest.fixture
def reference(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_compact_reference.py"
    spec = importlib.util.spec_from_file_location("compact_gdn_cpu_reference", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def step_inputs(batch=3, heads=2, key_width=8, value_width=6):
    generator = torch.Generator().manual_seed(881)
    base = (torch.randn(batch, heads, value_width, key_width, generator=generator) * 0.03).to(torch.bfloat16)
    query, key = (torch.randn(batch, 4, heads, key_width, generator=generator) * 0.1 for _ in range(2))
    value = torch.randn(batch, 4, heads, value_width, generator=generator) * 0.2
    gamma = torch.exp(-torch.rand(batch, 4, heads, generator=generator) * 0.2)
    beta = torch.sigmoid(torch.randn(batch, 4, heads, generator=generator))
    return base, query, key, value, gamma, beta


def test_scalar_binary_fraction_oracle(reference):
    base = torch.tensor([[[[0.25]]]], dtype=torch.bfloat16)
    q = k = torch.ones(1, 4, 1, 1)
    v = torch.tensor([0.5, -0.25, 0.75, -0.5]).reshape(1, 4, 1, 1)
    gamma = beta = torch.full((1, 4, 1), 0.5)
    # With k=q=1, gamma=beta=1/2: H_t = H_(t-1)/4 + v_t/2.
    expected = torch.tensor([0.3125, -0.046875, 0.36328125, -0.1591796875]).reshape(1, 4, 1, 1)
    snapshots, outputs = reference.dense_step(base, q, k, v, gamma, beta)
    log, compact_outputs = reference.emit_compact_step(base, q, k, v, gamma, beta)
    torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
    torch.testing.assert_close(compact_outputs, expected, rtol=0, atol=0)
    torch.testing.assert_close(snapshots.squeeze(-1), expected.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(reference.export_logical_states(log), snapshots, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [1, 2, 4, torch.tensor([1, 2, 4]), torch.tensor([4, 1, 2], dtype=torch.int32)])
def test_all_candidates_and_selected_prefix_match_dense(reference, accepted):
    inputs = step_inputs()
    candidates, expected_output = reference.dense_step(*inputs)
    state, output = reference.emit_compact_step(*inputs)
    torch.testing.assert_close(reference.export_logical_states(state), candidates, rtol=0, atol=0)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    prefixes = torch.full((3,), accepted) if isinstance(accepted, int) else accepted
    wanted = candidates[torch.arange(3), prefixes.long() - 1]
    restored = reference.restore_accepted(state, accepted)
    assert restored.dtype == torch.bfloat16
    torch.testing.assert_close(restored, wanted, rtol=0, atol=0)


def test_logs_keep_actual_fp32_operands_and_do_not_alias_sources(reference):
    inputs = step_inputs()
    before = tuple(value.clone() for value in inputs)
    state, output = reference.emit_compact_step(*inputs)
    candidates, dense_output = reference.dense_step(*inputs)
    restored, exported = reference.restore_accepted(state, 2), reference.export_logical_states(state)
    for actual, wanted in zip(inputs, before, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert state.base.dtype == torch.bfloat16
    assert state.valid.dtype == torch.bool and state.valid.all()
    for value in (state.gamma, state.keys, state.updates):
        assert value.dtype == torch.float32
        assert not torch.equal(value, value.bfloat16().float())
    torch.testing.assert_close(state.gamma, inputs[4], rtol=0, atol=0)
    torch.testing.assert_close(state.keys, inputs[2], rtol=0, atol=0)
    storage = {value.untyped_storage().data_ptr() for value in inputs}
    results = (
        state.base,
        state.gamma,
        state.keys,
        state.updates,
        state.valid,
        output,
        candidates,
        dense_output,
        restored,
        exported,
    )
    assert not storage.intersection(value.untyped_storage().data_ptr() for value in results)
    assert len({value.untyped_storage().data_ptr() for value in results}) == len(results)
    saved = tuple(value.clone() for value in (state.base, state.gamma, state.keys))
    inputs[0].fill_(9)
    inputs[4].zero_()
    inputs[2].fill_(11)
    for actual, wanted in zip((state.base, state.gamma, state.keys), saved, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    exported.zero_()
    torch.testing.assert_close(reference.export_logical_states(state), candidates, rtol=0, atol=0)


def test_initial_invalid_log_exposes_base_for_every_prefix(reference):
    base = step_inputs()[0]
    state = reference.initial_compact_state(base)
    assert not state.valid.any()
    torch.testing.assert_close(reference.export_logical_states(state), base[:, None].expand(-1, 4, -1, -1, -1))
    for accepted in (1, 2, 4):
        torch.testing.assert_close(reference.restore_accepted(state, accepted), base, rtol=0, atol=0)
    base.zero_()
    assert state.base.count_nonzero()


@pytest.mark.parametrize("seed,key_width,value_width,steps", [(41, 128, 128, 4), (992, 16, 12, 8), (6711, 8, 8, 6)])
def test_closed_loop_preserves_request_identity_inactive_null_and_mixed_prefixes(
    reference, seed, key_width, value_width, steps
):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = reference.qualify_closed_loop(
            seed, steps=steps, requests=7, heads=1, key_width=key_width, value_width=value_width
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert result["status"] == "passed_under_identical_CPU_FP32_arithmetic"
    assert result["candidate_elements_checked"] == 7 * 4 * value_width * key_width * steps
    assert {accepted for record in result["records"] for accepted in record["accepted"]} == {1, 2, 4}
    assert all(0 in record["inactive_request_ids"] for record in result["records"])
    assert any(record["active_request_ids"] != sorted(record["active_request_ids"]) for record in result["records"])
    active_sets = [set(record["active_request_ids"]) for record in result["records"]]
    assert any(active_sets[index] != active_sets[index + 1] for index in range(len(active_sets) - 1))


@pytest.mark.parametrize(
    "accepted",
    [
        0,
        5,
        True,
        1.5,
        torch.tensor([1, 2]),
        torch.tensor([1.0, 2.0, 4.0]),
        torch.tensor([[1, 2, 4]]),
        torch.tensor([1, 0, 4]),
    ],
)
def test_restore_rejects_invalid_accepted_prefixes(reference, accepted):
    state, _ = reference.emit_compact_step(*step_inputs())
    with pytest.raises(ValueError):
        reference.restore_accepted(state, accepted)


@pytest.mark.parametrize(
    "field",
    [
        "base_shape",
        "base_dtype",
        "query_shape",
        "key_dtype",
        "value_shape",
        "gamma_negative",
        "gamma_nan",
        "beta_dtype",
    ],
)
@pytest.mark.parametrize("operation", ["dense_step", "emit_compact_step"])
def test_step_rejects_invalid_tensor_contract(reference, field, operation):
    inputs = list(step_inputs())
    if field == "base_shape":
        inputs[0] = inputs[0][0]
    elif field == "base_dtype":
        inputs[0] = inputs[0].float()
    elif field == "query_shape":
        inputs[1] = inputs[1][:, :3]
    elif field == "key_dtype":
        inputs[2] = inputs[2].bfloat16()
    elif field == "value_shape":
        inputs[3] = inputs[3][..., :-1]
    elif field == "gamma_negative":
        inputs[4][0, 0, 0] = -1
    elif field == "gamma_nan":
        inputs[4][0, 0, 0] = float("nan")
    else:
        inputs[5] = inputs[5].double()
    with pytest.raises(ValueError):
        getattr(reference, operation)(*inputs)


@pytest.mark.parametrize("field", ["gamma", "keys", "updates", "valid"])
def test_restore_and_export_reject_lossy_or_malformed_logs(reference, field):
    state, _ = reference.emit_compact_step(*step_inputs())
    invalid = replace(state, **{field: getattr(state, field).bfloat16()})
    for operation in (lambda: reference.restore_accepted(invalid, 1), lambda: reference.export_logical_states(invalid)):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("field", ["base", "gamma", "keys", "updates"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_state_consumers_reject_nonfinite_base_or_log(reference, field, nonfinite):
    state, _ = reference.emit_compact_step(*step_inputs())
    invalid_value = getattr(state, field).clone()
    invalid_value.flatten()[0] = nonfinite
    invalid = replace(state, **{field: invalid_value})
    with pytest.raises(ValueError):
        reference.restore_accepted(invalid, 2)
    with pytest.raises(ValueError):
        reference.export_logical_states(invalid)


@pytest.mark.parametrize("kwargs", [dict(steps=0), dict(requests=-1), dict(heads=True), dict(key_width=0)])
def test_report_rejects_invalid_dimensions(reference, kwargs):
    with pytest.raises(ValueError):
        reference.qualify_closed_loop(**kwargs)
