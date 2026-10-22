# SPDX-License-Identifier: Apache-2.0
"""CPU routing contracts; these do not qualify the frozen device arithmetic."""

import importlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    assert importlib.util.find_spec("multi_graph_gdn_q14_layer_core") is not None, (
        "layer core helper is not implemented"
    )
    return importlib.import_module("multi_graph_gdn_q14_layer_core")


class TorchWithNPU:
    def __init__(self, events):
        self.events = events
        self.npu = self
        self.current = self.Stream()

    def __getattr__(self, name):
        return getattr(torch, name)

    def Stream(self):
        return SimpleNamespace(wait_event=lambda event: self.events.append("wait"))

    def Event(self, **kwargs):
        return SimpleNamespace(record=lambda stream: self.events.append("record"))

    def current_stream(self):
        return self.current

    def synchronize(self):
        self.events.append("synchronize")

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


def fixture(helper, mode="q14_ready"):
    from multi_graph_gdn_q4_log_backend import Q4LogState

    events, calls = [], []
    api = TorchWithNPU(events)
    state = Q4LogState.from_base(api, torch.zeros(3, 24, 128, 128), [[1, 2, 3, 4], [5, 6, 7, 8]])
    descriptor = helper.PackedLayerDescriptor(
        pack_indices=torch.tensor([0, 4, 5, 6, 7, 0, 0, 0]),
        inverse_indices=torch.tensor([0, 8, 8, 8, 1, 2, 3, 4]),
        active=torch.tensor([True] * 5 + [False] * 3),
        sequence=torch.tensor([0, 1, 4], dtype=torch.int32),
        ids=torch.tensor([2, 1, 1, 1, 1, 0, 0, 0], dtype=torch.int32),
        accepted=torch.tensor([4, 1], dtype=torch.int32),
    )
    causal = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 5], dtype=torch.int32),
        cache_indices=torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([4, 1], dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=8,
        num_prefills=0,
        num_decodes=0,
        spec_sequence_masks=torch.ones(2, dtype=torch.bool),
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=causal, actual_seq_lengths=descriptor.sequence.clone()),
    )

    def conv(x, conv_state, weight, *, out, **kwargs):
        # Model the native operator leaving its non-token output tail untouched.
        count = int(kwargs["query_start_loc"][-1])
        calls.append(("conv_before", out.clone(), kwargs["num_accepted_tokens"].clone()))
        out[:count].copy_(x[:count])
        return out

    def original_recurrence(*args, **kwargs):
        raise AssertionError("unexpected dense recurrence fallback")

    ops = SimpleNamespace(causal_conv1d_update=conv, recurrent_gated_delta_rule=original_recurrence)

    class Projection(torch.nn.Module):
        def forward(self, output):
            events.append("projection")
            return output

    class GDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = Projection()
            self.kv_cache = (torch.zeros(9, 6, 1), state.handle)
            self.fail = False

        def _forward_core(self, mixed, b, a, output):
            calls.append(("core", mixed.clone(), b.clone(), a.clone()))
            if self.fail:
                raise RuntimeError("core failure")
            convolved = ops.causal_conv1d_update(
                mixed,
                self.kv_cache[0],
                None,
                out=torch.full_like(mixed, float("nan")),
                query_start_loc=causal.query_start_loc,
                num_accepted_tokens=causal.num_accepted_tokens,
            )
            query = convolved[:, :1, None].expand(8, 8, 128).contiguous()
            value = convolved[:, :1, None].expand(8, 24, 128).contiguous()
            result = ops.recurrent_gated_delta_rule(
                query,
                query,
                value,
                state.handle,
                beta=b,
                g=a.float(),
                scale=128**-0.5,
                actual_seq_lengths=metadata.spec_decode_metadata.actual_seq_lengths,
                ssm_state_indices=causal.cache_indices.flatten(),
                num_accepted_tokens=causal.num_accepted_tokens,
            )
            output.copy_(result)

        def forward(self, mixed, b, a):
            output = torch.full((8, 24, 128), float("nan"), dtype=torch.bfloat16)
            self._forward_core(mixed, b, a, output)
            return self.out_proj(output)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gdn = GDN()

        def forward(self, *args):
            result = self.gdn(*args)
            events.append("mlp")
            return result

    def module(label):
        def solve(*args):
            events.append("solve:" + label)
            calls.append(("solve", args[8], args[9], args[10], args[-1]))
            output, value = args[12], args[5]
            count = int(args[8].sum())
            output[:count].copy_(value[:count])

        def commit(*args):
            events.append("commit:" + label)
            calls.append(("commit", args[1], args[2], args[3]))

        return SimpleNamespace(solve=solve, commit=commit)

    layer = Layer()
    kwargs = dict(
        torch=api,
        gdn=layer.gdn,
        metadata=metadata,
        state=state,
        descriptor=descriptor,
        old_module=module("old"),
        new_module=module("new"),
        deferred_module=module("deferred"),
        mode=mode,
        endpoint=layer,
        gdn_ops=ops,
    )
    values = torch.arange(1, 9, dtype=torch.bfloat16).view(8, 1)
    return layer, kwargs, (values, values.expand(8, 24).contiguous(), values.expand(8, 24).contiguous()), calls, events


def test_packed_core_uses_independent_compact_ids_and_inverse_sentinel(helper):
    layer, kwargs, inputs, calls, events = fixture(helper)
    with helper.installed(**kwargs) as adapter:
        result = layer(*inputs)
        assert adapter.backend._pending_commit is None
    descriptor = kwargs["descriptor"]
    assert result[:, 0, 0].tolist() == [1, 0, 0, 0, 5, 6, 7, 8]
    assert torch.isfinite(result).all()
    assert next(c for c in calls if c[0] == "core")[1].flatten().tolist() == [1, 5, 6, 7, 8, 0, 0, 0]
    for name in ("solve", "commit"):
        call = next(c for c in calls if c[0] == name)
        assert call[1] is descriptor.sequence and call[2] is descriptor.ids and call[3] is descriptor.accepted
    assert next(c for c in calls if c[0] == "solve")[-1] is True
    assert events.index("commit:new") < events.index("projection") < events.index("mlp") < len(events) - 1
    assert events[-1] == "wait"


def test_official_conv_output_is_cleared_before_partial_native_write(helper):
    layer, kwargs, inputs, calls, _ = fixture(helper)
    with helper.installed(**kwargs):
        layer(*inputs)
    conv_call = next(c for c in calls if c[0] == "conv_before")
    assert torch.equal(conv_call[1], torch.zeros_like(conv_call[1]))
    assert conv_call[2].tolist() == [4, 1]


@pytest.mark.parametrize("mode,label,ready", [("forced_q4", "old", True), ("q14_delayed", "new", False)])
def test_frozen_dispatch_and_forced_control_preserve_the_declared_mode(helper, mode, label, ready):
    layer, kwargs, inputs, calls, events = fixture(helper, mode)
    if mode == "forced_q4":
        kwargs["descriptor"].sequence.copy_(torch.tensor([0, 4, 4]))
        kwargs["descriptor"].ids.copy_(torch.tensor([2] * 4 + [1] * 4))
        kwargs["descriptor"].active.fill_(True)
        kwargs["descriptor"].inverse_indices.copy_(torch.arange(8))
        kwargs["metadata"].spec_decode_metadata.spec_causal_conv1d.query_start_loc.copy_(torch.tensor([0, 4, 8]))
    with helper.installed(**kwargs):
        result = layer(*inputs)
    assert "solve:" + label in events
    assert ("commit:deferred" if mode == "forced_q4" else "commit:new") in events
    assert next(c for c in calls if c[0] == "solve")[-1] is ready
    if mode == "forced_q4":
        assert result[:, 0, 0].tolist() == inputs[0].flatten().tolist()


def test_descriptor_updates_keep_buffers_and_restore_every_patch(helper):
    layer, kwargs, inputs, calls, _ = fixture(helper)
    gdn, ops = layer.gdn, kwargs["gdn_ops"]
    original_core, original_conv, original_recurrence = (
        gdn._forward_core,
        ops.causal_conv1d_update,
        ops.recurrent_gated_delta_rule,
    )
    with helper.installed(**kwargs) as adapter:
        first = layer(*inputs)
        pointers = adapter.buffer_pointers()
        kwargs["descriptor"].pack_indices[0] = 2
        second = layer(*inputs)
        assert first[0, 0, 0] == 1 and second[0, 0, 0] == 3
        assert adapter.buffer_pointers() == pointers
        assert ops.causal_conv1d_update is original_conv and ops.recurrent_gated_delta_rule is original_recurrence
    assert gdn._forward_core == original_core
    assert "_forward_core" not in gdn.__dict__
    assert not gdn.out_proj._forward_pre_hooks and not layer._forward_hooks


def test_core_exception_restores_global_ops_and_instance_method(helper):
    layer, kwargs, inputs, _, _ = fixture(helper)
    ops = kwargs["gdn_ops"]
    original_conv, original_recurrence = ops.causal_conv1d_update, ops.recurrent_gated_delta_rule
    layer.gdn.fail = True
    with pytest.raises(RuntimeError, match="core failure"), helper.installed(**kwargs):
        layer(*inputs)
    assert ops.causal_conv1d_update is original_conv and ops.recurrent_gated_delta_rule is original_recurrence
    assert "_forward_core" not in layer.gdn.__dict__
    assert not layer.gdn.out_proj._forward_pre_hooks and not layer._forward_hooks


def test_rejects_non_spec_metadata_before_installing_any_hook(helper):
    layer, kwargs, _, _, _ = fixture(helper)
    kwargs["metadata"].num_decodes = 2
    with pytest.raises(ValueError, match="speculative"), helper.installed(**kwargs):
        pass
    assert "_forward_core" not in layer.gdn.__dict__
    assert not layer.gdn.out_proj._forward_pre_hooks


def test_paused_request_zero_interval_and_persistent_reinstallation(helper):
    layer, kwargs, inputs, _, _ = fixture(helper)
    descriptor = kwargs["descriptor"]
    descriptor.sequence.copy_(torch.tensor([0, 1, 0]))
    descriptor.active.copy_(torch.tensor([True] + [False] * 7))
    descriptor.ids.copy_(torch.tensor([2] + [0] * 7))
    with helper.installed(**kwargs) as adapter:
        layer(*inputs)
        pointers = adapter.buffer_pointers()
    with helper.installed(adapter=adapter) as reused:
        layer(*inputs)
        assert reused is adapter
        assert adapter.buffer_pointers() == pointers


def test_real_custom_op_keyword_contract(helper):
    layer, kwargs, inputs, _, _ = fixture(helper)
    with helper.installed(**kwargs) as adapter:
        output = torch.empty((8, 24, 128), dtype=torch.bfloat16)
        layer.gdn._forward_core(mixed_qkv=inputs[0], b=inputs[1], a=inputs[2], core_attn_out=output)
        adapter.backend.launch_pending_commit()
        adapter.backend.finish_pending_commit()
        assert torch.isfinite(output).all()
