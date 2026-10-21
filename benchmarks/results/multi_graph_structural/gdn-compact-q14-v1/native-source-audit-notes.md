# Q1/Q4 native control source audit

Local source audit, 2026-10-21. No device experiment was run by this audit.
The installed snapshot is vLLM 0.30.0+empty, vLLM-Ascend 0.30.0rc1,
torch 2.10.0+cpu, torch_npu 2.10.0.post4 according to
`work/gdn-runner-compat-download-v1/inventory.json`.

## Installed dispatch and callable boundaries

Installed-source snapshot:
`work/gdn-runner-compat-download-v1/sources/vllm_ascend/ops/gdn.py`.

- Lines 564-575: Q4 speculative rows call `recurrent_gated_delta_rule`.
- Lines 706-720: pure Q1 calls that same recurrence without
  `num_accepted_tokens`.
- Installed builder
  `work/gdn-runner-compat-download-v1/sources/vllm_ascend/ops/gdn_attn_builder.py:777`
  promotes non-spec Q1 rows to prefill when Q4 spec rows coexist. Thus a mixed
  Q1/Q4 batch uses Q4 recurrence plus a Q1 chunk call. A single variable-width
  recurrence can be a separate arithmetic control, but is not this installed
  mixed dispatch.
- `gdn.py:620`: first preference is `DeviceOperator.fla_gdn_prefill` when
  PCP world size is one, hardware supports `FLA_GDN_PREFILL`, and
  `_get_fla_gdn_prefill_op()` returns a callable.
- `gdn.py:58`: `_get_fla_gdn_prefill_op()` imports
  `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`, returning `None` only on
  `ImportError`.
- `gdn.py:627`: second preference is the torch_npu fused chunk op if FLA is
  absent, the device is NPU, its smoke probe succeeds, and PCP world size is one.
- `gdn.py:682`: remaining preference is
  `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule`.

The current local checkout's hardware profile marks A2 as supporting
`FLA_GDN_PREFILL` (`work/vllm-ascend/vllm_ascend/device/hardware_profile.py:257`
and `:289`). The installed snapshot does not include this hardware-profile
file or the installed FLA Python wrapper. Its actual import/capability and
call signature must be recorded on the remote device before choosing the
branch. An attribute existing does not prove that its kernel runs there.

The observed installed recurrence call is:

```python
out = gdn.recurrent_gated_delta_rule(
    q, k, v, dense_state,  # [T,Nk,Dk], [T,Nk,Dk], [T,Nv,Dv], [S,Nv,Dv,Dk]
    g=g, beta=beta, scale=Dk ** -0.5,
    actual_seq_lengths=sequence,  # int32 [0, qlen0, qlen1, ...], NOT cumsum
    ssm_state_indices=ids,       # int32 [T], one destination per actual token
    # num_accepted_tokens=... only for Q4 below
)
```

`gdn_attn_builder.py:166-180` converts cumulative query locations to
`[0, *individual_lengths]`; the recurrence kernel accumulates those lengths.
By contrast, chunk APIs take cumulative `cu_seqlens`.

## Previous acceptance and continuous dense candidates

Do not pass previous accepted counts 2..4 to current Q1. The vendor kernel
checks `acceptedTokenNum <= seqLen` and otherwise returns before its work.
Exact checked source:
`work/reference-fla-npu-20261020/fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h:184-195`.
The vendored derivative contains the same check at
`work/vllm-ascend/benchmarks/multi_graph_gdn_compact_state/vendor_recurrent.h:197`.
This reference is corroborative source, not a claim that the remote binary was
built from this exact revision.

Maintain four dense slots per stable owner and a previous published slot list.
Before the current call, select the physical slot
`previous_slots[owner][accepted_previous - 1]`. This is metadata selection,
not a reconstruction from compact logs. Preserve each control's own dense
trajectory across the segment; never seed each step from another control.

```python
selected = {o: previous_slots[o][a - 1] for o, a in live_acceptances}
for owner in live_owners:
    first = selected[owner]
    current_slots[owner] = (
        [first] if width[owner] == 1 else
        [first] + [s for s in four_dense_slots[owner] if s != first]
    )

if all(width[o] == 1 for o in live_owners):
    # One token and selected input/destination slot per owner; no accepted arg.
    output = recurrent(q, k, v, dense_state,
                       actual_seq_lengths=[0] + [1] * n,
                       ssm_state_indices=[selected[o] for o in live_owners],
                       g=g, beta=beta, scale=scale)
else:
    # Q4 rows packed in their original relative order.
    output_q4 = recurrent(q4, k4, v4, dense_state,
                          actual_seq_lengths=[0] + [4] * n4,
                          ssm_state_indices=flatten(current_slots[o] for o in q4_owners),
                          num_accepted_tokens=ones(n4, int32),
                          g=g4, beta=beta4, scale=scale)
    # If n1 > 0, Q1 goes through the installed chunk branch below.
    # Gather the selected dense Q1 rows, chunk, scatter back to those slots.
    # Merge outputs into original packed actual-token order.

previous_slots.update(current_slots)  # publish current width; paused owners unchanged
```

All actual tensors and indices are on NPU; Python lists above describe
metadata. Precompute host metadata outside the device endpoint. Include any
device partition/index_select, dense gather, transpose/contiguous, chunk,
scatter, and output merge in the native endpoint timing. Q4 recurrence reads
the first current slot and writes all four current slots; there is no need
to copy the selected dense state to an arbitrary fixed slot.

## Saved-site normalization boundary

The saved recurrence sites intercept `Q4LogBackend.__call__`
(`work/vllm-ascend/benchmarks/multi_graph_gdn_ready_consumer_extract.py:30`).
Installed `gdn.py:562-563` applies `l2norm_fwd` before reaching that call.
Therefore saved query/key are already normalized, and raw pre-normalization
Q/K are not available in this fixture.

Use the same saved normalized Q/K for every control. Do not call a wrapper
that normalizes them again. Label the measurement as the recurrence site
after normalization. This preserves the installed native chunk operator,
state transforms, and dispatch, but excludes the earlier normalization
boundary for all controls. Calling full wrappers on the saved operands
would introduce another BF16 normalization and could change arithmetic.

### Preferred A2 FLA chunk call below normalization

Exact local wrapper:
`work/vllm-ascend/vllm_ascend/device/device_op.py:854-913`.
It normalizes Q/K at lines 866-867, converts sequence-major operands to
`BNSD` at 875-877, transposes state at 883, calls FLA at 888-906,
and returns final state transposed at 913. For saved normalized Q/K call
the same FLA op directly, retaining those transforms:

```python
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_

# q1/k1/v1: [1,n1,H,D]; g1/beta1: [1,n1,Nv].
# Q1 owners only: no empty/padded sequences in this direct call.
state1 = gather_ssm_states(dense_state, selected_q1_slots, has_state_true)
o1, final1, *_ = fla_chunk_fwd(
    q1.transpose(1, 2).contiguous(),
    k1.transpose(1, 2).contiguous(),
    v1.contiguous().transpose(1, 2).contiguous(),
    g1.to(torch.float32).contiguous(),
    beta1.to(v1.dtype).contiguous(),
    initial_state=state1.transpose(-1, -2).contiguous(),
    output_final_state=True, chunk_size=64,
    cu_seqlens=tuple(range(n1 + 1)),
    chunk_indices=tuple(x for i in range(n1) for x in (i, 0)),
    scale=scale, layout="BNSD", use_exp2=False,
    use_qk_l2norm_in_kernel=False, allow_neg_eigval=False,
    disable_recompute=True, state_v_first=False,
)
scatter_ssm_states_(dense_state, selected_q1_slots,
                    final1.transpose(-1, -2).contiguous().to(dense_state.dtype))
# o1 is already sequence-major [1,n1,Nv,Dv], per DeviceOperator wrapper.
```

The host tuples above are the installed builder's metadata for n1 sequences
of length one: `gdn_attn_builder.py:269-334`.
To obtain metadata via the builder instead, call
`_build_non_spec_chunked_prefill_metadata(builder, cu_seqlens_cpu, device,
build_device_metadata=False)` and use `cu_seqlens_host` and
`chunk_indices_chunk64_host`. The `builder` is not consulted for this
host-only mode. For the Triton fallback use
`build_device_metadata=True` and supply the actual builder/config; this
avoids per-call `.tolist()` and device metadata creation.

The native gather/scatter helpers are at
`work/vllm-ascend/vllm_ascend/ops/triton/mamba/state_index.py:124` and `:170`.
They support the padded batch stride of the installed cache and read only
selected rows. `gather_ssm_states(state, indices, bool_has_initial_state,
output_dtype=None)` returns contiguous dense rows. `scatter_ssm_states_(state,
indices, source)` writes the selected rows. Include these device operations
in the endpoint cost.

### Other chunk branches

If FLA is absent but the torch_npu fused smoke succeeds, call its native op
directly below normalization, corresponding to installed `gdn.py:243-268`:

```python
state1 = gather_ssm_states(dense_state, selected_q1_slots, has_state_true,
                          output_dtype=torch.bfloat16)
o1, final1 = torch_npu.npu_chunk_gated_delta_rule(
    q1.squeeze(0).contiguous(), k1.squeeze(0).contiguous(), v1.squeeze(0).contiguous(),
    beta=beta1.squeeze(0).to(v1.dtype).contiguous(),
    initial_state=state1.to(torch.bfloat16).contiguous(),
    actual_seq_lengths=ones(n1, dtype=torch.int32),  # no prefix element
    scale=scale, g=g1.squeeze(0).float().contiguous(),
)
scatter_ssm_states_(dense_state, selected_q1_slots, final1)
o1 = o1.unsqueeze(0)
```

This branch intentionally uses BF16 initial state because that is the
installed wrapper's stated operator restriction, even when the dense cache
is FP32. Report any resulting correctness difference.

If both native fused options are unavailable, the installed Triton call is:

```python
state1 = gather_ssm_states(dense_state, selected_q1_slots, has_state_true)
o1, final1 = gdn.chunk_gated_delta_rule(
    q=q1, k=k1, v=v1, g=g1, beta=beta1, scale=scale,
    initial_state=state1.transpose(-1, -2).contiguous(),
    output_final_state=True, cu_seqlens=cu1_device,
    prebuilt_meta=chunk_meta, head_first=False,
    use_qk_l2norm_in_kernel=False,  # saved site is already normalized
)
scatter_ssm_states_(dense_state, selected_q1_slots,
                    final1.transpose(-1, -2).contiguous().to(dense_state.dtype))
```

Exact current chunk signature:
`work/vllm-ascend/vllm_ascend/ops/triton/fla/chunk.py:259-275`.
The optional norm flag is honored at `:238-240`; disabling it is supported.
This pipeline obtains `get_forward_context()` (`:42`), so establish the
minimal valid forward context rather than assuming an isolated import call
is sufficient. Its metadata API should be inspected in the installed
environment before use.

## One bounded remote capability inspection

Run in the same Python environment and NPU device as the experiment. The
optional `_probe_fused_chunk()` smoke launches one small kernel and syncs;
it is not a benchmark or actual-Q1 shape qualification. The class caches
the outcome for the process.

```python
import inspect, json, torch, torch_npu
import vllm_ascend.ops.gdn as gdn
from vllm_ascend.device.hardware_profile import get_current_hardware_profile, HardwareCapability

profile = get_current_hardware_profile()
fla = gdn._get_fla_gdn_prefill_op()
print(json.dumps(dict(
    gdn_source=gdn.__file__,
    fla_capability=profile.supports(HardwareCapability.FLA_GDN_PREFILL),
    fla_op=None if fla is None else dict(module=fla.__module__,
        signature=str(inspect.signature(fla)), source=inspect.getsourcefile(fla)),
    torch_fused_attribute=hasattr(torch_npu, "npu_chunk_gated_delta_rule"),
    recurrence_signature=str(inspect.signature(gdn.recurrent_gated_delta_rule)),
    triton_chunk_signature=str(inspect.signature(gdn.chunk_gated_delta_rule)),
), default=str))
print(inspect.getsource(gdn.DeviceOperator.fla_gdn_prefill))
print("torch_fused_smoke", gdn.AscendGatedDeltaNetAttention._probe_fused_chunk())
```

Then one untimed actual Q1-chunk shape call and synchronization are required
before timing. Preserve errors and the selected branch in the result.
The local audit cannot establish which installed chunk callable succeeds
on the remote A2 or its actual numerical/performance behavior.

## Existing probe reuse limits

`work/vllm-ascend/benchmarks/multi_graph_gdn_q4_log_layer_probe.py:156-177`
loads a real single GDN layer and checks FP32 TP2 state `[*,24,128,128]`,
but line 166 requires Q4. Its fixture/snapshot approach is useful; its Q4
metadata and changing accepted counts are not a valid Q1 transition test.
The saved-site consumer timing probe
`work/vllm-ascend/benchmarks/multi_graph_gdn_ready_consumer_probe.py`
is explicitly a consumer sublayer endpoint; it does not establish a full
real-layer Q1/Q4 comparison. This audit recommends no runner installation
and no compact-to-dense bridge.
