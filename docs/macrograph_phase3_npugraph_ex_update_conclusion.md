# Macrograph Phase 3 npugraph_ex Update Conclusion

Date: 2026-06-17

## Context

The current Phase 3 work investigates whether mixed-request macrograph replay
can safely update FIA attention parameters through `graph_task_update` instead
of falling back to piecewise execution.

The target failure shape is the compact mixed-request macro key:

```text
(68, False, (36, 32), (36, 32), (5, 1), (0, 0),
 'matmul_serial_attention_parallel', 'mixed_request',
 'torch.bfloat16', 1, '/vllm-workspace/models/Qwen3-0.6B')
```

This corresponds to a `step_mixed` workload with four decode requests and two
32-token prefill arrivals:

```text
request_tokens = [1, 1, 1, 1, 32, 32]
split = 36 + 32
```

## Confirmed Findings

The failure is not caused by stale runtime `seq_lens`.

Diagnostics from `task_update` mode showed that the update path reads the new
runtime values. For example, capture step 2 split 0 used:

```text
seq_lens = [33, 33, 33, 33, 32]
```

and replay/update step 3 split 0 used:

```text
seq_lens = [34, 34, 34, 34, 32]
```

The failure is also not a general `npugraph_ex` inability to use
`ExternalEvent + graph_task_update`.

The diagnostic script `examples/diagnose_npugraph_ex_fia_update.py` verified:

- single custom-op FIA under `npugraph_ex` can be externally task-updated;
- dual-stream TND FIA with 28 layers and 56 handle/event records can be
  externally task-updated;
- forced serial replay still failed in full vLLM mixed macrograph, so the
  original MTE was not caused by the Python secondary stream fork/join alone.

## Backend Opaque Update Result

`VLLM_ASCEND_MACRO_GRAPH_NPUGRAPH_EX_OPAQUE_ATTENTION_UPDATE=1` was tested as a
candidate replacement for vLLM-side external FIA update.

The real `step_mixed` reproduction did hit the target mixed-request compact
macrograph and materialized the `npugraph_ex` macro graph:

```text
/vllm-workspace/prof_result/macro_phase3_step_mixed_backend_opaque_20260617_104614
```

However, backend opaque update failed inside `npugraph_ex`:

```text
CapturedGraphUpdateAndReplay.forward
  -> _run_updates(update_before_replay=True)
  -> torch.npu.graph_task_update_end(...)
```

The failing operator was not FIA. The runtime reported:

```text
current working operator name is aclnnInplaceCopy
Kernel Run failed. opType: 17, Slice
driver error: current driver version does not support to update this op
```

This means the opaque update target is too broad. Wrapping
`vllm::macro_unified_attention*` makes `npugraph_ex` try to update task groups
that include non-FIA work such as Slice/Copy, which is not supported by the
current driver.

## Current Root Cause

The Phase 3 failure is now narrowed to the update boundary.

The correct update region must be FIA-only. Updating the whole
`macro_unified_attention` custom op is unsafe because it includes unsupported
operators. Conversely, vLLM-side manual update of internal FIA handles is not a
stable backend contract for `npugraph_ex` macrograph replay.

## Safe Current Behavior

Keep the default safe path:

```bash
VLLM_ASCEND_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE=0
VLLM_ASCEND_MACRO_GRAPH_NPUGRAPH_EX_OPAQUE_ATTENTION_UPDATE=0
```

When mixed-request `npugraph_ex` macrograph would require attention update and
external update is disabled, fall back to piecewise execution rather than
running an unsafe macro replay.

## Recommended Next Step

Do not continue with `macro_unified_attention` opaque update as the production
path.

The next implementation should move toward one of these:

1. backend selective FIA update: `npugraph_ex` records and updates only FIA task
   handles inside the macro graph;
2. PTA C++ macro update API: Python passes a compact update descriptor, while
   PTA/backend owns replay launch, FIA task update, and event ordering;
3. keep mixed-request macrograph disabled/fallback by default until FIA-only
   backend update is available.

The immediate development target is to remove or keep experimental-only any
path that updates whole macro attention custom ops.
