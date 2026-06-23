# Macrograph Phase 3 npugraph_ex Update Conclusion

Date: 2026-06-17
Update: 2026-06-19

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

## Environment Guardrail

Mixed macrograph `npugraph_ex` validation must use the torch-npu environment
that actually registers the `npugraph_ex` torch.compile backend:

```bash
/vllm-workspace/.venv-torchnpu-280-post4/bin/python - <<'PY'
import torch
import torch_npu
print(torch_npu.__version__)
print(torch._dynamo.list_backends())
assert "npugraph_ex" in torch._dynamo.list_backends()
PY
```

Observed on 2026-06-19:

- container default `python`: `torch_npu 2.8.0.post1+gitc7b6b32`, available
  backends do not include `npugraph_ex`;
- `/vllm-workspace/.venv-torchnpu-280-post4/bin/python`:
  `torch_npu 2.8.0.post4+git5705b58`, available backends include
  `npugraph_ex`.

Do not use the default `python` for `npugraph_ex` macro smoke runs. A failure
like this is an environment miss, not a macrograph update result:

```text
RuntimeError: macro_graph_config.backend='npugraph_ex' requires torch-npu with
a registered 'npugraph_ex' torch.compile backend.
```

For `benchmark_split_tpot.py`, `workload=step_mixed` alone is not enough to
exercise mixed macrograph replay. The command must explicitly enable mixed
split and macrograph config:

```text
--enable-mixed-request-split
--mixed-request-split-execution-mode piecewise_attention_parallel
--macro-graph-config-json ...
```

Without these flags the run only validates the ordinary inplace parallel graph
path and must not be used to judge FIA update behavior.

The ACLGraph/FIA update parameter diagnostic flag is:

```text
VLLM_ASCEND_ACLGRAPH_UPDATE_PARAM_DIAG=1
```

Do not use `VLLM_ASCEND_ACL_GRAPH_UPDATE_PARAM_DIAG`; that spelling is not
read by `vllm_ascend/compilation/acl_graph.py`.

`benchmark_split_tpot.py` may still return shell status 0 after an internal
macrograph replay failure. Treat the terminal `[FAIL]` row, `results_*.json`,
and `debug.jsonl` as the source of truth for these smoke tests; do not classify
a run as passing from `$?` alone.

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

## Replay/Update Timing Boundary

The replay/update order is confirmed to be correct and should not be treated as
the current failure source.

Python intentionally launches macro graph replay first, then starts the FIA
parameter update. The captured graph does not execute FIA immediately: the FIA
task is guarded by captured `ExternalEvent.wait/reset`, and the update stream
records that event only after `graph_task_update_begin/out/end` finishes. This
makes the NPU-side order effectively:

```text
launch replay -> graph waits before FIA -> update captured FIA params ->
event.record(update_stream) -> FIA executes with updated params
```

The update target is the captured FIA task. The key updated parameters include
`seq_len` / `actual_seq_lengths_kv` style runtime sequence-length metadata, and
the purpose is to change FIA's tiling strategy for the current mixed step.
Therefore, mixed macrograph update failures should not be debugged by moving
the update before replay or by adding a pre-replay timing mode.

## 2026-06-19 exact68 Revalidation

After switching to the correct `npugraph_ex` environment
(`/vllm-workspace/.venv-torchnpu-280-post4/bin/python`), the exact mixed shape
still fails on the second mixed step:

```text
total_tokens = 68
split_actual_tokens = [36, 32]
split_graph_tokens = [36, 32]
split_num_reqs = [5, 1]
```

The one-step run materializes the graph and succeeds because the first
`npugraph_ex` call returns captured outputs and skips external FIA update.
The two-step run fails during the second replay at
`self.stream_main.synchronize()` with FIA MTE DDR out-of-range.

Diagnostics now confirm:

- capture/workspace/full-graph FIA seq_lens are templated;
- task_update sees runtime seq_lens and applies the update;
- block_tables and slot_mapping values are reasonable for block_size 128:
  split 0 uses blocks `[1, 2, 3, 4, 7]`, split 1 uses block `[8]`;
- capture and update pointers for query, block_tables, attn_output, and
  workspace are stable for sampled first/last layers;
- `event_only` mode also fails, but this is not proof of a Python timing bug:
  event-only releases captured FIA events without patching FIA task params, so
  it can fail with stale task parameters after runtime tensor binding;
- disabling runtime mixed update seq_lens templating with
  `VLLM_ASCEND_MACRO_GRAPH_MIXED_UPDATE_TEMPLATE_SEQ_LENS=0` still fails, so
  the MTE is not directly caused by changing the prefill tail to 128.
- split-level isolation does not clear either split:
  `VLLM_ASCEND_MACRO_GRAPH_ATTENTION_UPDATE_SPLIT_FILTER=split0` updates only
  split 0 and forces split 1 to event-only, yet still fails with the same FIA
  MTE;
  `VLLM_ASCEND_MACRO_GRAPH_ATTENTION_UPDATE_SPLIT_FILTER=split1` updates only
  split 1 and forces split 0 to event-only, yet still fails with the same FIA
  MTE.

This narrows the current failure away from environment selection, missing
mixed flags, block table refresh values, slot_mapping values, pointer rebinding,
the Python replay/update launch order, and a single obvious split-local
metadata error.

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

The Phase 3 failure is now narrowed to the `npugraph_ex` mixed macrograph FIA
task-update contract.

The correct update region must be FIA-only. Updating the whole
`macro_unified_attention` custom op is unsafe because it includes unsupported
operators. Conversely, vLLM-side manual update of internal FIA handles is not a
stable backend contract for `npugraph_ex` macrograph replay.

The next investigation should focus on whether the mixed macrograph records and
updates the intended FIA task handles with the exact runtime `seq_len`/tiling
inputs, not on Python replay/update timing.

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
