# Phase 10 Inplace Split Correctness Analysis

Date: 2026-05-24

This note tracks the current correctness failure for `inplace_serial` fixed batch
split. It is intentionally separate from the implementation plan and phase
completion report because the current NPU result is still a correctness failure.

## Current Status

The latest useful both-run artifact is:

```text
/tmp/vllm_ascend_inplace_phase10_lowmem/20260524_081934
```

Command shape:

```bash
python examples/test_split_batch_correctness_npu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split-mode inplace_serial \
  --fixed-batch-size 416 \
  --capture-sizes 256,384,512 \
  --max-tokens 8 \
  --validate-ptrs \
  --force-fixed-prompts \
  --expect-split 384,32 \
  --run both \
  --compare-mode subprocess \
  --gpu-memory-utilization 0.82 \
  --output-dir /tmp/vllm_ascend_inplace_phase10_lowmem
```

Result:

```text
summary.json status: FAIL
mismatch_total: 311 / 416
```

This means phase 10 is not complete. The enabled-only path can run, and the
lazy capture/replay trace looks structurally correct, but output equivalence
against the no-split baseline is not satisfied.

## Trace Facts

`split_trace_summary.json` reports no trace validation failures:

```text
failures: []
expected split observations: 5 steps
observed split histogram:
  384+32@384: 5
  384+31@384: 1
offset graph:
  lazy_capture_count: 1
  inferred_replay_count: 4
  unexpected_capture_count: 0
```

The `384+31@384` tail is expected near the end of generation as requests finish.
It should not be treated as the primary correctness failure.

Split-1 pointer stability is also passing:

```text
input_ids:       stable
positions:       stable
query_start_loc: stable
seq_lens:        stable
block_tables:    stable
slot_mapping:    stable
```

Conclusion: the current failure is not explained by repeated lazy capture,
missing offset replay, or split-1 metadata pointer drift.

## Output Diff Pattern

The serialized output comparison shows:

```text
mismatch_total: 311 / 416
mismatches before request index 384: 309
mismatches in request index 384..415: 2
```

First mismatching generated token position:

```text
position 1:  30 requests
position 2: 235 requests
position 3:   2 requests
position 4:  42 requests
position 5:   1 request
position 6:   1 request
```

Contiguous request intervals:

```text
0..0      mismatch, first mismatch token 3
1..91     mismatch, first mismatch token 2
92..92    mismatch, first mismatch token 3
93..236   mismatch, first mismatch token 2
237..265  mismatch, first mismatch token 1
266..266  mismatch, first mismatch token 5
267..295  match
296..337  mismatch, first mismatch token 4
338..413  match
414..414  mismatch, first mismatch token 6
415..415  mismatch, first mismatch token 1
```

Token-position equality across all 416 requests:

```text
generated token 0: 416 / 416 equal
generated token 1: 386 / 416 equal
generated token 2: 151 / 416 equal
generated token 3: 149 / 416 equal
generated token 4: 107 / 416 equal
generated token 5: 106 / 416 equal
generated token 6: 105 / 416 equal
generated token 7: 106 / 416 equal
```

Interpretation:

- The first generated token is identical for all requests.
- Divergence starts from later decode steps, mostly token position 2.
- The mismatch is concentrated in request indexes `<384`, i.e. the split-0
  request range.
- Some contiguous regions still match exactly, which is more consistent with a
  graph update / attention metadata ordering problem than with a simple merge or
  trim bug.

## Relevant Current Execution Semantics

For fixed batch 416 and capture sizes `256,384,512`, the inplace planner creates:

```text
split-0: 384 tokens, start_num_tokens=0
split-1: 32 tokens,  start_num_tokens=384
```

Key code paths:

- `create_inplace_split_batch_slices()` chooses the largest lower capture size
  as split-0. See `vllm_ascend/worker/ubatch_utils.py`.
- `_make_split_batch_metadata_inplace_serial()` dispatches each split separately.
  Split-0 gets the ordinary `384` graph key; split-1 gets the descriptor-aware
  offset key.
- `_run_split_batch_inplace_serial()` executes split-0 then split-1 serially on
  `stream_main`.
- `_run_inplace_serial_offset_capture()` only applies when an offset graph is
  missing. It performs warmup, capture, then immediately replays the captured
  graph and returns the replay output.
- `_update_attn_params_for_split_ubatch()` updates graph task attention params
  after each split replay.
- `get_graph_param_key()` uses plain `runtime_shape` for split-0, but uses the
  full `BatchDescriptor` for split-1 because `start_num_tokens > 0` and the
  descriptor has variant fields.

Important difference from no-split baseline:

```text
no-split baseline: 416 real requests replay through padded 512 graph
inplace split:     first 384 requests replay through 384 graph,
                   remaining requests replay through 32@offset384 graph
```

Even if both use the same attention backend (`fia` in this run), the graph task
params and metadata update path are not identical.

## Most Likely Problem Area

The highest-probability problem is in post-replay attention graph task update
semantics, not in lazy capture itself.

Reasons:

1. The first generated token is identical for every request. The corruption
   appears after the graph task params have been updated for later decode steps.
2. The majority of mismatches are in split-0 request indexes, not split-1.
3. Trace validation for split-1 offset graph capture/replay and pointer stability
   passes.
4. The current split update helper differs from the ordinary no-split helper:
   ordinary no-split update uses `self.update_stream`, while split uses
   `self.update_stream_main` for serial main-stream splits.
5. `update_attn_params_split()` always enables block table in-place refresh,
   including for split-0, even though split-0 uses the ordinary non-offset graph
   key.

## Working Hypotheses

### H1: split-0 should use the ordinary update path

Split-0 has `start_num_tokens=0` and dispatches to the same kind of ordinary
graph key as the non-split path. It currently goes through
`update_attn_params_split()`, which enables split-only block table refresh.

Risk:

- Refreshing the captured block table for split-0 may change semantics compared
  with the ordinary graph path.
- It may also interact badly with the later split-1 update if graph params share
  storage or update handles/events.

Diagnostic change:

- For `start_num_tokens == 0`, call ordinary `update_attn_params()` and use the
  same stream as the no-split path.
- Keep `update_attn_params_split()` only for offset split descriptors.

### H2: serial split update stream should match no-split ordering

No-split uses `self.update_stream`. The current serial split path uses
`self.update_stream_main` when `parallel_streams=False`.

Risk:

- The graph task update stream may have different ordering relative to
  `stream_main` and graph replay than the no-split path.
- The failure pattern begins after the first token, which matches an update
  ordering problem.

Diagnostic change:

- For serial `parallel_streams=False`, use `self.update_stream`.
- Reserve `self.update_stream_parallel` for true parallel-stream split.

### H3: split-1 update may overwrite shared graph params used by split-0

Split-1 uses descriptor-aware graph keys, but both split-0 and split-1 execute
on the same main stream and currently call the same split update helper. If any
graph param tables, block table tensors, handles, or workspaces are shared
unexpectedly, split-1 update can corrupt split-0 behavior on subsequent decode
steps.

Diagnostics:

- Enable focused debug for graph param key, graph param object id, block table
  ptr, workspace ptr, handle id, and event id for split-0 vs split-1.
- Verify split-0 key is `384` and split-1 key is the offset `BatchDescriptor`.
- Verify their graph params and block table captures do not alias unless
  intentionally shared.

### H4: 384 graph is not semantically equivalent to the first 384 rows of the 512 graph

The no-split baseline runs a padded 512 graph for 416 active requests. The split
path runs split-0 through a 384 graph. Backend match alone may be insufficient:
FIA metadata, `actual_seq_lengths_q`, `actual_seq_lengths_kv`, workspaces, or
captured block table layout may depend on graph size.

Diagnostics:

- Compare split-0 update metadata against a no-split run for the same decode
  step:
  - graph param key
  - attention backend decision
  - `actual_seq_lengths_q`
  - `actual_seq_lengths_kv` after `maybe_template_fia_seq_lens`
  - block table shape and ptr
  - workspace key and ptr
- Run an experiment where split-0 is disabled or forced through eager while
  split-1 remains offset graph, if feasible.

## Recommended Next Experiments

### Experiment A: split-0 ordinary update path

Status: tried and reverted on 2026-05-24.

Patch tested:

```text
split-0 start_num_tokens == 0:
  update_attn_params(self.update_stream, ...)
  no block_table refresh

split-1 start_num_tokens > 0:
  update_attn_params_split(self.update_stream_main, ...)
  block_table refresh remains enabled
```

Verification:

```text
py_compile: PASS
tests/ut/test_inplace_split_execution_helpers.py: 11 passed
NPU both-run summary: FAIL
artifact: /tmp/vllm_ascend_inplace_phase10_fix_split0_update/20260524_085141
```

NPU result:

```text
mismatch_total: 269 / 416
previous baseline mismatch_total: 311 / 416
mismatches before request index 384: 267
mismatches in request index 384..415: 2
split_trace_summary.failures: []
offset lazy_capture_count: 1
offset inferred_replay_count: 4
```

First mismatching generated token position after the patch:

```text
position 1:  17 requests
position 2: 248 requests
position 3:   2 requests
position 5:   1 request
position 6:   1 request
```

Contiguous request intervals after the patch:

```text
0..0      mismatch, first mismatch token 3
1..91     mismatch, first mismatch token 2
92..92    mismatch, first mismatch token 3
93..249   mismatch, first mismatch token 2
250..265  mismatch, first mismatch token 1
266..266  mismatch, first mismatch token 5
267..413  match
414..414  mismatch, first mismatch token 6
415..415  mismatch, first mismatch token 1
```

Conclusion:

- This patch is not sufficient and was reverted from the workspace.
- The mismatch reduction from `311` to `269` suggests split-0 update semantics
  are relevant, but not the whole failure.
- The next step should not re-apply this as a standalone fix. It should be
  combined with finer diagnostics of graph param aliasing, update ordering, and
  split-0 vs no-split FIA metadata.

Original backups used for the revert:

```text
.codex_backups/phase10_correctness_20260524_084904/model_runner_v3.py
.codex_backups/phase10_correctness_20260524_084904/test_inplace_split_execution_helpers.py
```

### Experiment B: clone split outputs before the next serial replay

Status: tried and reverted on 2026-05-24.

Patch tested:

```text
in _run_split_batch_inplace_serial:
  trim split_result
  clone the trimmed tensor / tuple / IntermediateTensors
  append cloned output to results
```

Rationale:

`ACLGraphWrapper` returns graph-pool backed outputs. The serial split path
executes split-0, stores the returned tensor view, then replays split-1 before
merging. If split-1 replay reuses graph-pool output memory, split-0 rows can be
corrupted before `_merge_split_outputs()`.

Verification:

```text
py_compile: PASS
tests/ut/test_inplace_split_execution_helpers.py: 10 passed
NPU both-run summary: FAIL
artifact: /tmp/vllm_ascend_inplace_phase10_fix_output_clone/20260524_090409
```

NPU result:

```text
mismatch_total: 45 / 416
previous baseline mismatch_total: 311 / 416
mismatches before request index 384: 43
mismatches in request index 384..415: 2
split_trace_summary.failures: []
```

Remaining mismatch indexes:

```text
0
298..339
414
415
```

Conclusion:

- This patch is not sufficient and was reverted from the workspace.
- It is a strong signal that graph-pool output lifetime is a major contributor:
  mismatch count drops from `311` to `45`.
- The remaining 298..339 block overlaps the region that Experiment A fixed,
  which suggests output lifetime and split-0 update semantics are independent
  contributors.

Original backups used for the revert:

```text
.codex_backups/phase10_output_clone_20260524_090243/model_runner_v3.py
.codex_backups/phase10_output_clone_20260524_090243/test_inplace_split_execution_helpers.py
```

### Experiment C: clone split outputs plus split-0 ordinary update

Status: tried and reverted on 2026-05-24.

Patch tested:

```text
1. Clone each trimmed serial split output before appending to results.
2. For split-0 start_num_tokens == 0:
     update_attn_params(self.update_stream, ...)
     no block_table refresh
3. For offset split start_num_tokens > 0:
     update_attn_params_split(self.update_stream_main, ...)
     block_table refresh remains enabled
```

Verification:

```text
py_compile: PASS
tests/ut/test_inplace_split_execution_helpers.py: 12 passed
NPU both-run summary: FAIL
artifact: /tmp/vllm_ascend_inplace_phase10_fix_combo_clone_update/20260524_090954
```

NPU result:

```text
mismatch_total: 3 / 416
mismatches before request index 384: 1
mismatches in request index 384..415: 2
split_trace_summary.failures: []
offset lazy_capture_count: 1
offset inferred_replay_count: 4
observed split histogram:
  384+32@384: 5
  384+31@384: 1
```

Remaining mismatches:

```text
index 0:   first mismatch at generated token position 4
index 414: first mismatch at generated token position 6
index 415: first mismatch at generated token position 1
```

Conclusion:

- This combination is still not sufficient and was reverted from the workspace.
- The result narrows the remaining problem from broad split-0 corruption to:
  one first-row split-0 mismatch and two tail/request-finish mismatches.
- The next useful experiment should target the tail `384+31@384` step and the
  first-row split-0 case separately instead of re-testing broad update/output
  changes alone.

Original backups used for the revert:

```text
.codex_backups/phase10_combo_clone_update_20260524_090804/model_runner_v3.py
.codex_backups/phase10_combo_clone_update_20260524_090804/test_inplace_split_execution_helpers.py
```

1. Add low-overhead diagnostic logging guarded by an env var, not hot-path JSONL
   by default. Capture only a few decode steps and only the first attention
   layer.

2. Run a diagnostic patch where:

```text
split-0 start_num_tokens == 0:
  update_attn_params(self.update_stream, ...)
  no block_table refresh

split-1 start_num_tokens > 0:
  update_attn_params_split(...)
  block_table refresh remains enabled
```

3. Re-run the same both-run command with `--gpu-memory-utilization 0.82`.

4. Accept the fix only when:

```text
summary.json status == PASS
split_trace_summary.failures == []
expected split 384+32@384 observed for multiple decode steps
offset lazy capture count == 1
offset replay count or inferred replay count >= 1
```

## Current Conclusion

The current implementation has progressed past the phase 9 blockers: offset
lazy capture is happening once, replay after capture is used, and split-1 pointer
stability passes. The remaining phase 10 blocker is output equivalence.

The evidence points away from lazy capture itself and toward graph task attention
parameter update semantics, especially split-0 using split-specific update logic
and a split-specific update stream even though split-0 is an ordinary
`start_num_tokens=0` graph.
