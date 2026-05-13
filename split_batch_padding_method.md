# Split Batch Padding Method

## Goal

Keep split execution enabled and make the split path handle unstable decode
shapes such as `67 + 64` by padding the odd split to the next supported shape:

- actual split: `67, 64`
- execution shape: `68, 64`
- merged output: trim back to `67, 64`

## Implementation

1. `vllm_ascend/worker/ubatch_utils.py`
   - In the no-cudagraph branch of `split_batch_split`, set each
     `SplitBatchSlice.padded_num_tokens` to `round_up(num_tokens, 2)`.
   - This makes `67` execute as `68`, while `64` stays `64`.
   - In the cudagraph branch, keep using the smallest capture size that is
     greater than or equal to the split size. With capture sizes including
     `64` and `68`, the same `67 -> 68`, `64 -> 64` rule is used.

2. `vllm_ascend/worker/model_runner_v4.py`
   - Do not zero-pad split 0 inside `_prepare_inputs`, because for `67 + 64`
     the shared full-batch buffer index `67` is split 1's first real token.
   - Add `_copy_and_pad_split_attn_metadata` to create split-local padded
     metadata without mutating the original metadata tree.
   - In `_run_split_batch_gr`, build split-local context and model inputs with
     `padded_num_tokens`, then trim each split output to its actual token count
     before merging.
   - If split 0 is padded in the shared prefix buffer, restore the overwritten
     full-batch range before slicing split 1. This keeps split 1's real input
     tokens intact while still letting split 0 execute at shape `68`.
   - In `_run_split_batch_parallel_two_graph_v1`, apply the same padding to
     the first lane so graph replay receives an actual `68`-token input rather
     than a `67`-token view with a `68`-token descriptor.
   - In the two-graph path, restore the overwritten shared-buffer padding range
     immediately after lane 0 replay and again in `finally` as a cleanup guard.

## Data Flow

For the failing decode step:

1. Split planning produces actual split sizes `67` and `64`.
2. The split descriptors keep the actual token counts for slicing and merging.
3. Execution shapes are set to `68` and `64`.
4. Split 0 input, positions, attention metadata, and batch descriptor are padded
   to `68`.
5. Split 1 remains `64` and continues to use its own real input slice.
6. Outputs are trimmed back to the actual split sizes before merge, so the final
   hidden state length is still `67 + 64`.

## Verification Notes

- Syntax check:
  `python -m py_compile vllm_ascend/worker/model_runner_v4.py vllm_ascend/worker/ubatch_utils.py`
- no-ACL split log:
  `/vllm-workspace/split_pad_debug_after_restore.log`
  - second decode split 0: `actual_num_tokens=67`, `num_tokens=68`
  - second decode split 1: `actual_num_tokens=64`, `num_tokens=64`
- ACL two-graph split log:
  `/vllm-workspace/split_pad_acl_after_restore.log`
  - second decode lane 0: `first_num_tokens=68`, `input_ids_shape=[68]`
  - second decode lane 1: `second_num_tokens=64`, `input_ids_shape=[64]`
  - merge trims by `sorted_actual_tokens=[67, 64]`
- Current strict correctness still has one remaining mismatch at prompt index 6
  on the third generated token:
  - disabled: `[220, 20, 12]`, text ` 5-`
  - enabled: `[220, 20, 47604]`, text ` 5 syll`
  This mismatch remains after the verified `67 -> 68` padding, so it is not the
  missing-padding failure mode.
