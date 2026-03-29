# Split-Only Block Table Refresh Record (2026-03-29)

## Goal
Enable block_table in-place refresh **only** on split replay path (`gr0/gr`) to fix second-split wrong outputs while avoiding unsplit regressions.

## Files Changed
- `vllm_ascend/compilation/acl_graph.py`
- `vllm_ascend/worker/model_runner_v3.py`

## Symbol-Level Changes
- Added helper: `_refresh_block_table_in_place(...)`
- Extended internals:
  - `_update_attn_pa_params(..., refresh_block_table: bool = False)`
  - `_update_attn_fia_params(..., refresh_block_table: bool = False)`
- Added split-only API:
  - `update_attn_params_split(...)` (enables `refresh_block_table=True`)
- Split path wiring:
  - `_update_attn_params_for_split_ubatch` now calls `update_attn_params_split(...)`
  - Unsplit path `_update_attn_params_for_wrapper` still uses `update_attn_params(...)`

## Rollback
1. Revert only these files:
   - `git checkout -- vllm_ascend/compilation/acl_graph.py vllm_ascend/worker/model_runner_v3.py`
2. Or revert symbols manually by removing `update_attn_params_split` and using `update_attn_params` in split path.
