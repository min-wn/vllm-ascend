# inplace split 阶段 5 完成报告

阶段 5 的目标是实现 inplace split planner，只生成 2-way split dry-run 计划，
不接入真实 inplace 执行。

本阶段已完成 planner helper、runner dry-run 接入、JSONL 字段扩展和单元测试。
`parallel_buffer` 旧路径继续使用现有 split planner；`inplace_serial` 和
`inplace_parallel` 只记录计划结果，不设置会触发现有 split execution 的
`split_ubatch_slices` / `split_batch_slices`。

## 完成内容

### 1. 扩展 split slice 表达

修改 `vllm_ascend/worker/ubatch_utils.py`：

- `SplitBatchSlice` 新增 `start_num_tokens: int = 0`。
- 新增 `graph_num_tokens` property，当前返回 `padded_num_tokens`。
- 旧构造调用不需要改参，默认 offset 为 0。

这为后续阶段的 descriptor-aware offset graph 准备了统一 slice 表达。

### 2. 新增 inplace planner helper

新增：

```python
create_inplace_split_batch_slices(...)
InplaceSplitPlan
```

planner 采用 token-first 规则：

```text
first = max(capture_size < total_tokens and capture_size % q == 0)
second = total_tokens - first
```

成功样例：

```text
q=1
capture_sizes=[256,384,512]
total=416

first token_slice = [0,384)
second token_slice = [384,416)
first start_num_tokens = 0
second start_num_tokens = 384
first graph_num_tokens = 384
second graph_num_tokens = 32
```

### 3. runner dry-run 接入

修改 `vllm_ascend/worker/model_runner_v3.py`：

- 读取 `split_batch_config.mode` 后拆分 planner 分支。
- `mode=parallel_buffer` 保持现有 request-based split planner。
- `mode in ("inplace_serial", "inplace_parallel")` 调用 inplace planner。
- inplace planner 成功时只记录 `inplace_split_dry_run`，不启用真实 split 执行。
- dry-run 日志中增加目标 descriptor 预期字段：
  - `num_tokens`
  - `num_reqs`
  - `uniform`
  - `has_lora`
  - `start_num_tokens`

### 4. debug 输出扩展

修改 `vllm_ascend/inplace_split_debug.py`：

- `split_slices_info()` 新增：
  - `graph_num_tokens`
  - `start_num_tokens`

runner 中 `split_slices` 事件新增：

- `is_inplace`
- `dry_run`

## fallback reason

阶段 5 已覆盖以下 planner reason：

```text
inplace_split_dry_run
no_split_inplace_disabled
no_split_non_uniform_decode
no_split_dbo_active
no_split_no_aclgraph
no_split_num_splits_not_two
no_split_exact_graph_hit
no_split_above_max_capture_size
no_split_no_lower_capture_size
no_split_first_not_request_aligned
no_split_second_empty
no_split_remainder_too_large
no_split_spec_decode
no_split_prefill_or_mixed
no_split_lora
no_split_mrope
no_split_mla
no_split_pcp_or_context_parallel
no_split_no_capture_sizes
no_split_invalid_query_len
```

## 测试覆盖

新增 `tests/ut/test_inplace_split_planner.py`：

- `416 -> 384 + 32`
- `q=2` request/token 边界换算
- exact graph hit fallback
- batch above max capture size fallback
- no lower capture size fallback
- query len 对齐过滤
- aligned lower capture size 选择
- `inplace_max_remainder_tokens` fallback

扩展 `tests/ut/test_inplace_split_debug.py`：

- `split_slices_info()` 输出 `graph_num_tokens`
- `split_slices_info()` 输出 `start_num_tokens`

## 验证结果

已执行：

```bash
python -m pytest tests/ut/test_inplace_split_planner.py tests/ut/test_inplace_split_debug.py tests/ut/test_ascend_config.py -q
```

结果：

```text
27 passed, 2 warnings
```

已执行：

```bash
python -m py_compile \
  vllm_ascend/worker/ubatch_utils.py \
  vllm_ascend/inplace_split_debug.py \
  vllm_ascend/worker/model_runner_v3.py \
  tests/ut/test_inplace_split_planner.py \
  tests/ut/test_inplace_split_debug.py
```

结果：通过。

## 未改变内容

阶段 5 明确未改变以下内容：

- 未启用真实 inplace input offset view。
- 未构造 stable metadata buffer。
- 未触发 second split lazy capture。
- 未启用 inplace serial replay。
- 未启用 inplace parallel 并发执行。
- 未放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP 支持。

## 阶段结论

阶段 5 已完成。

下一步进入阶段 6：实现 inplace input slicing。阶段 6 需要把 dry-run planner
结果真正转成 original buffer offset view，并继续避免复用 parallel-buffer 的
复制和 padding 语义。
