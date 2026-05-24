# inplace split 阶段 6 完成报告

阶段 6 的目标是实现 inplace input slicing 能力，把阶段 5 的 dry-run
planner 结果转换为原始 input buffer 的 offset view 观测能力，但不启用真实
`inplace_serial` / `inplace_parallel` 执行。

本阶段已完成通用 input slicing helper、二维 positions 切片修正、inplace
dry-run view JSONL 日志、parallel-buffer 写入保护和单元测试。

## 完成内容

### 1. 新增通用 input slicing helper

修改 `vllm_ascend/attention/utils.py`，新增：

```python
slice_positions_by_token(...)
slice_model_inputs_by_token(...)
```

行为：

- 一维 positions 使用 `positions[token_slice]`。
- 二维 positions 使用 `positions[:, token_slice]`。
- `input_ids` 和 `inputs_embeds` 均按 token slice 返回 view。
- 支持 `input_ids is None` 且 `inputs_embeds is not None`。
- 不执行 clone、contiguous 或 copy。

### 2. 替换分散的 positions 切片

已替换：

- `vllm_ascend/worker/model_runner_v3.py`
  - `_slice_split_batch_inputs()` 改用 `slice_model_inputs_by_token()`。
- `vllm_ascend/attention/utils.py`
  - split attention metadata 中 positions 切片改用 `slice_positions_by_token()`。
- `vllm_ascend/ascend_forward_context.py`
  - ubatch positions 切片改用 `slice_positions_by_token()`。

这样阶段 6 后一维和二维 positions 的 token 维切片语义保持一致。

### 3. inplace dry-run input view 日志

修改 `vllm_ascend/worker/model_runner_v3.py`：

- 当 `inplace_split_plan is not None` 且 debug 开启时，构造 split input views。
- 新增 JSONL 事件：

```text
inplace_input_views
```

每个 split 记录：

- `token_start`
- `token_stop`
- `start_num_tokens`
- `num_tokens`
- `padded_num_tokens`
- `input_ids`
- `positions`
- `inputs_embeds`

tensor view 信息包括：

- `data_ptr`
- `shape`
- `dtype`
- `stride`
- `device`
- `is_contiguous`
- `storage_offset`

固定样例 `416 -> 384 + 32` 中，second split 预期：

```text
token_start = 384
token_stop = 416
start_num_tokens = 384
num_tokens = 32
padded_num_tokens = 32
input_ids.storage_offset = 384
positions.storage_offset = 384
```

### 4. debug helper 扩展

修改 `vllm_ascend/inplace_split_debug.py`：

```python
tensor_view_info(...)
```

该 helper 复用原有 `tensor_info()`，但输出字段使用 `data_ptr`，便于和阶段 6
计划中的 view debug 字段保持一致。

### 5. parallel-buffer 写入保护

修改 `vllm_ascend/worker/model_runner_v3.py`：

- 第二段提前复制到 `*_parallel_streams` 的逻辑只在
  `split_mode == "parallel_buffer"` 时执行。
- `mode in ("inplace_serial", "inplace_parallel")` 不写：
  - `input_ids_parallel_streams`
  - `positions_parallel_streams`
  - `inputs_embeds_parallel_streams`

当前阶段 inplace 仍只 dry-run，不设置旧 split execution slices，因此不会进入
真实 inplace split 执行。

## 测试覆盖

新增 `tests/ut/test_inplace_split_input_slicing.py`：

- 一维 positions offset view。
- 二维 positions token 维切片。
- `inputs_embeds` offset view。
- `input_ids=None` 场景。

扩展 `tests/ut/test_inplace_split_debug.py`：

- `tensor_view_info()` 输出 `data_ptr`。
- `tensor_view_info(None)` 返回 `None`。

## 验证结果

已执行：

```bash
python -m py_compile \
  vllm_ascend/worker/model_runner_v3.py \
  vllm_ascend/attention/utils.py \
  vllm_ascend/ascend_forward_context.py \
  vllm_ascend/inplace_split_debug.py \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_debug.py
```

结果：通过。

已执行：

```bash
python -m pytest \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_debug.py \
  tests/ut/test_inplace_split_planner.py \
  -q
```

结果：

```text
20 passed, 2 warnings
```

## 未执行内容

未执行 NPU fixed decode dry-run 日志验证。本阶段验证集中在纯 tensor helper、
debug 输出和语法检查。

## 未改变内容

阶段 6 明确未改变以下内容：

- 未构造 stable attention metadata buffer。
- 未启用真实 `inplace_serial` 执行路径。
- 未触发 split-1 offset lazy capture。
- 未启用 `inplace_parallel` 并发执行。
- 未放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP 支持。
- 未改变 `parallel_buffer` 的 copy 和 padding 语义。

## 阶段结论

阶段 6 已完成。

下一步进入阶段 7：实现 inplace metadata stable buffer，确保 split-1 的
`query_start_loc`、`seq_lens`、`slot_mapping` 等 attention metadata tensor
地址在相同 offset graph key 下稳定。
