# inplace split 阶段 1 完成报告

阶段 1 的目标是新增 inplace split 的配置入口和模式枚举，同时保持当前
split 默认行为不变。

本阶段只落地配置解析、静态校验、日志字段和单元测试；没有接入 inplace
planner，也没有修改 `BatchDescriptor`、`CudagraphDispatcher`、`GraphParams`
或 split 执行路径。

## 完成内容

### 1. 新增 split mode 配置

修改文件：

```text
vllm_ascend/ascend_config.py
```

在 `SplitBatchConfig` 中新增字段：

```python
mode
enable_inplace_lazy_capture
inplace_serial_first
inplace_max_remainder_tokens
inplace_validate_metadata_ptrs
```

合法 `mode`：

```text
parallel_buffer
inplace_serial
inplace_parallel
```

默认值：

| 字段 | 默认值 | 阶段 1 行为 |
|---|---:|---|
| `mode` | `parallel_buffer` | 只解析、校验和记录 |
| `enable_inplace_lazy_capture` | `True` | 只解析、校验和记录 |
| `inplace_serial_first` | `True` | 只解析、校验和记录 |
| `inplace_max_remainder_tokens` | `None` | 只解析、校验和记录 |
| `inplace_validate_metadata_ptrs` | `False` | 只解析、校验和记录 |

### 2. 保持旧配置兼容

旧配置示例：

```json
{
  "split_batch_config": {
    "enabled": true,
    "enable_parallel_streams": true,
    "num_splits": 2
  }
}
```

阶段 1 解析结果：

```text
mode = parallel_buffer
enable_parallel_streams = true
```

兼容规则：

- 不设置 `mode` 时默认 `parallel_buffer`。
- `mode=parallel_buffer` 继续尊重 `enable_parallel_streams`。
- 旧的 `enable_parallel_streams=True` 不会隐式启用 inplace。
- `mode=parallel_buffer` 下仍允许现有 `num_splits > 2` 行为。

### 3. 新增配置校验

新增校验规则：

- `mode` 必须是 `parallel_buffer`、`inplace_serial` 或 `inplace_parallel`。
- `num_splits` 仍必须 `>= 2`。
- `min_batch_size_for_split` 仍必须 `>= 1`。
- `mode=inplace_serial` 或 `mode=inplace_parallel` 时，`num_splits` 必须等于 `2`。
- 如果设置 `inplace_max_remainder_tokens`，值必须 `>= 1`。

这些校验只依赖纯配置，不检查 runtime/model/backend 状态。以下条件留到后续
planner 和执行路径阶段处理：

- 是否 full ACL graph。
- 是否 uniform decode。
- 是否 MLA / M-RoPE / CP / PCP / DCP。
- 是否多 DP rank。
- 当前 batch 是否带 prefill 或 DBO。

### 4. 扩展 split planner 输入日志

修改文件：

```text
vllm_ascend/worker/model_runner_v3.py
```

在 phase 0 已有 `split_planner_input` JSONL 事件中新增字段：

```text
split_mode
enable_inplace_lazy_capture
inplace_serial_first
inplace_max_remainder_tokens
inplace_validate_metadata_ptrs
```

这些字段只用于可观测性，不参与 planner 决策。

### 5. 新增配置单测

修改文件：

```text
tests/ut/test_ascend_config.py
```

新增测试覆盖：

- 默认配置保持 `parallel_buffer`。
- 旧配置 `enable_parallel_streams=True` 兼容。
- `parallel_buffer` 下 `num_splits=3` 仍允许。
- `inplace_serial` 可解析。
- `inplace_parallel` 可解析。
- 非法 `mode` 报错。
- inplace mode 下 `num_splits != 2` 报错。
- 非法 `inplace_max_remainder_tokens` 报错。

## 未改变内容

阶段 1 明确没有改变以下内容：

- 未新增 `BatchDescriptor.start_num_tokens`。
- 未修改 `CudagraphDispatcher.dispatch()`。
- 未修改 `GraphParams` key 类型。
- 未修改 split planner 的 split/no split 决策逻辑。
- 未修改 `_run_split_batch_gr0()` 和 `_run_split_batch_parallel()` 的选择逻辑。
- 未新增 lazy capture。
- 未启用 inplace split 执行路径。

## 验证结果

已执行：

```bash
python -m py_compile vllm_ascend/ascend_config.py tests/ut/test_ascend_config.py vllm_ascend/worker/model_runner_v3.py
python -m pytest tests/ut/test_ascend_config.py
python -m pytest tests/ut/test_inplace_split_debug.py
```

结果：

```text
tests/ut/test_ascend_config.py: 13 passed
tests/ut/test_inplace_split_debug.py: 4 passed
```

额外尝试：

```bash
python -m pytest tests/ut/worker/test_model_runner_v2.py -k split
```

结果：未执行成功。collection 阶段失败，原因是当前仓库不存在：

```text
vllm_ascend.worker.model_runner_v2
```

该失败发生在测试收集阶段，与阶段 1 改动无关。

## 风险和注意事项

- `mode=inplace_serial` 和 `mode=inplace_parallel` 目前只是声明式配置，不代表
  inplace 已经启用。
- 后续阶段接入 planner 前，需要明确这些 mode 在不满足条件时的 fallback 行为。
- 工作区已有 phase 0 诊断改动，本阶段是在其基础上追加配置字段和日志字段。

## 阶段结论

阶段 1 已完成。

当前代码已经具备 inplace split 的配置入口、模式枚举、基础静态校验和日志
可观测性，同时保持默认行为不变。可以进入阶段 2：扩展 `BatchDescriptor`，
让 graph key 能表达 `start_num_tokens` offset。
