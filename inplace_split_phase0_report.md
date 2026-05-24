# inplace split 阶段 0 完成报告

阶段 0 的目标是在不改变当前 split 执行语义的前提下，冻结首版
inplace split 的支持范围，并补齐后续阶段需要的基线采集和 JSONL
诊断能力。

本阶段没有启用 inplace split，也没有修改 graph key、dispatcher key、
GraphParams key 或 split 执行路径。

## 范围冻结

首版 inplace split 只面向固定、低风险的 decode 场景。阶段 0 对暂不支持
场景做了明确记录，后续阶段不得隐式尝试支持。

| 场景 | 阶段 0 结论 | 后续动作 |
|---|---|---|
| non-uniform decode | 不进入 inplace | fallback 到 no split |
| 带 prefill | 不进入 inplace | fallback 到 no split |
| DBO active | 不进入 inplace | 保持 DBO 优先 |
| `num_splits != 2` | 首版 inplace 不支持 | planner 校验或 fallback |
| M-RoPE | 首版不支持 | 单独验证 positions slicing |
| MLA | 首版不支持 | 单独验证 MLA metadata |
| CP/PCP/DCP | 首版不支持 | 单独验证分布式 metadata |
| 多 DP rank | 第二批支持 | 记录 `_sync_metadata_across_dp()` 影响 |
| LoRA active | 暂不承诺 | 记录 `has_lora` descriptor 行为 |

## 当前代码事实

| 函数/模块 | 当前行为 | 与 inplace 的关系 | 风险 | 是否阻塞阶段 1-3 |
|---|---|---|---|---|
| `NPUModelRunner._prepare_inputs()` | 只在 uniform decode 且 DBO 未生成 ubatch 时考虑 split。 | 后续 inplace planner 的入口。 | 中 | 否 |
| `split_batch_split()` | 生成 request 对齐的 split slice，并按 capture size padding。 | inplace 可以复用 slice 结构，但 split-1 不能继续 padding。 | 中 | 否 |
| `_make_split_batch_metadata()` | 单流 split 使用原始 buffer，并按 padded split size dispatch descriptor。 | inplace serial 需要 offset-aware descriptor。 | 高 | 否 |
| `_make_split_batch_metadata_parallel_streams()` | 非首 split 绑定到 parallel buffer。 | inplace parallel 需要替换为原始 buffer offset view。 | 高 | 否 |
| `split_attn_metadata()` | 会创建新的 `query_start_loc`，request 内部切分时可能 clone `seq_lens`。 | inplace replay 需要同一 graph key 下 metadata 地址稳定。 | 高 | 否 |
| `ACLGraphWrapper.__call__()` | 主流和 parallel stream 使用独立 graph entries/pool，key 为 `BatchDescriptor`。 | lazy capture 必须把 offset 写进 key 和诊断信息。 | 高 | 否 |
| `GraphParams` | 主流/parallel 各有一套对象，但内部仍按 `int num_tokens` 索引。 | offset graph 需要 descriptor-aware key。 | 高 | 否 |

## 实际修改内容

阶段 0 已完成以下代码和工具改动。

### 1. 新增 split inplace JSONL 诊断工具

新增文件：

```text
vllm_ascend/inplace_split_debug.py
```

能力：

- 通过 `VLLM_ASCEND_SPLIT_INPLACE_DEBUG` 控制是否启用。
- 通过 `VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE` 指定 JSONL 输出路径。
- 每条事件记录 `event`、`ts_ns`、`rank`、`pid`、`step_id`。
- 提供 tensor 摘要方法，只记录指针、shape、dtype、stride、device、
  contiguity、storage offset，不 dump tensor 值。
- 提供 batch descriptor、split slice、metadata tensor 摘要方法。
- debug 关闭时 `log_event()` 直接返回，避免默认路径明显开销。

### 2. 新增环境变量声明

修改文件：

```text
vllm_ascend/envs.py
```

新增环境变量：

```text
VLLM_ASCEND_SPLIT_INPLACE_DEBUG
VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE
```

默认输出文件：

```text
/tmp/vllm_ascend_inplace_split.jsonl
```

### 3. 增加 planner 和 split slice 诊断

修改文件：

```text
vllm_ascend/worker/model_runner_v3.py
```

新增事件：

```text
split_planner_input
split_planner_decision
split_slices
```

记录内容包括：

- batch/request/token 数量
- 是否 uniform decode
- 是否 prefill
- DBO 是否生效
- ACL graph 是否可用
- main/parallel capture sizes
- split 配置
- planner 决策原因
- split-0/split-1 的 request/token 范围和 padded token 数

### 4. 增加 input buffer、metadata 和 descriptor 诊断

修改文件：

```text
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/attention/utils.py
```

新增事件：

```text
split_input_buffers
split_metadata
split_descriptor
```

记录内容包括：

- split index
- 当前路径是 `original_buffer` 还是 `parallel_buffer`
- input ids、positions、inputs embeds 的 tensor 摘要
- metadata 中常见 tensor 的地址和形状
- descriptor dispatch token 数和实际 token 数
- 是否处于 parallel stream

这些信息用于后续确认：

- split-1 当前是否仍走 parallel buffer。
- 哪些 metadata tensor 是 view，哪些是新分配。
- 固定 batch 连续 decode step 中 metadata ptr 是否漂移。

### 5. 增加 ACL graph capture/replay 诊断

修改文件：

```text
vllm_ascend/compilation/acl_graph.py
```

新增事件：

```text
acl_graph_capture
acl_graph_replay
```

记录内容包括：

- graph entry id
- batch descriptor
- ubatch 编号
- 是否 parallel stream
- runtime mode
- replay runtime shape
- capture 前后的输入 tensor 摘要

这些日志用于后续确认 split-0、split-1 分别命中哪个 graph entry，以及
lazy capture 接入后是否正确区分 offset graph。

### 6. 新增 baseline 环境快照脚本

新增文件：

```text
tools/inplace_split_phase0_baseline.py
```

默认输出：

```text
/tmp/vllm_ascend_inplace_phase0/baseline_env.json
```

记录内容包括：

- vllm-ascend git commit
- vLLM git commit
- Python 路径和版本
- torch / torch_npu / CANN 信息
- NPU 是否可用和设备名
- 关键环境变量
- model、compilation_config、split_batch_config 占位信息

### 7. 新增 debug 单测

新增文件：

```text
tests/ut/test_inplace_split_debug.py
```

覆盖内容：

- debug 关闭时不会创建 JSONL 文件。
- debug 开启时能写入 JSONL。
- tensor 摘要包含 shape、dtype、storage offset 等字段。
- metadata 摘要能识别 `query_start_loc`、`seq_lens`、`block_table_tensor`、
  `slot_mapping`、`positions`。
- `log_event()` 能使用当前 step id。

## 使用方式

开启诊断：

```bash
export VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1
export VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE=/tmp/vllm_ascend_inplace_split.jsonl
```

采集 baseline 环境：

```bash
python tools/inplace_split_phase0_baseline.py
```

默认输出：

```text
/tmp/vllm_ascend_inplace_phase0/baseline_env.json
```

## 验证结果

已执行：

```bash
python tools/inplace_split_phase0_baseline.py
python -m pytest tests/ut/test_inplace_split_debug.py
```

结果：

```text
baseline_env.json 已生成：
/tmp/vllm_ascend_inplace_phase0/baseline_env.json

tests/ut/test_inplace_split_debug.py: 4 passed
```

## 未完成或未执行项

当前工作区没有完成 NPU runtime 固定 batch 验证，因此以下项目仍标记为未执行：

- split disabled 与 split enabled correctness 对比。
- 当前 parallel-buffer split 性能基线。
- 固定 batch 下 split-0/split-1 capture/replay 的真实 NPU 日志归档。
- exact graph hit 场景的真实运行日志归档。

这些项目需要后续在固定 NPU decode 场景中运行，例如：

```text
capture_sizes=[256,384,512]
batch=416
split-0=384 actual/384 padded
split-1=32 actual/64 padded
```

## 阶段结论

阶段 0 已完成范围冻结、静态代码事实梳理、baseline 快照脚本和 JSONL
可观测性建设。默认行为保持不变；debug 关闭时不会输出诊断文件。

阶段 0 允许进入阶段 1。进入真正 inplace 实现前，仍需要在 NPU 环境补跑固定
batch correctness 和性能基线。
