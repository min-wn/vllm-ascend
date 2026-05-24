# vllm-ascend inplace split 策略实施计划

## 1. 背景和目标

本文档制定将 GPU `DUAL_INPLACE` split 策略迁移到 vllm-ascend 的分阶段实施方案。

输入依据：

- `/vllm-workspace/GPU_inplace模式实现调研.md`
- `/vllm-workspace/gpu代码总结.md`
- `/vllm-workspace/vllm-ascend/model_runner_v3_split_report.md`
- 当前 vllm-ascend 代码静态调研结果

当前 vllm-ascend 的 split-batch 已经实现了近似 `DUAL_PARALLEL` 的策略：

- split-0 使用主输入 buffer。
- split-1 拷贝到 `input_ids_parallel_streams` / `positions_parallel_streams` / `inputs_embeds_parallel_streams`。
- split-1 按 parallel capture size padding。
- 主流和 parallel stream 各自使用独立 graph pool、graph entries、GraphParams。

要新增的 inplace split 策略不是替代当前 parallel-buffer split，而是新增第三条路径：

```text
no split
parallel_buffer split      当前已有方案
inplace_serial split       第一阶段目标
inplace_parallel split     第二阶段优化目标
```

inplace split 的核心目标：

1. 大 uniform decode batch 拆成两个 split。
2. split-0 命中已有 ACL graph capture size。
3. split-1 使用真实 remainder，不 padding。
4. split-1 不复制到第二套完整输入 buffer，而是直接使用原始 input buffer 的 offset view。
5. split-1 的 graph key 必须包含 offset，即 `start_num_tokens=split_0_num_tokens`。
6. split-1 首次遇到某个 `(num_tokens, start_num_tokens)` 组合时 lazy capture，后续 replay。
7. 所有被 ACL graph 捕获或 graph task update 使用的 metadata tensor 地址必须稳定。

## 2. 最重要的技术判断

GPU inplace 的 `start_num_tokens` 主要解决 CUDA Graph entry key 区分问题。

在 vllm-ascend 上，仅加 `BatchDescriptor.start_num_tokens` 不够，还必须同步解决两个 Ascend 特有问题：

### 2.1 GraphParams key 必须 descriptor-aware

当前 Ascend full graph attention 的 `GraphParams` 按 `runtime_shape: int` 索引：

```python
graph_params.attn_params[num_tokens]
graph_params.handles[num_tokens]
graph_params.events[num_tokens]
graph_params.workspaces[num_tokens]
```

inplace 后，以下两个 graph 不应共用同一个 attention task params 桶：

```text
32 tokens at start=256
32 tokens at start=384
```

如果仍用 `32` 作为 key，就可能复用错误的 graph task handle、workspace、block table 或 seq_lens 相关参数。

因此必须把 GraphParams 的 key 从单纯 `int` 扩展为：

```text
普通 graph: int num_tokens
inplace offset graph: BatchDescriptor 或 GraphParamKey(num_tokens, start_num_tokens, stream_kind)
```

### 2.2 第二段 metadata 必须使用稳定地址

`split_attn_metadata()` 会创建新 tensor，尤其是 `query_start_loc - start` 这种操作。ACL graph replay 要求同一个 graph key 下被捕获的 tensor 地址稳定。

因此 split-1 必须有固定 metadata buffer 或等价机制，至少覆盖：

- `query_start_loc`
- `seq_lens`
- backend 实际捕获或 graph task update 使用的 block table 字段
- backend 实际捕获或 graph task update 使用的 slot mapping 字段

不能只依赖临时 tensor。

## 3. 总体落地顺序

推荐按下面顺序推进：

1. 阶段 0：冻结目标范围，建立基线和可观测性。
2. 阶段 1：新增配置和模式枚举，不改变默认行为。
3. 阶段 2：扩展 `BatchDescriptor`，让 graph key 能表达 offset。
4. 阶段 3：扩展 `CudagraphDispatcher`，支持 inplace offset lazy key。
5. 阶段 4：扩展 Ascend `GraphParams`，支持 descriptor-aware key。
6. 阶段 5：实现 inplace split planner，只产出 2-way split。
7. 阶段 6：实现 inplace input slicing，不 padding second。
8. 阶段 7：实现 inplace metadata stable buffer。
9. 阶段 8：实现 `inplace_serial` 执行路径。
10. 阶段 9：补全 lazy capture 安全开关和调试断言。
11. 阶段 10：固定 batch correctness 验证。
12. 阶段 11：实现 `inplace_parallel` 并发路径。
13. 阶段 12：性能 benchmark、灰度开关和回退策略。

每一阶段都应能独立验证，避免一次性引入 split planner、lazy capture、metadata buffer、GraphParams key、双流并发五类变量。

## 4. 阶段 0：基线和范围确认

### 4.1 目标

阶段 0 的目标不是实现 inplace，而是在不改变当前执行语义的前提下，把后续实现所需的事实、基线和观测能力一次性补齐。

阶段 0 完成后，应能回答四个问题：

1. 当前一次 decode step 为什么 split 或为什么不 split。
2. split 后每一段的真实 token/request 范围、padding 后 token 数、graph descriptor 和执行路径是什么。
3. 当前 parallel-buffer split 在固定场景下的 correctness、capture/replay 和性能基线是什么。
4. 后续引入 inplace 后，是否能通过同一套日志直接定位 planner、descriptor、GraphParams、metadata 地址、lazy capture 中的哪一环出错。

阶段 0 明确不做以下事情：

- 不新增 `BatchDescriptor.start_num_tokens`。
- 不修改 `CudagraphDispatcher.dispatch()` 语义。
- 不修改 `GraphParams` key 类型。
- 不改变当前 split planner 的决策结果。
- 不新增 inplace 执行路径。
- 不把当前默认模式从 parallel-buffer 改成 inplace。

### 4.2 第一版范围冻结

后续 `inplace_serial` 的第一版目标范围先冻结为：

```text
uniform decode
full ACL graph
no DBO
num_splits=2
普通 attention
非 M-RoPE
非 MLA
非 PCP/DCP
单 DP rank 优先
```

阶段 0 要把不在第一版范围内的场景记录为 fallback 或 deferred，而不是在实现中隐式尝试支持。

需要在阶段 0 产出一张范围表：

| 场景 | 阶段 0 结论 | 后续动作 |
|---|---|---|
| non-uniform decode | 不进入 inplace | fallback no split |
| with prefill | 不进入 inplace | fallback no split |
| DBO active | 不进入 inplace | 保持 DBO |
| `num_splits != 2` | 不进入 inplace | 启动校验或 planner fallback |
| M-RoPE | 第一版不支持 | 专项验证 positions slicing |
| MLA | 第一版不支持 | 专项验证 MLA metadata |
| CP/PCP/DCP | 第一版不支持 | 专项验证 distributed metadata |
| 多 DP rank | 第二批支持 | 先记录 `_sync_metadata_across_dp()` 影响 |
| LoRA active | 暂不承诺 | 记录 `has_lora` descriptor 行为 |

### 4.3 需要确认的代码事实

当前关键文件：

- `vllm_ascend/worker/model_runner_v3.py`
- `vllm_ascend/worker/ubatch_utils.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/ascend_config.py`
- `vllm_ascend/compilation/acl_graph.py`
- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/attention/mla_v1.py`
- `vllm_ascend/attention/attention_cp.py`
- `vllm_ascend/attention/mla_cp.py`
- `vllm/vllm/forward_context.py`
- `vllm/vllm/v1/cudagraph_dispatcher.py`

当前 vllm-ascend split 关键事实：

- `_prepare_inputs()` 中只在 `uniform_decode and ubatch_slices is None` 时 split。
- 当前 graph-aware split planner 已经倾向 2-way split，`custom_split_sizes` 也是二元列表。
- 当前 `enable_parallel_streams=True` 时，split-1 使用 parallel buffer。
- 当前 `ACLGraphWrapper` 已有主/parallel 两套 graph entries 和 graph pool。
- 当前 `GraphParams` 已有主/parallel 两套对象，但每套内部仍按 `int num_tokens` 索引。
- `split_attn_metadata()` 会创建新的 `query_start_loc` tensor，`seq_lens` 在 request 内部切分时也可能 clone。
- `_make_split_batch_metadata()` 单流路径会按 `split_slice.padded_num_tokens` dispatch descriptor，但 `_run_split_batch_gr0()` 后续更新 attention params 的 runtime shape 需要重点核对。
- `_make_split_batch_metadata_parallel_streams()` 非首 split 会把 input/position/embed 绑定到 parallel buffer。
- `ACLGraphWrapper.__call__()` 已有 capture/replay 入口和输入地址校验，是后续补 descriptor 和 ptr 日志的核心位置。

阶段 0 需要把以上事实落成一份短报告或文档小节，至少记录：

```text
函数名
当前行为
与 inplace 的关系
风险等级
是否阻塞阶段 1-3
```

### 4.4 基线快照

在加任何日志前，先记录当前工作区和运行环境，避免后续调试时无法复现实验条件。

建议记录到：

```text
/tmp/vllm_ascend_inplace_phase0/baseline_env.json
```

字段：

```json
{
  "vllm_ascend_git_commit": "...",
  "vllm_git_commit": "...",
  "python": "...",
  "torch": "...",
  "torch_npu": "...",
  "cann": "...",
  "device": "...",
  "model": "...",
  "compilation_config": {},
  "split_batch_config": {},
  "env": {
    "VLLM_USE_V1": "...",
    "VLLM_WORKER_MULTIPROC_METHOD": "...",
    "ASCEND_RT_VISIBLE_DEVICES": "...",
    "VLLM_ASCEND_ACLGRAPH_DIAG": "...",
    "VLLM_ASCEND_PERF_STATS_FILE": "..."
  }
}
```

如果没有 NPU 环境，阶段 0 仍可完成静态代码事实、单元测试和日志设计，但 NPU correctness/perf 基线要标记为未执行。

### 4.5 可观测性设计

在不改变行为前，先补 split 专用 JSONL 日志。建议新增环境变量：

```text
VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1
```

在 debug 开启时写 JSONL：

```text
VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE=/tmp/vllm_ascend_inplace_split.jsonl
```

实现要求：

- 默认关闭，关闭时不能引入明显 runtime overhead。
- 每行是一条 JSON object，带 `event`、`ts_ns`、`rank`、`pid`、`step_id`。
- tensor 指针只记录整数地址、shape、dtype、stride、device，不 dump tensor 内容。
- 日志写入失败只打 warning，不能影响默认推理。
- 同一 decode step 的 planner、metadata、graph 事件可以通过 `step_id` 或递增计数关联。

建议复用已有诊断开关：

- `VLLM_ASCEND_SPLIT_METADATA_DEBUG_FILE`
- `VLLM_ASCEND_ACLGRAPH_DIAG`
- `VLLM_ASCEND_ACL_GRAPH_DEBUG_FILE`
- `VLLM_ASCEND_PERF_STATS_FILE`

但 inplace 迁移相关事件建议独立写入 `VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE`，避免和已有临时 debug 文件混在一起。

### 4.6 日志事件清单

#### 4.6.1 planner 入口事件

位置：

```text
vllm_ascend/worker/model_runner_v3.py::_prepare_inputs()
```

事件名：

```text
split_planner_input
```

字段：

```json
{
  "event": "split_planner_input",
  "num_reqs": 416,
  "total_num_scheduled_tokens": 416,
  "num_tokens_unpadded": 416,
  "num_tokens_padded": 416,
  "uniform_decode": true,
  "uniform_decode_query_len": 1,
  "with_prefill": false,
  "enable_dbo": false,
  "use_aclgraph": true,
  "cudagraph_capture_sizes": [256, 384, 512],
  "parallel_capture_sizes": [64, 128],
  "split_enabled": true,
  "enable_parallel_streams": true,
  "num_splits": 2,
  "min_batch_size_for_split": 4,
  "force_split": false
}
```

#### 4.6.2 planner 决策事件

位置：

```text
vllm_ascend/worker/model_runner_v3.py::_prepare_inputs()
```

事件名：

```text
split_planner_decision
```

字段：

```json
{
  "event": "split_planner_decision",
  "decision": "split",
  "reason": "padding_saved_above_threshold",
  "main_reqs": 384,
  "parallel_reqs": 32,
  "custom_split_sizes": [384, 32],
  "original_padded": 512,
  "original_padding": 96,
  "remainder_padded": 64,
  "remainder_padding": 32,
  "padding_saved": 64,
  "threshold": 0
}
```

`decision` 建议使用固定枚举：

```text
split
no_split_exact_graph_hit
no_split_dbo_active
no_split_non_uniform
no_split_padding_saving_too_small
no_split_above_max_capture_size
no_split_no_capture_size
no_split_invalid_custom_split
```

#### 4.6.3 split slice 事件

位置：

```text
vllm_ascend/worker/model_runner_v3.py::_prepare_inputs()
vllm_ascend/worker/ubatch_utils.py::split_batch_split()
```

事件名：

```text
split_slices
```

字段：

```json
{
  "event": "split_slices",
  "num_splits": 2,
  "splits": [
    {
      "idx": 0,
      "request_start": 0,
      "request_stop": 384,
      "token_start": 0,
      "token_stop": 384,
      "num_requests": 384,
      "num_tokens": 384,
      "padded_num_tokens": 384
    },
    {
      "idx": 1,
      "request_start": 384,
      "request_stop": 416,
      "token_start": 384,
      "token_stop": 416,
      "num_requests": 32,
      "num_tokens": 32,
      "padded_num_tokens": 64
    }
  ]
}
```

#### 4.6.4 input buffer 事件

位置：

```text
vllm_ascend/worker/model_runner_v3.py::_prepare_inputs()
vllm_ascend/worker/model_runner_v3.py::_slice_split_batch_inputs()
vllm_ascend/worker/model_runner_v3.py::_make_split_batch_metadata_parallel_streams()
```

事件名：

```text
split_input_buffers
```

字段：

```json
{
  "event": "split_input_buffers",
  "idx": 1,
  "path": "parallel_buffer",
  "input_ids": {"ptr": 123, "shape": [64], "dtype": "torch.int32"},
  "positions": {"ptr": 456, "shape": [64], "dtype": "torch.int64"},
  "inputs_embeds": {"ptr": 789, "shape": [64, 4096], "dtype": "torch.float16"},
  "source_token_start": 384,
  "source_token_stop": 416,
  "num_tokens": 32,
  "padded_num_tokens": 64
}
```

阶段 0 只记录当前 parallel-buffer 行为；后续 inplace 引入后，同一事件的 `path` 可变成 `original_offset_view`。

#### 4.6.5 metadata 事件

位置：

```text
vllm_ascend/worker/model_runner_v3.py::_make_split_batch_metadata()
vllm_ascend/worker/model_runner_v3.py::_make_split_batch_metadata_parallel_streams()
vllm_ascend/attention/utils.py::split_attn_metadata()
```

事件名：

```text
split_metadata
```

字段：

```json
{
  "event": "split_metadata",
  "idx": 1,
  "num_tokens": 32,
  "padded_num_tokens": 64,
  "num_reqs": 32,
  "query_start_loc": {"ptr": 123, "shape": [33], "dtype": "torch.int32"},
  "seq_lens": {"ptr": 456, "shape": [32], "dtype": "torch.int32"},
  "block_table_tensor": {"ptr": 789, "shape": [32, 16], "dtype": "torch.int32"},
  "slot_mapping": {"ptr": 1000, "shape": [32], "dtype": "torch.int32"},
  "positions": {"ptr": 1001, "shape": [32], "dtype": "torch.int64"}
}
```

阶段 0 要用该事件确认：

- split-1 的哪些 metadata tensor 是 view。
- 哪些是新分配 tensor。
- 同一个固定 batch 连续 decode step 中，metadata ptr 是否漂移。

#### 4.6.6 descriptor 事件

位置：

```text
vllm_ascend/worker/model_runner_v3.py::execute_model()
vllm_ascend/worker/model_runner_v3.py::_make_split_batch_metadata()
vllm_ascend/worker/model_runner_v3.py::_make_split_batch_metadata_parallel_streams()
```

事件名：

```text
split_descriptor
```

字段：

```json
{
  "event": "split_descriptor",
  "idx": 1,
  "dispatch_num_tokens": 64,
  "actual_num_tokens": 32,
  "runtime_mode": "FULL",
  "batch_descriptor": {
    "num_tokens": 64,
    "num_reqs": 64,
    "uniform": true,
    "has_lora": false
  },
  "in_parallel_streams": true
}
```

阶段 0 先记录现状。阶段 2 后该事件必须包含 `start_num_tokens`。

#### 4.6.7 ACL graph 事件

位置：

```text
vllm_ascend/compilation/acl_graph.py::ACLGraphWrapper.__call__()
```

事件名：

```text
acl_graph_capture
acl_graph_replay
```

在已有 `_append_acl_graph_debug()` 基础上补齐：

```json
{
  "event": "acl_graph_replay",
  "runtime_mode": "FULL",
  "batch_descriptor": "...",
  "in_parallel_streams": true,
  "entry_exists_before_call": true,
  "graph_pool": "parallel",
  "input_ptrs": [],
  "metadata_ptrs": []
}
```

阶段 0 不要求完整 metadata ptr 校验，只要求能看出每个 split 是 capture 还是 replay，以及命中的 entry key。

#### 4.6.8 GraphParams 事件

位置：

```text
vllm_ascend/compilation/acl_graph.py::update_attn_params*
vllm_ascend/attention/attention_v1.py
vllm_ascend/attention/mla_v1.py
vllm_ascend/attention/attention_cp.py
vllm_ascend/attention/mla_cp.py
```

事件名：

```text
graph_params_update
```

字段：

```json
{
  "event": "graph_params_update",
  "backend": "pa",
  "runtime_shape": 64,
  "in_parallel_streams": true,
  "has_workspace": true,
  "num_handles": 1,
  "num_events": 1,
  "attn_params_len": 1
}
```

阶段 0 只记录当前 `runtime_shape: int` 行为，为阶段 4 的 descriptor-aware key 改造提供基线。

### 4.7 固定基线场景

阶段 0 至少要覆盖以下固定场景。若硬件或模型不可用，记录为 pending，不能把 pending 当作通过。

| 编号 | 配置 | batch | 预期 |
|---|---|---:|---|
| P0-S1 | split disabled | 416 | no split baseline |
| P0-S2 | split enabled, parallel off | 416 | 当前单流 split，输出等于 S1 |
| P0-S3 | split enabled, parallel on | 416 | 当前 parallel-buffer split，输出等于 S1 |
| P0-S4 | exact graph hit | 384 | 默认不 split |
| P0-S5 | exact graph hit + force | 384 或 512 | 强制 split，记录 second padding |
| P0-S6 | below threshold | 3 | 不 split |
| P0-S7 | DBO active | 416 | DBO 优先，split 不触发 |
| P0-S8 | `num_splits=3` | 416 | 记录当前行为，后续 inplace 禁止 |
| P0-S9 | non-uniform decode | mixed | 不进入 inplace 目标范围 |

推荐固定 capture sizes：

```json
{
  "compilation_config": {
    "level": 3,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [256, 384, 512],
    "cudagraph_split_pad_threshold": 0
  },
  "additional_config": {
    "split_batch_config": {
      "enabled": true,
      "num_splits": 2,
      "min_batch_size_for_split": 4,
      "enable_parallel_streams": true,
      "parallel_capture_sizes": [64, 128],
      "force_split": false
    }
  }
}
```

对于 `batch=416`，当前 parallel-buffer 目标应记录为：

```text
split-0: 384 actual, 384 padded, main graph
split-1: 32 actual, 64 padded, parallel graph
```

后续 inplace 的目标才是：

```text
split-0: 384 actual, 384 padded, start=0
split-1: 32 actual, 32 padded, start=384
```

### 4.8 correctness 基线

推荐优先使用已有脚本：

```bash
python examples/test_split_batch_correctness_npu.py \
  --model <model> \
  --max-tokens 64 \
  --batch-size 416 \
  --enable-parallel-streams \
  --parallel-capture-sizes 64,128 \
  --output-dir /tmp/vllm_ascend_inplace_phase0/correctness
```

需要保存：

```text
prompts.json
outputs_split_disabled.json
outputs_split_enabled.json
summary.json
metadata.json
console.log
vllm_ascend_inplace_split.jsonl
acl_graph_debug.jsonl
perf_stats.jsonl
```

correctness 判定：

- deterministic sampling 下 split disabled 与 split enabled 的 token ids 完全一致。
- 文本输出完全一致。
- `summary.json` 无 diff。
- 日志中 split 真实触发，不能只比较两个 no split run。

### 4.9 性能基线

阶段 0 不追求优化，只记录当前 no split、single-stream split、parallel-buffer split 的基线。

建议输出 CSV 或 JSONL：

```text
mode
batch
capture_sizes
parallel_capture_sizes
num_decode_steps
avg_prepare_ms
avg_forward_ms
avg_replay_ms
avg_tpot_ms
p50_tpot_ms
p90_tpot_ms
num_acl_graph_capture
num_acl_graph_replay
peak_memory
```

对比项：

| 模式 | 目的 |
|---|---|
| no split | 正确性和性能基准线 |
| split + parallel off | 串行 split 的 overhead 基线 |
| split + parallel on | 当前方案的性能上限参考 |
| force split | 放大 split 路径，方便收集日志 |

### 4.10 阶段 0 实施任务拆分

建议按下面顺序做，保证每一步都可以单独 review。

#### P0.1 范围和事实确认

产物：

- 更新本文档的阶段 0 范围表。
- 记录当前关键函数和风险点。
- 明确第一版 inplace 不支持的场景。

完成标准：

- reviewer 能从文档直接判断某个场景应走 inplace、parallel-buffer 还是 no split。

#### P0.2 debug helper

产物：

- 新增轻量 JSONL helper，例如 `_append_split_inplace_debug(event, payload)`。
- 新增 tensor 摘要 helper，例如 `_tensor_debug_info(t)`。
- helper 默认关闭。

完成标准：

- 环境变量关闭时不写文件。
- 环境变量开启时可写合法 JSONL。
- 写失败不影响推理。

#### P0.3 planner 日志

产物：

- `_prepare_inputs()` 输出 `split_planner_input`。
- `_prepare_inputs()` 输出 `split_planner_decision`。
- `split_batch_split()` 或 caller 输出 `split_slices`。

完成标准：

- no split 和 split 都有明确 reason。
- `batch=416` 能看出 planner 选择的 `384+32`。

#### P0.4 metadata 和 input 日志

产物：

- `_slice_split_batch_inputs()` 输出 view 的 ptr/shape。
- `_make_split_batch_metadata()` 输出每个 split 的 descriptor 和 metadata ptr。
- `_make_split_batch_metadata_parallel_streams()` 输出 parallel buffer rebind 后的 ptr。

完成标准：

- 能区分 split-1 当前是否使用 parallel buffer。
- 能看出 split metadata 中哪些 tensor 地址跨 step 变化。

#### P0.5 ACL graph 和 GraphParams 日志

产物：

- `ACLGraphWrapper.__call__()` 日志包含完整 `BatchDescriptor` 字符串和 capture/replay 状态。
- `update_attn_params*()` 或统一入口记录 `runtime_shape` 和 `in_parallel_streams`。

完成标准：

- 能确认 split-0、split-1 分别命中了 main/parallel graph entry。
- 能确认 GraphParams 当前仍按 `int num_tokens` 访问。

#### P0.6 基线运行和归档

产物：

- `/tmp/vllm_ascend_inplace_phase0/` 下保存 correctness、perf、debug logs。
- 写一份 `phase0_baseline_summary.md`，总结每个场景是否通过。

完成标准：

- 至少有一个固定 batch 场景触发当前 split。
- 至少有一个 exact graph hit 场景证明默认不 split。
- 若 NPU 不可用，summary 明确写出未执行原因。

### 4.11 阶段 0 验收标准

- 当前无 split、当前 parallel-buffer split 行为不变。
- 默认环境变量关闭时，无新增日志文件、无行为变化。
- 可在日志中看到当前 split 的 planner、slice、descriptor、metadata ptr、input ptr、ACL graph capture/replay 和 GraphParams key 信息。
- 能选出并跑通一个固定 batch 的验证场景，例如 `capture_sizes=[256,384,512]`、`batch=416`。
- split disabled 与 split enabled 的 deterministic correctness 对比通过。
- `batch=384` 这类 exact graph hit 默认不 split 的行为被日志证明。
- `num_splits > 2`、M-RoPE、MLA、CP/PCP/DCP、多 DP rank 等场景的第一版处理策略已经明确。
- 阶段 1 可以在不重新调研的情况下开始实现配置开关。

### 4.12 阶段 0 退出产物

阶段 0 结束时应提交或归档：

```text
inplace_split_implementation_plan.md          阶段 0 详细计划已更新
phase0_code_fact_report.md                    当前代码事实和风险点
phase0_baseline_summary.md                    correctness/perf/debug 结果摘要
/tmp/vllm_ascend_inplace_phase0/*.jsonl       原始 debug 日志
/tmp/vllm_ascend_inplace_phase0/*.json        baseline 环境和 correctness 结果
```

如果阶段 0 发现现有 parallel-buffer split 本身有 correctness 或 graph key 问题，应先修现有问题，再进入 inplace 阶段 1。

## 5. 阶段 1：新增配置和模式

### 5.1 阶段目标

阶段 1 只建立 inplace split 的配置面、模式语义和静态校验，不引入
planner、descriptor offset、metadata buffer、GraphParams key 或执行路径改动。

本阶段完成后应满足：

- 旧配置和默认配置行为完全不变，仍然走当前 `parallel_buffer` split 或 no split。
- 新配置能表达后续三条 split 路径：`parallel_buffer`、`inplace_serial`、`inplace_parallel`。
- 非法 inplace 配置在启动阶段明确失败，不等到 decode 运行时才暴露。
- phase 0 JSONL 日志能记录 `mode` 和 inplace 相关开关，便于后续阶段定位配置是否生效。
- `mode=inplace_serial` / `mode=inplace_parallel` 在阶段 1 只能作为“声明式开关”存在，不改变当前执行路径。

明确不做：

- 不新增 `BatchDescriptor.start_num_tokens`。
- 不修改 `CudagraphDispatcher.dispatch()` 签名。
- 不修改 `GraphParams` key 类型。
- 不改变 `_prepare_inputs()` 的 split planner 结果。
- 不改变 `_run_split_batch_gr0()` 或 `_run_split_batch_parallel()` 的选择逻辑。
- 不新增 lazy capture 行为。

### 5.2 输入依据

阶段 1 基于阶段 0 的范围冻结：

```text
uniform decode
full ACL graph
no DBO
num_splits=2
普通 attention
非 M-RoPE
非 MLA
非 PCP/DCP
单 DP rank 优先
```

这些限制在阶段 1 不需要全部做运行时探测，但配置层必须先把
`num_splits=2`、inplace mode 名称、lazy capture 开关等基础约束固定下来。

### 5.3 修改文件

必须修改：

```text
vllm_ascend/ascend_config.py
tests/ut/test_ascend_config.py
```

建议修改：

```text
vllm_ascend/worker/model_runner_v3.py
tests/ut/test_inplace_split_debug.py
```

`model_runner_v3.py` 只补日志字段，不改变分支条件和执行路径。

### 5.4 配置 schema

在 `SplitBatchConfig` 中新增字段：

```python
self.mode: str = str(split_batch_config.get("mode", "parallel_buffer"))
self.enable_inplace_lazy_capture: bool = bool(
    split_batch_config.get("enable_inplace_lazy_capture", True))
self.inplace_serial_first: bool = bool(
    split_batch_config.get("inplace_serial_first", True))
self.inplace_max_remainder_tokens: int | None = ...
self.inplace_validate_metadata_ptrs: bool = bool(
    split_batch_config.get("inplace_validate_metadata_ptrs", False))
```

字段含义：

| 字段 | 默认值 | 阶段 1 行为 | 后续使用阶段 |
|---|---:|---|---|
| `mode` | `parallel_buffer` | 只做解析、校验和日志记录 | 阶段 5、8、11 |
| `enable_inplace_lazy_capture` | `True` | 只做配置记录 | 阶段 3、9 |
| `inplace_serial_first` | `True` | 只做配置记录；用于声明先串行验证再并行 | 阶段 8、11 |
| `inplace_max_remainder_tokens` | `None` | 只做类型和范围校验 | 阶段 3、5、9 |
| `inplace_validate_metadata_ptrs` | `False` | 只做配置记录 | 阶段 7、9、10 |

合法 `mode`：

```text
parallel_buffer
inplace_serial
inplace_parallel
```

`mode` 命名约束：

- `parallel_buffer` 表示当前已有行为，包含单流 split 和 parallel-stream split 的现有实现。
- `inplace_serial` 表示后续只使用主流顺序执行两个 split。
- `inplace_parallel` 表示后续使用原始 buffer offset view，并发执行 second split。
- 阶段 1 不新增 `auto`，避免过早引入策略选择和回退语义。

### 5.5 兼容规则

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

阶段 1 必须解析为：

```text
mode = parallel_buffer
enable_parallel_streams = true
```

兼容规则：

- 未设置 `mode` 时，默认 `parallel_buffer`，所有现有 split 行为不变。
- `mode=parallel_buffer` 时，继续尊重 `enable_parallel_streams`：
  - `enable_parallel_streams=False`：当前单流 split。
  - `enable_parallel_streams=True`：当前 parallel-buffer split。
- `mode=inplace_serial` 时，阶段 1 不改变执行路径；后续阶段实际接线前必须在 runner 入口显式 fallback 到当前路径或直接禁用 split。
- `mode=inplace_parallel` 时，阶段 1 不改变执行路径；后续阶段实际接线前必须显式要求 `enable_inplace_lazy_capture=True` 和 full ACL graph。
- 不允许通过 `enable_parallel_streams=True` 隐式启用 inplace。

### 5.6 校验规则

在 `SplitBatchConfig.__init__()` 完成类型转换后做校验。

基础校验：

```python
valid_modes = ("parallel_buffer", "inplace_serial", "inplace_parallel")
if self.mode not in valid_modes:
    raise ValueError(
        "split_batch_config.mode must be one of "
        f"{valid_modes}, got {self.mode!r}")

if self.num_splits < 2:
    raise ValueError("split_batch_config.num_splits must be >= 2")

if self.min_batch_size_for_split < 1:
    raise ValueError(
        "split_batch_config.min_batch_size_for_split must be >= 1")
```

inplace 专属校验：

```python
if self.mode.startswith("inplace") and self.num_splits != 2:
    raise ValueError(
        "inplace split currently supports split_batch_config.num_splits=2 only")

if self.mode == "inplace_parallel" and self.inplace_serial_first:
    # 阶段 1 不强制失败；只保留语义：实现和验证顺序仍以 serial 为先。
    pass
```

`inplace_max_remainder_tokens` 解析规则：

```python
raw = split_batch_config.get("inplace_max_remainder_tokens", None)
if raw is None:
    self.inplace_max_remainder_tokens = None
else:
    self.inplace_max_remainder_tokens = int(raw)
    if self.inplace_max_remainder_tokens < 1:
        raise ValueError(
            "split_batch_config.inplace_max_remainder_tokens must be >= 1")
```

不在阶段 1 校验的条件：

- 是否 full ACL graph。
- 是否普通 attention。
- 是否 MLA / M-RoPE / CP / PCP / DCP。
- 是否多 DP rank。
- 当前 batch 是否 uniform decode。

原因：这些依赖 runtime/model/backend 状态，应该在阶段 5 planner 和阶段 8 执行路径中做 fallback 或失败保护。

### 5.7 日志扩展

在 phase 0 已有 `split_planner_input` 事件中补充：

```json
{
  "split_mode": "parallel_buffer",
  "enable_inplace_lazy_capture": true,
  "inplace_serial_first": true,
  "inplace_max_remainder_tokens": null,
  "inplace_validate_metadata_ptrs": false
}
```

要求：

- debug 关闭时无额外文件和明显开销。
- `split_cfg is None` 时字段使用 `None` 或默认值，但不能抛异常。
- 日志字段名固定使用 `split_mode`，避免与 Python `mode` 变量混淆。
- 阶段 1 不新增 planner decision 枚举，避免暗示 inplace 已经接入 planner。

### 5.8 实施步骤

#### P1.1 配置字段落地

在 `vllm_ascend/ascend_config.py::SplitBatchConfig` 中新增字段和默认值。

完成标准：

- `SplitBatchConfig({}).mode == "parallel_buffer"`。
- `SplitBatchConfig({}).enable_inplace_lazy_capture is True`。
- `SplitBatchConfig({}).inplace_serial_first is True`。
- `SplitBatchConfig({}).inplace_max_remainder_tokens is None`。
- `SplitBatchConfig({}).inplace_validate_metadata_ptrs is False`。

#### P1.2 配置校验落地

新增 `mode`、`num_splits`、`inplace_max_remainder_tokens` 的校验。

完成标准：

- `mode="bad"` 抛出包含 `split_batch_config.mode` 的 `ValueError`。
- `mode="inplace_serial", num_splits=3` 抛出明确错误。
- `mode="inplace_parallel", num_splits=3` 抛出明确错误。
- `inplace_max_remainder_tokens=0` 抛出明确错误。
- `mode="parallel_buffer", num_splits=3` 仍保持允许，避免改变当前非 inplace 行为。

#### P1.3 旧配置兼容验证

覆盖旧配置组合：

```text
enabled=false
enabled=true, enable_parallel_streams=false
enabled=true, enable_parallel_streams=true
enabled=true, parallel_capture_sizes=[64, 128]
enabled=true, force_split=true
```

完成标准：

- 新增字段不会影响旧字段的解析结果。
- 旧测试无需大规模改写；若测试断言“所有属性”，只追加新字段断言。

#### P1.4 runner 日志补字段

在 `model_runner_v3.py::_prepare_inputs()` 的 `split_planner_input` payload 中补充配置字段。

完成标准：

- 设置 `VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1` 时，日志中能看到 `split_mode`。
- 不设置 `mode` 时，日志值为 `parallel_buffer`。
- 设置 `mode=inplace_serial` 时，日志值为 `inplace_serial`。
- 除日志字段外，planner decision 和 split slices 与阶段 0 基线一致。

#### P1.5 单元测试

优先放在 `tests/ut/test_ascend_config.py`。

建议测试项：

```text
test_split_batch_config_defaults_keep_parallel_buffer
test_split_batch_config_legacy_parallel_streams_compat
test_split_batch_config_accepts_inplace_serial
test_split_batch_config_accepts_inplace_parallel
test_split_batch_config_rejects_invalid_mode
test_split_batch_config_rejects_inplace_num_splits_not_two
test_split_batch_config_rejects_invalid_inplace_max_remainder_tokens
```

如补日志测试，可在 `tests/ut/test_inplace_split_debug.py` 或 runner 相关测试中只验证 payload 构造，不启动 NPU runtime。

#### P1.6 静态回归检查

阶段 1 至少运行：

```bash
python -m pytest tests/ut/test_ascend_config.py
python -m pytest tests/ut/test_inplace_split_debug.py
```

如果 runner 相关测试在当前环境可运行，再补：

```bash
python -m pytest tests/ut/worker/test_model_runner_v2.py -k split
```

若当前环境缺少 NPU 或依赖导致无法运行，记录失败原因；阶段 1 的核心验收仍以配置单测为准。

#### P1.7 文档和示例配置

在阶段 1 结束时补充示例配置。

当前行为：

```json
{
  "split_batch_config": {
    "enabled": true,
    "mode": "parallel_buffer",
    "enable_parallel_streams": true,
    "num_splits": 2,
    "parallel_capture_sizes": [64, 128]
  }
}
```

后续串行 inplace 声明：

```json
{
  "split_batch_config": {
    "enabled": true,
    "mode": "inplace_serial",
    "num_splits": 2,
    "enable_inplace_lazy_capture": true,
    "inplace_validate_metadata_ptrs": true
  }
}
```

阶段 1 必须注明：第二段配置只声明目标模式，尚不代表 inplace 执行路径已经启用。

### 5.9 风险和处理

| 风险 | 影响 | 阶段 1 处理 |
|---|---|---|
| 用户设置 `mode=inplace_serial` 后误以为已启用 inplace | 预期不一致 | 日志和文档明确“阶段 1 仅配置声明” |
| `enable_parallel_streams` 与 `mode` 语义混淆 | 后续接线容易误走旧并行路径 | `mode=parallel_buffer` 才解释 `enable_parallel_streams` 为现有路径 |
| 过早校验 runtime 条件 | 无 NPU/无模型的配置测试变脆 | 阶段 1 只校验纯配置条件 |
| `num_splits=3` 旧行为被误禁 | 行为回归 | 只在 `mode.startswith("inplace")` 时要求 `num_splits=2` |
| 新字段未进入 debug 日志 | 后续无法判断配置是否生效 | phase 0 `split_planner_input` 追加固定字段 |

### 5.10 阶段验收标准

- 不设置 `split_batch_config.mode` 时，`mode` 默认为 `parallel_buffer`。
- 旧配置下 `enabled`、`enable_parallel_streams`、`num_splits`、`parallel_capture_sizes`、`force_split` 的解析结果不变。
- 非法 `mode`、inplace 下 `num_splits != 2`、非法 `inplace_max_remainder_tokens` 均能明确报错。
- `mode=inplace_serial` 和 `mode=inplace_parallel` 能被解析和记录，但不会改变当前 split planner 或执行路径。
- `split_planner_input` JSONL 包含新增配置字段。
- 配置单测通过；无法运行的 NPU 相关验证必须记录为未执行，而不是通过。

### 5.11 阶段退出产物

```text
vllm_ascend/ascend_config.py              新增 inplace split 配置字段和校验
vllm_ascend/worker/model_runner_v3.py     split_planner_input 日志补字段
tests/ut/test_ascend_config.py            配置默认值、兼容和非法值测试
tests/ut/test_inplace_split_debug.py      可选：日志字段测试
inplace_split_implementation_plan.md      阶段 1 详细计划更新
```

进入阶段 2 前必须确认：阶段 1 没有改变任何 `BatchDescriptor`、dispatcher、GraphParams 或 split 执行路径行为。

## 6. 阶段 2：扩展 BatchDescriptor

阶段 2 的目标是扩展 `BatchDescriptor`，让 graph key 具备表达
`start_num_tokens` offset 的能力。

本阶段只建立 key 语义和兼容性基础，不启用 inplace split，不修改 split
planner，不新增 lazy capture，不改 `GraphParams` key，也不改变当前
parallel-buffer split 的执行路径。

### 6.1 输入依据

阶段 2 基于以下已完成内容继续推进：

- 阶段 0 已完成范围冻结和 split/graph JSONL 诊断。
- 阶段 1 已新增 `split_batch_config.mode` 和 inplace 相关配置字段。
- 总实施计划已明确：split-1 的 graph key 后续必须能表达
  `start_num_tokens=split_0_num_tokens`。

当前关键事实：

- `BatchDescriptor` 定义在 `/vllm-workspace/vllm/vllm/forward_context.py`。
- 当前字段为 `num_tokens`、`num_reqs`、`uniform`、`has_lora`。
- `ACLGraphWrapper` 使用 `ForwardContext.batch_descriptor` 作为 graph entry
  dict key。
- 当前 `CudagraphDispatcher.dispatch()` 返回的 descriptor 不包含 offset。
- `vllm_ascend/inplace_split_debug.py::batch_descriptor_info()` 目前只序列化
  已知的四个字段，阶段 2 后需要包含 `start_num_tokens`。

### 6.2 阶段边界

本阶段要做：

- 在 `BatchDescriptor` 尾部新增 `start_num_tokens: int = 0`。
- 保证默认 descriptor 的 hash/equality 行为与旧路径兼容。
- 保证非零 offset descriptor 能与同 token 数的默认 descriptor 区分。
- 保证 `relax_for_mixed_batch_cudagraphs()` 保留 offset。
- 扩展 split inplace JSONL 诊断，使 `batch_descriptor` 输出包含
  `start_num_tokens`。
- 增加纯 Python/单元测试覆盖 descriptor 兼容性和 offset key 区分能力。

本阶段不做：

- 不修改 `CudagraphDispatcher.dispatch()` 函数签名。
- 不把 `start_num_tokens` 接入 dispatcher 产出的 runtime key。
- 不新增 `allow_inplace_lazy_key` 或 lazy capture 行为。
- 不修改 `GraphParams` 的 `int num_tokens` key。
- 不修改 `_make_split_batch_metadata()` 或 split planner 的实际决策。
- 不创建真实 inplace input view 或 metadata stable buffer。
- 不启用 `inplace_serial` / `inplace_parallel` 执行路径。

这些内容分别留给阶段 3、4、5、6、7、8。

### 6.3 变更文件

预计修改：

```text
/vllm-workspace/vllm/vllm/forward_context.py
vllm_ascend/inplace_split_debug.py
```

预计新增或修改测试：

```text
/vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py
tests/ut/test_inplace_split_debug.py
```

如果 vLLM upstream 测试中已有更合适的 `BatchDescriptor` 专项测试文件，优先把
descriptor 测试放到该文件；否则放在 `test_cudagraph_dispatch.py` 中，避免新增
过散的测试入口。

### 6.4 设计方案

#### 6.4.1 BatchDescriptor 字段

当前 `BatchDescriptor` 近似为：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
```

在尾部新增字段，保持现有 positional 调用兼容：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    start_num_tokens: int = 0
```

设计约束：

- 新字段必须放在最后，保证现有 positional 四参调用兼容。
- 默认值必须是 `0`，表示普通 graph 或 batch 起始位置。
- `start_num_tokens < 0` 没有合法语义，阶段 2 可只在测试中覆盖预期；
  是否增加 runtime assert 留到阶段 3 接入 dispatcher 时统一处理。
- `NamedTuple` 的 equality/hash 会自动包含新字段，因此不需要额外实现
  `__eq__` 或 `__hash__`。

#### 6.4.2 relax_for_mixed_batch_cudagraphs

`relax_for_mixed_batch_cudagraphs()` 必须保留 offset：

```python
return BatchDescriptor(
    self.num_tokens,
    num_reqs=None,
    uniform=False,
    has_lora=self.has_lora,
    start_num_tokens=self.start_num_tokens,
)
```

原因：

- 后续阶段 3 可能允许 offset key 走 lazy capture。
- 如果 relax 后丢掉 offset，`32 tokens at start=256` 和
  `32 tokens at start=384` 会重新碰撞。

#### 6.4.3 JSONL descriptor 输出

`vllm_ascend/inplace_split_debug.py::batch_descriptor_info()` 需要追加
`start_num_tokens`。

建议实现方式：

```python
for name in ("num_tokens", "num_reqs", "uniform", "has_lora",
             "start_num_tokens"):
    ...
```

阶段 2 不要求所有日志事件新增独立顶层字段；只要求现有
`batch_descriptor` object 内能看到 `start_num_tokens`。

### 6.5 实施步骤

#### 6.5.1 预检查

执行以下静态检查，记录是否存在依赖 `BatchDescriptor` tuple 长度的代码：

```bash
rg -n "BatchDescriptor\\(" /vllm-workspace/vllm/vllm /vllm-workspace/vllm/tests vllm_ascend tests
rg -n "len\\(.*batch_descriptor|tuple\\(.*batch_descriptor|_fields|num_tokens,.*num_reqs,.*uniform,.*has_lora" /vllm-workspace/vllm/vllm vllm_ascend tests
```

如果发现四字段 unpack，例如：

```python
num_tokens, num_reqs, uniform, has_lora = batch_descriptor
```

必须改成属性访问，不能依赖 tuple arity。

#### 6.5.2 修改 BatchDescriptor

在 `/vllm-workspace/vllm/vllm/forward_context.py` 中：

- 在 `BatchDescriptor` 尾部增加 `start_num_tokens: int = 0`。
- 增加字段注释，说明该字段表示 graph key 对应的 token 起始 offset。
- 修改 `relax_for_mixed_batch_cudagraphs()`，保留 offset。

阶段 2 暂不新增 helper 方法，避免过早固定阶段 3 的 dispatcher 接口。

#### 6.5.3 修改诊断输出

在 `vllm_ascend/inplace_split_debug.py` 中：

- `batch_descriptor_info()` 增加 `start_num_tokens`。
- 保持 debug 关闭时零行为变化。
- 保持对旧 descriptor 或 mock descriptor 的兼容：如果对象没有该属性，则不输出。

#### 6.5.4 增加测试

vLLM 侧测试覆盖：

```python
def test_batch_descriptor_start_num_tokens_defaults_to_zero():
    desc = BatchDescriptor(num_tokens=32)
    assert desc.start_num_tokens == 0


def test_batch_descriptor_start_num_tokens_participates_in_key():
    first = BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                            has_lora=False, start_num_tokens=256)
    second = BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                             has_lora=False, start_num_tokens=384)
    assert first != second
    assert len({first, second}) == 2


def test_batch_descriptor_relax_preserves_start_num_tokens():
    desc = BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                           has_lora=False, start_num_tokens=384)
    relaxed = desc.relax_for_mixed_batch_cudagraphs()
    assert relaxed.num_tokens == 32
    assert relaxed.num_reqs is None
    assert relaxed.uniform is False
    assert relaxed.has_lora is False
    assert relaxed.start_num_tokens == 384
```

vllm-ascend 侧测试覆盖：

```python
def test_batch_descriptor_info_includes_start_num_tokens():
    desc = BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                           start_num_tokens=384)
    info = split_debug.batch_descriptor_info(desc)
    assert info["start_num_tokens"] == 384
```

#### 6.5.5 兼容性回归

至少确认以下旧调用仍成立：

```python
BatchDescriptor(32)
BatchDescriptor(32, 32)
BatchDescriptor(32, 32, True)
BatchDescriptor(32, 32, True, False)
BatchDescriptor(num_tokens=32, uniform=True)
```

确认 `CudagraphDispatcher.dispatch()` 未传入 offset 时返回：

```python
key.start_num_tokens == 0
```

### 6.6 验证命令

建议执行：

```bash
python -m py_compile /vllm-workspace/vllm/vllm/forward_context.py \
  vllm_ascend/inplace_split_debug.py
python -m pytest /vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py -k "BatchDescriptor or batch_descriptor"
python -m pytest tests/ut/test_inplace_split_debug.py
```

如果测试环境没有安装 editable vLLM，需显式确认 `PYTHONPATH` 指向
`/vllm-workspace/vllm`，否则测试可能导入站点包中的旧 `vllm`。

NPU runtime correctness 不是阶段 2 的必需验收项；如果能运行，可以额外补跑阶段 0
的 fixed batch parallel-buffer 基线，预期行为不变。

### 6.7 阶段验收标准

- `BatchDescriptor._fields` 包含 `start_num_tokens`。
- 默认构造的 `BatchDescriptor(...).start_num_tokens == 0`。
- 仅 `start_num_tokens` 不同的 descriptor 彼此不相等，且可作为 dict/set 的不同 key。
- `relax_for_mixed_batch_cudagraphs()` 不丢失 `start_num_tokens`。
- 当前 `CudagraphDispatcher.dispatch()` 默认返回的 key 均为 `start_num_tokens=0`。
- split inplace JSONL 中的 `batch_descriptor` object 能输出 `start_num_tokens`。
- 当前 no split 和 parallel-buffer split 行为没有被启用或改变。
- 相关单元测试通过；无法执行的 NPU 验证明确记录为未执行。

### 6.8 风险和处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 代码依赖 `BatchDescriptor` tuple 长度 | 新增字段后 unpack 失败或逻辑错位 | 阶段 2 预检查并改为属性访问 |
| `relax_for_mixed_batch_cudagraphs()` 丢 offset | offset graph key 碰撞 | 单测强制覆盖 |
| JSONL 未输出新字段 | 后续阶段无法排查 key 是否正确 | 更新 `batch_descriptor_info()` 并加单测 |
| 测试导入了旧 vLLM 包 | 假通过或假失败 | 验证 `PYTHONPATH` / import path |
| 过早修改 dispatcher | 阶段边界扩大，增加行为回归风险 | 阶段 2 明确禁止 dispatcher 签名变更 |

### 6.9 阶段退出产物

阶段 2 完成后应补充 `inplace_split_phase2_report.md`，至少记录：

```text
修改文件
新增测试
验证命令和结果
是否发现 BatchDescriptor tuple unpack
默认 dispatch key 是否仍为 start_num_tokens=0
未执行项
进入阶段 3 的条件
```

进入阶段 3 前必须确认：

- descriptor 已能表达 offset。
- dispatcher 仍不会自动产生 offset key。
- GraphParams 仍未 descriptor-aware，不能启用真实 inplace replay。

## 7. 阶段 3：扩展 CudagraphDispatcher lazy key

### 7.1 目标

阶段 3 的目标是让 `CudagraphDispatcher` 具备生成 offset graph key 的能力：

1. split-1 可以用真实 remainder token 数作为 `BatchDescriptor.num_tokens`。
2. split-1 的 graph key 必须携带 `start_num_tokens=split_0_num_tokens`。
3. 当该 offset key 不在初始化 capture sizes 中时，dispatcher 可以在显式允许的情况下把它加入 FULL graph key 集合，供后续 lazy capture 使用。
4. 默认 no split、parallel-buffer split、普通 FULL/PIECEWISE dispatch 行为保持不变。

阶段 3 只改变 dispatcher 的 key 选择能力，不启用真实 inplace split。

### 7.2 非目标

阶段 3 明确不做以下事情：

- 不修改 Ascend `GraphParams` key 类型。
- 不修改 `ACLGraphWrapper.__call__()` capture/replay 语义。
- 不修改 split planner，不让 `mode=inplace_serial` 或 `mode=inplace_parallel` 进入执行路径。
- 不做 input offset view 或 metadata stable buffer。
- 不允许普通请求因为 remainder 未命中 capture sizes 而自动 lazy capture。
- 不把 PIECEWISE 或 mixed graph 扩展成 offset lazy key。

这些内容分别留给阶段 4、5、6、7、8、9。

### 7.3 前置条件

进入阶段 3 前必须满足：

- 阶段 1 已完成配置入口，且 `enable_inplace_lazy_capture` 只是配置字段，不参与执行路径。
- 阶段 2 已完成 `BatchDescriptor.start_num_tokens`，默认值为 `0`，并参与 equality/hash。
- `BatchDescriptor.relax_for_mixed_batch_cudagraphs()` 已保留 `start_num_tokens`。
- 当前 `CudagraphDispatcher.dispatch()` 未传入 offset 时仍返回 `start_num_tokens=0`。
- 当前 Ascend `GraphParams` 仍是 `int num_tokens` key，因此阶段 3 完成后仍不能接线真实 inplace replay。

### 7.4 修改文件

```text
vllm/vllm/v1/cudagraph_dispatcher.py
vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py
```

如果 vLLM 侧 pytest 仍因环境依赖无法 collection，需要补一个不依赖 pytest
conftest 的最小 Python 校验脚本或在阶段报告中记录等价断言输出。

### 7.5 接口扩展方案

当前 dispatch 签名：

```python
dispatch(
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    disable_full: bool = False,
) -> tuple[CUDAGraphMode, BatchDescriptor]
```

阶段 3 增加两个 keyword-only 可选参数，避免旧 positional 调用误传：

```python
dispatch(
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    disable_full: bool = False,
    *,
    start_num_tokens: int = 0,
    allow_inplace_lazy_key: bool = False,
) -> tuple[CUDAGraphMode, BatchDescriptor]
```

默认值保持旧行为：

| 参数 | 默认值 | 默认行为 |
|---|---:|---|
| `start_num_tokens` | `0` | 普通 graph key，不带 offset |
| `allow_inplace_lazy_key` | `False` | 不新增 runtime key |

### 7.6 offset key 语义

只有显式满足以下条件时，dispatcher 才能生成 offset lazy key：

```python
is_inplace_offset_key = (
    allow_inplace_lazy_key
    and start_num_tokens > 0
    and uniform_decode
    and not disable_full
    and self.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
)
```

生成的 descriptor 必须使用真实 split-1 token 数，不走 padding：

```python
batch_desc = BatchDescriptor(
    num_tokens=num_tokens,
    num_reqs=num_tokens // self.uniform_decode_query_len,
    uniform=True,
    has_lora=has_lora,
    start_num_tokens=start_num_tokens,
)
```

必须校验：

- `num_tokens > 0`
- `start_num_tokens > 0`
- `uniform_decode is True`
- `num_tokens % self.uniform_decode_query_len == 0`
- `disable_full is False`
- `decode_mode() == CUDAGraphMode.FULL`
- `num_tokens <= compilation_config.max_cudagraph_capture_size`

推荐在阶段 3 先用明确 `assert` 或 `ValueError` 暴露错误调用。后续阶段 9
再补用户配置级别的 fallback 和错误消息收敛。

### 7.7 dispatch 决策顺序

阶段 3 的决策顺序建议如下：

1. 保留现有早退：

```python
if (
    not self.keys_initialized
    or self.cudagraph_mode == CUDAGraphMode.NONE
    or num_tokens > self.compilation_config.max_cudagraph_capture_size
):
    return CUDAGraphMode.NONE, BatchDescriptor(
        num_tokens=num_tokens,
        start_num_tokens=start_num_tokens,
    )
```

2. 如果 `start_num_tokens == 0` 或 `allow_inplace_lazy_key == False`，完全走旧逻辑。

3. 如果 `start_num_tokens > 0` 但 `allow_inplace_lazy_key == False`，不能偷偷新增 key，返回 no graph：

```python
return CUDAGraphMode.NONE, BatchDescriptor(
    num_tokens=num_tokens,
    start_num_tokens=start_num_tokens,
)
```

4. 如果启用 offset lazy key 且满足约束，构造 offset descriptor。

5. 如果 offset descriptor 已存在于 `cudagraph_keys[CUDAGraphMode.FULL]`，直接返回 FULL。

6. 如果 offset descriptor 不存在，调用 `add_cudagraph_key(CUDAGraphMode.FULL, batch_desc)` 后返回 FULL。

7. offset key 不参与 relaxed PIECEWISE fallback，避免 `start_num_tokens` graph 被降级为可复用的 mixed graph key。

### 7.8 为什么阶段 3 不调用 `_create_padded_batch_descriptor()`

`_create_padded_batch_descriptor()` 会把 `num_tokens` padding 到最近的 capture
size。inplace split 的 split-1 目标正是使用真实 remainder，因此阶段 3 的
offset lazy key 必须手工构造 `BatchDescriptor`。

示例：

```text
原 batch tokens: 416
split-0: 384 tokens, start=0, 命中已有 FULL graph
split-1: 32 tokens, start=384, lazy FULL graph
```

split-1 的 key 必须是：

```python
BatchDescriptor(
    num_tokens=32,
    num_reqs=32,
    uniform=True,
    has_lora=False,
    start_num_tokens=384,
)
```

不能变成：

```python
BatchDescriptor(num_tokens=256, start_num_tokens=384)
```

也不能丢失 offset：

```python
BatchDescriptor(num_tokens=32, start_num_tokens=0)
```

### 7.9 与 `disable_full` 的关系

`disable_full=True` 通常表示当前 step 因 cascade attention 等原因不能使用 FULL
graph。offset lazy key 只允许 FULL decode graph，因此：

- `disable_full=True` 时，即使 `allow_inplace_lazy_key=True`，也不得新增 offset FULL key。
- 可以返回 no graph descriptor，保留 `start_num_tokens` 方便日志定位。
- 不 fallback PIECEWISE，因为 PIECEWISE key 不能表达 inplace split 的 FULL
  decode capture/replay 需求。

### 7.10 与 LoRA 和 speculative decode 的关系

LoRA：

- `has_lora` 继续写入 `BatchDescriptor.has_lora`。
- 如果当前配置会 specialize LoRA，offset key 也必须按 `has_lora=True/False`
  区分。
- 阶段 3 只验证 key 语义，不承诺 inplace 首版支持 LoRA；阶段 5 planner 仍可
  fallback。

Speculative decode：

- `num_reqs = num_tokens // self.uniform_decode_query_len`。
- 必须断言整除，否则该 batch 不能构造 uniform decode offset key。
- 阶段 3 单测至少覆盖 `uniform_decode_query_len=1`；如果现有 helper 容易设置
  speculative config，可增加非 1 场景，否则在阶段报告中标记未覆盖。

### 7.11 建议实现步骤

1. 在 `CudagraphDispatcher.dispatch()` 增加 keyword-only 参数
   `start_num_tokens` 和 `allow_inplace_lazy_key`。
2. 保持所有旧调用不需要修改。
3. 新增私有 helper，降低主 dispatch 分支复杂度：

```python
def _create_inplace_offset_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    start_num_tokens: int,
) -> BatchDescriptor:
    ...
```

4. 新增私有判定 helper：

```python
def _can_dispatch_inplace_offset_key(
    self,
    num_tokens: int,
    uniform_decode: bool,
    disable_full: bool,
    start_num_tokens: int,
    allow_inplace_lazy_key: bool,
) -> bool:
    ...
```

5. 在 dispatch 早退后、普通 padded descriptor 逻辑前处理 offset 分支。
6. offset 分支只访问 `cudagraph_keys[CUDAGraphMode.FULL]`。
7. offset descriptor 不存在时通过 `add_cudagraph_key()` 注册。
8. 保留阶段 0/2 诊断工具对 descriptor 的输出能力，无需新增日志字段；如果要加日志，只记录 key 变化，不记录 tensor 内容。

### 7.12 单元测试计划

修改：

```text
/vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py
```

新增测试建议：

1. 默认 dispatch 行为不变：

```python
rt_mode, key = dispatcher.dispatch(
    num_tokens=8,
    uniform_decode=True,
    has_lora=False,
)
assert key.start_num_tokens == 0
```

2. offset lazy key 可注册并返回 FULL：

```python
rt_mode, key = dispatcher.dispatch(
    num_tokens=3,
    uniform_decode=True,
    has_lora=False,
    start_num_tokens=8,
    allow_inplace_lazy_key=True,
)
assert rt_mode == CUDAGraphMode.FULL
assert key == BatchDescriptor(
    num_tokens=3,
    num_reqs=3,
    uniform=True,
    has_lora=False,
    start_num_tokens=8,
)
assert key in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
```

3. offset key 不 padding：

```python
assert key.num_tokens == 3
```

即使 capture sizes 是 `[1, 8]`，也不能返回 `8`。

4. 同一 `(num_tokens, start_num_tokens, has_lora)` 重复 dispatch 不增加重复 key。

5. 相同 `num_tokens`、不同 `start_num_tokens` 是两个 key。

6. `start_num_tokens > 0` 但 `allow_inplace_lazy_key=False` 不注册 key，返回 NONE。

7. `uniform_decode=False` 且允许 lazy 时应报错或返回 NONE；阶段 3 实现必须在测试中固定一种语义。

8. `disable_full=True` 且允许 lazy 时不注册 FULL offset key。

9. `CUDAGraphMode.PIECEWISE` 或 `CUDAGraphMode.NONE` 下不能注册 offset FULL key。

10. `num_tokens > max_cudagraph_capture_size` 时保持现有 NONE 行为，并保留 descriptor offset。

11. `num_tokens % uniform_decode_query_len != 0` 时拒绝 offset key。

### 7.13 静态兼容性检查

阶段 3 实现后需要检查：

```bash
grep -R "dispatch(" -n /vllm-workspace/vllm/vllm \
  /vllm-workspace/vllm-ascend/vllm_ascend \
  /vllm-workspace/vllm/tests \
  /vllm-workspace/vllm-ascend/tests
```

关注点：

- 是否存在 positional 调用依赖旧四参之后继续传第五参。
- 是否有 wrapper/mock 复制了旧签名。
- 是否有类型检查或 monkeypatch 需要同步参数。

因为新增参数是 keyword-only，旧调用不应受影响。

### 7.14 验证命令

优先执行：

```bash
PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m py_compile \
  /vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py \
  /vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py

PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m pytest \
  /vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py \
  -k "CudagraphDispatcher or batch_descriptor"
```

如果 pytest collection 仍因为环境依赖失败，需要执行等价 Python 脚本，至少覆盖：

- 默认 key offset 为 `0`。
- offset lazy key 返回 FULL。
- offset key 被加入 FULL key set。
- offset key 不 padding。
- 未允许 lazy 时不注册 key。

### 7.15 日志和报告要求

阶段 3 完成报告应记录：

```text
修改文件
dispatch 新签名
offset lazy key 触发条件
不触发条件
新增测试
实际执行的验证命令和结果
pytest 未执行成功时的替代校验结果
```

报告中必须明确写出：

- 阶段 3 后 dispatcher 已能生成 offset key。
- 阶段 3 后 Ascend `GraphParams` 仍未 descriptor-aware。
- 阶段 3 后仍不能启用真实 `inplace_serial` replay。
- 下一阶段必须先做 GraphParams descriptor-aware key。

### 7.16 阶段验收标准

给定 capture sizes `[256, 384, 512]`：

```python
rt_mode, key = dispatcher.dispatch(
    num_tokens=32,
    uniform_decode=True,
    has_lora=False,
    start_num_tokens=384,
    allow_inplace_lazy_key=True,
)
```

应返回：

```text
CUDAGraphMode.FULL
BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                has_lora=False, start_num_tokens=384)
```

并且：

- 该 key 存在于 `dispatcher.cudagraph_keys[CUDAGraphMode.FULL]`。
- `key.num_tokens == 32`，没有 padding 到 `256`。
- `BatchDescriptor(num_tokens=32, start_num_tokens=256)` 和
  `BatchDescriptor(num_tokens=32, start_num_tokens=384)` 是两个不同 key。
- 未传 `allow_inplace_lazy_key=True` 时不会注册 offset FULL key。
- 所有旧 dispatcher 单测保持通过。

### 7.17 进入阶段 4 的条件

进入阶段 4 前必须确认：

- dispatcher 已能返回 `start_num_tokens > 0` 的 FULL key。
- offset key 不会污染普通 padded dispatch。
- offset key 不会走 PIECEWISE relaxed fallback。
- 测试覆盖旧行为和 offset 行为。
- 阶段 3 完成报告已写明真实 inplace replay 仍被 GraphParams 阻塞。

## 8. 阶段 4：扩展 Ascend GraphParams key

### 8.1 目标

让 Ascend attention graph task params、handles、events、workspace 能按 descriptor-aware key 隔离。

阶段 4 完成后，同一个真实 token 数但不同 offset 的 graph 不再共用同一个
`GraphParams` 桶：

```text
32 tokens, start_num_tokens=256
32 tokens, start_num_tokens=384
```

二者必须分别拥有独立的：

- `events`
- `handles`
- `attn_params`
- `workspaces`

本阶段只解决 GraphParams key 能力，不启用真实 inplace split。

### 8.2 阶段边界

本阶段要做：

- 扩展 `GraphParams` key 类型，使其支持 `int` 和 offset `BatchDescriptor`。
- 新增统一 helper，从 `forward_context.batch_descriptor` 推导 GraphParams key。
- 修改 attention capture 写入点，按 descriptor-aware key 写入 graph task 相关参数。
- 修改 graph task update 读取点，按 descriptor-aware key 读取 params、handles、events、workspace。
- 保持 `start_num_tokens == 0` 时继续使用原有 `int runtime_shape` key。
- 保持主流和 parallel stream 两套 `GraphParams` 对象互相隔离。
- 补充单测或最小脚本验证 key 隔离和旧行为兼容。

本阶段不做：

- 不接入 inplace split planner。
- 不实现 input offset slicing。
- 不实现 metadata stable buffer。
- 不修改 `ACLGraphWrapper` 的 capture/replay 选择语义。
- 不放宽阶段 3 的 `allow_inplace_lazy_key` 条件。
- 不支持 MTP offset graph。
- 不承诺 MLA、CP、PCP、DCP 的 inplace 正式启用，只保证相关代码路径的 key 改造不回归旧行为。

### 8.3 前置条件

进入阶段 4 前必须确认：

- 阶段 2 已完成 `BatchDescriptor.start_num_tokens`，且该字段参与 equality/hash。
- 阶段 3 已完成 dispatcher offset lazy key，能够返回
  `BatchDescriptor(..., start_num_tokens > 0)`。
- 阶段 3 完成报告已明确：真实 inplace replay 仍被 Ascend `GraphParams`
  的 `int num_tokens` key 阻塞。
- 当前默认路径仍只产生 `start_num_tokens == 0` 的 key，旧行为基线可用于回归验证。

### 8.4 修改文件

```text
vllm_ascend/compilation/acl_graph.py
vllm_ascend/attention/attention_v1.py
vllm_ascend/attention/mla_v1.py
vllm_ascend/attention/attention_cp.py
vllm_ascend/attention/mla_cp.py
```

需要重点检查的调用方：

```text
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/worker/model_runner_v1.py
vllm_ascend/worker/npu_split_wrapper.py
vllm_ascend/spec_decode/mtp_proposer.py
```

这些调用方通常只传 `runtime_shape`，阶段 4 不应要求它们新增参数。

### 8.5 key 设计

第一版采用兼容型 key：

```python
GraphParamKey = int | BatchDescriptor
```

规则：

```text
start_num_tokens == 0  -> int runtime_shape
start_num_tokens > 0   -> BatchDescriptor
```

选择该规则的原因：

- 普通 graph 保持 `int` key，最大限度降低旧路径回归风险。
- offset graph 直接复用阶段 2/3 已完成的 `BatchDescriptor` equality/hash。
- `BatchDescriptor` 已包含 `num_tokens`、`num_reqs`、`uniform`、`has_lora`、
  `start_num_tokens`，足够区分阶段 4 需要隔离的 graph params。
- 主流和 parallel stream 已经使用 `_graph_params` / `_graph_params_parallel`
  两套对象隔离，key 本身暂不需要再包含 `in_parallel_streams`。

暂不新增独立 dataclass。若后续发现 `BatchDescriptor` 字段过宽或日志不便阅读，
可在阶段 9 后再引入 `ACLGraphParamKey`，但阶段 4 不增加这类迁移成本。

### 8.6 helper 函数

在 `vllm_ascend/compilation/acl_graph.py` 中新增：

```python
GraphParamKey = int | BatchDescriptor


def get_graph_param_key(forward_context: Any,
                        runtime_shape: int,
                        *,
                        allow_mtp_offset: bool = False) -> GraphParamKey:
    if getattr(forward_context, "is_mtp_model", False) and not allow_mtp_offset:
        return runtime_shape

    desc = getattr(forward_context, "batch_descriptor", None)
    start = int(getattr(desc, "start_num_tokens", 0) or 0)
    if start > 0:
        return desc
    return runtime_shape
```

新增：

```python
def ensure_graph_param_key(graph_params: GraphParams,
                           key: GraphParamKey) -> None:
    graph_params.events.setdefault(key, [])
    graph_params.handles.setdefault(key, [])
    graph_params.attn_params.setdefault(key, [])
    graph_params.workspaces.setdefault(key, None)
```

同时新增只用于读取路径的 helper，避免 update 时静默创建空桶导致
`zip(..., [], [], [])` 直接跳过 attention task update：

```python
def require_graph_param_key(graph_params: GraphParams,
                            key: GraphParamKey,
                            *,
                            op: str) -> None:
    if key not in graph_params.attn_params:
        raise KeyError(f"Missing GraphParams key for {op}: {key!r}")
    if key not in graph_params.handles or key not in graph_params.events:
        raise KeyError(f"Incomplete GraphParams key for {op}: {key!r}")
```

读取路径必须使用 `require_graph_param_key()`，写入和 workspace 初始化路径使用
`ensure_graph_param_key()`。

建议再新增日志格式化 helper：

```python
def graph_param_key_info(key: GraphParamKey) -> dict[str, Any]:
    if isinstance(key, BatchDescriptor):
        return {
            "kind": "batch_descriptor",
            "num_tokens": key.num_tokens,
            "num_reqs": key.num_reqs,
            "uniform": key.uniform,
            "has_lora": key.has_lora,
            "start_num_tokens": key.start_num_tokens,
        }
    return {"kind": "runtime_shape", "num_tokens": int(key)}
```

该 helper 只用于 debug JSONL 和异常消息，不参与关键路径计算。

### 8.7 修改 GraphParams 类型

当前：

```python
@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
```

改成：

```python
@dataclass
class GraphParams:
    events: dict[GraphParamKey, list[torch.npu.ExternalEvent]]
    workspaces: dict[GraphParamKey, torch.Tensor | None]
    handles: dict[GraphParamKey, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[GraphParamKey, list[tuple]]
```

`_make_graph_params(capture_sizes)` 仍初始化 int key：

```python
{size: [] for size in aclgraph_capture_sizes}
```

offset key 由 lazy capture 期间 `ensure_graph_param_key()` 动态创建。

`set_mtp_graph_params()` 可继续使用同一个 `GraphParams` 类型，但阶段 4 默认
`get_graph_param_key(..., allow_mtp_offset=False)`，因此 MTP 仍只使用 `int` key。

### 8.8 修改 update_graph_params_workspaces

当前：

```python
def update_graph_params_workspaces(num_tokens: int, workspace, in_parallel_streams=False)
```

改为兼容：

```python
def update_graph_params_workspaces(key_or_num_tokens: Any,
                                   workspace: torch.Tensor,
                                   in_parallel_streams: bool = False):
    target = ...
    ensure_graph_param_key(target, key_or_num_tokens)
    target.workspaces[key_or_num_tokens] = weak_ref_tensors(workspace)
```

注意事项：

- 函数参数名建议改为 `graph_param_key` 或 `key_or_num_tokens`，避免继续暗示只能传 `int`。
- 调用方如果已经传入 `weak_ref_tensors(workspace)`，不要重复包装到改变语义。
  阶段 4 可以先保留当前调用方风格，只保证 key 正确。
- `update_mtp_graph_params_workspaces()` 不必在阶段 4 支持 `BatchDescriptor`，
  除非同时明确启用 MTP offset graph。

### 8.9 capture 写入点改造

所有 attention graph capture 写入点都需要先计算 `param_key`，再写入
`events`、`attn_params`、`handles`、`workspaces`。

以 `attention_v1.py` 为例，当前 full graph FIA：

```python
num_tokens = attn_metadata.actual_seq_lengths_q[-1]
graph_params.events[num_tokens].append(event)
graph_params.attn_params[num_tokens].append(...)
graph_params.handles[num_tokens].append(handle)
```

目标：

```python
forward_context = get_forward_context()
num_tokens = attn_metadata.actual_seq_lengths_q[-1]
param_key = get_graph_param_key(forward_context, num_tokens)
ensure_graph_param_key(graph_params, param_key)

workspace = graph_params.workspaces.get(param_key)
...
update_graph_params_workspaces(param_key, workspace, ...)
graph_params.events[param_key].append(event)
graph_params.attn_params[param_key].append(...)
graph_params.handles[param_key].append(handle)
```

需要覆盖的 capture 写入点：

| 文件 | 函数/路径 | 当前 key | 阶段 4 key |
|---|---|---|---|
| `attention_v1.py` | `full_graph_fia()` | `num_tokens` | `param_key` |
| `attention_v1.py` | `full_graph_pa()` | `num_tokens` | `param_key` |
| `mla_v1.py` | MLA decode capture | `num_tokens` | `param_key`，MTP 仍为 `num_tokens` |
| `attention_cp.py` | DCP/PCP normal attention capture | `num_tokens` | `param_key` |
| `mla_cp.py` | DCP/PCP MLA capture | `num_tokens` | `param_key` |

阶段 4 的最小安全策略：

- 普通 attention 和 MLA 路径都完成 key 改造，避免后续 fallback 场景留下隐藏旧 key。
- 对 CP/PCP/DCP 路径也完成机械改造，但 planner 在阶段 5 仍不启用这些 inplace 场景。
- 对 `forward_context.is_mtp_model` 路径保留 `int` key，防止 speculative/MTP 语义被误扩展。

### 8.10 update 读取点改造

当前 `update_attn_params_split()` 最终会读取：

```python
graph_params.attn_params[runtime_shape]
graph_params.handles[runtime_shape]
graph_params.events[runtime_shape]
```

目标：

```python
param_key = get_graph_param_key(forward_context, runtime_shape)
require_graph_param_key(graph_params, param_key, op="update_attn_params")

graph_params.attn_params[param_key]
graph_params.handles[param_key]
graph_params.events[param_key]
graph_params.workspaces.get(param_key)
```

注意：

- `runtime_shape` 仍用于 padding seq_lens 或 workspace 参数计算。
- 字典 key 使用 `param_key`。
- `using_paged_attention(runtime_shape, vllm_config)` 仍使用 `runtime_shape`，
  不使用 `param_key`。
- update 读取路径不能用 `setdefault()` 静默创建 key，缺 key 应该立即失败。

需要覆盖的 update 读取点：

| 文件 | 函数 | 改造要求 |
|---|---|---|
| `acl_graph.py` | `_update_attn_pa_params()` | 用 `param_key` 读取 PA params/handles/events，workspace 仍按 `param_key` 取 |
| `acl_graph.py` | `_update_attn_fia_params()` | 用 `param_key` 读取 FIA params/handles/events/workspace |
| `acl_graph.py` | `update_mla_attn_params()` | 非 MTP 用 `param_key`，MTP 保持 `runtime_shape` |
| `acl_graph.py` | `update_attn_dcp_pcp_params()` | 用 `param_key` 读取 DCP/PCP params/handles/events/workspace |
| `acl_graph.py` | `update_mla_attn_dcp_pcp_params()` | 用 `param_key` 读取 MLA CP params/handles/events/workspace |

### 8.11 诊断和日志改造

当前 `_extract_graph_param_block_table(runtime_shape)` 和
`_build_replay_block_table_diag(forward_context, runtime_shape)` 默认按
`runtime_shape` 查 GraphParams。阶段 4 需要改成 descriptor-aware：

```python
param_key = get_graph_param_key(forward_context, runtime_shape)
graph_params.attn_params.get(param_key)
```

建议在 `acl_graph_replay` JSONL 中新增：

```json
{
  "runtime_shape": 32,
  "graph_param_key": {
    "kind": "batch_descriptor",
    "num_tokens": 32,
    "start_num_tokens": 384
  }
}
```

对普通 graph：

```json
{
  "runtime_shape": 384,
  "graph_param_key": {
    "kind": "runtime_shape",
    "num_tokens": 384
  }
}
```

诊断日志必须保持默认关闭，不 dump tensor 内容，只记录 ptr、shape、dtype、key。

### 8.12 实施步骤

1. 在 `acl_graph.py` 新增 `GraphParamKey` 类型别名和三个 helper：
   `get_graph_param_key()`、`ensure_graph_param_key()`、
   `require_graph_param_key()`。
2. 扩展 `GraphParams` dataclass typing，并确认 `_make_graph_params()`、
   `set_graph_params()`、`set_graph_params_parallel()` 仍初始化 int key。
3. 修改 `update_graph_params_workspaces()`，让 workspace 可以写入 `int` 或
   offset `BatchDescriptor` key。
4. 修改 `attention_v1.py` 的 `full_graph_fia()` 和 `full_graph_pa()`。
   这是阶段 4 的主路径，必须先完成。
5. 修改 `acl_graph.py` 中普通 attention update 读取点：
   `_update_attn_pa_params()`、`_update_attn_fia_params()`、
   `update_attn_params()`、`update_attn_params_split()`。
6. 修改 MLA 和 CP/PCP/DCP capture/update 读写点，保持第一版不启用这些
   inplace 场景，但避免后续代码路径仍按 `runtime_shape` 误共享。
7. 修改 replay/block table 诊断函数，使日志展示 `graph_param_key`。
8. 全仓搜索 `graph_params.attn_params[`、`graph_params.handles[`、
   `graph_params.events[`、`graph_params.workspaces.get(`，
   确认除 MTP 特意保留项外没有遗漏的 `runtime_shape`/`num_tokens` key。
9. 补测试和阶段 4 完成报告。

### 8.13 静态检查清单

阶段 4 完成后，下面搜索结果需要逐条解释：

```bash
rg -n "graph_params\\.(attn_params|handles|events)\\[[^\\]]+\\]" \
  vllm_ascend/compilation/acl_graph.py \
  vllm_ascend/attention/attention_v1.py \
  vllm_ascend/attention/mla_v1.py \
  vllm_ascend/attention/attention_cp.py \
  vllm_ascend/attention/mla_cp.py
```

允许保留的情况：

- 初始化字典时使用 int capture size。
- MTP 专用 graph params 仍按 `num_tokens`。
- 明确不参与 ACL attention graph task update 的普通字典访问。

不允许保留的情况：

- 非 MTP capture 路径继续 `graph_params.attn_params[num_tokens].append(...)`。
- 非 MTP update 路径继续 `graph_params.handles[runtime_shape]`。
- workspace 在 offset graph capture 时仍写入 `workspaces[num_tokens]`。

### 8.14 测试计划

优先补不依赖 NPU runtime 的单元测试或最小 Python 校验，覆盖 key 语义：

```python
desc_1 = BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                         has_lora=False, start_num_tokens=256)
desc_2 = BatchDescriptor(num_tokens=32, num_reqs=32, uniform=True,
                         has_lora=False, start_num_tokens=384)

assert get_graph_param_key(fake_context(desc_1), 32) == desc_1
assert get_graph_param_key(fake_context(desc_2), 32) == desc_2
assert get_graph_param_key(fake_context(None), 32) == 32
assert get_graph_param_key(fake_context(
    BatchDescriptor(32, start_num_tokens=0)), 32) == 32

graph_params = GraphParams({}, {}, {}, {})
ensure_graph_param_key(graph_params, desc_1)
ensure_graph_param_key(graph_params, desc_2)
graph_params.attn_params[desc_1].append("a")
graph_params.attn_params[desc_2].append("b")
assert graph_params.attn_params[desc_1] == ["a"]
assert graph_params.attn_params[desc_2] == ["b"]
```

旧行为兼容测试：

- `start_num_tokens == 0` 时 key 仍是 `int`。
- `_make_graph_params([256, 384])` 仍创建 `256`、`384` 两个 int key。
- `update_graph_params_workspaces(256, workspace)` 仍写入 int key。
- `update_graph_params_workspaces(desc_1, workspace)` 写入 offset key，不污染
  `workspaces[32]` 或 `workspaces[256]`。
- 主流 `_graph_params` 和 parallel `_graph_params_parallel` 使用相同 offset
  descriptor 时仍写入不同对象。
- update 读取缺失 offset key 时抛 `KeyError`，不能静默跳过。

若 NPU 环境可用，额外运行：

```text
no split decode baseline
parallel-buffer split baseline
普通 FULL ACL graph capture/replay
```

阶段 4 不要求运行真实 inplace split，因为 planner、input slicing 和 metadata
stable buffer 尚未接入。

### 8.15 回归风险和处理

| 风险 | 影响 | 阶段 4 处理 |
|---|---|---|
| update 路径用 `ensure` 创建空桶 | attention task update 被静默跳过 | 读取路径必须用 `require_graph_param_key()` |
| capture 写入 offset key，update 仍读 int key | replay 读取错误 params 或直接 KeyError | capture/update helper 统一从 forward_context 推导 key |
| workspace 仍按 `num_tokens` 共享 | 不同 offset graph 复用错误 workspace | workspace key 同步改为 `param_key` |
| MTP/spec 路径被误切到 offset key | speculative 语义扩大，难以验证 | `is_mtp_model` 默认返回 int key |
| CP/PCP/DCP 改造后被误认为已支持 inplace | 支持范围混淆 | 阶段 5 planner 仍对这些场景 fallback |
| 普通 graph key 从 int 变成 descriptor | 旧 capture/replay 命中率变化 | `start_num_tokens == 0` 强制返回 int key |

### 8.16 阶段验收标准

阶段 4 必须满足：

- `GraphParams` 能同时保存 int key 和 offset `BatchDescriptor` key。
- 相同 `num_tokens`、不同 `start_num_tokens` 的 offset key 互不共享
  `events`、`handles`、`attn_params`、`workspaces`。
- 普通 no-offset graph 仍使用 int key，旧路径行为不变。
- 主流和 parallel stream 的 `GraphParams` 仍是两套对象。
- 非 MTP capture/update 读写点都通过统一 helper 获取 key。
- MTP 路径未被默认扩展到 offset key。
- 缺失 GraphParams key 的 update 路径会显式失败，不会静默跳过。
- 静态搜索确认没有遗漏的非 MTP `graph_params.*[runtime_shape]` 或
  `graph_params.*[num_tokens]` 读写点。
- `py_compile` 通过。
- 能运行的单测或最小脚本通过；不能运行的 pytest/NPU 用例必须在阶段报告中记录原因。

### 8.17 阶段退出产物

阶段 4 完成后应补充：

```text
inplace_split_phase4_report.md
```

报告至少记录：

- 修改文件清单。
- 新增 helper 和 key 规则。
- capture/update 已改造的读写点清单。
- 保留 int key 的路径和原因，尤其是 MTP。
- 静态搜索结果摘要。
- 单测、py_compile、NPU 验证结果。
- 阶段 5 仍不能启用真实 inplace 的剩余阻塞项。

进入阶段 5 前必须确认：

- dispatcher offset key 和 Ascend GraphParams offset key 语义一致。
- offset key 不再复用同一个 attention params/handle/workspace 桶。
- 默认 `parallel_buffer` split 和 no split 行为未改变。
- 真实 inplace 仍等待阶段 5 planner、阶段 6 input slicing、阶段 7 metadata stable buffer。

## 9. 阶段 5：实现 inplace split planner

### 9.1 目标

在 `model_runner_v3.py` 中增加独立的 inplace split planner，只负责判断和生成
2-way inplace split 计划：

```text
first = lower captured graph size
second = real remainder
```

阶段 5 的核心目标是把“是否适合 inplace split”和“两个 split 的 token/request
边界”算清楚，并通过 JSONL 日志可观测；不在本阶段接入真实 inplace 执行。

阶段完成后应满足：

- `parallel_buffer` 现有 split 行为不变。
- `inplace_serial` / `inplace_parallel` 可以得到 dry-run planner 结果。
- planner 输出包含 first/second 的 request slice、token slice、actual token、
  graph token 和 `start_num_tokens`。
- 若任一前置条件不满足，planner 明确 fallback，并记录可诊断 reason。
- 不会把 inplace planner 结果误交给现有 parallel-buffer split 执行路径。

### 9.2 阶段边界

本阶段只做 planner，不做以下事情：

- 不修改 input slicing 为 original buffer offset view；这是阶段 6。
- 不构造第二段 stable metadata buffer；这是阶段 7。
- 不接入 `_run_split_batch_inplace_serial()` 或真实 replay；这是阶段 8。
- 不放开 lazy capture 安全开关；这是阶段 9。
- 不启用 `inplace_parallel` 并发执行；这是阶段 12。

因此，阶段 5 的 inplace 模式应该以 dry-run / no-execute 形式落地：

```text
mode=parallel_buffer:
    继续走现有 split_batch_split + 现有执行路径

mode in (inplace_serial, inplace_parallel):
    计算 InplaceSplitPlan
    记录 split_planner_decision / split_slices
    不设置会触发现有 split execution 的 split_ubatch_slices
    forward 仍走当前 no-split 或已存在 fallback 路径
```

如果实现时希望先返回 `split_batch_slices` 供后续阶段复用，必须额外增加
`is_inplace` 或等价标记，并确保当前执行分支不会把它当作
parallel-buffer split 执行。

### 9.3 阶段 4 交接条件

阶段 4 已完成 Ascend `GraphParams` descriptor-aware key。进入阶段 5 前假设：

- `BatchDescriptor` 已能表达 `start_num_tokens`。
- dispatcher offset key 和 GraphParams offset key 语义一致。
- offset key 不再复用普通 `int` key 的 attention params / handle / workspace。
- 缺失 GraphParams key 会显式抛错，不再静默跳过 update。
- MTP、CP、PCP、DCP 虽已有部分机械改造，但 planner 第一版仍必须 fallback。

### 9.4 修改文件

```text
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/worker/ubatch_utils.py
vllm_ascend/inplace_split_debug.py
tests/ut/worker/test_inplace_split_planner.py
tests/ut/test_inplace_split_debug.py
```

若当前测试目录没有 `tests/ut/worker/`，可以按已有 worker 测试布局放置，
但测试文件名应明确包含 `inplace_split_planner`。

### 9.5 新增数据结构

建议不要复用 `SplitBatchSlice.padded_num_tokens` 表达所有含义，增加字段或新结构：

```python
@dataclass
class InplaceSplitSlice:
    request_slice: slice
    token_slice: slice
    num_tokens: int
    graph_num_tokens: int
    start_num_tokens: int
```

也可以先复用 `SplitBatchSlice`，但要明确：

- `num_tokens`: actual token count
- `padded_num_tokens`: graph token count
- `start_num_tokens`: runtime token offset，用于 dispatcher / BatchDescriptor
- inplace first 中 `start_num_tokens == 0`
- inplace second 中 `start_num_tokens == first.num_tokens`
- inplace second 中 `padded_num_tokens == num_tokens`，但 descriptor 使用
  `(num_tokens=second.num_tokens, start_num_tokens=first.num_tokens)`，
  不再把 second padding 到 next graph size

建议新建 `SplitBatchSlice.start_num_tokens: int = 0`，保持兼容。

推荐的最小改造：

```python
@dataclass
class SplitBatchSlice:
    request_slice: slice
    token_slice: slice
    padded_num_tokens: int = 0
    start_num_tokens: int = 0

    @property
    def graph_num_tokens(self) -> int:
        return self.padded_num_tokens
```

也可以新增不可变结果对象，避免与现有 parallel-buffer split 混淆：

```python
@dataclass(frozen=True)
class InplaceSplitPlan:
    slices: list[SplitBatchSlice]
    total_num_tokens: int
    padded_num_tokens_without_split: int
    reason: str
```

阶段 5 推荐至少增加一个 helper，便于单测不依赖完整 runner：

```python
def create_inplace_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    uniform_decode_query_len: int,
    cudagraph_capture_sizes: set[int],
    inplace_max_remainder_tokens: Optional[int] = None,
) -> Optional[SplitBatchSlices]:
    ...
```

### 9.6 planner 接入位置

接入点仍在 `_prepare_inputs()`，位于以下事实都已经算出之后：

- `num_scheduled_tokens`
- `total_num_scheduled_tokens`
- `num_input_tokens = pad_for_cudagraph(total_num_scheduled_tokens)`
- `uniform_decode`
- `ubatch_slices`
- DP 同步后的 `enable_dbo`
- split config 和 debug step id

推荐把现有 split 逻辑拆成两个分支：

```text
if mode == "parallel_buffer":
    run existing parallel-buffer planner
elif mode in ("inplace_serial", "inplace_parallel"):
    run inplace planner dry-run
else:
    no split
```

阶段 5 不建议在原有 parallel-buffer planner 中继续堆条件，否则容易把
`main_reqs/parallel_reqs` 和 `first_tokens/second_tokens` 两套语义混在一起。

### 9.7 planner 前置规则

只在 `_prepare_inputs()` 里满足以下条件时进入：

```text
split_config.enabled
split_config.mode in ("inplace_serial", "inplace_parallel")
uniform_decode
ubatch_slices is None
use_aclgraph
cudagraph full decode available
num_splits == 2
total_num_scheduled_tokens <= max(capture_sizes)
pad_for_cudagraph(total_num_scheduled_tokens) > total_num_scheduled_tokens
```

补充限制：

```text
with_prefill is False
attn_state == DecodeOnly
scheduled_spec_decode_tokens is empty
lora_config is None or active LoRA is absent
pcp_size == 1
pipeline / context parallel special path is not active
model_config.use_mla is False unless MLA 已完成专项验证
M-RoPE positions 未验证前不启用
```

每个失败条件都要落一个稳定 reason 字符串，避免只记录
`no_split_not_attempted`：

```text
no_split_inplace_disabled
no_split_not_inplace_mode
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
```

### 9.8 token-first 拆分算法

计算：

```python
q = self.uniform_decode_query_len
total_tokens = total_num_scheduled_tokens
capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)

first_tokens = max(
    s for s in capture_sizes
    if s < total_tokens and s % q == 0
)
second_tokens = total_tokens - first_tokens
```

请求数：

```python
first_reqs = first_tokens // q
second_reqs = num_reqs - first_reqs
```

要求：

```python
first_tokens > 0
second_tokens > 0
first_reqs > 0
second_reqs > 0
first_tokens + second_tokens == total_tokens
```

生成 slice：

```python
first = SplitBatchSlice(
    request_slice=slice(0, first_reqs),
    token_slice=slice(0, first_tokens),
    padded_num_tokens=first_tokens,
    start_num_tokens=0,
)
second = SplitBatchSlice(
    request_slice=slice(first_reqs, num_reqs),
    token_slice=slice(first_tokens, total_tokens),
    padded_num_tokens=second_tokens,
    start_num_tokens=first_tokens,
)
```

这里 `first_tokens` 必须已经是 capture size，因此 first 的 `padded_num_tokens`
等于 `graph_num_tokens`。second 是真实 remainder，`padded_num_tokens` 不再向上
padding；后续 descriptor 应使用 offset key：

```text
BatchDescriptor(
    num_tokens=second_tokens,
    num_reqs=second_reqs,
    uniform=True,
    has_lora=False,
    start_num_tokens=first_tokens,
)
```

### 9.9 capture size 选择规则

capture size 必须来自 full decode graph capture sizes，不使用 parallel-stream
capture sizes：

```python
capture_sizes = sorted(
    s for s in self.compilation_config.cudagraph_capture_sizes
    if s > 0
)
```

选择 `first_tokens` 时：

- 只允许 `s < total_tokens`，避免 exact graph hit 拆出空 remainder。
- 只允许 `s % q == 0`，避免 request 边界不对齐。
- 若 `pad_for_cudagraph(total_tokens) == total_tokens`，直接 fallback。
- 若没有合法 lower capture size，fallback。
- 若配置了 `inplace_max_remainder_tokens`，要求
  `second_tokens <= inplace_max_remainder_tokens`。

第一版使用最大合法 lower capture：

```python
first_tokens = max(valid_lower_capture_sizes)
```

这能最大化 first 的已 capture graph 复用，并最小化 second lazy key 的大小。

### 9.10 与现有 parallel-buffer planner 的关系

现有 planner 主要按 request 数拆分，并会把每个 split padding 到对应 graph size。
inplace planner 必须与它分离：

| 维度 | parallel-buffer | inplace |
| --- | --- | --- |
| split basis | request count | token count |
| first graph | padded split size | lower capture size |
| second graph | padded split size | real remainder + `start_num_tokens` |
| input buffer | copied / padded split buffer | original buffer offset view，阶段 6 |
| metadata | per split current metadata | stable offset metadata，阶段 7 |
| 阶段 5 执行 | 真实执行保持不变 | dry-run only |

`force_split` 在阶段 5 只作用于 `parallel_buffer`；不要让它绕过 inplace
前置条件。若后续需要强制 inplace dry-run，可以新增单独 debug-only 开关，
但不建议在第一版加入。

### 9.11 为什么用 token 而不是 request

当前 split 代码主要按 request 数拆分。inplace 推荐以 token 数为主，因为 ACL graph capture size 是 token size。

在 uniform decode 下：

```text
tokens = reqs * uniform_decode_query_len
```

所以可以从 token size 精确反推 request range。

如果不是 uniform decode，token range 可能落在 request 内部，会导致 request-level
metadata、slot mapping、positions、KV block table 难以保持一致。因此阶段 5 不做
mixed prefill/decode 的 token 内部切分。

### 9.12 日志要求

继续复用阶段 0 的 JSONL debug helper，并扩展 `split_slices_info()` 输出：

```text
start_num_tokens
graph_num_tokens
is_inplace
```

`split_planner_input` 至少确认已有字段：

```text
split_mode
enable_inplace_lazy_capture
inplace_serial_first
inplace_max_remainder_tokens
inplace_validate_metadata_ptrs
uniform_decode_query_len
cudagraph_capture_sizes
use_aclgraph
enable_dbo
```

`split_planner_decision` 对 inplace 追加：

```text
decision
reason
dry_run
first_tokens
second_tokens
first_reqs
second_reqs
total_tokens
padded_tokens_without_split
lower_capture_size
remainder_tokens
capture_sizes_considered
fallback_to
```

`split_slices` 对阶段 5 的固定样例应类似：

```json
{
  "num_splits": 2,
  "is_inplace": true,
  "splits": [
    {
      "idx": 0,
      "request_start": 0,
      "request_stop": 384,
      "token_start": 0,
      "token_stop": 384,
      "num_requests": 384,
      "num_tokens": 384,
      "padded_num_tokens": 384,
      "graph_num_tokens": 384,
      "start_num_tokens": 0
    },
    {
      "idx": 1,
      "request_start": 384,
      "request_stop": 416,
      "token_start": 384,
      "token_stop": 416,
      "num_requests": 32,
      "num_tokens": 32,
      "padded_num_tokens": 32,
      "graph_num_tokens": 32,
      "start_num_tokens": 384
    }
  ]
}
```

### 9.13 实施步骤

#### P5.1 保护现有 `parallel_buffer` 行为

- 在 `_prepare_inputs()` 中先读取 `split_config.mode`。
- 明确只有 `mode == "parallel_buffer"` 才进入现有 split planner。
- 为现有 planner 补一个单测或日志断言：默认配置下输出不变。

#### P5.2 扩展 slice 表达

- 给 `SplitBatchSlice` 增加 `start_num_tokens: int = 0`。
- 视需要增加 `graph_num_tokens` property，返回 `padded_num_tokens`。
- 扩展 `split_debug.split_slices_info()` 输出新字段。
- 确认现有构造调用不需要改参，默认值保持旧路径兼容。

#### P5.3 新增 token-first helper

- 在 `ubatch_utils.py` 新增 `create_inplace_split_batch_slices()` 或等价 helper。
- helper 输入只包含可单测的纯数据，避免依赖 runner。
- helper 返回 `(split_slices, reason)` 或小 dataclass，便于记录 fallback。
- 覆盖 exact hit、above max、no lower、q 不整除、remainder limit 等分支。

#### P5.4 接入 `_prepare_inputs()` dry-run

- 在 `mode in ("inplace_serial", "inplace_parallel")` 时调用 helper。
- helper 成功时记录 `decision="inplace_split_dry_run"`。
- helper 失败时记录 `decision="no_split"` 和具体 reason。
- 阶段 5 不设置会触发旧 split execution 的 `split_ubatch_slices`。
- 若为了后续阶段需要保存 planner 结果，只保存到本地变量或 runner debug 字段，
  不改变 forward 执行路径。

#### P5.5 descriptor 预校验

- planner 成功后，构造或模拟两个目标 descriptor：
  - first: `(num_tokens=first_tokens, start_num_tokens=0)`
  - second: `(num_tokens=second_tokens, start_num_tokens=first_tokens)`
- 通过 dispatcher dry-run helper 或只记录预期 key，确认 key 语义与阶段 3/4 对齐。
- 不在阶段 5 触发 lazy capture。

#### P5.6 fallback 原因收敛

- 用常量或小枚举集中管理 reason 字符串。
- 避免同一原因在不同分支记录多个拼写。
- 对用户可配置限制和运行时限制分别记录，例如
  `no_split_remainder_too_large` 与 `no_split_no_lower_capture_size`。

#### P5.7 文档同步

- 在阶段 5 完成报告中记录：
  - 新 helper 和字段。
  - dry-run 语义。
  - 每个 fallback reason。
  - 固定样例 JSONL 输出。
  - 进入阶段 6 前仍阻塞的事项。

### 9.14 单元测试计划

优先写纯函数单测，避免 NPU runtime 依赖：

```text
q=1, capture=[256,384,512], total=416
=> first=384, second=32, start=384

q=2, capture=[256,384,512], total=416
=> first=384, second=32, first_reqs=192, second_reqs=16

total=384, capture=[256,384,512]
=> no_split_exact_graph_hit

total=640, capture=[256,384,512]
=> no_split_above_max_capture_size

total=128, capture=[256,384,512]
=> no_split_no_lower_capture_size

q=3, capture=[256,384,512], total=420
=> first=384 allowed, request aligned

q=5, capture=[256,384,512], total=420
=> 384 not aligned; fallback or choose lower aligned size if present

inplace_max_remainder_tokens=16, total=416, first=384
=> no_split_remainder_too_large
```

runner 级轻量测试：

- `parallel_buffer` 默认仍进入旧 planner。
- `inplace_serial` 成功时只记录 dry-run，不设置旧 execution slices。
- DBO active 时 inplace planner fallback 为 `no_split_dbo_active`。
- spec decode / prefill / MLA / PCP 场景返回对应 reason。

debug 测试：

- `split_slices_info()` 输出 `start_num_tokens` 和 `graph_num_tokens`。
- `batch_descriptor_info()` 已包含 `start_num_tokens`，保持覆盖。
- JSONL event 中 `dry_run=true`。

### 9.15 建议验证命令

```bash
python -m pytest tests/ut/test_inplace_split_debug.py -q
python -m pytest tests/ut/worker/test_inplace_split_planner.py -q
python -m pytest tests/ut/test_ascend_config.py -q
```

如果当前环境缺少上游 vLLM 依赖导致单测无法启动，需要在阶段报告中记录
import 失败栈和未覆盖风险。

有 NPU 环境时补充 dry-run 日志验证：

```bash
VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1 \
VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE=/tmp/vllm_ascend_inplace_split_p5.jsonl \
<fixed decode workload with total tokens 416 and capture sizes 256,384,512>
```

检查：

```bash
rg '"event": "split_planner_decision"|"event": "split_slices"' \
  /tmp/vllm_ascend_inplace_split_p5.jsonl
```

### 9.16 不支持场景

第一版明确不支持：

- mixed prefill/decode
- chunked prefill
- DBO 同时启用
- `num_splits > 2`
- batch 超过最大 capture size
- first split 无法落在 request 边界
- M-RoPE 未验证前默认关闭
- MLA/PCP/DCP 未验证前可先关闭或走 fallback
- speculative decode / MTP offset graph
- LoRA graph key 组合未验证
- DP ranks 之间 total token 不一致但需要协同 inplace split 的场景

### 9.17 阶段验收标准

固定：

```text
q=1
capture_sizes=[256,384,512]
total=416
```

planner 输出：

```text
first token_slice = [0,384)
second token_slice = [384,416)
first start_num_tokens = 0
second start_num_tokens = 384
first graph_num_tokens = 384
second graph_num_tokens = 32
```

阶段 5 dry-run planner 结果中的 intended/effective token count 等于 `416`，
不是 `512`，不是 `384+128`。由于本阶段不接入真实 inplace 执行，forward
实际 `num_input_tokens` 是否改为 `416` 放到阶段 6/8 验收。

补充验收：

- `mode=parallel_buffer` 下现有 split 单测和日志保持不变。
- `mode=inplace_serial` / `mode=inplace_parallel` 只产生 dry-run planner 结果，
  不触发现有 split execution。
- JSONL 能看到成功和 fallback reason，且 reason 字符串稳定。
- `SplitBatchSlice.start_num_tokens` 默认值不影响旧调用。
- 所有新增纯函数单测通过。

### 9.18 阶段退出产物

```text
inplace_split_phase5_report.md
```

报告至少包含：

- 阶段 5 实际修改文件和 helper 签名。
- 成功样例的 planner 输出。
- fallback reason 列表。
- 单测和 dry-run 验证结果。
- 明确说明真实 inplace 仍未启用，下一阶段进入 input slicing。

## 10. 阶段 6：实现 inplace input slicing

### 10.1 目标

阶段 6 的目标是把阶段 5 的 dry-run planner 结果转成可被后续执行路径使用的
inplace 输入切片能力，但仍不启用完整 `inplace_serial` graph replay。

阶段 6 完成后应满足：

1. split-0 使用原始 input buffer 的前缀 view。
2. split-1 使用原始 input buffer 的 offset view。
3. split-1 不复制到 `*_parallel_streams` buffer。
4. split-1 不按 graph size padding，shape 等于真实 remainder。
5. 所有 input view 的 token range、shape、stride、`storage_offset()` 和
   `data_ptr()` 能通过 JSONL 日志稳定观测。

阶段 6 明确不做：

- 不构造 stable attention metadata buffer，这是阶段 7。
- 不启用 `_run_split_batch_inplace_serial()`，这是阶段 8。
- 不触发 split-1 offset lazy capture。
- 不放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP。

### 10.2 修改文件

```text
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/attention/utils.py
vllm_ascend/ascend_forward_context.py
tests/ut/test_inplace_split_input_slicing.py
tests/ut/test_inplace_split_debug.py
```

### 10.3 当前行为

当前 parallel-buffer split 会在 `_prepare_inputs()` 中提前复制第二段：

```python
self.positions_parallel_streams.gpu[:second_num_tokens].copy_(...)
self.input_ids_parallel_streams.gpu[:second_num_tokens].copy_(...)
```

inplace 模式下这一步必须跳过。

当前真实 split execution 仍依赖已有 parallel-buffer 语义：

- `_prepare_inputs()` 只有 `split_batch_slices is not None` 时才会执行第二段拷贝。
- 阶段 5 中 `mode in ("inplace_serial", "inplace_parallel")` 成功后只记录 dry-run，
  不设置 `split_batch_slices` / `split_ubatch_slices`。
- `_run_split_batch_*` 中非首 split 仍会选择 `input_ids_parallel_streams` /
  `positions_parallel_streams`。

因此阶段 6 要先把 input slicing helper 和 debug 校验准备好，再在阶段 8 接入
新的 inplace execution path。

### 10.4 目标行为

新增判断：

```python
if split_mode == "parallel_buffer":
    copy second to parallel buffers
elif split_mode.startswith("inplace"):
    do not copy input_ids / positions / inputs_embeds to parallel input buffers
```

模型输入切片：

```python
first_input_ids = input_ids[:first_tokens]
second_input_ids = input_ids[first_tokens:first_tokens + second_tokens]
```

positions：

```python
if positions.ndim == 2:
    second_positions = positions[:, first_tokens:first_tokens + second_tokens]
else:
    second_positions = positions[first_tokens:first_tokens + second_tokens]
```

inputs_embeds：

```python
second_inputs_embeds = inputs_embeds[first_tokens:first_tokens + second_tokens]
```

要求：

- `first_tokens == split_slices[0].num_tokens`。
- `second_tokens == split_slices[1].num_tokens`。
- `split_slices[1].padded_num_tokens == second_tokens`。
- `split_slices[1].start_num_tokens == first_tokens`。
- `second_*` 的 shape 第一维或 token 维等于 `second_tokens`，不是
  `pad_for_cudagraph(second_tokens)`。

### 10.5 新增统一 input slicing helper

建议新增两个小 helper，便于单测和后续阶段复用：

```python
def slice_positions_by_token(
    positions: torch.Tensor,
    token_slice: slice,
) -> torch.Tensor:
    if positions.ndim == 2:
        return positions[:, token_slice]
    return positions[token_slice]


def slice_model_inputs_by_token(
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    inputs_embeds: Optional[torch.Tensor],
    token_slice: slice,
) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    sliced_input_ids = None if input_ids is None else input_ids[token_slice]
    sliced_positions = slice_positions_by_token(positions, token_slice)
    sliced_inputs_embeds = (
        None if inputs_embeds is None else inputs_embeds[token_slice]
    )
    return sliced_input_ids, sliced_positions, sliced_inputs_embeds
```

helper 要求：

- 只返回 view，不做 `.clone()`、`.contiguous()` 或 copy。
- 不修改输入 tensor。
- 支持 `input_ids is None` 且 `inputs_embeds is not None`。
- 支持普通一维 positions 和 M-RoPE 风格二维 positions 的静态切片校验，
  即使阶段 6 仍不放开 M-RoPE 执行。

### 10.6 修改 `_slice_split_batch_inputs`

确认 `_slice_split_batch_inputs()` 对二维 positions 的切片正确。当前函数本身处理了 `positions.ndim == 2`，但其他路径如 `attention/utils.py` 和 `ascend_forward_context.py` 仍有 `positions[token_slice]` 风险。

需要统一为 helper：

```python
def slice_positions(positions: torch.Tensor, token_slice: slice):
    if positions.ndim == 2:
        return positions[:, token_slice]
    return positions[token_slice]
```

并替换：

- `model_runner_v3.py`
- `attention/utils.py`
- `ascend_forward_context.py`

替换要求：

- `_slice_split_batch_inputs()` 内部改用 `slice_model_inputs_by_token()` 或
  `slice_positions_by_token()`。
- `attention/utils.py` 中 `split_attn_metadata_one_way()` 的
  `attn_metadata.positions[token_slice]` 改用统一 helper。
- `ascend_forward_context.py` 中 ubatch positions 切片改用统一 helper。
- parallel-buffer 旧路径的行为不应改变；如仍需要 padded buffer，应继续在旧路径
  使用 parallel buffer 的 `[0:padded_tokens]` view。

### 10.7 接入 `_prepare_inputs()` 的阶段 6 dry-run view

阶段 6 仍不设置会触发旧 split execution 的 `split_batch_slices`，但当
`inplace_split_plan is not None` 时，应额外构造一次 input view dry-run：

```python
if inplace_split_plan is not None:
    inplace_input_views = [
        slice_model_inputs_by_token(
            input_ids,
            positions,
            inputs_embeds,
            split_slice.token_slice,
        )
        for split_slice in inplace_split_plan.split_slices
    ]
```

该 dry-run view 只用于：

- 验证 view shape。
- 记录 ptr / storage offset / stride。
- 证明不会写 parallel stream buffers。

该 dry-run view 不用于：

- 构造 attention metadata。
- 调用 model forward。
- dispatch ACL graph。

### 10.8 禁止写 parallel buffer 的保护

把 `_prepare_inputs()` 中第二段拷贝逻辑收窄到 `mode == "parallel_buffer"`：

```python
if split_batch_slices is not None and len(split_batch_slices) > 1:
    if split_mode == "parallel_buffer":
        copy second split to parallel stream buffers
    elif split_mode in ("inplace_serial", "inplace_parallel"):
        assert inplace_split_plan is not None
        skip parallel buffer copy
```

由于阶段 5 当前没有为 inplace 设置 `split_batch_slices`，这个分支通常不会触发。
阶段 6 仍应补上保护，避免阶段 8 接入时误复用旧拷贝路径。

保护点：

- `input_ids_parallel_streams` 不写。
- `positions_parallel_streams` 不写。
- `inputs_embeds_parallel_streams` 不写。
- 旧 `parallel_buffer` 模式仍完整保留 copy 和 padding 行为。

### 10.9 view debug 字段

扩展 `VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1` JSONL，新增事件：

```text
inplace_input_views
```

每个 split 至少记录：

```json
{
  "split_idx": 1,
  "token_start": 384,
  "token_stop": 416,
  "start_num_tokens": 384,
  "num_tokens": 32,
  "padded_num_tokens": 32,
  "input_ids": {
    "shape": [32],
    "data_ptr": 123,
    "storage_offset": 384,
    "stride": [1],
    "is_contiguous": true
  },
  "positions": {
    "shape": [32],
    "data_ptr": 456,
    "storage_offset": 384,
    "stride": [1],
    "is_contiguous": true
  },
  "inputs_embeds": null
}
```

二维 positions 记录示例：

```json
{
  "positions": {
    "shape": [3, 32],
    "storage_offset": 384,
    "stride": [max_num_tokens, 1]
  }
}
```

debug helper 可新增：

```python
def tensor_view_info(tensor: Optional[torch.Tensor]) -> Optional[dict[str, Any]]:
    ...
```

注意：`data_ptr()` 对 offset view 本身已经包含 storage offset 后的首元素地址，
同时记录 `storage_offset()` 是为了定位 stride 和二维 positions 问题。

### 10.10 地址稳定验证

同一个 `start_num_tokens` 下，以下地址必须跨 step 不变：

```python
input_ids[start:stop].data_ptr()
positions[start:stop].data_ptr()
```

M-RoPE：

```python
positions[:, start:stop].data_ptr()
```

阶段 6 的地址稳定只验证 input tensor view。attention metadata 地址稳定留到阶段 7。

连续两步相同 fixed batch 应满足：

```text
split-0 input_ids.data_ptr       相同
split-1 input_ids.data_ptr       相同
split-0 positions.data_ptr       相同
split-1 positions.data_ptr       相同
split-1 storage_offset           等于 first_tokens
split-1 num_tokens               等于真实 remainder
```

如果 `inputs_embeds is not None`，同样验证 `inputs_embeds[token_slice]`。

### 10.11 单元测试计划

新增纯 tensor 单测，不依赖 NPU runtime：

```text
1D positions:
input_ids=torch.arange(416)
positions=torch.arange(416)
split=384+32
second input_ids == input_ids[384:416]
second positions == positions[384:416]
second storage_offset == 384
second shape == [32]
```

```text
2D positions:
positions=torch.arange(3*416).reshape(3, 416)
second positions == positions[:, 384:416]
second shape == [3, 32]
second storage_offset == 384
```

```text
inputs_embeds:
inputs_embeds=torch.arange(416*hidden).reshape(416, hidden)
second shape == [32, hidden]
second storage_offset == 384 * hidden
```

```text
input_ids None:
input_ids=None
inputs_embeds 非空
helper 返回 sliced_input_ids=None
```

debug 单测：

- `tensor_view_info(None)` 返回 `None`。
- `tensor_view_info(tensor_view)` 输出 shape、data_ptr、storage_offset、stride。
- `split_slices_info()` 中 `start_num_tokens` 与 view debug 的
  `token_start` 对齐。

runner 轻量测试：

- `mode=parallel_buffer` 时仍写 parallel buffer 事件或保持旧路径日志。
- `mode=inplace_serial` dry-run 成功时记录 `inplace_input_views`。
- `mode=inplace_serial` dry-run 成功时不设置旧 split execution slices。

### 10.12 建议验证命令

```bash
python -m pytest \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_debug.py \
  tests/ut/test_inplace_split_planner.py \
  -q
```

语法验证：

```bash
python -m py_compile \
  vllm_ascend/worker/model_runner_v3.py \
  vllm_ascend/attention/utils.py \
  vllm_ascend/ascend_forward_context.py \
  vllm_ascend/inplace_split_debug.py
```

有 NPU 环境时补充 fixed decode dry-run：

```bash
VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1 \
VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE=/tmp/vllm_ascend_inplace_split_p6.jsonl \
<fixed decode workload with total tokens 416 and capture sizes 256,384,512>
```

检查：

```bash
rg '"event": "inplace_input_views"|"event": "split_slices"' \
  /tmp/vllm_ascend_inplace_split_p6.jsonl
```

### 10.13 风险和处理

| 风险 | 处理 |
|---|---|
| offset view 被后续 `.contiguous()` 隐式复制 | 阶段 6 debug 记录 `data_ptr` 和 `storage_offset`，阶段 8 接入前复核 |
| 二维 positions 被按一维切片 | 统一 `slice_positions_by_token()` 并加单测 |
| inplace 模式误写 parallel buffer | copy 分支加 `split_mode == "parallel_buffer"` 限定 |
| second split 被 padding | 验证 `padded_num_tokens == num_tokens == remainder` |
| dry-run view 生命周期不代表执行路径 | 阶段 6 只证明 slicing 语义；阶段 8 再接入真实 forward |
| metadata 中 positions 仍错误切片 | `attention/utils.py` 和 `ascend_forward_context.py` 同步替换 helper |

### 10.14 阶段验收标准

inplace 模式下：

- 不写 `input_ids_parallel_streams`。
- 不写 `positions_parallel_streams`。
- 不写 `inputs_embeds_parallel_streams`。
- second split input 的 `data_ptr()` 等于原始 buffer offset view 地址。
- second split 不 padding，shape 为真实 remainder。
- 一维 positions 和二维 positions helper 单测通过。
- `inputs_embeds` 和 `input_ids=None` helper 单测通过。
- `parallel_buffer` 旧路径 split 行为不变。
- JSONL 中能看到 `inplace_input_views`，且 split-1 的
  `storage_offset` / `token_start` / `start_num_tokens` 一致。

### 10.15 阶段退出产物

```text
inplace_split_phase6_report.md
```

报告至少包含：

- 阶段 6 修改文件。
- input slicing helper 签名。
- 固定样例 `416 -> 384 + 32` 的 view debug 输出。
- parallel-buffer 兼容性说明。
- 单测和 py_compile 结果。
- 明确说明 metadata stable buffer 和真实 inplace execution 仍未启用。

## 11. 阶段 7：实现 inplace metadata stable buffer

### 11.1 目标

阶段 7 的目标是为 inplace split-1 准备 attention metadata stable buffer，并在
dry-run 中验证 stable metadata 的地址、shape 和内容来源。

阶段 7 完成后应满足：

1. split-1 的 `query_start_loc` / `query_start_loc_cpu` 指向 runner 预分配 buffer。
2. split-1 的 `seq_lens` / `seq_lens_cpu` 指向 runner 预分配 buffer。
3. split-1 的 `slot_mapping` 指向 runner 预分配 buffer。
4. split-1 的 `block_table_tensor` 第一版仍使用原始 request slice view，并记录
   ptr，后续如地址不稳定再加 dedicated buffer。
5. inplace dry-run 能输出原始 split metadata ptr 和 stabilized metadata ptr。
6. 相同 fixed batch 连续两步下 stable metadata ptr 不变。

阶段 7 明确不做：

- 不启用真实 `inplace_serial` forward。
- 不调用 builder 构造 per-layer offset graph metadata。
- 不触发 ACL graph capture/replay。
- 不放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP。

### 11.2 修改文件

```text
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/attention/utils.py
vllm_ascend/ascend_forward_context.py
vllm_ascend/inplace_split_debug.py
tests/ut/test_inplace_split_metadata_stabilization.py
tests/ut/test_inplace_split_debug.py
```

### 11.3 新增 buffer

在 runner 初始化中新增轻量 metadata buffer：

```python
self.inplace_query_start_loc_secondary = self._make_buffer(
    self.max_num_reqs + 1, dtype=torch.int32)

self.inplace_seq_lens_secondary = self._make_buffer(
    self.max_num_reqs, dtype=torch.int32)

self.inplace_num_computed_tokens_secondary = torch.empty(
    self.max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True)
```

block table 和 slot mapping 视 backend 决定：

第一版建议也准备固定 slot mapping buffer：

```python
self.inplace_slot_mapping_secondary = self._make_buffer(
    self.max_num_tokens, dtype=torch.int64)
```

block table 可以先用原始 block table 的 request slice view。如果地址断言失败，再增加：

```python
self.inplace_block_table_secondary_per_kv_group
```

阶段 7 第一版只新增 secondary buffer，不新增 per-kv-group block table buffer。
原因是第一版支持范围固定为普通 decode attention，block table 的 request slice view
通常来自长期存在的 block table storage；先记录 ptr，待阶段 8/10 的 capture
验证确认是否需要复制。

### 11.4 stable metadata 构造流程

当前逻辑：

```python
common_attn_metadata_list = split_attn_metadata(...)
builder.build(common_attn_metadata=common_attn_metadata_list[ubid])
```

inplace 目标：

1. 先得到逻辑正确的 `second_common_metadata`。
2. 把关键 tensor copy 到 stable buffer。
3. 构造新的 `AscendCommonAttentionMetadata`，字段指向 stable buffer view。
4. 再交给 builder.build。

伪代码：

```python
def _stabilize_inplace_common_attn_metadata(
    self,
    common: AscendCommonAttentionMetadata,
    split_idx: int,
) -> AscendCommonAttentionMetadata:
    if split_idx == 0:
        return common bound to original prefix buffers

    nreq = common.num_reqs
    ntok = common.num_actual_tokens

    self.inplace_query_start_loc_secondary.gpu[:nreq + 1].copy_(
        common.query_start_loc[:nreq + 1])
    self.inplace_query_start_loc_secondary.cpu[:nreq + 1].copy_(
        common.query_start_loc_cpu[:nreq + 1])

    self.inplace_seq_lens_secondary.gpu[:nreq].copy_(common.seq_lens[:nreq])
    self.inplace_seq_lens_secondary.cpu[:nreq].copy_(common.seq_lens_cpu[:nreq])

    self.inplace_slot_mapping_secondary.gpu[:ntok].copy_(
        common.slot_mapping[:ntok])

    return dataclasses.replace(
        common,
        query_start_loc=self.inplace_query_start_loc_secondary.gpu[:nreq + 1],
        query_start_loc_cpu=self.inplace_query_start_loc_secondary.cpu[:nreq + 1],
        seq_lens=self.inplace_seq_lens_secondary.gpu[:nreq],
        seq_lens_cpu=self.inplace_seq_lens_secondary.cpu[:nreq],
        slot_mapping=self.inplace_slot_mapping_secondary.gpu[:ntok],
    )
```

阶段 7 建议新增两个 runner helper：

```python
def _stabilize_inplace_common_attn_metadata(
    self,
    common: AscendCommonAttentionMetadata,
    *,
    split_idx: int,
) -> AscendCommonAttentionMetadata:
    ...


def _dry_run_inplace_stable_metadata(
    self,
    common_attn_metadata: AscendCommonAttentionMetadata,
    inplace_split_plan: InplaceSplitPlan,
) -> None:
    ...
```

`_dry_run_inplace_stable_metadata()` 只在 debug 或 inplace plan 存在时执行：

1. 用阶段 7 之前已有的 `split_attn_metadata()` 生成逻辑正确的 split metadata。
2. 对 split-0 保持原始 metadata view。
3. 对 split-1 调用 `_stabilize_inplace_common_attn_metadata()`。
4. 记录 `inplace_metadata_views` JSONL。
5. 不把 stable metadata 交给 builder，不改变实际 forward 路径。

### 11.4.1 copy 字段

split-1 需要复制到 stable buffer 的字段：

```text
query_start_loc.gpu[:nreq + 1]
query_start_loc.cpu[:nreq + 1]
seq_lens.gpu[:nreq]
seq_lens.cpu[:nreq]
slot_mapping.gpu[:ntok]
```

`num_computed_tokens_cpu` 第一版可使用原始 request slice view，因为它不在 NPU graph
输入中；但需要记录 ptr。如果后续 builder 或 replay 断言要求稳定 CPU ptr，再增加
dedicated pinned CPU buffer。

`positions` 继续使用阶段 6 的 input offset view，不复制到 metadata stable buffer。

### 11.4.2 返回 metadata 要求

返回的新 `AscendCommonAttentionMetadata` 要保持以下字段不变：

- `num_reqs`
- `num_actual_tokens`
- `num_input_tokens`
- `max_query_len`
- `decode_token_per_req`
- `actual_seq_lengths_q`
- `attn_state`
- `attn_mask`
- `spec_attn_mask`
- `prefill_context_parallel_metadata`

只替换需要稳定地址的 tensor 字段。

### 11.4.3 容量断言

在 copy 前增加断言：

```text
nreq + 1 <= max_num_reqs + 1
nreq <= max_num_reqs
ntok <= max_num_tokens
```

如果断言失败，记录 debug event 并 fallback dry-run，不应进入后续真实 inplace。

### 11.5 CPU tensor 注意

`seq_lens_cpu` 和 `query_start_loc_cpu` 可能被 builder 用于生成 Python list 或 CPU-side metadata。要保证：

- 内容正确。
- 生命周期稳定。
- 不会被下一段覆盖。

如果 serial 路径先跑 split-0 后跑 split-1，secondary buffer 不会和 split-0 竞争。parallel 路径需要 split-0 和 split-1 的 metadata buffer 互不覆盖。

阶段 7 只准备 split-1 secondary buffer。阶段 11 实现 `inplace_parallel` 前，需要再
确认 split-0 和 split-1 是否会并发读取同一 CPU pinned buffer。

### 11.6 per-layer metadata 地址

`AscendMetadata` 或 MLA/GDN metadata 可能把 common metadata 字段转换成 backend-specific 字段。例如：

- `block_tables`
- `seq_lens`
- `seq_lens_list`
- `actual_seq_lengths_q`
- `decode.seq_lens_list`
- `decode.block_table`

需要用 debug helper 收集每个 per-layer metadata 的 tensor ptr。

建议新增：

```python
def collect_attn_metadata_ptrs(attn_metadata: Any) -> dict[str, int]:
    ...
```

在 capture/replay 中记录和比较。

阶段 7 先实现 common metadata 层面的 ptr 收集，新增 debug helper：

```python
def common_metadata_tensor_info(common: Any) -> dict[str, Any]:
    ...
```

至少输出：

- `query_start_loc`
- `query_start_loc_cpu`
- `seq_lens`
- `seq_lens_cpu`
- `block_table_tensor`
- `slot_mapping`
- `num_computed_tokens_cpu`
- `positions`

per-layer metadata ptr 收集继续保留为阶段 8/9 的 capture 接入项。

### 11.6.1 JSONL 日志

新增事件：

```text
inplace_metadata_views
```

每个 split 记录：

```json
{
  "split_idx": 1,
  "stabilized": true,
  "num_reqs": 32,
  "num_tokens": 32,
  "start_num_tokens": 384,
  "original": {
    "query_start_loc": {"data_ptr": 111, "shape": [33]},
    "seq_lens": {"data_ptr": 222, "shape": [32]},
    "slot_mapping": {"data_ptr": 333, "shape": [32]}
  },
  "stable": {
    "query_start_loc": {"data_ptr": 444, "shape": [33]},
    "seq_lens": {"data_ptr": 555, "shape": [32]},
    "slot_mapping": {"data_ptr": 666, "shape": [32]}
  }
}
```

### 11.7 第一阶段支持范围

为了降低复杂度，第一版建议只支持普通 non-MLA decode attention：

```python
if self.model_config.use_mla:
    fallback parallel_buffer or no split
if self.pcp_size * self.dcp_size > 1:
    fallback parallel_buffer or no split
if self.uses_mrope:
    fallback parallel_buffer or no split
```

后续逐项打开。

### 11.8 阶段验收标准

相同 fixed batch 连续两步：

- second `query_start_loc.data_ptr()` 相同。
- second `seq_lens.data_ptr()` 相同。
- second `slot_mapping.data_ptr()` 相同，或确认未被 graph 捕获。
- per-layer metadata 中被 capture/update 使用的 tensor ptr 相同。

补充验收：

- split-1 stable `query_start_loc` 内容等于原始 split metadata。
- split-1 stable `seq_lens` 内容等于原始 split metadata。
- split-1 stable `slot_mapping` 内容等于原始 split metadata。
- split-0 不复制到 secondary buffer。
- `parallel_buffer` 旧路径行为不变。
- 阶段 7 仍不启用真实 inplace execution。

### 11.9 单元测试计划

新增纯 tensor 单测，不依赖 NPU runtime：

```text
构造 fake runner + AscendCommonAttentionMetadata
split = 384 + 32
split_attn_metadata 产生 original second metadata
stabilize second
验证 stable query_start_loc / seq_lens / slot_mapping 内容一致
验证 stable ptr 来自 runner secondary buffer
验证 repeated stabilize ptr 不变
```

debug 测试：

- `common_metadata_tensor_info()` 覆盖 GPU/CPU tensor 字段。
- `inplace_metadata_views` payload 可 JSON 序列化。

### 11.10 建议验证命令

```bash
python -m pytest \
  tests/ut/test_inplace_split_metadata_stabilization.py \
  tests/ut/test_inplace_split_debug.py \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_planner.py \
  -q
```

语法验证：

```bash
python -m py_compile \
  vllm_ascend/worker/model_runner_v3.py \
  vllm_ascend/attention/utils.py \
  vllm_ascend/inplace_split_debug.py
```

### 11.11 阶段退出产物

```text
inplace_split_phase7_report.md
```

报告至少包含：

- 阶段 7 修改文件。
- stable metadata helper 签名。
- stable buffer 字段和未稳定字段说明。
- 固定样例 ptr/content 验证结果。
- 单测和 py_compile 结果。
- 明确说明真实 inplace execution 仍未启用。

## 12. 阶段 8：实现 inplace_serial 执行路径

### 12.1 目标

阶段 8 的目标是实现第一版真实 `inplace_serial` 执行路径，把阶段 5 的 planner、
阶段 6 的 input offset view、阶段 7 的 stable metadata 串起来执行两个 split。

阶段 8 完成后应满足：

1. `mode=inplace_serial` 且 planner 成功时，`_prepare_inputs()` 返回真实
   `split_batch_slices` / `split_ubatch_slices`。
2. split-0 使用原始 input prefix view。
3. split-1 使用原始 input offset view，不复制到 parallel buffer，不复制回主 buffer
   前缀。
4. split-1 使用 stable common metadata 构造 per-layer metadata。
5. split-1 dispatch descriptor 带 `start_num_tokens=first_tokens`。
6. split-1 使用真实 remainder 作为 graph `num_tokens`，不 padding。
7. 两个 split 在 main stream 串行执行，结果 trim 后 concat。

阶段 8 明确不做：

- 不实现 `inplace_parallel` 并发。
- 不放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP。
- 不新增 block table dedicated stable buffer。
- 不扩大 lazy capture 安全策略；阶段 9 再补更严格开关和断言。

### 12.2 修改文件

```text
vllm_ascend/worker/model_runner_v3.py
tests/ut/test_inplace_split_execution_helpers.py
```

### 12.3 新增函数

```python
def _run_split_batch_inplace_serial(
    self,
    split_ubatch_slices,
    split_batch_slices,
    attn_metadata,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
    model_kwargs,
    batch_descriptor,
    aclgraph_runtime_mode,
) -> Any:
    ...
```

配套新增：

```python
def _make_split_batch_metadata_inplace_serial(...) -> list[AscendUbatchMetadata]:
    ...

def _select_split_execution_mode(...) -> str:
    ...
```

### 12.4 执行流程

1. 为 split-0 构造 context：
   - input view: `[0:first_tokens)`
   - descriptor: `start_num_tokens=0`
   - runtime mode: normal dispatch
   - GraphParams key: int 或 descriptor start=0，建议普通仍 int

2. 为 split-1 构造 context：
   - input view: `[first_tokens:first_tokens+second_tokens)`
   - descriptor: `start_num_tokens=first_tokens`
   - runtime mode: offset lazy dispatch
   - GraphParams key: descriptor

3. 执行 split-0：

```python
with torch.npu.stream(self.stream_main):
    with override_forward_context(ctx0):
        out0 = self.model(...)
        self._update_attn_params_for_split_ubatch(
            ctx0, first_graph_num_tokens, ...)
```

4. 同步或等待 split-0 完成。

5. 执行 split-1：

```python
with torch.npu.stream(self.stream_main):
    with override_forward_context(ctx1):
        out1 = self.model(...)
        self._update_attn_params_for_split_ubatch(
            ctx1, second_graph_num_tokens, ...)
```

6. trim output。

7. concat output。

### 12.4.1 `_prepare_inputs()` 接入

阶段 8 开始，`inplace_serial` 不再只是 dry-run。planner 成功后：

```python
if split_mode == "inplace_serial" and inplace_split_plan is not None:
    split_batch_slices = inplace_split_plan.split_slices
    split_ubatch_slices = [
        UBatchSlice(s.request_slice, s.token_slice)
        for s in split_batch_slices
    ]
```

`inplace_parallel` 仍保持 dry-run，不设置 execution slices。

`num_input_tokens` 对 inplace serial 应使用真实 total tokens：

```python
num_input_tokens = total_num_scheduled_tokens
```

而不是 no-split padding 后的 graph size。这样传入模型的原始 input buffer 范围是
`[0,total_tokens)`，split-1 offset view 才不会落到 padding 区。

### 12.4.2 metadata builder 接入 stable metadata

当前 split metadata 构造：

```python
common_attn_metadata_list = split_attn_metadata(...)
builder.build(common_attn_metadata=common_attn_metadata_list[ubid])
```

阶段 8 在 `mode=inplace_serial` 时改为：

```python
common_attn_metadata_list = split_attn_metadata(...)
common_attn_metadata_list = [
    self._stabilize_inplace_common_attn_metadata(common, split_idx=idx)
    for idx, common in enumerate(common_attn_metadata_list)
]
```

然后再交给 builder。split-0 保持原 view，split-1 使用 secondary buffer。

### 12.4.3 descriptor dispatch

`_make_split_batch_metadata_inplace_serial()` 中每个 split 的 descriptor：

```python
_, desc = self.cudagraph_dispatcher.dispatch(
    num_tokens=split_slice.graph_num_tokens,
    uniform_decode=batch_descriptor.uniform,
    has_lora=batch_descriptor.has_lora,
    start_num_tokens=split_slice.start_num_tokens,
    allow_inplace_lazy_key=split_slice.start_num_tokens > 0
        and split_cfg.enable_inplace_lazy_capture,
)
```

预期：

- split-0: `start_num_tokens=0`，命中已有 capture size。
- split-1: `start_num_tokens=first_tokens`，使用 offset lazy key。

### 12.4.4 执行模式选择

`execute_model()` 中 split execution 分支改为：

```python
if split_mode == "inplace_serial":
    hidden_states = self._run_split_batch_inplace_serial(...)
elif split_enable_parallel_streams:
    hidden_states = self._run_split_batch_parallel(...)
else:
    hidden_states = self._run_split_batch_gr0(...)
```

禁止 `inplace_parallel` 误入 parallel-buffer execution；在阶段 8 它仍是 dry-run。

### 12.4.5 日志

新增 JSONL event：

```text
inplace_serial_execution
```

记录：

- split idx。
- token range。
- `start_num_tokens`。
- actual tokens。
- graph tokens。
- descriptor。
- input view ptr。
- metadata ptr。
- runtime mode。

### 12.5 与当前 `_run_split_batch_gr0` 的区别

当前 `_run_split_batch_gr0`：

- 后续 split 会复制数据到主 buffer 前缀。
- 后续 split 用起始地址 replay。
- 需要备份和恢复主 buffer 前缀。

inplace serial：

- 不复制 second input。
- 不覆盖主 buffer 前缀。
- 不需要备份 input_ids/positions。
- second graph key 用 offset 区分。

### 12.6 model_kwargs 注意

当前 `execute_model()` 构造：

```python
model_kwargs = self._init_model_kwargs(maybe_padded_num_tokens)
```

如果 pooling model 或未来 kwargs 依赖 token 数，inplace second 应该使用对应 split token 数。第一版可限制非 pooling model，或者在函数内按 split 重建 kwargs。

### 12.7 阶段验收标准

在 fixed batch 下：

- 不调用 parallel buffer copy。
- split-1 使用 offset input view。
- split-1 descriptor 带 `start_num_tokens`。
- split-1 首次 lazy capture 成功。
- split-1 第二次 replay 成功。
- 输出与 no-split padding 模式一致。

可在无 NPU 环境验证的阶段 8 验收：

- `mode=inplace_serial` planner 成功后会设置 execution slices。
- `mode=inplace_parallel` planner 成功后仍 dry-run。
- `_make_split_batch_metadata_inplace_serial()` 生成 split-1 descriptor 时传入
  `start_num_tokens`。
- split-1 `AscendUbatchMetadata.num_tokens == split_slice.graph_num_tokens`。
- split-1 input/positions 使用 offset view，`storage_offset == first_tokens`。
- `_run_split_batch_inplace_serial()` 不调用 parallel buffer 和 prefix copy helper。

### 12.8 单元测试计划

优先覆盖纯 helper 和可 monkeypatch 的 runner 方法：

```text
_enable_inplace_execution_slices:
inplace_serial + plan -> returns split slices
inplace_parallel + plan -> dry-run only
```

```text
_make_split_batch_metadata_inplace_serial:
fake dispatcher 记录 start_num_tokens
fake create context 返回 descriptor
验证 second descriptor start=first_tokens
验证 second input view storage_offset=first_tokens
```

```text
_merge_split_outputs / _trim_split_output:
两个 split 输出 concat 后 token 顺序保持
```

NPU 环境补充：

```text
fixed decode 416 -> 384 + 32
第一次触发 split-1 lazy capture
第二次 replay
对比 no split 输出
```

### 12.9 阶段退出产物

```text
inplace_split_phase8_report.md
```

报告至少包含：

- 阶段 8 修改文件。
- `inplace_serial` execution path 入口。
- descriptor / input / metadata 三类日志示例。
- 单测和 py_compile 结果。
- NPU fixed decode 是否执行。
- 明确说明 `inplace_parallel` 仍未启用。

### 12.10 当前 416 decode 失败的修复设计

本小节补充 2026-05-23 定位到的 `inplace_serial` 真实执行失败方案。该问题不能通过
fallback 到 no-split 解决，也不能把当前 case 简单改成 PA。目标仍然是让
`_run_split_batch_inplace_serial()` 真实执行两个 split，并保持和 no-split baseline
一致的 attention backend。

#### 12.10.1 已观察到的失败形态

固定场景：

```text
model: Qwen/Qwen2.5-0.5B-Instruct
total decode tokens: 416
cudagraph_capture_sizes: [256, 384, 512]
planner result: 384 + 32
split-0: num_tokens=384, start_num_tokens=0
split-1: num_tokens=32, start_num_tokens=384
```

实际短实验中，第一条请求可能已经完成，所以复现样本也可能是：

```text
total decode tokens: 415
planner result: 384 + 31
split-0: num_tokens=384, start_num_tokens=0
split-1: num_tokens=31, start_num_tokens=384
```

两者触发的是同一个问题。

失败日志中的关键 NPU 报错：

```text
When layout is TND and PA not enabled, keyT(256) and valueT(256)
must be equal to the last element of actualSeqenceLengthKV(9)
```

这说明当前 split graph 中 attention 走到了 FIA/TND non-PA 分支，但传入的
`actual_seq_lengths_kv` 是 decode 的真实 per-request `seq_lens_list`，最后一个值
可能是 9；而该 NPU kernel 在 non-PA TND 模式下要求 `actual_seq_lengths_kv[-1]`
等于 key/value 的 T 维，例如 256。

还有一个次生失败：

```text
ValueError: too many values to unpack (expected 9)
```

这是因为 capture 时记录的是 FIA 的 13 元组 graph task params，update 时却按 PA 的
9 元组解析。这个错误通常来自“运行时强制 PA，但 graph entry / GraphParams 仍复用
普通 FIA capture”的错误修复方式。

#### 12.10.2 根因

2026-05-23 的实验修正了之前的判断：当前 case 的 no-split baseline 不是 PA，
而是 FIA。

实验证据：

```text
enabled inplace_serial:
  /tmp/codex_inplace_split_backend_guard/20260523_083845/summary.json
  status = ERROR
  step 4: 415 -> 384 + 31
  error = actualSeqenceLengthKV(9)

disabled no-split control:
  /tmp/codex_inplace_split_disabled_control/20260523_084122/summary.json
  status = DONE
  count = 416

additional_config.pa_shape_list 未设置，默认是 []
using_paged_attention(31/32/384/415/416/512) 全部 False
```

因此本 case 的真实根因不是“no-split 走 PA，split 后变 FIA”。真实根因是：

1. no-split baseline 走 FIA，但它使用启动阶段已经预捕获好的普通 graph。
2. 预捕获普通 graph 时使用的是 capture-safe 的 warmup metadata，因此不会触发
   CANN 的 TND non-PA tiling 约束。
3. `inplace_serial` 的 split-1 是 offset graph，启动阶段没有预捕获，首次运行会
   进入 lazy capture。
4. 当前 lazy capture 直接使用真实 decode metadata。FIA DecodeOnly 分支使用：

```python
actual_seq_lengths_kv = attn_metadata.seq_lens_list
```

5. 对当前 NPU FAI TND non-PA kernel，真实 `seq_lens_list` 的最后一个值可能是 9，
   不满足 `actual_seq_lengths_kv[-1] == keyT` 约束，导致 lazy capture 阶段
   tiling 失败。

所以当前问题的核心是：**FIA offset graph 的首次 capture 使用了不合法的真实
decode seqlen metadata**。backend mismatch 仍然是一个需要防护的独立问题，但不是
当前复现 case 的根因。

#### 12.10.3 不能采用的方案

以下方案不能作为最终修复：

| 方案 | 为什么不行 |
|---|---|
| `no_split_attention_backend_mismatch` fallback | 规避了 `_run_split_batch_inplace_serial()`，不满足 inplace 目标 |
| 当前 case 直接 forced PA | 改变了 no-split baseline 的 attention backend；当前 baseline 实验证明是 FIA |
| 只在 `using_paged_attention()` 中看到 `split_inplace_mode` 就返回 True | 会让 runtime update 走 PA，但已有 graph entry 可能是 FIA capture，导致 PA/FIA 参数元组不匹配 |
| 只把 FIA 的 `actual_seq_lengths_kv` 末尾 pad 到 keyT | 可能让 tiling 通过，但语义仍不是 decode paged KV；除非同时把 KV cache gather 成普通连续 TND KV |
| 把 split-1 padding 到 PA shape | 违背 inplace split “second 使用真实 remainder，不 padding”的核心目标 |

#### 12.10.4 推荐总体方案

推荐方案是：**backend-preserving inplace graph variant + FIA capture-safe
first-use。**

核心规则：

1. 先按 no-split padded graph shape 计算 baseline backend。
2. split-0 和 split-1 都必须保持这个 baseline backend。
3. 如果 baseline 是 FIA，split 也必须走 FIA；不能为了绕过错误改成 PA。
4. 如果 baseline 是 PA，split 才可以显式 forced PA，因为这是保持 baseline
   backend，不是改变 backend。
5. capture、replay、graph task update 必须使用同一个 backend 决策和同一个
   GraphParams key。

对当前 416/415 case：

```text
pa_shape_list = []
padded_without_split = 512
using_paged_attention(512) == False
=> baseline backend = FIA
=> inplace split backend = FIA
```

因此当前修复应优先打通 FIA offset graph，而不是 forced PA。

FIA offset graph 的首次运行需要新增 capture-safe 行为：

```text
entry missing + inplace backend FIA:
  1. 用合法的 template metadata capture FIA graph
  2. 不把 capture output 当作本次请求结果
  3. 立即 enqueue replay，返回 replay output buffer
  4. caller 继续调用 update_attn_params_split()，用真实 metadata 更新 graph task
  5. stream synchronize 后，本次输出来自 replay + update 后的真实 metadata
```

如果第一版实现无法安全保证“capture 后立即 replay + update ordering”，则退回到
更保守但正确的方案：在 warmup / capture 阶段预捕获需要的 inplace offset FIA
descriptor，运行时只 replay/update，不做真实请求上的 lazy capture。

#### 12.10.5 descriptor / graph key 设计

当前 `BatchDescriptor.start_num_tokens` 只用于区分 offset graph。`inplace_serial`
第一版要求 **split-0 必须复用启动阶段已经捕获的普通 graph**：

```text
split-0: start_num_tokens == 0 -> 使用普通 int runtime_shape key，例如 384
split-1: start_num_tokens > 0  -> 使用 inplace descriptor key，例如 32@384
```

因此 split-0 不能携带会改变 graph key 的 `graph_variant` /
`capture_metadata_mode`，也不能因为 `forced_attention_backend` 被放入 inplace
专属 key。它应该直接匹配普通 `384` graph 和普通 `384` GraphParams。

这个要求带来一个前置校验：split-0 普通 graph 的 attention backend 必须和 no-split
baseline backend 一致。若不一致，该 split plan 不能进入 `inplace_serial`，否则会
破坏 backend-preserving 语义。

需要扩展 graph key 表达能力。建议在 `BatchDescriptor` 尾部新增字段：

```python
graph_variant: str = ""
attention_backend: str = ""
capture_metadata_mode: str = ""
```

字段语义：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `graph_variant` | `""` | 普通 graph；`"inplace_serial"` 表示 inplace 专属 graph |
| `attention_backend` | `""` | 普通 graph 按原逻辑；inplace graph 明确为 `"pa"` 或 `"fia"` |
| `capture_metadata_mode` | `""` | 普通 graph 为空；FIA offset 可用 `"template"` 表示 capture 使用合法模板 metadata |

示例 key：

```python
# 普通 no-split / parallel-buffer graph，保持旧行为
BatchDescriptor(num_tokens=384, num_reqs=384, uniform=True, has_lora=False)

# inplace split-0 不新增 descriptor variant，必须复用普通 384 graph
BatchDescriptor(num_tokens=384, num_reqs=384, uniform=True, has_lora=False)

# inplace split-1，offset + backend + capture policy
BatchDescriptor(
    num_tokens=32,
    num_reqs=32,
    uniform=True,
    has_lora=False,
    start_num_tokens=384,
    graph_variant="inplace_serial",
    attention_backend="fia",
    capture_metadata_mode="template",
)
```

关键要求：

- `graph_variant`、`attention_backend`、`capture_metadata_mode` 参与 equality/hash。
- 默认值必须保持旧 graph key 完全兼容。
- split-0 不设置任一 inplace variant 字段，必须降级成普通 `int` key。
- 只有 offset descriptor 或明确非 split-0 的 inplace variant 才使用完整 descriptor
  key。
- `ACLGraphWrapper.concrete_aclgraph_entries` 使用完整 descriptor 作 entry key。

#### 12.10.6 GraphParams key 设计

当前阶段 4 的设计是：

```text
start_num_tokens == 0 -> int runtime_shape
start_num_tokens > 0  -> BatchDescriptor
```

对 inplace offset graph 不够，因为 split-1 需要 descriptor-aware key；但 split-0
必须继续复用普通 graph。

需要把 helper 改为：

```text
普通 graph 或 split-0 且 graph_variant == "" 且 attention_backend == "" 且 start_num_tokens == 0
    -> int runtime_shape

其他所有 descriptor-aware graph
    -> BatchDescriptor
```

伪代码：

```python
def get_graph_param_key(forward_context, runtime_shape):
    desc = getattr(forward_context, "batch_descriptor", None)
    if not isinstance(desc, BatchDescriptor):
        return runtime_shape

    is_descriptor_variant = (
        int(getattr(desc, "start_num_tokens", 0) or 0) > 0
        or bool(getattr(desc, "graph_variant", ""))
        or bool(getattr(desc, "attention_backend", ""))
        or bool(getattr(desc, "capture_metadata_mode", ""))
    )
    if is_descriptor_variant:
        return desc
    return runtime_shape
```

这样 split-0 继续读写普通 `384` 桶；split-1 / offset inplace capture 的
`attn_params` 不会写入普通 `32` 或 `384` 桶，也不会污染普通 graph task params。

#### 12.10.7 attention backend policy

新增显式 policy helper，避免每个 split 用自己的 shape 隐式判定 backend：

```python
def select_inplace_attention_backend(
    *,
    total_padded_graph_tokens: int,
    split_plan: InplaceSplitPlan,
    vllm_config: VllmConfig,
    attn_state: AscendAttentionState,
) -> str:
    ...
```

第一版规则：

1. 仅支持 `DecodeOnly`。
2. speculative / M-RoPE / MLA / PCP / DCP / sliding window 继续不进入第一版。
3. 使用 `padded_without_split` 调用原始 `using_paged_attention()` 得到 baseline。
4. baseline 为 FIA 时，split context 设置 `forced_attention_backend="fia"`。
5. baseline 为 PA 时，split context 设置 `forced_attention_backend="pa"`。
6. 如果某个 backend 的 capture-safe 机制尚未实现，则 planner 不能 fallback 到
   no-split；应明确报 unsupported，避免静默给出错误性能/正确性结论。

对 416 case：

```text
total_tokens=416
padded_without_split=512
pa_shape_list=[]
using_paged_attention(512) == False
=> inplace_attention_backend = "fia"
```

随后 split-0 和 split-1 都携带：

```python
forward_context.forced_attention_backend = "fia"
```

但 graph key 处理不同：

```text
split-0: 不写 batch_descriptor.attention_backend，复用普通 384 graph
split-1: batch_descriptor.attention_backend = "fia"，使用 offset descriptor graph
```

split-0 是否能复用普通 graph，需要在 dispatch 前校验：

```text
using_paged_attention(split0_graph_tokens) == forced_attention_backend
```

对当前 416 case，`using_paged_attention(384) == False`，与 baseline FIA 一致，
所以 split-0 可以直接复用启动阶段普通 384 graph。

#### 12.10.8 `using_paged_attention()` 修改方式

`using_paged_attention()` 可以支持 forced backend，但 forced backend 必须来自
backend-preserving policy，不能只看 `split_inplace_mode`：

```python
def using_paged_attention(runtime_shape, vllm_config, forward_context=None):
    forced = getattr(forward_context, "forced_attention_backend", None)
    if forced == "pa":
        return True
    if forced == "fia":
        return False

    desc = getattr(forward_context, "batch_descriptor", None)
    if getattr(desc, "attention_backend", "") == "pa":
        return True
    if getattr(desc, "attention_backend", "") == "fia":
        return False

    # 原有逻辑
    ...
```

约束：

- forced backend 必须在 capture 前设置。
- replay 和 `_update_attn_params_for_split_ubatch()` 使用同一个 forward context。
- 如果 descriptor 带 `attention_backend` 或 `capture_metadata_mode`，GraphParams key
  必须是 descriptor，不能是 `int runtime_shape`。
- 对当前 case，forced backend 应为 `"fia"`。

#### 12.10.9 dispatcher 修改

新增或扩展 inplace dispatch：

```python
dispatch(
    num_tokens=split_slice.graph_num_tokens,
    uniform_decode=True,
    has_lora=False,
    start_num_tokens=split_slice.start_num_tokens,
    graph_variant=("inplace_serial" if split_slice.start_num_tokens > 0 else ""),
    attention_backend=(inplace_attention_backend
                       if split_slice.start_num_tokens > 0 else ""),
    capture_metadata_mode=("template"
                           if split_slice.start_num_tokens > 0
                           and inplace_attention_backend == "fia" else ""),
    allow_inplace_lazy_key=True,
)
```

与当前 offset lazy key 的差异：

- split-0 必须继续使用普通 graph；不得为 split-0 生成 inplace descriptor。
- split-1 继续 lazy 注册 offset key。
- `allow_inplace_lazy_key` 只对 offset inplace descriptor 生效；split-0 不需要
  lazy capture，因为它必须命中启动阶段普通 graph。

建议改成：

```python
allow_inplace_lazy_capture = (
    split_cfg.enable_inplace_lazy_capture
    and descriptor.start_num_tokens > 0
    and descriptor.graph_variant == "inplace_serial"
    and descriptor.attention_backend in ("fia", "pa")
)
```

ACLGraphWrapper 中 lazy capture 许可应限制为“authorized offset inplace variant”。

#### 12.10.10 FIA capture-safe metadata 设计

FIA graph capture 不能直接使用真实 decode `seq_lens_list`。需要为 capture 提供
合法的 template metadata：

```text
actual_seq_lengths_q: 递增到 graph_num_tokens
actual_seq_lengths_kv: 最后一个值必须等于 keyT/valueT，例如 256
block_tables: 使用 shape 合法的稳定 tensor
seq_lens / seq_lens_list: capture 用 template；runtime update 用真实 metadata
```

关键要求：

- template metadata 只用于 graph capture。
- runtime graph task update 必须使用真实 split metadata。
- capture output 不能作为本次请求输出。
- 如果 entry 是运行时首次 capture，wrapper 必须在 capture 后立即 enqueue replay，
  让 caller 后续的 `_update_attn_params_for_split_ubatch()` 更新这次 replay 的 graph
  task。
- 如果无法证明上述 update ordering，必须改为 warmup 预捕获 offset graph。

2026-05-23 实施验证补充：

首轮修复把 FIA template 改写放在 `full_graph_fia()` 中，并依赖
`forward_context.capturing` 判断 capture 阶段。NPU 416 验证仍失败：

```text
When layout is TND and PA not enabled, keyT(256) and valueT(256)
must be equal to the last element of actualSeqenceLengthKV(9)
```

这说明真实 capture 路径进入 FAI tiling 时仍拿到了真实 decode
`seq_lens_list[-1] == 9`，template 改写没有覆盖实际生效点。

因此实现要求改为：

1. template seqlen 改写不能只放在 `full_graph_fia()`，否则可能绕不开实际
   FAI tiling 生效点。
2. 第二轮把改写前移到 `AscendAttention._get_fia_params()`，并改为依赖
   descriptor policy，而不是只看 `forward_context.capturing`：

```python
desc = getattr(get_forward_context(), "batch_descriptor", None)
if (
    attn_metadata.attn_state == AscendAttentionState.DecodeOnly
    and getattr(desc, "capture_metadata_mode", "") == "template"
    and getattr(desc, "attention_backend", "") == "fia"
):
    actual_seq_lengths_kv = list(actual_seq_lengths_kv)
    actual_seq_lengths_kv[-1] = int(key.shape[1])
```

3. 注意：DecodeOnly 中 `key` 已经 reshape 成 `[num_blocks, block_size, -1]`，
   所以正确值是 `key.shape[1]` / `block_size`，不是 `key.shape[0]`。
   `key.shape[0]` 是 num_blocks，曾经正好产生错误中的 `actualSeqenceLengthKV(9)`。
4. 最新运行仍出现 `actualSeqenceLengthKV(9)`，说明仅依赖 `_get_fia_params()`
   内部读取 Python forward context 不够稳，可能被 torch.compile 图专门化或没有覆盖
   capture/update 的实际参数来源。
5. 当前实现进一步把 FIA template 改写前移到
   `_make_split_batch_metadata_inplace_serial()`：

```python
if (
    split_slice.start_num_tokens > 0
    and inplace_attention_backend == "fia"
    and descriptor.capture_metadata_mode == "template"
):
    for layer_metadata in ubatch_attn_metadata.values():
        layer_metadata.seq_lens_list = list(layer_metadata.seq_lens_list)
        layer_metadata.seq_lens_list[-1] = self.block_size
```

6. 该逻辑只允许出现在 offset inplace descriptor 上。split-0 不带
   `capture_metadata_mode`，因此仍复用普通 graph 和普通 FIA 参数。
7. 第三轮修复把 template policy 固化为 ACL graph 层 helper：
   `maybe_template_fia_seq_lens(forward_context, seq_lens, block_size)`。
   `full_graph_fia()` 在计算 workspace、保存 `graph_params.attn_params`、capture
   FAI op 之前再次调用该 helper；`_update_attn_fia_params()` 在下发
   `actual_seq_lengths_kv` 前也调用同一个 helper。
8. helper 只在 descriptor 满足
   `attention_backend == "fia"` 且 `capture_metadata_mode == "template"` 时生效，
   并复制 `seq_lens` 后只改尾值，避免污染普通 graph 或 split-0 metadata。
9. 当前实现阶段接受“descriptor policy 下 capture/replay/update 都会看到 template
   seqlen 尾值”。如果后续 correctness 证明该尾值必须在 runtime update 恢复为真实
   seqlen，则需要新增独立的 update 参数重写策略或切换到 warmup 预捕获 offset graph，
   不能回退到 no-split。

#### 12.10.11 metadata stable buffer 修改

当前阶段 7 主要稳定 split-1 metadata。FIA offset graph 修复后，需要明确真实
metadata 与 capture template metadata 的地址策略。

第一版建议：

- split-0 可以继续使用原始 prefix buffers，但必须验证这些地址跨 step 稳定：
  - `seq_lens.gpu[:first_reqs]`
  - `block_table_tensor[:first_reqs]`
  - `slot_mapping[:first_tokens]`
  - positions / input_ids / inputs_embeds prefix view
- split-1 必须使用 secondary stable buffers：
  - `inplace_query_start_loc_secondary`
  - `inplace_seq_lens_secondary`
  - `inplace_slot_mapping_secondary`
  - `inplace_block_table_secondary`
- capture template 涉及 tensor 时必须使用独立 stable buffers，不能复用真实 runtime
  metadata tensor 后再原地改写；否则 capture/replay/update 的地址和语义会互相污染。
  当前 FIA 修正只改 Python `seq_lens_list`，实现时复制 list 后改尾值，不改
  `seq_lens` tensor 本体。

如果 split-0 prefix 地址校验失败，则新增 primary stable buffers：

```text
inplace_query_start_loc_primary
inplace_seq_lens_primary
inplace_slot_mapping_primary
inplace_block_table_primary
```

FIA graph task update 使用 per-layer metadata 中的 `seq_lens_list` /
`actual_seq_lengths_q` / `block_tables`。在 `capture_metadata_mode="template"` 的
offset FIA descriptor 下，`seq_lens_list[-1]` 当前会保持 block size 模板值；其余
metadata 仍来自真实 split metadata。reshape/cache 仍会使用 `slot_mapping`。因此
slot mapping 不能只在 common metadata 层稳定，也要确认 per-layer metadata 中引用的是
stable tensor。

#### 12.10.12 `_make_split_batch_metadata_inplace_serial()` 修改

该函数需要新增参数或从 runner 状态读取：

```python
inplace_attention_backend: Literal["fia", "pa"]
```

为每个 split 构造 context 时设置：

```python
setattr(split_forward_context, "split_inplace_mode", "inplace_serial")
setattr(split_forward_context, "forced_attention_backend",
        inplace_attention_backend)
```

并确保 descriptor 包含：

```python
if split_slice.start_num_tokens == 0:
    graph_variant = ""
    attention_backend = ""
    capture_metadata_mode = ""
else:
    graph_variant = "inplace_serial"
    attention_backend = inplace_attention_backend
    capture_metadata_mode = "template"  # only for FIA entries that need capture-safe capture
```

日志必须记录：

```json
{
  "event": "split_descriptor",
  "execution": "inplace_serial",
  "idx": 1,
  "graph_variant": "inplace_serial",
  "attention_backend": "fia",
  "capture_metadata_mode": "template",
  "start_num_tokens": 384,
  "dispatch_num_tokens": 32,
  "templated_fia_seq_lens": 24,
  "batch_descriptor": {}
}
```

#### 12.10.13 `_run_split_batch_inplace_serial()` 修改

执行逻辑仍然保持：

```text
split-0: self.model(...) -> update_attn_params_for_split_ubatch(...)
sync
split-1: self.model(...) -> update_attn_params_for_split_ubatch(...)
merge
```

但进入循环前必须断言：

```python
for idx, metadata in enumerate(ubatch_metadata):
    assert metadata.context.split_inplace_mode == "inplace_serial"
    assert metadata.context.forced_attention_backend in ("fia", "pa")
    desc = metadata.context.batch_descriptor
    if idx == 0:
        assert desc.start_num_tokens == 0
        assert getattr(desc, "graph_variant", "") == ""
        assert getattr(desc, "attention_backend", "") == ""
        assert getattr(desc, "capture_metadata_mode", "") == ""
    else:
        assert desc.start_num_tokens > 0
        assert desc.graph_variant == "inplace_serial"
        assert desc.attention_backend in ("fia", "pa")
```

`_update_attn_params_for_split_ubatch()` 不需要单独传 backend；它通过
`forward_context` 调用 `update_attn_params_split()`，后者再通过
`using_paged_attention(runtime_shape, vllm_config, forward_context)` 进入对应 update。

这样 FIA capture 写入的是 FIA 13 元组，update 也解析 FIA 13 元组；PA capture
写入的是 PA 9 元组，update 也解析 PA 9 元组。

#### 12.10.14 ACLGraphWrapper lazy capture 修改

当前 lazy capture 判断偏向：

```text
start_num_tokens > 0
```

新方案中，是否允许 lazy capture 取决于授权的 inplace descriptor，而不是只看
`start_num_tokens`。

建议改为：

```python
def _is_allowed_inplace_lazy_capture(ctx, desc, mode):
    return (
        mode == CUDAGraphMode.FULL
        and getattr(ctx, "allow_inplace_lazy_capture", False)
        and getattr(ctx, "split_inplace_mode", None) == "inplace_serial"
        and getattr(desc, "graph_variant", "") == "inplace_serial"
        and getattr(desc, "attention_backend", "") in ("fia", "pa")
    )
```

安全边界：

- 不允许普通 graph 借此开启 lazy capture。
- 不允许 backend 为空的 inplace graph 开启 lazy capture。
- capture 结束或异常后恢复原 `cudagraph_capturing_enabled`。
- FIA + `capture_metadata_mode="template"` 时，不能返回 capture output；必须
  capture 后 enqueue replay，或者使用 warmup 预捕获方案避免运行时 capture。

#### 12.10.15 为什么不默认 forced PA

当前实验已经证明 no-split baseline 是 FIA。如果默认 forced PA，虽然可能绕开
`actualSeqenceLengthKV` tiling 错误，但会改变 attention backend，不能作为
backend-preserving 的正确性验证。

PA variant 只在以下情况可用：

```text
using_paged_attention(padded_without_split) == True
```

也就是 no-split baseline 本来就是 PA。此时 split forced PA 是为了保持 baseline，
不是为了规避 FIA 问题。

#### 12.10.16 单元测试计划

新增或调整测试：

```text
test_inplace_descriptor_isolated_from_normal_graph
```

- normal descriptor 与携带 `graph_variant` / `attention_backend` /
  `capture_metadata_mode` 的 inplace descriptor 不相等。
- inplace descriptor hash 不同。

```text
test_graph_param_key_keeps_split_zero_as_int
```

- split-0 不设置 `graph_variant` / `attention_backend` /
  `capture_metadata_mode` 时，`get_graph_param_key()` 返回普通 int。
- offset split 设置 `graph_variant="inplace_serial"` 时返回 descriptor。

```text
test_using_paged_attention_honors_forced_backend
```

- `forced_attention_backend="pa"` 返回 True。
- `forced_attention_backend="fia"` 返回 False。
- 无 forced backend 时保持原 `pa_shape_list` 行为。

```text
test_inplace_context_sets_backend_and_variant
```

- `_make_split_batch_metadata_inplace_serial()` 为两个 split 都设置
  backend-preserving 的 `forced_attention_backend`。
- split-0 descriptor 不包含 `graph_variant` / `attention_backend`，用于复用普通
  graph。
- split-1 descriptor 包含 `graph_variant="inplace_serial"` 和
  `attention_backend in ("fia", "pa")`。

```text
test_aclgraph_lazy_capture_allows_authorized_offset_inplace
```

- offset inplace 专属 graph 在授权时可 lazy capture。
- split-0 不触发 lazy capture，必须命中普通 graph。
- 普通 graph 不受影响。

```text
test_update_attn_params_split_uses_backend_from_forced_context
```

- forced PA context 下进入 `_update_attn_pa_params()`。
- forced FIA context 下进入 `_update_attn_fia_params()`。
- split-0 GraphParams key 为 int；offset split GraphParams key 为 descriptor。

```text
test_fia_inplace_lazy_capture_uses_template_metadata
```

- FIA + `capture_metadata_mode="template"` 时，split-1 metadata 构造阶段会把
  每层 `seq_lens_list[-1]` 改成 `block_size`，避免真实 decode seqlen 进入 capture
  tiling。
- `_get_fia_params()` 中若保留兜底逻辑，尾值也必须使用 `key.shape[1]` /
  `block_size`，不能使用 `key.shape[0]`。
- `full_graph_fia()` 和 `_update_attn_fia_params()` 还会按 descriptor policy
  再次调用 `maybe_template_fia_seq_lens()`，确保 capture 存参和 replay/update 下发
  都不会重新使用真实 decode 尾值 9。
- runtime update 当前使用复制后的模板化尾值，不原地污染 metadata。
- capture output 被标记为不可直接返回，必须 replay 或预捕获。

```text
test_maybe_template_fia_seq_lens_copies_tail_to_block_size
test_update_attn_fia_params_templates_offset_seq_lens
```

- 覆盖 ACL graph helper 复制尾值到 block size。
- 覆盖 `_update_attn_fia_params()` 下发 `[9, 256]` 且保留原始
  `metadata.seq_lens_list == [9, 9]`。

#### 12.10.17 NPU 验证计划

固定先验证 416 case：

```text
prompts: 416
capture sizes: [256, 384, 512]
expected split: 384 + 32
mode: inplace_serial
```

必须在 debug JSONL 中看到：

```text
split_planner_decision = inplace_split_execute
inplace_attention_backend = fia
split-1 graph_variant = inplace_serial
split-1 attention_backend = fia
split-1 capture_metadata_mode = template
split-1 start_num_tokens = 384
split-1 GraphParams key = BatchDescriptor(...)
```

必须不能出现：

```text
no_split_attention_backend_mismatch
fallback_to = no_split
split update PA but capture FIA tuple
forced_attention_backend = pa  # 当前 pa_shape_list=[] 的 case 中不允许
```

运行验证：

1. 第一次 decode step：split-1 offset graph 受控 capture-safe capture 或已预捕获。
2. capture output 不作为本次请求输出。
3. 本次请求输出来自 replay + graph task update，或来自预捕获 graph 的 replay/update。
4. 第二次 decode step：split graph 正常 replay。
5. 输出 token 与 no-split padding baseline 一致。
6. 无 `actualSeqenceLengthKV` tiling 错误。
7. 无 PA/FIA tuple unpack 错误。

补充验证：

```text
additional_config.pa_shape_list=[512]
```

该配置下 no-split baseline 才是 PA。此时 split forced PA 是 backend-preserving，
需要看到 split capture/update 均为 PA。

#### 12.10.18 需要删除或避免的临时代码

最终实现前必须移除以下临时方向：

- `no_split_attention_backend_mismatch` fallback guard。
- 仅凭 `split_inplace_mode` 强制 `using_paged_attention()` 返回 True。
- 任何让 `inplace_serial` planner 成功但不进入
  `_run_split_batch_inplace_serial()` 的路径。

如果需要保留诊断能力，可以只保留日志字段：

```text
unsplit_attention_backend
split_attention_backend_without_force
forced_attention_backend
capture_metadata_mode
```

但不能用它们阻止 inplace execution。

#### 12.10.19 最小改动顺序

推荐按以下顺序实施，避免再次出现 capture/update backend 不一致：

1. 扩展 descriptor 字段：`graph_variant`、`attention_backend`。
2. 增加 `capture_metadata_mode` 字段或等价 context policy。
3. 扩展 `get_graph_param_key()`：split-0 仍返回普通 `int runtime_shape`；
   offset inplace descriptor 返回完整 descriptor。
4. 扩展 dispatcher：split-0 不生成 inplace descriptor，split-1 / offset split 生成
   inplace descriptor。
5. 新增 backend policy：416 case 选择 forced FIA；只有 baseline PA 时才选择
   forced PA。
6. 修改 `using_paged_attention()`：split-0 使用普通 graph backend 校验；offset split
   接受 context/descriptor forced backend。
7. 修改 `_make_split_batch_metadata_inplace_serial()`：设置 forced backend 和
   offset descriptor variant。
8. 为 FIA inplace offset graph 增加 capture-safe template metadata。
   该改写必须在 `_get_fia_params()` 生成 FIA 参数时完成，不能只放在
   `full_graph_fia()` 的 capture 分支里。
9. 修改 ACLGraphWrapper lazy capture guard：授权 offset inplace variant，而不是只
   授权 `start_num_tokens > 0`。
10. 实现 capture 后立即 replay，或改为 warmup 预捕获 offset graph。
11. 补齐 split-0 / split-1 metadata ptr 校验。
12. 跑单测。
13. 跑 NPU 416 correctness。

阶段 8A 完成定义：

- 416 case 必须真实进入 `_run_split_batch_inplace_serial()`。
- 不 fallback。
- 当前 `pa_shape_list=[]` 时，split capture/update 均为 FIA。
- FIA offset graph capture 不再使用非法真实 `seq_lens_list`。
- GraphParams 不发生 PA/FIA tuple 混用。
- 输出与 no-split padding baseline 一致。

## 13. 阶段 9：lazy capture 安全开关和调试断言

### 13.1 目标

让授权的 inplace graph 可以在运行时首次 lazy capture，同时避免意外 graph
capture 开放过大；补齐 `inplace_serial` 验证阶段必须的 input / metadata
地址断言，使 NPU fixed decode 验证失败时能直接定位到 descriptor、lazy
capture、input view 或 metadata stable buffer。

阶段 9 完成后应满足：

- load 阶段结束后全局 cudagraph capture 仍保持 disabled。
- 只有受控的 inplace FULL graph 首次运行时临时 enable capture。
- 临时 capture 结束或异常后恢复进入 wrapper 前的 capture enabled 状态。
- split graph 的 forward context 显式携带 lazy capture 许可。
- 开启 `inplace_validate_metadata_ptrs` 后，split replay 校验 input tensor
  地址和 attention metadata tensor 地址。
- 非 inplace graph、未授权 inplace graph 不会绕过全局 capture 禁止。

### 13.2 修改文件

```text
vllm_ascend/compilation/acl_graph.py
vllm_ascend/worker/model_runner_v3.py
tests/ut/compilation/test_acl_graph.py
tests/ut/test_inplace_split_execution_helpers.py
inplace_split_phase9_report.md
```

### 13.3 当前问题

`_capture_model()` 结束时调用：

```python
set_cudagraph_capturing_enabled(False)
```

但 offset graph 没有在 load 阶段预捕获，首次运行 split-1 时需要 capture。
12.10 修正后，split-0 必须复用启动阶段普通 graph；lazy capture 的授权对象应是
offset inplace descriptor，而不是 split-0。

阶段 8 已经在 `_make_split_batch_metadata_inplace_serial()` 中把 split-1
dispatch 成带 `start_num_tokens` 的 descriptor，但如果全局 capture disabled，
首次运行会被 `validate_cudagraph_capturing_enabled()` 拦截。

另外，阶段 8 只记录 input / metadata ptr 日志，没有把
`split_batch_config.inplace_validate_metadata_ptrs` 接到 replay 断言上。

### 13.4 建议方案

不要长期保持全局 capture enabled。只在 `ACLGraphWrapper.__call__()` 中发现
受控 inplace key 且 entry 为空时，临时打开。

受控 inplace key 必须同时满足：

```text
cudagraph_runtime_mode == CUDAGraphMode.FULL
forward_context.allow_inplace_lazy_capture is True
forward_context.split_inplace_mode in ("inplace_serial", "inplace_parallel")
batch_descriptor.graph_variant in ("inplace_serial", "inplace_parallel")
batch_descriptor.attention_backend in ("fia", "pa")
batch_descriptor.start_num_tokens > 0
```

未满足条件的 inplace entry 首次 capture 必须明确失败，不允许因为外部环境
偶然打开了 capture 而静默捕获。

伪代码：

```python
if entry.aclgraph is None and is_authorized_inplace_lazy:
    previous = compilation_monitor.cudagraph_capturing_enabled
    set_cudagraph_capturing_enabled(True)
    try:
        capture
    finally:
        set_cudagraph_capturing_enabled(previous)
```

### 13.5 context flag

在 inplace split context 构造后设置：

```python
split_forward_context.split_inplace_mode = "inplace_serial"
split_forward_context.allow_inplace_lazy_capture = (
    split_cfg.enable_inplace_lazy_capture
    and ubatch_cudagraph_mode == CUDAGraphMode.FULL
    and ubatch_batch_descriptor.start_num_tokens > 0
    and ubatch_batch_descriptor.graph_variant == "inplace_serial"
    and ubatch_batch_descriptor.attention_backend in ("fia", "pa")
)
split_forward_context.validate_inplace_input_ptrs = (
    split_cfg.inplace_validate_metadata_ptrs
    and ubatch_batch_descriptor.graph_variant == "inplace_serial"
)
split_forward_context.validate_inplace_metadata_ptrs = (
    split_cfg.inplace_validate_metadata_ptrs
    and ubatch_batch_descriptor.graph_variant == "inplace_serial"
)
```

当前 FIA baseline 只对 split-1 offset graph 打开 lazy capture；split-0 不打开
lazy capture，必须命中普通 graph。

### 13.6 安全限制

lazy capture 只允许：

- `split_inplace_mode in ("inplace_serial", "inplace_parallel")`
- `cudagraph_runtime_mode == FULL`
- descriptor 来自 dispatcher 的 authorized inplace FULL key
- `attention_backend in ("fia", "pa")`
- FIA + `capture_metadata_mode="template"` 时，不允许直接返回 capture output

以下安全限制由阶段 8 之前的 planner 入口继续保证：

- uniform decode
- `with_prefill == False`
- no DBO
- no spec decode
- no LoRA
- no M-RoPE
- no MLA
- no PCP/DCP/context parallel

### 13.7 调试断言

扩展 `ACLGraphEntry`，记录 capture 时的 metadata tensor 地址：

```python
attn_metadata_addresses: Optional[list[int]]
attn_metadata_tensor_infos: Optional[list[dict[str, Any]]]
```

新增 metadata tensor collector：

```python
_collect_attn_metadata_tensor_infos(attn_metadata)
```

第一版递归遍历：

- dict
- list / tuple
- dataclass
- 普通对象 `__dict__`

记录字段：

```text
path
ptr
shape
dtype
device
stride
is_contiguous
storage_offset
```

replay 时：

- `validate_inplace_input_ptrs=True`：即使不是 DEBUG 日志级别，也校验
  runnable tensor args 地址。
- `validate_inplace_metadata_ptrs=True`：校验 attention metadata tensor
  地址列表与 capture 时一致。
- 失败时写 split JSONL 事件并抛出 `AssertionError`。

### 13.8 JSONL 事件

阶段 9 新增或补强以下事件：

```text
inplace_lazy_capture_guard phase=enable|restore
inplace_lazy_capture_blocked
inplace_metadata_ptr_mismatch
split_descriptor.allow_inplace_lazy_capture
split_descriptor.validate_inplace_ptrs
```

其中 `inplace_lazy_capture_guard` 必须记录：

```json
{
  "entry_id": 123,
  "batch_descriptor": {
    "num_tokens": 32,
    "start_num_tokens": 384
  },
  "previous_capture_enabled": false
}
```

### 13.9 单测计划

新增或扩展单测：

```text
tests/ut/compilation/test_acl_graph.py
tests/ut/test_inplace_split_execution_helpers.py
```

覆盖：

- 全局 capture disabled 时，受控 offset key 首次运行会临时 enable capture。
- lazy capture 后恢复原始 capture enabled 状态。
- offset key 没有 context 许可时拒绝 capture。
- `validate_inplace_input_ptrs=True` 在 INFO 日志级别也能发现 input ptr 漂移。
- metadata collector 能递归收集常见 per-layer/common/decode metadata tensor。
- metadata ptr replay 漂移会触发 `AssertionError`。
- `inplace_serial` split-1 context 会设置 lazy capture 和 ptr validation flags。

### 13.10 阶段验收标准

- `python -m py_compile` 覆盖修改文件通过。
- ACLGraphWrapper 和 inplace execution helper 相关单测通过。
- debug JSONL 能区分 offset lazy capture 的 enable / restore / blocked。
- `inplace_validate_metadata_ptrs=True` 时，split-1 replay 会校验 input 和
  metadata ptr。
- 未执行 NPU fixed decode correctness；该项进入阶段 10。

## 14. 阶段 10：fixed batch correctness 修复与验证

### 14.1 目标

阶段 10 的目标不是“确认能跑完”，而是修复当前 `inplace_serial` fixed batch
的 correctness 错误，使它与 no-split padding baseline 在 token ids 和 text 上
严格一致。

当前 NPU 运行已经证明：

- `inplace_serial` 能进入 `384 + 32`。
- split-1 offset graph 可以 on-demand capture。
- 后续 step 可以继续执行，且没有 input / metadata ptr mismatch。

但这些只能说明 graph/capture/ptr 机制表面跑通，不能说明输出正确。当前已知
correctness 是错误的，阶段 10 必须把这个问题作为核心任务处理。

阶段 10 完成后必须能证明：

- disabled no-split baseline 与 enabled `inplace_serial` 输出严格一致。
- `384 + 32` 至少连续 3 个 decode step 真实触发。
- split-1 offset descriptor capture 后，后续 replay/update 的语义与 baseline
  等价。
- ptr validation 打开时无 input / metadata 地址漂移。

### 14.2 当前已知现象

最近 enabled-only smoke 命令：

```bash
python examples/test_split_batch_correctness_npu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split-mode inplace_serial \
  --fixed-batch-size 416 \
  --capture-sizes 256,384,512 \
  --max-tokens 8 \
  --validate-ptrs \
  --force-fixed-prompts \
  --run enabled \
  --output-dir /tmp/vllm_ascend_inplace_enabled_debug
```

观察到：

- step 4-8 连续 5 个 decode step 为 `384 + 32`。
- step 4 对 `(num_tokens=32,start_num_tokens=384)` 做了 lazy capture。
- step 5-8 没有再次 capture 同 descriptor，说明后续走已有 graph。
- split-1 input ptr 和 metadata ptr 在 step 4-8 稳定。
- step 9 出现 `384 + 31`，并对新的 `(31,start=384)` descriptor 做 lazy
  capture。
- 进程正常结束。

但该结果仍然是 correctness 错误场景，原因：

- 只跑了 `--run enabled`，没有 disabled baseline 对比。
- 之前或当前对比结果显示输出不等，不能把 enabled-only `DONE` 当作通过。
- capture 当次输出可能没有经过 replay/update，和后续 replay step 的语义不同。
- warmup/capture 使用真实请求 metadata，可能对 KV cache 有副作用。
- FIA TND `actual_seq_lengths_kv` 模板化、GraphParams update、block table、
  slot mapping 任何一个字段语义不一致，都可能在无 ptr mismatch 的情况下导致
  token 错。

### 14.3 优先假设

按风险从高到低排查：

1. **capture 当次输出未走 graph task update**

   `_run_inplace_serial_offset_capture()` 当前在 split-1 缺图时执行 warmup + capture，
   然后直接返回 capture output。普通 replay 路径会在 `self.model()` 返回后调用
   `_update_attn_params_for_split_ubatch()`，而 capture 当次没有同等 update/replay
   时序。若 capture output 使用的是 capture/template metadata 而不是 runtime
   updated params，首个 offset decode token 就可能错误。

2. **on-demand warmup 写真实 KV cache**

   warmup 使用真实 split-1 input / metadata，并写同一批 slot。即使同 token
   重复写理论上应幂等，也必须验证 cache op、attention backend 和 graph task
   update 没有非幂等副作用。

3. **FIA template seqlen 与 runtime seqlen 语义不一致**

   当前 offset graph 使用 `attention_backend="fia"`、
   `capture_metadata_mode="template"`。如果 capture、replay、update 阶段看到的
   `actual_seq_lengths_q/kv` 不一致，可能不报地址错误但输出错误。

4. **GraphParams key/update 使用了正确 key，但值来自错误 metadata**

   descriptor-aware key 已经存在，但仍要确认：

   - capture 写入 `(32,start=384)` key。
   - replay/update 读取同一个 key。
   - block table、workspace、handles、events 与 split-1 metadata 一致。

5. **merge 或 trim 顺序问题**

   split-0 和 split-1 输出 `torch.cat` 后必须等价于 no-split 前 416 个输出。
   如果 trim/cat 顺序或 shape 错误，会表现为局部 request 输出错。

### 14.4 修改文件

阶段 10 允许修改：

```text
examples/test_split_batch_correctness_npu.py
tests/ut/test_inplace_split_correctness_script.py
tests/ut/test_inplace_split_execution_helpers.py
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/compilation/acl_graph.py
vllm_ascend/attention/attention_v1.py
vllm_ascend/attention/utils.py
inplace_split_phase10_report.md
```

约束：

- 不实现 `inplace_parallel`。
- 不扩大支持范围到 non-uniform decode / DBO / MLA / CP。
- 不把 correctness 错误通过 fallback 到 no-split 或复制 input buffer 掩盖。
- 不默认打开 `VLLM_ASCEND_ACL_GRAPH_DEBUG=1`。

### 14.5 验证脚本要求

`examples/test_split_batch_correctness_npu.py` 需要支持：

```text
--split-mode parallel_buffer|inplace_serial|inplace_parallel
--fixed-batch-size 416
--capture-sizes 256,384,512
--validate-ptrs
--split-debug
--expect-split 384,32
--force-fixed-prompts
--fixed-prompt ...
--ignore-eos
--run both|disabled|enabled
--compare-mode subprocess|inproc
```

脚本行为：

- `--fixed-batch-size` 覆盖 `--batch-size`。
- fixed batch 且未显式 `--capture-sizes` 时，默认使用 `[256, 384, 512]`。
- fixed batch 自动保证 `max_num_seqs >= max(capture_sizes)`。
- `--split-mode inplace_serial` 自动打开 split inplace debug JSONL。
- `--validate-ptrs` 写入
  `split_batch_config.inplace_validate_metadata_ptrs=True`。
- `--force-fixed-prompts` 默认设置 `ignore_eos=True`。
- `--run both --compare-mode subprocess` 分别运行 disabled / enabled，再比较
  token ids 和 text。
- enabled-only 只能作为 smoke/debug，不能产出阶段 10 PASS。

### 14.6 正式验证命令

smoke 命令只用于复现和观察：

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
  --run enabled \
  --output-dir /tmp/vllm_ascend_inplace_phase10_smoke
```

阶段 10 验收必须使用 both subprocess：

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
  --output-dir /tmp/vllm_ascend_inplace_phase10
```

### 14.7 split trace 汇总

脚本从 `VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE` 生成：

```text
split_trace_summary.json
```

当前运行暴露一个重要细节：不能只用最后一个 observed split 判断是否通过。
`max_tokens=8` 的尾部可能出现 `384 + 31` 或 no-split 收尾 step。summary 应按
step 聚合，并记录 histogram。

必须检查：

- 至少 3 个 decode step 观察到 expected split `384 + 32`。
- expected split 的 split-1 descriptor 带：
  `num_tokens=32,start_num_tokens=384,graph_variant=inplace_serial`。
- `(32,start=384)` lazy capture 只出现一次。
- expected descriptor capture 后存在后续 execution，可计为 inferred replay。
- ptr stability 只在 expected descriptor 的 execution events 内计算。
- split-1 `input_ids` / `positions` / `query_start_loc` / `seq_lens` /
  `block_tables` / `slot_mapping` ptr 稳定。
- `384 + 31` 可以记录为额外 descriptor，但不能覆盖 expected split 判断。

建议字段：

```json
{
  "expected_split": {
    "total_tokens": 416,
    "first_tokens": 384,
    "second_tokens": 32,
    "second_start_num_tokens": 384
  },
  "expected_split_observations": {
    "count": 5,
    "step_ids": [4, 5, 6, 7, 8]
  },
  "observed_split_histogram": {
    "384+32@384": 5,
    "384+31@384": 1
  },
  "offset_graph": {
    "capture_count": 1,
    "inferred_replay_count": 4,
    "unexpected_capture_count": 0
  },
  "ptr_stability": {
    "input_ids": {"stable": true},
    "positions": {"stable": true},
    "query_start_loc": {"stable": true},
    "seq_lens": {"stable": true},
    "block_tables": {"stable": true},
    "slot_mapping": {"stable": true}
  },
  "failures": []
}
```

### 14.8 correctness 定位 artifacts

both run 失败时必须保存：

```text
prompts.json
outputs_split_disabled.json
outputs_split_enabled.json
diff.json
summary.json
metadata.json
console.log
split_inplace_debug_enabled.jsonl
split_trace_summary.json
```

`diff.json` 至少记录：

- mismatch total。
- 前 50 个 mismatch 的 request index。
- disabled/enabled token ids。
- disabled/enabled text preview。
- token mismatch 的第一个位置。

额外建议在调试版本中记录每个 decode step 的 first-token mismatch 分布：

```text
step_id
request_index
disabled_token_id
enabled_token_id
split_idx
start_num_tokens
descriptor
```

这样可以区分错误是否只从 split-1 capture 当次开始，还是 split-0 / merge
也被影响。

### 14.9 修复路线

#### P10.1 先固定比较入口

目标：

- both subprocess 必须产出 `PASS` 或可诊断的 `FAIL`。
- enabled-only 输出不能被误标记为 PASS。
- trace summary 不能因为尾部 `384 + 31` 覆盖 `384 + 32` 而误报。

完成标准：

- 单测覆盖 `384+32` 多次出现、尾部 `384+31` 的 synthetic trace。
- 单测覆盖 expected descriptor ptr stability。

#### P10.2 定位第一个错误 token

目标：

- 找出 disabled/enabled 第一个不一致 request。
- 判断错误发生在第几个 generated token。
- 映射该 token 对应的 split step 和 split idx。

判断：

- 如果第 1 个 decode token 就错，优先看 prefill 后首个 `384+32` capture/update。
- 如果第 2 个或后续 token 才错，优先看 offset graph replay/update。
- 如果只有 request index >= 384 错，优先看 split-1。
- 如果 request index < 384 也错，优先看 split-0 update、merge、全局 metadata。

#### P10.3 修复 capture 当次语义

当前 `_run_inplace_serial_offset_capture()` capture 后直接返回 capture output。
阶段 10 必须验证并修复：

- capture 当次是否需要立即 replay。
- capture 当次是否需要调用 `_update_attn_params_for_split_ubatch()`。
- capture output 是否可以作为真实请求输出使用。

可选方案：

1. capture 后立即 replay 同一 graph，再执行 graph task update，返回 replay output。
2. on-demand capture 使用 dummy/cache-safe metadata，真实请求随后走正常 replay/update。
3. 初始化或请求前预捕获 offset graph，真实请求永远只走 replay/update。

第一版优先选能最小改变当前路径且输出正确的方案。无论选哪种，都不能让
capture/template metadata 的输出直接绕过 runtime graph task update。

#### P10.4 验证 warmup 副作用

临时增加开关对比：

```text
inplace_lazy_capture_warmups = 0
inplace_lazy_capture_warmups = cudagraph_num_of_warmups
```

如果 warmup=0 正确而 warmup=1 错，说明真实 KV cache warmup 非幂等，需要改成
dummy warmup 或禁用 offset on-demand warmup。

如果 warmup=0 仍错，继续看 capture/replay/update 语义。

#### P10.5 验证 FIA metadata 语义

必须确认 capture、replay、update 三处使用一致的：

- `attention_backend`
- `capture_metadata_mode`
- `actual_seq_lengths_q`
- `actual_seq_lengths_kv`
- `query_start_loc`
- `seq_lens`
- `slot_mapping`
- `block_tables`
- GraphParams key

只允许在低层诊断时打开 `VLLM_ASCEND_ACL_GRAPH_DEBUG=1`。默认 correctness 命令
不能打开该开关，避免 replay 到 update 之间引入同步 CPU I/O。

#### P10.6 验证 merge/trim

增加轻量单测或 debug check：

- split-0 output shape 为 `[384, ...]`。
- split-1 output shape 为 `[32, ...]`。
- merge 后 shape 为 `[416, ...]`。
- request index `[0,383]` 来自 split-0。
- request index `[384,415]` 来自 split-1。

如果 only split-1 requests mismatch，merge/trim 大概率不是主因，但仍需保留检查。

### 14.10 输出一致性标准

与 split disabled baseline 对比：

- sampled token ids 严格一致。
- generated text 严格一致。
- no illegal memory access。
- no graph address mismatch。
- no metadata ptr mismatch。
- split trace validation 无 failure。

禁止把以下情况记为阶段 10 通过：

- 只跑 `--run enabled`。
- `summary.json.status != "PASS"`。
- 存在任何 token id mismatch。
- split trace 没有至少 3 个 `384 + 32` observation。
- split-1 offset descriptor 没有 capture 或没有后续 inferred replay。
- ptr validation 关闭。
- 通过 fallback no-split 绕过 `inplace_serial`。

### 14.11 单测计划

`tests/ut/test_inplace_split_correctness_script.py`：

- `416` + `[256,384,512]` 推导 `384 + 32`。
- `--capture-sizes` 覆盖 compilation config，且去重排序。
- fixed batch 自动把 `max_num_seqs` 提升到 `512`。
- trace summary 支持多次 expected observation。
- trace summary 不因尾部 `384 + 31` 失败。
- ptr stability 只按 expected descriptor 统计。
- 同 descriptor 多次 lazy capture 应失败。
- observation 少于 3 次应失败。

`tests/ut/test_inplace_split_execution_helpers.py`：

- capture 当次修复后的执行顺序。
- offset capture/replay/update 使用同一个 descriptor。
- split-1 output trim/merge 顺序。
- warmup=0 / warmup=1 行为开关不破坏 context 恢复。

### 14.12 阶段验收标准

- `python -m py_compile` 覆盖 correctness 脚本、runner、ACL graph 和新增单测通过。
- 相关单测通过。
- NPU 环境执行正式 both 命令后：
  - `summary.json.status == "PASS"`。
  - `diff.json` 不存在或 mismatch total 为 0。
  - `split_trace_summary.failures == []`。
  - 至少 3 个 decode step 为 `384 + 32`。
  - split-1 input / metadata ptr 稳定。
  - 日志无 illegal memory access、graph address mismatch、metadata ptr mismatch。

### 14.13 本地无 NPU 时

可以完成脚本、单元测试和静态检查，但真实 correctness 必须标记为未执行。
在 NPU both run 通过前，不能进入阶段 11。

## 15. 阶段 11：实现 inplace_parallel

### 15.1 目标

在 `inplace_serial` 正确后，打开主流和 parallel stream 并发 replay。

### 15.2 修改文件

```text
vllm_ascend/worker/model_runner_v3.py
vllm_ascend/compilation/acl_graph.py
```

### 15.3 复用当前能力

当前已有：

- `self.stream_main`
- `self.stream_parallel`
- `self.update_stream_main`
- `self.update_stream_parallel`
- `ACLGraphWrapper.concrete_aclgraph_entries`
- `ACLGraphWrapper.concrete_aclgraph_entries2`
- `graph_pool_parallel_streams`
- `_graph_params`
- `_graph_params_parallel`

这些都可继续用于 inplace_parallel。

### 15.4 与当前 parallel-buffer 路径的区别

当前 parallel-buffer:

- split-1 input 绑定到 `input_ids_parallel_streams`
- split-1 positions 绑定到 `positions_parallel_streams`
- split-1 inputs_embeds 绑定到 `inputs_embeds_parallel_streams`
- split-1 可 padding 到 parallel capture size

inplace_parallel:

- split-1 input 绑定到原始 input buffer offset view
- split-1 positions 绑定到原始 positions offset view
- split-1 inputs_embeds 绑定到原始 inputs_embeds offset view
- split-1 不 padding
- split-1 descriptor 带 `start_num_tokens`
- split-1 GraphParams key 带 offset
- split-1 metadata 使用 stable secondary metadata buffer

### 15.5 执行流程

可以新增：

```python
def _run_split_batch_inplace_parallel(...):
    ...
```

流程：

1. 构造 split-0 context，`in_parallel_streams=False`。
2. 构造 split-1 context，`in_parallel_streams=True`。
3. split-0 在 `stream_main` 上执行。
4. split-1 在 `stream_parallel` 上执行。
5. split-0 attention update 用 `_graph_params`。
6. split-1 attention update 用 `_graph_params_parallel`。
7. 分别 stream synchronize。
8. merge outputs。

### 15.6 并发安全检查

必须验证：

- split-0 和 split-1 的 `slot_mapping` 不重叠。
- split-0 和 split-1 的 block table refresh 不写同一 graph param tensor。
- split-1 使用 `_graph_params_parallel`。
- split-1 graph entry 进入 `concrete_aclgraph_entries2`。
- split-1 `cos_sin_slot_id` 不与 split-0 冲突。
- global cos/sin cache 不被并发覆盖。
- NPU attention workspace 按 stream/key 隔离。

### 15.7 阶段验收标准

在 `inplace_serial` 正确的 fixed batch 上：

- `inplace_parallel` 输出一致。
- split-0 和 split-1 确实在不同 stream replay。
- split-1 不使用 parallel input buffer。
- split-1 使用 parallel graph entry dict 和 parallel GraphParams。
- 无 metadata ptr mismatch。

## 16. 阶段 12：扩展支持范围

### 16.1 spec decode

要求：

- `uniform_decode_query_len = 1 + num_speculative_tokens`
- first_tokens 必须能被 `uniform_decode_query_len` 整除。
- first_reqs = first_tokens / query_len。
- second_reqs = second_tokens / query_len。

测试：

```text
num_speculative_tokens=1
query_len=2
capture_sizes=[256,384,512]
batch tokens=416
first=384
second=32
first_reqs=192
second_reqs=16
```

### 16.2 M-RoPE

需要修复所有 `positions[token_slice]`：

```python
positions[:, token_slice]
```

并验证：

- input positions ptr 稳定。
- cos/sin slicing 正确。
- `create_ascend_forward_context()` 不误切二维 positions。

### 16.3 MLA

MLA metadata 中可能嵌套：

- `decode`
- `prefill`
- `block_table`
- `seq_lens_list`
- `actual_seq_lengths_q`

需要扩展 stable metadata buffer 和 ptr collector。

### 16.4 PCP/DCP

PCP/DCP attention update 有独立函数：

- `update_attn_dcp_pcp_params`
- `update_mla_attn_dcp_pcp_params`

这些也必须使用 descriptor-aware GraphParams key。

第一版建议不打开 PCP/DCP，后续专项验证。

## 17. 阶段 13：性能 benchmark

### 17.1 目标

量化三种模式：

```text
PADDING
parallel_buffer
inplace_serial
inplace_parallel
```

### 17.2 关键指标

- TPOT
- replay_ms
- header_ms
- first graph replay time
- second graph replay/capture time
- second graph capture 次数
- padding token 数
- graph memory 增量
- NPU memory 峰值
- input copy 时间
- metadata copy 时间

### 17.3 理论 padding 对比

示例：

```text
capture_sizes=[256,384,512]
batch=416

PADDING:
  416 -> 512
  padding=96

parallel_buffer:
  384 + 32 padded to 128
  total=512
  padding=96
  benefit mainly comes from concurrency, not padding reduction

inplace:
  384 + 32
  total=416
  padding=0
```

### 17.4 benchmark 场景

建议 batch sizes：

```text
257, 300, 383, 385, 416, 480, 511
```

覆盖：

- 刚超过 bucket
- 接近下一个 bucket
- remainder 很小
- remainder 较大

### 17.5 阶段验收标准

- `inplace_serial` replay token 数少于 padding baseline。
- `inplace_parallel` 在 capture 稳定后 TPOT 优于或不差于 parallel_buffer。
- graph lazy capture 的一次性开销可从统计中分离。

## 18. 回退策略

### 18.1 配置回退

默认模式必须保持：

```text
parallel_buffer
```

如出现问题，用户可设置：

```json
{
  "split_batch_config": {
    "enabled": true,
    "mode": "parallel_buffer"
  }
}
```

或完全关闭：

```json
{
  "split_batch_config": {
    "enabled": false
  }
}
```

### 18.2 自动 fallback

以下情况自动 fallback 到当前 parallel-buffer 或 no split：

- `num_splits != 2`
- non-uniform decode
- DBO active
- with prefill
- no full ACL graph
- batch > max capture size
- first split 不命中 capture size
- second remainder <= 0
- metadata stable buffer 不支持当前 backend
- M-RoPE 未开启支持
- MLA/PCP/DCP 未开启支持

### 18.3 运行时错误回退

lazy capture 或 ptr validation 失败时：

- debug 模式直接 raise。
- production 可选 fallback 到 parallel_buffer，但要谨慎，因为同一步已经部分执行时不能安全重跑。

建议第一版 production 不自动吞错，直接报错并提示关闭 inplace。

## 19. 任务拆分清单

### 19.1 PR 1：配置和 descriptor

包含：

- `SplitBatchConfig.mode`
- `BatchDescriptor.start_num_tokens`
- `relax_for_mixed_batch_cudagraphs()` 保留 offset
- 单测

不包含执行路径变化。

### 19.2 PR 2：dispatcher lazy key

包含：

- dispatch 增加 `start_num_tokens`
- dispatch 增加 `allow_inplace_lazy_key`
- offset FULL graph lazy key
- 单测

不包含 model runner 修改。

### 19.3 PR 3：GraphParams descriptor-aware key

包含：

- `get_graph_param_key()`
- `ensure_graph_param_key()`
- `GraphParams` key 类型扩展
- `attention_v1.py` PA/FIA 修改
- `acl_graph.py` update functions 修改
- 先覆盖普通 attention

不包含 MLA/CP 全量支持。

### 19.4 PR 4：inplace planner 和 input slicing

包含：

- `_should_use_inplace_split()`
- `_make_inplace_split_slices()`
- second 不 padding
- second 使用 original buffer offset view
- debug 日志

暂不执行 inplace graph，可先 dry-run 输出 planner。

### 19.5 PR 5：metadata stable buffer

包含：

- secondary metadata buffers
- `_stabilize_inplace_common_attn_metadata()`
- metadata ptr collector
- 普通 attention decode 支持

### 19.6 PR 6：inplace_serial execution

包含：

- `_make_split_batch_metadata_inplace()`
- `_run_split_batch_inplace_serial()`
- execute_model mode dispatch
- fixed batch correctness test

### 19.7 PR 7：lazy capture safety

包含：

- offset graph 临时 capture enable
- lazy capture guard
- graph entry 日志补充 descriptor start
- input/metadata ptr validation

### 19.8 PR 8：inplace_parallel

包含：

- `_run_split_batch_inplace_parallel()`
- parallel stream 使用原始 offset view
- parallel GraphParams descriptor-aware key
- 并发 correctness test

### 19.9 PR 9：扩展模型和性能

包含：

- spec decode
- M-RoPE
- MLA
- PCP/DCP
- benchmark 脚本
- 文档

## 21. 最小完成定义

第一阶段最小完成定义是 `inplace_serial` 可用：

```text
uniform decode
full ACL graph
no DBO
num_splits=2
普通 attention
非 M-RoPE
非 MLA
非 PCP/DCP
```

必须满足：

- second 不 padding。
- second 不复制到 parallel input buffer。
- second graph key 包含 `start_num_tokens`。
- second GraphParams key 包含 offset。
- second metadata 地址稳定。
- second graph 可 lazy capture 并 replay。
- 输出与 padding baseline 一致。

第二阶段完成定义是 `inplace_parallel` 可用：

- 在第一阶段全部正确基础上，split-0 和 split-1 可以并发 replay。
- TPOT 相对 padding baseline 有收益。
- 无 graph params 竞争。
- 无 metadata 地址漂移。

## 22. 风险总表

| 风险 | 影响 | 应对 |
|---|---|---|
| 只改 BatchDescriptor，未改 GraphParams key | attention update 复用错误 handles/params | 阶段 4 必做 |
| second metadata 使用临时 tensor | replay 地址漂移或非法访问 | 阶段 7 stable buffer |
| lazy capture 被全局禁用 | second graph 首次运行失败 | 阶段 9 临时 enable |
| `num_splits > 2` | buffer 和 planner 语义不完整 | inplace 明确限制 2 |
| M-RoPE positions 切片错误 | 位置编码错误 | 第一版 fallback，后续专项支持 |
| parallel 双流共享资源竞争 | 输出错误或 NPU runtime error | 先 serial，再 parallel |
| offset graph key 过多 | graph memory 增长 | 限制 remainder 和 start 组合 |
| fallback 自动重跑不安全 | 同一步状态污染 | 第一版报错，不自动吞错 |

## 23. 推荐最终配置示例

### 23.1 当前默认行为

```json
{
  "split_batch_config": {
    "enabled": true,
    "mode": "parallel_buffer",
    "num_splits": 2,
    "enable_parallel_streams": true
  }
}
```

### 23.2 inplace serial 验证

```json
{
  "split_batch_config": {
    "enabled": true,
    "mode": "inplace_serial",
    "num_splits": 2,
    "min_batch_size_for_split": 4,
    "enable_inplace_lazy_capture": true,
    "inplace_validate_metadata_ptrs": true
  }
}
```

### 23.3 inplace parallel 优化

```json
{
  "split_batch_config": {
    "enabled": true,
    "mode": "inplace_parallel",
    "num_splits": 2,
    "min_batch_size_for_split": 4,
    "enable_inplace_lazy_capture": true,
    "inplace_validate_metadata_ptrs": false
  }
}
```

## 24. 结论

vllm-ascend 实现 inplace split 的正确路线是：

```text
先实现 offset graph 的正确性，再追求双流并发性能。
```

必须优先完成：

1. `BatchDescriptor.start_num_tokens`
2. dispatcher offset lazy key
3. Ascend GraphParams descriptor-aware key
4. second metadata stable buffer
5. inplace serial correctness

完成这些后，再把当前 parallel-buffer split 的双流框架迁移到 inplace input view 上，形成 `inplace_parallel`。

如果跳过 GraphParams key 或 metadata stable buffer，inplace 模式很可能出现表面能 capture、但 replay 使用错误 attention params 或 metadata 地址漂移的问题。该计划因此把这两项列为强制前置条件。
