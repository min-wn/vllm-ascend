# inplace split 阶段 10 完成报告

阶段 10 的目标是把固定 batch correctness 验证入口落地，用同一套脚本验证
`inplace_serial` 与 no-split baseline 的输出一致性，并汇总阶段 9 的 split
debug JSONL 来检查 offset graph capture / replay 和 ptr 稳定性。

## 完成内容

### 1. 阶段 10 详细计划

已更新 `inplace_split_implementation_plan.md` 的阶段 10 章节，明确：

- 推荐 NPU 验证命令。
- correctness 脚本参数。
- split trace 汇总字段。
- 输出一致性标准。
- 本地无 NPU 时的验收边界。

后续阶段已顺延：

- 阶段 11：实现 `inplace_parallel`。
- 阶段 12：扩展支持范围。
- 阶段 13：性能 benchmark。

### 2. correctness 脚本支持 inplace fixed batch

扩展 `examples/test_split_batch_correctness_npu.py`，新增参数：

```text
--split-mode parallel_buffer|inplace_serial|inplace_parallel
--fixed-batch-size
--capture-sizes
--validate-ptrs
--split-debug
--expect-split
--force-fixed-prompts
--fixed-prompt
--ignore-eos
```

推荐阶段 10 命令：

```bash
python examples/test_split_batch_correctness_npu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split-mode inplace_serial \
  --fixed-batch-size 416 \
  --capture-sizes 256,384,512 \
  --max-tokens 8 \
  --validate-ptrs \
  --force-fixed-prompts \
  --output-dir /tmp/vllm_ascend_inplace_phase10
```

脚本现在会：

- 使用 `--fixed-batch-size` 覆盖 `--batch-size`。
- fixed batch 下默认 capture sizes 为 `[256, 384, 512]`。
- 在 `inplace_serial` fixed batch 验证中自动收集 split debug JSONL。
- 把 `--validate-ptrs` 写入
  `split_batch_config.inplace_validate_metadata_ptrs=True`。
- 使用固定 prompt 和 `ignore_eos=True` 降低 decode batch 提前缩小风险。
- 要求 split trace validation 时 `--max-tokens >= 3`，以便观察一次 capture
  和后续 replay。

### 3. split trace 汇总

新增 debug JSONL 汇总逻辑，输出：

```text
split_trace_summary.json
```

汇总内容包括：

- event counts。
- 最新 planner decision。
- 最近 descriptor。
- expected / observed split。
- offset graph capture / replay 次数；默认不强制校验该项。
- split-1 input / metadata ptr 稳定性。
- failures 列表。

当 `--expect-split` 或 fixed batch 自动推导出 expected split 时，脚本会校验：

- observed split 等于 expected split。
- 连续 step 中 split-1 ptr 无漂移。

### 4. hot path 日志修复

阶段 10 验证时发现一个关键约束：`ACLGraphWrapper.__call__()` 运行在
`self.model()` 内部，而调用方会在 `self.model()` 返回后紧接着执行
`update_attn_params` / graph task update。因此 replay 到 update 之间不能有同步
CPU 文件 I/O。

已调整：

- `_append_acl_graph_debug(...)` 默认 no-op。
- 只有显式设置 `VLLM_ASCEND_ACL_GRAPH_DEBUG=1` 时才写
  `acl_graph_debug.log`。
- 正常 `acl_graph_replay` 不再写 split JSONL。
- 正常 `acl_graph_capture` 不再写 split JSONL。
- 不再在 replay 后构造 block table diag 并同步写 JSONL。
- ptr mismatch 等异常路径仍保留诊断写入后 raise。

阶段 10 correctness / benchmark 默认不应打开 `VLLM_ASCEND_ACL_GRAPH_DEBUG=1`。

### 5. fixed batch 验证发现的 descriptor 修复

enabled-only NPU 验证跑到 `inplace_serial` 后暴露：

```text
ValueError: inplace offset keys only support uniform decode
```

根因：

- planner 输入已经是 uniform decode：`num_reqs=416`、
  `total_num_scheduled_tokens=416`、每请求本 step 调度 1 token。
- 但 `_make_split_batch_metadata_inplace_serial()` 给 dispatcher 传的是外层
  `batch_descriptor.uniform`。
- 该外层 descriptor 在经历 prefill / 保守 dispatch 后可能仍是 `False`。
- split 执行路径本身只会在 planner 接受 uniform decode 后进入，因此这里应该
  传 split invariant：`uniform_decode=True`。

已修复：

```python
self.cudagraph_dispatcher.dispatch(
    num_tokens=split_slice.graph_num_tokens,
    uniform_decode=True,
    ...
)
```

并补充单测：即使外层 `BatchDescriptor.uniform=False`，inplace split dispatch
也必须传 `uniform_decode=True`。

### 6. enabled-only NPU 验证发现的 runtime mode 修复

enabled-only NPU 验证继续暴露一个调度问题：

```text
RuntimeError: ... current working operator name is ReshapeCacheOperation
ERR00100 PTA call acl api failed
```

同时 split trace 显示：

```json
{"event":"split_descriptor","idx":0,"runtime_mode":"NONE","batch_descriptor":{"num_tokens":384,"start_num_tokens":0}}
{"event":"split_descriptor","idx":1,"runtime_mode":"NONE","batch_descriptor":{"num_tokens":32,"start_num_tokens":384}}
```

根因：

- 外层 batch 是 416 token，普通 FULL graph dispatch 可能因为没有可用的外层
  416 graph key 而返回 `CUDAGraphMode.NONE`。
- 但 inplace serial 已经把该 batch 拆成 `384 + 32`：
  - 384 是已捕获的主 slice graph。
  - 32 是带 `start_num_tokens=384` 的 offset lazy graph。
- `_make_split_batch_metadata_inplace_serial()` 之前在 split 自己 dispatch 之后，
  又用外层 `aclgraph_runtime_mode == NONE` 强制把 split runtime mode 改回
  `NONE`。
- 结果两个 split 都绕过 ACLGraphWrapper，直接执行 torch compiled runnable，
  在 NPU 异步执行中报 `ReshapeCacheOperation`。

已修复：

- 删除 inplace serial metadata 构造中的外层 `NONE` 覆盖。
- split slice runtime mode 完全由 split 自己的 descriptor dispatch 决定。
- 如果全局没有 ACL graph 或 dispatcher 不支持对应 key，split dispatch 自身仍会
  返回 `NONE`；不需要外层 416 的结果覆盖。

补充单测：当外层 runtime mode 是 `CUDAGraphMode.NONE` 时，inplace serial 的
384/32 split 仍必须保留各自 dispatch 得到的 `CUDAGraphMode.FULL`。

### 7. offset lazy capture 零拷贝语义校正

继续运行 enabled-only 后，offset graph 已经进入 `ACLGraphWrapper` capture 分支，
但在首次 lazy capture 内报错：

```text
RuntimeError: copy_between_host_and_device_opapi ... aclrtMemcpy, error code is 107030
Not allow to synchronize captured-stream
When layout is TND and PA not enabled, keyT(256) and valueT(256) must be equal
to the last element of actualSeqenceLengthKV(9)
```

trace 显示：

- idx0/idx1 的 split descriptor 均已是 `runtime_mode="FULL"`。
- idx1 是 `allow_inplace_lazy_capture=true`，失败发生在 offset graph 首次 capture。
- idx1 的 `input_ids` / `positions` 是主 batch buffer 上的 view：
  `storage_offset=384`。
- idx1 的 stable `slot_mapping` dtype 与真实 metadata 不一致，之前为 `int64`，
  而真实 slot mapping 为 `int32`。

已确认 / 已修正：

- 按 GPU DUAL_INPLACE 语义，idx1 必须继续使用原 batch buffer 的 offset view，
  依靠 `BatchDescriptor.start_num_tokens=384` 区分 graph key；不能为了绕过
  capture 报错把 `input_ids` / `positions` 复制到 secondary input buffer。
- 已删除 offset split 输入稳定化拷贝逻辑，`inplace_serial` 重新保持零拷贝
  input view。
- `inplace_slot_mapping_secondary` dtype 改为 `torch.int32`，匹配当前 NPU
  slot mapping。
- 移除 `full_graph_pa` capture 分支中的临时 debug `tolist()/print`，避免 graph
  capture 路径产生 CPU 同步 / I/O。

补充单测：idx1 传入模型的 input / position view 必须保持
`storage_offset=384`，且 `data_ptr()` 等于原始 buffer offset 地址。

剩余问题仍应从 capture 内 attention metadata / graph params 更新路径定位，
不能通过复制模型输入 buffer 解决。

### 8. 单元测试

新增 `tests/ut/test_inplace_split_correctness_script.py`，覆盖：

- `416` + `[256, 384, 512]` 推导出 `384 + 32`。
- `--capture-sizes` 覆盖 compilation config。
- `inplace_serial` split config 正确写入 lazy capture 和 ptr validation。
- synthetic split JSONL 可以汇总出 expected split、capture/replay 和 ptr 稳定。

扩展 `tests/ut/test_inplace_split_execution_helpers.py`：

- 覆盖 inplace split metadata 构造时 dispatcher 收到的 `uniform_decode=True`。
- 覆盖外层 416 dispatch 为 `NONE` 时，split 级别 dispatch 不被覆盖。
- 覆盖 offset split 使用原始 input buffer 的非零 offset view，保持零拷贝语义。

## 验证结果

已执行：

```bash
python -m py_compile \
  vllm_ascend/worker/model_runner_v3.py \
  tests/ut/test_inplace_split_execution_helpers.py
```

结果：通过。

已执行：

```bash
python -m pytest tests/ut/test_inplace_split_execution_helpers.py -q
```

结果：

```text
6 passed, 2 warnings
```

已执行：

```bash
python -m py_compile \
  examples/test_split_batch_correctness_npu.py \
  tests/ut/test_inplace_split_correctness_script.py \
  vllm_ascend/compilation/acl_graph.py
```

结果：通过。

已执行：

```bash
python -m pytest tests/ut/test_inplace_split_correctness_script.py -q
```

结果：

```text
4 passed, 2 warnings
```

补充执行：

```bash
python -m pytest \
  tests/ut/compilation/test_acl_graph.py \
  tests/ut/test_inplace_split_correctness_script.py \
  -q
```

结果：`36 passed` 后有 2 个 PCP/DCP 测试因当前机器 NPU 显存被运行中的
EngineCore 占用而触发 `Memory_Allocation_Failure(EL0004)`。该失败发生在
`torch.npu.current_stream()` 初始化设备阶段，不是 hot path 日志改动导致的单测
断言失败。

## 未执行内容

当前环境未执行真实 NPU fixed decode correctness。因此以下仍需在 NPU 环境执行推荐
命令确认：

- split disabled 与 `inplace_serial` sampled token ids / text 完全一致。
- split-1 第一次 offset graph capture。
- split-1 后续 replay。
- split-1 input / metadata ptr 在连续 decode step 中稳定。
- `summary.json.status == "PASS"`。
- `split_trace_summary.failures == []`。

## 阶段结论

阶段 10 已完成验证工具、trace 汇总、计划文档和无 NPU 单元测试。

下一步进入阶段 11：在阶段 10 的 fixed batch correctness 通过后，打开
`inplace_parallel` 并验证并发 replay 的输出一致性和 GraphParams/graph entry 隔离。
