# inplace split 阶段 9 完成报告

阶段 9 的目标是补全 `inplace_serial` 所需的 lazy capture 安全开关和
debug 断言，使 split-1 offset graph 可以受控地首次运行时 capture，并在
后续 replay 时校验 input / metadata 地址稳定性。

## 完成内容

### 1. 阶段 9 详细计划

已更新 `inplace_split_implementation_plan.md` 的阶段 9 章节，明确：

- offset lazy capture 的允许条件。
- forward context flags。
- input / metadata ptr 断言。
- JSONL 事件。
- 单测计划和验收标准。

### 2. 受控 offset lazy capture

修改 `vllm_ascend/compilation/acl_graph.py`：

- 引入 `vllm.compilation.monitor`，读取并恢复原始
  `cudagraph_capturing_enabled` 状态。
- 只允许满足以下条件的 offset graph 首次 capture：
  - `batch_descriptor.start_num_tokens > 0`
  - `cudagraph_runtime_mode == CUDAGraphMode.FULL`
  - `forward_context.allow_inplace_lazy_capture is True`
  - `forward_context.split_inplace_mode in ("inplace_serial", "inplace_parallel")`
- 对允许的 offset graph，在 capture 前临时打开 cudagraph capture，capture
  后恢复原状态。
- 对未授权 offset graph，直接抛出明确 `RuntimeError`，避免意外 capture。

新增 JSONL 事件：

```text
inplace_lazy_capture_guard
inplace_lazy_capture_blocked
```

阶段 10 验证后补充约束：`inplace_lazy_capture_guard` 属于
`ACLGraphWrapper.__call__()` 内部热路径事件，默认不再写 split JSONL。只有显式
打开 `VLLM_ASCEND_ACL_GRAPH_DEBUG=1` 做低层诊断时才允许记录，避免在
`self.model()` replay 返回前引入同步 CPU 文件 I/O。

### 3. inplace context flags

修改 `vllm_ascend/worker/model_runner_v3.py`：

- 在 `_make_split_batch_metadata_inplace_serial(...)` 中为 split context 设置：
  - `split_inplace_mode`
  - `allow_inplace_lazy_capture`
  - `validate_inplace_input_ptrs`
  - `validate_inplace_metadata_ptrs`
- 只对 split-1 offset FULL graph 打开 lazy capture 和 ptr validation。
- `split_descriptor` debug payload 新增：
  - `allow_inplace_lazy_capture`
  - `validate_inplace_ptrs`

### 4. input / metadata ptr 断言

修改 `vllm_ascend/compilation/acl_graph.py`：

- `ACLGraphEntry` 新增 metadata ptr baseline 字段。
- 新增 `_collect_attn_metadata_tensor_infos(...)`，递归收集 attention metadata
  中的 tensor 地址和形状信息。
- `validate_inplace_input_ptrs=True` 时，即使日志级别不是 DEBUG，也校验
  replay input tensor arg 地址。
- `validate_inplace_metadata_ptrs=True` 时，校验 replay attention metadata
  tensor 地址与 capture 时一致。
- metadata ptr 漂移时记录 `inplace_metadata_ptr_mismatch` 并抛出
  `AssertionError`。

## 测试覆盖

新增/扩展：

```text
tests/ut/compilation/test_acl_graph.py
tests/ut/test_inplace_split_execution_helpers.py
```

覆盖点：

- 全局 capture disabled 时，受控 offset key 首次运行会临时 enable capture。
- lazy capture 后恢复原始 capture enabled 状态。
- offset key 没有 context 许可时拒绝 capture。
- `validate_inplace_input_ptrs=True` 在 INFO 日志级别也能发现 input ptr 漂移。
- metadata collector 能递归收集 common/decode metadata tensor。
- metadata ptr replay 漂移会触发 `AssertionError`。
- `inplace_serial` split-1 context 会设置 lazy capture 和 ptr validation flags。

## 验证结果

已执行：

```bash
python -m py_compile \
  vllm_ascend/compilation/acl_graph.py \
  vllm_ascend/worker/model_runner_v3.py \
  tests/ut/compilation/test_acl_graph.py \
  tests/ut/test_inplace_split_execution_helpers.py
```

结果：通过。

已执行：

```bash
python -m pytest \
  tests/ut/compilation/test_acl_graph.py \
  tests/ut/test_inplace_split_execution_helpers.py \
  -q
```

结果：

```text
38 passed, 2 warnings
```

## 未执行内容

未执行 NPU fixed decode correctness / real ACL lazy capture 验证。因此以下仍需
阶段 10 在 NPU 环境确认：

- split-1 第一次 offset graph 真实 capture。
- split-1 后续 replay。
- `inplace_validate_metadata_ptrs=True` 下连续 decode step 地址稳定。
- 输出与 no-split 或 `parallel_buffer` baseline 一致。

## 阶段结论

阶段 9 已完成 lazy capture 安全开关、context flags、input / metadata ptr
断言和无 NPU 单元测试。

下一步进入阶段 10：固定 batch correctness 验证，重点使用 batch size 416、
capture sizes `[256, 384, 512]` 验证 `384 + 32` 的 offset capture / replay
和输出一致性。
