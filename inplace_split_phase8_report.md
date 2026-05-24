# inplace split 阶段 8 完成报告

阶段 8 的目标是实现第一版真实 `inplace_serial` 执行路径，把阶段 5 planner、
阶段 6 input offset view、阶段 7 stable metadata 串起来执行两个 split。

本阶段已完成 `inplace_serial` execution slices 接入、stable metadata builder
接入、inplace serial metadata 构造、串行 split forward 路径和基础单元测试。
`inplace_parallel` 仍保持 dry-run。

## 完成内容

### 1. 阶段 8 详细计划

已更新 `inplace_split_implementation_plan.md` 的阶段 8 章节，补充：

- `inplace_serial` 目标和不做范围。
- `_prepare_inputs()` 接入规则。
- stable metadata builder 接入方式。
- descriptor dispatch 规则。
- execute_model 分支选择。
- JSONL 日志、验收标准、单测计划和阶段退出产物。

### 2. `inplace_serial` planner 结果进入真实 execution

修改 `vllm_ascend/worker/model_runner_v3.py`：

- 新增 `_inplace_plan_to_execution_slices(...)`。
- `mode=inplace_serial` 且 planner 成功时设置：
  - `split_batch_slices`
  - `split_ubatch_slices`
- `mode=inplace_parallel` planner 成功时仍保持 dry-run，不设置 execution slices。
- `inplace_serial` 下 `num_input_tokens` 使用真实 `total_num_scheduled_tokens`，
  不使用 no-split padding 后 token 数。

planner 成功时 decision 从阶段 5/6/7 的 dry-run 变为：

```text
inplace_split_execute
```

### 3. stable metadata 接入 builder

修改 `vllm_ascend/worker/model_runner_v3.py`：

- 新增 `_stabilize_inplace_common_attn_metadata_list(...)`。
- 在 split metadata 交给 attention builder 前，对 `inplace_serial` 的 split-1
  metadata 执行 stable buffer 绑定。
- split-0 保持原 metadata view。

这样真实 per-layer metadata 会基于阶段 7 的 stable common metadata 构造。

### 4. 新增 inplace serial metadata 构造

新增：

```python
_make_split_batch_metadata_inplace_serial(...)
```

行为：

- 每个 split 使用原始 input buffer view。
- split-1 不绑定 `*_parallel_streams` buffer。
- split-1 不复制回主 buffer 前缀。
- split descriptor 通过 dispatcher 生成。
- split-1 dispatch 传入：

```text
start_num_tokens = first_tokens
allow_inplace_lazy_key = enable_inplace_lazy_capture
```

JSONL 新增：

```text
inplace_serial_execution
```

记录 input view、metadata ptr、descriptor、token range 和 graph token 数。

### 5. 新增 inplace serial 执行路径

新增：

```python
_run_split_batch_inplace_serial(...)
```

执行流程：

1. 构造 inplace serial ubatch metadata。
2. split-0 在 main stream 执行。
3. 同步 main stream。
4. split-1 在 main stream 执行。
5. 每段按真实 token 数 trim。
6. concat 输出。

该路径不执行 parallel buffer copy，也不执行 `_run_split_batch_gr0()` 中的主 buffer
前缀搬运和恢复逻辑。

### 6. execute_model 分支

`execute_model()` 中 split 分支现在按模式选择：

```text
mode=inplace_serial     -> _run_split_batch_inplace_serial
parallel stream enabled -> _run_split_batch_parallel
otherwise               -> _run_split_batch_gr0
```

## 测试覆盖

新增 `tests/ut/test_inplace_split_execution_helpers.py`：

- `inplace_serial` planner 结果会转换成 execution slices。
- `inplace_parallel` planner 结果继续 dry-run。
- split output trim + merge 保持 token 顺序。

继续覆盖：

- `tests/ut/test_inplace_split_metadata_stabilization.py`
- `tests/ut/test_inplace_split_input_slicing.py`
- `tests/ut/test_inplace_split_debug.py`
- `tests/ut/test_inplace_split_planner.py`

## 验证结果

已执行：

```bash
python -m py_compile \
  vllm_ascend/worker/model_runner_v3.py \
  vllm_ascend/attention/utils.py \
  vllm_ascend/inplace_split_debug.py \
  tests/ut/test_inplace_split_execution_helpers.py \
  tests/ut/test_inplace_split_metadata_stabilization.py \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_debug.py \
  tests/ut/test_inplace_split_planner.py
```

结果：通过。

已执行：

```bash
python -m pytest \
  tests/ut/test_inplace_split_execution_helpers.py \
  tests/ut/test_inplace_split_metadata_stabilization.py \
  tests/ut/test_inplace_split_debug.py \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_planner.py \
  -q
```

结果：

```text
28 passed, 2 warnings
```

## 未执行内容

未执行 NPU fixed decode correctness / lazy capture 验证。因此以下仍需在 NPU 环境
确认：

- split-1 首次 offset lazy capture。
- split-1 第二次 replay。
- fixed batch 输出与 no-split padding 路径一致。
- GraphParams descriptor key 与实际 attention task update 完整匹配。

## 未改变内容

阶段 8 明确未改变以下内容：

- 未启用 `inplace_parallel` 并发路径。
- 未新增 block table dedicated stable buffer。
- 未放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP 支持。
- 未改变 `parallel_buffer` split 执行路径。
- 未扩大 lazy capture 安全策略；阶段 9 继续补安全开关和断言。

## 阶段结论

阶段 8 已完成代码接入和无 NPU 单测验证。

下一步进入阶段 9：补全 lazy capture 安全开关、debug 断言和 metadata/input ptr
校验，使 NPU fixed decode 验证失败时能快速定位 descriptor、GraphParams、metadata
或 input view 的具体问题。
