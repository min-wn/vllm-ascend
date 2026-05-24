# inplace split 阶段 7 完成报告

阶段 7 的目标是实现 inplace split-1 attention metadata stable buffer，并在
dry-run 中验证稳定化后的 metadata tensor 地址和内容。

本阶段已完成 stable metadata helper、runner secondary buffer、inplace dry-run
metadata ptr 日志和单元测试。真实 `inplace_serial` 执行仍未启用。

## 完成内容

### 1. 阶段 7 详细计划

已更新 `inplace_split_implementation_plan.md` 的阶段 7 章节，补充：

- 阶段目标和不做范围。
- secondary buffer 字段。
- stable metadata 构造流程。
- copy 字段和暂不稳定字段。
- JSONL 日志字段。
- 单元测试计划、验证命令和阶段退出产物。

### 2. stable metadata helper

修改 `vllm_ascend/attention/utils.py`，新增：

```python
stabilize_inplace_common_attn_metadata(...)
```

该 helper 对 split-1 执行：

- `query_start_loc` 复制到 stable GPU buffer。
- `query_start_loc_cpu` 复制到 stable CPU buffer。
- `seq_lens` 复制到 stable GPU buffer。
- `seq_lens_cpu` 复制到 stable CPU buffer。
- `slot_mapping` 复制到 stable GPU buffer。
- 返回新的 `AscendCommonAttentionMetadata`，上述字段指向 stable buffer view。

split-0 保持原 metadata，不复制到 secondary buffer。

容量断言：

- `nreq + 1 <= max_num_reqs + 1`
- `ntok <= max_num_tokens`

### 3. runner secondary buffer

修改 `vllm_ascend/worker/model_runner_v3.py`，新增：

```python
self.inplace_query_start_loc_secondary
self.inplace_seq_lens_secondary
self.inplace_slot_mapping_secondary
```

这些 buffer 只服务于 inplace split-1 stable metadata。

### 4. dry-run metadata 稳定化

修改 `vllm_ascend/worker/model_runner_v3.py`，新增：

```python
_stabilize_inplace_common_attn_metadata(...)
_dry_run_inplace_stable_metadata(...)
```

当 `inplace_split_plan is not None` 且 debug 开启时：

1. 使用 `split_attn_metadata()` 生成逻辑 split metadata。
2. split-0 保持原 metadata。
3. split-1 复制到 secondary buffer。
4. 记录 `inplace_metadata_views` JSONL。
5. 不把 stable metadata 交给 builder。
6. 不改变实际 forward 路径。

### 5. debug 输出扩展

修改 `vllm_ascend/inplace_split_debug.py`，新增：

```python
common_metadata_tensor_info(...)
```

输出字段：

- `query_start_loc`
- `query_start_loc_cpu`
- `seq_lens`
- `seq_lens_cpu`
- `block_table_tensor`
- `slot_mapping`
- `num_computed_tokens_cpu`
- `positions`

新增 JSONL event：

```text
inplace_metadata_views
```

固定样例 `416 -> 384 + 32` 中，second split 预期：

```text
split_idx = 1
stabilized = true
start_num_tokens = 384
num_reqs = 32
num_tokens = 32
stable.query_start_loc.shape = [33]
stable.seq_lens.shape = [32]
stable.slot_mapping.shape = [32]
```

## 测试覆盖

新增 `tests/ut/test_inplace_split_metadata_stabilization.py`：

- split-1 stable metadata 内容等于 original split metadata。
- split-1 stable metadata ptr 来自 secondary buffer。
- repeated stabilize 后 `query_start_loc` / `seq_lens` / `slot_mapping` ptr 不变。
- split-0 不复制，原样返回。
- `nreq` / `ntok` 容量断言。

扩展 `tests/ut/test_inplace_split_debug.py`：

- `common_metadata_tensor_info()` 输出 CPU/GPU common metadata 字段。

## 验证结果

已执行：

```bash
python -m py_compile \
  vllm_ascend/worker/model_runner_v3.py \
  vllm_ascend/attention/utils.py \
  vllm_ascend/inplace_split_debug.py \
  tests/ut/test_inplace_split_metadata_stabilization.py \
  tests/ut/test_inplace_split_debug.py
```

结果：通过。

已执行：

```bash
python -m pytest \
  tests/ut/test_inplace_split_metadata_stabilization.py \
  tests/ut/test_inplace_split_debug.py \
  tests/ut/test_inplace_split_input_slicing.py \
  tests/ut/test_inplace_split_planner.py \
  -q
```

结果：

```text
25 passed, 2 warnings
```

## 未执行内容

未执行 NPU fixed decode dry-run 日志验证。本阶段验证集中在 stable metadata
helper、debug 输出和语法检查。

## 未改变内容

阶段 7 明确未改变以下内容：

- 未启用真实 `inplace_serial` 执行路径。
- 未把 stable metadata 交给 attention builder。
- 未触发 split-1 offset lazy capture。
- 未启用 `inplace_parallel` 并发执行。
- 未新增 block table dedicated stable buffer。
- 未放开 MLA、M-RoPE、LoRA、spec decode、PCP/DCP/MTP 支持。
- 未改变 `parallel_buffer` 的 metadata 构造和执行语义。

## 阶段结论

阶段 7 已完成。

下一步进入阶段 8：实现 `inplace_serial` 执行路径，把阶段 6 的 input offset
view 和阶段 7 的 stable metadata 接入真实串行 split forward。
