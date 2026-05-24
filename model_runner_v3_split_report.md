# model_runner_v3.py split-batch 实现策略调研报告

## 1. 调研范围

本报告聚焦 `vllm_ascend/worker/model_runner_v3.py` 中当前 split-batch 的实现策略，并补充分析其依赖模块：

- `vllm_ascend/worker/model_runner_v3.py`
- `vllm_ascend/worker/ubatch_utils.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/ascend_config.py`
- `vllm_ascend/compilation/acl_graph.py`
- `vllm_ascend/ascend_forward_context.py`

调研重点包括：

- split-batch 的触发条件
- split slice 的生成策略
- attention metadata 的拆分方式
- 单流和并行流两套执行路径
- ACL graph 捕获、replay、GraphParams 的配套设计
- 当前实现的约束、隐患和建议验证项

## 2. 核心结论

当前 `model_runner_v3.py` 的 split-batch 是一个面向大规模 uniform decode batch 的优化路径。

它的基本策略是：

1. 只在 uniform decode 且 DBO/ubatch 未启用时尝试 split。
2. 按 request 连续区间拆分 batch，而不是按 token 任意切分。
3. 每个 split 生成独立的 `SplitBatchSlice`，并转换成 `UBatchSlice` 复用 ubatch 的 metadata 框架。
4. 每个 split 单独构建 attention metadata 和 forward context。
5. 每个 split 按 ACL graph capture size 做 padding。
6. 执行时分别 replay 每个 split 的 graph，输出 trim 掉 padding 后按原始顺序 concat。
7. 如果 `enable_parallel_streams=True`，split-0 走主流，split-1 走 parallel stream，并使用独立 buffer、graph pool 和 GraphParams，目标是并发 replay。

从实现形态看，当前代码虽然配置上支持 `num_splits >= 2`，但主路径和资源设计明显围绕 2-way split。尤其是并行流路径只有一套 parallel input buffer，prepare 阶段也只预处理了第二个 split，因此 `num_splits > 2` 需要额外验证，当前不应视为成熟支持。

## 3. 配置入口

配置对象定义在：

`vllm_ascend/ascend_config.py:314`

`SplitBatchConfig` 支持字段：

- `enabled`
- `enable_parallel_streams`
- `num_splits`
- `min_batch_size_for_split`
- `parallel_capture_sizes`
- `force_split`

默认行为：

- `enabled=False`
- `enable_parallel_streams=False`
- `num_splits=2`
- `min_batch_size_for_split=4`
- `parallel_capture_sizes=None`
- `force_split=False`

字段含义：

- `enabled`: 是否启用 split-batch。
- `enable_parallel_streams`: 是否使用主流 + parallel stream 并发 replay。
- `num_splits`: 拆分数量，默认 2，配置校验要求不小于 2。
- `min_batch_size_for_split`: request 数小于该值时不 split。
- `parallel_capture_sizes`: parallel stream 独立的 ACL graph capture sizes；未配置时复用主 capture sizes。
- `force_split`: 强制进入 split 路径，跳过 padding saving 判断，主要用于 benchmark 或排查。

## 4. 初始化阶段

### 4.1 并行流专用 buffer

`model_runner_v3.py` 初始化阶段新增了 parallel stream 专用输入 buffer：

`vllm_ascend/worker/model_runner_v3.py:610`

- `self.input_ids_parallel_streams`
- `self.inputs_embeds_parallel_streams`
- `self.positions_parallel_streams`
- `self.stream_parallel`

这些 buffer 用于 parallel split 的 replay 地址隔离。主 split 使用原始 `self.input_ids.gpu` / `self.positions.gpu`，非首 split 在并行路径里绑定到 parallel buffer。

### 4.2 模型 wrapper

`load_model()` 中的注释明确了当前策略：

`vllm_ascend/worker/model_runner_v3.py:3774`

- DBO: 使用 `AscendUBatchWrapper`
- Split-batch: 由 `execute_model()` 内部处理
- Full graph only: 使用 `ACLGraphWrapper`

也就是说，`model_runner_v3.py` 里 split-batch 没有使用旧的 `AscendSplitBatchWrapper` 作为外层模型 wrapper，而是在 `execute_model()` 中手动调度多个 split。

当 full ACL graph 启用且 DBO 未接管时，模型仍包在 `ACLGraphWrapper` 中：

- `self.update_stream`
- `self.update_stream_main`
- `self.update_stream_parallel`
- `self.model = ACLGraphWrapper(...)`

这意味着 split-batch 的每个 split replay 最终仍会进入 `ACLGraphWrapper.__call__()`，但 forward context 会告诉 wrapper 当前是否是 parallel stream，以及应使用哪个 `BatchDescriptor`。

### 4.3 attention metadata builder 数量

attention backend 初始化时会根据 split 配置预创建多个 metadata builder：

`vllm_ascend/worker/model_runner_v3.py:4249`

规则：

- DBO 开启时：2 个 builder。
- Split-batch 开启时：`split_config.num_splits` 个 builder。
- 默认：1 个 builder。

后续构建 split metadata 时，会按 `ubatch_id` 选择对应 builder：

`attn_group.get_metadata_builder(ubatch_id=ubid)`

## 5. split 触发条件

split 决策发生在 `_prepare_inputs()`：

`vllm_ascend/worker/model_runner_v3.py:932`

入口条件：

```python
if uniform_decode and ubatch_slices is None:
```

因此 split-batch 只面向以下场景：

1. 当前 batch 是 uniform decode。
2. DBO/ubatch 没有启用或没有成功产生 `ubatch_slices`。
3. `split_batch_config.enabled=True`。
4. request 数满足 `num_reqs >= min_batch_size_for_split`。

`uniform_decode` 的判断来自 `_prepare_inputs()` 中：

```python
uniform_decode = (
    max_num_scheduled_tokens == self.uniform_decode_query_len
    and total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens
)
```

也就是说，每个 request 的本轮 scheduled token 数都一致，常见 decode 场景是每个 request 1 token；spec decode 时可能是 `1 + num_speculative_tokens`。

当前实现注释强调 split-batch 与 DBO 不冲突：

- DBO 用于 overlapping compute/communication。
- Split-batch 用于拆大 uniform decode batch。
- DBO 已产生 `ubatch_slices` 时，split 不再触发。

## 6. split slice 生成策略

slice 结构定义在：

`vllm_ascend/worker/ubatch_utils.py:285`

`SplitBatchSlice` 包含：

- `request_slice`
- `token_slice`
- `padded_num_tokens`

其中：

- `num_requests = request_slice.stop - request_slice.start`
- `num_tokens = token_slice.stop - token_slice.start`
- 如果没有显式设置 `padded_num_tokens`，默认等于 `num_tokens`

### 6.1 基础拆分函数

基础拆分函数是：

`vllm_ascend/worker/ubatch_utils.py:395`

`split_batch_split()` 做以下事情：

1. 读取全局 `ascend_config.split_batch_config`。
2. 如果 `enabled=False`，直接返回 `(None, None)`。
3. 如果 `num_reqs < min_batch_size_for_split`，直接返回 `(None, None)`。
4. 调用 `create_split_batch_slices()` 生成 split slices。
5. 如果 split 数小于 2，返回 `(None, None)`。
6. 如果传入 `cudagraph_capture_sizes`，为每个 split 选择最小的可覆盖 capture size 作为 `padded_num_tokens`。

### 6.2 request 维度拆分

`create_split_batch_slices()` 逻辑在：

`vllm_ascend/worker/ubatch_utils.py:315`

默认策略：

- 根据 request 数做 ceil 均分：

```python
reqs_per_split = (num_reqs + num_splits - 1) // num_splits
```

- 对每个 split 计算：

```python
request_slice = slice(start_req, end_req)
token_slice = slice(cu_num_tokens[start_req], cu_num_tokens[end_req])
```

因为 split-batch 只在 uniform decode 下启用，按 request 拆分通常等价于按 token 拆分，并且不会切断单个 request。

如果传入 `custom_split_sizes`，则按指定 request 数拆分。该列表必须满足：

- `sum(custom_split_sizes) == num_reqs`
- `len(custom_split_sizes) == num_splits`

### 6.3 ACL graph padding

当传入 `cudagraph_capture_sizes` 时，`split_batch_split()` 会为每个 split 找到最小的 capture size：

`vllm_ascend/worker/ubatch_utils.py:452`

```python
padded_size = next(
    (cs for cs in sorted_capture_sizes if cs >= split_size),
    max_capture_size,
)
```

这意味着：

- 如果 split size 能被某个 capture size 覆盖，则 pad 到最小可覆盖 size。
- 如果 split size 超过最大 capture size，则 fallback 到 `max_capture_size`。

后者存在风险：如果 `split_size > max_capture_size`，则 `padded_num_tokens < split_size`，这在后续 graph replay 或 buffer 切片里是不安全的。正常路径会尽量避免这种情况，但 `force_split=True` 或配置不当时可能踩到。

## 7. graph-aware split 决策

`model_runner_v3.py` 在调用 `split_batch_split()` 前有一层 graph-aware 逻辑：

`vllm_ascend/worker/model_runner_v3.py:944`

它不是简单均分，而是试图让主流 split 命中已有 graph size，剩余部分交给 parallel stream。

关键变量：

- `sorted_main_sizes`: 主流 capture sizes。
- `max_main_size`: 最大主流 capture size。
- `main_reqs`: 最大的 `<= num_reqs` 的主流 capture size。
- `parallel_reqs`: `num_reqs - main_reqs`。
- `custom_split_sizes`: 如果决定 split，则通常为 `[main_reqs, parallel_reqs]`。

### 7.1 batch size 正好命中 graph

如果：

```python
main_reqs == num_reqs
```

默认不 split。原因是当前 batch 已经能直接命中主流 graph，不存在 padding 浪费。

如果 `force_split=True` 且存在更小 graph size，则使用次大的 graph size 做主 split：

```python
custom_split_sizes = [main_reqs, parallel_reqs]
```

这里的 `main_reqs` 会被重置为小于 `num_reqs` 的最大 graph size，`parallel_reqs` 是剩余部分。

### 7.2 batch size 未命中 graph 但可 split

如果：

```python
main_reqs > 0 and parallel_reqs > 0
```

则分几种情况：

- `force_split=True`: 直接 split。
- `num_reqs > max_main_size`: 默认不 split，因为超出最大 capture size 时普通阈值逻辑认为 split 收益不明确。
- 否则计算 padding saving。

padding saving 逻辑：

```python
original_padded = ceil_to_graph(num_reqs, sorted_main_sizes)
original_padding = original_padded - num_reqs
remainder_padded = ceil_to_graph(parallel_reqs, parallel_sizes)
remainder_padding = remainder_padded - parallel_reqs
padding_saved = original_padding - remainder_padding
```

只有：

```python
padding_saved > cudagraph_split_pad_threshold
```

才会 split。

`parallel_sizes` 优先使用：

```python
self.cudagraph_batch_sizes_parallel
```

否则回退到主流 capture sizes。

### 7.3 `num_splits` 的实际约束

这里有一个重要约束：graph-aware 路径生成的 `custom_split_sizes` 是二元列表：

```python
[main_reqs, parallel_reqs]
```

但 `create_split_batch_slices()` 要求：

```python
len(custom_split_sizes) == num_splits
```

因此，如果用户配置 `num_splits != 2`，graph-aware custom split 路径会出现不匹配。当前实现实际更适合 `num_splits=2`。

## 8. 输入准备策略

split slice 生成后，`_prepare_inputs()` 会继续准备真实 input tensor。

### 8.1 split 到 UBatchSlice 的转换

`split_batch_slices` 会转换为 `split_ubatch_slices`：

`vllm_ascend/worker/model_runner_v3.py:1048`

```python
split_ubatch_slices = [
    UBatchSlice(s.request_slice, s.token_slice)
    for s in split_batch_slices
]
```

这是复用 vLLM ubatch metadata 工具的关键。

### 8.2 parallel buffer 预拷贝

如果产生了 split，且 split 数大于 1，则 `_prepare_inputs()` 会提前处理第二个 split：

`vllm_ascend/worker/model_runner_v3.py:1174`

处理内容：

- 将第二个 split 的 positions 复制到 `positions_parallel_streams.gpu`。
- 将第二个 split 的 input_ids 复制到 `input_ids_parallel_streams.gpu`。
- 对第二个 split 的 tail padding 区域 fill 0。
- 对第一个 split 的 tail padding 区域也 fill 0。

这样在并行 replay 时：

- split-0 的 graph 看到主 buffer 起始地址。
- split-1 的 graph 看到 parallel buffer 起始地址。

这有助于避免两个 graph replay 共享输入地址或互相覆盖数据。

### 8.3 inputs_embeds 的特殊处理

`inputs_embeds` 没有在 `_prepare_inputs()` 中提前复制到 parallel buffer。代码注释说明原因是 embedding 可能在 `_prepare_inputs()` 返回后才填充。

因此 parallel 路径中，`inputs_embeds` 在 `_make_split_batch_metadata_parallel_streams()` 里复制：

`vllm_ascend/worker/model_runner_v3.py:1879`

这部分会在 `self.stream_parallel` 上执行，以期和主流工作重叠。

## 9. Attention metadata 拆分策略

### 9.1 attn_metadata 从 dict 变 list

如果启用 DBO 或 split，`attn_metadata` 会变成 list：

`vllm_ascend/worker/model_runner_v3.py:1230`

```python
if ubatch_slices is not None:
    attn_metadata = [dict() for _ in range(len(ubatch_slices))]
elif split_ubatch_slices is not None:
    attn_metadata = [dict() for _ in range(len(split_ubatch_slices))]
```

每个 list 元素是一个 per-layer metadata dict，对应一个 split。

### 9.2 split_attn_metadata

metadata 拆分函数在：

`vllm_ascend/attention/utils.py:422`

`split_attn_metadata()` 遍历每个 `UBatchSlice`，调用 `_make_metadata_with_slice()`。

`_make_metadata_with_slice()` 主要切分：

- `query_start_loc`
- `query_start_loc_cpu`
- `seq_lens`
- `seq_lens_cpu`
- `num_computed_tokens_cpu`
- `block_table_tensor`
- `slot_mapping`
- `positions`

关键语义：

- request 维度使用 `request_slice`。
- token 维度使用 `token_slice`。
- 如果 token slice 切到了 request 内部，会调整 `query_start_loc` 和 `seq_lens`。
- split-batch 正常是按 request 边界切，因此通常不会触发 request 内部切分逻辑。

### 9.3 per-layer metadata 构建

在 `_prepare_inputs()` 构造 per-layer metadata 时，如果 `split_ubatch_slices_for_metadata` 不为空，会对每个 split 单独 build：

`vllm_ascend/worker/model_runner_v3.py:1591`

GDN 分支和普通 full attention 分支都有相似逻辑：

```python
common_attn_metadata_list = split_attn_metadata(...)
for ubid, common_attn_metadata in enumerate(common_attn_metadata_list):
    attn_metadata_i = attn_group.get_metadata_builder(ubatch_id=ubid).build(...)
    attn_metadata[ubid][layer_name] = attn_metadata_i
```

同时 `_validate_split_attn_metadata_count()` 会校验拆出来的 metadata 数量是否等于 split 数：

`vllm_ascend/worker/model_runner_v3.py:395`

如果数量不匹配，会写 debug 文件并抛 `RuntimeError`。

## 10. Forward context 构造

split 执行前，每个 split 都需要独立 forward context。

### 10.1 单流 context 构造

单流路径使用：

`vllm_ascend/worker/model_runner_v3.py:1901`

`_make_split_batch_metadata()` 会为每个 split：

1. 选择该 split 的 `attn_metadata[i]`。
2. 通过 `cudagraph_dispatcher.dispatch(num_tokens=split_slice.padded_num_tokens, ...)` 获取匹配 capture 的 `BatchDescriptor`。
3. 调用 `create_ascend_forward_context(...)`。
4. 保存到 `AscendUbatchMetadata`。

其中 `ubatch_cudagraph_mode` 规则：

```python
CUDAGraphMode.FULL if aclgraph_runtime_mode != CUDAGraphMode.NONE else CUDAGraphMode.NONE
```

### 10.2 并行流 context 构造

并行路径使用：

`vllm_ascend/worker/model_runner_v3.py:1796`

`_make_split_batch_metadata_parallel_streams()` 做的事情类似，但额外设置：

- `in_parallel_streams=(i > 0)`
- `cos_sin_slot_id=i`
- 非首 split 使用 `self.stream_parallel` 构造 context
- 非首 split 的 input/position/embed 绑定到 parallel buffer

这使得 `ACLGraphWrapper` 能根据 forward context 选择主 graph entry 池或 parallel graph entry 池。

### 10.3 create_ascend_forward_context 的关键字段

`create_ascend_forward_context()` 定义在：

`vllm_ascend/ascend_forward_context.py:181`

关键字段：

- `attn_metadata`
- `dp_metadata`
- `cudagraph_runtime_mode`
- `batch_descriptor`
- `ubatch_slices`
- `in_parallel_streams`
- `cos_sin_slot_id`
- `dbo_enabled=True`

即使这里设置的是 `dbo_enabled=True`，split-batch 实际不是 DBO 的业务语义，而是复用了 ubatch 的 forward context 和 cos/sin slicing 机制。

## 11. 执行路径

执行入口在：

`vllm_ascend/worker/model_runner_v3.py:2743`

如果 `_prepare_inputs()` 返回 `split_batch_slices`，`execute_model()` 会再次转成 `split_ubatch_slices`，并根据配置选择执行路径：

- `enable_parallel_streams=True`: `_run_split_batch_parallel()`
- 否则：`_run_split_batch_gr0()`

### 11.1 单流路径 `_run_split_batch_gr0`

实现位置：

`vllm_ascend/worker/model_runner_v3.py:2010`

设计目标：

确保每个 split replay 时使用和 graph capture 一致的输入起始地址。

执行流程：

1. 调用 `_make_split_batch_metadata()` 为所有 split 准备 context 和 sliced input。
2. 备份原始 buffer 前缀：
   - input_ids
   - positions
   - inputs_embeds
   - slot_mapping
3. split-0 直接执行，因为数据已经在原始 buffer 起始位置。
4. 后续 split 执行前：
   - `torch.npu.synchronize()`
   - 把当前 split 的 input_ids/positions/inputs_embeds 复制到原始 buffer 起始位置。
   - 将 metadata 的 input tensor 重新绑定到原始 buffer 起始位置。
   - 将 split slot_mapping 复制到 base slot_mapping 前缀。
   - 重建 forward context，避免 stale context。
5. 每个 split 执行后调用 `_update_attn_params_for_split_ubatch()`。
6. 每个 split 输出通过 `_trim_split_output()` 去掉 padding。
7. 所有 split 输出通过 `_merge_split_outputs()` concat。
8. finally 恢复备份的原始 buffer 前缀。

单流路径的本质是“用同一套 graph capture 地址串行 replay 多个 split”。

### 11.2 并行流路径 `_run_split_batch_parallel_impl`

实现位置：

`vllm_ascend/worker/model_runner_v3.py:2355`

设计目标：

让 split-0 和 split-1 在不同 NPU stream 上并发 replay。

执行流程：

1. 调用 `_make_split_batch_metadata_parallel_streams()` 准备所有 split 的 metadata。
2. 设置主流和 parallel stream 的 core limit：

```python
torch.npu.set_stream_limit(self.stream_main, cube_num=15, vector_num=20)
torch.npu.set_stream_limit(self.stream_parallel, cube_num=15, vector_num=20)
```

3. 为每个 split 创建一个 Python thread。
4. thread 内根据 `slice_idx` 选择 stream：
   - `slice_idx == 0`: `stream_main`
   - `slice_idx > 0`: `stream_parallel`
5. 在对应 stream 上 override 当前 split 的 forward context 后调用 `self.model(...)`。
6. 如果是 full graph，调用 `_update_attn_params_for_split_ubatch()`。
7. trim 当前 split 的输出并写入 `results[slice_idx]`。
8. 所有 thread join。
9. 分别 synchronize 主流和 parallel stream。
10. merge outputs。

并行路径中 attention params 更新会传入：

```python
parallel_streams = slice_idx > 0
```

从而选择：

- `self.update_stream_main`
- `self.update_stream_parallel`

并在 `acl_graph.py` 里选择主 GraphParams 或 parallel GraphParams。

## 12. 输出合并策略

输出处理函数：

`vllm_ascend/worker/model_runner_v3.py:1972`

`_trim_split_output()` 支持：

- `torch.Tensor`
- `tuple`
- `IntermediateTensors`

它会按实际 `num_tokens` 裁剪掉 graph padding 区域。

合并函数：

`vllm_ascend/worker/model_runner_v3.py:1993`

`_merge_split_outputs()` 支持：

- `IntermediateTensors`: 每个 key 逐项 `torch.cat`
- `tuple`: tuple 内 tensor 逐项 cat
- 普通 tensor: 直接 `torch.cat(outputs, dim=0)`

因为 split slice 是连续 request/token 区间，按 split 顺序 concat 可以恢复原 batch token 顺序。

## 13. ACL graph 配套设计

### 13.1 parallel stream 独立 graph pool 和 entries

`ACLGraphWrapper` 中有两套 graph entry dict：

`vllm_ascend/compilation/acl_graph.py:340`

- `concrete_aclgraph_entries`
- `concrete_aclgraph_entries2`

并且有独立的 parallel graph pool：

```python
self.graph_pool_parallel_streams = torch.npu.graph_pool_handle()
```

在 `__call__()` 中，根据 forward context 的 `in_parallel_streams` 选择：

```python
current_concrete_aclgraph_entries = (
    self.concrete_aclgraph_entries2
    if in_parallel_streams
    else self.concrete_aclgraph_entries
)
selected_pool = (
    self.graph_pool_parallel_streams
    if in_parallel_streams
    else self.graph_pool
)
```

这避免了主流和 parallel stream 共享 graph pool 导致中间 tensor 地址冲突。

### 13.2 parallel stream 独立 GraphParams

`acl_graph.py` 维护两套 GraphParams：

`vllm_ascend/compilation/acl_graph.py:930`

- `_graph_params`
- `_graph_params_parallel`

初始化：

`vllm_ascend/compilation/acl_graph.py:949`

`set_graph_params_parallel()` 的注释说明了目的：parallel stream 会和主流并发运行，因此需要自己的 GraphParams，避免两边同时调用 `graph_task_update_begin/end` 时争用 handles/events。

获取逻辑：

```python
if in_parallel_streams and _graph_params_parallel is not None:
    return _graph_params_parallel
return _graph_params
```

### 13.3 parallel capture sizes

ACL graph capture 初始化中会设置主流和 parallel capture sizes：

`vllm_ascend/worker/model_runner_v3.py:4489`

主流：

```python
set_graph_params(self.cudagraph_batch_sizes)
```

parallel：

```python
_parallel_sizes = (
    _split_cfg.parallel_capture_sizes
    if _split_cfg is not None and _split_cfg.parallel_capture_sizes is not None
    else self.cudagraph_batch_sizes
)
self.cudagraph_batch_sizes_parallel = _parallel_sizes
set_graph_params_parallel(self.cudagraph_batch_sizes_parallel)
```

如果 `enable_parallel_streams=True`，`_capture_model()` 会做第二轮 capture：

`vllm_ascend/worker/model_runner_v3.py:4641`

第二轮 capture 使用：

- `self.cudagraph_batch_sizes_parallel`
- `in_parallel_streams=True`

这会让 `ACLGraphWrapper` 把 capture 结果放入 parallel graph entry 池。

### 13.4 parallel capture 时 clone block table

dummy capture 中，如果是 parallel stream，会 clone attn metadata 里的 block tables：

`vllm_ascend/worker/model_runner_v3.py:3660`

目的在注释中写得很明确：避免主流 `_graph_params` 和 parallel `_graph_params_parallel` 绑定同一块 device block_table storage。否则 runtime 两个 stream 同时刷新 block_table 前缀，会产生数据竞争，污染 KV cache lookup。

clone helper 定义在：

`vllm_ascend/worker/model_runner_v3.py:250`

`_clone_attn_metadata_block_tables()` 会 clone：

- metadata 自身的 `block_tables`
- `prefill` 子 metadata 的 `block_tables`
- `decode_meta` 子 metadata 的 `block_tables`

## 14. Attention params 更新策略

split 执行后会调用：

`vllm_ascend/worker/model_runner_v3.py:1736`

`_update_attn_params_for_split_ubatch()`。

它只在以下条件下执行：

- `forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL`
- 非 capture 阶段
- 非 sparse

根据模型类型选择：

- MLA: `update_mla_attn_params(...)`
- PCP/DCP: `update_attn_dcp_pcp_params(...)` 或 `update_mla_attn_dcp_pcp_params(...)`
- 普通 attention: `update_attn_params_split(...)`

普通 attention 的 split 专用更新函数是：

`vllm_ascend/compilation/acl_graph.py:713`

它和普通 `update_attn_params()` 的差异是启用：

```python
refresh_block_table=True
```

这样 `_update_attn_pa_params()` / `_update_attn_fia_params()` 会尝试把 runtime metadata 中的 block table 复制到 graph capture 时绑定的 block table buffer：

`vllm_ascend/compilation/acl_graph.py:532`

这是 split replay 正确性的关键，因为每个 split 的 block table / slot mapping 对应不同 request 区间。

## 15. Debug 和性能统计

### 15.1 split metadata debug

debug 文件路径来自环境变量：

```python
VLLM_ASCEND_SPLIT_METADATA_DEBUG_FILE
```

默认：

```text
vllm_ascend/split_metadata_debug.log
```

相关函数：

- `_append_split_metadata_debug`
- `_build_split_tensor_debug`
- `_validate_split_attn_metadata_count`

这些主要用于记录 split metadata 数量、tensor ptr、shape、head 等。

### 15.2 perf stats

`execute_model()` 中对 uniform decode 记录：

- `header_ms`
- `replay_ms`
- `batch_size`
- `is_split`

输出文件由环境变量控制：

```python
VLLM_ASCEND_PERF_STATS_FILE
```

为空则关闭。

### 15.3 隐式 dump 文件

单流和并行 split 执行路径都会在首次 split 成功后写：

```text
split_batch_merged_first_result_gg.json
```

路径是 `os.getcwd()`。

这对调试有用，但对生产路径而言是隐式 I/O 副作用，应考虑通过 env 开关控制或移除。

## 16. 当前实现的主要约束和风险

### 16.1 `num_splits > 2` 支持不完整

配置允许 `num_splits >= 2`，attention builder 也会按 `num_splits` 创建。

但实际主路径存在多个 2-way 假设：

1. graph-aware custom split 只生成 `[main_reqs, parallel_reqs]`。
2. `_prepare_inputs()` 只提前复制 `split_batch_slices[1]` 到 parallel buffer。
3. parallel path 只有一套 parallel buffer，所有 `i > 0` 都绑定同一地址。
4. parallel path 只有一个 `self.stream_parallel`，所有非首 split 都共享它。

因此当前实现实际可靠范围更接近：

```text
num_splits == 2
```

如果要支持更多 split，需要为每个非首 split 设计独立 buffer/stream/GraphParams，或者改为串行复用策略。

### 16.2 `force_split=True` 可能绕过安全收益判断

`force_split=True` 会跳过 padding saving 判断。

这对 benchmark 很有用，但如果 remainder 超过 parallel 最大 capture size，`split_batch_split()` 可能把 `padded_num_tokens` clamp 到 `max_capture_size`，导致 padded size 小于实际 token 数。

建议增加显式校验：

```python
if split_slice.padded_num_tokens < split_slice.num_tokens:
    raise ...
```

### 16.3 单流路径的 BatchDescriptor 可能使用 actual token

`_make_split_batch_metadata()` 中已经根据 `split_slice.padded_num_tokens` dispatch 过 batch descriptor。

但 `_run_split_batch_gr0()` 对后续 split 重建 context 时使用：

```python
BatchDescriptor(num_tokens=current_num_tokens, ...)
```

而不是 `split_slice.padded_num_tokens`。

如果该 split 需要 padding 才命中 graph，这里可能造成 replay key 或 attention params runtime shape 不一致。

并行路径中 attention update 明确使用：

```python
current_padded_num_tokens = split_slice.padded_num_tokens
```

单流路径也应检查是否需要统一成 padded shape。

### 16.4 M-RoPE / 2D positions 路径不一致

部分代码处理了二维 positions：

- `_prepare_inputs()` 拷贝 parallel positions 时检查 `self.positions.gpu.ndim == 2`
- `_slice_split_batch_inputs()` 检查 `positions.ndim == 2`

但也存在潜在不一致：

- `positions_parallel_streams` 初始化是一维 buffer。
- `attention/utils.py` 中 `_make_metadata_with_slice()` 使用 `attn_metadata.positions[token_slice]`，对二维 positions 可能应是 `positions[:, token_slice]`。
- `ascend_forward_context.py` 中 `create_ascend_forward_context()` 使用 `positions[token_slice]`，对二维 positions 也可能不正确。

如果 split-batch 要支持 M-RoPE 模型，需要专门验证。

### 16.5 parallel path 对并发正确性要求高

parallel path 同时依赖：

- 独立 input buffer
- 独立 graph pool
- 独立 GraphParams
- cloned block table
- separate update stream
- `in_parallel_streams` 正确传递
- `cos_sin_slot_id` 正确隔离

任何一个字段丢失都可能导致 graph replay 读写冲突或使用错误 attention params。

当前代码已经做了多处隔离，但整体复杂度较高，需要用 correctness test 和 debug ptr 日志持续覆盖。

### 16.6 隐式 debug dump 不适合生产默认开启

首次 split 后写 `split_batch_merged_first_result_gg.json`，这会带来：

- 额外同步或序列化开销
- 工作目录污染
- 多进程/多 rank 文件竞争风险

建议改为 env 开关控制。

### 16.7 split path 和 DBO 复用概念，语义容易混淆

split-batch 复用了 `UBatchSlice`、`AscendUbatchMetadata`、`create_ascend_forward_context()`，甚至设置 `dbo_enabled=True`。

实现上这是降低改造成本的做法，但语义上容易混淆：

- DBO 是 compute/communication overlap。
- split-batch 是 decode batch 分片和 graph padding 优化。

后续维护时要避免把 DBO 的假设直接套到 split-batch 上。

## 17. 建议验证项

建议至少覆盖以下场景：

1. `num_splits=2, enable_parallel_streams=False`
2. `num_splits=2, enable_parallel_streams=True`
3. batch size 正好命中主 graph size，确认默认不 split。
4. batch size 未命中 graph size，且 padding saving 大于阈值，确认 split。
5. batch size 未命中 graph size，且 padding saving 小于等于阈值，确认不 split。
6. `force_split=True` 下命中 graph size 的 batch，确认走 split。
7. `parallel_capture_sizes` 与主 capture sizes 不同，确认 parallel replay 使用 parallel graph pool。
8. `num_splits > 2`，确认当前行为是否失败或数据覆盖；若不支持，应显式禁止。
9. spec decode 下 `uniform_decode_query_len > 1` 的 split correctness。
10. M-RoPE 模型 split correctness。
11. PP 非首 rank 的 `IntermediateTensors` split/merge correctness。
12. sequence parallel enabled 时 `_slice_split_batch_inputs()` 的 intermediate tensor 切片 correctness。
13. 多 DP rank 下 split 与 `_sync_metadata_across_dp()` 的一致性。
14. block table clone 和 refresh 在 parallel path 下无数据竞争。
15. `VLLM_ASCEND_PERF_STATS_FILE` 下 split/non-split TPOT 对比。

## 18. 建议改进

### 18.1 明确限制 `num_splits=2`

如果短期目标是稳定 2-way split，建议在 split-batch 配置或 `_prepare_inputs()` 中显式限制：

```python
if split_config.num_splits != 2:
    raise ValueError("model_runner_v3 split-batch currently supports num_splits=2 only")
```

这样比隐式失败或并行 buffer 覆盖更安全。

### 18.2 对 padded size 增加安全校验

在 `split_batch_split()` 返回前增加：

```python
if split_slice.padded_num_tokens < split_slice.num_tokens:
    raise RuntimeError(...)
```

避免 graph size clamp 后产生非法 replay shape。

### 18.3 统一单流路径的 padded shape

检查 `_run_split_batch_gr0()` 中后续 split 重建 `BatchDescriptor` 时是否应使用：

```python
split_slice.padded_num_tokens
```

同时 attention update 也应考虑使用 padded shape，而不是 actual token count。

### 18.4 抽象 split runtime metadata

当前 split path 与 DBO path 共享 `UBatchSlice` / `AscendUbatchMetadata`，但语义不同。可以考虑新增更明确的结构：

- `SplitRuntimeSlice`
- `SplitRuntimeMetadata`
- `SplitReplayPlan`

这样可以把 request/token slice、actual tokens、padded tokens、stream kind、buffer kind、graph descriptor 显式组织起来。

### 18.5 为 parallel path 扩展资源模型

如果未来要支持 `num_splits > 2`，需要至少解决：

- 多套 parallel buffers
- 多个 streams 或串行复用策略
- 多套 graph pools 或明确共享策略
- 多套 GraphParams 或 per-stream key
- 输出排序和错误处理

### 18.6 移除或开关化 JSON dump

`split_batch_merged_first_result_gg.json` 建议通过 env 控制，例如：

```python
VLLM_ASCEND_SPLIT_DUMP_FIRST_RESULT=1
```

默认关闭。

## 19. 总结

当前 `model_runner_v3.py` 的 split-batch 实现是一个围绕 ACL full graph decode replay 的优化路径。它的设计目标不是通用 microbatch，而是针对大 uniform decode batch 的 graph padding 浪费和 replay 并发优化。

实现上，它把 batch 按 request 连续区间切分，复用 ubatch metadata 架构，为每个 split 构建独立 attention metadata 和 forward context。单流路径通过复制数据到 graph capture 的起始地址来保证 replay 地址一致；并行流路径通过独立 input buffer、独立 graph pool、独立 GraphParams 和 parallel capture 来支持 split-0 与 split-1 并发 replay。

当前代码最适合的运行模型是：

```text
uniform decode + no DBO + full ACL graph + num_splits=2
```

并行模式适合：

```text
split-0 命中主 graph size，split-1 使用 parallel graph size
```

需要重点关注的风险是：

- `num_splits > 2` 支持不完整
- `force_split` 可能绕过安全 shape 判断
- 单流路径 actual/padded shape 可能不一致
- M-RoPE/二维 positions 切片路径需要验证
- 隐式 JSON dump 不适合生产默认开启

建议在继续扩展前先把 2-way split 的 shape 校验、配置限制和 correctness tests 补齐，再考虑更通用的多 split 并行执行。
