# GPU DUAL_INPLACE 模式实现调研

本文基于当前工作区下列文件做静态代码调研：

- `/vllm-workspace/gpu_model_runner.py`
- `/vllm-workspace/cudagraph_dispatcher.py`
- `/vllm-workspace/cudagraph.py`
- `/vllm-workspace/gpu代码总结.md`
- 交叉检查：`/vllm-workspace/vllm/vllm/forward_context.py`
- 用户随后提供路径：`/vllm-workspace/forward_context.py`

结论先行：当前根目录三份实现文件已经形成了 `DUAL_INPLACE` 的主体链路：在 decode uniform batch 上按 cudagraph capture size 切成两段，第一段使用原始 buffer 前缀，第二段使用同一个原始 buffer 的 offset view，并通过 `BatchDescriptor.start_num_tokens` 区分第二张 offset graph。相比 `DUAL_PARALLEL`，它不为第二段分配完整 `micro_*` 输入/metadata buffer，不 padding 第二段真实 remainder，理论目标是减少显存、减少 copy、减少 padding token。

补充 `forward_context.py` 后，`BatchDescriptor` 主链路已经能闭合：`/vllm-workspace/forward_context.py` 中的 `BatchDescriptor` 包含 `num_tokens / uniform_decode / start_num_tokens`，并且 `non_uniform` 会保留 `start_num_tokens`。`ForwardContext` 也包含 `cudagraph_runtime_mode / batch_descriptor / ubatch_slices / stream_slot / timing_scheme_label`。因此此前关于 descriptor 字段缺失的阻塞项已解除。仍需确认实际运行时导入的是这份更新后的 `vllm.forward_context`，而不是包内旧版 `/vllm-workspace/vllm/vllm/forward_context.py`。

## 1. 模式目标

原始 CUDA Graph replay 的主问题是 batch size 不能完全命中已捕获图时，需要 padding 到下一个 capture size。例如 capture sizes 为 `256, 384, 512`，当前 decode batch 为 `416`：

- `PADDING`：重放 `512` 图，实际 token 为 `416`，padding 为 `96`。
- `DUAL_PARALLEL`：切成 `384 + 32`，第二段通常 padding 到 micro capture size，例如 `384 + 128`，仍有 padding，同时需要第二套 buffer。
- `DUAL_INPLACE`：切成 `384 + 32`，第一段重放正常 `384` 图，第二段按原 buffer offset 捕获或重放 `32@start=384` 图，不 padding 第二段。

inplace 的核心不是“同一张图处理两个区域”，而是“第二张图的输入 tensor 是原始 buffer 的 offset view，并且 graph key 额外携带 start offset”。也就是说第二张图仍然是独立 CUDA Graph entry，只是它的输入地址来自同一套持久 buffer 内部，而不是 `micro_*` buffer。

## 2. 关键数据结构

### 2.1 `StreamSlot`

`gpu_model_runner.py:130-133` 和 `cudagraph.py:25-28` 都定义了 `StreamSlot`：

```python
class StreamSlot(IntEnum):
    PRIMARY = 0
    SECONDARY = 1
```

它用于在 forward context 中告诉 `CUDAGraphWrapper` 当前执行的是第一段还是第二段。`cudagraph.py:142-145` 根据 `stream_slot` 选择 graph entry 字典和 graph pool。

注意：当前两处各自定义 `StreamSlot`，虽然枚举值相同，但类型不是同一个 class。代码依赖的是值比较和上下文透传，最好后续统一到一个模块，避免类型检查或跨模块比较出现隐性问题。

### 2.2 `DualModelInputs`

`gpu_model_runner.py:140-158` 定义 `DualModelInputs`，把双段执行需要的所有信息打包：

- 两段 token 数：`first_num_tokens`、`second_num_tokens`
- 两段 cudagraph runtime mode：`first_cudagraph_runtime_mode`、`second_cudagraph_runtime_mode`
- 两段 graph key：`first_batch_descriptor`、`second_batch_descriptor`
- 两段 model inputs：`input_ids / positions / inputs_embeds / intermediate_tensors`
- `timing_scheme_label`：例如 `384+32`

这个结构是 `_prepare_dual_model_inputs()` 和 `_execute_model_dual_mode()` 之间的边界。

### 2.3 `BatchDescriptor.start_num_tokens`

inplace 依赖 `BatchDescriptor.start_num_tokens`：

- 第一段：`start_num_tokens=0`
- 第二段：`start_num_tokens=first_num_tokens`

使用位置：

- 计时 child mode：`gpu_model_runner.py:743-759`
- 实际 dual input：`gpu_model_runner.py:3044-3061`
- dispatcher lazy key：`cudagraph_dispatcher.py:124-154`

设计含义：`num_tokens=32,start=384` 与 `num_tokens=32,start=256` 必须是不同图，因为 CUDA Graph replay 要求输入 tensor 地址与 capture 时一致；同样长度但不同 offset 的 view 起始地址不同，不能复用同一个 graph entry。

已确认：用户提供的 `/vllm-workspace/forward_context.py:37-56` 定义了 inplace 需要的 descriptor：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
    start_num_tokens: Optional[int] = 0

    @property
    def non_uniform(self) -> "BatchDescriptor":
        return BatchDescriptor(
            self.num_tokens,
            uniform_decode=False,
            start_num_tokens=self.start_num_tokens)
```

因为它是 `NamedTuple`，hash/equality 默认包含所有字段，所以 `(num_tokens=32, uniform_decode=False, start_num_tokens=384)` 和 `(num_tokens=32, uniform_decode=False, start_num_tokens=256)` 会是不同 key。这正好满足 inplace offset graph 的 key 隔离需求。

仍需注意：根目录文件导入的是 `from vllm.forward_context import BatchDescriptor`，不是相对导入 `/vllm-workspace/forward_context.py`。如果运行环境的 `PYTHONPATH` 指向包内 `/vllm-workspace/vllm/vllm/forward_context.py`，而包内版本仍是旧结构，则实际运行仍会失败。需要确认这份更新已同步到真正被 import 的 `vllm.forward_context`。

### 2.4 `ForwardContext` 对 inplace 的支撑

`/vllm-workspace/forward_context.py:266-291` 的 `ForwardContext` 已包含：

- `attn_metadata`
- `dp_metadata`
- `cudagraph_runtime_mode`
- `batch_descriptor`
- `ubatch_slices`
- `stream_slot`
- `timing_scheme_label`

`/vllm-workspace/forward_context.py:311-330` 的 `create_forward_context()` 会把这些字段全部写入 `ForwardContext`。

`/vllm-workspace/forward_context.py:349-381` 的 `set_forward_context()` 默认 `stream_slot=StreamSlot.PRIMARY`，双流执行时 `gpu_model_runner.py` 显式传入 `StreamSlot.PRIMARY` 或 `StreamSlot.SECONDARY`。虽然 `forward_context.py`、`gpu_model_runner.py`、`cudagraph.py` 各自定义了 `StreamSlot`，它们都是 `IntEnum` 且值为 `0/1`，运行时比较一般能按整数值成立；但工程上建议统一定义，避免类型和可读性问题。

## 3. 初始化阶段

### 3.1 原始持久 buffer

`gpu_model_runner.py:403-462` 初始化了 CUDA Graph 需要的原始持久 buffer：

- `self.input_ids`
- `self.positions`
- `self.query_start_loc`
- `self.seq_lens`
- `self.inputs_embeds`
- `self.is_token_ids`
- `self.discard_request_indices`
- 推测解码相关 buffer
- M-RoPE 相关 `self.mrope_positions`

这些 buffer 的生命周期覆盖整个 runner，保证 CUDA Graph capture/replay 的输入 tensor 地址稳定。

### 3.2 micro buffer 只服务 `DUAL_PARALLEL`

`gpu_model_runner.py:356-374` 只在 `ReplayMode.DUAL_PARALLEL` 下创建 `self.micro_input_batch`。

`gpu_model_runner.py:465-511` 也只在 `DUAL_PARALLEL` 下创建：

- `self.micro_input_ids`
- `self.micro_positions`
- `self.micro_query_start_loc`
- `self.micro_seq_lens`
- `self.micro_inputs_embeds`
- `self.micro_is_token_ids`
- `self.micro_discard_request_indices`
- `self.micro_num_decode_draft_tokens`
- `self.micro_num_accepted_tokens`
- `self.micro_mrope_positions`

这正是 `DUAL_INPLACE` 和 `DUAL_PARALLEL` 的第一层差异：inplace 不分配第二套完整输入/metadata buffer。

### 3.3 micro batch size

`gpu_model_runner.py:1284-1291` 的 `_get_micro_batch_size()` 取相邻 capture sizes 的最大间隔：

```python
return max(end - start for start, end in zip(
    cudagraph_batch_sizes, cudagraph_batch_sizes[1:]))
```

含义：第二段 remainder 的上限不能超过最大 bucket gap。`_get_split_point()` 会 assert 这一点。

例：capture sizes 为 `[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]`，最大 gap 是 `128`。batch `416` 切到 `384 + 32`，第二段 `32 <= 128`，合法。

### 3.4 graph streams

`gpu_model_runner.py:561-566` 在 `DUAL_MIXED / DUAL_PARALLEL / DUAL_INPLACE` 下创建两条 stream：

```python
self.graph_streams = {
    StreamSlot.PRIMARY: torch.cuda.Stream(),
    StreamSlot.SECONDARY: torch.cuda.Stream(),
}
```

当前 `DUAL_INPLACE` 也会走双 stream 并发执行。需要注意：因为两段共享原始 buffer，虽然读的 token 区间不同，但 attention metadata、KV cache 写入、通信 buffer、临时 workspace 是否完全无冲突，要由具体 kernel 和 metadata 地址确认。报告后面列为重点风险。

## 4. 切分判定

入口是 `_prepare_inputs()` 中的 `gpu_model_runner.py:1516-1524`：

```python
if self._should_split_for_cudagraph(total_num_scheduled_tokens,
                                    uniform_decode,
                                    vllm_config=self.vllm_config):
    split_point = self._get_split_point(total_num_scheduled_tokens)
    ubatch_slices = create_ubatch_slices(num_scheduled_tokens, split_point)
    split_for_cudagraph = True
```

### 4.1 `_should_split_for_cudagraph()`

`gpu_model_runner.py:1246-1282` 逐项过滤：

1. `enable_dbo` 为真则不切：`gpu_model_runner.py:1255-1258`
2. `ReplayMode.PADDING` 不切：`gpu_model_runner.py:1260-1262`
3. 只在 `uniform_decode=True` 时切：`gpu_model_runner.py:1264-1266`
4. 没有 `cudagraph_batch_sizes` 不切：`gpu_model_runner.py:1268-1269`
5. `_get_split_point()` 失败不切：`gpu_model_runner.py:1271-1273`
6. `_split_saves_enough_padding()` 不满足不切：`gpu_model_runner.py:1275-1279`

所以 inplace 当前只覆盖纯 decode 或 speculative decode 的 uniform decode 场景，不覆盖 mixed prefill/decode、chunked prefill、DBO。

### 4.2 `_get_split_point()`

`gpu_model_runner.py:1293-1304`：

- 从 `self.cudagraph_batch_sizes` 中找所有 `< num_input_tokens` 的 capture size
- 取最大值作为 first graph size
- `remainder = num_input_tokens - split_point`
- assert `remainder <= self.micro_batch_size`

这是一种 greedy lower bucket 策略。它保证第一段最大化，第二段尽量小。

### 4.3 `_split_saves_enough_padding()`

`gpu_model_runner.py:1306-1328`：

- 如果当前 batch 大于最大 capture size，不切。
- `original_padded_tokens = pad_for_cudagraph(num_input_tokens)`。
- 对 `DUAL_INPLACE`，只要求 `original_padded_tokens > num_input_tokens`。也就是只要原始 padding 模式会产生 padding，就值得切。
- 对其他 dual 模式，要计算 saved padding，并与 `cudagraph_split_pad_threshold` 比较。

这体现 inplace 的收益模型：第二段不 padding，因此只要原始单图存在 padding，就可能收益为正。

## 5. token 数和 padding 处理

`_preprocess()` 是实际决定传给模型多少 token 的地方。

### 5.1 普通单图路径

`gpu_model_runner.py:2638-2642`：

```python
num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
num_pad, num_tokens_after_padding = self.get_dp_padding(num_input_tokens)
num_input_tokens += num_pad
```

即先 cudagraph padding，再叠加 DP padding。

### 5.2 DUAL_INPLACE

`gpu_model_runner.py:2643-2650`：

```python
num_input_tokens = ubatch_slices[0].num_tokens + ubatch_slices[1].num_tokens
num_tokens_after_padding = None
```

这里没有调用 `_get_num_input_tokens()`，也没有 `pad_out_ubatch_slice()`，所以第二段保持真实 remainder。

关键后果：

- `num_input_tokens == 原始 scheduled token 数`
- 不会把第二段 token_slice 扩到 padding 后的末尾
- logits/sampling 后续仍按真实 token 输出

### 5.3 DUAL_PARALLEL

`gpu_model_runner.py:2651-2660`：

```python
num_input_tokens = first + self._get_num_input_tokens(second)
self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
```

parallel 的第二段会 padding 到可 capture 的 micro size，且 token_slice 被扩展。

## 6. attention metadata

这是 inplace 最敏感的部分。CUDA Graph replay 不只要求 `input_ids` / `positions` 地址稳定，也要求 attention metadata 中被 kernel 捕获的 tensor 地址稳定。

### 6.1 common metadata 基于原始 buffer 构造

`gpu_model_runner.py:1679-1695` 构造 `CommonAttentionMetadata`，主要字段来自原始 buffer：

- `query_start_loc`
- `seq_lens`
- `block_table_tensor`
- `slot_mapping`
- `num_actual_tokens`
- `max_query_len`
- `max_seq_len`
- `logits_indices_padded`

### 6.2 inplace 分支

`gpu_model_runner.py:1738-1795` 是 `DUAL_INPLACE` 专属分支：

1. 调用 `split_attn_metadata(ubatch_slices, common_attn_metadata)`。
2. 对每个 `ubid` 用 `attn_group.get_metadata_builder(ubatch_id=ubid).build(...)` 构造 metadata。
3. 对 `ubid == 0`，把 `attn_metadata_i.query_start_loc` 写回 `self.query_start_loc.gpu`，再用 `self.query_start_loc.gpu[:sub_num_reqs + 1]` 作为 `FlashAttentionMetadata.query_start_loc`。
4. 如果 `seq_lens` 地址不是 `self.seq_lens.gpu`，也写回 `self.seq_lens.gpu`。
5. 重新构造 `FlashAttentionMetadata`，强制 `query_start_loc` 和 `seq_lens` 使用持久 buffer view。

这里的意图是修复 `split_attn_metadata()` 内部切片/减法可能创建新 tensor，导致 replay 地址与 capture 不一致的问题。

### 6.3 inplace 第二段 metadata 的当前问题

inplace 分支只对 `ubid == 0` 做了持久地址修复。对 `ubid == 1`，当前代码直接把 `attn_metadata_i` 放入 `attn_metadata[1]`，没有像 parallel 分支那样复制到 `micro_*` buffer，也没有把第二段的 `query_start_loc/seq_lens` 写回一个稳定的独立 buffer。

这可能是当前实现最大的风险点：

- 第二段 input tensor 是原始 buffer 的 offset view，地址可由 `input_ids[first:]` 稳定得到。
- 但第二段 `query_start_loc` 可能来自 `split_attn_metadata()` 的新 tensor，而不是稳定持久 buffer。
- `cudagraph.py:176-203` 捕获时记录前两层 attn metadata tensor 地址；`cudagraph.py:289-332` replay 时比较地址，说明作者已经遇到或预期了 metadata 地址漂移。
- 如果第二段 metadata tensor 地址每步变化，`CUDAGraphWrapper` 的地址断言会失败，或者更糟糕地导致 CUDA illegal memory access。

建议补强：为 `DUAL_INPLACE` 的第二段也提供稳定 metadata 地址策略。可选方案：

1. 为 inplace 仅增加轻量 metadata buffer，不增加完整 `micro_input_ids/micro_positions`。例如 `inplace_second_query_start_loc`、`inplace_second_seq_lens`、必要时 `inplace_second_slot_mapping`。
2. 或保证 `split_attn_metadata()` 对第二段返回的是原始大 buffer 的 offset view，而不是新分配 tensor。但如果需要把 query_start_loc 归零，通常会产生新 tensor，很难完全避免。
3. 或按 `start_num_tokens` 直接构造第二段 metadata，使所有被 graph 捕获的 tensor 都来自固定 buffer。

### 6.4 parallel 分支对照

`gpu_model_runner.py:1796-1908` 是非 inplace 的 ubatch 分支。对 `ubid == 1`，`gpu_model_runner.py:1863-1902` 明确把第二段 metadata 拷贝到：

- `self.micro_query_start_loc.gpu`
- `self.micro_seq_lens.gpu`
- `self.micro_input_batch.block_table[...]`
- `self.micro_input_batch.block_table[...].slot_mapping.gpu`

然后重新构造 `FlashAttentionMetadata` 使用 micro buffer。这就是 `DUAL_PARALLEL` 地址稳定的来源。

inplace 当前没有对应的第二段稳定 metadata buffer，后续需要确认第二段是否在实际模型中不捕获这些字段，或者已有别处保证地址稳定。

## 7. dual model inputs 构造

`_prepare_dual_model_inputs()` 位于 `gpu_model_runner.py:3039-3107`。

### 7.1 第一段

`gpu_model_runner.py:3044-3054`：

```python
first_batch_descriptor = BatchDescriptor(
    num_tokens=first_num_tokens,
    uniform_decode=uniform_decode,
    start_num_tokens=0,
)
first_runtime_mode, first_batch_descriptor = dispatcher.dispatch(...)
first_input_ids, first_positions, ... = self._slice_model_inputs(
    ubatch_slices[0].token_slice, ...)
```

第一段是原始 buffer 前缀。只要 `token_slice.start == 0`，它的 tensor 起始地址与常规 capture 一致。

### 7.2 第二段

`gpu_model_runner.py:3056-3068`：

```python
second_batch_descriptor = BatchDescriptor(
    num_tokens=second_num_tokens,
    uniform_decode=uniform_decode,
    start_num_tokens=first_num_tokens if replay_mode == DUAL_INPLACE else 0,
)
```

inplace 第二段的 key 带 `start_num_tokens=first_num_tokens`。

然后 `gpu_model_runner.py:3077-3090`：

- `DUAL_PARALLEL`：调用 `_copy_second_ubatch_to_micro_buffers()`，把第二段输入复制到 micro buffer，第二段 tensor 从 micro buffer 的 0 开始。
- 非 `DUAL_PARALLEL`，包括 `DUAL_INPLACE`：直接使用 `_slice_model_inputs()` 得到的原始 buffer 切片。

这就是 inplace “in-place”的实际落点。

## 8. 双流执行

`_execute_model_dual_mode()` 位于 `gpu_model_runner.py:3109-3192`。

流程：

1. 取 `primary_stream` 和 `secondary_stream`。
2. 在 primary stream 上 `wait_stream(default_stream)`。
3. 设置 forward context：
   - `attn_metadata[0]`
   - `num_tokens=first_num_tokens`
   - `cudagraph_runtime_mode=first...`
   - `batch_descriptor=first...`
   - `stream_slot=PRIMARY`
4. 调用 `self.model(...)`。
5. 在 secondary stream 上同样设置 `attn_metadata[1]`、`stream_slot=SECONDARY` 并调用 `self.model(...)`。
6. default stream 等待两个 event。
7. `torch.cat([output_first, output_second], dim=0)` 合并输出。

对于 `DUAL_INPLACE`，两段并发执行时共享的资源包括：

- 原始 input buffer 的不同 token 区间
- 原始 block table / slot mapping / seq_lens / query_start_loc 的可能视图或派生 tensor
- KV cache，写入 slot 应该不同，但具体由 `slot_mapping` 保证
- 模型权重只读
- 某些 attention backend 的 workspace 或全局 scratch，需确认是否 stream-safe

因此 `DUAL_INPLACE` 的正确性不只取决于 Python tensor 切片，还取决于 attention backend 的 metadata 和 kernel 是否能安全并发。

## 9. dispatcher 和 lazy graph key

`cudagraph_dispatcher.py` 是 graph key 的唯一调度入口。

常规逻辑：

- 先查 FULL key：`cudagraph_dispatcher.py:110-112`
- 再查 `batch_descriptor.non_uniform` 的 FULL：`cudagraph_dispatcher.py:114-117`
- 再查 piecewise：`cudagraph_dispatcher.py:119-122`

inplace 新增 lazy key：`cudagraph_dispatcher.py:124-154`。

```python
inplace_offset_graph = (
    replay_mode == ReplayMode.DUAL_INPLACE
    and batch_descriptor.start_num_tokens is not None
    and batch_descriptor.start_num_tokens > 0)

if batch_descriptor.num_tokens in capture_sizes or inplace_offset_graph:
    ...
    self.add_cudagraph_key(..., batch_descriptor)
    return ..., batch_descriptor
```

含义：

- 第一段通常是 capture size，能命中预初始化 key。
- 第二段 remainder 未必是 capture size，例如 `32`、`80`、`160`。
- 只要是 `DUAL_INPLACE` 且 `start_num_tokens > 0`，就允许动态加入 key。
- 第一次遇到该 key 时，`CUDAGraphWrapper` 会 lazy capture；后续相同 `(num_tokens, start_num_tokens, uniform_decode)` 复用。

这也是当前实现支持 `384+32` 而不把 `32` padding 到 capture size 的关键。

## 10. CUDAGraphWrapper 行为

`cudagraph.py` 是 capture/replay 的执行器。

### 10.1 graph entry 字典

`cudagraph.py:103-108`：

- `concrete_cudagraph_entries`：primary graph entries
- `concrete_cudagraph_entries_secondary`：secondary graph entries

`cudagraph.py:142-145` 根据 `stream_slot` 选择 entry 字典和 graph pool。

注意：`graph_pool_secondary` 只在 `ReplayMode.DUAL_PARALLEL` 下创建；`DUAL_INPLACE` 的 secondary 会走 `current_graph_pool = self.graph_pool_secondary`，值为 `None`，后续 `torch.cuda.graph(cudagraph, pool=current_graph_pool)` 即 pool 为 `None`。同时 `set_graph_pool_id()` 会在 `current_graph_pool is None` 时设置一个新 handle。这里需要实际验证 PyTorch graph pool 与 pynccl allocator pool id 是否一致。如果 inplace secondary graph 也要独立 pool，建议显式为 `DUAL_INPLACE` 也创建 secondary pool，或确认共享 pool 不会造成地址/内存复用问题。

### 10.2 capture

`cudagraph.py:157-248`：

1. 首次 key 没有 graph 时进入 capture。
2. 记录 args/kwargs 中所有 tensor 的 `data_ptr()` 到 `entry.input_addresses`。
3. 记录前两层 attention metadata 中 tensor 字段地址到 `entry.attn_metadata_addresses`。
4. `torch.cuda.graph(cudagraph, pool=current_graph_pool)` 中运行 `self.runnable(*args, **kwargs)`。
5. 保存 weak-ref output 和 graph。

对 inplace 第二段来说，`input_ids[first:first+second]` 的 `data_ptr()` 必须在每次同一 key replay 时一致；这依赖原始 buffer 地址不变且 `start_num_tokens` 相同。

### 10.3 replay

`cudagraph.py:250-342`：

- 无论是否 debug，当前代码都会重新计算 args/kwargs tensor 地址并 assert 与 capture 相同：`cudagraph.py:262-272`。
- 如果记录了 attention metadata 地址，则 replay 时比较地址：`cudagraph.py:289-332`。
- 根据 `stream_slot` 调用 `entry.cudagraph.replay()` 并打 NVTX range：`cudagraph.py:334-341`。

这个断言对 inplace 很重要：它能直接暴露第二段 offset key 是否正确、metadata 是否稳定。

## 11. capture 流程

`capture_model()` 位于 `gpu_model_runner.py:4626-4704`。

### 11.1 普通 capture

非 `DUAL_PARALLEL` 下调用 `_capture_cudagraphs()`：

- mixed mode：`gpu_model_runner.py:4663-4678`
- decode full mode：`gpu_model_runner.py:4680-4702`

这意味着 `DUAL_INPLACE` 当前没有像 parallel 那样预先为 secondary 捕获第二套图。它依赖 dispatcher lazy key + wrapper 首次执行 capture。

### 11.2 DUAL_PARALLEL capture

`gpu_model_runner.py:4779-4860` 的 `_capture_cudagraphs_dual()` 只在 `ReplayMode.DUAL_PARALLEL` 下由 `capture_model()` 调用。

它会：

- primary 对所有 compilation cases warmup/capture。
- secondary 只对 `num_tokens <= self.micro_batch_size` 的图 warmup/capture。

对 inplace 来说，这段不走。inplace 第二段 offset graph 是运行时 lazy capture，而不是加载阶段穷举捕获。

### 11.3 lazy capture 的全局开关风险

`capture_model()` 末尾有注释掉的：

```python
# set_cudagraph_capturing_enabled(False)
```

注释说明为了 future lazy capturing，捕获后没有禁用全局 capture。`DUAL_INPLACE` 正好依赖这个行为。如果后续恢复禁用，inplace offset graph 首次执行可能会被 `validate_cudagraph_capturing_enabled()` 拦下。

## 12. 和 DUAL_PARALLEL 的逐点差异

| 维度 | DUAL_PARALLEL | DUAL_INPLACE |
|---|---|---|
| 第二段输入 | copy 到 `micro_input_ids/micro_positions/micro_inputs_embeds` | 原始 buffer offset slice |
| 第二段 graph key | `num_tokens + start=0` | `num_tokens + start=first_num_tokens` |
| 第二段 padding | padding 到 capture size | 不 padding，真实 remainder |
| 第二段 metadata | copy 到 `micro_query_start_loc/micro_seq_lens/micro_block_table/micro_slot_mapping` | 当前主要依赖 split 后 metadata，只有 ubid=0 做持久地址修复 |
| capture 时机 | load 阶段预捕获 primary/secondary | primary 预捕获，offset secondary 运行时 lazy capture |
| 显存 | 两套输入/metadata buffer，secondary graph pool | 少一套完整 micro buffer，但 offset graph entries 仍占 graph memory |
| copy 开销 | 第二段输入和 metadata 有 GPU->GPU copy | 输入 copy 少，metadata 可能仍需修复/copy |
| 风险 | 内存大但地址稳定 | 地址稳定和并发资源冲突风险更高 |

## 13. 当前实现的关键风险

### 13.1 `BatchDescriptor` 已补齐，但需确认 import 路径

用户提供的 `/vllm-workspace/forward_context.py` 已具备 inplace 所需字段/属性：

- `uniform_decode: bool`
- `start_num_tokens: int | None`
- `non_uniform` property，且保留 `start_num_tokens`

因此从源码设计上看，dispatcher 和 runner 对 descriptor 的使用是自洽的：

- `BatchDescriptor(num_tokens=..., uniform_decode=...)` 可构造。
- `batch_descriptor.non_uniform` 可用于 uniform decode 回退到 non-uniform graph。
- `batch_descriptor.start_num_tokens` 可作为 second inplace graph 的 offset key。
- `NamedTuple` 的 hash/equality 会把 `start_num_tokens` 纳入 key。

剩余风险是运行环境是否真正 import 这份文件。根目录实现写的是 `from vllm.forward_context import ...`，如果 `vllm.forward_context` 仍解析到包内旧版 `/vllm-workspace/vllm/vllm/forward_context.py`，则仍会出现字段不匹配。建议最终同步到包内源码，或用一次 `python -c "import vllm.forward_context as f; print(f.__file__, f.BatchDescriptor._fields)"` 验证。

### 13.2 缺少配置定义确认

当前根目录代码使用：

- `ReplayMode.DUAL_INPLACE`
- `ReplayMode.DUAL_PARALLEL`
- `ReplayMode.DUAL_MIXED`
- `ReplayMode.PADDING`
- `compilation_config.cudagraph_split_pad_threshold`
- `compilation_config.replay_mode`

我未在当前 vLLM package config 中定位到这些修改。需要提供对应 `vllm.config` 或 `vllm/vllm/config/compilation.py` 修改。

### 13.3 inplace 第二段 attention metadata 地址

如第 6 节，`DUAL_INPLACE` 对 `ubid == 1` 没有显式写回稳定 buffer。建议用 `cudagraph.py` 的地址断言实际验证：

- 第一次第二段 capture 时记录了哪些 metadata tensor。
- 第二次相同 key replay 时是否完全一致。
- 不同 `start_num_tokens` 是否分别 capture，不误用。

### 13.4 secondary graph pool 对 inplace 未显式创建

`cudagraph.py:106` 只对 `DUAL_PARALLEL` 创建 `graph_pool_secondary`。`DUAL_INPLACE` 的 secondary entry 字典存在，但 secondary graph pool 为 `None`。这可能是有意共享默认 pool，也可能是遗漏。

建议用两组实验确认：

1. `DUAL_INPLACE` secondary graph 使用 `None` pool 是否稳定。
2. 改成 `ReplayMode.DUAL_PARALLEL or DUAL_INPLACE` 都创建 secondary pool，比较 capture/replay 地址和显存。

### 13.5 双流并发和共享 metadata

`_execute_model_dual_mode()` 中 primary 和 secondary 是并发提交的。如果两个 metadata 共享同一底层 buffer 且其中一个在执行前被另一个分支覆盖，会出现竞态。当前 `_prepare_inputs()` 在执行前构造 metadata，但需要确认：

- `attn_metadata[0].query_start_loc` 与 `attn_metadata[1].query_start_loc` 是否指向不同地址或只读且不被覆盖。
- `slot_mapping` 是否为不重叠 view。
- attention backend 是否使用全局 workspace。

### 13.6 只适配 FlashAttentionMetadata

metadata 修复分支直接 import 并重建 `FlashAttentionMetadata`。如果模型使用 GDN、其他 attention backend、encoder-only/cross-attention、cascade attention 的特殊 metadata，当前逻辑可能不完整。

## 14. 建议补充的验证日志

为了把 inplace 跑通，建议至少记录下面字段：

- replay mode
- batch size
- split point
- first/second num_tokens
- first/second start_num_tokens
- first/second runtime mode
- first/second descriptor
- first/second input_ids data_ptr
- first/second positions data_ptr
- first/second query_start_loc data_ptr
- first/second seq_lens data_ptr
- first/second slot_mapping data_ptr
- graph entry 是 capture 还是 replay

当前已有基础：

- `gpu_model_runner.py:632-676` 附近有 execute timing CSV 字段。
- `cudagraph.py:170-174` 记录输入地址。
- `cudagraph.py:176-203` 记录 metadata 地址。
- `cudagraph.py:262-272` 强制检查输入地址。
- `cudagraph.py:289-332` 检查 metadata 地址。

建议把第二段 descriptor 的 `start_num_tokens` 加到 NVTX 或日志里，否则 `num_tokens` 相同但 offset 不同的图很难区分。

## 15. 最小跑通依赖清单

如果要让我继续从静态调研推进到修复/跑通，需要补齐或确认以下代码：

1. 确认实际 import 的 `vllm.forward_context.BatchDescriptor` 就是用户提供的新版本，而不是包内旧版。
2. 修改后的 `vllm.config.ReplayMode` 和 compilation config。
3. `split_attn_metadata()` 的实现，特别是第二段 query_start_loc、seq_lens、slot_mapping 是否 clone。
4. `create_ubatch_slices()` 的实现，确认 token_slice 对第二段是否是原始 offset。
5. 实际运行使用的是根目录这几份文件，还是 `/vllm-workspace/vllm/vllm/...` 包内文件。当前根目录文件和包内文件不是同一套实现。

## 16. 推荐后续实现动作

1. 先验证 `BatchDescriptor` 和 `ReplayMode` 的实际 import 版本，否则无法运行。
2. 给 `DUAL_INPLACE` secondary metadata 增加稳定 buffer 或确认 `split_attn_metadata()` 零分配稳定返回。
3. 在 `CUDAGraphWrapper` 的 graph key 日志中打印完整 descriptor，包括 `start_num_tokens`。
4. 对 `DUAL_INPLACE` 也考虑创建 `graph_pool_secondary`，至少做 A/B 验证。
5. 写一个固定 batch 的 replay 测试：
   - capture sizes 包含 `384,512`
   - batch 固定 `416`
   - 期望 split 为 `384+32`
   - 第一次 second graph capture，第二次 second graph replay
   - assert `num_input_tokens == 416`
   - assert second descriptor `start_num_tokens == 384`
   - assert input/metadata 地址一致

## 17. 当前实现的一句话判断

`DUAL_INPLACE` 的主干设计是成立的：用 `start_num_tokens` 把原始 buffer 内的 offset 子图纳入 graph key，避免第二段 padding 和 micro input buffer。补充的 `forward_context.py` 已经让 `BatchDescriptor` 这条链路自洽；当前最大剩余风险变成两类：一是运行时 import 是否同步到新 `forward_context` 和新 `ReplayMode`，二是第二段 attention metadata 地址稳定性没有像 `DUAL_PARALLEL` 那样被显式保障。

## 18. 面向迁移的实现方法

本节按“迁移到另一套 vLLM/推理框架时应该怎么做”来抽象，不依赖当前工作区能否直接运行。

### 18.1 迁移目标

要迁移的不是 `DUAL_PARALLEL` 的双 buffer 方案，而是 `DUAL_INPLACE` 的单 buffer offset graph 方案：

1. 大 batch 先拆成两个子 batch。
2. 第一段命中已有 capture size，例如 `384`。
3. 第二段使用真实 remainder，例如 `32`。
4. 第二段不复制到第二套 input buffer，而是直接使用原始输入 buffer 的 `[first:first+second]` view。
5. 第二段 graph key 必须包含 offset，即 `start_num_tokens=first`。
6. 第一次遇到某个 offset/remainder 组合时 lazy capture，后续复用。

这套方法的本质是把 CUDA Graph 的“地址固定”要求从“第二套 buffer 从 0 开始”转换成“同一套 buffer 的固定 offset 地址”。

### 18.2 必改模块一：配置和模式枚举

需要在配置层加入 replay mode，最小集合：

```python
class ReplayMode(Enum):
    PADDING = ...
    DUAL_PARALLEL = ...
    DUAL_INPLACE = ...
```

可选保留：

- `DUAL_MIXED`
- `DUAL_SERIAL`

`CompilationConfig` 至少需要：

- `replay_mode`
- `cudagraph_capture_sizes`
- `cudagraph_split_pad_threshold`

迁移时建议把 `ReplayMode` 的语义定义清楚：

- `PADDING`：原始单图 padding。
- `DUAL_PARALLEL`：第二段复制到 secondary buffer，第二段可 padding。
- `DUAL_INPLACE`：第二段保留在原始 buffer offset，第二段不 padding。

### 18.3 必改模块二：BatchDescriptor

CUDA Graph cache key 必须能表达 offset。推荐结构：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
    start_num_tokens: int | None = 0

    @property
    def non_uniform(self):
        return BatchDescriptor(
            self.num_tokens,
            uniform_decode=False,
            start_num_tokens=self.start_num_tokens)
```

关键要求：

- hash/equality 必须包含 `start_num_tokens`。
- `non_uniform` 不能丢掉 `start_num_tokens`。
- 第一段使用 `start_num_tokens=0`。
- 第二段使用 `start_num_tokens=first_num_tokens`。

如果迁移目标已有 `BatchDescriptor(num_tokens, num_reqs, uniform, has_lora)` 这类结构，不要直接覆盖旧字段含义；建议新增字段，或者新增一个 `graph_offset` 字段，避免影响 LoRA、request-aware graph key。

### 18.4 必改模块三：forward context

`CUDAGraphWrapper` 捕获/重放时需要从 forward context 取到：

- `cudagraph_runtime_mode`
- `batch_descriptor`
- `stream_slot`
- `attn_metadata`
- 可选：`ubatch_slices`
- 可选：`timing_scheme_label`

`stream_slot` 的作用是区分 primary/secondary 执行路径。迁移时建议只定义一次：

```python
class StreamSlot(IntEnum):
    PRIMARY = 0
    SECONDARY = 1
```

当前实现中 `forward_context.py`、`gpu_model_runner.py`、`cudagraph.py` 各自定义了 `StreamSlot`。迁移时最好统一到 `forward_context` 或一个公共模块，降低误用概率。

### 18.5 必改模块四：切分策略

迁移时可以直接采用当前策略：

```text
只在 uniform decode 下切分
split_point = max(capture_size < num_tokens)
remainder = num_tokens - split_point
```

触发条件：

- `replay_mode != PADDING`
- `uniform_decode == True`
- `num_tokens <= max(cudagraph_capture_sizes)`
- 存在小于 `num_tokens` 的 capture size
- 对 `DUAL_INPLACE`：只要 `pad_for_cudagraph(num_tokens) > num_tokens` 就值得切
- 对 `DUAL_PARALLEL`：还要扣除第二段 padding 后仍有收益

示例：

```text
capture_sizes = [256, 384, 512]
num_tokens = 416
split_point = 384
remainder = 32
DUAL_INPLACE scheme = 384 + 32
```

注意：当前根目录 `gpu_model_runner.py` 引用 `vllm.v1.worker.ubatch_splitting.create_ubatch_slices`，但当前工作区没有该文件。我从调用方式推断迁移所需语义如下：

```python
def create_ubatch_slices(num_scheduled_tokens, split_point) -> list[UBatchSlice]:
    # 返回两个 slice：
    # first.token_slice = slice(0, split_point 或请求边界修正后的点)
    # second.token_slice = slice(first_end, total_num_tokens)
    # request_slice 覆盖每段涉及的 request
```

如果 attention backend 不支持把一个 request 从中间切开，切分点必须回退到 request 边界。对于 pure decode，通常每个 request 一个 token，batch token index 与 request index 对齐，切分最简单。

### 18.6 必改模块五：preprocess 的 token 数规则

迁移时最容易写错的是 `num_input_tokens`。三种模式应严格区分：

```python
if replay_mode == PADDING:
    num_input_tokens = pad_for_cudagraph(total_tokens)

elif replay_mode == DUAL_PARALLEL:
    first = ubatch_slices[0].num_tokens
    second = pad_for_cudagraph(ubatch_slices[1].num_tokens)
    num_input_tokens = first + second
    pad_out_second_slice_to(num_input_tokens)

elif replay_mode == DUAL_INPLACE:
    first = ubatch_slices[0].num_tokens
    second = ubatch_slices[1].num_tokens
    num_input_tokens = first + second
    # 不 pad second，不扩 second token_slice
```

`DUAL_INPLACE` 的目标就是让 `num_input_tokens == total scheduled tokens`。如果迁移后这里仍然 padding 到 capture size，inplace 的主要收益就没了。

### 18.7 必改模块六：dispatcher lazy key

迁移时 dispatcher 要允许第二段 remainder 不在 capture sizes 中也能捕获：

```python
inplace_offset_graph = (
    replay_mode == DUAL_INPLACE
    and batch_descriptor.start_num_tokens is not None
    and batch_descriptor.start_num_tokens > 0
)

if batch_descriptor.num_tokens in capture_sizes or inplace_offset_graph:
    add_key(runtime_mode, batch_descriptor)
    return runtime_mode, batch_descriptor
```

这一步是 `DUAL_INPLACE` 能重放 `32@start=384` 的关键。没有 lazy key，第二段 `32` 如果不在 capture sizes 中就会退回 eager。

### 18.8 必改模块七：input slicing

迁移时 `DUAL_INPLACE` 第二段不能 copy 到 micro buffer：

```python
first_input_ids = input_ids[0:first]
second_input_ids = input_ids[first:first + second]

first_positions = positions[0:first]
second_positions = positions[first:first + second]
```

对于多维 positions，例如 M-RoPE：

```python
first_positions = positions[:, 0:first]
second_positions = positions[:, first:first + second]
```

CUDA Graph 对输入地址的要求：

- 同一个 graph key 捕获和重放时，`second_input_ids.data_ptr()` 必须相同。
- 因为原始 buffer 地址固定，且 `start_num_tokens` 固定，所以 view 的 `data_ptr()` 也固定。
- 如果 `start_num_tokens` 不同，即使 `num_tokens` 相同，也必须是不同 graph key。

### 18.9 必改模块八：attention metadata 稳定地址

这是迁移中最重要的一节。上游 `split_attn_metadata()` 明确会创建新 tensor：

```python
def slice_query_start_locs(...):
    """
    Creates a new tensor ...
    This will break cudagraph compatibility.
    """
```

它会对 `query_start_loc` 做切片并减去起点：

```python
query_start_loc[request_slice.start : request_slice.stop + 1]
    - query_start_loc[request_slice.start]
```

减法会分配新 tensor。对 CUDA Graph capture/replay 来说，这个地址不能每步变化。

因此迁移 `DUAL_INPLACE` 时有两种可靠方案。

方案 A：为第二段单独准备轻量 metadata buffer。

```text
inplace_second_query_start_loc
inplace_second_seq_lens
inplace_second_block_table  可选，若原始 view 地址稳定可不复制
inplace_second_slot_mapping 可选，若原始 view 地址稳定可不复制
```

流程：

1. 正常 `split_attn_metadata()` 得到逻辑正确的 metadata。
2. 把第二段 metadata 中 graph 会捕获的 tensor copy 到固定 buffer。
3. 重新构造 attention metadata，让字段指向固定 buffer view。

这是最稳的迁移方案，显存开销远小于 `DUAL_PARALLEL`，因为不需要第二套 input ids/positions/embeds，只需要 metadata 小 buffer。

方案 B：保证 split 后 metadata 全部是原始 buffer 的稳定 view。

这很难，因为 `query_start_loc` 通常需要归零，归零就会触发新 tensor 或原地改写共享 buffer。除非 attention backend 接受非零起点的 `query_start_loc`，否则不推荐。

当前根目录实现只对 `ubid == 0` 写回 `self.query_start_loc.gpu`，对 `ubid == 1` 没有同等稳定化处理。这是迁移时必须补强的点。

### 18.10 必改模块九：CUDAGraphWrapper

Wrapper 需要支持：

- 按 `BatchDescriptor` 缓存 graph entry。
- 按 `stream_slot` 区分 primary/secondary entry 字典。
- capture 时记录输入 tensor 地址。
- replay 时检查输入 tensor 地址。
- 最好也检查 attention metadata 地址。

迁移时的最小伪代码：

```python
entries = primary_entries if slot == PRIMARY else secondary_entries

entry = entries.get(batch_descriptor)
if entry is None:
    entry = entries[batch_descriptor] = CUDAGraphEntry(...)

if entry.graph is None:
    entry.input_ptrs = collect_tensor_ptrs(args, kwargs)
    entry.attn_ptrs = collect_attn_metadata_ptrs(context.attn_metadata)
    capture_graph(...)
else:
    assert collect_tensor_ptrs(args, kwargs) == entry.input_ptrs
    assert collect_attn_metadata_ptrs(context.attn_metadata) == entry.attn_ptrs
    entry.graph.replay()
```

对 `DUAL_INPLACE`，建议 secondary 也使用独立 graph pool 做 A/B 验证。独立 pool 不是 offset 方法的理论必要条件，但能减少和 primary graph pool 的内存复用干扰。

### 18.11 必改模块十：双流执行

迁移时可以保留当前双流结构：

```python
with torch.cuda.stream(primary_stream):
    set_forward_context(attn_metadata[0], descriptor=first, slot=PRIMARY)
    output_first = model(first_inputs)
    first_done.record()

with torch.cuda.stream(secondary_stream):
    set_forward_context(attn_metadata[1], descriptor=second, slot=SECONDARY)
    output_second = model(second_inputs)
    second_done.record()

default_stream.wait_event(first_done)
default_stream.wait_event(second_done)
output = torch.cat([output_first, output_second], dim=0)
```

迁移时必须确认：

- 两段写 KV cache 的 `slot_mapping` 不重叠。
- attention backend 的 workspace 是 stream-safe。
- model forward 内没有使用全局可变临时 buffer。
- 通信、NCCL、custom op 是否允许两个 stream 并发。

如果目标系统不能保证这些，可以先迁移成 `DUAL_INPLACE_SERIAL`：仍然单 buffer offset 和不 padding，但 primary/secondary 串行执行。这样先验证 graph key、metadata、输出正确性，再打开双流并发。

### 18.12 推荐迁移顺序

建议按下面顺序迁移，避免一次性引入太多变量：

1. 只做 descriptor + dispatcher lazy key，不开双流。
2. 实现 split 策略，让日志能打印 `first+second`。
3. 实现 `DUAL_INPLACE_SERIAL`：第二段使用原始 buffer offset view，串行执行。
4. 加 input address assert，确认第二段 offset graph 能 capture/replay。
5. 加 metadata 稳定 buffer，确认 metadata address assert 通过。
6. 对比 `PADDING` 与 `DUAL_INPLACE_SERIAL` 的输出一致性。
7. 再启用 primary/secondary 双流并发。
8. 最后做性能 benchmark。

### 18.13 最小验证用例

用固定 batch，避免调度干扰：

```text
capture_sizes = [256, 384, 512]
batch_size = 416
uniform_decode = True
expected split = 384 + 32
```

需要验证：

- `num_input_tokens == 416`
- first descriptor：`num_tokens=384,start=0`
- second descriptor：`num_tokens=32,start=384`
- 第一次运行 second graph 是 capture
- 第二次运行 second graph 是 replay
- second input ptr 两次相同
- second metadata ptr 两次相同
- 输出 token 与 `PADDING` 模式一致

### 18.14 迁移时可以舍弃的当前实现细节

以下不是方法本质，可以按目标系统重写：

- 当前 CSV timing 统计。
- NVTX 文案。
- 只记录前两层 attention metadata 地址的调试逻辑。
- `DUAL_PARALLEL` 的完整 micro input buffer。
- 当前 `StreamSlot` 多处重复定义。
- 当前 `FlashAttentionMetadata` 手工重建写法。

必须保留的是：

- offset 进入 graph key。
- 第二段不 padding。
- 第二段 input 使用原始 buffer offset view。
- 第二段 offset graph lazy capture。
- 被 graph 捕获的 metadata tensor 地址稳定。

## 19. 结合 GPU lazy capture 分析 NPU FIA 失败与规避方案

### 19.1 当前 NPU 失败现象

NPU `inplace_serial` 在固定 batch `416`、capture sizes `[256, 384, 512]`
下已经正确切成 `384 + 32`：

- split-0：普通 `384` graph，`start_num_tokens=0`
- split-1：offset `32` graph，`start_num_tokens=384`

最新 debug 已确认：

```text
split-0 384 graph replay 成功
split-1 descriptor:
  graph_variant = inplace_serial
  attention_backend = fia
  capture_metadata_mode = template
  templated_fia_seq_lens = 24
```

失败发生在 split-1 lazy capture 内部：

```text
ACLGraphWrapper.__call__
  with torch.npu.graph(...):
      output = self.runnable(...)
```

没有出现 `acl_graph_capture phase=post`，也没有进入
`lazy_replay_enqueued` 或 replay/update 阶段。

NPU runtime 报错：

```text
When layout is TND and PA not enabled, keyT(256) and valueT(256)
must be equal to the last element of actualSeqenceLengthKV(9)
```

这说明 FIA 在 TND + non-PA 模式下要求
`actual_seq_lengths_kv[-1] == block_size == 256`，但 lazy capture 时实际传入
FAI tiling 的尾值仍是 runtime decode 的真实 seqlen `9`。

### 19.2 为什么 GPU lazy capture 可以成立

GPU `DUAL_INPLACE` 的 lazy capture 关键点在于：第二段 offset graph 第一次运行时，
直接用真实 runtime batch capture。对 GPU FlashAttention 来说，这个真实 runtime
metadata 本身就是合法的。也就是说：

- dispatcher 允许 `start_num_tokens > 0` 的 lazy key；
- wrapper 第一次看到 key 就 capture；
- capture 使用的 input/metadata 与之后 replay 使用的是同一类合法 runtime 数据；
- 后续只需保证 tensor 地址稳定。

GPU 代码里的重点是地址一致性，而不是改写 metadata 语义：

- `BatchDescriptor.start_num_tokens` 进入 graph key；
- 第二段 input 使用原始 buffer offset view；
- wrapper 捕获时记录 input 地址；
- replay 时校验 input/attention metadata 地址；
- 对 metadata 地址不稳定的字段，需要写回持久 buffer 或使用稳定 view。

因此 GPU lazy capture 的前提是：

```text
真实 runtime metadata 可直接合法 capture
```

### 19.3 NPU FIA 与 GPU 的本质差异

NPU 当前失败不是 graph key 或地址稳定性问题，而是 capture 输入语义不合法：

```text
真实 split-1 decode seq_lens_list[-1] = 9
FIA TND non-PA capture 要求尾值 = block_size = 256
```

也就是说，NPU FIA offset lazy capture 不能像 GPU 那样直接拿真实 runtime
metadata capture。即使 Python 层已经把 `seq_lens_list[-1]` 改成 256，实际
torch.compile / ATB / FAI capture 路径仍然可能从另一条参数路径拿到 `9`。

因此只复制 GPU lazy key 机制是不够的。NPU 还需要一个 capture-safe metadata
策略，而且这个策略必须覆盖 FAI 实际使用的参数源头。

### 19.4 不做 offset graph 预捕获的可选解决方向

用户约束：不做预捕获 offset graph。因此不能在初始化阶段穷举
`32@start=384` 之类的 offset graph。可选方案只能在运行时第一次遇到 offset key
时处理。

#### 方案 A：FIA offset lazy capture 使用 capture-safe metadata overlay

保留 lazy capture，但 split-1 首次 capture 时临时切换到一份 capture-safe
metadata overlay：

```text
真实 runtime metadata:
  用于 scheduler / block table / slot mapping / 后续 replay update

capture-safe metadata overlay:
  只用于 torch.npu.graph 内首次 capture
  必须保证 FAI 实际看到 actual_seq_lengths_kv[-1] == block_size
```

实现要点：

1. overlay 不能只改 `AscendMetadata.seq_lens_list`。
2. 还要覆盖 FAI capture 实际使用的所有 seqlen 来源：
   - `seq_lens_list`
   - `seq_lens` / `seq_lens_cpu`
   - `actual_seq_lengths_q`
   - `common_attn_metadata` 中可能被 builder 或 compiled graph 捕获的字段
3. overlay 的 tensor 地址必须稳定，不能每次新建临时 tensor。
4. capture 结束后要恢复真实 metadata，再立即 replay 并走正常 update。

这个方向最贴近 GPU lazy capture：仍然是运行时 lazy capture offset graph，
只是 GPU 使用真实 metadata，NPU FIA 使用 capture-safe overlay metadata。

风险：

- 当前 evidence 表明 FAI 可能绕过 Python list，从 compiled graph 的另一条路径
  读取 seqlen。因此 overlay 必须前移到 metadata builder 或 common metadata 构造层，
  而不是只在 `full_graph_fia()` 里兜底。
- 如果 torch.compile 已经把 seqlen 专门化进子图，overlay 要在进入 compiled
  model 前生效。

#### 方案 B：FIA offset lazy capture 阶段强制走 PA，后续 offset graph 也按 PA replay

split-0 仍然复用普通 `384` FIA graph；只对 split-1 offset graph 使用 PA：

```text
split-0: FIA, start=0, replay existing 384 graph
split-1: PA, start=384, lazy capture 32@384 offset graph
```

这不是把当前 FIA baseline 全局改 PA，而是只让 offset graph 避开 FIA TND
non-PA 的 tiling 约束。

优点：

- 不需要预捕获 offset graph。
- 不需要欺骗 FIA seqlen。
- 语义上更像 GPU：真实 runtime metadata 可以直接 capture。

风险：

- split-1 PA graph key 必须带 `attention_backend="pa"`，避免复用 FIA params。
- `_update_attn_params_for_split_ubatch()` 必须按 descriptor/forced backend 进入
  PA update，不能出现 capture PA、update FIA 或相反。
- 要确认当前模型/shape 下 PA offset lazy capture 可用，且输出与 no-split/FIA
  baseline 对齐。

#### 方案 C：FIA offset lazy capture 禁用 ACL graph，只对 split-1 eager 执行

仍然保持 split：

```text
split-0: replay 384 graph
split-1: eager FIA, no ACL graph capture
```

这样可以避免 `512` padding，也能验证 split 输入、metadata、merge 逻辑，但
split-1 失去 graph replay 收益。

优点：

- 实现风险最低。
- 不需要 PA，也不需要预捕获 offset graph。
- 可作为中间验证路径，确认除了 lazy capture 外的 split 逻辑正确。

缺点：

- 性能收益有限。
- 每步 split-1 都走 eager，无法达到 GPU `DUAL_INPLACE` 的完整目标。

#### 方案 D：FIA offset lazy capture 时绕开 full-graph capture，只捕获非 attention 部分

如果当前 compilation 支持 piecewise 或 attention 外部更新，可以让 split-1
offset key 不捕获 FIA attention，attention 用 eager 或独立 update 路径。

优点：

- 避开 FAI TND capture 限制。
- 比完全 eager 可能保留部分 graph 收益。

风险：

- 当前 NPU 配置是 `FULL_DECODE_ONLY`，日志中也提示 use_inductor 不支持，只使用
  ACL Graph mode。是否能在 NPU 上稳定做到“非 attention graph + eager attention”
  需要单独验证。

### 19.5 推荐方案

在“不做预捕获 offset graph”的约束下，推荐按下面顺序推进：

1. **短期验证路径：方案 C**
   - split-0 继续复用普通 `384` graph。
   - split-1 FIA offset 不做 lazy capture，先 eager 执行。
   - 目标是确认 `384 + 32` 的 slicing、metadata、KV 写入、merge 输出都正确。

2. **可性能化路径：方案 B**
   - split-1 offset graph 改走 PA lazy capture。
   - split-0 不变，仍然是普通 FIA graph。
   - 这是最可能快速规避 `actualSeqenceLengthKV(9)` 的 graph 化方案。

3. **长期 FIA 原生路径：方案 A**
   - 做 capture-safe metadata overlay。
   - overlay 必须在进入 compiled model 前生效，且覆盖 tensor/list 两类 seqlen
     来源。
   - 这条路径保留 FIA offset lazy capture，但工程风险最高。

### 19.6 为什么不建议继续只在 `full_graph_fia()` 里修

当前已经尝试过：

- split-1 descriptor 增加 `capture_metadata_mode="template"`；
- split metadata 构造阶段把每层 `seq_lens_list[-1]` 改成 `block_size`；
- `_get_fia_params()` / `full_graph_fia()` / `_update_attn_fia_params()` 做兜底。

但是实际 lazy capture 仍然报 `actualSeqenceLengthKV(9)`。这说明 `full_graph_fia()`
附近的 Python list 修改不是 FAI tiling 的唯一参数来源，或者不是实际生效的参数源头。

继续在同一层补更多 list 兜底，收益会越来越低。更合理的分界是：

- 要么在 metadata builder/common metadata 源头构造 capture-safe overlay；
- 要么对 offset graph 换成不受该 tiling 约束的 backend；
- 要么 split-1 暂时 eager。

### 19.7 与 GPU lazy capture 的迁移结论

GPU lazy capture 可以照搬的部分：

- offset 进入 graph key；
- runtime 首次遇到 offset key 时 lazy capture；
- input 使用原始 buffer offset view；
- capture/replay 都要检查 input 和 metadata tensor 地址；
- 不 padding第二段。

NPU 不能直接照搬的部分：

- GPU 可以用真实 runtime metadata capture；
- NPU FIA TND non-PA 不能用真实 split-1 decode seqlen capture。

因此 NPU 的正确设计应当是：

```text
GPU:
  offset lazy key +真实 metadata +地址稳定

NPU FIA:
  offset lazy key + capture-safe metadata/backend policy +地址稳定
```

在不做 offset graph 预捕获的前提下，最实际的路径是先让 split-1 eager 跑通，
再将 split-1 offset graph 切到 PA lazy capture；FIA 原生 lazy capture 需要
单独做 capture-safe metadata overlay，不能只依赖 `seq_lens_list` 局部改写。
