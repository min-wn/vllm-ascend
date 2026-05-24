下面这版可以作为后续同学接手的设计说明/工程导览。

**整体目标**
原始 vLLM 的 CUDA Graph replay 是单图 padding 逻辑：当前 decode batch 不能精确命中已捕获图时，会 padding 到下一个 cudagraph size，然后重放一张更大的图。你的改动是在这个基础上增加“双图切分 replay”：

- `PADDING`：原始单图 padding replay。
- `DUAL_PARALLEL`：把 batch 切成两段，分别放到两套独立 buffer / 两个 graph pool / 两条 stream 上并行 replay。
- `DUAL_INPLACE`：仍然切成两段，但不额外分配第二套完整输入 buffer，而是在同一个 buffer 内按 offset 区分子图，用 `start_num_tokens` 标识第二段起点，从而复用单 buffer，降低显存和 copy 开销。

**核心切分思路**
切分入口在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:1169)：

```text
_should_split_for_cudagraph()
_get_split_point()
```

逻辑是：

- 只在 decode 且 `uniform_decode=True` 时切分。
- `PADDING` 不切分。
- batch 小于等于 `micro_batch_size` 不切分。
- split point 取“小于当前 batch 的最大 cudagraph capture size”。

例如 batch=416，capture sizes 有 `256,384,512`：

- `PADDING`：416 -> 512，padding=96。
- `DUAL_PARALLEL`：416 -> 256 + 256，第二段 160 padding 到 256，padding=96。
- `DUAL_INPLACE`：416 -> 256 + 160，第二段不 padding，padding=0。

**DUAL_PARALLEL**
`DUAL_PARALLEL` 是更传统的双图并行方案，工程上分三阶段。

1. 初始化阶段：分配两套独立持久化 buffer  
位置在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:337) 和 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:445)。

主要新增/使用：

```text
micro_input_ids
micro_positions
micro_query_start_loc
micro_seq_lens
micro_inputs_embeds
micro_input_batch
```

这些是 secondary stream / 第二张图专用的输入和 metadata buffer。这样第二段 replay 的地址稳定，满足 CUDA Graph replay 对输入地址一致性的要求。

2. 捕获阶段：两个 graph pool / 两套 graph  
位置在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:4959)：

```text
_capture_cudagraphs_dual()
```

`capture_model()` 中如果是 `DUAL_PARALLEL / DUAL_SERIAL`，会走双图 capture 路径，位置在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:4849)。

同时在 [cuda_graph.py](D:/Project/vllm/vllm/compilation/cuda_graph.py:164) 增加了 secondary graph pool：

```text
graph_pool_secondary
```

这样 primary / secondary 可以在不同 graph pool 上捕获，避免两套图的内存地址和 pool 管理互相干扰。

3. 推理阶段：两条 stream 并行 replay  
执行入口在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:3493)：

```text
_execute_model_dual_cudagraph()
```

具体实现在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:2989)。

主要流程：

- 第一段使用原始 input buffer。
- 第二段先 copy 到 `micro_*` buffer。
- 分别构造两个 `BatchDescriptor`。
- 分别 dispatch 到 cudagraph runtime mode。
- primary stream 跑第一张图。
- secondary stream 跑第二张图。
- 最后 `torch.cat([output_first, output_second])` 聚合输出。

关键 stream 在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:540) 初始化：

```text
self.graph_streams[StreamSlot.PRIMARY]
self.graph_streams[StreamSlot.SECONDARY]
```

**DUAL_INPLACE**
`DUAL_INPLACE` 的目标是解决 `DUAL_PARALLEL` 的额外显存和 copy 成本。

核心区别是：它不使用第二套完整输入 buffer 来承载第二段，而是在原始 input buffer 中直接按 offset 切分。

执行入口在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:3540)：

```text
_execute_model_dual_inplace()
```

具体实现在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:3127)。

关键点：

- 第一段 descriptor：

```text
BatchDescriptor(num_tokens=first_num_tokens, start_num_tokens=0)
```

- 第二段 descriptor：

```text
BatchDescriptor(num_tokens=second_num_tokens,
                start_num_tokens=first_num_tokens)
```

也就是第二张图不仅由 `num_tokens` 区分，还由 `start_num_tokens` 区分。这样同样是 `160 tokens`，如果它起始 offset 不同，可以被识别为不同 graph key。

`start_num_tokens` 是新增在 [forward_context.py](D:/Project/vllm/vllm/forward_context.py:37) 的：

```text
BatchDescriptor.start_num_tokens
```

并且 `non_uniform` 也保留这个字段，位置在 [forward_context.py](D:/Project/vllm/vllm/forward_context.py:52)。

CUDA Graph wrapper 侧使用完整 `BatchDescriptor` 作为 graph entry key，位置在 [cuda_graph.py](D:/Project/vllm/vllm/compilation/cuda_graph.py:161)：

```text
concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]
```

所以 `num_tokens + uniform_decode + start_num_tokens` 会共同决定是否命中已有图。

**INPLACE 的懒捕获**
为了支持 offset 子图，dispatcher 增加了 lazy key 逻辑，位置在 [cudagraph_dispatcher.py](D:/Project/vllm/vllm/v1/cudagraph_dispatcher.py:124)。

普通 cudagraph 只能 dispatch 到预先注册的 capture sizes。但 `DUAL_INPLACE` 的第二段可能是真实 remainder，比如 `160`，不一定在 capture sizes 里。因此这里允许：

```text
replay_mode == DUAL_INPLACE
start_num_tokens > 0
```

时动态加入 cudagraph key。第一次遇到该 offset 子图时捕获，后续相同切分可以复用。

这就是 `DUAL_INPLACE` 的核心收益：

- 不需要 `micro_input_ids / micro_positions` 那种第二套完整输入区。
- 不需要把第二段 copy 到 micro buffer。
- 不需要把 remainder padding 到 bucket。
- 对稳定 batch 来说，第一次 capture 后，后续 decode step 可以复用同一个 offset 子图。

**attention metadata**
切分后 metadata 也必须按 ubatch 拆开，否则 attention kernel 看到的 query range、slot mapping、seq lens 会和子图输入不一致。

主要处理位置在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:1619) 附近：

- `DUAL_INPLACE` 有单独 metadata 处理。
- `DUAL_PARALLEL` 会为第二段构造 micro metadata。
- `split_attn_metadata()` 用于按 ubatch 拆分 common attention metadata。

这部分是最容易出 CUDA illegal memory access 的地方，因为 CUDA Graph replay 要求：

```text
输入 tensor 地址稳定
metadata tensor 地址稳定
shape / range 与 capture 时一致
```

**num_input_tokens 处理**
关键位置在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:2506) 的 `_preprocess()`。

三种模式区别：

```text
PADDING:
num_input_tokens = pad_for_cudagraph(num_scheduled_tokens)

DUAL_PARALLEL:
num_input_tokens = first + pad_for_cudagraph(second)
然后 pad_out_ubatch_slice() 扩展第二段

DUAL_INPLACE:
num_input_tokens = first + real_second
不 padding 第二段
```

也就是说 `DUAL_INPLACE` 的理论目标已经从之前的 `256+256` 修正成 `256+160` 这种真实 remainder。

**测试统计**
execute_model 统计在 [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:592) 附近。

记录字段包括：

```text
replay_mode
top_runtime_mode
first_runtime_mode
second_runtime_mode
input_batch_size
seq_len
target_total
target_first
target_second
uniform_decode
split_for_cudagraph
num_input_tokens
first_num_tokens
second_num_tokens
elapsed_ms
```

benchmark 脚本在 [split_mode_benchmark.py](D:/Project/vllm/tests_script/split_mode_benchmark.py:123) 计算每种模式的理论 scheme：

```text
PADDING: target_total = padded batch

DUAL_PARALLEL:
target_first = lower bucket
target_second = padded remainder

DUAL_INPLACE:
target_first = lower bucket
target_second = real remainder
```

匹配逻辑在 [split_mode_benchmark.py](D:/Project/vllm/tests_script/split_mode_benchmark.py:211)：

只统计实际执行和理论目标一致的 decode rows：

```text
PADDING:
num_input_tokens == target_total

DUAL_*:
num_input_tokens == target_total
first_num_tokens == target_first
second_num_tokens == target_second
```

绘图脚本在 [plot_split_mode_execute.py](D:/Project/vllm/tests_script/plot_split_mode_execute.py:104)，会过滤匹配行并统计平均时间，同时计算 padding tokens。

**给后续同学的入手顺序**
建议按这个顺序看：

1. [forward_context.py](D:/Project/vllm/vllm/forward_context.py:37)  
   先理解 `BatchDescriptor` 为什么要加 `start_num_tokens`。

2. [cudagraph_dispatcher.py](D:/Project/vllm/vllm/v1/cudagraph_dispatcher.py:97)  
   看 graph key 是如何 dispatch 的，以及 `DUAL_INPLACE` offset 子图如何 lazy 加入。

3. [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:1169)  
   看是否切分、怎么选 split point。

4. [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:2506)  
   看不同模式下 `num_input_tokens` 和 padding 怎么处理。

5. [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:2989)  
   看 `DUAL_PARALLEL` 两套 buffer + 双 stream replay。

6. [gpu_model_runner.py](D:/Project/vllm/vllm/v1/worker/gpu_model_runner.py:3127)  
   看 `DUAL_INPLACE` 单 buffer + offset replay。

7. [cuda_graph.py](D:/Project/vllm/vllm/compilation/cuda_graph.py:161)  
   看 CUDA Graph entry 如何按 `BatchDescriptor` 缓存和复用。

8. [split_mode_benchmark.py](D:/Project/vllm/tests_script/split_mode_benchmark.py:123)  
   看理论目标和实验统计如何对齐。

一句话总结就是：`DUAL_PARALLEL` 用“空间换并行”，两套 buffer、两套图、两条 stream；`DUAL_INPLACE` 用“offset 标识换空间”，单 buffer 内按切分位置捕获 offset 子图，减少 padding、copy 和显存占用。