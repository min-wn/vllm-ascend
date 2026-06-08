# Chunked Prefill / Mixed Batch Split 切分并行调研

日期：2026-06-08

## 结论先行

相比当前基于 `FULL_DECODE_ONLY` 的纯 decode split，`chunked_prefill` 下的 prefill / mixed batch 更值得继续调研 split 切分，但收益边界必须限定清楚：

1. **有潜在收益的场景**：同一个 engine step 中存在多个互相独立的 prefill 请求，或 decode 请求与多个 chunked prefill 请求混排；split 按 request 边界切，两个 split 的 attention piece 可并行，非 attention / MatMul piece 串行，避免纯 full-graph 双流里已经观察到的 cube 竞争。
2. **不应作为第一阶段目标的场景**：单个长 prefill request 被从 token 中间切开。prefill 的后半段 token 依赖前半段 token 的 K/V 与 causal mask，直接把一个 request 的 query token 切成两段并行执行很容易破坏语义，除非 attention kernel 明确支持“q-range 切分但 K/V 覆盖完整上下文”的执行模式。
3. **当前代码不能直接打开开关验证**：`model_runner_v3.py` 的 inplace split precheck 显式拒绝 `uniform_decode=False` 和 `with_prefill=True`，`ubatch_utils.py` 的 inplace planner 也要求所有 request 的 scheduled token 数都等于 `uniform_decode_query_len`。因此 mixed/prefill split 是一个新方向，不是现有 split 配置的简单参数组合。
4. **相对 vllm-ascend 基线的收益假设更合理**：vllm-ascend 默认会把上游 vLLM v1 的 `FULL_AND_PIECEWISE` 调整为 `PIECEWISE`，所以 mixed/prefill baseline 本来就走 piecewise；在这个 baseline 上做 piecewise-aware split，不会额外把纯 decode 从 full graph 拉进 piecewise，代价结构比纯 decode 方向更合理。

一句话判断：**值得做 request-level mixed/prefill split 原型，优先验证“MatMul 串行、Attention 并行”的 piecewise split；不要从 single-request prefill token split 开始。**

## 本地代码事实

### 1. vLLM / vLLM-Ascend 的图模式语义

上游 vLLM 的 `CUDAGraphMode` 中：

- `FULL_DECODE_ONLY = (FULL, NONE)`：decode 走 full graph，mixed prefill-decode 不走 graph。
- `FULL_AND_PIECEWISE = (FULL, PIECEWISE)`：decode 走 full graph，prefill / mixed prefill-decode 走 piecewise graph。
- 注释明确写到 `FULL_AND_PIECEWISE` 是 v1 default，且通常是多数模型更好的性能模式。

代码位置：

- [vllm/vllm/config/compilation.py](/vllm-workspace/vllm/vllm/config/compilation.py:52)
- [vllm/vllm/config/compilation.py](/vllm-workspace/vllm/vllm/config/compilation.py:465)

vllm-ascend 平台层当前会把默认的 `FULL_AND_PIECEWISE` 改成 `PIECEWISE`：

- [vllm-ascend/vllm_ascend/platform.py](/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:219)

这意味着当前 Ascend 默认路径下，mixed/prefill 不是 full graph，而是 piecewise graph；而已有 split benchmark 脚本为了测纯 decode，显式设置为 `FULL_DECODE_ONLY` 并关闭 `chunked_prefill`：

- [benchmark_split_tpot.py](/vllm-workspace/benchmark_split_tpot.py:77)
- [benchmark_dynamic_batch_tpot.py](/vllm-workspace/benchmark_dynamic_batch_tpot.py:101)

### 2. 当前 split 实现只覆盖 uniform decode

`model_runner_v3.py` 的 inplace split precheck 有以下硬门槛：

- `not uniform_decode` 返回 `no_split_non_uniform_decode`。
- `with_prefill` 或 attention state 不是 decode-only 返回 `no_split_prefill_or_mixed`。
- MLA、PCP/DCP、默认 spec decode、LoRA 等也会被拒绝。

代码位置：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:1416)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:1456)

planner 层也再次固定为 uniform decode：

- `create_inplace_split_batch_slices()` 要求 `np.all(num_scheduled_tokens_per_request == q)`。
- `create_macro_inplace_split_batch_slices()` 同样要求所有 request 都等于 `q`。
- 普通 `split_batch_split()` 的注释也写明 split batch 是为 uniform decode batch 设计。

代码位置：

- [ubatch_utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/ubatch_utils.py:517)
- [ubatch_utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/ubatch_utils.py:842)
- [ubatch_utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/ubatch_utils.py:984)

因此，当前“split 各种实现方式”的结论只能外推到纯 decode / speculative uniform decode，不能直接代表 mixed/prefill。

### 3. 当前 piecewise split scheduler 已经提供了合适的调度雏形

`inplace_parallel_replay_policy="piecewise_attention_parallel"` 已经存在。其调度模型是：

- 捕获两个 split 对同一个 compiled model 的 piecewise runtime args。
- 对 `is_splitting_graph=False` 的子图串行执行。
- 对 `is_splitting_graph=True` 的 attention 子图并行执行。
- 支持 `event_chain` 避免每个 piece 都 host synchronize，也支持 persistent attention worker。

代码位置：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5590)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:6127)
- [inplace_piecewise_scheduler.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/inplace_piecewise_scheduler.py:187)
- [inplace_piecewise_scheduler.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/inplace_piecewise_scheduler.py:532)

这个结构比“两个 stream 同时 replay 完整 full graph”更适合 mixed/prefill，因为它能避免非 attention MatMul 的重叠竞争，只保留 attention overlap。

### 4. mixed batch 的 request 顺序假设

v2 model runner 中明确按 scheduled token 数排序，注释为 `Decode first, then prefill`：

- [v2/model_runner.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/v2/model_runner.py:106)

attention utils 也假设 batch 已 reorder，并用第一个 `query_lens > decode_threshold` 的位置作为 decode / prefill 分界：

- [attention/utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py:238)

这对 request-level split 是利好：如果保持 request 边界切分，每个 split 仍可保持“decode 前缀 + prefill 后缀”的局部顺序。第一阶段不需要重排单个 request 内部 token。

## 为什么纯 decode 不适合 piecewise split

纯 decode 的特点是每个 request 通常只有 1 个 query token：

- attention 本身更像短 query 的 flashdecode / FIA 路径；
- full graph 可以把 Python launch 与 metadata 更新压进完整 replay；
- piecewise 会把 attention 从 graph 中拆出来，增加 piece 调度、context 切换和 host 侧开销；
- full graph 双流 replay 会让每个 split 的 MatMul/MLP 同时竞争 cube，已有 profiling 已经显示 384+64 这类不均等 split 会拉低 primary MatMul 效率；
- split 的主要收益只剩“少 padding”或“并行 replay”，但少 padding 在 exact graph hit 时为零，并行 replay 又容易被 cube 竞争抵消。

这就是为什么当前文档中“纯 decode + piecewise split”不是一个好目标。它把最适合 full graph 的 decode 场景强行拉到更碎的图粒度上，收益来源不稳定。

## 为什么 mixed/prefill 可能更适合 split

### 1. baseline 已经是 piecewise

在 vllm-ascend 默认 `PIECEWISE` 下，mixed/prefill baseline 已经按 piecewise 子图执行。也就是说，split 原型不需要先承担“从 full graph 退化到 piecewise”的额外代价，而是在已有 piecewise baseline 上改变调度粒度：

```text
baseline:
  whole mixed batch:
    non-attn piece -> attention piece -> non-attn piece -> ...

request-level split:
  split0 / split1:
    non-attn piece 串行
    attention piece 并行
    non-attn piece 串行
    ...
```

这与现有 `inplace_piecewise_scheduler.py` 的设计完全一致。

### 2. prefill attention 的占比和可并行空间更大

mixed/prefill 中，prefill request 的 query length 通常大于 1，且存在 chunked context attention、slot mapping、block table、actual seq lengths 等更重的 attention metadata。相比纯 decode 的 q=1，prefill attention 更可能成为可 overlap 的部分。

如果两个 split 是不同 request 集合，它们之间没有跨 request attention 依赖，attention piece 并行具备语义基础。尤其是多个中等长度 chunked prefill request 混在一个 batch 中时，请求级切分可能带来：

- 降低单个 attention op 的最长尾部；
- 将 prefill attention 与 decode attention 分摊到两条 stream；
- 在不重叠 MatMul 的前提下保留 attention overlap；
- 减少某些 piecewise graph bucket 的 padding；
- 降低单次 attention workspace 峰值。

### 3. mixed batch 的形状更不规则，baseline padding/调度浪费更明显

纯 decode 的 batch size 是请求数，shape 离散且容易命中 capture bucket。mixed/prefill 的 shape 由 `sum(num_scheduled_tokens)`、`max_query_len`、decode/prefill 边界共同决定，更容易出现不均衡：

```text
decode requests: 128 * 1 token
prefill requests: 8 * 256 token chunk
total tokens: 2176
```

这种情况下，按 request boundary 切成两个 token workload 接近的 split，比纯 decode 的 `384+64` 这种小 remainder 更有希望形成有效 overlap。

## chunked_prefill 下的分块大小

### 1. 分块大小不是固定常量

vLLM v1 scheduler 没有独立的“prefill phase / decode phase”常量。每个调度步先拿到一个全局 token budget：

```text
token_budget = max_num_scheduled_tokens
```

这个值本质上来自 `scheduler_config.max_num_batched_tokens`。随后 scheduler 会先调度 `RUNNING` 请求，再调度 `WAITING` 请求；每个请求本步拿到的 token 数是：

```text
request_remaining = request.num_tokens_with_spec - request.num_computed_tokens
if long_prefill_token_threshold > 0:
    request_remaining = min(request_remaining, long_prefill_token_threshold)
scheduled_tokens_for_request = min(request_remaining, remaining_token_budget)
remaining_token_budget -= scheduled_tokens_for_request
```

对 waiting prefill 请求，公式等价为：

```text
request_remaining = request.num_tokens - num_computed_tokens
if long_prefill_token_threshold > 0:
    request_remaining = min(request_remaining, long_prefill_token_threshold)
if not enable_chunked_prefill and request_remaining > remaining_token_budget:
    do not schedule this request in this step
scheduled_tokens_for_request = min(request_remaining, remaining_token_budget)
```

代码位置：

- [scheduler.py](/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:235)
- [scheduler.py](/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:266)
- [scheduler.py](/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:271)
- [scheduler.py](/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:526)
- [scheduler.py](/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:531)
- [scheduler.py](/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:637)

因此，chunked prefill 的“块大小”应理解为每个 prefill request 在当前 engine step 中实际 scheduled 的 token 数，而不是固定 128/256/512 之类的编译常量。

### 2. 默认 max_num_batched_tokens 的典型范围

`SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` 只是测试/兜底默认值，注释明确说真实使用会在 `EngineArgs.create_engine_config` 里设置。

`EngineArgs.get_batch_defaults()` 的当前逻辑是：

- 设备显存 >= 70GiB 且不是 A100：`LLM_CLASS=16384`，`OPENAI_API_SERVER=8192`。
- 其他设备：`LLM_CLASS=8192`，`OPENAI_API_SERVER=2048`。
- CPU 还有单独默认，不适用于 NPU 推理判断。

代码位置：

- [scheduler.py](/vllm-workspace/vllm/vllm/config/scheduler.py:44)
- [arg_utils.py](/vllm-workspace/vllm/vllm/engine/arg_utils.py:1770)
- [arg_utils.py](/vllm-workspace/vllm/vllm/engine/arg_utils.py:1799)
- [arg_utils.py](/vllm-workspace/vllm/vllm/engine/arg_utils.py:1811)
- [arg_utils.py](/vllm-workspace/vllm/vllm/engine/arg_utils.py:1942)

对 910B3 64GB 这类卡，如果平台上报显存低于 70GiB 且用户没有显式传 `--max-num-batched-tokens`，典型默认值更接近：

```text
OpenAI server serving: 2048 tokens / step
LLM offline/class usage: 8192 tokens / step
```

如果用户或 benchmark 显式设置，例如 `max_num_batched_tokens=65536`，prefill chunk 就可以远大于上述默认值；当前本地 `benchmark_dynamic_batch_tpot.py` 默认还设置了 `enable_chunked_prefill=False`，所以不能用它的默认参数代表 chunked prefill mixed baseline。

### 3. mixed step 中典型 prefill 占比

在 decode-first mixed batch 中，decode 请求通常每个本步只消耗 1 个 token；prefill chunk 消耗剩余 budget。简化估算：

```text
B = max_num_batched_tokens
D = 本 step decode 请求数
T = long_prefill_token_threshold，0 表示不额外限制

prefill_chunk <= B - D
if T > 0:
    prefill_chunk <= min(T, B - D)
```

例如：

```text
B=2048, D=128, T=0   => 单个 prefill chunk 最多约 1920 tokens
B=2048, D=512, T=0   => 单个 prefill chunk 最多约 1536 tokens
B=8192, D=512, T=0   => 单个 prefill chunk 最多约 7680 tokens
B=8192, D=1024, T=0  => 单个 prefill chunk 最多约 7168 tokens
B=8192, D=512, T=512 => 单个 prefill chunk 最多约 512 tokens
```

这里的“最多”还会被请求剩余 prompt tokens、KV block 可分配量、encoder/multimodal 约束、max model len 约束继续压小。

配置默认值：

- `max_num_partial_prefills=1`
- `max_long_partial_prefills=1`
- `long_prefill_token_threshold=0`
- `enable_chunked_prefill=True`

代码位置：

- [scheduler.py](/vllm-workspace/vllm/vllm/config/scheduler.py:64)
- [scheduler.py](/vllm-workspace/vllm/vllm/config/scheduler.py:68)
- [scheduler.py](/vllm-workspace/vllm/vllm/config/scheduler.py:74)
- [scheduler.py](/vllm-workspace/vllm/vllm/config/scheduler.py:78)
- [scheduler.py](/vllm-workspace/vllm/vllm/config/scheduler.py:235)

这意味着默认思路下，最常见的 mixed shape 不是很多个 256-token prefill 均匀混排，而更可能是：

```text
decode prefix: N * 1 token
prefill suffix: 1 个大 chunk，大小约为 B - N
```

如果开启 concurrent partial prefill 或手动设置 `long_prefill_token_threshold`，才更容易得到多个中等大小 prefill request 共同占据一个 step 的形状，例如 `8 * 256` 或 `4 * 512`。这类形状才更适合 request-level split。

### 4. vllm-ascend dynamic batch 的特殊情况

vllm-ascend 的 `SchedulerDynamicBatch` 会在普通 token budget 之上做一次查表调整：

```text
token_budget = budget_refiner.refine_budget(self.running, token_budget)
```

它还会把 running 队列重排成 decode-first：

```text
self.running = decode_requests + prefill_requests
```

代码位置：

- [scheduler_dynamic_batch.py](/vllm-workspace/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py:37)
- [scheduler_dynamic_batch.py](/vllm-workspace/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py:89)
- [scheduler_dynamic_batch.py](/vllm-workspace/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py:161)
- [scheduler_dynamic_batch.py](/vllm-workspace/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py:165)
- [platform.py](/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:303)

因此 dynamic batch 下的 prefill chunk 不能只看 `max_num_batched_tokens`，还要看 `profile_table.csv` 对当前 decode context / decode request 数给出的 chunk_size。当前仓库未包含该表文件，所以本地只能确认机制，不能给出固定表值。

### 5. 对 split 的直接约束

为了“不破坏请求”，第一阶段 split planner 应使用 scheduler 输出的 `num_scheduled_tokens` 作为真实边界，而不是自己重新按固定 chunk size 切 token：

```text
request_i scheduled token range:
  [request_start, request_start + num_scheduled_tokens[i])

合法 split:
  只能在 request_i 和 request_{i+1} 之间切

非法 split:
  从 request_i 的 scheduled token range 中间切开
```

如果默认 mixed step 只有一个大 prefill chunk，request-boundary split 的选择通常只有：

```text
split0: decode requests
split1: the single prefill chunk
```

或者：

```text
split0: decode requests + the single prefill chunk
split1: empty
```

前者可能 token 负载极不均衡，后者没有 split 意义。因此，真正适合 mixed request-level split 的 workload 应满足：

- 至少两个 prefill request 在同一个 step 中被调度；或
- decode token 数足够大，decode prefix 本身能成为一个有意义的 split；或
- 手动设置较小 `long_prefill_token_threshold`，把多个长请求限制成中等 chunk；或
- dynamic batch 表本身给出较小 chunk_size，形成更多可组合 request 边界。

如果 workload 是“一个长 prefill 请求 + 少量 decode 请求”，应直接 fallback baseline，除非后续实现 intra-request q-range split。

## 关键限制：不能直接切开单个 prefill request

prefill 与 decode 最大差别在于，同一个 prefill chunk 内的后续 token 会 attend 到前面的新 token。若把一个 request 的 token range 直接切成：

```text
request A prefill tokens: [0..1023]
split0: [0..511]
split1: [512..1023]
```

split1 的 attention 需要看到 split0 产生的 K/V，且 causal mask、slot_mapping、query_start_loc、actual_seq_lengths_kv 都要表达“query 只在后半段，但 key/value 覆盖完整前缀 + 前半段新 token”。当前 split metadata 不是为这种 intra-request q-range 模式设计的。

因此第一阶段必须采用更保守的约束：

- 只按 request boundary 切；
- 不允许一个 prefill request 被拆到两个 split；
- 如果某个单独 prefill request 占总 token 的比例过高，直接 fallback baseline；
- split 后每个 split 内部仍保持 decode 在前、prefill 在后；
- 先不支持 MLA、PCP/DCP、LoRA、spec decode。

## 推荐切分方案细化

### 1. 第一版目标形态

建议第一版实现一个独立的 `mixed_request_split` 路径，而不是改造现有 uniform decode inplace planner：

```text
适用 batch:
  prefill-only 多请求
  mixed decode + prefill

图模式:
  CUDAGraphMode.PIECEWISE

split 数:
  2-way split

切分单位:
  request boundary

执行策略:
  non-attention piece 串行
  attention piece 双流并行
```

不要让第一版同时解决以下问题：

- 单 request prefill token range split；
- MLA / PCP / DCP；
- LoRA；
- speculative decode；
- 多于 2 个 split；
- full graph mixed split；
- dynamic graph key 自由 lazy capture。

这些都属于第二阶段之后的扩展项。

### 2. 为什么不能直接复用现有 offset split

当前 `SplitBatchSlice` 已经能表达：

- `request_slice`
- `token_slice`
- `padded_num_tokens`
- `start_num_tokens`

代码位置：

- [ubatch_utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/ubatch_utils.py:284)

但现有 inplace offset graph key 有一个硬限制：`start_num_tokens > 0` 时 dispatcher 会走 `_create_inplace_offset_batch_descriptor()`，而这个函数明确要求 `uniform_decode=True`：

- [cudagraph_dispatcher.py](/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py:90)
- [cudagraph_dispatcher.py](/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py:106)
- [cudagraph_dispatcher.py](/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py:125)

这意味着 mixed/prefill split 不能照搬当前 uniform decode 的 offset-key 设计：

```text
错误方向:
  split1 token_slice = [first_tokens, total_tokens)
  split1 start_num_tokens = first_tokens
  dispatch(uniform_decode=False, start_num_tokens>0)

结果:
  dispatcher offset key 路径直接不支持 non-uniform/mixed
```

因此第一版 mixed split 应采用“紧凑 split buffer”语义：

```text
原始 batch token:
  [decode tokens][prefill0 tokens][prefill1 tokens]...

split0 compact buffer:
  copy/view selected request tokens -> [0, actual0)
  pad tail -> [actual0, graph0)

split1 compact buffer:
  copy/view selected request tokens -> [0, actual1)
  pad tail -> [actual1, graph1)

graph descriptor:
  num_tokens = graphN
  uniform = False
  num_reqs = None after relax
  start_num_tokens = 0
  graph_variant = "mixed_request_split" or empty
```

这样可以继续复用 mixed PIECEWISE 的 `cudagraph_capture_sizes` bucket，而不是为每个 runtime token offset 生成新的图 key。

### 3. planner 输入

新增 planner 建议放在 `ubatch_utils.py`，独立于 `create_inplace_split_batch_slices()`：

```python
create_mixed_request_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    cudagraph_capture_sizes: Iterable[int],
    *,
    decode_threshold: int,
    pad_for_cudagraph: Callable[[int], int],
    min_total_tokens: int = 128,
    min_tokens_per_split: int = 64,
    max_padding_tokens_per_split: int | None = None,
    max_padding_ratio_per_split: float = 0.25,
    max_single_request_ratio: float = 0.70,
    min_prefill_reqs_for_prefill_split: int = 2,
    prefer_prefill_balance: bool = True,
) -> tuple[Optional[InplaceSplitPlan], str]
```

必要输入来自当前 `_prepare_inputs()` 已经构造出的数据：

- `num_scheduled_tokens`：每个 request 本 step 的真实 token 数。
- `total_num_scheduled_tokens`：真实总 token 数。
- `decode_threshold`：判断 decode / prefill 的 query length 阈值。
- `cudagraph_capture_sizes`：piecewise 图可用 token bucket。

代码位置：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:1999)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:2058)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:2072)

### 4. planner 候选边界

只枚举 request 边界：

```python
cu_tokens = np.concatenate([[0], np.cumsum(num_scheduled_tokens)])

for split_req in range(1, num_reqs):
    left_req_slice = slice(0, split_req)
    right_req_slice = slice(split_req, num_reqs)
    left_token_slice = slice(0, cu_tokens[split_req])
    right_token_slice = slice(cu_tokens[split_req], total_num_tokens)
```

由于 vllm-ascend 当前 batch 顺序假设是 decode first then prefill，任意 request 边界切分都不会产生“prefill 在前、decode 在后”的局部乱序：

- 切在 decode 区内部：`split0=decode`，`split1=decode+prefill`
- 切在 decode/prefill 边界：`split0=decode`，`split1=prefill`
- 切在 prefill 区内部：`split0=decode+prefill`，`split1=prefill`

相关代码：

- [attention/utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py:236)
- [attention/utils.py](/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py:263)

### 5. 候选过滤条件

每个候选边界先做硬过滤：

```text
1. split0 / split1 都非空。
2. 每个 split actual tokens >= min_tokens_per_split。
3. 每个 split graph tokens <= max_cudagraph_capture_size。
4. 每个 split padding 不超过 max_padding_tokens / max_padding_ratio。
5. 不切开任何 request。
6. batch 中不能存在单个 request token 占比过高。
7. split 后 attention backend 选择不能和 unsplit baseline 不兼容。
8. 不支持 LoRA / MLA / PCP / DCP / spec decode / mrope。
```

单 request 占比过滤很重要：

```text
single_request_tokens / total_tokens > 0.70
=> no_split_single_request_dominates
```

因为这种 batch 只有 intra-request split 才可能真正平衡，而第一版明确不做 intra-request split。

### 6. graph token bucket 选择

对每个 split：

```python
actual_tokens = token_slice.stop - token_slice.start
graph_tokens = pad_for_cudagraph(actual_tokens)
padding_tokens = graph_tokens - actual_tokens
```

如果 `graph_tokens` 不在可捕获范围，直接 fallback eager/piecewise baseline，不建议第一版 lazy capture 任意新 bucket。

mixed piecewise dispatch 必须使用：

```text
uniform_decode = False
disable_full = True
start_num_tokens = 0
```

原因是 mixed/prefill 要命中 relaxed PIECEWISE descriptor：

- [forward_context.py](/vllm-workspace/vllm/vllm/forward_context.py:68)
- [cudagraph_dispatcher.py](/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py:271)
- [cudagraph_dispatcher.py](/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py:285)

不要在第一版给 mixed split 引入 `start_num_tokens` offset key，否则会同时带来“dispatcher 不支持 non-uniform offset”和“图 key 爆炸”两个问题。

### 7. 候选评分

第一版建议用保守评分，优先平衡 attention-heavy workload，其次控制 padding：

```python
score = (
    max(left_attention_cost, right_attention_cost),
    abs(left_attention_cost - right_attention_cost),
    left_padding + right_padding,
    max(left_actual_tokens, right_actual_tokens),
)
```

attention cost 可以先用简化模型：

```python
decode_tokens = sum(q for q <= decode_threshold)
prefill_tokens = sum(q for q > decode_threshold)
attention_cost = decode_tokens * decode_weight + prefill_tokens * prefill_weight
```

第一版可以取：

```text
decode_weight = 1
prefill_weight = 4
```

更精细的第二版再引入 `seq_lens`：

```text
per_request_attention_cost ~= query_len * seq_len
```

这样可以避免把一个 token 数看似均衡但 prefill context 极不均衡的方案误选为最优。

### 8. metadata slicing

这是实现风险最高的部分，应新增 mixed 专用函数，不要复用当前 uniform decode 隐含假设：

```python
_make_mixed_request_split_metadata_piecewise(
    split_batch_slices,
    common_attn_metadata,
    input_ids,
    positions,
    inputs_embeds,
    intermediate_tensors,
)
```

每个 split 需要重新构造：

```text
query_lens_split = num_scheduled_tokens[request_slice]
query_start_loc_split = [0] + cumsum(query_lens_split)
seq_lens_split = seq_lens[request_slice]
block_table_split = block_table[request_slice]
num_computed_tokens_split = num_computed_tokens[request_slice]
slot_mapping_split = slot_mapping[token_slice]
positions_split = positions[token_slice]
input_ids_split = input_ids[token_slice]
max_query_len_split = max(query_lens_split)
num_actual_tokens_split = sum(query_lens_split)
num_input_tokens_split = graph_tokens
```

同时要重新判断 split 内的 attention state：

```text
all query_lens <= decode_threshold     => DecodeOnly
all query_lens > decode_threshold      => PrefillOnly / ChunkedPrefill
otherwise                              => Mixed
```

padding tail 必须满足：

- padding token 不参与 logits；
- padding token 不写有效 KV；
- padding token 的 slot mapping 使用无效值或 backend 可接受的 dummy slot；
- `query_start_loc` 不为 padding token 创建 fake request；
- `num_actual_tokens` 仍然是真实 token 数，不是 graph token 数。

### 9. 输入 buffer 策略

由于 mixed split 不应使用 non-zero `start_num_tokens` offset key，第一版建议新增 compact input path：

```text
split0_input_buffer[:actual0] = original_input[token_slice0]
split0_input_buffer[actual0:graph0] = 0

split1_input_buffer[:actual1] = original_input[token_slice1]
split1_input_buffer[actual1:graph1] = 0
```

`positions`、`inputs_embeds`、`intermediate_tensors` 同理。

这样做的代价是多一次小规模 copy，但收益是：

- graph descriptor 不依赖 runtime token offset；
- mixed PIECEWISE capture bucket 可复用；
- split 内 metadata 全部从 0 开始，`query_start_loc` 和 logits index 更容易校验；
- 避免改动 dispatcher 的 uniform decode offset key 语义。

### 10. replay 调度

执行层复用当前 `piecewise_attention_parallel` 思路：

```text
for each split:
  capture_piecewise_model_call(...)

piece scheduler:
  non-attention piece:
    split0 on main stream
    split1 on main or secondary stream after event

  attention piece:
    split0 on main stream
    split1 on secondary stream
```

当前 `_run_split_batch_inplace_parallel_piecewise()` 已经要求 `CUDAGraphMode.PIECEWISE` 且只支持 2 splits，这正好符合第一版范围：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5590)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5604)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5608)

但 `_make_split_batch_metadata_inplace_parallel()` 当前 dispatch 时仍写死 `uniform_decode=True`，mixed 路径必须另起函数或加显式分支：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:4733)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:4736)

### 11. fallback reason

建议把 no-split 原因做细，避免 benchmark 时只看到“不生效”：

```text
no_split_not_piecewise
no_split_no_prefill_or_not_target
no_split_single_request_dominates
no_split_too_few_request_boundaries
no_split_padding_too_large
no_split_graph_bucket_missing
no_split_attention_backend_mismatch
no_split_unsupported_mla
no_split_unsupported_pcp_dcp
no_split_unsupported_lora
no_split_unsupported_spec_decode
no_split_copy_buffer_not_ready
```

这些字段应该进入 split debug JSONL，和现有 `target_batch_descriptors` 类似记录每个 split 的：

```text
actual_num_tokens
graph_num_tokens
padding_tokens
num_reqs
num_decode_reqs
num_prefill_reqs
max_query_len
start_num_tokens
uniform
runtime_mode
```

## decode FULL + prefill PIECEWISE 的 hybrid split 评估

### 1. 方案含义

这个想法不是上游 `FULL_AND_PIECEWISE` 的现有语义。上游语义是：

```text
decode-only batch:
  FULL graph

mixed prefill-decode batch:
  PIECEWISE graph
```

代码位置：

- [compilation.py](/vllm-workspace/vllm/vllm/config/compilation.py:61)
- [compilation.py](/vllm-workspace/vllm/vllm/config/compilation.py:488)

而这里讨论的是把同一个 mixed batch 拆成两个子 batch：

```text
split0: decode requests only
  runtime mode = FULL

split1: prefill requests only
  runtime mode = PIECEWISE
```

这属于新的 per-split hybrid runtime mode，不是当前 dispatcher / piecewise scheduler 的简单参数组合。

### 2. 潜在收益

这个方向的性能直觉是成立的：

- decode q=1 天然适合 FULL graph；
- FULL decode 可以把 attention、MLP、metadata 更新、launch 开销全部包进完整 replay；
- prefill/mixed 仍然保留 PIECEWISE，避免 prefill full graph 不支持或代价过大；
- 如果 decode 请求很多，decode 子 batch 用 FULL 可能比把 decode 也放进 PIECEWISE 更低延迟；
- 如果系统目标更偏 resident decode ITL，而不是 prefill TTFT，先跑 decode FULL 再跑 prefill PIECEWISE 可能有 latency isolation 价值。

### 3. 当前实现为什么不能直接支持

当前 `piecewise_attention_parallel` 要求两个 split 都是 `CUDAGraphMode.PIECEWISE`：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5590)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5604)

它还要求两个 split capture 出来的 piecewise runtime call 使用同一个 runtime handle：

- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5635)
- [model_runner_v3.py](/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v3.py:5643)

decode FULL + prefill PIECEWISE 会得到两个不同的执行模型：

```text
decode:
  opaque full graph replay

prefill:
  piecewise runtime call + piece scheduler
```

因此现有 `InplacePiecewiseSplitScheduler` 不能直接调度它。需要新建 hybrid scheduler，支持一个 split 是 opaque full replay，另一个 split 是 piecewise replay。

此外，vllm-ascend 平台层当前会把默认 `FULL_AND_PIECEWISE` 改成 `PIECEWISE`：

- [platform.py](/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:219)

所以默认 Ascend 路径下 decode FULL 甚至不是 mixed baseline 的自然状态。要做 hybrid，需要显式恢复/支持 full decode graph capture，并处理额外图内存。

### 4. 最大风险：FULL graph 是 opaque 的

如果把 decode FULL 和 prefill PIECEWISE 并行跑，decode FULL graph 内部包含：

```text
attention + MLP/MatMul + norm + residual + metadata相关更新
```

它是一个 opaque replay，调度器看不到里面哪些阶段是 attention，哪些阶段是 MatMul。因此无法像 `piecewise_attention_parallel` 那样做到：

```text
non-attention 串行
attention 并行
```

结果很可能变成：

```text
decode FULL 的 MatMul
  overlap
prefill PIECEWISE 的 MLP/MatMul
```

这会重新引入当前纯 decode full graph 双流已经观察到的 cube 竞争问题。也就是说，hybrid 并行不一定比 all-piecewise attention-only parallel 更好。

### 5. 三种可行变体

#### A. 串行 hybrid

```text
decode FULL
prefill PIECEWISE
merge outputs
```

优点：

- decode 子 batch 用 FULL，q=1 路径最高效；
- 避免 FULL 与 prefill MLP 并行抢 cube；
- 实现比并行 hybrid 简单。

缺点：

- 没有 attention overlap；
- prefill TTFT 会被 decode FULL 额外排在前面；
- 如果 engine 不能提前返回 decode 输出，只是内部先算完 decode，并不会直接改善用户可见 ITL。

适合做 benchmark control，不适合作为最终优化形态。

#### B. 直接并行 hybrid

```text
stream0: decode FULL
stream1: prefill PIECEWISE
```

优点：

- 实现直觉简单；
- decode 和 prefill 端到端可能有 overlap。

缺点：

- FULL graph opaque，无法避免 MatMul/MLP 重叠；
- 资源竞争风险最大；
- profiler 很可能看到 cube 利用率互相干扰；
- 输出 merge、KV 写入、graph pool 隔离都更复杂。

不建议作为第一版。

#### C. 调度感知 hybrid

理想形态是：

```text
decode FULL replay 作为一个 opaque critical section
prefill PIECEWISE non-attention 不与 decode FULL 重叠
prefill PIECEWISE attention 尽量与 decode FULL 的 attention窗口重叠
```

但因为 FULL graph 内部不可见，除非 PTA/NPU graph 层能暴露更细粒度的 event 或 graph segment，否则无法可靠实现。这个方向更接近 PTA 下沉后的长期能力，不适合 Python 层第一阶段。

### 6. 建议优先级

建议优先级如下：

```text
P0: all-piecewise request-level split
    prefill/decode 都走 PIECEWISE
    non-attention 串行，attention 并行

P1: serial hybrid control
    decode FULL + prefill PIECEWISE 串行
    用于验证 decode FULL 是否明显降低 decode 子批耗时

P2: parallel hybrid experiment
    decode FULL 与 prefill PIECEWISE 并行
    只在 profiler 证明 cube 竞争可控时继续

P3: PTA-level scheduled hybrid
    需要 graph segment / event 原语支持
```

判断 hybrid 是否值得推进，应满足：

- decode 请求数足够大，例如 decode tokens >= 128/256；
- decode FULL graph 能稳定命中；
- prefill 侧不是单个巨大 request 独占，否则 request-level split 仍不平衡；
- profiler 显示直接并行没有明显拉低 MatMul/MLP 效率；
- 输出路径能真正让 decode 结果更早返回，否则 latency isolation 的收益只停留在内部时间线上。

### 7. 当前结论

`decode FULL + prefill PIECEWISE` 是值得作为第二阶段 benchmark control 的方向，但不建议作为第一版 mixed split 的主线。

第一版更应该先做：

```text
mixed batch request-level split
both splits use PIECEWISE
non-attention serial
attention parallel
```

原因是它和当前 Ascend 默认 PIECEWISE baseline 更一致，图 key 更简单，也能避免 FULL graph opaque 带来的 cube 竞争。

## 与 vllm-ascend 基线的预期收益/风险

### 可能收益

1. **TTFT / mixed step latency**：多个 prefill 请求并行执行 attention piece，有机会缩短 prefill-heavy step。
2. **decode tail latency**：mixed step 中已有 decode 请求可能被 prefill attention 拖慢；split 后 decode-heavy split 与 prefill-heavy split 的 attention overlap 可能降低 resident decode 请求的阻塞。
3. **更符合当前 piecewise baseline**：不再为了 split 把 mixed/prefill 改成 full graph，而是复用 baseline 的 piecewise 编译结果。
4. **更接近 PTA 下沉目标**：双流原语不应只服务纯 decode full graph replay，更应该抽象成 piecewise attention overlap 原语。

### 主要风险

1. **metadata slicing 复杂度高**：`query_start_loc` 要 rebased，`seq_lens`、`block_table`、`slot_mapping`、`positions`、`logits_indices` 都要按 request/token slice 保持一致。
2. **graph key 爆炸**：mixed/prefill shape 比纯 decode 多，若每个 `start_num_tokens + num_tokens + max_query_len` 都 lazy capture，图数量和内存会快速膨胀。
3. **attention backend 差异**：当前 inplace precheck 直接拒绝 MLA；DeepSeek/MLA 的 chunked prefill metadata 更复杂，应晚于普通 attention backend 验证。
4. **scheduler 可能没有真实 mixed batch**：当前动态 benchmark 默认 `prefill_only_on_arrival=1`，新请求到达时先单独 prefill，再恢复 decode，因此不是 mixed split 实验。
5. **收益可能被非 attention 占比限制**：如果 mixed/prefill step 的耗时主要在 MLP/MatMul，piecewise 策略故意串行非 attention，收益上限会受限。

## 推荐实现路线

### Phase 0：先补观测，不改行为

在 perf/debug JSONL 中增加 mixed/prefill 维度：

- `with_prefill`
- `attn_state`
- `uniform_decode`
- `num_reqs`
- `num_decode_reqs`
- `num_prefill_reqs`
- `num_decode_tokens`
- `num_prefill_tokens`
- `max_query_len`
- `cudagraph_runtime_mode`
- `piecewise_total_graphs / attention_pieces / capturable_pieces`

目的：先确认默认调度下真实 mixed batch 的比例、形状分布和 baseline latency，否则容易在“其实没有 mixed step”的 benchmark 上得出错误结论。

### Phase 1：实现 request-level mixed split planner

新增一个独立 planner，不复用 `create_inplace_split_batch_slices()` 的 uniform decode 假设：

```python
create_mixed_request_split_slices(
    num_scheduled_tokens_per_request,
    decode_threshold,
    capture_sizes,
    *,
    min_tokens_per_split,
    max_single_request_ratio,
    split_policy="balanced_tokens",
)
```

核心规则：

1. 只枚举 request 边界，不枚举 token 边界。
2. 只接受每个 split token 数都大于阈值的方案。
3. 保持每个 split 内 decode/prefill 相对顺序。
4. 如果存在单个 request token 数超过总 token 的阈值，例如 60%，直接 no split。
5. planner 评分先最小化 `max(split0_graph_tokens, split1_graph_tokens)`，再最小化 padding。

第一阶段建议只支持 `CUDAGraphMode.PIECEWISE`，避免 `FULL_AND_PIECEWISE` 下 offset key 误走 decode full graph 语义。

### Phase 2：扩展 metadata slicing

现有 `_make_split_batch_metadata_inplace_parallel()` 对 uniform decode 做了较多隐含假设，例如 dispatch 时强制 `uniform_decode=True`。mixed split 需要新的 metadata 构造路径：

- `BatchDescriptor.uniform=False`
- `BatchDescriptor.num_reqs=None` 或 split 内真实 request 数，根据 piecewise key 是否需要 request 数决定
- `start_num_tokens` 保留 token offset
- `graph_variant="mixed_request_split"`
- `attention_backend` 第一阶段留空或按 baseline
- `capture_metadata_mode` 记录 mixed split 的 metadata 策略

需要重点验证：

- `query_start_loc` 对 split 内 request 重新从 0 开始累计；
- `positions` 使用 token slice；
- `slot_mapping` 使用 token slice，且两个 split 不重叠；
- `block_table` 使用 request slice；
- `seq_lens` 使用 request slice；
- `logits_indices` 只保留每个 request 的最后 scheduled token；
- prefill partial request 的 sampled token 仍按 baseline 忽略。

### Phase 3：复用 piecewise_attention_parallel 调度

mixed/prefill split 不建议走 full graph 双流 replay，而应直接走：

```text
inplace_parallel_replay_policy = piecewise_attention_parallel
cudagraph_mode = PIECEWISE
```

调度策略：

- 非 attention piece：split0 后 split1，串行。
- attention piece：split0 与 split1 并行。
- 使用 `event_chain`，避免 per-piece host sync。
- 使用 `persistent_thread`，避免每个 attention piece 建临时线程。

这条路径已经有主体实现，主要缺口是 planner 和 mixed metadata。

### Phase 4：只在 request-level 成功后，再评估 intra-request prefill split

如果 request-level split 证明收益明显，再考虑单 request prefill token split。但这需要 attention kernel / metadata 能表达：

- query 是局部 token range；
- key/value 覆盖完整可见上下文；
- causal mask 正确限制同 chunk 后续 token；
- KV cache 写入与读取顺序不破坏依赖。

如果 kernel 不支持这个语义，单 request prefill split 不应推进。

## Benchmark 设计

### baseline 与实验组

建议至少对比：

1. `baseline_piecewise`：vllm-ascend 默认 `PIECEWISE`，`chunked_prefill=True`，split disabled。
2. `mixed_split_serial`：request-level split，两个 split 串行，验证 correctness 和 metadata。
3. `mixed_split_piecewise_attention_parallel`：request-level split，非 attention 串行、attention 并行。
4. `decode_full_decode_only_control`：现有纯 decode split，用于说明新方向不是靠 decode benchmark 获益。

不要用当前 `padding / aclgraph / inplace_parallel` 的 `FULL_DECODE_ONLY` 作为 mixed baseline，因为它会把 mixed batch 的 graph 语义改变掉。

### workload

建议构造四类：

1. **多 prefill request mixed**：已有 128/256 个 decode 请求正在生成，同时每隔 N 个 decode step 注入 8/16/32 个新请求，不启用 `prefill_only_on_arrival`。
2. **prefill-only 多请求**：同一 step 里多个 chunked prefill request，验证纯 prefill request-level split。
3. **单长 prefill request**：一个超长 chunked prefill，预期 fallback，不追求收益。
4. **纯 decode 对照**：验证新 planner 不误触纯 decode，或触发后无明显收益时 fallback。

当前 `run_dynamic_batch_tpot.sh` 默认 `PREFILL_ONLY_ON_ARRIVAL=1`，会把到达请求的 prefill 与老请求 decode 隔离，不能作为 mixed batch benchmark。mixed 实验应设置：

```bash
PREFILL_ONLY_ON_ARRIVAL=0
```

并保持 `enable_chunked_prefill=True` 或不覆盖 vLLM 默认值。

### 指标

必须新增 mixed/prefill 视角的指标，不能只看 decode TPOT：

- mixed step latency
- resident decode request 在 mixed step 的 inter-token latency
- prefill chunk latency
- TTFT
- total tokens/s
- split hit rate
- fallback reason histogram
- graph capture / replay count
- per-piece attention vs non-attention 时间
- NPU profiler 中 cube/vector 利用率、MatMul 耗时、FIA/PA 耗时、stream overlap
- HBM peak 与 graph pool 占用

### 正确性验证

最小正确性用例：

```text
decode reqs: 4, each q=1
prefill reqs: 2, each q=64
total tokens: 132
split candidates:
  split0: decode reqs + prefill0 = 68 tokens
  split1: prefill1 = 64 tokens
```

需要验证：

- baseline piecewise 与 split serial 输出一致；
- split serial 与 split attention parallel 输出一致；
- 每个 request 的 sampled token 与 baseline 一致；
- KV cache slot mapping 两个 split 无重叠；
- `query_start_loc` split 内 rebased 正确；
- lazy capture 后第二次运行命中 replay；
- graph key 不随 host metadata 地址漂移。

## 对 PTA 迁移的含义

这个方向对 PTA 下沉更有价值，因为它把双流原语从“两个完整 decode 图并行 replay”提升为“piecewise 图上的 attention overlap 原语”：

```text
PTA primitive:
  replay_piecewise_split(
      graph_handle,
      split_descriptors,
      schedule = serial_non_attention_parallel_attention,
      stream_main,
      stream_secondary,
  )
```

PTA 层应抽象的不是 vLLM 的 `split_batch_config`，而是更稳定的语义：

- split 数量和 request/token 边界；
- 每个 split 的 actual tokens / graph tokens / start tokens；
- 是否允许 token-in-request split；
- 每个 piece 的资源类型：attention 可并行，MatMul/MLP 串行；
- graph key 的 descriptor；
- metadata 更新与 stream/event 同步策略。

这样既能覆盖当前 decode split，也能覆盖更有潜力的 mixed/prefill split，并且不把 PTA API 绑死到 vLLM Python 层配置。

## 最终建议

1. 短期不要继续把主要精力放在“纯 decode + piecewise split”上；它天然不占优。
2. 立刻补 mixed/prefill baseline 观测，确认默认 `chunked_prefill` 下真实 mixed step 的形状和占比。
3. 做一个严格 request-boundary 的 mixed split 原型，先串行验证 correctness，再启用 piecewise attention parallel。
4. benchmark baseline 使用 vllm-ascend 默认 `PIECEWISE + chunked_prefill`，不要使用 `FULL_DECODE_ONLY + enable_chunked_prefill=False`。
5. 如果 request-level mixed split 没有明显收益，再评估是否值得做高风险的 intra-request prefill token split；不要反过来从最复杂路径开始。
