# Split Batch Logits Root Cause Report

## 当前结论修订

### 2026-05-12 验证更新

用户把 `ubatch_utils.py` 中的 exact-shape 逻辑恢复为：

```python
if exact_shape_num_tokens is not None:
```

也就是即使存在 `cudagraph_capture_sizes`，`VLLM_ASCEND_SPLIT_EXACT_SHAPE=1`
仍然会覆盖每个 split 的 `padded_num_tokens` 和 `physical_num_tokens`。
恢复后，同一 correctness 测试完全正确。

这个结果把根因重新锁回 execution shape：

- split/merge 路径保留。
- ACL graph 路径保留。
- parallel stream 路径保留。
- 改变的是每个 split lane 的实际执行 shape：不再使用 `68 + 64` 最小图，
  而是使用与 full path 对齐的 full execution shape。
- 结果从 mismatch 变为完全一致。

因此，目前最强结论是：剩余 token mismatch 的必要触发条件是 split lane 使用
不同于 full path 的 physical execution shape。Ascend 上不同 shape 会导致
logits 产生 bf16 量级差异；near-tie token 被 greedy 放大成 token mismatch。

这不是 sampler 使用 full size 能解决的问题。sampler 只消费 forward 输出的
logits；如果 logits 已经来自 `68 + 64` 数值路径，它无法恢复 full `[132]` 或
full graph `[256]` 路径下的 bit-exact logits。

### 2026-05-12 eager layer diff 复核

为了验证差异是否真的来自 forward 本身，而不是 ACL graph replay、sampler 或
lm_head，本次又做了一轮 eager 模式层级对比：

- disabled：
  `--run disabled --enforce-eager --max-tokens 2`
- enabled：
  `--run enabled --enforce-eager --max-tokens 2`
- 环境：
  `VLLM_ASCEND_LAYER_DIFF_DUMP=1`
  `VLLM_ASCEND_SPLIT_PARALLEL_FORCE_EAGER=1`
  `VLLM_ASCEND_SPLIT_EXACT_SHAPE=0`
  `VLLM_ASCEND_SPLIT_GPU_LIKE_CONTEXT=1`

结果文件：

- disabled dump：`/tmp/layer_diff_disabled_eager_2tok.jsonl`
- enabled dump：`/tmp/layer_diff_enabled_eager_2tok.jsonl`

forward 结构如下：

- disabled `forward_id=3`：decode full batch，`num_tokens=132`
- enabled `forward_id=3`：decode split0，`num_tokens=68`
- enabled `forward_id=4`：decode split1，`num_tokens=64`

对同一批 global rows `0..67`，直接比较：

- disabled `forward_id=3`
- enabled `forward_id=3`

结论：

- layer `0` 到 `3` 的 sampled hidden states 完全一致。
- 第一个可观测差异出现在 layer `4` 输出。
- layer `4` 的首个 sampled max diff 是 `0.001953125`。
- 差异随后逐层放大，layer `22` 到 `27` 已增长到 `0.125`、`0.25`，最终
  layer `27` sampled max diff 达到 `1.5`。

这条证据非常关键，因为它说明：

1. 差异在 transformer 中间层就已经出现，不是只发生在 lm_head。
2. 这轮验证运行在 eager 模式，`cudagraph` 已关闭，因此 ACL graph replay
   不是必要条件。
3. split0 自己就会对比 full path 产生层内差异，因此 split1 的并发、第二路 graph
   地址绑定、secondary stream replay 都不是这个差异的必要触发条件。

这会把候选根因进一步缩小到两类：

- 最强解释：不同 execution shape (`132` vs `68`) 改变了 Ascend eager forward
  的底层数值路径。
- 次强解释：split lane0 的某个 metadata/build contract 仍和 full path 存在细微差异，
  这种差异在 eager forward 第 `4` 层开始被放大。

结合 `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1` 能恢复完全正确，以及 ACL graph/地址
校验路径已经单独验证过，本报告仍然把“execution shape 改变 forward 数值路径”
视为当前最强结论。

### 2026-05-12 layer 4 子模块复核

为继续判断第一次偏移到底来自 layer 4 的前半段还是后半段，又追加了一轮更细的
eager dump，只记录 layer `4` 的两个子模块输出：

- `self_attn`
- `mlp`

使用环境：

```text
VLLM_ASCEND_LAYER_DIFF_DUMP=1
VLLM_ASCEND_LAYER_DIFF_TARGET_LAYERS=4
VLLM_ASCEND_LAYER_DIFF_SUBMODULES=self_attn,mlp
VLLM_ASCEND_SPLIT_PARALLEL_FORCE_EAGER=1
VLLM_ASCEND_SPLIT_EXACT_SHAPE=0
VLLM_ASCEND_SPLIT_GPU_LIKE_CONTEXT=1
```

结果文件：

- `/tmp/layer_diff_l4_disabled.jsonl`
- `/tmp/layer_diff_l4_enabled.jsonl`

比较对象仍然是 decode step：

- disabled `forward_id=3`，full batch `132`
- enabled `forward_id=3`，split0 `68`

对同一批 global rows `0..67` 的 sampled 数据比较结果：

- layer4 `self_attn` output：`max_diff = 0.001953125`
- layer4 `mlp` output：`max_diff = 0.001953125`
- layer4 final output：`max_diff = 0.001953125`

关键结论：

1. 第一次偏移已经出现在 layer 4 的 `self_attn` 输出。
2. 因此“layer 4 前半段完全一致，只是后半段 MLP 开始分叉”的假设可以排除。
3. 当前最早可观测边界已经收缩到：
   - `input_layernorm + residual` 到 `self_attn` 之间
   - 或 `self_attn` 内部本身

也就是说，第一次偏移发生在 attention half，而不是 layer 内后半段。

需要注意一个边界条件：

- 当前 layer hook 记录的是 layer 输出里的第一个 tensor，也就是 `hidden_states`；
  没有单独记录 `residual`。
- 当前子模块 hook 记录的是 `self_attn` 输出，而不是 `input_layernorm` 输出。

所以这轮复核还不能把根因最终缩到“attention core 数值路径”本身；它只能证明：

```text
第一次偏移不晚于 layer4.self_attn output，
并且一定早于 layer4.mlp output。
```

如果还要再往里收敛，下一步应只看 layer 4 的：

- `input_layernorm` 输出
- `self_attn` 输出

如果 `input_layernorm` 已不同，说明 residual/path contract 仍可能参与；
如果 `input_layernorm` 仍一致而 `self_attn` 开始不同，说明 attention kernel /
attention metadata 数值路径就是第一触发点。

### 2026-05-12 input_layernorm / self_attn 复核

按上面的下一步建议，又做了一轮更窄的 eager 对比，只记录 layer `4` 的：

- `input_layernorm`
- `self_attn`

结果文件：

- `/tmp/layer_diff_l4_inln_disabled.jsonl`
- `/tmp/layer_diff_l4_inln_enabled.jsonl`

比较对象仍然是 decode step：

- disabled `forward_id=3`，full batch `132`
- enabled `forward_id=3`，split0 `68`

比较结果：

- `input_layernorm` output：`max_diff = 0.0`
- `self_attn` output：`max_diff = 0.001953125`
- layer4 final output：`max_diff = 0.001953125`

这轮复核把边界继续收紧为：

1. split0 和 full path 在进入 layer4 attention 前，`input_layernorm` 输出仍然一致。
2. 第一个可观测差异从 layer4 `self_attn` 输出开始出现。
3. 因此 residual 合并、`input_layernorm`、以及 layer4 之前的 hidden-state sampled
   路径，不是第一次偏移的触发点。

当前最强定位可以写成：

```text
第一次可观测分叉点位于 layer4 attention path 内部。
```

这里的 “attention path 内部” 还包含两种可能：

- `self_attn` 内部算子/tiling/累加路径随 shape 改变
- `self_attn` 消费的 attention metadata 在 full 与 split 之间仍有未发现的细微差异

但至少可以明确排除：

- sampler
- lm_head
- ACL graph replay
- split1 并发
- layer4 MLP
- layer4 input_layernorm / residual pre-attn 路径

### 继续下钻：attention 内部边界

基于 `/tmp/attn_path_disabled.jsonl` 与 `/tmp/attn_path_enabled.jsonl` 的
`forward_id = 3`（第一轮真实 decode，full batch 132 vs split0 batch 68）,
可以把 layer4 attention 内部边界继续收紧。

#### 1. qkv_proj 输出一致

`self_attn.qkv_proj` 的 sampled 输出在 full 与 split0 上对齐。例如第一行前 8 个值
都为：

```text
[-0.11572265625, 0.296875, -0.15234375, 0.251953125,
 -0.154296875, 0.2041015625, 0.15625, 0.19921875]
```

这说明进入 RoPE 之前，attention 主输入张量没有先分叉。

#### 2. attention metadata 前缀对齐，但 shape 与 RoPE context 来源不同

同一轮 `forward_id = 3` 下：

- full path:
  - `query_start_loc_shape = [133]`
  - `seq_lens_shape = [132]`
  - `slot_mapping_shape = [132]`
  - `forward_context.cos/sin = null`
- split0:
  - `query_start_loc_shape = [69]`
  - `seq_lens_shape = [68]`
  - `slot_mapping_shape = [68]`
  - `forward_context.cos/sin` 非空，`cos_shape = sin_shape = [1, 68, 1, 128]`

更重要的是，三类 metadata 的前缀值一致：

- `query_start_loc` 前缀都是 `[0,1,2,3,4,5,6,7]`
- `seq_lens` 前缀都是 `[6,6,13,12,8,11,9,11]`
- `slot_mapping` 前缀也一致

这说明 split0 没有出现明显的 request/token 对齐错误；差异不在这些字段的值本身，
而在于 **full 与 split 使用了不同的 RoPE context 来源**。

#### 3. rotary_emb 是第一个高风险点

从代码看，`vllm_ascend/ops/rotary_embedding.py` 的 NPU fast path 在
`head_size == 128` 且 `is_neox_style == True` 时，会走：

- `forward_context.dbo_enabled == True`:
  `torch_npu.npu_apply_rotary_pos_emb(query, key, forward_context.cos, forward_context.sin)`
- 否则：
  `torch_npu.npu_apply_rotary_pos_emb(query, key, global_cos, global_sin)`

而 `vllm_ascend/ascend_forward_context.py` 里 split-local context 的构造会显式：

- `new_forward_context.dbo_enabled = True`
- 基于 split-local `positions` 构造并 `clone()` 出 `forward_context.cos/sin`

结合上面的 trace：

- full path 的 `forward_context.cos/sin` 为 `null`
- split0 的 `forward_context.cos/sin` 非空且 shape 为 `[1, 68, 1, 128]`

可以得到一个高置信度判断：

```text
full 与 split0 在 rotary_emb 处很可能走了不同的 cos/sin 输入源分支。
```

这里的“不同”不一定意味着数学上错误，更可能是：

- full 走全局 cos/sin slice
- split0 走 local cloned cos/sin
- 两者喂给 `npu_apply_rotary_pos_emb` 的 tensor shape / layout / storage 来源不同
- 导致 RoPE 后的 bf16 数值路径开始出现微小偏移

#### 4. attn core 只有极小差异，self_attn 输出开始放大

`self_attn.attn` 输出 sampled 值在 full 与 split0 上只出现了很小偏移。比如第二行
第 4 个值：

- full: `0.0032196044921875`
- split0: `0.003204345703125`

这说明 attention core 本身不是“第一处大偏移”，更像是在消费已经轻微偏移过的
Q/K 后继续传播。

而 `self_attn` 最终输出的 sampled 值偏移已经明显大于 `attn` core。例如第二行
第一列：

- full: `0.01446533203125`
- split0: `0.01470947265625`

也就是说：

```text
qkv_proj 前一致
-> rotary_emb / RoPE 分支处开始引入微小偏移
-> attn core 保持微小偏移
-> o_proj / self_attn 输出把偏移放大到 1e-4 ~ 1e-3 量级
```

### 当前判断强度

到这一步，可以把“第一次偏移”的候选按强弱排序为：

1. **RoPE 输入源分支差异**  
   full 使用全局 cos/sin，split0 使用 split-local `forward_context.cos/sin`。
2. **同一 NPU RoPE/attention kernel 在不同 execution shape 下的数值路径差异**  
   即使数学输入等价，`68` 与 `132` 的物理 shape 仍可能改变 tile/累加顺序。
3. attention metadata 隐式合约问题  
   目前证据最弱，因为 `query_start_loc / seq_lens / slot_mapping` 的值前缀是对齐的。

因此，当前比“纯 batch shape 数值路径差异”更具体的根因候选是：

```text
split path 在 layer4 attention 里不仅改变了 batch shape，
还把 RoPE 的 cos/sin 输入源从 full path 的 global slice
切换成了 split-local forward_context.cos/sin。
第一次可观测偏移很可能就从这个分支切换开始。
```

### 还差的最后一锤

为了把这个候选从“高置信度”提升到“完全锁定”，还需要一次新的 trace，补上最新
hook 已经添加但当前文件里没有的字段：

- `self_attn.rotary_emb` 的 `input_tensors`
- `attn_metadata.forward_context.dbo_enabled`

如果重跑后看到：

- `rotary_emb` 输入 Q/K 一致
- full `dbo_enabled = False`
- split0 `dbo_enabled = True`
- 输出从 `rotary_emb` 开始产生偏移

就可以把 RoPE 分支差异作为第一次偏移的直接触发点锁定下来。

## 最新验证：RoPE 分支差异已被运行时 trace 证实

本节基于重新生成的 trace：

- `/tmp/attn_path4_disabled.jsonl`
- `/tmp/attn_path4_enabled.jsonl`

这次 trace 使用了最新 hook 代码，已经包含：

- `input_tensors`
- `attn_metadata.forward_context.dbo_enabled`

关注对象仍然是 `forward_id = 3`，也就是第一轮真实 decode，对比：

- full path: batch shape `132`
- split0 path: batch shape `68`

### 1. qkv_proj 输入输出一致

对 `self_attn.qkv_proj`：

- full `dbo_enabled = False`
- split0 `dbo_enabled = True`
- 但 `input_tensors[0]`（即进入 `qkv_proj` 的 hidden states）sample 完全一致
- `qkv_proj` 输出 sample 也完全一致

例如第一行前 8 个值，full 与 split0 都是：

```text
[-0.11572265625, 0.296875, -0.15234375, 0.251953125,
 -0.154296875, 0.2041015625, 0.15625, 0.19921875]
```

这一步把“进入 attention 主干前 hidden states 已经不同”的可能性排除了。

### 2. rotary_emb 输入一致，但运行分支不同

对 `self_attn.rotary_emb` 的同一组记录：

- full:
  - `dbo_enabled = False`
  - `forward_context.cos_shape = None`
  - `forward_context.sin_shape = None`
- split0:
  - `dbo_enabled = True`
  - `forward_context.cos_shape = [1, 68, 1, 128]`
  - `forward_context.sin_shape = [1, 68, 1, 128]`

同时，`rotary_emb` 的三个输入 sample 在 full 与 split0 的前几条记录上是对齐的：

- input `0`: positions / index tensor
- input `1`: query
- input `2`: key

例如第一条 `rotary_emb` 记录：

- input `0` 前缀都为 `[5, 5, 12, 11, 7, 10, 8, 10]`
- input `1` 前 8 个值一致
- input `2` 前 8 个值一致

这说明：

```text
进入 RoPE 前，positions / q / k 这三个直接输入没有先分叉。
```

### 3. 第一次可观测偏移首先出现在 rotary_emb

在 `forward_id = 3` 的 `rotary_emb` 记录序列中：

- 前几条记录 full 与 split0 sample 完全一致
- 第一个 sample 级非零差异出现在第 5 条相关 `rotary_emb` 记录

该条记录的 sample 第 2 行前 8 个值：

```text
disabled:
[-2.21875, -1.4453125, 0.109375, 0.81640625,
 -5.15625, 0.474609375, -0.1884765625, -0.208984375]

enabled:
[-2.21875, -1.4375, 0.1083984375, 0.8125,
 -5.15625, 0.466796875, -0.193359375, -0.212890625]
```

也就是说，在我们能看到的 attention 子模块边界上：

```text
第一次真实偏移先出现在 rotary_emb 输出，
不是 qkv_proj，不是 attn metadata 前缀字段。
```

### 4. attn 与 self_attn 只是后续传播和放大

随后，同一 `forward_id = 3` 下：

- `self_attn.attn` 的 sample 首次出现差异，最大 sample 偏移为
  `2.6702880859375e-05`
- `self_attn` 最终输出的 sample 最大偏移扩大到 `0.001953125`

对应 `self_attn.attn` 的一行差异：

```text
disabled:
[0.0517578125, -0.057373046875, -0.158203125, 0.0032196044921875,
 0.12060546875, 0.00063323974609375, 0.02783203125, -0.17578125]

enabled:
[0.0517578125, -0.057373046875, -0.158203125, 0.003204345703125,
 0.12060546875, 0.000606536865234375, 0.02783203125, -0.17578125]
```

对应 `self_attn` 的一行差异：

```text
disabled:
[0.01446533203125, -0.001983642578125, -0.09326171875, 0.27734375,
 0.1552734375, -0.0810546875, -0.1162109375, -0.0269775390625]

enabled:
[0.01470947265625, -0.0023345947265625, -0.09326171875, 0.279296875,
 0.1552734375, -0.08154296875, -0.1162109375, -0.027099609375]
```

这和前面已有判断完全一致：

```text
RoPE 先引入微小偏移
-> attention core 传播微小偏移
-> o_proj / self_attn 输出把偏移继续放大
```

## 更新后的结论强度

现在可以把根因表述从“高置信度候选”提升为“已验证链路”：

```text
full 与 split0 在 layer4 attention 中的第一次可观测偏移，
先出现在 rotary_emb 输出。

在这一步之前：
- qkv_proj 输入输出一致
- positions / q / k 输入一致
- query_start_loc / seq_lens / slot_mapping 前缀一致

而运行时唯一明确不同的执行条件是：
- full: dbo_enabled=False，RoPE 不使用 split-local forward_context.cos/sin
- split0: dbo_enabled=True，RoPE 使用 split-local forward_context.cos/sin
```

因此，当前最直接、最具体的根因是：

```text
split path 在 RoPE 处切换到了 DBO/local-cos-sin 分支，
这一步先产生了微小数值偏移；后续 attention 与 o_proj 继续传播并放大该偏移。
```

需要强调的是，这个结论并不自动说明“local cos/sin 数值错了”。
更精确的说法是：

- full 与 split 在 RoPE 处走了不同执行分支
- 该分支切换是第一次已验证的偏移触发点
- 偏移可能来自 local/global cos-sin source、layout、clone 后 storage、
  或底层 kernel 在该分支下的实现差异

但无论细分是哪一种，**第一次偏移已经被锁定在 rotary_emb 这一步**。

### 可行修复方向

如果要求 `split_batch` 与 full path 的 token 输出严格一致，可行方案只有几类：

#### 方案 1：Exact-shape correctness mode

保持当前恢复后的逻辑：当 `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1` 时，即使有
`cudagraph_capture_sizes`，每个 split 也 padding 到 full path execution shape。

优点：

- 已被验证能恢复 token exact。
- 不需要改 sampler。
- 不依赖 top1/top2 margin。
- 保留 split 代码路径、metadata 构造、双流执行和 merge。

缺点：

- 不能使用最小 graph；两个 split 都会按 full shape 跑。
- 性能收益主要来自保留双流结构，不来自减少单路 graph shape。

这是当前唯一已经验证有效的严格 correctness 解法。

#### 方案 2：Near-tie fallback recompute

默认使用最小 graph `68 + 64` 跑 split；采样前检查每个 request 的 top1/top2
margin。如果 margin 小于阈值，例如 `0.25` 或 `0.5`，只对这些高风险 step
触发 full-shape 纠偏。

纠偏不能只改 sampler tie-break。必须重新获得 full-shape 数值路径下的 logits。
否则 sampler 仍然只能基于 split-shape logits 做决定。

可能实现：

- 快路径：最小 graph split forward。
- 检测：对 logits 做 top2 margin 检查。
- 慢路径：当存在 near-tie row 时，用 full execution shape 重新 forward 当前
  decode step，然后用 full-shape logits 采样。

优点：

- 大多数 step 仍可走最小 graph。
- 只有 near-tie 时才付出 full-shape 代价。

缺点：

- 实现复杂，必须处理 KV cache 写入幂等性或使用 scratch KV，避免同一 decode
  step 被重复写入造成副作用。
- 如果一个 batch 中任意 row near-tie，为了匹配 full path，最稳妥的纠偏通常
  是重跑整个 full execution shape，而不是只重跑单行。
- 需要保存/恢复当前 step 的 KV、slot mapping 或确保重复写同一 slot 是严格安全的。

这个方案可以在性能和 correctness 之间折中，但需要单独设计和验证。

#### 方案 3：让最小 graph 内部使用 full-shape 等价 kernel

理论上可以保留外部 graph descriptor 为 `68/64`，但让 logits 相关的关键 kernel
内部按 full shape 的 tile/accumulate 规则执行，例如强制 matmul/attention/lm_head
使用与 full graph 相同的 tiling 和累加顺序。

优点：

- 理想情况下同时获得最小 graph 和 bit-exact。

缺点：

- 需要 Ascend kernel/算子层支持 shape-invariant deterministic path。
- Python 层很难保证，因为差异来自底层 bf16 kernel tile、workspace 或累加顺序。
- 即使只修 lm_head，也不一定够；hidden states 在前面层可能已经分叉。

这是长期方向，不是当前最小改动。

#### 方案 4：稳定 tie-break policy

在 top1/top2 margin 很小时，不直接用当前 logits top1，而是使用固定规则，例如
按 token id 或历史 full-path 偏好选择。

优点：

- 实现成本低。
- 可以减少 near-tie 翻转。

缺点：

- 不能保证匹配现有 full path。它只是定义一个新的 deterministic policy。
- full path 和 split path 都必须使用同一新 policy，测试基准也要随之改变。

如果目标是“匹配当前 full path 输出”，这个方案不成立。

### 推荐

短期推荐保留两个模式：

- correctness/debug：`VLLM_ASCEND_SPLIT_EXACT_SHAPE=1`，严格 token exact。
- performance：默认最小 graph split，接受 logits tolerance 或 token 非严格一致。

中期如果必须同时要“多数时候最小 graph”和“最终 token exact”，实现
near-tie fallback recompute。这个方案的核心不是采样时用 full size，而是在采样前
确保 near-tie row 的 logits 来自 full-shape 等价 forward。

## GPU 为什么没有同类问题

对照 `/vllm-workspace/gpu.py` 和 `/vllm-workspace/cudagraph.py` 后，可以看到
GPU 路径做了两类事情。

第一类是状态正确性保护：

- dual graph runtime 明确区分 `StreamSlot.PRIMARY` 和 `StreamSlot.SECONDARY`。
- primary/secondary 分别使用独立的 graph entry 字典和 graph pool。
- secondary split 的输入被拷贝到 `micro_*` buffer，再用 secondary graph replay。
- 两条 stream replay 前都会 `wait_stream(default_stream)`，保证 default stream
  上的 buffer 填充对 graph stream 可见。
- replay 时强制校验 input tensor 地址必须与 capture 时一致。
- 还记录并比较 attention metadata tensor 地址，避免 graph 读到 capture 地址而
  runtime metadata 指向另一套 tensor。
- dual graph runtime context 使用 `ubatch_slices=None`，依赖已经 split 好的
  per-lane metadata 和 per-slot buffer。

这些机制主要防止输入地址、metadata、stream 同步和 graph pool 混用错误。
NPU 当前实现已经部分对齐：有两套 ACL graph entry、两套 graph pool、第二路
micro buffer、`in_parallel_streams`、以及 DEBUG 下的 input address check。
仍可补齐的工程差异包括：

- secondary stream replay 前显式 `parallel_stream1.wait_stream(default_stream)`。
- 把 ACL graph input address check 从 DEBUG 诊断提升为可配置 hard check。
- 增加 attention metadata tensor 地址 capture/replay 校验。
- 尽量让 dual graph runtime context 与 GPU 一样使用 split-local metadata +
  `ubatch_slices=None`，只单独处理 RoPE cos/sin。

但这些只保证状态路径正确，不保证数值 bit-exact。

第二类是数值路径行为：

GPU 能在最小图下 exact，说明在该 GPU 组合中，decode forward 的 row 级输出
对 batch M 维变化足够稳定，或者至少不会跨过 greedy top1/top2 边界。CUDA graph
本身没有提供“不同 batch shape bit-exact”的保证；它只是 replay capture 的同一
shape graph。

NPU 的实验证据不同：

- split 使用 `68 + 64` 最小 execution shape 时 mismatch。
- 用户恢复 `exact_shape_num_tokens` 覆盖 ACL capture sizes 后，split 仍走同一套
  split/ACL/parallel/merge 代码，但 execution shape 与 full path 对齐，结果完全正确。

这说明 NPU 的主要差异来自 physical execution shape 触发的底层 kernel/tile/
workspace/累加顺序变化，而不是 GPU 已经防住的那些状态错误。

## 能否把 NPU 改成 GPU 一样“最小图且完全一致”

结论：可以继续把 NPU 的工程结构改得更像 GPU，但这不一定能解决 logits 差异。
要同时满足“最小图”和“完全一致”，需要让 NPU 最小 shape forward 产生 full-shape
等价 logits。Python 层可选方案如下。

### A. 先补齐 GPU 同款状态保护

建议做，但它更像健壮性修复，不是已验证的 logits 修复：

- 在 second lane replay 前加 `parallel_stream1.wait_stream(default_stream)`。
- ACL graph replay 默认校验 input addresses。
- 增加 attention metadata tensor 地址校验。
- 对齐 GPU 的 `ubatch_slices=None` runtime context 模式。

如果补齐后最小图仍 mismatch，则可彻底排除状态路径差异。

当前实现状态：

- `model_runner_v4.py`
  - 默认启用 secondary stream replay 前等待 default stream：
    `VLLM_ASCEND_SPLIT_SECONDARY_WAIT_DEFAULT_STREAM=1`。
  - 新增 GPU-like split context 实验开关：
    `VLLM_ASCEND_SPLIT_GPU_LIKE_CONTEXT=1`。开启后 split runtime context
    尽量使用 split-local metadata + `ubatch_slices=None`。
  - 新增 layer diff hook 诊断：
    `VLLM_ASCEND_LAYER_DIFF_DUMP=1`。
- `ascend_forward_context.py`
  - 当 `ubatch_slices=None` 且传入的是 split-local `positions` 时，也会基于这些
    local positions 重建 RoPE cos/sin，避免 GPU-like context 丢失 RoPE。
- `acl_graph.py`
  - `VLLM_ASCEND_ACLGRAPH_ADDR_CHECK=1` 时，ACL graph replay 会强校验 input
    tensor capture/replay 地址。
  - `VLLM_ASCEND_ACLGRAPH_ATTN_METADATA_ADDR_CHECK=1` 时，会记录并比较 attention
    metadata tensor 地址。

### B. 定位 logits 差异从哪一层开始

这是判断能否低成本修复的关键：

- 如果 hidden states 在进入 lm_head 前完全一致，只是 lm_head logits 不一致，
  可以只把 lm_head 做 full-shape 等价计算或高精度 deterministic 计算。
- 如果第一层或中间层 hidden states 已经不同，则必须修整整段 transformer
  forward 的数值路径；只改 sampler 或 lm_head 没用。

建议加一次层级 dump：

- split disabled full path 的 selected rows hidden state。
- split enabled minimal path 的同 rows hidden state。
- 分层比较 max/mean abs diff，找第一个出现差异的 layer。

已新增诊断开关：

```text
VLLM_ASCEND_LAYER_DIFF_DUMP=1
VLLM_ASCEND_LAYER_DIFF_FILE=/tmp/layer_diff_disabled.jsonl
VLLM_ASCEND_LAYER_DIFF_RUN_NAME=disabled
VLLM_ASCEND_LAYER_DIFF_MAX_LAYERS=40
VLLM_ASCEND_LAYER_DIFF_MAX_ROWS=4
VLLM_ASCEND_LAYER_DIFF_MAX_COLS=8
```

注意：ACL graph replay 不会执行 Python layer hook，因此 layer diff 应配合
`VLLM_ASCEND_SPLIT_PARALLEL_FORCE_EAGER=1` 或 eager 配置使用。否则只能记录
capture/warmup，不代表 replay runtime 的逐层输出。

### C. Near-tie fallback recompute

这是最现实的“多数时候最小图，同时 token exact”的方案：

1. 先用最小图 split forward。
2. 采样前检查 logits top1/top2 margin。
3. 只有存在 near-tie row 时，触发 full-shape forward 纠偏。
4. 用 full-shape logits 采样，并确保 KV cache 中保留 full-shape forward 的写入结果。

难点是 KV cache 副作用。decode forward 会写当前 step 的 KV；如果先跑 split
再跑 full，需要保证 full recompute 能覆盖同一 slot，或使用 scratch KV，或把
KV commit 延后到确认不需要 fallback 之后。

### D. 底层 deterministic/shape-invariant kernel

这是理论上的最佳方案，但需要 Ascend kernel/算子层支持：

- 对 attention、MLP、lm_head 等 logits 相关算子使用 shape-invariant tiling。
- 或用更高精度累加，减少 batch shape 引起的 bf16 rounding 分叉。
- 或提供 deterministic mode，保证同一 row 不随 batch M 维变化。

如果没有底层支持，仅靠 Python 选择 `68/64` graph 很难保证和 `132/256`
full graph bit-exact。

前一版报告把剩余 mismatch 直接归因于“不同 batch shape 的数值非
bit-exact”，这个判断原本证据不足。加入本次 eager layer diff 后，更严谨的
结论变为：

1. `68 + 64` 这类最小 graph split 会触发 logits 差异，并且差异经常落在
   `0.0625` 到 `0.125` 量级；当 top1/top2 margin 很小时，greedy token 会翻转。
2. `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1` 曾经让同一批测试完全通过，说明 split slice、
   merge 和大部分 metadata 路径并不是必然错位。
3. eager layer diff 已验证：在关闭 ACL graph 后，split0 (`68`) 对比 full (`132`)
   的同一批 rows，会在 transformer layer `4` 输出首次出现 hidden-state 差异，
   并在后续层逐步放大。
4. 这说明 sampler、lm_head、ACL graph replay、split1 并发路径都不是必要条件。
5. 但 GPU 平台的 split/cudagraph 实现可以在最小 graph 下做到完全一致，因此
   “只要 batch shape 不同就必然不一致”不是跨平台定律。
6. Ascend 当前剩余的未彻底排除项，主要只剩两类：shape 触发的底层数值路径差异，
   或 lane0 split-local metadata/build contract 的细微偏差。

因此，当前根因状态应标记为：

```text
强假设：Ascend 最小 graph split 改变 physical execution shape 后，
transformer forward 本身在第 4 层开始走出不同 hidden-state 数值路径，
最终导致 logits 不再 bit-exact。

未定案：这个差异究竟是不可避免的 shape 数值路径，还是 Ascend split lane0
metadata/build contract 中仍有一个尚未发现的细微不一致。
```

后续分析必须继续用 falsifiable test 区分这两类原因，不能把 near-tie logits
翻转本身当作根因。

## GPU 对照发现

用户提供的 `/vllm-workspace/gpu.py` 和 `/vllm-workspace/cudagraph.py`
显示，GPU 参考实现在 dual graph 路径上有几处 Ascend 需要逐项对齐验证的机制：

- GPU 使用显式 `StreamSlot.PRIMARY/SECONDARY`，并按 slot 选择独立 graph entry
  字典和 graph pool；Ascend 使用 `in_parallel_streams` boolean 做类似选择。
- GPU 在 primary 和 secondary stream replay 前都执行
  `stream.wait_stream(default_stream)`，保证 default stream 上的输入 buffer 填充
  对非默认 stream 可见。
- GPU replay 强制校验输入 tensor 地址必须与 capture 时一致；Ascend 目前只有在
  `VLLM_LOGGING_LEVEL=DEBUG` 时才检查输入地址。
- GPU 还记录并比较前两层 attention metadata tensor 地址；Ascend 当前没有同等
  强度的 metadata 地址校验。
- GPU dual graph 的 runtime context 使用 `ubatch_slices=None`，但 capture 时按
  `stream_slot` 使用不同 buffer 和不同 metadata builder；Ascend 运行时使用
  split-local `UBatchSlice(slice(0, n), slice(0, n))` 重建 context。
- GPU 第二路输入显式拷贝到 `micro_*` buffer 后用 secondary graph replay；
  Ascend 也在 `_prepare_inputs` 中预拷贝到 `micro_*`，但当前二路 replay 前没有
  看到等价的 `parallel_stream1.wait_stream(default_stream)`。

这些差异不一定都是 bug，但它们足以说明：GPU exact 不能直接反推 Ascend 的
mismatch 只能由数学 shape 差异造成。

## 当前待验证根因候选

### H1: Ascend shape 数值路径差异

现有证据支持该假设：exact-shape split 曾经 PASS，而最小 graph `68 + 64`
FAIL；logits 首个分叉点多为 top1/top2 near-tie。

反证条件：如果在 Ascend 最小 graph 下补齐 stream wait、地址校验、metadata
地址绑定后可以完全一致，则 H1 不是根因，或者最多只是症状放大因素。

### H2: secondary stream 缺少 default stream wait

Ascend 二路 replay 使用 `torch.npu.Stream(device=self.device)`，但当前代码中
进入 `parallel_stream1` 后没有显式等待 default stream。第二路输入和 metadata
buffer 的填充发生在 default stream 或前序 stream 上，如果 NPU 非默认 stream
不隐式同步，则 secondary graph 可能读到旧数据或未完全可见的数据。

验证方法：只加诊断/实验性开关，在 `with torch.npu.stream(parallel_stream1):`
前或内部执行 `parallel_stream1.wait_stream(default_stream)`，比较 `68 + 64`
最小 graph token exact 是否恢复。

### H3: ACL graph replay 地址绑定不如 GPU 严格

GPU 无条件 assert replay input addresses 与 capture 一致，并额外比较 attention
metadata tensor 地址。Ascend 当前只在 DEBUG 下检查输入地址，且没有完整 metadata
地址校验。若某个 graph 实际读取的是 capture 时的静态 buffer，而 runtime
metadata 对象指向了另一个 tensor，日志中的值对齐不一定代表 graph 读取地址对齐。

验证方法：把 Ascend 的 input address check 和 attention metadata address check
临时提升为 always-on 诊断，分别记录 primary/secondary、64/68/128 graph 的
capture/replay 地址。

### H4: split-local context 与 GPU `ubatch_slices=None` 语义不一致

GPU dual graph runtime context 传 `ubatch_slices=None`，依赖已经构造好的
per-slot metadata；Ascend 当前为了 RoPE 和 local metadata 使用重建的
`UBatchSlice(slice(0, n), slice(0, n))`。这可能影响 RoPE、DBO 标志、
`num_tokens`/`padded_num_tokens`、或某些 layer 通过 forward context 读取的
shape 语义。

验证方法：增加只用于实验的模式，使 Ascend dual graph runtime context 更接近
GPU：metadata 已经 split-local 时，尽量传 `ubatch_slices=None`，并单独处理
RoPE cos/sin 切片，观察最小 graph 是否恢复 exact。

## 证据来源

本报告基于以下测试结果：

- 结果目录：
  `/vllm-workspace/split_batch_correctness_results/20260511_024549`
- 关键文件：
  - `diff.json`
  - `logits_debug_disabled.jsonl`
  - `logits_debug_enabled.jsonl`
  - `logits_debug_report.json`
- 本次输出：
  - `disabled_rows = 139`
  - `enabled_rows = 139`
  - `common_rows = 139`
  - `divergence_total = 44`
  - `diff.json` mismatch 总数为 `13`

本次 trace 覆盖了 9 个 mismatch prompt 的首个采样分叉点：

| prompt index | first flip step | disabled token | enabled token | disabled margin | enabled margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 15344 | 9625 | 0.25 | 0.0 |
| 11 | 5 | 11483 | 829 | 0.125 | 0.0 |
| 18 | 5 | 315 | 10525 | 0.0 | 0.125 |
| 65 | 1 | 448 | 429 | 0.125 | 0.0 |
| 72 | 4 | 429 | 304 | 0.125 | 0.0 |
| 92 | 4 | 63505 | 9237 | 0.125 | 0.0 |
| 117 | 7 | 4396 | 829 | 0.125 | 0.0 |
| 122 | 8 | 5461 | 8109 | 0.0 | 0.125 |
| 131 | 5 | 911 | 13 | 0.125 | 0.0 |

这些首个分叉点的共同特征是：

- top1/top2 margin 全部非常小，范围是 `0.0` 到 `0.25`。
- disabled/enabled 的 top-k 候选集合几乎相同。
- 分叉点通常只是 top1/top2 互换，而不是完全不同的 logits 分布。

例如 prompt `1` 的 first flip 发生在 decode step `5`：

```text
disabled:
  token 15344 = 16.75
  token  9625 = 16.50
  margin      = 0.25

enabled:
  token  9625 = 16.625
  token 15344 = 16.625
  margin      = 0.0
```

这里 split enabled 只引入了 `+0.125/-0.125` 量级的 logits 变化，但因为
原始 margin 很小，greedy top1 被翻转。后续 step 中 disabled/enabled 的
top-k 开始大幅不同，是因为前一步生成 token 已经不同。

再看 prompt `65`：

```text
decode step 1:

disabled:
  token 448 = 19.375
  token 429 = 19.25
  margin    = 0.125

enabled:
  token 429 = 19.25
  token 448 = 19.25
  margin    = 0.0
```

同样是 near-tie，被 `0.125` 量级差异触发了 top1 翻转。

## 因果链

1. Full path 用一个 full batch 执行 decode forward。
2. Split path 把同一个 logical batch 拆成两个 physical batch，例如
   `[68] + [64]`。
3. 对每个 request 来说，输入 token、position、slot mapping、seq lens 和
   merge 顺序都能对齐。
4. 但 NPU 上不同 batch shape 的 bf16 kernel/tile/累加路径不保证 bit-exact。
5. 因此 split logits 和 full logits 出现 `0.0625` 或 `0.125` 量级差异。
6. 当 full path 的 top1/top2 margin 很小，split 的小差异足以交换 top1/top2。
7. Greedy decoding 选择不同 token。
8. 后续 decode 基于不同 token 历史继续生成，差异快速放大。

这个链路解释了为什么很多 mismatch 不是第一个 token 就错，而是在第 5 到第 9
个生成 token 才分叉；也解释了为什么分叉后的 logits 分布会完全不同。

## 已排除项

以下方向已经被测试或日志证据排除为主因：

- `dummy_run` 中的 `split_batch_split`：用户注释后测试结果不变。
- replay 前更新和 `update_attn_params` refresh：用户注释后测试结果不变。
- ACL graph replay/capture：关闭 ACL graph 后仍有 `13/132` mismatch。
- parallel stream 调度：关闭 parallel streams 后 split 串行路径仍有 mismatch。
- merge 顺序：日志显示按 split0 后 split1 合并，token 顺序对齐。
- slot mapping 覆盖：日志中 `first_sm_eq=True`，split0 metadata 和主 buffer 对齐。
- block table refresh：日志显示 graph/meta block table ptr 一致，refresh 不是必要条件。
- `67 -> 68` padding 缺失：padding 修复后仍有剩余 mismatch。
- padding 到 `128` 的 fake row KV 写入：修过 `num_actual_tokens` 语义后仍有 mismatch。

这些排除项说明，当前问题不再像是某个 metadata 字段错位，而是 split 改变
physical execution shape 后产生的数值非一致性。

## 为什么 strict token exact 会失败

`temperature=0` 只保证采样过程确定，不保证不同数值路径的 logits 完全一致。
如果两个 token 的 logits 非常接近，任何 `0.0625` 或 `0.125` 的 bf16 级别差异
都可能改变 greedy top1。

因此，对于当前 split 方案：

- logits 大部分 step 是接近的。
- token exact 在 near-tie 请求上不稳定。
- 一旦某一步 token 不同，后续 output 文本不同是必然的。

这不是随机采样问题，也不是 prompt/output 对齐问题，而是 deterministic greedy
对 near-tie logits 极其敏感。

## 修改方案

### 方案 A：Exact-shape split 验证模式

目标：验证并尽量恢复 full path 的 bit-exact 行为。

核心做法：

- 保留 split 逻辑和 split merge。
- 但在 correctness/debug 模式下，不让每个 split 使用自己的最小执行 shape。
- 每个 split lane 都 padding 到 full batch 当前使用的 execution shape。
  - no ACL 时，full batch shape 可能是 `132`。
  - ACL graph 时，full batch `132` 可能被捕获/执行为 `256`。
- split0 执行 `[full_exec_shape]`，只取前 `split0_actual` 行。
- split1 也执行 `[full_exec_shape]`，只取前 `split1_actual` 行。
- fake rows 只参与计算 padding，不参与真实 KV 写入；`num_actual_tokens` 必须保持真实值。

预期：

- 如果 mismatch 消失，说明 batch shape 数值路径就是剩余根因。
- 如果仍 mismatch，再继续查 row index、metadata local/global 坐标或 fake row 语义。

优点：

- 最直接验证根因。
- 不绕过 split 路径，仍然经过 split slice、split metadata、split merge。
- 实现风险比改底层 kernel 小。

缺点：

- 性能会明显变差，因为每个 split 都按 full execution shape 跑。
- 更适合作为 correctness/debug mode，不适合作为默认高性能模式。

开关：

```text
VLLM_ASCEND_SPLIT_EXACT_SHAPE=1
```

已实现：

- `vllm_ascend/worker/ubatch_utils.py`
  - `split_batch_split(...)` 新增参数 `exact_shape_num_tokens`。
  - split planning 阶段新增 exact-shape padding 策略。
  - 当前实现已调整：
    当存在 `cudagraph_capture_sizes` 时，不再让
    `exact_shape_num_tokens` 覆盖 `SplitBatchSlice.padded_num_tokens`；
    split 仍然保留 per-split 的最小匹配 capture size，避免影响 ACL
    graph 选择。
  - `exact_shape_num_tokens` 现在只在无 graph capture size 可用时，
    才会统一 split execution shape。
  - 增加保护：如果 exact shape 小于 full padded tokens，或小于任一 split
    实际 token 数，则忽略该 override，回退原有 split padding 逻辑。
- `vllm_ascend/worker/model_runner_v4.py`
  - 新增环境开关解析：
    `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1`
  - 在 split planning 点把当前 full path 的 `num_input_tokens`
    传给 `split_batch_split(...)` 作为 exact padding target。
    - no ACL / eager：`num_input_tokens` 就是真实 full batch shape。
    - ACL graph：`num_input_tokens` 已经是 full path 当前实际使用的
      capture-aligned execution shape，例如 `256`。
  - 继续按 actual tokens trim output。
  - 当前实现仍保留此前对 fake rows 的约束：
    fake rows 只参与执行 shape padding，不应扩展真实 `num_actual_tokens`
    语义，也不应写入真实 KV。
  - two-graph 并行路径增加 micro buffer guard：
    如果 second split 的 exact shape 超过 `micro_batch_size`，直接报错，
    避免 silently overflow 第二路 micro buffer。

代码位置：

- `/vllm-workspace/vllm-ascend/vllm_ascend/worker/ubatch_utils.py`
- `/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v4.py`

### 方案 A 验证结果

实现后，使用与原失败场景一致的 prompt 集和 batch 配置进行了 ACL graph
验证：

- 验证目录：
  `/vllm-workspace/split_batch_correctness_results/exact_shape_validation_acl/20260511_030950`
- 关键输出：
  - `summary.json`
  - `console.log`
- 配置特征：
  - model: `Qwen/Qwen3-0.6B`
  - batch_size: `132`
  - max_tokens: `10`
  - split enabled
  - parallel streams enabled
  - compare-mode: `subprocess`
  - cudagraph capture sizes: `[64, 68, 128, 256, 512]`
  - env: `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1`

验证结果：

```text
PASS: 132/132 outputs match exactly
```

这次结果说明：

1. split slice / metadata / merge 路径本身可以保持正确。
2. 当两个 split lane 被强制到与 full path 一致的 physical execution
   shape 后，之前的 token mismatch 消失。
3. 这进一步确认剩余根因就是 batch shape 改变引入的数值路径差异，而不是
   metadata/KV/merge 错位。

### 方案 A 的使用方式

用于 correctness/debug 验证时，设置：

```text
VLLM_ASCEND_SPLIT_EXACT_SHAPE=1
```

推荐搭配：

- `compare-mode=subprocess`
  - 避免 inproc 连续初始化两个 engine 时的 NPU 内存碎片噪声。
- 复用同一组 prompts
  - 保证前后对比只变化 split execution shape。

已观察到：

- `compare-mode=inproc` 在当前环境中可能因为第二个 engine 初始化时
  KV cache 可用显存不足而失败。
- 该问题属于验证方式噪声，不影响方案 A 本身的 correctness 结论。

### 方案 B：Near-tie fallback

目标：保留 split 性能，同时降低 greedy 分叉概率。

核心做法：

- split forward 后计算 logits。
- 对 greedy 请求检查 `top1 - top2` margin。
- 如果某些请求 margin 小于阈值，例如 `0.25` 或 `0.5`，触发 fallback。

可选 fallback 粒度：

- batch 级 fallback：本 decode step 重新用 full batch forward 得到 logits 并采样。
- request 级 fallback：只对 near-tie request 做更高精度或 full-shape 重算。

风险：

- 如果之前 step 的 KV 已经由 split shape 写入，那么只在当前 step fallback 到 full
  logits 不一定能完全恢复 bit-exact，因为历史 KV 也可能已有微小差异。
- 要做到严格 exact，fallback 需要从一开始就维持 full-shape KV，或者使用方案 A。

因此，方案 B 更适合“减少分叉概率”的产品策略，不适合作为证明 strict correctness
的第一步。

### 方案 C：调整 correctness 测试判定

目标：让测试反映数值近似而不是 token bit-exact。

做法：

- 对 split/full 比较 logits top-k 和 margin。
- 如果 token 不同但 top1/top2 margin 小于阈值，并且候选集合一致、logit delta
  在 bf16 预期范围内，则标记为 numerical tie divergence，而不是 metadata failure。

优点：

- 能避免把 near-tie 数值差异误判为 split metadata bug。

缺点：

- 不能证明输出 token exact。
- 如果业务要求 split/full 文本完全一致，这个方案不够。

### 方案 D：底层数值一致性优化

目标：让不同 batch shape 的 kernel 也尽量 bit-exact。

可能方向：

- 强制相同 matmul/attention kernel 策略。
- 提高局部累加精度。
- 固定 tile/reduction order。

风险：

- 改动范围大。
- 可能影响全局性能。
- 很难保证所有模型、所有 shape 都 bit-exact。

该方案不建议作为当前第一修复方向。

## 推荐执行顺序

1. 先实现方案 A 的 `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1` 验证模式。
2. 用当前失败配置重跑：
   - no ACL
   - ACL graph
   - parallel streams on/off
3. 如果 exact-shape 模式通过，确认根因为 batch shape 数值差异。
4. 再决定生产策略：
   - strict correctness 优先：保留 exact-shape 模式作为可配置路径。
   - 性能优先：默认继续 per-split shape，但测试改为 logits/margin-aware。
   - 折中：加入 near-tie fallback，降低输出分叉概率。

## 当前建议

方案 A 已经完成，并且在 ACL graph 配置下验证通过。

当前更合理的下一步是把该模式从“验证开关”升级为明确的产品策略之一：

- 如果目标是 strict token exact：
  - 保留 `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1` 作为 correctness 优先模式。
- 如果目标是默认性能：
  - 继续使用 per-split 最小 shape，但接受 near-tie request 上的数值分叉，
    同时把 correctness 判定升级为 logits/margin-aware。
- 如果目标是折中：
  - 研究 near-tie fallback，但它更像性能/一致性折中策略，不是根因修复本身。

## 最小图且无额外开销避免 RoPE 偏移的可行性

基于最新 trace，第一次可观测偏移已经收敛到 `rotary_emb`：

- `qkv_proj` 输入一致。
- `qkv_proj` 输出一致。
- `rotary_emb` 的直接输入 `positions/q/k` 一致。
- full path 使用 `dbo_enabled=False`，RoPE 读取全局 `get_cos_and_sin_slice()`。
- split path 使用 `dbo_enabled=True`，RoPE 读取 `forward_context.cos/sin`。
- 偏移从 `rotary_emb` 输出开始出现。

因此，“最小图 + 不增加额外计算 + 避免 RoPE 偏移”只有一种可能成立的条件：

```text
当前偏移必须主要来自 RoPE 输入源/分支差异，
而不是来自 68/132 physical shape 本身触发的底层 kernel 数值路径。
```

如果这个条件成立，可以尝试做一个无额外执行开销的修复：让 full 和 split 在 RoPE
处使用同一种 cos/sin storage、layout 和 kernel 分支，同时仍保留 split graph
shape 为 `68 + 64`。

### 候选 1：split RoPE 使用全局 cos/sin 分支

实验目标：

```text
保持 split 最小 graph shape 不变，
但让 split0 的 RoPE 不走 forward_context.cos/sin clone 分支，
而是走与 full path 相同的 global cos/sin 分支。
```

可能做法：

- 在 split lane 调用 `self.model(...)` 前，基于该 lane 的 local positions 调用
  `update_cos_sin(local_positions)`。
- 在 `rotary_embedding.py` 中让该实验模式下的 GQA RoPE 忽略
  `forward_context.dbo_enabled`，直接使用 `get_cos_and_sin_slice()` 返回的全局
  cos/sin view。

这条路径的优点是没有额外 forward，也不需要 full-shape padding；甚至可以去掉
当前 split context 中的 `clone()` 开销。

风险：

- 全局 cos/sin buffer 是共享状态。若 lane0/lane1 真并发运行，lane1 更新全局
  cos/sin 可能覆盖 lane0 尚未消费完的值。
- 当前 two-graph 路径为了避免这个风险，已经把 lane1 context 延后创建，并且可
  通过 `parallel_stream1.wait_stream(default_stream)` 串住可见性；这会削弱并发。
- 如果要保留真实并发，就不能只有一套全局 cos/sin buffer。

因此这个候选适合作为第一验证实验，但不是最终并发形态。

### 候选 2：每个 stream slot 使用预分配 RoPE buffer

更接近最终形态的做法是：

- 保留 `68/64` 最小 graph。
- 为 primary/secondary stream slot 各预分配一套 RoPE cos/sin buffer。
- 每个 lane 只更新自己 slot 的 cos/sin buffer。
- `forward_context.cos/sin` 不再使用 `clone()` 临时张量，而是引用对应 slot 的
  预分配 buffer slice。

这样可以避免：

- full/global 与 split/local 的 storage/layout 差异；
- 每步 clone 分配；
- 两个 lane 覆盖同一套全局 cos/sin 的并发风险。

这条路径理论上可以做到“不比当前更贵”，因为当前 split path 已经有
`update_cos_sin + clone`；预分配 slot buffer 后，运行时至少可以去掉 `clone()`。

但它仍然不能保证 strict exact。原因是 `npu_apply_rotary_pos_emb` 看到的 shape
仍是 `68` 或 `64`，如果 NPU RoPE kernel 本身会因为第二维不同而选择不同实现，
那么只统一 storage/layout 仍不够。

### 候选 3：full path 也统一走同一 RoPE 分支

另一个实验是反过来：让 full path 也使用和 split 相同的 RoPE 分支，例如都通过
`forward_context.cos/sin` 进入 `npu_apply_rotary_pos_emb`。

这能回答一个关键问题：

```text
差异是 local/global 分支不一致导致的，
还是同一分支下 132 与 68 shape 仍会产生不同结果。
```

如果 full 和 split 都走同一 RoPE 分支后最小图通过，那么根因就是 RoPE 分支合约
不一致，可以继续做无额外开销修复。

如果仍然失败，说明即使 RoPE 分支统一，`68` 与 `132` 的 physical shape 仍然会
在 RoPE 或后续 attention/o_proj 中产生不可忽略偏移。此时 Python 层想同时满足
“最小图、零额外开销、strict token exact”基本不现实。

### 结论

当前不能直接承诺“保持最小图、不引入额外开销、同时 strict exact”一定可行。
可行性取决于下一轮实验：

- 如果统一 RoPE 分支后 PASS：可以做无额外执行开销修复，优先实现 stream-slot
  RoPE buffer，替代当前 `forward_context.cos/sin.clone()`。
- 如果统一 RoPE 分支后 FAIL：偏移不是单纯 RoPE source/layout 问题，而是最小
  physical shape 的数值路径问题。此时 strict exact 只能靠 exact-shape、
  fallback recompute，或底层 shape-invariant kernel；这些都会引入额外开销或
  需要算子层支持。

下一步最小验证应先做两个实验：

1. `split_min_graph + RoPE force global branch`。
2. `full_and_split + RoPE force same DBO/local branch`。

这两个实验都不需要先改采样器，也不需要引入 full recompute；它们能直接判断
RoPE 微弱偏移是否可以在最小图内无额外开销消除。

## 2026-05-12 RoPE 分支实验结果

为验证上面的假设，新增了三个只由环境变量控制的实验开关，默认关闭：

- `VLLM_ASCEND_ROPE_FORCE_GLOBAL=1`
  - 即使 `forward_context.dbo_enabled=True`，RoPE 也强制使用
    `get_cos_and_sin_slice()` 返回的 global cos/sin。
- `VLLM_ASCEND_ROPE_FORCE_CONTEXT=1`
  - RoPE 强制优先使用 `forward_context.cos/sin`。
- `VLLM_ASCEND_ROPE_CONTEXT_NO_CLONE=1`
  - 构造 context cos/sin 时不 clone，直接引用 global slice。

测试公共配置：

```text
model = Qwen/Qwen3-0.6B
batch_size = 132
max_tokens = 10
num_splits = 2
enable_parallel_streams = true
cudagraph_capture_sizes = [64, 68, 128, 256, 512]
VLLM_ASCEND_SPLIT_EXACT_SHAPE = 0
VLLM_ASCEND_SPLIT_GPU_LIKE_CONTEXT = 1
```

### 实验 1：split 强制 global RoPE

环境：

```text
VLLM_ASCEND_ROPE_FORCE_GLOBAL=1
VLLM_ASCEND_ROPE_FORCE_CONTEXT=0
VLLM_ASCEND_ROPE_CONTEXT_NO_CLONE=0
```

结果：

```text
FAIL: 14 mismatches
result_dir = /vllm-workspace/split_batch_correctness_results/20260512_131353
mismatch indices = [1, 3, 11, 18, 46, 48, 65, 72, 77, 92, 106, 117, 122, 131]
```

结论：

```text
只把 split RoPE 从 context cos/sin 分支切回 global cos/sin 分支，
不能恢复最小图 strict correctness。
```

这排除了“唯一原因是 split 使用 context 分支而 full 使用 global 分支”的简单解释。

### 实验 2：full/split 都强制 context RoPE，且使用 clone

环境：

```text
VLLM_ASCEND_ROPE_FORCE_GLOBAL=0
VLLM_ASCEND_ROPE_FORCE_CONTEXT=1
VLLM_ASCEND_ROPE_CONTEXT_NO_CLONE=0
```

结果：

```text
FAIL: 132 mismatches
result_dir = /vllm-workspace/split_batch_correctness_results/20260512_131806
```

现象：

- disabled full path 的输出本身也明显改变。
- enabled split path 也改变，但两者没有对齐。

结论：

```text
强行让 full path 使用 cloned context cos/sin 不是可用修复。
它不仅不能让 split 对齐 full，还会改变 full path 自身的数值行为。
```

这说明 context clone 可能改变 graph capture/replay 或 RoPE 输入合约，不适合作为
无额外开销修复方向。

### 实验 3：full/split 都强制 context RoPE，但不 clone

环境：

```text
VLLM_ASCEND_ROPE_FORCE_GLOBAL=0
VLLM_ASCEND_ROPE_FORCE_CONTEXT=1
VLLM_ASCEND_ROPE_CONTEXT_NO_CLONE=1
```

结果：

```text
FAIL: 14 mismatches
result_dir = /vllm-workspace/split_batch_correctness_results/20260512_132210
mismatch indices = [1, 3, 11, 18, 46, 48, 65, 72, 77, 92, 106, 117, 122, 131]
```

结论：

```text
关闭 clone 后 mismatch 集合回到和实验 1 一样。
因此 clone/storage 不是当前 14 个 mismatch 的主因。
```

### 更新后的判断

这三轮实验把 RoPE 分支假设进一步拆开后，得到更强结论：

1. 第一次可观测偏移仍然出现在 `rotary_emb` 输出。
2. 但单纯统一 RoPE 的 global/context 分支，不能消除最小图 mismatch。
3. 单纯去掉 context clone，也不能消除 mismatch。
4. `context clone` 反而会让 full path 自身改变，不能作为修复手段。

因此，当前最可能的根因已经从：

```text
RoPE local/global 输入源分支差异
```

收敛为：

```text
最小 physical shape 68/64 下的 RoPE 或其后续 attention/o_proj 数值路径，
与 full physical shape 132/256 不 bit-exact。
```

也就是说，RoPE 是第一次可观测偏移点，但不是因为 Python 层选择了错误的
cos/sin source；更可能是 NPU `npu_apply_rotary_pos_emb` 或后续 attention path
在不同 physical shape 下使用了不同 kernel/tile/layout 数值路径。

### 对“最小图、零额外开销、strict exact”的影响

在当前 Python 层可控范围内，这个组合基本不可保证：

- `exact-shape` 能过，是因为把 physical execution shape 对齐到了 full path。
- 最小图 `68 + 64` 只要保持 physical shape 不变，即使统一 RoPE 分支和 cos/sin
  storage，仍然产生同一批 mismatch。
- sampler、tie-break、lm_head 调整都不能恢复已经从 layer4 attention path 开始的
  hidden-state 偏移。

剩余可行方向只剩：

1. **strict correctness**：继续使用 `VLLM_ASCEND_SPLIT_EXACT_SHAPE=1`。
2. **性能优先**：最小图运行，但接受 near-tie token 分叉或改测试为 logits/margin-aware。
3. **折中**：near-tie full-shape recompute，但这会引入额外开销，并需要处理 KV cache
   写入副作用。
4. **底层修复**：让 Ascend RoPE/attention/o_proj kernel 提供 shape-invariant
   deterministic path；这是算子层问题，不是当前 Python split 逻辑能直接保证的。
