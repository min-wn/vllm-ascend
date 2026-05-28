# Inplace Parallel Rotary Failure Analysis

Date: 2026-05-25

## 结论摘要

本次 `inplace_parallel` 失败不是普通的 ACL graph replay 失败，也不是
attention metadata 更新之后的数值正确性问题。异常发生在 split-1 offset
graph 的 lazy capture warmup 阶段，位置是
`_run_inplace_serial_offset_capture()` 中把 `cudagraph_runtime_mode` 临时切到
`NONE` 后执行 eager model forward。

直接报错来自 `torch_npu.npu_apply_rotary_pos_emb`：

```text
op[ApplyRotaryPosEmb], all input dim1 must equal
RuntimeError: inplace parallel replay worker failed at slice_idx=1
```

根因判断是：`inplace_parallel` 试图用不同 `cos_sin_slot_id` 隔离两个 split
的 rotary cos/sin，但 torch compile 生成的 backbone graph 已经把 rotary
cos/sin 访问特化到了全局 slot0。split-1 运行时应使用 slot1 的短 cos/sin
切片，却实际走到了 slot0。当前 batch=400 的 split 形态是 `384 + 16`，因此
split-1 的 query/key dim1 是 16，而 slot0 仍对应 split-0 或全 batch 的长度，
NPU rotary op 检查 dim1 相等时直接失败。

`inplace_serial` 没有触发这个问题，不代表编译图对 rotary slot 的处理正确。
它是在当前执行时序下复用了 slot0，并且没有 split-0/split-1 并发写读同一个
全局 slot，因此规避了这次 dim1 mismatch。

历史 `inplace_parallel` 测试通过也不矛盾。历史 gate 主要使用
`Qwen/Qwen2.5-0.5B-Instruct`，该模型 `hidden_size=896`、
`num_attention_heads=14`，head_dim 为 64，不满足当前失败路径中
`head_size == 128` 的 `npu_apply_rotary_pos_emb` 快路径条件。本次失败使用
`Qwen/Qwen3-0.6B`，配置中 `head_dim=128`，正好进入这条对 cos/sin dim1
强约束的路径。

## 现象与影响

失败日志：

```text
/vllm-workspace/benchmark_split_results/inplace_input_sweep_20260525_134033/logs/inplace_parallel_mt16_bs400-448-496.log
```

同一组输入下的对比结果：

```text
aclgraph:
  bs=400  mt=16  seq=427  OK  steps=15(split=15)
  bs=448  mt=16  seq=209  OK  steps=15(split=15)
  bs=496  mt=16  seq=365  OK  steps=15(split=15)

inplace_serial:
  bs=400  mt=16  seq=427  OK  steps=15(split=15)
  bs=448  mt=16  seq=209  OK  steps=15(split=15)
  bs=496  mt=16  seq=365  OK  steps=15(split=15)

inplace_parallel:
  bs=400  mt=16  seq=427  FAIL  steps=0(split=0)
  bs=448  mt=16  seq=209  FAIL  steps=0(split=0)
```

perf 文件也支持这个判断：

```text
aclgraph_mt16_bs400-448-496.jsonl          15 lines
inplace_serial_mt16_bs400-448-496.jsonl    15 lines
inplace_parallel_mt16_bs400-448-496.jsonl   0 lines
```

也就是说 `inplace_parallel` 第一组 case 尚未进入稳定 decode 统计，split-1
lazy capture warmup 就失败了。后续 bs=448 的失败不能独立看待，因为首次异常
后同一个 engine 状态已经被破坏。

## 关键证据

### 1. 失败发生在 split-1 lazy capture warmup

调用栈显示 split-1 worker 进入：

```text
_run_split_batch_inplace_parallel()
  -> _run_inplace_parallel_worker(slice_idx=1)
    -> _run_inplace_serial_offset_capture(... parallel_streams=True)
      -> self.model(...)  # warmup, line 3165
```

对应代码位置：

```text
vllm_ascend/worker/model_runner_v3.py
  3160: with torch.npu.stream(target_stream):
  3161:     for warmup_idx in range(warmups):
  3162:         context.cudagraph_runtime_mode = CUDAGraphMode.NONE
  3163:         context.capturing = False
  3164:         with override_forward_context(context):
  3165:             _ = self.model(...)
```

这说明失败不是 ACL graph replay 已经完成后更新 graph task params 引起的，也不是
merge output 或 trim output 的问题。它在 offset graph 捕获前的 eager warmup
阶段就已经触发。

### 2. parallel 明确给 split-1 设置了 slot1

`_make_split_batch_metadata_inplace_parallel()` 中，split-0 使用 main stream，
split-1 使用 parallel stream，并把 `cos_sin_slot_id=i` 传入 context 构造：

```text
vllm_ascend/worker/model_runner_v3.py
  2948: ctx_stream = self.stream_parallel if in_parallel_streams else self.stream_main
  2951: split_forward_context = create_ascend_forward_context(...)
  2961: in_parallel_streams=in_parallel_streams,
  2962: cos_sin_slot_id=i,
```

`create_ascend_forward_context()` 会按该 slot 更新 cos/sin，并把 clone 后的
切片挂到 forward context：

```text
vllm_ascend/ascend_forward_context.py
  272: new_forward_context.dbo_enabled = True
  274: new_forward_context.cos_sin_slot_id = cos_sin_slot_id
  291: update_cos_sin(positions, slot_id=cos_sin_slot_id)
  292: cos_slice, sin_slice = get_cos_and_sin_slice(slot_id=cos_sin_slot_id)
  293: new_forward_context.cos = cos_slice.clone()
  294: new_forward_context.sin = sin_slice.clone()
```

按设计，split-1 应该使用 slot1 的 `context.cos/context.sin`，长度应等于
split-1 token 数。

### 3. rotary 实现同时存在 context cos/sin 和全局 slot cos/sin 两种路径

`_rope_forward_oot()` 先根据 forward context 读取 slot，再根据
`dbo_enabled` 选择参数来源：

```text
vllm_ascend/ops/rotary_embedding.py
  179: forward_context = get_forward_context()
  180: slot_id = forward_context.cos_sin_slot_id
  181: cos, sin = get_cos_and_sin_slice(slot_id=slot_id)
  182: if is_neox_style and self.head_size == 128 ...
  191:     if forward_context.dbo_enabled:
  192:         query, key = torch_npu.npu_apply_rotary_pos_emb(
  193:             query, key, forward_context.cos, forward_context.sin)
  194:     else:
  197:         query, key = torch_npu.npu_apply_rotary_pos_emb(
  198:             query, key, cos, sin)
```

当前 split context 中 `dbo_enabled=True`，理论上应该走 192-193 行。但失败栈和
编译图都指向 197 行附近，即非 DBO 的全局 `cos/sin` 路径。这是一个强信号：
torch compile 生成 backbone graph 时，已经在普通 decode context 下把分支和
全局变量访问特化了，后续 split context 的 `dbo_enabled=True` 和
`cos_sin_slot_id=1` 没有被可靠表达为运行时输入。

### 4. torch compile cache 中只有 slot0，没有 slot1

当前 cache 文件：

```text
/root/.cache/vllm/torch_compile_cache/68ccba701d/rank_0_0/backbone/computation_graph.py
```

函数签名中出现：

```text
G_import_vllm_ascend_dot_ops_dot_rotary_embedding_cos_slice_slots_0_
G_import_vllm_ascend_dot_ops_dot_rotary_embedding_sin_slice_slots_0_
```

每一层 rotary 调用都使用这两个 slot0 输入，例如：

```text
torch.ops.npu.npu_apply_rotary_pos_emb(
    view_56,
    view_57,
    g_import_vllm_ascend_dot_ops_dot_rotary_embedding_cos_slice_slots_0_,
    g_import_vllm_ascend_dot_ops_dot_rotary_embedding_sin_slice_slots_0_)
```

没有 `cos_slice_slots_1`，也没有 `forward_context.cos` /
`forward_context.sin` 作为 per-split runtime tensor input。这与
`inplace_parallel` 的 slot 隔离设计冲突。

## 执行链路复盘

### 普通准备阶段

`_prepare_inputs()` 会先对整个 batch 调用一次全局：

```text
update_cos_sin(positions)
```

默认 slot 是 0。因此在 split metadata 生成前，slot0 可能对应全 batch 的
positions 长度。

### inplace_parallel metadata

以 bs=400 为例，planner 选择 `384 + 16`：

1. split-0 context 构造时 `cos_sin_slot_id=0`，slot0 被更新为 384 token。
2. split-1 context 构造时 `cos_sin_slot_id=1`，slot1 被更新为 16 token。
3. slot0 仍保留 384 token 语义，slot1 才是 split-1 应使用的 16 token 语义。

### inplace_parallel 执行

`_run_split_batch_inplace_parallel()` 为每个 split 起一个 Python thread：

```text
vllm_ascend/worker/model_runner_v3.py
  3739: for slice_idx in range(num_splits):
  3740:     worker = threading.Thread(...)
  3745:     worker.start()
```

split-1 因为 offset graph 不存在，进入 lazy capture warmup：

```text
vllm_ascend/worker/model_runner_v3.py
  3682: if self._needs_inplace_serial_offset_capture(metadata):
  3683:     split_result = self._run_inplace_serial_offset_capture(
  3687:         parallel_streams=parallel_streams,
```

此时 split-1 的 query/key dim1 是 16，但编译图中的 rotary 参数仍读 slot0
而不是 slot1。slot0 长度与 split-1 不一致，NPU 算子校验失败。

## 根因判断

根因不是单纯的“parallel stream 不同步”，也不是 split-1 输入 view 地址不稳定。
这些问题可能仍需要单独验证，但不能解释当前异常，因为当前异常在
`npu_apply_rotary_pos_emb` 参数校验阶段已经发生。

更准确的根因是：

1. `inplace_parallel` 的正确性依赖 per-split rotary cos/sin 隔离。
2. 运行时 context 已经为 split-1 准备了 slot1 和 `context.cos/context.sin`。
3. torch compile 生成的 graph 没有把 per-split cos/sin 表达为运行时输入，而是
   捕获了全局 slot0。
4. split-1 lazy warmup 中，query/key token 维度来自 split-1，cos/sin 维度来自
   slot0，二者不一致。
5. `aclnnApplyRotaryPosEmbV2` 要求 query、key、cos、sin 的 dim1 相等，因此报：
   `all input dim1 must equal`。

## 为什么 serial 没有这个问题

`inplace_serial` 和 `inplace_parallel` 的关键差异不是模型本身，而是 slot 使用和
执行时序。

`_make_split_batch_metadata_inplace_serial()` 构造 context 时没有传
`cos_sin_slot_id=i`：

```text
vllm_ascend/worker/model_runner_v3.py
  2771: split_forward_context = create_ascend_forward_context(...)
  2781: in_parallel_streams=False,
```

`create_ascend_forward_context()` 的默认参数是：

```text
cos_sin_slot_id: int = 0
```

因此 serial 的两个 split 都复用 slot0：

1. split-0 context 构造时，slot0 被更新为 split-0 长度。
2. split-1 context 构造时，slot0 又被更新为 split-1 长度。
3. 后续执行是严格串行的，每个 split 在 main stream 上执行并 synchronize。

当前失败发生在 split-1 offset graph lazy warmup。对 serial 来说，进入 split-1
warmup 前，slot0 已经被 split-1 context 构造刷新成 split-1 长度。即使编译图
错误地读取 slot0，也不会出现 split-1 query dim1 与 slot0 dim1 不一致。

这不是一个健康的设计保证，而是时序上的规避：

- serial 没有 split-0/split-1 同时执行，也没有 slot0/slot1 隔离需求。
- serial 复用 slot0，反而掩盖了编译图“固定读 slot0”的问题。
- 如果未来 serial 的执行顺序、lazy capture 时机、context 构造顺序改变，或者
  某条路径让 split-0 eager warmup 在 slot0 已被 split-1 改写后运行，serial 也
  可能暴露类似问题。

因此，serial 通过不能证明 rotary slot 设计正确，只能说明当前 serial 路径没有
触发 slot0/slot1 不一致。

## 为什么历史 parallel 测试通过

历史 Phase 11 文档记录的 `inplace_parallel` NPU correctness 主要覆盖：

```text
model: Qwen/Qwen2.5-0.5B-Instruct
batch: 416, max_tokens=7, split 384+32
batch: 400, max_tokens=8, split 384+16
prompt: "The capital of France is"
```

这些测试通过与当前失败不冲突，主要原因如下。

### 1. 模型 head_dim 不同，rotary op 路径不同

历史模型 `Qwen/Qwen2.5-0.5B-Instruct` 配置：

```json
{
  "hidden_size": 896,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "torch_dtype": "bfloat16"
}
```

其 head_dim 为 `896 / 14 = 64`。

当前失败模型 `Qwen/Qwen3-0.6B` 配置：

```json
{
  "head_dim": 128,
  "hidden_size": 1024,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "torch_dtype": "bfloat16"
}
```

`_rope_forward_oot()` 中失败路径的条件包含：

```text
is_neox_style
self.head_size == 128
self.cos_sin_cache.shape[-1] == 128
cos is not None and sin is not None
```

Qwen3-0.6B 满足 `head_size == 128`，进入
`torch_npu.npu_apply_rotary_pos_emb` 路径；Qwen2.5-0.5B head_dim 为 64，
不会进入这条路径，而会走后面的 `_npu_rotary_embedding` 路径。后者使用
`positions` 和 `self.cos_sin_cache`，不依赖同样的全局 slot0/slot1 cos/sin
切片，因此不会触发本次 dim1 mismatch。

这是历史测试通过的最强解释。

### 2. 历史测试是稳定 prompt 小矩阵，不覆盖当前 benchmark 条件

历史 Phase 11 文档也明确写了：

```text
This is a smoke-level signal only.
Formal performance conclusions need a larger matrix...
The current validation proves correctness for the stable prompt
"The capital of France is" under fixed-batch decode.
```

当前 benchmark 条件更宽：

```text
model: Qwen/Qwen3-0.6B
max_tokens: 16
seq_len: random in [100,500]
dataset: LongBench-v2
batch: 400,448,496
parallel_capture_sizes: [1,2,4,8,16,32,64,128]
max_num_batched_tokens: 1024000
```

这组条件同时覆盖了 Qwen3 head_dim=128 rotary 快路径、offset graph lazy capture、
parallel stream split-1，以及更大的 decode 步数。历史 smoke/correctness gate
没有覆盖这个组合。

### 3. compile cache 和首次编译上下文会影响是否暴露

当前 cache 中 backbone graph 明确使用 slot0。该 graph 可能是在普通 decode /
非 DBO context 下首次编译，随后被 split context 复用。只要 graph 中已经把
`forward_context.dbo_enabled` 和 `cos_sin_slot_id` 相关逻辑特化掉，后续 runtime
context 设置 slot1 就不会生效。

因此这个问题具有条件性，但不是随机小概率问题。满足以下条件时风险很高：

- `split_batch_config.mode = "inplace_parallel"`。
- split-1 `start_num_tokens > 0`，需要 offset graph。
- offset graph 首次使用，需要 lazy capture warmup。
- 模型 rotary 进入 `npu_apply_rotary_pos_emb` 快路径，典型条件是 head_dim=128、
  neox style、cos/sin cache last dim=128。
- torch compile graph 中 rotary cos/sin 被特化为全局 slot0。
- split-0 和 split-1 使用不同 slot，且 slot0 长度与 split-1 长度不同。

不满足这些条件时，测试可能通过，例如 head_dim=64 模型、没有进入该 rotary
快路径、没有触发 lazy capture、或执行顺序让 slot0 恰好等于当前 split 长度。

## 修改方案

### 方案 A：把 rotary cos/sin 变成显式 graph/runtime 输入

推荐修复方向是让 split 的 rotary cos/sin 不再通过模块全局变量或
thread-local forward context 在编译区间内隐式读取，而是作为当前 model forward
的真实 tensor input 进入 graph。

目标形态：

- split-0 调用 graph 时传入 split-0 的 cos/sin。
- split-1 调用 graph 时传入 split-1 的 cos/sin。
- 编译图签名中不再只有
  `G_import_vllm_ascend_dot_ops_dot_rotary_embedding_cos_slice_slots_0_`。
- 对 `inplace_parallel`，slot 只用于准备 per-split tensor，不作为编译图内的
  隐式数据源。

实现注意点：

1. 不能只在 `_rope_forward_oot()` 内继续读取 `get_forward_context()`，因为
   torch compile 可能继续把 `dbo_enabled`、`cos_sin_slot_id` 或全局 slot 访问
   特化。
2. `forward_context.cos/context.sin` 也要以 graph 可见的 tensor 参数形式传递，
   否则仍可能被当作编译时常量或全局捕获。
3. 可以考虑通过 model kwargs、attention metadata、或一个明确的 runtime
   rotary input carrier 传递 cos/sin，但最终标准是编译图可检查：split-1 不应
   再读 slot0。
4. graph cache key 需要包含足够的 variant 信息，避免普通 decode graph 和
   DBO/split graph 错误复用。

这是最干净的修复，能同时解决 parallel stream 场景和未来更多 split 数、更多
模型结构下的同类问题。

### 方案 B：为 inplace_parallel split graph 建独立编译 variant

可以为 `inplace_parallel` 的 split-1 使用单独的 torch compile / graph variant，
确保首次编译发生在 `dbo_enabled=True`、`cos_sin_slot_id=1` 的上下文中。

这个方案比方案 A 风险更高：

- 如果编译结果仍捕获 `_cos_slice_slots[1]` 这种全局变量，本质上仍是对 slot 的
  特化，只是从 slot0 换成 slot1。
- graph cache key、lazy capture key、parallel graph pool 都需要额外维护。
- 对多个 split 或动态 slot 数不自然。

它可以作为过渡方案，但不应作为长期设计。

### 方案 C：临时 fallback / guard

在正式修复前，应避免让已知高风险组合进入 `inplace_parallel`：

- 当模型满足 `head_size == 128` 且会进入 `npu_apply_rotary_pos_emb` 路径时，
  `inplace_parallel` 回退到 `inplace_serial`。
- 或者在启动时检查 torch compile graph，如果发现 parallel split graph 只引用
  `cos_slice_slots_0`，直接禁用 `inplace_parallel`，给出明确 fallback reason。

这会牺牲 parallel 性能验证，但能避免 crash 和 engine 状态污染。

### 方案 D：诊断增强

为了后续验证和排错，建议加入低开销 debug 信息：

- split index、`cos_sin_slot_id`。
- `context.cos.shape`、`context.sin.shape`。
- 全局 slot0/slot1 当前 slice shape。
- rotary 调用前 query/key dim1。
- lazy capture warmup 是否处于 parallel stream。
- 编译图是否包含 slot0/slot1/global context cos/sin。

这些日志应只在 split debug 或显式诊断开关下启用。

## 验证计划

### 编译图检查

修复后先清理或切换 torch compile cache，然后重新运行最小 case。检查
`computation_graph.py`：

- 不应出现 split-1 仍只读 `cos_slice_slots_0`。
- 如果采用方案 A，期望看到 per-call cos/sin tensor input，而不是模块全局
  `_cos_slice_slots`。
- 如果采用方案 B，至少应能区分 split-0/split-1 graph variant，且 split-1 不读
  slot0。

### NPU 最小复现

优先跑当前失败的最小矩阵：

```text
model: Qwen/Qwen3-0.6B
mode: inplace_parallel
batch: 400
max_tokens: 16
seq_len range: [100,500]
parallel_capture_sizes: [1,2,4,8,16,32,64,128]
```

期望：

- 不再在 split-1 lazy capture warmup 中触发 rotary dim1 mismatch。
- perf jsonl 有正常 decode step 记录。
- split steps 等于 decode steps。

### 回归矩阵

至少覆盖：

```text
Qwen/Qwen3-0.6B:
  inplace_parallel bs=400,448,496 mt=16
  inplace_serial   bs=400,448,496 mt=16
  aclgraph         bs=400,448,496 mt=16

Qwen/Qwen2.5-0.5B-Instruct:
  inplace_parallel bs=400 mt=8
  inplace_parallel bs=416 mt=7
```

Qwen2.5-0.5B 用于保证 head_dim=64 的旧路径不回退；Qwen3-0.6B 用于覆盖
head_dim=128 的失败路径。

### 执行顺序压力测试

需要分别验证：

- 清空 compile cache 后先跑 `inplace_parallel`。
- 先跑 `aclgraph` / `inplace_serial`，再跑 `inplace_parallel`。
- 重复运行同一 case，确认不是首次 compile context 偶然通过。
- 开启 split debug，确认 split-1 使用的 cos/sin 长度始终等于 split-1 token 数。

## 临时结论

当前问题是特定条件下触发的真实实现缺陷，不是单纯的偶发 NPU runtime 抖动。

历史 `inplace_parallel` 能通过，是因为测试覆盖的模型和路径没有触发
`head_size == 128` 的 `npu_apply_rotary_pos_emb` slot 隔离问题。当前
Qwen3-0.6B + `inplace_parallel` + split-1 lazy capture 正好命中该缺陷。

在修复前，建议不要把 `inplace_parallel` 视为对所有 RoPE/GQA 模型可用；对
head_dim=128 的 Qwen3 类模型，应先 fallback 到 `inplace_serial` 或禁用 split
parallel，直到编译图可以明确使用 per-split cos/sin runtime input。
