# PTA 迁移方案提案

> **目标**：分析 vllm-ascend-hust 当前在 Python 层实现的功能，识别哪些适合下沉到 PTA（PyTorch Adapter for Ascend）层，输出迁移方案。
>
> **作者**：wty
> **日期**：2026-05-23


> ## 执行摘要
> 
> 本文档分析 vllm-ascend-hust（inplace 分支）当前的 7 个自研模块（共 ~3749 行 Python 代码），
> 识别其中适合下沉到 PTA（PyTorch Adapter for Ascend）层的功能，并输出分阶段迁移方案。
> 
> **核心结论**：最高优先级的迁移候选是将 compilation/acl_graph.py 中的 **6 个 attention 参数更新函数**
> 合并为 1 个 PTA C++ API，把 for-layer 循环从 Python 移到 PTA 内部。
> 预计可将 Python→PTA 边界穿越次数从 **O(4L)/step 减少到 O(1)/step**
>（LLaMA-70B + 2 uBatches 场景：每步 640 次 → 1 次，~160ms 节省）。
> 
> **推荐实施路线**：Phase 1 (Attention 更新) → Phase 2 (Dual-Stream 原语) → Phase 3 (UBatch 同步) → Phase 4 (Core 分配)
> 
---
## 目录

1. [背景与现状](#1-背景与现状)
2. [软件架构全景](#2-软件架构全景)
3. [迁移候选分析](#3-迁移候选分析)
4. [详细方案：Phase 1 — Attention 参数更新](#4-详细方案phase-1--attention-参数更新)
5. [详细方案：Phase 2 — Dual-Stream Overlap 原语](#5-详细方案phase-2--dual-stream-overlap-原语)
6. [详细方案：Phase 3 — UBatch 线程同步](#6-详细方案phase-3--ubatch-线程同步)
7. [详细方案：Phase 4 — Core 分配控制](#7-详细方案phase-4--core-分配控制)
8. [不建议迁移的部分](#8-不建议迁移的部分)
9. [推荐实施路线](#9-推荐实施路线)
10. [收益预估](#10-收益预估)

---

## 1. 背景与现状

### 1.1 什么是 PTA

**PTA（PyTorch Adapter for Ascend）** 是 PyTorch 与昇腾 CANN 之间的适配层，使用 C++ 实现。它提供了 `torch.npu.*` 和 `torch_npu.*` 等 Python API 背后的底层实现。

- 项目地址：`https://gitee.com/ascend/pytorch`
- 语言：C++（核心）+ Python（绑定层）
- 位置：介于 PyTorch Python 前端和 CANN C 接口之间

### 1.2 当前的分界线

```
┌────────────────────────────────────────┐
│  vllm-ascend (Python)                   │
│  ├─ model_runner_v1.py / v3.py         │
│  ├─ npu_split_wrapper.py               │   ← 双流调度
│  ├─ npu_ubatch_wrapper.py              │   ← uBatching
│  ├─ ubatching.py                       │   ← 线程同步
│  └─ acl_graph.py                       │   ← 图管理 + Attention 更新
├────────────────────────────────────────┤  ← 分界线
│  torch_npu / torch.npu (Python 绑定层)   │
├────────────────────────────────────────┤
│  PTA (C++)                              │
│  ├─ graph_task_update_begin/end         │
│  ├─ NPUGraph capture/replay             │
│  ├─ npu_fused_infer_attention_score     │
│  └─ set_stream_limit / ExternalEvent    │
├────────────────────────────────────────┤
│  CANN (C)                               │
│  └─ ACL (Ascend Compute Language)       │
└────────────────────────────────────────┘
```

### 1.3 核心问题

> **当前 Python 层在每层 forward 后都要遍历所有 Transformer 层，逐层调用 PTA API。**
> 这种"每层一问"的模式导致大量 Python↔C++ 边界穿越。

具体来说，一次 decode step 的 attention 参数更新流程：

```python
# Python 循环 (当前)
for layer in layers:               # ← Python 遍历
    torch.npu.graph_task_update_begin(...)   # → PTA
    torch_npu.npu_fused_infer_attention(...) # → PTA
    torch.npu.graph_task_update_end(...)     # → PTA
    event.record(...)                        # → PTA
```

以 LLaMA-70B (80层) + 2 ubatches 估算：
- 每步穿越：80 × 2 × 4 = **640 次**
- 1000 步 decode：**640,000 次穿越**
- 穿越开销 (~0.5μs/次)：**约 320ms**

---

## 2. 软件架构全景

### 2.1 涉及的自研模块

以下 7 个文件是 vllm-ascend-hust（`inplace` 分支）的核心自研代码，共 **~3749 行**：

| 文件 | 行数 | 角色 |
|------|------|------|
| `worker/npu_split_wrapper.py` | 336 | **AscendSplitBatchWrapper** — 双流 batch 切分调度 |
| `worker/npu_ubatch_wrapper.py` | 439 | **AscendUBatchWrapper** — 多线程 uBatching + DBO |
| `worker/ubatching.py` | 299 | **AscendUBatchContext** — 流/事件/线程同步上下文 |
| `worker/ubatch_utils.py` | 661 | uBatching 工具（UBatchSlices 等） |
| `compilation/acl_graph.py` | 1369 | **ACLGraphWrapper** + 6 个 `update_attn_params*` 函数 |
| `ascend_forward_context.py` | 417 | **AscendForwardContext** — 前向上下文管理 |
| `inplace_split_debug.py` | 228 | JSONL 诊断日志工具 |

### 2.2 核心执行流

```
模型 forward (decode)
  │
  ├─ AscendSplitBatchWrapper.__call__()
  │    ├─ ubatch 切分 ─────► 2 个 BatchDescriptor + ForwardContext
  │    │
  │    ├─ ubatch[0]: ACLGraphWrapper.replay()     [default stream]
  │    │    └─ update_attn_params()                [update stream] ← 热点！
  │    │
  │    ├─ ubatch[1]: ACLGraphWrapper.replay()     [default stream]
  │    │    └─ update_attn_params()                [update stream] ← 热点！
  │    │
  │    └─ 结果合并 (cat + SP unpadding)
  │
  └─（非 split 路径直接走 ACLGraphWrapper）
```

---

## 3. 迁移候选分析

### 3.1 候选总览

| 优先级 | 模块 | 当前位置 | 迁移理由 | 预期收益 |
|--------|------|---------|---------|---------|
| **P0** | Attention 参数更新 | `acl_graph.py` (6 个函数) | 每层都穿越 PTA，热点路径 | 减少 **99%** 穿越次数 |
| **P1** | Dual-Stream 重叠调度 | `npu_split_wrapper.py` | Python 管理流的同步/切换 | 消除 Python 调度开销 |
| **P2** | UBatch 线程同步 | `ubatching.py` | threading.Event/Barrier 可被硬件事件替代 | 降低同步延迟 |
| **P3** | ACL Graph capture/replay 分发 | `acl_graph.py` | 多 entry 管理可下沉 | 消除输入地址校验的 Python 开销 |
| **P4** | Core 分配控制 | `npu_ubatch_wrapper.py` | 可集成到 Stream 属性 | API 简洁性 |

### 3.2 详细分析

#### P0: Attention 参数更新 ⭐ 最高优先级

**涉及文件**：`acl_graph.py` 第 871–1260 行

**6 个函数一览**：

| 函数 | 用途 | PTA 调用模式 |
|------|------|-------------|
| `_update_attn_pa_params` | PagedAttention 更新 | `begin → paged_attention → end → record` |
| `_update_attn_fia_params` | FlashInferenceAttention 更新 | `begin → fused_attention → end → record` |
| `update_attn_params` | PA/FIA 分发入口 | 同上（分发器） |
| `update_attn_params_split` | Split 版 (含 block_table refresh) | 同上 + `refresh_block_table` |
| `update_mla_attn_params` | MLA 更新 | `begin → multi_head_latent_attention → end` |
| `update_attn_dcp_pcp_params` | DCP/PCP 更新 | `begin → fused_attention → end` |
| `update_mla_attn_dcp_pcp_params` | MLA + DCP/PCP 更新 | `begin → multi_head_latent_attention → end` |

**所有函数共享的公共模式**：

```python
with torch.npu.stream(update_stream):
    for key, param, handle, event in zip(
        forward_context.attn_metadata,       # dict: {layer_id: metadata}
        graph_params.attn_params[param_key],  # list[tuple]: 静态参数
        graph_params.handles[param_key],     # list[handle]: graph task handles
        graph_params.events[param_key],      # list[event]: ExternalEvent
    ):
        # 1. 从 metadata 取实时数据
        seq_lens = forward_context.attn_metadata[key].seq_lens
        # 2. (可选) 刷新 block_table
        if refresh_block_table:
            _refresh_block_table_in_place(block_table, metadata.block_table)
        # 3. 三段式 PTA 调用
        torch.npu.graph_task_update_begin(stream, handle)
        torch_npu.<attention_kernel>(...)
        torch.npu.graph_task_update_end(stream)
        event.record(stream)
```

**迁移方案**：
创建一个 PTA 层的 `npu_batch_update_attention()` 函数，将 `for layer` 循环移到 C++ 内部。

```python
# 迁移前：6 个函数 × L 次穿越
update_attn_params(stream, fc, shape, config)    # Python 循环 L 次

# 迁移后：1 个函数 × 1 次穿越
torch_npu.batch_update_attention(
    stream, handles, param_batch, events, mode)  # PTA 内部循环 L 次
```

**参数说明**：
- `handles`：所有层的 graph task handles（来自 `graph_params.handles`）
- `param_batch`：包含 `query`, `key_cache`, `value_cache`, `block_table`, `seq_lens`, `output` 等每层参数
- `seq_lens`：这是关键——在 Python 侧从 `forward_context.attn_metadata` 提取后打包传入
- `mode`：`PA | FIA | MLA | PA_DCP_PCP | MLA_DCP_PCP`

**迁移前后接口对比**：

| 当前 | 迁移后 |
|------|--------|
| 6 个 Python 函数 | 1 个 PTA API |
| 每层调用 4 次 PTA | 1 次批量调用 |
| Python 负责遍历 + 参数提取 | Python 只负责打包参数 |
| 调用点需要 if/else 分发 | 统一 `mode` 参数 |

---

#### P1: Dual-Stream Overlap 原语

**涉及文件**：`npu_split_wrapper.py` 第 98–186 行 (`_run_ubatches`)

**当前实现**：

```python
def _run_ubatches(self, *args, **kwargs):
    for ubatch_id, md in enumerate(ubatch_metadata):
        # 1. default stream 上重放 graph
        model_output = self.aclgraph_wrapper(...)
        
        # 2. update_stream 上更新参数
        if fc.cudagraph_runtime_mode == CUDAGraphMode.FULL and not fc.capturing:
            self._update_attn_params_for_ubatch(fc, md.num_tokens, ubatch_id)
```

**迁移方案**：
PTA 提供 `npu_dual_stream_execute` 原语：
- 自动在 primary_stream 上执行主计算
- 同时在 secondary_stream 上执行更新
- 在指定同步点自动同步

```python
# 迁移后
torch_npu.dual_stream_execute(
    primary_stream=default_stream,
    secondary_stream=update_stream,
    primary_fn=lambda: self.aclgraph_wrapper(...),
    secondary_fn=lambda: update_attn_params(fc, num_tokens, ubatch_id),
    sync_points=[SyncPoint.AFTER_PRIMARY],  # 等主计算完了再继续
)
```

---

#### P2: UBatch 线程同步

**涉及文件**：`ubatching.py` 第 1–299 行

**当前实现**（纯 Python 线程同步）：

```python
class AscendUBatchContext:
    def __enter__(self):
        self.ready_barrier.wait()         # threading.Barrier
        self.cpu_wait_event.wait()        # threading.Event
        self._restore_context()
    
    def yield_(self):
        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()        # CPU 级 yield
        self._restore_context()
```

**迁移方案**：
PTA 提供轻量级 micro-batch 调度器，用 GPU 事件替代 `threading.Event`：

| 当前 (Python) | 迁移 (PTA) |
|---------------|-----------|
| `threading.Barrier(3)` | PTA 内部屏障 |
| `threading.Event.wait()` | `torch.npu.Event.synchronize()` |
| `threading.Event.set()` | `torch.npu.Event.record()` |
| `cpu_signal_event` + `cpu_wait_event` 交替 | 单 GPU event 同步 |
| `set_stream_limit(cube=comm, vector=comm)` | Stream 创建时绑定配置 |

---

#### P3: ACL Graph Capture/Replay 分发

**涉及文件**：`acl_graph.py` 第 135–770 行

**当前实现**：
```python
class ACLGraphWrapper:
    def __call__(self, *args, **kwargs):
        if is_new_entry:
            # capture
            with torch.npu.graph(aclgraph, pool=pool):
                output = self.runnable(...)
        else:
            # validate input addresses (Python)
            if new_addresses != entry.input_addresses:
                raise AssertionError(...)
            entry.aclgraph.replay()
```

**迁移方案**：
PTA 提供多 entry graph manager，内部管理：
- capture/replay 分发（基于 key）
- 地址校验（C++ 层直接比较指针）
- 双 pool 支持（main + parallel streams）
- in-place 偏移图管理（`start_num_tokens > 0`）

---

#### P4: Core 分配控制

**涉及文件**：`npu_ubatch_wrapper.py` 第 42–86 行

**当前实现**：
```python
class NPUCoreControlContextManager:
    def __enter__(self):
        torch.npu.set_stream_limit(self.current_stream,
                                   cube_num=self.comm_aic_core,
                                   vector_num=self.comm_aiv_core)
```

**迁移方案**：
将 core 分配集成到 `torch.npu.Stream` 创建时：

```python
# 当前
stream = torch.npu.Stream()
with NPUCoreControlContextManager(comm_aic=4, comm_aiv=2, stream):
    ...

# 迁移后
stream = torch.npu.Stream(cube_limit=4, vector_limit=2)
```

---

## 4. 详细方案：Phase 1 — Attention 参数更新

### 4.1 目标

将 6 个 `update_attn_params*` Python 函数合并为 1 个 PTA C++ API，消除 `for layer` 循环的 Python↔PTA 边界穿越。

### 4.2 新 API 设计

```cpp
// PTA 层新增 (C++)

enum class AttentionUpdateMode {
    PA,          // PagedAttention
    FIA,         // FlashInferenceAttention
    MLA,         // Multi-head Latent Attention
    PA_DCP_PCP,  // PA + Data/Context Parallel
    MLA_DCP_PCP, // MLA + DCP/PCP
};

struct AttentionParamBatch {
    // 所有层的参数 (vector size = num_layers)
    std::vector<at::Tensor> query;
    std::vector<at::Tensor> key_cache;
    std::vector<at::Tensor> value_cache;
    std::vector<at::Tensor> block_table;
    std::vector<at::Tensor> seq_lens;       // 来自 metadata (实时数据!)
    std::vector<at::Tensor> output;
    
    // 公共标量
    int64_t num_kv_heads;
    int64_t num_heads;
    double scale;
    
    // FIA 特有
    std::vector<at::Tensor> attn_mask;
    int64_t block_size;
    std::vector<at::Tensor> softmax_lse;
    
    // MLA 特有
    std::vector<at::Tensor> q_pe;
    std::vector<at::Tensor> k_pe;
    std::vector<at::Tensor> k_nope;
    
    // DCP/PCP 特有
    int64_t dcp_size;
    std::vector<at::Tensor> cp_seq_len;
};

/// @brief 批量更新 Graph Task 的 Attention 参数
/// @param stream      目标 NPU stream
/// @param handles     所有层的 graph task handles [L]
/// @param params      所有层的参数 batch
/// @param events      记录完成事件的 events [L]
/// @param mode        更新模式
void npu_batch_update_attention(
    at::Stream stream,
    std::vector<NPUTaskGroupHandle> handles,
    AttentionParamBatch params,
    std::vector<at::npu::ExternalEvent> events,
    AttentionUpdateMode mode
);
```

### 4.3 Python 侧封装

```python
# acl_graph.py

@dataclass
class AttentionParamBatch:
    handles: list
    events: list
    mode: AttentionUpdateMode
    # PA/FIA
    query: list[torch.Tensor]
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]
    block_table: list[torch.Tensor]
    seq_lens: list[torch.Tensor]   # from metadata
    output: list[torch.Tensor]
    num_kv_heads: int
    num_heads: int
    scale: float
    # FIA
    attn_mask: list[torch.Tensor] | None = None
    block_size: int = 0
    softmax_lse: list[torch.Tensor] | None = None
    # MLA
    q_pe: list[torch.Tensor] | None = None
    k_pe: list[torch.Tensor] | None = None
    k_nope: list[torch.Tensor] | None = None
    q_nope: list[torch.Tensor] | None = None
    # DCP/PCP
    dcp_size: int = 1

def _build_param_batch(forward_context, graph_params, param_key,
                       mode, refresh_block_table=False) -> AttentionParamBatch:
    """从 forward_context 和 graph_params 构建参数 batch。"""
    batch = AttentionParamBatch(...)
    for key, param, handle, event in zip(...):
        # 静态参数来自 graph_params
        # 实时数据 (seq_lens) 来自 forward_context.attn_metadata
        batch.seq_lens.append(forward_context.attn_metadata[key].seq_lens)
        # 可选: block_table refresh
        if refresh_block_table:
            _refresh_block_table_in_place(...)
    return batch

# 统一的 update 入口 (替换原来的 6 个函数)
def update_attn_params(update_stream, forward_context, runtime_shape,
                       vllm_config, in_parallel_streams=False,
                       split_mode=False):
    graph_params = get_graph_params(in_parallel_streams)
    param_key = get_graph_param_key(forward_context, runtime_shape)
    mode = _determine_attention_mode(runtime_shape, vllm_config, forward_context)
    
    param_batch = _build_param_batch(
        forward_context, graph_params, param_key, mode,
        refresh_block_table=split_mode)
    
    torch_npu.batch_update_attention(
        stream=update_stream,
        handles=param_batch.handles,
        param_batch=param_batch,
        events=param_batch.events,
        mode=param_batch.mode,
    )
```

### 4.4 调用点变化

| 调用位置 | 当前 | 迁移后 |
|---------|------|--------|
| `npu_split_wrapper.py` | `update_attn_params(..., fc, num_tokens, config)` | 同上（接口不变） |
| `model_runner_v1.py` | 4 个 if/else 分支 | 1 个统一调用 |
| `model_runner_v3.py` | 2 个 wrapper 函数 | 1 个统一调用 |
| `mtp_proposer.py` | 单独传 `update_mla_attn_params` | 传 `mode=MLA` |

### 4.5 收益预估

| 指标 | 迁移前 | 迁移后 |
|------|--------|--------|
| Python→PTA 穿越/step | L × 4 (80×4=320) | 1 |
| 穿越耗时 (0.5μs/次) | 160μs | 0.5μs |
| 1000 step 穿越总耗时 | 160ms | 0.5ms |
| 调用代码行数 | ~400 行 (6 个函数) | ~100 行 (1 个函数 + 辅助) |


### 4.6 各函数差异对比

6 个 update 函数的本质差异在于使用的 attention kernel 和 seq_lens 数据来源：

| 维度 | PA | FIA | MLA | DCP_PCP | MLA_DCP_PCP | Split 版 |
|------|----|-----|-----|---------|-------------|----------|
| **Attention kernel** | `_npu_paged_attention` | `npu_fused_infer_attention_score` | `npu_multi_head_latent_attention` | `npu_fused_infer_attention_score` | `npu_multi_head_latent_attention` | 同对应 kernel |
| **seq_lens 来源** | `metadata.seq_lens` | `metadata.seq_lens_list` | `decode.seq_lens_list` + `actual_seq_lengths_q` | `decode_meta.num_computed_tokens_of_pcp_dcp` | `decode.cp_seq_len` | 同左 |
| **block_table refresh** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **额外参数** | workspace (GQA bug workaround) | attn_mask, block_size, sparse_mode, softmax_lse | q_pe, k_pe, spec_attn_mask | dcp_size, pcp_rank, dcp_rank | cp_seq_len | 同对应 kernel |
| **使用的** `graph_params` | 普通 | 普通 | 普通 / MTP | 普通 | 普通 | 普通 |

> 尽管参数有差异，但**所有函数共享同一个三段式 PTA 调用模式**：
> `graph_task_update_begin → attention_kernel → graph_task_update_end → event.record`
> 因此可以合并为同一个 PTA API，通过 `mode` 参数区分。


---

## 5. 详细方案：Phase 2 — Dual-Stream Overlap 原语

### 5.1 目标

将 `AscendSplitBatchWrapper` 中的双流重叠调度模式（同时执行 graph replay 和参数更新）封装为 PTA 原语。

### 5.2 当前执行流程

以 `_run_ubatches()`（`npu_split_wrapper.py:98-186`）为例，当前双流的完整执行时序：

```
时间 →
                                                     ubatch[0] 完成
                                                     │  ubatch[1] 开始
                                                     │  │          ubatch[1] 完成
                                                     │  │          │
default stream:  [==== ACL graph replay ubatch[0] ====]  [==== ACL graph replay ubatch[1] ====]
                                                     │             │
update stream:                        [= update attn params =]    [= update attn params =]
                                       ↑             ↑             ↑
                                     当前 ubatch    更新完成后     当前 ubatch
                                     结果已产出     (为下一轮     结果已产出
                                                   做准备)
```

**Python 层的调度开销**：

```python
# 当前循环（每次迭代 Python 控制流切换）
for ubatch_id, md in enumerate(ubatch_metadata):
    # Step 1: 切换 forward context（Python）
    with override_forward_context(fc):
        # Step 2: replay graph（PTA 内执行，Python 等待）
        model_output = self.aclgraph_wrapper(...)
    
    results.append(model_output)
    
    # Step 3: 切换 stream（Python）
    # Step 4: 更新 attention 参数（Python 循环 L 层 × 4 次 PTA 调用）
    if fc.cudagraph_runtime_mode == CUDAGraphMode.FULL and not fc.capturing:
        self._update_attn_params_for_ubatch(fc, md.num_tokens, ubatch_id)

# merge（Python）
with override_forward_context(original_forward_context):
    # SP unpadding + cat
    result = torch.cat(sorted_results, dim=0)
```

**问题**：ubatch[0] replay 完成后，Python 需要做：结果收集 → context 切换 → stream 切换 → 参数更新 → ... → 才到 ubatch[1] replay。每一步都是 Python 代码执行，产生了不必要的 CPU 调度延迟。

### 5.3 原语设计

将整个双循环下沉到 PTA，一次调用完成所有 ubatch 的调度：

```python
# PTA API
torch_npu.dual_stream_execute(
    primary_stream=Stream,          # default stream
    secondary_stream=Stream,        # update stream
    primary_fn=Callable,            # graph replay (可调用多次)
    secondary_fn=Callable,          # param update (与 primary 交替)
    num_cycles=2,                   # ubatch 数量
    sync_strategy="after_primary",  # 等 primary 完成后再继续
)
```

PTA 内部实现时序：

```
PTA 内部循环（无 Python 介入）：
  cycle 0:
    primary:   [graph replay ubatch 0]
    secondary: [update attn params 0]     ← 与下一个 primary 重叠
  cycle 1:
    primary:   [graph replay ubatch 1]
    secondary: [update attn params 1]
  merge: 结果合并（可留在 Python）
```

同步点由 PTA 内部管理，不再经过 Python 的 stream/context 切换。

### 5.4 与 uBatching 的关系

当前有两套双流机制：

| 特性 | AscendSplitBatchWrapper (npu_split_wrapper.py) | AscendUBatchWrapper (npu_ubatch_wrapper.py) |
|------|-----------------------------------------------|--------------------------------------------|
| **执行模式** | 单线程，两个 ubatch 顺序执行 | **多线程**（2 个线程 + main 线程） |
| **流切换** | 同一个线程切换 stream | 每个线程绑定各自的 stream |
| **同步机制** | 无显式同步（顺序执行） | `threading.Barrier(3)` + `threading.Event` |
| **参数更新** | `update_attn_params` 在 update_stream 上 | DBO 同步原语（`dbo_yield`, `dbo_switch_to_comm` 等） |
| **适用场景** | Inplace Serial 模式 | DBO（Decoder Block Overlap）模式 |

**迁移方案对两者的影响**：
- **SplitBatchWrapper**：双流原语直接替换 `_run_ubatches` 中的 for 循环
- **UBatchWrapper**：需要先完成 Phase 3（线程同步下沉）再引入双流原语

---

## 6. 详细方案：Phase 3 — UBatch 线程同步

### 6.1 目标

将 Python 的 `threading.Event/Barrier` 同步替换为 PTA 硬件事件，消除多线程调度中的 CPU 同步开销。

### 6.2 当前线程模型

`make_ubatch_contexts()`（`ubatching.py:195-299`）创建 2 个 `AscendUBatchContext`，每个在一个独立线程中运行：

```
Main Thread                UBatch Thread 0            UBatch Thread 1
    │                            │                         │
    │ 创建 2 个 ubatch ctx       │                         │
    │ 启动 2 个线程 ────────────►│                         │
    │ 启动 2 个线程 ──────────────────────────────────────►│
    │                            │                         │
    │ Barrier.wait()             │ Barrier.wait()          │ Barrier.wait()
    │（等待线程就绪）            │（初始化 cuda 上下文后等待）│（初始化 cuda 上下文后等待）
    │                            │                         │
    ├── [开始 capture/replay] ──►│── context.__enter__()   │── context.__enter__()
    │ 设置 cpu_wait_event[0]     │   Barrier.wait()        │   Barrier.wait()
    │ 设置 cpu_wait_event[1]     │   cpu_wait_event[0]     │   cpu_wait_event[←1→]
    │                            │   .wait()               │   实际是 events[0]!
    │                            │                         │
    │                            │  [compute on stream 0]  │  [compute on stream 1]
    │                            │                         │
    │                            │  yield():               │  yield():
    │                            │   signal_event.set()    │   signal_event.set()
    │                            │   wait_event.wait()     │   wait_event.wait()
    │                            │                         │
    │                            │  switch_to_comm():      │  switch_to_compute():
    │                            │   update_stream(comm)   │   update_stream(compute)
    │                            │                         │
    │                            │  ... DBO 同步 ...       │  ... DBO 同步 ...
    │                            │                         │
    │←─── thread.join() ◄───────┤◄────────────────────────┤
    │                            │                         │
```

**关键同步原语**（12 个注册函数，`ubatching.py:188-240`）：

| 函数 | 作用 | 底层实现 |
|------|------|---------|
| `dbo_yield` | 让出 CPU，等待另一个线程的信号 | `threading.Event.wait()` / `set()` |
| `dbo_yield_and_switch_from_compute_to_comm` | 切到 comm stream + yield | `Event` + `torch.npu.stream()` |
| `dbo_yield_and_switch_from_comm_to_compute` | 切回 compute stream + yield | `Event` + `torch.npu.stream()` |
| `dbo_switch_to_comm` | 无同步切换 | `update_stream(comm_stream)` |
| `dbo_switch_to_compute` | 无同步切换 | `update_stream(compute_stream)` |
| `dbo_switch_to_comm_sync` | 同步后切到 comm | `signal_compute_done` → `wait_compute_done` |
| `dbo_switch_to_compute_sync` | 同步后切回 compute | `signal_comm_done` → `wait_comm_done` |
| `dbo_record_current_stream` | 记录当前 stream 完成事件 + 设置 core limit | `torch.npu.Event.record()` + `set_stream_limit()` |
| `dbo_wait_current_stream_and_yield` | 等待 completion + yield + 设置 core limit | `wait_compute_done` → `cpu_yield` → `set_stream_limit` |
| `dbo_maybe_run_recv_hook` | 执行接收 hook（用于 DBO 通信） | Python callable |
| `dbo_register_recv_hook` | 在另一个 ubatch 上注册接收 hook | 跨线程变量赋值 |
| `dbo_get_previous_event` | 在上一个 ubatch 的 compute stream 上执行 | `with torch.npu.stream(ctx.compute_stream)` |

### 6.3 流分配逻辑

`make_ubatch_contexts()` 中的关键设计——**交错分配**计算流和通信流：

```python
for i in range(num_micro_batches):
    if i == 0:
        current_microbatch_stream = compute_stream   # ubatch 0 用默认 compute 流
        other_microbatch_stream = comm_stream         # 另一路用 comm 流
    else:
        current_microbatch_stream = comm_stream       # ubatch 1 用 comm 流
        other_microbatch_stream = compute_stream       # 另一路用 compute 流
```

**目**的：两个 ubatch 分别占用不同 NPU 硬件流（compute / comm），通过 DBO 同步原语实现计算和通信的重叠。

### 6.4 迁移方案

PTA 提供 `NPUMicroBatchScheduler`，在 C++ 层管理线程/事件/流分配：

```python
# 当前 (Python)
ctxs = make_ubatch_contexts(
    num_micro_batches=2,
    compute_stream=compute_stream,
    comm_stream=comm_stream,
    forward_contexts=forward_contexts,
    ready_barrier=threading.Barrier(3),
    schedule="default",
)

# 迁移后 (PTA)
scheduler = torch_npu.MicroBatchScheduler(
    num_batches=2,
    compute_stream=default_stream,
    comm_stream=comm_stream,
    forward_contexts=forward_contexts,
)
```

PTA 内部自动处理：
- 线程创建和销毁（替代 Python `threading.Thread`）
- Event 同步（替代 `threading.Event` / `threading.Barrier`）
- NPU 流切换（`torch.npu.stream()` 调用移入 PTA）
- Core 分配（`torch.npu.set_stream_limit()` 移入 PTA）

| 当前 (Python) | 迁移后 (PTA) |
|---------------|-------------|
| `threading.Barrier(3)` | PTA 内部屏障 |
| `threading.Event.wait()` | `torch.npu.Event.synchronize()` |
| `threading.Event.set()` | `torch.npu.Event.record()` |
| `_cpu_yield()` (Event switch) | GPU event 同步 |
| 12 个 `dbo_*` 注册函数 | PTA 内部统一调度 |

---

## 7. 详细方案：Phase 4 — Core 分配控制

### 7.1 目标

将 `NPUCoreControlContextManager` 的 AIV/AIC core 分配逻辑集成到 PTA Stream 的创建时配置中，消除 `set_stream_limit` 的运行时调用。

### 7.2 当前实现

当前 `NPUCoreControlContextManager`（`npu_ubatch_wrapper.py:42-86`）在 ubatch 同步的关键点控制 NPU 核心分配：

```python
class NPUCoreControlContextManager:
    def __enter__(self):
        # 进入时：减少当前 stream 可用的 AIC/AIV 核心数（留给通信）
        torch.npu.set_stream_limit(self.current_stream,
                                   cube_num=self.comm_aic_core,
                                   vector_num=self.comm_aiv_core)
    
    def __exit__(self):
        # 退出时：恢复全部核心
        torch.npu.reset_stream_limit(self.current_stream)
```

**调用位置**（在 `AscendUBatchContext` 的 DBO 同步点中）：

```python
# ubatching.py:137-154
def record_current_stream(self, event=UBatchEventKey.DEFAULT):
    if self.comm_cube_core != -1 or self.comm_vector_core != -1:
        torch.npu.set_stream_limit(dbo_current_stream(),
                                   cube_num=self.comm_cube_core,
                                   vector_num=self.comm_vector_core)
    self._signal_compute_done(event)

def wait_current_stream_and_yield(self, event=UBatchEventKey.DEFAULT, wait=True):
    if wait:
        self._wait_compute_done(event)
    self._cpu_yield()
    if self.comm_cube_core != -1 or self.comm_vector_core != -1:
        torch.npu.set_stream_limit(dbo_current_stream(),
                                   cube_num=self.comp_cube_core,
                                   vector_num=self.comp_vector_core)
```

### 7.3 迁移方案

将 core 分配变成 Stream 的**创建时属性**，而不是运行时的动态切换：

```python
# 当前：每次 DBO 同步点都要调用 set_stream_limit
stream = torch.npu.Stream()
with NPUCoreControlContextManager(comm_aic=4, comm_aiv=2, stream):
    ...

# 迁移后：Stream 创建时绑定额度
comm_stream = torch.npu.Stream(cube_limit=4, vector_limit=2)
compute_stream = torch.npu.Stream(cube_limit=12, vector_limit=6)
```

PTA 内部实现：
- `torch.npu.Stream()` 新增 `cube_limit` 和 `vector_limit` 参数
- Stream 切换时自动应用 core 限制（而非每次调用 `set_stream_limit`）
- `NPUCoreControlContextManager` 不再需要

### 7.4 与 Phase 3 的交互

Phase 3 的 `MicroBatchScheduler` 创建时会接收 core 限制参数：

```python
scheduler = torch_npu.MicroBatchScheduler(
    num_batches=2,
    compute_stream=compute_stream,   # 已绑定 core 额度
    comm_stream=comm_stream,         # 已绑定 core 额度
)
```

> 因此 Phase 4 可以在 Phase 3 的基础上自然集成，不需要额外实现。

---

## 8. 不建议迁移的部分


| 模块 | 理由 | 建议 |
|------|------|------|
| `ascend_forward_context.py` | 纯配置逻辑，随模型/量化/并行度频繁变化 | 留在 Python |
| `inplace_split_debug.py` | Debug 诊断工具 | 留在 Python |
| `_slice_model_inputs()` | 框架层 tensor 切分，与 vLLM 数据流强耦合 | 留在 Python |
| MoE 通信策略选择 (`select_moe_comm_method`) | 策略性逻辑，需灵活调整 | 留在 Python |
| `ubatch_utils.py` 中的 UBatchSlices | 纯数据类，无性能热点 | 留在 Python |
| 模型加载/权重处理 | 与框架耦合深 | 留在 Python |

---

## 9. 推荐实施路线

```
Phase 1 (P0): Attention 参数更新下沉
  ├─ 目标: 6→1 个 PTA API，减少 ~99% 穿越
  ├─ 改动: acl_graph.py + PTA 新增 1 个 API
  └─ 风险: 低 — 逻辑不变，只改调用方式

       ↓

Phase 2 (P1): Dual-Stream 原语
  ├─ 目标: 简化 npu_split_wrapper.py 调度
  ├─ 改动: npu_split_wrapper.py + PTA 新增 1 个 API
  └─ 风险: 中 — 需要重新设计同步语义

       ↓

Phase 3 (P2+P3): UBatch 同步 + Graph 管理
  ├─ 目标: 用 GPU 事件替代 threading.Event
  ├─ 改动: ubatching.py + PTA 新增 MicroBatchScheduler
  └─ 风险: 高 — 线程模型变化大，需充分测试

       ↓

Phase 4 (P4): Core 分配集成
  ├─ 目标: set_stream_limit → Stream 构造参数
  ├─ 改动: npu_ubatch_wrapper.py + PTA 修改 Stream 类
  └─ 风险: 低 — API 简化
```

### 9.1 各阶段依赖关系

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
   ↓            ↓
 无依赖     依赖 Phase 1 (update_attn_params
             是 secondary_fn 的核心)
```

---

## 10. 收益预估

### 10.1 性能收益

| 阶段 | 每步减少的 Python↔PTA 穿越 | LLaMA-70B 每步节省 |
|------|---------------------------|-------------------|
| Phase 1 | 4L → 1 (减少 4L-1 次) | 319 次 (~160μs) |
| Phase 2 | 流切换 Python 逻辑消除 | ~10μs |
| Phase 3 | Event/Barrier 消除 | ~5μs |
| **合计** | | **~175μs/step** |

### 10.2 代码质量收益

| 指标 | 当前 | 迁移后 |
|------|------|--------|
| 自研 Python 代码行数 | ~3749 行 | ~3000 行 (-20%) |
| `acl_graph.py` 中 update 函数数 | 6 个 | 1 个 |
| 调用点 if/else 分支 | 4 路 | 1 路 |
| PTA C++ 新增代码 | 0 行 | ~500 行 |

### 10.3 架构收益

- **关注点分离**：性能关键路径入 PTA，策略逻辑留 Python
- **复用性**：PTA API 可供其他 PyTorch 推理框架（非 vLLM）使用
- **可维护性**：减少 Python 层面的并发/同步复杂性

---

## 附录

### A. 关键代码位置速查

| 文件 | 关键行号 | 内容 |
|------|---------|------|
| `acl_graph.py` | 871–957 | `_update_attn_pa_params` |
| `acl_graph.py` | 959–1028 | `_update_attn_fia_params` |
| `acl_graph.py` | 1030–1061 | `update_attn_params` (分发) |
| `acl_graph.py` | 1062–1137 | `update_mla_attn_params` |
| `acl_graph.py` | 1138–1203 | `update_attn_dcp_pcp_params` |
| `acl_graph.py` | 1204–1255 | `update_mla_attn_dcp_pcp_params` |
| `acl_graph.py` | 1258–1262 | `GraphParams` dataclass |
| `npu_split_wrapper.py` | 61–95 | `_update_attn_params_for_ubatch` |
| `npu_split_wrapper.py` | 98–186 | `_run_ubatches` (核心调度) |
| `ubatching.py` | 50–160 | `AscendUBatchContext` |
| `ubatching.py` | 195–299 | `make_ubatch_contexts` |

### B. 当前遍历模式代码段

```python
# acl_graph.py:891-930
for key, param, handle, event in zip(
    forward_context.attn_metadata,       # dict
    graph_params.attn_params[param_key], # list[tuple]
    graph_params.handles[param_key],     # list[handle]
    graph_params.events[param_key],      # list[event]
):
    seq_lens = forward_context.attn_metadata[key].seq_lens
    # ... block_table refresh (可选) ...
    torch.npu.graph_task_update_begin(update_stream, handle)
    torch_npu._npu_paged_attention(query=query, ...)
    torch.npu.graph_task_update_end(update_stream)
    event.record(update_stream)
```

---

*本方案基于 `vllm-ascend` 仓库 `inplace` 分支 (commit 9cd05f24) 分析。*
