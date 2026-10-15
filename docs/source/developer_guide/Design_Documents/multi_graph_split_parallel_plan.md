# 基于官方主线的多图切分并行实现方案

## 1. 基线与本次交付

- 官方仓库：[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- 基线：`upstream/main`，`33a8b92bdc00dc83c6088d51a397f0cab2f98761`。
- 基线提交时间：2026-10-13 11:28:45（Asia/Shanghai）。
- 本地开发分支：`feature/multi-graph-split-parallel`。
- 本地仓库：当前工作区下的 `work/vllm-ascend`。
- `origin` 保留原 min-wn fork，`upstream` 指向官方仓库。本次未 push、未 commit。
- 已接入 P1 与 P2/P3 实验代码；真实 NPU 验收尚未完成，不能视为已验证可用。

最初拉取的主线为 00627a11，随后快进到上述提交；两者差异仅为 CI test selector。本文所引用图执行代码在这两次提交间未变化。

### 1.1 当前实施状态

P1 配置/planner 与 P2/P3 实验实现已接入，模块为 `vllm_ascend/worker/multi_graph/{config,types,support,planner,validation,buffers,metadata,manager,npu_backend,runner_v1}.py`。

- 配置默认关闭，支持两 lane、exact、startup；校验大小、entry 上限和显式预算。
- 每 entry 独立 input/metadata/output、UpdatableGraph 和 pool；每 lane 独立 compute/update stream。图 key 包含 lane、模型/配置签名、generation、dtype、FIA 和 KV layout/block size。
- V1 启动期串行捕获未编译 forward；运行前全部 layout 预检和 task resolve，replay 后 update，原序合并 hidden states，沿用一次 logits/sampling。
- query 边界重基准，KV block/slot 与长度保持；CPU block metadata 检查重复写入 slot；KV 地址变化使图失效。
- 跨步 host fence 保护缓冲区和输出；提交后失败进入 FAILED，不 eager 重跑、不等待可能未解除的外部事件。
- 自动 KV 分配前预留预算，捕获超预算启动失败；显式 KV 字节预算、热权重/配置变更拒绝。
- `force` 用于实验验收；`measured` 已实现严格成本表加载、样本数/收益阈值和签名失效检查，无匹配记录时保持既有执行路径。benchmark 已实现同步 decode forward 测量和配对导出，尚未执行 NPU 采集。

P1 的 NotImplementedError 门禁已替换为具体准入验证。首版限制为 V1、标准 Ascend adaptor、LlamaForCausalLM/Qwen2ForCausalLM、FP16/BF16、标准 FIA、单 KV group、单设备纯 decode；compilation mode=0、FULL_DECODE_ONLY、npugraph_ex 关闭。完整启用说明见仓库 `multi_graph_split_parallel_usage.md`。

186 项 CPU 测试通过，tensor 测试使用真实 PyTorch 2.10.0+cpu；adapter 的 vLLM/NPU 导入依赖被 mock。新增成本持久化、启动捕获模拟、事件生命周期和 worker 门禁测试。完整插件集成及真实 NPU capture/replay、精度、overlap、性能尚未验证；P0/P2/P3 硬件验收待执行，不能将代码写完视为阶段已验收。

新增 `test_multi_graph_fia.py`、`test_multi_graph_split.py` 和 `benchmarks/multi_graph_split.py`，提供真实 FIA 多轮更新、四模式模型对照、同步 decode 成本导出和可选 NPU trace。P5 的 exact 成本表选路与离线契约已实现，性能收益仍待硬件验收。P4 padding/异步事件优化、P5 KV 分桶插值/在线采集、P6 mixed/V2/分布式属于后续扩展，不扩大当前运行支持矩阵。

### 1.2 依赖基准

以当前仓库锁定依赖建立 Linux Ascend 验证环境：

| 依赖 | 仓库依据 |
| --- | --- |
| PyTorch | requirements.txt：2.10.0 |
| torch-npu | requirements.txt：2.10.0.post4 |
| Triton Ascend | requirements.txt：3.2.2 |
| vLLM 发布基线 | .github/vllm-release-tag.commit：v0.30.0 |
| vLLM main 验证基线 | .github/vllm-main-verified.commit：8cc9aa5ad350f6f80cf3bebb298d410803ff75a9 |

vLLM 两个版本作为分别运行的兼容矩阵，不混用。CANN、HDK、固件、NPU 型号以及实际 wheel/build 信息写入测试结果；本文不猜测未验证的匹配版本。

## 2. 目标与架构决策

目标：将一次模型 forward 的 batch 按完整请求边界拆成多个子 batch，每份独立捕获、更新、回放模型图；在不同 NPU stream 上执行，恢复原请求顺序后只调用一次原有 logits/sampling 流程。

首版实现两图，两图通过后再推广到 N 图。首版选择：

**纯 decode + 固定 lane 输入缓冲区 + 每图独立 UpdatableGraph + 两条计算流 + 显式元数据更新。**

这里 lane 指固定执行通道，包含输入、图缓存、内存池和流。不是两个模型实例：权重只读共享、KV cache 按实际 slot 访问，中间张量与图资源独立。

不把旧 fork 的 model_runner_v3/v4 整体移植到主线。新功能以 runner 组合组件接入，不另造复制版 runner。

多图整模型并行与 attention-only 并行分开：

| 形态 | 含义 | 本方案阶段 |
| --- | --- | --- |
| serial_split | 两份完整模型图依次执行 | 正确性基准 |
| parallel_split | 两份完整模型图分流回放 | 核心交付 |
| piecewise_attention_parallel | 非 attention 分段串行，attention 分段并行 | 后续独立策略 |
| single_graph_attention_parallel | 保持完整 batch，在一个图内仅 attention 分叉 | 独立优化，不冒充多图成果 |

完整模型并行不保证性能提升：两个 MatMul 可能竞争同一组 Cube 和内存带宽。最终是否启用某种切分由真实双流测量决定，不依据“padding 更少”推导。

## 3. 最新主线中可复用的基础

以下源码链接固定到本次基线。

| 位置 | 已有能力 | 对方案的影响 |
| --- | --- | --- |
| [acl_graph.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/compilation/acl_graph.py) | ACLGraphWrapper、BatchDescriptor 缓存、UpdatableGraph 捕获、默认全局 graph pool | 可复用 wrapper 生命周期；必须增加显式独立 pool 注入 |
| [updatable_graph.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/compilation/updatable_graph.py) | 每图 tasks/providers/resources、ContextSource、resolve_tasks、update | FIA/PA 不必再建全局 main/parallel GraphParams |
| [attention_v1.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/attention/attention_v1.py) | FIA/FIAV2/PA provider、register_task、metadata builder | 复用算子更新契约，按 lane 提供 metadata |
| [model_runner_v1.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/worker/model_runner_v1.py) | 输入准备、batch dispatch、metadata、_model_forward、capture_model | 采用少量入口挂接 manager |
| [v2/aclgraph_utils.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/worker/v2/aclgraph_utils.py) | ModelAclGraphManager、UpdatableGraph replay | V2 后续使用单独 adapter |
| [compiler_interface.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/compilation/compiler_interface.py) | npugraph_ex 的 FX 编译优化及 force_eager 配置 | 不能等同旧 fork 的 npugraph_ex 宏图后端 |
| [platform.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/platform.py) | 当前将 enable_dbo 重置为 False | 新功能不能依赖打开 DBO 实现 |
| [rotary_embedding.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/ops/rotary_embedding.py) | 普通路径用显式 positions/cache，仍有全局 cos/sin 辅助函数 | 普通 dense 模型可先做；Minimax/310P 等全局读取路径先排除 |
| [device_metadata.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/worker/device_metadata.py) | 单 executor 提交、阶段事件、release 协议 | 不能让两份图共享同一次 producer/consumer 生命周期 |
| [sleep_mem_optimized.py](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/vllm_ascend/device_allocator/sleep_mem_optimized.py) | wrapper/graph manager 清理和恢复 | 新 manager 后续必须加入同等生命周期 |

当前平台在 VLLM_USE_V2_MODEL_RUNNER 未显式设置时返回 False，因此首版落在 V1 有实际依据；不能据此宣称 V2 不受支持，也不能让开启本功能的 V2 静默走未经实现的路径。

### 3.1 UpdatableGraph 的正确使用边界

每个图拥有自己的 GraphUpdateTask；其中保存 operation、kwargs、provider、handle、ExternalEvent。捕获时：

1. 图中记录 ExternalEvent.wait/reset。
2. graph_task_group_begin/end 只包围可更新算子。
3. replay 时先 resolve 本轮参数，提交图，更新流执行 task update，再 record 对应事件。

query/output/workspace 等捕获参数不会因为 Python 换了变量就自动重绑；provider 只更新自己声明的字段。FIA provider 当前主要更新 query/KV 长度和 block table，FIAV2、PA 各有不同字段集合。

use_updatable_graph 只识别 AscendAttentionBackend 体系。MLA/SFA 等仍有旧 GraphParams 路径，不能假设全仓库已完成同一种重构。

## 4. 首版支持矩阵

| 项目 | 首版处理 |
| --- | --- |
| 执行器 | V1 |
| 设备 | 910B/910C 分别实测启用；不自动覆盖 310P 或未验证设备 |
| 模型 | 已验证的 dense decoder-only、GQA/MHA、无特殊状态 |
| batch | uniform decode，每请求每步 1 token |
| graph mode | FULL_DECODE_ONLY 的 FULL decode 路径 |
| attention | 标准 AscendAttentionBackend；FIA 为首个已验收算子路径，PA/FIAV2 单独验证后加入 |
| 分布式 | TP=PP=DP=PCP=DCP=1，无 EP |
| 精度 | BF16/FP16 非量化，KV cache 非量化 |
| 图数量 | 2 条 lane，有限且预先声明的 capture sizes |
| capture | 启动期串行预捕获；在线只 replay，不 lazy capture |
| 编译 | 首先验证不含隐藏输入复制/内部图缓存的直接 forward；npugraph_ex/FX 融合另设验证阶段 |
| 状态功能 | 不支持 LoRA、spec decode、Mamba、KV sharing/cascade、Engram、跨层 KV 复用、动态 offload、KV transfer |
| 其他图模式 | 不支持 ENPU、breakable graph、Xlite、super-kernel 的组合 |
| device metadata | 有活动 producer/executor 的路径首版不进入多图 |
| async scheduling | 首版使用明确的跨 step 完成边界；异步扩展后才进入支持矩阵 |
| sleep/wake、权重热更新 | 启用多图时首版拒绝组合；完成失效/重捕获机制后加入 |

配置冲突在启动时报出具体原因；正常动态 batch 不满足条件、未命中图、收益不足则回到原单图/eager 路径。未覆盖功能属于明确边界，不伪装为已兼容。

## 5. 组件、职责和接口

### 5.1 新增模块

| 文件（拟新增） | 主要职责 |
| --- | --- |
| vllm_ascend/worker/multi_graph/types.py | 不可变 SlicePlan、ExecutionPlan、GraphKey、结果/拒绝原因 |
| vllm_ascend/worker/multi_graph/planner.py | 无 NPU 依赖的请求切分、图匹配、代价比较 |
| vllm_ascend/worker/multi_graph/buffers.py | lane 固定输入、metadata、输出收集、padding 生命周期 |
| vllm_ascend/worker/multi_graph/metadata.py | 从请求区间生成 lane common metadata，调用现有 backend builder |
| vllm_ascend/worker/multi_graph/manager.py | capture、资源所有权、串行/并行 replay、状态机 |
| vllm_ascend/worker/multi_graph/runner_v1.py | V1 输入输出适配，不复制 execute_model |
| vllm_ascend/worker/multi_graph/runner_v2.py | 后续 V2 adapter，首版不实现 |

新增类型的逻辑形态：

- SlicePlan：request_start/stop、token_start/stop、actual_tokens、graph_tokens、actual_reqs、graph_reqs、lane_id。
- ExecutionPlan：原始请求顺序、两个 SlicePlan、总 padding、graph keys、是否允许并行、成本来源。
- LaneState：compute_stream、update_stream、固定 buffers、图表、events、capture generation。
- MultiGraphEntry：graph、capture signature、捕获 tensor 地址签名、输出、资源 owner、状态。
- MultiGraphManager：supports、plan、capture、prepare、run_serial、run_parallel、invalidate、close。

主要待实现接口：

```text
plan(batch_summary, available_entries, cost_table) -> ExecutionPlan | NoSplit
prepare(plan, prepared_inputs, request_metadata) -> PreparedSplitBatch
capture(capture_specs, model_callable, scratch_kv_state) -> CaptureSummary
run(prepared_batch, execution_mode) -> full_order_hidden_states
invalidate(reason, new_generation) -> None
```

所有持久可变对象由 runner/manager 实例拥有，不增加 _graph_params_parallel 一类模块全局变量。不新增多处读取的实验环境变量；配置放入 AscendConfig 的 typed sub-config。

### 5.2 图 key 与隔离

首版固定 lane buffer 偏移为 0，避免 offset 图组合爆炸。图 key 包括：

- lane_id、buffer_generation。
- graph_tokens、graph_reqs、query_len=1。
- graph mode、dtype、attention backend 与实际算子变体。
- KV layout/block size、模型/编译配置签名。

request ID、实际 KV seq_len、物理 block ID 不进入 key，而是每 step 更新。一个 manager 对应固定模型和设备，不跨实例共享图。

例如两条 lane 都执行 256 tokens，仍必须是不同图实例；不能用同一张 graph 对象在两条流并发 replay。缓存可采用每 lane 单独字典，BatchDescriptor 无需修改上游 dataclass。

### 5.3 模型 callable 与内存池

ACLGraphWrapper 增加可选 graph_pool 参数，默认行为不变。第一阶段每个 MultiGraphEntry 使用独立 capture pool；证明同 lane 所有图严格串行且输出已消费后，才考虑同 lane 共享 pool 节省内存。

两个外层 wrapper 包住同一 compiled callable 并不自动安全：内层可能复用静态输入、输出、中间缓冲区或 piecewise graph。P0 必须验证调用链：

1. 首版捕获直接 tensor forward，权重只读共享，warmup 完成模型 lazy 初始化。
2. 不通过嵌套的既有 ACLGraphWrapper 再 capture；也不复制权重构造两个模型。
3. 检查编译器返回 callable 的静态 tensor 所有权和重入能力。
4. 若共享 compiled callable 具有可变缓存，采用 lane 独立 compiled execution state，或在该组合中禁用 split。
5. 所有 capture 串行执行，禁止两个 Python 线程同时改 capture context。

workspace、output 和更新用 tensor 的存活由 capture pool 和 entry owner 明确保证。现有 weak_ref_tensors 不能当成额外的强引用保活机制；debug 模式检查跨 entry 可写区间不重叠。

## 6. planner：切 batch，不切断请求

### 6.1 首版策略

输入为已有 CPU request/token 摘要，不在热路径对 NPU tensor 调用 item()/tolist()。

首版每请求 1 token，枚举两个连续非空请求区间。先提供 exact 策略：

- 每个区间长度必须存在预捕获图。
- 所有请求恰好覆盖一次、无交叉、不重排。
- 不能用一张容量 384 的图执行 385 个真实 token。
- 若总 batch 已有较优单图，保持单图。

示例：真实 batch=416，单图容量512。若已有 lane0=384、lane1=32，执行384+32；若只有256图，exact 策略不把416假装成256+256。

第二阶段支持有界 padding，枚举实际区间与各 lane 最小覆盖容量；例如208+208对应256+256，节省 padding 为0，并不自动进入双流。

### 6.2 代价模型

```text
T_single = 原单图端到端时间
T_split = 输入/metadata准备 + 实测双图执行及更新 + 输出收集
收益 = T_single - T_split
```

max(T_graph0, T_graph1) 只能作为乐观下界，不能替代双流测量。按设备、模型、dtype、图 pair、KV 长度区间保存测量结果。首版 benchmark 可强制执行，但默认策略没有有效测量则不声称有收益。

图数量预先有上限、显存预算、stream/event 资源预算。在线不生成无限尺寸、不无界缓存 offset。推广到 N 图后，组合搜索需受限，不能简单枚举所有 size 的 N 次笛卡尔积。

## 7. 输入、metadata 和 KV cache 的处理

### 7.1 首版使用稳定缓冲区

两条 lane 各自预分配 input_ids、positions、query_start_loc、seq_lens、slot_mapping、block_tables，以及所需的 pinned CPU 侧镜像。数据从本轮已准备输入复制到 lane，保持目标地址不变。

这是有意识地以少量输入/metadata 复制换取地址契约清楚；避免一开始把 offset 图、动态地址、inplace padding、并发全部叠加。后续才能加入输入 view/offset 优化，且必须独立验收。

执行层只拆 forward，不再次执行 scheduler bookkeeping、KV 分配、请求状态推进或 sampling。

### 7.2 逐字段规则

| 字段 | lane 规则 |
| --- | --- |
| input_ids、positions | 按连续 token 范围复制；positions 保留原绝对位置 |
| query_start_loc | 截取请求范围后减去该范围 token 起点，首项为0 |
| actual_seq_lengths_q | 根据 lane query_start_loc 生成，FIA TND 末项符合图 query 长度 |
| seq_lens、seq_lens_list | 取本请求真实 KV 长度，不减去 token offset |
| block_tables | 截取请求行，保留物理 block ID，复制到稳定 lane buffer |
| slot_mapping | 截取 token 范围，保留物理 slot 编号，不减去 offset |
| attention mask | 可只读共享的静态 mask 共享；动态/可写 mask 独立 |
| backend metadata | 通过现有 builder 构造，按 layer/group 保持映射 |
| logits_indices | 保留原全 batch 索引；merge 恢复原序后交回原 runner |

不要只调用一次全局 metadata builder 再浅拷贝两份对象：内部 tensor、Python list 和 builder 工作缓冲区可能仍然别名。按 layer/provider 验证 tensor 地址和可变列表独立。

### 7.3 padding 必须同时满足算子与 KV 写入语义

首个端到端版本只接受 exact size，先避开 padded request 歧义。支持 padding 时：

- capture 中固化的 num_actual_tokens、query/output shape 与 replay 必须一致。
- fake tokens 的 slot_mapping=-1；先用独立算子测试确认目标设备 kernel 对此不写 KV。
- fake request 的 block table/seq_len 使用已分配的专用 dummy KV 状态；不猜测 block0 永远安全。
- FIA query 累积长度、KV 长度列表、block table 行数、output view 必须相容。
- PA、FIAV2 的 padding 不复用 FIA 的未经验证假设。
- 真实 KV 写入互不重叠；共享 prefix 只能共享只读已完成区域，COW 必须已完成。
- merge 只复制 actual_tokens，fake output 不进入 logits。

不能仅将 padded 输入填0而不处理 cache write；图 capture 固化的切片长度不会因 Python 的 num_actual_tokens 改动自动变化。

### 7.4 RoPE 与模型状态审计

主线普通 rope_forward_oot 使用 positions 和 cos_sin_cache，优先覆盖此路径。仍读取全局 get_cos_and_sin_slice 的路径、共享可变 scratch 的模型，进入支持矩阵前必须显式改成 lane 所属 tensor 或明确拒绝。

warmup/capture/replay 用不同 positions，覆盖 head_dim=64/128、交替等长与异长 split，验证 compiler 没有把 lane0 数据特化成 lane1 的隐含输入。

## 8. capture、更新与回放协议

### 8.1 启动捕获

1. 创建完整功能能力检查结果与有限 capture specs。
2. 在 KV cache 分配前预留多图内存预算。
3. 使用独立 scratch 请求/slot 做 warmup/capture，不污染真实请求 KV。
4. 串行捕获每条 lane、每个 size；捕获期间 forward context、buffers、pool 明确对应 entry。
5. 记录每图 provider/task 数量、输入/metadata/output 指针签名、workspace 大小。
6. 至少跑两轮不同 seq_len 的 replay/update，对比 eager。
7. 所有资源和验收成功后 entry 才从 CAPTURING 转为 READY。

捕获数量、额外 pool 内存、workspace、event/stream 资源纳入 capture_model 返回及 worker 内存统计。预留值与实际值分别报告，避免 KV 分配吃掉多图所需空间，也避免重复扣除。

### 8.2 正确性首版的主机调度

首先使用单 Python 调度线程，串行下发两条计算流。设备流可并行，不要求 Python 双线程。

每步顺序：

1. 在发出本轮任何 replay 前，完成上一轮所有 lane 及更新流的必要 fence。
2. 准备全部 lane 输入与 metadata；在 prep stream 记录 inputs_ready。
3. 在任何图 launch 前，为所有图 resolve_tasks，并验证 shape、provider 数量、图状态。
4. 每条 compute/update stream 等待 inputs_ready。
5. compute0 提交 graph0.replay；update0 提交 graph0.update。
6. compute1 提交 graph1.replay；update1 提交 graph1.update。
7. 两条 compute stream 分别记录 graph_done。
8. merge stream 等待两个 graph_done，按原序复制真实输出到 runner-owned output buffer。
9. 原 runner 在该输出上执行 logits、采样和后处理一次。

图内 ExternalEvent 保证 FIA 在自己的参数更新完成后运行。保留非 ENPU 主线的 replay→update 顺序。

重要约束：

- 不能 graph0 replay 后先 synchronize(graph0) 再 update(graph0)，图会等待尚未触发的外部事件。
- 首版跨 step barrier 要放在所有本轮 replay 之前，不能在部分图已 launch、尚未 update 时做设备全局同步。
- wrapper 已拥有 UpdatableGraph 更新时，manager 不得再重复更新；新增 lane 执行入口统一明确唯一更新 owner。
- 先以每个 lane 执行完整 replay/update 的包装接口保证正确，再经 trace 决定是否改成“两图都 launch 后批量提交 update”；不依赖旧 fork 私有 update_many API。

### 8.3 事件化优化的准入条件

当前 ACLGraphWrapper 的 FULL 路径有 synchronize，它保护上一轮 replay 与本轮 task 更新之间的关系。不能简单删除换成 wait_stream。

后续将主机 fence 改成事件 DAG，必须证明：

- 输入覆盖发生在上一轮使用这些输入的 kernel 完成之后。
- 同一图的 task handle 更新不会与前一轮该 task 执行重叠。
- 更新流等待的 prev_done 指向上一轮，绝不等待本轮被自身更新卡住的 graph_done。
- output 在 merge 读取完前不会被下次 replay 覆盖。
- step n 和 n+1 的事件代次可区分，避免重复 record/reset 串代。
- graph_task_update_begin/end 的 host API 语义允许这种队列排序；若仅设备 wait 不能保护 host 侧更新，保留主机 fence。
- metadata Python list、pinned H2D 源缓冲区直到后端消费完成前不复用。

拟用每 lane 双代事件 prev_graph_done、inputs_ready、graph_done、update_done，以及输出 consumer_done。先实现每步最多一轮 in-flight，再支持更深流水。

### 8.4 异常与回退

安全回退只允许发生在本轮尚未 launch 前，例如未命中图、功能不支持、资源/shape 预检失败。

任一图已 launch 后，另一图失败或 update 失败时，不再把整个 batch 重跑 eager；否则可能发生重复 KV 写入、未解除 ExternalEvent 等待或污染设备状态。manager 标记 FAILED，走现有 worker 异常处理。不能无限等待完成事件以掩盖 NPU 异常；故障注入测试使用子进程超时结束。

状态机：NEW → WARMED → CAPTURING → READY → IN_FLIGHT → READY；异常进入 FAILED，地址或权重 generation 变化进入 INVALID。

## 9. 主线接入位置

### 9.1 配置与平台

新增 AscendConfig.multi_graph_config，遵循当前 @config/pydantic dataclass 风格，用 validator 检查范围和组合。影响编译行为的配置进入现有配置哈希/缓存标识流程，不能让 single 与 split 误用缓存。

以下为拟议配置，当前尚不可运行：

```json
{
  "multi_graph_config": {
    "enabled": true,
    "execution": "parallel",
    "num_lanes": 2,
    "workload": "uniform_decode",
    "input_layout": "lane_buffer",
    "capture_sizes_per_lane": [[32, 64, 128, 256, 384], [32, 64, 128, 256, 384]],
    "padding_policy": "exact",
    "capture_policy": "startup",
    "miss_policy": "single_graph",
    "selection_policy": "measured",
    "max_graph_entries": 10,
    "graph_memory_budget_bytes": 1073741824
  }
}
```

1 GiB 是示例配置值，不是通用显存估算或已验证默认。enabled 默认 false；execution 支持 serial/parallel；开发验收可用 force 策略，服务默认不强制切分。

### 9.2 V1 runner

- load_model：建立 manager 与经过验证的 callable，传递配置和模型信息。
- 输入/metadata 阶段：在 scheduler bookkeeping 已完成、真实 token 范围已知时规划 split；避免先把整个 batch 固化为单图 padded metadata 再硬切。
- _build_attention_metadata：抽取可复用的“给定 common metadata 构造 backend metadata”部分，lane 独立准备，原路径保持原行为。
- _model_forward：命中多图时调用 adapter；未命中仍执行现有 run_model。
- 返回值：首版只支持标准 hidden_states tensor；tuple、aux hidden、IntermediateTensors 在覆盖前不进入此路径。
- capture_model/profile_run：纳入多图捕获和预算；不能仅在 capture 后发现内存不足再继续服务。
- 后处理：沿用原 logits_indices 和 sampling，不为每 split 单独采样。
- device metadata 的 submit/release 仍成对；首版有活动 executor 时明确不切。

### 9.3 V2 后续适配

复用 planner、buffers、manager 的核心状态机，通过 ModelAclGraphManager/BatchExecutionDescriptor 接入。V2 的 model_state.attn_metadata、InputBuffers、slot_mapping、device_metadata.prepare/finish 使用其自己的 adapter，不通过修改全局对象伪装成 V1。

V2 admission 明确未实现时启动拒绝该组合。V1/V2 两套集成测试独立通过，不能把一套的结果当成另一套的证明。

## 10. 后续扩展及可靠性边界

### 10.1 混合 prefill/decode

按完整请求边界切，绝不截断同一 prefill 请求的 query。planner 以 sum(query_len × estimated_kv_len) 近似 attention 工作量；prefill 与 decode 分组需保留原序映射。

新 key 至少包含请求容量、query 容量及影响捕获控制流的 attention state。不能只按 total_tokens 复用不同请求组成的图。metadata builder 的不同算子选择、FIA workspace 最大值和 layout 分别捕获。

当前主线 attention_v1 声明可支持 mixed graph，但这只证明现有单图契约，不能直接推出新多图的 metadata/并发路径正确。先做 mixed serial split，再 mixed parallel。旧宏图 exact68 首次通过/第二次越界的形态纳入回归。

### 10.2 Piecewise 与 attention-only

只有端到端测量显示整模型图的 Cube 竞争限制收益时，再增加独立策略：

- piecewise 方案需从当前锁定 vLLM 的正式编译接口取得可调度片段；不能假定旧 fork 的 PiecewiseRuntimeHandle API 当前存在。
- 区分“两份非 attention 顺序执行”与“完整 batch 非 attention 执行一次”。
- attention-only 优化可保持一个完整 batch 图，只在 attention 中分流，将 output 直接写入不重叠 view。
- 同一个 UpdatableGraph 内两份并行 FIA 不能共享当前按算子名称索引的 workspace；resource key 要增加 lane/并发域。
- 同层两份 FIA 注册相同 provider 时，ContextSource 当前只返回一组参数；必须新增带 split_id 的 provider/source 契约，满足 provider_sizes 的数量校验。
- 任务更新只覆盖 FIA/PA 等受支持算子，不包整个含 Slice/Copy/cache-write 的 opaque attention 区域。

### 10.3 TP/PP/DP、MoE 与 N 图

TP 不能简单复制两条流并让同一 HCCL communicator 随机交错 collective。必须保证所有 rank 的 split plan、collective 顺序、shape 一致，或使用明确隔离且后端支持的通信域。

PP 需要 intermediate tensor 分片及顺序契约；DP 需要 padding/plan 协调；MoE/EP 需要路由、通信、EPLB 热度统计隔离。作为单独里程碑，不与首版混做。

推广 N 图前固定每 lane 资源上限、最大 in-flight 数与 planner 搜索复杂度；不是仅把 for range(2) 改成 N。

### 10.4 offset/inplace 与生命周期

lane buffer 版本稳定后才减少复制。offset 版本图 key 要加入 input storage identity/generation、offset、stride；相同 shape 不代表同一 capture 地址。按有限 offset plan 预捕获，在线 miss 回退。

sleep/wake、KV cache 重分配、权重重新加载、编译缓存变化均使 generation 失效。先 drain、清图资源，再恢复/重捕获；不能仅清 ACLGraphWrapper 字典而留 manager 中失效 handle。

## 11. 实施拆分与验收门槛

| 阶段 | 具体变更 | 完成条件 |
| --- | --- | --- |
| P0 基线与算子验证 | 固定依赖；确认两张 UpdatableGraph、独立 pool、共享只读权重可重复 replay；审计 compile/RoPE/workspace | 两图等长/异长、不同输入和 seq_len 的多轮结果正确，无跨图可写别名 |
| P1 配置与 planner | typed config、纯 CPU planner、支持矩阵、明确拒绝原因 | 边界与属性测试通过；功能关闭无行为变化 |
| P2 exact 串行多图 | lane buffers、metadata、串行 capture、serial_split、输出 merge | eager/单图/serial_split 对照通过；请求顺序与 KV 写入一致 |
| P3 exact 并行多图 | 两 compute/update stream、单线程提交、上一轮主机 fence、错误状态机 | parallel 与 serial 一致，trace 证明确有设备重叠；无死锁/泄漏 |
| P4 padding 与事件优化 | dummy KV、容量图复用、生命周期事件、受控省 copy | padding 数据不写真实 KV；异步压力与事件代次验证通过 |
| P5 性能选路 | 实测 pair cost、KV 长度分桶、收益阈值、资源预算 | 可复现性能收益；无收益 case 正确回到单图 |
| P6 扩展 | mixed serial→parallel、V2、分布式分别交付 | 各自新增支持矩阵和硬件验证，不借用 P3 结论 |

P3 是“多图切分并行”的最小完整交付，P5 是有性能选路依据的服务化交付。每阶段单独可评审，未通过前不扩大功能覆盖。

## 12. 测试设计

### 12.1 单元测试（拟新增）

- tests/ut/worker/multi_graph/test_planner.py：覆盖无重叠/无遗漏、图容量、无空 split、奇数 batch、图 miss、成本退回、确定性。
- tests/ut/worker/multi_graph/test_metadata.py：query_start 重基准、KV 长度不减 offset、slot/block ID 保留、list 与 tensor ownership。
- tests/ut/worker/multi_graph/test_manager.py：先全量预检后 launch、pool 隔离、相同 size 不同 graph、update 唯一 owner、异常不得 eager 重跑。
- tests/ut/worker/multi_graph/test_lifecycle.py：地址 generation、entry 状态、下一轮写入和 output 消费顺序。
- 扩展 tests/ut/compilation/test_acl_graph.py：pool 注入默认行为不变、UpdatableGraph provider 数量/resolve 失败、输出所有权。
- 扩展 tests/ut/worker/test_model_runner_v1.py 与配置测试：开启/关闭路径、postprocess 一次、支持矩阵拒绝。

Mock 测试只能证明 Python 调用顺序；不能证明 NPU task update 的设备语义。

### 12.2 NPU 正确性矩阵

新建 tests/e2e/pull_request/one_card/aclgraph/test_multi_graph_split.py，后续加入 two_card/夜间压力。

| 维度 | 必测项目 |
| --- | --- |
| 模型 | Qwen2.5-0.5B、Qwen3-0.6B、Qwen3-4B；记录实际 backend/head dimension |
| 图 pair | 相同容量、384+32、256+128、极小余数、两边交替变化 |
| batch | 捕获档位前后±1、非命中尺寸、请求逐步结束、插入新请求 |
| KV | 短/长上下文、跨 block 边界、prefix cache/COW 单独后续启用 |
| 输入 | 各请求不同 prompt/position；不能只用完全相同 prompt |
| 次数 | 首次 capture、至少第二次 replay、1000步压力、连续多个独立进程 |
| 精度 | BF16、FP16，固定参数与 seed；logits误差、top1、token序列 |
| 故障 | prelaunch shape/resolve 失败、部分 launch 后 update 失败、预算不足 |
| 回归 | 单图功能关闭、exact→fallback→exact 切换、旧 mixed exact68 后续回归 |

对照必须使用独立 engine 或重建相同 KV 初态，不能在同一份已经写过的 KV 上依次执行 eager 和 split 再假称等价。teacher-forcing 固定输入逐步比较 logits，避免早期 token 分歧放大后无法定位。

BF16/FP16 容差按算子精度合同与基线误差确定并写入测试；不任意放宽到“生成文本看起来一样”。top1 接近并列时报告 margin 和数值误差，不能靠更改 seed 掩盖差异。

### 12.3 性能、压力和验收

四组对照：原单图、exact serial split、exact parallel split、padded parallel split。相同设备、模型、请求序列、上下文长度、输出长度、KV 预算与编译选项。

报告：

- 端到端吞吐、每请求 TPOT 的 p50/p95/p99、TTFT。
- CPU 规划/metadata/提交时间。
- 设备 copy、attention、MatMul、update、event wait、merge。
- 实际双流重叠比例；不能用 host thread 同时活跃替代 NPU trace。
- capture 时间、图数量、显存峰值、KV capacity、长期内存趋势。
- split attempt/hit/miss/fallback reason、实际走的 execution mode。
- 性能与 correctness 脚本遇到失败返回非0；结果 JSON 标明是否发生 fallback。

TPOT 不使用 wall_time/全 batch token 数冒充单请求 token 间隔；吞吐与请求延迟分别统计。

每个性能 case 至少5次独立测量，warmup与capture单列。提议服务选路阈值为收益超过测量噪声且至少5%，非最终性能承诺。未达阈值维持单图；强制 split 的实验结果仍完整报告。

### 12.4 后续实现时的验证命令

以下命令仅在对应文件实现及依赖安装后运行：

```bash
pytest -q tests/ut/worker/multi_graph tests/ut/compilation/test_acl_graph.py
pytest -q tests/ut/worker/test_model_runner_v1.py
pytest -q tests/e2e/pull_request/one_card/aclgraph/test_multi_graph_split.py
bash format.sh ci
```

使用项目现有风格检查；新增功能需要 UT、NPU E2E 与性能验证。AGENTS.md 中 model_runner 行为和 patch 的架构评审作为后续代码评审要求，不把它理解成撰写方案前需额外批准。

## 13. 风险与判定依据

| 风险 | 具体失败形态 | 控制方式 |
| --- | --- | --- |
| 图池/compiled scratch 别名 | 双图输出随机串写 | 独立 entry pool、callable审计、指针区间测试 |
| 动态 metadata 与捕获参数不一致 | 第二轮才错或越界 | lane固定地址、provider字段审计、多轮不同seq_len验证 |
| task update 时序 | 提前更新旧任务、ExternalEvent死锁 | 保留跨step fence、唯一更新owner、事件代次测试 |
| padding污染KV | fake token写真实slot | exact先行、-1语义实测、dummy KV隔离 |
| output过早复用 | merge读到下一轮结果 | graph_done→merge→consumer_done依赖 |
| 全局RoPE/模型状态 | 模型或head_dim相关错误 | 显式支持列表、不同positions测试、排除隐含全局路径 |
| 多图不提速 | Cube/带宽竞争大于padding收益 | 实测pair成本，不用理论max时间承诺性能 |
| 资源膨胀 | capture OOM、stream资源耗尽 | 图数量/内存/事件预算、启动期验收 |
| 分布式collective乱序 | rank间死锁 | 首版单卡；分布式另行明确顺序协议 |
| 失败后错误回退 | 重复KV写、挂死worker | 只允许prelaunch回退，launch后进入失败处理 |

## 14. 当前结论与未完成事项

官方最新主线已经提供每图独立的 UpdatableGraph 和 provider 更新抽象，使多图方案不必重复旧 fork 的全局 graph params 改造。可靠的实现顺序是：隔离资源和 metadata，先完成 exact 串行验证，再引入双流，最后处理 padding 与事件优化。

本次已完成基线获取、新分支建立、源码核对及方案交付。未实现 runtime 功能、未安装 NPU 运行环境、未执行 NPU 测试；本文所有性能收益和扩展支持均为待验证目标。首要 P0 不确定性是锁定依赖下共享模型 callable 的可重入性、task update 的跨流时序、以及实际双流资源竞争，必须以硬件结果闭环。
