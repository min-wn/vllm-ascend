# GDN 请求组 ready：实际 runner 兼容性核验

日期：2026-10-21。结论：**暂缓直接接入实际 runner；本轮不实施服务集成。**

已有离线真实自回归生成循环的 **2.6687%** 收益保留。它是在同一 compact FP32 算法、强制 Q4、BF16 head 下，ready 相对 delayed 的调度收益；包含 prefill 的完整批次收益仍不稳定。本轮没有得到新的原生生成性能数据，也没有否定已经观测到的 ready 收益。

暂缓有三个具体依据：原生请求实际可能为 Q1 或混合 Q1/Q4，现有 compact 只支持 Q4；状态 owner/物理 block 的动态生命周期适配尚不存在；这次原生默认图初始化在生成前耗尽 stream 资源。另外，原生缓存池的账本不支持直接复用旧入口的 B56 驻留批次。即使先排除图初始化故障，前两个契约缺口仍存在。

## 一、冻结范围与证据性质

保持原内核、compact FP32 状态、20 AIC/40 AIV 资源、Q4、BF16 head 和默认图目录；没有实施消费侧候选、路由器、状态转换器或变长 recurrence。全部官方数值检查、真实文本分布初筛及既有任务质量失败继续归档，不以本次性能核验覆盖。

复用上轮四条真实短文本各 14 次，共 56 请求，prompt 长度 33..37，greedy，正常 EOS，每请求最多输出 256 token，ngram min/max=2/4、最多 3 个候选。模型为 Qwen3.5-27B TP2，HF32 关闭，FP32 matmul highest，最大上下文 2048。冻结 workload SHA：

```text
520c70cd7415449c2e8b53b78b8492a1e130b1c544d5a31a6bb3f3c35cb73181
```

本轮严格区分三类证据：

| 证据 | 实际执行了什么 | 可以支持什么 |
| --- | --- | --- |
| 上轮 compact 生成 ledger | 已完成的真实独立生成、实际 draft/接受/EOS，239 步 | 旧 Q4 入口真实工作量及调度对照 |
| 本轮原生 runner 探针 | 一次 TP2 初始化、模型/缓存准备、部分默认图捕获；随后失败 | 实际缓存布局、默认目录、初始化错误；没有原生生成分布 |
| 本轮 CPU 元数据核验 | 已安装 dispatcher、GDN builder、KVCacheManager 处理旧 ledger 的条件重建 | query/图/spec 分流规则和分配账本；没有执行模型或算子 |

条件重建假定沿用旧生成的 token、draft 和 EOS，没有新 admission、抢占、恢复、prefix hit 或批次调度改变。使用的 quartet ID 是合成元数据，**不是观察到的原生物理槽位**。原生 admission 不同还可能改变后续批次及生成轨迹，不能把这个重建标为原生实测。

冻结安装版本为 vLLM `0.30.0+empty`，HEAD `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；vllm-ascend `0.30.0rc1`，HEAD `cf67a3cc9f6ffd7891cceb8b84869f7acedbb195`；torch `2.10.0+cpu`，torch-npu `2.10.0.post4`。设备是 SSH `vllm-ascend-030` 的 910B3 卡 0、1。Ascend 工作区含既有实验修改，不能把安装版本号当作纯净上游代码的证明；本轮保存实际源码及逐文件 SHA，以它们作为判定依据。

## 二、实际 draft 与三类填充

旧入口的计数直接来自 `gdn-ready-generation-v1/pair0.ready/result.rank0.json`，ledger SHA 为 `4db07f4e6ebcd704ad462231105052e91e126a43602c7143d4cbf5edd8162588`。56 个 owner 的逐请求 draft 数和模式切换次数保存在本轮 `analysis.json:ledger_inventory.per_request`。

| 数量 | 旧 compact 真实 ledger | 沿相同 ledger 的原生规则条件重建 |
| --- | ---: | ---: |
| 请求验证次数 | 9,608 | 9,608 |
| 实际 draft=0 / draft=3 的请求验证 | 8,711 / 897 | 同一份 draft 输入 |
| 实际提出的候选 token | 2,691 | 2,691 |
| 为固定 Q4 增加的 filler token | 26,133 | 稳态条件重建不增加；新/恢复分支另述 |
| 每请求当前 query 长度 | 全部为 4 | Q1：8,711 次；Q4：897 次 |
| 未加图 padding 的 query 行 | 38,432 | 12,299 |
| 图容量 padding 行 | 88 | 709 |
| 图容量累计行 | 38,520 | 13,008 |
| LM head 实际投影行 / 按代码重建行 | 38,520 | 12,299 |

Q4 filler 是不足三个真实 draft 时为固定内核补入的计算行；图 padding 是 live Q4 行之外为命中默认容量而加的行，两者不能合并。旧入口对全部图容量行运行 head 后再裁出 live 预测；原生正常路径先按 `logits_indices` 选择 hidden，再投影。这里未启用 fine-grained LM-head TP 的额外行 padding；普通 TP2 的词表分片/通信仍存在。

表中原生 head 行数是按 `sum(1+draft)` 与 `logits_indices` 源码规则重建；CPU 脚本没有调用原生 head 或 `_calc_spec_decode_metadata`，不能称为 head 执行观测。

接受前缀 1/2/3/4 出现次数为 9416/79/56/57，它是**上一验证的接受状态选择**，不是当前 query 长度。90.664% 的请求验证没有真实 draft；不能因为它上一步接受了 4 个 token，就把当前零 draft 的 query 也标为 Q4。

原生 scheduler 为新加入或恢复的单 token decode 请求保留特殊分支：在满足条件时补成 4 行，并写入三个 `-1` placeholder。这是另一类原生 filler，不能笼统声称原生永不填充。此次没有跑到原生生成，不能测得该分支的实际频率；上述 CPU 重建不包含它。

## 三、图容量与 GDN 分流

原生配置的默认数值目录共有 51 个 size：

```text
1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,
120,128,136,144,152,160,168,176,184,192,200,208,216,
224,232,240,248,256,272,288,304,320,336,352,368,384,
400,416,432,448,464,480,496,512
```

这是容量数值目录，不等于图实例数量。模式为 FULL_AND_PIECEWISE；本配置 ngram 使 async scheduling 被禁用。原生 `uniform_decode_query_len=4`，仅在所有已完成 prefill 的请求当前 query 都为 4 时判为 uniform decode；Q1 和混合请求不能沿用旧入口的全 Q4 FULL 图选择。

| 图容量 | 旧 Q4 实际步数 | 原生元数据条件重建步数 |
| ---: | ---: | ---: |
| 16 | 0 | 18 |
| 24 | 0 | 6 |
| 32 | 0 | 41 |
| 40 | 9 | 16 |
| 48 | 1 | 29 |
| 56 | 5 | 71 |
| 64 | 8 | 10 |
| 72 | 1 | 8 |
| 80 | 0 | 6 |
| 88 | 2 | 4 |
| 96 | 2 | 0 |
| 104 | 0 | 26 |
| 112 | 66 | 2 |
| 136 | 6 | 1 |
| 144 | 7 | 1 |
| 168 | 34 | 0 |
| 224 | 98 | 0 |
| 合计 | 239 | 239 |

重建中 157 步全 Q1、80 步混合 Q1/Q4、2 步全 Q4；dispatcher 选择 PIECEWISE 237 步、FULL 2 步。FULL 两步是第 145、157 步，B28、容量 112。

GDN builder 在去除图 padding 后覆盖全部 12,299 个真实 query 行：

| GDN 子路径 | 请求验证次数 | query 行 | 长度语义 |
| --- | ---: | ---: | --- |
| spec recurrence | 897 | 3,588 | 每请求实际 query=4 |
| 全零 draft batch 的 non-spec recurrence | 6,383 | 6,383 | 每请求实际 query=1 |
| mixed batch 的 non-spec chunk 分支 | 2,328 | 2,328 | 每请求实际 query=1，有初态；不是新 prompt |

mixed 中的 Q1 被 builder 重分类进 `num_prefills`，在 GDN core 中走 chunk 路径。这只是执行分支名称，不表示重算整个 prompt。具体 chunk 后端还受硬件能力及可用性检查影响；本轮没有执行模型来确认其最终设备算子。

例如第 0 步 B56、零 draft：原生规则是 query/head=56、容量 56、56 个 Q1 non-spec recurrence；旧入口仍为 224 行 Q4。第 1 步 B56、14 个请求各有三个 draft：原生规则 query/head=98、容量 104，GDN 分为 56 个 spec query 行和 42 个 non-spec chunk 行；旧入口仍为容量 224。

上述计数揭示工作量差异，**没有给出其时延比例**。Q1/chunk/FULL/PIECEWISE、head、TP、状态访问和调度边界都不同，不能由 12,299/38,520 推算性能收益。

## 四、实际缓存布局及驻留预算

原生初始化已经分配出下列真实布局。两个 TP rank 均记录相同规格；每个 GDN 层：

| 项 | 实际值 |
| --- | --- |
| cache mode | none |
| 全局 pool block 数 / CPU 账本初始可用数 | 389 / 388；389 来自原生配置，388 来自 CPU BlockPool 保留一个 null block 的规则 |
| FIA group | 16 层，层 3,7,...63；page token size=896，BF16，2 KV heads、head width=256 |
| GDN groups | 3 组，每组 16 层；各层组有独立 block table |
| 每 GDN 请求候选槽 | 接受态基槽＋3 个 speculative 槽 |
| GDN spec 的逻辑 block size | 2048 |
| conv Tensor | BF16，[389,6,5120]，stride=[917504,5120,1] |
| SSM Tensor | FP32，[389,24,128,128]，stride=[458752,16384,128,1] |
| padded page bytes | 1,835,008 |

不能从 `auto` 配置名称推断 SSM 为 BF16；本次实际 Tensor 是 FP32。原生 FIA 页长 896 与旧离线显式分配的 128 不同，不能复用旧物理页号或 slot 算式。

以本批最大 query end=297、无 prefix hit、无抢占为条件，将实际缓存规格交给已安装 KVCacheManager：每请求需要 FIA 1 block，加三个 GDN group 各 4 blocks，共 **13 blocks**。56 请求需要 **728 blocks**，超过实际可用 388；其相同条件下驻留数上界为 **29**。

这是原生分配器的 CPU 账本核验，未实际申请这些物理块；388 不是原生运行中观测的 free-block 计数，也未观测到“原生实际 batch=29”。其他预留可能进一步降低上界；调度器仍可排队处理全部 56 请求。它说明原生 admission 不能直接保持旧 B56 同时驻留轨迹。核验用各组 block size 的 LCM=14336 满足 coordinator 的构造契约；这个元数据粒度不是新的 query 长度或图容量。

## 五、最小状态生命周期映射

必须按 request 的生命周期建立映射，不能把 persistent batch 行号或物理 block ID 当成永久 owner。所需最小记录如下，属于**待实现的适配契约**：

| 键或资源 | 原生来源 / 所需映射 | 当前 compact 的限制 |
| --- | --- | --- |
| request 身份 | request ID＋本次 incarnation/epoch | 现有 fixture owner 是固定 0..N-1；没有生产 allocate/release API |
| batch 行 | 本步 req_id_to_index；spec/non-spec 重排后再 gather owner | 行号可变化，不能直接作 compact base 下标 |
| GDN 接受态/conv 槽 | 每个 GDN group 的 block table quartet，物理 ID 可不连续 | 当前 lookup 在初始化时绑定固定 quartet |
| compact owner | 同请求跨 48 层稳定；每层 base[owner+1]＋logs/valid | opaque handle 不能交给官方 dense Tensor consumer |
| FIA KV | 独立 group block table＋实际 token position | 与 compact owner、GDN quartet 属于独立语义 namespace |
| 接受与位置 | 上次实际 accepted、当前已消费 token 数、当前 query 边界 | 三者必须分开；不能只上传 accepted 代替 query |
| in-flight 工作 | pending commit、ready completion、图引用、epoch | 现有 host generations 只数 fixture 步，不能替代槽位回收协议 |

按阶段需要的操作：

1. **新请求 prefill。** 原生合法 prompt prefill 写入 dense SSM/conv；首次 decode 前选出真正的接受前缀，把一份 FP32 接受态导入或合法绑定为 compact base，清空 logs/valid，建立各 GDN group 的 quartet→owner 映射；conv 按四候选契约初始化，FIA 继续使用原生页表。prefill 预测出的首 token 与已被 recurrence 消费的位置不能混淆。现有离线 bundle 只允许整批一次 `transition_to_decode()`，并不支持服务期间随时插入新 prefill。
2. **连续 Q4。** 沿用现有 compact base＋日志，按上一真实 accepted 选择，再处理当前 Q4；不需要每步物化四份 dense SSM。batch 压缩或重排只改变本步映射，owner 保持稳定。
3. **Q4→Q1。** 必须先等待相关工作完成，并把最后真实接受前缀闭合到一个可消费接受态。若 Q1 继续走官方路径，还需将该态放到官方指定 dense 槽，并按接受位置恢复 conv 的语义。官方 Q1 不会自动读 compact 日志；只把 opaque handle 传过去会被拒绝，使用旧 dense cache 则会读到过期状态。
4. **Q1→Q4。** 如果 Q1 由官方路径更新了 dense state，compact 不能继续使用旧 base；必须导入最新接受态，重置日志并重建有效映射。只有显式共享存储/所有权适配才可能省掉复制，当前实现没有这种桥接。
5. **暂停与恢复。** 未释放资源的短暂停顿可保留 owner、日志和接受信息；persistent 行变化需重映射。若抢占释放了 block 或重新 prefill，必须先解决 in-flight 引用，再废止旧 epoch，按新 block table 重建或重新导入，不能把旧物理槽位当作仍有效。
6. **结束、释放和复用。** 现有离线入口在批次结束做真实 accepted 的 terminal commit，并审计日志清零。生产 none/no-prefix-cache 模式若永不再消费此状态，可以在确认全部设备引用结束后显式丢弃；若要保留/审计最终状态则必须闭合。两种策略都需要明确的 per-request release 和 epoch 更新，清理 base/logs/valid、conv、lookup，再与原生 block 释放/复用协调。当前没有该 API。

在原生 none 模式，`mamba_get_block_table_tensor` 返回请求 quartet；Q1 consumer 读取 dense 槽。align 模式的状态 copy 不在本次配置路径上，不能假定它替 compact 完成桥接。本轮没有证据认定官方 Q1/Q4 切换存在正确性 bug；这里确认的是私有 compact 接口与原生 consumer 之间的缺口。

## 六、接入选择及可承担的额外成本

| 方案 | 对原生工作量的影响 | 所需状态操作 | 本轮判断 |
| --- | --- | --- | --- |
| 直接替换 recurrence，仅切 delayed/ready | 原生仍有 Q1、mixed chunk、后续 prefill | 现有 compact 不接受这些 consumer；动态 owner 生命周期未接 | 契约不满足，不能静默 fallback |
| 全部请求强制扩成 Q4 | 沿旧 ledger 需补 26,133 行 filler；图和 head 工作也变化 | 大体沿现有 Q4 算法，仍需生产 owner 适配 | 可构造受限研究入口，但不再是原生仅调度切换的接入，不实施 |
| dense/compact 双状态桥接 | 保留原生 Q1/spec 选择，但增加转换/同步 | 每次模式切换闭合或重新导入接受态，conv 也需对齐 | 额外成本尚未实测，不能认定能保住 2.67% |
| 共享接受态的显式 consumer 适配，或变长 compact recurrence | 有机会减少 filler 或复制 | 改变状态 ABI/consumer，或新增 recurrence 支持范围 | 新研究范围，本轮不顺手实现 |

沿旧 ledger，owner Q1→spec 为 831 次，spec→Q1 为 829 次，共 **1660 次 owner 模式切换**。这是逐请求事件数，不是 1660 次整批切换，也不是已测得的原生事件数。

作为可复算的双存储情景：每次切换搬一份接受态，每卡每 owner 的 48 层 FP32 SSM 为：

```text
48 × 24 × 128 × 128 × 4 bytes = 72 MiB
1660 × 72 MiB = 116.71875 GiB 单向状态 payload / rank
若按一次 copy 的读＋写计，共 233.4375 GiB / rank
```

旧生成循环均值 delayed/ready 为 21.325973/20.756844 秒，总节省约 **0.569129 秒**。若该收益能完全保留，全部新增桥接平均每个 owner 切换只能花约 **0.34285 ms**；上述 copy 情景单是读写就需平均约 **440.41 GB/s** 的额外有效吞吐，还没包括 flush 算术、日志、conv、元数据、launch 和同步。

这里的频率来自旧轨迹，字节量来自固定形状，**转换时间尚未实测**，带宽数字是收支预算而非测得带宽或硬件峰值比较。共享存储可改变复制量，批处理可减少 launch；所以它不是所有设计的不可避免下界，也不能据此断言桥接必定失败。原生工作量/批次改变后，0.569 秒本身也不能直接外推。现阶段只能判断桥接开销足以成为主要风险，不能承诺净收益。

已有 ready 一次导入约 0.093304 秒、终末提交约 0.009514 秒，都是旧完整入口的阶段成本，包含 workspace/管理等工作；不能按 owner 数或切换次数线性外推成桥接延迟。本轮没有为不兼容的桥接方案新增搬运实验。

## 七、一次原生探针的失败与边界

原生探针没有安装 compact。参数沿用上述短文本设置，`max_num_seqs=256`、`max_num_batched_tokens=2048`、`gpu_memory_utilization=0.6`、无 prefix caching、默认编译和图数值目录，额外保留既有 `enable_npugraph_ex=False`。独立 TP2 进程组设 900 秒上限，实际约 250.016 秒后失败，没有重试。

错误发生在 mixed prefill-decode PIECEWISE 图捕获进度 14/49 附近：

```text
Resource_Error(EE1023): Alloc Stream resource failed.
Reason: Too many streams are created.
aclrtStreamWaitEvent: 207005
capture_end: 507903 (后续 capture invalidated)
```

两个 rank 的实际生成 steps 都为零。没有原生每请求 query、head、接受或 EOS 的运行观察。日志还有已有 conv import fallback 警告，但未证明它导致 stream 耗尽，不能混为同一根因；也没有足够证据把这次错误进一步归因为驱动版本。

旧离线入口只捕获 B1..56 可达的 29 个 Q4 默认容量，范围 4..224；原生会初始化自己的 FULL_AND_PIECEWISE 图集合。它们使用默认容量来源，但捕获范围与执行路径并未证明等价。不能用旧捕获成功替代原生初始化资格；本轮也不通过缩减目录或改变默认模式来绕过失败。

## 八、源码位置与复核入口

下表行号对应本轮保存的**已安装源码快照**，不是根据版本号猜测的上游代码；完整文件在三个 `installed-*.zip` 中，`source-excerpts.json` 保存关键片段和文件 SHA。

| 源码位置 | 核验内容 |
| --- | --- |
| vllm/v1/core/sched/scheduler.py:656、812、1040、1266 | 真实 scheduled query、draft 裁剪、新/恢复请求 filler |
| vllm_ascend/worker/model_runner_v1.py:1739、1766、2724、3446 | draft mask、logits_indices、先选 hidden 再 head、uniform decode |
| vllm_ascend/ops/gdn_attn_builder.py:62、706、717、777 | 去除图 padding、物理 quartet、spec mask、mixed Q1 分流 |
| vllm_ascend/ops/gdn.py:561、639、710 | spec recurrence、mixed non-spec chunk、纯 non-spec recurrence |
| vllm/v1/attention/backends/utils.py:1140 | none/all/align 的 block table 语义 |
| vllm/v1/worker/gpu_model_runner.py:1214、1461、1590 | 请求清理、重排/恢复时 block IDs、实际接受数更新 |
| vllm/v1/worker/mamba_utils.py:1385、1433、1540 | 状态映射清理及其他 cache mode 的 copy 边界 |
| vllm/v1/core/kv_cache_coordinator.py:94、170 | 各组共享 block pool、所需 block 数求和 |
| vllm/v1/core/single_type_kv_cache_manager.py:1713 | Mamba speculative scratch block 分配 |
| vllm_ascend/utils.py:958、ascend_config.py:1316 | fine-grained LM-head TP 开关及默认值 |

本仓库的对应限制：`multi_graph_gdn_q4_log_backend.py` 的 `Q4LogState.from_base/prepare_step/export`、`Q4LogBackend.__call__` 和 `CompactSSMRegistry.__call__`；`multi_graph_gdn_q4_full_metadata.py` 的独立页表与一次 decode 转换；`multi_graph_gdn_ready_generation_probe.py` 的全容量 head、实际 draft ledger 和 terminal commit。

新增分析器 `benchmarks/multi_graph_gdn_ready_runner_compat.py` 只做计数、元数据审计及情景预算。CPU 定向测试覆盖 accepted/query 区分、三类数量、暂停/重排 owner、丢行/换工作量/伪造实测 provenance 的拒绝和成本情景边界；22 项通过。定向 Ruff、格式和语法检查通过。全仓 `bash format.sh ci` 已执行，在既有 CRLF 的第 22 行 shell 语法处失败，未进入 pre-commit，不声称全仓检查通过；原始输出与退出码见 `validation.json` 和 `format-ci.log`。

证据目录：`benchmarks/results/multi_graph_structural/gdn-ready-runner-compat-v1`。包含实际版本/源码 SHA、原生参数及失败日志、两 rank 缓存布局、239 步 CPU 元数据核验、逐请求统计、转换预算、收口时无模型 worker 与冻结内核一致性证明。原始日志和快照保持字节不变，工具与内容清单可复核。

## 九、收口决定

**本轮完成兼容性分析，暂缓接入。** 尚未取得的是“原生 runner 默认图初始化成功后，同一短文本真实 admission/query/head 分布”的运行证据；即使补齐它，也仍需解决 Q1/mixed consumer 和动态状态生命周期契约，才能进行只切 delayed/ready 的公平原生对照。

现有 ready 作为 Q4 compact 调度机制的约 2.67% 生成循环收益保留；不升级为服务收益或部署资格。本轮没有改内核、精度门槛或质量记录，不继续启动生成、内核候选、桥接或完整服务工作。
