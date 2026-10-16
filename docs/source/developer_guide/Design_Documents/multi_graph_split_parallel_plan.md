# 基于官方主线的多图切分并行实现方案

## 当前优化执行方案（2026-10-16）

本节取代下文早期快照中的实施进度和下一步安排。下文保留最初的架构依据，不能将其中“未实现”“未测试”“未推送”等历史描述当作当前状态。

### 目标与测量边界

切分是本项工作的既定目标：优化请求边界切分后的多图执行，使其在明确的负载范围内，相对单图 padding 取得可重复的单次 replay 性能优势。成本选路和 fallback 是服务化保护，不能代替切分优化本身。

当前仅考察 target model 一次 forward/replay，不包含 draft model、接受率、采样、调度和完整生成耗时。推测验证抽象为每请求固定 query 长度 Q；Q=1、2、4、8 分别对应普通 decode 与不同验证 token 数，Q 包含本次所有 target 输入 token。以请求数 B、实际 token 数 B×Q、图容量 C×Q 分别记账，不能把 batch 和 token 数混为一谈。

单次 replay 必须区分两种口径：

1. **replay＋必要 update（默认主对照）**：保留现有 UpdatableGraph 更新协议，使用相同模型、输入和图容量；报告逐层更新数、host 时间与设备完成时间。当前 A2 FIA 路径使用 host 端真实 seq_len，长度变化必须更新；已捕获的 ExternalEvent 依赖也不能直接跳过。单次 replay 的性能结论必须包含这条更新链。
2. **固定元数据 replay（可选消融）**：输入地址、query 长度、KV 长度、页表和 slot 固定，重新捕获直接记录 FIA 的独立实验图，不建立每层 task update 与 ExternalEvent。只用于估计更新协议的成本；不能将其视为动态长度服务的可用优化，也不能以此宣称实际切分优于 padding。通过 `--protocols update frozen` 显式启用。

当前固定参数基准在计时前 resolve task 参数，计时包含 task update 提交、事件依赖、图执行与切分输出合并；尚不包含动态 metadata 构造和逐步参数解析。不能把该时间等同于完整服务步耗时。

两个口径中单图和切分必须使用同一种 attention 实现。不能用优化后的切分对比未优化的单图并归因于切分。每个样本仅执行一次 forward，末端同步；事件计时报告设备调度区间，wall time 报告 host 提交至完成。并行必须包含两图完成后的输出合并；事件区间包含可能的 host 供给空洞，不能冒充 kernel 时间之和。

### 原方案主线与现有证据

| 原方案环节 | 已完成内容 | 当前仍缺少的证据 |
| --- | --- | --- |
| 图与状态隔离 | 独立 pool、输入、metadata、输出、事件，完整请求切分，V1 单卡路径 | 多 token 验证的服务接入与压力测试 |
| 降低 CPU 准备成本 | 共享 metadata 每 lane 每步只刷新一次 | 大 batch 和 Q>1 下的具体占比 |
| 双流并行 | 独立 compute/update stream、受控提交顺序实验 | 现有 trace 未证明计算重叠，不能将双 stream 等同加速 |
| 减少 attention update | A2 设备长度 Triton 原型通过 40 个数值用例 | 大 batch/长 KV kernel 效率不足，尚未接入模型 |
| 图容量与切分策略 | exact、padding、长度分桶等基础组件 | 有性能优势的容量组合与稳定适用范围 |

已有数据：metadata 去重使 B=7 的完整生成 p50 从约 521 ms 降至 294 ms，但同轮单图约 210 ms。提前 replay 无明确收益，逐 task 交错更慢。设备长度原型的小负载算子收益不能替代全模型 replay 结论。这些均是历史证据，不是本轮单次 replay 的测量值。

### 按顺序执行的优化步骤

| 顺序 | 实施内容 | 要回答的问题与交付标准 |
| --- | --- | --- |
| O1 固定验证 replay 基准 | 全模型固定 B/Q/KV；独立图资源；单图、串行切分、双流切分；固定 FIA 与 update FIA 两种协议；对照 eager hidden states | 首先排除接受率与生成链路干扰，准确分离切分计算代价、update 代价和调度代价 |
| O2 优化必要更新链 | 主对照保留 FIA update，研究参数准备去重、更新提交与计算重叠；免 update 仅作固定参数消融，设备长度算子另行验证；不复用旧长度冒充新长度 | 减少更新开销而保持真实长度与事件协议正确；动态路径需长度变化后的精度与生命周期证据 |
| O3 调度与输出路径 | 基于 O1 trace 优化 fan-out/fan-in、事件复用、提交批量化；保留跨轮缓冲区保护；再尝试合并输出拷贝或输出地址复用 | 找出造成两图不重叠的依赖，证明实际 kernel overlap，并测量包含 merge 的最终 replay |
| O4 kernel 与切分粒度 | 平衡切分、不平衡切分、主图＋尾图；保留 FIA 基线，研究 GQA 的 KV 复用、向量/Cube 分工、分段归约成本 | 避免小子图 MatMul 效率下降抵消 padding 节省；同 attention 下串行/并行消融 |
| O5 固定 Q 的验证图 | Q=2/4/8，按完整请求切分，query 前缀和重基准，正确 causal mask 与 KV 写入位置 | 实验图先通过 eager/单图/双图的所有有效 token 对照；不能直接解除生产 query_len=1 门禁 |
| O6 动态扩展 | 在有收益区间接入动态长度、请求变动、padding 复用与安全异步 fence；再接服务选路 | 单次 replay 优势已明确后扩展，mixed/V2/分布式另行验收 |

O1 和 O5 的固定形状实验可以同时建立，但不意味着生产推测解码已经接入。若某一步负收益，保留结果并定位原因，继续优化下一处可消除的切分成本；不能用默认退回 padding 作为优化完成的结论。

### 第一轮负载与验收

- B：7/32/33/64/65/128/129；明确 C，既包括 C=B 的无 padding 对照，也包括跨容量边界的受控 padding。指定的 2 的幂容量是实验条件，不能声称为官方所有配置的默认 bucket。
- Q：1/2/4/8；KV 长度至少覆盖 64/512/2048，并要求 KV≥Q。
- 切分：先均衡两图，再比较约 3:1、主图＋尾图；每个请求完整保留，所有子图实际 token 总数等于 B×Q。
- 每组先做精度与执行路径检查，再 warmup，五轮交替模式顺序，分别报告 p50/p95、单图/切分比值、事件时间与 wall time；不在 profiler 内采集性能数字。
- KV 使用 disjoint 页与 slot；padding 请求有独立 dummy KV。每个模式恢复相同 KV 初态；比较所有有效 token 的 hidden states，固定容差，不因失败而放宽。
- 独立 trace 判断 kernel overlap，不能从 stream 数或耗时推断。记录图数量、每图 task 数、捕获时间和峰值内存。
- 至少重复独立进程确认收益超过噪声；5% 是候选收益门槛，不是承诺。结论必须写明 B/Q/KV、模型、硬件和软件版本。

实施记录与结果追加在本节后续条目；早期阶段 P0–P6 继续作为架构与安全合同。

### 阶段检查：基准已建立，性能验收未完成

新增 `benchmarks/multi_graph_replay.py` 与 CPU workload 合同、CPU 测试及硬件测试入口。默认保留 update；免 update 仅显式消融。KV 初态由模型真实 prefill 生成，再复制至各请求独立物理页。当前未改动生产 FIA 的 update 协议。

在 910B3、Qwen2.5-0.5B-Instruct、BF16 上的阶段结果：

| 用例 | 精度与测量状态 |
| --- | --- |
| B=7、C=8、Q=1、KV=64 | eager 重跑、FIA out、padding、update 单图/串行双图/并行双图均通过，最大绝对误差均为 0 |
| 同一 Q=1 用例的初步计时 | 单图 4.396 ms、串行双图 8.379 ms、并行双图 8.373 ms；仅一轮三个样本的 wall p50，是烟测而非正式性能验收 |
| B=33、C=64、Q=4、KV=512 | eager 重跑与 FIA out 一致，但 eager padding 对照最大绝对误差 2.5，27.4% 元素不满足 rtol=0.02、atol=0.02；在图捕获及双流测量前停止 |

Q=4 的失败在真实 prefill KV 下仍复现，不能仅归因于随机 KV；根因尚未确认，也不能据此认定是双流、驱动或官方 padding 路径的问题。该用例不输出有效性能报告，不放宽容差。下一阶段先定位基准中的容量变化精度问题，再运行完整 B/Q/KV 矩阵与独立 overlap trace。

本地隔离 CPU suite 309 项通过，新增文件 ruff check/format 与 diff whitespace 检查通过。硬件参数化 e2e 尚未整组通过，完整 format.sh 检查此前受缺少 pre-commit 阻塞，不能把此阶段称为全部完成。

### 后续进展：数值控制与大负载验证

普通 BF16 路径的逐层对照显示：第一层 QKV 投影已有小幅差异，之后差异逐层积累。启用官方 batch-invariant 模式后，B=33/C=64/Q=4/KV=512 的 eager、padding 与两种切分 replay 均得到最大绝对误差 0。这支持“batch 形状相关数值行为”的解释，尚不能锁定某一个算子；没有修改精度容差，也没有据此宣称普通模式已通过。

`--batch-invariant` 显式启用这一独立实验条件，JSON 记录实际开关和 attention operator。当前镜像安装了 `batch_invariant_ops`，启用后使用 `FusedInferAttentionScoreBatchInvariant`，并改变部分 MatMul/归一化路径。因此本节结果仅在这一模式内比较单图和切分，不与普通 FIA 的单图耗时混合计算加速比。所有主实验保留每层 update。

在 Qwen2.5-0.5B-Instruct、BF16、910B3、B=129/C=256/Q=8/KV=512、lane=65+64 上，每个模式测量五轮，每轮二十次单次 replay。下表为五个轮次 wall p50 的中位数，精度对照全部通过：

| 输出合并方式 | 单图 padding | 串行双图 | 并行双图 |
| --- | --- | --- | --- |
| 完成后 `cat` | 24.957 ms | 20.750 ms | 18.631 ms |
| 图内独立 slice copy | 24.952 ms | 20.589 ms | 18.533 ms |

第一组切分并行相对同模式单图减少约 25.3% 延迟；串行切分本身已有收益，说明减少 padding 计算是重要组成部分。图内 copy 相比 `cat` 的初步改善仅约 0.5%，不足以宣称稳定优化，仍保留 `cat` 默认值。图内 copy 是基准内的实验实现，没有接入生产 runner。

独立 trace 的三个并行采样中，两张图的设备 kernel 时间区间重叠分别为 6.094/6.134/6.142 ms；三个串行采样均为 0。统计通过 captured Model Id 区分子图，排除 EVENT、MEM_WAIT、NOP 及 host 事件，对每图 kernel 区间取并集后求交集。它证明设备 kernel 区间重叠，不代表同类计算单元都并行，也不将 profiler 内时间当作性能样本。并行时每张图的 kernel busy 时间变长，存在资源竞争。

可复现命令（模型路径按机器实际位置设置）：

```bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/multi_graph_replay.py \
  --model /workspace/models/Qwen2.5-0.5B-Instruct --batch-invariant \
  --batch 129 --capacity 256 --query 8 --kv-length 512 \
  --protocols update --rounds 5 --repeats 20 --merge cat \
  --output /workspace/multi-graph-results/fixed-replay/reproduce.json
```

将 `--merge cat` 改为 `--merge captured --trace` 可验证图内输出拷贝和设备重叠；trace 在正式计时后单独采集。JSON 区分 `running`、`failed`、`passed`，失败报告不包含有效性能样本，避免旧结果文件造成误判。硬件测试可通过 `python tools/run_multi_graph_replay_tests.py --replay-model /workspace/models/Qwen2.5-0.5B-Instruct` 独立运行，覆盖 Q=1/2/4/8、两种输出合并和两种更新协议。

独立进程与边界对照均已补测（五轮、每轮二十个样本，保留 update，以下均为 `cat`）：

| B / C / Q / KV | 单图 | 串行双图 | 并行双图 | 结论 |
| --- | --- | --- | --- | --- |
| 129 / 256 / 8 / 512，独立进程复测 | 25.162 ms | 20.902 ms | 18.752 ms | 相对 padding 延迟降低 25.5%，复现收益 |
| 129 / 129 / 8 / 512，无 padding | 16.289 ms | 20.965 ms | 18.821 ms | 切分慢于无 padding 单图，不能宣称固有并行优势 |
| 65 / 128 / 4 / 512 | 10.737 ms | 12.387 ms | 11.693 ms | 此负载切分仍慢，padding 浪费不足以覆盖额外成本 |

复测 trace 的三个并行 kernel 重叠为 6.037/6.185/6.092 ms，串行仍为 0。普通非 batch-invariant 模式的 B=129/C=256/Q=8 仍在 eager padding 对照失败，最大绝对误差 4.0，35.7% 元素超出固定容差；失败报告已保存，不能使用它的性能。

本地和容器 CPU suite 均为 314 项通过；A2 硬件参数化测试 8 项通过（Q=1/2/4/8 × cat/captured，每项包含 update/frozen × single/serial/parallel）。新增图捕获校验要求每张图具有与 Qwen2 attention 层数相同的 FIA task，并追加一项 Q=8 硬件回归通过；该模型每图 24 个，单图每步 24 次更新，双图每步共 48 次。

完整 `format.sh ci` 已安装 pre-commit 后重试，但在 actionlint 环境安装阶段遇到 `BadZipFile: File is not a zip file`，未完成全量检查；新增文件的独立 Ruff、Markdown 和 whitespace 检查通过。

下一步重点是保留真实长度 update 的前提下，缩短双图的 48 次 host 更新链并优化子图规模；普通模式仍需独立数值验收。此结果不代表动态长度服务、生产推测解码或普通 FIA 已获得同等加速。

### 普通 FIA：执行正确性与跨形状数值诊断分开

此前的跨形状 hidden-state `allclose` 失败不能直接归因于 FIA 错误。普通模式同形状重复执行一致，batch-invariant 又同时改变 MatMul 和 attention 等实现，尚不足以定位唯一算子。后续普通 FIA 实验采用显式 `--qualification same-shape`：每张图严格对照相同形状的 eager，split 对照各 lane eager 输出按原序拼接；跨形状误差仅作描述。默认 `cross-shape` 检查不变，生产准入不变。

报告记录 `qualification`、`cross_shape_qualified`、各模式的最大绝对误差、relative RMS、平均 cosine 和超逐元素容差比例。`same-shape` 下 `status=passed` 只表示执行检查通过，不能解释为模型质量验收完成。可选 `--logit-check` 在计时外比较每请求最后一个 query token 的 top-1 一致率、平均 KL 和最大 logprob 差异；不运行采样或 draft，不预设接受率，也不以诊断结果自动放宽准入。

普通 FIA 的 B=129/C=256/Q=8/KV=512 数据（BF16、同一 Qwen2.5-0.5B、五轮各二十次 replay，全数保留 update）：

| 切分与提交方式 | 单图 padding | 串行双图 | 并行双图 |
| --- | --- | --- | --- |
| 65+64，逐 lane 提交 | 20.713 ms | 17.666 ms | 16.528 ms |
| 65+64，先提交两张 replay 再 update | 20.804 ms | 17.681 ms | 16.460 ms |
| 96+33 | 20.482 ms | 16.824 ms | 15.707 ms |
| 128+1 | 20.212 ms | 16.427 ms | 15.217 ms |
| 128+1，独立进程复测并检查 logits | 20.971 ms | 16.845 ms | 15.599 ms |
| 128+1，图内输出拷贝 | 20.629 ms | 16.259 ms | 15.041 ms |

128+1 两次相对同轮 padding 的延迟降幅为 24.7% 和 25.6%；均衡切分约 20.2%。不同进程的单图时间也存在波动，必须比较同轮比值；图内 copy 与 replay-first 暂不改为默认策略。每张图与同形状 eager 最大误差为 0。均衡 trace 的并行 kernel 重叠约 4.2 ms；主图尾图约 6.0 ms，串行对照均为 0。

较小的普通 FIA 负载 B=65/C=128/Q=4/KV=512：64+1 的单图/串行/并行分别为 9.620/10.070/9.904 ms；均衡 33+32 分别为 9.424/10.357/10.253 ms。主图尾图减少了切分成本，但仍未超过同轮 padding；不可将大负载的优势外推到该区间。

主图尾图复测的实际 batch eager 对照：padding relative RMS 约 1.324%，切分约 1.321%；最后 query token 的 top-1 一致率分别为 94.6% 和 96.1%，平均 KL 分别约 0.00138 和 0.00134。该输入是重复文本 prefix 加固定 token ID 后缀的合成负载，不能代表真实业务质量。结果表明该样本的切分数值偏移没有明显超过 padding 基线，不等于跨形状精度已经验收。

可复现普通路径命令：

```bash
VLLM_BATCH_INVARIANT=0 ASCEND_RT_VISIBLE_DEVICES=0 \
python benchmarks/multi_graph_replay.py \
  --model /workspace/models/Qwen2.5-0.5B-Instruct \
  --batch 129 --capacity 256 --query 8 --kv-length 512 --lanes 128 1 \
  --qualification same-shape --protocols update --logit-check \
  --rounds 5 --repeats 20 --trace \
  --output /workspace/multi-graph-results/native-optimization/reproduce.json
```

### 基于测量的优化空间与优先级

| 优先级 | 方法 | 当前依据与可行性 | 验证边界 |
| --- | --- | --- | --- |
| 1 | 用实际子图成本选切分点，保留主图尾图与非均衡候选 | 128+1 在该负载优于 65+64，且串行子图总成本也下降；现有基准可直接控制 lane，落地成本低 | 不将请求数均分当作时间均分；跨 B/Q/KV 搜索，不把 Q=8 结果写入生产 Q=1 成本表 |
| 2 | 减少两图对计算与带宽的竞争，按关键路径调整提交 | 并行时每图 kernel busy 时间增加；replay-first 初测仅约 0.4% 改善，没有稳定证据 | 必须看每图耗时与 kernel overlap，不以 stream 数或单纯重叠量作目标；资源限额 API 仍需当前驱动验证 |
| 3 | 缩短必要 update 的 host 提交链 | 两图 48 次 host attention 调用在 profiler 中累计约 3 ms，单图约 1.7 ms；批量 C++ 提交可研究，Python 交错已有负面历史结果 | 这些调用与设备执行重叠，3 ms 不是可直接兑现的收益；不得取消真实 seq_len 更新、ExternalEvent 或缓冲区生命周期约束 |
| 4 | 图内输出拷贝与中间数据搬运融合 | 图内 copy 已实现并通过精度，收益仍受噪声影响；两图 kernel 中 copy 类累计约 0.8–1.2 ms | 图内 copy 仍有拷贝成本；进一步融合 QKV split/RoPE/KV 写入需独立 kernel、页表/slot、因果性和别名测试，不能直接套用 MLA 专用算子 |
| 条件项 | 若需要 batch-invariant，优化其带 bias GEMM | 大图 Addmm 从普通约 0.59 ms 变为 Triton invariant 约 6.51 ms，而 attention 约 5.79→5.01 ms；这是该模式的明确热点 | 属于独立算子替换实验，需要验证 bias 累加精度与 batch invariance；普通 FIA 优化不依赖它 |

上述 kernel 分类是独立 profiler 内的累计 duration，包含并发时的重叠，不可相加当作 wall time。当前优先交付普通 FIA 的切分成本证据和可复现实验，不扩大到动态服务、V2 或分布式。

本轮校验：314 项 CPU 测试通过；新增普通 FIA same-shape/logits/replay-first/captured 组合与既有 Q=8 batch-invariant 严格检查共 2 项硬件回归通过。普通模式完整性能用例均单独通过同形状 eager 检查。新增检查没有改变原有跨形状测试的默认要求。

### 结构性原型：共享 Dense，只切分 Attention

新增 `benchmarks/multi_graph_shared_dense.py`。这一实验改变切分边界：每层先按真实 B×Q 一次执行 LayerNorm、QKV 投影和 RoPE，再按完整请求分成两路 Attention，写入同一输出张量的互不重叠切片，汇合后一次执行 O 投影、残差归一化和 FFN。目标是保留请求切分，同时避免 Dense 跟随请求一起碎片化。

与整层切分相比，真实 token 的 Dense FLOPs 没有减半；减少的是小矩阵执行、重复 kernel 调度和额外输出合并。相对容量 C×Q 的 padding 图，还省去假 token 的 Dense 计算。Attention 对真实 host seq_len 的依赖保持不变，每层两路 FIA 均保留 task update。

本轮实现八个对照，均使用普通 FIA、相同模型权重和真实 prefill KV：

| 模式 | Dense 执行方式 | Attention 执行方式 | 每次回放提交图数 | FIA update 数 |
| --- | --- | --- | --- | --- |
| single_padded | 容量 batch | 单路 | 1 | 1 |
| single_exact | 真实 batch | 单路 | 1 | 1 |
| full_serial / full_parallel | 两份子 batch | 整层两图串行 / 并行 | 2 | 2 |
| shared_serial / shared_parallel | 真实 batch，共享前后段 | 两张 Attention 图串行 / 并行 | 4 | 2 |
| joined_serial / joined_parallel | 真实 batch，共享前后段 | 同一捕获图内两分支串行 / 跨流 | 1 | 2 |

joined_parallel 将分叉、两路 Attention 和汇合录入同一张外层图，不再逐层由 Python 提交四张图。请求边界切分仍然存在，但它不是“两个独立整模型图”的原执行形式。当前驱动与软件组合已通过普通事件跨流捕获探测，随后通过实际 FIA task update 的单层实验。

最终对照默认 `--merge captured`，整层切分也使用此前已有的图内输出合并，不能只用带外部 `cat` 的较弱基线计算新结构收益。`--merge cat` 保留为显式消融。此处整层双图合并 hidden 和 residual 两个输出；整模型旧路径可以让 residual 留在各 lane 内继续后续层，因此该单层合并成本不能逐层外推。

实现中的必要约束：

- 两路 FIA workspace 按 lane 隔离，并断言两个不同存储地址；不能直接沿用同层共享 workspace key。
- 同层 provider 会注册两次任务，参数 source 按真实注册顺序和 lane 分别绑定长度、页表，不能将整 batch 的元数据重复用于两路。
- 图持有输入、元数据和闭包，保留跨图张量、输出、pool 和事件的生命周期。初版遇到设备无效地址；补齐 ownership 后，后续正确性、输入变化与性能测试未复现该错误，不据此归因于驱动。
- 捕获图的各条路径对照各自同形状 eager，检查 hidden 和 residual 两个输出；单路直接 backend 另与官方 Qwen2DecoderLayer 比较。改变固定存储中的 hidden 值后重新生成 eager 对照并检查全部八种 replay，防止旧中间值被误复用。
- 保留跨结构的误差诊断，但 `cross_architecture_qualified=false`；不将同形状通过写成跨形状或全模型精度通过。

测量仅覆盖 Qwen2.5-0.5B-Instruct 第一层，输入为 embedding，BF16、910B3/A2。每种模式五轮、每轮三十次，交替正反顺序；表格使用各轮 wall p50 的中位数。时间包含 Python 图提交、必要 FIA update、事件依赖、执行和输出汇合，排除模型加载、prefill、capture、KV 恢复及 profiler。长度与地址固定，尚未包含逐步 metadata 生成和解析。

| B / C / Q / KV | lanes | padding 单图 ms | exact 单图 ms | 整层并行＋图内合并 ms | 共享 Dense 图内跨流 ms | 相对同 lanes 整层并行降幅 |
| --- | --- | --- | --- | --- | --- | --- |
| 129 / 256 / 8 / 512 | 65+64 | 1.180 | 0.894 | 1.089 | 0.968 | 11.1% |
| 同上，独立进程复测 | 65+64 | 1.190 | 0.908 | 1.094 | 0.988 | 9.7% |
| 129 / 256 / 8 / 512 | 128+1 | 1.179 | 0.898 | 0.966 | 0.921 | 4.7% |
| 65 / 128 / 4 / 512 | 33+32 | 0.793 | 0.715 | 1.021 | 0.847 | 17.1% |
| 129 / 256 / 8 / 2048 | 65+64 | 1.544 | 1.114 | 1.226 | 1.123 | 8.5% |

共享 Dense 相对 padding 在两个大 batch 短 KV 均衡试验中减少约 17.0–17.9% 延迟，在长 KV 试验中减少约 27.3%；但 B=65/Q=4 仍比 padding 慢约 6.7%。在已优化的 128+1 对照上，4.7% 的单次进程结果尚不足以认定稳定优势。尤其不能把均衡方案的 9.7–11.1% 改善解释成相对旧方案所有切分点中最好的结果也改善这么多。

四张分段图的 shared_parallel 在第一组大 batch 实验为 1.324 ms，慢于 1.180 ms 的 padding。改为一次图回放后降至 0.968 ms。joined_serial 同组为 0.968 ms、复测为 0.988 ms，几乎与 joined_parallel 相同：当前有证据支持共享 Dense 与减少提交的作用，尚无稳定证据支持两路 FIA 并发本身提速。

短 KV 的初步 trace 表明：整层切分每层 8 次 MatMul，共享 Dense 后是 4 次；两路 FIA 未出现明显 kernel 区间重叠。因此不能把共享结构的收益全部归因于 attention 并行。四张分段图尽管设备计算减少，但 host 提交代价使其 wall time 更差；一次捕获图是此原型的重要组成部分。

最终 trace 的三个 joined_parallel 样本中，KV=512 时每路 FIA kernel 约 73–82 us，两次 host FIA 调用起点相隔约 260–277 us，两路设备 FIA 之间还有约 160–191 us 空档。KV=2048 时 FIA 约 215–254 us，空档缩至约 17–41 us；两个长度下采集到的 FIA overlap 都为 0。该现象与更新供给不及时相符，但 profiler 会放大 host 开销，尚不能排除设备资源约束，更不能将空档全部作为可兑现收益。

下一项性能实验应保持真实 seq_len 更新，尝试把两路 task update 的 begin、FIA 调用、end、event record 放到更紧凑的 host 提交链中，比较非 profiler 的 wall time，并再验证 FIA overlap。仅在这项实验后才能判断 C++ 批量提交是否值得接入。另一个条件方向是共享 QKV、后段按就绪分支执行 O/FFN 的流水：用一部分 Dense 分块代价交换更少的汇合等待，必须以多层 trace 中的等待占比为依据。

复现命令：

```bash
source /etc/profile.d/ascend-multigraph.sh
VLLM_BATCH_INVARIANT=0 ASCEND_RT_VISIBLE_DEVICES=0 \
python benchmarks/multi_graph_shared_dense.py \
  --model /workspace/models/Qwen2.5-0.5B-Instruct \
  --batch 129 --capacity 256 --query 8 --kv-length 512 --lanes 65 64 --merge captured \
  --rounds 5 --repeats 30 --trace --output /workspace/shared-dense.json

ASCEND_RT_VISIBLE_DEVICES=0 python tools/run_multi_graph_replay_tests.py \
  --replay-model /workspace/models/Qwen2.5-0.5B-Instruct -k shared_dense
```

边界与下一阶段：单层每次末端同步，不能将延迟乘以层数当作整模型结果，也不能与历史整模型 15–21 ms 的数据直接比较。整层双图能够跨层推进，共享 Dense 则每层需要汇合；该差异可能改变全模型排序。下一阶段需把多个 decoder layer 的残差传递及分叉/汇合放入同一外层图，再比较完整 target replay。动态 seq_len、服务接入、其他模型、TP/V2 均不在本轮通过范围。

本轮验证：318 项隔离 CPU 测试通过；2 项共享 Dense 硬件测试覆盖 Q=1/Q=8、cat/captured 合并、非均衡分支与输入值变化，另有 1 项既有普通 FIA 整模型回归通过。五组最终性能实验的所有同形状检查（hidden 和 residual）最大绝对误差均为 0，所有结果的脚本 SHA256 与提交源码对应。Ruff、Markdown、拼写、修改文件的禁止导入及 diff 检查通过。完整 `format.sh ci` 已执行，但 Windows 缺少 `/bin/bash`、shellcheck、python3 命令，以及已有 `device_length_decode.py` 直接 import Triton 的检查失败仍阻止全套通过；这些失败未作为本次功能通过的依据。

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
