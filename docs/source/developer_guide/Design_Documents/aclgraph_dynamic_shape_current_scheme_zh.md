# 动态 Shape ACLGraph 推理优化：当前方案说明

更新日期：2026-10-19。

本文说明目前真正实现了什么、现有证据支持什么，以及下一步为什么优先修改
attention consumer。详细代码与原始结果分别见文末链接；本文不替代各阶段的
冻结协议和审计记录。

## 1. 目前的主线是什么

目前主线已经从“把一个 batch 切成多张图并行执行”，转为：
**在默认 padding 图机制下，由设备端真实描述符产生共享的原生 attention 计划，
再通过完整 request/head 归属改善 attention 调度与执行。**

二者都服务于动态 shape 推理，但属于不同层次的优化：

| 层次 | 优化对象 | 当前状态 |
| --- | --- | --- |
| 图层 | 将动态 batch 分成多个已有图，并用多流并行回放 | 前期探索方向；当前结果不证明其优于 padding。 |
| 计划层 | 根据每步真实长度和页表，决定各核承担哪些 attention 工作 | 当前已实现受条件保护的 whole-row 原生策略。 |
| 算子层 | 改善同一核内部的分块、累加和数据搬运 | running-O UB 驻留补丁正确性通过；初轮有限改善在反向配对中未复现，暂不扩大。 |

当前仍保留默认 capture sizes，图执行完整的合法 padding 容量。
没有为某个测试 batch 增加专用 size，也没有通过删除物理 padding 制造优势。
未来如果重新比较切分和 padding，双方必须使用同样最优、合法的 attention
计划和 consumer，才能把收益归因于切分。

## 2. 为什么固定图仍需要每步真实信息

ACLGraph 固定的是算子结构、缓冲区地址及允许的物理形状。
每步实际请求数、KV 长度、请求归属和缓存页路由仍可能变化。
只保留一个固定 shape，并不能让 attention 忽略这些变化。

以当前测试的 Q4 为例，每个请求有四个 query token。
原始 B98 记录映射到物理容量 400，即 100 个物理请求位置；
B128 映射到容量 512，即 128 个位置。
这里的 400/512 是物理 token 数，不是 batch 数。
padding 位置的 KV 长度为零，attention 必须真实写出零结果。

因此，方案区分两类信息：

- **图的静态容量**：沿用默认桶，保持捕获地址、物理 extent 和图生命周期稳定。
- **当前精确描述符**：保留真实 KV 长度、Q 边界和页表，不能用桶上限替代。

legacy FIA 路径依赖的真实 host 长度 update 仍必须执行。
我们没有以省略必要 update 的方式测量 legacy。
sink 路径则把真实长度交给设备端原生 producer，每次 replay 都重新生成计划；
必要的描述符准备、H2D、producer 和最终同步都计入当前完整 wall 端点。

## 3. 已实现的执行链

当前实验的一步可概括为：

```text
本步真实请求、KV长度、页表
        │
        ▼
固定地址描述符填充及必要H2D
        │
        ▼
图内原生AICPU producer：根据当前描述符重建计划
        │
        ▼
稳定的int32[1024]原生计划
        │
        ├── attention occurrence 0：私有Q/K/V与workspace
        ├── attention occurrence 1：私有Q/K/V与workspace
        ├── ...
        └── attention occurrence 15：私有Q/K/V与workspace
        │
        ▼
全部执行完成及最终join
```

十六次 attention 顺序执行，共享兼容的只读计划。
这不是十六个流并行，也不意味着共享权重、KV 内容或可变 workspace。
共享计划的前提是这些 occurrence 的长度、页路由语义和几何配置相容。

当前已验证目标是 A2 路径上的 BF16/TND、Q4、Q12/KV2、D256、page128、
sparse mode 3，使用实际 20 个 Cube/40 个 Vector 核限制。
为适配 D256，隔离包缩小了 MM1/MM2 的子块并增加容量检查，保留原 L1 布局。
这属于具体目标的工程适配，不等于官方已全面支持所有 D256 组合。

### whole-row 策略具体改变什么

上游 producer 已有代价估计和切分搜索，并能把一个 head 的 KV 工作拆给多个核，
之后通过 FlashDecode（FD）归并部分结果。

我们的策略把完整的 `(request, KV head)` 作为不可再拆的工作单元。
它沿用原生代价权重，在连续的工作单元序列上做 minimax 分段：
目标是尽量降低最重核的估计工作量，然后输出合法的原生端点。
构造的新候选将 FD 和归并字段置零，完整覆盖所有物理位置；
不满足几何、容量或端点合法性时，回退到上游策略。

这带来一个可利用的条件：同一个 request/head 的累加生命周期可以归属同一核。
代价是完整 head 无法继续拆小，长请求可能使负载更难均衡。
现有 whole-row 与 no-FD 同时变化，当前结果不能单独证明收益全部来自取消 FD。

**当前 producer 每次都完整重建计划，不是增量计划维护。**
已有 Python 增量原型只是 CPU 假设验证：高 merge 代价下，127 个后续 step
中有 126 个回退，且 Python 路径慢于其完整重建对照。
该负面结果保留，没有接入原生 NPU producer。

## 4. 现有性能证据究竟到哪一层

目前实验使用十六组私有、固定、驻留的 synthetic Q/K/V。
动态轨迹改变真实描述符和页路由，但没有加载完整模型，
没有根据模型输出生成新 Q/K/V，也没有持续写入真实生成 token 的 KV。
因此，下表是 **attention 链** 的结果。

| 实验 | whole 相对 upstream 的结果 | 可以支持的结论 |
| --- | --- | --- |
| Stage 6：四条固定记录 | 完整 wall 均值下降 5.665%；有一轮退化 | 小范围策略收益存在，不能代表完整动态轨迹。 |
| Stage 7：原始 128-step 时间顺序轨迹、17 桶 | 均值下降 1.788171%，约 89.354 µs | 当前最完整的动态描述符轨迹证据，收益有限且不稳定。 |
| consumer 诊断：两条原始记录、六个 fresh process | B98/B128 分别下降 9.418%/10.095% | 重复定点协议下观察到更大收益，不能替换 Stage 7。 |
| catalog bridge：同样两条记录、保留 2 桶 | 分别下降 6.086%/5.401% | 与 17 桶共同用于排查图目录上下文。 |
| catalog bridge：同样两条记录、保留 17 桶 | 分别下降 7.580%/8.316% | 本轮不支持桶数或保留内存本身造成 Stage 7 收益减弱。 |

Stage 7 保留了全部 3072 次 timing 和 1536 对匹配样本。
四轮均值降幅为 1.957489%、2.083247%、**-0.117017%** 和 3.269628%。
整体 P99 从 6.550212 ms 增加到 6.631752 ms。
四轮中一轮变慢，尾延迟也变差，不能只报告均值收益。

Stage 7 的独立正确性阶段通过 2048 个 step 检查和 6144 个 FP32 抽样行比较，
最大绝对误差为 0.0003158003091812134。
这是独立 qualification pass 的抽样，不是对每个 timed replay 的逐元素验证。

bridge 保持相同回放记录，只改变捕获并保留的图目录。
未使用的图没有额外回放；144 个 FP32 抽样行通过，全部 80 次 timing 保留。
每个组合仅一个 fresh process，进程顺序、时段、设备状态和分配位置仍然混杂。
因此不能说“更多图更快”，也不能确认 replay 历史就是收益差异的原因。
动态证据仍以 Stage 7 的 1.788171% 为准。

这些结果目前都不支持以下结论：

- 已取得完整模型端到端或 serving 吞吐收益。
- 切分并行优于默认 padding。
- 已实现原生增量计划、通用多流调度或独属于切分的新算法。
- 只要使用 Q4 或更大模型，就能稳定复现较大的点测收益。

## 5. 为什么下一步改 consumer

当前诊断已经把目标缩小到算子执行内部。
分别 profiling 的十六次 attention kernel 总时长如下：

| 物理容量 | whole 比 legacy 多用的 kernel 时间 | sink kernel 之间的间隙总和 | whole producer 时间 |
| --- | ---: | ---: | ---: |
| 400 | 269.705 µs，慢 7.675% | 37.711 µs | 74.128 µs |
| 512 | 370.147 µs，慢 8.226% | 37.444 µs | 112.112 µs |

producer、间隙、kernel 和 host 区间可能重叠，不能相加重建 wall。
上表也不保证所有差额都能消除：legacy 和 sink 的实现及 tiling 不同。
但它说明继续削减 launch 间隙的预算较小，consumer 内部更值得做源码消融。

小规模硬件计数器进一步发现，whole 相对 legacy 的 Vector scalar、AIC MTE2
和 fixpipe 时间增加，比 MAC 时间增加更明显。
这只是定位线索；流水线可以并发，不能把这些时间相加，
也不能据此认定 HBM 带宽饱和或承诺某个可回收百分比。

### 下一项源码消融：减少 running-O 往返

源码位置是固定版本 `bec19b8dea1e` 的
`fia_block_vec_nonquant_sink.h::DealBmm2ResBaseBlock`。
现有路径每个 S2 分块读取新的 MM2 partial；非首块还要从 GM workspace
读回上一块的 running-O，做缩放和相加；非末块再通过 UB staging 写回 GM。
running-O 就是在线 softmax 过程中持续更新的输出累加值。

有两个可独立验证的修改：

1. **先去除中间 UB staging**：仍保留 GM 累加，只检查多余复制能否安全移除。
   必须等待 MTE3 完成，保证输入缓冲在搬运结束前不被复用。
2. **再尝试 running-O 在 UB 驻留**：以完整 head 的生命周期为单位，
   减少跨 S2 的 GM 读回和写回，保留 MM2 的既有 workspace 交接。

目前优先实施第二项；去 staging 仍是未实施的备选消融。
running-O 补丁和隔离构建工具已实现；两记录结果及复现限制见本节后半部分，
尚不能把它计入完整动态轨迹或模型收益。
受限实现解决以下约束：

- Q4/GQA6 对应 24 行，现有两侧 AIV 按 16/8 行分工；
  D256 的 FP32 累加器分别需要 16/8 KiB，不能按 12/12 行估算。
- 三任务缓存中的 Vec2 实际按创建顺序串行消费，S2 是最内层。
  因此独立于 Vec1 的一个累加器足够；补丁仍检查 head 身份、loop 连续性、
  AIV 行区间及首末块，防止读取未初始化状态。
- 原静态 UB 为 184 KiB/AIV。把 outputQue1 从 32 KiB 缩到 16 KiB，
  再分配独立 16 KiB 累加区，总量保持 184 KiB。
  Vec1 的 BF16 输出最多 16 KiB，最终 BF16 输出最多 8 KiB；
  缩容模式的 FP32 fallback staging 限制为 16 行，最多 16 KiB。
- 非末块删掉原输出队列和 GM store 后，显式保留 V→MTE3 依赖，
  防止原 cross-core 完成信号过早释放 MM2 workspace。
- 不改变 softmax 递推、真实长度、页路由、MM2 交接和最终输出格式。
  FD、容量或生命周期不满足要求时保留原路。

快路径目前仅限定 A2、BF16/TND/page128、Q12/KV2、D256、max-S1=4、
preload=2、S2 块不超过 512、无 sink/RoPE 的组合。
这里的 max-S1=4 是计划中的单请求 Q 上限，物理 token 容量仍为 400/512。
不满足静态条件的形状保留原 32 KiB 队列和 workspace 路径；
FD 等不满足 task 条件的 head 继续走 workspace。
合法且保持 head shape 的状态失配会先同步 spill 到原 workspace 后回退；
这不保证违反原生合同的描述符在 head 中途改变 Q/M 形状仍然安全。

源码补丁以完整 SHA 和 anchor 拒绝版本漂移，并支持同源宏 0/1 对照。
构建工具只在新实验目录编译一个 BF16 binary，给 upstream/whole 两种 producer
装入同一 consumer，不安装系统包，也不修改既有 vendor。
本轮小实验只编译目标路径的 `104000000010222223` 和
`104000000010322223` 两个 tiling key，并严格核对产物中的 key 集合。
这是受限实验 binary；源码保留 fallback 不代表该 binary 包含其他 shape 所需的模板。
whole 计划中的 FD head 数可以为零，但 host 仍可能选用 FD-capable 模板，
因此不能根据策略名字删除该编译变体。
源码、CPU 测试和静态预算通过并不等于设备正确性或性能通过。

**资格按实际 plan 和生命周期判断，不能按 `whole` 策略名字判断。**
如果 upstream 计划也满足条件，必须获得同一 consumer 快路径。
实验需要分别看 consumer 收益、原生调度收益及二者的交互，
避免把只给 whole 使用的算子优化误报为 whole 调度优势。

先用原 B98/B128 两条记录做小实验，保留全部 pre/post 正确性门槛和完整 wall，
profiling 与 timing 分开。只有端点收益与相关执行证据一致，才扩大动态验证。

### 本轮源码消融结果

`v3` 已完成四个独立进程，顺序为 control/upstream、resident/upstream、
resident/whole、control/whole。每组使用相同的原始两条记录、seed1703、
16 份私有 Q/K/V、5 次 warmup、每容量 10 次计时和一次单独 profile。
完整 wall 包含描述符准备、必要 H2D、真实 producer、replay 和最终 join。
每组仅一个 fresh process，重复 replay 不等于独立进程复现。

| 调度策略 | 原始记录 / token 容量 | control 均值 µs | resident 均值 µs | 观测均值下降 | 单次 profile 的 16 个 attention 总计减少 µs |
| --- | --- | ---: | ---: | ---: | ---: |
| upstream | B98 / 400 | 4517.510 | 4357.540 | 3.541% | 101.74 |
| upstream | B128 / 512 | 5673.378 | 5543.449 | 2.290% | 127.86 |
| whole | B98 / 400 | 4250.731 | 4186.755 | 1.505% | 49.02 |
| whole | B128 / 512 | 5609.524 | 5311.093 | 5.320% | 73.00 |

**最后一行不能解释为稳定的 5.32% 改善。** control/whole 的 B128 有一次
7896.423 µs 尖峰；该组中位数为 5362.983 µs，resident 为 5309.507 µs，
对应约 0.997% 的中位数改善。全部十次观测保留，没有删异常值。
profile 的 73.00 µs 与均值节省 298.43 µs 也不相符，不能全部归因于 consumer。
profile 本身每组每容量仅一次，不能作为稳定延迟估计。

本轮 80 次计时、144 行独立 FP32 抽样和 24 次完整计划检查通过独立审计；
抽样最大绝对误差为 0.0002998188。相同 policy/capacity 的 control/resident
在 pre/post 中，16 份完整 BF16 输出哈希全部一致，1024-word 计划也完全相同。
因此本轮没有观察到 consumer 修改引入数值变化；这不代替全部动态形状正确性证明。

实际计划中，upstream 的完整 head 资格占比为 93.88%/95.31%，whole 为 100%。
这是根据计划和源码推导的资格，不是设备执行计数。
upstream 同样得到 consumer 改善，所以片上累加不是 whole 策略独有的收益。
双方均用 resident consumer 时，whole 相对 upstream 的本轮 wall 均值下降为
3.919%/4.192%；仍受固定进程顺序等混杂因素影响。

目前判断是：该补丁值得保留为受保护的实验候选，但没有显示大的独有收益，
也没有证明 GM 带宽饱和或取得完整模型收益。它减少原 running-O 搬运及相关依赖，
同时改变队列预算、状态检查和同步，不能将全部差值归为单一 GM 流量因素。
因此执行了下文的反向 whole control/resident 配对，先区分约 1% 的
实际改善与进程波动。当前不扩大模型或完整性能矩阵。

原始结果与两份分析分别保存在
`benchmarks/results/multi_graph_structural/fia-sink-running-o-v1/`、
`fia-sink-running-o-independent-audit.json` 和
`fia-sink-running-o-primary-recompute.json`。首次编译失败、第二次主动停止的
全变体 CPU 编译另行保存，均没有 NPU 计时，不计入上述数据。

### 反向配对：没有复现稳定收益

随后复用原 `v3` binary、输入和 frozen probe，只运行 whole 的 control→resident
两个独立进程，反转初轮 whole 的 resident→control 顺序。
warmup、10 次计时、单独 profile 和前后检查协议均不变，没有重新编译。

| 原始记录 / 容量 | control 均值 µs | resident 均值 µs | 均值下降 | 中位数下降 | 单次 profile 的 attention 总计减少 µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| B98 / 400 | 4248.401 | 4271.701 | -0.548% | -0.723% | 4.841 |
| B128 / 512 | 5333.839 | 5411.841 | -1.462% | -1.331% | -56.381 |

负值代表退化。初轮的约 1% 典型改善未能复现，不能将 running-O 宣称为稳定优化。
两轮分别报告，不把它们简单合并成一个看似精确的 replay 平均值。
补丁继续作为隔离候选保存，目前不推广到正式完整模型路径。
本次追加 40 次计时、72 行 FP32 抽样和 12 次完整计划检查通过独立审计，
同 variant 与初轮的全部输入、输出哈希、计划和 native artifact 均一致；
NPU0 结束后空闲。原始结果、审计工具和故障注入检查位于
`fia-sink-running-o-v1/confirmation/`。

这一结果也要求调整瓶颈判断：根据真实描述符，单次 attention 逻辑 K/V 读取
约为 193.34/271.27 MB，而 whole 可避免的 running-O workspace 逻辑读写
约为 13.27/18.97 MB，即前者的约 6.9%/7.0%。这些都是逻辑工作量，
不包含 cache、重复搬运或实际带宽测量；但它说明仅消除 running-O 往返的收益上限
需要谨慎判断，不能从“有 GM 往返”直接推导“是主瓶颈”。
后续若继续 consumer 优化，应先定位更大的 paged-K/V 搬运、布局转换及 Cube/Vector
交接关键路径，再决定是否值得做同步与分块大改，而不是继续为此补丁扩展性能矩阵。

## 6. 与已有工作的区别，以及毕设能写什么

下表依据仓库已有的来源核查记录，详细源码锚点见
[贡献边界审计](aclgraph_fia_sink_contribution_scope.md)。
本文没有开展新的全面查新，也不据此声称研究优先权。

| 已有工作或机制 | 我们复用的内容 | 目前可以明确区分的实现 |
| --- | --- | --- |
| 官方 FIA sink PR 15336 | 设备真实长度、AICPU plan/run、1024-word ABI、兼容层共享计划、原生 cost 搜索与 FD | A2 D256 适配；完整 request/head 的连续 minimax 策略；合法端点输出与回退。 |
| FlashInfer plan/run 与 attention 调度 | 规划与执行分离、跨兼容层复用计划属于已有原则 | 当前使用 A2 AICPU producer 和既有 endpoint ABI；不是 kernel-owned ticket 调度。 |
| FlashInfer balanced GQA 等设备调度 | 设备长度驱动、拆分长请求及归并已有先例 | 当前限定 A2/D256/GQA6/page128，采用完整有序 request/head；硬件差异本身不证明算法新颖。 |
| 在线 softmax 与片上输出累加 | 递推累加和保留片上中间值是已有 attention 方法 | 待验证的是与原生 whole-head 归属、UB 容量和在途 task 生命周期结合的受保护实现。 |
| 固定图池、固定缓冲和默认桶调度 | 属于 graph 推理基础设施 | 当前贡献是具体适配和可复核实验，不是发明固定地址或 padding 图。 |

当前毕设可以扎实描述三部分：

1. **问题与约束**：动态真实长度、固定图容量、原生 ABI、A2 片上存储与流水线
   生命周期之间的关系，以及为何不能省略必要 update 或随意添加 size。
2. **已实现机制**：受条件保护的 whole-row 原生调度、D256 适配、严格覆盖与回退。
   完整 head 所有权如何改变 FD 与负载均衡，是可解释的机制差异。
3. **证据与边界**：相同 consumer 对照、真实 producer-inclusive 端点、
   默认桶动态轨迹、正确性抽样、独立审计、负轮次和 P99 退化。

下一阶段可以继续检验“调度归属与 consumer 数据驻留的联合设计”。
其价值必须来自与上游同等 consumer 优化后的比较，不能把既有片上累加方法本身
写成首创。目前的小实验观察到有限 consumer 改善，但 upstream 也能受益，
尚不足以形成稳定联合优势或毕设级新颖性结论。

## 7. 从当前小实验到完整模型的门槛

顺序是先验证 consumer 修改，再判断是否值得接完整模型，避免立即扩大模型矩阵。
若小实验有效，仍需要保留原 17 桶和 capture 顺序，区分时间顺序回放历史与
进程/设备波动；两条记录的较大收益不能直接带入完整动态轨迹。

完整模型接入已有 [Stage 8 协议](aclgraph_fia_sink_stage8_protocol.md)，目前只是协议，
没有实施模型 sink adapter，也没有 Stage 8 NPU 结果。
计划在 Qwen3.5-27B TP2 的真实 QKV、KV 写入、GDN 状态、残差和通信路径上，
依次验证：

- 三个原始 step 的真实模型正确性与关键路径预算。
- 一次完整 128-step 连续状态资格验证。
- 通过前两项后，再做正式的三对照性能实验。

三对照必须分开解释：

| 对照差值 | 实际回答的问题 |
| --- | --- |
| `legacy_pad` → `upstream_sink_pad` | 采用设备计划/共享计划等既有 sink 机制的整体影响。 |
| `upstream_sink_pad` → `whole_sink_pad` | 同一 sink consumer 下，原生调度策略的独立影响。 |
| 将来的 padding → split | 双方拿到同样最优方案后，切分本身是否仍有收益。 |

真实模型端点必须计入必要准备、真实 KV 写入、producer、输出适配、TP 通信及
最终 join。attention 链的 89.354 µs 不能直接乘到模型层数，
也不能把当前 attention 百分比称作完整推理加速。
若真实关键路径可暴露的收益不足，应报告预算限制并停止扩大该路线的性能矩阵。

## 8. 详细记录入口

- [贡献边界与已有工作](aclgraph_fia_sink_contribution_scope.md)
- [Stage 5：原生 whole-row 策略](aclgraph_fia_sink_stage5.md)
- [Stage 6：受限 attention 链对照](aclgraph_fia_sink_stage6.md)
- [Stage 7：完整动态描述符轨迹](aclgraph_fia_sink_stage7.md)
- [当前 consumer 瓶颈与下一项消融](aclgraph_fia_sink_consumer_diagnostic.md)
- [2/17 桶上下文 bridge](aclgraph_fia_sink_catalog_bridge.md)
- [Stage 8：有条件的真实模型接入协议](aclgraph_fia_sink_stage8_protocol.md)
- [既有调研记录](aclgraph_dynamic_shape_research.md)
- [CPU 增量原型提案](aclgraph_incremental_attention_thesis.md)
