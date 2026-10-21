# GDN ready 剩余消费成本与 beta/gate 复用候选结果

本轮结论：**保留现有 compact FP32 ready，结束本轮内核优化。** 唯一候选通过同算法正确性检查，但没有超过重复波动的稳定净收益；未接入完整层或完整模型，没有新增 head 或整网性能数字。旧约 4.55% 的含 BF16 head 固定轨迹回放收益保留，原全部数值和质量失败也保留。

## 实验对象与复用范围

使用原 Qwen3.5-27B TP2 模型、910B3、CANN 9.1.0、torch 2.10.0+cpu、torch-npu 2.10.0.post4。内核输入 Q/K/V/beta 为 BF16，gate、base/log/products 为 FP32，HF32 关闭，FP32 matmul 为 highest。冻结原状态算法、递推与规约次序、AIC/AIV 资源、图目录和同步槽数。原 BF16 head 基线保持不变；本轮单层 recurrence 探针不执行 LM head 或 TP 通信。

先检查已有 `gdn-q4-tile-solve-v3-balanced.json`：B100 合成输入的闭合 recurrence 包含 separate、delayed、ready，分别约 1110.99、835.98、664.82 us，但不含预生成产品后的独立消费端点，也不是这三个真实层现场。因此复用入口和实现，未将旧耗时直接与新样本相减。

选定 layer 4，原 Q4 轨迹 step 7（真实 B115、默认容量 464 token，即 116 个 Q4 请求槽）、step 31（真实 B86、容量 352 token，即 88 槽），以及已有低负载轨迹 step 7（B26、容量 104 token，即 26 槽）。两个 rank 的真实层输入及连续 compact 状态均已保存。现场对应同一冻结 token、owner、接受前缀轨迹，从相同真实整网 prefill 连续推进至相应 step 后采集，没有生成评测或 128 步全量审计。

原 profile 未保存这三处完整输入，故补做原轨迹前 32 步、低负载前 8 步的不计时采集。这里的模型输入沿用原整网 fixture 的确定性 token，不是新增真实文本质量评测。采集是 eager 真实层路径，未声称与旧 profile 未保存的中间张量逐字节复现。

下列微测量把两个 TP rank 的本地算子输入分别放到同一物理卡 0 执行。它不是同时运行的 TP2 整层端点；表内两个 rank 输入的均值不能叫 TP2 critical-rank 时延。

## 第一阶段：成本判别

单位为 us。每个 rank 输入均测 3 轮、每轮 12 次，表内先取各轮中位数的中位数，再对两个 rank 输入取均值。

| 样本 | 纯 products | 预生成后 consume | products+consume 分离 | mixed delayed | mixed ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 Q4 step 7 | 161.990 | 695.885 | 832.815 | 508.205 | 395.400 |
| 原 Q4 step 31 | 127.260 | 535.955 | 635.055 | 403.925 | 322.695 |
| 低负载 step 7 | 63.220 | 187.080 | 219.535 | 153.255 | 134.045 |

这张表只计对应核心 kernel：Q/K FP32 张量预先分配，清零、提交和数据恢复在表外。每次计时前都使用同样的状态恢复、清零与完整 producer 预热，预生成产品不计入 consume-only。分离路径在计时内重新生产再消费，mixed 两模式也在计时内重新生产；没有把预生成产品当成部署方案。

资源核查：纯生产入口使用 20 AIC，纯消费绑定传入 40 AIV（host launcher 为 2×20），mixed 为 20 个 AIC 与其配对的 40 个 AIV。两侧使用同一消费 header、同一请求组遍历与三个 value-head 分工、相同产品形状 `[B,24,128,8]`、相同 FP32 products 与公开提交。源码和启动数量不等于实测占用率或指令时间；本轮没有测得纯 AIV 与 mixed 内逐指令调度完全相同。

**预生成后 consume 反而比 mixed ready 慢。** 因而不能把独立消费当作 mixed 的可达到下界，也不能把二者差值全部解释成同步或生产等待。相同预热只控制外部准备，不能保证 kernel 内部缓存和编译调度相同。这个结果否定了“直接用 consume-only 与 mixed 的差额估算可删同步成本”的做法，没有据此增加槽数或减少握手。

## 唯一候选：同请求组 beta/gate 复用

`q4_tile_solve.h` 的 `LoadInputs()` 原来每个 value head 都加载完整 4×24 beta 和 gate，将 beta 上转 FP32，并对全部 96 个 gate 做 Exp。每个 AIV 在同一请求组内处理三个 head，输入标量实际相同。

候选只把这份标量结果保留在独立 768-byte UB 中，按 `begin` 在下一请求失效；组内后两个 head 复用。全请求的加载、转换与 Exp 从 24 份减少为 8 份，即减少约 2/3 的这些特定操作。**这只是删除操作数，不是测得其占总耗时的 2/3。** Q/K/V 加载、Gram、Projection、递推、规约、输出转换和状态提交保持不变。

必须单独存储：`WriteLog()` 将原 `scalarBuffer_` 前部用于 DMA flag，直接缓存到原位置会破坏后续 head 的 gamma。候选避开该生命周期冲突。AIC 生产代码、两组在途、ready/consumed 握手及核数不变；delayed 和 ready 使用同一份新消费代码。

候选仅在独立构建目录中测试，源码、补丁和二进制身份归档于本轮证据。默认 kernel 源文件与旧运行库没有被覆盖。它是常规组内输入复用，不作为新的状态算法或独属于多图的原创机制。

## 第二阶段：正确性与增量计时

六个独立进程按 old→new、new→old、old→new 运行；每个进程覆盖两个 rank 的三个冻结现场，每模式 3 轮×12 次。计时包括 Q/K 转换、输出清零、mixed solve、完整 accepted commit 及最终设备完成；同样的恢复和 producer 预热在表外。这里是**单层 recurrence 的闭合端点**，不含层内 norm/投影/MLP、TP、上传或 LM head，不能称为完整层或服务收益。

完整状态校验覆盖输出、新日志/valid、接受后 base/log/valid、所有四个候选、inactive/null owner 和所有驻留 owner 的终末物化；旧/新、delayed/ready 的字节 SHA 全部一致，graph 的输出和接受态也与 eager 相同。共有 72 条 eager 生命周期核对及 72 次 graph 输出/接受态核对通过。它们不替代旧跨算法数值或质量资格。

单位 us。对三个进程对、两个 rank 输入的中位数取均值；收益按两侧均值之比计算，正值为减少耗时。

| 样本 | 旧 ready | 新 ready | ready 收益 | 旧 delayed | 新 delayed | delayed 收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 Q4 step 7 | 760.700 | 759.365 | +0.175% | 873.712 | 870.170 | +0.405% |
| 原 Q4 step 31 | 598.362 | 597.567 | +0.133% | 678.572 | 681.030 | -0.362% |
| 低负载 step 7 | 233.995 | 234.883 | -0.380% | 253.787 | 261.342 | -2.977% |

不是所有正均值都构成通过。下表保留逐 rank 的独立配对及旧实现重复波动；没有删掉退化轮次。预先冻结的准入要求是每个代表现场三对均改善、均值改善超过 1% 且超过旧实现重复跨度的两倍。跨度为三个旧进程中位数的极差除以其中位数，不是置信区间。

| 现场 | 配对 1 ready 收益 | 配对 2 | 配对 3 | 旧实现重复跨度 | 准入门槛 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original.step7.rank0 | -0.425% | +0.466% | +0.704% | 0.604% | 1.207% | 不通过 |
| original.step31.rank0 | -1.291% | +0.655% | +1.050% | 0.224% | 1.000% | 不通过 |
| low.step7.rank0 | -3.163% | -1.122% | +0.407% | 2.046% | 4.092% | 不通过 |
| original.step7.rank1 | -1.200% | +1.076% | +0.426% | 0.880% | 1.759% | 不通过 |
| original.step31.rank1 | -1.690% | +1.304% | +0.762% | 1.083% | 2.167% | 不通过 |
| low.step7.rank1 | +2.087% | -1.975% | +1.468% | 1.868% | 3.737% | 不通过 |

两个原轨迹样本聚合改善只有约 0.18%、0.13%，低负载聚合略退化。每一个 rank 现场都至少有一对退化；六个现场全部未达到冻结门槛。低负载 delayed 的退化更明显，说明不能把消费代码修改当成 ready 流水自身的收益。即使去掉人为的 1% 最低门槛，这些有正有负的配对也没有提供稳定净收益证据。

## 成本解释与去留

已证实的是：原实现有可以删除的重复 beta/gate 操作；这份特定复用代码语义正确；但在当前真实层输入和闭合 recurrence 中，删除这些操作没有变成稳定的净延迟收益。

候选增加了 768-byte UB、缓存键判断以及不同的缓冲布局。它们是实际代码差别，但本轮没有独占计时证据证明究竟哪一项抵消了所删工作；不能断言 Exp 已被完全隐藏，也不能断言新增分支或 UB 访问是主因。现有 PMU 仍不足以把剩余消费算术、生产供给、flag 等待拆成可独立相减的延迟。

据此**拒绝该候选，保留旧 ready，并结束本轮内核优化**。未运行完整层或完整模型增量实验，未更改在途槽数、握手、精度、状态算法、图目录或路由器。结论不外推成“ready 已无任何性能空间”，但这条组内 beta/gate 复用没有提供足够的继续投入依据。本轮不枚举布局、分支或 Exp 变体。

## 证据、源码与工程检查

- 基线源码：`benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/q4_tile_solve.h` 的 `LoadInputs`、`Solve`、`WriteLog`；同目录 `tile_pipeline_kernel.cpp` 的 `ConsumeSolve` 与 launch；`torch_binding.cpp` 的 `consume`。
- 采集入口：`benchmarks/multi_graph_gdn_ready_consumer_extract.py`；探针：`multi_graph_gdn_ready_consumer_probe.py`；独立配对审计：`multi_graph_gdn_ready_consumer_analyze.py`。
- 证据根目录：`benchmarks/results/multi_graph_structural/gdn-ready-consumer-v1`。`diagnostic.rank*.json`、`pair*.json` 保存全部计时与状态 SHA；`candidate-analysis.json` 保存门槛判断；`candidate-status.json` 保存进程顺序及命令。
- `candidate-source/`、`scalar-reuse.patch` 保存唯一候选，`candidate-binary-identity.json` 记录旧/新扩展及 ldd 实际 kernel 库 SHA。四组原扩展以旧冻结版本为基线，未覆盖。
- 六份大型现场张量保留在远端 `/workspace/gdn-ready-consumer-v1/sites/`，路径、大小和 SHA 见 `frozen-site-files.json`，没有放进 Git。归档的 67 份原文件逐份 SHA 校验通过，分析已本地重算一致。
- 首次采集入口遗漏了上一轮的任务队列、HCCL 和线程环境，在完成 step 2 后停止进展；已主动终止并保留于 `failed-extraction-v1`，没有计时样本。补齐原环境后的一次有界重试成功。该观察不构成驱动问题或环境因素的独立因果证明。另保留一个在状态文件创建前启动的队列读取失败日志；没有重复启动模型或计时进程。
- 全部本轮 worker 已退出，见 `closeout.json`。同算法分析与原消费契约共 17 项 CPU 测试通过；新增三个入口及测试的 Ruff check、format --check 与语法检查通过。全仓 `bash format.sh ci` 返回 2，在现有 CRLF 的第 22 行 shell 语法处退出，未进入 pre-commit；原始输出与退出码保存于 `format-ci.log`、`format-ci-status.json`。未修改全仓换行，定向检查不代表全仓通过。

全部旧数值与质量失败继续有效。旧完整模型 BF16-head 基线及其适用边界没有被本轮单层微测量覆盖或放宽。
