# GDN ready：原 Q4 128 步整网核验与主线收敛

日期：2026-10-20。Qwen3.5-27B，TP2，A2 / 910B3；BF16 激活、FP32 SSM 与日志。
FP16 排查维持结束；没有新增精度方案、路由器或模型。

补记：随后按限定边界完成了 [FP32 首步诊断](aclgraph_gdn_fp32_step0_zh.md)，
未发现明确的有限实现修复，compact 部署候选维持结束；下文的原始计时、审计与失败记录保持不变。

## 1. 结论

**同算法 delayed/ready 的完整轨迹状态审计通过；独立官方整网数值筛选未通过。**
默认图目录下，ready 相对 compact delayed 的三轮闭合成本改善率均值为
**4.801%**，3/3 轮为正。
按成本均值之比计算为 4.801%，两种统计口径分别保留。
本进程的三轮配对均观察到 ready 净收益；仍缺另一完全隔离进程的复现，不能称为跨进程稳定的整网加速。

原 0.005 数值筛选未通过；保留原阈值和全部失败结果，不把调度一致性通过写成官方等价。当前 compact 配置不能作为已获数值资格的官方整网替代。

主线收敛为：**A2 GDN 请求组 ready 是可单独归因的执行调度机制；
compact 状态算法复用已有工作，整网替代资格与调度收益分开判断。**
本轮完成这一配置的证据闭环，停止继续扩大性能矩阵。
若数值筛选失败，暂不将 compact 接入生产整网或新增精度变体来继续追逐收益。
它可以作为执行流水研究及同源消融结果，不能写成“已证明优于官方的完整推理方案”。

## 2. 简单解释机制及与已有工作的区别

Q4 验证先计算四个候选，下一步才知道接受几个。
compact 用一份大状态加小日志表示候选；这类状态表示、动态接受前缀、
日志修正投影与延迟物化已有 RecoverSSM、ReplaySSM、TreeWY 等工作覆盖。

这里改变的是同一紧凑算法在 A2 上的执行顺序：Cube 每完成一个请求的两个 key head 产品，
配对的两个 AIV 就消费六个 value head，同时 Cube 可继续生产下一组。
最多两个组在途；ready 与 consumed flag 约束发布、读取和退出。
null/无效任务仍参与握手，避免动态请求有效性破坏进度。
生产消费流水本身不是新概念；独立差异限定为这条 Q4 动态请求路径上的 A2 映射、
局部发布/背压协议及完整状态生命周期的同源消融，不能写成首创流水或新的状态算法。

delayed 使用相同混合核、产品矩阵、分组和核数，只把发布时机改为本 AIC 的整批任务完成后。
这个控制是**每个生产者的局部批次屏障**，不是设备全局屏障。
两者共享 FP32 算术、输出投影期间的延迟提交、全部 join 和终末物化。
所以 delayed/ready 差值可以归因到请求组 ready 时序，不能重复归因到状态压缩或延迟提交。

| 对象 | 已有内容与本轮区别 |
| --- | --- |
| ReplaySSM 固定社区源码 | PR #49887、提交 `5b96cdbceefcdec6d60d588420ee127d76bd8f97` 已有环形日志、设备游标和周期 flush；本轮没有把这些当作原创。 |
| 社区 flush 依赖 | flush 后的状态被当前输出直接使用；既有 A2 机制对照保留前置 flush，未直接套用 compact 的延迟提交。 |
| A2 ready 机制 | 所审固定源码没有本实现的 AIC:2×AIV 请求组 flag 协议；这只支持具体执行映射差异，不证明所有文献都没有流水。 |
| 本机官方路径 | 保留真实 conv、门控、norm、输出投影、TP、MLP 和 FIA；compact 仅替换 speculative recurrence。 |
| Padding 与切分 | 当前两控制都在同一合法整桶图内运行。Padding 可原样使用 ready 算子；没有多图切分独有收益结论。 |

社区源码、SHA 与上轮完整对照沿用
[GDN ready 动态范围记录](aclgraph_gdn_ready_ranges_zh.md)和
[ReplaySSM A2 对照](aclgraph_replayssm_a2_review_zh.md)。
这是固定版本核查，不声称本轮重新全面查新或优于社区最佳实现。

## 3. 冻结轨迹与覆盖范围

轨迹 SHA256：`0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`。
原 128 步、128 个稳定 owner、Q4、token seed、位置、接受长度及请求重排完全保留。
batch 为 64～128，均值 97.6015625，共 58 个不同取值。
17 个实际使用的默认容量为 256、272、…、512 token；没有新增专用容量。
例如 batch=98、Q4 时，两控制都按默认目录使用 400 token 图。
先由安装版 EngineArgs 默认值和 dispatcher 决定容量，再复用这些容量捕获两模式共 34 张图。
每张整网图必须拥有全部 16 个 FIA 真实长度更新任务。
304 个 decoder 线性权重共用同一 NZ 格式；权重转换属于共同模型初始化，不计为 ready 的优化。

共同前缀对全部 128 owner 使用官方真实模型初始化。
后续输入为原轨迹的固定词表随机 token 和预定接受长度，属于 teacher-forced 回放，
不是自回归文本生成或真实 draft 接受率测试。
执行全部 64 个真实 decoder 和最终 norm，端点是 hidden/residual；
未包含 LM head、sampling、服务调度器、draft model、请求驱逐或槽位回收。

## 4. 同算法全轨迹状态审计

两 rank × 两模式 × 128 步，共 512 条资格记录。
每一步都从上一完整状态继续，以 delayed eager 轨迹对照 delayed/ready ACLGraph：

- 输出 hidden/residual 使用 `rtol=atol=0`。
- 扫描全部 GDN base、log、valid、提交暂存、卷积和全部 FIA K/V；
  每步约 17.94 GB/rank，流式比较原始字节，并检查 null/暂停 owner 及非活动页面不变。
- 核对 registry、状态/暂存身份、generation、initialized、last_step、pending 等 host 契约。
- 每步重建全部 owner 的四个逻辑候选，检查有限及 null 为空。
- 两模式的完整 pending 终态及最终接受态物化结果一致。
  每个计时样本收尾后再次与资格参考全状态比较，共 12 次。

CPU 数组扫描只加速计时外的审计，保留有限性、零值、字节相等和检查范围。
资格阶段 CPU threads=8，计时前恢复为 1；CPU 与数组两种审计路径均有失败样本回归。
最初因 CPU 扫描慢而主动停止的未完成尝试保留在 `incomplete-original/`，不计为通过。

delayed 的完整 128 步来自此前进程的已闭合审计；ready 在移除诊断 watchdog 后从相同初态重跑全部 128 步。
复用通过前缀、默认目录、完整元数据、源码 SHA / 算术函数 AST 和四扩展二进制校验，
没有拼接此前 ready 的 21 步。新进程另外重跑完整 delayed eager 轨迹，
其 pending 与终末接受态均与新 ready 全状态比较。
原 delayed 记录与其 SHA 链保留在 `replay-delayed-source/`；不把继承记录说成新扫描。

本轮相关 CPU 回归为 249 项通过，Ruff 格式/静态检查和 `git diff --check` 通过。
仓库要求的 `bash format.sh ci` 已尝试，但现有 CRLF 脚本在执行 hooks 前发生语法错误；
原始日志保留在证据目录，不把它标为全仓格式检查通过。

## 5. 完整生命周期成本

下表为三轮的 TP 闭合关键 rank 成本均值，单位 ms。
每一轮、每一模式的全部列取自**同一个闭合耗时最大的 rank**，保证可相加。
执行列 = 从第 0 步准备到第 127 步完成的墙钟；终末物化单独扣出。

| 模式 | 128 步执行 | 一次导入/直接转换 | 终末接受态物化 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| compact delayed | 18524.283 | 118.671 | 24.059 | 18667.013 |
| compact ready | 17628.192 | 118.671 | 23.920 | 17770.784 |

“一次导入/直接转换”是把已初始化的一份前缀 base 转成 compact decode：
分配和填充卷积候选槽、日志/valid、owner lookup、提交缓冲和注册信息。
**它不是从常驻四份 dense 候选压缩导入**；每 rank 只有一次观测，没有稳定性分布。
两控制复用同一转换观测；表内可能的导入列差异来自闭合关键 rank 选择，不代表模式初始化速度不同。
主表来自独立数值进程和原审计进程全部退出后的隔离计时，导入、执行和收尾均隔离，
见 `timing/timing-isolation.json`。

资格进程的先行计时仍完整保留在 `replay/`，但其单次转换与另一数值进程初始化并行，
因此不与主表合并为额外的完整成本重复试验。
主表补齐终末 CPU 决策张量构造和 H2D 上传后再 replay 收尾图。
它通过源码/函数 AST、二进制、默认目录、前缀和轨迹校验复用此前 512 条逐步审计，
重新运行一条完整 delayed eager 参考，并在全部 12 个 rank/样本端点做全状态一致性检查；
没有把复用资格写成再次实施逐步扫描。

终末决策预先固定为每 owner 接受 `1+owner%4` 个候选。
所有 128 owner 均实际支付最后接受态物化，包括暂停 owner；log/valid 清零。
卷积与 FIA 缓存保留，接受长度留给下一次卷积历史选择，不把它们的清除当成物化。

| 配对轮次 | delayed 合计 ms | ready 合计 ms | ready 改善 |
| --- | ---: | ---: | ---: |
| 1 | 18671.657 | 17763.416 | 4.864% |
| 2 | 18660.092 | 17766.702 | 4.788% |
| 3 | 18669.292 | 17782.234 | 4.751% |

三轮按 delayed→ready、ready→delayed、delayed→ready 交替。
轮间恢复实验初态、加载模型、共同前缀、捕获、诊断和 CPU 状态审计不计时。
执行包含原 token/描述符准备及上传、FIA update、全部真实层与 TP 通信、commit、逐步同步和 join。
没有逐步额外 TP barrier；每条轨迹起点对齐。

逐步延迟仅作补充：两 rank 重算的均值 delayed/ready 为
144.734/137.735 ms，
P99 为 172.836/162.996 ms。
逐步最大值之和不替代闭合端点；原始 `critical_rank_us` 字段在此阶段是本地值，分析器已重算。

## 6. 独立官方整网数值对照

官方与 compact 各从同一个相干前缀开始，各自连续推进原 128 步；
卷积、FIA 和 SSM 历史彼此独立，没有每步把官方状态注入 compact。
完整四候选 dense 常驻会增加约 27 GiB/rank，因此官方 FP32 候选库放在 CPU，
每层原样搬到一个 NPU scratch，运行安装版未替换的 `recurrent_gated_delta_rule` 后搬回。
层钩子仅改变状态存储位置，不改变官方算术；全部 48 个真实 GDN 层必须逐个经过。
**这些搬运只是正确性工具，其时间不参与任何性能比较。**

每一步比较最终 hidden/residual、48 层所有 owner 的四个候选与卷积缓存、
16 层全部 FIA K/V 页面，最后比较所有 owner 的实际接受态；共 70336 条数值记录。
阈值固定为最差请求 NRMSE ≤0.005，没有因失败而中止后续观察或放宽阈值。
FIA 的 owner 轴实际是页面，表中明确记为 page。

| 观察对象 | 最大 NRMSE | 超线记录/全部记录 | 最大值位置：step、层、owner/page |
| --- | ---: | ---: | --- |
| 最终 hidden | 0.06227756 | 256/256 | 74，hidden，owner 55 |
| 最终 residual | 0.04848308 | 256/256 | 74，residual，owner 55 |
| 48 层四个逻辑候选 | 0.15539885 | 45589/49152 | 74，gdn62/candidate，owner 55 |
| 48 层终末接受态 | 0.01962793 | 91/96 | 128，gdn60/terminal，owner 55 |
| 48 层卷积缓存 | 0.01891886 | 11132/12384 | 12，gdn32/conv，owner 47 |
| 16 层 FIA K/V 页面 | 0.02356244 | 8168/8192 | 10，fia39/value/page，page 643 |

最终输出在第 0 步已经超线：hidden/residual 的最差请求 NRMSE 分别为 0.01181611/0.01007352。这不是只在终末状态才出现的误差，也不是 ready 调度改变了同算法结果。

NRMSE 是相对输出或状态张量的范数误差，不是 token 错误率、任务精度损失或接受率。
“调度一致性通过”只说明 ready 改变时序没有改变同算法结果；
“跨算法数值筛选未通过”说明 compact 与官方之间的累积差异尚未满足原冻结门槛。
两者不矛盾，不能互相替代。
独立官方对照端点为 eager；另一项完整状态审计建立 compact eager 与图 replay 的零容差一致性，
因此这份数值比较可用于该 compact replay 的算术资格，不能用作官方图性能基线。

首次独立官方对照在完成第 0～123 步后发生 SIGSEGV，完整状态审计也在 delayed 128 步、ready 21 步后发生 SIGSEGV。
两次最后打印的 Python 主线程位置分别是 LayerNorm 和 CPU 扫描，不能据此判断算子损坏。
宿主机 core 的实际故障线程都在 `libpython3.12.so.1.0+0x2ea414`，
符号化为 `PyType_HasFeature → dump_frame → dump_traceback`，属于定时诊断打印路径。
两个 core 均被系统截为 2 GiB，因此不能恢复完整调用栈；现有寄存器、映射和符号足以确认直接故障位置。
官方文档说明该功能使用 watchdog 线程；CPython 也记录过异步读取其他线程 frame 的崩溃类别，
参见 [faulthandler 文档](https://docs.python.org/3.12/library/faulthandler.html)与
[CPython #116008](https://github.com/python/cpython/issues/116008)。
这里确认的是本机直接故障路径，不把其他版本的 issue 当成本机具体竞态的完整证明。

修复只移除 `dump_traceback_later(180, repeat=True)`，保留自然崩溃诊断和所有审计。
第一次原字节记录保留在 `numeric-incomplete-sigsegv/` 和 `replay-delayed-source/`；
诊断期间为避免相同风险主动停止的第二次数值尝试保留在 `numeric-stopped-watchdog/`。
本节完整数值表来自修复诊断后的单次连续重试，没有拼接不同运行的状态，也没有改精度或放宽门槛。
core 映射、符号化、原压缩 core SHA 及限制说明见 `native-diagnostics/`。

## 7. 主线与适用边界

保留的研究对象是**同算术、同资源下的 A2 请求组 ready 执行流水**。
已有真实激活单层中、高负载证据继续有效；本轮整网结果只支持这里的配置与原 Q4 回放。
不再把紧凑日志表示、环形游标、周期 flush 或延迟物化本身包装成新算法。

本轮结束后不继续扩大模型、负载或精度矩阵，不增加路由器；FP16 归因保持结束。
该配置未通过原官方数值门槛，整网部署方向在本轮收口；若要重新开启，决定性证据首先应是
**在不放宽冻结门槛的条件下取得独立官方连续整网数值资格**。
在它之前，扩大服务压测或继续微调 kernel 都不能补上正确性缺口。

当前没有完整服务吞吐提升、优于官方图模式、切分独占优势或生产可用性结论。
可交付的是受限执行机制、同源调度消融、完整状态审计、独立数值对照和明确失败边界。

## 8. 源码及证据

| 位置 | 职责 |
| --- | --- |
| `multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp`，`Producer::Process` / `ConsumeSolve` | 两组在途、局部 delayed 与逐组 ready，AIC/AIV 握手。 |
| `multi_graph_gdn_q4_log_backend.py`，`Q4LogBackend` / `installed_log_registry` | 同算术派发、投影期间提交及 join、状态事务。 |
| `multi_graph_gdn_q4_full_probe.py`，`replay_trajectories` | 默认桶捕获、全部 128 步审计、交替计时及终末比较。 |
| `multi_graph_gdn_q4_full_stream_audit.py`，`StreamingCheckpoint` | 流式逐字节状态与 host 契约核查。 |
| `multi_graph_gdn_cpu_audit.py` | BF16 有限性及原始字节 CPU 数组扫描，计时外使用。 |
| `multi_graph_gdn_full_numeric.py`，`OfficialBanks` / `qualify` | 独立官方逐层状态搬运、连续数值误差记录。 |
| `multi_graph_gdn_ready_full_lifecycle.py`，`TerminalDrain` | 所有 owner 的最终接受态物化和闭合成本。 |
| `multi_graph_gdn_ready_full_analyze.py` | 完整性、来源校验、TP 可加成本拆分及跨算法误差分析。 |

证据目录：
[gdn-ready-full-v1](../../../../benchmarks/results/multi_graph_structural/gdn-ready-full-v1/README.md)。
原始报告、日志、精确被测源码压缩包、扩展二进制 SHA、环境、命令和轨迹均已保存。
本地仓库 HEAD 与被测本机安装提交分开记录，不把当前分支提交伪装成本机安装版本。
