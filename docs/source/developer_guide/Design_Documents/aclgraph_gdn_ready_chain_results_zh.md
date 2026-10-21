# GDN 请求组 ready：含 BF16 head 的固定轨迹完整模型回放

本轮比较同一 compact FP32 算法的 delayed 与 ready 调度。模型为 Qwen3.5-27B，TP2，原 BF16 head，默认图目录。输入 token、接受长度、状态推进、核数和工作区相同。差异限定为请求组中间乘积的发布、消费与背压时序。

环境为 910B3 的物理卡 0、1，CANN 9.1.0，torch 2.10.0+cpu、torch-npu 2.10.0.post4、vLLM 0.30.0+empty、vLLM-Ascend 0.30.0rc1。HF32 关闭，matmul precision 为 highest；既有驱动软件兼容设置保持不变。四个扩展及实际 kernel 动态库使用原冻结二进制，未重新编译。

## 完整成本表

单位：毫秒；每格是两次独立进程运行的算术均值。每次先按合计选出最慢的同一个 TP rank，再从该 rank 提取所有列，避免不同 rank 的分项相加不闭合。

| 负载 | 调度 | 执行阶段 | 一次初始化/导入 | 终末物化 | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| 原 Q4 动态轨迹 | delayed | 19549.419 | 111.536 | 24.119 | 19685.074 |
| 原 Q4 动态轨迹 | ready | 18654.315 | 111.413 | 23.991 | 18789.719 |
| 低负载 | delayed | 2765.312 | 111.620 | 8.669 | 2885.601 |
| 低负载 | ready | 2711.499 | 116.132 | 8.814 | 2836.445 |
| 中负载 | delayed | 4319.268 | 116.630 | 14.105 | 4450.004 |
| 中负载 | ready | 4185.512 | 110.941 | 14.046 | 4310.500 |

执行阶段包含每步 token 构造和上传、元数据准备和上传、真实 FIA 图更新、64 层 decoder、final norm、BF16 全词表 head、TP 通信、状态更新和最终等待。head 对默认容量的全部 Q4 行投影（含 padding 行），两种调度一致。

一次初始化是共同真实 prefill 后的 compact 状态转换与工作区准备。两模式执行相同初始化代码；表内差异是独立进程的实测波动，不代表 ready 改善了初始化。终末物化包含所有 128 个驻留 owner 的最终接受前缀生成、上传、SSM 物化与等待，保留卷积和 FIA 历史。模型加载、共同 prefill、图捕获、快照恢复和正确性下载在表外。

合计为一次初始化、连续轨迹执行、终末物化三个实测组件之和；初始化先于捕获与回归，正式执行恢复共同初态后单独计时。它不是一次连续服务冷启动的墙钟时间，也不是加载模型到响应结束的时延。

## 负载收益表

每档两对独立 TP2 进程，顺序分别为 delayed→ready、ready→delayed；12 个进程组全部顺序运行。百分比为闭合合计耗时减少率。均值收益用两侧均值耗时计算，不直接平均比率。

| 负载 | 步数 | 实际 batch 最小/均值/最大 | 实际 KV 长度范围 | 默认 token 容量子集 | 配对 1 | 配对 2 | 闭合均值收益 |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 原 Q4 动态轨迹 | 128 | 64/97.60/128 | 516～1795 | 256～512，间隔 16，共 17 个 | 4.67% | 4.43% | 4.55% |
| 低负载 | 40 | 22/24.00/26 | 20～97 | 88、96、104 | 2.38% | 1.03% | 1.70% |
| 中负载 | 40 | 54/57.00/60 | 20～111 | 216、224、232、240 | 2.85% | 3.42% | 3.13% |

低、中负载复用原 40 步请求、长度与接受前缀序列，包含重排和暂停恢复；128 个驻留 owner 中未参与者休眠，prefix 为 16。输入采用原整网 fixture 的确定性随机 token，不能称为此前单层真实文本激活实验的完全复现。原轨迹的输入和 prefix 保持不变。三档同时包含 batch、KV 长度及请求池大小的差别，不能把档间收益差异全归因于 batch，也不能据此推导普遍的 batch 阈值。

每个进程的计时终态均与其不计时完整轨迹终态相同。配对双方逐 rank 核对了全部 step 的完整 hidden、residual、logits 字节 SHA，以及初态、终末各 320 个状态张量和 host 生命周期契约，均完全一致。本轮没有重新扫描每一步的全部状态；这里只核对同算法的调度一致性，之前的逐步全状态审计另行保留。

## 关键路径与 profile

ready 让某请求组的中间乘积准备好后，就交给配对 AIV 做后续递推；AIC 可以继续生产下一组。delayed 等该 AIC 的本地整批乘积准备完才发布，并非设备全局屏障。两侧都保留已有的输出投影期间延迟提交和最终 join，它们不能再次计为 ready 独有的收益。

两者均为 20 AIC、40 AIV，同一缓冲容量，每请求四组、每组两个 key head，配对 AIV 各处理三个 value head。ready 最多两组在途，null/padding 组也参加握手。源码位置为 `benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp` 的 Producer::Process 与 ConsumeSolve；共同计算在 `q4_tile_solve.h`，资源与时序选择在 `multi_graph_gdn_q4_log_backend.py`。

选择原轨迹 step 7（batch 115、默认容量 464）和 step 31（batch 86、默认容量 352），选择依据是输入大小，先于查看 profile 结果。无 profile 的 12 个进程组全部结束、配对审计通过后，才在两个新进程组中采集 delayed/ready。未修改内核或增加性能变体。

下表单位为毫秒，取两个 rank 的 profile 值均值，只用于解释位置，不能替代无 profile 的闭合计时。每步均识别到 48 个 GDN mixed、48 个提交、304 个 decoder matmul、16 个 FIA、1 个 head 投影及 130 次 TP collective。

| step | 调度 | GDN mixed 合计 | 304 个 dense 合计 | FIA 合计 | head 投影 | head gather | decoder 至 head 跨度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | delayed | 26.977 | 57.905 | 4.427 | 2.741 | 6.124 | 155.664 |
| 7 | ready | 17.902 | 57.898 | 4.431 | 2.695 | 6.094 | 146.844 |
| 31 | delayed | 20.839 | 58.095 | 3.784 | 2.336 | 4.610 | 137.480 |
| 31 | ready | 14.588 | 58.187 | 3.788 | 2.368 | 4.632 | 131.251 |

各类 kernel、通信与提交可重叠，不能把这些列直接相加作为整步耗时。主机 final_wait 包含尚未完成的设备工作，也不能重复计费。完整计时没有扣除上传或首个 TP。

profile 单独启动采集会使两 rank 进入首个 embedding TP 的时间不同：例如 delayed step 7 的 rank 0 首个 all-reduce 为 19.151 ms，rank 1 为 0.281 ms；ready step 31 的 rank 0 为 18.576 ms。因此保留原始完整跨度和首个通信记录，但用“第一个 decoder matmul 到最后 head/gather”的相同区间比较后续关键路径，避免把 profiler 启动偏差当成 ready 的通信收益。这个区间不充当完整成本端点。

- step 7：GDN mixed 减少 9.074 ms（33.64%），decoder 至 head 跨度减少 8.820 ms。
- step 31：GDN mixed 减少 6.251 ms（30.00%），decoder 至 head 跨度减少 6.229 ms。

减少量及其位置相吻合，304 个 dense、FIA 和 head 的工作量与耗时基本相同，支持收益主要来自 GDN 组内发布与消费时序。它不是来自少跑层、减少 token、降低 head 精度或减少接受态提交。

### Cube/AIV 与等待的可测证据

下表单位为微秒，是 48 层、两个 rank 的每层每核平均 PMU 时间。AIC/AIV 生命周期包括核内等待；矩阵、向量和标量管线可相互重叠，不能相加得到生命周期，也不能由两项累计时间相减还原真实交集。

| step | 调度 | AIC 生命周期 | AIV 生命周期 | AIC 矩阵指令 | AIV 向量指令 | AIC 标量指令 | AIV 标量指令 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | delayed | 214.922 | 552.336 | 29.129 | 215.727 | 92.162 | 157.749 |
| 7 | ready | 365.598 | 364.641 | 29.129 | 215.727 | 184.726 | 157.812 |
| 31 | delayed | 150.138 | 404.961 | 21.785 | 165.409 | 72.141 | 122.560 |
| 31 | ready | 281.377 | 280.483 | 21.784 | 165.409 | 136.804 | 122.558 |

矩阵与向量算术耗时几乎不变；AIV 生命周期分别缩短约 187.70、124.48 us，说明同一消费工作更早完成。AIC 生命周期反而增长，ready 中接近 AIV 生命周期，与生产者受到 consumed 背压、跟随消费者推进的源码协议一致。这支持“减少原整批发布造成的空等并形成流水”的解释，不是 Cube 算术本身加快。

ready 的代价也实际存在：AIC 标量管线计数分别从 92.16 增到 184.73 us、72.14 增到 136.80 us。该增量与新增握手、背压同在一条冻结代码分支内，但包含等待与执行关系的影响，不能把它全部解释成可删除的 flag 开销。正式计时已经支付这些代价，净收益仍为正。

静态协议调用数可精确核对：delayed 每 GDN 层为 20 次 ready_set 加 40 次 ready_wait，共 60 次；ready 包含 null/padding 组，容量 G 对应 G 次 ready_set、2G 次 ready_wait、2G 次 consumed_set、G 次 consumed_wait，共 6G 次。容量 464 为 2784 次，容量 352 为 2112 次。这些是 Ascend C API 调用数，不是硬件指令数或耗时。

本轮直接测到的是任务区间、PMU 管线时间、AIV 生命周期变化及上述同步调用数量。**精确的 mixed kernel 内 Cube/AIV 指令时间交集、CrossCoreWaitFlag 独占等待和新增握手的独占耗时仍未分离，保留为未知值。** 不以 host final_wait、运行时 EVENT_WAIT 或累计管线时间冒充它们。字段解释依据[昇腾 PipeUtilization 文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/opdev/optool/atlasopdev_16_0099.html)，本机原始字段和数值另行保存。

两模式还共有 AIV 提交与 AIC dense 的任务区间重叠：step 7 约 3.938/4.078 ms，step 31 约 3.562/3.800 ms（delayed/ready）。这是跨 kernel 的区间交集，不能称作 mixed kernel 内算术管线交集，也不能把共有的延迟提交重算为 ready 的独立收益。

### 是否据此再改性能

本轮足以确认请求组 ready 在冻结完整模型轨迹中有独立净收益。当前 GDN 已从整批等待转为 AIV 消费与生产者背压接近同速；同时整网还保留约 58 ms 的 304 个 dense 投影，以及相同 TP 和 head 成本。不能把原来被掩盖的生产阶段重复当成后续可消除预算。

本轮不实施下一项性能修改。现有 PMU 还不能确定剩余主要可消除成本究竟是消费侧哪段工作，或同步背压中的哪部分；单凭标量计数上升就减少握手、调整在途槽数，缺少独占耗时证据。若继续，首先需要针对这个已定位的消费/背压边界补足直接证据，再决定修改点；不把它写成已证明的新收益。

## 结论边界

本结果准确名称为“固定轨迹的完整模型执行回放”。采样、draft 生成与真实服务调度没有执行，不能据此报告端到端服务收益；本轮只比较同一 compact 算法的调度。每档只有两个独立配对，负载与波动范围均有限。

head 接入现有手工整网回放图，使用官方默认图目录中该轨迹需要的容量；head 覆盖全部 padding Q4 行。这里没有证明标准服务会采用完全相同的 head 捕获范围或 logits 行选择方式。

原 0.5% 数值检查失败、真实文本分布初筛失败、全部局部退化及任务质量记录保持有效。本轮性能证据不授予 compact 部署数值或质量资格；不再开展 AIME、首次分歧或 FP32 head 优化。旧约 4.80% 不含 head，本轮以完整成本表为准。

## 与已有机制的关系

compact 状态算法本轮冻结，不能把这份性能结果当成新增状态算法的贡献。被消融的是请求组中间乘积发布与背压流水：delayed 也保留同样的延迟提交、投影重叠与 join，这些共同能力不重复计入 ready 的独立收益。

此前社区环形日志、动态游标和周期 flush 的能力也不计入本轮调度收益。本轮没有重新比较 ReplaySSM 与 compact 的完整算法，不能据此宣布 compact 优于社区整套路径。这个执行机制可以被单图或 Padding 复用，不把多图独享性作为主线成立的前提。

## 证据与复现

- 冻结协议：`benchmarks/results/multi_graph_structural/gdn-ready-chain-v1/frozen-protocol.md`。
- 逐 rank 结果、顺序及命令：同目录各独立运行子目录及 `protocol.json`、`status.json`。
- 原始结果下载 SHA：`fetch-manifest.json`；冻结四组扩展与其真实 kernel 动态库、源码摘要：`kernel-source-and-libraries.json`。
- 为避免 Windows 路径长度限制，下载时缩短了 profiler 时间戳目录；`fetch-manifest.json` 保留原远端路径到本地路径的映射，每文件原字节 SHA 未改变。完整 CANN 原始数据库仍保存在远端。
- 同算法配对审计和闭合成本：`analysis.json`。profile 解释见 `profile-analysis.json` 和对应原始时间线、CSV。
- 195 份下载原文件的 SHA 及 8 份时间线 SHA 全部核对通过，profile 进程的完整不计时轨迹与正式回放参考相同。Git 归档显式关闭原证据的换行转换，并核对暂存 blob 的原字节 SHA；见 `postprocessing-audit.json`。全部 worker 退出的记录见 `closeout.json`。
- 回放入口：`benchmarks/multi_graph_gdn_ready_chain_probe.py`；隔离进程顺序入口：`multi_graph_gdn_ready_chain_runner.py`；分析入口：`multi_graph_gdn_ready_chain_analyze.py`。

机制所据的构建快照 `tile_pipeline_kernel.cpp` 和 `q4_tile_solve.h` 与当前仓库按 LF 归一后相同；构建快照 `torch_binding.cpp` 与当前仓库不是同一 SHA，不把现有源树冒充完整原构建来源。两模式实际复用同一冻结二进制，完整身份记录保留。

入口开发时 44 项相关 CPU 测试通过；适配本机 CANN 时间线后的定向 profile 解析回归 5 项通过。后者以 `--noconftest --import-mode=importlib --basetemp=../gdn-chain-profile-ut-v2` 运行，避免本地缺少 vLLM 的全仓 fixture 及既有临时目录权限问题。新改 Python 的 Ruff 和语法检查通过。全仓 `bash format.sh ci` 已尝试，但在已有 CRLF 的 shell 入口处失败，未声称全仓检查通过。实际 TP2 回放以本轮各 rank 的闭合与字节审计为依据。
