# 默认桶内并行的归因与第二轮优化

后续主机准备与集中提交实测见
[主机提交瓶颈记录](multi_graph_host_submission_bottlenecks.md)。

## 比较契约

本轮承接 `multi_graph_bucket_internal_parallel.md`，使用相同的 Qwen3-4B BF16、
Q1、KV512、默认 104 桶与 B98 → B97 → B103 → B98 动态轨迹。
请求分界固定为 56，attention 捕获容量为默认集合中的 56+48，Dense 容量 104。
不增加专用容量；每种策略每桶一张图；每步执行全部 FIA update。
原 ND eager 精度门槛保持 rtol=0.02、atol=0.02，hidden 和 residual 均检查。

增加 `--serial-control` 后，三种策略在同一进程比较：padding、切分串行、切分并行。
串行对照的两路 FIA 位于同一条流，使用完整资源；限核和独立 update 流仅作用于并行策略。
每种策略仍只捕获一张图；报告中的三张图对应三种策略，不是三种动态 batch。
所有正式运行每模式五轮、每轮三十二步，轮间反转模式执行顺序。
Trace 在正式计时后单独采集，每模式三次，不能以 profiler 内的主机耗时替代正式数据。

## 新增实现

### 共享 KV 写入与直接 FIA 输出

`--shared-cache` 先在完整 batch 上调用现有的 KV cache 写入实现，再分叉到两路 FIA。
两路继续通过各自的页表视图读取相同物理 KV，不搬迁或复制历史 cache。
FIA out 直接写入共享输出的两段，省掉通用 wrapper 中的输出自复制。
同样改造 padding 的 cache 写入和 FIA 输出路径，确保通用优化不被归为切分收益。
通过每个 block 的 trace 验证，三种策略均只有一个 ScatterPaKvCache。

该路径限定普通、未量化、本地 Qwen2/Qwen3 FIA；不支持 PCP、KV sharing、
KV transfer、sinks 等特殊语义。不修改生产 backend 或 runner。
参考结果在启用这些优化和 NZ 之前保存；图 replay 仍须通过原 ND 参考。

### 资源与更新时序

`--attention-cores 16` 将两路 attention 分别限制为 16 Cube/32 Vector 核。
`--lane-cores 16 12` 使用非对称资源，同时为两路创建匹配资源限制的 update 流。
`--independent-updates` 可在全核情况下单独检查 update 流拆分的成本与影响。
所有 capture/update 资源匹配；不把额外 event、流切换与等待从正式计时中剔除。

### 固定任务计划的长度绑定

`--static-task-plan` 在创建原生计划时校验 task、event、固定参数与 tensor 句柄，
每步仅绑定新的实际 KV 长度列表，继续提交每一个 FIA update。
固定 query 边界、图输入地址、页表 tensor 地址不变；页表内容随 H2D 更新。
它适用于本实验的固定桶、Q1 与固定缓冲区契约，不是通用的动态 task 替代接口。
若需要替换 tensor、query 边界或算子固定参数，调用方必须重建计划。
新接口检查 task 数、每个任务的长度行数和负长度；所有检查通过后才修改计划。
padding 与 split 同时使用该优化。

### 重新构思的方案：提前执行分支输出投影

若两路 FIA 的重叠不足以覆盖拆分成本，单独调 update/限核无法保证净收益。
新增 `--pipeline-output-projection`，将 O 投影移入对应的 attention 分支：

```text
共享 Norm/QKV/RoPE → 统一写入 KV → 分支 A：FIA A → O A ┐
                                分支 B：FIA B → O B ┴→ 共享 Norm/FFN
```

该方案让先完成的请求组执行 O 投影，尝试与另一路尚未结束的 FIA 重叠。
保持 QKV 和两个大 FFN 矩阵共享，避免重新引入完整 decoder 两路各算一遍的 Dense 成本。
代价是 O 从一次 MatMul 变为两次，并增加两次投影结果装配 copy；这些都纳入正式计时。
Trace 应显示 padding 四次 MatMul、split 五次 MatMul。
仍使用同一默认桶和固定请求分界，动态输入未知不影响捕获复用。
这是一项待实测的结构假设，不预设一定优于共享 O。

## 归因指标

除完整 block 固定/动态总耗时与单步 p50/p95 外，trace 报告以下内容：

- FIA 首次启动至全部完成的跨度、两路间空档和实际重叠。
- 所有计算 kernel 的名称、次数与耗时总和，包含 contiguous、copy 和 cache 写入。
- prepare/H2D/resolve/bind、replay、update、join 与最终同步的主机区间。
- FIA query 的捕获 shape/stride、每种策略的 task/workspace/update 流数量。

主机提交区间和设备运行区间可以重叠，不能把它们简单相加后解释为完整 wall latency。
两路位于不同流不代表同时计算；观察到重叠也不代表已经获得净收益。

## 复现

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_block.py \
  --model /models/Qwen3-4B --nz --native-update --serial-control \
  --shared-cache --static-task-plan --trace --output /tmp/bucket-plan.json
```

分别去掉 `--shared-cache --static-task-plan`、增加 `--independent-updates`、
增加 `--attention-cores 16` 或 `--lane-cores 16 12`，可复现各项对照。
增加 `--pipeline-output-projection` 验证新的 O 投影流水方案。
每次使用新的输出文件，失败结果与日志同样保留。

## 首轮实测结果（2026-10-16）

下表为动态轨迹的五轮单步 p50 中位数。前三种模式各自完整计入准备、更新、依赖和同步。
所有列出的正式运行均通过原 ND eager 检查；默认尺寸、分界与实际请求数不变。

| 配置 | padding ms | 切分串行 ms | 切分并行 ms | 并行相对 padding 增时 |
| --- | ---: | ---: | ---: | ---: |
| 原生提交基线 | 1.5766 | 1.6148 | 1.6038 | 1.73% |
| Python 提交基线 | 1.6396 | 1.6645 | 1.6524 | 0.78% |
| 统一 KV、直接 FIA | 1.5370 | 1.5668 | 1.5654 | 1.85% |
| 统一 KV、独立 update 流 | 1.5194 | 1.5539 | 1.5786 | 3.89% |
| 统一 KV、16 核 | 1.5275 | 1.5618 | 1.6003 | 4.76% |
| 统一 KV、16+12 核 | 1.5261 | 1.5393 | 1.6663 | 9.19% |
| 最终源码、常规绑定对照 | 1.5569 | 1.5906 | 1.5761 | 1.24% |
| 固定任务计划、仅绑定 KV 长度 | 1.4289 | 1.4348 | 1.4253 | -0.25% |
| 固定任务计划、独立进程复测 | 1.4599 | 1.4824 | 1.4497 | -0.70% |
| 固定任务计划、O 投影流水 | 1.4508 | 1.4798 | 1.4699 | 1.32% |

原生基线的一个 trace 中，padding FIA 为 312.5 μs；串行切分两路 FIA 总和为 315.75 μs，
加上间隙后跨度为 328.75 μs；并行切分重叠 49 μs，但 FIA 总时长升至 363.25 μs，
最终跨度仍为 314.25 μs。它说明拆分和重叠都已经发生，关键路径却没有缩短。
这些仅是独立 trace 的具体样本，不用于估计正式计时的主机/设备占比。

统一 KV 后，trace 中写入 kernel 从两次变成一次；通用输出复制也被删除。
padding 同样得到优化，因此该变化没有形成切分独有收益。
16 核与非对称限核没有改善净耗时，独立 update 流增加的依赖也未带来收益。
O 投影流水的五轮动态总耗时均比 padding 更长，不作为后续默认路径。

固定任务计划的两组动态 32 步总耗时中位数为：
46.6485 → 45.7791 ms（减少 1.86%）、47.1392 → 46.5742 ms（减少 1.20%）。
两组各有四轮总耗时更短、一轮更长；固定 B98 的单步中位数则分别慢 0.88%、1.52%。
因此这是较小且依赖测量轨迹的收益信号，不能推论完整模型或服务吞吐已经更优。
另做仅包含两种模式、六轮每轮 64 步的等次序确认，结果单独记录。

### 两模式等次序确认

最终版本关闭串行对照，每轮只执行 padding 与切分并行；两种模式各有三轮先执行。
使用共享 KV、直接 FIA 输出、固定任务计划和全核配置，独立进程运行两次。

| 运行 | 轨迹 | padding 单步 ms | split 单步 ms | padding 64 步 ms | split 64 步 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 确认 | 固定 B98 | 1.427733 | 1.433128 | 91.729821 | 91.869771 |
| 确认 | 动态 batch | 1.419865 | 1.421230 | 91.256644 | 91.510184 |
| 独立复测 | 固定 B98 | 1.449413 | 1.455273 | 92.976416 | 93.352636 |
| 独立复测 | 动态 batch | 1.451705 | 1.454416 | 94.101824 | 93.427688 |

动态单步中位数分别慢 0.10%、0.19%，总耗时分别增 0.28%、减 0.72%。
因此三模式阶段的小幅动态收益没有在等次序双模式比较中得到稳定复现。
不能将前面最有利的一次结果作为切分优于 padding 的结论。

本轮保留固定计划与统一 KV 路径用于后续实验；它们也优化 padding。
限核、独立 update 流和 O 流水均保留为显式实验选项，不成为默认生产策略。
当前证据支持“该负载下接近持平”，尚不满足进入完整 decoder 的净收益准入条件。

## 下一次结构调整的约束

本轮已经尝试从“FIA 两路并发”改为“FIA 与分支 O 投影流水”，也未获得净收益。
后续不能只增加流或延长流水线，就预设能够获益。
可检验的新方向是让图切分对应不同的 attention 工作分配，而不仅是均匀拆开同一个 FIA：
先从实际 kernel 的 tiling、访存与执行区间验证分支是否能减少关键路径上的工作或等待，
再实现匹配该差异的分支算子和调度。
当前只有 kernel 时间证据，尚没有足够的硬件计数器数据判定瓶颈是带宽还是计算资源。

保持现有三种对照、默认容量、动态输入、精度和完整 block 计时契约。
若后续更换 FIA 布局或融合入口，padding 同时应用通用改进；
先证明新分支算子的单独执行成本和并行后的实际跨度，再验证完整 block。
这属于下一阶段算子级研究，当前交付不将它描述成已经实现或必然更快的方案。

## 验证范围与后续准入

最终资格检查覆盖动态计时实际使用的每一种 batch/KV 状态，包括 B97/KV512，
并在整个计时前后确认图、长度缓冲区、分支输入视图、页表、slot、workspace 和输出地址不变。
长度变化资格检查逐次恢复原 prefill KV；正式动态计时的 KV 长度保持 512。
尚未覆盖持续生成的 KV 增长、跨默认桶切换、完整 decoder 或在线服务吞吐。

CPU multi-graph 回归 363 项通过；NPU 原生计划边界检查及四种 block 配置共五项通过，
最终加强资格检查的共享 KV＋固定任务计划＋独立 update 路径另有一项 NPU 回归通过。
Ruff、clang-format、markdownlint、变更文件导入检查和直接运行的长函数/symbolic meta 检查通过。
全仓 `format.sh ci` 已执行，仍存在 Windows 缺少 `/bin/bash`、shellcheck、部分 `python3` hook
启动失败，以及既有 `device_length_decode.py` 直接导入 Triton 的失败，不能视为全仓检查通过。

全部正式数据、源码版本、trace、日志、XML 与哈希校验器保存在工作区
`outputs/multi_graph_bucket_attribution/`。早期 SSH banner 超时的日志也保留，重连后完成硬件检查。
