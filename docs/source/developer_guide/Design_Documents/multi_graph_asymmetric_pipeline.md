# 请求切分并行的结构优化计划

## 依据与目标

2026-10-16。承接 `multi_graph_host_submission_bottlenecks.md`。
现有 56+48 FIA 并行存在实际重叠，但分支变慢抵消了重叠收益。
共享 Dense 后，切分没有减少 Dense 行数，也未明显减少 FIA 主要读取量。
输入打包和原生提交同时改善 padding，不能作为切分独有收益。
计数器尚未证明整卡 HBM 饱和。

目标是验证切分能否通过不同阶段重叠或不同请求的执行策略产生净收益。
不再将增加流、调整核数或公共主机优化当作主要结构方案。

## 固定比较契约

- 主样本保持 Qwen3-4B BF16、Q1、KV512、B98/97/103/98、默认 104 桶。
- 分支容量必须属于实际默认 capture sizes，且总和等于桶容量；禁止专用 49/98 容量。
- 不切 attention head，不提前知道输入；固定容量与地址，每步更新实际内容与长度。
- FIA 全部 update 保留；padding 对称使用通用优化。
- 原 ND eager 参考，hidden/residual 均按 rtol=0.02、atol=0.02 检查。
- 正式结果计入准备、传输、update、replay、装配和同步；profiler 另行运行。
- 六轮等执行次序、每轮 64 步，独立进程复测；保留失败与负收益。
- 当前仅为 embedding 加第一 decoder block，不外推服务或完整模型。

## 第一阶段：非对称跨算子流水

共享 QKV/RoPE/KV 写入后，小分支执行 FIA/O/Norm/FFN，
同时大分支执行 FIA 并继续自身 O/Norm/FFN，最后汇合。
小分支提前进入 Dense，尝试与大分支 attention 重叠。
相比已有 O-only 流水，额外窗口来自 FFN；代价是小 GEMM、重复权重读取和输出装配。
已有 O-only 失败结果是风险证据，不能预设完整 FFN 流水有效。

先验证真实形状的资源互补：FIA 大分支与预备输入的小分支 Dense 独立/并发耗时，
以及共享 Dense 与拆分 Dense 的代价。预备输入只用于诊断，不作为推理结果。
允许以真实依赖的完整 block 实验直接检验净收益，其结论优先于理想重叠上界。
比较相同结构串行和并行，以区分拆分损失与实际并行收益。
预先选择默认容量中的 8+96、24+80、56+48；不为每个实际 batch 选形状。

若并行不能稳定覆盖拆分成本，停止扩大该路线；不直接扩展全部层。
若明显获益，再增加动态输入、KV 增长及未调参 batch 的验证。

## 第二阶段：按 KV 工作量分组

保持请求集合和总 KV token 数相同，比较混合长度与分组执行。
padding 使用相同排序；输入重排和结果恢复计入成本，历史 KV 通过页表寻址而不搬迁。
首先证明混合长度存在 tiling 或负载不均损失，再考虑分支专用策略。
等长 KV512 仍作为固定对照，不以换负载后的收益替代原场景结论。
若 FIA 已充分平衡混合长度，则停止该方向。

## 第三阶段：统一 attention 任务调度

仅在前两阶段证据支持时，考虑一个执行入口统一分配分支任务，减少独立启动、
静态资源分配及尾部等待。它需要算子工程，不假设现有 FIA 公开接口支持。
单路 padding 也应获得相同通用改进；只有分支差异或流水产生的部分才算切分收益。
设备长度读取必须单独验证，不跳过当前 FIA 的 host 长度 update。

## 当前进度

已直接实现真实依赖的完整 block 流水，未将预备输入的理想重叠作为正式成绩。
新增 `--pipeline-ffn`、`--branch-capacities`；默认行为不变。
共享 QKV 后每路执行 FIA/O/Norm/FFN，两个输出各装配两次，总共四次显式 copy。
每 block 的 MatMul 从四次变为七次；所有 FIA update 保留。
新增 trace 指标统计 FIA 与 MatMul 区间交集的并集，防止重复计算重叠时间。

### 首轮与独立双模式复测

动态轨迹六轮、每轮 64 步，以下为各轮单步 p50 的中位数。
首轮含 padding、相同结构串行切分、并行切分；复测只含 padding 和并行。

| 配置 | padding ms | 切分串行 ms | 切分并行 ms | 并行比 padding 增时 |
| --- | ---: | ---: | ---: | ---: |
| 8+96 | 1.086984 | 1.267985 | 1.238370 | 13.93% |
| 24+80 | 1.085064 | 1.283857 | 1.230605 | 13.41% |
| 56+48 | 1.091677 | 1.256134 | 1.168174 | 7.01% |
| 8+96 独立复测 | 1.096436 | — | 1.246152 | 13.65% |
| 56+48 独立复测 | 1.102724 | — | 1.165762 | 5.72% |

以上全部通过原 ND 参考、动态 batch/长度资格检查与稳定地址检查。
并行确实改善相同结构串行耗时，但不抵消拆分代价。
在 56+48 首轮 trace 中，FIA/MatMul 重叠中位数 131.5 us；
MatMul 时长总和从 padding 189.5 us 增至切分串行 318.5 us、切分并行 585.25 us。
后者包含并发导致的延长，不能相加或相减来推算 wall latency。
设备计算跨度分别为 699.75、828、738.75 us；最终仍以不带 profiler 的整步计时为准。

### 后续定位：update 提交顺序

8+96 trace 中，大分支 FIA 比小分支晚约 150 us 启动；小 FIA 早已结束。
这尚不能证明全部空档都来自主机，但提供了可检验的因果干预点。
新增 `--large-lane-first`，只改变 update 提交优先级，不改变请求映射、图拓扑、
实际长度、更新数量或结果。绑定参数继续按原 lane 身份索引。
对 8+96、24+80 做优先级实验，并独立复测及以相同源码反向跑默认顺序对照。
结果如下，仍为六轮每轮 64 步的动态单步 p50 中位数：

| 配置 | padding ms | 切分并行 ms | 比 padding 增时 |
| --- | ---: | ---: | ---: |
| 8+96 大分支优先 | 1.095667 | 1.217287 | 11.10% |
| 24+80 大分支优先 | 1.100784 | 1.202164 | 9.21% |
| 8+96 大分支优先独立复测 | 1.101044 | 1.227540 | 11.49% |
| 8+96 最终源码默认顺序对照 | 1.122634 | 1.261987 | 12.41% |

优先更新大分支后，8+96 首轮 trace 的 FIA 跨度从默认顺序约 439 us 降到约 323 us，
但并发下 FIA/MatMul 仍存在明显时长延长，完整 block 没有净收益。
主机更新顺序确实影响设备时间线；它不能消除拆分 Dense 的成本和并发干扰。
不同进程的绝对时间有波动，不能将上述相减视为精确的独立 update 节省量。

第一阶段结论：停止把完整 FFN 请求流水扩展到全部层或生产 runner。
保留显式 benchmark 开关与失败证据，不将其设为默认路径。
后续转向第二阶段的混合 KV 长度损失检验，进度与结果见
[混合 KV 长度与请求分组实验](multi_graph_mixed_length_partition.md)。
也尚未实现统一 attention 任务调度或独立的预备输入资源互补 microbenchmark；
本阶段使用真实依赖完整 block 结果直接否定当前流水实现的净收益。

### 验证与记录说明

CPU multi-graph 回归 382 项通过；新增 NPU FFN 流水复用测试通过，覆盖全部 update、
原 ND 精度、动态 batch/长度、独立更新流、固定地址和 4/7 MatMul trace 检查。
九组 4B 正式运行及一组 0.6B smoke 均通过各自的资格和计时后精度检查。
大分支优先顺序在 4B 正式运行中完成 NPU 验证。

`format.sh ci` 已执行，Ruff、格式、拼写、clang-format、markdownlint 等通过。
Windows 缺少 `/bin/bash`、shellcheck、部分 `python3` hook 启动失败及既有
`device_length_decode.py` 禁止直接导入 Triton 的问题仍存在，不能称全仓检查通过。
长函数和 symbolic meta 检查改用 venv Python 直接运行通过。

首轮源码的 `explicit_output_merge_copies` 总字段仍为旧值 0，
其新增的 `final_assembly_copies_per_split_replay=4` 与实际代码一致。
复测前修正总字段；保留首轮原始 JSON，不回写修改已测证据。
首轮、复测和最终优先级源码分别归档并核对 source SHA256。

### 复现

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_block.py \
  --model /models/Qwen3-4B --nz --native-update --native-replay \
  --static-task-plan --shared-cache --packed-metadata \
  --pipeline-ffn --branch-capacities 8 96 --large-lane-first \
  --rounds 6 --repeats 64 --trace --output /tmp/asymmetric-priority.json
```

去掉 `--large-lane-first` 为提交顺序对照，增加 `--serial-control` 为相同结构串行对照。
原始 JSON/trace、运行日志、被测源码快照和汇总保存在工作区
`outputs/multi_graph_asymmetric/`，不得覆盖失败或负收益记录。
