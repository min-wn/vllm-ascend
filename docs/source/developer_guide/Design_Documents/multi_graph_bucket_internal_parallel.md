# 默认桶内并行图改造方案

## 目标和架构契约

当前两张完整 decoder 图重复执行 Dense，小矩阵成本和并发竞争抵消了少量 padding 节省。
改造目标是每个默认捕获桶持有固定内部并行拓扑：Dense 合并执行，attention 按请求分支执行。
这保留默认容量和动态 batch 复用，放弃直接拼接两张原有完整 decoder 图的执行形式。
不增加 49/98 等专用容量，不按 attention head 切分，不为每个实际 batch 或分支组合捕获图。

104 桶固定使用 56+48 attention 容量，分界永远在第 56 行。
B98 对应真实请求 56+42，B97 对应 56+41，B103 对应 56+47。
Dense 始终执行 104 行，dummy 行 KV 长度为零、slot 为 -1。
请求槽位与物理 KV 页表保持一一对应，改变 B 不移动分界或 KV cache。

每层共享 Norm/QKV，然后在图内分叉到两路 attention，结果写入共享输出的非重叠区域，
图内汇合后共享 O/FFN。图执行期间不回到 Python 逐层 gather、cat、launch 或同步。
实际 FIA 内部是否引入 contiguous/copy 仍需 trace 检查，不能只凭 tensor view 宣称零拷贝。

## 按阶段实施

### P0：单个真实 decoder block 的复用资格

新增独立 benchmark，复用现有真实模型加载、prefill KV 和默认 dispatcher 配置。
比较官方单路 block 与固定桶内并行 block，包含 embedding、完整第一层、hidden/residual 输出。
Qwen3-4B BF16、Q1、KV512、910B3 为主样本，初始 B98，不重新抽取有利样本。
两边使用相同布局，先保存原 ND 权重 eager 参考，再对称启用 NZ。

一次捕获后运行 B98/KV512 → B97/KV511 → B103/KV512 → B98/KV512。
检查两路容量来自默认集合、图数不随 B 变化、输入/输出/页表/slot 地址不变，
每步按真实长度 resolve 并执行全部 FIA update。容差保留 rtol=0.02、atol=0.02。
同结构 eager 精度与跨结构误差分开记录，生产准入保持原有条件。

### P1：验证完整 block 的净收益

计时包含输入和 metadata 准备、H2D、任务解析、所有 update、图 replay、依赖与最终同步。
不在正式计时开启 profiler，五轮每模式三十二次，覆盖八个完整动态周期，反转测量次序，独立进程复测。
padding 和 split 使用相同两路 attention 流拓扑对照规则：默认均为全核，
仅 split attention 的受限配置另作实验，Dense 始终使用完整资源。
capture 和 update 的资源设置必须一致；限核仍对照原精度门槛，失败不报告有效收益。

同时报告固定 B98 和预先定义的桶内动态轨迹总耗时、p50/p95。
动态性能轨迹改变 batch，KV 长度固定为 512；长度变更资格检查逐次恢复相同 prefill KV。
这样避免在固定请求的 KV 上先回退长度、覆写旧前缀再增长，造成基准参考状态失配。
它不代表真实生成时持续增长的 KV 长度轨迹已完成性能验证。
逐层共享投影、主机区间缩短或 isolated attention 收益都不能替代完整 block 的结果。
只有明确的完整 block 净收益才进入 P2；负收益应保留报告并定位其来源。

### P2：完整 decoder 与动态默认图库

通过 P1 后扩展至所有层，将每个默认桶的内部拓扑固定为一个离线选择的默认容量对。
每种执行策略每个桶一张图，跨桶选图不建立组合缓存。
在未用于调参的 batch 序列验证桶内、跨桶复用，保留 B98 固定样本，不用极端 padding 替代。
统计完整 step 时间、尾延迟、捕获次数、图内存和每步 metadata 成本。

### 后续条件方向

1. 融合 Q/K norm、RoPE 与分支布局生成，直接生产固定分支输入；通用融合也应用于 padding。
2. 在固定容量槽位内按 KV 工作量分配请求，只重排 metadata，不搬迁物理 KV。
   对 padding 应用相同排序，并计入输入排列及最终输出恢复的成本。
3. 设备长度 attention 属于独立算子工程；现有向量原型不能直接替代 FIA。
   原生提交实验没有获得整步收益，不能以免 host update 推断算子重写必然获益。

## 环境与实施边界

使用现有 910B3、驱动 24.1.rc3、CANN 9.1 环境；不升级驱动、不重启、不跳过 FIA update。
先交付 benchmark 与回归测试，不改生产 runner 的默认执行策略。
小桶或接近桶上限的负载不承诺普遍加速；104 到 98 的行数差本身只占 104 的 5.77%，
Dense 同为 104 时，超越 padding 的直接来源必须是 attention 等阶段的净执行效率改善。

## 已实施内容与实测结果（2026-10-16）

P0 的 benchmark 和回归测试已实现。入口为 `benchmarks/multi_graph_bucket_block.py`，
支持原 ND eager 参考、对称 NZ、对称原生 FIA 提交、attention 单独限核、独立 trace。
每种策略捕获一张图，padding 一个 FIA 更新任务，split 两个 FIA 更新任务和独立 workspace。
检查图、输入、position、页表、slot 和输出地址在资格轨迹前后保持不变。
这里未逐个断言所有临时张量地址，也没有将 block 的跨结构误差作为生产准入证明。

P1 在 Qwen3-4B、BF16、Q1、KV512、默认 104 桶完成七组独立进程实验。
以下为动态轨迹 B98 → B97 → B103 → B98 的结果：
单步列为五轮 p50 的中位数，32 步列为五轮总耗时的中位数；所有正式轮次均为 32 步。
NZ 权重在两边共享，原生提交对两边同时启用，每次 replay 保留全部 FIA update。

| 配置 | padding 单步 ms | split 单步 ms | split 增时 | padding 32 步 ms | split 32 步 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python 全核，首轮 | 1.6286 | 1.6509 | 1.37% | 52.3371 | 53.8828 |
| Python 全核，独立复测 | 1.6148 | 1.6429 | 1.74% | 51.8865 | 52.8826 |
| Python attention 10 核 | 1.5928 | 1.7045 | 7.02% | 51.5841 | 54.7729 |
| 最终源码 Python 全核对照 | 1.6279 | 1.6657 | 2.32% | 52.3870 | 53.5685 |
| 原生提交，全核 | 1.5483 | 1.5708 | 1.45% | 49.6987 | 50.4032 |
| 原生提交，全核独立复测 | 1.5373 | 1.5781 | 2.65% | 49.4402 | 50.6945 |
| 原生提交，attention 10 核 | 1.5338 | 1.5686 | 2.26% | 49.4267 | 50.3371 |

固定 B98 的七组实验中 split 也均更慢，单步增时范围为 0.46%–6.65%。
全部正式运行通过原 ND eager 参考检查和长度变化资格检查，容差未调整。
这些数据只覆盖 embedding 与第一层真实 decoder block，不能乘以层数推算完整模型时间。
不同进程间会有运行状态波动，原生提交的绝对降幅也不能直接解释为独立因果收益。

独立 trace 证实两边每个 block 均只有四个 MatMul，split 的两个 FIA 位于两条流。
Python 全核的三次 split trace 未观察到 FIA 重叠；原生全核为 0、31.75、45.00 μs，
但 split kernel 时间并集仍约 609.5–612.0 μs，padding 约 602.5–604.25 μs。
原生提交加 10 核后重叠增加至 168.0–188.25 μs，
两路 FIA 时长总和却增至 547.5–553.75 μs，block kernel 并集约 657.75–682.0 μs。
因此已经观察到设备重叠，但当前资源限制引起的单算子变慢超过了并行收益。
不能据此断言硬件完全饱和，也不能把尚未逐项分析的内部 copy 视为零开销。

首次 smoke 曾因在同一 KV cache 上反复缩短再增长长度而失败，失败日志保留。
修正后的资格检查每次恢复 prefill KV；正式动态计时固定 KV512，仅改变 batch。
后续真实生成、KV 持续增长、跨桶切换和完整 decoder 性能仍需单独验证。

P0 已交付，P1 的当前结论为未获得完整 block 净收益，P2 的推进条件尚未满足。
下一轮应针对 FIA 分支执行成本及输入布局研究可验证改动，再用同一 block 判据验收。
当前实现作为独立实验入口保留，尚未接入生产 runner。

验证记录：CPU multi-graph 回归 355 项通过；910B3 上 Qwen3-0.6B 的 ND、NZ、
NZ＋原生提交三组 NPU 回归全部通过，包含固定分界、动态 batch/长度与图复用检查。
Ruff、markdownlint 与变更文件的导入检查通过。
全仓 `format.sh ci` 已执行，仍受 Windows 缺少 `/bin/bash`、shellcheck、
部分 hook 的 `python3` 启动失败，以及既有 `device_length_decode.py` 直接导入 Triton 影响；
因此全仓检查未全绿。长函数检查和 symbolic meta 检查另用本地 Python 执行通过。

## 复现方式

在现有远端容器加载 `/etc/profile.d/ascend-multigraph.sh` 后，从仓库根目录执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_block.py \
  --model /models/Qwen3-4B --nz --native-update \
  --rounds 5 --repeats 32 --trace --output /tmp/bucket-block-native.json
```

去掉 `--native-update` 为 Python 提交对照；增加 `--attention-cores 10` 限制分支 attention。
每次使用新的输出路径，正式计时完成后才采集 trace。
报告保存默认 capture sizes、固定分界、每步真实请求数、所有样本、精度检查、源码 SHA256。
本地原始数据和源码快照保存在工作区 `outputs/multi_graph_bucket_block/`，
`summary.json` 汇总七组固定及动态轨迹，`summarize_bucket_block.py` 校验快照哈希与采样契约。
