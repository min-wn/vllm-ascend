# FP32 compact 首步数值失败：有边界的诊断与收口

日期：2026-10-20。对象：Qwen3.5-27B TP2，A2 / 910B3，原 Q4 轨迹第 0 步。

## 1. 结论与下一阶段资格

**当前证据支持：recurrence 的 FP32 计算重排产生极小局部差异，
随后在 BF16 激活、norm、投影、MLP 及后续层中扩散和放大。**
没有发现可直接最小修复的初态、owner 索引、输出对应关系或精度配置错误。
没有修改状态算法、算子实现、精度配置或原门槛。
这次诊断没有使部署候选重新获得数值资格，**维持 compact 整网部署候选结束，不进入完整数值复验**。

可以定位到“首次差异发生于 recurrence，随后经过哪些真实模块放大”；
尚未逐条拆清 native Cube 规约、向量规约、exp 和 scale 重排分别贡献了多少差异。
不将当前证据写成“已找到一条有缺陷的 native 指令”，也不将前两层的局部对照推广为全部层均无实现问题。

已有 delayed/ready 调度一致性和 **4.801%** 的三对闭合成本改善保留，
它们仍是同算法执行机制的证据，不等于 compact 相对官方整网的数值资格或服务吞吐收益。
FP16 排查、共享规划与变长 query attention 筛选保持结束；本轮不自动开展新的候选搜索。

## 2. 本轮实际做了什么

沿用原模型、默认图目录、BF16 权重和激活、FP32 SSM/日志，HF32 关闭。
第 0 步为 **B98、392 个有效 token、默认 400-token 桶**，没有为诊断缩小 batch 或另造捕获尺寸。
所有 128 个 owner 的共同官方前缀重新生成一次；已有目录没有可复用的 FP32 首步张量快照。
前缀 token SHA、模型加载器、完整默认目录及调度结果均与上一轮原始证据相同。

本轮只进行一对官方/compact 首步整网执行；每 rank 在 64 个 decoder 中记录 **1408 条**边界误差，
包括全部 48 次 GDN recurrence 和 hidden/residual 的 128 个 decoder 输出。
随后在第 0、1 层分别使用相同输入和相同初态做局部对照，最后做一次首层 recurrence 输出替换。
没有重跑完整 128 步、捕获整套图、增加性能轮次或扩大负载矩阵。
这是 eager 诊断；原图回放的 ready 调度资格继续引用上一轮证据。

另用保存的四份局部快照完成 CPU 算术核对。
FP64 只用于离线检验公式和舍入敏感性，**不是新增部署精度方案，也没有替代原官方对照或精度门槛**。

## 3. 起点与输出对应关系核对

两 rank 的起点审计均通过：

| 核对对象 | 证据与结果 |
| --- | --- |
| 初始接受态 | 全部 48 层的官方四候选均由同一真实前缀 FP32 base 复制；逐元素、逐字节与 compact base 对应一致。四候选初始相同，因此首步接受长度选择不会引入不同历史。 |
| 日志与生命周期 | compact 日志和 valid 全零，generation 为 0，last_step 为 -1，pending 为空；null base 和 null candidate 为零。 |
| 卷积历史 | 全部 48 层的独立 conv 缓存一致；官方对照运行后再次检查，compact conv 没有被它改写。 |
| FIA KV 历史 | 全部 16 层、共 32 份 K/V 历史一致；两个对照持有独立缓存，官方执行没有改写 compact 历史。 |
| owner 与候选 | owner i 对应 compact 行 i+1，官方候选为 1+4i 至 4+4i；设备 lookup、真实候选 ID、接受长度一致。batch 行号没有作为 owner 使用。 |
| 位置与描述符 | 每个请求的首个 query 位置等于其真实前缀长度；位置、FIA slot、接受数和 query 长度按原轨迹构造，recurrence 两侧的元数据另作 exact 核对。 |
| 输出对应 | 按相同输入行顺序取前 B×4 个 token，再 reshape 为 [B,4,...]；owner 轴严格使用原 request_ids。norm 展平的 head 轴恢复为同一 token 轴后比较。 |
| 比较时机 | hook 当场复制到 CPU，避免后续 fused residual 原地修改已经观察的参考值。官方与 compact 分别保存最终 norm 的真实 hidden/residual。 |

首步 hidden 最坏请求 NRMSE 再现为 **0.0118161086**，上一轮为 0.0118161095；
约 9.3×10⁻¹⁰ 的差别来自指标归约位置，不改变失败结论。
residual 为 **0.0100735165**。原门槛始终为 **0.005**。

## 4. 首次差异及其放大位置

第一层的 embedding、input norm、输入投影、卷积后的 Q/K/V、g 和 beta 全部 exact。
**最早差异位于 layer 0 recurrence 输出**：每 rank 的 1,204,224 个有效元素中，
分别有 63、86 个 BF16 元素不同；最坏请求 NRMSE 分别为 2.086×10⁻⁶、2.765×10⁻⁶。
这不是 recurrence 自身在第一层就产生百分比误差。

下面展示 rank 0 的真实整网传播；每行均为该边界的最坏请求值，不表示始终是同一个 owner，
因而不能把这些数相除解释为固定请求的条件数。

| 边界，层号从 0 开始 | 最坏请求 NRMSE |
| --- | ---: |
| layer 0 recurrence 输出 | 0.000002086 |
| layer 0 gated norm 输出 | 0.000064035 |
| layer 0 attention 输出投影 | 0.000060444 |
| layer 0 post-attention norm | 0.000351511 |
| layer 0 MLP gate/up 投影 | 0.000756025 |
| layer 0 MLP 激活 | 0.000979717 |
| layer 0 MLP/down 及 decoder hidden | 0.004002500 |
| layer 1 recurrence 输出，已含上游输入差异 | 0.003623889 |
| layer 1 gated norm 输出 | 0.006067574 |
| layer 1 attention 输出投影 | 0.008084437 |
| layer 1 MLP/down 及 decoder hidden | 0.011550651 |
| 最终 hidden | 0.011816109 |
| 最终 residual | 0.010073517 |

两 rank 第一个超过同一数值门槛的被观察模块均为 **layer 1 的 gated norm**；
第一个超线 decoder 为 layer 1，TP all-reduce 后的 hidden 误差一致。
norm/投影/MLP 沿用相同官方实现；它们两侧输入已不同，不能据此说这些模块存在实现错误。

### 同输入、同初态的局部对照

局部对照取官方整网执行在该层入口的真实 hidden/residual，复制相同 conv 和初始 FP32 state，
分别执行真实 decoder。两个对照在 recurrence 前的 Q/K/V、g、beta 均 exact。

| 层与 TP rank | recurrence 局部误差 | decoder hidden 局部误差 |
| --- | ---: | ---: |
| layer 0，rank 0 | 2.086×10⁻⁶ | 0.004002500 |
| layer 0，rank 1 | 2.765×10⁻⁶ | 0.004002500 |
| layer 1，rank 0 | 1.836×10⁻⁵ | 0.001421250 |
| layer 1，rank 1 | 2.172×10⁻⁵ | 0.001421250 |

layer 1 在整网中的 hidden 误差为 0.011550651，而相同入口输入的局部对照为 0.001421250。
这把很大一部分整网差异与该层局部计算差异区分开：上游传播不能算成该层 recurrence 本身的错误。
同时该层仍有自己的微小 recurrence 差异，不能认为只修第一层就会消除所有后续差异。

### 唯一一次中间结果替换

恢复所有原始前缀状态，只把两个 TP rank 的 layer 0 recurrence 输出替换为各自官方输出；
compact 的状态计算与提交照常执行，不修改其他层。本步每层状态独立且只消费一次，
该层的提交结果不会在同一步被它自己再次消费。

替换后 hidden 的 pooled NRMSE 从 0.009270932 降到 0.009184925，
但最坏请求仍为 **0.011816109**；residual 最坏值仍为 **0.010073517**。
因此本次替换不足以解释或修复最终超线。
其作用仅是排除“修好第一层一个孤立点即可过线”的判断；没有做第二次替换。

## 5. 算术证据与源码位置

官方按照 token 顺序执行：先对状态施加 decay，再计算 delta、更新完整状态，最后做 query 投影。
compact 先计算初始状态与四个 K/Q 的产品，再通过 Gram 项和三角关系求出更新与输出；
官方先缩放 query，compact 在输出末尾缩放。这些在实数算术中等价，但 FP32 的乘加与规约顺序不同。
即使大状态始终是 FP32，recurrence 输出及后续激活仍是 BF16。

在实际冻结的 98 个请求、前两层、两 rank 四份快照上：

- FP64 两套公式的最坏请求 NRMSE 为 **2.17×10⁻¹⁶～2.71×10⁻¹⁶**，支持公式等价，没有显示大尺度公式或布局错误。
- CPU FP32 两套公式的舍入前误差为 **2.11×10⁻⁷～3.51×10⁻⁷**；
  转为 BF16 后约为 **2.11×10⁻⁶～2.12×10⁻⁵**，与实测 native 局部差异同量级。
- native 官方和 compact 都接近离线 FP64 的 BF16 舍入结果；这不是仅 compact 相对高精度出现巨大偏差的现象。
- 不同元素大多相邻，但不是全部仅差 1 ULP。四份 native 输出的最大 BF16 ULP 距离为 **3、84、25、23**；
  对应整体误差仍很小，不能用“全是一个 ULP”作过度简化。

CPU exp 和规约并不逐位模拟 Ascend，因此这里的结论是支持计算重排和数值传播这一原因类别，
不是声称已经精确复现每条 native 指令的误差。没有借用 FP16 排查的日志或归因。

| 源码对象 | 可复核位置 |
| --- | --- |
| 本机官方 GDN 调用、L2 norm、scale、真实门控与 norm | 固定安装版 `vllm_ascend/ops/gdn.py` 的 speculative recurrence 分支及 `forward`；原件收录于本轮源码归档的 `source-audit/installed-gdn.py`。 |
| 本机官方 recurrence | `source-audit/installed-recurrent.h` 中 `CopyInQKV`、`ReduceSumDispatch` 和 `Compute`，尤其先 decay、delta、state update、query 投影的顺序。安装头文件 SHA 为 `7661897a77d6cf8f4e2c3f9c44d24c1fdb3547ee3d3e55435b0857b1192cb17f`。 |
| compact 产品计算 | `benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp` 的 `Producer::Produce`；FP32 输入并显式 `SetHF32(false,0)`。 |
| compact 三角计算与 BF16 边界 | `benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/q4_tile_solve.h` 的 `ComputeGram`、`Projection`、`Solve` 和 `WriteOutput`。 |
| 状态、owner 与调用契约 | `benchmarks/multi_graph_gdn_q4_log_backend.py` 的 `Q4LogState.from_base`、`Q4LogBackend.__call__`。 |
| 独立官方缓存与原阈值 | `benchmarks/multi_graph_gdn_full_numeric.py` 的 `OfficialBanks`、`error_record`，未修改原 0.005 门槛。 |
| 本轮诊断 | `benchmarks/multi_graph_gdn_fp32_step0.py` 的 `audit_start`、`local_pair`、`diagnose`；离线核对在 `multi_graph_gdn_fp32_arithmetic.py`，证据验证在 `multi_graph_gdn_fp32_analyze.py`。 |

本机官方使用 `fla_npu` A2 26.9.1+deva4a7958，安装源文件和运行库哈希均保存。
来源以本机实际安装字节为准，不把社区最新分支或不同提交的头文件混作当前执行源码。
本轮没有提出新的状态表示或精度方法；与已有工作的差异仍限定为此前 A2 请求组 ready 的执行时序，
不因本次数值诊断扩大创新声明。

## 6. 有限修复路径与收口判断

目前没有已定位、能通过一次有限代码修补恢复原精度门槛的错误。
单改 owner 映射、接受数、初始状态或比较索引没有证据依据；这些项目已通过核对。
简单调整某一处缩放或规约顺序能否消除全部层的注入、并使 128 步通过，没有证据，不能称为明确修复路径。

若要保证官方运算的舍入行为，需要进一步改变计算组织或制定并验证新的数值策略；
这超出本轮边界，因此没有实施，也没有继续枚举 kernel 微调。
剩余疑点是 native 内部各数值操作的精确贡献及未做同输入隔离的其余层，
不是初态或缓存起点尚未核对。

**下一步资格判断：维持当前 compact 部署候选结束。**
不安排完整数值复验，不扩大模型、负载、精度或路由器。
即使将来在另行授权下修复第 0 步，也只能据此重新申请完整连续轨迹的数值资格验证，不能直接宣布整网通过。

## 7. 证据、复现与检查

本轮证据目录：`benchmarks/results/multi_graph_structural/gdn-fp32-step0-v1/`。
原始报告 `step0.rank0.json`、`step0.rank1.json` 保存起点哈希、逐层观测、局部对照和一次替换结果；
`arithmetic.json` 保存四份快照的离线核对，`analysis.json` 给出资格判断。
17 项原始证据和 156 项归档源码均完成 SHA-256 与字节数校验。
四份完整张量快照共约 1.28 GB，保留于远端 `/workspace/gdn-fp32-step0-v2/`，
本地 `snapshot-manifest.json` 记录原路径、大小及哈希，不把大张量强塞入 Git。

复核本地证据：

```bash
python benchmarks/multi_graph_gdn_fp32_analyze.py \
  benchmarks/results/multi_graph_structural/gdn-fp32-step0-v1 \
  --original benchmarks/results/multi_graph_structural/gdn-ready-full-v1 \
  --output /tmp/gdn-fp32-analysis.json
```

原首步启动命令及四扩展路径完整保存在 `command.json`，没有改动被测 SO。
只需重做诊断时使用 `--phase fp32-step0 --dense-nz`，不要调用完整 `numeric` 阶段来替代首步复验。
相关 **87 项 CPU 检查通过**，涉及七个文件的 Ruff 检查与格式检查通过，结果单独保存在 `validation.json`；
全仓 `format.sh ci` 的既有 CRLF 入口失败原样保存，不记作全仓格式通过。
