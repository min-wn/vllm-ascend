# ReplaySSM 最小 A2 对照：ready 的独立收益与主线资格

日期：2026-10-20。

## 1. 结论

**在保留环形日志、动态游标和周期 flush 的 ReplaySSM 算法上，
A2 请求组 ready 消费仍有独立净收益。这个结论仅在本轮通过精度门槛的 FP32 日志变体上成立。**

同一混合核只改变产品发布与消费时序，两个独立进程共六轮，
40 步真实单层轨迹计入一次导入、所有 flush 和末尾接受态物化后，
ready 相对 delayed 的墙钟缩短 **4.30%～7.40%**，逐轮百分比均值 **5.53%**。
按两种模式总成本均值计算为 5.55%，与逐轮百分比均值的统计口径不同。
每条轨迹平均节省约 7.10 ms，摊到 40 步约 178 μs/步。

但有两个明确限制：

1. 社区 BF16 激活配 FP16 d/k 日志的原配置，在第二步的真实层精度门槛失败。
   失败结果保留，没有放宽阈值或继续统计性能。后续 FP32 d/k 是另立的归因变体。
2. 当前 FP32 环形 ready 版的闭合成本仍比既有 compact ready 高 **13.77%～15.70%**，
   常驻状态也更多，不能作为替换 compact 的更优方案。

因此，**请求组 ready 可以继续作为有条件的 A2 执行机制研究主线；
ReplaySSM 原配置的可用性、优于社区最佳或完整模型收益均未成立。**
本轮补上的是“同社区算法的 A2 ready 消融”，不是新的紧凑状态算法。
共享规划和变长 query attention 筛选保持结束；没有增加路由器或扩大模型。

## 2. 社区依据与独立差异

固定审阅 [vLLM PR #49887](https://github.com/vllm-project/vllm/pull/49887)，
提交 `5b96cdbceefcdec6d60d588420ee127d76bd8f97`。
核查时已关闭、未合并，仍属于必须对照的已有工作。
本轮未把前缀缓存 PR #56466 或 RecoverSSM 当作此算法的替代基线。

社区核心文件：
`vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py`。
原字节源码及 SHA 沿用
[上一轮社区源码清单](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/community-source-manifest.json)。

| 内容 | 社区源码位置与已有机制 | 本轮处理及独立差异 |
| --- | --- | --- |
| 状态表示 | 核第 98 行起，checkpoint 加 d/k/g 环形日志，逻辑位置映射到物理环。 | 保留算法；不是新表示。A2 为每个 value head 复制 key，并填充 g/cursor 到 32 字节。 |
| 动态接受前缀 | 第 360 行起，`_advance_gdn_spec_cursors_kernel` 按接受长度推进游标。 | 接受长度留在设备；同一固定图读变化值。不是独立创新。 |
| 周期 flush | 第 180～280 行，flush 物化旧历史，再用新状态计算当前投影；第 443 行起有固定 launch 路径。 | 保留触发规则和当前输出依赖；A2 使用独立前置 AIV flush，而非 GPU 原核布局。 |
| 日志精度 | BF16 激活时 d/k 用 FP16，g 和 checkpoint 用 FP32。 | 首先照此做，真实层未过门槛；再独立测试 FP32 d/k。不能称后者是社区原配置。 |
| ready 流水 | 本次审阅的 GPU 核没有这里的 A2 Cube/AIV 请求组跨核发布协议。 | Cube 每组生成两个 key head 的产品，配对 AIV 各处理三个 value head，逐组提前消费。 |
| 对照强度 | 社区 GPU 核有自己的平台优化。 | 本轮只证明相同 A2 移植算法、相同资源下的时序贡献，不证明与 GPU 核性能等价或优于社区最佳 A2 实现。 |

本轮真正可辨认的贡献是：**将环形状态算法中的 checkpoint 产品生成与历史修正/求解，
映射到 A2 的 Cube/AIV 配对流水，并用同算术 delayed 控制隔离消费时序的收益。**
算法中的日志、回滚、投影修正、周期物化均不作为原创。
“所审源码没有同一协议”也不能外推为所有工作都没有类似流水。

## 3. flush 为什么不能直接延迟提交

GDN 每个 token 的状态为：

```text
S[t] = exp(g[t]) * S[t-1] + d[t] * k[t]^T
```

非 flush 步从旧 checkpoint 计算当前 Q/K 的产品，
再把已经接受的历史日志作用到这些小产品上，得到当前窗口的输入。
触发 flush 后，社区核先生成已接受的状态 `sw_f`，
当前步的 `hw_q`、`hw_k` 直接消费它（固定源码第 239、245、246 行）。
这个依赖不能用旧 compact 版的输出投影延迟提交钩子代替。

本轮执行顺序为：

```text
上一窗口接受长度到达
    → phase 0：推进动态游标
    → phase 1：需要 flush 的 owner 先物化 checkpoint
    → Cube：checkpoint × 当前 Q/K
    → AIV：历史日志修正、小三角求解、当前候选输出及 d/k/g 入环
    → 官方 norm / out projection / TP / MLP
    → 本步完成
```

`ring_delayed` 等对应 Cube 的全部分配任务完成后才放行 AIV。
`ring_ready` 每个请求组完成即发布，最多两个未消费组在途；
null 任务也参与 flag 协议，避免有效性分支破坏进度。
两者使用同一套 ring 求解、游标、前置 flush、核数和数据布局。
本轮没有借用旧 compact 的输出投影提交重叠来解释 ring ready 的收益。

末尾对每个 owner 接受最后候选前缀，再物化剩余历史并清空 pending/cursor。
暂停的 owner 也要收尾，不能在 live 较小的末步直接丢弃其状态。

## 4. 冻结的输入和精度契约

沿用 Qwen3.5-27B TP2，第 4 层（从 0 编号）的真实权重及官方 conv、门控、
norm、TP 通信和 MLP；只执行这一层。
hidden/residual 为固定种子的合成输入，状态在连续轨迹内持续推进。
这不是任务文本、完整模型或服务负载测试。

| 项目 | 冻结值 |
| --- | --- |
| query | Q4，包含 bonus 的接受前缀为 1～4。 |
| 图容量 | 400 token，100 个请求 owner；不为候选新增专用容量。 |
| 轨迹 | 40 步；前 8 步沿用原 Q4 live 序列 `98,97,99,98,98,97,99,98`。 |
| 后 32 步 | 固定 `98,100,99,100,98,100,99,100` 周期，使暂停 owner 可以恢复。 |
| 接受长度 | `1 + (owner + 已生成窗口次数) % 4`；首次无 pending 不提交。 |
| 随机种子 | `20719 + step`，每步固定。 |
| 环 | B=16，T=4，逻辑长度 L=20，物理长度 32。 |
| 连续覆盖 | owner 各有 2～7 次 flush；前 99 个 owner 均至少一次回绕；末位 owner 有暂停恢复。 |
| flush 实际支付 | 40 步中 23 步有 owner flush，合计 692 个 owner-step flush；不在轨迹中 reset。 |
| 精度门槛 | 对官方路径逐步最差请求 hidden/residual NRMSE ≤0.005，最终接受态同阈值，全部有限。 |
| 同算法控制 | delayed/ready 每步输出、卷积状态和全部 ring mutable 必须 bit-exact。 |

### 4.1 原 FP16 配置：失败并保留

最小算子 40 步通过 delayed/ready exact 与独立物化参照，
但进入真实单层，第二步 hidden 最差请求 NRMSE 为 **0.00520601496**，超过 0.005。
旧 compact 同步的误差较低。正式资格立即停止，没有该配置的性能结论。

两步诊断中，取实际 recurrence 输入，按同一环日志精度逐 token 物化的参照与原生输出，
两 rank NRMSE 分别约 `8.31e-6`、`5.20e-7`，delayed/ready 仍 exact。
这支持日志量化误差经真实层后续运算放大的解释，
但没有证明所有误差完全来自量化，也不说明社区 GPU 算法存在错误。

该配置的源码、失败日志和诊断保留在 `fp16-attempt.tar.gz`。
顶层 `diagnostic.json`、`diagnostic.log` 也是这次 FP16 诊断，**不是 FP32 性能证据**。

### 4.2 FP32 日志变体：门槛通过

独立冻结 d/k、g、checkpoint 均 FP32，激活仍 BF16，关闭 HF32，
保持算法、轨迹、阈值和对照不变。
FP32 增加日志存储及传输成本，必须在该配置自身下计时。

两个进程累计 640 条 rank/step/mode 资格记录全部通过。
每轮计时后还把全部 mutable 与非计时资格轨迹的闭合参考做 exact 比较，累计 48 次。

| 误差指标 | compact ready | ring delayed / ready |
| --- | ---: | ---: |
| 40 步最坏请求输出 NRMSE | 0.00313131 | 0.00329969 |
| 最终接受状态最坏请求 NRMSE | 1.65e-7 | 1.69e-7 |

ring delayed/ready 自身逐步 exact；null/暂停行不变，dense 初始化 SSM 不被 ring 路径写入。
独立算子物化参照的最大输出 NRMSE 为 `7.85e-5`，终末状态为 `8.37e-8`。
这只是冻结筛选门槛下的连续正确性证据，不能替代文本质量或生产可用性认证。

## 5. 完整状态生命周期的成本

计时包括 CPU 描述符拷贝打包及 H2D、输入 D2D、整层图 replay、
游标维护、全部周期 flush、每步同步、末尾接受态物化、一次状态导入。
每轮内部连续执行 40 步，不重置历史；轮间恢复初始态不计时。
模型加载、输入合成、编译/捕获及诊断快照不计时。
输入上传位于事件区间内，表中的设备跨度不是纯 kernel 执行时间。

两个独立进程各三轮，执行顺序正/反/正。
每个 mode、每个 round 取两 rank 的最大成本，再配对比较。
一次导入每进程/rank 仅测一个值，不能据此声称初始化分布稳定。

| 路径 | 40 步及收尾墙钟均值，不含导入（ms） | 加一次导入的闭合墙钟均值（ms） | 设备跨度均值（ms） | 末尾收尾设备跨度均值（ms） |
| --- | ---: | ---: | ---: | ---: |
| 官方逐候选状态 | 129.515 | 129.515 | 122.921 | 0.109 |
| 既有 compact ready | 102.707 | 105.852 | 96.373 | 0.415 |
| ReplaySSM A2 FP32 delayed | 125.276 | 128.065 | 118.629 | 0.790 |
| ReplaySSM A2 FP32 ready | 118.168 | 120.962 | 111.785 | 0.788 |

官方端点已经持有全部候选 dense 状态，最终接受态由索引标识；其收尾没有额外状态恢复核，
表中这段仍包含共同的终末描述符上传及事件端点。
ring/compact 的延迟状态必须实际物化，不能只比较未支付负债的前向输出。

| 独立进程 / 轮次 | ready 相对 delayed 闭合墙钟改善 | ready 相对 compact ready 闭合墙钟改善 |
| --- | ---: | ---: |
| 1 / 0 | 6.27% | -13.77% |
| 1 / 1 | 5.77% | -14.06% |
| 1 / 2 | 4.79% | -13.85% |
| 2 / 0 | 7.40% | -15.70% |
| 2 / 1 | 4.30% | -14.17% |
| 2 / 2 | 4.67% | -14.11% |

ring ready 相对官方端点六轮改善 4.88%～9.48%，
但其中同时包含状态算法变化与执行流水，不能全归因给 ready。
能归因给 ready 的是同算法 delayed/ready 对照。

一次导入：ring 约 2.77～2.81 ms，compact 约 3.11～3.16 ms。
不含初始化分布、owner 回收和新请求复用；这里的“完整生命周期”限定为
**固定已有 owner 从 dense 初态导入，到最后接受态物化并清 pending 的闭合成本**。

### 5.1 存储账本与差距

每 rank 常驻 mutable：ring **229.63 MiB**，compact **161.34 MiB**，差约 68.29 MiB。
这是冻结布局的状态张量，不含图池、临时产品缓冲、原 dense 初始化夹具和模型权重，
也不是社区原生共享 key 布局的内存成本。
FP32 d/k 相比本端口的 FP16 d/k 多约 37.88 MiB；收益结论已包含该变体的实际运行成本。

本端口的长历史修正、前置 flush、复制 key 和填充标量均可能影响绝对成本；
本轮没有 profile 将 ring/compact 约 15.11 ms 的轨迹差额逐项归因。
不能把全部差额说成算法固有 flush 开销，也不能把全部差额视为可消除预算。
ready 消融已经消除其中约 7.10 ms，但没有反转最终优劣。

## 6. 主线准入与尚缺的决定性证据

| 判断 | 本轮结论 |
| --- | --- |
| 社区算法上 ready 是否仍有独立净收益 | **有，在通过门槛的 FP32 A2 变体上六轮均成立。** |
| 是否把环形 ReplaySSM 换成当前默认方案 | **否，当前比 compact ready 更慢且更占状态内存。** |
| 原社区精度配置是否可用 | **尚未成立，本次 A2 原配置未过真实层门槛。** |
| 是否具有研究主线资格 | **有条件保留 A2 ready 执行机制主线；不确立新的状态算法或生产主线。** |

如果下一阶段要把结论推进到社区原配置，决定性缺口是：
**在不放宽阈值、不偷延迟 flush、不额外复制完整状态的条件下，
解释并解决 FP16 日志的真实层精度失败，然后在同一轨迹重做同算法闭合对照。**
若保留 FP32 作为设计的一部分，必须明确它的资源成本和适用边界；
若要替换现有 compact，还需要直接证明更低的闭合成本，当前证据恰好相反。
这些缺口不需要先扩大模型或新增路由器。

## 7. 源码、证据和复算

| 内容 | 仓库位置 |
| --- | --- |
| 冻结轨迹与游标参照 | `benchmarks/multi_graph_replayssm_protocol.py` |
| 状态适配与 flush 依赖 | `benchmarks/multi_graph_replayssm_backend.py` |
| Cube/AIV delayed/ready | `benchmarks/multi_graph_replayssm_kernel/replay_kernel.cpp` |
| 环历史修正与当前窗口求解 | `benchmarks/multi_graph_replayssm_kernel/ring_solve.h` |
| advance / flush / drain / reset | `benchmarks/multi_graph_replayssm_kernel/state_kernel.cpp` |
| 独立算子物化参照与连续资格 | `benchmarks/multi_graph_replayssm_probe.py` |
| 真实单层精度和闭合成本 | `benchmarks/multi_graph_replayssm_layer_probe.py` |
| 原始证据校验与统计复算 | `benchmarks/multi_graph_replayssm_analyze.py` |
| 结果目录 | `benchmarks/results/multi_graph_structural/replayssm-a2-v1/` |

结果目录保留两个生命周期 JSON、原始日志、FP16 失败源码归档、
FP32 被测源码归档、环境及二进制 SHA、归档逐文件校验。
被测源码按原字节归档，之后的格式检查不改写历史证据。

```bash
python benchmarks/multi_graph_replayssm_analyze.py \
  benchmarks/results/multi_graph_structural/replayssm-a2-v1
pytest --noconftest tests/ut/worker/multi_graph/test_replayssm_protocol.py -q
```

硬件环境仍为 910B3/A2、CANN 9.1.0、Torch/Torch-NPU 2.10、FLA NPU A2
`26.9.1+deva4a7958`，安装源码 revision 与实际使用的二进制 SHA 分开记录。
本轮只在隔离 benchmark 中加载新扩展，没有替换生产算子。

针对性检查：7 个 CPU 协议测试、Ruff 检查/格式、clang-format 21.1.2 和原始结果复算通过。
测量后 C++ 只有格式差异，已用相同 clang-format 规范化两份源码核对；Python 被测源码的 LF SHA 相同。
`bash format.sh ci` 已尝试，在本机 WSL 读取既有 CRLF `format.sh` 时退出，未能进入 hooks；
不把它记为全库 CI 通过。原始失败输出和检查结果见 `format-ci.log`、`validation.json`。
