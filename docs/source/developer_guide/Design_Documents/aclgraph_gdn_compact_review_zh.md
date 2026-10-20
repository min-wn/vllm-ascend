# GDN 紧凑状态复核：社区差异与完整状态生命周期

日期：2026-10-20。

## 0. 后续最小 ReplaySSM A2 对照已完成

同社区环形算法的 delayed/ready 对照已有结果：**通过精度门槛的 FP32 日志变体中，
两个独立进程六轮闭合生命周期墙钟均改善 4.30%～7.40%，逐轮改善均值 5.53%。**
保留动态游标、多周期 flush，并在本步 Q/K 投影前完成 flush，没有套用 compact 的输出投影延迟提交。
这补上了本报告原先缺少的同算法 A2 调度消融。

原社区 BF16 激活搭配 FP16 d/k 日志配置在真实层第二步未过 0.005 NRMSE 门槛，
失败与归因变体分开保留；不能说原配置已经通过。
当前 FP32 ring ready 的闭合成本仍比 compact ready 高 13.77%～15.70%。
因此保留有条件的 A2 ready 执行机制主线，不把环形版替换为默认方案，也不确立新的状态算法。
完整结果、源码差异、精度限制及尚缺证据见
[ReplaySSM 最小 A2 对照](aclgraph_replayssm_a2_review_zh.md)。

以下保留上一轮八步探针及其当时结论。

## 1. 上一轮结论

**保留为有条件的研究主线候选，暂不把它确立为新的状态算法。**
现有实现有可重复的真实单层净收益，值得保留；
但紧凑候选表示、动态接受前缀、日志修正投影和延迟物化均已有相关工作。
本轮能辨认的差异收紧到 **A2 上的请求组 ready 消费与状态提交调度**。
这属于执行机制和工程实现差异，不能等同于已经证明论文意义的首创。

最缺的一项决定性证据是：**在同一 A2、相同精度和状态生命周期下，
与最接近的社区紧凑状态实现对照，证明 ready 调度仍有稳定净收益和持续正确性。**
当前最强非 ready 对照仍是我们自己的实现；它具有同源消融价值，
不能替代 RecoverSSM/ReplaySSM 的公平实测。

共享规划、变长 query attention 两轮筛选保持结束。本轮没有增加路由器、
扩大模型规模或增加完整模型实验；沿用 Qwen3.5-27B 权重加载器，只执行已有的一个真实 GDN decoder 层。

## 2. 社区已有内容与本方案的边界

### 2.1 固定核查版本

| 对象 | 固定版本及状态 | 源码依据 |
| --- | --- | --- |
| vLLM Ascend main | `8237ee6d690d1399bc6eb48ef5977ac2bd470f3f` | `vllm_ascend/ops/gdn.py`，投机路径调用 FLA NPU recurrence。 |
| FLA NPU main | `47d7c97ce439228e0ead6fab4dad8b0440b663a8` | `fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_kernel/`。 |
| 本机插件 | `cf67a3cc9f6ffd7891cceb8b84869f7acedbb195` | 安装源码和二进制 SHA 已保存。 |
| 本机 FLA NPU A2 | `26.9.1+deva4a7958` | 稳定 ABI 包装器调用原生 recurrence；保留库 SHA。 |
| vLLM PR #49887 | `5b96cdbceefcdec6d60d588420ee127d76bd8f97`，已关闭、未合并 | GDN ReplaySSM；保留公开源码，不因未合并而忽略已有工作。 |
| vLLM PR #56466 | `2efc0c896555006911b0edf0e0f089e8b2273094`，开放草稿 | ReplaySSM 前缀缓存原型。 |
| vLLM PR #58863 | `3694757f68bc9dc82f29be876a9de4c4da2b051a`，开放、未合并 | GDN RecoverSSM 的 verify、接受态恢复与 runner 提交。 |

本机 wheel 的版本后缀可以对应公开提交 `a4a7958`，
但不能仅凭版本号证明安装二进制由该源码无修改构建。
本报告区分“本机包装器及实际测量”和“最新公开核源码”，不把两者视为同一个已验证产物。

### 2.2 四项机制逐项复核

| 机制 | 社区已有实现 | 本方案独立差异及限制 |
| --- | --- | --- |
| 一个基态加候选日志 | RecoverSSM 保存 FP32 correction/key/decay；ReplaySSM 使用 checkpoint 和环形日志。 | 我们的一个 FP32 基态加四条日志不是新表示；固定 Q4 和带填充日志布局是适配。 |
| 动态接受前缀 | 原生 FLA 已按设备端接受长度选旧候选；RecoverSSM 按接受前缀恢复；ReplaySSM 更新游标。 | 固定图每次读变化的接受长度不是独立创新。 |
| 延迟提交 | ReplaySSM 周期 flush；RecoverSSM 在采样后恢复；TreeWY 三角求解后重建接受态。 | 我们将上一窗口的物化放到下一次 GDN 的输出投影期间，利用旧基态产品加日志修正解除输出依赖；这是提交时序差异。日志修正投影本身 ReplaySSM 已有。 |
| ready 消费流水 | 当前审阅的 FLA A2 recurrence 是 AIV 路径，逐 token 保存完整候选；审阅的 GPU 紧凑状态核没有同一 A2 跨核发布协议。 | 我们的混合核按请求组发布 Cube 产品，配对 AIV 提前求解；相同混合核的 delayed/ready 控制仅改变消费时序。差异仅限已审源码，不能证明全社区从未有类似流水。 |

RecoverSSM 还支持它所声明的对齐边界、卷积提交和模型接入。
我们的真实层继续使用官方卷积，未实现这些生产级缓存语义；
不能只比 recurrence 时延就声称功能等价。

主要外部依据：

- [RecoverSSM PR #58863](https://github.com/vllm-project/vllm/pull/58863)：verify 缓存小记录，commit 按接受前缀恢复状态。
- [ReplaySSM PR #49887](https://github.com/vllm-project/vllm/pull/49887)：日志、旧 checkpoint 投影修正、固定图与周期 flush。
- [ReplaySSM 前缀缓存原型 #56466](https://github.com/vllm-project/vllm/pull/56466)：块复用、前缀恢复和游标维护。
- [TreeWY，arXiv:2608.20961v1，第 3～4 节](https://arxiv.org/html/2608.20961v1)：三角求解与接受态按需重建；本文不重复声明这部分代数为原创。

源码、状态元数据、固定提交链接和逐文件 SHA 均保存在
[社区源码清单](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/community-source-manifest.json)。

## 3. 当前方案具体在做什么

对一个注意力头，GDN 的状态更新可以写成：

```text
S[t] = decay[t] * S[t-1] + update[t] * key[t]^T
```

原生 Q4 投机路径保存四份完整候选矩阵。
紧凑方案保存一份 FP32 基态，以及四条 key、update、decay 日志；
四个候选仍然在逻辑上存在，不能丢弃未被选中的候选来简化正确性。

上一轮接受长度到达后，当前轮先计算旧基态对当前 Q/K 的产品，
再把已接受日志作用到小产品上，从而得到当前求解需要的投影。
它不必先等待完整基态写回。
求解结束后，上一窗口的接受态物化与当前层的输出投影、TP 归约及 MLP 重叠，
并在层端点显式汇合。

```text
基态 + 上一窗口日志 + 设备端接受长度
             |
             v
Cube：旧基态 × 当前 Q/K，按请求组发布产品
             |
             v
配对 Vector：等待本组 ready → 日志修正 → 当前 Q4 求解
             |
             +----> 当前输出 → gated Norm → 输出投影 / TP / MLP
             |
             +----> 新日志 staging
                          |
上一窗口物化 + 发布新日志 --+--> 层端点 join → 下一步可复用
```

ready 不是向另一张图暴露未完成张量。
生产和消费在同一个 `KERNEL_TYPE_MIX_AIC_1_2` 核内配对执行：
每组包含两个 key head、对应六个 value head；每对 AIV 分担消费，
最多保留两组 outstanding 标志，null 任务也发布标志，避免有效性分支造成等待不匹配。
生产端在 `PIPE_FIX` 发布，消费端逐组等待，最终等待输出写入完成。
这些是当前固定硬件、固定几何下的实现契约，不能据此宣称任意多图轮询安全。

本方案可直接置于完整 Padding 图中。
本轮不要求多图独有收益，也不把该混合核的收益归因于独立图切分。

### 3.1 可复核源码位置

| 环节 | 仓库源码位置 |
| --- | --- |
| FP32 表示和 CPU 数学参照 | `benchmarks/multi_graph_gdn_q4_log_reference.py`：`Q4LogCache`、`_ordered_selected`、`q4_log_step`。 |
| 状态导入、所有权和前缀范围 | `benchmarks/multi_graph_gdn_q4_log_backend.py`：`Q4LogState`、`prepare_step`。 |
| 旧基态产品、日志求解与延迟提交 | 同文件：`Q4LogBackend.__call__`、`launch_pending_commit`、`finish_pending_commit`。 |
| 小产品日志修正 | `benchmarks/multi_graph_gdn_q4_log_deferred_kernel/q4_log_deferred_kernel.cpp`：`Projection`、`ProcessCommit`。 |
| ready 协议 | `benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp`：`Producer::Process`、`ConsumeSolve`。 |
| 完整真实层和官方对照 | `benchmarks/multi_graph_gdn_q4_log_layer_probe.py`：`experiment`，新增 `--lifecycle`。 |
| 连续推进、元数据上传和终末物化 | `benchmarks/multi_graph_gdn_lifecycle_probe.py`：`lifecycle_samples`。 |

接受长度为 1～4，包含一个真实/bonus token。
“接受 0 个 draft”对应长度 1；null 或暂停请求通过零 query 长度和 null ID 表示。
本实现不支持把一个活跃 Q4 请求的总推进长度随意设置为 0。

## 4. 本轮怎样计入状态生命周期

复用已有真实层：Qwen3.5-27B、TP2，第 4 层（0 起始编号），
每卡 8 个 key head、24 个 value head、K=V=128。
固定默认物理容量 400 token，即 100 个 Q4 请求位置；
八步 live 数为 `98,97,99,98,98,97,99,98`，接受前缀随 owner 的生成次数变化。
输入是固定种子的合成 hidden/residual，权重、卷积、门控、Norm、MLP 和 TP 归约均走真实层。

每轮计时包含：

1. 固定轨迹描述符的 CPU staging 打包，以及每步 4,808 字节 H2D。
2. device 常驻层输入的 D2D staging、官方卷积状态更新及本层全部执行。
3. 四个候选日志生成、上一窗口接受态提交、事件依赖与每步完成同步。
4. 八步连续推进；中途不恢复快照，不把每步当成独立初始状态。
5. 末尾为所有 owner（含暂停者）物化最后接受前缀并清空 pending 标志，再结束端点。
6. 单独测量一次 dense→compact 导入，并另报“导入 + 八步 + 终末交接”的总成本。

官方对照已经持有完整候选，终末消费者可以通过接受索引访问状态，
没有人为要求它拷贝出连续大矩阵。
紧凑方案则真实支付末尾的物化和日志失效成本。
因此末尾 drain 没有被推到计时边界以外，也没有通过导出四个候选矩阵夸大对照开销。

冻结输入的生成、图捕获和诊断快照不计入热路径。
每步事件计时是流上的完成跨度，包含上传和可能的提交空隙，不是各 kernel 时长之和。
墙钟还包含 CPU 打包、提交及完成等待。两者分别取关键 rank，不能相加。

一次性图与缓冲区准备时间不属于服务热路径；本轮保留其内存占用，
没有声称完成生产 KV 管理器的分配、驱逐和回收成本验证。

## 5. 实测结果与拆分归因

先运行 v1，再进行仅修改报告说明/打印的 v2 重复。
两次各三轮，模式顺序正向/反向/正向；每个模式每轮连续推进八步。
v1 的被测源码副本单独保留；v2 与当前提交源码的 SHA 一致。

### 5.1 v2 的完整八步端点，折算每步中位数

以下包含终末交接摊销，暂未加一次性导入；单位为 ms。

| 对照 | 流事件端点 | 主机墙钟端点 | 作用 |
| --- | ---: | ---: | --- |
| 官方完整候选 | 3.044 | 3.224 | 实际安装的普通投机状态路径。 |
| 紧凑、融合串行 | 2.757 | 2.919 | 表示与产品/求解算法的共同变化，不能单归因于压缩。 |
| 紧凑、拆阶段串行提交 | 2.831 | 2.995 | 增加 staging 后的提交串行控制。 |
| 紧凑、延迟提交 | 2.595 | 2.764 | 与上一行比较，分离提交重叠。 |
| 同一混合核、延迟消费 | 2.596 | 2.762 | ready 的同源调度控制。 |
| 同一混合核、ready 消费 | **2.425** | **2.592** | 只让各请求组产品提前被消费。 |

关键比较：

- **ready 对同一混合核 delayed**：v2 三轮事件端点改善 `8.96%,6.59%,6.47%`，
  墙钟改善 `8.51%,6.15%,6.07%`。第一轮 delayed 较慢，全部轮次保留。
- **ready 对按全轮平均墙钟选出的最强非 ready 控制（延迟提交）**：
  v2 墙钟改善 `5.47%,6.20%,6.13%`；v1 为 `5.57%,9.11%,6.43%`。
- 更苛刻地让控制每轮选择最小墙钟，两个进程的六轮改善仍为 `5.47%～9.11%`。
  这只是已测控制集合内的结果，不是超过所有社区实现。
- **延迟提交对同表示拆阶段串行提交**：v2 墙钟约减少 `7.67%～7.93%`；
  差值包含重叠对资源竞争的实际影响，不能把理论可重叠窗口全当作收益。

### 5.2 一次性和末尾成本并非免费

每个独立进程测到一次状态导入约 **3.60～3.61 ms/每卡本层**。
ready 的末尾物化约 **0.372～0.380 ms/整条八步轨迹**。
它们是实付成本；导入只在每个进程测一次，不能称为稳定初始化分布。

ready 相对官方路径，八步热端点的 v2 墙钟改善约 `18.70%～19.89%`；
把一次性导入全部压到这八步以后，仅剩 `4.56%～6.10%`。
v1 同口径为 `5.07%～7.38%`。
ready 对已紧凑的延迟提交控制，双方都加相同导入成本后，v2 改善 `4.71%～5.33%`。
因此不能把热端点的约 19% 解释成短生命周期请求也能稳定得到的收益。

原始数据与可重算结果：
[v1](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/lifecycle-v1.json)、
[v2](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/lifecycle-v2.json)、
[分析](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/analysis.json)。

### 5.3 正确性和内存边界

每个进程通过 96 条 rank×step×mode 的 eager/回放检查，
以及 36 条 rank×round×mode 的连续生命周期检查。
同紧凑算法的所有调度控制和终末状态逐元素一致；
终末接受态对官方路径的最坏请求 NRMSE 为 `9.40e-8`，原筛选阈值为 `0.005`。
暂停/null 保持检查、四个逻辑候选及有限值检查沿用真实层探针。
这不是长序列 logits、生成质量或服务正确性证明。

FP32 三角求解重排了归约顺序；保留 FP32 状态不等于与官方 recurrence bit-exact。
RecoverSSM 源码还专门按原序逐 token 恢复来保证其定义下的精确状态，
不能省略精度语义差异后直接比较性能。

本层每卡 canonical base/old-log/valid 为 169,175,808 字节，
另外还需新日志 staging、产品、cast/output 临时区和终末测试缓冲；
终末缓冲额外 10,318,948 字节。
对照进程同时保留 dense 缓存、诊断历史与多个图，
**本轮没有测出生产 KV 池可释放多少内存或可提升多少并发。**

## 6. 是否能成为毕设主线

可以保留的研究问题是：

> 固定 ACLGraph 下，如何让动态接受前缀的紧凑 GDN 状态，与 A2 的 Cube/Vector
> 提前消费和延迟提交共同执行，在支付完整生命周期成本后减少真实层完成时间。

当前证据支持“调度机制有可测价值”：ready 的同源控制六轮均正收益，
提交重叠也有独立消融；末尾物化和一次性导入没有被隐藏。
现有结果不足以支持“新的紧凑状态数学算法”或“已经优于社区最佳方案”。

下一门槛只保留一个决定性对照：
在当前真实层、同一冻结轨迹和 A2 精度约束下，复现最接近的社区紧凑状态基线，
再单独打开 ready/提交调度，比较闭合生命周期总成本与持续状态正确性。
尤其要允许 ReplaySSM 的周期 flush，不能强迫它每步提交后再声称我们更快。
这项对照不需要扩大模型或新增路由器。

若同环境社区基线抵消当前增量，或为了数值契约必须恢复串行大状态依赖，
就结束“独立研究主线”资格，把现有核保留为工程成果。
本轮不继续枚举细碎 kernel 调参，也不重启已结束的两轮筛选。

## 7. 重现与保存

运行入口沿用 `multi_graph_gdn_q4_log_layer_probe.py --lifecycle`，
要求 TP2 和已有 log/products/deferred/tile 扩展库。
环境为 910B3/A2、CANN 9.1.0、Torch 2.10、torch-npu 2.10.0.post4；
固定 `TASK_QUEUE_ENABLE=0`、`HCCL_DETERMINISTIC=true`。

完整冻结协议在
[frozen-protocol.json](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/frozen-protocol.json)，
安装文件和库 SHA 在
[installed-manifest.json](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/installed-manifest.json)。
CPU 重算命令：

```bash
python benchmarks/multi_graph_gdn_review_analyze.py benchmarks/results/multi_graph_structural/gdn-review-v1
```

本轮源码只增加探针和分析，没有替换生产算子。
54 项针对性 CPU 测试和本轮代码的 Ruff 检查通过。
已尝试 `bash format.sh ci`，本机 WSL 提示及既有 CRLF 脚本解析错误阻断该流程，
不能宣称完整格式 CI 通过；错误记录保存在证据目录。
社区/安装源码和原始日志按原字节放入 `source-snapshots.tar.gz`，
解包后的路径与源码清单对应，成员及归档 SHA 另存。
