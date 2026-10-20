# 固定 ACLGraph 下变长 query attention：选题验证与结束记录

日期：2026-10-20。

**本轮筛选结束，未立项新算子。**
共享计划优化候选保持结束状态。新轨迹、当前社区和本机源码核查、
最小算子探针、原 Q4 回归均已完成。
找到的具体重复步骤是 attention 核内的输出初始化及其核间同步；
但没有取得能够单独归因、达到准入门槛的可消除时间预算。
因此没有实现候选，不继续枚举 kernel 微调。
这不是证明核内初始化成本为零，也不是证明变长 query attention 没有任何优化可能。

## 1. 先冻结应用与轨迹

目标应用是**每请求验证长度可变的投机验证 attention**。
这里验证算子机制，不研究 draft 接受率，不声称已运行动态投机服务。

轨迹是明确标注的合成快照：沿用原 Q4 轨迹的 128 步请求次序、请求 ID、
batch 和 query 之前的上下文长度，以 seed `20261020` 独立生成 Q1～Q4。
预设概率为 5%、10%、15%、70%，不能解释为测得的置信度或接受率。
每个快照独立成立，不构成接受/回退过程的连续服务记录。

在第一次 NPU 测量前，已经保存
[输入轨迹](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/input-trace.json)，
并在其中冻结探针步骤、profile 步骤、语义和准入条件。
轨迹规范化 SHA256 为：

```text
17a08ae8f66864c08aafd94396fa0026fab1c41e76b373b3040a735f602a0a00
```

原 Q4 规范化 SHA256 保持：

```text
0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524
```

实际分布为 Q1/2/3/4 各 647/1249/1873/8724 个请求快照，
合计 43660 个有效 query token，每步有效 token 为 223～448。
只对预先指定的前 8 步测量，profile 固定用第 0、7 步，不根据结果挑输入。
其余变长快照已冻结，未声称都经过设备资格化。

固定几何为 BF16、Q12/KV2、D256、page128、128 个物理请求、512 个物理 query token。
512 来自原默认 capture 目录；始终使用同一张图，不添加专用容量。
本轮没有比较不同容量下的性能，也没有把物理空行比例算成收益。

语义契约如下：

| 项目 | 约定 |
| --- | --- |
| Query 存储 | 固定 TND，活跃 query 紧凑放在前缀，尾部仍是物理存储 |
| Q 描述符 | 128 个 int64 累积终点，不含开头的零；不能当成每请求长度 |
| KV 描述符 | 128 个 int64 每请求局部长度，包含当前 query |
| 因果关系 | 第 j 个 query 只读到 `kv_length - q_length + j` |
| 分页 KV | 每个请求拥有固定且互不重叠的页，随请求 ID 更新页表次序 |
| 无效请求 | 新轨迹为 Q0/KV0；累积 Q 终点允许重复 |
| 输出 | 整个物理输出有限值，有效域之外精确为零，地址保持不变 |
| 计时输入 | Q/K/V 驻留；本步长度和页表内容改变，必须上传 |

## 2. 当前社区与本机究竟走什么路径

2026-10-20 重新拉取并固定了两个社区版本：

| 对象 | 版本和执行路径 | 对本课题的含义 |
| --- | --- | --- |
| 官方 main | `23e8c5afe172fdcb2d6896180eced8d275620f97`；普通 FIA 通过 `register_task`、`FIAParamProvider` 交给 UpdatableGraph 更新参数 | 这是安装路径核查的参照，不能将旧更新开销算成新 sink 算子的原创收益 |
| 社区 sink PR #15336 | 重新获取的 PR head 仍为 `bec19b8dea1e538e64838f3064f7e91a3fa9ee72`；设备长度生成元数据，attention 消费稳定元数据缓冲 | 设备长度、计划与执行分离、跨层复用都已有社区实现；PR 当前为 Open，不能写成已合入 main |
| 本机安装插件 | repo `cf67a3cc9f6ffd7891cceb8b84869f7acedbb195`；普通 FIA 仍为任务注册更新路径 | 没有把隔离扩展误报为已安装生产路径 |
| 本轮实际探针 | 单独加载前阶段 D256 兼容移植的社区 sink，保留社区 FD 规划 | 不是 stock CANN，也不是已结束候选使用的 Q4 完整头变体 |

社区路径见[当前 main 的 FIA](https://github.com/vllm-project/vllm-ascend/blob/23e8c5afe172fdcb2d6896180eced8d275620f97/vllm_ascend/attention/attention_v1.py#L654)、
[sink PR](https://github.com/vllm-project/vllm-ascend/pull/15336)、
[固定版本的计划封装](https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/vllm_ascend/ops/fia_v2_sink.py#L23)。
封装会 clone 两份长度，为异步 AICPU 读取保留生命周期，再将计划写入稳定缓冲。
这些是每次 metadata build 的工作，不能不经证明按每层重复计费，也不能直接删掉生命周期保护。

本机为 910B3/A2，20 Cube/40 Vector；CANN 9.1.0，
Torch `2.10.0+cpu` 加 `torch-npu 2.10.0.post4`，
vLLM `0.30.0+empty`、插件 `0.30.0rc1`。
这些是包版本标识，不表示本轮运行了 CPU attention 或完整模型。

隔离探针复用 stage2 的 D256 内部分块适配，没有新编译算子。
已逐文件核对：输出初始化所在的 `fia_kernel_nonquant_sink.h`、Vector consumer
与当前 PR head 字节完全一致；Cube 文件含已有 D256 适配；
AICPU 文件仅有既有 include 路径修正，保留社区算法。
实际解析的 OpAPI 地址、扩展 schema、加载库路径和 SHA256 已记录。

可复核材料：
[社区源码清单和行号](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/source-audit.json)、
[本机源码和二进制清单](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/native-audit.json)。
原有兼容适配说明见[stage2](aclgraph_fia_sink_stage2.md)。

## 3. 最小探针与实际成本

代码为
[输入生成及校验](../../../../benchmarks/aclgraph_varlen_query_trace.py)、
[固定图算子探针](../../../../benchmarks/aclgraph_varlen_query_probe.py)、
[结果审计](../../../../benchmarks/aclgraph_varlen_query_analyze.py)。
没有接完整模型，也没有增加路由器或新的 attention 实现。

同进程四轮交替顺序，每模式每轮使用同一份 8 步输入，总计 128 个端点。
正式结果如下，单位 μs，中位数：

| 模式 | 设备队列跨度 | CPU wall | 用途 |
| --- | ---: | ---: | --- |
| 闭合：上传＋原生规划＋attention＋等待 | 367.08 | 508.30 | 本轮可执行端点 |
| 生产者：上传＋原生规划＋等待 | 140.91 | 271.35 | 预算诊断；没有重新优化共享计划 |
| consumer：上传＋attention＋等待 | 303.06 | 452.19 | 本步原生计划提前准备，仅诊断，不能当作可执行加速方案 |
| 上传及提交诊断 | 86.94 | 232.37 | 含长度 `add_(0)` 以形成可捕获的匹配图，不是纯 DMA 时延 |

队列跨度包含上传、主机提交空隙和图执行，**不是纯 kernel 时间**。
CPU wall 从把已有本步输入复制到 pinned 缓冲开始，到最后等待完成；
合成快照的离线构造、随机 Q/K/V 生成不在计时内。
本轮没有测生产调度器、Q/K/V 生成、KV 写回、模型其它层或整网吞吐。
各模式重复上传、提交且计时边界不同，不能通过相减直接归因规划或初始化成本。

初次诊断将合成页表的 Python 构造也计入，wall 约 2.6 ms；
该数字是夹具生成开销，没有作为生产瓶颈。
正式版把快照当作上游已经提供的输入，只计必要 pinned 准备与上传。
这项口径修正没有改变冻结轨迹、探针步序或算法，不是性能优化结果。

独立 profile 在第 0、7 步各回放 3 次。每次都包含 3 个 H2D、
1 个 metadata 清零核、1 个 AICPU 计划核、1 个融合 attention 核。
每类 kernel 的中位数如下：

| 步骤 | metadata 清零 | AICPU 计划 | 融合 attention | 三份 H2D 总和/次 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.10 | 78.44 | 226.34 | 8.01 |
| 7 | 1.02 | 89.78 | 259.15 | 8.70 |

H2D 总和/次由同一 profile 三次回放的实际 DMA 总时长除以 3，
不是包含主机空隙的上传端点。
`NOTIFY_WAIT` 与图流上的实际工作重叠，是汇合跨度，不能再加到 kernel 总和上。
profiler 给出 schedule 提醒；两份导出均核对到完整的三次预期任务清单，
只用于本轮任务归因，正式端点不启用 profiler。

**没有观察到每层独立执行的 cumsum、cast、repack、重新规划或 host update 任务。**
这只适用于本轮隔离路径和已测输入，不覆盖全部框架或所有算子。
一次共享计划约 78～90 μs 的 profile 预算仍存在，
但它是已经结束的候选，不能乘以 16，也没有因此重新立项。

原始端点和清单见
[probe](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/probe.json)、
[analysis](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/analysis.json)、
[step0 profile](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/profile-step-0.json)、
[step7 profile](../../../../benchmarks/results/multi_graph_structural/varlen-query-v1/profile-step-7.json)。
正式报告中的两份被测脚本和输入文件 hash 已与本地提交内容逐一匹配。

## 4. 找到的具体机制，以及为什么没有实现

### 4.1 尚存的重复步骤

当前 PR 的
[`fia_kernel_nonquant_sink.h:333`](https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_kernel_nonquant_sink.h#L333)
中，TND 只有在无 attention mask 且实际 T 等于物理 T 时才跳过输出初始化。
本轮 causal sparse3 带 mask，因此每次 attention 都进入
[`InitOutputSingleCore:345`](https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_kernel_nonquant_sink.h#L345)，
按物理 T 分工清零输出，并在 383 行执行 `SyncAll()`。
521 行在 AIV 初始化路径调用它。
本轮每份物理输出为 `512 × 12 × 256 × 2 = 3145728` 字节。

这段初始化**在融合 attention 核内部**。
profile 中约 1 μs 的独立清零是 metadata 清零，不能拿它冒充输出初始化成本。

长度解析也仍在 consumer 内发生，但
[`memory_copy.h:442`](https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/memory_copy.h#L442)
直接取相邻累积终点之差，分页 KV 直接取本请求长度，
不需要逐层再做一次全 batch 前缀和。
没有测得一个足够大的独立长度变换任务。

### 4.2 具体但未立项的机制

机制可以明确为：**按最终输出写入者分工，只清理未覆盖区域**。

在本轮受限域中，活跃请求满足 `KV >= Q > 0`，至少有一个可见 KV；
query 紧凑放在 `[0, L)`，其余 `[L, T)` 必须为零。
若能证明每个有效输出元素都被最终 attention 写入者完整覆盖，
则有效域的提前清零是重复写；由已有 AIV 工作单元清理尾部，
可尝试消除初始化阶段针对有效域的写入及相应的全核等待。

必须保留真实 Q/KV、因果 mask、相同 softmax 计算与 FD 归约。
尤其不能将 FD 尚未生成的部分结果视为有效输出，
不能删除保护其它依赖的同步，也不能通过图外提前清零隐藏工作。
空 KV、全 masked 行、LSE、其它布局和更大 query 不在本轮机制证明范围内。
任何实现都仍须满足尾部精确为零、全部有效输出覆盖、无跨 replay 残留。

### 4.3 门槛与决定

冻结轨迹已预先要求：一个**社区仍未消除的重复成本**，
其可归因可消除预算同时达到每次 attention 至少 10 μs、至少占 attention 端点 5%。
以本轮 consumer 队列端点 303.06 μs 为参照，5% 对应约 15.15 μs。
这只是立项阈值，不是观察到的潜在收益。

目前只有融合 attention 总时长，未单独测得初始化和这一处屏障的因果耗时。
3 MiB 写入量不能用峰值带宽直接转换成可消除的关键路径时间；
整个 226～259 μs 核时长也不能当作初始化预算。
因此这条具体机制**没有取得明确预算，不实现**。
不能把“预算未知”写成“肯定不够”，也不能把它写成“有大收益待兑现”。

## 5. 与已有工作的区别

| 工作 | 已经具备 | 本轮的增量与边界 |
| --- | --- | --- |
| 当前 FIA sink PR | 设备长度、元数据生产、稳定缓冲、attention 消费与复用；当前初始化代码已融合在核中 | 对固定图下真实变化的 Q 做受限资格化和成本筛选；未发明设备描述符或 plan/run |
| FlashInfer | 变长 query/paged KV、`qo_indptr`、plan/run 和图模式稳定缓冲已有 API | 仅比较机制，未在 A2 执行 CUDA kernel 或宣称跨硬件性能领先 |
| 本分支 Q4 共享计划原型 | 完整元数据契约、精确规划和负面生产者端点 | 本轮保留社区 FD 计划，检查变长 Q consumer；没有重开 AIV 共享计划候选 |
| 上述按写入者清理机制 | 消除覆盖前冗余写是一般算子优化思想，不能直接声称首创 | 本轮只定位到具体源码和约束；没有实现、净收益或足够预算，不作为已完成的毕设创新点 |

FlashInfer 的当前公开 API 见
[变长 attention、plan/run 与图缓冲约束](https://docs.flashinfer.ai/api/attention.html)，
检索日期为 2026-10-20，页面版本为 0.7.2。
不将其接口与 Ascend 的具体流水、屏障或零尾部契约混为一谈。

## 6. 正确性、保存范围与最终状态

8 个变长输入均检查全部输出有限值、物理尾部精确零、输入描述符和页表不变、
捕获输入/输出地址固定；每步抽取首、中、末三个请求做独立 FP32 causal reference。
计时后再次检查最后一步；query guard 与 poisoned page0 保持不变。
这是抽样数值参考，不是所有有效输出的逐项 oracle。

原 Q4 的 128 步长度和请求顺序全部回放，物理图固定 512，
Q4 padding 请求保留 Q4/KV0 语义；每步检查有限值、零尾部、固定地址，
首末步增加 FP32 抽样参考。
Q/K/V 是本轮驻留随机输入，未复现旧实验的全部随机张量、16 层消费或模型执行。
因此只称**原轨迹的算子级回归**。

本地 8 项 CPU 契约与证据检查通过，限定 Ruff 检查通过。
使用 `pytest --noconftest` 隔离纯 CPU 检查：本机没有 vLLM 包，
仓库上层 conftest 会导入设备插件，不能据此宣称完整 UT 通过。
按仓库要求尝试 `bash format.sh ci`，在本机 WSL 提醒后被工作树
`format.sh` 的 CRLF 解析错误阻断，退出码为 2；没有宣称完整 CI 通过。

本轮没有得到端到端收益，没有实现新算子，也没有证明新的独有机制贡献。
交付的是被冻结输入、源代码位置、实际路径、计时、社区区别和明确的未立项原因。
当前变长 query attention 选题筛选至此结束；不接模型、不增加路由器、
不继续枚举 kernel 参数或重新包装已结束的共享计划方案。
