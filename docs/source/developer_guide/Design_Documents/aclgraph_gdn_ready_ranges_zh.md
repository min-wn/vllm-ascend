# GDN 请求组 ready 流水：真实激活、动态负载与精度归因

日期：2026-10-20。主实现为 compact FP32 ready；ReplaySSM FP32 delayed/ready 为机制对照。

## 1. 当前结论

**真实激活驱动的单层动态范围验证通过。中、高负载下，ready 的同算法收益在两个独立进程的六轮测量中均为正；低负载收益不稳定。**

这里的收益指一条 40 步连续轨迹从状态导入到最后接受态物化的闭合墙钟成本降低。
不是服务吞吐、文本生成质量或多图切分独有收益。

| 负载 | 每步有效请求数 | compact ready 相对 compact delayed | ReplaySSM ready 相对 ReplaySSM delayed | 六轮为正次数，compact / ReplaySSM |
| --- | ---: | ---: | ---: | ---: |
| 低 | 22～26 | 平均 0.43% | 平均 0.68% | 5/6、4/6 |
| 中 | 54～60 | 平均 3.12% | 平均 2.82% | 6/6、6/6 |
| 高 | 94～100 | 平均 5.30% | 平均 4.73% | 6/6、6/6 |

百分比是六个配对轮次的百分比均值。各轮先对每种模式取两个 TP rank 的最大闭合耗时，再计算改善率。
六轮来自两个进程，每进程三轮；不能把相邻 step 当作独立重复试验。

**这足以继续验证整网调度贡献，不足以宣称全负载稳定加速。**
本轮没有增加路由器，也没有用负载选择器隐藏低负载退化。

## 2. 方案用简单语言解释

GDN 会保留每个请求的递推状态。投机验证一次处理四个 query，之后才知道其中接受几个。
已有紧凑状态算法用一个大状态加小日志表示这四个候选，避免反复保存四份大矩阵。

本方案研究的是这套算法的执行顺序：Cube 生成一组请求需要的小矩阵产品后，
让配对 AIV 立即消费这一组，而其他请求组仍在生产。
delayed 控制使用相同算法、相同分组、相同资源，但等待其分配的产品全部完成后才消费。
这样 delayed/ready 的差异可以归因到组间就绪流水，而不是精度、状态表示或输入变化。

compact 两个控制还共享相同的输出投影期间延迟提交及最终 join。
因此它们之间的收益不能再次归因到“增加了延迟提交”。
ReplaySSM 则保留自己的环形日志、动态游标和前置周期 flush，使用同一组 delayed/ready 时序对照。

## 3. 与已有工作的边界

社区依据沿用已归档源码及固定提交，而不是把本机移植版当作社区最佳实现。

| 机制 | 已有工作覆盖 | 本方案的研究范围 |
| --- | --- | --- |
| checkpoint 加小日志、动态接受前缀、日志修正投影 | RecoverSSM、ReplaySSM、TreeWY 等已覆盖，不能当作新算法。 | 复用其思想，作为执行流水的工作负载。 |
| 环形日志、设备游标和周期 flush | [ReplaySSM PR #49887](https://github.com/vllm-project/vllm/pull/49887)，固定提交 `5b96cdbceefcdec6d60d588420ee127d76bd8f97`。 | 保留输出依赖，在 A2 上建立同算法时序消融。 |
| compact 输出投影期间提交 | 既有 compact 实现已具备。 | 两个 compact 控制均开启，不能重复计作 ready 创新。 |
| A2 请求组 ready 消费 | 所审固定社区源码没有这里的 Cube/AIV 配对发布协议。 | 本轮实际比较的独立差异；不推断所有社区工作都没有类似流水。 |
| 请求图切分或 Padding | 两者都可以使用同一 ready 算子。 | 这是公开可复用的执行机制；不要求多图独占，也不宣称切分必要性已成立。 |

固定社区核为 `vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py`：
日志布局从第 98 行起，flush 输出依赖在第 180～280 行，游标推进从第 360 行起。
原始文件和 SHA 见
[社区源码清单](../../../../benchmarks/results/multi_graph_structural/gdn-review-v1/community-source-manifest.json)。
更完整的前置依赖分析见 [ReplaySSM A2 对照](aclgraph_replayssm_a2_review_zh.md)。

本机路径仍是 `vllm_ascend/ops/gdn.py` 的真实 conv、门控、norm、输出投影、TP 和 MLP，
只替换 speculative recurrence；没有另造简化层来替代真实单层。
被测本机文件和扩展的 SHA 见本轮 `environment.json`，原字节脚本见 `measured-sources.tar.gz`。

## 4. 预先冻结的输入

协议 SHA256：`e952a78fd7d8d977de9c6ff67d17cb020719fb54a05d926b64be0b1bdb50a697`。
原始协议见
[frozen-protocol.json](../../../../benchmarks/results/multi_graph_structural/gdn-ready-range-v1/frozen-protocol.json)。

| 项目 | 冻结设置 |
| --- | --- |
| 环境 | A2 / 910B3，Qwen3.5-27B，TP2，BF16 激活，FP32 状态与日志，HF32 关闭。 |
| 测量对象 | 第 4 层 GDN，层编号从 0 开始。 |
| 固定图 | 400 token、100 owner、Q4；低中高档均使用这一张容量。 |
| 轨迹 | 每档 40 步；低/中/高 owner 池分别为 32/64/100。 |
| 请求次序 | 按 `(step*7+offset)%pool` 取 owner，奇数步反转。请求身份与批内行号分离。 |
| 接受长度 | `1+(owner+该 owner 已生成窗口数)%4`，首次没有 pending 不提交。 |
| 动态行为 | 三档均有重排、暂停与恢复；暂停请求保留未接受完的候选及状态。 |
| 初始前缀 | 每个 owner 的同源真实文本前 16 token，官方 prefill 初始化目标层 conv/SSM。 |
| 终末接受 | 按冻结的最终 owner generation 生成；所有 owner 收尾，不能只清末步 live 请求。 |
| 比较模式 | official、compact delayed、compact ready、ring delayed、ring ready。 |

**固定 400 token 图是本轮动态有效负载范围实验的边界。**
它没有比较默认 dispatcher 在低负载选择更小 bucket 的性能，也不能用低负载相对官方的差距推断默认小图收益。

### 4.1 真实激活从哪里来

为每个 owner 构造固定英文自然文本，内容分为递推网络说明、研究报告、Python 移动平均和异步生产消费四类。
使用本地模型 tokenizer，加 owner 前缀，保留 256 token。

每个 step 根据已接受的文本长度，运行真实 embedding 和官方第 0～3 层，
取末尾四个 token 的 hidden/residual，作为第 4 层的共同输入。
第 3 层 FIA 使用真实位置和长度，目标层先用同一文本前缀获得相干的初始状态。
目标层在连续轨迹内推进；重算上游层时不改它的状态。

这不是随机 hidden。五个控制消费完全相同的预先采集激活，避免把上游数值差异混入目标层消融。
两个进程的 token SHA 和每步 hidden/residual SHA 全部一致。
激活采集和共同 prefill 不计入目标层时延。

它也不是完整生成实验：接受长度是固定调度输入，未执行第 5 层以后的模型，未测接受率或文本质量。

## 5. 正确性结果

逐步对照官方层，最坏请求的 hidden/residual NRMSE 门槛保持 0.005，没有放宽。
同算法 delayed/ready 的输出、dense/compact/ring 状态和提交暂存逐值相等，使用 `rtol=atol=0`。
暂停/null owner 及其候选卷积槽保持不变；所有观测张量有限。

| 负载 | 全部控制最坏请求输出 NRMSE | 最终接受态最大 NRMSE |
| --- | ---: | ---: |
| 低 | 0.00284950 | 1.93913e-7 |
| 中 | 0.00334544 | 1.91406e-7 |
| 高 | 0.00295662 | 1.98727e-7 |

两进程共 2400 条 rank/step/mode 资格记录通过；180 个计时样本在终末 drain 后与资格参考 exact。
每轮内部不重置；每个计时样本末尾实际支付最后接受态物化并清除有效 pending/cursor。
这是一份冻结轨迹下的精度证据，不能外推为所有输入的精度保证。

## 6. 完整生命周期性能

计时包含：CPU 描述符暂存拷贝及 H2D、输入 D2D、真实整层 replay、
全部 cursor/flush/commit、逐步同步、末尾接受态物化、一次状态导入。
实验轮间恢复初态、模型加载、共同 prefill、激活采集、捕获和诊断不计时。
设备事件覆盖上传，因此其跨度也不能称作纯 kernel 时间。

| 路径 | 低负载 40 步闭合墙钟（ms） | 中负载（ms） | 高负载（ms） |
| --- | ---: | ---: | ---: |
| 官方逐候选状态 | 95.472 | 110.740 | 127.216 |
| compact delayed | 92.188 | 101.742 | 112.080 |
| **compact ready** | **91.794** | **98.567** | **106.141** |
| ReplaySSM FP32 delayed | 97.348 | 111.725 | 127.548 |
| ReplaySSM FP32 ready | 96.685 | 108.575 | 121.513 |

compact ready 相对官方的逐轮改善率均值为 3.84%、10.99%、16.56%。
这些同时包含状态算法与执行方式变化；**能够独立归因到 ready 的仍是第一节同算法消融的 0.43%、3.12%、5.30%。**

单次导入观测：compact 为 3.257～3.312 ms，ring 为 2.815～3.049 ms。
每进程/rank 只采一次，不能声称导入分布已经稳定。
这里的完整生命周期限定为固定 owner 从相干初态导入，到最后接受态物化；未包含 owner 回收、新请求复用或服务排队。

低负载有退化轮次，现有数据没有证明稳定净收益。
中、高档各六轮为正，支持继续整网验证，但局部收益不能按 GDN 层数直接相乘。

## 7. FP16 首个失败样本：一次分阶段归因并收口

只复现上一轮第 1 步、owner 34（均从 0 编号），token 切片 `[136:140]`。
沿用原 FP16 日志内核和两步固定种子输入，没有新增生产精度方案，也没有该配置的性能测量。
这个诊断输入是上一轮首失败样本，不与本轮真实文本性能数据混用。

Q/K/V、beta/g 在该 owner 上与官方 exact。以下按两 rank 中较大值列出 NRMSE。

| 阶段或替换实验 | 误差 | 解释 |
| --- | ---: | --- |
| 已接受状态重建对官方状态 | 9.33e-5 | FP16 日志所代表的状态已出现近似。 |
| 原生 recurrence 对“同 FP16 日志逐 token 物化参照” | 8.31e-6 | 独立物化基本复现原生核心。 |
| FP16 日志物化参照对官方 recurrence | 4.09e-4 | 状态表示差异进入当前输出。 |
| gated norm | 9.07e-4 | 原生路径开始放大差异。 |
| 输出投影 | 0.00205603 | 包括真实 TP 后的结果。 |
| post-attention norm hidden | 0.00247858 | 继续传播。 |
| MLP gate/up | 0.00290574 | 分 rank 观测。 |
| MLP activation | 0.00461311 | 非线性继续放大。 |
| MLP down / 最终 hidden | **0.00520601** | 超过 0.005 门槛，失败复现。 |
| 用同 FP16 日志物化参照替换 recurrence 后的最终 hidden | **0.00519926** | 仍失败，主要误差不依赖原生环求解的具体实现。 |
| 从官方 dense 状态起算的独立逐 token参照，最终 hidden | 0.00113969 | 低于门槛；微小舍入仍可能经 BF16/TP/MLP 放大。 |
| 直接注入官方 recurrence 输出 | **0，所有后续阶段 exact** | 将误差来源定位在 recurrence 输出之前。 |

归因结论：**对此固定样本，主要误差来自 FP16 日志表示的状态近似，再被后续层放大；没有发现应修复的索引、游标或调度实现错误。**
不能由一个样本证明所有误差只来自量化，也不能据此声称社区 GPU 算法存在 bug。
按照约定，保留原配置失败并结束 FP16 方向，不再枚举精度补丁。

诊断原始结果见
[attribution.json](../../../../benchmarks/results/multi_graph_structural/gdn-ready-range-v1/fp16-first-failure/attribution.json)。

## 8. 本轮之后的整网验证

单层精度通过、中高负载收益可复现，允许继续已有整网回放，不新增模型。
整网沿用 Qwen3.5-27B TP2、64 层、48 GDN 和 16 FIA，以及原 Q4 128 步轨迹和默认图目录。
它仍是固定输入轨迹的模型 forward，不是自回归服务。

新增的 `--ready-lifecycle` 路径只比较 compact delayed/ready，采用同一个 delayed eager 连续轨迹作为参考：

1. 每一步检查输出、全部 compact 状态/暂存、conv、FIA KV、host generation、暂停/null 保护，并重建全部逻辑候选检查有限性。
2. 两种图模式都与该参考逐步 exact；最后 pending 状态及终末接受态再跨模式 exact。
3. 正式计时计入真实 FIA 更新、所有提交、最后接受态 drain 和一次直接 compact 转换。
4. 每轮计时后再比较全部闭合状态；CPU 审计、轮间恢复、加载及捕获在计时外。

整网结果尚待本轮正在运行的审计完成后写入。
同算法 replay 正确性与 compact 对独立官方 dense 整网轨迹的数值一致性是两个问题；本次补的是前者。
单层相对官方的误差门槛，不能代替整网数值质量评估。

## 9. 代码与复核入口

| 文件 | 职责 |
| --- | --- |
| `benchmarks/multi_graph_gdn_ready_range_protocol.py` | 冻结三档输入、接受语义与协议 SHA。 |
| `benchmarks/multi_graph_gdn_ready_activations.py` | 同源真实激活、相干前缀初态。 |
| `benchmarks/multi_graph_gdn_ready_range_probe.py` | 五控制的单层连续资格与闭合计时。 |
| `benchmarks/multi_graph_gdn_ready_range_analyze.py` | 原始证据、源码、精度、配对成本审计。 |
| `benchmarks/multi_graph_replayssm_fp16_attribution.py` | 仅首个 FP16 失败样本的分阶段替换归因。 |
| `benchmarks/multi_graph_gdn_q4_full_probe.py` | 既有整网回放及新增完整 ready 消融入口。 |
| `benchmarks/multi_graph_gdn_ready_full_lifecycle.py` | 全 owner 终末接受态和两 rank 闭合成本配对。 |

本轮原始证据位于 `benchmarks/results/multi_graph_structural/gdn-ready-range-v1/`。
可在仓库根运行：

```bash
python benchmarks/multi_graph_gdn_ready_range_analyze.py \
  benchmarks/results/multi_graph_structural/gdn-ready-range-v1
```

该分析器重新校验原始文件 SHA、被测源码归档、冻结协议、两进程激活一致性、全部精度记录和收尾成本。
共享规划及变长 query attention 筛选继续保持结束；本报告不重新开启这些候选。
