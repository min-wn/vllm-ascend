# Q1/Q4 共用 compact 状态：完整 GDN 层有界验证

日期：2026-10-22。保留全部历史数值、质量及性能失败；本轮不重新评价模型质量。

## 结论与去留

**正确性检查通过，完整层性能准入失败。结束本轮完整层扩展，不推进离线生成或 runner。**

两个独立性能进程均退化。主指标合并中位数：强制 Q4 ready **41.037763 ms**，Q1/Q4 ready **47.093439 ms**；完整16步耗时增加 **6.055675 ms（14.7563%）**。冻结噪声门槛为 0.279721%，`stable=false`。

此前 recurrence 对称 out/out 的约2.29%收益继续有效，但不能外推为完整 GDN 层收益；已有 Q4 ready 调度成果也不被本轮覆盖。没有修改冻结内核、精度、状态 ABI、核资源、默认容量或质量门槛。

## 范围、来源与共同条件

- Ascend 910B3 卡0/1，TP2；Qwen3.5-27B-original 的 decoder layer 4。torch 2.10.0+cpu、torch_npu 2.10.0.post4；vLLM 0.30.0+empty / vLLM-Ascend 0.30.0rc1+multigraph.732514948。
- BF16 层输入及 Q/K/V/beta/输出，FP32 gate、compact base/log/products；HF32关闭，FP32 matmul highest。内部格式保持已安装默认值，两侧相同。
- 原 manifest 的54～69步、56 owners、每步224行共同 dense容量。10个Q1-only步（56有效行），6个mixed步（42 Q1＋14 Q4，即98有效行）。主轨迹接受全1且无暂停；接受1～4、重排与暂停恢复由额外5步短链检查。
- 没有找回原 ledger step54 的 hidden/residual、conv、SSM。原资料只有token、调度和状态hash。因此本轮明确是**新构造的 teacher-forced 层级对照**，不是原生成轨迹精确重放，也不是自回归生成。
- 使用已有真实文本入口，同文16-token官方前缀得到匹配的目标层conv/SSM初态；每步从owner已接受位置取接下来四token，通过embedding和官方layers0～3全前缀重算取得变化的hidden/residual。两侧共用有效输入，目标层各自独立连续推进compact状态。
- 输入文件每rank189,602,819 bytes，远端保存在 `/workspace/gdn-compact-q14-layer-v1/artifacts/inputs.rank0.pt` 和 `inputs.rank1.pt`，保存hash并在两性能进程启动时核验。主轨迹每rank的16份hidden及16份residual各有16个不同hash，没有重复一个recurrence site代替逐步输入。

## 最小完整层适配与 conv 语义

共同dense层计算保持224行：输入norm、QKV/BA/Z投影、gated norm、输出投影、TP规约、残差及MLP均复用原层实现。强制Q4执行每请求四行；Q1/Q4仅在投影之后按实际宽度打包core输入，core输出还原到相同dense quartet位置。没有添加Q1有效filler计算。

conv显式沿用官方spec varlen调用，保留当前Q1/Q4边界和上一轮接受长度。纯Q1也不切到丢失该selector的普通conv分支。官方生产GDN使用 `cache_indices[:,0]`；接受长度a在该槽内部选择 `conv[a-1:a+2]` 三项历史。Q1只发布前三项有效历史，Q4可以保留其他候选；因此跨模式不比较无效conv尾部。

这属于benchmark适配，未改变官方conv或recurrence内核。没有据此断言官方原生runner存在错误；本轮使用的是独立compact状态契约。

## 设备正确性证据

两rank的 `correctness4` 均通过，各93条记录：每模式两组有限filler的5步graph链、一组5步eager链，以及原16步连续graph链，共3种模式（强制Q4、Q1/Q4 ready、Q1/Q4 delayed）。

- 短链包含Q4→接受1→纯Q1→Q4，也包含接受4后纯Q1；Q1步前接受长度为 `[4,3,2,1,4,3,2,1]`。另覆盖mixed、重排、owner2/5暂停与恢复。
- 将无效filler分别设为0.125和−0.375，合法层输出、接受态和逻辑conv历史逐位一致；随后Q4续算也一致。
- 强制Q4与Q1/Q4比较有效hidden/residual、K/delta/gamma日志前缀、接受态、逻辑conv窗口及终末物化；发布记录按旧Q4布尔valid与新Q1/Q4宽度valid各自契约检查。
- delayed/ready比较全部六份状态张量及终末，eager/graph短链比较输出和状态；暂停/null原始状态不变；每条链结束后logs/valid清零。
- 输入、源码、三个冻结扩展及依赖共享库SHA记录保留。性能入口要求本rank正确性成功、输入与源码/二进制身份一致，外部启动器同时核对installed依赖没有变化。

审计中间张量只在不计时轮下载到CPU比较，未逐份落盘；正确性结论依赖已核查SHA的失败即抛出断言路径及保存的输入。跨算法数值和任务质量失败仍然有效，本轮不恢复其资格。

## 正式完整成本表

先冻结判据，再计时：两个独立TP2性能进程，顺序互换；每配置2条16步链预热、8条正式链。共64条rank原始样本，按相同process/config/sample选择完整生命周期较大rank的**整条记录**，得到32条TP2主样本。没有把不同rank各分项最大值相加，没有删样本或另选结果。

每条样本包含一次状态导入、连续16步执行及终末物化。下表为所选样本的**均值**（ms），分项可相加；独立分项中位数不作为相加成本。

| 配置 | 一次导入 | 执行链 | 终末物化 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 强制 Q4 ready | 0.372300 | 38.757604 | 1.930000 | 41.059904 |
| Q1/Q4 ready | 0.370269 | 44.788098 | 1.883574 | 47.041941 |

正式执行包含CPU元数据/索引构造、必要上传、入口hidden/residual/position复制、Q1/Q4投影后打包/结果还原、完整decoder层、状态commit及最终完成等待。强制Q4不上传没有使用的pack/inverse/mask三个描述符；两侧使用共同host契约构造入口记录，Q1/Q4额外适配成本留在计时内。没有再加入上一轮五份recurrence输入的fixture gather重建。

一次导入在预分配资源上执行同文prefix conv→四物理槽展开、SSM复制、日志及staging清零。静态分配和权重加载不计入导入。终末按最后真实接受长度commit SSM、清日志，并提取每owner逻辑conv历史和等待；没有上传按channel重复扩张的巨大索引，仅上传56×3个offset并使用view广播。

以下为主判据的完整生命周期中位数（ms）；改善为负表示退化。

| 独立进程及顺序 | 强制Q4 | Q1/Q4 | 绝对节省 | 改善 |
| --- | ---: | ---: | ---: | ---: |
| 0（forced_q4→q14_ready） | 41.084194 | 47.047553 | -5.963359 | -14.5150% |
| 1（q14_ready→forced_q4） | 41.026773 | 47.108949 | -6.082176 | -14.8249% |

两配置跨进程中位数相对跨度分别为 0.139861%、0.130413%；样本相对MAD分别为 0.106780%、0.094186%。判据要求两进程均正，且合并收益超过两倍最大跨度/MAD；本次明确失败。

模型加载日志约10秒/rank、25.1767GB权重/rank，均在计时外。性能图构建（含单步eager预热）分别记录在两rank报告的 `capture_ms`，不混入请求链。公共真实输入生成、prefill、静态分配、图构建和正式链预热均在计时外。输入在接入调试时已保存，本次资格/性能复用；原生成区间数字未保存在成功报告中，`input_generation_ms=null`，不能当成0成本。

## Q1-only / mixed 分段与成本来源

下表来自每性能进程另行运行的一小条事件诊断链，先选同一步完整端点较大rank，再按段求和，最后平均两个进程。它不参与正式主指标，不能与无插桩wall样本混算。

| 分段 | 配置 | 准备端点 ms | 图内层执行端点 ms | 分段总端点 ms |
| --- | --- | ---: | ---: | ---: |
| Q1-only，10步 | 强制Q4 | 8.748760 | 16.825610 | 25.574370 |
| Q1-only，10步 | Q1/Q4 | 12.090260 | 17.456270 | 29.546530 |
| mixed，6步 | 强制Q4 | 5.075510 | 10.152220 | 15.227730 |
| mixed，6步 | Q1/Q4 | 6.988940 | 10.680050 | 17.668990 |

在这条诊断中，16步准备增量为 **5.254930 ms**，图内层执行增量为 **1.158490 ms**；两部分均退化。准备区间包括CPU构造、提交、H2D及设备等待主机的空隙，不能解释成纯H2D内核耗时。上述三列存在包含关系，不将total再与prepare/layer相加。

源码能确定的额外工作是：Q1/Q4每步多上传pack/inverse/packed-active三个描述符（合计3,808 bytes/step，16步48次额外小上传），图中另有三份投影后输入的index_select/mask、固定输出清零及逆映射。其余dense计算形状相同。没有profiling把这些单项分别隔离，不能把全部准备差额归给copy，也不能从图内增量直接推断某个Cube/AIV算子瓶颈。

证据支持的判断是：**当前完整层适配没有保留净收益；准备及图执行端点均增加，与新增适配抵消潜在节省一致，不能仅用共同dense计算稀释解释。具体单项占比尚未隔离确定。** 即使假设消除诊断中的全部准备增量，图内端点仍未显示收益；这只是同一诊断的预算观察，不是已实现的可部署优化结果。

小上传合并是一个可检查的工程方向，但本轮没有证据证明单靠它能恢复完整层净收益。没有进一步已验证的可消除主要成本，故按范围结束扩展，不追加参数搜索、新布局、内核候选、完整模型或runner实验。

## 接入调试、验证与证据位置

前四次接入尝试的失败日志全部保留：manifest envelope读取、NPU配置属性读取、custom-op关键字签名、旧Q4 valid语义误判。它们是验证入口/检查器问题，不改写为模型精度失败；最终正确性使用相同冻结内核通过。正式性能只运行上述两个独立进程，没有根据结果重测。

本机定向CPU检查32项通过，Ruff检查及格式检查通过。`bash format.sh ci` 已执行，因仓库既有CRLF在第22行报语法错误而失败；未修改公共脚本。

仓库证据目录：`benchmarks/results/multi_graph_structural/gdn-compact-q14-layer-v1/`。含四份rank性能报告、正确性报告、完整原始样本与事件端点、analysis.json、源码/依赖/下载SHA、全部接入失败及运行日志。原始输入和每次source归档保留于远端 `/workspace/gdn-compact-q14-layer-v1/`，未将约380MB输入张量加入Git。

复算入口：`python benchmarks/analyze_gdn_q14_layer.py --input-dir benchmarks/results/multi_graph_structural/gdn-compact-q14-layer-v1 --output <新的JSON路径>`。

实现位于 `multi_graph_gdn_q14_layer_inputs.py`（真实输入）、`multi_graph_gdn_q14_layer_core.py`（有限适配）、`multi_graph_gdn_q14_layer_probe.py`（审计/完整端点）、`multi_graph_gdn_q14_layer_protocol.py`（冻结契约/门槛）。原Q4和Q1/Q4内核、原fixture out探针及旧结果均未修改。

`source-sha256.json` 是首次部署清单，保留其原始内容；最终通过版本以 `correctness4`、`final-source-sha256.json` 的 `source_sha256` / `fingerprint` 为准。历史source归档与失败日志均未覆盖。
