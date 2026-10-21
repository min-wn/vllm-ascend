# Q1/Q4 原型准备区间退化的一次有界归因

## 结论与边界

**主要定位到输入 fixture 的小尺寸 `copy_` 主机准备/提交路径，未定位到共用 compact 状态必需的新增工作。**
新 Q/K 及 Q1-only 的 V copy 改走 SDMA 后，设备搬运更快，但主机 `aten::copy_` 调用更长。
在唯一一次诊断采集中，每条 16 步链的五份输入 copy 主机调用净增加 1.734515 ms，
其中 enqueue 前增加 1.528890 ms，约占这项增量的 88.1%。
不能将这些 profile 数字当作原正式准备增量 2.324233 ms 的精确组成或已实现可追回收益。

本轮不改内核、状态算法、精度、布局、图或冻结原型，不运行整网，也不实施工程修复。
此前性能准入失败与所有质量失败记录保持有效，不推进完整层或 runner。

## 先核对的正式证据

冻结 `multi_graph_gdn_q14_probe.py` 的规范化 SHA256 为
`bde7a1c1e4d9e401f9ce37528e2442f018cd9c067ecec3877494c557d7872b5a`。
原 `measure()` 每步按 `begin.record → prepare → compute_begin.record → graph.replay → end.record` 提交，
只在链尾同步。因此 prepare event 区间包含设备没有后继工作时等主机提交的空隙，并非纯 CPU 或 DMA 时间。

下面仍是原三轮、每轮八样本的未 profile 均值，单位 ms；本轮没有重新测正式性能。

| 原正式指标 | 强制 Q4 ready | Q1/Q4 ready | 新减旧 |
| --- | ---: | ---: | ---: |
| 准备 event 区间 | 11.537582 | 13.861816 | +2.324233 |
| 图 event 区间 | 5.085942 | 3.936800 | -1.149142 |
| 执行 wall | 16.765155 | 17.943019 | +1.177863 |
| 初始化＋执行＋终末物化 | 17.429836 | 18.591420 | +1.161583 |

原稳定收益判据仍失败，三轮生命周期中位数合并退化 6.7574%。
event 两段之和与执行 wall 的差分别约 0.141631 / 0.144403 ms，为链边界及步间未覆盖成本，未塞入 H2D 或 gather。

## 源码差异与工作分类

原探针 `Trial.prepare()` 位于 `benchmarks/multi_graph_gdn_q14_probe.py:180`；
`metadata()` 位于 `benchmarks/multi_graph_gdn_q14_protocol.py:90`。
两侧每步均一次 metadata 构造、一次 CPU tensor＋H2D、一次 `.long()`、五次 `index_select`、五次 `copy_`，并更新相同56个 owner 的记录。
只有 lengths、有效行索引及相关 tensor/view shape 随模式变化，没有 Q14 专属的额外准备调用。

| 工作 | 分类 | 依据 |
| --- | --- | --- |
| 当前 query 长度、owner、上一接受前缀独立表达，检查上一发布宽度 | 共用状态所需语义 | `sequence / ids / accepted`；当前 Q1 不限制上一接受 1～4 |
| 固定数组、packed owner 重复、padding、每步 CPU 字典/列表与诊断字段 | 当前适配形式 | 语义不强制由 CPU 重建；两侧当前都执行，不能只从新侧删除 |
| 当前 ABI 的 sequence/ids/accepted 上传 | 当前实现需要 | 57＋224＋56 个 int32，1,348 B/step，两侧相同 |
| source indices 构造、padding 及混入上传 | fixture 重建 | 224 个 int32，896 B/step；不是 compact 状态信息 |
| source indices 的设备 `.long()` | fixture 重建 | 旧224个，新56/98个，不是 recurrence 的状态转换 |
| 五份已在设备上的输入 gather、临时结果及固定 buffer copy | fixture 重建/适配 | 从同一保存 site 按 owner 重建本步输入；并非五份激活 H2D |

两侧共同 H2D 为 2,244 B/step、35,904 B/16步，上传次数均16。
五份输入单行 payload 为10,384 B（Q/K/V/beta BF16，gate FP32）。
旧每步224行；新Q1-only为56行、mixed为98行。16步 payload 从37,216,256 B降至11,920,832 B。
这些是源码 payload，不是实测 HBM 流量；较少字节不代表较少主机调度成本。

## 一次诊断如何采集

只用原54～69步、56 owners、224行容量、同一真实 recurrence site 和共同合法初态。
一个进程、一个 CPU＋NPU Level0 profiler 会话，固定 ABBA：旧→新→新→旧，每条仍是同一16步连续链。
capture/预热位于 profiler 外；链内保留原三 event/step，无逐步 synchronize；仅 restore、链尾、terminal 同步。
四条链 terminal 逐位一致。这里不是新的稳定性能判据。

诊断脚本仅给原 `prepare()` 插入 host trace ranges；剥离 ranges 后，AST 与原19条语句严格相同。
tensor＋copy 和五组 gather＋copy 表达式未拆开。阶段划分依赖 profiler 内部 aten 调用及关联任务。
`record_function`、标签和列表记录都增加主机成本，故保留 marker/residual，不把它们冒充真实算法成本。

原始 trace 的192个 event record全部对应原64步×3；时间戳用 Decimal 减共同原点后分析，
与保存的 NPU event 时长最大差小于0.075微秒。704个准备设备任务均关联，
包括64次cast、320次gather、320次copy；未把 HostToDevice 关联箭头当成传输方向。
profiler 日志含默认 RECORD 停止可能不完整的提示；本次所需的四条链、事件和上述任务计数均通过核对，未因此重采。

## 诊断成本：CPU 与设备分别报告

以下均为各模式两条诊断链均值，单位 ms/16步，不能代替原未 profile 数字。
CPU 表按互不相交的范围拆分，列间可闭合为 host prepare；阶段包含标记开销。
H2D 行是主机 copy 调用 wall；本次 Level0 没有独立可关联的该同步上传 DMA 任务，不能称为纯总线时长。

| 诊断 host prepare 排他分段 | 旧 | 新 | 新减旧 |
| --- | ---: | ---: | ---: |
| CPU 元数据构造范围 | 2.557480 | 2.222665 | -0.334815 |
| CPU tensor 构造前缀，含标记开销 | 2.602830 | 2.557325 | -0.045505 |
| metadata H2D 的主机 copy 调用 | 1.807960 | 1.709115 | -0.098845 |
| H2D 范围尾部/标记 | 0.822290 | 0.822030 | -0.000260 |
| 索引转换范围，含slice/cast/分配提交 | 2.803265 | 2.772765 | -0.030500 |
| 五份输入 gather 的主机调用 | 3.777505 | 3.846205 | +0.068700 |
| 五份输入 copy 的主机调用 | 3.339130 | 5.073645 | +1.734515 |
| input 范围其他工作，含slice/分配/标记 | 8.560985 | 9.028505 | +0.467520 |
| CPU 宽度与接受记录更新范围 | 1.356105 | 1.263525 | -0.092580 |
| prepare 内范围间空隙/诊断开销 | 4.909905 | 4.959125 | +0.049220 |
| host prepare 合计 | 32.537455 | 34.254905 | +1.717450 |

下面使用设备 event 窗口；busy 是可见 KERNEL/SDMA 工作区间的并集，idle 是窗口补集，二者恰好闭合。
不把 CPU 表、DMA、kernel 和 idle 再相加。

| 诊断准备窗口 | 旧 | 新 | 新减旧 |
| --- | ---: | ---: | ---: |
| 准备 event 窗口 | 35.224854 | 36.965158 | +1.740304 |
| 可见设备工作并集 | 1.253524 | 1.097543 | -0.155981 |
| 无可见设备工作的窗口补集 | 33.971330 | 35.867615 | +1.896285 |

此处 idle 不等于设备故障或全部属于主机。继续按后继任务的CPU调用起点、runtime launch关联起点分割，得到互斥分段：

| idle 的另一种排他视角 | 旧 | 新 | 新减旧 |
| --- | ---: | ---: | ---: |
| 尚未进入后继 host operation | 28.465260 | 29.962906 | +1.497646 |
| host operation 至 runtime launch 起点 | 3.185241 | 3.431503 | +0.246262 |
| runtime launch 起点至设备任务开始 | 2.320828 | 2.473206 | +0.152378 |

第一行表示设备暂无工作而主机尚未进入后继调用，可能覆盖当前/前一阶段的处理与诊断开销。
第二、三行不能进一步冒充纯 Python、纯runtime队列或纯设备依赖：本次只有launch关联起点，没有底层API完成边界。
同一 idle 与各 host stage 的交集单独保存在 `window-analysis.json`，是另一种切法，不能和本表重复相加。

## 主要来源与排除的解释

| 五份输入的诊断指标 | 旧 | 新 | 新减旧 |
| --- | ---: | ---: | ---: |
| copy 主机调用 | 3.339130 | 5.073645 | +1.734515 |
| 其中 enqueue 前主机范围 | 2.324880 | 3.853770 | +1.528890 |
| copy 设备任务 | 0.274285 | 0.107213 | -0.167073 |
| gather 设备任务 | 0.961519 | 0.971920 | +0.010401 |

copy enqueue 前是主机调用的子区间，不和整段再相加。输入copy主机增量以Q/K为主，分别增加0.693400和0.646665 ms，V增加0.452120 ms；beta减少0.057810 ms、gate近乎不变。

设备路径发生明确变化：旧 Q/K/V 均为 AIV TensorMove；新 Q/K均为SDMA，V在Q1-only走SDMA、mixed仍为AIV；beta/gate两侧均为SDMA。
新 mixed V 同样是部分storage视图却仍走AIV，故不能将路径变化直接归因于“部分view”。
数据更符合小尺寸copy触发不同后端，但没有核实具体阈值或CANN内部函数，止步于已观测到的提交前路径差异。

新SDMA的148/148次、旧SDMA的64/64次，主机copy均在本次设备copy开始前返回。
所以“新copy在同步等待本次搬运完成”不成立。与前序gather存在重叠，但仅此不能证明或排除内部对前序工作的等待。
copy内可见嵌套CPU op只有 `aclnnInplaceCopy`，没有额外 `empty/reshape/contiguous` 证据。
没有以 `non_blocking=True` 作为已成立的修复。

CPU metadata与H2D调用在这次诊断中没有增加；五份gather设备合计只增加0.010401 ms，copy设备合计反而减少0.167073 ms。
这否定“上传更多字节”“多一套共用状态准备”“copy设备带宽变慢”作为主要解释。
观察到的准备退化主要落在 fixture 小copy的主机路径，而不是recurrence本体或必需状态语义。
不过 profile 将原约11.5/13.9 ms准备区间放大到约35.2/37.0 ms，不能按比例把诊断分量移植回正式2.324233 ms。

## 小工程修复与最大可挽回预算

明确且范围小的候选只在 fixture 适配层：两侧同时将
`target.copy_(source.index_select(0, select))` 改成 `torch.index_select(source, 0, select, out=target)`，
其中target仍是同一固定buffer的有效前缀。source与target不别名，shape/dtype可保持不变；状态ABI与内核均无需改变。
本机已安装torch_npu为 `aten::index_select.out` 注册PrivateUse1内核，证据在 `out-registration.json`。
注册只说明入口存在，不证明该形状的out实现内部无copy，也不证明替代开销为零。本轮不实施、不另采集。

**现有证据不能给出正式链“最多确定追回多少ms”的精确答案。** 能给出的范围与假设是：

- 只看已定位的enqueue前差分，诊断工作预算为1.528890 ms/16步；把全部copy主机差分都理想消除，则是1.734515 ms/16步。
- 这些是同一profile下的差分主机工作预算，未建立完整反事实依赖DAG，不能称为严格的关键路径收益上界。
- 单侧新实现全部copy主机wall为5.073645 ms，旧侧也有3.339130 ms；公平修改会同时影响旧侧，不能只给新侧减5ms。
- 所有320次设备copy均在对应host input范围结束前完成，设备copy时间已被其他主机工作覆盖，不能再加0.107/0.274 ms到可省时间。
- 因此不能用2.324233减1.734515得到“已解释后的剩余”，也不能把优化后的假想数值用于重新判定准入。

这次已找到可处理的主要来源，范围限定为fixture的冗余copy提交；其是否有正式净收益尚未验证。
原型、原性能失败及原质量记录继续冻结，本轮诊断到此收口，不自动实施修复或扩大调查。

## 证据与复核

- 原冻结探针/内核/结果均未修改；新增诊断入口为 `benchmarks/multi_graph_gdn_q14_prepare_diagnostic.py`。
- 窗口并集分析为 `benchmarks/multi_graph_gdn_q14_prepare_analysis.py`，排他区间及事件计时闭合有断言。
- 证据目录为 `benchmarks/results/multi_graph_structural/gdn-compact-q14-prepare-diagnostic-v1/`。
- 原始trace压缩为 `trace.json.gz`；解压SHA与采集记录一致；`download-sha256.json`核对远端文件身份。
- `window-analysis.json`保留逐步窗口、busy/idle、主机范围与launch分段；`copy-analysis.py/json`用独立flow路径核对640个输入任务。
- `summary.json`分开保存正式与诊断数字；`SHA256SUMS.json`校验本轮证据。只进行了一次设备诊断采集。

本地复核：两个新增Python入口的Ruff check和format check通过；区间并集/补集/相交自检、
prepare AST等价、trace解压SHA、执行脚本身份与中文BOM副本核对通过。
仓库统一 `bash format.sh ci` 仍被原有CRLF脚本第22行语法错误阻断，未宣称统一CI通过。
