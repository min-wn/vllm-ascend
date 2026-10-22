# Q1/Q4 共用 compact 状态：fixture out 有界验证

> 状态更新（2026-10-22）：当前compact Q1/Q4部署扩展已正式结束。本文是历史阶段记录，旧候选资格、runner接入及补救建议不再是待办；数值、性能和失败数据保持原样。当前边界见[最终收口结论](aclgraph_gdn_compact_closeout_zh.md)。

日期：2026-10-22。历史性能失败、数值与任务质量失败记录全部保留。

## 结论

**本次对称 out/out 对照通过原稳定收益门槛，恢复 Q1/Q4 原型的单层候选资格。**
这不是完整 GDN 层、整网、runner 或服务性能结论，也不恢复跨算法数值与任务质量资格。
原内核、状态 ABI、精度、资源、轨迹、原探针与旧准备方式均保持不变；本轮只新增一个 fixture 写入方式。
本轮完成后收口，不自动进入完整层或 runner，不搜索其他搬运变体。

## 实验对象与唯一变化

原准备方式为 `target.copy_(source.index_select(0, select))`，新方式为
`torch.index_select(source, 0, select, out=target)`。target仍为同一固定buffer的有效前缀。
新增 `OutTrial` 的 prepare 与原 Trial 的 AST 除这一条表达式外完全相同；冻结原探针的规范化SHA保持
`bde7a1c1e4d9e401f9ce37528e2442f018cd9c067ecec3877494c557d7872b5a`。
五份输入仍在每一步按当前owner/query即时gather；metadata构造、CPU tensor、H2D、索引long转换与接受记录更新仍在计时内。
没有预生成整条轨迹的激活、将准备移出计时或只优化Q1/Q4侧。

硬件为 Ascend910B3 卡0、CANN 9.1，torch 2.10.0+cpu / torch_npu 2.10.0.post4。
20 AIC/40 AIV、Q/K/V/beta及输出BF16、gate与compact状态/日志/产品FP32、HF32关闭及highest精度设置沿用原配置。
使用原已编译模块，无重新编译；二进制与执行源码身份保存在证据目录。

输入为原manifest第54～69步、56 owners、224行图容量，10步Q1-only、6步mixed（42 Q1＋14 Q4）。
复用原Qwen3.5-27B TP2的rank0/layer4真实recurrence site；连续推进的是本实验状态，
不是原整网连续层激活的精确复现。保持 `input_mode=real_site_reuse`、
`initial_state_matches_original_ledger_step=False`。边界从Q/K归一化后的输入开始，不含conv、投影、norm、TP和head。

## 设备正确性

- 先在8-owner原5步序列，再在冻结56-owner完整16步进行输入准备新旧逐模式对照；覆盖forced、ready、delayed。
- 五份有效输入按原始字节比较，不只使用浮点容差；两侧未写入尾部通过不同step的sentinel检查。
- dispatch前target shape精确匹配；out返回前缀的pointer、storage pointer/bytes、shape、stride、offset及dtype不变，无resize或相关警告。
- 共315次out调用通过直接安全检查，覆盖实际56/98/224行及小型count变化。固定完整buffer的descriptor也逐步检查。
- 5步eager与ACLGraph、16步连续ACLGraph审计均逐模式比较legacy/out：输出、base、完整日志、valid、新日志和候选导出一致。
- 同时核对ready/delayed一致、强制Q4有效前缀一致、接受1～4、重排、暂停恢复、有效日志、终末物化及债务清零。
- 原16步实际接受长度全部为1；接受2～4和暂停重排的覆盖来自已有小型序列，不冒充真实ledger分布。

运行日志保留CANN/HDK需要32-padding处理并分配内存的警告；它不是out resize警告。
通过地址与shape检查不代表后端完全无临时分配、内部copy或格式转换。
本轮没有额外采trace，也不据API名字声称底层零copy。

## 正式计时方法

正确性通过且全部来源SHA匹配后，在另一进程启动唯一一次正式测量。四配置在该性能进程中各测3轮×8样本，共96个样本。
顺序预先固定为旧强制Q4→旧Q1/Q4→out强制Q4→outQ1/Q4；第二轮完全逆序，第三轮恢复。
全部样本保留，不因异常值重新采样或删除。

每个样本计入实测restore/初态导入、连续16步prepare/replay及链尾等待、终末物化及等待。
每步保留原begin/compute_begin/end三个event，仅链尾synchronize；无profiler、record_function或逐步下载/审计。
初始化使用预分配资源；公共fixture构建及source初次上传、buffer静态分配、eager预热与图捕获位于计时外，沿用原边界。
这些一次性公共搭建成本本轮没有单独测量，表中初始化不能解释为从磁盘加载site或模型的完整初始化。
每配置同样预热/捕获，默认容量不变。event分段不与wall或其他重叠区间再相加。

## 完整成本

单位ms/16步；以下为每配置24个样本的算术均值，主判据另用每轮中位数。

| 配置 | 初始化/导入 | 16步执行＋等待 | 终末物化＋等待 | 生命周期合计 |
| --- | ---: | ---: | ---: | ---: |
| 旧准备＋强制 Q4 ready | 0.189679 | 16.930273 | 0.464728 | 17.584680 |
| 旧准备＋Q1/Q4 ready | 0.182846 | 17.803545 | 0.458564 | 18.444955 |
| out 准备＋强制 Q4 ready | 0.187340 | 14.010703 | 0.465957 | 14.664000 |
| out 准备＋Q1/Q4 ready | 0.186728 | 13.687640 | 0.454550 | 14.328918 |

## 原稳定门槛

主比较只使用out-Q1/Q4对out-强制Q4，禁止将out-Q1/Q4对旧准备强制Q4的差值称为增量收益。
直接调用原 `stability_screen()`：三轮均正收益、汇总改善大于1%、且大于两侧较大轮间相对跨度的2倍。

| 轮次 | out强制Q4 生命周期中位数 | outQ1/Q4 生命周期中位数 | 改善 |
| --- | ---: | ---: | ---: |
| 1 | 14.615982 | 14.276996 | 2.319283% |
| 2 | 14.673607 | 14.343276 | 2.251195% |
| 3 | 14.713957 | 14.374360 | 2.307992% |

三轮端点汇总改善 **2.292803%**；两侧较大相对跨度为
0.678816%，其两倍为1.357633%。原门槛通过。
以全样本均值计算，生命周期减少0.335083 ms；
它与中位数端点汇总百分比是不同口径，不混用。

本轮旧准备Q1/Q4对旧准备强制Q4仍退化4.749681%。
历史原正式轮退化6.7574%及准备增量2.324233ms原样保留；当前旧准备对照的波动和不同数值不能覆盖历史结果。
同一次四配置测量中，out准备相对各自旧准备的完整成本改善：强制Q4为
16.423654%，Q1/Q4为
22.042621%。这两项是fixture修改收益，不是Q1/Q4增量收益。

## 准备与图区间的解释

以下来自同一无profiler测量的互不相交event分段，单位ms/16步。
prepare包含主机提交空隙，graph包含固定容量Q/K转换、solve、commit及启动，均不是kernel-only时间。

| 配置 | 准备event | 图event |
| --- | ---: | ---: |
| 旧准备＋强制 Q4 ready | 11.770112 | 5.028192 |
| 旧准备＋Q1/Q4 ready | 13.628028 | 4.047243 |
| out 准备＋强制 Q4 ready | 8.778281 | 5.105585 |
| out 准备＋Q1/Q4 ready | 9.476448 | 4.082663 |

当前旧准备两侧差为1.857916ms，out两侧差缩为0.698168ms；out-Q1/Q4仍有额外准备成本，
但其图区间节省1.022922ms，在当前生命周期边界内足以产生净收益。
这支持“当前fixture显式临时结果加copy路径掩盖了部分变长计算收益”的解释，
不证明底层所有copy均被消除，也不能把原profile的1.53～1.73ms预算按比例移植到本轮。
CPU、设备及等待空隙没有重复相加；本轮未增加机制trace或新kernel变体。

同一连续链的负载分段如下；这是含准备的每step event端点，不是逐负载独立生命周期测量。

| 负载 | out强制Q4 每step | outQ1/Q4 每step | 改善 |
| --- | ---: | ---: | ---: |
| Q1-only，56请求，10步 | 0.865061 | 0.831944 | 3.828306% |
| mixed，42 Q1＋14 Q4，6步 | 0.872210 | 0.873279 | -0.122629% |

收益主要来自Q1-only的10步；mixed六步端点均值略退化，不能宣称本轮对所有混合负载均有收益。
不因这项分段增加负载搜索；资格结论仍只针对预先冻结的完整16步链。

## 收口与下一阶段边界

本轮接口安全与同源连续状态检查通过，并满足原稳定门槛，因此恢复**单层候选资格**。
适用结论限定于上述16步、56-owner复用真实输入的recurrence原型，收益约2.29%，不是大幅整网收益。
尚缺完整GDN层中真实输入生产、其他算子和状态生命周期的净收益证据；本轮不自动补做。
全部旧质量失败保持有效，没有运行官方数值对照、任务评测或修改精度门槛。

## 复现与检查

- 候选：`benchmarks/multi_graph_gdn_q14_fixture_out.py`；入口：`benchmarks/multi_graph_gdn_q14_fixture_out_probe.py`。
- 证据：`benchmarks/results/multi_graph_structural/gdn-compact-q14-fixture-out-v1/`，含原始correctness/performance、96样本、命令、日志、源归档、SHA及本地汇总。
- `correctness.json`与`performance.json`来源SHA完全一致；性能入口检查已通过资格及其SHA，禁止未资格或换源码后直接计时。
- 来源门槛包含五份Python、site/manifest、三份扩展及新内核依赖库；旧Q4与deferred依赖库的SHA仅在结束后补录，未在两阶段分别再次核对。没有重新编译或修改它们，但不声称每阶段都重新校验了全部依赖二进制。
- 下载12个远端文件逐一核对SHA；源归档逐文件与本地LF规范化源码、报告sources核对一致。旧原型和内核未改。
- 本地CPU契约及原协议共71项通过；新增测试先观察缺少候选的4项预期失败，再实现。后续修正了一处测试AST定位，未改变冻结实现。
- 默认pytest conftest依赖本机缺少的vllm，Windows默认pytest临时目录另有权限问题；隔离检查使用 `--noconftest` 与本工作区显式basetemp后71项通过。
- 三个新Python文件Ruff check及format check通过。统一 `bash format.sh ci` 仍在原CRLF脚本第22行失败；未宣称全仓CI通过。
