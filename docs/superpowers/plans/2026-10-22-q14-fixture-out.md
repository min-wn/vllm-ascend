# Q1/Q4 fixture out 有界验证计划

**目标：** 在不修改冻结内核、ABI、精度和轨迹的前提下，判断对称采用 `index_select(out=...)` 后是否恢复单层候选资格。

**设计：** 保留原 `Trial`；新增仅替换一条 gather/copy 表达式的 `OutTrial`。独立入口负责设备正确性与四配置计时，不改原探针与历史结果。用户本轮六项要求为执行范围；不搜索其他搬运方式。

## 固定条件

- Ascend910B3 卡0、原已编译二进制、20 AIC/40 AIV、Q/K/V/beta与输出BF16、gate与状态FP32；HF32关闭。
- 原manifest第54～69步、56 owners、224行容量；复用原真实recurrence site，非原连续整网激活。
- 完整生命周期为每样本实测restore＋连续16步prepare/replay/最终等待＋terminal/等待；公共fixture搭建、静态分配、图捕获在计时外。
- 顺序：legacy-forced、legacy-q14、out-forced、out-q14；第二轮完全逆序，第三轮恢复。每轮每配置8样本，共96样本。
- 原 `stability_screen`：三轮均正收益、汇总改善>1%、改善>两侧较大轮间相对跨度的2倍。主要比较out-q14与out-forced。
- profiler与诊断插桩关闭。若正式结果已足够决定去留，不额外采trace。

## 执行与检查

- [x] 先添加CPU契约测试并记录缺少候选时失败；确认新prepare除一条表达式外AST与冻结实现一致。
- [x] 新增OutTrial与独立入口；小型及原16步设备检查两种宽度模式的新旧准备、五份有效输入、尾部、地址/shape/stride/storage及out返回视图，无resize。
- [x] 使用原5步8-owner序列做eager/graph连续状态对照：接受1～4、count变化、重排、暂停恢复；逐模式比较legacy/out及delayed/ready、有效前缀和终末物化。
- [x] 对原16步做graph状态审计；只有正确性全部通过，才启动唯一一组正式四配置测量。
- [x] 保留全部原始样本、执行源码/二进制哈希与命令；给出完整成本、每轮门槛和准备/图区间分解。
- [x] 依据原门槛恢复单层候选资格；本轮收口，不自动接完整层或runner。汇总改善2.292803%，mixed分段略退化亦保留。

## 文件

- 新增 `benchmarks/multi_graph_gdn_q14_fixture_out.py`：独立OutTrial及固定四配置/统计。
- 新增 `benchmarks/multi_graph_gdn_q14_fixture_out_probe.py`：设备审计与原边界计时。
- 新增 `tests/ut/worker/multi_graph/test_gdn_q14_fixture_out.py`：冻结差异与统计契约。
- 新增同名证据目录及中文报告；原内核、协议、探针、失败记录均不修改。
