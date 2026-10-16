# 默认桶并行图的主机准备与提交瓶颈

## 实测范围

本轮继续使用 Qwen3-4B BF16、Q1、KV512、默认 104 桶与 56+48 attention 容量。
动态轨迹仍为 B98 → B97 → B103 → B98，固定分界不变，每种策略一张图。
每次正式运行六轮、每轮 64 步，padding 和切分先执行的轮数相同。
输入准备、全部 FIA update、replay、依赖与同步均包含在正式计时中。
原 ND eager 参考、rtol=0.02/atol=0.02、动态资格检查与地址检查均保留。

测量对象仍是 embedding 和第一层真实 decoder block。
本轮优化了该独立测试入口的控制路径，不能直接宣称完整模型或服务获得同等百分比收益。

## 已确认的较大开销

### 输入准备与多次小块 H2D

原入口每步将 ids、positions、lengths、pages、slots 五组 Python 数据转换为临时 CPU Tensor，
复制到五块 pinned staging，再分别提交五次 H2D。
脱离 profiler 的分段诊断中，4B 输入准备区间中位数约为 595–615 μs，
正式完整 block 单步约为 1.42 ms。这是一个值得优先压缩的固定开销。

新增 `PackedStepBuffers` 和 `--packed-metadata`：

1. 捕获前分配连续 pinned host buffer 与连续 device buffer，按 64 字节对齐建立固定类型视图。
2. 每步把实际输入写入 NumPy 视图，不临时构造 CPU Tensor，不预存有利 batch 的输入结果。
3. 一次 H2D 上传整个包，捕获图直接读取 device views，不增加图内 unpack kernel。
4. lengths 的 CPU Tensor 直接引用对应 host view；FIA 的真实长度列表仍逐步更新。

打包后的输入准备诊断中位数约为 359–408 μs；诊断本身位于正式计时之外。
独立 trace 验证，padding 的全部 DMA 事件从每步六次降为两次，切分从七次降为三次。
其中额外的一次/两次是 FIA 更新相关传输，依然保留；不能把“一次输入上传”描述为“全部传输只有一次”。

这也揭示了旧测试入口包含的准备开销。生产 runner 已使用可复用的 CPU/NumPy buffers，
因此“删除临时 CPU Tensor”的收益不应直接外推到生产；其多次 H2D 是否值得合并需独立测量。
本入口每步最终同步，确保复用 pinned 包时前一次 H2D 已完成；异步服务接入还需要完成 fence 或双缓冲。

### replay、事件与 update 的碎片化主机提交

旧入口在 Python 中逐次操作 ready event、replay stream、graph.replay、update stream、
updated event、done event 和主流等待；原生适配器仅覆盖 FIA update。
打包输入后，padding 的 replay/update/join 诊断区间中位数分别约 137/161/25 μs，
切分约 165/339/28 μs。两路 FIA 仍有额外的供给成本。
这些主机区间可以与设备执行重叠，不能相加后视作整步可节省时间。

新增 benchmark-only `NativeReplay` 与 `--native-replay`，集中处理上述控制动作。
它保留输入就绪、每一个 FIA update、每个 update 流的完成依赖和主流回收等待。
NPUGraph 的 Python 对象布局没有公共定义，因此保留一次公共 `graph.replay` 绑定调用，
不通过猜测私有结构取得 C++ graph 指针。图和 timing event 的所有者均被保留。

两组原生提交实验中，切分的 graph-plus-update-ready 事件跨度约为 693/672 μs，
此前同源码对照约为 835 μs；该跨度包含图内等待及 update 就绪，不是纯 kernel 时间。

## 正式结果（2026-10-16）

下表为六轮单步 p50 的中位数；各进程启动与 capture 不计入 replay。
打包阶段按照原入口 → 打包 → 打包复测 → 原入口复测执行，避免只采集单向的有利变化。

| 配置 | 动态 padding ms | 动态切分 ms |
| --- | ---: | ---: |
| 原入口 | 1.423610 | 1.424278 |
| 原入口反向复测 | 1.420928 | 1.428755 |
| 打包上传 | 1.195964 | 1.190599 |
| 打包上传复测 | 1.199267 | 1.184392 |
| 原生提交同源码对照（已打包） | 1.188382 | 1.199507 |
| 打包＋原生集中提交 | 1.115717 | 1.131059 |
| 打包＋原生集中提交复测 | 1.087159 | 1.099384 |
| 原生提交对照反向复测（已打包） | 1.188397 | 1.187697 |

打包阶段的切分单步约减少 16%–17%；增加原生控制后，相比本轮原入口约减少 21%–23%。
这是两边共同的控制路径优化，不能作为切分相对 padding 获胜的证明。
原生集中提交两组中，切分仍分别比 padding 慢约 1.38%、1.12%。

完整 block 的 CPU 输入准备只发生一次，设备层计算则会在完整模型的各层重复。
禁止用上述单层百分比推算全模型同比加速，也不能把节省时间简单乘以层数。

## 设备侧计数与后续方向

另外采集 Level1 MemoryAccess，原始 kernel CSV 中存在实际计数：

| FIA 查询容量 | 样本数 | kernel 中位数 μs | AIC read-main-memory KB | AIV read-main-memory KB | Block/Mix Block |
| --- | ---: | ---: | ---: | ---: | --- |
| 104 | 3 | 305.547 | 205129.875 | 9027.250 | 20/40 |
| 56 | 3 | 179.764 | 117319.625 | 5624.875 | 20/40 |
| 48 | 3 | 159.543 | 88131.500 | 4492.625 | 20/40 |

切分两路计数相加没有减少主要读取量，两路也仍然各自启动 20/40 个 block。
Block 数不等同于利用率；这里尚不能据此断言 HBM 已饱和，或把 main-memory 计数直接当作 HBM 实测带宽。
不过，下一步必须针对约 300 μs 的 FIA 及其大规模 KV 读取寻找能减少实际工作量的变化，
继续增流或复制 Dense 并不会消除这部分读取。

额外采集的 PipeUtilization 三个样本中位数如下；比例列按各自 pipeline 统计，存在重叠，不能相加。

| FIA 查询容量 | AIC MTE2 活跃比例 | AIC MAC 比例 | AIC scalar 比例 | AIV vector 比例 |
| --- | ---: | ---: | ---: | ---: |
| 104 | 77.6% | 6.3% | 75.2% | 4.1% |
| 56 | 79.5% | 6.4% | 72.3% | 4.1% |
| 48 | 78.9% | 6.1% | 72.5% | 4.0% |

较高的 MTE2 活跃和低 MAC 比例支持优先研究搬运与控制开销，而非增加矩阵计算并发。
CSV 的 `cube_utilization(%)` 不能替代 MAC pipeline 比例，也不能直接解释为接近峰值 FLOPS。
后续有价值的算子实验应针对 Q1/GQA 的 KV tiling、预取与布局，验证能否减少搬运等待；
它们必须对 padding 同样应用，并用现有原 ND 精度门槛验证。
若不能在保持 BF16 的前提下降低关键路径，减少 KV 字节量是另一种实质性方向，
但低精度 KV 需要独立的算子支持与精度验证，当前未实现，不能预设其收益或直接启用。

两次高等级 profiler 的 CSV 采集成功，但原分析器只识别 Level0 的 `KERNEL_*` 标签，
未识别 Level1 的 `AI_CORE`、`AI_VECTOR_CORE`、`MIX_AIC`，报告因此保留为失败。
本轮补充标签识别与 CPU 回归，用原始归档 trace 离线重分析后，
两份 trace 均通过六个范围、每 block 四个 MatMul 和一/两个 FIA 的检查。
初次失败日志及原报告保留；离线检查和计数汇总另存 `hardware-analysis.json`。
这些 profiler 计数仅用于诊断，不用于给出正式 wall latency 收益。
仍需 L2/带宽等证据才能进一步区分具体的数据搬运层级，不能宣称已经证明整卡 HBM 饱和。

## 复现与检查

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_block.py \
  --model /models/Qwen3-4B --nz --native-update --static-task-plan \
  --shared-cache --packed-metadata --native-replay \
  --phase-diagnostics --rounds 6 --repeats 64 --trace \
  --output /tmp/bucket-host-native.json
```

去掉 `--native-replay` 为同源码控制提交对照；再去掉 `--packed-metadata` 为原准备路径。
`--phase-diagnostics` 在正式计时后另收集每种模式 32 步主机分段和 NPU event 跨度。
所有原始数据、源码快照、独立复测和 profiler 失败日志位于工作区
`outputs/multi_graph_bucket_packed/`，哈希和采样契约由对应汇总脚本验证。

CPU multi-graph 回归 369 项通过；NPU 原生计划边界检查、打包后的三模式动态图复用、
独立 update 流及整数视图实际传输共四项通过。
全仓 `format.sh ci` 已执行；Windows `/bin/bash`、shellcheck、部分 `python3` hook 启动问题，
以及既有 `device_length_decode.py` 的 Triton 导入检查仍失败，不能称为全仓检查通过。
变更文件的 Ruff、clang-format、markdownlint 和导入检查，以及直接运行的长函数和 symbolic meta 检查通过。
