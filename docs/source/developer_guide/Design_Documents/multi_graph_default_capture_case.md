# 默认捕获档位下的固定随机 Batch

## 样本与测量口径

本轮先固定一个随机 batch，再分析与优化；不按收益重新抽样。
Python `random.Random(20261016).randint(64, 256)` 的第一次结果为 B=98。
模型为 Qwen3-4B BF16，单卡 910B3，普通 decode Q=1，KV=512，切分为 49+49。

通过当前容器真实的 `EngineArgs.create_engine_config()` 和
`CudagraphDispatcher` 解析配置与选图。设置 `max_num_seqs=256`、
`max_model_len=2048`，不指定 capture sizes 或最大捕获大小。
生成的 `max_num_batched_tokens=2048`，图模式为 FULL_AND_PIECEWISE，
捕获列表为 `[1, 2, 4, 8, 16, ..., 248, 256]`。
B=98 的 uniform decode 实际 dispatch 到 FULL、104-token 图。
因此 padding 为 6 个请求，额外槽位比例为 6/98=6.12%，并非手动选择的容量。

沿用完整 decoder 基准，包括 embedding、36 层、final norm 以及必要 FIA update。
不包含 LM head、draft、sampling、调度器和输入准备时间。
两路切分每次执行 72 个 FIA update，padding 执行 36 个，没有省略真实长度更新。
固定 KV 长度，但每次仍执行 update；检查包括各自结构的 eager/replay 和 changed-input。

默认配置在本轮决定的是 padding 对照的容量。
计时仍由控制基准加载 eager 模型后手动捕获 decoder，未完整复刻默认编译与服务路径。
两路 Attention 使用精确 49+49 形状，共享 Dense 使用 98 形状；
这些不是默认捕获列表里的既有图，也不代表动态 batch 图缓存复用已经完成。
因此该实验回答的是普通固定样本的切分潜力，不是在线推理性能结论。

## 顺序复测结果

所有性能实验在物理卡 0 顺序运行。每个进程内 padding 与 split 交替五轮，
每轮三十次；表内为各轮 wall p50 的中位数。先后次序与表格一致。

| 实验 | padding ms | 切分 ms | 相对 padding 延迟下降 |
| --- | --- | --- | --- |
| 原结构，不限核，第一次 | 23.833 | 23.810 | 0.10% |
| 两路各 10 Cube / 20 Vector，第一次 | 23.782 | 22.603 | 4.96% |
| 两路各 20 Cube / 40 Vector，拓扑对照 | 23.854 | 23.758 | 0.40% |
| 两路各 10 Cube / 20 Vector，独立复测 | 23.875 | 22.668 | 5.06% |
| 原结构，不限核，独立复测 | 23.810 | 23.733 | 0.32% |

原结构和全核双流对照基本持平，不能认定稳定优势。
10 Cube 的两次复测均节省约 1.2 ms，说明上轮限核优化在这个固定随机样本上也有效。
该结果没有以改变 batch 或扩大 padding 为前提。
资源上限控制的是流可用核心数量，不保证两条流物理独占。
共享 Dense 保持整卡资源，捕获 Attention 的两条流及 FIA update 流使用一致限额。

## Trace 定位与优化含义

两个配置各采集 padding 和 split 的三个完整 replay marker。
每个 marker 均为 144 个 MatMul；padding 为 36 个 FIA，split 为 72 个 FIA。
以下为各 trace 第一个完整 marker；profiler 数据不替代上述无 profiler 配对计时。

| 指标 | padding，不限核对照 | 原切分，不限核 | 切分，每路 10 Cube |
| --- | --- | --- | --- |
| FIA 累计 duration | 11.335 ms | 17.313 ms | 20.256 ms |
| FIA 区间并集 | 11.336 ms | 11.504 ms | 10.462 ms |
| FIA 重叠时间 | 约 0 | 5.809 ms | 9.794 ms |
| MatMul 累计 duration | 8.057 ms | 8.278 ms | 8.235 ms |
| 所有 kernel 区间并集 | 23.327 ms | 23.387 ms | 22.034 ms |

原切分虽然重叠执行 FIA，但 FIA 区间并集没有比 padding 更短。
限核后单个 FIA 更慢、累计 duration 更大，但重叠增加，区间并集缩短约 1 ms。
三次切分 marker 的 FIA 区间并集分别为：不限核 11.504/11.495/11.450 ms，
限核 10.462/10.490/10.693 ms；改善并非仅见于一个 marker。
Dense MatMul 的累计时间基本不变，因此该样本的进一步收益主要与 Attention 阶段有关。
这些证据支持资源受控并行的方向，但不能完全分离 tiling 变化与硬件资源竞争的贡献。
Profiler 日志有停止时处于 RECORD 的通用警告；已核对六个 marker 的上述 kernel 数量，
只对这些完整 marker 作分析，不宣称日志中的所有活动都完整。

本轮将既有 `--attention-cores 10` 优化应用于固定随机样本并完成复测，
没有新增或省略 FIA update，也未修改生产默认值。
继续优化应以该样本的 Attention 关键路径为对象，
包括资源限额与 FIA tiling 的匹配、分支 KV 写入及固定同步开销；
不能用更大的 padding、更多流或更碎的 Dense 自动推定改善。

## 复现

~~~bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_default_capture.py \
  --model /models/Qwen3-4B \
  --seed 20261016 --batch-min 64 --batch-max 256 --max-num-seqs 256 \
  --cores 0 10 20 10 0 --rounds 5 --repeats 30 \
  --output-dir /workspace/default-capture-reproduction
~~~

输出目录必须为空，以保留历史证据。`selection.json` 保存完整捕获列表与实际 dispatch。
各次独立进程保存参数、日志和原始报告；`summary.json` 汇总配对结果。
脚本保存开始时的 benchmark SHA，并检查运行中源码未改变。
使用 `--select-only` 可以只解析配置和选图，不加载模型进行性能测量。

本轮原始矩阵由归档的配置探测与 runner 分步启动，
上述脚本将相同流程整理为仓库内可复现入口。
原始数据保存在工作区 `outputs/multi_graph_default_capture/`。

新增默认档位选择 NPU 测试通过；新入口还完成 Qwen3-0.6B 的完整 subprocess
冒烟运行，检查报告、源码哈希及汇总落盘。该冒烟样本 B7 的性能不作为优化证据。
五组正式测量的 replay、各轮输出与 changed-input 检查均通过，
但这些同结构检查不证明跨形状数值等价。
修改文件的 Ruff、Markdown 与 diff 检查通过。
完整 `format.sh ci` 仍受 Windows `/bin/bash`、shellcheck、python3 入口缺失，
以及既存 `device_length_decode.py` 禁止导入失败影响，未记为全量通过。

## 后续判断标准

固定 B98/C104/Q1/KV512 作为优化样本，不用更换 batch 掩盖退化。
以配对 wall time 判断优化，trace 用于寻找关键路径；累计 kernel duration 不能直接当作收益。
在固定样本上的优化完成后，再用独立 batch 和真实动态序列检验泛化。
当前结果不证明默认编译服务、精度跨形状等价、多卡或所有 batch 均有优势。
