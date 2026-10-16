# 默认档位切分的性能差距定位

## 范围与方法

接续[默认档位完整模型图基线](multi_graph_legal_bucket_replay.md)。
固定 Qwen3-4B BF16、B98、Q1、KV512、单卡 910B3，padding104，
切分仅使用默认集合中的 96+2 或 56+48；不新增专用图，不做 head 切分。
所有 FIA update、输入准备与复制、输出合并、最终同步仍在计时内。

本轮增加三类诊断：

- profiler 阶段标记：准备、图提交、每路 update、输出合并、最终同步。
- 同一进程中单独运行已经捕获的每条分支，诊断并发带来的执行变化。
- 可关闭的分块交替 update、固定 pinned 输入缓冲区、流核数限制对照。

单分支结果只用于归因，不是新增的竞争方案。
正式对照始终只有 padding 与切分并行。
CPU 阶段耗时与设备执行相互重叠，不能把二者直接相加；
profiler 数据用于定位，正式优化率来自未开启 profiler 的 wall 测量。
逐层 update 的大量标记还会改变 host 提交节奏，进而改变设备重叠关系；
其 profile wall 甚至可能低于未插桩结果，不能把 profile 当作无扰动的执行重现。

## 已定位的差距

### 第二路 FIA 的启动延迟

原实现先 replay 两张图，再依次更新第一路的全部 36 层、第二路的全部 36 层。
图内 FIA 等待对应 update 的 external event，因此图已经提交并不等于 FIA 已可执行。

balanced 基线的第一个 profile replay 中：

- 第一路首个 FIA 在 marker 开始后 3.98 ms 执行。
- 第二路首个 FIA 在 12.88 ms 才执行。
- 第二路从首个到末个 kernel 之间有约 9.67 ms 的无 kernel 区间。

按四层交替 update 后，第二路首个 FIA 提前到 5.22 ms，
第二路内部空档降至约 1.95 ms。
这验证了 update 顺序导致的就绪延迟，但不能把追回的空档全部算作总时延收益。

### 并发时算子执行变长

在同一进程中，56 容量图（50 个真实请求）单独运行约 18.44 ms，
48 容量图单独运行约 17.34 ms；两路并发仍需约 32.36 ms。
这不是理想的 `max(18.44, 17.34)`。

对应 pinned/chunk4 profile 的三个 replay 中位数：

| 指标 | 56 图单独运行 | 48 图单独运行 | 两图并发 |
| --- | --- | --- | --- |
| MatMul 时长之和 ms | 7.840 | 7.538 | 26.689 |
| FIA 时长之和 ms | 5.789 | 5.607 | 15.210 |
| 所有 kernel 区间并集 ms | 17.261 | 16.174 | 30.534 |

两图并发的 MatMul 累计时长比两张图单独运行之和明显更长。
同时，提前交替 update 后，两路整体 kernel 重叠增大，但各自执行时间也增大。
因此“第二路更早就绪”只能解决一部分差距。
这些证据说明并发执行下存在明显的执行效率损失；
尚不能细分带宽、算力、缓存与设备调度等待的贡献，没有据此声称已证明 HBM 饱和。

96+2 的小分支也并非免费：在对应独立诊断中，96 图单独约 23.97 ms，
2 图单独仍约 10.43 ms。两条分支都经过全部 decoder 层和 Dense 权重。
对应 2 图的 MatMul 累计约 6.94 ms，而 96 图约 7.81 ms，
表明缩小 batch 没有按比例降低这部分执行时间。
减少 6 个 dummy 请求不能抵消第二条完整模型分支的固定成本。
MatMul 调用数从 144 增至 288，但这不等于 FLOPs 翻倍。

### 输入准备不是主要差距

balanced baseline profile 中准备阶段约 1.70 ms。
固定 pinned staging buffer 并用 `non_blocking=True` 提交复制后约 1.43 ms，
仍每步重建当前输入、长度、页表与 slots，仍每步 resolve 和 update。
没有缓存固定请求内容，也没有把复制移到计时外。
主流 ready event 保证图在复制完成后读取，最终同步保证下一步重填 staging 时没有未完成读取。
这个改动能减少一些 host 成本，无法单独消除设备端数毫秒的差距。

## 优化与未通过的尝试

`--update-chunk-size N` 在每条图内保持任务顺序，交替提交两路的 N 个 update。
0 保留逐图完整更新，1 逐层交替，4 每四层交替。
每条 update 流只等待一次输入 ready；所有 task update、external event、完成依赖均保留。

`--pinned-inputs` 使用各图固定持有的 pinned CPU staging buffer，
把每步的阻塞复制改为受 ready event 约束的异步复制；padding 同样使用该优化。
默认参数仍复现原基线，优化需显式开启。

`--lane-cores 10 10` 为两路完整图及各自 update 流使用相同的资源限制。
该尝试未通过原有同形状、全核 eager 参考的精度检查：
18.1% 元素超出 `rtol=0.02, atol=0.02`，最大绝对误差 1.0。
核数变化可能改变 tiling 和累加顺序，但本轮没有证明它只是可接受的舍入差异，
也没有放宽阈值或用受限参考替换原参考。因此不报告其性能收益，不纳入可用优化。
该参数仅保留为会执行完整精度检查的诊断选项。

## 完整复测

完整复测使用同一版 benchmark，在物理卡 0 串行运行。
每个进程五轮、每轮三十次，padding 与切分交替计时；
两次独立重复均按 balanced baseline、balanced optimized、exact baseline、exact optimized 顺序。
optimized 使用逐层交替 update 与 pinned 输入，两种方案都保留相同计时边界。

| 组合 / 重复 | 原 padding ms | 原切分 ms | 优化 padding ms | 优化切分 ms | 切分自身耗时降低 | 优化切分仍慢于 padding |
| --- | --- | --- | --- | --- | --- | --- |
| 56+48 / 1 | 24.700 | 33.801 | 24.623 | 31.833 | 5.82% | 29.28% |
| 56+48 / 2 | 24.809 | 33.664 | 24.711 | 32.063 | 4.76% | 29.75% |
| 96+2 / 1 | 24.835 | 32.937 | 24.789 | 31.271 | 5.06% | 26.15% |
| 96+2 / 2 | 24.773 | 32.878 | 24.762 | 31.300 | 4.80% | 26.41% |

切分自身改善使用同一重复中原切分和优化切分的比值；
最后一列使用优化配置进程内的配对 padding，不能混用分母。
每个数是五轮 wall p50 的中位数。
全部八个正式报告的源码 SHA256 一致，图容量合法性、动态复用与各轮输出检查通过。
源码 SHA256 为 `bdcf1dc7dab381117b6254975300999b90759068df02544597f19f9cbd149f7c`。

这次优化稳定追回了部分等待成本，尚未让切分优于 padding。
若继续缩小剩余差距，需要针对两张完整图的 Dense 执行成本和并发下的算子效率做实验，
并在同一默认图库约束下验证；仅优化输入复制和 host 调用已不足以解决当前差距。

## 复现与检查

~~~bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_replay.py \
  --model /models/Qwen3-4B --strategy balanced \
  --update-chunk-size 1 --pinned-inputs --diagnose-lanes --trace \
  --output /workspace/bucket-optimized.json

python benchmarks/multi_graph_bucket_trace.py \
  /workspace/bucket-optimized.trace.json --output /workspace/bucket-analysis.json
~~~

替换 `--strategy exact` 检查 96+2；去掉 update 和 pinned 参数得到当前原始协议对照。
分析器裁剪 marker 内的 kernel 区间，排除控制任务和 CPU 同名事件，
输出各流活跃区间、首个 FIA 时间、空档、算子时长之和与区间并集。
这些指标描述时间区间，不能视作硬件利用率。

333 项 CPU 测试通过，包括交替 update 的完整性、每路顺序以及 trace 区间裁剪。
四项 Qwen3-0.6B NPU 回归通过，覆盖两种合法组合、默认协议和 pinned/chunk4 协议。
4B 复测仍逐次检查 B98/512 → B97/511 → B98/512 的图复用和同形状 eager 精度。
这不等于跨形状精度认证，也不等于已完成服务端动态 batch 轨迹验证。
修改文件的 Ruff、Markdown、长函数和禁止导入检查通过。
完整 `format.sh ci` 已执行，仍有既存 Windows 工具缺失和
`device_length_decode.py` 的禁止导入问题，未通过全量 CI。

本轮改动位于 benchmark、诊断工具和测试中，没有自动修改生产 runner 的默认策略。
原始报告、trace 与执行脚本保存在工作区 `outputs/multi_graph_bucket_diagnosis/`。
