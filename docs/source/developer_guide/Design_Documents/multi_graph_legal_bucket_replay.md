# 默认档位完整模型图的切分复用

## 约束与目标

本实验恢复动态 shape 图推理的原始约束：仅使用既定捕获档位，
每条分支复用完整 decoder 图，不为实际 batch、Attention 分支或组合新增专用形状。
不使用 head 切分或共享 Dense 的精确形状图。

固定随机样本仍为 B98，Qwen3-4B BF16，Q1，KV512，单卡 910B3。
当前默认配置 `max_num_seqs=256`、`max_model_len=2048`，
真实 dispatcher 选择 padding 104；默认档位为 `[1, 2, 4, 8, 16, ..., 256]`。

| 对照或策略 | 捕获容量 | B98 时真实请求分配 | dummy 槽位 |
| --- | --- | --- | --- |
| padding | 104 | 98 | 6 |
| exact 切分 | 96 + 2 | 96 + 2 | 0 |
| balanced 切分 | 56 + 48 | 50 + 48 | 6 |

exact 在默认档位中选择能精确覆盖 B 的最均衡两项；
balanced 在总容量不超过 padding 的合法两项中优先选择容量差最小者。
两者都不从测量结果挑选 batch，候选不存在时直接报错，不创造新档位。
策略枚举在捕获前完成，本轮不是在线路由性能测试。

## 图与测量协议

每个进程只捕获本次比较需要的三张图：一张 padding、两张完整 decoder 分支。
各图在独立流上捕获，含 embedding、全部 decoder 层和 final norm；
两路分支共享模型权重，但完整执行各自的 Dense 与 Attention。
没有为 96+2 或 56+48 创建组合外层图。
本轮只使用默认集合的子集，尚未测量完整默认图库的捕获时间与内存成本。

计时从输入准备前开始，到输出完成后结束，包含：

- 创建当前请求的 CPU 输入和长度、页表、slot 数据并复制到固定图缓冲区。
- 根据当前 metadata 重新 resolve FIA task。
- 提交两张图，逐图执行必要 FIA update，并等待完成。
- 裁去 dummy 输出，按请求顺序合并两路结果。
- 最终设备同步。

这是显式 CPU 准备与复制的参考实现，尚未优化为生产 runner 的 pinned buffer 管线。
不能把其准备耗时直接归为实际 vLLM 服务的固定开销。
padding 每步 36 个 FIA update，切分每步 72 个；所有 update 都在计时范围内。
本轮不包含 LM head、sampling、draft、调度器，也没有完整复制默认编译融合路径。

计时前用相同图完成 B98/长度512 → B97/长度511 → B98/长度512 的复用检查。
同时改变输入 token、positions、slot 和分支页表；每次与同容量 eager 路径比较。
断言图对象、输入与输出缓冲区地址未变，期间不重新捕获。
这证明这些已捕获图能处理相邻 batch 与长度变化，不证明所有动态序列或跨形状精度合格。
正式性能计时仍固定 B98/KV512，每进程五轮、每轮三十次，交替两种模式。

## 配对实测

物理卡 0 按 exact、balanced、exact、balanced 顺序执行四个独立进程。
下表为各轮 wall p50 的中位数，包括上述 host 准备和输出合并成本。

| 组合与运行 | padding ms | 切分并行 ms | 切分相对 padding |
| --- | --- | --- | --- |
| 96+2，第一次 | 24.827 | 32.789 | 慢 32.07% |
| 56+48，第一次 | 24.818 | 33.946 | 慢 36.78% |
| 96+2，独立复测 | 24.750 | 32.881 | 慢 32.85% |
| 56+48，独立复测 | 24.787 | 33.648 | 慢 35.75% |

两种组合均未在固定随机样本上获益。96+2 虽避免六个 dummy 请求，
却多执行一条完整模型分支；56+48 虽更均衡，但没有节省总捕获容量。
不能从合法组合存在直接推定它比单图更快。
本结果不与之前不含输入准备、使用自定义中间形状的约 23.8 ms 基线直接作优化率比较。

325 项 CPU 测试通过，包括默认档位组合、无法组合时拒绝以及分支容量覆盖检查。
两项 Qwen3-0.6B NPU 回归通过，分别覆盖 4+2 精确组合和 4+4 带 padding 组合；
四组 4B 测量的动态复用与每轮输出检查也均通过。
Ruff、Markdown、修改文件 diff 与长函数检查通过。
完整 `format.sh ci` 仍有既存 Windows 工具缺失以及
`device_length_decode.py` 的禁止导入失败，不能记为全量 CI 通过。

## Trace 证据与优化边界

正式测量之后，在同一物理卡分别采集 exact 和 balanced 的独立 profile。
每种模式均有三个完整 replay marker；padding 每次包含 144 个 MatMul、36 个 FIA，
切分每次包含 288 个 MatMul、72 个 FIA，与完整 decoder 分支数量一致。
下表各列为三个 marker 对应指标的中位数，profile 耗时不混入正式性能统计。

| profile / 模式 | kernel 活跃区间并集 ms | MatMul 时长之和 ms | FIA 时长之和 ms | FIA 重叠时长 ms |
| --- | --- | --- | --- | --- |
| exact / padding | 23.305 | 8.054 | 11.358 | 约 0 |
| exact / 96+2 | 30.955 | 20.933 | 14.178 | 0.223 |
| balanced / padding | 23.113 | 8.055 | 11.282 | 约 0 |
| balanced / 56+48 | 31.738 | 20.993 | 13.606 | 约 0 |

kernel 活跃区间并集消除了不同流同时执行区间的重复计数；
算子时长之和则会包含重叠，不能直接当作关键路径时间。
FIA 重叠定义为 FIA 时长之和减去其执行区间并集；微小负值来自 trace 时间戳精度，视为约零。
MatMul 调用数翻倍也不等于 FLOPs 翻倍，因为每条分支的 batch 更小。

这些数据支持以下判断：

- 设备 kernel 活跃时间本身增加约 7.7–8.6 ms，不能把退化全部归因于 Python 输入准备。
- MatMul 的累计执行时长显著增加；更小矩阵和并发执行并未在这次实验中带来总体效率提升。
- 两路 FIA 的执行重叠很少，提交到两个流不代表 Attention 已有效并行。
- 仅凭时长无法区分带宽竞争、算力竞争、算子选择与调度等待各自的贡献；尚无硬件计数器证据。

后续优化必须继续保持默认档位和完整图复用约束。
首先需要解释不同流之间实际的 kernel 排布、FIA update 就绪时刻与分支等待，
并把 host 准备、update 提交和设备执行分别归因；单纯减少 Python 调用不能解释全部差距。
任何提交或资源调度改动都应在同一 B98、相同档位、相同计时口径下复测，
不能通过新增 49/98 图或 head 切分替代这个问题。
本轮建立了合法基线和可复用性检查，尚未获得性能优化，也未完成实际 serving 的动态 batch 轨迹验证。

## 复现

~~~bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_replay.py \
  --model /models/Qwen3-4B --strategy exact \
  --output /workspace/legal-exact.json

ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_replay.py \
  --model /models/Qwen3-4B --strategy balanced \
  --output /workspace/legal-balanced.json
~~~

输出路径不得覆盖已有文件。`--trace` 在性能计时后额外采集两种模式各三个 replay。
报告保存完整默认捕获列表、实际图容量、请求分配、update 次数和复用检查。
原始证据位于工作区 `outputs/multi_graph_legal_buckets/`。

此前[固定随机样本的结构实验](multi_graph_default_capture_case.md)使用专门的 49+49
Attention 与 98 Dense 形状；其约 5% 收益不满足本实验约束，不能作为本方案的性能证据。
