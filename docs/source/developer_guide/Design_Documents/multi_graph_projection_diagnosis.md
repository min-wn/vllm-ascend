# 默认档位切分的逐算子性能归因

## 问题与证据边界

本轮接续[图切分差距定位](multi_graph_bucket_diagnosis.md)，
继续使用 Qwen3-4B BF16、B98、Q1、KV512、910B3 单卡。
完整 decoder 仍是 padding104 对照 96+2 或 56+48，长度 update 保留。
本轮没有修改完整 decoder 执行策略，也没有新增自定义捕获档位。

结论是：已经定位到小 batch Dense 的成本不随请求数同比下降、
MLP 投影的并发执行膨胀，以及部分向量算子与 FIA 同时执行时的长区间。
这些结论比总耗时更具体，但尚未获得 HBM 流量、缓存命中率、Cube/Vector 利用率计数器，
不能宣称已经证明某一种微架构资源饱和。

## MatMul 的归属校验

当前 TP1 Qwen3 每层顺序为 QKV、FIA、输出投影、gate/up、down。
分析器要求每条流恰好出现 36 次 `MatMul, FIA, MatMul, MatMul, MatMul`，
顺序或数量不符即拒绝归因。不是仅按所有 MatMul 序号取模后无条件标记。
模型 forward 代码和实际加载的权重形状交叉验证如下：

| 投影 | 每层权重形状 `[N,K]` | 输入形状 |
| --- | --- | --- |
| QKV | `[6144,2560]` | `[M,2560]` |
| Attention 输出 | `[2560,4096]` | `[M,4096]` |
| MLP gate/up | `[19456,2560]` | `[M,2560]` |
| MLP down | `[2560,9728]` | `[M,9728]` |

M 为该图的默认捕获容量，与真实请求数可能不同。
该映射只适用于已验证的四投影、每层一 FIA 的当前模型结构。

## 完整图中的主要损失位置

以下为前轮 `confirm-balanced-optimized-1.trace.json` 的三个 replay 中位数，
协议为 pinned 输入加逐层交替 update。表中数值为 36 层对应 kernel 时长之和，单位 ms。
并发列含两条分支，单独两列分别运行已经捕获的同一张分支图。

| 投影 | padding104 | 56 单独 | 48 单独 | 56+48 并发 |
| --- | --- | --- | --- | --- |
| QKV | 1.675 | 1.516 | 1.400 | 4.389 |
| 输出投影 | 0.932 | 0.893 | 0.853 | 2.709 |
| gate/up | 3.556 | 3.659 | 3.490 | 11.133 |
| down | 1.891 | 1.786 | 1.812 | 5.872 |

相对两条分支单独执行时长之和，MLP 两类投影约占新增 MatMul 累计时长的 72%。
前轮 chunk4 trace 的对应占比约 64%，两份 trace 都指向 MLP 是主要位置。
这不是说 MLP 贡献了相同比例的 wall 退化：重叠 kernel 的累计时长不能直接换算为关键路径。

逐层 update 的 96+2 trace 中，gate/up 执行区间约 94.7% 与另一流 MatMul 重叠，
与另一流 FIA 重叠仅约 1.8%；down 的对应比例约为 86.5% 和约零。
因此不能把 MatMul 退化全部归于 FIA 占用资源。
这些比例是时间重叠的相关性，不能单凭它们断言每个 kernel 的因果关系。

### 向量算子也有长执行区间

在逐层 update 的 56+48 trace 中，AddRmsNormBias 累计约 5.42 ms，
单独两图之和约 1.52 ms；SwiGLU 约 2.98 ms，单独之和约 1.00 ms。
并发时 AddRmsNormBias 的执行区间约 74.4% 与另一流 FIA 重叠，SwiGLU 约 67.4%。
chunk4 trace 同样出现该现象，且更明显。

这表明还需要关注 FIA 与向量算子的相互影响，不能只检查 Cube MatMul。
但 profiler duration 可能包含设备调度/抢占等影响，不等于纯算术执行时间。
96+2 与 56+48 的重叠结构不同，不能把一种组合的竞争比例直接套给另一种。

## 没有 FIA 时，Dense 是否仍然落后

新增 `multi_graph_dense_probe.py` 作为归因对照：

- 使用同一模型真实的 36 层权重，单个诊断图顺序执行每层的一次指定投影。
- 输入为固定随机 BF16 张量；各层独立，不运行 Attention、norm、激活或 update。
- 遍历 36 个不同权重，避免重复一个权重形成不现实的全热缓存循环。
- 捕获容量仍来自真实默认集合；比较 padding、两路并发，并测单路作归因。
- 五轮、每轮三十次、交替测量顺序；profile 在正式计时之后执行。
- 捕获输出与 eager 比较；最终版本还检查并发测量后的每个输出。

这是算子诊断，不是新的推理模式。它省略 FIA/update 是为了隔离变量，
不能替代完整 decoder 的性能或精度认证；更不能把四个独立 sweep 的时间直接相加当成推理时延。

首轮 ND 结果，单位 ms，每个数字包含 36 层同类投影：

| 投影 | padding104 | 56+48 并发 | 慢于 padding | 96+2 并发 | 慢于同进程 padding |
| --- | --- | --- | --- | --- | --- |
| QKV | 1.790 | 2.247 | 25.5% | 2.222 | 23.6% |
| 输出投影 | 1.016 | 1.441 | 41.8% | 1.396 | 39.4% |
| gate/up | 4.158 | 4.941 | 18.8% | 4.956 | 19.1% |
| down | 2.062 | 2.926 | 41.9% | 2.711 | 33.2% |

96+2 来自独立进程，百分比使用该进程自己的 padding，未混用表中 balanced padding。
两路并发仍快于各自单独执行耗时之和，因此不是完全没有并行；
但该收益不足以抵消拆成两次投影的成本，且所有四类投影在无 FIA 时仍落后于单图。

尤其 B2 的 gate/up 仍约 3.20 ms，而 padding104 约 4.16 ms。
请求数缩小约五十倍并没有让 kernel 成本同比缩小。
这与每个小 batch 承担较高的权重访问/算子固定成本相符，但尚不能区分两者的具体占比。
共享同一份权重存储也不等于两张图能把同一权重的读取和计算合并一次完成。

独立进程复测使用最终版脚本，仍为五轮、每轮三十次：

| 投影 | 56+48 的 padding / split ms | 96+2 的 padding / split ms |
| --- | --- | --- |
| QKV | 1.799 / 2.241 | 1.787 / 2.234 |
| 输出投影 | 1.016 / 1.463 | 1.000 / 1.402 |
| gate/up | 4.164 / 5.004 | 4.153 / 4.988 |
| down | 2.050 / 2.921 | 2.063 / 2.745 |

复测四类投影继续全部落后，排除了首轮偶然波动作为主要解释。

## ND / NZ 布局对照

加载权重的实测 format ID 为 2（ND），这是当前默认 `weight_nz_mode=1`
对 BF16 的预期行为：该模式仅转换量化权重。
官方 `weight_nz_mode=2` 才同时允许 BF16/FP16 转 NZ，不是配置漏失。
依据为 `AscendConfig.weight_nz_mode`、`utils._should_trans_nz` 和官方 additional_config 文档。

诊断脚本在捕获前把同一权重转成 format 29（FRACTAL_NZ），
实际 kernel 从 `aclnnMatmul_MatMulCommon_MatMulV2` 切换为
`aclnnMatmulWeightNz_MatMulCommon_MatMulV2`。
所有布局对照仍与原 ND 权重计算的 eager 结果比较，未放宽 `rtol=0.02, atol=0.02`。

首轮 56+48 对照：

| 投影 | ND padding / split ms | NZ padding / split ms |
| --- | --- | --- |
| QKV | 1.790 / 2.247 | 1.683 / 1.992 |
| 输出投影 | 1.016 / 1.441 | 0.922 / 1.167 |
| gate/up | 4.158 / 4.941 | 4.540 / 4.754 |
| down | 2.062 / 2.926 | 1.709 / 2.243 |

布局会影响结果，但效果因投影而异。gate/up 的 NZ padding 反而变慢，
其相对差距缩小不能全部归为 split 本身的提升。
本轮没有把 NZ 用于完整 decoder，因此不声称其已缩短完整推理时间或解决切分劣势。
独立 NZ 复测的 padding / split 分别为 QKV 1.668 / 1.982 ms、
输出投影 0.934 / 1.161 ms、gate/up 4.556 / 4.761 ms、down 1.720 / 2.258 ms，
与首轮方向一致，四项 split 仍慢于各自的 NZ padding。

## 复现与验证

~~~bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_dense_probe.py --model /models/Qwen3-4B \
  --strategy balanced --weight-format loaded --trace --output /workspace/dense-nd.json

ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_dense_probe.py --model /models/Qwen3-4B \
  --strategy balanced --weight-format nz --output /workspace/dense-nz.json

python benchmarks/multi_graph_projection_trace.py /workspace/decoder.trace.json \
  --output /workspace/projection-analysis.json
~~~

当前结果是对既有退化的进一步归因，未修改生产 runner 或默认配置。
原始结果、源码快照与分析脚本位于工作区 `outputs/multi_graph_dense_diagnosis/`。
全部 dense profile 的每个 marker 已验证为单路 36 个、双路 72 个 MatMul，
每种模式三个完整 marker；采集警告不作为完整性通过的依据。
profiler 会扰动提交节奏，重叠比例仅描述对应 profile，不能当作未插桩运行的精确比例。

336 项 CPU 测试通过，两项 Qwen3-0.6B NPU 回归通过，覆盖 ND 与 NZ 诊断及输出校验。
六个 4B 诊断进程完成：ND 的两种组合、NZ 的 balanced 组合各一次初测和一次独立复测。
修改文件的 Ruff、Markdown、长函数和禁止导入检查通过。
完整 `format.sh ci` 仍受既存 Windows 工具缺失及 `device_length_decode.py` 禁止导入问题影响，
不能记为全量 CI 通过。
