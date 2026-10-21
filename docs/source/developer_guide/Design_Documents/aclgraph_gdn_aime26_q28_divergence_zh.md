# AIME26 第 28 题首次输出分歧诊断

日期：2026-10-21。对象：原 Qwen3.5-27B TP2、BF16 权重与激活、FP32 SSM、HF32 关闭。实现、权重、精度、原门槛均未改变；本轮没有新任务评分或性能实验。

**结论：首次分歧已复现，ready 调度未引入额外差异。该真实层样本中，compact 物化后的 FP32 状态误差比官方略大，BF16 recurrence 输出误差基本相同。最终 token 选择的直接触发点是共同 LM head 的 BF16 舍入形成并列；尚无证据支持一个有限的 recurrence 修复。**

## 1. 首次分歧及原运行条件

原始输出的最长共同前缀为 175 个生成 token；第 176 个输出（从零计数 175）首次不同：官方 `866 = " any"`，compact `660 = " all"`。附近文本为：

```text
官方：This means for any $s \\in S$, $f(s) \\notin S$.
候选：This means for all $s \\in S$, $f(s) \\notin S$.
```

| 条件 | 记录 |
| --- | --- |
| Prompt 长度 | 169 token |
| Q4 整网步 / 验证位置 | 从零计数 150 / 第 0 个 query |
| Batch / 有效 token / 默认目录容量 | 6 / 24 / 24 |
| 请求 / batch 行 / compact base 行 | owner 3 / 行 3 / 行 4 |
| 官方候选状态槽 | 13、14、15、16 |
| 输入 token | `[364, 248044, 248044, 248044]` |
| 输入位置 / 预测输出位置 | 343–346 / 344（均从零计数） |
| FIA seq_len | 347 |
| 上一步 / 当步接受长度 | 1 / 1；本步 ngram proposals 为空 |
| 活跃题号 | 25–30 |
| 当步 batch 变化 / 请求退出 | 无 / 无 |

原两侧第 28 题在 0–150 步的输入、位置、接受长度全部相同。其他请求此前已经走向不同生成路径：例如该步第三行 seq_len 为官方 401、compact 410。因此诊断将六请求的完整输入、接受长度和 Q4 划分统一为原官方轨迹。

## 2. 同一共同前缀重放与调度核对

三条路径各自从零状态完成相同 prompt 的 prefill，随后独立推进自己的 SSM、conv 和 KV。prefill 后全部 GDN base/conv 和有效 prompt KV 的哈希一致。连续 compact 路径没有导入官方中间状态。仅在下节孤立算子对照中共享局部初态。

两张卡 0、1 上顺序执行三路径，每条只执行 151 个 Q4 步，到首次分歧即停。官方完整复现原 151 步全部 24 个预测；两条 compact 路径在第 28 题此前实际接受的 175 个输出上都与原官方相同，均在第 150 步给出 `all`。统一其他请求历史后依然复现，故原其他行历史不同并非该分歧的必要条件，无需再重放原 compact 全轨迹。

两个 TP rank 均确认 delayed/ready 的以下内容逐位一致：151 步完整 hidden、全部预测，以及末步 logits、全部 48 层的 SSM base/log/valid 与卷积状态。每条路径每 rank 的每个 GDN 层均调用 Q4 recurrence 151 次，无回退。该证据支持本次诊断中的调度一致性，不证明所有运行条件下都不存在同步问题。

首次分歧处的 BF16 logits：

| 路径 | `any` logit / 排名 | `all` logit / 排名 | `every` logit | top-1/2 间隔 | argmax |
| --- | --- | --- | --- | --- | --- |
| 官方 | 27.000 / 1 | 26.875 / 2 | 26.625 | 0.125 | any |
| compact delayed | 27.000 / 并列 1 | 27.000 / 并列 1 | 26.625 | 0 | all |
| compact ready | 27.000 / 并列 1 | 27.000 / 并列 1 | 26.625 | 0 | all |

完整 top-10、各候选排名和并列数保存在逐 rank JSON。argmax 对并列值选较小 token ID，此处 660 小于 866。

针对已观察到的舍入边界，仅取原 top-3 的相同 BF16 LM head 权重，与各路径自己的实际末层 BF16 hidden 做 CPU FP64 点积；未替换模型 hidden、状态或生产计算：

| 路径 | 舍入前 any | 舍入前 all | any − all |
| --- | --- | --- | --- |
| 官方 | 26.970242893 | 26.934366536 | +0.035876357 |
| compact delayed/ready | 26.978363008 | 26.947555345 | +0.030807663 |

这些点积转回 BF16 后，三个候选均逐位复现设备 logits。`all` 越过相邻 BF16 值 26.875 与 27.000 的中点 **26.9375**，从而出现并列。所检查三候选在舍入前均由 `any` 领先；这不是完整 FP64 模型的预测，也不是一个更改 LM head 精度后的质量评测。

## 3. 实际设备实现到同一 FP64 参考的距离

取第 150 步第 0 层的真实完整 B6 Q4 输入，在同一个已接受的官方 FP32 canonical state 上做孤立局部对照。官方把它复制到对应四候选；compact 逐位复制同一个 FP32 base，并从零日志开始。两侧接受元数据相同。三条连续模型路径的状态仍保持独立。

FP64 参考将这份 FP32 初态以及真实 BF16 Q/K/V/beta、FP32 gate 精确提升到 FP64，按逐 token 的 decay → delta → 状态更新 → query 投影计算。公式等价检查另做：FP64 compact 与 sequential 输出 NRMSE 为约 `1.49–1.50e-16`。**公式等价不代替下面的实际设备误差比较。**

下表合并两张卡的本题行、完整 Q4；状态为四个逻辑前缀的更新后 FP32 状态，compact 使用冻结 native export 物化。NRMSE 定义为 `||设备结果 − FP64参考||₂ / ||FP64参考||₂`。

| 设备实现 vs 同一 FP64 参考 | recurrence 输出 max abs | 输出 NRMSE | 更新后状态 max abs | 状态 NRMSE |
| --- | --- | --- | --- | --- |
| 官方 | 0.0077527363 | 1.51792557e-3 | 4.66904715e-6 | 6.48999625e-8 |
| compact delayed | 0.0077527363 | 1.51792557e-3 | 6.60718489e-6 | 7.13901785e-8 |
| compact ready | 0.0077527363 | 1.51792557e-3 | 6.60718489e-6 | 7.13901785e-8 |

该样本 compact 状态 NRMSE 高 **10.0003%**，最大绝对误差高 **41.5103%**。只看本步真实接受的第一个前缀，状态 NRMSE 为官方 `4.09446043e-8`、compact `4.47666662e-8`，高 **9.3347%**，两者最大绝对误差同为 `2.34176177e-6`。这些是此层此样本的局部结果，不能推成所有层、整段历史或最终任务质量的误差排序。

输出已经过设备 BF16 转换，误差主要由该边界的量化尺度决定；两种设备输出相对 FP64 的总体误差基本相同。相对正确舍入的 FP64 输出，官方有 3/24576 个元素相差 1 BF16 ULP，compact 有 1/24576 个，不能据此声称 compact 的任务质量更好。本轮未插桩采集设备转换前的 FP32 recurrence 输出，不能判定该边界前的输出误差排序。

导入、表示及物化均单列核对：FP32 base 导入逐位准确，没有 BF16 状态转换；真实存储日志为 K、update 和 FP32 gamma。把同一组设备日志在 FP64 下应用，可区分日志系数误差与 FP32 物化舍入。native export 与 CPU 逐条 FP32 应用日志的结果逐位一致；表格已经包含物化差异。分项误差不能简单线性相加。

## 4. 已定位的环节、修复判断与边界

第 150 步第 0 层 Q/K/V、gate、beta 和全部 Q4 元数据逐位相同；此前独立推进的 logical state 已有约 `2.57–2.85e-7` 的相对差异。该步首 query 的 recurrence 输出在 rank0 仅一个元素相差 `4.76837158e-7`，rank1 相同。第 1、2 层的 recurrence 差异开始增大；最终该 query 的 final-norm hidden NRMSE 为 `0.01102634`。逐层完整记录在分析 JSON 中。

源码上，官方按 token 直接更新 dense FP32 状态后求投影；compact 使用 base 的 H0K/H0Q、旧日志投影修正、Gram/三角计算，再在使用时物化日志。官方在 query 上预乘 scale，compact 在输出末端乘 scale。当前证据支持这些重排与状态历史造成数值路径差异，并明确观察到 LM head BF16 舍入触发首次不同 token；尚未把最终差异唯一归到某一次 reduction、Exp、缩放或舍入操作。也未做逐算子状态历史的 FP64 全模型追踪。

没有发现具体的 owner/索引、状态生命周期、回退或 ready 同步错误，因此**没有修改实现，也没有进行 kernel 变体搜索**。改变 LM head 精度可作为未来单独假设，但不属于本轮保持精度不变的修复；更不能由这三个候选的离线点积推断最终答案会恢复。

当前证据不支持直接重启完整数值资格复验。原 0.5% 检查失败、真实文本分布失败、AIME 实际退化记录及所有门槛保留；此前约 4.80% 的调度闭合耗时改善也保留。本轮只解释生成路径首次如何分开，不能据此解释最终 107→115 的全部原因。到此收口，不扩其他题、全套评测或性能实验。

## 5. 复核入口与证据

- 对齐：`benchmarks/multi_graph_gdn_aime26_divergence.py`；最小重放：`benchmarks/multi_graph_gdn_aime26_divergence_probe.py`；离线分析：`benchmarks/multi_graph_gdn_aime26_divergence_analyze.py`。
- 复用的 FP64 算术：`benchmarks/multi_graph_gdn_fp32_arithmetic.py`、`multi_graph_gdn_q4_log_reference.py`。
- compact 算术与日志布局：`benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/q4_tile_solve.h` 的 `Projection/Solve/WriteOutput/WriteLog`；物化：`multi_graph_gdn_q4_log_kernel/q4_log_kernel.cpp` 的 `Export/ApplyOldLog`。
- 官方冻结入口：原 `installed-sources/gdn.py:564` → `fla_npu.ops.ascendc.recurrent_gated_delta_rule`；公共 LM head：原 `installed-sources/logits_processor.py:144`。
- 本轮目录：`benchmarks/results/multi_graph_structural/gdn-aime26-q28-divergence-v1/`。`alignment.json` 为原日志定位；`analysis-with-head.json` 为最终分析；`common-official/*.rank*.json` 为各路径逐步记录。
- `diagnostic-metadata.tar.gz` 保留诊断源码、日志及来源；两份完整真实激活归档保留于远端 `/workspace/gdn-aime26-q28-divergence-v1/snapshots.rank{0,1}.tar.gz` 和本地工作区 `work/gdn-aime26-q28-divergence-v1-raw/`，SHA256 与下载核验见 `snapshot-archives.json`、`local-archive-verification.json`。大体积激活归档不并入 Git。
- 冻结工具、native 库及权重 size/mtime 在运行前后核对；原正式 AIME 归档未改。入口的首次启动曾有字符串转义错误，未执行模型，日志另存；修正后一次三路径重放正常退出。没有失败样本重试或择优取结果。
- 10 项 CPU 单测、四个新增 Python 文件的 Ruff 检查及格式检查、Git 差异空白检查通过。`format.sh ci` 因当前 Git Bash 环境缺少 `pre-commit` 命令退出，不能报告全仓格式检查通过。
