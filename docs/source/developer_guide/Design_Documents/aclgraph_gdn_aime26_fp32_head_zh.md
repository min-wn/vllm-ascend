# 第 28 题 FP32 LM head 独立诊断

当前状态：原现场完整词表投影及成本诊断完成，第 28 题一次成对真实生成正在进行。
本文件后续仅补本轮生成与审计结果，不启动新配置搜索。

## 现场结论

**真正的 FP32 head 消除了这个保存现场的 `any/all` 舍入并列。**
两侧完整 248320 词表都唯一选择 ` any`，但这不代表完整生成相同或解题质量恢复。

复用原首次分歧现场两张 TP rank 的末层 hidden：完整 B6/Q4，形状 `[24,5120]`，
选中第 12 行，对应第 28 题、第 176 个输出 token 的预测器。没有重跑原 128 步审计。
同一加载模型中的 BF16 head 权重同时用于两种投影；四个 query 的原完整 BF16 logits
逐项精确复现。两张卡的结果一致。

| 输入 hidden 来源 | head 输出 | `any` (866) | `all` (660) | top-1/top-2 间隔 | 完整词表 argmax |
| --- | --- | ---: | ---: | ---: | --- |
| 官方 | 原 BF16 | 27.0 | 26.875 | 0.125 | `any` |
| compact ready | 原 BF16 | 27.0 | 27.0 | 0 | `all`，两个词并列，较小 ID 获选 |
| 官方 | FP32 | 26.9702339172 | 26.9343757629 | 0.0358581543 | `any`，唯一 |
| compact ready | FP32 | 26.9783630371 | 26.9475631714 | 0.0307998657 | `any`，唯一 |

FP32 下，compact hidden 导致 `any` 增加约 0.008129、`all` 增加约 0.013187，
二者相对间隔缩小但仍为正。BF16 的 26.9375 舍入中点使 compact 的 `all` 也变为 27.0，
从而把原本不足以改变赢家的扰动变成并列后的 argmax 变化。这说明此现场存在舍入放大。
它不表示全部模型误差来自 head，也不单独解释最终答案 107→115 的整条推理变化。

FP32 投影结果转回 BF16 时，以上两个候选词复现原设备值；完整现场目标 query 中，
每侧另有 43/248320 个词的舍入结果与原 BF16 投影不同，因此没有宣称两条投影只差一次
输出 cast。原 top-3 的 FP64 点积仅作参考；本轮判断依据是实际设备完整词表投影。

## 确认配置实际生效

使用当前 vLLM 已有配置 `hf_overrides={"head_dtype":"float32"}`，没有替换 checkpoint。
模型主体、末层 hidden 和存储的 LM head 权重仍是 BF16；GDN SSM 仍是 FP32。

- 本机 `ModelConfig.head_dtype` 与实例 `logits_processor.head_dtype` 均为 `torch.float32`。
- 实际实例为 Ascend 的 logits processor；其 `_apply_head` 调用 vLLM 原实现。
- Ascend 落到原 `F.linear(hidden_states.to(float32), lm_head.weight.to(float32), ...)` 分支。
  直接记录到的两个乘法操作数均为 FP32，局部输出 `[24,124160]` 为 FP32，
  TP gather 后 `[24,248320]` 仍为 FP32。不是 BF16 logits 的事后上转。
- `torch.npu.matmul.allow_hf32=False`，`torch.get_float32_matmul_precision()='highest'`。
- 单 rank 权重 `[124160,5120]`，BF16→FP32→BF16 的全表 roundtrip 精确一致。
  权重上转不会补充原 BF16 权重中不存在的精度。

源码位置（以归档 SHA 为准）：

1. `/vllm-workspace/vllm/vllm/config/model.py` 的 `ModelConfig.head_dtype`、`_get_head_dtype`。
2. `/vllm-workspace/vllm/vllm/model_executor/layers/logits_processor.py:136` 的 `_apply_head`。
3. `/vllm-workspace/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:542` 的
   `AscendLogitsProcessor`，以及 `:613` 的常规 TP gather 路径。

本机 `TorchDispatchMode` 没有暴露该 custom-op 内部的 aten 事件，第一份 site 记录
因审计信息缺失而没有放行生成。补充审计仅临时包装并直接调用原 `F.linear` 与
`Tensor.to`，记录操作数类型、shape 和 storage；没有改值、缓存权重或替换算子。
补充审计产生的两侧四个 query 完整 FP32/BF16 logits 与第一份记录逐位一致。
生成只在两张卡均通过补充审计后启动。所有早期入口/审计失败记录保留。

## 实际 head 成本

910B3，TP2，少量 head 独立计时，每项两次 warmup、五次测量；表中是每次取两 rank
较大 wall 耗时后的中位数。包含现有路径的转换、投影与 TP gather，不含 decoder。
补充操作数审计复用这组成本，没有重新增加性能轮次。

| 投影行数 | 原 BF16 head | 现有 FP32 head | 增量 |
| ---: | ---: | ---: | ---: |
| 1（prefill 最后一个 hidden） | 1.525 ms | 5.806 ms | +4.281 ms |
| 4（本轮 B1/Q4） | 1.566 ms | 5.942 ms | +4.376 ms |
| 24（原 B6/Q4 现场） | 1.930 ms | 6.405 ms | +4.475 ms |

每次调用确实重新转换**整份本卡 LM head shard**。单卡 BF16 存储权重占
1,271,398,400 bytes（约 1.184 GiB）；临时 FP32 副本占 2,542,796,800 bytes
（约 2.368 GiB）。B1/Q4 实测单次 head 的峰值新增 allocated 约 2.545 GB/卡，
包含临时副本及操作中间量。它是调用时的临时开销，不能当作永久缓存。

拆分诊断中，仅转换权重的 wall 中位数约 3.182 ms；已上转权重的局部 FP32 投影
约 2.420 ms（不含转换和 TP gather）。这些是独立动作，不能把两者的和当成端到端计时。
上转副本只用于成本拆分，没有装入生成模型。本轮不实施权重缓存或其他性能改动。

旧约 4.80% 是 decoder+finalnorm 的闭合调度收益，不含 LM head、采样及服务。
本轮未测新配置的端到端净收益，不能沿用旧百分比。

## 第 28 题独立生成范围

两侧只生成第 28 题一次，均为 B1/Q4，从原 prompt 分别零态 prefill，然后各自根据实际
生成历史提出 ngram draft、验证及提交接受前缀。不指定 `any`，不注入官方 hidden、状态
或答案 token。保留原 thinking、greedy、seed42、EOS、81920 输出上限及 82944 上下文
容量，未使用原 128 步上限。默认图目录不增桶，仍为原 eager 质量入口。

本轮 B1 与旧 30 题中的 B6 运行条件不同，因此不能把新旧最终答案的变化全部归因于
FP32 head；本轮有效的配对比较是同一 B1/FP32-head 配置下的官方与 compact。
完整长度、结束原因、最终答案及新的首次分歧须待这一次配对完成后报告。

## 边界与保留记录

原 0.5% hidden 数值门槛失败、真实文本 KL/TV/NLL 初筛失败、原 AIME26 官方 26/30
对 compact 23/30，以及全部局部退化记录继续有效。FP32 head 是单独的新配置。
即使第 28 题答对，也只是已知失败题的一次诊断观察，不作为总体质量或生产可用证明。

新证据目录：`benchmarks/results/multi_graph_structural/gdn-aime26-q28-fp32-head-v1/`。
完整词表 tensor 的已校验归档保存在工作区
`work/gdn-aime26-q28-fp32-head-v1-raw/site-evidence.tar.gz`，另在远端保留；
Git 中保存 JSON、协议、哈希和运行源码，不把大 tensor 当作普通文本文件提交。
