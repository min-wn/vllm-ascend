# 第 28 题 FP32 LM head 独立诊断

当前状态：保存现场的完整词表投影及成本诊断完成，B1/FP32-head 配对生成也已完成。
compact 正常 EOS，答案 107，正确；官方在 81920 token 上限截断，没有给出最终整数答案，
按原规则未得分。两侧完整提交审计通过。唯一的 compact/B1/BF16-head 对照已启动，
其结果出来前，完整解题恢复尚未完成对 head 精度的归因。本轮不启动新配置搜索。

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
计时输入均截取自保存的官方末层 hidden；M=1/4 匹配新入口的投影形状，
没有重新运行 prefill/decoder 或测量完整生成性能。

| 投影行数 | 原 BF16 head | 现有 FP32 head | 增量 |
| ---: | ---: | ---: | ---: |
| 1（对应 prefill 投影行数） | 1.525 ms | 5.806 ms | +4.281 ms |
| 4（对应本轮 B1/Q4 行数） | 1.566 ms | 5.942 ms | +4.376 ms |
| 24（原 B6/Q4 现场） | 1.930 ms | 6.405 ms | +4.475 ms |

每次调用确实重新转换**整份本卡 LM head shard**。单卡 BF16 存储权重占
1,271,398,400 bytes（约 1.184 GiB）；临时 FP32 副本占 2,542,796,800 bytes
（约 2.368 GiB）。B1/Q4 实测单次 head 的峰值新增 allocated 约 2.545 GB/卡，
包含临时副本及操作中间量。它是调用时的临时开销，不能当作永久缓存。
这里的 allocated 是 PyTorch 统计口径，不包含其未跟踪的 CANN/通信库内存，
因此不能把这个数当作整卡物理占用的上限。

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
FP32 head。同 B1 的 head 归因需要再比较 compact FP32/BF16，而不能直接用旧 B6 代替。

| 本轮 B1/Q4 路径 | 原规则答案与判定 | 生成 token 数 | 结束原因 | Q4 step 数 | 回退 |
| --- | --- | ---: | --- | ---: | ---: |
| 官方 FP32 head | 无最终整数答案，未得分 | 81920 | 长度上限截断 | 44873 | 0 |
| compact FP32 head | 107，正确 | 23242 | 正常 EOS | 15484 | 0 |
| compact BF16 head | 唯一条件匹配对照正在运行 | 待定 | 待定 | 待定 | 待定 |

官方退出码为 0，表示程序完成，并不表示成功解题或正常 EOS。它在思考阶段逐项枚举
到约 `n=323` 后触及原上限，尚未输出最终答案。原冻结评分器从全文抽取了早先的
`n > 12`，`integer_only=True` 按 `integer_literal_required` 判为不匹配。
保留这一原始抽取及截断记录，没有改用人工答案、只评最后一句或重跑更长生成。
compact 的末尾为 `\boxed{107}`，原规则判正确；两侧都没有运行错误或算子回退。

旧 B6 是独立记录，另列如下：

| 原 B6/Q4 路径 | 答案 | 判定 | 生成 token 数 | 结束原因 |
| --- | ---: | --- | ---: | --- |
| 官方 BF16 head | 107 | 正确 | 19587 | EOS |
| compact BF16 head | 115 | 错误 | 35069 | EOS |

### B1/FP32-head 首次分歧

两侧最长共同输出前缀为 41 token，第 42 个输出 token 首次不同：官方为 ` =`（283），
compact 为 `$.`（12576）。附近文本分别为 `$|S| = |T| = n$` 和
`$|S| = |T|$. Let $n = |S|$`。这不是保存 B6 现场的第 176 个 token `any/all` 分歧。

两侧这一位置均属于 Q4 step 30（从 0 开始）、batch=1、owner=0、query offset=0、
seq_len=213、预测位置 209、输出位置 210；上一轮及本轮接受长度均为 1，draft 为空。
此前至 step 30 的 Q4 输入元数据、draft 和接受划分逐步相同；该位置没有批次变化或
其他请求退出。本轮只对齐真实生成日志，没有为新分歧另跑局部
算术实验。因此它说明两条生成路径在何处分开，不能单独解释最终解题差异。

### 完整调用与终末提交审计

两侧两张 TP rank 的 head 调用均覆盖一次 prefill 及每个 Q4 step；官方为 44874 次，
compact 为 15485 次。生成记录及 head 审计均显示零回退。实际乘法类型审计覆盖首次
prefill/decode，后续每次调用检查 hidden、输出及存储权重 dtype，保留 FP32 head 配置。

逐步审计重建了独立生成历史、ngram draft、真实 greedy 验证、接受前缀、位置及 owner。
48 层的调用数逐层等于各自 Q4 step 数，日志完整，最大实际位置分别为 82089 和 23412，
均在 82944 上下文容量内。末次接受选择器为官方 2、compact 1；官方保留相应候选态，
compact 等待 ready 完成并物化真实最终接受前缀，检查 compact 日志/valid 清零。
审计通过的是每条独立历史的提交契约，不是不同生成历史下的终末状态数值相等证明。

### 唯一的 B1/BF16-head 归因对照

配对生成及终末审计完成后，检索了远端既有的第 28 题 compact 完整生成记录：仅发现
本轮 B1/FP32 和旧 B6/BF16，没有完全匹配的 B1/BF16，故按原授权只补一条。
它与 compact B1/FP32 使用相同设备 2,3、TP2、原 prompt、同一套 BF16 checkpoint 数值、
FP32 compact ready、Q4、greedy、seed42、thinking、EOS、81920/82944 上限及零态初始化。
入口只增加 `head_dtype` 选择及对应类型审计，差异完整归档；实验变量是 head 精度配置。
它从原 prompt 独立连续推进，不强制 token，不注入官方状态或 hidden，不调整 GDN。

该对照仅用于判断本题答案是否因 head 配置而改变。即使 compact FP32 正确、BF16 错误，
也不能推断普遍无损；若两者都正确，则本题不提供 B1 答案准确率恢复的证据。
投影精度配置同时影响投影算术与输出舍入，不能把完整生成对照描述为只隔离最后一次 cast。

## 边界与保留记录

原 0.5% hidden 数值门槛失败、真实文本 KL/TV/NLL 初筛失败、原 AIME26 官方 26/30
对 compact 23/30，以及全部局部退化记录继续有效。FP32 head 是单独的新配置。
即使第 28 题答对，也只是已知失败题的一次诊断观察，不作为总体质量或生产可用证明。

FP32 配对证据目录：`benchmarks/results/multi_graph_structural/gdn-aime26-q28-fp32-head-v1/`，
评分及首次分歧见 `paired-result-with-text.json`，完整调用/提交审计见 `pair-lifecycle-audit.json`。
BF16 对照单独使用 `gdn-aime26-q28-bf16-head-control-v1/`，不覆盖 FP32 和原 B6 记录。
完整词表 tensor 的已校验归档保存在工作区
`work/gdn-aime26-q28-fp32-head-v1-raw/site-evidence.tar.gz`，另在远端保留；
Git 中保存 JSON、协议、哈希和运行源码，不把大 tensor 当作普通文本文件提交。
