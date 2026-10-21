# 第 28 题 FP32 LM head 独立诊断协议

在查看本轮现场投影与生成结果前固定。旧 BF16 配置全部失败记录保留，
本轮不改变 GDN 内核、compact FP32 状态表示、ready 调度、权重数值或质量门槛。

1. 首先读取原首次分歧现场的官方和 compact ready 末层 BF16 hidden，
   两张卡各保留完整 B6/Q4 的 `[24,5120]` 输入。使用同一份已加载的 BF16 head 权重，
   对比原 head 与 vLLM `hf_overrides={"head_dtype":"float32"}` 的实际执行路径。
   必须复现原四个 query 的全部 BF16 logits，并审计 FP32 matmul 的输入、权重、输出，
   记录 HF32 开关和 matmul precision。比较完整 248320 词表，而非候选词子集。
2. 生成准入条件：两张卡均复现原 BF16 logits；FP32 乘法操作数与结果确为 FP32；
   权重精确上转且 roundtrip 无损；官方与 compact 的现场 FP32 完整词表均唯一选择 866
   （` any`）。条件失败则不生成，不枚举其他精度方案。
3. 若通过，只做第 28 题各一次真实连续生成，两侧均 B1/Q4、相同 FP32 head 配置。
   B1 是本轮独立对照的固定 batch，不重新生成原 B6 中其他五题；因此新旧最终答案变化
   不能仅归因于 head。原现场 B6 投影证据和新 B1 生成结果分别报告。
4. 保留原 prompt、思考模板、greedy、seed42、EOS `[248046,248044]`、
   `max_new_tokens=81920`、`max_model_len=82944`、原 ngram 2..4 / 最多 3 draft。
   默认图目录不增桶；继续原 eager 质量入口和真实 Q4 native 算子，不声称生成在 ACLGraph
   内测得性能。两侧分别零态 prefill，按自己的预测提出/验证 draft、提交实际接受前缀；
   不向 compact 注入官方状态、hidden 或答案 token，不指定任何要选的词。
5. 每次 head 投影检查 hidden BF16、存储权重 BF16、logits FP32，记录调用计数；
   每个 decoder step 必须执行 48 次被测 GDN Q4 recurrence，回退数为零。
   结束后完成实际最后接受前缀的状态提交。所有失败/截断样本保留。
6. 评分沿用原 EvalScope 1.13.0 / AIME26 v1.1 规则，只做离线抽取，生成进程不读答案。
   报告两侧首个分歧、输出长度、结束原因、最终答案；不重做 30 题或调整解码参数。
7. 仅附少量 head 成本诊断：M=1/4/24，每项两次 warmup、五次计时；
   同时记录 wall/device 耗时和峰值新增 allocated 内存。完整 head 含原实现的权重转换、
   投影、TP gather；另外拆分一次权重转换和预上转权重的局部投影成本。
   预上转副本只用于成本拆分，不装入生成模型；本轮不实施权重缓存优化。
   旧约 4.80% 的 decoder+finalnorm 收益不能作为新配置的端到端收益。

原生成 manifest SHA256：
`851b74dbdc6cb8a8e1cecd26617e06080562896bcf882c9fbd15362d1b192fa8`。
模型 `/models/Qwen3.5-27B-original`，TP2，304 个 decoder NZ29 权重转换维持不变。
本轮结果使用全新目录 `gdn-aime26-q28-fp32-head-v1`，不覆盖原证据。
