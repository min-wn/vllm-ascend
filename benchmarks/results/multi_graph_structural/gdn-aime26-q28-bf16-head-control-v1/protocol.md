# 第 28 题 compact/B1/BF16 head 条件匹配对照

仅在原 FP32 配对完成、旧记录检索未找到完全匹配的 compact/B1/BF16-head 后启动一次。
对照为已完成的 compact/B1/FP32-head，唯一实验变量是 hf_overrides.head_dtype。
同一模型 /models/Qwen3.5-27B-original、同一 BF16 权重、TP2、同一设备 2,3；
compact FP32 ready 内核及状态算法不变，304 个 decoder NZ29 投影不变；
同一原 Q28 prompt、thinking、greedy、seed42、Q4、ngram 2..4 / 最多 3 draft、
max_new_tokens=81920、max_model_len=82944、原 EOS [248046,248044]，默认图目录不增桶，原 eager 质量入口。
零态独立 prefill，并依据自己的生成结果连续推进；无强制 token、无状态或 hidden 替换。
TASK_QUEUE_ENABLE=0，HCCL_DETERMINISTIC=true，HF32=False，highest，CPU threads=8。
原 FP32 源码留在独立目录，本目录入口只增加 dtype 配置和对应审计，具体差异完整归档。
新增生成仅此一次；运行失败和截断保留，不自动重试。沿用原冻结 EvalScope 规则评分。
不扩大 AIME、不优化 head、不调整 GDN、不放宽原门槛，保留原 head 成本及全部失败记录。
