# GDN AIME26 独立实际生成协议

本轮仅回答 compact FP32 ready 在这 30 道数学题上观察到怎样的实际解题变化。
原 hidden NRMSE 门槛失败、真实文本 KL/TV/局部 NLL 失败与 FP32 诊断全部保留。
任务得分不覆盖这些记录，不作为普遍无损或生产可用证明。

## 冻结对象和数据

- 模型：原 Qwen3.5-27B，TP2，BF16 权重和激活、FP32 SSM、HF32 关闭。
- 官方 recurrence 与 compact FP32 `deferred_tile_ready` 四个 native SO 及依赖保持原哈希。
- 304 个 decoder 投影沿用两侧共同的 NZ29 格式；LM head 使用同一官方实现。
- 数据：ModelScope `evalscope/aime26`，revision `12ae93b463353acd4d272dbbc7dbecbcfb902642`，
  `aime2026.jsonl` SHA256 `52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee`，test 全部 30 题。
- 工具：EvalScope 1.13.0 wheel，AIME adapter evaluation_version `v1.1`。
  原始 wheel、源码和依赖清单留档；数据顺序不筛选、不打乱。
- 提示：逐字复用 wheel 中 AIME `PROMPT_TEMPLATE`，包括开头换行；0-shot、无 system prompt。
- 聊天模板：原 checkpoint `chat_template.jinja`，`enable_thinking=True`，
  `add_generation_prompt=True`。生成 manifest 只有题面及 prompt token，不包含参考答案。

## 生成设置（查看任务输出前冻结）

- 每题每版本一次，greedy，temperature 0，top_p 1，top_k 禁用，所有 penalty 为中性值。
- 模型支持确定性的 argmax 解码；模型卡推荐采样，但没有禁止 greedy。
  本轮依用户优先减少采样变化，使用 greedy，不用于复现模型卡成绩。
- seed 42；输出最多 **81,920 token**，与模型卡对竞赛题建议的预算一致。
- EOS 为 checkpoint generation_config 的 `[248046, 248044]`；不加文本 stop，
  不强行截断思考，不因出现 boxed 答案提前结束。
- max_model_len **82,944**；输入长度必须 ≤1,020，额外覆盖末次 Q4 的推测尾部。
- 固定每组最多 8 题，按原数据顺序分成 8/8/8/6；两侧同组、同 owner、独立进程和独立缓存。
- 沿用 eager 整网质量入口，仅增加真实生成和容量适配。本轮不声称验证长上下文 ACLGraph
  生产服务，也不记录性能收益。

## 真实 Q4 接受与状态语义

各侧从自己的零状态和相同 prompt 完成官方 prefill，取末 token 的 LM head argmax。
之后输入为“上次末输出 + 本侧历史的最多 3 个 ngram 提案”，采用安装版 vLLM 的
longest matched ngram 方法（长度 2..4）。不足的提案补齐 Q4 输入，填充部分不能被接受。

当步四个 logits 的 argmax 与提案顺序比较，接受连续匹配前缀并输出首个纠正或 bonus token。
EOS、输出预算可在前缀内部终止。下一步 convolution 选择和 compact/dense SSM 选择都使用
该实际接受数；FIA 的逻辑位置只前进接受数，拒绝尾部由下一步覆盖。
每步必须满足“已计算状态长度 = prompt 长度 + 已输出 token 数 − 1”。
终末提交使用本请求最后一次真实接受数，保留末输出 token 尚未作为输入计算的通常解码语义。
不调用旧固定轨迹的 `terminal_counts`，不注入另一侧状态或参考 token。

每个 GDN 层记录 Q4 调用数，要求与实际 decode 步数一致；未知 handle 或非 Q4 路径直接报错，
不回退。共同的官方 prefill 单独记录，不算 compact decode 回退。

## 容量与接入资格

正式 30 题开始前，运行非题集的有界生成接入检查，验证真实接受、EOS 和末次提交。
预分配各 owner 到 82,944 位置的独立 FIA 页；检查所有页和末端 slot 无越界、无别名。
在最后位置执行 FIA 容量探针及完整 decoder 有限性探针，记录位置、shape 和分配内存。
该探针只证明地址和算子可执行，不取代真实长生成的正确性记录。
生成过程中每步检查长度、位置、接受数、48 层调用数、提交 join 和输出有限性。

## 规则评分与完整记录

使用 EvalScope AIME adapter 的规则分支：`extract_answer` 后
`compare_answers(..., integer_only=True)`，依赖 Math-Verify 0.9.0。
关闭 judge，不使用解题模型评分。生成完毕才向评分器提供目标答案。
为避免导入服务和 agent，独立载入固定 wheel 的 parser/contracts 原函数体，
仅解析依赖导入边，不改抽取和比较代码。

保存 generated token、原始含 special token 文本，以及
`decode(skip_special_tokens=True)` 得到的评分文本；不删除思考内容、不手工修答案。
保存抽取文本、prediction_valid、整数限制和判等原因。
解析失败及运行失败计为错误，全部 30 题进入分母；截断仍按同一规则评分并单列标记。
变化题逐项检查抽取与原文是否对应，区分数学错误、格式问题、截断和运行问题，不改评分。

交付两侧正确数/准确率、四格配对统计，以及逐题 token 数、结束原因、长度限制、
解析/运行失败、Q4 调用和回退记录。只做一次配对，不做显著性或普遍无损断言。
如接入资格失败，先报告具体缺口；不能用官方服务、teacher forcing 或部分题分数替代。
本轮完成后收口，不增加数据集、模型、重复轮次或有利配置搜索。
