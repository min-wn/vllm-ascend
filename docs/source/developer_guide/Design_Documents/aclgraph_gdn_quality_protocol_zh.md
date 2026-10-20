# GDN 真实文本预测质量评估：预注册协议 v1

本文在查看本轮模型输出前确定。旧 hidden NRMSE ≤0.5% 检查失败及全部诊断仍然有效；本轮评估它是否伴随可检测的预测质量损失。准入预算是本研究预先选定的工程筛选条件，不是社区统一认可的误差容限。

## 不变项与对照

复用 Qwen3.5-27B、TP2、BF16 权重和激活、FP32 SSM，四份既有 native SO、compact FP32 ready、官方 dense recurrence 和原安装版本。只新增评估脚本，不改内核、状态算法、精度或 ready 调度。decoder 的 NZ 配置对两侧一致。官方与 compact 从同一次真实文本官方 prefill 分出独立 SSM、卷积及 FIA KV 历史。官方 CPU 状态换入换出仅用于正确性，不能用作性能对照。

调用官方 `language_model.compute_logits`，经过同一 LM head、TP2 词表合并及词表 padding 移除后比较全部 248320 个词条。温度固定为 1，不截断 top-k，不施加采样过滤。分布归一化及 NLL 使用 FP32，统计聚合使用 CPU FP64。输入不套聊天模板，不添加 BOS/EOS，不省略词表中的特殊 token。记录实际 logits dtype、shape 和运行时 gather 配置。

## 初筛文本与连续状态

数据为 [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext)，`wikitext-103-raw-v1` 的 test split。下载时冻结 Git revision、原始 Parquet SHA256、文章文本 SHA256、模型 tokenizer 文件 SHA256 和最终 token manifest SHA256；输入准备阶段不得加载模型或查看 logits。

按原始行顺序识别一级文章标题，文章内部原样连接行文本。按文章顺序贪心分配 128 个互不重叠 token 窗口：每个 owner 所需长度由原 Q4 轨迹的最大位置加一个 next-token target 决定；当前文章余量不足则跳到下一篇。窗口不得跨文章、重复文本或循环填充，不按模型表现挑样本。若语料不够或不足 16 篇来源文章，协议不成立，应报告输入准备失败，不悄悄更换集合。

保持原 Q4 SHA `0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524` 的全部 128 步、128 owner、前缀长度、请求排列、暂停恢复、接受长度和默认桶。只替换 token 来源。每个 owner 在整条轨迹中连续推进，两侧独立演化，不逐步重置到官方状态。

输入位置 p 的 hidden 经 LM head 预测真实 token p+1。每个非末次出现取本步接受前缀 a 对应的前 a 个输出；该 owner 最后一次出现使用既有终末物化的 `1+owner%4` 接受数。只计已经接受上下文产生的预测，每个目标位置恰好一次，不计 padding、重叠或拒绝候选。首次 decode 输入 token 本身的概率来自相同官方 prefill，因此不计入两侧 decode 筛选。困惑度为这段连续真实文本 continuation 的 token 加权条件 PPL，不是 WikiText 全测试集标准 PPL。

## 预先确定的初筛准入条件

以下条件须全部满足，缺数据、非有限值或 owner/位置不一致均不得通过：

| 条件 | 上限或要求 |
|---|---:|
| 全部 128 owner、128 步和预定计分 token 完整 | 必须 |
| 来源文章数 | 至少 16 |
| compact/官方 PPL 比的一侧 95% 上界 | ≤1.01 |
| 平均 KL(官方‖compact) | ≤0.001 nat/token |
| 单 token KL 的 P99 | ≤0.01 nat |
| 平均 total variation | ≤0.01 |
| 单 token total variation 的 P99 | ≤0.05 |
| greedy top-1 不一致比例 | ≤2% |
| 任一 owner 的平均 NLL 增量 | ≤0.05 nat/token |

PPL 比由 exp(配对平均 NLL 差)得到；按来源文章聚类进行 10000 次配对 bootstrap，seed=20261020，取 95 分位作为一侧上界，保留同文章窗口的相关性。文章数量有限，区间仅代表这个固定英语语料的抽样不确定性，不等同于所有任务的保证。同步报告 JS、真实 token top-1 命中率、top-1 分歧时的官方 margin、逐步及逐 owner NLL，避免只给总均值。原 NRMSE 作为诊断列保留，不在本轮重新解释成通过。

## 通过初筛才执行的有限任务对照

使用 [EleutherAI/LAMBADA OpenAI](https://huggingface.co/datasets/EleutherAI/lambada_openai) test 的固定前 128 条合格英语样本，冻结数据 revision、原文件和选定行号。合格条件只依赖输入：最后一个以空白分隔的词在当前 tokenizer 下由 1～4 token 构成，分词边界明确，prompt 至少 2 token 且连同 Q4 不超过 2048。不按回答正确与否筛选。

预填充 prompt 到最后一个词之前（保留 prompt 最后一个 token 给 Q4），用同源 prompt/词序列运行两条路径。所有目标词 token 的 teacher-forced greedy 预测均正确才算该样本通过，并同时报告整个目标词 NLL。官方前缀与 compact 公式都执行真实层。该任务是有限 cloze 对照，不是自由生成或通用能力测试。

任务筛选条件：128 例必须完整；compact exact-word accuracy 相对官方下降不超过 2 个百分点，且“官方正确、compact 错误”的新增失败不超过总样本的 3%。给出成对计数和样本量限制，不把小样本无显著差异称为等价证明。不通过不得扩大样本寻找通过结果。

## 收口规则

本轮不新增性能轮次、不改门槛、不改算子、不扩模型或精度方案。初筛失败则报告失败指标及位置，停止任务对照；初筛通过才执行上述有限任务。两级均通过仅说明该部署候选在冻结条件下具有初步质量可接受证据，原数值门槛仍失败，仍缺广泛任务与服务验证。无论结果如何都保留约 4.80% 的既有 decoder+final norm 闭合计时观察，不将它扩张为含 LM head 的端到端收益。
