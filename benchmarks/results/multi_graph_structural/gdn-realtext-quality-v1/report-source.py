import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent.parent
repo=root/'work/vllm-ascend'
evidence=repo/'benchmarks/results/multi_graph_structural/gdn-realtext-quality-v1'
analysis=json.loads((evidence/'analysis.json').read_bytes())
frozen=json.loads((evidence/'manifest.json').read_bytes())
raw=json.loads((evidence/'screen.rank0.json').read_bytes())
summary=analysis['summary']
means=summary['means']
paired=analysis['paired_target_accuracy']
worst_owner=max(analysis['owners'],key=lambda row:row['means']['delta_nll'])
worst_owner_ppl=(math.exp(worst_owner['means']['delta_nll'])-1)*100
if analysis['status']!='audited_real_text_quality': raise AssertionError('incomplete quality audit')
if summary['screen_passed'] and not analysis['task_run']: raise AssertionError('limited task remains required')
screen='通过' if summary['screen_passed'] else '未通过'
task=analysis.get('task')
if task:
    conclusion=('固定条件下取得初步预测与任务质量可接受证据；原数值门槛仍失败。' if task['passed'] else
                '连续文本初筛通过，但有限任务对照未通过，当前不能将误差判为已可接受。')
else:
    conclusion='预注册连续文本初筛未全部通过；当前不能将误差判为已可接受，按协议未进入任务对照。'
interpretation=('PPL 的预注册非劣性条件通过。这支持“本固定续写集合上未出现超过预算的平均预测损失”，不能扩张为所有任务无损。'
                if summary['gates']['ppl_noninferiority'] else
                'PPL 的预注册非劣性条件未通过。这表示未取得预算内非劣性证据；单凭区间上界未过，不能自动宣称已证明统计显著劣化。')
table=[]
for title,value,threshold,gate in [
    ('PPL 比单侧 95% 上界',summary['ppl_ratio_upper95'],'≤1.01','ppl_noninferiority'),
    ('平均 KL(官方‖compact)',means['kl'],'≤0.001','mean_kl'),
    ('KL P99',summary['p99_kl'],'≤0.01','p99_kl'),
    ('平均 TV',means['tv'],'≤0.01','mean_tv'),
    ('TV P99',summary['p99_tv'],'≤0.05','p99_tv'),
    ('top-1 不一致比例',means['top1_disagreement'],'≤0.02','top1'),
    ('最差 owner 平均 NLL 增量',summary['max_owner_delta_nll'],'≤0.05','owner_tail')]:
    table.append(f'| {title} | {value:.10g} | {threshold} | {"通过" if summary["gates"][gate] else "失败"} |')
segments=[]
for row in analysis['segments']:
    m=row['means']
    segments.append(f'| {row["start_step"]}～{row["stop_step"]} | {row["tokens"]} | {m["delta_nll"]:.9g} | {m["kl"]:.9g} | {m["tv"]:.9g} | {m["top1_disagreement"]*100:.5f}% |')
tails=[]
for row in analysis['largest_kl_observations'][:5]:
    tails.append(f'| {row["step"]} | {row["owner"]} | {row["position"]} | {row["kl"]:.8g} | {row["tv"]:.8g} | {row["official_nll"]:.8g} | {row["compact_nll"]:.8g} |')
if task:
    task_text=f'''已在初筛通过后执行预先冻结的 128 例 LAMBADA OpenAI 英语目标词对照。
官方 {task['official_correct']}/128（{task['official_accuracy']*100:.5f}%），compact {task['compact_correct']}/128（{task['compact_accuracy']*100:.5f}%）；
新增失败 {task['new_failures']} 例，恢复正确 {task['recovered']} 例。任务筛选{'通过' if task['passed'] else '未通过'}。
每个目标词的全部 reference token 都满足 greedy 预测才算该词正确；这不是自由生成或通用能力测评。
样本量有限，无显著下降也不能证明所有任务等价。逐例配对与目标词 NLL 保存在 analysis.json。
'''
else:
    task_text='初筛未满足全部预注册条件，因此没有运行 LAMBADA。数据与前 128 个合格样本已在模型输出前冻结，没有用任务结果反选样本。不能将“未运行任务”写成任务质量通过或下降。\n'
text=f'''# GDN compact FP32：连续真实文本预测质量评估

日期：2026-10-20。模型：Qwen3.5-27B，TP2，A2 / 910B3。

## 1. 本轮结论

**{conclusion}**

原 hidden NRMSE ≤0.5% 的数值检查失败和全部首步诊断保持原样；本轮没有修改算法、内核、ready 调度或精度。
本次判断依据是 LM head 后的真实文本预测分布、NLL、条件困惑度与预先确定的质量预算，不再单独以 hidden NRMSE 决定候选。
预注册筛选：**{screen}**。阈值是本研究的工程预算，不是通用任务质量定律；筛选失败本身也不能证明任务能力下降。

## 2. 实际评估对象与连续状态

复用 BF16 权重/激活、FP32 SSM、304 个 decoder 投影的共同 NZ 配置和四份既有 native 扩展。
10 个运行时二进制哈希与上一轮 FP32 诊断完全相同；安装提交、二进制及源码清单均归档。
官方路径仅换入换出原始 FP32 dense 状态库，不改变 recurrence 计算；这部分搬运不能用来声称性能收益。
两条路径从同一次真实文本官方 prefill 分出各自 SSM、conv 和 FIA KV 历史，此后连续推进，未逐步重置 compact。

原 Q4 的 128 个 owner、128 步、请求重排、暂停恢复、前缀长度、接受前缀与默认图目录不变；
只将随机 token 替换为 WikiText test 的真实文本。来源为 {summary['source_articles']} 篇文章的互不重叠窗口，
预填充 {raw['prefill']['tokens']} token，decode 正式计分 **{summary['tokens']} 个唯一 next-token 位置**。
每次只计接受前缀对应输出，owner 末次接受数与既有终末物化一致；排除 padding 和拒绝/重叠候选的重复计分。
每个输入位置 p 的输出预测真实 token p+1。官方和 compact 都完成了 128 步，compact 两卡终末状态债务闭合。

调用安装版 `Qwen3_5ForCausalLMBase.compute_logits`，沿用官方 LM head 和 TP2 all-gather，
完整词表 {raw['lm_head']['vocabulary']}，logits 为 {raw['lm_head']['logits_dtype']}，温度 1，无 top-k/采样过滤。
两侧以相同的 32-token 分块调用 head；分布与 NLL 用 FP32 计算，CPU Python FP64 聚合。
NPU 上的 `.double()` 观测转换曾提示退回 FP32；评价协议本来就使用 FP32 分布运算，未因此更改模型精度。
本轮为 eager 数值对照，沿用默认桶形状，没有新增图捕获或性能轮次。

## 3. NLL、困惑度与分布结果

| 指标 | 官方 | compact |
|---|---:|---:|
| token 加权 NLL（nat/token） | {means['official_nll']:.12g} | {means['compact_nll']:.12g} |
| 条件 PPL | {summary['official_ppl']:.12g} | {summary['compact_ppl']:.12g} |
| 真实 next-token top-1 命中率 | {means['official_correct']*100:.7f}% | {means['compact_correct']*100:.7f}% |

配对 NLL 增量 **{means['delta_nll']:.12g} nat/token**，PPL 点估计相对变化 **{(summary['ppl_ratio']-1)*100:+.7f}%**。
按 {summary['source_articles']} 篇来源文章聚类、seed=20261020、10000 次配对 bootstrap 的 PPL 比单侧 95% 上界为 **{summary['ppl_ratio_upper95']:.12g}**。
这是固定窗口 continuation 的条件 PPL，未覆盖整份 WikiText test，不能与公开全测试集 PPL 排行直接比较。

| 准入项 | 观测值 | 预注册标准 | 结果 |
|---|---:|---:|---|
{chr(10).join(table)}

所有 128 owner 和预定 token 完整；来源文章数也满足至少 16 篇。平均 JS 为 {means['js']:.10g}。
{interpretation}
NLL 衡量真实目标 token 的预测损失，KL/TV 衡量整个预测分布的偏移；二者可能给出不同信号。
即使平均 NLL 没有明显变差，也不能用它抹掉少数位置的分布突变；KL/TV 预算失败本身同样不等于任务能力已经下降。

具体到最差请求，owner {worst_owner['owner']}（文章 {worst_owner['article']}，{worst_owner['tokens']} 个计分 token）
的 NLL 从 {worst_owner['means']['official_nll']:.10g} 变为 {worst_owner['means']['compact_nll']:.10g}，
该窗口条件 PPL 点估计增加 **{worst_owner_ppl:.5f}%**。这是单个冻结请求的观测，不是所有请求或任务均退化该比例的结论。

目标 token 的成对命中计数：两侧均正确 {paired['both_correct']}，均错误 {paired['both_wrong']}；
官方正确而 compact 错误 {paired['official_correct_compact_wrong']}，官方错误而 compact 正确 {paired['official_wrong_compact_correct']}。
greedy 预测有 {analysis['disagreeing_tokens']} 个位置不一致；这些位置的官方 top-1/top-2 logit margin
平均 {analysis['disagreement_official_margin']['mean']:.9g}，最大 {analysis['disagreement_official_margin']['maximum']:.9g}。
这用于解释预测排序变化，不把接近并列的分歧自动当作正确性失败，也不自动豁免既定 top-1 预算。

| 连续步段 | 计分 token | 平均 ΔNLL | 平均 KL | 平均 TV | top-1 不一致 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(segments)}

最大的五个 KL 位置如下；它们没有从均值、PPL 或门禁中剔除。每个位置的正确 token、两侧 top-1 及命中情况保存在原始记录中。

| step | owner | 目标位置 | KL | TV | 官方 NLL | compact NLL |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(tails)}

本轮 hidden 诊断最大 NRMSE 为 {analysis['hidden_diagnostic_max']:.12g}，
{analysis['hidden_diagnostic_screen_failed_steps']}/128 步至少一个 rank 超出旧 0.005 门槛。
这是诊断列，不替代上面的质量指标；也不覆盖或更改上一轮原随机轨迹的数值失败记录。

## 4. 有限任务对照

{task_text}
## 5. 资格与适用边界

本轮没有证明原数值门槛通过，也没有改变首步“FP32 重排差异经 BF16 后续网络传播”的诊断。
只对冻结的英语真实文本、≤2048 上下文、Q4 动态接受和当前 BF16/FP32 配置作判断。
teacher forcing 固定上下文，有利于分离计算差异，但不覆盖自由生成分叉、中文、工具使用或更长上下文。

既有 delayed/ready 约 4.80% 闭合改善仍然是 decoder+final norm 的同算法调度观察，
没有计入本轮新增 LM head、采样或服务开销；本轮没有新性能结论。
共享规划、变长 attention 筛选及 FP16 排查维持结束，本轮完成后收口，不搜索新候选、不改阈值。

## 6. 复现与证据

首个模型执行前已写定[中文协议](aclgraph_gdn_quality_protocol_zh.md)并冻结输入，启动记录保存了二者哈希；
代码与冻结输入随后提交并推送，早于首次查看聚合质量指标。
首次 launcher 字符串语法错误发生在模型启动前，失败日志保留，正式模型筛选只有一条完整轨迹。
协议和输入 SHA256：

- protocol：`{frozen['payload']['protocol_sha256']}`。
- manifest：`{hashlib.sha256((evidence/'manifest.json').read_bytes()).hexdigest()}`。
- 原 Q4 trace：`{frozen['payload']['trace_sha256']}`。

数据来源：[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext)，revision `{frozen['payload']['sources']['wikitext']['revision']}`；
[EleutherAI/LAMBADA OpenAI](https://huggingface.co/datasets/EleutherAI/lambada_openai)，revision `{frozen['payload']['sources']['lambada']['revision']}`。
原始文件哈希、tokenizer 哈希、文章哈希、选定窗口及任务行号均在 manifest.json；原始语料另保存在本地准备目录及远端输入目录。
冻结时的输入准备源码与格式化后的仓库源码 AST 一致，原始准备字节单独保存，未回写旧输入 manifest。

主要源码：`benchmarks/multi_graph_gdn_quality_inputs.py`（输入）、
`multi_graph_gdn_quality_probe.py`（真实模型+LM head）、`multi_graph_gdn_quality.py`（指标与门禁）、
`multi_graph_gdn_quality_analyze.py`（独立对齐与计分审计）。
官方入口：安装版 `qwen3_5.py:426` 和 `logits_processor.py:180`；原文件及 SHA 已保存。

证据目录：`benchmarks/results/multi_graph_structural/gdn-realtext-quality-v1`；
远端：`/workspace/gdn-quality-v1`。独立审计重新计算全部 token 对齐、PPL、bootstrap 及门禁，拒绝不完整轨迹或被改写的结论。
派生统计的跨平台复算只允许 1e-12 的浮点末位差异；门槛值与全部通过/失败标志仍要求完全一致。
质量逻辑 11 项 CPU 检查及 Ruff 已通过；全仓 `bash format.sh ci` 因既有 CRLF 脚本解析失败，不能宣称全仓格式检查通过。
'''
destination=repo/'docs/source/developer_guide/Design_Documents/aclgraph_gdn_quality_result_zh.md'
destination.write_text(text,encoding='utf-8')
print(destination,len(text))
