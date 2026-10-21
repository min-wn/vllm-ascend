# GDN 请求组 ready：离线真实自回归生成冻结协议

本轮只比较同一 compact FP32 算法的 delayed/ready 调度。保留原 BF16 head、Q4、AIC/AIV 资源、默认图目录与原全部数值、质量失败。上一轮 beta/gate 复用候选已拒绝，生产内核冻结，不再进行内核或精度搜索。

## 连接范围与工作负载

复用 `GenerationRequest`、原 vLLM ngram proposer、`FullMetadataFactory`、`UpdatableGraph` 和现有 deferred commit。原完整回放分配器的 128 owner 限制是 fixture 守卫；增加显式 `expected_owners` 参数，原默认值仍为 128。本轮声明 56 owner，底层既有 1..128 owner 状态 ABI 不变，没有改 recurrence 或规约算术。

固定 56 个请求，位于已有中负载 54..60 范围。复用已有范围实验的四条自然语言任务 prompt，每条重复 14 次，附相同的简洁回答说明；没有随机词表输入，也没有数学解题。冻结本机 tokenizer、chat template 和 token IDs，关闭 thinking，greedy，正常 EOS，每请求最多输出 256 token。输入长度和每 owner 的 KV 上界在首个模型运行前记录到 `workload.json`。上下文上限保持 2048，额外预留四个 speculative query 的写入尾部。

四种 prompt 只提供一个有限、可复现的性能批次，不代表广泛的请求分布；prefix caching 关闭。所有可能的活跃请求数 1..56 只用本机默认 dispatcher 决定 bucket，捕获相应默认容量，不增加 bucket 或路由器。运行时接受长度完全来自本请求的真实 draft 与模型预测；无 draft 时只填充 Q4，不能接受填充 token。

## 正确性与计时

三对独立 TP2 进程，顺序 delayed→ready、ready→delayed、delayed→ready。每进程独立零态 prefill，不从另一侧注入状态。捕获后，先从自身 prefill 快照运行一个不计时的完整生成验证轮，再恢复同一自身快照做正式生成。

验证轮记录逐步 token、draft、预测、实际接受长度、owner、位置、bucket、结束原因和 hidden 字节摘要，核对终末 SSM、卷积和 FIA 状态。正式轮只保留生成所需的 CPU ngram、metadata/token H2D、FIA host 长度更新、图执行、BF16 head、TP、argmax/broadcast/D2H、接受判断、提交和完成等待；详细张量下载、有限性检查和 JSON 落盘在计时外。内存中的逐步轻量 ledger 及计时调用保留在正式计时中，属于本入口的共同开销。

分别测量真实 prefill（含首个 head、首 token 回传）、一次 compact 导入、生成循环、真实最后接受前缀的终末提交。批次成本定义为这四个实测阶段之和，捕获和验证造成的阶段间间隔剔除，**不是从 prompt 到结束的一段连续 wall clock**。模型加载、缓存与扩展准备、捕获另列；恢复和审计不计入性能。prefill 是新进程中的一次实测，生成则经过自身验证轮预热，两者的缓存条件明确分开。

生成循环输出吞吐分子是该循环实际输出的 token（不含 prefill 产生的 56 个首 token）；完整批次吞吐分子包含这 56 个 token。EOS 属于输出 token，未输出的 Q4 候选不计吞吐。分别报告输出 token、步数、接受分布和三个独立进程对的波动，不把验证步时延当单 token 时延。

每对完成后先严格核对两侧实际工作和状态；不一致立即按同算法调度正确性问题停止，不能报告不同工作量的性能收益。稳定收益的预设判据为：三对生成循环耗时均改善，合计分母计算的均值改善超过 1%，且超过两侧重复测量相对跨度的两倍。该标准用于决定是否值得规划 runner 接入，不表示质量或部署资格。

若收益缩小或消失，报告 draft、host 上传与提交、模型回放等待、预测回传及接受判断在闭环中的占比。阶段计时反映带等待的 host 边界，异步设备成本不能跨阶段独占归因。不返回内核枚举、不改变停止条件、不扩大工作负载。
