# 完整 Decoder 的模型规模与多卡实验

## 测量范围

`benchmarks/multi_graph_model_scaling.py` 只计时 padding 和切分并行。
切分采用共享 Dense、每层两路 Attention、逐层汇合的外层图。
包含 embedding、所有 decoder 层和最终 norm，不包含 LM head、采样、draft、调度器。
Q=8 表示每请求一次处理 8 个验证 token，不表示运行了 draft model。
长度固定，但每次 replay 都执行真实 host seq_len 所需的 FIA task update。

padding 按 C×Q 执行，真实请求数为 B；假请求的 KV 长度为 0，slot 为 -1。
切分按真实 B×Q 执行共享 Dense，在完整请求边界分为两路 Attention。
两路 workspace 隔离，跨层汇合后复用；所有 TP collective 留在共享计算路径，
两条 Attention 流不各自发起 collective。

实验环境：Ascend910B3/A2，64 GB/卡，驱动 24.1.rc3，CANN 9.1，
torch 2.10.0、torch-npu 2.10.0.post4、vLLM 0.30.0、插件 0.30.0rc1。
模型为现成 BF16 权重，普通 FIA，未启用 batch-invariant。
五轮，每轮三十次，每轮交换两个模式的顺序；结果取各轮 wall p50 的中位数。
时间包括 host 提交、update、事件、计算和 TP 通信；排除加载、prefill、capture、
KV 恢复和 profiler。TP 每个样本取各 rank 的 wall time 最大值，计时前 CPU barrier 不计入。

## 结果

以下数据为完整 decoder 实测，不能与此前单层耗时直接比较。

| 模型 | TP | B / C / Q / KV | lanes | padding ms | 切分并行 ms | 延迟下降 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 129 / 256 / 8 / 512 | 65+64 | 40.176 | 27.581 | 31.35% |
| Qwen3-4B | 1 | 同上 | 65+64 | 105.723 | 65.237 | 38.29% |
| Qwen3-8B | 1 | 同上 | 65+64 | 163.187 | 94.331 | 42.19% |
| Qwen3-8B | 2 | 同上 | 65+64 | 156.961 | 88.703 | 43.49% |
| Qwen3-8B | 4，确定性 HCCL | 同上 | 65+64 | 95.105 | 57.713 | 39.32% |
| Qwen3-4B | 1 | 65 / 128 / 4 / 512 | 33+32 | 40.156 | 29.800 | 25.79% |
| Qwen3-8B | 1 | 同上 | 33+32 | 60.139 | 46.277 | 23.05% |

结论边界：

- 大工作量下，模型增大与更大的收益同时出现；共享 Dense 省去假 token 的计算是重要因素。
  不应把这一收益全部归因于两路 FIA kernel 并发，当前实验没有采集证明重叠的 trace。
- 中等工作量下，8B 的相对降幅小于 4B，不能建立“模型越大，收益必然越大”的规律。
- 8B 双卡相比单卡的切分延迟只下降约 6.0%，padding 约 3.8%。
  当前设置没有表现出接近线性的 TP 加速；需进一步分解每卡矩阵和通信开销，不能只归因于通信。
- padding 容量几乎是真实 batch 的两倍，属于 padding 浪费较多的场景。
  更密集的 capture 容量表、B 更接近 C 时，结果可能变化。
- 没有改变真实长度更新要求，没有以取消 update 获得这些数字。

每个模式按各自结构的 eager 对照检查，另将未切分的重写 decoder 与官方 decoder 比较。
改变输入 token 后重新生成 eager 参考，再检查 replay，覆盖捕获存储与中间值更新。
以上通过的七组实验全部同形状检查最大误差为 0。
切分 eager 与同 batch 官方输出在这些样本中也为 0；padding 与真实 batch 的官方输出存在
跨形状数值差异。因此输出仍明确标记 `cross_architecture_qualified=false`，
不能把这些测试解释为完整模型质量、logits 或跨形状精度验收。

## 四卡数值定位

Qwen3-8B TP4 首次实验在计时前的官方 decoder 对照检查失败，未输出有效性能结果。
追加逐层探测发现官方 eager 自身重复执行也有最大绝对差 1.5。
此证据排除了“只有重写 decoder 才出现差异”的解释，但尚不能单凭它证明切分正确。
使用 `HCCL_DETERMINISTIC=true` 后，官方重复差异为 0，逐层与完整 decoder 对照通过。
随后正式脚本完成五轮测量，两种模式全部 replay 和 changed-input 检查通过。
四卡结果为 padding 95.105 ms、切分 57.713 ms，降幅 39.32%。
这仍使用普通 FIA，没有启用 batch-invariant，但通信设置与单/双卡默认测试不同；
不能将四卡与其他卡数间的差异全部归因于卡数。

## 较新模型的真实兼容性探测

`benchmarks/multi_graph_model_probe.py` 仅用于官方路径的基础兼容性，不计入性能表。

| 模型与配置 | 实测结果 | 能证明的范围 |
| --- | --- | --- |
| Qwen3.5-9B，单卡 eager | 加载并完成两条文本生成 | 当前 A2 可执行该模型 |
| Qwen3.5-27B，四卡 eager | 加载并完成两条文本生成 | 当前 A2 四卡可执行 27B，TP 通信可用 |
| Qwen3.5-9B，官方 FULL_DECODE_ONLY | capture 后首次 decode replay 失败 | 当前配置尚不具备图性能比较条件 |

Qwen3.5-9B 的图错误在 `aclnnFusedInferAttentionScoreV3` update 内触发：
`Slice` / `aclnnContiguous`，错误码 207019，运行时明确报告
`current driver version does not support to update this op`。
此前 dense 模型使用连续 KV 存储绕过相同类别的问题；Qwen3.5 是 GDN/full-attention 混合结构，
具有多个 KV/state group 与 padded page，现有连续 KV 方案明确不支持这种布局。
不能解除 guard 后直接认为适用，也不能用 eager 的时间冒充 padding 图时间。

启动日志另提示 `causal_conv1d_update_npu` 导入失败，使用带逐请求同步的 PyTorch fallback。
这是需要独立检查的 GDN 图执行限制；本次实际失败栈指向 FIA/Slice update，
不能把两个问题混为一项已证实的根因。

机器上的插件文档已列出 Qwen3.5/3.6-27B 的 A2/A3 支持，
但支持矩阵不保证旧驱动与当前软件组合的所有图模式都通过。
本轮实际加载的是 Qwen3.5 权重，没有下载或测试 Qwen3.6，亦没有获得 Qwen3.5 切分性能数字。

## 复现

```bash
source /etc/profile.d/ascend-multigraph.sh
VLLM_BATCH_INVARIANT=0 ASCEND_RT_VISIBLE_DEVICES=0 \
python benchmarks/multi_graph_model_scaling.py \
  --model /models/Qwen3-8B --batch 129 --capacity 256 --query 8 --kv-length 512 \
  --rounds 5 --repeats 30 --output /workspace/qwen3-8b-tp1.json

VLLM_BATCH_INVARIANT=0 ASCEND_RT_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/multi_graph_model_scaling.py \
  --model /models/Qwen3-8B --tensor-parallel-size 2 \
  --batch 129 --capacity 256 --query 8 --kv-length 512 \
  --rounds 5 --repeats 30 --output /workspace/qwen3-8b-tp2.json

VLLM_BATCH_INVARIANT=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_DETERMINISTIC=true \
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/multi_graph_model_scaling.py \
  --model /models/Qwen3-8B --tensor-parallel-size 4 \
  --batch 129 --capacity 256 --query 8 --kv-length 512 \
  --rounds 5 --repeats 30 --output /workspace/qwen3-8b-tp4-deterministic.json

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python benchmarks/multi_graph_model_probe.py \
  --model /models/Qwen3.5-27B-original --tensor-parallel-size 4 \
  --output /workspace/qwen35-27b-probe.json

ASCEND_RT_VISIBLE_DEVICES=0 python tools/run_multi_graph_replay_tests.py \
  --replay-model /models/Qwen3-0.6B -k full_decoder
```

新增的 eager 连续 KV 准入仅允许 dense Qwen3 和 TP 布局资格测试。
生产 `multi_graph_config.enabled=true` 对 Qwen3、TP、混合架构的拒绝仍保留。
322 项 CPU 测试通过；3 项硬件回归通过，覆盖 Qwen3-0.6B、Qwen2.5-0.5B 完整 decoder
以及既有普通 FIA 的 Qwen2 整模型测试。
Ruff 与修改文件禁止导入检查通过。完整 `format.sh ci` 的现有 Windows 工具缺失与
既有 `device_length_decode.py` 禁止导入失败仍存在，不能记为全量 CI 通过。

原始报告与日志保存在工作区 `outputs/multi_graph_model_scaling/`；
`summary.json` 汇总各轮 p50，`provenance.json` 保存源码哈希及通信环境。
七组正式测试脚本 SHA256 均为
`cb89e79dbdec2ad0cc2fa1ac92f9c3edee81d4b62a55538c602fa1f2dc734af1`，
与本地源码和容器部署文件一致。容器源码采用文件部署，因此报告中的远端 Git HEAD
并不代表已部署文件的完整内容，复核以哈希及本次本地提交为准。
