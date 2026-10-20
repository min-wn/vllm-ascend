# ReplaySSM 最小 A2 对照原始证据

日期：2026-10-20。真实层为 Qwen3.5-27B TP2 的第 4 层（从 0 编号），不是完整模型或服务。

核心结论：FP32 日志变体六轮闭合墙钟相对同算法 delayed 改善 4.30%～7.40%；
原社区 FP16 日志配置未过精度门槛；当前 ring ready 比 compact ready 更慢。

| 文件 | 含义 |
| --- | --- |
| `lifecycle-v1.json`、`lifecycle-v2.json` | FP32 日志变体，两个独立进程，每个三轮；各含逐步精度、两 rank 数据和完整闭合成本。 |
| `layer.log`、`layer2.log` | 对应上述两次正式运行的原始日志。 |
| `qualification.json`、`probe.log` | FP32 独立最小算子，40 步图 replay 与逐 token 物化参照；含终末接受态。 |
| `fp16-attempt.tar.gz` | 原 FP16 配置的源码、真实层失败、算子资格和两步诊断，按原字节归档。 |
| `diagnostic.json`、`diagnostic.log` | **FP16 配置**的两步诊断，不计性能，不是 FP32 结果。 |
| `fp32-measured-sources.tar.gz` | 被测 FP32 源码，包含相关既有 benchmark 依赖与离线 tiling；不随格式化改写。 |
| `measured-source-manifest.json` | 上述被测源码的逐文件 SHA；真实层 JSON 内源 SHA 可与之核验。 |
| `deployed-sources.json` | 测量时上传的 Python 源码 SHA。 |
| `environment.json` | 安装 revision、实际库及模型配置 SHA、包版本和运行参数。安装工作树保留既有修改。 |
| `evidence-manifest.json` | 远程取回的 13 个原始文件的大小与 SHA，分析器会检查。 |
| `fp16-archive-members.json` | FP16 原始归档成员清单。 |
| `analysis.json` | 从两份原始生命周期 JSON 复算的派生结果。 |
| `validation.json` | 本地针对性检查及当前提交源码与被测源码的关系。 |

原始文件的校验基于 Git blob/LF 字节。Windows 若自动转成 CRLF，复算前应保留原始换行。
报告链接：[机制、差异及主线资格](../../../../docs/source/developer_guide/Design_Documents/aclgraph_replayssm_a2_review_zh.md)。

在仓库根目录复算：

```bash
python benchmarks/multi_graph_replayssm_analyze.py \
  benchmarks/results/multi_graph_structural/replayssm-a2-v1
```

`analysis.json` 只证明所声明的 FP32 A2 变体和固定轨迹。
每个进程/rank 的 import 仅一次观测，不代表稳定初始化分布；不推断 P99 或文本质量。
