# GDN ready 真实激活动态范围证据

日期：2026-10-20。冻结协议 SHA：
`e952a78fd7d8d977de9c6ff67d17cb020719fb54a05d926b64be0b1bdb50a697`。

- `ranges-v1.json`、`ranges-v2.json`：两个独立进程的资格、激活 SHA、精度、原始时间样本。
- `ranges-v1.log`、`ranges-v2.log`：原始运行日志。
- `frozen-protocol.json`：观察结果前固定的三档输入及比较契约。
- `measured-sources.tar.gz`、`measured-source-manifest.json`：实际上传的 Python、FP16 诊断源码及 FP32 ring 核原字节。
- `environment.json`：本机版本、模型索引、本机 GDN、实际扩展二进制 SHA；不把单独本地 Git HEAD 当作完整环境。
- `evidence-manifest.json`：所有原始证据的长度和 SHA；不含后生成的分析、文档和质量检查。
- `analysis.json`：从两进程原始证据复算出的配对结果，低负载退化保持在结果中。
- `fp16-first-failure/attribution.json`：只针对旧首失败样本的阶段归因；不提供 FP16 性能结果。

原始取回包 SHA：`eb51c344be10c5173c2a1866ce918ba4bbd7f4482159fd9ce124c8be61d06297`。
被测 ready 协议、激活与探针源码在归档中的 SHA 与原始结果一致；分析器、报告和整网审计在采样后补充。
既有 compact 扩展源码见仓库对应 `multi_graph_gdn_q4_*_kernel` 目录和前轮证据；本轮记录实际二进制 SHA。

范围限制：三档都使用同一个 400 token 图；真实上游激活但只计第 4 层；无服务、文本质量、全负载稳定加速或切分独有收益声明。
详细中文解释见
[GDN ready 动态范围验证](../../../../docs/source/developer_guide/Design_Documents/aclgraph_gdn_ready_ranges_zh.md)。

在仓库根复算：

```bash
python benchmarks/multi_graph_gdn_ready_range_analyze.py \
  benchmarks/results/multi_graph_structural/gdn-ready-range-v1
```
