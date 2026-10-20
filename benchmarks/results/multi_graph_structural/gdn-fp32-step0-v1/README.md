# FP32 compact 第 0 步诊断证据

原 Qwen3.5-27B TP2、Q4 B98、默认 400-token 桶；BF16 激活与 FP32 SSM/日志。
两 rank 的完整起点审计通过，原 hidden 0.011816 数值失败复现。
第一层 recurrence 产生约 10⁻⁶ 的局部误差，随后经真实 norm、投影与 MLP 扩散；
两层同输入对照和一次首层输出替换没有显示明确有限的实现修复路径。
保持部署候选结束，不进入完整数值复验，不增加性能轮次或新候选。

中文说明见 `docs/source/developer_guide/Design_Documents/aclgraph_gdn_fp32_step0_zh.md`。

- `step0.rank0.json` / `step0.rank1.json`：每 rank 1408 条整网边界误差，以及起点、局部和一次替换证据。
- `arithmetic.json`：四份保存快照的离线 FP32/FP64 算术核对；FP64 不是新精度方案或接受门槛。
- `analysis.json` / `analysis-source.py`：证据门禁与收口结论，独立保留原 ready 调度收益。
- `measured-sources.tar.gz` / `measured-source-manifest.json`：156 项原始源码、实际安装源码及其哈希。
- `evidence-manifest.json` / `transport.json`：17 项原始证据及转运 SHA-256/字节数校验。
- `snapshot-manifest.json`：约 1.28 GB 的四份完整张量快照保留在远端，不提交巨大张量。
- `runtime-binaries.json`：绑定扩展及其 kernel 库、FLA 与 Python 库哈希，与既有核验相同。
- `validation.json` / `format-ci.log`：87 项 CPU 检查、Ruff 通过；全仓格式入口因既有 CRLF 在 hooks 前失败。

没有完整 128 步新审计、性能结果、新模型、FP16 排查或新的路由器。
官方与 compact 的两个首步整网控制、两层局部配对和一次替换均 exit 0，原 0.005 门槛未放宽。
