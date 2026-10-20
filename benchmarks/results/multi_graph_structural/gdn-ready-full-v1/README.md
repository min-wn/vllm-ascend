# GDN ready 整网闭合核验：原始证据

Qwen3.5-27B / TP2 / 910B3，原 Q4 128 步与默认图目录。
同算法完整状态审计通过；独立官方数值筛选未通过。
ready 相对 compact delayed 的三对闭合成本改善均值 4.801%，
3/3 对为正。不是相对官方的性能改善，也不是完整服务吞吐。

| 目录/文件 | 内容 |
| --- | --- |
| `replay/` | remote full-v5：34 图，继承完整 delayed / 新审计全部 ready，合计 512 条资格记录与先行计时。 |
| `replay-delayed-source/` | remote full-v2：delayed 全部 128 步；ready 21 步后诊断线程崩溃，后者不拼接。 |
| `timing/` | remote full-v6：隔离直接转换、终末 CPU 决策上传和三对计时；复用已验证逐步资格。 |
| `numeric/` | remote numeric-v3：独立官方/compact 连续历史与终末状态；诊断修复后的完整记录。 |
| `numeric-incomplete-sigsegv/` | remote numeric-v1：124 步后 native 崩溃，未完成、未拼接。 |
| `numeric-stopped-watchdog/` | remote numeric-v2：识别诊断风险后主动停止，未完成、未拼接。 |
| `native-diagnostics/` | 实际 PC/模块映射、addr2line 符号化、core SHA 与截断限制。 |
| `incomplete-original/` | CPU 扫描较慢而主动结束的旧审计；不计为通过。 |
| `timing-wait-retired/` | 尚未加载模型的旧等待控制；旧数值路径崩溃后被替换。 |
| `measured-sources.tar.gz` | 每次尝试实际部署的 Python、四扩展 C++/头文件、安装版路径与模型配置。 |
| `measured-source-manifest.json` | 压缩包内每个源文件 SHA256。 |
| `environment.json` | 包版本、安装提交、硬件、运行选项和实际二进制 SHA。 |
| `local-provenance.json` | 本地收尾提交的父提交、功能分支及另列的被测安装版提交。 |
| `evidence-manifest.json` | 远程收集时全部原始文件 SHA256 与大小；原字节保留。 |
| `analysis.json` | 完整性、来源校验、可相加 TP 成本和独立数值误差；分析源码 SHA 随附。 |
| `analysis-source.py` | 生成本次分析时的源码原字节，与 `analysis_source_sha256` 对应。 |
| `generated-manifest.json` | 本地生成的分析、说明、验证和来源文件 SHA256。 |
| `validation.json` / `format-ci.log` | 相关 CPU / Ruff / diff 检查结果及全仓格式入口失败原日志。 |

全部路径以本目录为基准。性能主表仅使用 `timing/`，没有把原审计进程的先行计时视为额外独立复现。
执行、一次直接转换、终末物化三列均取同一闭合关键 rank；逐步 TP max 不求和代替闭合端点。
官方 CPU 状态库只用于独立数值核验，没有官方性能基线。

在仓库根目录复算：

```bash
python benchmarks/multi_graph_gdn_ready_full_analyze.py \
  benchmarks/results/multi_graph_structural/gdn-ready-full-v1 \
  --output benchmarks/results/multi_graph_structural/gdn-ready-full-v1/analysis.json
```

详细中文机制、成本、社区差异及结束边界见
[整网核验文档](../../../../docs/source/developer_guide/Design_Documents/aclgraph_gdn_ready_full_zh.md)。
FP16 排查保持结束，没有新增精度方案、路由器或模型。
