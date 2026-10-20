# 精确规划收紧可行上界：原始证据

日期：2026-10-20。结论：上界收紧有作用，但无生产者净收益，结束当前共享计划候选。
没有接 16 次消费链、不扩完整模型、不增加路由器、不转为近似分区。

- `cpu-audit.json`：16,221 个穷举组合、128 步默认容量轨迹和七代设备输入的可行性、
  精确最优值及规范化恢复验证。内含测量前声明的稳定收益门槛。
- `producer-result.json`：同进程 AICPU、原 AIV、收紧上界 AIV 的七代全 1024 字资格、
  固定地址/输入只读检查、上界诊断、120 个原始样本及计时后全字段检查。
  对应当前三个 descriptor kernel/probe 源码；源文件和二进制 SHA 由审计核对。
- `producer-profile-result.json`、`profile/`：每模式仅一次诊断 replay，不含正式端点样本。
  AICPU 清零/计划核为 2.30/104.34 μs，原 AIV 303.23 μs，收紧上界 AIV 244.86 μs。
  三份长度 DMA 约 2.5 μs；不要重复累加依赖等待，也不要用受扰动核时长替换无 profiler 端点。
- `producer-audit.json`：哈希、样本完整性、七代诊断与 CPU 一致性、四轮设备/wall 门槛、
  同一历史共享链内的乐观预算比例及候选结束状态。
- `validation.json`、`format-ci.log`：限定检查与完整格式脚本的环境阻断。

本轮设备端点中位数：AICPU 115.49 μs、原 AIV 186.20 μs、新 AIV 163.05 μs。
对应 wall 中位数：297.02、371.18、347.75 μs。
新 AIV 较原 AIV 少约 12.43%，但较 AICPU 仍慢约 41.18%，四轮均退化。

上界在固定核内构造，不在 CPU 预计算；必要 KV 长度 H2D 计入两类端点。
图外 oracle 和插桩版本只用于资格，不作为正式核的输入或计时样本。
保留旧精确目标与恢复，不直接导出近似前缀分区。

固定设备资格为物理 100 请求/400 token、Q4/GQA6/KV2/D256/page128、20/40 核、KV0..4097。
七代输入复用上一轮 `dynamic-descriptor-v1/plan-cases.json`，按记录的 SHA 校验，不覆盖原文件。
只用原 step0、step6 的变化长度计时，全零或全同成本输入没有用于报告收益。

复算：

```bash
python benchmarks/aclgraph_descriptor_bound_audit.py \
  --producer benchmarks/results/multi_graph_structural/dynamic-descriptor-bound-v1/producer-result.json
python -m pytest --noconftest -q tests/ut/worker/multi_graph/test_descriptor_bound.py
```

远端隔离目录为 `/workspace/aclgraph-descriptor-bound-v1/`；
已资格化 vendor 与 extension 沿用前一轮 README 中的路径和三项强制 SHA 校验。
远端启动参数与前一轮一致，仅替换脚本为 `aclgraph_descriptor_bound_probe.py`。
加 `--profile-dir` 时只诊断，不执行正式端点测量。
基线是本分支已资格化的完整头导出，不能将结果解释为通用官方 FIA 的性能结论。
