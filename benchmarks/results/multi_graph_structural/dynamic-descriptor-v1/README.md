# 固定 ACLGraph 动态描述符候选筛选：证据索引

日期：2026-10-20。无路由器、无完整模型扩展、无驱动/镜像更换。

## 结论与范围

三候选复用原始 Q4 128 步轨迹及默认容量。dense 无效行仅占 1.41256%；
动态 GMM 在本轮完整输出契约下没有净收益。通算融合的必要 AG/归约不能自动删除。
选定 attention 共享计划生产者；AIV 原型虽保持全部原生导出，却比 AICPU 更慢，停止扩大。

实测设备为远端容器的 A2/910B3，驱动 24.1.rc3。
软件版本以每份结果的 `versions` 为准；没有调用生产模型路径。
生产者只替换元数据生成，不包含 attention 的 Q/K/V 计算。

## 有效报告

- `dense-result.json`：社区 GMM 的七代有效行数资格和 120 个端点样本。
  对应字节源码 `dense-probe-success.py`。392/400 的性能比较中，
  动态完整输出 259.80 μs，社区完整容量 GMM 214.74 μs。
- `plan-cases.json`：两代原轨迹及五代边界/恢复情况，固定 100 请求、400 token。
  `aclgraph_descriptor_audit.py` 重建其语义，但不会覆盖已 hash 的输入文件。
- `plan-result-v4.json`：首次选对 fresh whole 基线后，七代完整 1024 字资格与 80 个样本。
  对应 `plan-probe-v4.py`、`plan-kernel-v2.py`。AICPU/AIV 设备中位数 125.42/199.42 μs。
- `plan-result-v5.json`：补齐计时后全字段检查，保留同一算法，80 个样本。
  对应当前 `benchmarks/aclgraph_descriptor_plan_probe.py` 与 kernel。
  AICPU/AIV 设备中位数 126.29/199.25 μs，wall 中位数 315.45/387.88 μs。
  两次运行每轮均为负，不选择最快样本。
- `plan-profile-result.json`、`profile/`：每模式一次诊断 replay，非性能端点。
  对应 `plan-probe-profile.py`、`plan-kernel-v2.py`。
  新核内部工作显著，不能把 304.15 μs 的受扰动核时长替换为无 profiler 的端点。
- `inventory.json`：本机相关 schema/doc 清单，支持接口和契约核查。
- `audit.json`：复算成本、16221 个 CPU 计划等价、样本完整性、全部源码/库 hash 及任务清单。
- `validation.json`、`format-ci.log`：限定检查结果与完整格式脚本的环境阻断记录。

源码快照按原始字节保存。当前 benchmark 按 LF 保存；dense 源码只规范换行，
审计同时核对归一化语义源码，避免把换行差异误当成机制变化。
`plan-kernel-v2.py` 与当前 kernel 也仅存在换行差异。

## 无效尝试，不能纳入性能统计

- `dense-result-initial-failed.json`、`dense-probe-initial.py`：首次 Matmul NZ 基线格式构造错误。
  三维父张量的二维 view 不适合直接转换为 NZ，修正 clone 后另起进程成功。
- `plan-result-initial-failed.json`、`plan-kernel-initial.py`：安装版后端不支持 int32 gather。
  受限整数域改用精确数值 FP32 转换后通过，不放宽整数前缀和比较。
- `plan-result-abi-failed.json`、`plan-result-diagnostic.json`、`plan-probe-initial.py`：
  探针误选旧 vendor，旧库 FD 字段不符合 fresh whole 契约。这不是已确认算法错误，
  也没有有效性能数据。v4/v5 强制匹配三个已资格化二进制 hash 并重新全字段验证。

正确 vendor：
`/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e/stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3`。
误选旧 AICPU SHA 为 `46bafeb84baa095b84063a2d0c262f04aef1fd5d8ba51b975ca7d44a618078f2`，
正确 AICPU SHA 为 `1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90`。
公共 OpAPI 与 extension 的 SHA 由报告和审计共同检查。

## 复算与远端命令

仓库根目录 CPU 复算：

```bash
python benchmarks/aclgraph_descriptor_audit.py
python -m pytest --noconftest -q tests/ut/worker/multi_graph/test_descriptor_audit.py
```

远端容器，先加载既有软件环境，脚本和 cases 放在本轮隔离目录：

```bash
source /etc/profile.d/ascend-multigraph.sh
python /workspace/aclgraph-descriptor-selection-v1/aclgraph_descriptor_plan_probe.py \
  --cases /workspace/aclgraph-descriptor-selection-v1/plan-cases.json \
  --output /workspace/aclgraph-descriptor-selection-v1/new-result.json \
  --opp /workspace/aclgraph-sink-stage2-d256-bec19b8dea1e/stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3 \
  --extension /workspace/aclgraph-sink-stage2-d256-bec19b8dea1e/extension-build/thesis_fia_sink_bec19b8dea1e.so
```

追加 `--profile-dir /workspace/aclgraph-descriptor-selection-v1/new-profile` 时只诊断两次 replay，
跳过端点计时，避免把带 profiler 的数据混进性能报告。
基线是本分支已资格化的完整头生产者，非未经修改的官方全功能 FIA。
设备描述符、plan/run、元数据 AIV 生成本身已有社区工作，不能单独作为原创性结论。
