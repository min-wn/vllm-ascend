# FIA host update 并发最小实验

2026-10-17。承接混合长度实验后的候选，不扩大完整测试矩阵。

## 假设与实现

已有独立 update 流仍由主机顺序调用。若第二路图等待 host update，
让两个更新调用并行可能提前启动第二路 FIA。

新增 benchmark-only `--parallel-host-update`，要求 `--native-replay --independent-updates`。
切分的两份 NativeLane 分别由两个持久 C++ 工作线程执行；padding 保持单更新。
工作线程不调用 Python，不每步创建线程，各自在 inference mode 和对应 NPU stream guard 下调用 FIA。
输入 ready 依赖、实际 KV 长度绑定、所有 task update/event 和最终 stream join 均保留。
主线程等待两份更新提交返回后才继续操作计划，防止下一次 bind 与工作线程读参数并发。
工作线程在计划/Python 所有者销毁之前退出；设备完成仍由原有 event 和每步同步保证。
该实验只适用于现有 inference benchmark，不作为通用线程池或生产 runtime 接口。

## 最小判据

Qwen3-4B、默认 104 桶、56+48、B98/97/103/98、KV512。
比较相同两条 update 流下的主机串行和并行，避免把增加 stream 的变化混入效果。
每进程两轮，每轮 16 步，交换 padding/split 执行顺序。
执行并行、串行、并行复测三组，保留已有原 ND 精度与动态图复用资格检查。
trace 在计时后单独采集，用于检查实际设备 FIA 启动和主机调用。
没有明显净收益即停止扩大；不因微小变化跑完整回归矩阵。

## 复现

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_block.py \
  --model /models/Qwen3-4B --nz --native-update --native-replay \
  --shared-cache --static-task-plan --packed-metadata \
  --independent-updates --parallel-host-update \
  --rounds 2 --repeats 16 --trace --output /tmp/parallel-host.json
```

去掉 `--parallel-host-update` 为主机串行对照，独立 update 流保持不变。

## 最小实验结果

所有运行均通过原 ND 精度、真实长度 update 和动态图地址复用检查。
以下仅为两轮每轮 16 步的探索性结果，不作为稳定加速结论。

| 提交配置 | padding ms | 切分 ms | 切分相对 padding 增时 |
| --- | ---: | ---: | ---: |
| 默认队列，host 并行 | 1.157132 | 1.151857 | -0.46% |
| 默认队列，host 串行 | 1.127049 | 1.192567 | 5.81% |
| 默认队列，host 并行复测 | 1.131842 | 1.148152 | 1.44% |
| 关闭队列，host 并行 | 1.121567 | 1.110382 | -1.00% |
| 关闭队列，host 串行 | 1.139769 | 1.153842 | 1.23% |

明确发现：默认队列 trace 中，两路 FIA 的 Enqueue 位于两个不同的工作线程，
但两次 Dequeue 均位于同一个 torch_npu 线程。上层并发不等于底层处理并发。
这证明该 trace 中 dequeue 处理串行，不证明运行时所有工作都受同一全局锁限制。

安装版 OptionsManager.h 将 TASK_QUEUE_ENABLE=0 明确定义为 close，
因此追加两组进程级因果对照：在相同命令前设置 `TASK_QUEUE_ENABLE=0`。
它是现有 torch_npu 环境选项，不修改服务配置。关闭后的 trace 不再出现对应入队/出队事件。
默认队列的具体等级未单独读取，不能把未设置环境变量解释为已确认的某个等级。

设备 trace 中，默认队列 host 并行复测的两路 FIA 重叠约 141 us，
时长总和约 445.5 us，整体跨度约 309 us；同进程 padding 约 303.75 us。
关闭队列且 host 并行时，两路跨度约 314.75 us，padding 约 309.25 us。
更早供给增加了并发，但执行延长抵消了大部分重叠，没有明确的设备关键路径缩短。
计数均为独立 trace 样本的中位数，不用它们相加推导正式 wall 时间。

结论：保留 opt-in 实验开关，不启用为默认，不扩大线程池方案。
关闭队列的约 1% 信号未独立复测，不能作为切分已经胜过 padding 的依据。
后续应优先研究能减少实际 attention 工作或并发干扰的变化。

### 下一方向的约束：先核算 KV 搬运下界

核对实际 Qwen3-4B 配置为 32 query heads、8 KV heads、head_dim=128、BF16。
B98/KV512 的 K/V 逻辑数据量为
`98 * 512 * 8 * 128 * 2(K,V) * 2(bytes) = 205520896 bytes = 200704 KiB`。
此前 MemoryAccess 的单路 FIA AIC read-main-memory 约 205130 KB，
与一份逻辑 KV 的量级接近，另有 AIV 读取约 9027 KB。
计数口径不等同于实际 HBM 流量，不能据此证明最优；
但当前没有证据支持“GQA 把同一 KV 在主存重复读了四遍”这一大幅优化假设。
因此不直接启动 GQA kernel 重写。

后续先用匹配页面布局与工作集的访存诊断和可用硬件计数定位搬运等待层级，
确认 FIA 距离该机器实际可达到的带宽/延迟有多大差距。
只有明确存在空转或资源分配损失，再考虑共享任务调度、prefetch 或专用 tiling。
若已经接近搬运下界，同精度、同 KV 字节量下简单增加分支不会产生大幅收益。

## 验证与归档

本轮不跑完整回归矩阵。五组 4B 最小实验均含现有资格检查，所有结果通过。
增加独立 NPU 回归入口和单 lane 拒绝检查供后续保留/合并时使用，本轮未另外运行这些 pytest 项。
Ruff、C++ 格式和 diff 检查执行；原始 JSON、trace、日志、被测源码和汇总位于
`outputs/multi_graph_host_parallel/`。未修改生产 runner。
按仓库要求执行格式 CI；仍有既有 Windows 工具缺失、python3 hook 启动和 Triton 导入失败，
不宣称全仓 CI 通过。本轮未运行完整 CPU/NPU 回归矩阵。
