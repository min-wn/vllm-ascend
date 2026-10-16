# 多图切分：原生 FIA 提交与共享默认桶投影实验

## 问题与对照边界

本轮验证两个假设：把逐层 FIA update 的 Python 提交改成一次 C++ 批量提交，
是否能够降低完整 decoder replay 时延；把两个切分分支的 Dense 投影汇集到
同一个默认容量图，是否值得进一步实现为完整 decoder。

硬件为单张 910B3，驱动 24.1.rc3、CANN 9.1，torch 2.10.0 + torch_npu，
vLLM 0.30 / Ascend 0.30rc1 环境；模型为 Qwen3-4B BF16，普通 FIA，Q1、KV512。
vLLM 基础提交为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
沿用随机选出的 B98，padding 使用默认 104 桶，切分使用默认 56+48 桶，
真实请求分别为 50+48。没有引入 49、98 等额外图，没有按 attention head 切分。

完整 decoder 对照两边均使用四类投影 NZ 权重、pinned 输入和逐层交错更新。
每张图有 36 个 FIA 更新任务，切分仍提交全部 72 个任务，不省略真实长度更新。
正式计时不打开 profiler，采用五轮、每轮每模式 30 次 replay，轮间反转模式顺序。
表格统计量为各轮 p50 的中位数。每次计时包含 CPU metadata 准备、H2D、任务解析、
全部 FIA 更新、完整 decoder 图执行、输出合并及最终设备同步。

每份完整 decoder 报告先检查 B98/KV512 → B97/KV511 → B98/KV512 的六次回放，
保持图、输入及输出缓冲区身份；与原始 ND 权重的同 shape eager 参考比较，
容差保持 `rtol=0.02, atol=0.02`。这些检查不认证跨 shape 数值一致性或生产 serving。

## 原生提交实现

`benchmarks/multi_graph_native_update.cpp` 是可选的基准扩展。
Python 端仍按每步真实 metadata 调用 `resolve_tasks`，并检查任务只包含 FIA v1 `.out`，
动态参数只包含 `actual_seq_lengths`、`actual_seq_lengths_kv` 和 `block_table`。
扩展初始化时转换固定参数，每步重新绑定动态参数；同一步内相同长度对象的转换可复用。
绑定检查任务数、事件、handle、固定参数对象身份及参数集合。

`submit` 释放 GIL，以 C++ 调用 graph task update begin/end、dispatcher FIA `.out`
以及 NPU event record。两条 lane 逐层交错，保留每图的任务顺序和依赖事件。
没有从 C++ 回调 Python `task.apply`，也没有将长度转换或参数绑定移出完整 replay 计时。
padding 使用同一原生优化，避免只优化切分产生不公平对照。

扩展按源文件 SHA256 命名，针对已安装 torch_npu 的头文件和共享库编译。
事件桥接依赖安装头文件中的 `THNPEvent` 布局并持有事件所有者，不复制或销毁原事件。
这有明确的 torch_npu ABI 依赖，不能视为稳定的跨版本接口。
固定参数检查只适用于基准中不可变的捕获参数，不是可处理任意原位参数变更、异常恢复
或并发重绑定的通用执行器。该扩展不接入生产服务路径。

首版逐步重新转换全部参数，虽然缩短 update 提交区间，却明显增加 prepare 开销。
第二版缓存固定参数，并单独验证扩展编译和加载后进行完整复测。
首版探索数据、第二版编译失败日志及修复后的测量均分别保留，不能混作同源码对照。

## 主机区间的含义

`--host-timing` 在计时起点、准备后、图提交后、更新后、收尾后和最终同步后取六个时间戳，
形成 `prepare`、`submit`、`update`、`finish`、`synchronize` 五个互斥主机区间。
逐样本检查五段之和等于完整 wall time；另以关闭区间计时的独立进程作对照。

设备执行可与这些主机区间重叠。因此 update 区间变短并不等于设备关键路径同量缩短；
同步等待变长也不意味着硬件本身变慢。应以完整 replay 对照判断是否获益。
各阶段分别计算的中位数也不保证相加等于总时间的中位数。

## 完整 decoder 结果

单位 ms；所有行均使用同一最终 benchmark 源码。

| 提交方式 | 主机分段计时 | padding 104 | split 56+48 |
| --- | --- | ---: | ---: |
| Python | 关闭 | 25.257 | 28.928 |
| C++ 缓存参数 | 关闭 | 25.311 | 29.439 |
| C++ 缓存参数独立复测 | 关闭 | 25.355 | 29.228 |
| Python | 开启 | 25.284 | 29.100 |
| C++ 缓存参数 | 开启 | 25.373 | 29.564 |

关闭分段计时的 Python 切分比 padding 慢约 14.53%；原生两次分别慢约 16.31% 和 15.28%。
原生切分相对 Python 基线的绝对时延分别增加约 1.77% 和 1.04%。
本轮结果不支持将原生提交作为当前场景的性能优化上线。

带分段计时的主机区间如下，单位 ms。这些是 CPU wall intervals，不是设备独占耗时。

| 模式 | prepare | submit | update | finish | synchronize |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python padding | 0.803 | 0.166 | 13.254 | 0.040 | 11.003 |
| 原生 padding | 1.152 | 0.178 | 4.018 | 0.049 | 19.995 |
| Python split | 1.238 | 0.273 | 21.774 | 0.111 | 5.601 |
| 原生 split | 1.876 | 0.284 | 6.837 | 0.129 | 20.413 |

切分的 update 主机区间减少约 68.6%，但最终同步等待增加，prepare 也增加约 0.64 ms。
这表明直接把主机 update 的 21.77 ms 当成可消除的整步时间是不成立的。
该实验没有移除设备侧任务、update 机制或双图重复投影，不能据此否定改变这些结构的潜力。

首版全参数转换的探索结果为 padding 29.126 / split 34.005 ms（关闭分段计时）；
它说明原生化必须计入数据绑定成本。这不是与最终版同源码的正式对照。

## 共享 Dense 投影诊断

`benchmarks/multi_graph_dense_probe.py --shared-input-probe` 使用模型各层真实投影权重，
独立合成输入隔离矩阵计算。四类投影分别遍历 36 层不同权重，避免反复只读一层权重。
原始 ND 权重的 eager 结果用于精度对照。

- `padding`：默认 104 容量的投影图。
- `split_parallel`：默认 56 和 48 容量两图并行。
- `shared_bucket`：两次输入拷贝汇集到 104 buffer，复用 padding 图执行 36 个投影，
  以预建 view 提供两份输出。计时包含拷贝、事件依赖、图执行及同步。
- `shared_staged`：每层分别使用一个 104 容量图，每层都付出两次输入拷贝、ready 事件、
  一次图启动、done 事件及主流等待的成本。一类投影共 36 张默认容量图。

逐层图在同一条流上按捕获顺序执行，共享一个图池并保留所有输出。
初次每层独立图池的版本在 `AclrtReserveMemAddress` 失败，错误码 207001，未产生有效计时；
共享图池版本单独留存为 `shared-staged-nz-v2.json`，不覆盖失败报告。

`shared_bucket` 每次只汇集一次，却执行 36 个权重，明显摊薄了逐层汇集开销；
`shared_staged` 补充测量直接逐层编排的代价。二者均没有真实 attention 和层间隐藏态依赖，
没有跳过 padding 行，也不代表新的 serving 执行模式。
这里汇集的分界为容量 56，完整 decoder 的真实请求分界为 50，不能直接照搬这个张量布局。
共享输出 view 通过精度检查，但不能据此推断 FIA 可零拷贝消费非连续视图。

该诊断仅用于判断后续结构调整的成本，不将四类投影的收益相加并称作 decoder 收益。
本轮没有硬件带宽计数器，也不从投影差异推断已经证明 HBM 饱和。

共享图池版结果如下，单位 ms，每列均为完整 36 个权重的投影 sweep。

| 投影 | padding | split parallel | 一次汇集共享图 | 逐层汇集与调度 |
| --- | ---: | ---: | ---: | ---: |
| QKV | 1.691 | 2.007 | 1.842 | 8.370 |
| O | 0.908 | 1.176 | 1.014 | 7.590 |
| Gate/Up | 4.600 | 4.766 | 4.719 | 8.499 |
| Down | 1.719 | 2.264 | 1.765 | 6.262 |

一次汇集控制组相对 split 的改善约为 8.2%、13.8%、1.0%、22.0%，
说明共享默认容量矩阵计算有值得验证的空间；但逐层主机编排完全吃掉这部分收益。
它也仍慢于对应的 padding 投影，没有证明完整 decoder 可以超越 padding。

若继续实现共享 Dense、切分 attention，应首先验证常驻共享激活缓冲区、
图内或原生编排依赖，以及与现有 FIA 布局兼容的分支输入输出。
目标是避免每层回到 Python 做两次拷贝、启动图及设置事件，而不是直接扩展本次逐层控制组。
真实层间依赖、FIA update、stride/布局、padding 无效行及图池生命周期都必须保留在后续完整验证中。
本轮尚未实现或测得这种完整 decoder 架构的收益。

## 验证记录

五份最终完整 decoder 报告和一份共享投影报告已通过原始数据核验，
包括源码哈希、默认容量、每图 36 个 FIA 任务、六项精度检查、固定图和缓冲区复用、
正式样本数，以及每个样本五段主机区间覆盖完整 replay。
CPU multi-graph 单元测试 344 项通过。

Qwen3-0.6B 的相关 NPU 回归六项通过：native exact/balanced 图复用各一项，
loaded/NZ/NZ+shared 投影诊断三项，以及原生参数绑定保护一项。
首轮 `-k 'native or dense_projection_diagnostic'` 同时选中了旧的
`test_native_same_shape_replay_does_not_claim_cross_shape_accuracy`；
该用例的资格检查限定 Qwen2，使用本轮 Qwen3 模型时被拒绝。因此原始 pytest 汇总为
`1 failed, 6 passed, 23 deselected`，失败属于测试模型不适用，不能写成整次测试全绿。
后续复现可精确筛选
`(complete_decoder_default_bucket_reuse and native) or native_fia_binding_guards or dense_projection_diagnostic`。

改动 Python 文件的 Ruff 检查与格式检查、C++ clang-format、文档 markdownlint
和 `git diff --check` 通过。

全量 `format.sh ci` 在本机未通过：缺少 `/bin/bash` hook 入口与 ShellCheck，
部分 system hook 的 `python3` 返回 9009，另有既有 `device_length_decode.py` 的 Triton import 错误。
完整日志已保存，不将全量 CI 描述为通过。

## 复现与原始记录

~~~bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_replay.py --model /models/Qwen3-4B \
  --strategy balanced --update-chunk-size 1 --pinned-inputs \
  --nz-projections qkv o gate_up down --native-update --host-timing \
  --output /workspace/native-host.json

# 分别移除 --native-update、--host-timing 得到 Python 对照和计时控制组。
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_dense_probe.py --model /models/Qwen3-4B \
  --strategy balanced --weight-format nz --shared-input-probe \
  --output /workspace/shared-staged.json
~~~

原始 JSON、日志、测量版本源码、实验驱动和验证脚本保存在工作区
`outputs/multi_graph_native_submission/`，远端为
`/workspace/multi-graph-results/native-submit/`。正式缓存版以 `cached-` 命名，
逐层汇集报告为 `shared-staged-nz-v2.json`；`summarize_native_submission.py` 检查源码哈希、
图容量、任务数、精度检查、样本数和主机区间覆盖后生成 `summary.json`。
