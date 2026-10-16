# 默认桶完整 Decoder 的编译与权重布局验证

本文继续 [投影归因](multi_graph_projection_diagnosis.md)，
检验两项可实施方向：官方编译路径对齐、完整 decoder 的选择性或全量 NZ 权重布局。
硬件为 910B3，驱动 24.1.rc3，CANN 9.1，torch 2.10，vLLM 0.30 / Ascend 0.30rc1。
实验日期为 2026-10-16，未更新驱动。

## 对照边界

固定 Qwen3-4B BF16、Q1、KV512，默认捕获集合的随机样本 B98。
padding 为 104，balanced 为 56+48（真实请求 50+48），exact 为 96+2。
所有容量来自真实默认集合，没有专门的 49 或 98 图，也没有切分 head 或组合外图。

计时包含 CPU 准备、输入拷贝、task resolve、每层 FIA update、完整 decoder replay、
输出合并和同步。使用 pinned 输入与 `--update-chunk-size 1`；
每项正式对照五轮、每轮三十次，交替 padding 与 split 的测量顺序。
所有正式布局比较在物理 NPU 0 上按进程顺序执行；NPU 回归使用物理 NPU 7。
计时包含完整 decoder，不包含 LM head、采样、调度器或网络服务。

## ND 与 NZ

ND 是当前加载的普通稠密布局，实测 format ID 为 2。
FRACTAL_NZ 是适合矩阵计算的分块布局，format ID 为 29。
两者都保存 BF16 权重，NZ 不是量化，也不减少数学运算量。
转换在捕获前进行，不计入稳态 replay；因此结果不能描述启动成本。

`benchmarks/multi_graph_bucket_replay.py` 新增 `--nz-projections`：

- 不传：四类线性投影保留原 ND。
- `qkv o down`：只转换这三类投影，gate/up 保留 ND。
- `qkv o gate_up down`：转换四类投影。

padding 和两张子图使用同一个模型、同一份权重，布局优化对三张图同时生效。
没有为 split 单独选择一个更有利的 padding 基线。
转换前保存每个图容量的原 ND eager 输出，转换后仍用 `rtol=0.02, atol=0.02` 检查。
复用顺序为 B98/KV512 → B97/KV511 → B98/KV512，输入值与真实长度均变化，
图和输入输出地址保持不变，不重新捕获。
这只认证相同捕获 shape 的输出一致性，尚未认证不同 shape 的 FIA 输出完全一致。

## 官方编译路径

原基准使用 `enforce_eager=True`、`skip_compiled=True` 和 `disable_compilation`，
因此旧数据不能代表官方编译服务的性能。
新增 `--compiled` 显式启用 `VLLM_COMPILE`（mode 3）及 `FULL_AND_PIECEWISE`，
选择当前对照所需的合法默认容量作为引擎启动捕获集合。
不使用该选项时，既有非编译基准行为不变。

Ascend 在图模式为 `NONE` 时会关闭编译；所以不能仅修改 compile mode。
编译实验使用官方默认 KV 布局，因为现有连续 KV workaround 的准入限制不支持该组合。
没有放宽生产 `validate_vllm_config` 的模型、编译或布局检查。

实际日志确认经过 Dynamo 转换、`AscendCompiler` 和图编译，
`torch.compile` 总耗时约 30.37 秒，随后三张完整 decoder 图捕获成功。
但首次 FIA task update 失败，错误码 `207019`；日志中的关键内容为：

~~~text
current driver version does not support to update this op, please upgrade the driver
KernelLaunch failed: .../slice/Slice_..._high_performance.o
~~~

这与非连续 KV 在 FIA update 内引入 Slice/Contiguous 的已有驱动限制一致。
因此本轮没有有效的 compiled padding/split 性能对照，也不能声称已完成编译融合优化。
失败结果与日志保留；没有省略 update 或放宽精度继续计时。
错误后的进程未自动释放 NPU，确认 PID 后仅终止本次失败的实验进程，再启动正式对照。

报告中的 `graph_count=3` 只表示基准自身持有的图。
编译模式下引擎启动还会捕获自己的图，不能据此声称整个进程仅有三张图。
`graph_count_scope` 明确记录该边界，`compiled_serving_qualified` 始终为 false。

## 结果解释

下表单位为 ms，数值为五轮各自 p50 的中位数；“慢于 padding”为同布局的相对差距。

| 切分容量 | 权重布局 | padding 104 | split parallel | 慢于 padding |
| --- | --- | ---: | ---: | ---: |
| 56+48 | ND | 24.676 | 31.952 | 29.49% |
| 56+48 | QKV/o/down NZ | 24.648 | 30.690 | 24.51% |
| 56+48 | 四类投影全 NZ | 25.181 | 28.760 | 14.21% |
| 96+2 | ND | 24.645 | 31.259 | 26.84% |
| 96+2 | QKV/o/down NZ | 24.544 | 30.322 | 23.54% |
| 96+2 | 四类投影全 NZ | 25.435 | 29.895 | 17.54% |

独立进程复测全 NZ：56+48 为 padding 25.194 / split 28.828（慢 14.42%），
96+2 为 padding 25.183 / split 29.856（慢 18.56%）。
相比各自 ND，首轮全 NZ 的 split 绝对时延分别降低约 9.99% 和 4.37%。
但首轮 padding 也分别变慢约 2.05% 和 3.20%，96+2 的 padding 在复测中另有约 1% 波动。

布局优化必须同时比较绝对 split 时延与同布局 padding 时延；
如果某种布局让 padding 变慢，不能把全部相对差距缩小都算作切分优化。
当前完整 decoder 结果支持“权重布局可以降低切分时延”，尚不支持“切分优于 padding”。
选择性 NZ 在先前孤立投影诊断中的表现，也不能直接外推到完整 decoder。

计时结束后的独立 trace 验证了实际算子分派：选择性 NZ 时每张完整图为
108 个 MatmulWeightNz、36 个 ND Matmul 和 36 个 FIA；
全 NZ 时为 144 个 MatmulWeightNz 和 36 个 FIA。
两张切分图对应上述数量的两倍。trace 使用六个完整测量标记核验，未用其扰动后的耗时替代正式计时。
本轮没有硬件带宽计数器，不据此声称已经证明 HBM 饱和。

## 验证记录

八份成功正式报告均通过默认容量、源码哈希、每图 36 个更新任务、
六项 ND eager 精度检查和图/缓冲区复用身份检查。
精度容差保持 `rtol=0.02, atol=0.02`。
Qwen3-0.6B 的 NPU 回归八项通过，覆盖 baseline、pinned、选择性 NZ 和全 NZ，
每种布局包含 exact 和 balanced 两种容量组合；CPU multi-graph 单测 339 项通过。
这些验证不等价于不同捕获 shape 之间的数值认证，也不等价于生产服务认证。

改动 Python 文件的 Ruff 检查与格式检查、文档 markdownlint 和 `git diff --check` 通过。
全量 `format.sh ci` 未通过：本地缺少 `/bin/bash` hook 入口与 ShellCheck，
部分 system hook 的 `python3` 返回 9009，另有既有 `device_length_decode.py` 的 Triton import 检查错误。
完整日志保存在实验目录，没有将全量 CI 描述为通过。

## 复现

~~~bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_replay.py --model /models/Qwen3-4B \
  --strategy balanced --update-chunk-size 1 --pinned-inputs \
  --nz-projections qkv o gate_up down --output /workspace/full-nz.json

# 编译兼容性诊断；当前驱动预期在 FIA update 处遇到上述限制。
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 \
python benchmarks/multi_graph_bucket_replay.py --model /models/Qwen3-4B \
  --strategy balanced --compiled --update-chunk-size 1 --pinned-inputs \
  --output /workspace/compiled-probe.json
~~~

原始 JSON、日志、trace、测量版本源码和运行命令保存在工作区
`outputs/multi_graph_compiled_layout/`，远端目录为
`/workspace/multi-graph-results/compiled-layout/`。
