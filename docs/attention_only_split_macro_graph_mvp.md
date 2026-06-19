# Attention-only split macro graph MVP 实施说明

## 目标

本轮先落地一个可开关的实验 runner，用来验证“非 attention piece 串行、attention piece 双流并行”的执行形态，避免整图 inplace split 中 MatMul 在双流上重叠竞争 cube。

当前实现还不是 PTA/C++ 层的多流宏图 capture。它是 Python runner 上的 piecewise 调度 MVP：通过 vLLM `PiecewiseRuntimeHandle` 取得 `split_gm`，逐 piece 执行；非 attention piece 在 main stream 上执行一次，attention splitting piece 按 split0/split1 切片后在 main/parallel 两条 stream 上执行，再把 attention 输出按 token 维 merge 回完整 batch。

## 启用方式

推荐使用结构化配置：

```json
{
  "model_runner_config": {
    "type": "attention_only_macro"
  },
  "split_batch_config": {
    "enabled": true,
    "mode": "inplace_parallel",
    "enable_parallel_streams": true,
    "enable_inplace_lazy_capture": false,
    "macro_graph_config": {
      "enabled": true,
      "schedule": "attention_only_split",
      "capture_timing": "pre",
      "miss_policy": "error",
      "plan_source": "explicit",
      "require_exact_graph_tokens": true
    }
  }
}
```

也可以用环境变量临时覆盖 runner：

```bash
VLLM_ASCEND_MODEL_RUNNER=attention_only_macro
```

## 运行条件

- 必须是 `CUDAGraphMode.PIECEWISE`，并且模型已走 vLLM piecewise compile。
- 必须是 `split_batch_config.mode=inplace_parallel`。
- 当前只支持 `num_splits=2`。
- 当前要求 `graph_num_tokens == num_tokens`，也就是 attention split 不带 padded tail。
- 当前不支持 MLA、MROPE、PCP/DCP。
- 当前拒绝 `cudagraph_copy_inputs`，因为它会在 piecewise callable 内部重绑 symbolic tensor buffer，破坏按 runtime token 维切片的假设。
- 当前要求 `enable_inplace_lazy_capture=false`，避免 profiling 中混入 offset graph lazy capture。

## 当前执行形态

运行时 `_run_split_batch_inplace_parallel()` 被实验 runner 接管：

1. 使用原始 inplace split planner 产生两个 split。
2. 为两个 split 构造仅服务 attention piece 的 forward context。
3. 捕获本次模型调用的 `PiecewiseRuntimeCall`，拿到真实 compiled args 和 `PiecewiseRuntimeHandle`。
4. 用 `torch.fx.Interpreter` 逐 piece 执行 `split_gm`。
5. 普通 piece：main stream 执行一次，处理完整 batch。
6. attention splitting piece：对 token-major tensor 参数按 split 切片，两个线程分别在 main/parallel stream 执行 attention，再按 token 维 merge 输出。
7. 下一个普通 piece 等待 parallel stream 完成后继续在 main stream 执行完整 batch。

因此本版可以验证 MatMul 是否不再双流重叠，以及 attention-only 并行是否有潜在收益；但 trace 中仍会看到 Python 调度导致的 stream dependency event 和 attention 输出 merge copy。

## 后续 PTA 宏图版本

下一阶段应把当前 Python 调度语义下沉为 PTA 多流宏图：

1. 预先根据 `macro_graph_config.capture_plans` 构造 capture plan，不做 runtime lazy capture。
2. capture 内部记录 main stream 的非 attention piece，attention piece 记录 main/secondary stream 两条分支。
3. graph 内部保留必要的 stream dependency，运行时只 replay 一个 macro graph。
4. attention 输出优先写入 full output view，减少 Python MVP 中的 merge copy。
5. miss policy 保持 fail-fast，避免静默回退到整图 split 或 padding 路径。
