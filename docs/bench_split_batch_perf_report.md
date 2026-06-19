# Split-Batch (Dual-Inplace) 性能对比测试报告

> 测试日期: 2026-06-10
> 环境: Ascend 910B3 × 1, wn-vllm-ascend-inplace 容器

---

## 1. 测试脚本

**路径**: `examples/bench_split_batch_perf.py`

### 功能

单次运行脚本，支持两种模式：
- **原生单流 (baseline)**: split 关闭，整个 batch 直接 padding 后一次 graph replay
- **Dual-Inplace**: 开启 split，将 batch 拆分为 gear(命中 graph size) + recapture(剩余部分)，走 parallel stream 并发 replay

通过环境变量 `VLLM_ASCEND_PERF_STATS_FILE` 采集每步 decode 的 `replay_ms` / `header_ms` / `is_split` 等细粒度性能数据。

### 脚本源码

```python
#!/usr/bin/env python3
"""Benchmark split-batch 性能 — 单次运行脚本。

用法:
  # 1. 原生单流 (baseline)
  python examples/bench_split_batch_perf.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --batch-size 416 --capture-sizes 256,384,512 \\
    --max-tokens 128 --tag baseline_nosplit

  # 2. dual-inplace parallel
  python examples/bench_split_batch_perf.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --batch-size 416 --capture-sizes 256,384,512 \\
    --max-tokens 128 --tag dual_inplace \\
    --split-mode inplace_parallel --enable-parallel-streams --force-split

  结果会输出到终端 + JSON 文件。
"""

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=416)
    parser.add_argument("--capture-sizes", type=str, default="256,384,512")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tag", type=str, default="run", help="标识此次运行的名称")
    parser.add_argument("--output-dir", type=str, default="./bench_results")
    # Split config
    parser.add_argument("--split-mode", type=str, default=None,
                        choices=[None, "parallel_buffer", "inplace_serial", "inplace_parallel"])
    parser.add_argument("--enable-parallel-streams", action="store_true")
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--ascend-device", type=str, default="3")
    args = parser.parse_args()

    capture_sizes = sorted(set(
        int(s.strip()) for s in args.capture_sizes.split(",") if s.strip()))

    split_enabled = args.split_mode is not None

    # Build additional_config
    additional_config = {}
    if split_enabled:
        split_cfg = {
            "enabled": True,
            "mode": args.split_mode,
            "num_splits": args.num_splits,
            "enable_parallel_streams": args.enable_parallel_streams,
            "min_batch_size_for_split": 4,
            "force_split": args.force_split,
            "enable_inplace_lazy_capture": True,
            "inplace_split_planner_policy": "largest_lower",
        }
        additional_config["split_batch_config"] = split_cfg

    print(f"\\n{'='*60}")
    print(f"  Tag:             {args.tag}")
    print(f"  Model:           {args.model}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Capture sizes:   {capture_sizes}")
    print(f"  Max tokens:      {args.max_tokens}")
    print(f"  Warmup:          {args.warmup}")
    print(f"  Split enabled:   {split_enabled}")
    if split_enabled:
        print(f"  Split mode:      {args.split_mode}")
        print(f"  Parallel stream: {args.enable_parallel_streams}")
        print(f"  Force split:     {args.force_split}")
    print(f"{'='*60}")

    # Expected split info
    if split_enabled and args.batch_size not in capture_sizes:
        lower = [s for s in capture_sizes if s < args.batch_size]
        if lower:
            gear = max(lower)
            recapture = args.batch_size - gear
            print(f"  >> Expected split: gear={gear}, recapture={recapture}")
    print()

    # Set PERF_STATS
    os.makedirs(args.output_dir, exist_ok=True)
    perf_path = os.path.join(args.output_dir, f"perf_{args.tag}.jsonl")
    os.environ["VLLM_ASCEND_PERF_STATS_FILE"] = perf_path
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.ascend_device

    # Build LLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_num_seqs=args.batch_size + 128,
        gpu_memory_utilization=0.9,
        compilation_config={
            "level": 3,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        },
        additional_config=additional_config,
    )

    # Fixed-batch prompts
    prompt = "Write one concise sentence about deterministic batch scheduling."
    prompts = [prompt] * args.batch_size

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Warmup
    print(f"Warmup ({args.warmup} steps)...")
    for i in range(args.warmup):
        llm.generate(prompts, sampling)
        print(f"  warmup {i+1}/{args.warmup} done")

    # Benchmark
    print(f"\\nBenchmarking...")
    torch.npu.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    # Stats
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tpot = elapsed / total_tokens * 1000 if total_tokens > 0 else 0

    print(f"\\n--- Results ({args.tag}) ---")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Elapsed:             {elapsed:.3f}s")
    print(f"  Throughput:          {total_tokens/elapsed:.1f} tok/s")
    print(f"  TPOT:                {tpot:.3f} ms/tok")
    print(f"  Throughput:          {args.batch_size/elapsed:.1f} req/s")

    # Load perf stats
    perf_stats = []
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        perf_stats.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if perf_stats:
        decode_steps = [s for s in perf_stats if "replay_ms" in s]
        if decode_steps:
            replay_times = [s["replay_ms"] for s in decode_steps]
            header_times = [s["header_ms"] for s in decode_steps if "header_ms" in s]
            split_steps = [s for s in decode_steps if s.get("is_split")]

            print(f"\\n  Per-step perf stats ({len(decode_steps)} decode steps, {len(split_steps)} split):")
            print(f"    Avg replay_ms:  {sum(replay_times)/len(replay_times):.2f}")
            print(f"    P50 replay_ms:  {sorted(replay_times)[len(replay_times)//2]:.2f}")
            print(f"    P99 replay_ms:  {sorted(replay_times)[int(len(replay_times)*0.99)]:.2f}")
            print(f"    Min/Max replay: {min(replay_times):.2f}/{max(replay_times):.2f}")
            print(f"    Avg header_ms:  {sum(header_times)/len(header_times):.2f}" if header_times else "")
            print(f"\\n  First 20 steps replay_ms:")
            for i in range(min(20, len(decode_steps))):
                sp = " [SPLIT]" if decode_steps[i].get("is_split") else ""
                print(f"    step {i:>3}: {decode_steps[i]['replay_ms']:>8.2f} ms {sp}")

    result_path = os.path.join(args.output_dir, f"result_{args.tag}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\\nResult saved to: {result_path}")
    print(f"Perf stats:      {perf_path}")


if __name__ == "__main__":
    main()
```

---

## 2. 运行方式

### 前置条件

```bash
cd /vllm-workspace/vllm-ascend
```

### 运行 baseline

```bash
python examples/bench_split_batch_perf.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 416 --capture-sizes 256,384,512 \
  --max-tokens 128 --tag baseline_nosplit
```

### 运行 dual-inplace

```bash
python examples/bench_split_batch_perf.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 416 --capture-sizes 256,384,512 \
  --max-tokens 128 --tag dual_inplace \
  --split-mode inplace_parallel --enable-parallel-streams --force-split
```

### 切换模型

本地已有模型路径：
- `/vllm-workspace/models/Qwen3-4B` — Qwen3 4B (~7.7GB)
- `/vllm-workspace/models/Qwen3-4B` — 直接替换 `--model` 参数即可

```bash
# 用本地模型
python examples/bench_split_batch_perf.py \
  --model /vllm-workspace/models/Qwen3-4B \
  --batch-size 416 --capture-sizes 256,384,512 \
  --max-tokens 128 --tag qwen3-4b_baseline_nosplit
```

结果自动保存到 `./bench_results/result_<tag>.json` 和 `./bench_results/perf_<tag>.jsonl`。

---

## 3. 测试结果

### 3.1 Qwen2.5-0.5B-Instruct（~1.4GB，极小模型）

**配置**: batch=416, capture_sizes=[256,384,512], max_tokens=128
**预期 split**: gear=384, recapture=32 (padding 节省: 416→512 原本浪费 96 tokens)

| Metric | Baseline (单流) | Dual-Inplace | Diff |
|--------|:---:|:---:|:---:|
| **Throughput** | 14,135.8 tok/s | 11,807.5 tok/s | **-16.5%** |
| **TPOT** | 0.071 ms/tok | 0.085 ms/tok | +19.7% |
| **Avg replay_ms** (含第1步) | 7.36 | 14.46¹ | +96% |
| **Replay 稳态**² | 7.36 ms | **~12.5 ms** | **+70%** |
| **P50 replay_ms** | 7.32 | 12.86 | +76% |
| **P99 replay_ms** | 8.31 | 14.17 | +70% |
| **Avg header_ms** | 2.84 | 5.11 | +80% |
| **Split decode steps** | 0/508 | **501/508** | ✅ |
| 第1步 (lazy capture) | — | 191.95 ms | 不计入 |
| **GPU memory (weights)** | 0.93 GB | 0.93 GB | — |
| **Graph capture** | 3 graphs, 0.15 GiB | 3+3 graphs, 0.33 GiB | +0.18 GiB |

> ¹ Avg replay 含第1步 lazy capture 191ms
> ² 稳态 = 排除第1步: (14.46×508 − 191.95) / 507 ≈ **12.5 ms**

**按步骤细节 (dual, 前20步):**

```
step   0:   191.95 ms  [SPLIT]  ← lazy capture
step   1:    12.47 ms  [SPLIT]
step   2:    12.59 ms  [SPLIT]
...
```

### 3.2 Qwen3-4B（~7.7GB，中等模型）

**配置**: batch=416, capture_sizes=[256,384,512], max_tokens=128
**预期 split**: gear=384, recapture=32

| Metric | Baseline (单流) | Dual-Inplace | Diff |
|--------|:---:|:---:|:---:|
| **Throughput** | 6,482.0 tok/s | 5,772.5 tok/s | **-10.9%** |
| **TPOT** | 0.154 ms/tok | 0.173 ms/tok | +12.3% |
| **Avg replay_ms** (含第1步) | 10.97 | 52.18¹ | +376% |
| **Replay 稳态**² | 10.97 ms | **~51.5 ms** | **+369%** |
| **P50 replay_ms** | 10.90 | 51.39 | +371% |
| **P99 replay_ms** | 12.10 | 54.53 | +351% |
| **Avg header_ms** | 2.85 | 5.13 | +80% |
| **Split decode steps** | 0/508 | **504/508** | ✅ |
| 第1步 (lazy capture) | — | 228.11 ms | 不计入 |
| **GPU memory (weights)** | 7.56 GB | 7.56 GB | — |
| **Graph capture** | 3 graphs, 0.35 GiB | 3+3 graphs, 0.58 GiB | +0.23 GiB |

> ¹ Avg replay 含第1步 lazy capture 228ms
> ² 稳态 = 排除第1步: (52.18×508 − 228.11) / 507 ≈ **51.5 ms**

---

## 4. 分析与结论

### 4.1 Dual-Inplace 当前性能劣于单流

在两个模型上 dual-inplace 都比单流慢：
- **0.5B**: throughput 降 **16.5%**，稳态 replay **+70%**
- **4B**: throughput 降 **10.9%**，稳态 replay **+369%**（异常）

### 4.2 问题诊断

4B 模型上 dual replay 从 10.97ms 跳到 51.5ms（4.7x）是不正常的。理论上 gear(384) + recapture(32) 的计算量 ≤ 单次 512。可能原因：

1. **Parallel stream 资源限制**
   ```python
   torch.npu.set_stream_limit(self.stream_main, cube_num=15, vector_num=20)
   torch.npu.set_stream_limit(self.stream_parallel, cube_num=15, vector_num=20)
   ```
   cube/vector 配额从默认降到了 15/20，每个 stream 只能使用 `15/40 ≈ 37.5%` 的 cube 资源。两个 stream 合计最多使用 75%，但分离后的 cache 局部性可能更差。

2. **同步开销**
   - Python thread create/join 每次 decode 都有
   - 两个 stream 的 `torch.npu.synchronize()` 等待
   - output merge 的 `torch.cat` 和 trim

3. **Attention params update 成本**
   - 每个 split 都要更新 block_table（`refresh_block_table=True`）
   - 每步 `_update_attn_params_for_split_ubatch()` 在两个 split 上分别执行

### 4.3 预期

Dual-inplace 的设计目标是大模型（27B+）场景，当单次 graph replay 需要 50-100ms+ 时，split 节省的 padding 和并行计算的收益才可能覆盖 overhead。
