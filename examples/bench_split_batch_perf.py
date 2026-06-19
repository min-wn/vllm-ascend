#!/usr/bin/env python3
"""Benchmark split-batch 性能 — 单次运行脚本。

用法:
  # 1. 原生单流 (baseline)
  python examples/bench_split_batch_perf.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 416 --capture-sizes 256,384,512 \
    --max-tokens 128 --tag baseline_nosplit

  # 2. dual-inplace parallel
  python examples/bench_split_batch_perf.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 416 --capture-sizes 256,384,512 \
    --max-tokens 128 --tag dual_inplace \
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

    print(f"\n{'='*60}")
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
    print(f"\nBenchmarking...")
    torch.npu.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    # Stats
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tpot = elapsed / total_tokens * 1000 if total_tokens > 0 else 0  # ms per token

    print(f"\n--- Results ({args.tag}) ---")
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

            print(f"\n  Per-step perf stats ({len(decode_steps)} decode steps, {len(split_steps)} split):")
            print(f"    Avg replay_ms:  {sum(replay_times)/len(replay_times):.2f}")
            print(f"    P50 replay_ms:  {sorted(replay_times)[len(replay_times)//2]:.2f}")
            print(f"    P99 replay_ms:  {sorted(replay_times)[int(len(replay_times)*0.99)]:.2f}")
            print(f"    Min/Max replay: {min(replay_times):.2f}/{max(replay_times):.2f}")
            print(f"    Avg header_ms:  {sum(header_times)/len(header_times):.2f}" if header_times else "")
            print(f"\n  First 20 steps replay_ms:")
            for i in range(min(20, len(decode_steps))):
                sp = " [SPLIT]" if decode_steps[i].get("is_split") else ""
                print(f"    step {i:>3}: {decode_steps[i]['replay_ms']:>8.2f} ms {sp}")

    # Save result
    result = {
        "tag": args.tag,
        "model": args.model,
        "batch_size": args.batch_size,
        "capture_sizes": capture_sizes,
        "max_tokens": args.max_tokens,
        "split_enabled": split_enabled,
        "split_mode": args.split_mode,
        "enable_parallel_streams": args.enable_parallel_streams,
        "force_split": args.force_split,
        "total_output_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "throughput_tok_per_sec": round(total_tokens / elapsed, 1),
        "tpot_ms": round(tpot, 3),
        "num_perf_steps": len(perf_stats),
    }
    if perf_stats:
        result["perf_summary"] = {
            "avg_replay_ms": round(sum(replay_times)/len(replay_times), 2),
            "p50_replay_ms": round(sorted(replay_times)[len(replay_times)//2], 2),
            "p99_replay_ms": round(sorted(replay_times)[int(len(replay_times)*0.99)], 2),
        }

    result_path = os.path.join(args.output_dir, f"result_{args.tag}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {result_path}")
    print(f"Perf stats:      {perf_path}")


if __name__ == "__main__":
    main()
