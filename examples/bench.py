#!/usr/bin/env python3
# isort: skip_file
"""Performance benchmark for Ascend vLLM.

目标:
- 测试不同 batch size 和序列长度下的吞吐量 (throughput) 和延迟 (latency)
- 支持 split-batch 开启/关闭的性能对比

运行方式:
  python vllm-ascend/examples/bench.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-tokens 128 \
    --batch-size 16 \
    --num-iters 10

输出:
- 会在 --output-dir 下创建时间戳子目录，保存:
  - metadata.json (配置信息)
  - results.json (性能指标)
  - console.log (stdout/stderr)
"""

import gc
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Match existing offline example behavior.
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "7")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_ASCEND_SPLIT_TWO_GRAPH_DIAG", "1")

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
import vllm.logger as vllm_logger_module  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


def create_parser() -> FlexibleArgumentParser:
    """Create a CLI parser in the same style as vLLM offline examples."""
    parser = FlexibleArgumentParser()

    # Add all standard engine/vllm args (model, tokenizer, compilation_config, etc).
    EngineArgs.add_cli_args(parser)

    # Match previous behavior in this script.
    parser.set_defaults(trust_remote_code=True)

    # Default compilation config.
    cudagraph_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    compilation_config = {
        "level": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
    }
    parser.set_defaults(compilation_config=compilation_config)

    bench_group = parser.add_argument_group("Benchmark settings")
    bench_group.add_argument("--max-tokens", type=int, default=128)
    bench_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Single batch size for benchmark.",
    )
    bench_group.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated list of batch sizes (e.g., '1,2,4,8,16'). Overrides --batch-size.",
    )
    bench_group.add_argument("--num-iters", type=int, default=10,
        help="Number of iterations for benchmark (first iteration is warmup)")
    bench_group.add_argument("--input-len", type=int, default=32,
        help="Input prompt length (pad/truncate to this length)")
    bench_group.add_argument(
        "--run-mode",
        type=str,
        default="both",
        choices=["both", "disabled", "enabled"],
        help="Which run(s) to execute: both (compare), disabled only, enabled only.",
    )

    split_group = parser.add_argument_group("Split-batch settings")
    split_group.add_argument("--num-splits", type=int, default=2)
    split_group.add_argument("--min-batch-size-for-split", type=int, default=4)
    split_group.add_argument(
        "--enable-parallel-streams",
        action="store_true",
        help="Enable split parallel streams (if supported).",
    )

    output_group = parser.add_argument_group("Output settings")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./bench_results",
        help="Base output directory.",
    )

    return parser


def _generate_prompt(input_len: int) -> str:
    """Generate a fixed prompt of specified length."""
    # Use a simple repeating pattern
    base = "Hello world. "
    while len(base) < input_len:
        base += base
    return base[:input_len]


def _build_split_additional_config(
    *,
    enabled: bool,
    num_splits: int,
    enable_parallel_streams: bool,
    min_batch_size_for_split: int,
) -> dict[str, Any]:
    return {
        "split_batch_config": {
            "enabled": enabled,
            "num_splits": num_splits,
            "enable_parallel_streams": enable_parallel_streams,
            "min_batch_size_for_split": min_batch_size_for_split,
        }
    }


def _build_llm_from_args(
    args: dict[str, Any],
    additional_config: dict[str, Any] | None = None,
) -> LLM:
    """Build LLM engine from parsed arguments."""
    engine_args = {}
    for key in dir(EngineArgs):
        if key.startswith("_"):
            continue
        val = args.get(key)
        if val is not None:
            engine_args[key] = val

    if additional_config:
        engine_args["additional_config"] = additional_config

    return LLM(**engine_args)


def _run_benchmark(
    llm: LLM,
    prompts: list[str],
    sampling: SamplingParams,
    num_iters: int,
) -> dict[str, Any]:
    """Run benchmark and collect metrics."""
    warmup_iter = 1
    total_iters = warmup_iter + num_iters

    # Warmup
    print(f"Warming up with {warmup_iter} iteration(s)...")
    _ = llm.generate(prompts, sampling)
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    gc.collect()

    # Benchmark iterations
    print(f"Running {num_iters} benchmark iterations...")
    iter_times = []
    total_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0

    for i in range(num_iters):
        iter_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        iter_end = time.perf_counter()

        iter_time = iter_end - iter_start
        iter_times.append(iter_time)

        # Count tokens
        for output in outputs:
            generated_tokens = len(output.outputs[0].token_ids)
            total_tokens += generated_tokens

        print(f"  Iter {i+1}/{num_iters}: {iter_time:.3f}s")

    # Calculate statistics
    avg_time = sum(iter_times) / len(iter_times)
    min_time = min(iter_times)
    max_time = max(iter_times)

    total_time = sum(iter_times)
    throughput_tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0
    throughput_samples_per_s = (len(prompts) * num_iters) / total_time if total_time > 0 else 0.0

    return {
        "num_iters": num_iters,
        "batch_size": len(prompts),
        "iter_times_s": iter_times,
        "avg_time_s": avg_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "throughput_samples_per_s": throughput_samples_per_s,
    }


def _cleanup_llm(llm: LLM) -> None:
    """Release engine resources before starting the next run."""
    try:
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    finally:
        if hasattr(torch, "npu"):
            if hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
        gc.collect()


def main() -> int:
    parser = create_parser()
    args: dict[str, Any] = vars(parser.parse_args())

    max_tokens = int(args.pop("max_tokens"))
    batch_size_arg = args.pop("batch_size")
    batch_sizes_arg = args.pop("batch_sizes")
    num_iters = int(args.pop("num_iters"))
    input_len = int(args.pop("input_len"))
    run_mode = str(args.pop("run_mode"))
    num_splits = int(args.pop("num_splits"))
    min_batch_size_for_split = int(args.pop("min_batch_size_for_split"))
    enable_parallel_streams = bool(args.pop("enable_parallel_streams"))
    output_dir_base = str(args.pop("output_dir"))

    # Parse batch sizes: --batch-sizes takes precedence over --batch-size
    if batch_sizes_arg:
        batch_sizes = [int(x.strip()) for x in batch_sizes_arg.split(",")]
    elif batch_size_arg is not None:
        batch_sizes = [int(batch_size_arg)]
    else:
        batch_sizes = [16]  # default

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir_base, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Build split configs (shared across all batch sizes)
    split_disabled_cfg = _build_split_additional_config(
        enabled=False,
        num_splits=num_splits,
        enable_parallel_streams=False,
        min_batch_size_for_split=min_batch_size_for_split,
    )
    split_enabled_cfg = _build_split_additional_config(
        enabled=True,
        num_splits=num_splits,
        enable_parallel_streams=enable_parallel_streams,
        min_batch_size_for_split=min_batch_size_for_split,
    )

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model": args.get("model"),
        "tokenizer": args.get("tokenizer") or args.get("model"),
        "batch_sizes": batch_sizes,
        "max_tokens": max_tokens,
        "input_len": input_len,
        "num_iters": num_iters,
        "num_splits": num_splits,
        "enable_parallel_streams": enable_parallel_streams,
        "min_batch_size_for_split": min_batch_size_for_split,
        "compilation_config": args.get("compilation_config"),
        "run_mode": run_mode,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    # Console log
    console_path = os.path.join(out_dir, "console.log")
    console_f = open(console_path, "a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = type("TeeIO", (), {
        "write": lambda self, s: (old_stdout.write(s), console_f.write(s)),
        "flush": lambda self: (old_stdout.flush(), console_f.flush()),
    })()
    sys.stderr = type("TeeIO", (), {
        "write": lambda self, s: (old_stderr.write(s), console_f.write(s)),
        "flush": lambda self: (old_stderr.flush(), console_f.flush()),
    })()

    # Re-bind vLLM logger streams
    try:
        vllm_logger_module._configure_vllm_root_logger()
        for _name in ("vllm", "vllm_ascend"):
            _lg = logging.getLogger(_name)
            for _h in _lg.handlers:
                if isinstance(_h, logging.StreamHandler):
                    _h.setStream(sys.stderr)
    except Exception as _e:
        print(f"Warning: failed to rebind vLLM loggers: {_e}")

    try:
        print("=== Benchmark Configuration ===")
        print(f"Model: {args.get('model')}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Max tokens: {max_tokens}")
        print(f"Input length: {input_len}")
        print(f"Iterations: {num_iters} (first iter is warmup)")
        print(f"Output dir: {out_dir}")
        print()

        # Generate base prompt
        base_prompt = _generate_prompt(input_len)

        # Setup sampling params
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        results = {}

        for bs in batch_sizes:
            prompts = [base_prompt] * bs
            print(f"\n{'='*50}")
            print(f"=== Batch size: {bs} ===")
            print(f"{'='*50}")

            bs_key = f"bs_{bs}"
            results[bs_key] = {}

            # Run disabled
            if run_mode in ("both", "disabled"):
                print(f"--- Split disabled (batch_size={bs}) ---")
                llm_disabled = _build_llm_from_args(args, additional_config=split_disabled_cfg)
                results[bs_key]["disabled"] = _run_benchmark(
                    llm_disabled, prompts, sampling, num_iters
                )
                results[bs_key]["disabled"]["config"] = "split_disabled"
                results[bs_key]["disabled"]["split_batch_config"] = split_disabled_cfg["split_batch_config"]
                print(f"Avg time: {results[bs_key]['disabled']['avg_time_s']:.3f}s")
                print(f"Throughput: {results[bs_key]['disabled']['throughput_tokens_per_s']:.1f} tokens/s")

                if run_mode == "both":
                    _cleanup_llm(llm_disabled)

            # Run enabled
            if run_mode in ("both", "enabled"):
                print(f"--- Split enabled (batch_size={bs}) ---")
                llm_enabled = _build_llm_from_args(args, additional_config=split_enabled_cfg)
                results[bs_key]["enabled"] = _run_benchmark(
                    llm_enabled, prompts, sampling, num_iters
                )
                results[bs_key]["enabled"]["config"] = "split_enabled"
                results[bs_key]["enabled"]["split_batch_config"] = split_enabled_cfg["split_batch_config"]
                print(f"Avg time: {results[bs_key]['enabled']['avg_time_s']:.3f}s")
                print(f"Throughput: {results[bs_key]['enabled']['throughput_tokens_per_s']:.1f} tokens/s")

            # Compare if both
            if run_mode == "both" and "disabled" in results[bs_key] and "enabled" in results[bs_key]:
                disabled_time = results[bs_key]["disabled"]["avg_time_s"]
                enabled_time = results[bs_key]["enabled"]["avg_time_s"]
                speedup = disabled_time / enabled_time if enabled_time > 0 else 0.0
                throughput_ratio = (
                    results[bs_key]["enabled"]["throughput_tokens_per_s"] /
                    results[bs_key]["disabled"]["throughput_tokens_per_s"]
                    if results[bs_key]["disabled"]["throughput_tokens_per_s"] > 0 else 0.0
                )
                print(f"Speedup: {speedup:.2f}x, Throughput ratio: {throughput_ratio:.2f}x")
                results[bs_key]["comparison"] = {
                    "speedup": speedup,
                    "throughput_ratio": throughput_ratio,
                }

        # Summary table
        print(f"\n{'='*60}")
        print("=== Summary ===")
        print(f"{'='*60}")
        print(f"{'Batch Size':<12} {'Split':<10} {'Avg Time':<12} {'Throughput':<15} {'Speedup':<10}")
        print("-" * 60)
        for bs in batch_sizes:
            bs_key = f"bs_{bs}"
            if "disabled" in results[bs_key]:
                d = results[bs_key]["disabled"]
                print(f"{bs:<12} {'disabled':<10} {d['avg_time_s']:<12.3f} {d['throughput_tokens_per_s']:<15.1f} {'-':<10}")
            if "enabled" in results[bs_key]:
                e = results[bs_key]["enabled"]
                speedup_str = f"{results[bs_key]['comparison']['speedup']:.2f}x" if "comparison" in results[bs_key] else "-"
                print(f"{bs:<12} {'enabled':<10} {e['avg_time_s']:<12.3f} {e['throughput_tokens_per_s']:<15.1f} {speedup_str:<10}")

        # Save results
        results_path = os.path.join(out_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print()
        print(f"Results saved to: {results_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        console_f.close()


if __name__ == "__main__":
    sys.exit(main())
