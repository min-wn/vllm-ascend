"""
Offline vLLM Serving Benchmark Script

Directly loads the model via vLLM's offline LLM engine (no server needed),
runs inference on ShareGPT prompts, and measures TTFT / TPOT / ITL / E2EL
using the per-request metrics exposed by RequestOutput.metrics.

Usage:
    # Single NPU card
    python bench_serve_offline.py \
        --model /workspace/data/models/qwen3-0.6b \
        --dataset-path /workspace/data/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1 \
        --temperature 0 \
        --device 0

    # Multi-card (e.g. NPU 0,1 for TP=2)
    python bench_serve_offline.py \
        --model /workspace/data/models/qwen3-0.6b \
        --dataset-path ... \
        --device 0,1 \
        --tensor-parallel-size 2

    --device sets ASCEND_RT_VISIBLE_DEVICES (Ascend NPU) or
    CUDA_VISIBLE_DEVICES (NVIDIA GPU) automatically before model loading.

Dependencies:
    pip install vllm aiohttp   # vllm provides LLM engine + tokenizer
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkConfig:
    model: str
    dataset_path: str | None
    num_prompts: int
    temperature: float
    top_p: float
    max_tokens: int | None
    seed: int
    ignore_eos: bool
    trust_remote_code: bool
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    num_warmups: int
    save_result: str | None
    device: str | None
    additional_config: dict | None
    compilation_config: dict | None
    input_len: int | None
    output_len: int | None


def load_sharegpt_dataset(
    dataset_path: str, num_prompts: int, seed: int
) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    data = [
        entry
        for entry in data
        if "conversations" in entry and len(entry["conversations"]) >= 2
    ]
    random.seed(seed)
    random.shuffle(data)
    samples = []
    for entry in data:
        if len(samples) >= num_prompts:
            break
        prompt = entry["conversations"][0]["value"]
        completion = entry["conversations"][1]["value"]
        samples.append({"prompt": prompt, "completion": completion})
    if len(samples) < num_prompts:
        print(
            f"WARNING: Only found {len(samples)} valid samples, "
            f"requested {num_prompts}"
        )
    return samples


SYNTHETIC_PROMPTS = [
    "Write a short essay about the importance of artificial intelligence in modern society.",
    "Explain the concept of machine learning in simple terms.",
    "What are the key differences between supervised and unsupervised learning?",
    "Describe the impact of climate change on global ecosystems.",
    "Summarize the history of the internet in a few paragraphs.",
    "What are the advantages and disadvantages of renewable energy sources?",
    "Explain how neural networks work to someone with no technical background.",
    "Discuss the role of education in economic development.",
    "What are the ethical considerations in artificial intelligence research?",
    "Describe the process of photosynthesis in detail.",
    "Compare and contrast democracy and authoritarianism as forms of government.",
    "What are the main challenges facing healthcare systems worldwide?",
    "Explain the theory of relativity in accessible language.",
    "Discuss the social and economic effects of globalization.",
    "What are the potential benefits and risks of genetic engineering?",
    "Describe the water cycle and its importance to life on Earth.",
    "How does the human immune system protect the body from disease?",
    "What role does creativity play in scientific discovery?",
    "Explain the difference between correlation and causation.",
    "Discuss the future of space exploration and its significance for humanity.",
]


def build_synthetic_dataset(
    num_prompts: int, input_len: int | None, seed: int
) -> list[dict]:
    random.seed(seed)
    samples = []
    for i in range(num_prompts):
        prompt = SYNTHETIC_PROMPTS[i % len(SYNTHETIC_PROMPTS)]
        samples.append({"prompt": prompt, "completion": ""})
    return samples


def safe_mean(vals):
    return statistics.mean(vals) if vals else 0.0


def safe_median(vals):
    return statistics.median(vals) if vals else 0.0


def safe_stdev(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def safe_percentile(vals, p):
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f_idx = math.floor(k)
    c_idx = math.ceil(k)
    if f_idx == c_idx:
        return sorted_vals[int(k)]
    return sorted_vals[f_idx] * (c_idx - k) + sorted_vals[c_idx] * (k - f_idx)


def extract_metrics_from_output(
    output,
    wall_clock_e2el: float | None = None,
) -> dict[str, Any] | None:
    has_output = output.outputs and len(output.outputs) > 0
    num_output_tokens = 0
    if has_output:
        num_output_tokens = len(output.outputs[0].token_ids)
    if num_output_tokens == 0 and output.prompt_token_ids is None and not has_output:
        return None

    prompt_len = len(output.prompt_token_ids) if output.prompt_token_ids else 0
    generated_text = output.outputs[0].text if has_output else ""

    metrics = output.metrics
    ttft = None
    decode_time = None

    if metrics is not None:
        ttft = getattr(metrics, "first_token_latency", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        if (
            ttft is not None
            and first_token_ts is not None
            and last_token_ts is not None
        ):
            decode_time = max(0.0, last_token_ts - first_token_ts)

    if ttft is not None and decode_time is not None:
        e2el = ttft + decode_time
    elif wall_clock_e2el is not None:
        e2el = wall_clock_e2el
        if num_output_tokens > 0:
            prefill_ratio = prompt_len / max(prompt_len + num_output_tokens, 1)
            ttft = e2el * prefill_ratio
            decode_time = e2el - ttft
        else:
            ttft = e2el
            decode_time = 0.0
    else:
        e2el = 0.0
        ttft = 0.0
        decode_time = 0.0

    tpot = 0.0
    if num_output_tokens > 1 and decode_time > 0:
        tpot = decode_time / (num_output_tokens - 1)

    return {
        "ttft": ttft,
        "e2el": e2el,
        "tpot": tpot,
        "decode_time": decode_time,
        "num_output_tokens": num_output_tokens,
        "prompt_len": prompt_len,
        "generated_text": generated_text,
        "metrics_source": "engine" if (metrics is not None and ttft is not None) else "wall_clock",
    }


def detect_platform() -> str:
    try:
        import torch_npu  # noqa: F401
        return "npu"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "unknown"


def setup_visible_devices(device: str | None) -> None:
    if device is None:
        return
    platform = detect_platform()
    if platform == "npu":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
        os.environ.setdefault("VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE", "npu:0")
        print(f"[device] ASCEND_RT_VISIBLE_DEVICES={device}")
    elif platform == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"[device] CUDA_VISIBLE_DEVICES={device}")
    else:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"[device] ASCEND_RT_VISIBLE_DEVICES={device}, CUDA_VISIBLE_DEVICES={device}")


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    setup_visible_devices(config.device)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TextPrompt, TokensPrompt

    print(f"Loading model: {config.model}")
    llm_kwargs: dict[str, Any] = dict(
        model=config.model,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        seed=config.seed,
    )
    if config.additional_config is not None:
        llm_kwargs["additional_config"] = config.additional_config
    if config.compilation_config is not None:
        llm_kwargs["compilation_config"] = config.compilation_config
    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()

    if config.dataset_path:
        samples = load_sharegpt_dataset(
            config.dataset_path, config.num_prompts, config.seed
        )
    else:
        samples = build_synthetic_dataset(
            config.num_prompts, config.input_len, config.seed
        )
    if not samples:
        print("ERROR: No valid samples found.")
        sys.exit(1)

    prompts = []
    sampling_params_list = []
    prompt_lens = []
    output_lens = []

    for sample in samples:
        prompt_ids = tokenizer(sample["prompt"]).input_ids
        prompt_len = len(prompt_ids)

        use_token_ids = False
        if config.input_len is not None and prompt_len != config.input_len:
            if prompt_len < config.input_len:
                prompt_ids = prompt_ids + [0] * (config.input_len - prompt_len)
            else:
                prompt_ids = prompt_ids[: config.input_len]
            prompt_len = config.input_len
            use_token_ids = True

        if sample["completion"]:
            completion_ids = tokenizer(sample["completion"]).input_ids
            output_len = len(completion_ids)
        else:
            output_len = config.output_len or config.max_tokens or 128

        if use_token_ids:
            prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
        else:
            prompts.append(TextPrompt(prompt=sample["prompt"]))
        max_tok = config.max_tokens if config.max_tokens is not None else output_len
        sp = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=max_tok,
            ignore_eos=config.ignore_eos,
        )
        sampling_params_list.append(sp)
        prompt_lens.append(prompt_len)
        output_lens.append(output_len)

    dataset_label = "ShareGPT" if config.dataset_path else "synthetic"
    print(f"\nLoaded {len(prompts)} prompts ({dataset_label}).")
    for i in range(len(prompts)):
        print(
            f"  Prompt {i}: input_len={prompt_lens[i]}, "
            f"expected_output_len={output_lens[i]}"
        )

    if config.num_warmups > 0:
        print(f"\nWarming up with {config.num_warmups} request(s)...")
        warmup_prompts = prompts[: config.num_warmups]
        warmup_sp = sampling_params_list[: config.num_warmups]
        llm.generate(warmup_prompts, warmup_sp, use_tqdm=False)
        print("Warmup complete.")

    print(f"\nRunning benchmark with {len(prompts)} prompt(s)...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params_list, use_tqdm=True)
    end_time = time.perf_counter()
    benchmark_duration = end_time - start_time

    per_request_e2el = None
    if len(prompts) == 1:
        per_request_e2el = benchmark_duration

    ttfts = []
    tpots = []
    e2els = []
    itls = []
    total_input = 0
    total_output = 0
    completed = 0
    failed = 0
    metrics_sources = {"engine": 0, "wall_clock": 0}

    for i, output in enumerate(outputs):
        wc_e2el = per_request_e2el
        if wc_e2el is None and len(prompts) > 1:
            wc_e2el = benchmark_duration / len(prompts)

        m = extract_metrics_from_output(output, wall_clock_e2el=wc_e2el)
        if m is not None and m["num_output_tokens"] > 0:
            completed += 1
            ttfts.append(m["ttft"])
            e2els.append(m["e2el"])
            tpots.append(m["tpot"])
            total_input += m["prompt_len"]
            total_output += m["num_output_tokens"]
            metrics_sources[m["metrics_source"]] += 1

            if m["num_output_tokens"] > 1 and m["decode_time"] > 0:
                avg_itl = m["decode_time"] / (m["num_output_tokens"] - 1)
                itls.extend([avg_itl] * (m["num_output_tokens"] - 1))
        else:
            failed += 1
            total_input += prompt_lens[i]
            if output.outputs:
                num_out = len(output.outputs[0].token_ids)
                total_output += num_out

    result = {
        "completed": completed,
        "failed": failed,
        "duration_s": benchmark_duration,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / benchmark_duration
        if benchmark_duration > 0
        else 0,
        "output_throughput": total_output / benchmark_duration
        if benchmark_duration > 0
        else 0,
        "total_token_throughput": (total_input + total_output) / benchmark_duration
        if benchmark_duration > 0
        else 0,
        "ttft": {
            "mean_ms": safe_mean(ttfts) * 1000,
            "median_ms": safe_median(ttfts) * 1000,
            "std_ms": safe_stdev(ttfts) * 1000,
            "p99_ms": safe_percentile(ttfts, 99) * 1000,
        },
        "tpot": {
            "mean_ms": safe_mean(tpots) * 1000,
            "median_ms": safe_median(tpots) * 1000,
            "std_ms": safe_stdev(tpots) * 1000,
            "p99_ms": safe_percentile(tpots, 99) * 1000,
        },
        "itl": {
            "mean_ms": safe_mean(itls) * 1000,
            "median_ms": safe_median(itls) * 1000,
            "std_ms": safe_stdev(itls) * 1000,
            "p99_ms": safe_percentile(itls, 99) * 1000,
        },
        "e2el": {
            "mean_ms": safe_mean(e2els) * 1000,
            "median_ms": safe_median(e2els) * 1000,
            "std_ms": safe_stdev(e2els) * 1000,
            "p99_ms": safe_percentile(e2els, 99) * 1000,
        },
        "note": "ITL in offline mode is approximated as avg decode_time / (output_tokens - 1) "
        "per request, not per-token streaming measurement.",
        "metrics_sources": metrics_sources,
    }

    return result


def print_results(result: dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("{s:{c}^{n}}".format(s=" Offline Serving Benchmark Result ", n=55, c="="))
    print("=" * 55)
    print("{:<45} {:<10}".format("Successful requests:", result["completed"]))
    print("{:<45} {:<10}".format("Failed requests:", result["failed"]))
    print(
        "{:<45} {:<10.2f}".format(
            "Benchmark duration (s):", result["duration_s"]
        )
    )
    print(
        "{:<45} {:<10}".format("Total input tokens:", result["total_input_tokens"])
    )
    print(
        "{:<45} {:<10}".format(
            "Total generated tokens:", result["total_output_tokens"]
        )
    )
    print(
        "{:<45} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<45} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<45} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_token_throughput"]
        )
    )

    for metric_key, metric_name, metric_header in [
        ("ttft", "TTFT", "Time to First Token"),
        ("tpot", "TPOT", "Time per Output Token (excl. 1st token)"),
        ("itl", "ITL", "Inter-token Latency (approx)"),
        ("e2el", "E2EL", "End-to-end Latency"),
    ]:
        m = result[metric_key]
        print("{s:{c}^{n}}".format(s=metric_header, n=55, c="-"))
        print("{:<45} {:<10.2f}".format(f"Mean {metric_name} (ms):", m["mean_ms"]))
        print(
            "{:<45} {:<10.2f}".format(f"Median {metric_name} (ms):", m["median_ms"])
        )
        print("{:<45} {:<10.2f}".format(f"P99 {metric_name} (ms):", m["p99_ms"]))

    print("=" * 55)
    if "note" in result:
        print(f"\nNote: {result['note']}")
    src = result.get("metrics_sources", {})
    if src.get("wall_clock", 0) > 0:
        print(
            f"Metrics source: engine={src.get('engine', 0)}, "
            f"wall_clock_fallback={src.get('wall_clock', 0)}"
        )
        print(
            "  (wall_clock: TTFT estimated by prefill/total token ratio, "
            "not streaming measurement)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline vLLM Serving Benchmark (no server needed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ShareGPT dataset JSON file. "
        "If not set, uses synthetic prompts (--input-len/--output-len).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to sample from dataset.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input token length for synthetic prompts. "
        "Only used when --dataset-path is not set.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Expected output token length for synthetic prompts. "
        "Only used when --dataset-path is not set.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens per request. "
        "If not set, uses the ShareGPT completion length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset shuffling.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag in requests.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type (auto, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup requests before benchmarking.",
    )
    parser.add_argument(
        "--save-result",
        type=str,
        default=None,
        help="Path to save benchmark result as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Visible device IDs, e.g. '0' or '0,1'. "
        "Sets ASCEND_RT_VISIBLE_DEVICES (Ascend NPU) or "
        "CUDA_VISIBLE_DEVICES (NVIDIA GPU) automatically.",
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help="Additional engine config as JSON string, "
        'e.g. \'{"split_batch_config": {"enabled": true}}\'',
    )
    parser.add_argument(
        "--compilation-config",
        type=str,
        default=None,
        help="Compilation config as JSON string, "
        'e.g. \'{"cudagraph_mode": "FULL_DECODE_ONLY", '
        '"cudagraph_capture_sizes": [8, 16, 32]}\'',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = BenchmarkConfig(
        model=args.model,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        num_warmups=args.num_warmups,
        save_result=args.save_result,
        device=args.device,
        additional_config=json.loads(args.additional_config) if args.additional_config else None,
        compilation_config=json.loads(args.compilation_config) if args.compilation_config else None,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    print("Configuration:")
    print(f"  Model:                {config.model}")
    print(f"  Num prompts:          {config.num_prompts}")
    print(f"  Temperature:          {config.temperature}")
    print(f"  Top-p:                {config.top_p}")
    print(f"  Max tokens:           {config.max_tokens}")
    print(f"  Dataset:              {config.dataset_path or 'synthetic'}")
    if not config.dataset_path:
        print(f"  Input len:            {config.input_len or '<natural>'}")
        print(f"  Output len:           {config.output_len or config.max_tokens or 128}")
    print(f"  Device:               {config.device or '<auto>'}")
    print(f"  Tensor parallel:      {config.tensor_parallel_size}")
    print(f"  GPU mem utilization:  {config.gpu_memory_utilization}")
    print(f"  Dtype:                {config.dtype}")
    print(f"  Num warmups:          {config.num_warmups}")
    if config.additional_config:
        print(f"  Additional config:    {json.dumps(config.additional_config, separators=(',', ':'))}")
    if config.compilation_config:
        print(f"  Compilation config:   {json.dumps(config.compilation_config, separators=(',', ':'))}")

    result = run_benchmark(config)
    print_results(result)

    if config.save_result:
        with open(config.save_result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to: {config.save_result}")


if __name__ == "__main__":
    main()