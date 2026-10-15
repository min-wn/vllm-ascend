"""Compare padded single graphs and exact split graphs in isolated processes."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from vllm_ascend.worker.multi_graph.benchmark import (
    inspect_device_trace,
    resolve_workload,
    summarize,
    verify_decode_steps,
    verify_outputs,
)


def run_child(args) -> None:
    import torch
    import torch_npu
    from vllm import LLM, SamplingParams
    from vllm.forward_context import get_forward_context

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.multi_graph.runner_v1 import MultiGraphV1Adapter

    state = {"recording": False, "step": None, "runner": None, "split_steps": 0}
    steps, samples, signatures = [], [], []
    forward = NPUModelRunner._model_forward
    replay = ACLGraphWrapper._updatable_graph_replay
    try_run = MultiGraphV1Adapter.try_run

    def observed_replay(self, context, graph):
        if state["step"] is not None:
            state["step"]["single_replays"].append(context.batch_descriptor.num_tokens)
        return replay(self, context, graph)

    def observed_split(self, *positional, **keywords):
        before = dict(self.fallback_counts) if state["step"] is not None else {}
        result = try_run(self, *positional, **keywords)
        if state["recording"] and result is not None:
            state["split_steps"] += 1
        if state["step"] is not None:
            if result is not None:
                state["step"]["lane_sizes"] = [part.actual_tokens for part in self.last_plan.slices]
                if not signatures:
                    signatures.append(dataclasses.asdict(self.signature))
            else:
                reasons = [name for name, count in self.fallback_counts.items() if count > before.get(name, 0)]
                state["step"]["fallback"] = reasons or ["preflight_or_ineligible"]
        return result

    def observed_forward(self, *positional, **keywords):
        state["runner"] = self
        if not state["recording"] or args.measurement == "generation":
            return forward(self, *positional, **keywords)
        context = get_forward_context()
        metadata = context.attn_metadata
        first = next(iter(metadata.values())) if isinstance(metadata, dict) and metadata else None
        count = len(self.input_batch.req_ids)
        eligible = (
            first is not None
            and first.attn_state == AscendAttentionState.DecodeOnly
            and first.max_query_len == 1
            and not first.num_prefills
            and first.num_decodes >= count > 0
        )
        if not eligible:
            return forward(self, *positional, **keywords)
        lengths = tuple(first.seq_lens_list[:count])
        step = {
            "actual_requests": count,
            "kv_lengths": lengths,
            "dispatched_capacity": getattr(context.batch_descriptor, "num_tokens", None),
            "lane_sizes": None,
            "single_replays": [],
            "fallback": [],
        }
        state["step"] = step
        try:
            if args.measurement == "decode":
                torch.npu.synchronize()
                start = time.perf_counter_ns()
            result = forward(self, *positional, **keywords)
            if args.measurement == "decode":
                torch.npu.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
                step["elapsed_us"] = elapsed
                samples.append({"kv_lengths": lengths, "elapsed_us": elapsed, "lane_sizes": step["lane_sizes"]})
        finally:
            state["step"] = None
        capacity = sum(step["lane_sizes"]) if step["lane_sizes"] else sum(step["single_replays"])
        step["padding_tokens"] = max(0, capacity - count) if capacity else 0
        step["padding_fraction"] = step["padding_tokens"] / capacity if capacity else 0.0
        steps.append(step)
        return result

    NPUModelRunner._model_forward = observed_forward
    ACLGraphWrapper._updatable_graph_replay = observed_replay
    MultiGraphV1Adapter.try_run = observed_split
    config = {
        "ascend_compilation_config": {"enable_npugraph_ex": False},
        "pa_shape_list": [],
        "multi_graph_config": {"kv_cache_layout": args.kv_cache_layout},
    }
    if args.mode in ("serial", "parallel"):
        config["multi_graph_config"].update(
            enabled=True,
            execution=args.mode,
            selection_policy="force",
            capture_sizes_per_lane=[[size] for size in args.lane_sizes],
            max_graph_entries=2,
            graph_memory_budget_bytes=args.graph_budget,
        )
    startup = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.single_capacity,
        max_num_batched_tokens=args.max_batched_tokens,
        block_size=128,
        num_gpu_blocks_override=args.kv_blocks,
        enable_prefix_caching=args.prefix_cache == "on",
        seed=args.seed,
        enforce_eager=args.mode == "eager",
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [args.single_capacity],
        },
        additional_config=config,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    startup_s = time.perf_counter() - startup
    tokenizer = llm.get_tokenizer()
    base_ids = tokenizer.encode("The algorithm uses simple steps.", add_special_tokens=False)
    if not base_ids:
        raise ValueError("tokenizer produced an empty synthetic seed")
    prompts = [
        {"prompt_token_ids": (base_ids * ((length + len(base_ids) - 1) // len(base_ids)))[:length]}
        for length in args.prompt_lengths
    ]
    sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True, logprobs=1)
    for _ in range(args.warmups):
        llm.generate(prompts, sampling, use_tqdm=False)
    torch.npu.synchronize()
    runner = state["runner"]
    if runner.kv_cache_config.num_blocks != args.kv_blocks:
        raise ValueError("actual KV capacity differs from requested common block count")
    memory_before = {
        "allocated_bytes": torch.npu.memory_allocated(),
        "reserved_bytes": torch.npu.memory_reserved(),
    }
    torch.npu.reset_peak_memory_stats()
    state["recording"] = True
    timings, generations, token_counts = [], [], []
    repeats = 1 if args.measurement == "trace" else args.repeats
    profiler = None
    if args.measurement == "trace":
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        )
        profiler.start()
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.npu.synchronize()
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
        if len(outputs) != args.batch_size:
            raise ValueError("generation returned an incomplete request batch")
        requests = []
        for result in outputs:
            output = result.outputs[0]
            if len(output.token_ids) != args.output_tokens:
                raise ValueError("generation length differs from requested workload")
            requests.append(
                {
                    "tokens": list(output.token_ids),
                    "selected_logprobs": [
                        float(step[token].logprob) for step, token in zip(output.logprobs, output.token_ids)
                    ],
                }
            )
        generations.append(requests)
        token_counts.append(sum(len(request["tokens"]) for request in requests))
    state["recording"] = False
    trace_info = None
    if profiler is not None:
        profiler.step()
        profiler.stop()
        trace_path = Path(args.result).with_suffix(".trace.json")
        profiler.export_chrome_trace(str(trace_path))
        trace_info = inspect_device_trace(trace_path)
    if args.mode in ("serial", "parallel") and not state["split_steps"]:
        raise ValueError("test never entered split execution")
    if args.measurement != "generation":
        expected_steps = repeats * (args.output_tokens - 1) if args.prefix_cache == "off" else None
        verify_decode_steps(steps, args.mode, args.batch_size, args.single_capacity, args.lane_sizes, expected_steps)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    resource = {
        "kv_blocks": runner.kv_cache_config.num_blocks,
        "block_size": 128,
        "memory_after_warmup": memory_before,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        "graph_reservation_bytes": args.graph_budget if args.mode in ("serial", "parallel") else 0,
        "startup_seconds": startup_s,
    }
    driver_path = Path("/usr/local/Ascend/driver/version.info")
    cann_path = Path("/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info")
    payload = {
        "schema_version": 2,
        "mode": args.mode,
        "measurement": args.measurement,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "single_capacity": args.single_capacity,
        "lane_sizes": args.lane_sizes,
        "kv_cache_layout": args.kv_cache_layout,
        "prompt_lengths": args.prompt_lengths,
        "prefix_cache": args.prefix_cache,
        "max_batched_tokens": args.max_batched_tokens,
        "output_tokens": args.output_tokens,
        "warmups": args.warmups,
        "seed": args.seed,
        "latency_ms": timings,
        "generation_latency_ms": summarize(timings),
        "output_tokens_per_second": sum(token_counts) / (sum(timings) / 1000),
        "generation_metric": "synchronized_decode_instrumented_generation"
        if args.measurement == "decode"
        else "profiled_generation"
        if args.measurement == "trace"
        else "whole_generation_wall_time",
        "split_steps": state["split_steps"],
        "requests": generations[-1],
        "generation_outputs": generations,
        "decode_steps": steps,
        "decode_metric": "synchronized_decode_forward_us" if args.measurement == "decode" else None,
        "decode_samples": samples,
        "capture_signatures": signatures,
        "trace": trace_info,
        "decode_forward_us": summarize([sample["elapsed_us"] for sample in samples]) if samples else None,
        "resources": resource,
        "environment": {
            "source_commit": revision,
            "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "architecture": platform.machine(),
            "os": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
            },
            "device": torch.npu.get_device_name(),
            "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "driver_info": driver_path.read_text() if driver_path.exists() else None,
            "cann_info": cann_path.read_text() if cann_path.exists() else None,
        },
    }
    Path(args.result).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--single-capacity", type=int)
    parser.add_argument("--lane-sizes", type=int, nargs=2)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--prompt-spread", type=int, default=0)
    parser.add_argument("--prompt-pattern", choices=("uniform", "staggered", "bimodal"), default="uniform")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-batched-tokens", type=int)
    parser.add_argument("--kv-blocks", type=int)
    parser.add_argument("--prefix-cache", choices=("off", "on"), default="off")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--graph-budget", type=int, default=12 << 30)
    parser.add_argument("--kv-cache-layout", choices=("default", "contiguous"), default="default")
    parser.add_argument("--mode", choices=("eager", "single", "serial", "parallel"))
    parser.add_argument(
        "--mode-order",
        nargs=4,
        choices=("eager", "single", "serial", "parallel"),
        default=["eager", "single", "serial", "parallel"],
    )
    parser.add_argument("--measurement", choices=("generation", "decode", "trace"), default="generation")
    parser.add_argument("--result")
    parser.add_argument("--trace", action="store_true", help="run a separate profiler process per mode")
    parser.add_argument("--decode-costs", action="store_true", help="run separate synchronized decode processes")
    parser.add_argument("--output-dir", type=Path, default=Path("multi_graph_results"))
    args = parser.parse_args()
    try:
        resolve_workload(args)
        if len(set(args.mode_order)) != 4:
            raise ValueError("mode-order must contain each mode exactly once")
    except ValueError as error:
        parser.error(str(error))
    if args.mode:
        if not args.result:
            parser.error("--result is required in child mode")
        Path(args.result).parent.mkdir(parents=True, exist_ok=True)
        run_child(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forwarded = [
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--single-capacity",
        str(args.single_capacity),
        "--lane-sizes",
        *map(str, args.lane_sizes),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--prompt-spread",
        str(args.prompt_spread),
        "--prompt-pattern",
        args.prompt_pattern,
        "--output-tokens",
        str(args.output_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-batched-tokens",
        str(args.max_batched_tokens),
        "--kv-blocks",
        str(args.kv_blocks),
        "--prefix-cache",
        args.prefix_cache,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--graph-budget",
        str(args.graph_budget),
        "--kv-cache-layout",
        args.kv_cache_layout,
    ]
    summary = {"workload": vars(args).copy(), "measurements": {}}
    summary["workload"]["output_dir"] = str(args.output_dir)
    passes = ["generation", *(["decode"] if args.decode_costs else []), *(["trace"] if args.trace else [])]
    for measurement in passes:
        results = {}
        for mode in args.mode_order:
            stem = mode if measurement == "generation" else f"{mode}.{measurement}"
            result = args.output_dir / f"{stem}.json"
            with (args.output_dir / f"{stem}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        *forwarded,
                        "--mode",
                        mode,
                        "--measurement",
                        measurement,
                        "--result",
                        str(result),
                    ],
                    check=True,
                    timeout=1800,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
                )
            results[mode] = json.loads(result.read_text(encoding="utf-8"))
            print(f"Completed {measurement}: {mode}", flush=True)
        if any(result["resources"]["kv_blocks"] != args.kv_blocks for result in results.values()):
            raise ValueError("resource fairness check failed")
        correctness = verify_outputs(results, args.dtype)
        baseline = results["single"]["generation_latency_ms"]["p50"]
        summary["measurements"][measurement] = {
            "correctness": correctness,
            "results": {
                mode: {
                    "generation_latency_ms": result["generation_latency_ms"],
                    "output_tokens_per_second": result["output_tokens_per_second"],
                    "decode_forward_us": result["decode_forward_us"],
                    "resources": result["resources"],
                    "generation_speedup_vs_single": baseline / result["generation_latency_ms"]["p50"]
                    if measurement == "generation"
                    else None,
                }
                for mode, result in results.items()
            },
        }
        if measurement == "decode":
            from pydantic import TypeAdapter

            from vllm_ascend.worker.multi_graph.costs import DecodeForwardSample, build_cost_document
            from vllm_ascend.worker.multi_graph.types import CaptureSignature

            for mode in ("serial", "parallel"):
                result = results[mode]
                if len(result["capture_signatures"]) != 1:
                    raise ValueError("cost export requires one capture identity")
                signature = TypeAdapter(CaptureSignature).validate_python(result["capture_signatures"][0])
                adapter = TypeAdapter(tuple[DecodeForwardSample, ...])
                document = build_cost_document(
                    signature,
                    mode,
                    adapter.validate_python(results["single"]["decode_samples"]),
                    adapter.validate_python(result["decode_samples"]),
                )
                (args.output_dir / f"{mode}.costs.json").write_text(
                    document.model_dump_json(indent=2), encoding="utf-8"
                )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
