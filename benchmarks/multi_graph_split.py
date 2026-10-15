"""Run each mode in a fresh process to compare exact decode correctness and latency."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_child(args) -> None:
    import torch
    import torch_npu
    from vllm import LLM, SamplingParams
    from vllm.forward_context import get_forward_context

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.multi_graph.runner_v1 import MultiGraphV1Adapter

    original = MultiGraphV1Adapter.try_run
    counts = {"split_steps": 0}
    samples = []
    capture_signatures = []
    recording = False

    forward = NPUModelRunner._model_forward

    def timed_forward(self, *positional, **keywords):
        context = get_forward_context()
        metadata = context.attn_metadata
        first = next(iter(metadata.values())) if isinstance(metadata, dict) and metadata else None
        count = len(self.input_batch.req_ids)
        eligible = (
            args.decode_costs
            and recording
            and first is not None
            and first.attn_state == AscendAttentionState.DecodeOnly
            and first.max_query_len == 1
            and not first.num_prefills
            and first.num_decodes >= count > 0
        )
        if not eligible:
            return forward(self, *positional, **keywords)
        lengths = tuple(first.seq_lens_list[:count])
        adapter = getattr(self, "multi_graph_adapter", None)
        previous_count = adapter.execution_count if adapter is not None else 0
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        result = forward(self, *positional, **keywords)
        torch.npu.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1000
        sizes = None
        if adapter is not None and adapter.execution_count > previous_count:
            sizes = [part.actual_tokens for part in adapter.last_plan.slices]
            if not capture_signatures:
                capture_signatures.append(dataclasses.asdict(adapter.signature))
        samples.append({"kv_lengths": lengths, "elapsed_us": elapsed, "lane_sizes": sizes})
        return result

    NPUModelRunner._model_forward = timed_forward

    def counted_forward(self, *positional, **keywords):
        result = original(self, *positional, **keywords)
        if result is not None:
            counts["split_steps"] += 1
        return result

    MultiGraphV1Adapter.try_run = counted_forward
    config = {
        "ascend_compilation_config": {"enable_npugraph_ex": False},
        "pa_shape_list": [],
    }
    if args.mode in ("serial", "parallel"):
        config["multi_graph_config"] = {
            "enabled": True,
            "execution": args.mode,
            "selection_policy": "force",
            "capture_sizes_per_lane": [[2, 3, 4], [2, 3, 4]],
            "max_graph_entries": 6,
            "graph_memory_budget_bytes": args.graph_budget,
        }
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        seed=0,
        enforce_eager=args.mode == "eager",
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2, 4, 6, 8]},
        additional_config=config,
        gpu_memory_utilization=0.8,
    )
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, logprobs=1)
    prompts = ["Explain " + "a simple algorithm " * (index + 1) for index in range(args.batch_size)]
    llm.generate(prompts, sampling, use_tqdm=False)
    counts["split_steps"] = 0
    recording = True
    timings = []
    outputs = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        timings.append((time.perf_counter() - start) * 1000)
    assert outputs is not None
    recording = False
    if args.mode in ("serial", "parallel") and counts["split_steps"] == 0:
        raise AssertionError("test never entered multi-graph execution")
    requests = []
    for result in outputs:
        output = result.outputs[0]
        requests.append(
            {
                "tokens": list(output.token_ids),
                "selected_logprobs": [
                    float(step[token].logprob) for step, token in zip(output.logprobs, output.token_ids)
                ],
            }
        )
    if args.trace:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            llm.generate(prompts, sampling, use_tqdm=False)
        profiler.export_chrome_trace(str(Path(args.result).with_suffix(".trace.json")))
    Path(args.result).write_text(
        json.dumps(
            {
                "mode": args.mode,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "latency_ms": timings,
                "split_steps": counts["split_steps"],
                "requests": requests,
                "torch_version": torch.__version__,
                "torch_npu_version": torch_npu.__version__,
                "device": torch.npu.get_device_name(),
                "decode_metric": "synchronized_decode_forward_us" if args.decode_costs else None,
                "decode_samples": samples,
                "capture_signatures": capture_signatures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, choices=(6, 7, 8), default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph-budget", type=int, default=1 << 30)
    parser.add_argument("--mode", choices=("eager", "single", "serial", "parallel"))
    parser.add_argument("--result")
    parser.add_argument("--trace", action="store_true", help="export a separate NPU profiler run for each mode")
    parser.add_argument("--decode-costs", action="store_true", help="synchronize each decode forward and export costs")
    parser.add_argument("--output-dir", type=Path, default=Path("multi_graph_results"))
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.mode:
        if not args.result:
            parser.error("--result is required in child mode")
        run_child(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("eager", "single", "serial", "parallel"):
        result = args.output_dir / f"{mode}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                args.model,
                "--dtype",
                args.dtype,
                "--batch-size",
                str(args.batch_size),
                "--repeats",
                str(args.repeats),
                "--graph-budget",
                str(args.graph_budget),
                "--mode",
                mode,
                "--result",
                str(result),
                *(["--trace"] if args.trace else []),
                *(["--decode-costs"] if args.decode_costs else []),
            ],
            check=True,
            timeout=600,
            env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
        )
    if args.decode_costs:
        from pydantic import TypeAdapter

        from vllm_ascend.worker.multi_graph.costs import DecodeForwardSample, build_cost_document
        from vllm_ascend.worker.multi_graph.types import CaptureSignature

        baseline = json.loads((args.output_dir / "single.json").read_text(encoding="utf-8"))
        for mode in ("serial", "parallel"):
            result = json.loads((args.output_dir / f"{mode}.json").read_text(encoding="utf-8"))
            baseline_tokens = [request["tokens"] for request in baseline["requests"]]
            split_tokens = [request["tokens"] for request in result["requests"]]
            if baseline_tokens != split_tokens:
                raise ValueError("cost export requires identical baseline/split output tokens")
            if len(result["capture_signatures"]) != 1:
                raise ValueError("cost export requires one capture identity")
            signature = TypeAdapter(CaptureSignature).validate_python(result["capture_signatures"][0])
            sample_adapter = TypeAdapter(tuple[DecodeForwardSample, ...])
            document = build_cost_document(
                signature,
                mode,
                sample_adapter.validate_python(baseline["decode_samples"]),
                sample_adapter.validate_python(result["decode_samples"]),
            )
            (args.output_dir / f"{mode}.costs.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
