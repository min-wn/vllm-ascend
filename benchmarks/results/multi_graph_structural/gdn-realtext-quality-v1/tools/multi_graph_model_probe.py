"""Check official model execution before attempting architecture-specific splitting.

This is a compatibility probe, not a split/padding performance benchmark.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        scope="official_compatibility_only_not_split_performance",
        graph_requested=args.graph,
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            enforce_eager=not args.graph,
            max_model_len=512,
            max_num_seqs=8,
            max_num_batched_tokens=512,
            gpu_memory_utilization=0.6,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=42,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE",
                "cudagraph_capture_sizes": [1, 2, 4, 8] if args.graph else [],
            },
            additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
        )
        results = llm.generate(
            ["The capital of France is", "One plus one equals"], SamplingParams(temperature=0, max_tokens=2)
        )
        report.update(status="passed", outputs=[result.outputs[0].text for result in results])
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
