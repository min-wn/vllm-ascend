"""Diagnostic projection sweeps using real model weights and default batch buckets.

Independent synthetic inputs isolate Dense cost; this is not decoder performance.
Each graph visits every layer's weight once, avoiding a one-weight hot-cache loop.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def experiment(args, workload, runner, model, inputs, context, reset_kv):
    # No attention or task update is present in this diagnostic workload.
    import torch
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
    from vllm_ascend.worker.multi_graph.benchmark import summarize

    main = torch.npu.current_stream()
    layers = model.model.layers
    projections = {
        "qkv": [layer.self_attn.qkv_proj for layer in layers],
        "o": [layer.self_attn.o_proj for layer in layers],
        "gate_up": [layer.mlp.gate_up_proj for layer in layers],
        "down": [layer.mlp.down_proj for layer in layers],
    }
    results = []
    for name, modules in projections.items():
        if any(module.bias is not None for module in modules):
            raise ValueError("probe requires bias-free projections")
        weights = [module.weight for module in modules]
        if any(w.shape != weights[0].shape or w.dtype != torch.bfloat16 for w in weights):
            raise ValueError("probe requires uniform BF16 weights")
        reference_weights = weights
        if args.weight_format == "nz":
            weights = [torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ) for weight in weights]
        entries = []
        for capacity in (workload.capacity, *args.pair):
            stream = torch.npu.Stream()
            data = torch.randn((capacity, weights[0].shape[1]), device="npu", dtype=torch.bfloat16)
            stream.wait_stream(main)
            with torch.npu.stream(stream):
                expected = [torch.nn.functional.linear(data, weight) for weight in reference_weights]
            stream.synchronize()
            graph, pool = torch.npu.NPUGraph(), torch.npu.graph_pool_handle()
            with torch.npu.graph(graph, pool=pool, stream=stream):
                outputs = [torch.nn.functional.linear(data, weight) for weight in weights]
            stream.synchronize()
            graph.replay()
            torch.npu.synchronize()
            for observed, reference in zip(outputs, expected):
                torch.testing.assert_close(observed, reference, rtol=0.02, atol=0.02)
            entries.append(
                dict(graph=graph, pool=pool, stream=stream, data=data, outputs=outputs, done=torch.npu.Event())
            )

        def replay(mode, captured=entries):
            active = (
                captured[1:] if mode == "split_parallel" else [captured[{"padding": 0, "lane0": 1, "lane1": 2}[mode]]]
            )
            for entry in active:
                with torch.npu.stream(entry["stream"]):
                    entry["graph"].replay()
                entry["done"].record(entry["stream"])
            for entry in active:
                main.wait_event(entry["done"])

        modes = ("padding", "lane0", "lane1", "split_parallel")
        for mode in modes:
            for _ in range(5):
                replay(mode)
                torch.npu.synchronize()
        measurements = []
        for repeat in range(args.rounds):
            for mode in modes if repeat % 2 == 0 else reversed(modes):
                samples = []
                for _ in range(args.repeats):
                    started = time.perf_counter_ns()
                    replay(mode)
                    torch.npu.synchronize()
                    samples.append((time.perf_counter_ns() - started) / 1000)
                measurements.append(dict(mode=mode, round=repeat, samples_us=samples, summary_us=summarize(samples)))
        # Validate concurrent replay against the original loaded-weight reference.
        for entry in entries:
            for observed, weight in zip(entry["outputs"], reference_weights):
                torch.testing.assert_close(
                    observed, torch.nn.functional.linear(entry["data"], weight), rtol=0.02, atol=0.02
                )
        if args.trace:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
            ) as profiler:
                for mode in modes:
                    for _ in range(3):
                        with torch.profiler.record_function(f"dense.{name}.{mode}"):
                            replay(mode)
                            torch.npu.synchronize()
            profiler.export_chrome_trace(str(args.output.with_suffix(f".{name}.trace.json")))
        row = dict(
            projection=name,
            weight_shape=list(weights[0].shape),
            weight_stride=list(weights[0].stride()),
            weight_format=torch_npu.get_npu_format(weights[0]),
            layers=len(weights),
            logical_weight_bytes=sum(w.numel() * w.element_size() for w in weights),
            measurements=measurements,
            summary_ms={
                mode: statistics.median(r["summary_us"]["p50"] for r in measurements if r["mode"] == mode) / 1000
                for mode in modes
            },
            passed=True,
        )
        results.append(row)
        print(name, row["summary_ms"], flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                selection=args.selection,
                pair=args.pair,
                source_sha256=args.source_sha256,
                requested_weight_format=args.weight_format,
                scope="synthetic_input_real_weight_projection_sweep_not_decoder",
                results=results,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    from multi_graph_bucket_replay import bucket_pair, lane_counts
    from multi_graph_default_capture import select_case
    from multi_graph_replay import run

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", choices=("exact", "balanced"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--weight-format", choices=("loaded", "nz"), default="loaded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rounds, args.repeats) < 1 or args.output.exists():
        parser.error("positive counts and a new output path required")
    args.seed, args.batch_min, args.batch_max, args.max_num_seqs = 20261016, 64, 256, 256
    args.kv_length, args.dtype, args.batch_invariant = 512, "bfloat16", False
    args.selection = select_case(args)
    capacity, args.pair = bucket_pair(args.selection["batch"], args.selection["capture_sizes"], args.strategy)
    args.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    workload = ReplayWorkload(
        args.selection["batch"], capacity, 1, 512, lane_counts(args.selection["batch"], args.pair)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="running")), encoding="utf-8")
    try:
        run(args, workload, experiment=experiment)
    except Exception as error:
        args.output.write_text(json.dumps(dict(status="failed", error=str(error))), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
