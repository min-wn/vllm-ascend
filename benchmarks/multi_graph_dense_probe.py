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
        canonical = (
            torch.randn((workload.capacity, weights[0].shape[1]), device="npu", dtype=torch.bfloat16)
            if args.shared_input_probe
            else None
        )
        for index, capacity in enumerate((workload.capacity, *args.pair)):
            stream = torch.npu.Stream()
            offset = args.pair[0] if index == 2 else 0
            data = (
                canonical[offset : offset + capacity].clone()
                if args.shared_input_probe
                else torch.randn((capacity, weights[0].shape[1]), device="npu", dtype=torch.bfloat16)
            )
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

        ready = torch.npu.Event()
        # Aliases are created once, independently of the number of live requests.
        shared_views = [
            [output[: args.pair[0]], output[args.pair[0] : sum(args.pair)]] for output in entries[0]["outputs"]
        ]
        staged = []
        if args.shared_input_probe:
            # These graphs replay serially in capture order on the same stream.
            # Share one pool instead of reserving a segment for every layer.
            staged_pool = torch.npu.graph_pool_handle()
            for weight in weights:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=staged_pool, stream=entries[0]["stream"]):
                    output = torch.nn.functional.linear(entries[0]["data"], weight)
                staged.append(dict(graph=graph, pool=staged_pool, output=output))
            entries[0]["stream"].synchronize()

        def replay(mode, captured=entries, ready=ready, staged=staged):
            if mode == "shared_staged":
                # Each layer pays its own gather, graph launch and dependency edges.
                # This control exposes costs hidden by a 36-weight monolithic sweep.
                entry = captured[0]
                for stage in staged:
                    entry["data"][: args.pair[0]].copy_(captured[1]["data"])
                    entry["data"][args.pair[0] : sum(args.pair)].copy_(captured[2]["data"])
                    ready.record(main)
                    entry["stream"].wait_event(ready)
                    with torch.npu.stream(entry["stream"]):
                        stage["graph"].replay()
                    entry["done"].record(entry["stream"])
                    main.wait_event(entry["done"])
                return
            if mode == "shared_bucket":
                captured[0]["data"][: args.pair[0]].copy_(captured[1]["data"])
                captured[0]["data"][args.pair[0] : sum(args.pair)].copy_(captured[2]["data"])
                ready.record(main)
                captured[0]["stream"].wait_event(ready)
            active = (
                captured[1:]
                if mode == "split_parallel"
                else [captured[{"padding": 0, "shared_bucket": 0, "lane0": 1, "lane1": 2}[mode]]]
            )
            for entry in active:
                with torch.npu.stream(entry["stream"]):
                    entry["graph"].replay()
                entry["done"].record(entry["stream"])
            for entry in active:
                main.wait_event(entry["done"])

        modes = (
            "padding",
            "lane0",
            "lane1",
            "split_parallel",
            *(("shared_bucket", "shared_staged") if args.shared_input_probe else ()),
        )
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
        if args.shared_input_probe:
            for branches, weight in zip(shared_views, reference_weights):
                for observed, entry in zip(branches, entries[1:]):
                    torch.testing.assert_close(
                        observed, torch.nn.functional.linear(entry["data"], weight), rtol=0.02, atol=0.02
                    )
            for stage, weight in zip(staged, reference_weights):
                torch.testing.assert_close(
                    stage["output"], torch.nn.functional.linear(entries[0]["data"], weight), rtol=0.02, atol=0.02
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
            shared_bucket_capacity=workload.capacity if args.shared_input_probe else None,
            shared_bucket_reuses_padding_graph=args.shared_input_probe,
            shared_bucket_includes_input_copies=args.shared_input_probe,
            shared_bucket_output_alias_checks=len(shared_views) * 2 if args.shared_input_probe else 0,
            shared_staged_graph_count=len(staged),
            shared_staged_capture_capacity=workload.capacity if staged else None,
            shared_staged_input_gathers_per_replay=len(staged),
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
                shared_input_probe=args.shared_input_probe,
                shared_bucket_scope="default_capacity_dense_only_no_active_row_skipping_no_attention_or_serving",
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
    parser.add_argument("--shared-input-probe", action="store_true")
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
