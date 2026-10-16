"""Full target-model replay with fixed query/KV lengths; no draft or sampling.

The default protocol retains attention task updates and their event dependencies.
Optional frozen FIA graphs omit updates only because all metadata is immutable.
This benchmark never relaxes the production multi-graph admission checks.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from vllm_ascend.worker.multi_graph.benchmark import summarize
from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload, inspect_replay_trace


def run(args, workload):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    import torch
    import torch_npu
    import vllm.envs as vllm_envs
    from vllm import LLM
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.multi_graph.runner_v1 import disable_compilation

    class FrozenGraph(UpdatableGraph):
        def __init__(self):
            super().__init__()
            self.frozen_tasks = []

        def register_task(self, operation, kwargs, provider):
            # Keep tensors/workspaces alive; there are no mutable host parameters.
            self.frozen_tasks.append((operation, kwargs))
            operation(**kwargs)

    observed = []
    original = NPUModelRunner._model_forward

    def observe(self, *values, **options):
        if not observed:
            observed.append(self)
        return original(self, *values, **options)

    NPUModelRunner._model_forward = observe
    try:
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            enforce_eager=True,
            max_model_len=max(512, workload.kv_length + workload.query),
            max_num_seqs=workload.capacity,
            max_num_batched_tokens=max(512, workload.capacity * workload.query),
            block_size=128,
            num_gpu_blocks_override=max(256, workload.kv_blocks),
            enable_prefix_caching=False,
            gpu_memory_utilization=0.8,
            seed=42,
            additional_config={
                "ascend_compilation_config": {"enable_npugraph_ex": False},
                "pa_shape_list": [],
                "multi_graph_config": {"kv_cache_layout": "contiguous"},
            },
        )
    finally:
        NPUModelRunner._model_forward = original
    runner = observed[0]
    model = runner.get_model()
    layers = runner.compilation_config.static_forward_context
    if runner.model_config.hf_config.model_type != "qwen2" or len(runner.kv_cache_config.kv_cache_groups) != 1:
        raise ValueError("this qualification is limited to dense Qwen2 with one KV group")
    torch.manual_seed(42)
    caches = []
    pointers = set()
    for layer in layers.values():
        for cache in getattr(layer, "kv_cache", ()):
            if cache.data_ptr() not in pointers:
                pointers.add(cache.data_ptr())
                cache.zero_()
                caches.append((cache, cache.clone()))
    if not caches:
        raise ValueError("no KV storage found")

    def reset_kv():
        torch.npu.synchronize()
        for cache, initial in caches:
            cache.copy_(initial)
        torch.npu.synchronize()

    def inputs(start, stop):
        values = workload.graph_values(start, stop)
        boundaries = torch.tensor(values["boundaries"], dtype=torch.int32)
        lengths = torch.tensor(values["lengths"], dtype=torch.int32)
        metadata = AscendMetadata(
            attn_mask=torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu_(1),
            attn_state=AscendAttentionState.DecodeOnly if workload.query == 1 else AscendAttentionState.ChunkedPrefill,
            num_actual_tokens=(stop - start) * workload.query,
            num_decode_tokens=(stop - start) * workload.query,
            num_decodes=stop - start,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_list=values["lengths"],
            seq_lens_gpu=lengths.npu(),
            actual_seq_lengths_q=values["boundaries"][1:],
            query_start_loc=boundaries,
            query_start_loc_gpu=boundaries.npu(),
            max_query_len=workload.query,
            block_tables=torch.tensor(values["pages"], dtype=torch.int32, device="npu"),
            slot_mapping=torch.tensor(values["slots"], dtype=torch.int64, device="npu"),
            model_runner_type="generate",
        )
        ids = torch.arange(start * workload.query, stop * workload.query, device="npu", dtype=torch.int64) + 100
        # TND query/output extents match capacity; dummy rows have zero KV and slot -1.
        positions = torch.tensor(values["positions"], dtype=torch.int64, device="npu")
        return ids, positions, {name: metadata for name in layers}

    def context(data):
        tokens = data[0].numel()
        return set_ascend_forward_context(
            data[2],
            runner.vllm_config,
            num_tokens=tokens,
            num_actual_tokens=next(iter(data[2].values())).num_actual_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(
                num_tokens=tokens,
                num_reqs=len(next(iter(data[2].values())).seq_lens_list),
                uniform=True,
            ),
            model_instance=model,
            skip_compiled=True,
        )

    def forward(data, capturing=False):
        with context(data):
            get_forward_context().capturing = capturing
            return model(input_ids=data[0], positions=data[1])

    # Build coherent model KV outside timing; random per-layer KV is not a prefill.
    prefix_length = workload.kv_length - workload.query
    if prefix_length:
        base = next(iter(inputs(0, 1)[2].values()))
        lengths = torch.tensor([prefix_length], dtype=torch.int32)
        boundaries = torch.tensor([0, prefix_length], dtype=torch.int32)
        prefix_metadata = dataclasses.replace(
            base,
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_actual_tokens=prefix_length,
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=1,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_list=[prefix_length],
            seq_lens_gpu=lengths.npu(),
            actual_seq_lengths_q=[prefix_length],
            query_start_loc=boundaries,
            query_start_loc_gpu=boundaries.npu(),
            max_query_len=prefix_length,
            slot_mapping=torch.arange(prefix_length, dtype=torch.int64, device="npu"),
        )
        seed_ids = llm.get_tokenizer().encode("The algorithm uses simple steps.", add_special_tokens=False)
        prefix_ids = (seed_ids * ((prefix_length + len(seed_ids) - 1) // len(seed_ids)))[:prefix_length]
        prefix_data = (
            torch.tensor(prefix_ids, dtype=torch.int64, device="npu"),
            torch.arange(prefix_length, dtype=torch.int64, device="npu"),
            {name: prefix_metadata for name in layers},
        )
        with torch.inference_mode(), disable_compilation(model):
            forward(prefix_data)
        torch.npu.synchronize()
        width = workload.pages_per_request
        for cache, initial in caches:
            if cache.shape[0] < workload.kv_blocks:
                raise ValueError("expected page-major contiguous KV")
            for request in range(1, workload.capacity):
                cache[request * width : (request + 1) * width].copy_(cache[:width])
            initial.copy_(cache)
        torch.npu.synchronize()
    actual = inputs(0, workload.batch)
    reset_kv()
    with torch.inference_mode(), disable_compilation(model):
        reference = forward(actual).clone()
    torch.npu.synchronize()
    reference_cpu = reference.float().cpu()
    controls = []
    with torch.inference_mode(), disable_compilation(model):
        for label, data, capture_path in (
            ("eager_repeat", actual, False),
            ("fia_out", actual, True),
            ("eager_padded", inputs(0, workload.capacity), False),
        ):
            reset_kv()
            control = forward(data, capture_path)[: workload.batch * workload.query].float().cpu()
            print(f"control {label}: max_abs_error={(control - reference_cpu).abs().max().item()}", flush=True)
            torch.testing.assert_close(control, reference_cpu, rtol=0.02, atol=0.02)
            controls.append(dict(name=label, max_abs_error=(control - reference_cpu).abs().max().item(), passed=True))
    rows, correctness, resources = [], [], []
    trace_analysis = None
    captured = {}
    with torch.inference_mode(), disable_compilation(model):
        merged = torch.empty_like(reference)
        for protocol in args.protocols:
            entries = []
            for graph_index, (start, stop) in enumerate(
                ((0, workload.capacity), (0, workload.lanes[0]), (workload.lanes[0], workload.batch))
            ):
                reset_kv()
                data = inputs(start, stop)
                stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        forward(data)
                stream.synchronize()
                graph = FrozenGraph() if protocol == "frozen" else UpdatableGraph()
                pool = torch.npu.graph_pool_handle()
                began = time.perf_counter()
                with context(data):
                    get_forward_context().capturing = True
                    with torch.npu.graph(graph, pool=pool, stream=stream):
                        output = model(input_ids=data[0], positions=data[1])
                        if graph_index and args.merge == "captured":
                            # Disjoint request slices; each lane's done event covers its copy.
                            merged[start * workload.query : stop * workload.query].copy_(output)
                stream.synchronize()
                registered_tasks = graph.tasks if protocol == "update" else graph.frozen_tasks
                if len(registered_tasks) != len(layers):
                    raise ValueError("expected exactly one FIA task per Qwen2 attention layer")
                entries.append(
                    dict(
                        data=data,
                        graph=graph,
                        pool=pool,
                        stream=stream,
                        update=update_stream,
                        output=output,
                        tasks=graph.resolve_tasks(ContextSource(data[2])),
                        done=torch.npu.Event(),
                        updated=torch.npu.Event(),
                    )
                )
                resources.append(
                    dict(
                        protocol=protocol,
                        requests=stop - start,
                        capture_seconds=time.perf_counter() - began,
                        update_tasks=len(graph.tasks),
                        frozen_tasks=len(getattr(graph, "frozen_tasks", ())),
                    )
                )
            captured[protocol] = entries
        ready = torch.npu.Event()
        main = torch.npu.current_stream()

        def replay(protocol, mode):
            entries = captured[protocol][:1] if mode == "single" else captured[protocol][1:]
            ready.record(main)
            for index, entry in enumerate(entries):
                entry["stream"].wait_event(ready)
                if mode == "serial" and index:
                    entry["stream"].wait_event(entries[index - 1]["done"])
                with torch.npu.stream(entry["stream"]):
                    entry["graph"].replay()
                if protocol == "update":
                    entry["update"].wait_event(ready)
                    entry["graph"].update(entry["update"], entry["tasks"])
                    entry["updated"].record(entry["update"])
                    entry["stream"].wait_event(entry["updated"])
                entry["done"].record(entry["stream"])
            for entry in entries:
                main.wait_event(entry["done"])
            if mode == "single":
                return entries[0]["output"][: workload.batch * workload.query]
            if args.merge == "cat":
                torch.cat([entry["output"] for entry in entries], out=merged)
            return merged

        def check(protocol, mode):
            reset_kv()
            result = replay(protocol, mode)
            torch.npu.synchronize()
            observed_cpu = result.float().cpu()
            error = (observed_cpu - reference_cpu).abs().max().item()
            print(f"check {protocol}/{mode}: max_abs_error={error}", flush=True)
            torch.testing.assert_close(observed_cpu, reference_cpu, rtol=0.02, atol=0.02)
            correctness.append(dict(protocol=protocol, mode=mode, max_abs_error=error, passed=True))

        combinations = [(p, m) for p in args.protocols for m in ("single", "serial", "parallel")]
        for protocol, mode in combinations:
            check(protocol, mode)
            for _ in range(args.warmups):
                replay(protocol, mode)
                torch.npu.synchronize()
        start_event, end_event = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for round_id in range(args.rounds):
            order = combinations if round_id % 2 == 0 else list(reversed(combinations))
            for protocol, mode in order:
                wall, device = [], []
                reset_kv()
                for _ in range(args.repeats):
                    start_event.record(main)
                    began = time.perf_counter_ns()
                    result = replay(protocol, mode)
                    end_event.record(main)
                    end_event.synchronize()
                    wall.append((time.perf_counter_ns() - began) / 1000)
                    device.append(start_event.elapsed_time(end_event) * 1000)
                torch.testing.assert_close(result.float().cpu(), reference_cpu, rtol=0.02, atol=0.02)
                rows.append(
                    dict(
                        round=round_id,
                        protocol=protocol,
                        mode=mode,
                        wall_us=wall,
                        device_interval_us=device,
                        wall_summary_us=summarize(wall),
                        device_summary_us=summarize(device),
                    )
                )
                print(f"{protocol}/{mode} round={round_id}: {summarize(wall)['p50']:.2f} us", flush=True)
        if args.trace:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            ) as profiler:
                for protocol, mode in combinations:
                    for _ in range(3):
                        with torch.profiler.record_function(f"replay.{protocol}.{mode}"):
                            replay(protocol, mode)
                            torch.npu.synchronize()
            profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
            trace_analysis = inspect_replay_trace(json.loads(args.output.with_suffix(".trace.json").read_text()))
    payload = dict(
        status="passed",
        workload=dataclasses.asdict(workload),
        model=args.model,
        dtype=args.dtype,
        batch_invariant=bool(vllm_envs.VLLM_BATCH_INVARIANT),
        merge=args.merge,
        resolved_tasks="outside_timed_region_fixed_metadata",
        controls=controls,
        trace_analysis=trace_analysis,
        metric="one_target_forward_fixed_metadata_including_split_merge",
        draft_invocations=0,
        kv_initialization="model_prefill_shared_content_disjoint_pages",
        correctness=correctness,
        measurements=rows,
        graphs=resources,
        memory=dict(peak_allocated_bytes=torch.npu.max_memory_allocated(), kv_blocks=runner.kv_cache_config.num_blocks),
        environment=dict(
            attention_operation=str(torch_npu.npu_fused_infer_attention_score),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            device=torch.npu.get_device_name(),
            packages={name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "vllm", "vllm-ascend")},
        ),
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    # Retain the engine until every captured graph has finished.
    torch.npu.synchronize()
    del llm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--batch-invariant",
        action="store_true",
        help="use official batch-invariant operators for all modes; results are a separate numerical regime",
    )
    parser.add_argument("--batch", type=int, default=33)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--query", type=int, default=1)
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--lanes", nargs=2, type=int)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("frozen", "update"),
        default=["update"],
        help="update is the primary comparison; frozen is an optional immutable-metadata ablation",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--merge",
        choices=("cat", "captured"),
        default="cat",
        help="cat merges after both graphs; captured copies each lane to a disjoint output slice in its graph",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        workload = ReplayWorkload(
            args.batch,
            args.capacity,
            args.query,
            args.kv_length,
            tuple(args.lanes or ((args.batch + 1) // 2, args.batch // 2)),
        )
        if min(args.rounds, args.repeats, args.warmups) < 1 or len(set(args.protocols)) != len(args.protocols):
            raise ValueError("positive measurement counts and unique protocols required")
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Replace stale success output before touching hardware; failed runs are not measurements.
    report = dict(status="running", workload=dataclasses.asdict(workload), model=args.model)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        run(args, workload)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
