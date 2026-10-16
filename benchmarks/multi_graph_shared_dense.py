"""First-layer Qwen2 experiment: full-layer splitting versus shared dense computation.

All attention branches retain FIA update. This is a fixed-metadata experiment,
not a production runner or a full-model performance claim.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

from multi_graph_replay import run

from vllm_ascend.worker.multi_graph.benchmark import summarize
from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload, inspect_shared_dense_trace


def experiment(args, workload, runner, model, inputs, context, reset_kv):
    import torch
    import torch_npu
    import vllm.envs as vllm_envs
    from vllm.forward_context import get_forward_context

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

    layer = model.model.layers[0]
    attention = layer.self_attn
    backend = attention.attn
    if attention.qk_norm or attention.dual_chunk_attention_config:
        raise ValueError("this prototype requires standard Qwen2 attention")
    main = torch.npu.current_stream()
    data = inputs(0, workload.capacity)
    hidden = model.model.embed_tokens(data[0]).clone()
    token_count = workload.batch * workload.query
    intervals = ((0, workload.lanes[0]), (workload.lanes[0], workload.batch))
    resources, checks = [], []

    def pre(x, positions):
        normalized = layer.input_layernorm(x)
        qkv, _ = attention.qkv_proj(normalized)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        q, k = attention.rotary_emb(positions, q, k)
        return q, k, v, x

    def attend(q, k, v, metadata, output):
        count = q.shape[0]
        return backend.impl.forward(
            backend,
            q.view(count, attention.num_heads, attention.head_dim),
            k.view(count, attention.num_kv_heads, attention.head_dim),
            v.view(count, attention.num_kv_heads, attention.head_dim),
            backend.kv_cache,
            metadata,
            output=output.view(count, attention.num_heads, attention.head_dim),
        )

    def post(output, residual):
        projected, _ = attention.o_proj(output)
        normalized, residual = layer.post_attention_layernorm(projected, residual)
        return layer.mlp(normalized), residual

    def full(x, entry_data):
        q, k, v, residual = pre(x, entry_data[1])
        output = torch.empty_like(q)
        attend(q, k, v, next(iter(entry_data[2].values())), output)
        return post(output, residual)

    def cpu_pair(pair):
        return tuple(value.float().cpu().clone() for value in pair)

    def assert_pair(observed, expected, name):
        errors = []
        for actual, reference in zip(cpu_pair(observed), expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
            errors.append((actual - reference).abs().max().item())
        checks.append(dict(name=name, max_abs_errors=errors, passed=True))

    def eager(fn, entry_data):
        reset_kv()
        with context(entry_data):
            return cpu_pair(fn())

    actual_data = inputs(0, workload.batch)
    x_actual = hidden[:token_count]
    official = eager(lambda: layer(actual_data[1], x_actual, None), actual_data)
    reset_kv()
    with context(actual_data):
        assert_pair(full(x_actual, actual_data), official, "direct_backend_vs_official_layer")

    def capture(name, fn, entry_data, expected_tasks, graph=None, source=None):
        print(f"capturing {name}", flush=True)
        reset_kv()
        stream, update = torch.npu.Stream(), torch.npu.Stream()
        stream.wait_stream(main)
        with torch.npu.stream(stream), context(entry_data):
            for _ in range(3):
                fn()
        stream.synchronize()
        graph = graph if graph is not None else UpdatableGraph()
        pool = torch.npu.graph_pool_handle()
        with context(entry_data):
            get_forward_context().capturing = True
            with torch.npu.graph(graph, pool=pool, stream=stream):
                output = fn()
        stream.synchronize()
        if len(graph.tasks) != expected_tasks:
            raise ValueError(f"{name}: expected {expected_tasks} FIA tasks, found {len(graph.tasks)}")
        workspace_pointers = [value.data_ptr() for value in graph.capture_resources.values()]
        if expected_tasks == 2 and len(set(workspace_pointers)) != 2:
            raise ValueError("joined attention branches must own distinct FIA workspaces")
        resources.append(
            dict(name=name, update_tasks=len(graph.tasks), distinct_workspaces=len(set(workspace_pointers)))
        )
        return dict(
            name=name,
            # Captured task arguments can be weak references; retain their owners.
            data=entry_data,
            fn=fn,
            graph=graph,
            pool=pool,
            stream=stream,
            update=update,
            tasks=graph.resolve_tasks(source if source is not None else ContextSource(entry_data[2])),
            output=output,
            done=torch.npu.Event(),
            updated=torch.npu.Event(),
        )

    def submit(entry, dependency):
        entry["stream"].wait_event(dependency)
        with torch.npu.stream(entry["stream"]):
            entry["graph"].replay()
        if entry["tasks"]:
            entry["update"].wait_event(dependency)
            entry["graph"].update(entry["update"], entry["tasks"])
            entry["updated"].record(entry["update"])
            entry["stream"].wait_event(entry["updated"])
        entry["done"].record(entry["stream"])

    padded_reference = eager(lambda: full(hidden, data), data)
    padded = capture("padded_layer", lambda: full(hidden, data), data, 1)
    exact = capture("exact_layer", lambda: full(x_actual, actual_data), actual_data, 1)
    merged = tuple(torch.empty_like(value) for value in exact["output"])
    lanes, lane_references = [], []
    for index, (start, stop) in enumerate(intervals):
        lane_data = inputs(start, stop)
        lane_x = hidden[start * workload.query : stop * workload.query]

        def fn(lane_x=lane_x, lane_data=lane_data, start=start, stop=stop):
            output = full(lane_x, lane_data)
            if args.merge == "captured":
                for value, destination in zip(output, merged, strict=True):
                    destination[start * workload.query : stop * workload.query].copy_(value)
            return output

        lane_references.append(eager(fn, lane_data))
        lanes.append(capture(f"full_lane_{index}", fn, lane_data, 1))
    full_split_reference = tuple(torch.cat([pair[index] for pair in lane_references]) for index in range(2))

    shared_pre = capture("shared_dense_pre", lambda: pre(x_actual, actual_data[1]), actual_data, 0)
    q, k, v, residual = shared_pre["output"]
    shared_attention = torch.empty_like(q)
    branches = []
    for index, (start, stop) in enumerate(intervals):
        left, right = start * workload.query, stop * workload.query
        branch_data = inputs(start, stop)
        metadata = next(iter(branch_data[2].values()))
        fn = lambda left=left, right=right, metadata=metadata: attend(
            q[left:right], k[left:right], v[left:right], metadata, shared_attention[left:right]
        )
        branches.append(capture(f"attention_lane_{index}", fn, branch_data, 1))
    shared_post = capture("shared_dense_post", lambda: post(shared_attention, residual), actual_data, 0)

    class BranchGraph(UpdatableGraph):
        def __init__(self):
            super().__init__()
            self.lane = None
            self.task_lanes = []

        def get_capture_resource(self, key, factory, use_max_workspace=False):
            # FIA workspace cannot be shared by concurrently executing branches.
            return super().get_capture_resource((self.lane, key), factory, use_max_workspace)

        def register_task(self, operation, kwargs, provider):
            super().register_task(operation, kwargs, provider)
            self.task_lanes.append(self.lane)

    class BranchSource:
        def __init__(self, graph, branch_inputs):
            self.graph, self.branch_inputs = graph, branch_inputs

        def get(self, provider):
            return tuple(
                provider.resolve(self.branch_inputs[lane][2])
                for task, lane in zip(self.graph.tasks, self.graph.task_lanes, strict=True)
                if task.provider == provider
            )

    def capture_joined(parallel):
        graph = BranchGraph()
        branch_inputs = [inputs(start, stop) for start, stop in intervals]
        branch_stream = torch.npu.Stream()
        fork, join = torch.npu.Event(), torch.npu.Event()

        def joined():
            capturing = get_forward_context().capturing
            root_stream = torch.npu.current_stream()
            q_joined, k_joined, v_joined, residual_joined = pre(x_actual, actual_data[1])
            out = torch.empty_like(q_joined)

            def branch(index):
                start, stop = intervals[index]
                left, right = start * workload.query, stop * workload.query
                graph.lane = index
                with context(branch_inputs[index]):
                    get_forward_context().capturing = capturing
                    attend(
                        q_joined[left:right],
                        k_joined[left:right],
                        v_joined[left:right],
                        next(iter(branch_inputs[index][2].values())),
                        out[left:right],
                    )

            if parallel:
                fork.record(root_stream)
                branch_stream.wait_event(fork)
                # Register the main lane first so its update can feed the larger task early.
                branch(0)
                with torch.npu.stream(branch_stream):
                    branch(1)
                    join.record(branch_stream)
                root_stream.wait_event(join)
            else:
                branch(0)
                branch(1)
            return post(out, residual_joined)

        entry = capture(
            "joined_shared_parallel" if parallel else "joined_shared_serial",
            joined,
            actual_data,
            2,
            graph=graph,
            source=BranchSource(graph, branch_inputs),
        )
        entry["branch_inputs"] = branch_inputs
        return entry

    joined_serial, joined_parallel = capture_joined(False), capture_joined(True)

    # The architecture has its own eager oracle; cross-shape equality is a separate diagnostic.
    reset_kv()
    with context(actual_data):
        q_ref, k_ref, v_ref, residual_ref = pre(x_actual, actual_data[1])
        attention_ref = torch.empty_like(q_ref)
    for start, stop in intervals:
        left, right = start * workload.query, stop * workload.query
        branch_data = inputs(start, stop)
        with context(branch_data):
            attend(
                q_ref[left:right],
                k_ref[left:right],
                v_ref[left:right],
                next(iter(branch_data[2].values())),
                attention_ref[left:right],
            )
    with context(actual_data):
        hybrid_reference = cpu_pair(post(attention_ref, residual_ref))

    ready = torch.npu.Event()
    modes = (
        "single_padded",
        "single_exact",
        "full_serial",
        "full_parallel",
        "shared_serial",
        "shared_parallel",
        "joined_serial",
        "joined_parallel",
    )

    def replay(mode):
        ready.record(main)
        if mode.startswith("joined_"):
            entry = joined_parallel if mode == "joined_parallel" else joined_serial
            submit(entry, ready)
            main.wait_event(entry["done"])
            return entry["output"]
        if mode in ("single_padded", "single_exact"):
            entry = padded if mode == "single_padded" else exact
            submit(entry, ready)
            main.wait_event(entry["done"])
            return tuple(value[:token_count] for value in entry["output"])
        if mode.startswith("full_"):
            for index, entry in enumerate(lanes):
                dependency = lanes[index - 1]["done"] if index and mode.endswith("serial") else ready
                submit(entry, dependency)
            for entry in lanes:
                main.wait_event(entry["done"])
            if args.merge == "cat":
                for index, destination in enumerate(merged):
                    torch.cat([entry["output"][index] for entry in lanes], out=destination)
            return merged
        submit(shared_pre, ready)
        for index, entry in enumerate(branches):
            dependency = branches[index - 1]["done"] if index and mode.endswith("serial") else shared_pre["done"]
            submit(entry, dependency)
        for entry in branches:
            shared_post["stream"].wait_event(entry["done"])
        submit(shared_post, shared_pre["done"])
        main.wait_event(shared_post["done"])
        return shared_post["output"]

    references = {
        "single_padded": tuple(value[:token_count] for value in padded_reference),
        "single_exact": official,
        "full_serial": full_split_reference,
        "full_parallel": full_split_reference,
        "shared_serial": hybrid_reference,
        "shared_parallel": hybrid_reference,
        "joined_serial": hybrid_reference,
        "joined_parallel": hybrid_reference,
    }
    numerical_diagnostics = {}
    for mode, pair in references.items():
        numerical_diagnostics[mode] = []
        for value, reference in zip(pair, official, strict=True):
            delta = value - reference
            numerical_diagnostics[mode].append(
                dict(
                    max_abs_error=delta.abs().max().item(),
                    relative_rms=(delta.square().mean() / reference.square().mean().clamp_min(1e-30)).sqrt().item(),
                    cosine=torch.nn.functional.cosine_similarity(value, reference, dim=-1).mean().item(),
                )
            )
    gc.collect()
    for mode in modes:
        print(f"checking {mode}", flush=True)
        reset_kv()
        assert_pair(replay(mode), references[mode], mode)
        for _ in range(args.warmups):
            replay(mode)
            torch.npu.synchronize()

    measurements = []
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for round_id in range(args.rounds):
        for mode in modes if round_id % 2 == 0 else reversed(modes):
            reset_kv()
            wall, device = [], []
            for _ in range(args.repeats):
                begin.record(main)
                started = time.perf_counter_ns()
                output = replay(mode)
                end.record(main)
                end.synchronize()
                wall.append((time.perf_counter_ns() - started) / 1000)
                device.append(begin.elapsed_time(end) * 1000)
            assert_pair(output, references[mode], f"{mode}/round{round_id}")
            measurements.append(
                dict(
                    mode=mode,
                    round=round_id,
                    wall_us=wall,
                    device_interval_us=device,
                    wall_summary_us=summarize(wall),
                    device_summary_us=summarize(device),
                )
            )
            print(f"{mode} round={round_id}: {summarize(wall)['p50']:.2f} us", flush=True)
    trace_summary = None
    if args.trace:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for mode in modes:
                for _ in range(3):
                    with torch.profiler.record_function(f"layer.{mode}"):
                        replay(mode)
                        torch.npu.synchronize()
        trace_path = args.output.with_suffix(".trace.json")
        profiler.export_chrome_trace(str(trace_path))
        trace_summary = inspect_shared_dense_trace(json.loads(trace_path.read_text()))

    # Change values in the captured storage to detect stale intermediates or aliases.
    hidden.mul_(0.75)
    changed_official = eager(lambda: layer(actual_data[1], x_actual, None), actual_data)
    changed_padded = eager(lambda: full(hidden, data), data)
    changed_lanes = [eager(entry["fn"], entry["data"]) for entry in lanes]
    changed_split = tuple(torch.cat([pair[index] for pair in changed_lanes]) for index in range(2))
    changed_shared = eager(joined_serial["fn"], actual_data)
    for mode in modes:
        if mode == "single_padded":
            reference = tuple(value[:token_count] for value in changed_padded)
        elif mode == "single_exact":
            reference = changed_official
        elif mode.startswith("full_"):
            reference = changed_split
        else:
            reference = changed_shared
        reset_kv()
        assert_pair(replay(mode), reference, f"{mode}/changed_input")
    payload = dict(
        status="passed",
        experiment="first_qwen2_decoder_layer_with_residual_output",
        scope="fixed_metadata_first_layer_host_submission_included_not_full_model",
        workload=dataclasses.asdict(workload),
        full_split_merge=args.merge,
        model=args.model,
        dtype=args.dtype,
        batch_invariant=bool(vllm_envs.VLLM_BATCH_INVARIANT),
        checks=checks,
        numerical_diagnostics=numerical_diagnostics,
        cross_architecture_qualified=False,
        graphs=resources,
        trace_summary=trace_summary,
        measurements=measurements,
        environment=dict(
            device=torch.npu.get_device_name(),
            attention_operation=str(torch_npu.npu_fused_infer_attention_score),
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch", type=int, default=129)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--query", type=int, default=8)
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--lanes", nargs=2, type=int)
    parser.add_argument("--merge", choices=("cat", "captured"), default="captured")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
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
        if min(args.rounds, args.repeats, args.warmups) < 1:
            raise ValueError("measurement counts must be positive")
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", workload=dataclasses.asdict(workload))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        run(args, workload, experiment=experiment)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
