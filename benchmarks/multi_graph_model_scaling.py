"""Full dense decoder replay: padding versus shared Dense / parallel Attention.

Includes embeddings, every decoder layer, final norm and required FIA updates.
Excludes LM head, sampling, draft and scheduler. Launch TP with torchrun.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from multi_graph_replay import run

from vllm_ascend.worker.multi_graph.benchmark import summarize
from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload


def experiment(args, workload, runner, model, inputs, context, reset_kv):
    import torch
    import torch.distributed as dist
    import vllm.envs as vllm_envs
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

    if vllm_envs.VLLM_BATCH_INVARIANT:
        raise ValueError("model scaling measures ordinary FIA only")
    layers = model.model.layers
    for layer in layers:
        attn = layer.self_attn
        if getattr(attn, "dual_chunk_attention_config", None) or getattr(attn, "qk_norm", False):
            raise ValueError("unsupported Qwen attention variant")
    rank = dist.get_rank()
    world = dist.get_world_size()
    cpu_group = get_world_group().cpu_group
    main = torch.npu.current_stream()
    tokens = workload.batch * workload.query
    padded_data, actual_data = inputs(0, workload.capacity), inputs(0, workload.batch)
    intervals = ((0, workload.lanes[0]), (workload.lanes[0], workload.batch))
    branch_data = [inputs(start, stop) for start, stop in intervals]
    branch_stream = torch.npu.Stream()
    events = [(torch.npu.Event(), torch.npu.Event()) for _ in layers]
    checks = []

    class BranchGraph(UpdatableGraph):
        def __init__(self):
            super().__init__()
            self.lane = None
            self.task_lanes = []

        def get_capture_resource(self, key, factory, use_max_workspace=False):
            # Layers join before reuse; concurrent lanes must never alias workspace.
            return super().get_capture_resource((self.lane, key), factory, use_max_workspace)

        def register_task(self, operation, kwargs, provider):
            super().register_task(operation, kwargs, provider)
            self.task_lanes.append(self.lane)

    graph = BranchGraph()

    class BranchSource:
        def get(self, provider):
            return tuple(
                provider.resolve(branch_data[lane][2])
                for task, lane in zip(graph.tasks, graph.task_lanes, strict=True)
                if task.provider == provider
            )

    def split_forward(parallel, partition=True):
        active_intervals = intervals if partition else ((0, workload.batch),)
        active_data = branch_data if partition else [actual_data]
        hidden = model.model.embed_tokens(actual_data[0])
        residual = None
        capturing = get_forward_context().capturing
        root = torch.npu.current_stream()
        for index, layer in enumerate(layers):
            attention = layer.self_attn
            backend = attention.attn
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            qkv, _ = attention.qkv_proj(hidden)
            q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
            if hasattr(attention, "q_norm"):
                q = attention.q_norm(q.view(-1, attention.num_heads, attention.head_dim)).view(q.shape)
                k = attention.k_norm(k.view(-1, attention.num_kv_heads, attention.head_dim)).view(k.shape)
            q, k = attention.rotary_emb(actual_data[1], q, k)
            output = torch.empty_like(q)

            def branch(lane, backend=backend, attention=attention, q=q, k=k, v=v, output=output):
                start, stop = active_intervals[lane]
                left, right = start * workload.query, stop * workload.query
                count = right - left
                graph.lane = lane
                with context(active_data[lane]):
                    get_forward_context().capturing = capturing
                    backend.impl.forward(
                        backend,
                        q[left:right].view(count, attention.num_heads, attention.head_dim),
                        k[left:right].view(count, attention.num_kv_heads, attention.head_dim),
                        v[left:right].view(count, attention.num_kv_heads, attention.head_dim),
                        backend.kv_cache,
                        next(iter(active_data[lane][2].values())),
                        output=output[left:right].view(count, attention.num_heads, attention.head_dim),
                    )

            if parallel:
                fork, join = events[index]
                fork.record(root)
                branch_stream.wait_event(fork)
                branch(0)
                with torch.npu.stream(branch_stream):
                    branch(1)
                    join.record(branch_stream)
                root.wait_event(join)
            else:
                for lane in range(len(active_intervals)):
                    branch(lane)
            hidden, _ = attention.o_proj(output)
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp(hidden)
        hidden, _ = model.model.norm(hidden, residual)
        return hidden

    def padded_forward():
        return model(input_ids=padded_data[0], positions=padded_data[1])

    def eager(fn, data):
        reset_kv()
        with context(data):
            return fn().float().cpu().clone()

    def check(output, reference, name):
        actual = output[:tokens].float().cpu()
        torch.testing.assert_close(actual, reference[:tokens], rtol=0.02, atol=0.02)
        checks.append(dict(name=name, passed=True, max_abs_error=(actual - reference[:tokens]).abs().max().item()))

    references = {
        "padding": eager(padded_forward, padded_data),
        "split_parallel": eager(lambda: split_forward(False), actual_data),
    }
    official = eager(lambda: model(input_ids=actual_data[0], positions=actual_data[1]), actual_data)
    check(eager(lambda: split_forward(False, partition=False), actual_data), official, "direct_backend_vs_official")
    diagnostics = {}
    for name, value in references.items():
        delta = value[:tokens] - official
        diagnostics[name] = dict(
            max_abs_error=delta.abs().max().item(),
            relative_rms=(delta.square().mean() / official.square().mean().clamp_min(1e-30)).sqrt().item(),
            cosine=torch.nn.functional.cosine_similarity(value[:tokens], official, dim=-1).mean().item(),
            outside_tolerance=(~torch.isclose(value[:tokens], official, rtol=0.02, atol=0.02)).float().mean().item(),
        )
    entries = {}
    for name, fn, data, current_graph, source, count in (
        ("padding", padded_forward, padded_data, UpdatableGraph(), ContextSource(padded_data[2]), len(layers)),
        ("split_parallel", lambda: split_forward(True), actual_data, graph, BranchSource(), 2 * len(layers)),
    ):
        print(f"rank={rank} capturing {name}, layers={len(layers)}", flush=True)
        reset_kv()
        stream, update = torch.npu.Stream(), torch.npu.Stream()
        stream.wait_stream(main)
        with torch.npu.stream(stream), context(data):
            for _ in range(3):
                fn()
        stream.synchronize()
        pool = torch.npu.graph_pool_handle()
        with context(data):
            get_forward_context().capturing = True
            with torch.npu.graph(current_graph, pool=pool, stream=stream):
                output = fn()
        stream.synchronize()
        if len(current_graph.tasks) != count:
            raise ValueError(f"{name}: expected {count} FIA tasks; got {len(current_graph.tasks)}")
        pointers = {value.data_ptr() for value in current_graph.capture_resources.values()}
        if name == "split_parallel" and len(pointers) != 2:
            raise ValueError("parallel branches require two distinct FIA workspaces")
        entries[name] = dict(
            graph=current_graph,
            stream=stream,
            update=update,
            pool=pool,
            data=data,
            fn=fn,
            tasks=current_graph.resolve_tasks(source),
            output=output,
            updated=torch.npu.Event(),
            done=torch.npu.Event(),
            workspaces=len(pointers),
        )
    ready = torch.npu.Event()

    def replay(name):
        entry = entries[name]
        ready.record(main)
        entry["stream"].wait_event(ready)
        with torch.npu.stream(entry["stream"]):
            entry["graph"].replay()
        entry["update"].wait_event(ready)
        entry["graph"].update(entry["update"], entry["tasks"])
        entry["updated"].record(entry["update"])
        entry["stream"].wait_event(entry["updated"])
        entry["done"].record(entry["stream"])
        main.wait_event(entry["done"])
        return entry["output"]

    for name in entries:
        reset_kv()
        check(replay(name), references[name], name)
        for _ in range(args.warmups):
            replay(name)
            torch.npu.synchronize()
    rows = []
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for round_id in range(args.rounds):
        for name in entries if round_id % 2 == 0 else reversed(entries):
            reset_kv()
            wall, device = [], []
            for _ in range(args.repeats):
                # Barrier excluded from timing; report the slowest rank for each sample.
                if world > 1:
                    dist.barrier(group=cpu_group)
                begin.record(main)
                started = time.perf_counter_ns()
                output = replay(name)
                end.record(main)
                end.synchronize()
                wall.append((time.perf_counter_ns() - started) / 1000)
                device.append(begin.elapsed_time(end) * 1000)
            check(output, references[name], f"{name}/round{round_id}")
            rows.append(dict(mode=name, round=round_id, wall_us=wall, device_interval_us=device))
            print(f"rank={rank} {name} round={round_id}: {summarize(wall)['p50']:.2f} us", flush=True)

    padded_data[0].add_(1)
    actual_data[0].add_(1)
    for name, fn, data in (
        ("padding", padded_forward, padded_data),
        ("split_parallel", lambda: split_forward(False), actual_data),
    ):
        reference = eager(fn, data)
        reset_kv()
        check(replay(name), reference, f"{name}/changed_input")
    rank_result = dict(rank=rank, checks=checks, measurements=rows, numerical_diagnostics=diagnostics)
    all_ranks = [None] * world
    dist.all_gather_object(all_ranks, rank_result, group=cpu_group)
    if rank == 0:
        measurements = []
        for index, row in enumerate(rows):
            maximum = [max(r["measurements"][index]["wall_us"][i] for r in all_ranks) for i in range(args.repeats)]
            measurements.append(
                dict(mode=row["mode"], round=row["round"], wall_us=maximum, wall_summary_us=summarize(maximum))
            )
        config = runner.model_config.hf_config
        payload = dict(
            status="passed",
            scope="full_decoder_including_embedding_and_final_norm_no_lm_head",
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=world,
            workload=dataclasses.asdict(workload),
            layers=len(layers),
            model_geometry={
                key: getattr(config, key, None)
                for key in (
                    "model_type",
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                )
            },
            update_included=True,
            cross_architecture_qualified=False,
            measurements=measurements,
            ranks=all_ranks,
            graphs=[
                dict(mode=name, update_tasks=len(e["tasks"]), workspaces=e["workspaces"]) for name, e in entries.items()
            ],
            environment=dict(
                device=torch.npu.get_device_name(),
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        )
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch", type=int, default=129)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--query", type=int, default=8)
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--lanes", nargs=2, type=int)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batch_invariant = False
    try:
        workload = ReplayWorkload(
            args.batch,
            args.capacity,
            args.query,
            args.kv_length,
            tuple(args.lanes or ((args.batch + 1) // 2, args.batch // 2)),
        )
        if min(args.rounds, args.repeats, args.warmups, args.tensor_parallel_size) < 1:
            raise ValueError("counts must be positive")
        if int(os.environ.get("WORLD_SIZE", "1")) != args.tensor_parallel_size:
            raise ValueError("torchrun WORLD_SIZE must equal --tensor-parallel-size")
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        args.output.write_text(json.dumps(dict(status="running")), encoding="utf-8")
    try:
        run(args, workload, experiment=experiment)
    except Exception as error:
        failure = args.output.with_suffix(f".rank{rank}.failed.json")
        failure.write_text(
            json.dumps(dict(status="failed", error_type=type(error).__name__, error=str(error))), encoding="utf-8"
        )
        if rank == 0:
            args.output.write_text(failure.read_text(encoding="utf-8"), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
