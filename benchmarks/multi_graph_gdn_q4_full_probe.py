# SPDX-License-Identifier: Apache-2.0
"""Stage complete Qwen3.5 TP2 caches, coherent prefixes and Q4 model replay.

Allocation/prefix/eager stages are explicit integration gates, not performance
claims. All 64 real decoders and final normalization run through model.forward.
The immutable random trajectory supplies ownership, lengths and accepted counts.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


def deterministic_token_ids(torch, length, vocabulary_size, seed):
    """CPU generation is independent of TP rank and accelerator RNG state."""
    if length < 1 or vocabulary_size < 1:
        raise ValueError("positive token count and vocabulary size required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocabulary_size, (length,), generator=generator, dtype=torch.int64)


def tensor_record(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), bytes=tensor.numel() * tensor.element_size())


def memory_record(torch, label):
    torch.npu.synchronize()
    return dict(
        phase=label,
        allocated_bytes=torch.npu.memory_allocated(),
        reserved_bytes=torch.npu.memory_reserved(),
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
        peak_reserved_bytes=torch.npu.max_memory_reserved(),
    )


def verify_layer_visits(visits):
    if len(visits) != 64 or any(count != 1 for count in visits):
        raise RuntimeError(f"complete forward must visit each of 64 real decoders exactly once: {visits}")


def latency_statistics(samples):
    if not samples:
        raise ValueError("latency samples must be nonempty")
    values = sorted(samples)

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return dict(
        mean_us=statistics.mean(values), p50_us=percentile(0.5), p95_us=percentile(0.95), p99_us=percentile(0.99)
    )


def verify_fia_tasks(graph, layer_names):
    expected = set(layer_names)
    actual = [getattr(task.provider, "layer_name", None) for task in graph.tasks]
    if len(layer_names) != 16 or len(expected) != 16 or len(actual) != 16 or set(actual) != expected:
        raise RuntimeError(f"every full-model FIA occurrence must have one true host update task: {actual}")


def cache_binding_identity(bundle):
    gdn = []
    for item in bundle.gdn_entries:
        if item.gdn.kv_cache[0] is not item.conv or item.gdn.kv_cache[1] is not item.state.handle:
            raise RuntimeError(f"GDN layer {item.index} no longer owns its declared conv/compact handle")
        gdn.append((item.conv.data_ptr(), item.base.data_ptr(), id(item.state.handle)))
    fia = []
    for item in bundle.fia_entries:
        if (
            item.backend.kv_cache[0] is not item.key
            or item.backend.kv_cache[1] is not item.value
            or item.impl.key_cache is not item.key
            or item.impl.value_cache is not item.value
        ):
            raise RuntimeError(f"FIA layer {item.index} cache binding changed")
        fia.append((item.key.data_ptr(), item.value.data_ptr()))
    return tuple(gdn), tuple(fia)


def complete_snapshot(torch, bundle, registry):
    registry.finish_pending_commits()
    torch.npu.synchronize()
    return dict(
        compact=registry.snapshot(destination="cpu"),
        conv=tuple(item.conv.to(device="cpu", copy=True) for item in bundle.gdn_entries),
        fia=tuple(
            (item.key.to(device="cpu", copy=True), item.value.to(device="cpu", copy=True))
            for item in bundle.fia_entries
        ),
    )


def restore_complete(torch, bundle, registry, saved):
    registry.restore(saved["compact"])
    for item, conv in zip(bundle.gdn_entries, saved["conv"], strict=True):
        item.conv.copy_(conv)
    for item, pair in zip(bundle.fia_entries, saved["fia"], strict=True):
        item.key.copy_(pair[0])
        item.value.copy_(pair[1])
    torch.npu.synchronize()


def replay_trajectories(args, torch, dist, bundle, factory, registry, forward, vocabulary, report):
    """Full default-bucket graphs; real FIA updates and all state joins included."""
    from multi_graph_gdn_q4_full_checks import verify_untouched_owners
    from vllm.distributed.parallel_state import get_world_group

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

    trace, selection = bundle.trace, args.selection
    cache_identity = cache_binding_identity(bundle)
    cpu_group = get_world_group().cpu_group
    main = torch.npu.current_stream()
    modes = ("deferred_overlap", "deferred_pipeline")
    initial = complete_snapshot(torch, bundle, registry)
    frames, tokens = {}, {}
    for capacity in selection["required_tokens"]:
        index = selection["dispatched_tokens"].index(capacity)
        frames[capacity] = factory.decode(trace.steps[index], capacity)
        tokens[capacity] = torch.zeros(capacity, dtype=torch.int64, device="npu")

    def prepare(step, capacity):
        if cache_binding_identity(bundle) != cache_identity:
            raise RuntimeError("complete-model cache storage changed between graph buckets")
        frame = frames[capacity]
        frame.refresh(step)
        token_cpu = torch.zeros(capacity, dtype=torch.int64)
        count = step["batch"] * 4
        token_cpu[:count] = deterministic_token_ids(torch, count, vocabulary, step["input_seed"])
        tokens[capacity].copy_(token_cpu)
        return frame

    entries = {}
    # All graphs execute exclusively. Shared pool storage may alias across
    # captures; endpoints are consumed before another graph is submitted.
    pool = torch.npu.graph_pool_handle()
    capture_start = time.perf_counter()
    for mode in modes:
        registry.mode = mode
        for capacity in sorted(frames, reverse=True):
            index = selection["dispatched_tokens"].index(capacity)
            frame = prepare(trace.steps[index], capacity)
            restore_complete(torch, bundle, registry, initial)
            root, update = torch.npu.Stream(), torch.npu.Stream()
            root.wait_stream(main)
            with torch.npu.stream(root):
                for _ in range(2):
                    forward(frame, tokens[capacity], registry=registry)
            torch.npu.synchronize()
            restore_complete(torch, bundle, registry, initial)
            dist.barrier(group=cpu_group)
            root.wait_stream(main)
            graph = UpdatableGraph()
            with torch.npu.graph(graph, pool=pool, stream=root):
                output = forward(frame, tokens[capacity], registry=registry, capturing=True)
            torch.npu.synchronize()
            verify_fia_tasks(graph, [item.backend.layer_name for item in bundle.fia_entries])
            entries[mode, capacity] = dict(
                graph=graph,
                root=root,
                update=update,
                output=output,
                prepared=torch.npu.Event(),
                updated=torch.npu.Event(),
            )
            print(f"rank={dist.get_rank()} full-model captured mode={mode} tokens={capacity}", flush=True)
    report["capture"] = dict(
        graphs=len(entries),
        required_default_buckets=selection["required_tokens"],
        fia_tasks_per_graph=16,
        elapsed_seconds=time.perf_counter() - capture_start,
        partition="native GDN request products/solve ranges inside each legal full-model bucket",
    )
    report["memory"].append(memory_record(torch, "all_required_default_graphs_captured"))

    def replay(entry, frame):
        tasks = entry["graph"].resolve_tasks(ContextSource(frame.mapping))
        entry["prepared"].record(main)
        entry["root"].wait_event(entry["prepared"])
        entry["update"].wait_event(entry["prepared"])
        with torch.npu.stream(entry["root"]):
            entry["graph"].replay()
        entry["graph"].update(entry["update"], tasks)
        entry["updated"].record(entry["update"])
        entry["root"].wait_event(entry["updated"])

    def state_exact(expected, label):
        from multi_graph_gdn_q4_full_checks import compare_complete_snapshots

        actual = complete_snapshot(torch, bundle, registry)
        compare_complete_snapshots(torch, actual, expected, label)
        del actual

    # Same-mode eager/graph controls keep different floating point algorithms
    # separate. Continue from the complete preceding step, including paused IDs.
    for mode in modes:
        registry.mode = mode
        restore_complete(torch, bundle, registry, initial)
        for index in range(args.qualification_steps):
            step, capacity = trace.steps[index], selection["dispatched_tokens"][index]
            before = complete_snapshot(torch, bundle, registry)
            frame = prepare(step, capacity)
            eager = tuple(value.cpu() for value in forward(frame, tokens[capacity], registry=registry))
            torch.npu.synchronize()
            expected = complete_snapshot(torch, bundle, registry)
            verify_untouched_owners(torch, before, expected, step["request_ids"], bundle.pages, f"{mode}/step{index}")
            restore_complete(torch, bundle, registry, before)
            frame = prepare(step, capacity)
            replay(entries[mode, capacity], frame)
            torch.npu.synchronize()
            for name, actual, wanted in zip(
                ("hidden", "residual"), entries[mode, capacity]["output"], eager, strict=True
            ):
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0, msg=f"{mode}/step{index}/{name}")
                if not torch.isfinite(actual).all().item():
                    raise RuntimeError(f"nonfinite full-model replay endpoint {mode}/step{index}/{name}")
            state_exact(expected, f"{mode}/step{index}")
            report["checks"].append(dict(label="complete_same_schedule_eager_replay_exact", mode=mode, step=index))
            del before, expected, eager
    samples = []
    for trial in range(args.trials):
        for mode in modes if trial % 2 == 0 else reversed(modes):
            registry.mode = mode
            restore_complete(torch, bundle, registry, initial)
            for index, step in enumerate(trace.steps):
                capacity = selection["dispatched_tokens"][index]
                dist.barrier(group=cpu_group)
                start = time.perf_counter_ns()
                registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
                frame = prepare(step, capacity)
                replay(entries[mode, capacity], frame)
                torch.npu.synchronize()
                registry.finish_step(step["step"])
                elapsed_us = (time.perf_counter_ns() - start) / 1000
                critical = torch.tensor(elapsed_us, dtype=torch.float64)
                dist.all_reduce(critical, op=dist.ReduceOp.MAX, group=cpu_group)
                samples.append(
                    dict(
                        trial=trial,
                        mode=mode,
                        step=index,
                        batch=step["batch"],
                        capacity=capacity,
                        elapsed_us=elapsed_us,
                        critical_rank_us=critical.item(),
                    )
                )
    latency = {
        mode: latency_statistics([item["critical_rank_us"] for item in samples if item["mode"] == mode])
        for mode in modes
    }
    means = {mode: latency[mode]["mean_us"] for mode in modes}
    paired_trials = []
    for trial in range(args.trials):
        times = {
            mode: statistics.mean(
                item["critical_rank_us"] for item in samples if item["trial"] == trial and item["mode"] == mode
            )
            for mode in modes
        }
        paired_trials.append(
            dict(trial=trial, mean_us=times, reduction_percent=100 * (1 - times[modes[1]] / times[modes[0]]))
        )
    report.update(
        status="measured_complete_replay_pending_full_correctness_audit",
        samples=samples,
        mean_critical_rank_us=means,
        latency=latency,
        paired_trials=paired_trials,
        mean_reduction_percent=100 * (1 - means[modes[1]] / means[modes[0]]),
        required_goal_reduction_percent=5,
        complete_goal_achieved=False,
        correctness_limitations=(
            "initial continued eager/replay steps only; full candidate/state/crossalgorithm audit still required"
        ),
    )
    report["memory"].append(memory_record(torch, "complete_dynamic_trials"))


@contextmanager
def model_observer(model):
    """Observe actual layer traversal and the real final hidden/residual pair."""
    observation = dict(visits=[0] * 64, endpoint=None)
    handles = []
    layers = model.language_model.model.layers
    if len(layers) != 64:
        raise ValueError("the complete-model probe requires all 64 decoders on every TP rank")
    for index, layer in enumerate(layers):

        def visited(module, inputs, output, index=index):
            observation["visits"][index] += 1

        handles.append(layer.register_forward_hook(visited))

    def endpoint(module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("the real final norm must return hidden and accumulated residual")
        observation["endpoint"] = output

    handles.append(model.language_model.model.norm.register_forward_hook(endpoint))
    try:
        yield observation
    finally:
        for handle in handles:
            handle.remove()


def experiment(args, trace, runner, model, config):
    import torch
    import torch.distributed as dist
    import torch_npu
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_full_metadata import FullMetadataFactory, allocate_full_caches
    from multi_graph_gdn_q4_log_backend import CompactSSMRegistry, Q4LogBackend, installed_log_registry
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    rank = dist.get_rank()
    report = dict(
        status="integration_in_progress_not_a_performance_result",
        requested_phase=args.phase,
        scope="all_64_decoders_and_final_norm",
        rank=rank,
        trace_sha256=trace.digest,
        loader=args.loader_metadata,
        selection=args.selection,
        memory=[memory_record(torch, "weights_loaded")],
        checks=[],
        sources={Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    )
    args.partial_report = report
    bundle = allocate_full_caches(
        torch,
        model,
        config,
        trace,
        device="npu",
        capture_sizes=args.selection["capture_sizes"],
        format_cast=lambda tensor: torch_npu.npu_format_cast(tensor, 2),
    )
    factory = FullMetadataFactory(torch, bundle, config)
    vocabulary = config.model_config.hf_text_config.vocab_size
    report["cache_layout"] = dict(
        gdn=[
            dict(index=item.index, prefix=item.prefix, base=tensor_record(item.base), conv=tensor_record(item.conv))
            for item in bundle.gdn_entries
        ],
        fia=[
            dict(index=item.index, key=tensor_record(item.key), value=tensor_record(item.value))
            for item in bundle.fia_entries
        ],
        no_dense_four_candidate_ssm=True,
    )
    report["memory"].append(memory_record(torch, "all_prefill_caches_allocated"))
    report["checks"].append(
        dict(label="real_layer_cache_counts", gdn=len(bundle.gdn_entries), fia=len(bundle.fia_entries))
    )

    def save():
        text = json.dumps(report, indent=2) + "\n"
        args.output.with_suffix(f".rank{rank}.json").write_text(text, encoding="utf-8")
        if rank == 0:
            args.output.write_text(text, encoding="utf-8")

    if args.phase == "allocation":
        bundle.transition_to_decode()
        report["memory"].append(memory_record(torch, "all_decode_caches_allocated"))
        report["status"] = "qualified_full_cache_allocation_only"
        save()
        return report

    with model_observer(model) as observation:

        def forward(frame, token_ids, *, registry=None, capturing=False):
            observation["visits"][:] = [0] * 64
            observation["endpoint"] = None
            with set_ascend_forward_context(
                frame.mapping,
                config,
                num_tokens=frame.capacity_tokens,
                num_actual_tokens=frame.capacity_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.FULL if capturing else CUDAGraphMode.NONE,
                model_instance=model,
                skip_compiled=True,
            ):
                ctx = get_forward_context()
                ctx.slot_mapping = {item.backend.layer_name: frame.slot_mapping for item in bundle.fia_entries}
                ctx.capturing = capturing
                output = model(input_ids=token_ids, positions=frame.positions)
                if registry is not None:
                    registry.finish_pending_commits()
                if observation["endpoint"] is None:
                    raise RuntimeError("complete model did not execute its real final normalization")
                hidden, residual = observation["endpoint"]
                if output is not hidden:
                    raise RuntimeError("unexpected model output wrapper or auxiliary hidden-state path")
                if hasattr(frame, "active"):
                    hidden = torch.where(frame.active, hidden, 0)
                    residual = torch.where(frame.active, residual, 0)
            verify_layer_visits(observation["visits"])
            return hidden, residual

        prefix_start = time.perf_counter()
        prefix_hash = hashlib.sha256()
        prefix_records = []
        prefixes = trace.initial_prefix_by_request
        for owner, length in enumerate(prefixes):
            if args.prefix_limit is not None and owner >= args.prefix_limit:
                break
            frame = factory.prefix(owner, length)
            tokens = deterministic_token_ids(torch, length, vocabulary, trace.seed + owner)
            prefix_hash.update(tokens.numpy().tobytes())
            result = forward(frame, tokens.to("npu"))
            torch.npu.synchronize()
            for value in result:
                if not torch.isfinite(value).all().item():
                    raise RuntimeError(f"nonfinite full-model prefix endpoint for owner {owner}")
            prefix_records.append(dict(owner=owner, tokens=length))
            if rank == 0 and (owner % 8 == 0 or owner + 1 == len(prefixes)):
                print(f"full-model prefix owner={owner + 1}/{len(prefixes)} tokens={length}", flush=True)
        report["prefix"] = dict(
            completed_requests=len(prefix_records),
            required_requests=len(prefixes),
            completed_tokens=sum(item["tokens"] for item in prefix_records),
            token_sha256=prefix_hash.hexdigest(),
            elapsed_seconds=time.perf_counter() - prefix_start,
            records=prefix_records,
            full_real_model=True,
        )
        report["memory"].append(memory_record(torch, "real_prefixes_completed"))
        if len(prefix_records) != len(prefixes):
            report["status"] = "qualified_partial_prefix_diagnostic_only"
            save()
            return report
        if args.phase == "prefix":
            report["status"] = "qualified_all_real_model_prefixes_only"
            save()
            return report

        bundle.transition_to_decode()
        log = load_extension("gdn_q4_log", args.extension)
        products = load_extension("gdn_q4_products", args.products_extension)
        deferred = load_extension("gdn_q4_log_deferred", args.deferred_extension)
        registry = CompactSSMRegistry()
        for item in bundle.gdn_entries:
            backend = Q4LogBackend(item.state, log, products, "deferred_overlap", deferred_module=deferred)
            backend.reserve_exclusive_capacity(max_batch=max(args.selection["required_tokens"]) // 4)
            registry.register(item.state.handle, backend, item.gdn.out_proj)
        report["memory"].append(memory_record(torch, "compact_decode_transition"))
        if args.phase == "replay":
            with installed_log_registry(registry):
                replay_trajectories(args, torch, dist, bundle, factory, registry, forward, vocabulary, report)
            save()
            return report
        step = trace.steps[0]
        capacity = args.selection["dispatched_tokens"][0]
        frame = factory.decode(step, capacity)
        token_cpu = torch.zeros(capacity, dtype=torch.int64)
        token_cpu[: step["batch"] * 4] = deterministic_token_ids(
            torch, step["batch"] * 4, vocabulary, step["input_seed"]
        )
        tokens = token_cpu.to("npu")
        with installed_log_registry(registry):
            output = forward(frame, tokens, registry=registry)
            torch.npu.synchronize()
        for label, value in zip(("hidden", "residual"), output, strict=True):
            if not torch.isfinite(value).all().item():
                raise RuntimeError(f"nonfinite complete eager Q4 {label}")
            report["checks"].append(dict(label=f"complete_eager_{label}_finite", **tensor_record(value)))
        report["checks"].append(dict(label="all_layers_visited_once", visits=observation["visits"]))
        report["memory"].append(memory_record(torch, "complete_eager_step"))
        report["status"] = "qualified_real_prefix_and_complete_eager_q4_not_replay"
        save()
        return report


def main():
    from multi_graph_gdn_q4_full_loader import run_full_model
    from multi_graph_gdn_q4_full_metadata import load_fixed_trace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(__file__).parent / "results/multi_graph_structural/structural-dynamic-trace-v1.json",
    )
    parser.add_argument("--phase", choices=("allocation", "prefix", "eager", "replay"), default="eager")
    parser.add_argument(
        "--prefix-limit", type=int, help="partial initialization diagnostic; never a full-model qualification"
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--deferred-extension", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--qualification-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.prefix_limit is not None and not 1 <= args.prefix_limit <= 128):
        parser.error("require a new output path and prefix-limit in 1..128")
    if args.trials < 2 or not 2 <= args.qualification_steps <= 128:
        parser.error("at least two paired trials and 2..128 continued qualification steps required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        trace = load_fixed_trace(args.trace)
        run_full_model(args, trace, experiment=experiment)
    except Exception as error:
        report = getattr(args, "partial_report", {})
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        rank = int(os.environ.get("RANK", "0"))
        text = json.dumps(report, indent=2) + "\n"
        args.output.with_suffix(f".rank{rank}.failure.json").write_text(text, encoding="utf-8")
        if rank == 0:
            args.output.write_text(text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
