# SPDX-License-Identifier: Apache-2.0
"""Stage complete Qwen3.5 TP2 caches, coherent prefixes and Q4 model replay.

Allocation/prefix/eager stages are explicit integration gates, not performance
claims. All 64 real decoders and final normalization run through model.forward.
The immutable random trajectory supplies ownership, lengths and accepted counts.
"""

import argparse
import faulthandler
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager, nullcontext
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


def replay_attribution(means, *, tile_enabled):
    """Do not credit a full-bucket kernel pipeline to request-graph splitting."""
    if tile_enabled:
        expected = {"deferred_overlap", "deferred_tile_separate", "deferred_tile_delayed", "deferred_tile_ready"}
        variant = "deferred_tile_ready"
        public_modes = expected
        scope = "public_full_bucket_kernel_pipeline"
    else:
        expected = {"deferred_overlap", "deferred_pipeline"}
        variant = "deferred_pipeline"
        public_modes = {"deferred_overlap"}
        scope = "request_range_pipeline_without_strongest_tile_padding_control"
    if set(means) != expected or any(not 0 < value < float("inf") for value in means.values()):
        raise ValueError("attribution requires exactly the declared controls and finite positive latencies")
    public_baseline = min(sorted(public_modes), key=means.__getitem__)
    control = min(sorted(expected - {variant}), key=means.__getitem__)
    return dict(
        scope=scope,
        measured_variant=variant,
        algorithm_control=control,
        algorithm_reduction_percent=100 * (1 - means[variant] / means[control]),
        public_padding_modes=sorted(public_modes),
        fastest_public_padding=public_baseline,
        split_parallel_candidate=None,
        split_incremental_reduction_percent=None,
        split_attribution_proven=False,
        limitation=(
            "Ready consumption runs in one native solve over the whole legal bucket, inside one full-model graph; "
            "padding can use it unchanged. The ready/delayed difference is a public kernel pipeline benefit."
            if tile_enabled
            else "The range-pipeline comparison does not include the public ready-tile padding kernel."
        ),
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


def replay_trajectories(
    args, torch, dist, bundle, factory, registry, forward, vocabulary, report, *, checkpoint, schedule=None
):
    """Full default-bucket graphs; real FIA updates and all state joins included."""
    from multi_graph_gdn_q4_full_stream_audit import StreamingCheckpoint
    from vllm.distributed.parallel_state import get_world_group

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

    trace, selection = bundle.trace, args.selection
    cache_identity = cache_binding_identity(bundle)
    cpu_group = get_world_group().cpu_group
    main = torch.npu.current_stream()
    modes = (
        ("deferred_overlap", "deferred_tile_separate", "deferred_tile_delayed", "deferred_tile_ready")
        if args.tile_extension is not None
        else ("deferred_overlap", "deferred_pipeline")
    )
    candidate = "deferred_tile_ready" if args.tile_extension is not None else "deferred_pipeline"
    diagnostic = args.phase == "diagnostic"
    steps = trace.steps[: args.diagnostic_steps] if diagnostic else trace.steps
    capacities = sorted(set(selection["dispatched_tokens"][: len(steps)]))
    if diagnostic:
        modes = ("deferred_tile_ready",)
    if schedule is not None:
        modes = schedule.modes
        report["split_schedule"] = dict(
            boundaries=schedule.boundaries,
            packed=schedule.packed,
            modes=modes,
            shared_kernel="deferred_tile_ready",
            pairs={
                str(capacity): list(
                    __import__("aclgraph_full_core_join").legal_pair(capacity, selection["capture_sizes"])
                )
                for capacity in capacities
            },
            intermediate_contract="paired consumer waits each row event; two output copies included",
        )

    def select_mode(mode):
        registry.mode = mode if schedule is None else "deferred_tile_ready"
        if schedule is not None:
            schedule.assert_joined()
            schedule.mode = mode

    initial = complete_snapshot(torch, bundle, registry)
    frames, tokens = {}, {}
    for capacity in capacities:
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
    # Keep the shared graph allocator on one capture stream across buckets.
    capture_root, update_stream = torch.npu.Stream(), torch.npu.Stream()
    capture_start = time.perf_counter()
    # Finish all shape initialization before any graph capture. This also
    # keeps subsequent captures free of intervening eager model launches.
    for mode in modes:
        select_mode(mode)
        for capacity in sorted(frames, reverse=True):
            index = selection["dispatched_tokens"].index(capacity)
            frame = prepare(trace.steps[index], capacity)
            restore_complete(torch, bundle, registry, initial)
            capture_root.wait_stream(main)
            with torch.npu.stream(capture_root):
                for _ in range(2):
                    forward(frame, tokens[capacity], registry=registry)
                    torch.npu.synchronize()
            print(f"rank={dist.get_rank()} full-model warmup mode={mode} tokens={capacity}", flush=True)
    for mode in modes:
        select_mode(mode)
        for capacity in sorted(frames, reverse=True):
            index = selection["dispatched_tokens"].index(capacity)
            frame = prepare(trace.steps[index], capacity)
            restore_complete(torch, bundle, registry, initial)
            root, update = capture_root, update_stream
            print(f"rank={dist.get_rank()} full-model capture begin mode={mode} tokens={capacity}", flush=True)
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
        required_default_buckets=capacities,
        diagnostic_subset=diagnostic,
        fia_tasks_per_graph=16,
        elapsed_seconds=time.perf_counter() - capture_start,
        partition=(
            "native GDN request-tile readiness inside each legal full-model bucket"
            if args.tile_extension is not None
            else "native GDN request products/solve ranges inside each legal full-model bucket"
        ),
    )
    report["memory"].append(memory_record(torch, "all_required_default_graphs_captured"))
    report["status"] = "all_default_graphs_captured_replay_unqualified"
    checkpoint()

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

    # Same-mode eager/graph controls keep different floating point algorithms
    # separate. Continue from the complete preceding step, including paused IDs.
    candidate_scratch = torch.empty((129, 4, 24, 128, 128), dtype=torch.float32, device=bundle.device)

    def check_candidates(scratch, label):
        for item in bundle.gdn_entries:
            item_backend = registry.backends[item.state.handle]
            item_backend.deferred_module.export_candidates(item.state.base, item.state.logs, item.state.valid, scratch)
            if not torch.isfinite(scratch).all().item():
                raise AssertionError(f"{label}/layer{item.index}: nonfinite logical candidate")
            if torch.count_nonzero(scratch[0]).item():
                raise AssertionError(f"{label}/layer{item.index}: nonzero null candidate")

    qualified_outputs = {}
    for mode in modes:
        select_mode(mode)
        restore_complete(torch, bundle, registry, initial)
        audit = StreamingCheckpoint(torch, initial)
        for index in range(args.qualification_steps):
            step, capacity = trace.steps[index], selection["dispatched_tokens"][index]
            registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
            frame = prepare(step, capacity)
            eager = tuple(value.cpu() for value in forward(frame, tokens[capacity], registry=registry))
            torch.npu.synchronize()
            registry.finish_step(step["step"])
            eager_audit = audit.capture_eager_and_restore(
                bundle, registry, step["request_ids"], label=f"{mode}/step{index}"
            )
            torch.npu.synchronize()
            registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
            frame = prepare(step, capacity)
            replay(entries[mode, capacity], frame)
            torch.npu.synchronize()
            registry.finish_step(step["step"])
            for name, actual, wanted in zip(
                ("hidden", "residual"), entries[mode, capacity]["output"], eager, strict=True
            ):
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0, msg=f"{mode}/step{index}/{name}")
                if not torch.isfinite(actual).all().item():
                    raise RuntimeError(f"nonfinite full-model replay endpoint {mode}/step{index}/{name}")
            replay_audit = audit.compare_replay(bundle, registry, label=f"{mode}/step{index}")
            check_candidates(candidate_scratch, f"{mode}/step{index}")
            report["checks"].append(
                dict(
                    label="complete_same_schedule_eager_replay_exact",
                    mode=mode,
                    step=index,
                    eager_audit=eager_audit,
                    replay_audit=replay_audit,
                )
            )
            report["status"] = "partial_continued_replay_qualification_not_a_performance_result"
            checkpoint()
            print(f"rank={dist.get_rank()} full-model qualified mode={mode} step={index}", flush=True)
            if schedule is not None:
                qualified_outputs[mode, index] = eager
                if mode != "padding":
                    for field, actual, reference in zip(
                        ("hidden", "residual"), eager, qualified_outputs["padding", index], strict=True
                    ):
                        delta = actual.float() - reference.float()
                        report.setdefault("cross_padding_diagnostics", []).append(
                            dict(
                                mode=mode,
                                step=index,
                                field=field,
                                max_abs=delta.abs().max().item(),
                                nrmse=(delta.square().mean() / reference.float().square().mean().clamp_min(1e-30))
                                .sqrt()
                                .item(),
                                tolerance_failures=(delta.abs() > 0.02 + 0.02 * reference.float().abs()).sum().item(),
                            )
                        )
                if mode == "split_early":
                    for actual, reference in zip(eager, qualified_outputs["split_barrier", index], strict=True):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            del eager
        del audit
    del candidate_scratch
    samples = []
    for trial in range(args.trials):
        for mode in modes if trial % 2 == 0 else reversed(modes):
            select_mode(mode)
            restore_complete(torch, bundle, registry, initial)
            for index, step in enumerate(steps):
                capacity = selection["dispatched_tokens"][index]
                dist.barrier(group=cpu_group)
                start = time.perf_counter_ns()
                registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
                frame = prepare(step, capacity)
                entry = entries[mode, capacity]
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record(entry["root"])
                replay(entry, frame)
                end.record(entry["root"])
                torch.npu.synchronize()
                device_us = begin.elapsed_time(end) * 1000
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
                        device_us=device_us,
                        critical_rank_us=critical.item(),
                    )
                )
            report.update(status="partial_complete_replay_timing_not_a_goal_result", samples=samples)
            checkpoint()
    latency = {
        mode: latency_statistics([item["critical_rank_us"] for item in samples if item["mode"] == mode])
        for mode in modes
    }
    means = {mode: latency[mode]["mean_us"] for mode in modes}
    if diagnostic:
        from aclgraph_full_path_profile import profile_replays

        report.update(
            status="diagnostic_timing_complete_profiles_pending",
            samples=samples,
            latency=latency,
            diagnostic_steps=len(steps),
            complete_goal_achieved=False,
            correctness_limitations="only the declared continued qualification steps are fully state-audited",
        )
        checkpoint()
        report["profiles"] = profile_replays(
            args,
            torch,
            dist,
            registry,
            steps,
            selection,
            entries,
            prepare,
            replay,
            restore=lambda: restore_complete(torch, bundle, registry, initial),
            cpu_group=cpu_group,
            modes=modes,
            select_mode=select_mode,
            selected_indices=[0] if schedule is not None else None,
        )
        report["status"] = (
            "split_full_model_diagnostic_complete_not_serving"
            if schedule is not None
            else "diagnostic_complete_not_a_speedup_comparison"
        )
        checkpoint()
        return
    attribution = replay_attribution(means, tile_enabled=args.tile_extension is not None)
    baseline = attribution["algorithm_control"]
    paired_trials = []
    for trial in range(args.trials):
        times = {
            mode: statistics.mean(
                item["critical_rank_us"] for item in samples if item["trial"] == trial and item["mode"] == mode
            )
            for mode in modes
        }
        paired_trials.append(
            dict(trial=trial, mean_us=times, algorithm_reduction_percent=100 * (1 - times[candidate] / times[baseline]))
        )
    report.update(
        status="measured_complete_replay_pending_full_correctness_audit",
        samples=samples,
        mean_critical_rank_us=means,
        latency=latency,
        paired_trials=paired_trials,
        attribution=attribution,
        algorithm_control=baseline,
        measured_variant=candidate,
        algorithm_reduction_percent=attribution["algorithm_reduction_percent"],
        same_kernel_readiness_reduction_percent=(
            100 * (1 - means[candidate] / means["deferred_tile_delayed"]) if args.tile_extension is not None else None
        ),
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
        tile = load_extension("gdn_q4_tile_pipeline", args.tile_extension) if args.tile_extension is not None else None
        registry = CompactSSMRegistry()
        # Complete-model invocations are exclusive; ordered commits from
        # different layers can share one side stream instead of binding the
        # entire stream pool separately in every captured bucket.
        shared_commit_stream = torch.npu.Stream()
        for item in bundle.gdn_entries:
            backend = Q4LogBackend(
                item.state, log, products, "deferred_overlap", deferred_module=deferred, tile_module=tile
            )
            backend.reserve_exclusive_capacity(
                max_batch=max(args.selection["required_tokens"]) // 4, commit_stream=shared_commit_stream
            )
            registry.register(item.state.handle, backend, item.gdn.out_proj)
        report["memory"].append(memory_record(torch, "compact_decode_transition"))
        if args.phase in ("replay", "diagnostic"):
            schedule_context = nullcontext(None)
            if getattr(args, "split_boundaries", None):
                from aclgraph_full_core_join import CoreJoinSchedule

                schedule_context = CoreJoinSchedule(
                    model,
                    args.split_boundaries,
                    args.selection["capture_sizes"],
                    packed=args.packed_join,
                ).installed()
            with installed_log_registry(registry), schedule_context as schedule:
                replay_trajectories(
                    args,
                    torch,
                    dist,
                    bundle,
                    factory,
                    registry,
                    forward,
                    vocabulary,
                    report,
                    checkpoint=save,
                    schedule=schedule,
                )
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
    parser.add_argument("--phase", choices=("allocation", "prefix", "eager", "replay", "diagnostic"), default="eager")
    parser.add_argument("--split-boundaries", type=int, nargs="+")
    parser.add_argument("--packed-join", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=6)
    parser.add_argument(
        "--prefix-limit", type=int, help="partial initialization diagnostic; never a full-model qualification"
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--deferred-extension", type=Path, required=True)
    parser.add_argument("--tile-extension", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--qualification-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.prefix_limit is not None and not 1 <= args.prefix_limit <= 128):
        parser.error("require a new output path and prefix-limit in 1..128")
    if args.trials < 2 or not 2 <= args.qualification_steps <= 128:
        parser.error("at least two paired trials and 2..128 continued qualification steps required")
    if args.phase == "diagnostic" and (
        args.tile_extension is None or not args.qualification_steps <= args.diagnostic_steps <= 128
    ):
        parser.error("diagnostic requires the public ready-tile kernel and qualification_steps <= steps <= 128")
    if args.split_boundaries and args.phase != "diagnostic":
        parser.error("split boundaries currently require the bounded diagnostic phase")
    if args.packed_join:
        parser.error("packed join not implemented yet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
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
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
