# SPDX-License-Identifier: Apache-2.0
"""Independent-process compact delayed/ready full-model replay with BF16 head.

This fixture consumes fixed tokens and acceptance counts. It does not execute
sampling, draft generation or a service scheduler, and makes no quality claim.
"""

import argparse
import faulthandler
import hashlib
import time
from pathlib import Path

from multi_graph_gdn_q4_full_loader import run_full_model
from multi_graph_gdn_q4_full_probe import (
    cache_binding_identity,
    complete_snapshot,
    deterministic_token_ids,
    restore_complete,
    verify_fia_tasks,
)
from multi_graph_gdn_q4_full_probe import (
    experiment as base_experiment,
)
from multi_graph_gdn_ready_chain_protocol import LOADS, MODES, critical_cost, digest, workload
from multi_graph_gdn_ready_full_lifecycle import TerminalDrain


class ChainTerminal(TerminalDrain):
    def __init__(self, torch, bundle, registry, counts):
        super().__init__(torch, bundle, registry)
        if len(counts) != len(bundle.pages.pages) or any(type(n) is not int or not 1 <= n <= 4 for n in counts):
            raise ValueError("one final fixed acceptance count per resident owner required")
        self.contract["accepted"] = list(counts)
        self.accepted.copy_(torch.tensor(counts, dtype=torch.int32))


def tensor_digest(torch, tensor):
    cpu = tensor.detach().to(device="cpu", copy=True).contiguous()
    from multi_graph_gdn_cpu_audit import finite

    if cpu.is_floating_point() and not finite(torch, cpu):
        raise AssertionError("nonfinite compact replay state or output")
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(torch, bundle, registry):
    registry._require_joined()
    torch.npu.synchronize()
    tensors, hosts = {}, []
    for entry in bundle.gdn_entries:
        backend, state = registry.backends[entry.state.handle], entry.state
        for index, tensor in enumerate(state.mutable_tensors):
            tensors[f"gdn{entry.index}/state{index}"] = tensor_digest(torch, tensor)
        for index, buffers in enumerate(backend._staging_buffers().values()):
            for key in ("new_logs", "new_valid"):
                tensors[f"gdn{entry.index}/staging{index}/{key}"] = tensor_digest(torch, buffers[key])
        tensors[f"gdn{entry.index}/conv"] = tensor_digest(torch, entry.conv)
        hosts.append(
            dict(
                layer=entry.index,
                generations=state.generations,
                initialized=state.initialized,
                last_step=state.last_step,
                pending=state.pending,
            )
        )
    for entry in bundle.fia_entries:
        for key, tensor in (("key", entry.key), ("value", entry.value)):
            tensors[f"fia{entry.index}/{key}"] = tensor_digest(torch, tensor)
    return dict(tensors=tensors, host=digest(hosts), tensor_count=len(tensors))


def replay_chain(
    args,
    torch,
    dist,
    bundle,
    factory,
    registry,
    forward,
    vocabulary,
    report,
    *,
    checkpoint,
    schedule=None,
    model,
    config,
):
    from vllm.distributed.parallel_state import get_world_group

    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

    if schedule is not None:
        raise ValueError("no split router or additional model schedule is part of this control")
    mode = "deferred_tile_" + args.mode
    registry.mode = mode
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    if model.language_model.lm_head.weight.dtype != torch.bfloat16:
        raise ValueError("retain existing BF16 head weight values")
    if model.language_model.logits_processor.head_dtype not in (None, torch.bfloat16):
        raise ValueError("this stage forbids FP32 head configuration")
    report.update(
        scope="fixed-trajectory complete-model execution replay including original BF16 LM head",
        mode=args.mode,
        load=args.load,
        load_contract=args.load_contract,
        exclusions=[
            "sampling",
            "draft generation",
            "service scheduler",
            "model loading",
            "common coherent prefill",
            "capture",
            "state audits",
            "fixture reset",
        ],
        initialization_definition="one direct compact transition after common coherent prefill",
        head=dict(
            rows="all Q4 padded bucket rows",
            vocabulary=vocabulary,
            input_dtype="torch.bfloat16",
            weight_dtype="torch.bfloat16",
            output_dtype="torch.bfloat16",
            actual_projection_audited=False,
        ),
    )
    trace, selection = bundle.trace, args.selection
    group, main = get_world_group().cpu_group, torch.npu.current_stream()
    root, update = torch.npu.Stream(), torch.npu.Stream()
    pool = torch.npu.graph_pool_handle()
    identity = cache_binding_identity(bundle)
    initial = complete_snapshot(torch, bundle, registry)
    report["initial_state_digest"] = state_digest(torch, bundle, registry)
    frames, tokens, entries = {}, {}, {}
    for capacity in selection["required_tokens"]:
        index = selection["dispatched_tokens"].index(capacity)
        frames[capacity] = factory.decode(trace.steps[index], capacity)
        tokens[capacity] = torch.zeros(capacity, dtype=torch.int64, device=bundle.device)

    def prepare(step, capacity):
        if cache_binding_identity(bundle) != identity:
            raise AssertionError("cache binding changed")
        frame = frames[capacity].refresh(step)
        cpu = torch.zeros(capacity, dtype=torch.int64)
        cpu[: step["batch"] * 4] = deterministic_token_ids(torch, step["batch"] * 4, vocabulary, step["input_seed"])
        tokens[capacity].copy_(cpu)
        return frame

    def operation(frame, ids, capturing=False):
        hidden, residual = forward(frame, ids, registry=registry, capturing=capturing)
        logits = model.language_model.compute_logits(hidden)
        if logits.dtype != torch.bfloat16 or tuple(logits.shape) != (frame.capacity_tokens, vocabulary):
            raise AssertionError("require original full-vocabulary BF16 projection and TP gather")
        return hidden, residual, logits

    # Audit one real operation outside all measured and captured execution.
    from multi_graph_gdn_aime26_fp32_head import projection_audit, real_projection

    capacity = selection["dispatched_tokens"][0]
    frame = prepare(trace.steps[0], capacity)
    hidden, _ = forward(frame, tokens[capacity], registry=registry)
    with projection_audit(torch, model.language_model.lm_head.weight) as audit:
        logits = model.language_model.compute_logits(hidden)
    torch.npu.synchronize()
    if (
        not real_projection(audit.operations, "torch.bfloat16")
        or logits.dtype != torch.bfloat16
        or not any(event.get("weight_is_original_storage") for event in audit.operations)
    ):
        raise AssertionError("original BF16 matrix operands and logits must be exposed")
    report["head"].update(actual_projection_audited=True, operations=audit.operations)
    restore_complete(torch, bundle, registry, initial)
    capture_start = time.perf_counter()
    for capacity in sorted(frames, reverse=True):
        index = selection["dispatched_tokens"].index(capacity)
        frame = prepare(trace.steps[index], capacity)
        restore_complete(torch, bundle, registry, initial)
        root.wait_stream(main)
        with torch.npu.stream(root):
            for _ in range(2):
                operation(frame, tokens[capacity])
                torch.npu.synchronize()
    for capacity in sorted(frames, reverse=True):
        index = selection["dispatched_tokens"].index(capacity)
        frame = prepare(trace.steps[index], capacity)
        restore_complete(torch, bundle, registry, initial)
        dist.barrier(group=group)
        root.wait_stream(main)
        graph = UpdatableGraph()
        with torch.npu.graph(graph, pool=pool, stream=root):
            output = operation(frame, tokens[capacity], capturing=True)
        torch.npu.synchronize()
        verify_fia_tasks(graph, [item.backend.layer_name for item in bundle.fia_entries])
        entries[capacity] = dict(graph=graph, output=output, prepared=torch.npu.Event(), updated=torch.npu.Event())
        print(f"rank={dist.get_rank()} chain captured {args.load}/{args.mode} tokens={capacity}", flush=True)
    terminal = ChainTerminal(torch, bundle, registry, args.terminal_counts)
    restore_complete(torch, bundle, registry, initial)
    root.wait_stream(main)
    with torch.npu.stream(root):
        terminal.run()
    torch.npu.synchronize()
    drain = torch.npu.NPUGraph()
    with torch.npu.graph(drain, pool=pool, stream=root):
        terminal.run()
    torch.npu.synchronize()
    report["capture"] = dict(
        graphs=len(entries),
        required_default_buckets=selection["required_tokens"],
        fia_tasks_per_graph=16,
        terminal_graphs=1,
        elapsed_seconds=time.perf_counter() - capture_start,
    )
    report["terminal_contract"] = terminal.contract
    checkpoint()

    def submit(entry, frame):
        tasks = entry["graph"].resolve_tasks(ContextSource(frame.mapping))
        entry["prepared"].record(main)
        root.wait_event(entry["prepared"])
        update.wait_event(entry["prepared"])
        with torch.npu.stream(root):
            entry["graph"].replay()
        entry["graph"].update(update, tasks)
        entry["updated"].record(update)
        root.wait_event(entry["updated"])

    def close():
        terminal.stage_acceptance()
        root.wait_stream(main)
        with torch.npu.stream(root):
            drain.replay()
        torch.npu.synchronize()

    def trajectory(audit_outputs=False):
        output_hashes, times = [], []
        for index, step in enumerate(trace.steps):
            start = time.perf_counter_ns()
            capacity = selection["dispatched_tokens"][index]
            registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
            frame = prepare(step, capacity)
            submit(entries[capacity], frame)
            torch.npu.synchronize()
            registry.finish_step(step["step"])
            times.append((time.perf_counter_ns() - start) / 1000)
            if audit_outputs:
                # Full vocabulary, every step; all downloads and hashes untimed.
                output_hashes.append(
                    dict(step=index, fields=[tensor_digest(torch, x) for x in entries[capacity]["output"]])
                )
        return output_hashes, times

    restore_complete(torch, bundle, registry, initial)
    output_hashes, _ = trajectory(audit_outputs=True)
    close()
    terminal.require_closed()
    reference = state_digest(torch, bundle, registry)
    report["correctness"] = dict(
        output_hashes=output_hashes,
        closed_state=reference,
        scope="same compact algorithm; cross-process output/state byte hashes only",
    )
    checkpoint()
    if args.profile_steps:
        profile_chain(
            args,
            torch,
            dist,
            group,
            bundle,
            registry,
            trace,
            selection,
            prepare,
            submit,
            entries,
            restore=lambda: restore_complete(torch, bundle, registry, initial),
            close=close,
            report=report,
        )
        report["status"] = "profile_complete_not_an_unprofiled_timing_sample"
        checkpoint()
        return
    restore_complete(torch, bundle, registry, initial)
    dist.barrier(group=group)
    begin = time.perf_counter_ns()
    _, per_step = trajectory()
    executed = time.perf_counter_ns()
    close()
    ended = time.perf_counter_ns()
    execution_us, terminal_us = (executed - begin) / 1000, (ended - executed) / 1000
    local = dict(
        rank=dist.get_rank(),
        execution_us=execution_us,
        terminal_us=terminal_us,
        initialization_us=report["compact_transition_us"],
        total_us=(ended - begin) / 1000 + report["compact_transition_us"],
        per_step_us=per_step,
    )
    terminal.require_closed()
    if state_digest(torch, bundle, registry) != reference:
        raise AssertionError("timed final state differs from own complete untimed trajectory")
    gathered = [None, None]
    dist.all_gather_object(gathered, local, group=group)
    report.update(
        status="closed_complete_model_replay_pending_cross_process_pair_audit",
        cost_ranks=gathered,
        critical_cost=critical_cost(gathered),
        timed_final_state_exact=True,
        covered="token construction and metadata H2D, graph update, 64 layers, BF16 head, TP, commits, all waits",
        head_precision_changes=False,
        cross_algorithm_quality_evaluated=False,
    )
    checkpoint()


def profile_chain(
    args, torch, dist, group, bundle, registry, trace, selection, prepare, submit, entries, *, restore, close, report
):
    import torch_npu

    restore()
    profiles = []
    for index, step in enumerate(trace.steps):
        capacity = selection["dispatched_tokens"][index]
        if index not in args.profile_steps:
            registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
            frame = prepare(step, capacity)
            submit(entries[capacity], frame)
            torch.npu.synchronize()
            registry.finish_step(step["step"])
            continue
        dist.barrier(group=group)
        destination = args.output.parent / f"profile.step{index}.rank{dist.get_rank()}"
        experimental = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=experimental,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
        ) as prof:
            with torch.profiler.record_function("ready_chain/prepare_and_upload"):
                registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
                frame = prepare(step, capacity)
            with torch.profiler.record_function("ready_chain/replay_including_BF16_head_and_TP"):
                submit(entries[capacity], frame)
            with torch.profiler.record_function("ready_chain/final_wait"):
                torch.npu.synchronize()
            with torch.profiler.record_function("ready_chain/host_state_finish"):
                registry.finish_step(step["step"])
        path = args.output.parent / f"step{index}.rank{dist.get_rank()}.trace.json"
        prof.export_chrome_trace(str(path))
        profiles.append(
            dict(
                step=index,
                batch=step["batch"],
                capacity=capacity,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_directory=str(destination),
            )
        )
    close()
    report["profiles"] = profiles


def experiment(args, trace, runner, model, config):
    def run(*positional, **keywords):
        return replay_chain(*positional, **keywords, model=model, config=config)

    return base_experiment(args, trace, runner, model, config, replay_runner=run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-steps", type=int, nargs="+")
    for name in ("extension", "products-extension", "deferred-extension", "tile-extension"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.tensor_parallel_size, args.phase, args.prefix_limit, args.dense_nz = 2, "replay", None, True
    args.split_boundaries, args.packed_join = None, False
    trace, args.terminal_counts, args.load_contract = workload(args.trace, args.load)
    if args.output.exists() or args.output.with_suffix(".rank0.json").exists():
        parser.error("preserve prior runs; use a new output directory")
    if args.profile_steps and (
        len(set(args.profile_steps)) != len(args.profile_steps)
        or any(not 0 <= n < len(trace.steps) for n in args.profile_steps)
    ):
        parser.error("profile indices must be unique and inside the fixed trajectory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    run_full_model(args, trace, experiment=experiment)


if __name__ == "__main__":
    main()
