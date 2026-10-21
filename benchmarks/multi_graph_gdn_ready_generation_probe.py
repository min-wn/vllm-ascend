# SPDX-License-Identifier: Apache-2.0
"""Offline actual Q4 autoregressive delayed/ready generation, never a service."""

import argparse
import hashlib
import time
from pathlib import Path

from multi_graph_gdn_aime26 import GenerationRequest, digest, terminal_metadata
from multi_graph_gdn_aime26_probe import save_json
from multi_graph_gdn_q4_full_loader import run_full_model
from multi_graph_gdn_q4_full_probe import (
    complete_snapshot,
    model_observer,
    restore_complete,
    verify_fia_tasks,
    verify_layer_visits,
)
from multi_graph_gdn_ready_chain_probe import state_digest, tensor_digest
from multi_graph_gdn_ready_generation import (
    COMPLETE,
    MAX_MODEL_LEN,
    MODES,
    capacity_trace,
    critical_cost,
    load_manifest,
)


class GenerationTerminal:
    """Existing deferred commit ABI, with actual final host acceptance counts."""

    def __init__(self, torch, bundle, registry):
        self.torch, self.bundle, self.registry = torch, bundle, registry
        count = len(bundle.pages.pages)
        values = terminal_metadata([1] * count)
        self.sequence, self.ids, self.accepted = [
            torch.tensor(values[k], dtype=torch.int32, device=bundle.device) for k in ("sequence", "ids", "accepted")
        ]
        state = bundle.gdn_entries[0].state
        self.empty_logs, self.empty_valid = torch.zeros_like(state.logs), torch.zeros_like(state.valid)

    def stage(self, requests):
        self.accepted.copy_(self.torch.tensor([r.accepted for r in requests], dtype=self.torch.int32))

    def run(self):
        self.registry._require_joined()
        for entry in self.bundle.gdn_entries:
            state = entry.state
            self.registry.backends[state.handle].deferred_module.commit(
                state.base,
                self.sequence,
                self.ids,
                self.accepted,
                state.logs,
                state.valid,
                self.empty_logs,
                self.empty_valid,
            )

    def require_closed(self):
        for entry in self.bundle.gdn_entries:
            if self.torch.count_nonzero(entry.state.valid).item() or self.torch.count_nonzero(entry.state.logs).item():
                raise AssertionError("final accepted-state debt remains")


def experiment(args, trace, runner, model, config):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_npu
    from multi_graph_gdn_aime26_fp32_head import projection_audit, real_projection
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_full_metadata import FullMetadataFactory, allocate_full_caches
    from multi_graph_gdn_q4_log_backend import CompactSSMRegistry, Q4LogBackend, installed_log_registry
    from vllm.config import CUDAGraphMode
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context
    from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

    rank, group = dist.get_rank(), get_world_group().cpu_group
    report = dict(
        status="preparing_offline_generation",
        rank=rank,
        mode=args.mode,
        workload_sha256=args.manifest["payload_sha256"],
        selection=args.selection,
        loader=args.loader_metadata,
        model_load_and_loader_seconds=(time.perf_counter_ns() - args.load_start) / 1e9,
        scope="offline actual autoregressive generation; no online service or cross-algorithm quality claim",
        fallback_calls=0,
        kernel_identities={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.extensions},
    )

    def checkpoint():
        save_json(args.output.with_suffix(f".rank{rank}.json"), report)

    checkpoint()
    torch.set_num_threads(1)
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.npu.config.allow_internal_format = True
    converted = []
    for index, layer in enumerate(model.language_model.model.layers):
        for name, module in layer.named_modules():
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 2:
                weight.data = torch_npu.npu_format_cast(weight.data, 29)
                converted.append(dict(layer=index, name=name, format=torch_npu.get_npu_format(weight)))
    if len(converted) != 304 or any(r["format"] != 29 for r in converted):
        raise AssertionError("retain the existing 304 NZ29 decoder projections")
    report["dense_weights"] = converted
    head = model.language_model.lm_head.weight
    if head.dtype != torch.bfloat16 or model.language_model.logits_processor.head_dtype not in (None, torch.bfloat16):
        raise AssertionError("the original BF16 head is frozen")
    setup_start = time.perf_counter_ns()
    bundle = allocate_full_caches(
        torch,
        model,
        config,
        trace,
        device="npu",
        capture_sizes=args.selection["capture_sizes"],
        format_cast=lambda t: torch_npu.npu_format_cast(t, 2),
        expected_owners=len(trace.initial_prefix_by_request),
    )
    factory = FullMetadataFactory(torch, bundle, config)
    modules = [
        load_extension(name, path)
        for name, path in zip(
            ("gdn_q4_log", "gdn_q4_products", "gdn_q4_log_deferred", "gdn_q4_tile_pipeline"),
            args.extensions,
            strict=True,
        )
    ]
    torch.npu.synchronize()
    report["cache_and_extension_setup_seconds"] = (time.perf_counter_ns() - setup_start) / 1e9
    registry = None
    rows = args.manifest["payload"]["inputs"]

    def new_requests():
        return [GenerationRequest(r["owner"], tuple(r["prompt_tokens"]), max_new_tokens=256) for r in rows]

    def predictions(logits):
        result = logits.argmax(-1).to(torch.int64)
        dist.broadcast(result, src=0)
        return result.cpu().tolist()

    with model_observer(model) as observation:

        def forward(frame, ids, *, capturing=False):
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
                context = get_forward_context()
                context.slot_mapping = {e.backend.layer_name: frame.slot_mapping for e in bundle.fia_entries}
                context.capturing = capturing
                output = model(input_ids=ids, positions=frame.positions)
                if registry is not None:
                    registry.finish_pending_commits()
                hidden, residual = observation["endpoint"]
                if output is not hidden:
                    raise AssertionError("unexpected final-normalization model endpoint")
                hidden = torch.where(frame.active, hidden, 0)
                residual = torch.where(frame.active, residual, 0)
            verify_layer_visits(observation["visits"])
            return hidden, residual

        requests = new_requests()
        dist.barrier(group=group)
        started = time.perf_counter_ns()
        for owner, request in enumerate(requests):
            frame = factory.prefix(owner, len(request.prompt))
            hidden, _ = forward(frame, torch.tensor(request.prompt, dtype=torch.int64, device=bundle.device))
            request.begin(predictions(model.language_model.compute_logits(hidden[-1:]))[0])
        torch.npu.synchronize()
        prefill_us = (time.perf_counter_ns() - started) / 1000
        with projection_audit(torch, head) as audit:
            logits = model.language_model.compute_logits(hidden[-1:])
        torch.npu.synchronize()
        if not real_projection(audit.operations, "torch.bfloat16") or logits.dtype != torch.bfloat16:
            raise AssertionError("the actual full-vocabulary BF16 projection was not observed")
        report["head"] = dict(input="BF16", weight="BF16", output="BF16", vocabulary=248320, actual_projection=True)
        first_tokens = [r.tokens[0] for r in requests]
        started = time.perf_counter_ns()
        bundle.transition_to_decode()
        registry, commit_stream = CompactSSMRegistry(), torch.npu.Stream()
        for entry in bundle.gdn_entries:
            backend = Q4LogBackend(
                entry.state,
                modules[0],
                modules[1],
                "deferred_tile_" + args.mode,
                deferred_module=modules[2],
                tile_module=modules[3],
            )
            backend.reserve_exclusive_capacity(
                max_batch=max(args.selection["required_tokens"]) // 4, commit_stream=commit_stream
            )
            registry.register(entry.state.handle, backend, entry.gdn.out_proj, name=f"gdn{entry.index}")
        torch.npu.synchronize()
        state_import_us = (time.perf_counter_ns() - started) / 1000
        initial = complete_snapshot(torch, bundle, registry)
        report["initial_state"] = state_digest(torch, bundle, registry)
        checkpoint()
        main = torch.npu.current_stream()
        root, update = torch.npu.Stream(), torch.npu.Stream()
        pool = torch.npu.graph_pool_handle()
        capacity_by_live = dict(zip(range(1, len(rows) + 1), args.selection["dispatched_tokens"], strict=True))
        entries = {}

        def operation(frame, ids, *, capturing=False):
            hidden, _ = forward(frame, ids, capturing=capturing)
            logits = model.language_model.compute_logits(hidden)
            if logits.dtype != torch.bfloat16 or tuple(logits.shape) != (frame.capacity_tokens, 248320):
                raise AssertionError("retain all padded rows and the original TP-gathered vocabulary")
            return hidden, logits

        capture_start = time.perf_counter_ns()
        with installed_log_registry(registry):
            for capacity in sorted(args.selection["required_tokens"], reverse=True):
                live = max(b for b, c in capacity_by_live.items() if c == capacity)
                step = dict(
                    step=0,
                    batch=live,
                    request_ids=tuple(range(live)),
                    seq_lengths=tuple(len(rows[o]["prompt_tokens"]) + 4 for o in range(live)),
                    accepted_from_previous=(1,) * live,
                )
                frame = factory.decode(step, capacity)
                ids = torch.zeros(capacity, dtype=torch.int64, device=bundle.device)
                restore_complete(torch, bundle, registry, initial)
                root.wait_stream(main)
                with torch.npu.stream(root):
                    for _ in range(2):
                        operation(frame, ids)
                torch.npu.synchronize()
                restore_complete(torch, bundle, registry, initial)
                dist.barrier(group=group)
                root.wait_stream(main)
                graph = UpdatableGraph()
                with torch.npu.graph(graph, pool=pool, stream=root):
                    output = operation(frame, ids, capturing=True)
                torch.npu.synchronize()
                verify_fia_tasks(graph, [e.backend.layer_name for e in bundle.fia_entries])
                entries[capacity] = dict(
                    graph=graph,
                    output=output,
                    frame=frame,
                    ids=ids,
                    prepared=torch.npu.Event(),
                    updated=torch.npu.Event(),
                )
                print(f"rank={rank} generation captured {args.mode} capacity={capacity}", flush=True)
            terminal = GenerationTerminal(torch, bundle, registry)
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
                capacities=sorted(entries),
                terminal_graphs=1,
                seconds=(time.perf_counter_ns() - capture_start) / 1e9,
            )
            checkpoint()

            def generate(*, audit_outputs):
                requests = new_requests()
                for request, token in zip(requests, first_tokens, strict=True):
                    request.begin(token)
                records, output_hashes, steps = [], [], 0
                components = dict(
                    draft_us=0.0,
                    upload_submit_us=0.0,
                    replay_wait_us=0.0,
                    prediction_transfer_us=0.0,
                    acceptance_us=0.0,
                )
                begin = time.perf_counter_ns()
                while active := [o for o, r in enumerate(requests) if not r.finish_reason]:
                    start = time.perf_counter_ns()
                    capacity = capacity_by_live[len(active)]
                    entry = entries[capacity]
                    cpu_ids = torch.zeros(capacity, dtype=torch.int64)
                    proposals = []
                    for row, owner in enumerate(active):
                        request = requests[owner]
                        history = np.asarray([*request.prompt, *request.tokens], dtype=np.int32)
                        proposed = _find_longest_matched_ngram_and_propose_tokens(
                            history, 2, 4, MAX_MODEL_LEN, 3
                        ).tolist()
                        proposals.append(proposed)
                        cpu_ids[row * 4 : (row + 1) * 4] = torch.tensor(request.input_tokens(proposed))
                    drafted = time.perf_counter_ns()
                    step = dict(
                        step=steps,
                        batch=len(active),
                        request_ids=tuple(active),
                        seq_lengths=tuple(requests[o].processed + 4 for o in active),
                        accepted_from_previous=tuple(requests[o].accepted for o in active),
                    )
                    registry.prepare_step(steps, active, step["accepted_from_previous"])
                    frame = entry["frame"].refresh(step)
                    entry["ids"].copy_(cpu_ids)
                    tasks = entry["graph"].resolve_tasks(ContextSource(frame.mapping))
                    entry["prepared"].record(main)
                    root.wait_event(entry["prepared"])
                    update.wait_event(entry["prepared"])
                    uploaded = time.perf_counter_ns()
                    with torch.npu.stream(root):
                        entry["graph"].replay()
                    entry["graph"].update(update, tasks)
                    entry["updated"].record(update)
                    root.wait_event(entry["updated"])
                    torch.npu.synchronize()
                    registry.finish_step(steps)
                    executed = time.perf_counter_ns()
                    target = predictions(entry["output"][1][: len(active) * 4])
                    predicted = time.perf_counter_ns()
                    decisions = [
                        requests[o].advance(p, target[row * 4 : (row + 1) * 4])
                        for row, (o, p) in enumerate(zip(active, proposals, strict=True))
                    ]
                    accepted = time.perf_counter_ns()
                    for key, left, right in zip(
                        components,
                        (start, drafted, uploaded, executed, predicted),
                        (drafted, uploaded, executed, predicted, accepted),
                        strict=True,
                    ):
                        components[key] += (right - left) / 1000
                    records.append(
                        dict(
                            **step,
                            proposals=proposals,
                            predictions=target,
                            decisions=decisions,
                            capacity_tokens=capacity,
                        )
                    )
                    if audit_outputs:
                        output_hashes.append(tensor_digest(torch, entry["output"][0]))
                    steps += 1
                generated = time.perf_counter_ns()
                terminal.stage(requests)
                root.wait_stream(main)
                with torch.npu.stream(root):
                    drain.replay()
                torch.npu.synchronize()
                finished = time.perf_counter_ns()
                return (
                    dict(
                        requests=[
                            dict(
                                owner=r.index,
                                tokens=r.tokens,
                                finish_reason=r.finish_reason,
                                accepted=r.accepted,
                                processed=r.processed,
                                q4_steps=r.q4_steps,
                            )
                            for r in requests
                        ],
                        steps=records,
                        output_tokens=sum(len(r.tokens) for r in requests),
                        loop_output_tokens=sum(len(r.tokens) - 1 for r in requests),
                    ),
                    dict(
                        generation_us=(generated - begin) / 1000,
                        terminal_us=(finished - generated) / 1000,
                        component_us=components,
                    ),
                    output_hashes,
                )

            restore_complete(torch, bundle, registry, initial)
            validation, _, output_hashes = generate(audit_outputs=True)
            terminal.require_closed()
            closed_state = state_digest(torch, bundle, registry)
            report["validation"] = dict(
                generation_sha256=digest(validation), output_hashes=output_hashes, closed_state=closed_state
            )
            checkpoint()
            print(
                f"rank={rank} {args.mode} validation steps={len(validation['steps'])} "
                f"output={validation['output_tokens']}",
                flush=True,
            )
            restore_complete(torch, bundle, registry, initial)
            dist.barrier(group=group)
            generated, measured, _ = generate(audit_outputs=False)
            terminal.require_closed()
            if generated != validation or state_digest(torch, bundle, registry) != closed_state:
                raise AssertionError("timed continuous generation differs from its own untimed validation")
            local = dict(
                rank=rank,
                prefill_us=prefill_us,
                state_import_us=state_import_us,
                generation_us=measured["generation_us"],
                terminal_us=measured["terminal_us"],
                batch_us=prefill_us + state_import_us + measured["generation_us"] + measured["terminal_us"],
                component_us=measured["component_us"],
            )
            ranks = [None, None]
            dist.all_gather_object(ranks, local, group=group)
            report.update(
                status=COMPLETE,
                generation=generated,
                cost_ranks=ranks,
                critical_cost=critical_cost(ranks),
                timed_matches_validation=True,
                terminal_contract=dict(
                    accepted=[r["accepted"] for r in generated["requests"]],
                    cleared_logs=True,
                    conv_and_fia="retained; actual final selectors recorded",
                ),
                timing_boundary=(
                    "phase sum: real prefill/first head + one state import + "
                    "post-capture warmed generation + final commit/wait"
                ),
                timing_exclusions=[
                    "loading",
                    "cache allocation",
                    "capture",
                    "validation",
                    "snapshot/restore",
                    "state downloads",
                    "JSON writes",
                ],
                capture_and_validation_gap_excluded=True,
            )
            checkpoint()
            print(f"rank={rank} {args.mode} COMPLETE generation_us={measured['generation_us']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("extension", "products-extension", "deferred-extension", "tile-extension"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.manifest = load_manifest(args.manifest)
    args.extensions = [
        getattr(args, k) for k in ("extension", "products_extension", "deferred_extension", "tile_extension")
    ]
    if args.output.with_suffix(".rank0.json").exists():
        parser.error("preserve prior generation evidence")
    for name, expected in args.manifest["payload"]["model_files"].items():
        if hashlib.sha256((Path(args.model) / name).read_bytes()).hexdigest() != expected:
            parser.error("frozen model/tokenizer file changed: " + name)
    args.tensor_parallel_size, args.load_start = 2, time.perf_counter_ns()
    run_full_model(args, capacity_trace(args.manifest), experiment=experiment)


if __name__ == "__main__":
    main()
