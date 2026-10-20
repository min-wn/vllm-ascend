# SPDX-License-Identifier: Apache-2.0
"""Offline actual autoregressive AIME26 generation using unchanged Q4 kernels.

No teacher forcing, answer access, cross-control state injection, graph change,
precision change, timing comparison or automatic failed-sample retry.
"""

import argparse
import gc
import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

from multi_graph_gdn_aime26 import MAX_MODEL_LEN, QUERY, GenerationRequest, digest, load_manifest, terminal_metadata


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def allocate_generation_caches(torch, torch_npu, model, config, inputs, capture_sizes):
    from multi_graph_gdn_q4_full_metadata import (
        BLOCK_SIZE,
        FIACacheEntry,
        FrozenTrace,
        FullCacheBundle,
        GDNCacheEntry,
        reserve_owner_pages,
    )

    count = len(inputs)
    trace = FrozenTrace(
        (dict(request_ids=tuple(range(count)), seq_lengths=(MAX_MODEL_LEN,) * count),),
        tuple(len(r["prompt_tokens"]) for r in inputs),
        42,
        digest(dict(ids=[r["id"] for r in inputs], capacity=MAX_MODEL_LEN)),
    )
    pages = reserve_owner_pages(trace)
    # Every owner's reservation includes all padded speculative tails.
    all_pages = [page for group in pages.pages for page in group]
    if len(all_pages) != len(set(all_pages)) or pages.width * BLOCK_SIZE < MAX_MODEL_LEN:
        raise AssertionError("long-context FIA reservations overlap or are too short")
    for owner in range(count):
        pages.slot(owner, MAX_MODEL_LEN - 1)
    gdn_entries, fia_entries = [], []
    for index, layer in enumerate(model.language_model.model.layers):
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            conv_shape, ssm_shape = gdn.get_state_shape()
            conv_dtype, ssm_dtype = gdn.get_state_dtype()
            if tuple(ssm_shape) != (24, 128, 128) or ssm_dtype != torch.float32 or conv_dtype != torch.bfloat16:
                raise ValueError("unchanged TP2 FP32 SSM/BF16 conv contract required")
            base = torch.zeros((count + 1, *ssm_shape), dtype=ssm_dtype, device="npu")
            conv = torch.zeros((count + 1, *conv_shape), dtype=conv_dtype, device="npu")
            gdn.kv_cache = (conv, base)
            gdn_entries.append(GDNCacheEntry(index, gdn.prefix, gdn, base, conv))
        elif layer.layer_type == "full_attention":
            attention = layer.self_attn
            backend, impl = attention.attn, attention.attn.impl
            if (
                impl.use_bnsd_kv_cache
                or impl.pcp_enabled
                or impl.enable_c8_quant
                or impl.sinks is not None
                or impl.sliding_window is not None
                or impl.kv_sharing_target_layer_name is not None
                or attention.num_kv_heads != 2
                or attention.head_dim != 256
            ):
                raise ValueError("unchanged TP2 BF16 ND FIA path required")
            shape = (pages.page_count, BLOCK_SIZE, 2, 256)
            key, value = (
                torch_npu.npu_format_cast(torch.zeros(shape, dtype=torch.bfloat16, device="npu"), 2) for _ in range(2)
            )
            backend.kv_cache = (key, value)
            impl.key_cache, impl.value_cache = key, value
            fia_entries.append(FIACacheEntry(index, attention, backend, impl, key, value))
    if len(gdn_entries) != 48 or len(fia_entries) != 16:
        raise ValueError("all 64 decoder layers must be retained")
    return FullCacheBundle(torch, gdn_entries, fia_entries, pages, capture_sizes, torch.device("npu"), trace)


def transition_official(torch, bundle):
    count = len(bundle.pages.pages)
    for entry in bundle.gdn_entries:
        shape = (1 + count * QUERY, *entry.base.shape[1:])
        dense = torch.zeros(shape, dtype=torch.float32, device=bundle.device)
        dense[1:].view(count, QUERY, *entry.base.shape[1:]).copy_(entry.base[1:, None])
        conv = torch.zeros((1 + count * QUERY, *entry.conv.shape[1:]), dtype=entry.conv.dtype, device=bundle.device)
        conv[1:].view(count, QUERY, *entry.conv.shape[1:]).copy_(entry.conv[1:, None])
        entry.conv = conv
        entry.base = None
        entry.gdn.kv_cache = (conv, dense)
    bundle.decode_ready = True


@contextmanager
def official_call_counter(bundle):
    import vllm_ascend.ops.gdn as gdn_module

    original = gdn_module.recurrent_gated_delta_rule
    counts = {entry.index: 0 for entry in bundle.gdn_entries}
    identities = {entry.gdn.kv_cache[1].data_ptr(): entry.index for entry in bundle.gdn_entries}

    def counted(q, k, v, state, **kwargs):
        if kwargs.get("num_accepted_tokens") is None or state.data_ptr() not in identities:
            raise ValueError("official decode unexpectedly left the Q4 FP32 recurrence path")
        result = original(q, k, v, state, **kwargs)
        counts[identities[state.data_ptr()]] += 1
        return result

    gdn_module.recurrent_gated_delta_rule = counted
    try:
        yield counts
    finally:
        gdn_module.recurrent_gated_delta_rule = original


def compact_registry(torch, bundle, modules, capacity):
    from multi_graph_gdn_q4_log_backend import CompactSSMRegistry, Q4LogBackend

    registry, stream = CompactSSMRegistry(), torch.npu.Stream()
    for entry in bundle.gdn_entries:
        backend = Q4LogBackend(
            entry.state,
            modules[0],
            modules[1],
            "deferred_tile_ready",
            deferred_module=modules[2],
            tile_module=modules[3],
        )
        backend.reserve_exclusive_capacity(max_batch=capacity // QUERY, commit_stream=stream)
        registry.register(entry.state.handle, backend, entry.gdn.out_proj, name=f"gdn{entry.index}")
    return registry


def terminal_commit(torch, bundle, registry, requests):
    if registry is None:
        return dict(kind="official selected candidate retained", selectors=[r.accepted for r in requests])
    metadata = terminal_metadata([r.accepted for r in requests])
    sequence, ids, accepted = [
        torch.tensor(metadata[key], dtype=torch.int32, device=bundle.device) for key in ("sequence", "ids", "accepted")
    ]
    registry._require_joined()
    for entry in bundle.gdn_entries:
        state = entry.state
        registry.backends[state.handle].deferred_module.commit(
            state.base,
            sequence,
            ids,
            accepted,
            state.logs,
            state.valid,
            torch.zeros_like(state.logs),
            torch.zeros_like(state.valid),
        )
    torch.npu.synchronize()
    if any(
        torch.count_nonzero(e.state.valid).item() or torch.count_nonzero(e.state.logs).item()
        for e in bundle.gdn_entries
    ):
        raise AssertionError("terminal compact state acceptance debt remains")
    return dict(
        kind="actual final accepted prefix materialized; logs cleared", selectors=[r.accepted for r in requests]
    )


def execute_group(args, rows, torch, torch_npu, dist, model, config, tokenizer, modules, group_id, *, boundary=False):
    import numpy as np
    from multi_graph_gdn_q4_full_metadata import FullMetadataFactory
    from multi_graph_gdn_q4_log_backend import installed_log_registry
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    rank = dist.get_rank()
    capacity = min(size for size in args.capture_sizes if size % QUERY == 0 and size >= len(rows) * QUERY)
    bundle = allocate_generation_caches(torch, torch_npu, model, config, rows, args.capture_sizes)
    factory = FullMetadataFactory(torch, bundle, config)
    registry = None

    def forward(frame, ids):
        with set_ascend_forward_context(
            frame.mapping,
            config,
            num_tokens=frame.capacity_tokens,
            num_actual_tokens=frame.capacity_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=model,
            skip_compiled=True,
        ):
            context = get_forward_context()
            context.slot_mapping = {item.backend.layer_name: frame.slot_mapping for item in bundle.fia_entries}
            context.capturing = False
            hidden = model(input_ids=ids, positions=frame.positions)
            if registry is not None:
                registry.finish_pending_commits()
        if not torch.isfinite(hidden).all().item():
            raise AssertionError("nonfinite decoder output")
        return hidden

    def predictions(hidden, frame):
        with set_ascend_forward_context(
            frame.mapping,
            config,
            num_tokens=len(hidden),
            num_actual_tokens=len(hidden),
            model_instance=model,
            skip_compiled=True,
        ):
            logits = model.language_model.compute_logits(hidden)
        if logits is None or logits.shape != (len(hidden), 248320) or not torch.isfinite(logits).all().item():
            raise AssertionError("LM head must expose the full finite official vocabulary on both TP ranks")
        result = logits.argmax(-1).to(torch.int64)
        dist.broadcast(result, src=0)
        return result.cpu().tolist()

    requests = [
        GenerationRequest(
            row["index"],
            tuple(row["prompt_tokens"]),
            max_new_tokens=(25 + owner) if args.qualify else args.manifest["payload"]["generation"]["max_new_tokens"],
        )
        for owner, row in enumerate(rows)
    ]
    for owner, request in enumerate(requests):
        frame = factory.prefix(owner, len(request.prompt))
        hidden = forward(frame, torch.tensor(request.prompt, dtype=torch.int64, device=bundle.device))
        request.begin(predictions(hidden[-1:], frame)[0])
    if args.variant == "compact":
        bundle.transition_to_decode()
        registry = compact_registry(torch, bundle, modules, capacity)
        counter_context = nullcontext()
        install_context = installed_log_registry(registry)
    else:
        transition_official(torch, bundle)
        counter_context = official_call_counter(bundle)
        install_context = nullcontext()
    boundary_record = None
    # Synthetic maximum-position check is entirely before formal question runs.
    # Its arbitrary KV/SSM history is never reused for actual generation.
    if boundary:
        boundary_record = boundary_fia_probe(torch, bundle, factory, config, model, capacity)
    stem = f"{args.variant}.shard{args.shard}.group{group_id}"
    journal = None
    if rank == 0:
        journal = (args.output / f"{stem}.steps.jsonl").open("x", encoding="utf-8")
    steps, maximum_position = 0, 0
    try:
        with counter_context as official_counts, install_context:
            while active := [owner for owner, request in enumerate(requests) if not request.finish_reason]:
                step = dict(
                    step=steps,
                    batch=len(active),
                    request_ids=tuple(active),
                    seq_lengths=tuple(requests[o].processed + QUERY for o in active),
                    accepted_from_previous=tuple(requests[o].accepted for o in active),
                )
                if max(step["seq_lengths"]) > MAX_MODEL_LEN:
                    raise AssertionError("actual Q4 query exceeds the preallocated long-context boundary")
                drafts, ids = [], torch.zeros(capacity, dtype=torch.int64)
                for row, owner in enumerate(active):
                    request = requests[owner]
                    history = np.asarray([*request.prompt, *request.tokens], dtype=np.int32)
                    proposed = _find_longest_matched_ngram_and_propose_tokens(history, 2, 4, MAX_MODEL_LEN, 3).tolist()
                    drafts.append(proposed)
                    ids[row * QUERY : (row + 1) * QUERY] = torch.tensor(request.input_tokens(proposed))
                frame = factory.decode(step, capacity)
                if registry is not None:
                    registry.prepare_step(steps, active, step["accepted_from_previous"])
                hidden = forward(frame, ids.to(bundle.device))
                if registry is not None:
                    registry.finish_step(steps)
                target_tokens = predictions(hidden[: len(active) * QUERY], frame)
                decisions = [
                    requests[owner].advance(proposal, target_tokens[row * QUERY : (row + 1) * QUERY])
                    for row, (owner, proposal) in enumerate(zip(active, drafts, strict=True))
                ]
                maximum_position = max(maximum_position, max(step["seq_lengths"]) - 1)
                if registry is not None:
                    counts = [b.calls for _, b, _, _ in registry.entries]
                else:
                    counts = list(official_counts.values())
                if counts != [steps + 1] * 48:
                    raise AssertionError("every real GDN layer must execute exactly one Q4 recurrence per model step")
                if rank == 0:
                    record = dict(
                        **step,
                        proposals=drafts,
                        predictions=target_tokens,
                        decisions=decisions,
                        question_indices=[requests[o].index for o in active],
                        q4_calls=48,
                        fallback_calls=0,
                    )
                    journal.write(json.dumps(record, separators=(",", ":")) + "\n")
                    for owner, decision in zip(active, decisions, strict=True):
                        if decision["finish_reason"]:
                            write_request(args, tokenizer, requests[owner], group_id)
                    if steps % 128 == 0:
                        journal.flush()
                        save_json(
                            args.output / f"{stem}.progress.json",
                            dict(
                                steps=steps + 1,
                                active=len(active),
                                generated=[len(r.tokens) for r in requests],
                                maximum_position=maximum_position,
                                q4_layer_calls=counts,
                                fallback_calls=0,
                            ),
                        )
                        print(
                            f"aime26 {stem} step={steps} active={len(active)} "
                            f"tokens={[len(r.tokens) for r in requests]}",
                            flush=True,
                        )
                steps += 1
            terminal = terminal_commit(torch, bundle, registry, requests)
            if boundary:
                # After completion only: full decoder at maximum position. This
                # corrupts this disposable qualification cache and is never a
                # starting state for any AIME question.
                synthetic = dict(
                    step=steps,
                    batch=len(rows),
                    request_ids=tuple(range(len(rows))),
                    seq_lengths=(MAX_MODEL_LEN,) * len(rows),
                    accepted_from_previous=tuple(r.accepted for r in requests),
                )
                if registry is not None:
                    registry.prepare_step(steps, synthetic["request_ids"], synthetic["accepted_from_previous"])
                frame = factory.decode(synthetic, capacity)
                hidden = forward(frame, torch.zeros(capacity, dtype=torch.int64, device=bundle.device))
                predictions(hidden, frame)
                if registry is not None:
                    registry.finish_step(steps)
                boundary_record["full_decoder_finite_at_max_position"] = True
                boundary_record["positions"] = [MAX_MODEL_LEN - QUERY, MAX_MODEL_LEN - 1]
        result = dict(
            variant=args.variant,
            shard=args.shard,
            group=group_id,
            indices=[r.index for r in requests],
            steps=steps,
            q4_layer_calls=[b.calls for _, b, _, _ in registry.entries] if registry else list(official_counts.values()),
            per_question_q4_steps={r.index: r.q4_steps for r in requests},
            fallback_calls=0,
            terminal=terminal,
            maximum_position=maximum_position,
            boundary=boundary_record,
            capacity_tokens=capacity,
            kv_page_count=bundle.pages.page_count,
            kv_shape=list(bundle.fia_entries[0].key.shape),
            hbm_allocated=torch.npu.memory_allocated(),
            hbm_peak=torch.npu.max_memory_allocated(),
            status="complete",
            independent_zero_initialized_prefill=True,
        )
        if rank == 0:
            for request in requests:
                write_request(args, tokenizer, request, group_id)
            save_json(args.output / f"{stem}.result.json", result)
        return result
    except BaseException as error:
        if rank == 0:
            for request in requests:
                write_request(args, tokenizer, request, group_id, error=repr(error))
        raise
    finally:
        if journal is not None:
            journal.close()
        # Release the entire group; no inter-question or cross-control reuse.
        for entry in bundle.gdn_entries:
            entry.gdn.kv_cache = None
        for entry in bundle.fia_entries:
            entry.backend.kv_cache = None
            entry.impl.key_cache = entry.impl.value_cache = None


def write_request(args, tokenizer, request, group, error=None):
    raw = tokenizer.decode(request.tokens, skip_special_tokens=False)
    score_text = tokenizer.decode(request.tokens, skip_special_tokens=True)
    record = dict(
        index=request.index,
        variant=args.variant,
        group=group,
        prompt_tokens=len(request.prompt),
        generated_tokens=request.tokens,
        generated_token_count=len(request.tokens),
        raw_output=raw,
        scorer_text=score_text,
        finish_reason="runtime_error" if error else request.finish_reason or "runtime_error",
        output_stop_reason=request.finish_reason,
        error=error,
        length_limited=request.finish_reason == "length",
        q4_steps=request.q4_steps,
        processed=request.processed,
        final_accepted=request.accepted,
        fallback_calls=0,
        independent_history=True,
        manifest_sha256=args.manifest["payload_sha256"],
    )
    save_json(args.output / f"{args.variant}.question{request.index:02}.json", record)


def boundary_fia_probe(torch, bundle, factory, config, model, capacity):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    # Directly address the highest reserved owner page. Only the last query
    # should see its nonzero sentinel: this tests both sparse causal alignment
    # and access beyond the old 2048-token validation boundary.
    entry = bundle.fia_entries[0]
    key = torch.zeros_like(entry.key)
    value = torch.zeros_like(entry.value)
    owner = len(bundle.pages.pages) - 1
    last_slot = bundle.pages.slot(owner, MAX_MODEL_LEN - 1)
    value.reshape(-1, 2, 256)[last_slot].fill_(128)
    count = len(bundle.pages.pages)
    step = dict(
        batch=count,
        request_ids=tuple(range(count)),
        seq_lengths=(MAX_MODEL_LEN,) * count,
        accepted_from_previous=(1,) * count,
    )
    frame = factory.decode(step, capacity)
    query = torch.zeros(
        (capacity, entry.impl.num_heads, entry.impl.head_size), dtype=torch.bfloat16, device=bundle.device
    )
    output = torch.empty_like(query)
    prior = entry.impl.key_cache, entry.impl.value_cache
    try:
        entry.impl.key_cache, entry.impl.value_cache = key, value
        with set_ascend_forward_context(
            frame.mapping,
            config,
            num_tokens=capacity,
            num_actual_tokens=capacity,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=model,
            skip_compiled=True,
        ):
            output = entry.impl.forward_fused_infer_attention(query, key, value, frame.fia, output, (key, value))
    finally:
        entry.impl.key_cache, entry.impl.value_cache = prior
    observed = output.float().cpu()
    expected = 128 / MAX_MODEL_LEN
    final_query = (owner + 1) * QUERY - 1
    if not torch.isfinite(observed).all() or observed[:final_query].abs().max().item() > 1e-6:
        raise AssertionError("long-context FIA causal sentinel leaked to earlier queries")
    error = (observed[final_query] - expected).abs().max().item() / expected
    if error > 0.02:
        raise AssertionError("long-context FIA did not consume the last reserved page")
    return dict(
        sequence_length=MAX_MODEL_LEN,
        last_slot=last_slot,
        relative_error=error,
        sentinel_expected=expected,
        all_queries_finite=True,
        dispatch="installed real FIA forward_fused_infer_attention",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("official", "compact"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    parser.add_argument("--qualify", action="store_true")
    for flag, path in (
        ("extension", "q4-log-build-v3-ranges/gdn_q4_log.so"),
        ("products-extension", "q4-products-build-v9-ranges/gdn_q4_products.so"),
        ("deferred-extension", "q4-log-deferred-build-v1/gdn_q4_log_deferred.so"),
        ("tile-extension", "q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so"),
    ):
        parser.add_argument("--" + flag, type=Path, default=Path("/workspace/multi-graph-results") / path)
    args = parser.parse_args()
    args.manifest = load_manifest(args.manifest)
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("exactly TP2 ranks required")
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob(f"{args.variant}.shard{args.shard}.group*.result.json")):
        parser.error("refusing to repeat an existing paired generation")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import torch.distributed as dist
    import torch_npu
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_full_loader import engine_options
    from transformers import AutoTokenizer
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher

    torch.set_num_threads(8)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for name, identity in args.manifest["payload"]["model_files"].items():
        if hashlib.sha256((Path(args.model) / name).read_bytes()).hexdigest() != identity:
            raise ValueError(f"frozen model/tokenizer config changed: {name}")
    for row in args.manifest["payload"]["inputs"]:
        prompt = tokenizer.apply_chat_template(
            [dict(role="user", content=row["prompt"])],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if prompt != row["prompt_tokens"]:
            raise ValueError("frozen chat template tokenization mismatch")
    options = engine_options(args.model, 2, eager=False)
    options["max_model_len"] = MAX_MODEL_LEN
    default_config = EngineArgs(**options).create_engine_config()
    args.capture_sizes = tuple(default_config.compilation_config.cudagraph_capture_sizes)
    options = engine_options(args.model, 2, eager=True)
    options["max_model_len"] = MAX_MODEL_LEN
    config = EngineArgs(**options).create_engine_config()
    executor = ExecutorWithExternalLauncher(config)
    try:
        runner = executor.driver_worker.model_runner
        model = runner.get_model()
        if getattr(runner, "kv_caches", None):
            raise ValueError("weight-only loader unexpectedly allocated production caches")
        converted = 0
        for layer in model.language_model.model.layers:
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    weight.data = torch_npu.npu_format_cast(weight.data, 29)
                    converted += 1
        if converted != 304:
            raise ValueError("both controls must retain the established common 304 NZ29 projections")
        modules = []
        if args.variant == "compact":
            modules = [
                load_extension(name, path)
                for name, path in (
                    ("gdn_q4_log", args.extension),
                    ("gdn_q4_products", args.products_extension),
                    ("gdn_q4_log_deferred", args.deferred_extension),
                    ("gdn_q4_tile_pipeline", args.tile_extension),
                )
            ]
        groups = [
            args.manifest["payload"]["inputs"][i : i + 8]
            for i in range(args.shard * 16, min(30, (args.shard + 1) * 16), 8)
        ]
        if args.qualify:
            text = "Diagnostic only: write the integers from one to twenty, then stop."
            prompt = tokenizer.apply_chat_template(
                [dict(role="user", content=text)],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            groups = [
                [
                    dict(index=-i - 1 - offset * 8, id=f"qualification-{offset}-{i}", prompt_tokens=prompt)
                    for i in range(count)
                ]
                for offset, count in enumerate((8, 6))
            ]
        with torch.inference_mode(), set_current_vllm_config(config):
            for offset, rows in enumerate(groups):
                execute_group(
                    args,
                    rows,
                    torch,
                    torch_npu,
                    dist,
                    model,
                    config,
                    tokenizer,
                    modules,
                    args.shard * 2 + offset,
                    boundary=args.qualify,
                )
                gc.collect()
                torch.npu.empty_cache()
                dist.barrier()
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
