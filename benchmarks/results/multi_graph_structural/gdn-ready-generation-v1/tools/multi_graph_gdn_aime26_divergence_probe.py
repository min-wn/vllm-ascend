# SPDX-License-Identifier: Apache-2.0
"""Untimed, bounded Q28 common-input replay and native local-state diagnosis.

The frozen native libraries, model, cache allocator and forward calls are reused.
Each continuous control performs its own zero-state prefill and never imports
another control's state. Only isolated recurrence calls share a selected state.
"""

import argparse
import gc
import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path


def cpu(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(torch, v) for v in value]
    return value


def clone(torch, value):
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def sha(torch, value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def logits_record(torch, logits, tokenizer, candidates):
    values = logits.float().cpu()
    top_values, top_ids = values.topk(10)
    return dict(
        dtype=str(logits.dtype),
        argmax=int(values.argmax()),
        top1_top2_gap=float(top_values[0] - top_values[1]),
        top10=[
            dict(token=int(i), text=tokenizer.decode([int(i)]), logit=float(v))
            for i, v in zip(top_ids, top_values, strict=True)
        ],
        candidates={
            str(i): dict(
                logit=float(values[i]),
                rank_strict_greater_plus_one=int((values > values[i]).sum()) + 1,
                tied_tokens=int((values == values[i]).sum()),
            )
            for i in candidates
        },
    )


@contextmanager
def observe_final(torch, model, bundle, registry, observed, target_row):
    import vllm_ascend.ops.gdn as gdn

    original = gdn.recurrent_gated_delta_rule
    visits = []

    def call(query, key, value, state, **kwargs):
        entry = bundle.gdn_entries[len(visits)]
        visits.append(entry.index)
        if entry.index == 0:
            observed["recurrence_inputs"] = {
                k: clone(torch, v) for k, v in dict(query=query, key=key, value=value, **kwargs).items()
            }
            if registry is None:
                slots = kwargs["ssm_state_indices"].reshape(-1, 4).long()
                counts = kwargs["num_accepted_tokens"].reshape(-1).long()
                selected = slots[torch.arange(len(counts), device=state.device), counts - 1]
                observed["canonical_base"] = state.index_select(0, selected).clone()
            elif registry.mode == "deferred_tile_ready":
                observed["continuous_compact_initial"] = [t.clone() for t in entry.state.mutable_tensors]
        result = original(query, key, value, state, **kwargs)
        observed.setdefault("recurrences", {})[entry.index] = result[target_row * 4 : (target_row + 1) * 4].clone()
        return result

    handles = []
    gdn.recurrent_gated_delta_rule = call
    try:
        for index, layer in enumerate(model.language_model.model.layers):

            def hook(module, args, result, index=index):
                values = result if isinstance(result, (tuple, list)) else (result,)
                observed.setdefault("decoders", {})[index] = [
                    v[target_row * 4 : (target_row + 1) * 4].clone() if isinstance(v, torch.Tensor) else v
                    for v in values
                ]

            handles.append(layer.register_forward_hook(hook))
        yield
        if visits != [entry.index for entry in bundle.gdn_entries]:
            raise AssertionError("final step did not visit every GDN layer exactly once")
    finally:
        gdn.recurrent_gated_delta_rule = original
        for handle in handles:
            handle.remove()


def state_hashes(torch, bundle, registry):
    result = {}
    for entry in bundle.gdn_entries:
        tensors = entry.state.mutable_tensors if registry is not None else (entry.gdn.kv_cache[1],)
        result[str(entry.index)] = dict(state=[sha(torch, t) for t in tensors], conv=sha(torch, entry.conv))
    return result


def run_control(torch, torch_npu, dist, model, config, tokenizer, modules, trace, capture_sizes, mode, output):
    from multi_graph_gdn_aime26_probe import (
        allocate_generation_caches,
        compact_registry,
        official_call_counter,
        transition_official,
    )
    from multi_graph_gdn_q4_full_metadata import FullMetadataFactory
    from multi_graph_gdn_q4_log_backend import installed_log_registry
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    rows, site = trace["rows"], trace["site"]
    capacity = min(n for n in capture_sizes if n % 4 == 0 and n >= len(rows) * 4)
    if capacity != site["capacity_tokens"] or site["step"] != 150 or len(trace["steps"]) != 151:
        raise ValueError("retain the original B6/default-24-token first-divergence boundary")
    bundle = allocate_generation_caches(torch, torch_npu, model, config, rows, capture_sizes)
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
            context.slot_mapping = {e.backend.layer_name: frame.slot_mapping for e in bundle.fia_entries}
            context.capturing = False
            hidden = model(input_ids=ids, positions=frame.positions)
            if registry is not None:
                registry.finish_pending_commits()
        return hidden

    def head(frame, hidden):
        with set_ascend_forward_context(
            frame.mapping,
            config,
            num_tokens=len(hidden),
            num_actual_tokens=len(hidden),
            model_instance=model,
            skip_compiled=True,
        ):
            return model.language_model.compute_logits(hidden)

    initial, first = {}, []
    for owner, row in enumerate(rows):
        frame = factory.prefix(owner, len(row["prompt_tokens"]))
        hidden = forward(frame, torch.tensor(row["prompt_tokens"], dtype=torch.int64, device=bundle.device))
        first.append(int(head(frame, hidden[-1:]).argmax(-1).cpu()[0]))
    for entry in bundle.gdn_entries:
        initial[str(entry.index)] = dict(base=sha(torch, entry.base), conv=sha(torch, entry.conv))
    for entry in bundle.fia_entries:
        # Only initialized prompt positions; independent capacity remains unchanged.
        initial[f"fia{entry.index}"] = [
            sha(
                torch,
                tensor.reshape(-1, 2, 256)[
                    torch.tensor(
                        [
                            bundle.pages.slot(o, p)
                            for o, row in enumerate(rows)
                            for p in range(len(row["prompt_tokens"]))
                        ],
                        device=bundle.device,
                    )
                ],
            )
            for tensor in (entry.key, entry.value)
        ]
    if first != [trace["original_first_tokens"][str(row["index"])] for row in rows]:
        raise AssertionError("same zero-state prefill did not reproduce the original first tokens")
    if mode == "official":
        transition_official(torch, bundle)
    else:
        bundle.transition_to_decode()
        registry = compact_registry(torch, bundle, modules, capacity)
        registry.mode = "deferred_tile_ready" if mode == "compact_ready" else "deferred_tile_delayed"
    installation = installed_log_registry(registry) if registry is not None else nullcontext()
    counter = official_call_counter(bundle) if registry is None else nullcontext()
    observed, steps = {}, []
    try:
        with installation, counter as official_counts:
            for step in trace["steps"]:
                frame = factory.decode(step, capacity)
                if registry is not None:
                    registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
                observer = (
                    observe_final(torch, model, bundle, registry, observed, site["row"])
                    if step["step"] == site["step"]
                    else nullcontext()
                )
                with observer:
                    hidden = forward(frame, torch.tensor(step["input_ids"], dtype=torch.int64, device=bundle.device))
                if registry is not None:
                    registry.finish_step(step["step"])
                logits = head(frame, hidden[: step["batch"] * 4])
                predictions = logits.argmax(-1).to(torch.int64)
                dist.broadcast(predictions, src=0)
                actual = predictions.cpu().tolist()
                steps.append(
                    dict(
                        step=step["step"],
                        predictions=actual,
                        all_predictions_equal_original=actual == step["expected_predictions"],
                        hidden_sha256=sha(torch, hidden),
                    )
                )
                if step["step"] == site["step"]:
                    observed["hidden"] = hidden.clone()
                    observed["logits"] = logits[site["row"] * 4 : (site["row"] + 1) * 4].clone()
                    selected_logits = logits[site["row"] * 4 + site["verification_offset"]]
                    top = logits_record(torch, selected_logits, tokenizer, (866, 660))
                if step["step"] % 50 == 0:
                    print(f"rank={dist.get_rank()} {mode} common-trace step={step['step']}", flush=True)
        torch.npu.synchronize()
        counts = [b.calls for _, b, _, _ in registry.entries] if registry is not None else None
        if counts is not None and counts != [151] * 48:
            raise AssertionError("a compact GDN layer left the tested Q4 path")
        if official_counts is not None and set(official_counts.values()) != {151}:
            raise AssertionError("an official GDN layer left the tested Q4 path")
        report = dict(
            mode=mode,
            schedule=trace["schedule_source"],
            independent_zero_prefill=True,
            independent_continuous_state=True,
            initial=initial,
            steps=steps,
            top_logits=top,
            final_states=state_hashes(torch, bundle, registry),
            compact_q4_calls=counts,
            official_q4_calls=official_counts,
            no_fallback=True,
            capacity_tokens=capacity,
            diagnostic_steps=151,
        )
        observed = cpu(torch, observed)
        torch.save(observed, output / f"{mode}.rank{dist.get_rank()}.pt")
        (output / f"{mode}.rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n")
        return observed, report
    finally:
        if registry is not None:
            registry._require_joined()
        for entry in bundle.gdn_entries:
            entry.gdn.kv_cache = None
        for entry in bundle.fia_entries:
            entry.backend.kv_cache = None
            entry.impl.key_cache = entry.impl.value_cache = None


def local_comparison(torch, modules, saved, target_row):
    from multi_graph_gdn_q4_log_backend import Q4LogBackend, Q4LogState

    import vllm_ascend.ops.gdn as gdn

    args = saved["recurrence_inputs"]
    kwargs = {
        k: v.to("npu") if isinstance(v, torch.Tensor) else v
        for k, v in args.items()
        if k not in ("query", "key", "value")
    }
    query, key, value = [args[k].to("npu") for k in ("query", "key", "value")]
    canonical = saved["canonical_base"].to("npu")
    count = len(canonical)
    candidates = tuple(tuple(1 + 4 * owner + t for t in range(4)) for owner in range(count))
    dense = torch.zeros((1 + 4 * count, 24, 128, 128), dtype=torch.float32, device="npu")
    dense[1:].view(count, 4, 24, 128, 128).copy_(canonical[:, None])
    official = gdn.recurrent_gated_delta_rule(query, key, value, dense, **kwargs)
    torch.npu.synchronize()
    result = dict(
        canonical_base=canonical.cpu(),
        recurrence_inputs=args,
        official_output=official.cpu(),
        official_candidates=dense[1:].view(count, 4, 24, 128, 128)[target_row : target_row + 1].cpu(),
        target_row=target_row,
        fresh_compact_logs=True,
        canonical_origin="selected FP32 official state before the final Q4; isolated calls only",
    )
    for mode in ("deferred_tile_delayed", "deferred_tile_ready"):
        base = torch.zeros((count + 1, 24, 128, 128), dtype=torch.float32, device="npu")
        base[1:].copy_(canonical)
        state = Q4LogState.from_base(torch, base, candidates)
        backend = Q4LogBackend(state, modules[0], modules[1], mode, deferred_module=modules[2], tile_module=modules[3])
        backend.reserve_exclusive_capacity(max_batch=count, commit_stream=torch.npu.Stream())
        imported_exact = torch.equal(base[1:], canonical)
        state.prepare_step(0, tuple(range(count)), kwargs["num_accepted_tokens"].cpu().tolist())
        actual = backend(query, key, value, state.handle, **kwargs)
        backend.launch_pending_commit()
        backend.finish_pending_commit()
        state.finish_step(0)
        exported = state.export(modules[0])
        torch.npu.synchronize()
        result[mode] = dict(
            output=actual.cpu(),
            candidates=exported[target_row + 1 : target_row + 2].cpu(),
            represented=[t[target_row + 1 : target_row + 2].cpu() for t in state.mutable_tensors],
            import_exact=bool(imported_exact),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("official", "compact_delayed", "compact_ready"),
        default=["official", "compact_delayed", "compact_ready"],
    )
    args = parser.parse_args()
    if int(os.environ.get("RANK", "0")) == 0:
        if args.output.exists():
            parser.error("use a new diagnostic output directory; no overwrites")
        args.output.mkdir(parents=True)
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
    trace = json.loads(args.trace.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if int(os.environ.get("RANK", "0")) == 0:
        (args.output / "token-text.json").write_text(
            json.dumps(
                dict(
                    tokens={str(t): tokenizer.decode([t]) for t in (866, 660)},
                    context={
                        v: tokenizer.decode(
                            json.loads(
                                Path("/workspace/gdn-aime26-v1/formal").joinpath(f"{v}.question27.json").read_bytes()
                            )["generated_tokens"][155:196]
                        )
                        for v in ("official", "compact")
                    },
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    default = engine_options(args.model, 2, eager=False)
    default["max_model_len"] = 82944
    captures = tuple(EngineArgs(**default).create_engine_config().compilation_config.cudagraph_capture_sizes)
    options = engine_options(args.model, 2, eager=True)
    options["max_model_len"] = 82944
    config = EngineArgs(**options).create_engine_config()
    executor = ExecutorWithExternalLauncher(config)
    try:
        runner = executor.driver_worker.model_runner
        model = runner.get_model()
        if getattr(runner, "kv_caches", None):
            raise AssertionError("retain the existing weight-only loader and explicitly owned caches")
        converted = 0
        for layer in model.language_model.model.layers:
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    weight.data = torch_npu.npu_format_cast(weight.data, 29)
                    converted += 1
        if converted != 304:
            raise AssertionError("preserve the common 304 decoder NZ29 weights")
        roots = Path("/workspace/multi-graph-results")
        modules = [
            load_extension(name, roots / suffix)
            for name, suffix in (
                ("gdn_q4_log", "q4-log-build-v3-ranges/gdn_q4_log.so"),
                ("gdn_q4_products", "q4-products-build-v9-ranges/gdn_q4_products.so"),
                ("gdn_q4_log_deferred", "q4-log-deferred-build-v1/gdn_q4_log_deferred.so"),
                ("gdn_q4_tile_pipeline", "q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so"),
            )
        ]
        official = None
        with torch.inference_mode(), set_current_vllm_config(config):
            for mode in args.modes:
                observed, report = run_control(
                    torch, torch_npu, dist, model, config, tokenizer, modules, trace, captures, mode, args.output
                )
                if mode == "official":
                    official = dict(
                        canonical_base=observed["canonical_base"], recurrence_inputs=observed["recurrence_inputs"]
                    )
                del observed, report
                gc.collect()
                torch.npu.empty_cache()
                dist.barrier()
            if official is not None:
                local = local_comparison(torch, modules, official, trace["site"]["row"])
                torch.save(local, args.output / f"local-layer0.rank{dist.get_rank()}.pt")
            dist.barrier()
        print(f"rank={dist.get_rank()} bounded Q28 diagnostic complete", flush=True)
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
