# SPDX-License-Identifier: Apache-2.0
"""Real-text LM-head quality audit; no kernel or precision changes, no timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

from multi_graph_gdn_q4_full_metadata import FrozenTrace, load_fixed_trace
from multi_graph_gdn_quality import LIMITS, SCHEMA, digest, score_layout, summarize

HEAD_CHUNK_TOKENS = 32


def load_inputs(path, protocol, trace):
    document = json.loads(path.read_text())
    data = document["payload"]
    if document["payload_sha256"] != digest(data) or data["schema"] != SCHEMA:
        raise ValueError("quality input manifest identity changed")
    if data["trace_sha256"] != trace.digest or data["limits"] != LIMITS:
        raise ValueError("trace or predeclared quality limits changed")
    if data["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
        raise ValueError("predeclared protocol changed")
    for name in ("screen", "task"):
        if len(data[name]) != 128 or [r["owner"] for r in data[name]] != list(range(128)):
            raise ValueError("exactly 128 stable request owners required")
    return document


def task_trace(data):
    prefixes = tuple(item["prefix"] for item in data["task"])
    step = MappingProxyType(
        dict(
            step=0,
            batch=128,
            request_ids=tuple(range(128)),
            first_use=(True,) * 128,
            seq_lengths=tuple(n + 4 for n in prefixes),
            accepted_from_previous=(1,) * 128,
            accepted_after_step=(4,) * 128,
            input_seed=0,
        )
    )
    return FrozenTrace((step,), prefixes, 42, digest(dict(prefixes=prefixes, kind="frozen_lambada_cloze")))


def task_summary(rows):
    owners = {}
    for row in rows:
        owners.setdefault(row["owner"], []).append(row)
    if set(owners) != set(range(128)):
        raise ValueError("incomplete limited task audit")
    paired = []
    for owner, tokens in owners.items():
        paired.append(
            dict(
                owner=owner,
                source_row=tokens[0]["article"],
                official=all(r["official_correct"] for r in tokens),
                compact=all(r["compact_correct"] for r in tokens),
                official_nll=sum(r["official_nll"] for r in tokens),
                compact_nll=sum(r["compact_nll"] for r in tokens),
            )
        )
    official = sum(r["official"] for r in paired)
    compact = sum(r["compact"] for r in paired)
    lost = sum(r["official"] and not r["compact"] for r in paired)
    gained = sum(r["compact"] and not r["official"] for r in paired)
    return dict(
        examples=128,
        official_correct=official,
        compact_correct=compact,
        official_accuracy=official / 128,
        compact_accuracy=compact / 128,
        new_failures=lost,
        recovered=gained,
        pairs=paired,
        passed=(official - compact) / 128 <= LIMITS["task_accuracy_drop"]
        and lost / 128 <= LIMITS["task_new_failure_fraction"],
    )


def experiment(args, trace, runner, model, config):
    import torch
    import torch.distributed as dist
    import torch_npu
    from multi_graph_gdn_full_numeric import OfficialBanks, error_record
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_full_metadata import FullMetadataFactory, allocate_full_caches
    from multi_graph_gdn_q4_log_backend import CompactSSMRegistry, Q4LogBackend, installed_log_registry
    from multi_graph_gdn_quality import distribution_metrics
    from multi_graph_gdn_ready_full_lifecycle import TerminalDrain
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    rank = dist.get_rank()
    torch.set_num_threads(8)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = True
    converted = 0
    for layer in model.language_model.model.layers:
        for module in layer.modules():
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 2:
                weight.data = torch_npu.npu_format_cast(weight.data, 29)
                converted += 1
    if converted != 304:
        raise ValueError("the established common decoder NZ format must remain unchanged")
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
    data = args.frozen["payload"]
    inputs = data[args.phase]
    vocabulary = config.model_config.hf_text_config.vocab_size
    report = dict(
        status="real_text_quality_in_progress",
        phase=args.phase,
        rank=rank,
        rows=[],
        hidden_diagnostics=[],
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        payload_sha256=args.frozen["payload_sha256"],
        loader=args.loader_metadata,
        selection=args.selection,
        precision=dict(weights="bfloat16", activations="bfloat16", ssm="float32", hf32=False),
        reference="unchanged installed official recurrence, independent CPU FP32 candidate banks",
        candidate="unchanged native compact FP32 deferred_tile_ready",
        scope="all 64 decoders + final norm + official LM head; eager accuracy, no performance claim",
        old_hidden_screen="0.005 failure retained; not a quality acceptance rule",
        lm_head=dict(
            weight_dtype=str(model.language_model.lm_head.weight.dtype),
            use_all_gather=model.language_model.logits_processor.use_all_gather,
            scale=model.language_model.logits_processor.scale,
            soft_cap=model.language_model.logits_processor.soft_cap,
            vocabulary=vocabulary,
            chunk_tokens=HEAD_CHUNK_TOKENS,
        ),
    )
    args.partial_report = report

    def save():
        path = args.output.with_suffix(f".rank{rank}.json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    def forward(frame, ids, registry=None):
        with set_ascend_forward_context(
            frame.mapping,
            config,
            num_tokens=frame.capacity_tokens,
            num_actual_tokens=frame.capacity_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=model,
            skip_compiled=True,
        ):
            ctx = get_forward_context()
            ctx.slot_mapping = {item.backend.layer_name: frame.slot_mapping for item in bundle.fia_entries}
            ctx.capturing = False
            hidden = model(input_ids=ids, positions=frame.positions)
            if registry is not None:
                registry.finish_pending_commits()
            return hidden

    prefix_hash = hashlib.sha256()
    for owner, length in enumerate(trace.initial_prefix_by_request):
        frame = factory.prefix(owner, length)
        ids = torch.tensor(inputs[owner]["tokens"][:length], dtype=torch.int64)
        prefix_hash.update(ids.numpy().tobytes())
        hidden = forward(frame, ids.to(bundle.device))
        if not torch.isfinite(hidden).all().item():
            raise ValueError("nonfinite common official prefill")
        if rank == 0 and (owner % 16 == 0 or owner == 127):
            print(f"quality {args.phase} prefill {owner + 1}/128", flush=True)
    torch.npu.synchronize()
    report["prefill"] = dict(
        owners=128,
        tokens=sum(trace.initial_prefix_by_request),
        token_sha256=prefix_hash.hexdigest(),
        split="both independent histories cloned from the same complete official prefill",
    )
    bundle.transition_to_decode()
    banks = OfficialBanks(torch, bundle)
    modules = [
        load_extension(name, path)
        for name, path in (
            ("gdn_q4_log", args.extension),
            ("gdn_q4_products", args.products_extension),
            ("gdn_q4_log_deferred", args.deferred_extension),
            ("gdn_q4_tile_pipeline", args.tile_extension),
        )
    ]
    registry = CompactSSMRegistry()
    stream = torch.npu.Stream()
    for item in bundle.gdn_entries:
        backend = Q4LogBackend(
            item.state,
            modules[0],
            modules[1],
            "deferred_tile_ready",
            deferred_module=modules[2],
            tile_module=modules[3],
        )
        backend.reserve_exclusive_capacity(max_batch=max(args.selection["required_tokens"]) // 4, commit_stream=stream)
        registry.register(item.state.handle, backend, item.gdn.out_proj)
    if args.phase == "screen":
        layout = score_layout(trace)
    else:
        layout = [
            [
                dict(flat=owner * 4 + j, owner=owner, position=item["prefix"] + 1 + j)
                for owner, item in enumerate(inputs)
                for j in range(len(item["target_tokens"]))
            ]
        ]
    token_hash = hashlib.sha256()
    for step, positions in zip(trace.steps, layout, strict=True):
        capacity = args.selection["dispatched_tokens"][step["step"]]
        frame = factory.decode(step, capacity)
        ids = torch.zeros(capacity, dtype=torch.int64)
        for row, (owner, length) in enumerate(zip(step["request_ids"], step["seq_lengths"], strict=True)):
            ids[row * 4 : row * 4 + 4] = torch.tensor(inputs[owner]["tokens"][length - 4 : length])
        token_hash.update(ids.numpy().tobytes())
        ids = ids.to(bundle.device)
        with banks.installed(model, step["request_ids"]):
            reference = forward(frame, ids).clone()
            torch.npu.synchronize()
        registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
        with installed_log_registry(registry):
            actual = forward(frame, ids, registry)
            torch.npu.synchronize()
        registry.finish_step(step["step"])
        report["hidden_diagnostics"].append(
            error_record(
                torch,
                actual[: step["batch"] * 4].reshape(step["batch"], 4, -1),
                reference[: step["batch"] * 4].reshape(step["batch"], 4, -1),
                label="hidden",
                step=step["step"],
                owners=step["request_ids"],
            )
        )
        for offset in range(0, len(positions), HEAD_CHUNK_TOKENS):
            group = positions[offset : offset + HEAD_CHUNK_TOKENS]
            indices = torch.tensor([r["flat"] for r in group], device=bundle.device)
            targets = torch.tensor([inputs[r["owner"]]["tokens"][r["position"]] for r in group], device=bundle.device)
            # Both TP ranks must execute both calls, including gather-only ranks.
            with set_ascend_forward_context(
                frame.mapping,
                config,
                num_tokens=len(group),
                num_actual_tokens=len(group),
                model_instance=model,
                skip_compiled=True,
            ):
                ref_logits = model.language_model.compute_logits(reference.index_select(0, indices))
                logits = model.language_model.compute_logits(actual.index_select(0, indices))
            if rank == 0:
                if ref_logits is None or logits is None or ref_logits.shape != (len(group), vocabulary):
                    raise ValueError("official LM head did not return the complete unpadded vocabulary on rank 0")
                report["lm_head"]["logits_dtype"] = str(logits.dtype)
                metrics = distribution_metrics(torch, ref_logits, logits, targets)
                for j, item in enumerate(group):
                    report["rows"].append(
                        dict(
                            step=step["step"],
                            owner=item["owner"],
                            position=item["position"],
                            article=inputs[item["owner"]]["article"],
                            target=targets[j].item(),
                            **{key: value[j] for key, value in metrics.items()},
                        )
                    )
            del logits, ref_logits
        report.update(completed_steps=step["step"] + 1, token_sha256=token_hash.hexdigest())
        save()
        if rank == 0:
            print(f"quality {args.phase} step={step['step']} scored={len(report['rows'])}", flush=True)
    terminal = TerminalDrain(torch, bundle, registry)
    terminal.run()
    torch.npu.synchronize()
    terminal.require_closed()
    report["terminal_closed"] = True
    if rank == 0:
        report["summary"] = (
            summarize(report["rows"], expected_tokens=data["screen_expected_tokens"])
            if args.phase == "screen"
            else task_summary(report["rows"])
        )
    report["status"] = "complete_real_text_quality_audit"
    save()
    dist.barrier()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "task"), required=True)
    parser.add_argument("--screen-result", type=Path)
    for flag, path in (
        ("extension", "q4-log-build-v3-ranges/gdn_q4_log.so"),
        ("products-extension", "q4-products-build-v9-ranges/gdn_q4_products.so"),
        ("deferred-extension", "q4-log-deferred-build-v1/gdn_q4_log_deferred.so"),
        ("tile-extension", "q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so"),
    ):
        parser.add_argument("--" + flag, type=Path, default=Path("/workspace/multi-graph-results") / path)
    args = parser.parse_args()
    trace = load_fixed_trace(args.trace)
    args.frozen = load_inputs(args.manifest, args.protocol, trace)
    if args.phase == "task":
        if args.screen_result is None:
            parser.error("task comparison requires the completed predeclared screen")
        screen = json.loads(args.screen_result.read_text())
        if (
            screen["status"] != "complete_real_text_quality_audit"
            or not screen["summary"]["screen_passed"]
            or screen["payload_sha256"] != args.frozen["payload_sha256"]
        ):
            parser.error("initial quality screen did not pass for this exact frozen dataset")
        trace = task_trace(args.frozen["payload"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if args.output.with_suffix(f".rank{rank}.json").exists():
        parser.error("refusing to overwrite a prior quality run")
    from multi_graph_gdn_q4_full_loader import run_full_model

    try:
        run_full_model(args, trace, experiment=experiment)
    except BaseException as error:
        failure = getattr(args, "partial_report", {})
        failure.update(status="failed_real_text_quality_audit", error=repr(error))
        args.output.with_suffix(f".rank{rank}.failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
