# SPDX-License-Identifier: Apache-2.0
"""Bounded Q28 LM-head diagnosis; no changes to decoder or state algorithms.

``site`` projects the saved B6/Q4 hidden states through the existing vLLM
head_dtype path. ``generate`` is gated by both ranks' complete-vocabulary site
results and calls the frozen autoregressive harness for Q28 only, at B1/Q4.
The B1 comparison is independent of the old B6 evaluation.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

VOCABULARY = 248320
SITE_ROW = 12
EXPECTED_TOKEN = 866
BF16 = "torch.bfloat16"
FP32 = "torch.float32"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generation_gate(reports):
    """Fail closed: a post-logit cast cannot pass the operator audit."""
    if len(reports) != 2 or {r["rank"] for r in reports} != {0, 1}:
        return False
    for report in reports:
        if not report["weight_roundtrip_exact"] or report["weight_dtype"] != BF16:
            return False
        for name in ("official", "compact_ready"):
            sample = report["samples"][name]
            projection = sample["float32"]
            if not sample["bfloat16"]["saved_logits_exact"]:
                return False
            if not projection["fp32_operands_and_output"] or projection["dtype"] != FP32:
                return False
            if projection["vocabulary"] != VOCABULARY or projection["argmax"] != EXPECTED_TOKEN:
                return False
            if not projection["finite"] or projection["top1_top2_gap"] <= 0:
                return False
    return True


def tensor_info(torch, value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, (tuple, list)):
        return [tensor_info(torch, item) for item in value]
    return str(value)


@contextmanager
def projection_audit(torch, weight):
    """Observe the real F.linear operands, not the post-projection cast.

    This Ascend build suppresses TorchDispatchMode events inside the installed
    custom op. These scoped wrappers delegate once to each unchanged original
    function and record only shape/dtype/storage identity. No tensor values are
    changed, and no converted weight is cached or installed.
    """
    import torch.nn.functional as functional

    observed = SimpleNamespace(operations=[])
    original_linear, original_to = functional.linear, torch.Tensor.to
    weight_pointer = weight.data_ptr()

    def linear(inputs, projected_weight, bias=None):
        result = original_linear(inputs, projected_weight, bias)
        observed.operations.append(
            dict(
                op="torch.nn.functional.linear",
                inputs=tensor_info(torch, (inputs, projected_weight)),
                output=tensor_info(torch, result),
                weight_is_original_storage=projected_weight.data_ptr() == weight_pointer,
            )
        )
        return result

    def convert(value, *positional, **keywords):
        result = original_to(value, *positional, **keywords)
        if value.data_ptr() == weight_pointer:
            observed.operations.append(
                dict(
                    op="lm_head.weight.to",
                    inputs=tensor_info(torch, (value,)),
                    output=tensor_info(torch, result),
                    allocated_new_storage=result.data_ptr() != value.data_ptr(),
                    output_bytes=result.numel() * result.element_size(),
                )
            )
        return result

    with patch.object(functional, "linear", linear), patch.object(torch.Tensor, "to", convert):
        yield observed


def real_fp32_projection(operations):
    return any(
        event["op"] in ("aten.mm.default", "torch.nn.functional.linear")
        and len(event["inputs"]) == 2
        and all(item.get("dtype") == FP32 for item in event["inputs"])
        and event["output"].get("dtype") == FP32
        for event in operations
    )


def measure(torch, dist, function, *, warmup=2, repeats=5):
    for _ in range(warmup):
        result = function()
        torch.npu.synchronize()
        del result
    rows = []
    for _ in range(repeats):
        dist.barrier()
        torch.npu.synchronize()
        initial = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        begin = time.perf_counter_ns()
        start.record()
        result = function()
        end.record()
        torch.npu.synchronize()
        elapsed = (time.perf_counter_ns() - begin) / 1e6
        rows.append(
            dict(
                wall_ms=elapsed,
                device_ms=start.elapsed_time(end),
                initial_allocated=initial,
                peak_extra_allocated=torch.npu.max_memory_allocated() - initial,
                retained_extra_allocated=torch.npu.memory_allocated() - initial,
            )
        )
        del result
    return dict(
        warmup=warmup,
        repeats=repeats,
        median_wall_ms=statistics.median(row["wall_ms"] for row in rows),
        median_device_ms=statistics.median(row["device_ms"] for row in rows),
        samples=rows,
    )


def site_probe(args, torch, dist, model, config, tokenizer):
    import torch.nn.functional as functional
    from multi_graph_gdn_aime26_divergence_probe import logits_record

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    language_model = model.language_model
    processor, head = language_model.logits_processor, language_model.lm_head
    rank = dist.get_rank()
    weight = head.weight
    if weight.dtype != torch.bfloat16 or processor.head_dtype != torch.float32:
        raise AssertionError("existing hf-overrides must select FP32 projection while retaining BF16 storage")
    result = dict(
        rank=rank,
        model_dtype=str(config.model_config.dtype),
        config_head_dtype=str(config.model_config.head_dtype),
        processor_type=f"{type(processor).__module__}.{type(processor).__name__}",
        processor_head_dtype=str(processor.head_dtype),
        weight_dtype=str(weight.dtype),
        weight_shape=list(weight.shape),
        weight_bytes=weight.numel() * weight.element_size(),
        weight_fp32_bytes=weight.numel() * 4,
        matmul=dict(
            allow_hf32=torch.npu.matmul.allow_hf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            allow_internal_format_requested=True,
        ),
        samples={},
    )
    weight32 = weight.float()
    result["weight_roundtrip_exact"] = bool(torch.equal(weight32.to(torch.bfloat16), weight))
    del weight32
    torch.npu.synchronize()

    def project(hidden, dtype):
        processor.head_dtype = dtype
        with set_ascend_forward_context(
            None,
            config,
            num_tokens=len(hidden),
            num_actual_tokens=len(hidden),
            model_instance=model,
            skip_compiled=True,
        ):
            return language_model.compute_logits(hidden)

    for name in ("official", "compact_ready"):
        source = args.snapshots / f"{name}.rank{rank}.pt"
        saved = torch.load(source, map_location="cpu", weights_only=False)
        hidden = saved["hidden"].to("npu")
        if tuple(hidden.shape) != (24, 5120) or hidden.dtype != torch.bfloat16:
            raise AssertionError("retain all original B6/Q4 hidden rows, without recasting or substitutions")
        sample = dict(
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            hidden=tensor_info(torch, hidden),
        )
        tensors = {}
        for label, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
            with projection_audit(torch, weight) as audit:
                logits = project(hidden, dtype)
            torch.npu.synchronize()
            if tuple(logits.shape) != (24, VOCABULARY) or logits.dtype != dtype:
                raise AssertionError("actual head output dtype/complete vocabulary mismatch")
            selected = logits[SITE_ROW].cpu()
            record = logits_record(torch, selected, tokenizer, (866, 660))
            record.update(
                vocabulary=logits.shape[-1],
                finite=bool(torch.isfinite(logits).all()),
                operations=audit.operations,
                fp32_operands_and_output=real_fp32_projection(audit.operations),
                all_query_argmax=logits[SITE_ROW : SITE_ROW + 4].argmax(-1).cpu().tolist(),
            )
            if label == "bfloat16":
                record["saved_logits_exact"] = bool(torch.equal(logits[SITE_ROW : SITE_ROW + 4].cpu(), saved["logits"]))
            tensors[label] = logits[SITE_ROW : SITE_ROW + 4].cpu()
            sample[label] = record
            del logits
        result["samples"][name] = sample
        torch.save(tensors, args.output / f"{name}.full-vocabulary.rank{rank}.pt")
        del tensors, saved, hidden
    if args.reuse_costs is not None:
        previous = json.loads((args.reuse_costs / f"site.rank{rank}.json").read_bytes())
        for name in result["samples"]:
            actual = torch.load(args.output / f"{name}.full-vocabulary.rank{rank}.pt", weights_only=False)
            earlier = torch.load(args.reuse_costs / f"{name}.full-vocabulary.rank{rank}.pt", weights_only=False)
            if any(not torch.equal(actual[key], earlier[key]) for key in actual):
                raise AssertionError("operand-only observation changed previously measured logits")
        result["costs"] = previous["costs"]
        result["cost_scope"] = previous["cost_scope"]
        result["costs_reused_from"] = str(args.reuse_costs)
        result["previous_full_logits_exact"] = True
        write_json(args.output / f"site.rank{rank}.json", result)
        finish_site(args, dist)
        return
    hidden = torch.load(args.snapshots / f"official.rank{rank}.pt", map_location="cpu", weights_only=False)[
        "hidden"
    ].to("npu")
    costs = {}
    for count in (1, 4, 24):
        inputs = hidden[:count]
        row = {}
        for label, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
            row[f"complete_head_{label}"] = measure(
                torch, dist, lambda dtype=dtype, inputs=inputs: project(inputs, dtype)
            )
        # These two actions decompose the same configured path. The converted
        # weight is never installed in the model or used by actual generation.
        if count == 4:
            row["weight_conversion_only"] = measure(torch, dist, lambda: weight.float())
            weight32, inputs32 = weight.float(), inputs.float()
            row["fp32_local_projection_only"] = measure(
                torch, dist, lambda inputs32=inputs32, weight32=weight32: functional.linear(inputs32, weight32)
            )
            del weight32, inputs32
        costs[str(count)] = row
    processor.head_dtype = torch.float32
    result["costs"] = costs
    result["cost_scope"] = "head-only; 2 warmups/5 samples; wall and device; no decoder timing"
    write_json(args.output / f"site.rank{rank}.json", result)
    finish_site(args, dist)


def finish_site(args, dist):
    rank = dist.get_rank()
    dist.barrier()
    reports = [json.loads((args.output / f"site.rank{r}.json").read_bytes()) for r in range(2)]
    if rank == 0:
        write_json(
            args.output / "gate.json",
            dict(
                allow_generation=generation_gate(reports),
                rule="both ranks reproduce saved BF16 logits; FP32 operands/output; full vocab unique argmax=866",
            ),
        )
    print(f"rank={rank} site complete, generation gate={generation_gate(reports)}", flush=True)


def generate(args, torch, torch_npu, dist, model, config, tokenizer, modules):
    from multi_graph_gdn_aime26_probe import execute_group

    reports = [json.loads((args.site / f"site.rank{r}.json").read_bytes()) for r in range(2)]
    if not generation_gate(reports):
        raise ValueError("complete-vocabulary scene verification did not authorize the bounded generation")
    rows = [args.manifest["payload"]["inputs"][27]]
    if rows[0]["index"] != 27:
        raise AssertionError("only Q28 is in scope")
    original = model.language_model.compute_logits
    counter = dict(
        calls=0,
        rows=0,
        input_dtype=BF16,
        output_dtype=FP32,
        weight_dtype=str(model.language_model.lm_head.weight.dtype),
        weight_shape=list(model.language_model.lm_head.weight.shape),
        configured_head_dtype=str(model.language_model.logits_processor.head_dtype),
        allow_hf32=torch.npu.matmul.allow_hf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        audited_calls=[],
        fallback_calls=0,
    )

    def checked(hidden, *positional, **keywords):
        if counter["calls"] < 2:
            with projection_audit(torch, model.language_model.lm_head.weight) as audit:
                logits = original(hidden, *positional, **keywords)
            if not real_fp32_projection(audit.operations):
                raise AssertionError("first prefill and decode must expose real FP32 matmul")
            counter["audited_calls"].append(dict(rows=len(hidden), operations=audit.operations))
        else:
            logits = original(hidden, *positional, **keywords)
        if hidden.dtype != torch.bfloat16 or logits.dtype != torch.float32:
            raise AssertionError("all generation head calls must be real configured FP32 projections")
        if model.language_model.lm_head.weight.dtype != torch.bfloat16:
            raise AssertionError("do not change stored head weights")
        counter["calls"] += 1
        counter["rows"] += len(hidden)
        return logits

    model.language_model.compute_logits = checked
    try:
        result = execute_group(args, rows, torch, torch_npu, dist, model, config, tokenizer, modules, 0, boundary=False)
        if counter["calls"] != result["steps"] + 1:
            raise AssertionError("every prefill/decode head call must be counted")
        write_json(args.output / f"{args.variant}.head-audit.rank{dist.get_rank()}.json", counter)
    finally:
        model.language_model.compute_logits = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("site", "generate"))
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/gdn-aime26-v1/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--reuse-costs", type=Path)
    parser.add_argument(
        "--snapshots", type=Path, default=Path("/workspace/gdn-aime26-q28-divergence-v1/common-official")
    )
    parser.add_argument("--variant", choices=("official", "compact"), default="official")
    args = parser.parse_args()
    args.shard, args.qualify = 0, False
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("retain TP2")
    if args.stage == "generate" and args.site is None:
        parser.error("generation requires complete-vocabulary site evidence")
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    import torch
    import torch.distributed as dist
    import torch_npu
    from multi_graph_gdn_aime26 import MAX_MODEL_LEN, load_manifest
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_full_loader import engine_options
    from transformers import AutoTokenizer
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher

    torch.set_num_threads(8)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = True
    args.manifest = load_manifest(args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for name, expected in args.manifest["payload"]["model_files"].items():
        if hashlib.sha256((Path(args.model) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"model/tokenizer config differs from the frozen evaluation: {name}")
    row = args.manifest["payload"]["inputs"][27]
    if (
        tokenizer.apply_chat_template(
            [dict(role="user", content=row["prompt"])],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        != row["prompt_tokens"]
    ):
        raise ValueError("retain the frozen thinking template and original Q28 prompt")
    options = engine_options(args.model, 2, eager=False)
    options.update(max_model_len=MAX_MODEL_LEN, hf_overrides={"head_dtype": "float32"})
    args.capture_sizes = tuple(EngineArgs(**options).create_engine_config().compilation_config.cudagraph_capture_sizes)
    options = engine_options(args.model, 2, eager=True)
    options.update(max_model_len=MAX_MODEL_LEN, hf_overrides={"head_dtype": "float32"})
    config = EngineArgs(**options).create_engine_config()
    executor = ExecutorWithExternalLauncher(config)
    try:
        runner = executor.driver_worker.model_runner
        model = runner.get_model()
        if getattr(runner, "kv_caches", None):
            raise AssertionError("retain explicit real-layer cache ownership")
        converted = 0
        for layer in model.language_model.model.layers:
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    weight.data = torch_npu.npu_format_cast(weight.data, 29)
                    converted += 1
        if converted != 304:
            raise AssertionError("retain the same 304 decoder NZ29 projections")
        if config.model_config.head_dtype != torch.float32:
            raise AssertionError("head_dtype=float32 did not take effect")
        modules = []
        if args.stage == "generate" and args.variant == "compact":
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
        with torch.inference_mode(), set_current_vllm_config(config):
            if args.stage == "site":
                site_probe(args, torch, dist, model, config, tokenizer)
            else:
                generate(args, torch, torch_npu, dist, model, config, tokenizer, modules)
        dist.barrier()
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
