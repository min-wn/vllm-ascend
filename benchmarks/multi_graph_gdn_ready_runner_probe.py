# SPDX-License-Identifier: Apache-2.0
"""One bounded native-runner workload inventory; no compact installation or timing."""

import argparse
import functools
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    output = args.root / f"native.rank{rank}.json"
    if output.exists():
        raise FileExistsError("one native attempt only; preserve existing result")
    report = dict(status="initializing", rank=rank, steps=[], compact_installed=False, performance_measurement=False)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        import torch
        import torch_npu  # noqa: F401
        from vllm import LLM, SamplingParams

        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        torch.npu.matmul.allow_hf32 = False
        torch.set_float32_matmul_precision("highest")
        manifest = json.loads(args.manifest.read_bytes())
        report["workload_sha256"] = manifest["payload_sha256"]
        current = {}
        originals = {}

        def patch(name, wrapper):
            old = getattr(NPUModelRunner, name)
            originals[name] = old
            setattr(NPUModelRunner, name, functools.wraps(old)(wrapper(old)))

        def execute_wrapper(old):
            def execute(self, scheduler_output, *rest, **kwargs):
                if len(report["steps"]) >= 1024:
                    raise RuntimeError("native inventory exceeded frozen 1024-step bound")
                row = dict(
                    step=len(report["steps"]),
                    scheduled=json_value(scheduler_output.num_scheduled_tokens),
                    scheduled_drafts=json_value(scheduler_output.scheduled_spec_decode_tokens),
                    finished=sorted(scheduler_output.finished_req_ids),
                    resumed=sorted(scheduler_output.scheduled_cached_reqs.resumed_req_ids),
                    preempted=sorted(scheduler_output.preempted_req_ids or []),
                )
                current["row"] = row
                report["steps"].append(row)
                try:
                    result = old(self, scheduler_output, *rest, **kwargs)
                    row["execute_returned"] = True
                    return result
                finally:
                    current.pop("row", None)
                    save()

            return execute

        def prepare_wrapper(old):
            def prepare(self, scheduler_output, counts):
                result = old(self, scheduler_output, counts)
                row = current.get("row")
                if row is not None:
                    n = self.input_batch.num_reqs
                    requests = []
                    for i, req_id in enumerate(self.input_batch.req_ids):
                        req = self.requests[req_id]
                        requests.append(
                            dict(
                                request_id=req_id,
                                row=i,
                                computed=int(self.input_batch.num_computed_tokens_cpu[i]),
                                prompt=int(self.input_batch.num_prompt_tokens[i]),
                                previous_accepted=int(self.input_batch.num_accepted_tokens_cpu[i]),
                                blocks=json_value(req.block_ids),
                            )
                        )
                    row.update(
                        requests=requests,
                        query_lengths=json_value(counts),
                        gdn_query_start_loc=json_value(self.gdn_query_start_loc.cpu[: n + 1]),
                        decode_draft_counts=json_value(self.num_decode_draft_tokens.cpu[:n]),
                        logits_rows=int(result[0].numel()),
                    )
                return result

            return prepare

        def determine_wrapper(old):
            def determine(self, *values, **kwargs):
                result = old(self, *values, **kwargs)
                row = current.get("row")
                if row is not None:
                    descriptor = result[1]
                    row["graph"] = dict(
                        mode=str(result[0]),
                        descriptor={
                            name: json_value(getattr(descriptor, name, None))
                            for name in ("num_tokens", "num_reqs", "uniform", "has_lora")
                        },
                    )
                return result

            return determine

        def metadata_wrapper(old):
            signature = inspect.signature(old)

            def metadata(self, *values, **kwargs):
                result = old(self, *values, **kwargs)
                row = current.get("row")
                if row is not None:
                    bound = signature.bind(self, *values, **kwargs)
                    row["metadata_counts"] = {
                        k: json_value(bound.arguments.get(k))
                        for k in ("num_tokens", "num_reqs", "num_tokens_padded", "num_reqs_padded", "use_spec_decode")
                    }
                    table = result[0]
                    if isinstance(table, list):
                        table = table[0]
                    for key, meta in table.items():
                        if hasattr(meta, "num_spec_decodes"):
                            summary = dict(layer=key)
                            for field in (
                                "num_prefills",
                                "num_decodes",
                                "num_spec_decodes",
                                "num_spec_decode_tokens",
                                "num_decode_tokens",
                                "num_prefill_tokens",
                                "spec_query_start_loc",
                                "non_spec_query_start_loc",
                                "spec_state_indices_tensor",
                                "non_spec_state_indices_tensor",
                                "num_accepted_tokens",
                                "spec_sequence_masks",
                            ):
                                summary[field] = json_value(getattr(meta, field, None))
                            for field, nested in (
                                ("spec_actual_seq_lengths", "spec_decode_metadata"),
                                ("non_spec_actual_seq_lengths", "non_spec_decode_metadata"),
                            ):
                                parent = getattr(meta, nested, None)
                                summary[field] = json_value(getattr(parent, "actual_seq_lengths", None))
                            row["gdn"] = summary
                            break
                return result

            return metadata

        def sample_wrapper(old):
            def sample(self, logits, spec_decode_metadata):
                if report["steps"]:
                    row = report["steps"][-1]
                    row["head"] = dict(rows=int(logits.shape[0]), dtype=str(logits.dtype), vocab=int(logits.shape[-1]))
                result = old(self, logits, spec_decode_metadata)
                if report["steps"]:
                    report["steps"][-1]["sampled_token_ids"] = json_value(result.sampled_token_ids)
                    save()
                return result

            return sample

        def cache_wrapper(old):
            def initialize(self, kv_cache_config, *rest, **kwargs):
                report["cache_config"] = dict(
                    mamba_cache_mode=self.cache_config.mamba_cache_mode,
                    num_blocks=kv_cache_config.num_blocks,
                    groups=[
                        dict(layers=group.layer_names, spec=str(group.kv_cache_spec))
                        for group in kv_cache_config.kv_cache_groups
                    ],
                )
                save()
                result = old(self, kv_cache_config, *rest, **kwargs)
                state_layouts = []
                for name, layer in self.compilation_config.static_forward_context.items():
                    if hasattr(layer, "kv_cache") and "linear_attn" in name:
                        cache = layer.kv_cache
                        state_layouts.append(
                            dict(
                                layer=name,
                                tensors=[
                                    dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype))
                                    for t in cache
                                    if isinstance(t, torch.Tensor)
                                ],
                            )
                        )
                report["state_layouts"] = state_layouts
                save()
                return result

            return initialize

        patch("execute_model", execute_wrapper)
        patch("_prepare_inputs", prepare_wrapper)
        patch("_determine_batch_execution_and_padding", determine_wrapper)
        patch("_build_attention_metadata", metadata_wrapper)
        patch("_sample", sample_wrapper)
        patch("initialize_kv_cache", cache_wrapper)
        options = dict(
            model="/models/Qwen3.5-27B-original",
            tensor_parallel_size=2,
            distributed_executor_backend="external_launcher",
            dtype="bfloat16",
            max_model_len=2048,
            max_num_seqs=256,
            max_num_batched_tokens=2048,
            speculative_config=dict(method="ngram", num_speculative_tokens=3, prompt_lookup_min=2, prompt_lookup_max=4),
            gpu_memory_utilization=0.6,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=42,
            additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
        )
        report["engine_options"] = options
        report["status"] = "constructing_native_engine"
        save()
        llm = LLM(**options)
        config = llm.llm_engine.vllm_config
        report["effective"] = dict(
            graph_mode=str(config.compilation_config.cudagraph_mode),
            capture_sizes=list(config.compilation_config.cudagraph_capture_sizes),
            mamba_cache_mode=config.cache_config.mamba_cache_mode,
            mamba_cache_dtype=config.cache_config.mamba_cache_dtype,
            mamba_ssm_cache_dtype=config.cache_config.mamba_ssm_cache_dtype,
            async_scheduling=config.scheduler_config.async_scheduling,
        )
        report["status"] = "native_generating"
        save()
        prompts = [{"prompt_token_ids": item["prompt_tokens"]} for item in manifest["payload"]["inputs"]]
        params = SamplingParams(
            temperature=0, max_tokens=256, ignore_eos=False, stop_token_ids=[248046, 248044], seed=42
        )
        results = llm.generate(prompts, params, use_tqdm=False)
        report["outputs"] = [
            dict(
                request_id=r.request_id,
                prompt_token_ids=list(r.prompt_token_ids),
                token_ids=list(r.outputs[0].token_ids),
                finish_reason=r.outputs[0].finish_reason,
                stop_reason=r.outputs[0].stop_reason,
            )
            for r in results
        ]
        report["status"] = "complete_native_workload_inventory"
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save()
    except BaseException as error:
        report.update(status="failed_native_inventory_no_retry", error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
