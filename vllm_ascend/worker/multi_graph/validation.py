from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_ascend.worker.multi_graph.config import MultiGraphConfig


def validate_vllm_config(config: MultiGraphConfig, vc: Any, ascend: Any) -> None:
    if not config.enabled and config.kv_cache_layout == "default":
        return
    model = vc.model_config
    architectures = getattr(model, "architectures", ())
    if not architectures or any(name not in ("LlamaForCausalLM", "Qwen2ForCausalLM") for name in architectures):
        raise ValueError(
            "multi_graph_config requires LlamaForCausalLM or Qwen2ForCausalLM for the experimental V1 path"
        )
    failures: list[str] = []
    if vc.use_v2_model_runner:
        failures.append("V2 model runner")
    parallel = vc.parallel_config
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field, 1) != 1:
            failures.append(field)
    if parallel.enable_expert_parallel:
        failures.append("expert parallelism")
    if (config.enabled or not model.enforce_eager) and vc.compilation_config.cudagraph_mode.name != "FULL_DECODE_ONLY":
        failures.append("graph mode must be FULL_DECODE_ONLY")
    if vc.compilation_config.mode != 0:
        failures.append("compilation mode must be NONE (0)")
    if (model.enforce_eager and config.enabled) or str(model.dtype) not in ("torch.float16", "torch.bfloat16"):
        failures.append("requires graph execution with FP16/BF16")
    for field in ("is_moe", "is_hybrid", "is_encoder_decoder", "enable_sleep_mode", "use_mla", "enable_prompt_embeds"):
        if getattr(model, field, False):
            failures.append(field)
    for field in ("quantization", "multimodal_config"):
        if getattr(model, field, None):
            failures.append(field)
    text_config = getattr(model, "hf_text_config", None)
    if text_config is not None:
        parameters = (getattr(text_config, "rope_parameters", None), getattr(text_config, "rope_scaling", None))
        if any(
            value
            and (
                not isinstance(value, Mapping)
                or set(value) - {"rope_type", "type", "rope_theta"}
                or value.get("rope_type", value.get("type", "default")) != "default"
            )
            for value in parameters
        ):
            failures.append("unverified RoPE scaling/state")
    for field in ("speculative_config", "lora_config", "kv_transfer_config", "engram_config"):
        if getattr(vc, field, None) is not None:
            failures.append(field)
    if getattr(vc, "weight_transfer_config", None) is not None:
        failures.append("hot weight transfer")
    cache = vc.cache_config
    if cache.kv_cache_memory_bytes:
        failures.append("explicit kv_cache_memory_bytes; use automatic sizing to reserve graph memory")
    if cache.cache_dtype not in ("auto", "float16", "bfloat16") or cache.kv_sharing_fast_prefill:
        failures.append("quantized/shared KV")
    if getattr(cache, "kv_offloading_size", None):
        failures.append("KV offload")
    offload = getattr(vc, "offload_config", None)
    if offload is not None and (
        offload.uva.cpu_offload_gb or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)
    ):
        failures.append("weight offload/prefetch")
    compilation = ascend.ascend_compilation_config
    if any((compilation.enable_npugraph_ex, compilation.enable_static_kernel, compilation.enable_super_kernel)):
        failures.append("disable npugraph_ex/static/super kernels")
    if ascend.xlite_graph_config.enabled or ascend.sparse_kv_offload_config.enabled or ascend.kvpp_config.size != 1:
        failures.append("Xlite/offload/KVPP")
    if any(
        size > min(vc.scheduler_config.max_num_seqs, vc.scheduler_config.max_num_batched_tokens)
        for sizes in config.capture_sizes_per_lane
        for size in sizes
    ):
        failures.append("capture sizes exceed scheduler request/token limits")
    if failures:
        raise ValueError("unsupported multi_graph_config combination: " + "; ".join(failures))


def reserve_graph_memory(available_bytes: int, config: MultiGraphConfig) -> int:
    """Reserve graph resources exactly once before automatic KV sizing."""
    if not config.enabled:
        return available_bytes
    remaining = available_bytes - config.graph_memory_budget_bytes
    if remaining <= 0:
        raise ValueError("multi-graph memory reservation leaves no KV cache budget")
    return remaining
