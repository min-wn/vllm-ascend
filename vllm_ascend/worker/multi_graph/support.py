from dataclasses import dataclass
from enum import StrEnum

from vllm_ascend.worker.multi_graph.types import CaptureSignature


class UnsupportedFeature(StrEnum):
    SPECULATIVE = "speculative decoding"
    LORA = "LoRA"
    HYBRID = "hybrid/Mamba state"
    MOE = "MoE"
    QUANTIZATION = "quantized weights or KV cache"
    ENCODER_DECODER = "encoder/decoder or multimodal inputs"
    KV_TRANSFER = "KV transfer"
    KV_OFFLOAD = "KV offload"
    KV_SHARING = "KV sharing or cross-layer reuse"
    CASCADE = "cascade attention"
    GLOBAL_ROPE = "global mutable RoPE state"
    ENGRAM = "Engram"
    ENPU = "ENPU"
    BREAKABLE = "breakable graph"
    XLITE = "Xlite"
    SUPER_KERNEL = "super kernel"
    DEVICE_METADATA = "active device metadata executor"
    SLEEP = "sleep/wake"
    WEIGHT_RELOAD = "hot weight reload"
    COMPILED_STATE = "unverified compiled mutable state"


@dataclass(frozen=True)
class RuntimeProfile:
    """Facts supplied by a runner adapter, scoped to a verified model/device."""

    runner_version: str
    device_family: str
    model_verified: bool
    parallel_sizes: tuple[int, ...]
    expert_parallel: bool
    configured_graph_mode: str
    unsupported_features: frozenset[UnsupportedFeature]


def rejection_reasons(profile: RuntimeProfile, signature: CaptureSignature) -> tuple[str, ...]:
    """Return all known admission failures without importing vLLM or torch."""
    reasons: list[str] = []
    if profile.runner_version != "v1":
        reasons.append("requires V1 model runner")
    if profile.device_family not in ("910B", "910C"):
        reasons.append("requires verified 910B/910C hardware")
    if not profile.model_verified:
        reasons.append("model/device capture ownership has not been verified")
    if len(profile.parallel_sizes) != 5 or any(size != 1 for size in profile.parallel_sizes):
        reasons.append("requires TP=PP=DP=PCP=DCP=1")
    if profile.expert_parallel:
        reasons.append("expert parallelism is unsupported")
    if profile.configured_graph_mode != "FULL_DECODE_ONLY":
        reasons.append("requires FULL_DECODE_ONLY graph mode")
    if signature.dtype not in ("float16", "bfloat16"):
        reasons.append("requires FP16/BF16")
    if signature.attention_backend != "AscendAttentionBackend" or signature.attention_variant != "FIA":
        reasons.append("requires standard AscendAttentionBackend FIA")
    if signature.graph_mode != "FULL" or signature.attention_state != "DecodeOnly" or signature.query_len != 1:
        reasons.append("requires FULL uniform decode capture with query_len=1")
    reasons.extend(feature.value for feature in sorted(profile.unsupported_features, key=lambda feature: feature.value))
    return tuple(reasons)
