from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.multi_graph.config import MultiGraphConfig
from vllm_ascend.worker.multi_graph.validation import reserve_graph_memory, validate_vllm_config


@pytest.fixture
def configs(config):
    vc = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"], dtype="torch.bfloat16", enforce_eager=False),
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"), mode=0),
        cache_config=SimpleNamespace(cache_dtype="auto", kv_sharing_fast_prefill=False, kv_cache_memory_bytes=None),
        scheduler_config=SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=512),
    )
    ascend = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=False,
            enable_static_kernel=False,
            enable_super_kernel=False,
        ),
        xlite_graph_config=SimpleNamespace(enabled=False),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        kvpp_config=SimpleNamespace(size=1),
    )
    return config, vc, ascend


def test_supported_config(configs):
    validate_vllm_config(*configs)


def test_transformers_default_rope_alias_is_unscaled(configs):
    _, vc, _ = configs
    parameters = {"rope_theta": 1000000.0, "rope_type": "default"}
    vc.model_config.hf_text_config = SimpleNamespace(rope_parameters=parameters, rope_scaling=parameters)
    validate_vllm_config(*configs)


@pytest.mark.parametrize(
    "parameters",
    [
        {"rope_type": "default", "factor": 2},
        {"rope_type": "dynamic"},
        {"full_attention": {"rope_type": "default"}},
    ],
)
def test_scaled_or_per_layer_rope_still_rejected(configs, parameters):
    _, vc, _ = configs
    vc.model_config.hf_text_config = SimpleNamespace(rope_parameters=parameters)
    with pytest.raises(ValueError, match="RoPE"):
        validate_vllm_config(*configs)


@pytest.mark.parametrize(
    "component,field,value",
    [
        ("model_config", "architectures", ["Qwen3ForCausalLM"]),
        ("model_config", "enable_sleep_mode", True),
        ("model_config", "is_hybrid", True),
        ("model_config", "quantization", "int8"),
        ("model_config", "hf_text_config", SimpleNamespace(rope_scaling={"rope_type": "dynamic"})),
        ("parallel_config", "tensor_parallel_size", 2),
        ("cache_config", "kv_cache_memory_bytes", 1024),
        ("cache_config", "cache_dtype", "int8"),
        ("compilation_config", "mode", 3),
        ("scheduler_config", "max_num_seqs", 64),
    ],
)
def test_unsupported_startup_combinations(configs, component, field, value):
    config, vc, ascend = configs
    setattr(getattr(vc, component), field, value)
    with pytest.raises(ValueError):
        validate_vllm_config(config, vc, ascend)


def test_disabled_config_never_requires_model_runtime():
    validate_vllm_config(MultiGraphConfig(), None, None)


def test_contiguous_layout_control_requires_supported_runtime(configs):
    _, vc, ascend = configs
    config = MultiGraphConfig(kv_cache_layout="contiguous")
    validate_vllm_config(config, vc, ascend)
    vc.model_config.enforce_eager = True
    vc.compilation_config.cudagraph_mode.name = "NONE"
    validate_vllm_config(config, vc, ascend)
    vc.kv_transfer_config = object()
    with pytest.raises(ValueError, match="kv_transfer_config"):
        validate_vllm_config(config, vc, ascend)


def test_graph_budget_deducted_before_kv_sizing(config):
    budget = config.graph_memory_budget_bytes
    assert reserve_graph_memory(budget + 1024, config) == 1024
    assert reserve_graph_memory(1024, replace(config, enabled=False)) == 1024
    with pytest.raises(ValueError, match="no KV cache budget"):
        reserve_graph_memory(budget, config)


def test_qwen3_tp_layout_control_does_not_enable_split_admission(configs):
    config, vc, ascend = configs
    vc.model_config.architectures = ["Qwen3ForCausalLM"]
    vc.model_config.enforce_eager = True
    vc.parallel_config.tensor_parallel_size = 2
    vc.compilation_config.cudagraph_mode.name = "NONE"
    validate_vllm_config(MultiGraphConfig(kv_cache_layout="contiguous"), vc, ascend)
    with pytest.raises(ValueError, match="requires one of"):
        validate_vllm_config(config, vc, ascend)
    vc.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="requires one of"):
        validate_vllm_config(MultiGraphConfig(kv_cache_layout="contiguous"), vc, ascend)


@pytest.mark.parametrize("field,value", [("is_hybrid", True), ("is_moe", True), ("quantization", "int8")])
def test_qwen3_layout_control_rejects_unsupported_variants(configs, field, value):
    _, vc, ascend = configs
    vc.model_config.architectures = ["Qwen3ForCausalLM"]
    vc.model_config.enforce_eager = True
    setattr(vc.model_config, field, value)
    with pytest.raises(ValueError, match=field):
        validate_vllm_config(MultiGraphConfig(kv_cache_layout="contiguous"), vc, ascend)
