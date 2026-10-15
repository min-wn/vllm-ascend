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


def test_graph_budget_deducted_before_kv_sizing(config):
    budget = config.graph_memory_budget_bytes
    assert reserve_graph_memory(budget + 1024, config) == 1024
    assert reserve_graph_memory(1024, replace(config, enabled=False)) == 1024
    with pytest.raises(ValueError, match="no KV cache budget"):
        reserve_graph_memory(budget, config)
