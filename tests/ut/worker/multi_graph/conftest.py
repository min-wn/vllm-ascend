import pytest

from tests.ut.worker.multi_graph.test_manager import execution as execution
from vllm_ascend.worker.multi_graph.config import MultiGraphConfig
from vllm_ascend.worker.multi_graph.support import RuntimeProfile
from vllm_ascend.worker.multi_graph.types import CaptureSignature, GraphKey


@pytest.fixture
def signature():
    return CaptureSignature(
        buffer_generation=0,
        model_config_hash="test-model-config",
        device_identity="test-npu-0",
        dtype="bfloat16",
        attention_backend="AscendAttentionBackend",
        attention_variant="FIA",
        kv_layout="test-kv-layout",
        block_size=128,
    )


@pytest.fixture
def profile():
    return RuntimeProfile(
        runner_version="v1",
        device_family="910B",
        model_verified=True,
        parallel_sizes=(1, 1, 1, 1, 1),
        expert_parallel=False,
        configured_graph_mode="FULL_DECODE_ONLY",
        unsupported_features=frozenset(),
    )


@pytest.fixture
def config():
    return MultiGraphConfig(
        enabled=True,
        selection_policy="force",
        capture_sizes_per_lane=((1, 2, 3, 4, 32, 64, 128, 256, 384),) * 2,
        max_graph_entries=18,
        graph_memory_budget_bytes=1 << 30,
    )


@pytest.fixture
def entries(config, signature):
    return frozenset(
        GraphKey(lane_id, size, size, signature)
        for lane_id, sizes in enumerate(config.capture_sizes_per_lane)
        for size in sizes
    )
