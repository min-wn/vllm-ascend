import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm_ascend.worker.multi_graph.config import MultiGraphConfig


def enabled_config(**changes):
    values = {
        "enabled": True,
        "capture_sizes_per_lane": [[32, 64, 128, 256, 384], [32, 64, 128, 256, 384]],
        "graph_memory_budget_bytes": 1 << 30,
    }
    values.update(changes)
    return MultiGraphConfig(**values)


def test_disabled_default_needs_no_capture_resources():
    config = MultiGraphConfig()
    assert not config.enabled
    assert config.capture_sizes_per_lane == ((), ())
    assert config.graph_memory_budget_bytes == 0


def test_json_roundtrip_and_deeply_immutable_sizes():
    source = [[32, 64], [32, 64]]
    config = enabled_config(capture_sizes_per_lane=source)
    source[0].append(128)
    assert config.capture_sizes_per_lane == ((32, 64), (32, 64))
    payload = json.dumps(dataclasses.asdict(config))
    assert TypeAdapter(MultiGraphConfig).validate_json(payload) == config
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.enabled = False


@pytest.mark.parametrize(
    "changes",
    [
        {"capture_sizes_per_lane": [[], [32]]},
        {"capture_sizes_per_lane": [[32], []]},
        {"capture_sizes_per_lane": [[32]]},
        {"capture_sizes_per_lane": [[32], [32], [32]]},
        {"capture_sizes_per_lane": [[32, 32], [32]]},
        {"capture_sizes_per_lane": [[64, 32], [32]]},
        {"capture_sizes_per_lane": [[0], [32]]},
        {"capture_sizes_per_lane": [[-1], [32]]},
        {"capture_sizes_per_lane": [[True], [32]]},
        {"capture_sizes_per_lane": [[32.0], [32]]},
        {"capture_sizes_per_lane": [["32"], [32]]},
        {"graph_memory_budget_bytes": 0},
        {"graph_memory_budget_bytes": -1},
        {"graph_memory_budget_bytes": True},
        {"max_graph_entries": 9},
        {"max_graph_entries": 0},
        {"execution": "attention_parallel"},
        {"num_lanes": 3},
        {"padding_policy": "bounded"},
        {"capture_policy": "lazy"},
        {"selection_policy": "heuristic"},
        {"unknown_flag": True},
        {"minimum_cost_samples": 0},
        {"minimum_gain_fraction": -0.01},
        {"minimum_gain_fraction": 1},
        {"minimum_gain_fraction": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValidationError):
        enabled_config(**changes)


def test_entry_limit_counts_equal_size_graphs_in_both_lanes():
    enabled_config(capture_sizes_per_lane=[[32], [32]], max_graph_entries=2)
    with pytest.raises(ValidationError, match="across both lanes"):
        enabled_config(capture_sizes_per_lane=[[32], [32]], max_graph_entries=1)


def test_capture_hash_is_stable_and_covers_all_settings():
    baseline = enabled_config()
    assert baseline.compute_hash() == enabled_config().compute_hash()
    changes = {
        "enabled": False,
        "execution": "serial",
        "capture_sizes_per_lane": [[32], [64]],
        "selection_policy": "force",
        "max_graph_entries": 11,
        "graph_memory_budget_bytes": 2 << 30,
    }
    for field, value in changes.items():
        assert enabled_config(**{field: value}).compute_hash() != baseline.compute_hash()


def test_imports_do_not_load_device_or_vllm_runtime():
    script = (
        "import sys, types; "
        "sys.modules['vllm_ascend.logger'] = types.ModuleType('vllm_ascend.logger'); "
        "from vllm_ascend.worker.multi_graph.planner import ExactDecodePlanner; "
        "assert not any(name in sys.modules for name in ('torch', 'torch_npu', 'vllm.config'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[4])


def test_routing_settings_do_not_change_capture_identity():
    baseline = enabled_config(selection_policy="force")
    measured = dataclasses.replace(
        baseline, selection_policy="measured", execution="serial", cost_table_path="costs.json", minimum_cost_samples=10
    )
    assert baseline.capture_hash() == measured.capture_hash()
    assert baseline.compute_hash() != measured.compute_hash()
