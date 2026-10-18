# SPDX-License-Identifier: Apache-2.0
"""Check hybrid-state byte arithmetic against independent tensor extents."""

import copy
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


@pytest.fixture
def budget(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    name = "hybrid_memory_budget_cpu_test"
    spec = importlib.util.spec_from_file_location(name, directory / "multi_graph_hybrid_memory_budget.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def workload(budget):
    trace = dict(
        query=4,
        initial_prefix_by_request={"0": 129, "1": 256},
        steps=[dict(batch=1, request_ids=[0], seq_lengths=[260]), dict(batch=1, request_ids=[1], seq_lengths=[257])],
    )
    return dict(trace=trace, trace_sha256=budget.trace_digest(trace))


def inventory():
    return dict(
        tensor_payload_bytes=10000,
        payload_bytes_by_group=dict(text_decoder=6000, text_embedding=1000, lm_head=1000, vision=2000),
        config=dict(
            text_config=dict(
                dtype="bfloat16",
                mamba_ssm_dtype="float32",
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                num_hidden_layers=4,
                linear_num_value_heads=48,
                linear_num_key_heads=16,
                num_attention_heads=24,
                num_key_value_heads=4,
                linear_value_head_dim=128,
                linear_key_head_dim=128,
                linear_conv_kernel_dim=4,
                head_dim=256,
            )
        ),
    )


def test_live_request_pauses_do_not_free_retained_prefixes(budget):
    observed = budget.trace_residency(workload(budget)["trace"], 128)
    assert observed["logical_requests"] == 2
    assert observed["candidate_slots_including_null"] == 9
    assert observed["initial_prefix_tokens"] == 385
    assert observed["maximum_retained_tokens"] == 517
    assert observed["initial_fia_blocks"] == 4
    assert observed["maximum_fia_blocks"] == 6


def test_fp32_state_page_alignment_and_tp8_kv_replication(budget):
    report = budget.estimate_budget(inventory(), workload(budget))
    tp2, tp4, tp8 = report["estimates"]
    assert tp2["ssm_bytes_per_slot_per_layer"] == 24 * 128 * 128 * 4
    assert tp2["conv_bytes_per_slot_per_layer"] == (2 * 8 * 128 + 24 * 128) * 6 * 2
    assert tp2["estimated_aligned_gdn_page_bytes"] == 458752 * 4
    assert tp2["ssm_state_bytes"] == 9 * 3 * 24 * 128 * 128 * 4
    assert tp2["ideal_text_checkpoint_shard_bytes"] == 4000
    assert tp2["initial_fia_bytes"] == 5 * 128 * 2 * 2 * 256 * 2
    assert tp2["maximum_fia_bytes"] == 7 * 128 * 2 * 2 * 256 * 2
    assert tp8["ssm_state_bytes"] * 4 == tp2["ssm_state_bytes"]
    assert tp8["maximum_fia_bytes"] == tp4["maximum_fia_bytes"]
    assert tp8["fia_kv_heads_replicated"] and not tp4["fia_kv_heads_replicated"]
    assert all(value is None for value in report["unknown_additional_device_bytes"].values())
    assert report["status"] == "cpu_capacity_estimate_not_allocation_or_oom_evidence"


def test_digest_detects_mutated_input_before_estimating(budget):
    document = workload(budget)
    document["trace"]["steps"][0]["seq_lengths"][0] += 1
    with pytest.raises(ValueError, match="digest mismatch"):
        budget.estimate_budget(inventory(), document)


@pytest.mark.parametrize("tp", [0, True, 3, 5])
def test_invalid_shard_layouts_rejected(budget, tp):
    with pytest.raises(ValueError):
        budget.estimate_budget(inventory(), workload(budget), tensor_parallel_sizes=(tp,))


def test_actual_manifest_budget_retains_all_requests_and_reports_unknown_peak(budget):
    root = Path(__file__).resolve().parents[4]
    path = root / "benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json"
    document = json.loads(path.read_text())
    report = budget.estimate_budget(inventory(), document)
    assert report["workload_residency"]["logical_requests"] == 128
    assert report["workload_residency"]["candidate_slots_including_null"] == 513
    assert report["workload_residency"]["initial_prefix_tokens"] == 125001
    assert report["workload_residency"]["max_sequence_length"] == 1795


def test_inventory_reads_only_headers_and_checks_index_consistency(budget, tmp_path):
    name = "model.language_model.layers.0.weight"
    header = json.dumps({name: dict(dtype="BF16", shape=[2, 3], data_offsets=[0, 12])}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 12)
    (tmp_path / "config.json").write_text(json.dumps(inventory()["config"]))
    index = dict(metadata=dict(total_size=12), weight_map={name: path.name})
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))
    actual = budget.checkpoint_inventory(tmp_path)
    assert actual["bytes_read_from_safetensors"] == len(header) + 8
    assert actual["tensor_payload_bytes"] == 12
    assert actual["payload_bytes_by_group"] == dict(text_decoder=12)
    assert not actual["tensor_payloads_read"]
    changed = copy.deepcopy(index)
    changed["metadata"]["total_size"] += 1
    index_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="total_size"):
        budget.checkpoint_inventory(tmp_path)
