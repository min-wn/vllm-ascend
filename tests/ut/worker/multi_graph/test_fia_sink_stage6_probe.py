# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for isolated policy identity and matched producer timing."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("fia_stage6_probe_cpu", root / "aclgraph_fia_sink_stage6_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streamed_bf16_identity_hashes_every_byte(probe, monkeypatch):
    monkeypatch.setattr(probe, "HASH_CHUNK_BYTES", 7)
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3).bfloat16()
    actual = probe.tensor_identity(torch, tensor)
    expected = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert actual == dict(shape=[4, 2, 3], dtype="torch.bfloat16", sha256=expected)
    changed = tensor.clone()
    changed[-1, -1, -1] += 1
    assert probe.tensor_identity(torch, changed)["sha256"] != expected


def test_all_private_occurrence_inputs_participate_in_identity(probe):
    layers = [
        {
            name: torch.full((2, 3), index + offset, dtype=torch.bfloat16)
            for offset, name in enumerate(("query", "key", "value"))
        }
        for index in range(16)
    ]
    expected = probe.input_identity(torch, layers)
    assert len(expected["occurrences"]) == 16
    changed = [{name: tensor.clone() for name, tensor in layer.items()} for layer in layers]
    changed[-1]["value"][-1, -1] += 1
    actual = probe.input_identity(torch, changed)
    assert actual["sha256"] != expected["sha256"]
    assert actual["occurrences"][:-1] == expected["occurrences"][:-1]


def test_private_storage_ranges_reject_cross_occurrence_aliases(probe):
    layers = [dict(query=torch.zeros(3), key=torch.zeros(4), value=torch.zeros(4)) for _ in range(2)]
    assert len(probe.private_storage_ranges(layers)) == 6
    layers[1]["key"] = layers[0]["value"]
    with pytest.raises(ValueError, match="overlaps"):
        probe.private_storage_ranges(layers)


def test_stop_child_waits_for_own_process_group_and_does_not_touch_completed(probe, monkeypatch):
    calls = []

    class Process:
        pid = 123

        def __init__(self, done=False):
            self.done = done

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            self.done = True

    monkeypatch.setattr(probe.os, "killpg", lambda pid, number: calls.append(("killpg", pid, number)), raising=False)
    probe.stop_child(Process())
    assert calls == [("killpg", 123, probe.signal.SIGTERM), ("wait", 10)]
    calls.clear()
    probe.stop_child(Process(done=True))
    assert not calls


@pytest.mark.parametrize("policy", ["upstream", "whole"])
def test_vendor_policy_pins_dispatch_name_and_both_hashes(probe, tmp_path, monkeypatch, policy):
    vendor = tmp_path / "vendor"
    name, digest = probe.POLICIES[policy]
    config = vendor / "op_impl/cpu/config/cust_aicpu_kernel.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"FusedInferAttentionScoreV2SinkMetadata": {"opInfo": {"kernelSo": name}}}))
    selected = vendor / "op_impl/cpu/aicpu_kernel/impl" / name
    monkeypatch.setattr(
        probe.baseline.isolated, "sha256", lambda path: digest if path == selected else probe.OPAPI_SHA256
    )
    result = probe.vendor_policy(vendor, policy)
    assert result["library_sha256"] == digest
    assert result["opapi_sha256"] == probe.OPAPI_SHA256
    config.write_text(json.dumps({"FusedInferAttentionScoreV2SinkMetadata": {"opInfo": {"kernelSo": "wrong.so"}}}))
    with pytest.raises(ValueError, match="dispatcher"):
        probe.vendor_policy(vendor, policy)


def test_vendor_policy_rejects_stale_library_and_changed_consumer(probe, tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    name, digest = probe.POLICIES["whole"]
    config = vendor / "op_impl/cpu/config/cust_aicpu_kernel.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"FusedInferAttentionScoreV2SinkMetadata": {"opInfo": {"kernelSo": name}}}))
    monkeypatch.setattr(probe.baseline.isolated, "sha256", lambda _: "stale")
    with pytest.raises(ValueError, match="policy library hash"):
        probe.vendor_policy(vendor, "whole")
    monkeypatch.setattr(probe.baseline.isolated, "sha256", lambda path: digest if path.name == name else "different")
    with pytest.raises(ValueError, match="common qualified consumer"):
        probe.vendor_policy(vendor, "whole")


def test_four_representative_records_retain_default_trace_contract(probe):
    schedule = probe.load_trace_schedule()
    records = probe.representative_records(schedule)
    assert tuple(record["physical_tokens"] for record in records) == (256, 400, 496, 512)
    assert all(record["page_count"] == schedule["owner_pages"]["page_count"] for record in records)
    assert [record["physical_tokens"] for record in probe.representative_records(schedule, screen=True)] == [400]


def test_metadata_preflight_is_before_attention_graph_capture(probe):
    source = Path(probe.__file__).read_text()
    tree = ast.parse(source)
    capture = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "capture")
    calls = [node for node in ast.walk(capture) if isinstance(node, ast.Call)]
    evidence_line = next(
        node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "plan_evidence"
    )
    graph_line = next(
        node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "SinkGraph"
    )
    assert evidence_line < graph_line


def test_whole_metadata_requires_structural_coverage_and_upstream_retains_fd(probe, monkeypatch):
    from aclgraph_fia_sink_stage4_whole_plan import whole_plan

    record = dict(physical_tokens=8, actual_seq_qlen=[4, 8], kv_lengths=[129, 0])
    template = [0] * 1024
    template[576:586] = [512, 129, 8, 0, 1, 0, 4, 129, 2, 2]
    words = whole_plan(record, template)
    tensor = torch.tensor(words, dtype=torch.int32)
    cores = dict(cube_core_num=20, vector_core_num=40)
    monkeypatch.setattr(probe.baseline.validation, "metadata_summary", lambda *_: dict(fd_present=False))
    result = probe.plan_evidence(tensor, record, "whole", cores)
    assert result["words"] == words and result["whole_coverage"]["used_cores"] == 2
    bad = tensor.clone()
    bad[4] = 1
    with pytest.raises(ValueError, match="zero FD fields"):
        probe.plan_evidence(bad, record, "whole", cores)
    assert "whole_coverage" not in probe.plan_evidence(bad, record, "upstream", cores)
