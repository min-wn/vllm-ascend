# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the retained-catalog bridge, with no native execution."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    parent = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(parent))
    spec = importlib.util.spec_from_file_location(
        "fia_catalog_bridge_probe_cpu", parent / "aclgraph_fia_sink_catalog_bridge_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalogs_keep_original_first_use_order(probe):
    schedule = probe.six.load_trace_schedule()
    original = copy.deepcopy(schedule)
    two = probe.catalog_records(schedule, 2)
    assert [(row["physical_tokens"], row["step"]) for row in two] == [(400, 0), (512, 29)]
    expected = probe.seven.capture_records(probe.seven.trace_records(schedule))
    seventeen = probe.catalog_records(schedule, 17)
    assert seventeen == expected and len(seventeen) == 17
    assert [row["physical_tokens"] for row in seventeen] != sorted(row["physical_tokens"] for row in seventeen)
    assert set(row["physical_tokens"] for row in seventeen).issubset(schedule["default_capture_sizes"])
    assert schedule == original


@pytest.mark.parametrize("count", (True, 1, 3, 16, 18, "17"))
def test_noncanonical_catalog_counts_are_rejected(probe, count):
    with pytest.raises(ValueError, match="exactly"):
        probe.catalog_records(probe.six.load_trace_schedule(), count)


def test_catalog_rejects_changed_defaults_and_missing_first_uses(probe):
    schedule = copy.deepcopy(probe.six.load_trace_schedule())
    schedule["default_capture_sizes"] = []
    with pytest.raises(ValueError, match="catalog"):
        probe.catalog_records(schedule, 17)
    schedule = copy.deepcopy(probe.six.load_trace_schedule())
    schedule["required_capture_sizes"] = tuple(schedule["required_capture_sizes"][:-1])
    with pytest.raises(ValueError, match="catalog"):
        probe.catalog_records(schedule, 17)


class HostTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return json.loads(json.dumps(self.data))


def audit_fixture(probe, monkeypatch):
    records = probe.catalog_records(probe.six.load_trace_schedule(), 2)
    frames, entries = {}, {}
    status = dict(captured_addresses={}, metadata_preflight=[])
    for record in records:
        capacity = record["physical_tokens"]
        tensors = {
            name: HostTensor(record[source])
            for name, source in (
                ("q_lengths", "actual_seq_qlen"),
                ("kv_lengths", "kv_lengths"),
                ("block_table", "block_table"),
            )
        }
        frames[capacity] = [], tensors
        entries[capacity] = SimpleNamespace(metadata=[HostTensor(dict(sha256=f"plan{capacity}"))])
        status["captured_addresses"][str(capacity)] = {"pointer": capacity}
        status["metadata_preflight"].append(dict(capacity=capacity, sha256=f"plan{capacity}"))
    monkeypatch.setattr(probe.consumer, "tracked_tensors", lambda views, tensors, entry, mode: tensors)
    monkeypatch.setattr(probe.six, "pointers", lambda tensors: {"pointer": len(tensors["kv_lengths"].data) * 4})
    monkeypatch.setattr(probe.seven, "current_plan", lambda metadata, record, mode, cores: metadata.data)
    return records, frames, entries, status


def test_catalog_audit_reads_existing_graph_state_without_replay(probe, monkeypatch):
    records, frames, entries, status = audit_fixture(probe, monkeypatch)
    for entry in entries.values():
        entry.metadata[0] = object()  # Any cpu() read of an uninitialized output fails.
    syncs, saves = [], []
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: syncs.append("sync")))
    result = probe.audit_catalog(
        torch,
        {},
        SimpleNamespace(mode="whole"),
        records,
        frames,
        entries,
        status,
        lambda: saves.append("save"),
        "catalog_pre_checks",
    )
    assert len(result) == len(saves) == 2
    assert syncs == ["sync"]
    assert all(row["replay_performed"] is False for row in result)
    assert [row["plan"]["sha256"] for row in result] == ["plan400", "plan512"]
    assert all(row["plan_source"] == probe.PREFLIGHT_SOURCE for row in result)
    assert all(row["graph_replays_after_capture"] == 0 for row in result)
    assert all(row["graph_metadata_semantically_initialized"] is False for row in result)


def test_post_catalog_audit_reads_selected_actual_plan_but_not_unused_output(probe, monkeypatch):
    records, frames, entries, status = audit_fixture(probe, monkeypatch)
    unused = copy.deepcopy(records[0])
    unused.update(physical_tokens=304, step=1)
    records.append(unused)
    frames[304] = [], dict(frames[400][1])
    entries[304] = SimpleNamespace(metadata=[object()])
    status["captured_addresses"]["304"] = {"pointer": 400}
    status["metadata_preflight"].append(dict(capacity=304, sha256="plan304"))
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    result = probe.audit_catalog(
        torch, {}, SimpleNamespace(mode="whole"), records, frames, entries, status, lambda: None, "catalog_post_checks"
    )
    assert [row["graph_replays_after_capture"] for row in result] == [17, 17, 0]
    assert [row["graph_metadata_semantically_initialized"] for row in result] == [True, True, False]
    assert [row["plan_is_captured_graph_output"] for row in result] == [True, True, False]
    assert [row["plan_source"] for row in result] == [probe.REPLAY_SOURCE, probe.REPLAY_SOURCE, probe.PREFLIGHT_SOURCE]


@pytest.mark.parametrize("mutation", ("address", "descriptor", "plan"))
def test_catalog_audit_rejects_mutated_state(probe, monkeypatch, mutation):
    records, frames, entries, status = audit_fixture(probe, monkeypatch)
    if mutation == "address":
        status["captured_addresses"]["400"] = {"pointer": 0}
    elif mutation == "descriptor":
        frames[400][1]["kv_lengths"] = HostTensor([0] * 100)
    else:
        entries[400].metadata[0].data = dict(sha256="different")
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    with pytest.raises(AssertionError):
        probe.audit_catalog(
            torch,
            {},
            SimpleNamespace(mode="whole"),
            records,
            frames,
            entries,
            status,
            lambda: None,
            "catalog_post_checks" if mutation == "plan" else "catalog_pre_checks",
        )


@pytest.mark.parametrize("catalog_count", (2, 17))
def test_experiment_retains_catalog_but_qualifies_and_times_only_two_records(probe, monkeypatch, catalog_count):
    schedule = probe.six.load_trace_schedule()
    events, status, sources = [], {}, {}
    args = SimpleNamespace(
        trial=0,
        repeats=10,
        warmups=5,
        profile_repeats=0,
        seed=1703,
        mode="whole",
        trace=None,
        selection=None,
        model="model",
        catalog_count=catalog_count,
    )
    torch = SimpleNamespace(set_num_threads=lambda count: events.append(("threads", count)))
    monkeypatch.setattr(probe.six.baseline, "installed_catalog", lambda *args: {"installed_catalog_revalidated": True})

    def capture(torch, cores, args, schedule, records, status, save):
        events.append(("capture", len(records)))
        for record in records:
            capacity = record["physical_tokens"]
            sources[capacity] = SimpleNamespace(summary=lambda cap=capacity: dict(capacity=cap))
        return [], {}, sources, {}, {}

    monkeypatch.setattr(probe.consumer, "capture", capture)
    monkeypatch.setattr(
        probe, "verify_resident_identities", lambda *args, post=False: events.append(("identity", post))
    )
    monkeypatch.setattr(probe, "audit_catalog", lambda *args: events.append((args[-1], len(args[3]))))
    monkeypatch.setattr(
        probe.consumer, "qualify", lambda *args: events.append((args[-1], [row["step"] for row in args[3]]))
    )
    monkeypatch.setattr(
        probe.consumer, "measure", lambda *args: events.append(("measure", [row["step"] for row in args[2]]))
    )
    probe.experiment(torch, dict(cube_core_num=20, vector_core_num=40), args, status, lambda: None)
    assert events == [
        ("threads", 4),
        ("capture", catalog_count),
        ("identity", False),
        ("catalog_pre_checks", catalog_count),
        ("pre_checks", [0, 29]),
        ("measure", [0, 29]),
        ("post_checks", [0, 29]),
        ("catalog_post_checks", catalog_count),
        ("identity", True),
    ]
    assert status["capture_records"] == probe.catalog_records(schedule, catalog_count)
    assert len(status["prepared_descriptors"]) == catalog_count
    assert status["profiles"] == [] and status["additional_catalog_replays_after_capture"] == 0
    assert status["trial"] == 0 and status["state"] == "completed"
    assert all(
        status[flag] is False for flag in ("full_trace_measured", "full_model_qualified", "end_to_end_gain_established")
    )


@pytest.mark.parametrize(
    "field,value", (("trial", 1), ("repeats", 11), ("warmups", 4), ("profile_repeats", 3), ("seed", 0))
)
def test_experiment_rejects_timing_protocol_drift(probe, field, value):
    args = SimpleNamespace(trial=0, repeats=10, warmups=5, profile_repeats=0, seed=1703, mode="whole")
    setattr(args, field, value)
    with pytest.raises(ValueError, match="same fixed"):
        probe.experiment(None, dict(cube_core_num=20, vector_core_num=40), args, {}, lambda: None)


def test_resident_identity_rejects_capture_mutation_before_overwriting_original(probe, monkeypatch):
    monkeypatch.setattr(probe.six, "input_identity", lambda *args: {"sha256": "mutated"})
    status = dict(input_identity={"sha256": "original"})
    with pytest.raises(AssertionError, match="original resident"):
        probe.verify_resident_identities(None, [], {}, status)
    assert status["input_identity"]["sha256"] == "original"


@pytest.mark.parametrize("catalog_count", (2, 17))
def test_cli_forwards_fixed_identity(probe, monkeypatch, tmp_path, catalog_count):
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    args = [
        "probe",
        "--root",
        str(tmp_path),
        "--opp-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--trace",
        str(trace),
        "--selection",
        str(trace),
        "--mode",
        "whole",
        "--catalog-count",
        str(catalog_count),
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(probe, "launcher", lambda args: args)
    parsed = probe.main()
    assert parsed.catalog_count == catalog_count
    assert (parsed.trial, parsed.repeats, parsed.warmups, parsed.profile_repeats) == (0, 10, 5, 0)
