# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the bounded current-consumer diagnostic."""

import copy
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "fia_consumer_probe_cpu", root / "aclgraph_fia_sink_consumer_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_are_first_unchanged_default_buckets(probe):
    schedule = probe.six.load_trace_schedule()
    records = probe.selected_records(schedule)
    assert [(r["physical_tokens"], r["step"]) for r in records] == [(400, 0), (512, 29)]
    for record in records:
        assert record == dict(schedule["records"][record["step"]], page_count=schedule["owner_pages"]["page_count"])
    assert all("page_count" not in r for r in schedule["records"])


def test_records_reject_changed_default_catalog_and_first_use(probe):
    schedule = copy.deepcopy(probe.six.load_trace_schedule())
    schedule["default_capture_sizes"] = tuple(size for size in schedule["default_capture_sizes"] if size != 400)
    with pytest.raises(ValueError, match="catalog"):
        probe.selected_records(schedule)
    schedule = copy.deepcopy(probe.six.load_trace_schedule())
    schedule["records"][29]["physical_tokens"] = 496
    with pytest.raises(ValueError, match="first-use"):
        probe.selected_records(schedule)


def test_record_order_rotates_each_trial_and_repeat_without_mutation(probe):
    records = [dict(physical_tokens=400), dict(physical_tokens=512)]
    assert probe.ordered_records(records, 0, 0) == records
    assert probe.ordered_records(records, 0, 1) == records[::-1]
    assert probe.ordered_records(records, 1, 0) == records[::-1]
    assert probe.ordered_records(records, 1, 1) == records
    assert [r["physical_tokens"] for r in records] == [400, 512]


def legacy_entry(task_count=16, *, skip_last=False, fail_index=None):
    calls = []

    class Task:
        def __init__(self, index):
            self.index = index
            self.kwargs = dict(actual_seq_lengths=[4, 8], actual_seq_lengths_kv=[129, 0])
            self.operation = "actual-native-op"

        def apply(self, stream):
            calls.append(("native_apply", self.index, stream))
            if self.index == fail_index:
                raise RuntimeError("native task failed")

    tasks = tuple(Task(index) for index in range(task_count))

    def update(stream, resolved):
        calls.append(("original_update", stream, len(resolved)))
        for task in resolved[: -1 if skip_last else None]:
            assert task.operation == "actual-native-op"
            task.apply(stream)
        return "native-result"

    entry = SimpleNamespace(
        graph=SimpleNamespace(update=update),
        outputs=[object() for _ in range(task_count)],
        tasks=tasks,
        workspace_bytes=lambda: [4096],
    )
    return entry, calls


def test_legacy_audit_counts_real_delegated_native_task_applications(probe):
    entry, calls = legacy_entry()
    audit = probe.LegacyAudit(entry)
    assert audit.snapshot() == (0, 0)
    result = entry.graph.update("update-stream", entry.tasks)
    assert result == "native-result"
    assert audit.snapshot() == (1, 16)
    assert calls == [("original_update", "update-stream", 16)] + [
        ("native_apply", i, "update-stream") for i in range(16)
    ]
    assert probe.update_delta(audit, (0, 0)) == dict(legacy_update_calls_delta=1, legacy_updated_tasks_delta=16)
    evidence = audit.evidence()
    assert evidence["task_count"] == 16 and evidence["workspace_bytes"] == [4096]
    assert evidence["current_query_lengths"] == [4, 8] and evidence["current_kv_lengths"] == [129, 0]


def test_legacy_audit_rejects_registered_tasks_without_actual_apply(probe):
    entry, _ = legacy_entry(skip_last=True)
    audit = probe.LegacyAudit(entry)
    with pytest.raises(AssertionError, match="did not apply"):
        entry.graph.update("stream", entry.tasks)
    assert audit.snapshot() == (0, 15)


def test_legacy_audit_propagates_native_failure_without_claiming_update_success(probe):
    entry, _ = legacy_entry(fail_index=3)
    audit = probe.LegacyAudit(entry)
    with pytest.raises(RuntimeError, match="native task failed"):
        entry.graph.update("stream", entry.tasks)
    assert audit.snapshot() == (0, 3)


def test_legacy_audit_rejects_missing_task_and_disagreeing_true_lengths(probe):
    entry, calls = legacy_entry()
    probe.LegacyAudit(entry)
    with pytest.raises(AssertionError, match="each private"):
        entry.graph.update("stream", entry.tasks[:-1])
    entry.tasks[-1].kwargs["actual_seq_lengths_kv"] = [128, 0]
    with pytest.raises(AssertionError, match="true host lengths"):
        entry.graph.update("stream", entry.tasks)
    assert calls == []


def test_delta_never_equates_registered_task_count_to_actual_updates(probe):
    entry, _ = legacy_entry()
    audit = probe.LegacyAudit(entry)
    with pytest.raises(AssertionError, match="omitted"):
        probe.update_delta(audit, (0, 0))
    assert probe.update_delta(None, (0, 0)) == dict(legacy_update_calls_delta=0, legacy_updated_tasks_delta=0)


def test_disabled_audit_restores_exact_original_update_and_reports_no_measured_counter(probe):
    entry, calls = legacy_entry()
    original = entry.graph.update
    audit = probe.LegacyAudit(entry)
    audit.set_enabled(False)
    assert entry.graph.update is original
    entry.graph.update("timed-stream", entry.tasks)
    assert calls[0] == ("original_update", "timed-stream", 16)
    assert len(calls) == 17 and audit.snapshot() == (0, 0)
    assert probe.update_delta(audit, (0, 0)) == dict(
        legacy_update_calls_delta=None, legacy_updated_tasks_delta=None, legacy_counter_instrumented=False
    )
    audit.set_enabled(True)
    entry.graph.update("qualification-stream", entry.tasks)
    assert audit.snapshot() == (1, 16)


def fake_runtime(probe, calls):
    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, stream):
            calls.append(("event", stream))

        def synchronize(self):
            calls.append(("event_join",))

        def elapsed_time(self, _other):
            return 0.3

    class Source:
        def __init__(self, capacity):
            self.capacity, self.inflight = capacity, False

        def prepare(self, record):
            assert not self.inflight
            assert record["physical_tokens"] == self.capacity
            self.inflight = True
            calls.append(("prepare", self.capacity))
            return [4], [128]

        def complete(self):
            assert self.inflight
            calls.append(("complete", self.capacity))
            self.inflight = False

    class Entry:
        def __init__(self, capacity):
            self.capacity, self.stream = capacity, f"stream{capacity}"

        def replay(self, q, kv):
            assert q == [4] and kv == [128]
            calls.append(("replay", self.capacity))

        def synchronize(self):
            calls.append(("join", self.capacity))

    records = [dict(physical_tokens=cap, step=step, batch=cap // 4) for cap, step in ((400, 0), (512, 29))]
    sources = {cap: Source(cap) for cap in probe.CAPACITIES}
    entries = {cap: Entry(cap) for cap in probe.CAPACITIES}
    npu = SimpleNamespace(Event=Event, synchronize=lambda: calls.append(("device_join",)))
    return SimpleNamespace(npu=npu), records, sources, entries, dict.fromkeys(probe.CAPACITIES)


def test_formal_endpoint_has_real_prepare_and_final_join_with_separate_warmups(probe, monkeypatch):
    calls, saves = [], []
    torch, records, sources, entries, audits = fake_runtime(probe, calls)
    clock = iter(range(0, 100000, 1000))
    monkeypatch.setattr(probe.time, "perf_counter_ns", lambda: next(clock))
    args = SimpleNamespace(warmups=5, repeats=10, trial=1, mode="whole")
    status = {}
    probe.measure(torch, args, records, sources, entries, audits, status, lambda: saves.append(True))
    assert len(status["samples"]) == 20 and len(saves) == 10
    assert [s["capacity"] for s in status["samples"]] == [
        cap for i in range(10) for cap in ([512, 400] if i % 2 == 0 else [400, 512])
    ]
    assert status["warmups"] == [dict(capacity=cap, replays=5, outside_timing=True) for cap in probe.CAPACITIES]
    for cap in probe.CAPACITIES:
        assert calls.count(("prepare", cap)) == 15
        assert calls.count(("replay", cap)) == 15
        assert calls.count(("complete", cap)) == 15
    for index, call in enumerate(calls):
        if call[0] == "complete":
            assert calls[index - 1][0] in ("device_join", "event_join")
    assert all(s["wall_us"] == 3 and s["queue_span_us"] == 300 for s in status["samples"])
    assert not any(source.inflight for source in sources.values())


def test_profile_keeps_preparation_outside_replay_labels_without_creating_timings(probe, tmp_path, monkeypatch):
    calls = []
    torch, records, sources, entries, audits = fake_runtime(probe, calls)

    class Profiler:
        def __enter__(self):
            calls.append(("profile_enter",))
            return self

        def __exit__(self, *_args):
            calls.append(("profile_exit",))

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps(dict(traceEvents=[])))

    def record_function(label):
        calls.append(("label", label))
        return nullcontext()

    torch.profiler = SimpleNamespace(record_function=record_function)
    fake_npu = SimpleNamespace(
        profiler=SimpleNamespace(
            profile=lambda **_kwargs: Profiler(), ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu")
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    args = SimpleNamespace(output_dir=tmp_path, mode="upstream", profile_repeats=3)
    status = dict(samples=[dict(unchanged=True)])
    probe.profiles(torch, args, records, sources, entries, audits, status, lambda: None)
    assert status["samples"] == [dict(unchanged=True)]
    assert len(status["profiles"]) == 2
    for profile in status["profiles"]:
        cap = profile["capacity"]
        assert profile["outside_timing"] is True and profile["repeats"] == 3
        assert profile["labels"] == [f"consumer/upstream/cap{cap}/repeat{i}" for i in range(3)]
        path = tmp_path / profile["path"]
        assert path.is_file() and profile["sha256"] == probe.six.baseline.isolated.sha256(path)
        assert calls.count(("prepare", cap)) == 3 and calls.count(("replay", cap)) == 3
        assert profile["preparation_in_trace_outside_label"] is True
    for index, call in enumerate(calls):
        if call[0] == "label":
            assert calls[index - 1] == ("device_join",)
            assert calls[index - 2][0] == "prepare"
            assert calls[index + 1][0] == "replay"
    assert not any(source.inflight for source in sources.values())


def test_profile_missing_device_export_fails_instead_of_claiming_trace(probe, tmp_path, monkeypatch):
    calls = []
    torch, records, sources, entries, audits = fake_runtime(probe, calls)
    torch.profiler = SimpleNamespace(record_function=lambda _label: nullcontext())
    profiler = SimpleNamespace(export_chrome_trace=lambda _path: None)

    class Context:
        def __enter__(self):
            return profiler

        def __exit__(self, *_args):
            pass

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            profiler=SimpleNamespace(profile=lambda **_: Context(), ProfilerActivity=SimpleNamespace(CPU=1, NPU=2))
        ),
    )
    args = SimpleNamespace(output_dir=tmp_path, mode="whole", profile_repeats=3)
    status = {}
    with pytest.raises(RuntimeError, match="real CPU/NPU trace"):
        probe.profiles(torch, args, records, sources, entries, audits, status, lambda: None)
    assert status["profiles"] == []


def test_child_preserves_failure_report_and_never_falls_back(probe, tmp_path, monkeypatch):
    target = tmp_path / probe.REPORT_NAME
    target.write_text(json.dumps(dict(state="queued", performance_measured=False)))
    args = SimpleNamespace(output_dir=tmp_path)
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(
        probe.six.baseline.validation,
        "initialize_native",
        lambda *_: (_ for _ in ()).throw(RuntimeError("bad native consumer")),
    )
    assert probe.child(args) == 1
    report = json.loads(target.read_text())
    assert report["state"] == "failed" and report["returncode"] == 1
    assert "bad native consumer" in report["error"] and report["performance_measured"] is False
