# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for chronological native-policy dynamic-trace qualification."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("fia_stage7_probe_cpu", root / "aclgraph_fia_sink_stage7_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_trace_keeps_all_original_steps_and_default_buckets(probe):
    schedule = probe.six.load_trace_schedule()
    records = probe.trace_records(schedule)
    assert [record["step"] for record in records] == list(range(128))
    assert len({record["physical_tokens"] for record in records}) == 17
    assert all(record["page_count"] == schedule["owner_pages"]["page_count"] for record in records)
    for original, selected in zip(schedule["records"], records, strict=True):
        assert selected == dict(original, page_count=schedule["owner_pages"]["page_count"])
    assert all("page_count" not in record for record in schedule["records"])
    screen = probe.trace_records(schedule, screen=True)
    assert screen == records[:8]


@pytest.mark.parametrize("change", ["rotation", "truncation", "catalog", "required"])
def test_changed_chronology_or_default_catalog_is_rejected(probe, change):
    schedule = copy.deepcopy(probe.six.load_trace_schedule())
    if change == "rotation":
        schedule["records"] = schedule["records"][1:] + schedule["records"][:1]
    elif change == "truncation":
        schedule["records"] = schedule["records"][:127]
    elif change == "catalog":
        schedule["default_capture_sizes"] = []
    else:
        schedule["required_capture_sizes"] = []
    with pytest.raises(ValueError, match="chronological|catalog"):
        probe.trace_records(schedule)


def test_capture_uses_first_original_record_for_every_bucket(probe):
    records = probe.trace_records(probe.six.load_trace_schedule())
    selected = probe.capture_records(records)
    assert len(selected) == 17
    assert [r["step"] for r in selected] == sorted(r["step"] for r in selected)
    for record in selected:
        assert record is next(r for r in records if r["physical_tokens"] == record["physical_tokens"])
    assert len(probe.capture_records(records[:8])) == len({r["physical_tokens"] for r in records[:8]})


def test_pass_boundary_is_not_pooled_with_within_trace_transitions(probe):
    record = dict(step=0, physical_tokens=400)
    previous = dict(step=127, physical_tokens=512)
    assert probe.transition(record, previous) == dict(
        previous_capacity=512, capacity_transition=True, transition_class="pass_start"
    )
    assert probe.transition(dict(step=1, physical_tokens=400), record)["transition_class"] == "same_capacity"
    assert probe.transition(dict(step=1, physical_tokens=496), record)["transition_class"] == "capacity_transition"


@pytest.mark.parametrize("kv_length", [0, 1, 128, 129, 259])
def test_compact_oracle_preserves_original_page_order_duplicates_and_partial_tails(probe, kv_length):
    generator = torch.Generator().manual_seed(1703)
    key = torch.randn((5, 128, 2, 2), generator=generator).bfloat16()
    value = torch.randn((5, 128, 2, 2), generator=generator).bfloat16()
    query = torch.randn((4, 4, 2), generator=generator).bfloat16()
    # The third logical page intentionally repeats the first physical page.
    table = [3, 1, 3, 4]
    record = dict(kv_lengths=[kv_length], block_table=[table])
    expected = probe.six.baseline.validation.cpu_reference(
        torch, query, key.flatten(2), value.flatten(2), torch.tensor([table]), [4], [kv_length], True
    )
    actual, provenance = probe.compact_reference(torch, dict(key=key, value=value), query, record, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    count = (kv_length + 127) // 128
    assert provenance == dict(kv_length=kv_length, original_page_ids=table[:count], local_page_ids=list(range(count)))


def test_compact_oracle_rejects_invalid_current_page_prefix(probe):
    layer = dict(key=torch.zeros((2, 128, 2, 2)), value=torch.zeros((2, 128, 2, 2)))
    query = torch.zeros((4, 4, 2))
    for length, table in ((129, [1]), (129, [1, 2]), (1, [-1]), (-1, [])):
        with pytest.raises(ValueError, match="page prefix|nonnegative"):
            probe.compact_reference(torch, layer, query, dict(kv_lengths=[length], block_table=[table]), 0)


def test_compact_oracle_gathers_on_device_before_copying_only_selected_pages(probe):
    calls = []

    class Cache:
        shape = (8, 128, 2, 2)
        device = torch.device("cpu")

        def index_select(self, dimension, indices):
            calls.append(("gather", dimension, indices.tolist()))

            class Gathered:
                def cpu(self):
                    calls.append(("cpu", len(indices)))
                    return torch.zeros((len(indices), 128, 2, 2))

            return Gathered()

    probe.compact_reference(
        torch,
        dict(key=Cache(), value=Cache()),
        torch.zeros((4, 4, 2)),
        dict(kv_lengths=[257], block_table=[[7, 2, 7, 6]]),
        0,
    )
    assert calls == [("gather", 0, [7, 2, 7]), ("cpu", 3), ("gather", 0, [7, 2, 7]), ("cpu", 3)]


def test_current_plan_rejects_stale_exact_length_or_bucket_base(probe, monkeypatch):
    record = dict(physical_batch=100, kv_lengths=[4097, 0])
    base = dict(maxS1=4, maxS2=4097, actualQDims=100, actualKVDims=100)
    evidence = dict(words=[0] * 1024, metadata=dict(base=base))
    monkeypatch.setattr(probe.six, "plan_evidence", lambda *_: copy.deepcopy(evidence))
    assert probe.current_plan(None, record, "upstream", {})["current_exact_base_verified"]
    for field in base:
        evidence["metadata"]["base"][field] -= 1
        with pytest.raises(AssertionError, match="exact lengths"):
            probe.current_plan(None, record, "upstream", {})
        evidence["metadata"]["base"][field] += 1
    evidence["words"].pop()
    with pytest.raises(AssertionError, match="1024"):
        probe.current_plan(None, record, "upstream", {})


def test_frame_identity_covers_every_bucket_query(probe):
    frames = {8: ([dict(query=torch.ones((8, 2))) for _ in range(2)], {}), 12: ([dict(query=torch.zeros((12, 2)))], {})}
    expected = probe.frame_query_identity(torch, frames)
    assert len(expected["8"]) == 2 and len(expected["12"]) == 1
    frames[8][0][-1]["query"][-1, -1] += 1
    actual = probe.frame_query_identity(torch, frames)
    assert actual["8"][-1] != expected["8"][-1]
    assert actual["8"][0] == expected["8"][0] and actual["12"] == expected["12"]


def test_measure_has_full_chronological_warmup_before_each_unrotated_pass(probe, monkeypatch):
    calls = []

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, stream):
            calls.append(("event", stream))

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 1.0

    class Prepared:
        def prepare(self, record):
            calls.append(("prepare", record["step"]))
            return [4], [5]

        def complete(self):
            calls.append(("complete",))

    class Entry:
        stream = "stream"

        def replay(self, _q, _kv):
            calls.append(("replay",))

        def synchronize(self):
            calls.append(("join",))

    records = [dict(step=index, batch=2, physical_tokens=8 if index % 2 == 0 else 12) for index in range(8)]
    torch_mock = SimpleNamespace(npu=SimpleNamespace(Event=Event, synchronize=lambda: None))
    args = SimpleNamespace(repeats=3, trial=7, policy="whole")
    status = {}
    counter = iter(range(1000))
    monkeypatch.setattr(probe.time, "perf_counter_ns", lambda: next(counter) * 1000)
    probe.measure(
        torch_mock, args, records, {8: Prepared(), 12: Prepared()}, {8: Entry(), 12: Entry()}, status, lambda: None
    )
    assert [item for item in calls if item[0] == "prepare"] == [("prepare", step) for step in list(range(8)) * 6]
    assert len(status["warmup_passes"]) == 3
    assert all(pass_["steps"] == list(range(8)) and pass_["completed"] for pass_ in status["warmup_passes"])
    assert [(s["repeat"], s["step"]) for s in status["samples"]] == [(r, step) for r in range(3) for step in range(8)]
    assert all(s["wall_us"] == 3 and s["queue_span_us"] == 1000 for s in status["samples"])
    assert [s["transition_class"] for s in status["samples"] if s["step"] == 0] == ["pass_start"] * 3
    assert all(s["previous_capacity"] == 12 for s in status["samples"] if s["step"] == 0)


@pytest.mark.parametrize("corruption", [None, "last_output", "length", "table", "missing_output"])
def test_qualification_checks_all_outputs_and_current_descriptors_and_rotates_oracle(probe, monkeypatch, corruption):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    tensors = dict(
        q_lengths=torch.tensor([4, 8, 12]),
        kv_lengths=torch.tensor([129, 5, 0]),
        block_table=torch.tensor([[1, 2], [3, 0], [0, 0]], dtype=torch.int32),
    )
    views = [dict(query=torch.zeros((12, 4, 2))) for _ in range(16)]
    records = [
        dict(
            step=step,
            batch=2,
            physical_tokens=12,
            physical_batch=3,
            live_tokens=8,
            actual_seq_qlen=[4, 8, 12],
            kv_lengths=[129, 5, 0],
            block_table=[[1, 2], [3, 0], [0, 0]],
        )
        for step in (0, 1, 17)
    ]

    class Prepared:
        def prepare(self, record):
            tensors["q_lengths"].copy_(torch.tensor(record["actual_seq_qlen"]))
            tensors["kv_lengths"].copy_(torch.tensor(record["kv_lengths"]))
            tensors["block_table"].copy_(torch.tensor(record["block_table"]))
            return list(record["actual_seq_qlen"]), list(record["kv_lengths"])

        def complete(self):
            pass

    class Entry:
        metadata = [torch.zeros(1024, dtype=torch.int32)]
        outputs = [torch.zeros((12, 4, 2)) for _ in range(15 if corruption == "missing_output" else 16)]

        def replay(self, _q, _kv):
            if corruption == "last_output":
                self.outputs[-1][-1, -1, -1] = torch.nan
            if corruption == "length":
                tensors["kv_lengths"][-1] += 1
            if corruption == "table":
                tensors["block_table"][-1, -1] += 1

        def synchronize(self):
            pass

    entry = Entry()
    status = dict(captured_addresses={"12": probe.six.pointers(probe.tracked_tensors(views, tensors, entry))})
    monkeypatch.setattr(probe, "current_plan", lambda *_: dict(words=[0] * 1024, current_exact_base_verified=True))
    oracle_calls = []

    def reference(_torch, layer, _query, record, row):
        oracle_calls.append((record["step"], layer["occurrence"], row))
        return torch.zeros((4, 4, 2)), dict(kv_length=record["kv_lengths"][row])

    monkeypatch.setattr(probe, "compact_reference", reference)
    arguments = (
        torch,
        {},
        SimpleNamespace(occurrences=16, policy="whole"),
        records,
        [dict(occurrence=index) for index in range(16)],
        {12: (views, tensors)},
        {12: Prepared()},
        {12: entry},
        status,
        lambda: None,
        "pre_checks",
    )
    if corruption:
        with pytest.raises(AssertionError, match="nonfinite|exact length|routing|every private"):
            probe.qualify(*arguments)
        return
    checks = probe.qualify(*arguments)
    assert [check["step"] for check in checks] == [0, 1, 17]
    assert [check["oracle_occurrence"] for check in checks] == [0, 1, 1]
    assert all(len(check["output_hashes"]) == 16 and check["all_outputs_finite"] for check in checks)
    assert oracle_calls == [(step, step % 16, row) for step in (0, 1, 17) for row in (0, 1)]
