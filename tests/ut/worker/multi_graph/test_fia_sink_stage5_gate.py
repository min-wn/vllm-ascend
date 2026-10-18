# SPDX-License-Identifier: Apache-2.0
"""Persist failed native payloads and reject unsafe plans before attention launch."""

import array
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
BENCHMARKS = ROOT / "benchmarks"
METADATA_WORDS = 1024


class FakeMetadata:
    def __init__(self, words, *, cpu_error=None):
        self.words = list(words)
        self.dtype = "torch.int32"
        self.shape = (len(words),)
        self.cpu_error = cpu_error
        self.cpu_calls = 0

    def data_ptr(self):
        return 123456

    def stride(self):
        return (1,)

    def cpu(self):
        self.cpu_calls += 1
        if self.cpu_error is not None:
            raise self.cpu_error
        return self

    def tolist(self):
        return self.words.copy()

    def numpy(self):
        return array.array("i", self.words)


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    # The gate imports a runner for function references. A stub makes accidental
    # execution/import of its native stack fail independently of installed Torch.
    baseline = ModuleType("aclgraph_fia_sink_stage3_budget")
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    path = BENCHMARKS / "aclgraph_fia_sink_stage5_gate.py"
    spec = importlib.util.spec_from_file_location("stage5_gate_cpu", path)
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def schedule(module):
    return module.load_trace_schedule()


@pytest.fixture
def record(module, schedule):
    return module.gate_records(schedule)[0][1]


@pytest.fixture
def words(record):
    planner = importlib.import_module("aclgraph_fia_sink_stage4_whole_plan")
    template = [0] * METADATA_WORDS
    batch = record["physical_batch"]
    maximum = max(record["kv_lengths"]) or 1024
    template[576:586] = [512, min(512, maximum), 8, 0, 20, 0, 4, maximum, batch, batch]
    return planner.whole_plan(record, template)


@pytest.fixture
def args():
    return SimpleNamespace(
        trace=Path("canonical-trace.json"),
        selection=Path("canonical-selection.json"),
        model="unused-model-config",
        seed=1703,
        opapi_lib=Path("/isolated/op_api/lib/libcust_opapi.so"),
    )


@pytest.fixture
def persisted(tmp_path):
    status = dict(extension_sha256="a" * 64, opp_artifacts={"op_api/lib/libcust_opapi.so": "b" * 64})
    path, snapshots = tmp_path / "gate.json", []

    def save():
        snapshots.append(copy.deepcopy(status))
        path.write_text(json.dumps(status), encoding="ascii")

    return status, path, snapshots, save


def assert_payload_persisted(path, words, expected_status="payload_saved_before_validation"):
    diagnostic = json.loads(path.read_text())["native_plan_diagnostics"][-1]
    assert diagnostic["status"] == expected_status
    assert diagnostic["word_count"] == METADATA_WORDS
    assert diagnostic["words"] == words
    assert diagnostic["payload_sha256"] == hashlib.sha256(array.array("i", words).tobytes()).hexdigest()
    assert diagnostic["base_metadata"] == words[576:586]
    assert diagnostic["aic_slots"] == [words[index * 10 : (index + 1) * 10] for index in range(20)]
    assert diagnostic["metadata_descriptor"] == dict(pointer=123456, shape=[1024], stride=[1])
    return diagnostic


def test_invalid_plan_is_saved_in_full_before_validation(module, record, words, args, persisted, monkeypatch):
    status, path, snapshots, save = persisted
    words[579] = 1
    words[1000] = 7654321
    events = []
    validate = module.validate_whole_plan

    def provenance(selected):
        assert selected == args.opapi_lib
        diagnostic = assert_payload_persisted(path, words)
        assert diagnostic["extension_sha256"] == status["extension_sha256"]
        assert diagnostic["staged_opp_artifacts"] == status["opp_artifacts"]
        events.append("provenance")
        return {"selected_library_handle_retained": True}

    def reject(record_arg, actual, *, cores):
        assert_payload_persisted(path, words)
        assert json.loads(path.read_text())["native_plan_diagnostics"][-1]["native_provenance"] == {
            "selected_library_handle_retained": True
        }
        events.append("validation")
        return validate(record_arg, actual, cores=cores)

    monkeypatch.setattr(module.baseline, "final_native_provenance", provenance, raising=False)
    monkeypatch.setattr(module, "validate_whole_plan", reject)
    with pytest.raises(ValueError, match="no-FD"):
        module.inspect_native_plan(FakeMetadata(words), record, "invalid_fd", args, status, save)
    diagnostic = assert_payload_persisted(path, words, expected_status="failed")
    assert "ValueError" in diagnostic["error"]
    assert diagnostic["label"] == "invalid_fd" and diagnostic["record"]["step"] == record["step"]
    assert events == ["provenance", "validation"]
    assert snapshots[0]["native_plan_diagnostics"][-1]["status"] == "reading_native_payload"
    assert snapshots[1]["native_plan_diagnostics"][-1]["words"] == words
    assert snapshots[-1]["native_plan_diagnostics"][-1]["status"] == "failed"


def test_provenance_failure_keeps_payload_and_skips_validator(module, record, words, args, persisted, monkeypatch):
    status, path, snapshots, save = persisted
    validations = []

    def reject_provenance(_selected):
        assert_payload_persisted(path, words)
        raise RuntimeError("injected alternate native library")

    monkeypatch.setattr(module.baseline, "final_native_provenance", reject_provenance, raising=False)
    monkeypatch.setattr(module, "validate_whole_plan", lambda *a, **k: validations.append((a, k)))
    with pytest.raises(RuntimeError, match="alternate native library"):
        module.inspect_native_plan(FakeMetadata(words), record, "provenance_failure", args, status, save)
    diagnostic = assert_payload_persisted(path, words, expected_status="failed")
    assert "injected alternate native library" in diagnostic["error"]
    assert "native_provenance" not in diagnostic
    assert validations == []
    assert len(snapshots) == 3


def test_successful_inspection_returns_validated_words_and_keeps_prior_failures(
    module, record, words, args, persisted, monkeypatch
):
    status, path, snapshots, save = persisted
    prior = dict(label="earlier_failure", status="failed", words=[7] * METADATA_WORDS)
    status["native_plan_diagnostics"] = [copy.deepcopy(prior)]
    monkeypatch.setattr(module.baseline, "final_native_provenance", lambda _: {"checked": True}, raising=False)
    actual, check = module.inspect_native_plan(FakeMetadata(words), record, "valid", args, status, save)
    assert actual == words and check["physical_heads"] == record["physical_batch"] * 2
    assert check["used_cores"] == 20
    assert status["native_plan_diagnostics"][0] == prior
    diagnostic = assert_payload_persisted(path, words, expected_status="validated")
    assert diagnostic["native_provenance"] == {"checked": True}
    assert diagnostic["validation"] == check
    assert len(snapshots) == 4


def test_payload_read_failure_still_records_descriptor_and_error(module, record, args, persisted):
    status, path, snapshots, save = persisted
    metadata = FakeMetadata([0] * METADATA_WORDS, cpu_error=RuntimeError("injected D2H failure"))
    with pytest.raises(RuntimeError, match="D2H failure"):
        module.inspect_native_plan(metadata, record, "read_failure", args, status, save)
    diagnostic = json.loads(path.read_text())["native_plan_diagnostics"][-1]
    assert diagnostic["status"] == "failed"
    assert diagnostic["metadata_descriptor"]["shape"] == [1024]
    assert "injected D2H failure" in diagnostic["error"]
    assert "words" not in diagnostic
    assert len(snapshots) == 2


@pytest.mark.parametrize("failure", ["invalid_plan", "provenance"])
def test_preflight_rejection_never_constructs_attention_graph(
    module, schedule, words, args, persisted, monkeypatch, failure
):
    status, path, _snapshots, save = persisted
    events = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("attention graph resources must not be constructed before successful native-plan preflight")

    torch = SimpleNamespace(
        npu=SimpleNamespace(
            synchronize=lambda: events.append("synchronize"),
            graph_pool_handle=forbidden,
            Stream=forbidden,
        )
    )
    monkeypatch.setattr(module, "load_trace_schedule", lambda *_args: schedule)
    monkeypatch.setattr(module.baseline, "installed_catalog", lambda *_args: {"checked": True}, raising=False)
    monkeypatch.setattr(module.baseline, "allocate_inputs", lambda *_args: (["layer"], "base"), raising=False)
    monkeypatch.setattr(module.baseline, "frame_inputs", lambda *_args: (["view"], {"q_lengths": "q"}), raising=False)
    monkeypatch.setattr(module.baseline, "prepare", lambda *_args: events.append("prepare"), raising=False)

    class Prepared:
        def __init__(self, *_args, **_kwargs):
            events.append("prepared_init")

        def complete(self):
            events.append("prepared_complete")

    monkeypatch.setattr(module, "PreparedDescriptors", Prepared)
    if failure == "invalid_plan":
        words[579] = 1
    metadata = FakeMetadata(words)
    monkeypatch.setattr(
        module.baseline,
        "graph_helpers",
        SimpleNamespace(
            metadata_call=lambda *_args: metadata,
        ),
        raising=False,
    )
    monkeypatch.setattr(module.baseline, "SinkGraph", forbidden, raising=False)

    def provenance(_selected):
        assert_payload_persisted(path, words)
        events.append("provenance")
        if failure == "provenance":
            raise RuntimeError("injected provenance preflight rejection")
        return {"checked": True}

    monkeypatch.setattr(module.baseline, "final_native_provenance", provenance, raising=False)
    error, message = (ValueError, "no-FD") if failure == "invalid_plan" else (RuntimeError, "preflight rejection")
    with pytest.raises(error, match=message):
        module.run_gate(torch, {"cube_core_num": 20, "vector_core_num": 40}, args, status, save)
    assert status["stage"] == "native_metadata_preflight"
    assert status["npu_work_executed"] is True
    assert metadata.cpu_calls == 1
    assert_payload_persisted(path, words, expected_status="failed")
    assert events.index("prepared_complete") < events.index("provenance")


@pytest.mark.parametrize(
    "cores",
    [
        {"cube_core_num": 24, "vector_core_num": 48},
        {"cube_core_num": 20, "vector_core_num": 39},
        {},
    ],
)
def test_unqualified_core_allocation_rejected_before_native_work(module, args, cores, monkeypatch):
    monkeypatch.setattr(module, "load_trace_schedule", lambda *_args: pytest.fail("core check must be first"))
    with pytest.raises(RuntimeError, match="20/40"):
        module.run_gate(None, cores, args, {}, lambda: pytest.fail("no evidence mutation before core rejection"))


def test_gate_records_change_lengths_clear_kv_and_recover_without_mutating_canonical(module, schedule):
    before = copy.deepcopy(schedule)
    records = module.gate_records(schedule)
    assert [label for label, _ in records] == ["initial", "changed", "zero_kv", "recover"]
    first, later, empty, recovered = [record for _, record in records]
    selected = [record for record in schedule["records"] if record["physical_tokens"] == 400]
    assert first["step"] == 0 and later["step"] == selected[1]["step"]
    assert first["kv_lengths"] != later["kv_lengths"]
    assert first["request_ids"] != later["request_ids"]
    assert first["block_table"] != later["block_table"]
    assert recovered == first
    for record in (first, later, empty, recovered):
        assert record["physical_tokens"] == 400 and record["physical_batch"] == 100
        assert record["q_lengths"] == (4,) * 100
        assert record["actual_seq_qlen"] == tuple(range(4, 401, 4))
        assert record["page_count"] == schedule["owner_pages"]["page_count"]
    assert empty["batch"] == empty["live_tokens"] == 0
    assert empty["request_ids"] == ()
    assert empty["kv_lengths"] == (0,) * 100
    assert empty["block_table"] == tuple((0,) * len(row) for row in later["block_table"])
    assert first is not selected[0] and later is not selected[1] and empty is not later
    assert schedule == before
    empty["batch"] = 17
    later["page_count"] = -1
    assert schedule == before


@pytest.mark.parametrize("mutation", ["no_revisit", "missing_initial", "wrong_bucket"])
def test_gate_records_requires_initial_and_revisited_original_bucket(module, schedule, mutation):
    changed = copy.deepcopy(schedule)
    if mutation == "no_revisit":
        changed["records"] = [changed["records"][0]]
    elif mutation == "missing_initial":
        changed["records"] = changed["records"][1:]
    else:
        changed["records"] = [{**record, "physical_tokens": 392} for record in changed["records"]]
    with pytest.raises(ValueError, match="capacity400"):
        module.gate_records(changed)
