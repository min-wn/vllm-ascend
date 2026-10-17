"""CPU checks for the raw conditional graph qualification protocol."""

import ctypes
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_acl_conditional_probe.py"
    spec = importlib.util.spec_from_file_location("acl_conditional_probe_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAcl:
    """Model default-condition reset and branch execution, without ACL loading."""

    def __init__(self, *, unsupported=False, ignore_condition_copy=False, run_both=False):
        self.unsupported, self.ignore_condition_copy, self.run_both = unsupported, ignore_condition_copy, run_both
        self.memory, self.operations = {}, {}
        self.next_address, self.capture, self.condition = 1000, None, 999
        self.children, self.finalized = (201, 202), False

    @staticmethod
    def pointer(value):
        return value.value if isinstance(value, ctypes.c_void_p) else value

    @staticmethod
    def assign(output, value):
        output._obj.value = value

    def aclGetRecentErrMsg(self):
        return b"TS firmware does not support aclgraph condition operator"

    def aclInit(self, config):
        return 0

    def aclFinalize(self):
        self.finalized = True
        return 0

    def aclrtSetDevice(self, device):
        return 0

    def aclrtCreateStream(self, output):
        self.next_address += 8
        self.assign(output, self.next_address)
        return 0

    def aclrtDestroyStream(self, stream):
        return 0

    def aclrtMalloc(self, output, size, policy):
        assert size == 8
        self.next_address += 8
        self.assign(output, self.next_address)
        self.memory[self.next_address] = 0
        return 0

    def aclrtFree(self, pointer):
        return 0

    def aclrtMemcpy(self, destination, maximum, source, count, kind):
        assert maximum == count == 8
        if kind == 1:
            self.memory[self.pointer(destination)] = source._obj.value
        elif kind == 2:
            destination._obj.value = self.memory[self.pointer(source)]
        else:
            raise AssertionError("unexpected synchronous transfer kind")
        return 0

    def aclrtMemcpyAsync(self, destination, maximum, source, count, kind, stream):
        assert maximum == count == 8 and kind == 3 and self.capture is not None
        self.operations.setdefault(self.capture, []).append((self.pointer(destination), self.pointer(source)))
        return 0

    def aclmdlRICaptureBegin(self, stream, mode):
        self.capture = 100
        return 0

    def aclmdlRICaptureGetInfo(self, stream, status, model):
        self.assign(status, 1)
        self.assign(model, 100)
        return 0

    def aclmdlRICondHandleCreate(self, model, default, flag, handle):
        if self.unsupported:
            return 207000
        assert default == 0 and flag == 1
        self.assign(handle, 99)
        self.memory[self.condition] = default
        return 0

    def aclmdlRICondHandleGetCondPtr(self, handle, output):
        self.assign(output, self.condition)
        return 0

    def aclmdlRIAddCondTask(self, params, stream, flags):
        assert params.type == 0 and params.size == 2 and flags == 0
        for index, value in enumerate(self.children):
            params.models[index] = value
        return 0

    def aclmdlRICaptureToModelRIBegin(self, stream, model, mode):
        self.capture = self.pointer(model)
        return 0

    def aclmdlRICaptureEnd(self, stream, model):
        self.capture = None
        return 0

    def aclmdlRIExecuteAsync(self, model, stream):
        self.memory[self.condition] = 0
        for destination, source in self.operations[100]:
            if not self.ignore_condition_copy:
                self.memory[destination] = self.memory[source]
        selected = self.children if self.run_both else (self.children[0 if self.memory[self.condition] else 1],)
        for child in selected:
            for destination, source in self.operations[child]:
                self.memory[destination] = self.memory[source]
        return 0

    def aclrtSynchronizeStreamWithTimeout(self, stream, timeout):
        assert timeout == 5000
        return 0

    def aclmdlRIDestroy(self, model):
        return 0


def test_conditional_ctypes_struct_matches_64_bit_acl_layout(probe):
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(probe.CondTaskParams) == 24
    assert probe.CondTaskParams.handle.offset == 0
    assert probe.CondTaskParams.type.offset == 8
    assert probe.CondTaskParams.size.offset == 12
    assert probe.CondTaskParams.models.offset == 16


@pytest.mark.parametrize("condition,selected,values,result", [(0, 1, (701, 202), 202), (1, 0, (101, 702), 101)])
def test_expected_if_branch_and_unselected_canary(probe, condition, selected, values, result):
    assert probe.expected_branch_values(condition) == (selected, values, result)


@pytest.mark.parametrize("condition", [-1, 2, None])
def test_expected_if_rejects_non_boolean_value(probe, condition):
    with pytest.raises(ValueError):
        probe.expected_branch_values(condition)


def run_fake(probe, monkeypatch, tmp_path, **options):
    fake = FakeAcl(**options)
    monkeypatch.setattr(probe, "bind_acl", lambda library: fake)
    args = SimpleNamespace(device=0, library="fake", repeats=2, output=tmp_path / "report.json", sync_timeout_ms=5000)
    status = probe.worker(args)
    return status, json.loads(args.output.read_text()), fake


def test_worker_replays_both_conditions_after_default_reset(probe, monkeypatch, tmp_path):
    code, report, fake = run_fake(probe, monkeypatch, tmp_path)
    assert code == 0 and report["status"] == "qualified_raw_conditional_IF"
    assert report["captures"] == 1 and report["replays"] == 12
    assert {check["condition"] for check in report["checks"]} == {0, 1}
    assert all(check["unselected_canary_unchanged"] for check in report["checks"])
    assert report["cleanup_complete"] and fake.finalized
    assert report["no_driver_or_device_reset"]
    assert not any("Reset" in call["api"] for call in report["calls"])


def test_worker_records_unsupported_firmware_without_qualifying(probe, monkeypatch, tmp_path):
    code, report, fake = run_fake(probe, monkeypatch, tmp_path, unsupported=True)
    assert code == 0 and report["status"] == "unsupported"
    assert report["api"] == "aclmdlRICondHandleCreate" and report["error_code"] == 207000
    assert "TS firmware" in report["recent_error"]
    assert "replays" not in report and report["cleanup_complete"] and fake.finalized


@pytest.mark.parametrize("option", ["ignore_condition_copy", "run_both"])
def test_worker_rejects_constant_condition_or_both_branch_execution(probe, monkeypatch, tmp_path, option):
    code, report, fake = run_fake(probe, monkeypatch, tmp_path, **{option: True})
    assert code == 1 and report["status"] == "failed"
    assert report["error_type"] == "ValueError"
    assert report["cleanup_complete"] and fake.finalized
