"""CPU checks for scoped NPU stream resource limits."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def budget_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    import multi_graph_tp_stream_budget

    return multi_graph_tp_stream_budget


class FakeNpu:
    def __init__(self):
        self.limits = {"cube_core_num": 20, "vector_core_num": 40}
        self.calls = []
        self.partial_failure = False
        self.ignore_vector = False

    def get_stream_limit(self, stream):
        assert stream == "stream"
        return self.limits.copy()

    def set_stream_limit(self, stream, cube_num=-1, vector_num=-1):
        assert stream == "stream"
        self.calls.append((cube_num, vector_num))
        if cube_num != -1:
            self.limits["cube_core_num"] = cube_num
        if self.partial_failure:
            self.partial_failure = False
            raise RuntimeError("setter failed after changing Cube count")
        if vector_num != -1 and not self.ignore_vector:
            self.limits["vector_core_num"] = vector_num


def test_vector_only_keeps_cube_and_restores(budget_module):
    npu = FakeNpu()
    with budget_module.stream_budget("stream", vector_num=20, npu=npu) as report:
        assert report["active"] == {"cube_core_num": 20, "vector_core_num": 20}
        assert report["status"] == "applied"
    assert npu.calls == [(-1, 20), (20, 40)]
    assert report["status"] == "restored"
    assert report["restored"] == report["before"]


def test_owned_stream_configuration_persists(budget_module):
    npu = FakeNpu()
    report = budget_module.configure_stream_budget("stream", vector_num=20, npu=npu)
    assert report["active"] == {"cube_core_num": 20, "vector_core_num": 20}
    assert report["before"] == {"cube_core_num": 20, "vector_core_num": 40}
    assert report["status"] == "applied"
    assert npu.calls == [(-1, 20)]


def test_nested_scopes_restore_enclosing_limit(budget_module):
    npu = FakeNpu()
    with budget_module.stream_budget("stream", vector_num=30, npu=npu):
        with budget_module.stream_budget("stream", cube_num=10, vector_num=20, npu=npu):
            assert npu.limits == {"cube_core_num": 10, "vector_core_num": 20}
        assert npu.limits == {"cube_core_num": 20, "vector_core_num": 30}
    assert npu.limits == {"cube_core_num": 20, "vector_core_num": 40}


def test_body_exception_restores(budget_module):
    npu = FakeNpu()
    with (
        pytest.raises(LookupError, match="body failed"),
        budget_module.stream_budget("stream", vector_num=10, npu=npu),
    ):
        raise LookupError("body failed")
    assert npu.limits == {"cube_core_num": 20, "vector_core_num": 40}


def test_partial_setter_failure_restores(budget_module):
    npu = FakeNpu()
    npu.partial_failure = True
    with (
        pytest.raises(RuntimeError, match="setter failed"),
        budget_module.stream_budget("stream", cube_num=10, npu=npu),
    ):
        pytest.fail("body must not run")
    assert npu.limits == {"cube_core_num": 20, "vector_core_num": 40}


def test_unapplied_runtime_limit_is_not_silently_accepted(budget_module):
    npu = FakeNpu()
    npu.ignore_vector = True
    with (
        pytest.raises(RuntimeError, match="did not apply"),
        budget_module.stream_budget("stream", vector_num=10, npu=npu),
    ):
        pytest.fail("body must not run")


@pytest.mark.parametrize("value", [0, -2, 1.5, "20", True, None])
@pytest.mark.parametrize("resource", ["cube_num", "vector_num"])
def test_invalid_resource_limit_rejected_before_api(budget_module, resource, value):
    npu = FakeNpu()
    with (
        pytest.raises(ValueError, match="positive integer"),
        budget_module.stream_budget("stream", npu=npu, **{resource: value}),
    ):
        pytest.fail("body must not run")
    assert not npu.calls


def test_missing_api_report_and_failure(budget_module):
    npu = SimpleNamespace(get_stream_limit=lambda stream: {})
    assert not budget_module.stream_budget_support(npu=npu)["available"]
    with pytest.raises(RuntimeError, match="API unavailable"), budget_module.stream_budget("stream", npu=npu):
        pytest.fail("body must not run")


@pytest.mark.parametrize("limits", [{}, {"cube_core_num": 20}, (20, 40), {"cube_core_num": -1, "vector_core_num": 40}])
def test_unknown_runtime_result_rejected(budget_module, limits):
    npu = FakeNpu()
    npu.get_stream_limit = lambda stream: limits
    with (
        pytest.raises(RuntimeError, match="Unsupported|positive integer"),
        budget_module.stream_budget("stream", npu=npu),
    ):
        pytest.fail("body must not run")
    assert not npu.calls


def test_import_failure_is_reported_without_npu_import_at_module_load(budget_module, monkeypatch):
    def missing(name):
        assert name == "torch_npu"
        raise ModuleNotFoundError("torch_npu unavailable")

    monkeypatch.setattr(budget_module, "import_module", missing)
    assert budget_module.stream_budget_support() == {
        "available": False,
        "reason": "torch_npu unavailable",
        "apis": {},
    }
