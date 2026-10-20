"""Scoped stream resource limits for the split-parallel benchmark.

A reported runtime limit does not establish physical core partitioning or prove
that a custom kernel chose a smaller block dimension. Device traces must answer
those questions separately.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
from typing import Any


def _npu_api(npu: Any | None) -> Any:
    return import_module("torch_npu").npu if npu is None else npu


def stream_budget_support(*, npu: Any | None = None) -> dict[str, Any]:
    """Describe Python API availability without creating a stream or NPU work."""
    try:
        api = _npu_api(npu)
    except ImportError as error:
        return {"available": False, "reason": str(error), "apis": {}}
    apis = {name: callable(getattr(api, name, None)) for name in ("get_stream_limit", "set_stream_limit")}
    return {
        "available": all(apis.values()),
        "apis": apis,
        "scope": "runtime stream limits; custom kernel launch dimensions require separate verification",
    }


def _validate_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or (value != -1 and value <= 0):
        raise ValueError(f"{name} must be -1 (unchanged) or a positive integer, got {value!r}")


def _read_limits(api: Any, stream: Any) -> dict[str, int]:
    limits = api.get_stream_limit(stream)
    keys = ("cube_core_num", "vector_core_num")
    if not isinstance(limits, Mapping) or any(key not in limits for key in keys):
        raise RuntimeError(f"Unsupported get_stream_limit result: {limits!r}")
    result = {key: limits[key] for key in keys}
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result.values()):
        raise RuntimeError(f"Stream limits must report positive integer core counts: {result!r}")
    return result


def configure_stream_budget(
    stream: Any,
    *,
    vector_num: int = -1,
    cube_num: int = -1,
    npu: Any | None = None,
) -> dict[str, Any]:
    """Configure an exclusively owned stream for its entire graph lifetime.

    Successful configuration persists until changed or the stream is destroyed.
    Use stream_budget for borrowed streams that require scoped restoration.
    A failing setter restores the previous values, including partial changes.
    """
    _validate_limit("cube_num", cube_num)
    _validate_limit("vector_num", vector_num)
    if stream is None:
        raise ValueError("stream cannot be None")
    api = _npu_api(npu)
    support = stream_budget_support(npu=api)
    if not support["available"]:
        raise RuntimeError(f"Stream budget API unavailable: {support['apis']}")
    previous = _read_limits(api, stream)
    expected = {
        "cube_core_num": previous["cube_core_num"] if cube_num == -1 else cube_num,
        "vector_core_num": previous["vector_core_num"] if vector_num == -1 else vector_num,
    }
    report: dict[str, Any] = {
        "status": "applying",
        "before": previous,
        "requested": {"cube_num": cube_num, "vector_num": vector_num},
    }
    try:
        # Even a partially failing setter can change one resource before raising.
        api.set_stream_limit(stream, cube_num=cube_num, vector_num=vector_num)
        active = _read_limits(api, stream)
        if active != expected:
            raise RuntimeError(f"Runtime did not apply requested stream budget: {active!r} != {expected!r}")
        return dict(report, status="applied", active=active)
    except BaseException:
        api.set_stream_limit(stream, cube_num=previous["cube_core_num"], vector_num=previous["vector_core_num"])
        raise


@contextmanager
def stream_budget(
    stream: Any,
    *,
    vector_num: int = -1,
    cube_num: int = -1,
    npu: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Apply and verify a stream limit, then restore its previous values.

    Keep the scope around graph capture or eager submission. The installed API's
    -1 leaves a resource unchanged, so vector-only experiments retain all Cube
    cores. This helper neither synchronizes streams nor asserts graph replay
    behavior; the calling experiment must verify capture/replay effects.
    """
    api = _npu_api(npu)
    report = configure_stream_budget(stream, vector_num=vector_num, cube_num=cube_num, npu=api)
    previous = report["before"]
    try:
        yield report
    finally:
        # reset_stream_limit would discard an enclosing scope's nondefault budget.
        api.set_stream_limit(stream, cube_num=previous["cube_core_num"], vector_num=previous["vector_core_num"])
        restored = _read_limits(api, stream)
        report["restored"] = restored
        if restored != previous:
            raise RuntimeError(f"Runtime did not restore prior stream budget: {restored!r} != {previous!r}")
        report["status"] = "restored"
