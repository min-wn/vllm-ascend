#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate torchair tagged-event multistream compilation.

This follows the torchair pattern:
  - create tagged events in the module
  - record/wait them around stream switches
  - compile with torchair reduce-overhead backend
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch_npu  # noqa: F401

try:
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
except Exception as exc:  # pragma: no cover - environment dependent
    tng = None
    CompilerConfig = None
    _TORCHAIR_IMPORT_ERROR = exc
else:
    _TORCHAIR_IMPORT_ERROR = None


class TaggedStreamModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        if tng is None:
            raise RuntimeError("torchair is not available")
        self.tagged_event1 = tng.ops.npu_create_tagged_event(tag="66")
        self.tagged_event2 = tng.ops.npu_create_tagged_event(tag="77")

    def forward(self, in1, in2, in3, in4):
        add_result = torch.add(in1, in2)
        tng.ops.npu_tagged_event_record(self.tagged_event1)
        with tng.scope.npu_stream_switch("1"):
            tng.ops.npu_tagged_event_wait(self.tagged_event1)
            mm_result = torch.mm(in3, in4)
            tng.ops.npu_tagged_event_record(self.tagged_event2)
            tmp = torch.add(in3, in4)
            tng.ops.npu_record_tagged_stream(tmp, "1")
        mm1 = torch.mm(in3, in4)
        with tng.scope.npu_stream_switch("2"):
            tng.ops.npu_tagged_event_wait(self.tagged_event2)
            add2 = torch.add(in3, in4)
        return add_result, mm_result, mm1, add2


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate torchair tagged-event multistream compile.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"],
                        default="float16")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _payload(**kwargs: Any) -> str:
    return json.dumps(kwargs, indent=2, sort_keys=True)


def _max_diff(outputs, refs) -> float:
    diffs = [
        (out.float() - ref.float()).abs().max().item()
        for out, ref in zip(outputs, refs)
    ]
    return float(max(diffs))


def main() -> int:
    args = _make_parser().parse_args()
    if _TORCHAIR_IMPORT_ERROR is not None:
        print(
            _payload(
                success=False,
                phase="import",
                error_type=type(_TORCHAIR_IMPORT_ERROR).__name__,
                error=str(_TORCHAIR_IMPORT_ERROR),
            ))
        return 2
    if not torch.npu.is_available():
        print(_payload(success=False, phase="init", reason="npu_not_available"))
        return 2

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = _dtype(args.dtype)
    size = int(args.size)
    if size < 1:
        raise ValueError("--size must be >= 1")

    torch.manual_seed(0)
    in1 = torch.randn(size, size, dtype=dtype, device=device)
    in2 = torch.randn(size, size, dtype=dtype, device=device)
    in3 = torch.randn(size, size, dtype=dtype, device=device)
    in4 = torch.randn(size, size, dtype=dtype, device=device)
    refs = (
        in1 + in2,
        torch.mm(in3, in4),
        torch.mm(in3, in4),
        in3 + in4,
    )

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    backend = tng.get_npu_backend(compiler_config=config)
    model = TaggedStreamModel().to(device)
    model = torch.compile(
        model,
        backend=backend,
        dynamic=False,
        fullgraph=True,
    )

    try:
        compile_start = time.perf_counter()
        outputs = model(in1, in2, in3, in4)
        torch.npu.synchronize()
        compile_and_first_run_ms = (time.perf_counter() - compile_start) * 1000
        max_abs_diff = _max_diff(outputs, refs)
        for _ in range(max(0, int(args.warmups))):
            outputs = model(in1, in2, in3, in4)
        torch.npu.synchronize()
        replay_count = max(1, int(args.replays))
        replay_start = time.perf_counter()
        for _ in range(replay_count):
            outputs = model(in1, in2, in3, in4)
        torch.npu.synchronize()
        avg_replay_ms = (
            (time.perf_counter() - replay_start) * 1000.0 / replay_count)
    except Exception as exc:
        print(
            _payload(
                success=False,
                phase="compile_or_run",
                error_type=type(exc).__name__,
                error=str(exc),
            ))
        return 1

    success = max_abs_diff <= float(args.atol)
    print(
        _payload(
            success=success,
            phase="complete",
            device=args.device,
            size=size,
            dtype=str(dtype),
            replays=replay_count,
            compile_and_first_run_ms=compile_and_first_run_ms,
            avg_replay_ms=avg_replay_ms,
            max_abs_diff=max_abs_diff,
            atol=float(args.atol),
            note=(
                "Use profiler/msprof to confirm tagged stream scopes are "
                "lowered to separate NPU streams."),
        ))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
