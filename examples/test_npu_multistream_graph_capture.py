#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate whether torch.npu.NPUGraph captures secondary-stream work.

This is the first backend gate for the multistream macro graph plan:

capture begin on stream0
  stream0: mm0
  stream1 waits stream0
  stream1: mm1
  stream0 waits stream1
  stream0: add
capture end
replay once or many times
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch_npu  # noqa: F401


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate NPU multistream graph capture.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"],
                        default="float16")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _run_eager(
    x: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.mm(torch.mm(x, w0), w1) + bias


def _capture_graph(
    *,
    stream0: torch.npu.Stream,
    stream1: torch.npu.Stream,
    x: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    bias: torch.Tensor,
    tmp0: torch.Tensor,
    tmp1: torch.Tensor,
    out: torch.Tensor,
) -> torch.npu.NPUGraph:
    graph = torch.npu.NPUGraph()
    event0 = torch.npu.ExternalEvent()
    event1 = torch.npu.ExternalEvent()
    with torch.npu.graph(
            graph,
            stream=stream0,
            capture_error_mode="thread_local",
    ):
        with torch.npu.stream(stream0):
            torch.mm(x, w0, out=tmp0)
            event0.record(stream0)
        event0.wait(stream1)
        with torch.npu.stream(stream1):
            torch.mm(tmp0, w1, out=tmp1)
            event1.record(stream1)
        event1.wait(stream0)
        with torch.npu.stream(stream0):
            torch.add(tmp1, bias, out=out)
    return graph


def _result_payload(**kwargs: Any) -> str:
    return json.dumps(kwargs, indent=2, sort_keys=True)


def main() -> int:
    args = _make_parser().parse_args()
    if not torch.npu.is_available():
        print(_result_payload(success=False, reason="npu_not_available"))
        return 2

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = _dtype(args.dtype)
    size = int(args.size)
    if size < 1:
        raise ValueError("--size must be >= 1")

    torch.manual_seed(0)
    x = torch.randn(size, size, device=device, dtype=dtype)
    w0 = torch.randn(size, size, device=device, dtype=dtype)
    w1 = torch.randn(size, size, device=device, dtype=dtype)
    bias = torch.randn(size, size, device=device, dtype=dtype)
    tmp0 = torch.empty_like(x)
    tmp1 = torch.empty_like(x)
    out = torch.empty_like(x)

    # NPUGraph capture must begin on a non-default stream. Replay may still be
    # submitted from the default stream.
    stream0 = torch.npu.Stream(device=device)
    stream1 = torch.npu.Stream(device=device)

    # Warm up kernels and allocator before capture.
    for _ in range(max(1, int(args.warmups))):
        out.copy_(_run_eager(x, w0, w1, bias))
    torch.npu.synchronize()
    ref = _run_eager(x, w0, w1, bias)
    torch.npu.synchronize()

    try:
        graph = _capture_graph(
            stream0=stream0,
            stream1=stream1,
            x=x,
            w0=w0,
            w1=w1,
            bias=bias,
            tmp0=tmp0,
            tmp1=tmp1,
            out=out,
        )
    except Exception as exc:
        print(
            _result_payload(
                success=False,
                phase="capture",
                error_type=type(exc).__name__,
                error=str(exc),
            ))
        return 1

    try:
        graph.replay()
        torch.npu.synchronize()
        max_abs_diff = float((out.float() - ref.float()).abs().max().item())
        success = max_abs_diff <= float(args.atol)
        replay_count = max(1, int(args.replays))
        start = time.perf_counter()
        for _ in range(replay_count):
            graph.replay()
        torch.npu.synchronize()
        avg_replay_ms = (
            (time.perf_counter() - start) * 1000.0 / replay_count)
        print(
            _result_payload(
                success=success,
                phase="replay",
                device=args.device,
                size=size,
                dtype=str(dtype),
                replays=replay_count,
                max_abs_diff=max_abs_diff,
                atol=float(args.atol),
                avg_replay_ms=avg_replay_ms,
                note=(
                    "Use msprof/torch profiler to confirm stream1 kernels "
                    "appear under one graph replay."),
            ))
        return 0 if success else 1
    except Exception as exc:
        print(
            _result_payload(
                success=False,
                phase="replay",
                error_type=type(exc).__name__,
                error=str(exc),
            ))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
