#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate npugraph_ex multistream torch.compile lowering."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch_npu  # noqa: F401


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate npugraph_ex multistream compile.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"],
                        default="float16")
    parser.add_argument("--parallel-pieces", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--clone-output", action="store_true")
    parser.add_argument("--deadlock-check", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _record_stream_tree(value: Any, stream: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "npu":
            value.record_stream(stream)
        return
    if isinstance(value, dict):
        for child in value.values():
            _record_stream_tree(child, stream)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _record_stream_tree(child, stream)


class NpuGraphExMultiStreamModel(torch.nn.Module):

    def __init__(self, parallel_pieces: int) -> None:
        super().__init__()
        self.parallel_pieces = int(parallel_pieces)

    def forward(self, x, w_main, w_side, bias):
        main = x + bias
        side = x - bias
        for _ in range(self.parallel_pieces):
            secondary = torch.npu.Stream()
            fork = torch.npu.Event()
            join = torch.npu.Event()
            fork.record()

            with torch.npu.stream(secondary):
                fork.wait(secondary)
                _record_stream_tree((side, w_side, bias), secondary)
                side = torch.mm(side, w_side)
                side = side + bias
                _record_stream_tree(side, secondary)
                join.record()

            main = torch.mm(main, w_main)
            main = main + bias
            join.wait(torch.npu.current_stream())
        return main, side, main + side


def _payload(**kwargs: Any) -> str:
    return json.dumps(kwargs, indent=2, sort_keys=True)


def _max_diff(outputs, refs) -> float:
    return float(
        max((out.float() - ref.float()).abs().max().item()
            for out, ref in zip(outputs, refs)))


def main() -> int:
    args = _make_parser().parse_args()
    try:
        backends = list(torch._dynamo.list_backends())
    except Exception as exc:
        print(
            _payload(
                success=False,
                phase="backend_check",
                error_type=type(exc).__name__,
                error=str(exc),
            ))
        return 2
    if "npugraph_ex" not in backends:
        print(
            _payload(
                success=False,
                phase="backend_check",
                available_backends=backends,
                reason="npugraph_ex_not_registered",
            ))
        return 2
    if not torch.npu.is_available():
        print(_payload(success=False, phase="init", reason="npu_not_available"))
        return 2

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = _dtype(args.dtype)
    size = int(args.size)
    parallel_pieces = int(args.parallel_pieces)
    if size < 1:
        raise ValueError("--size must be >= 1")
    if parallel_pieces < 2:
        raise ValueError("--parallel-pieces must be >= 2")

    torch.manual_seed(0)
    x = torch.randn(size, size, dtype=dtype, device=device)
    w_main = torch.randn(size, size, dtype=dtype, device=device)
    w_side = torch.randn(size, size, dtype=dtype, device=device)
    bias = torch.randn(size, size, dtype=dtype, device=device)

    eager_model = NpuGraphExMultiStreamModel(parallel_pieces).to(device)
    refs = eager_model(x, w_main, w_side, bias)
    torch.npu.synchronize()

    backend_options: dict[str, Any] = {}
    if args.clone_output:
        backend_options["clone_output"] = True
    if args.deadlock_check:
        backend_options["deadlock_check"] = True

    compiled_model = torch.compile(
        NpuGraphExMultiStreamModel(parallel_pieces).to(device),
        backend="npugraph_ex",
        dynamic=False,
        fullgraph=bool(args.fullgraph),
        options=backend_options,
    )

    try:
        first_start = time.perf_counter()
        outputs = compiled_model(x, w_main, w_side, bias)
        torch.npu.synchronize()
        first_call_ms = (time.perf_counter() - first_start) * 1000.0
        max_abs_diff = _max_diff(outputs, refs)

        for _ in range(max(0, int(args.warmups))):
            outputs = compiled_model(x, w_main, w_side, bias)
        torch.npu.synchronize()

        replay_count = max(1, int(args.replays))
        replay_start = time.perf_counter()
        for _ in range(replay_count):
            outputs = compiled_model(x, w_main, w_side, bias)
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
                available_backends=backends,
                backend_options=backend_options,
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
            fullgraph=bool(args.fullgraph),
            backend_options=backend_options,
            parallel_pieces=parallel_pieces,
            replays=replay_count,
            first_call_ms=first_call_ms,
            avg_replay_ms=avg_replay_ms,
            max_abs_diff=max_abs_diff,
            atol=float(args.atol),
        ))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
