#!/usr/bin/env python3
# isort: skip_file
"""Offline correctness check for Ascend split-batch.

目标:
- 验证开启 split-batch 不会改变生成结果（token_ids 与 text 严格一致）。

运行方式（推荐：主控模式，子进程分别跑 disabled/enabled，再做严格对比）:
  python vllm-ascend/examples/test_split_batch_correctness_npu.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 64 --batch-size 8

输出:
- 会在 --output-dir 下创建时间戳子目录，保存:
  - prompts.json
  - outputs_split_disabled.json
  - outputs_split_enabled.json
  - diff.json (仅失败时)
  - metadata.json
  - summary.json
  - console.log （抓取 stdout/stderr）

备注:
- 为确保 split-batch 实际生效，batch_size 应 >= min_batch_size_for_split（默认 4）。
- 默认强制确定性解码（temperature=0），并使用 seed 生成固定 prompts。
"""

import gc
import logging
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import torch
import torch_npu

# Match existing offline example behavior.
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Ensure a supported All2All backend is selected before vLLM config validation.
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "3")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
import vllm.logger as vllm_logger_module  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


CONFIGS: dict[str, dict[str, Any]] = {
    "split_disabled": {
        "enabled": False,
        "description": "Split-batch disabled",
    },
    "split_enabled": {
        "enabled": True,
        "description": "Split-batch enabled",
    },
}


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise ValueError("all values must be positive integers")
    return values


def _parse_start_graph_caps(raw: str | None) -> dict[int, int] | None:
    if raw is None or not raw.strip():
        return None
    result: dict[int, int] = {}
    for item in raw.strip().strip("'\"").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "expected start:max_graph_tokens entries, got "
                f"{item!r}")
        start_raw, cap_raw = item.split(":", 1)
        start = int(start_raw.strip())
        cap = int(cap_raw.strip())
        if start < 0 or cap <= 0:
            raise ValueError(
                "start:max_graph_tokens entries must use non-negative starts "
                "and positive graph token caps")
        result[start] = cap
    return result or None


def _parse_start_graph_allowed_sizes(raw: str | None
                                     ) -> dict[int, list[int]] | None:
    if raw is None or not raw.strip():
        return None
    result: dict[int, list[int]] = {}
    for item in raw.strip().strip("'\"").split(";"):
        item = item.strip().strip("'\"")
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "expected start:size|size entries, got "
                f"{item!r}")
        start_raw, sizes_raw = item.split(":", 1)
        start = int(start_raw.strip().strip("'\""))
        sizes = [
            int(size.strip())
            for size in sizes_raw.strip().strip("'\"").replace(
                ",", "|").split("|")
            if size.strip()
        ]
        if start < 0 or not sizes or any(size <= 0 for size in sizes):
            raise ValueError(
                "start:size|size entries must use non-negative starts and "
                "positive graph token sizes")
        result[start] = sorted(set(sizes))
    return result or None


def _parse_json_object(raw: str | None, *, arg_name: str) -> dict[str, Any] | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return parsed


def _apply_capture_sizes(args: dict[str, Any], capture_sizes: list[int]) -> None:
    compilation_config = args.get("compilation_config") or {}
    if not isinstance(compilation_config, dict):
        try:
            compilation_config = json.loads(str(compilation_config))
        except Exception as exc:
            raise ValueError(
                "--capture-sizes requires compilation_config to be a dict or "
                "JSON object string") from exc
    compilation_config = dict(compilation_config)
    compilation_config["cudagraph_capture_sizes"] = sorted(
        {int(size) for size in capture_sizes})
    args["compilation_config"] = compilation_config


def _apply_cudagraph_mode(args: dict[str, Any], cudagraph_mode: str) -> None:
    compilation_config = args.get("compilation_config") or {}
    if not isinstance(compilation_config, dict):
        try:
            compilation_config = json.loads(str(compilation_config))
        except Exception as exc:
            raise ValueError(
                "--cudagraph-mode requires compilation_config to be a dict "
                "or JSON object string") from exc
    compilation_config = dict(compilation_config)
    compilation_config["cudagraph_mode"] = str(cudagraph_mode)
    args["compilation_config"] = compilation_config


def _ensure_fixed_batch_graph_capacity(
    args: dict[str, Any],
    *,
    batch_size: int,
    capture_sizes: list[int] | None,
) -> None:
    """Keep padded graph cases from being filtered by max_num_seqs.

    vLLM derives valid decode graph sizes from max_num_seqs. For fixed-batch
    split probes such as 416 -> 512 padding, max_num_seqs must cover the padded
    graph size, not only the real request count.
    """
    required = int(batch_size)
    if capture_sizes:
        required = max(required, max(int(size) for size in capture_sizes))

    max_num_seqs = args.get("max_num_seqs")
    if max_num_seqs is None:
        args["max_num_seqs"] = required
    elif int(max_num_seqs) < required:
        print(
            "ERROR: --fixed-batch-size with --capture-sizes requires "
            f"--max-num-seqs >= {required}, got {max_num_seqs}",
            file=sys.stderr,
        )
        raise ValueError("fixed batch graph capacity is too small")


def _expected_inplace_split(batch_size: int,
                            capture_sizes: list[int]) -> dict[str, int] | None:
    sizes = sorted({int(size) for size in capture_sizes if int(size) > 0})
    if not sizes or batch_size <= 0 or batch_size > sizes[-1]:
        return None
    if batch_size in sizes:
        return None
    lower = [size for size in sizes if size < batch_size]
    if not lower:
        return None
    first = max(lower)
    second = batch_size - first
    if second <= 0:
        return None
    return {
        "total_tokens": int(batch_size),
        "first_tokens": int(first),
        "second_tokens": int(second),
        "first_start_num_tokens": 0,
        "second_start_num_tokens": int(first),
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _tensor_ptr(payload: dict[str, Any], *path: str) -> int | None:
    cur = _tensor_payload(payload, *path)
    if cur is None:
        return None
    ptr = cur.get("data_ptr", cur.get("ptr"))
    return int(ptr) if ptr is not None else None


def _tensor_payload(payload: dict[str, Any],
                    *path: str) -> dict[str, Any] | None:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if not isinstance(cur, dict):
        return None
    return cur


def _split_from_slices(row: dict[str, Any]) -> dict[str, int] | None:
    splits = row.get("splits")
    if not isinstance(splits, list) or len(splits) < 2:
        return None
    return {
        "total_tokens":
        int(sum(int(s.get("num_tokens", 0) or 0) for s in splits)),
        "first_tokens":
        int(splits[0].get("num_tokens", 0) or 0),
        "second_tokens":
        int(splits[1].get("num_tokens", 0) or 0),
        "first_start_num_tokens":
        int(splits[0].get("start_num_tokens", 0) or 0),
        "second_start_num_tokens":
        int(splits[1].get("start_num_tokens", 0) or 0),
    }


def _split_histogram_key(split: dict[str, int]) -> str:
    return (
        f"{split['first_tokens']}+{split['second_tokens']}"
        f"@{split['second_start_num_tokens']}"
    )


def _descriptor_matches_expected(row: dict[str, Any],
                                 expected_split: dict[str, int],
                                 expected_graph_variant: str) -> bool:
    descriptor = row.get("batch_descriptor") or {}
    return (
        int(descriptor.get("num_tokens", row.get("actual_num_tokens", 0)) or 0)
        == int(expected_split["second_tokens"])
        and int(descriptor.get("start_num_tokens", 0) or 0)
        == int(expected_split["second_start_num_tokens"])
        and str(descriptor.get("graph_variant", expected_graph_variant))
        == expected_graph_variant
    )


def _execution_matches_expected(row: dict[str, Any],
                                expected_split: dict[str, int]) -> bool:
    return (
        int(row.get("num_tokens", 0) or 0) == int(expected_split["second_tokens"])
        and int(row.get("start_num_tokens", 0) or 0)
        == int(expected_split["second_start_num_tokens"])
    )


def _summarize_split_debug_trace(
    path: str,
    *,
    expected_split: dict[str, int] | None,
    split_mode: str = "inplace_serial",
    expected_no_split_reason: str | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    event_counts: dict[str, int] = {}
    for row in rows:
        event = str(row.get("event"))
        event_counts[event] = event_counts.get(event, 0) + 1

    planner_decisions = [
        row for row in rows if row.get("event") == "split_planner_decision"
    ]
    fallback_reason_histogram: dict[str, int] = {}
    for row in planner_decisions:
        reason = row.get("reason")
        fallback_to = row.get("fallback_to")
        if reason is None:
            continue
        if fallback_to is not None or str(reason).startswith("no_split_"):
            reason_key = str(reason)
            fallback_reason_histogram[reason_key] = (
                fallback_reason_histogram.get(reason_key, 0) + 1)
    split_slices = [row for row in rows if row.get("event") == "split_slices"]
    descriptors = [row for row in rows if row.get("event") == "split_descriptor"]
    inplace_exec_events = {"inplace_serial_execution"}
    expected_graph_variant = "inplace_serial"
    if split_mode == "inplace_parallel":
        inplace_exec_events.add("inplace_parallel_execution")
        expected_graph_variant = "inplace_parallel"
    inplace_exec = [
        row for row in rows if row.get("event") in inplace_exec_events
    ]
    captures = [
        row for row in rows
        if row.get("event") == "acl_graph_capture"
        and row.get("phase") == "post"
        and int((row.get("batch_descriptor") or {}).get(
            "start_num_tokens", 0) or 0) > 0
    ]
    acl_replays = [
        row for row in rows
        if row.get("event") == "acl_graph_replay"
        and row.get("phase") == "post"
        and int((row.get("batch_descriptor") or {}).get(
            "start_num_tokens", 0) or 0) > 0
    ]
    lazy_capture_complete = [
        row for row in rows
        if row.get("event") == "inplace_lazy_capture_complete"
        and int((row.get("batch_descriptor") or {}).get(
            "start_num_tokens", 0) or 0) > 0
    ]

    observed_splits: list[dict[str, Any]] = []
    observed_split_histogram: dict[str, int] = {}
    for row in split_slices:
        split = _split_from_slices(row)
        if split is None:
            continue
        record = {
            "step_id": row.get("step_id"),
            **split,
        }
        observed_splits.append(record)
        key = _split_histogram_key(split)
        observed_split_histogram[key] = observed_split_histogram.get(key, 0) + 1

    expected_observations: list[dict[str, Any]] = []
    expected_descriptors: list[dict[str, Any]] = []
    expected_exec: list[dict[str, Any]] = []
    expected_captures: list[dict[str, Any]] = []
    expected_lazy_captures: list[dict[str, Any]] = []
    if expected_split is not None:
        expected_observations = [
            row for row in observed_splits
            if {
                "total_tokens": row["total_tokens"],
                "first_tokens": row["first_tokens"],
                "second_tokens": row["second_tokens"],
                "first_start_num_tokens": row["first_start_num_tokens"],
                "second_start_num_tokens": row["second_start_num_tokens"],
            } == expected_split
        ]
        expected_descriptors = [
            row for row in descriptors
            if _descriptor_matches_expected(row, expected_split,
                                            expected_graph_variant)
        ]
        expected_exec = [
            row for row in inplace_exec
            if _execution_matches_expected(row, expected_split)
        ]
        expected_captures = [
            row for row in captures
            if _descriptor_matches_expected(row, expected_split,
                                            expected_graph_variant)
        ]
        expected_lazy_captures = [
            row for row in lazy_capture_complete
            if _descriptor_matches_expected(row, expected_split,
                                            expected_graph_variant)
        ]

    second_exec = expected_exec if expected_split is not None else [
        row for row in inplace_exec
        if int(row.get("start_num_tokens", 0) or 0) > 0
    ]
    ptr_fields = {
        "input_ids": ("input_ids", ),
        "positions": ("positions", ),
        "query_start_loc": ("metadata", "query_start_loc"),
        "seq_lens": ("metadata", "seq_lens"),
        "block_tables": ("metadata", "block_tables"),
        "slot_mapping": ("metadata", "slot_mapping"),
    }
    ptr_stability: dict[str, Any] = {}
    for name, keys in ptr_fields.items():
        tensor_payloads = [
            info for info in (_tensor_payload(row, *keys)
                              for row in second_exec) if info is not None
        ]
        values = []
        shapes: list[list[int]] = []
        ndims: list[int] = []
        storage_offsets: list[int] = []
        for info in tensor_payloads:
            ptr = info.get("data_ptr", info.get("ptr"))
            if ptr is not None:
                values.append(int(ptr))
            shape = info.get("shape")
            if isinstance(shape, list):
                shapes.append([int(v) for v in shape])
                ndims.append(int(info.get("ndim", len(shape))))
            elif "ndim" in info:
                ndims.append(int(info["ndim"]))
            if info.get("storage_offset") is not None:
                storage_offsets.append(int(info["storage_offset"]))
        unique_shapes = sorted({tuple(shape) for shape in shapes})
        ptr_stability[name] = {
            "count": len(values),
            "unique_count": len(set(values)),
            "stable": bool(values) and len(set(values)) == 1,
            "values": sorted(set(values))[:8],
            "shapes": [list(shape) for shape in unique_shapes[:8]],
            "ndims": sorted(set(ndims))[:8],
            "storage_offsets": sorted(set(storage_offsets))[:8],
        }

    observed_split = None
    if expected_observations:
        observed_split = {
            k: expected_observations[-1][k]
            for k in (
                "total_tokens",
                "first_tokens",
                "second_tokens",
                "first_start_num_tokens",
                "second_start_num_tokens",
            )
        }
    elif observed_splits:
        observed_split = {
            k: observed_splits[-1][k]
            for k in (
                "total_tokens",
                "first_tokens",
                "second_tokens",
                "first_start_num_tokens",
                "second_start_num_tokens",
            )
        }

    expected_capture_count = (
        len(expected_lazy_captures) if expected_lazy_captures
        else len(expected_captures)
    )
    inferred_replay_count = (
        max(0, len(expected_exec) - expected_capture_count)
        if expected_split is not None else 0
    )
    parallel_gate = {
        "descriptor_parallel_stream_count": sum(
            1 for row in expected_descriptors
            if bool(row.get("in_parallel_streams", False))),
        "execution_parallel_stream_count": sum(
            1 for row in second_exec
            if str(row.get("stream", "")) == "parallel"),
        "original_offset_view_count": sum(
            1 for row in second_exec
            if row.get("buffer_source") == "original_offset_view"),
        "parallel_graph_entry_pool_count": sum(
            1 for row in second_exec
            if row.get("graph_entry_pool") == "parallel"),
        "parallel_graph_params_pool_count": sum(
            1 for row in second_exec
            if row.get("graph_params_pool") == "parallel"),
    }

    failures: list[str] = []
    if expected_no_split_reason is not None:
        observed = fallback_reason_histogram.get(expected_no_split_reason, 0)
        if observed < 1:
            failures.append(
                "expected no-split fallback reason not observed: "
                f"{expected_no_split_reason}")
        if inplace_exec:
            failures.append(
                "inplace execution observed despite expected no-split "
                f"fallback: {len(inplace_exec)} events")
    if expected_split is not None:
        if len(expected_observations) < 3:
            failures.append(
                "expected split observed fewer than 3 decode steps: "
                f"{len(expected_observations)}")
        if not expected_descriptors:
            failures.append(
                f"no split-1 descriptor observed for {expected_split}")
        if expected_capture_count < 1:
            failures.append(
                f"no offset lazy capture observed for {expected_split}")
        if expected_capture_count > 1:
            failures.append(
                f"offset descriptor captured more than once: "
                f"{expected_capture_count}")
        if expected_capture_count >= 1 and inferred_replay_count < 1:
            failures.append(
                "no inferred replay observed after offset lazy capture")
    if expected_split is not None and second_exec:
        unstable = [
            name for name, info in ptr_stability.items()
            if info["count"] > 1 and not info["stable"]
        ]
        if unstable:
            failures.append(f"unstable split-1 ptrs: {unstable}")
        missing = [
            name for name, info in ptr_stability.items()
            if info["count"] == 0
        ]
        if missing:
            failures.append(f"missing split-1 ptr observations: {missing}")
    if expected_split is not None and not second_exec:
        failures.append("no split-1 inplace execution event observed")
    if expected_split is not None and split_mode == "inplace_parallel":
        if parallel_gate["descriptor_parallel_stream_count"] == 0:
            failures.append("split-1 descriptor did not use parallel stream")
        if parallel_gate["execution_parallel_stream_count"] == 0:
            failures.append("split-1 execution did not use parallel stream")
        if parallel_gate["original_offset_view_count"] == 0:
            failures.append(
                "split-1 inplace_parallel execution did not report "
                "original_offset_view buffer source")
        if parallel_gate["parallel_graph_entry_pool_count"] == 0:
            failures.append(
                "split-1 inplace_parallel execution did not use parallel "
                "graph entry pool")
        if parallel_gate["parallel_graph_params_pool_count"] == 0:
            failures.append(
                "split-1 inplace_parallel execution did not use parallel "
                "GraphParams pool")

    return {
        "path": path,
        "exists": os.path.exists(path),
        "num_events": len(rows),
        "event_counts": event_counts,
        "latest_planner_decision":
        planner_decisions[-1] if planner_decisions else None,
        "fallback_reason_histogram": fallback_reason_histogram,
        "expected_no_split_reason": expected_no_split_reason,
        "latest_descriptors": descriptors[-4:],
        "expected_split": expected_split,
        "expected_graph_variant": expected_graph_variant,
        "expected_split_observations": {
            "count": len(expected_observations),
            "step_ids": [
                row.get("step_id") for row in expected_observations
                if row.get("step_id") is not None
            ],
        },
        "observed_splits": observed_splits[-16:],
        "observed_split_histogram": observed_split_histogram,
        "observed_split": observed_split,
        "offset_graph": {
            "capture_count": expected_capture_count,
            "acl_capture_count": len(expected_captures),
            "lazy_capture_count": len(expected_lazy_captures),
            "replay_count": len(acl_replays),
            "inferred_replay_count": inferred_replay_count,
            "unexpected_capture_count": max(0, expected_capture_count - 1),
        },
        "ptr_stability": ptr_stability,
        "parallel_gate": parallel_gate,
        "failures": failures,
    }


class _TeeTextIO:
    """Write to multiple text streams (used to tee stdout/stderr to a file)."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
            except Exception:
                # Best-effort: keep other streams working.
                pass
        return n

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for st in self._streams:
            try:
                if hasattr(st, "isatty") and st.isatty():
                    return True
            except Exception:
                continue
        return False


def create_parser() -> FlexibleArgumentParser:
    """Create a CLI parser in the same style as vLLM offline examples."""
    parser = FlexibleArgumentParser()

    # Add all standard engine/vllm args (model, tokenizer, compilation_config, etc).
    EngineArgs.add_cli_args(parser)

    # Match previous behavior in this script.
    parser.set_defaults(trust_remote_code=True)

    # Default compilation config: pass explicitly so Ascend platform sees it.
    # IMPORTANT: Only include keys that exist on vLLM's `CompilationConfig`.
    cudagraph_sizes = [1,2,3,4,5,6,7,8]
    compilation_config = {
        "level": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
    }
    parser.set_defaults(compilation_config=compilation_config)

    test_group = parser.add_argument_group("Split-batch correctness test")
    test_group.add_argument("--max-tokens", type=int, default=64)
    test_group.add_argument("--batch-size", type=int, default=8)
    test_group.add_argument(
        "--fixed-batch-size",
        type=int,
        default=None,
        help=(
            "Fixed request batch size. Overrides --batch-size and, when "
            "--capture-sizes is omitted, defaults capture sizes to "
            "256,384,512."
        ),
    )
    test_group.add_argument(
        "--fixed-batch-query-len",
        type=int,
        default=1,
        help=(
            "Scheduled token count per request for fixed-batch trace "
            "expectations. Use 1 for normal decode and "
            "1 + num_speculative_tokens for uniform spec decode."
        ),
    )
    test_group.add_argument("--num-splits", type=int, default=2)
    test_group.add_argument("--min-batch-size-for-split", type=int, default=4)
    test_group.add_argument(
        "--split-mode",
        type=str,
        default="parallel_buffer",
        choices=["parallel_buffer", "inplace_serial", "inplace_parallel"],
        help="split_batch_config.mode for the enabled run.",
    )
    test_group.add_argument(
        "--capture-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated cudagraph capture sizes. Phase-10 example: "
            "--capture-sizes 256,384,512."
        ),
    )
    test_group.add_argument(
        "--pa-shape-list",
        type=str,
        default=None,
        help=(
            "Comma-separated ordinary graph shapes routed through paged "
            "attention. Used for backend-policy diagnostics, for example "
            "--pa-shape-list 512."
        ),
    )
    test_group.add_argument(
        "--enable-parallel-streams",
        action="store_true",
        help="Enable split parallel streams (if supported).",
    )
    test_group.add_argument(
        "--parallel-capture-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated capture sizes for the parallel-stream graph pool "
            "(split_batch_config.parallel_capture_sizes). "
            "When omitted the parallel pool reuses cudagraph_capture_sizes. "
            "Example: --parallel-capture-sizes 1,2,4,8,16,32,64,128"
        ),
    )
    test_group.add_argument(
        "--inplace-split-planner-policy",
        "--inplace-split-first-tokens-policy",
        dest="inplace_split_planner_policy",
        choices=["largest_lower", "balanced", "macro_cube_balanced"],
        default="largest_lower",
        help=(
            "Inplace split planner policy. largest_lower preserves the "
            "current largest-lower main graph split; balanced chooses a more "
            "even split among valid lower main graphs; macro_cube_balanced "
            "uses split_batch_config.macro_graph_config."
        ),
    )
    test_group.add_argument(
        "--macro-graph-config-json",
        type=str,
        default="",
        help=(
            "JSON object assigned to split_batch_config.macro_graph_config. "
            "When enabled, inplace lazy capture is disabled automatically."
        ),
    )
    test_group.add_argument(
        "--inplace-parallel-replay-policy",
        choices=["full_graph_parallel", "piecewise_attention_parallel"],
        default="full_graph_parallel",
        help=(
            "Replay policy for split_mode=inplace_parallel. "
            "piecewise_attention_parallel requires --cudagraph-mode "
            "PIECEWISE."
        ),
    )
    test_group.add_argument(
        "--piecewise-scheduler-sync-policy",
        choices=["host_sync", "event_chain"],
        default="event_chain",
        help=(
            "Synchronization policy for piecewise_attention_parallel. "
            "event_chain uses device events; host_sync preserves the original "
            "per-piece stream synchronize behavior."
        ),
    )
    test_group.add_argument(
        "--piecewise-attention-enqueue-policy",
        choices=["per_piece_thread", "persistent_thread"],
        default="persistent_thread",
        help=(
            "CPU enqueue policy for attention pieces in "
            "piecewise_attention_parallel."
        ),
    )
    test_group.add_argument(
        "--enable-mixed-request-split",
        action="store_true",
        help=(
            "Enable request-boundary mixed/prefill split planner in "
            "split_batch_config."
        ),
    )
    test_group.add_argument(
        "--mixed-request-split-execution-mode",
        choices=["dry_run", "serial", "piecewise_attention_parallel"],
        default="dry_run",
        help=(
            "Execution mode for mixed request split. serial is the minimal "
            "correctness path; piecewise_attention_parallel exercises the "
            "dual-stream attention overlap path."
        ),
    )
    test_group.add_argument(
        "--mixed-request-min-total-tokens",
        type=int,
        default=128,
        help="split_batch_config.mixed_request_min_total_tokens.",
    )
    test_group.add_argument(
        "--mixed-request-min-tokens-per-split",
        type=int,
        default=64,
        help="split_batch_config.mixed_request_min_tokens_per_split.",
    )
    test_group.add_argument(
        "--mixed-request-max-single-request-ratio",
        type=float,
        default=0.70,
        help="split_batch_config.mixed_request_max_single_request_ratio.",
    )
    test_group.add_argument(
        "--mixed-request-max-padding-ratio-per-split",
        type=float,
        default=0.0,
        help="split_batch_config.mixed_request_max_padding_ratio_per_split.",
    )
    test_group.add_argument(
        "--mixed-request-min-prefill-reqs-for-prefill-split",
        type=int,
        default=2,
        help=(
            "split_batch_config."
            "mixed_request_min_prefill_reqs_for_prefill_split."
        ),
    )
    test_group.add_argument(
        "--cudagraph-mode",
        choices=[
            "NONE",
            "PIECEWISE",
            "FULL",
            "FULL_DECODE_ONLY",
            "FULL_AND_PIECEWISE",
        ],
        default=None,
        help=(
            "Override compilation_config.cudagraph_mode for this run. "
            "Use PIECEWISE for piecewise attention parallel profiling."
        ),
    )
    test_group.add_argument(
        "--inplace-offset-capture-sizes",
        type=str,
        default="",
        help=(
            "Optional comma-separated graph sizes for inplace offset buckets. "
            "Example: 32,64,128,256."
        ),
    )
    test_group.add_argument(
        "--inplace-offset-max-graph-tokens-by-start",
        type=str,
        default="",
        help=(
            "Optional comma-separated start:max_graph_tokens caps for inplace "
            "offset graphs. Example: 128:64,384:64."
        ),
    )
    test_group.add_argument(
        "--inplace-offset-min-graph-tokens",
        type=int,
        default=1,
        help="Minimum graph size allowed for inplace offset graphs.",
    )
    test_group.add_argument(
        "--inplace-offset-allowed-graph-tokens-by-start",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated exact start:size-list mapping for "
            "inplace offset graphs. Example: 32:16|32;64:16|32|64."
        ),
    )
    test_group.add_argument(
        "--force-split",
        action="store_true",
        help=(
            "Force split-batch for every decode step regardless of padding "
            "savings (split_batch_config.force_split). Useful for benchmarking "
            "the split path on all batch sizes including exact graph hits."
        ),
    )
    test_group.add_argument(
        "--validate-ptrs",
        action="store_true",
        help=(
            "Enable inplace input/metadata pointer validation for the enabled "
            "run (split_batch_config.inplace_validate_metadata_ptrs)."
        ),
    )
    force_pa_group = test_group.add_mutually_exclusive_group()
    force_pa_group.add_argument(
        "--inplace-force-pa-for-offset",
        dest="inplace_force_pa_for_offset",
        action="store_true",
        default=False,
        help=(
            "Diagnostic only: force offset inplace serial microbatches to use "
            "PA instead of the selected attention backend."
        ),
    )
    force_pa_group.add_argument(
        "--no-inplace-force-pa-for-offset",
        dest="inplace_force_pa_for_offset",
        action="store_false",
        help=(
            "Diagnostic only: keep offset inplace serial microbatches on the "
            "selected attention backend. This is the default."
        ),
    )
    test_group.add_argument(
        "--enable-inplace-spec-decode",
        action="store_true",
        help=(
            "Allow inplace split for uniform speculative decode. Default is "
            "off so spec decode keeps the Phase-12 fallback gate."
        ),
    )
    test_group.add_argument(
        "--enable-inplace-mrope",
        action="store_true",
        help=(
            "Allow inplace_serial split for M-RoPE. Default is off so M-RoPE "
            "keeps the Phase-12 fallback gate."
        ),
    )
    test_group.add_argument(
        "--split-debug",
        action="store_true",
        help="Collect VLLM_ASCEND_SPLIT_INPLACE_DEBUG JSONL for child runs.",
    )
    test_group.add_argument(
        "--expect-split",
        type=str,
        default=None,
        help=(
            "Expected first,second split token counts, for example 384,32. "
            "If omitted with --fixed-batch-size and --split-mode inplace_serial, "
            "the value is derived from --capture-sizes."
        ),
    )
    test_group.add_argument(
        "--expect-no-split-reason",
        type=str,
        default=None,
        help=(
            "Expected split_planner_decision reason when validating an "
            "intentional inplace fallback, for example no_split_mrope. "
            "When set, trace validation requires no offset inplace execution."
        ),
    )
    test_group.add_argument(
        "--force-fixed-prompts",
        action="store_true",
        help=(
            "Use the same prompt for every request and ignore EOS by default, "
            "so decode steps keep a fixed batch longer."
        ),
    )
    test_group.add_argument(
        "--fixed-prompt",
        type=str,
        default=(
            "Write one concise sentence about deterministic batch scheduling."
        ),
        help="Prompt used when --force-fixed-prompts is set.",
    )
    test_group.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set SamplingParams(ignore_eos=True).",
    )
    test_group.add_argument(
        "--run",
        type=str,
        default="both",
        choices=["both", "disabled", "enabled"],
        help=(
            "Which run(s) to execute: both (compare), disabled only, enabled only. "
            "Coordinator mode runs children when run=both (default)."
        ),
    )
    test_group.add_argument(
        "--compare-mode",
        type=str,
        default="subprocess",
        choices=["subprocess", "inproc"],
        help=(
            "When --run=both: 'subprocess' (recommended) runs two child processes and "
            "compares saved outputs; 'inproc' runs two engines in-process."
        ),
    )
    test_group.add_argument(
        "--output-dir",
        type=str,
        default="./split_batch_correctness_results",
        help=(
            "Base output directory. This script will always create a timestamp subdir "
            "and write all results there (including console.log)."
        ),
    )
    test_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "JSON file containing list[str] prompts. In coordinator mode, this is "
            "auto-generated and passed to child runs."
        ),
    )
    test_group.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Single-run output JSON path (used by coordinator children). If set, "
            "the output directory is derived from this path."
        ),
    )

    prof_group = parser.add_argument_group("PyTorch Profiler (torch_npu)")
    prof_group.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Enable torch_npu profiler and write TensorBoard traces. "
            "NOTE: For meaningful NPU kernel traces with vLLM V1, run with "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 so model executes in-process."
        ),
    )
    prof_group.add_argument(
        "--profile-dir",
        type=str,
        default="./torch_profiler",
        help="Output directory for TensorBoard traces.",
    )
    prof_group.add_argument(
        "--profile-target",
        type=str,
        default="enabled",
        choices=["disabled", "enabled", "both"],
        help="Which run to profile: split disabled, enabled, or both.",
    )
    prof_group.add_argument(
        "--profile-record-shapes",
        action="store_true",
        help="Record operator input shapes (more overhead).",
    )
    prof_group.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Record Python call stacks (more overhead).",
    )
    return parser


def _build_llm_from_args(
    llm_args: dict[str, Any],
    *,
    additional_config: dict[str, Any],
) -> LLM:
    args = dict(llm_args)
    args["additional_config"] = additional_config
    return LLM(**args)


def _extract_first_output(output) -> tuple[list[int], str]:
    # vLLM RequestOutput: output.outputs is a list (n=best_of)
    if not output.outputs:
        return [], ""
    o0 = output.outputs[0]
    token_ids = list(getattr(o0, "token_ids", []) or [])
    text = str(getattr(o0, "text", "") or "")
    return token_ids, text


def _default_prompts() -> list[str]:
    return [
        "Hello, my name is",
        "The capital of France is",
        "Explain in one sentence: what is split-batch?",
        "Write a short list of 3 items about: apples",
        "The president of the United States is",
        "Once upon a time in a land far away,",
        "In computer science, a binary tree is",
        "The quick brown fox jumps over the lazy dog.",
    ]


def _generate_prompts(*, batch_size: int, seed: int | None) -> list[str]:
    base = _default_prompts()
    prompts = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]

    # Make prompt order deterministic but seed-dependent.
    rng = random.Random(int(seed or 0))
    rng.shuffle(prompts)
    return prompts


def _generate_fixed_prompts(*, batch_size: int, prompt: str) -> list[str]:
    return [prompt for _ in range(batch_size)]


def _load_prompts(prompts_file: str) -> list[str]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ValueError("--prompts-file must be a JSON list[str]")
    return list(prompts)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _serialize_outputs(prompts: list[str], outputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, out in enumerate(outputs):
        token_ids, text = _extract_first_output(out)
        rows.append(
            {
                "index": i,
                "prompt": prompts[i] if i < len(prompts) else None,
                "token_ids": token_ids,
                "text": text,
            }
        )
    return rows


def _compare_serialized(
    disabled_rows: list[dict[str, Any]],
    enabled_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    if len(disabled_rows) != len(enabled_rows):
        mismatches.append(
            {
                "index": None,
                "reason": "output length differs",
                "disabled_len": len(disabled_rows),
                "enabled_len": len(enabled_rows),
            }
        )

    n = min(len(disabled_rows), len(enabled_rows))
    for i in range(n):
        a = disabled_rows[i]
        b = enabled_rows[i]
        if a.get("prompt") != b.get("prompt"):
            mismatches.append(
                {
                    "index": i,
                    "reason": "prompt differs",
                    "disabled_prompt": a.get("prompt"),
                    "enabled_prompt": b.get("prompt"),
                }
            )
            continue

        if a.get("token_ids") != b.get("token_ids") or a.get("text") != b.get("text"):
            disabled_token_ids = list(a.get("token_ids") or [])
            enabled_token_ids = list(b.get("token_ids") or [])
            first_token_mismatch = None
            for token_idx, (disabled_token, enabled_token) in enumerate(
                    zip(disabled_token_ids, enabled_token_ids)):
                if disabled_token != enabled_token:
                    first_token_mismatch = {
                        "position": token_idx,
                        "disabled": disabled_token,
                        "enabled": enabled_token,
                    }
                    break
            if first_token_mismatch is None and (
                    len(disabled_token_ids) != len(enabled_token_ids)):
                first_token_mismatch = {
                    "position": min(
                        len(disabled_token_ids), len(enabled_token_ids)),
                    "disabled_len": len(disabled_token_ids),
                    "enabled_len": len(enabled_token_ids),
                }
            mismatches.append(
                {
                    "index": i,
                    "prompt": a.get("prompt"),
                    "first_token_mismatch": first_token_mismatch,
                    "disabled": {
                        "token_ids": disabled_token_ids,
                        "text": a.get("text"),
                    },
                    "enabled": {
                        "token_ids": enabled_token_ids,
                        "text": b.get("text"),
                    },
                }
            )

    return mismatches


def _cleanup_llm(llm: LLM) -> None:
    """Release engine resources before starting the next run."""
    try:
        # vLLM's `LLM` does not expose a stable `shutdown()` method.
        # For the V1 engine, we must explicitly shutdown the underlying
        # EngineCore client to terminate the EngineCore subprocess(es),
        # otherwise the first run can keep NPU memory reserved and make the
        # second run see `Available memory: 0`.
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    finally:
        # Best-effort memory release
        if hasattr(torch, "npu"):
            if hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
        gc.collect()


def _run_with_torch_profiler(
    *,
    enabled: bool,
    profile_dir: str,
    run_name: str,
    record_shapes: bool,
    with_stack: bool,
    fn,
):
    if not enabled:
        return fn()

    try:
        from torch_npu import profiler as npu_profiler
    except Exception as e:
        raise RuntimeError(
            "--profile was set but torch_npu.profiler is not available"
        ) from e

    os.makedirs(profile_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(profile_dir, f"{ts}_{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    activities = [npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU]
    trace_handler = npu_profiler.tensorboard_trace_handler(out_dir)

    # Use an explicit schedule to avoid "stop while RECORD" edge cases.
    schedule = npu_profiler.schedule(wait=0, warmup=0, active=1, repeat=1)

    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text
            ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        mstx=False,    # 原参数名msprof_tx改为mstx，新版本依旧兼容原参数名msprof_tx
        mstx_domain_include=[],
        mstx_domain_exclude=[],
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False
    )

    with npu_profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_modules=True,
        on_trace_ready=trace_handler,
        profile_memory=False,
        experimental_config=experimental_config
    ) as prof:
        result = fn()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        prof.step()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        return result


def _build_split_additional_config(
    *,
    enabled: bool,
    split_mode: str,
    num_splits: int,
    enable_parallel_streams: bool,
    min_batch_size_for_split: int,
    parallel_capture_sizes: list[int] | None = None,
    force_split: bool = False,
    validate_ptrs: bool = False,
    inplace_force_pa_for_offset: bool = False,
    enable_inplace_spec_decode: bool = False,
    enable_inplace_mrope: bool = False,
    inplace_split_planner_policy: str = "largest_lower",
    inplace_offset_capture_sizes: list[int] | None = None,
    inplace_offset_max_graph_tokens_by_start: dict[int, int] | None = None,
    inplace_offset_min_graph_tokens: int = 1,
    inplace_offset_allowed_graph_tokens_by_start: (
        dict[int, list[int]] | None) = None,
    inplace_parallel_replay_policy: str = "full_graph_parallel",
    piecewise_scheduler_sync_policy: str = "event_chain",
    piecewise_attention_enqueue_policy: str = "persistent_thread",
    enable_mixed_request_split: bool = False,
    mixed_request_split_execution_mode: str = "dry_run",
    mixed_request_min_total_tokens: int = 128,
    mixed_request_min_tokens_per_split: int = 64,
    mixed_request_max_single_request_ratio: float = 0.70,
    mixed_request_max_padding_ratio_per_split: float = 0.0,
    mixed_request_min_prefill_reqs_for_prefill_split: int = 2,
    macro_graph_config: dict[str, Any] | None = None,
    pa_shape_list: list[int] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": enabled,
        "mode": split_mode,
        "num_splits": num_splits,
        "enable_parallel_streams": enable_parallel_streams,
        "min_batch_size_for_split": min_batch_size_for_split,
    }
    macro_graph_enabled = bool(
        macro_graph_config is not None
        and macro_graph_config.get("enabled", False))
    if parallel_capture_sizes is not None:
        cfg["parallel_capture_sizes"] = parallel_capture_sizes
    if force_split:
        cfg["force_split"] = True
    if split_mode.startswith("inplace"):
        cfg["enable_inplace_lazy_capture"] = not macro_graph_enabled
        cfg["inplace_validate_metadata_ptrs"] = bool(validate_ptrs)
        cfg["inplace_force_pa_for_offset"] = bool(
            inplace_force_pa_for_offset)
        cfg["enable_inplace_spec_decode"] = bool(enable_inplace_spec_decode)
        cfg["enable_inplace_mrope"] = bool(enable_inplace_mrope)
        cfg["inplace_split_planner_policy"] = inplace_split_planner_policy
        cfg["inplace_split_first_tokens_policy"] = (
            inplace_split_planner_policy)
        cfg["inplace_parallel_replay_policy"] = (
            inplace_parallel_replay_policy)
        cfg["piecewise_scheduler_sync_policy"] = (
            piecewise_scheduler_sync_policy)
        cfg["piecewise_attention_enqueue_policy"] = (
            piecewise_attention_enqueue_policy)
        if enabled and enable_mixed_request_split:
            cfg["enable_mixed_request_split"] = True
            cfg["mixed_request_split_execution_mode"] = (
                mixed_request_split_execution_mode)
            cfg["mixed_request_min_total_tokens"] = int(
                mixed_request_min_total_tokens)
            cfg["mixed_request_min_tokens_per_split"] = int(
                mixed_request_min_tokens_per_split)
            cfg["mixed_request_max_single_request_ratio"] = float(
                mixed_request_max_single_request_ratio)
            cfg["mixed_request_max_padding_ratio_per_split"] = float(
                mixed_request_max_padding_ratio_per_split)
            cfg["mixed_request_min_prefill_reqs_for_prefill_split"] = int(
                mixed_request_min_prefill_reqs_for_prefill_split)
        cfg["inplace_offset_match_policy"] = "bucket"
        offset_capture_sizes = (
            inplace_offset_capture_sizes
            if inplace_offset_capture_sizes is not None else
            parallel_capture_sizes)
        if offset_capture_sizes is not None:
            cfg["inplace_offset_capture_sizes"] = list(offset_capture_sizes)
        cfg["inplace_offset_min_graph_tokens"] = int(
            inplace_offset_min_graph_tokens)
        cfg["inplace_offset_max_padding_tokens"] = 127
        cfg["inplace_offset_max_padding_ratio"] = 8.0
        if inplace_offset_max_graph_tokens_by_start:
            cfg["inplace_offset_max_graph_tokens_by_start"] = (
                inplace_offset_max_graph_tokens_by_start)
        if inplace_offset_allowed_graph_tokens_by_start:
            cfg["inplace_offset_allowed_graph_tokens_by_start"] = (
                inplace_offset_allowed_graph_tokens_by_start)
        if macro_graph_config is not None:
            cfg["macro_graph_config"] = macro_graph_config
    additional_config: dict[str, Any] = {"split_batch_config": cfg}
    if pa_shape_list is not None:
        additional_config["pa_shape_list"] = list(pa_shape_list)
    return additional_config


def _run_single(
    *,
    llm_args: dict[str, Any],
    prompts: list[str],
    sampling: SamplingParams,
    additional_config: dict[str, Any],
    profile_enabled: bool,
    profile_dir: str,
    profile_target: str,
    profile_record_shapes: bool,
    profile_with_stack: bool,
    run_name: str,
):
    llm = _build_llm_from_args(llm_args, additional_config=additional_config)
    outputs = _run_with_torch_profiler(
        enabled=profile_enabled and profile_target in (run_name, "both"),
        profile_dir=profile_dir,
        run_name=f"split_{run_name}",
        record_shapes=profile_record_shapes,
        with_stack=profile_with_stack,
        fn=lambda: llm.generate(prompts, sampling),
    )
    return outputs, llm


def _coordinator_subprocess(*, base_args: dict[str, Any], prompts: list[str], **ctx) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(str(ctx["output_dir_base"]), timestamp)
    os.makedirs(out_dir, exist_ok=True)

    prompts_path = os.path.join(out_dir, "prompts.json")
    _save_json(prompts_path, prompts)

    metadata = {
        "timestamp": timestamp,
        "model": base_args.get("model"),
        "tokenizer": base_args.get("tokenizer") or base_args.get("model"),
        "seed": base_args.get("seed"),
        "batch_size": ctx["batch_size"],
        "fixed_batch_size": ctx["fixed_batch_size"],
        "fixed_batch_query_len": ctx["fixed_batch_query_len"],
        "fixed_batch_total_tokens": ctx["fixed_batch_total_tokens"],
        "max_tokens": ctx["max_tokens"],
        "max_model_len": base_args.get("max_model_len"),
        "gpu_memory_utilization": base_args.get("gpu_memory_utilization"),
        "split_mode": ctx["split_mode"],
        "num_splits": ctx["num_splits"],
        "enable_parallel_streams": ctx["enable_parallel_streams"],
        "min_batch_size_for_split": ctx["min_batch_size_for_split"],
        "validate_ptrs": ctx["validate_ptrs"],
        "enable_inplace_spec_decode": ctx["enable_inplace_spec_decode"],
        "enable_inplace_mrope": ctx["enable_inplace_mrope"],
        "inplace_split_planner_policy": ctx["inplace_split_planner_policy"],
        "inplace_parallel_replay_policy": (
            ctx["inplace_parallel_replay_policy"]),
        "piecewise_scheduler_sync_policy": (
            ctx["piecewise_scheduler_sync_policy"]),
        "piecewise_attention_enqueue_policy": (
            ctx["piecewise_attention_enqueue_policy"]),
        "enable_mixed_request_split": ctx["enable_mixed_request_split"],
        "mixed_request_split_execution_mode": (
            ctx["mixed_request_split_execution_mode"]),
        "mixed_request_min_total_tokens": (
            ctx["mixed_request_min_total_tokens"]),
        "mixed_request_min_tokens_per_split": (
            ctx["mixed_request_min_tokens_per_split"]),
        "mixed_request_max_single_request_ratio": (
            ctx["mixed_request_max_single_request_ratio"]),
        "mixed_request_max_padding_ratio_per_split": (
            ctx["mixed_request_max_padding_ratio_per_split"]),
        "mixed_request_min_prefill_reqs_for_prefill_split": (
            ctx["mixed_request_min_prefill_reqs_for_prefill_split"]),
        "inplace_offset_capture_sizes": ctx["inplace_offset_capture_sizes"],
        "inplace_offset_max_graph_tokens_by_start": (
            ctx["inplace_offset_max_graph_tokens_by_start"]),
        "inplace_offset_min_graph_tokens": (
            ctx["inplace_offset_min_graph_tokens"]),
        "inplace_offset_allowed_graph_tokens_by_start": (
            ctx["inplace_offset_allowed_graph_tokens_by_start"]),
        "split_debug": ctx["split_debug"],
        "expected_split": ctx["expected_split"],
        "expected_no_split_reason": ctx["expected_no_split_reason"],
        "pa_shape_list": ctx["pa_shape_list"],
        "compilation_config": base_args.get("compilation_config"),
        "compare_mode": "subprocess",
    }
    _save_json(os.path.join(out_dir, "metadata.json"), metadata)

    outputs_disabled_path = os.path.join(out_dir, "outputs_split_disabled.json")
    outputs_enabled_path = os.path.join(out_dir, "outputs_split_enabled.json")
    debug_disabled_path = os.path.join(out_dir,
                                       "split_inplace_debug_disabled.jsonl")
    debug_enabled_path = os.path.join(out_dir,
                                      "split_inplace_debug_enabled.jsonl")

    script_path = str(Path(__file__).resolve())

    # Forward argv, but remove coordinator-only flags and --run.
    forward_argv: list[str] = []
    skip_next = False
    coordinator_flags_with_value = {
        "--run",
        "--compare-mode",
        "--output-dir",
        "--prompts-file",
        "--output-file",
    }
    profiler_flags_with_value = {
        "--profile-dir",
        "--profile-target",
    }
    profiler_flags_bool = {
        "--profile",
        "--profile-record-shapes",
        "--profile-with-stack",
    }

    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in coordinator_flags_with_value:
            skip_next = True
            continue
        # Don't forward profiler flags by default; we attach per-child below.
        if tok in profiler_flags_with_value:
            skip_next = True
            continue
        if tok in profiler_flags_bool:
            continue
        forward_argv.append(tok)

    def _child_cmd(*, child_run: str, child_out: str) -> list[str]:
        cmd = [
            sys.executable,
            script_path,
            *forward_argv,
            "--run",
            child_run,
            "--compare-mode",
            "inproc",
            "--prompts-file",
            prompts_path,
            "--output-file",
            child_out,
            "--output-dir",
            out_dir,  # ensure children write console.log into the same run dir
        ]

        if ctx["profile_enabled"] and ctx["profile_target"] in (child_run, "both"):
            cmd.append("--profile")
            cmd.extend(["--profile-dir", ctx["profile_dir"]])
            cmd.extend(["--profile-target", child_run])
            if ctx["profile_record_shapes"]:
                cmd.append("--profile-record-shapes")
            if ctx["profile_with_stack"]:
                cmd.append("--profile-with-stack")
        return cmd

    def _child_env(*, child_run: str) -> dict[str, str]:
        env = os.environ.copy()
        if ctx["split_debug"]:
            env["VLLM_ASCEND_SPLIT_INPLACE_DEBUG"] = "1"
            env["VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"] = (
                debug_enabled_path
                if child_run == "enabled" else debug_disabled_path)
        return env

    print("=== Coordinator (subprocess) ===")
    print("Output dir:", out_dir)

    print("\n=== Run 1/2: split disabled (child process) ===")
    r0 = subprocess.run(
        _child_cmd(child_run="disabled", child_out=outputs_disabled_path),
        env=_child_env(child_run="disabled"))
    if r0.returncode != 0:
        print(f"FAIL: child(disabled) exit code={r0.returncode}", file=sys.stderr)
        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "FAIL",
                "reason": "child(disabled) non-zero exit",
                "returncode": r0.returncode,
            },
        )
        return r0.returncode or 1

    print("\n=== Run 2/2: split enabled (child process) ===")
    r1 = subprocess.run(
        _child_cmd(child_run="enabled", child_out=outputs_enabled_path),
        env=_child_env(child_run="enabled"))
    if r1.returncode != 0:
        print(f"FAIL: child(enabled) exit code={r1.returncode}", file=sys.stderr)
        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "FAIL",
                "reason": "child(enabled) non-zero exit",
                "returncode": r1.returncode,
            },
        )
        return r1.returncode or 1

    with open(outputs_disabled_path, "r", encoding="utf-8") as f:
        disabled_obj = json.load(f)
    with open(outputs_enabled_path, "r", encoding="utf-8") as f:
        enabled_obj = json.load(f)

    disabled_rows = disabled_obj.get("outputs", [])
    enabled_rows = enabled_obj.get("outputs", [])

    split_trace_summary = _summarize_split_debug_trace(
        debug_enabled_path,
        expected_split=ctx["expected_split"],
        split_mode=ctx["split_mode"],
        expected_no_split_reason=ctx["expected_no_split_reason"],
    ) if ctx["split_debug"] else None
    trace_summary_path = None
    if split_trace_summary is not None:
        trace_summary_path = os.path.join(out_dir, "split_trace_summary.json")
        _save_json(trace_summary_path, split_trace_summary)

    mismatches = _compare_serialized(disabled_rows, enabled_rows)
    if mismatches:
        diff_path = os.path.join(out_dir, "diff.json")
        _save_json(diff_path, {"mismatches": mismatches[:50], "total": len(mismatches)})
        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "FAIL",
                "mismatch_total": len(mismatches),
                "diff_path": diff_path,
                "trace_summary_path": trace_summary_path,
                "split_trace_summary": split_trace_summary,
            },
        )
        print(f"\nFAIL: {len(mismatches)} mismatches. See {diff_path}", file=sys.stderr)
        return 1

    if split_trace_summary is not None:
        if split_trace_summary["failures"]:
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "FAIL",
                    "reason": "split trace validation failed",
                    "trace_summary_path": trace_summary_path,
                    "trace_failures": split_trace_summary["failures"],
                    "outputs_disabled_path": outputs_disabled_path,
                    "outputs_enabled_path": outputs_enabled_path,
                },
            )
            print(
                "\nFAIL: split trace validation failed. See "
                f"{trace_summary_path}",
                file=sys.stderr)
            return 1

    sample = enabled_rows[0] if enabled_rows else {}
    _save_json(
        os.path.join(out_dir, "summary.json"),
        {
            "status": "PASS",
            "count": len(enabled_rows),
            "split_trace_summary": split_trace_summary,
            "sample": {
                "token_ids_len": len(sample.get("token_ids", []) or []),
                "text_preview": (sample.get("text") or "")[:200],
            },
        },
    )

    print(f"\nPASS: {len(prompts)}/{len(prompts)} outputs match exactly")
    print("Sample output[0] token_ids_len=", len(sample.get("token_ids", []) or []))
    print("Sample output[0] text=", repr((sample.get("text") or "")[:200]))
    return 0


def _make_run_dir(*, output_dir_base: str, output_file: str | None) -> tuple[str, str]:
    """Return (out_dir, timestamp). If output_file is set, derive out_dir from it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        out_dir = os.path.dirname(str(output_file)) or "."
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, timestamp
    out_dir = os.path.join(str(output_dir_base), timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, timestamp


def main() -> int:
    parser = create_parser()
    args: dict[str, Any] = vars(parser.parse_args())

    max_tokens = int(args.pop("max_tokens"))
    batch_size = int(args.pop("batch_size"))
    fixed_batch_size = args.pop("fixed_batch_size")
    if fixed_batch_size is not None:
        batch_size = int(fixed_batch_size)
    fixed_batch_query_len = int(args.pop("fixed_batch_query_len"))
    if fixed_batch_query_len < 1:
        print("ERROR: --fixed-batch-query-len must be >= 1",
              file=sys.stderr)
        return 2
    fixed_batch_total_tokens = (
        int(batch_size) * int(fixed_batch_query_len)
        if fixed_batch_size is not None else None
    )
    num_splits = int(args.pop("num_splits"))
    min_batch_size_for_split = int(args.pop("min_batch_size_for_split"))
    split_mode = str(args.pop("split_mode"))
    _capture_sizes_raw = args.pop("capture_sizes")
    capture_sizes = _parse_int_list(_capture_sizes_raw)
    if fixed_batch_size is not None and capture_sizes is None:
        capture_sizes = [256, 384, 512]
    if capture_sizes is not None:
        _apply_capture_sizes(args, capture_sizes)
    cudagraph_mode = args.pop("cudagraph_mode")
    if cudagraph_mode is not None:
        _apply_cudagraph_mode(args, cudagraph_mode)
    pa_shape_list = _parse_int_list(args.pop("pa_shape_list"))
    enable_parallel_streams = bool(args.pop("enable_parallel_streams"))
    _parallel_capture_sizes_raw = args.pop("parallel_capture_sizes")
    parallel_capture_sizes = _parse_int_list(_parallel_capture_sizes_raw)
    inplace_split_planner_policy = str(args.pop(
        "inplace_split_planner_policy"))
    try:
        macro_graph_config = _parse_json_object(
            str(args.pop("macro_graph_config_json") or ""),
            arg_name="--macro-graph-config-json",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    inplace_parallel_replay_policy = str(args.pop(
        "inplace_parallel_replay_policy"))
    piecewise_scheduler_sync_policy = str(args.pop(
        "piecewise_scheduler_sync_policy"))
    piecewise_attention_enqueue_policy = str(args.pop(
        "piecewise_attention_enqueue_policy"))
    enable_mixed_request_split = bool(
        args.pop("enable_mixed_request_split"))
    mixed_request_split_execution_mode = str(
        args.pop("mixed_request_split_execution_mode"))
    mixed_request_min_total_tokens = int(
        args.pop("mixed_request_min_total_tokens"))
    mixed_request_min_tokens_per_split = int(
        args.pop("mixed_request_min_tokens_per_split"))
    mixed_request_max_single_request_ratio = float(
        args.pop("mixed_request_max_single_request_ratio"))
    mixed_request_max_padding_ratio_per_split = float(
        args.pop("mixed_request_max_padding_ratio_per_split"))
    mixed_request_min_prefill_reqs_for_prefill_split = int(
        args.pop("mixed_request_min_prefill_reqs_for_prefill_split"))
    _inplace_offset_capture_sizes_raw = str(
        args.pop("inplace_offset_capture_sizes") or "")
    inplace_offset_capture_sizes = (
        _parse_int_list(_inplace_offset_capture_sizes_raw)
        if _inplace_offset_capture_sizes_raw.strip() else None)
    inplace_offset_max_graph_tokens_by_start = _parse_start_graph_caps(
        str(args.pop("inplace_offset_max_graph_tokens_by_start") or ""))
    inplace_offset_min_graph_tokens = int(
        args.pop("inplace_offset_min_graph_tokens"))
    inplace_offset_allowed_graph_tokens_by_start = (
        _parse_start_graph_allowed_sizes(
            str(args.pop("inplace_offset_allowed_graph_tokens_by_start")
                or "")))
    if (split_mode.startswith("inplace") and inplace_offset_capture_sizes
            and parallel_capture_sizes is not None):
        parallel_capture_sizes = sorted(
            set(parallel_capture_sizes) | set(inplace_offset_capture_sizes))
    force_split = bool(args.pop("force_split"))
    validate_ptrs = bool(args.pop("validate_ptrs"))
    inplace_force_pa_for_offset = bool(
        args.pop("inplace_force_pa_for_offset"))
    enable_inplace_spec_decode = bool(args.pop("enable_inplace_spec_decode"))
    enable_inplace_mrope = bool(args.pop("enable_inplace_mrope"))
    split_debug = bool(args.pop("split_debug"))
    expect_split_raw = args.pop("expect_split")
    expected_no_split_reason = args.pop("expect_no_split_reason")
    force_fixed_prompts = bool(args.pop("force_fixed_prompts"))
    fixed_prompt = str(args.pop("fixed_prompt"))
    ignore_eos = bool(args.pop("ignore_eos"))
    run_mode = str(args.pop("run"))
    compare_mode = str(args.pop("compare_mode"))
    output_dir_base = str(args.pop("output_dir"))
    prompts_file = args.pop("prompts_file")
    output_file = args.pop("output_file")

    profile_enabled = bool(args.pop("profile"))
    profile_dir = str(args.pop("profile_dir"))
    profile_target = str(args.pop("profile_target"))
    profile_record_shapes = bool(args.pop("profile_record_shapes"))
    profile_with_stack = bool(args.pop("profile_with_stack"))

    if split_mode.startswith("inplace"):
        split_debug = True if (split_debug or validate_ptrs
                               or fixed_batch_size is not None
                               or expected_no_split_reason) else split_debug
    if force_fixed_prompts:
        ignore_eos = True

    if fixed_batch_size is not None:
        try:
            _ensure_fixed_batch_graph_capacity(
                args,
                batch_size=int(fixed_batch_total_tokens),
                capture_sizes=capture_sizes,
            )
        except ValueError:
            return 2

    expected_split = None
    if expect_split_raw:
        expected_values = _parse_int_list(str(expect_split_raw))
        if expected_values is None or len(expected_values) != 2:
            print("ERROR: --expect-split must be first,second",
                  file=sys.stderr)
            return 2
        expected_split = {
            "total_tokens": int(sum(expected_values)),
            "first_tokens": int(expected_values[0]),
            "second_tokens": int(expected_values[1]),
            "first_start_num_tokens": 0,
            "second_start_num_tokens": int(expected_values[0]),
        }
    elif (fixed_batch_size is not None and split_mode.startswith("inplace")
          and capture_sizes is not None):
        expected_split = _expected_inplace_split(
            int(fixed_batch_total_tokens), capture_sizes)

    if expected_no_split_reason is not None:
        if not split_mode.startswith("inplace"):
            print("ERROR: --expect-no-split-reason requires inplace split mode",
                  file=sys.stderr)
            return 2
        if expect_split_raw:
            print(
                "ERROR: --expect-no-split-reason cannot be combined with "
                "--expect-split",
                file=sys.stderr,
            )
            return 2
        if expected_split is not None and not expect_split_raw:
            expected_split = None

    if expected_split is not None and max_tokens < 3:
        print(
            "ERROR: split trace validation needs --max-tokens >= 3 to observe "
            "one offset capture and later replay",
            file=sys.stderr,
        )
        return 2

    # In vLLM V1, setting VLLM_ENABLE_V1_MULTIPROCESSING=0 forces in-proc execution.
    # In this mode, releasing NPU memory fully between two separate engine instantiations
    # is unreliable (weights/KV cache/graphs may stay resident).
    v1_multiproc_env = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING")
    inproc_v1 = v1_multiproc_env is not None and str(v1_multiproc_env) in (
        "0",
        "false",
        "False",
    )

    if batch_size < 1:
        print("ERROR: --batch-size must be >= 1", file=sys.stderr)
        return 2

    seed = args.get("seed")
    if prompts_file:
        prompts = _load_prompts(str(prompts_file))
    elif force_fixed_prompts:
        prompts = _generate_fixed_prompts(batch_size=batch_size,
                                          prompt=fixed_prompt)
    else:
        prompts = _generate_prompts(batch_size=batch_size, seed=seed)

    if len(prompts) != batch_size:
        print(
            f"ERROR: prompts length ({len(prompts)}) != batch_size ({batch_size})",
            file=sys.stderr,
        )
        return 2

    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if ignore_eos:
        sampling_kwargs["ignore_eos"] = True
    sampling = SamplingParams(**sampling_kwargs)

    split_disabled_cfg = _build_split_additional_config(
        enabled=False,
        split_mode=split_mode,
        num_splits=num_splits,
        enable_parallel_streams=enable_parallel_streams,
        min_batch_size_for_split=min_batch_size_for_split,
        parallel_capture_sizes=parallel_capture_sizes,
        force_split=force_split,
        validate_ptrs=validate_ptrs,
        inplace_force_pa_for_offset=inplace_force_pa_for_offset,
        enable_inplace_spec_decode=enable_inplace_spec_decode,
        enable_inplace_mrope=enable_inplace_mrope,
        inplace_split_planner_policy=inplace_split_planner_policy,
        inplace_offset_capture_sizes=inplace_offset_capture_sizes,
        inplace_offset_max_graph_tokens_by_start=(
            inplace_offset_max_graph_tokens_by_start),
        inplace_offset_min_graph_tokens=inplace_offset_min_graph_tokens,
        inplace_offset_allowed_graph_tokens_by_start=(
            inplace_offset_allowed_graph_tokens_by_start),
        inplace_parallel_replay_policy=inplace_parallel_replay_policy,
        piecewise_scheduler_sync_policy=piecewise_scheduler_sync_policy,
        piecewise_attention_enqueue_policy=piecewise_attention_enqueue_policy,
        enable_mixed_request_split=enable_mixed_request_split,
        mixed_request_split_execution_mode=(
            mixed_request_split_execution_mode),
        mixed_request_min_total_tokens=mixed_request_min_total_tokens,
        mixed_request_min_tokens_per_split=(
            mixed_request_min_tokens_per_split),
        mixed_request_max_single_request_ratio=(
            mixed_request_max_single_request_ratio),
        mixed_request_max_padding_ratio_per_split=(
            mixed_request_max_padding_ratio_per_split),
        mixed_request_min_prefill_reqs_for_prefill_split=(
            mixed_request_min_prefill_reqs_for_prefill_split),
        macro_graph_config=None,
        pa_shape_list=pa_shape_list,
    )
    split_enabled_cfg = _build_split_additional_config(
        enabled=True,
        split_mode=split_mode,
        num_splits=num_splits,
        enable_parallel_streams=enable_parallel_streams,
        min_batch_size_for_split=min_batch_size_for_split,
        parallel_capture_sizes=parallel_capture_sizes,
        force_split=force_split,
        validate_ptrs=validate_ptrs,
        inplace_force_pa_for_offset=inplace_force_pa_for_offset,
        enable_inplace_spec_decode=enable_inplace_spec_decode,
        enable_inplace_mrope=enable_inplace_mrope,
        inplace_split_planner_policy=inplace_split_planner_policy,
        inplace_offset_capture_sizes=inplace_offset_capture_sizes,
        inplace_offset_max_graph_tokens_by_start=(
            inplace_offset_max_graph_tokens_by_start),
        inplace_offset_min_graph_tokens=inplace_offset_min_graph_tokens,
        inplace_offset_allowed_graph_tokens_by_start=(
            inplace_offset_allowed_graph_tokens_by_start),
        inplace_parallel_replay_policy=inplace_parallel_replay_policy,
        piecewise_scheduler_sync_policy=piecewise_scheduler_sync_policy,
        piecewise_attention_enqueue_policy=piecewise_attention_enqueue_policy,
        enable_mixed_request_split=enable_mixed_request_split,
        mixed_request_split_execution_mode=(
            mixed_request_split_execution_mode),
        mixed_request_min_total_tokens=mixed_request_min_total_tokens,
        mixed_request_min_tokens_per_split=(
            mixed_request_min_tokens_per_split),
        mixed_request_max_single_request_ratio=(
            mixed_request_max_single_request_ratio),
        mixed_request_max_padding_ratio_per_split=(
            mixed_request_max_padding_ratio_per_split),
        mixed_request_min_prefill_reqs_for_prefill_split=(
            mixed_request_min_prefill_reqs_for_prefill_split),
        macro_graph_config=macro_graph_config,
        pa_shape_list=pa_shape_list,
    )

    # Preferred: coordinator mode spawns 2 child processes then compares.
    if run_mode == "both" and compare_mode == "subprocess":
        return _coordinator_subprocess(
            base_args=args,
            prompts=prompts,
            output_dir_base=output_dir_base,
            batch_size=batch_size,
            fixed_batch_size=fixed_batch_size,
            fixed_batch_query_len=fixed_batch_query_len,
            fixed_batch_total_tokens=fixed_batch_total_tokens,
            max_tokens=max_tokens,
            split_mode=split_mode,
            num_splits=num_splits,
            enable_parallel_streams=enable_parallel_streams,
            min_batch_size_for_split=min_batch_size_for_split,
            validate_ptrs=validate_ptrs,
            enable_inplace_spec_decode=enable_inplace_spec_decode,
            enable_inplace_mrope=enable_inplace_mrope,
            inplace_split_planner_policy=inplace_split_planner_policy,
            inplace_parallel_replay_policy=inplace_parallel_replay_policy,
            piecewise_scheduler_sync_policy=piecewise_scheduler_sync_policy,
            piecewise_attention_enqueue_policy=(
                piecewise_attention_enqueue_policy),
            enable_mixed_request_split=enable_mixed_request_split,
            mixed_request_split_execution_mode=(
                mixed_request_split_execution_mode),
            mixed_request_min_total_tokens=mixed_request_min_total_tokens,
            mixed_request_min_tokens_per_split=(
                mixed_request_min_tokens_per_split),
            mixed_request_max_single_request_ratio=(
                mixed_request_max_single_request_ratio),
            mixed_request_max_padding_ratio_per_split=(
                mixed_request_max_padding_ratio_per_split),
            mixed_request_min_prefill_reqs_for_prefill_split=(
                mixed_request_min_prefill_reqs_for_prefill_split),
            inplace_offset_capture_sizes=inplace_offset_capture_sizes,
            inplace_offset_max_graph_tokens_by_start=(
                inplace_offset_max_graph_tokens_by_start),
            inplace_offset_min_graph_tokens=inplace_offset_min_graph_tokens,
            inplace_offset_allowed_graph_tokens_by_start=(
                inplace_offset_allowed_graph_tokens_by_start),
            pa_shape_list=pa_shape_list,
            split_debug=split_debug,
            expected_split=expected_split,
            expected_no_split_reason=expected_no_split_reason,
            profile_enabled=profile_enabled,
            profile_dir=profile_dir,
            profile_target=profile_target,
            profile_record_shapes=profile_record_shapes,
            profile_with_stack=profile_with_stack,
        )

    # For all non-coordinator paths: always create an output dir and write all artifacts.
    out_dir, timestamp = _make_run_dir(output_dir_base=output_dir_base, output_file=output_file)
    prompts_path = os.path.join(out_dir, "prompts.json")
    _save_json(prompts_path, prompts)

    metadata = {
        "timestamp": timestamp,
        "model": args.get("model"),
        "tokenizer": args.get("tokenizer") or args.get("model"),
        "seed": seed,
        "batch_size": batch_size,
        "fixed_batch_size": fixed_batch_size,
        "fixed_batch_query_len": fixed_batch_query_len,
        "fixed_batch_total_tokens": fixed_batch_total_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "force_fixed_prompts": force_fixed_prompts,
        "max_model_len": args.get("max_model_len"),
        "gpu_memory_utilization": args.get("gpu_memory_utilization"),
        "split_mode": split_mode,
        "num_splits": num_splits,
        "enable_parallel_streams": enable_parallel_streams,
        "min_batch_size_for_split": min_batch_size_for_split,
        "validate_ptrs": validate_ptrs,
        "enable_inplace_spec_decode": enable_inplace_spec_decode,
        "enable_inplace_mrope": enable_inplace_mrope,
        "inplace_split_planner_policy": inplace_split_planner_policy,
        "inplace_parallel_replay_policy": inplace_parallel_replay_policy,
        "piecewise_scheduler_sync_policy": piecewise_scheduler_sync_policy,
        "piecewise_attention_enqueue_policy": (
            piecewise_attention_enqueue_policy),
        "enable_mixed_request_split": enable_mixed_request_split,
        "mixed_request_split_execution_mode": (
            mixed_request_split_execution_mode),
        "mixed_request_min_total_tokens": mixed_request_min_total_tokens,
        "mixed_request_min_tokens_per_split": (
            mixed_request_min_tokens_per_split),
        "mixed_request_max_single_request_ratio": (
            mixed_request_max_single_request_ratio),
        "mixed_request_max_padding_ratio_per_split": (
            mixed_request_max_padding_ratio_per_split),
        "mixed_request_min_prefill_reqs_for_prefill_split": (
            mixed_request_min_prefill_reqs_for_prefill_split),
        "inplace_offset_capture_sizes": inplace_offset_capture_sizes,
        "inplace_offset_max_graph_tokens_by_start": (
            inplace_offset_max_graph_tokens_by_start),
        "inplace_offset_min_graph_tokens": inplace_offset_min_graph_tokens,
        "inplace_offset_allowed_graph_tokens_by_start": (
            inplace_offset_allowed_graph_tokens_by_start),
        "split_debug": split_debug,
        "expected_split": expected_split,
        "expected_no_split_reason": expected_no_split_reason,
        "pa_shape_list": pa_shape_list,
        "compilation_config": args.get("compilation_config"),
        "run": run_mode,
        "compare_mode": compare_mode,
        "profile_enabled": profile_enabled,
        "profile_dir": profile_dir,
        "profile_target": profile_target,
        "profile_record_shapes": profile_record_shapes,
        "profile_with_stack": profile_with_stack,
        "output_file": output_file,
        "output_dir": out_dir,
        "notes": {
            "split_mode": split_mode,
            "inproc_v1": inproc_v1,
        },
    }
    _save_json(os.path.join(out_dir, "metadata.json"), metadata)

    console_path = os.path.join(out_dir, "console.log")
    debug_path = os.path.join(out_dir, f"split_inplace_debug_{run_mode}.jsonl")
    if split_debug:
        os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG"] = "1"
        os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"] = debug_path
    os.makedirs(out_dir, exist_ok=True)
    console_f = open(console_path, "a", encoding="utf-8")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _TeeTextIO(old_stdout, console_f)
    sys.stderr = _TeeTextIO(old_stderr, console_f)

    # Re-bind vLLM logger streams after tee so diagnostics land in console.log.
    try:
        vllm_logger_module._configure_vllm_root_logger()
        for _name in ("vllm", "vllm_ascend"):
            _lg = logging.getLogger(_name)
            for _h in _lg.handlers:
                if isinstance(_h, logging.StreamHandler):
                    _h.setStream(sys.stderr)
    except Exception as _e:
        print(f"Warning: failed to rebind vLLM loggers to tee stream: {_e}")

    try:
        print("=== Output ===")
        print("Output dir:", out_dir)
        print("Console log:", console_path)

        print("\n=== Config ===")
        print(
            json.dumps(
                {
                    "model": args.get("model"),
                    "tokenizer": args.get("tokenizer") or args.get("model"),
                    "seed": seed,
                    "max_model_len": args.get("max_model_len"),
                    "gpu_memory_utilization": args.get("gpu_memory_utilization"),
                    "batch_size": batch_size,
                    "max_tokens": max_tokens,
                    "ignore_eos": ignore_eos,
                    "split_batch_config": split_enabled_cfg["split_batch_config"],
                    "pa_shape_list": pa_shape_list,
                    "compilation_config": args.get("compilation_config"),
                    "expected_split": expected_split,
                    "expected_no_split_reason": expected_no_split_reason,
                    "enable_inplace_spec_decode":
                    enable_inplace_spec_decode,
                    "enable_inplace_mrope": enable_inplace_mrope,
                    "split_debug_file": debug_path if split_debug else None,
                    "run": run_mode,
                    "compare_mode": compare_mode,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        # In-process comparison path or single-run path.
        out0 = None
        out1 = None

        outputs_disabled_path = os.path.join(out_dir, "outputs_split_disabled.json")
        outputs_enabled_path = os.path.join(out_dir, "outputs_split_enabled.json")

        if run_mode in ("both", "disabled"):
            print("\n=== Run: split disabled ===")
            out0, llm0 = _run_single(
                llm_args=args,
                prompts=prompts,
                sampling=sampling,
                additional_config=split_disabled_cfg,
                profile_enabled=profile_enabled,
                profile_dir=profile_dir,
                profile_target=profile_target,
                profile_record_shapes=profile_record_shapes,
                profile_with_stack=profile_with_stack,
                run_name="disabled",
            )

            disabled_payload = {
                "config": "split_disabled",
                "description": CONFIGS["split_disabled"]["description"],
                "split_batch_config": split_disabled_cfg["split_batch_config"],
                "prompts_file": prompts_file,
                "prompts_path": prompts_path,
                "outputs": _serialize_outputs(prompts, out0),
            }
            # Always write results; also honor --output-file if provided.
            _save_json(outputs_disabled_path, disabled_payload)
            if output_file and run_mode == "disabled":
                _save_json(str(output_file), disabled_payload)

            if run_mode == "both":
                if inproc_v1:
                    print(
                        "WARNING: VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-proc) may not support "
                        "running two engines back-to-back reliably due to NPU memory fragmentation.",
                        file=sys.stderr,
                    )
                _cleanup_llm(llm0)

        if run_mode in ("both", "enabled"):
            print("\n=== Run: split enabled ===")
            out1, llm1 = _run_single(
                llm_args=args,
                prompts=prompts,
                sampling=sampling,
                additional_config=split_enabled_cfg,
                profile_enabled=profile_enabled,
                profile_dir=profile_dir,
                profile_target=profile_target,
                profile_record_shapes=profile_record_shapes,
                profile_with_stack=profile_with_stack,
                run_name="enabled",
            )

            enabled_payload = {
                "config": "split_enabled",
                "description": CONFIGS["split_enabled"]["description"],
                "split_batch_config": split_enabled_cfg["split_batch_config"],
                "prompts_file": prompts_file,
                "prompts_path": prompts_path,
                "outputs": _serialize_outputs(prompts, out1),
            }
            # Always write results; also honor --output-file if provided.
            _save_json(outputs_enabled_path, enabled_payload)
            if output_file and run_mode == "enabled":
                _save_json(str(output_file), enabled_payload)

            if run_mode == "both":
                _cleanup_llm(llm1)

            if out0 is None:
                split_trace_summary = None
                trace_summary_path = None
                if split_debug:
                    split_trace_summary = _summarize_split_debug_trace(
                        debug_path,
                        expected_split=expected_split,
                        split_mode=split_mode,
                        expected_no_split_reason=expected_no_split_reason,
                    )
                    trace_summary_path = os.path.join(
                        out_dir, "split_trace_summary.json")
                    _save_json(trace_summary_path, split_trace_summary)
                    if split_trace_summary["failures"]:
                        print(
                            "\nFAIL: split trace validation failed. See "
                            f"{trace_summary_path}",
                            file=sys.stderr)
                        print(
                            json.dumps(split_trace_summary["failures"],
                                       ensure_ascii=False,
                                       indent=2),
                            file=sys.stderr)
                        _save_json(
                            os.path.join(out_dir, "summary.json"),
                            {
                                "status": "FAIL",
                                "reason": "split trace validation failed",
                                "trace_summary_path": trace_summary_path,
                                "trace_failures":
                                split_trace_summary["failures"],
                                "outputs_enabled_path": outputs_enabled_path,
                                "console_log": console_path,
                            },
                        )
                        _cleanup_llm(llm1)
                        return 1
                sample_ids, sample_text = _extract_first_output(out1[0])
                print("\nDONE: ran split enabled only")
                print("Sample output[0] token_ids_len=", len(sample_ids))
                print("Sample output[0] text=", repr(sample_text[:200]))
                _save_json(
                    os.path.join(out_dir, "summary.json"),
                    {
                        "status": "DONE",
                        "run": "enabled",
                        "count": len(out1),
                        "sample": {
                            "token_ids_len": len(sample_ids),
                            "text_preview": sample_text[:200],
                        },
                        "outputs_path": outputs_enabled_path,
                        "console_log": console_path,
                        "split_debug_file": debug_path if split_debug else None,
                        "split_trace_summary": split_trace_summary,
                    },
                )
                _cleanup_llm(llm1)  # ensure clean shutdown in enabled-only mode
                return 0

        if out1 is None:
            sample_ids, sample_text = _extract_first_output(out0[0])
            print("\nDONE: ran split disabled only")
            print("Sample output[0] token_ids_len=", len(sample_ids))
            print("Sample output[0] text=", repr(sample_text[:200]))
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "DONE",
                    "run": "disabled",
                    "count": len(out0),
                    "sample": {
                        "token_ids_len": len(sample_ids),
                        "text_preview": sample_text[:200],
                    },
                    "outputs_path": outputs_disabled_path,
                    "console_log": console_path,
                    "split_debug_file": debug_path if split_debug else None,
                },
            )
            _cleanup_llm(llm0)  # ensure clean shutdown in disabled-only mode
            return 0

        if len(out0) != len(out1):
            msg = f"FAIL: output length differs: {len(out0)} vs {len(out1)}"
            print(msg, file=sys.stderr)
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "FAIL",
                    "reason": "output length differs",
                    "disabled_len": len(out0),
                    "enabled_len": len(out1),
                    "outputs_disabled_path": outputs_disabled_path,
                    "outputs_enabled_path": outputs_enabled_path,
                    "console_log": console_path,
                },
            )
            return 1

        # Compare and write diff if needed.
        disabled_rows = _serialize_outputs(prompts, out0)
        enabled_rows = _serialize_outputs(prompts, out1)
        split_trace_summary = None
        trace_summary_path = None
        if split_debug:
            split_trace_summary = _summarize_split_debug_trace(
                debug_path,
                expected_split=expected_split,
                split_mode=split_mode,
                expected_no_split_reason=expected_no_split_reason,
            )
            trace_summary_path = os.path.join(out_dir,
                                              "split_trace_summary.json")
            _save_json(trace_summary_path, split_trace_summary)

        mismatches = _compare_serialized(disabled_rows, enabled_rows)

        if mismatches:
            diff_path = os.path.join(out_dir, "diff.json")
            _save_json(diff_path, {"mismatches": mismatches[:50], "total": len(mismatches)})
            print(f"\nFAIL: {len(mismatches)}/{len(prompts)} mismatches", file=sys.stderr)
            print(json.dumps(mismatches[:3], ensure_ascii=False, indent=2), file=sys.stderr)
            if len(mismatches) > 3:
                print(f"... (and {len(mismatches) - 3} more)", file=sys.stderr)

            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "FAIL",
                    "mismatch_total": len(mismatches),
                    "diff_path": diff_path,
                    "trace_summary_path": trace_summary_path,
                    "split_trace_summary": split_trace_summary,
                    "outputs_disabled_path": outputs_disabled_path,
                    "outputs_enabled_path": outputs_enabled_path,
                    "console_log": console_path,
                },
            )
            return 1

        if split_trace_summary is not None:
            if split_trace_summary["failures"]:
                print(
                    "\nFAIL: split trace validation failed. See "
                    f"{trace_summary_path}",
                    file=sys.stderr)
                print(
                    json.dumps(split_trace_summary["failures"],
                               ensure_ascii=False,
                               indent=2),
                    file=sys.stderr)
                _save_json(
                    os.path.join(out_dir, "summary.json"),
                    {
                        "status": "FAIL",
                        "reason": "split trace validation failed",
                        "trace_summary_path": trace_summary_path,
                        "trace_failures": split_trace_summary["failures"],
                        "outputs_disabled_path": outputs_disabled_path,
                        "outputs_enabled_path": outputs_enabled_path,
                        "console_log": console_path,
                    },
                )
                return 1

        print(f"\nPASS: {len(prompts)}/{len(prompts)} outputs match exactly")
        sample_ids, sample_text = _extract_first_output(out1[0])
        print("Sample output[0] token_ids_len=", len(sample_ids))
        print("Sample output[0] text=", repr(sample_text[:200]))

        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "PASS",
                "count": len(prompts),
                "sample": {
                    "token_ids_len": len(sample_ids),
                    "text_preview": sample_text[:200],
                },
                "outputs_disabled_path": outputs_disabled_path,
                "outputs_enabled_path": outputs_enabled_path,
                "console_log": console_path,
                "split_trace_summary": split_trace_summary,
            },
        )
        return 0

    except Exception as e:
        try:
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "ERROR",
                    "error": repr(e),
                    "console_log": console_path,
                },
            )
        except Exception:
            pass
        raise
    finally:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        finally:
            try:
                console_f.flush()
                console_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
