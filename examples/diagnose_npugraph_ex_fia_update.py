#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Diagnose FIA graph-task update semantics on explicit NPUGraph vs npugraph_ex.

This is intentionally independent from vLLM model execution.  It isolates the
Phase 3 question: does a graph containing FIA correctly wait on an
ExternalEvent and then run with updated FIA parameters?
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from typing import Any

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


_CUSTOM_FIA_RECORDS: list[dict[str, Any]] = []
_TND_FIA_CONFIGS: dict[int, dict[str, Any]] = {}


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": tuple(int(dim) for dim in tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "ptr": int(tensor.data_ptr()),
    }


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _fia(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    actual_seq_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=actual_seq_lengths,
    )


def _fia_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    actual_seq_lengths: list[int],
) -> None:
    torch_npu.npu_fused_infer_attention_score.out(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=actual_seq_lengths,
        workspace=workspace,
        out=[output, softmax_lse],
    )


def _fia_tnd_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: torch.Tensor,
    *,
    block_size: int,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    num_kv_heads: int,
    num_heads: int,
    scale: float,
) -> None:
    torch_npu.npu_fused_infer_attention_score.out(
        query=query,
        key=key,
        value=value,
        atten_mask=attn_mask,
        block_table=block_table,
        input_layout="TND",
        block_size=int(block_size),
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=int(num_kv_heads),
        num_heads=int(num_heads),
        scale=float(scale),
        sparse_mode=3,
        workspace=workspace,
        out=[output, softmax_lse],
    )


def _make_inputs(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(f"npu:{int(args.device)}")
    dtype = _dtype(args.dtype)
    torch.manual_seed(int(args.seed))
    query = torch.randn(
        1,
        int(args.num_heads),
        int(args.query_len),
        int(args.head_dim),
        dtype=dtype,
        device=device,
    )
    key = torch.randn(
        1,
        int(args.num_heads),
        int(args.kv_len),
        int(args.head_dim),
        dtype=dtype,
        device=device,
    )
    value = torch.randn_like(key)
    scale = 1.0 / math.sqrt(float(args.head_dim))
    return {
        "query": query,
        "key": key,
        "value": value,
        "num_heads": int(args.num_heads),
        "scale": scale,
    }


def _make_workspace(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    actual_seq_lengths: list[int],
) -> torch.Tensor:
    return torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=actual_seq_lengths,
    )


def _make_tnd_workspace(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    *,
    block_size: int,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    num_kv_heads: int,
    num_heads: int,
    scale: float,
) -> torch.Tensor:
    return torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query=query,
        key=key,
        value=value,
        atten_mask=attn_mask,
        block_table=block_table,
        input_layout="TND",
        block_size=int(block_size),
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=int(num_kv_heads),
        num_heads=int(num_heads),
        sparse_mode=3,
        scale=float(scale),
    )


@torch.library.custom_op(
    "vllm_diag::manual_fia_out",
    mutates_args=("output", "softmax_lse"),
    device_types="npu",
)
def _manual_fia_out_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: torch.Tensor,
    actual_seq_len: int,
    num_heads: int,
    scale: float,
) -> None:
    stream = torch.npu.current_stream()
    capturing = bool(torch.npu.is_current_stream_capturing())
    event = None
    if capturing:
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)

    _fia_out(
        query,
        key,
        value,
        output,
        softmax_lse,
        workspace,
        num_heads=int(num_heads),
        scale=float(scale),
        actual_seq_lengths=[int(actual_seq_len)],
    )

    if capturing:
        handle = torch.npu.graph_task_group_end(stream)
        _CUSTOM_FIA_RECORDS.append({
            "event": event,
            "handle": handle,
            "query": query,
            "key": key,
            "value": value,
            "output": output,
            "softmax_lse": softmax_lse,
            "workspace": workspace,
            "capture_len": int(actual_seq_len),
            "num_heads": int(num_heads),
            "scale": float(scale),
            "stream_id": getattr(stream, "stream_id", None),
        })


@torch.library.register_fake("vllm_diag::manual_fia_out")
def _manual_fia_out_custom_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: torch.Tensor,
    actual_seq_len: int,
    num_heads: int,
    scale: float,
) -> None:
    del (query, key, value, output, softmax_lse, workspace, actual_seq_len,
         num_heads, scale)


@torch.library.custom_op(
    "vllm_diag::manual_tnd_fia_out",
    mutates_args=("output", "softmax_lse"),
    device_types="npu",
)
def _manual_tnd_fia_out_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: torch.Tensor,
    config_id: int,
    split_idx: int,
    layer_idx: int,
) -> None:
    config = _TND_FIA_CONFIGS[int(config_id)]
    split_config = config["splits"][int(split_idx)]
    stream = torch.npu.current_stream()
    capturing = bool(torch.npu.is_current_stream_capturing())
    event = None
    if capturing:
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)

    _fia_tnd_out(
        query,
        key,
        value,
        block_table,
        attn_mask,
        output,
        softmax_lse,
        workspace,
        block_size=int(config["block_size"]),
        actual_seq_lengths_q=split_config["actual_seq_lengths_q"],
        actual_seq_lengths_kv=split_config["actual_seq_lengths_kv"],
        num_kv_heads=int(config["num_kv_heads"]),
        num_heads=int(config["num_heads"]),
        scale=float(config["scale"]),
    )

    if capturing:
        handle = torch.npu.graph_task_group_end(stream)
        _CUSTOM_FIA_RECORDS.append({
            "event": event,
            "handle": handle,
            "query": query,
            "key": key,
            "value": value,
            "block_table": block_table,
            "attn_mask": attn_mask,
            "output": output,
            "softmax_lse": softmax_lse,
            "workspace": workspace,
            "config_id": int(config_id),
            "split_idx": int(split_idx),
            "layer_idx": int(layer_idx),
            "stream_id": getattr(stream, "stream_id", None),
        })


@torch.library.register_fake("vllm_diag::manual_tnd_fia_out")
def _manual_tnd_fia_out_custom_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: torch.Tensor,
    config_id: int,
    split_idx: int,
    layer_idx: int,
) -> None:
    del (query, key, value, block_table, attn_mask, output, softmax_lse,
         workspace, config_id, split_idx, layer_idx)


def run_explicit_npugraph(args: argparse.Namespace) -> dict[str, Any]:
    params = _make_inputs(args)
    query = params["query"]
    key = params["key"]
    value = params["value"]
    num_heads = params["num_heads"]
    scale = params["scale"]
    capture_len = [int(args.length)]
    update_len = [int(args.length_new)]

    ref_capture = _fia(query,
                       key,
                       value,
                       num_heads=num_heads,
                       scale=scale,
                       actual_seq_lengths=capture_len)
    ref_update = _fia(query,
                      key,
                      value,
                      num_heads=num_heads,
                      scale=scale,
                      actual_seq_lengths=update_len)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    event = torch.npu.ExternalEvent()
    update_stream = torch.npu.Stream()
    output = torch.empty_like(ref_capture[0])
    softmax_lse = torch.empty_like(ref_capture[1])
    workspace = _make_workspace(query,
                                key,
                                value,
                                num_heads=num_heads,
                                scale=scale,
                                actual_seq_lengths=capture_len)

    with torch.npu.graph(graph):
        stream = torch.npu.current_stream()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        _fia_out(query,
                 key,
                 value,
                 output,
                 softmax_lse,
                 workspace,
                 num_heads=num_heads,
                 scale=scale,
                 actual_seq_lengths=capture_len)
        handle = torch.npu.graph_task_group_end(stream)

    def record_update_or_event_only() -> None:
        with torch.npu.stream(update_stream):
            if args.update_mode == "task_update":
                torch.npu.graph_task_update_begin(update_stream, handle)
                _fia_out(query,
                         key,
                         value,
                         output,
                         softmax_lse,
                         workspace,
                         num_heads=num_heads,
                         scale=scale,
                         actual_seq_lengths=update_len)
                torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

    start = time.perf_counter()
    if args.order == "update-before-replay":
        record_update_or_event_only()
        graph.replay()
    else:
        graph.replay()
        record_update_or_event_only()
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    expected = ref_update if args.update_mode == "task_update" else ref_capture
    expected_lse = ref_update[1] if args.update_mode == "task_update" else ref_capture[1]
    max_abs_diff = _max_abs_diff(output, expected[0])
    max_lse_abs_diff = _max_abs_diff(softmax_lse, expected_lse)
    success = (max_abs_diff <= float(args.atol)
               and max_lse_abs_diff <= float(args.atol))
    return {
        "case": "explicit_npugraph",
        "success": bool(success),
        "update_mode": str(args.update_mode),
        "order": str(args.order),
        "capture_len": capture_len,
        "update_len": update_len,
        "elapsed_ms": elapsed_ms,
        "max_abs_diff": max_abs_diff,
        "max_lse_abs_diff": max_lse_abs_diff,
        "atol": float(args.atol),
        "query": _tensor_summary(query),
        "key": _tensor_summary(key),
        "output": _tensor_summary(output),
    }


class FiaModule(nn.Module):

    def __init__(self, num_heads: int, scale: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.scale = float(scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, actual_seq_len: int) -> tuple[torch.Tensor,
                                                                   torch.Tensor]:
        return torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale=self.scale,
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False,
            actual_seq_lengths=[actual_seq_len],
        )


def run_npugraph_ex_fia(args: argparse.Namespace) -> dict[str, Any]:
    params = _make_inputs(args)
    query = params["query"]
    key = params["key"]
    value = params["value"]
    num_heads = params["num_heads"]
    scale = params["scale"]
    capture_len = int(args.length)
    update_len = int(args.length_new)

    ref_update = _fia(query,
                      key,
                      value,
                      num_heads=num_heads,
                      scale=scale,
                      actual_seq_lengths=[update_len])
    torch.npu.synchronize()

    module = FiaModule(num_heads, scale).npu().eval()
    backend_options = {
        "clone_input": False,
        "clone_output": bool(args.clone_output),
    }
    compiled = torch.compile(module,
                             backend="npugraph_ex",
                             dynamic=False,
                             fullgraph=True,
                             options=backend_options)

    start = time.perf_counter()
    first = compiled(query, key, value, capture_len)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    output, softmax_lse = compiled(query, key, value, update_len)
    torch.npu.synchronize()
    second_ms = (time.perf_counter() - start) * 1000.0

    max_abs_diff = _max_abs_diff(output, ref_update[0])
    max_lse_abs_diff = _max_abs_diff(softmax_lse, ref_update[1])
    success = (max_abs_diff <= float(args.atol)
               and max_lse_abs_diff <= float(args.atol))
    return {
        "case": "npugraph_ex_fia",
        "success": bool(success),
        "backend_options": backend_options,
        "capture_len": capture_len,
        "update_len": update_len,
        "first_call_ms": first_ms,
        "second_call_ms": second_ms,
        "max_abs_diff": max_abs_diff,
        "max_lse_abs_diff": max_lse_abs_diff,
        "atol": float(args.atol),
        "first_output": _tensor_summary(first[0]),
        "second_output": _tensor_summary(output),
    }


class ManualCustomFiaModule(nn.Module):

    def __init__(self, num_heads: int, scale: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.scale = float(scale)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
        workspace: torch.Tensor,
        actual_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.ops.vllm_diag.manual_fia_out(
            query,
            key,
            value,
            output,
            softmax_lse,
            workspace,
            actual_seq_len,
            self.num_heads,
            self.scale,
        )
        return output, softmax_lse


def _record_custom_update(record: dict[str, Any],
                          update_len: int,
                          update_mode: str) -> dict[str, Any]:
    event = record["event"]
    handle = record["handle"]
    update_stream = torch.npu.Stream()
    detail = {
        "update_mode": str(update_mode),
        "update_len": int(update_len),
        "update_stream_id": getattr(update_stream, "stream_id", None),
        "capture_stream_id": record.get("stream_id"),
        "event_repr": repr(event),
        "handle_repr": repr(handle),
    }
    start = time.perf_counter()
    with torch.npu.stream(update_stream):
        if update_mode == "task_update":
            torch.npu.graph_task_update_begin(update_stream, handle)
            _fia_out(
                record["query"],
                record["key"],
                record["value"],
                record["output"],
                record["softmax_lse"],
                record["workspace"],
                num_heads=int(record["num_heads"]),
                scale=float(record["scale"]),
                actual_seq_lengths=[int(update_len)],
            )
            torch.npu.graph_task_update_end(update_stream)
        event.record(update_stream)
    detail["record_ms"] = (time.perf_counter() - start) * 1000.0
    return detail


def run_npugraph_ex_custom_manual_fia(args: argparse.Namespace) -> dict[str,
                                                                        Any]:
    _CUSTOM_FIA_RECORDS.clear()
    params = _make_inputs(args)
    query = params["query"]
    key = params["key"]
    value = params["value"]
    num_heads = params["num_heads"]
    scale = params["scale"]
    capture_len = int(args.length)
    update_len = int(args.length_new)

    ref_capture = _fia(query,
                       key,
                       value,
                       num_heads=num_heads,
                       scale=scale,
                       actual_seq_lengths=[capture_len])
    ref_update = _fia(query,
                      key,
                      value,
                      num_heads=num_heads,
                      scale=scale,
                      actual_seq_lengths=[update_len])
    torch.npu.synchronize()

    output = torch.empty_like(ref_capture[0])
    softmax_lse = torch.empty_like(ref_capture[1])
    workspace = _make_workspace(query,
                                key,
                                value,
                                num_heads=num_heads,
                                scale=scale,
                                actual_seq_lengths=[capture_len])
    module = ManualCustomFiaModule(num_heads, scale).npu().eval()
    backend_options = {
        "clone_input": False,
        "clone_output": bool(args.clone_output),
        "return_captured_outputs_on_first_run": True,
    }
    compiled = torch.compile(module,
                             backend="npugraph_ex",
                             dynamic=False,
                             fullgraph=True,
                             options=backend_options)

    start = time.perf_counter()
    first = compiled(query, key, value, output, softmax_lse, workspace,
                     capture_len)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - start) * 1000.0
    records_after_first = len(_CUSTOM_FIA_RECORDS)
    if records_after_first < 1:
        return {
            "case": "npugraph_ex_custom_manual_fia",
            "success": False,
            "reason": "no_custom_fia_capture_record",
            "backend_options": backend_options,
            "records_after_first": int(records_after_first),
        }

    start = time.perf_counter()
    second = compiled(query, key, value, output, softmax_lse, workspace,
                      capture_len)
    update_detail = _record_custom_update(_CUSTOM_FIA_RECORDS[-1], update_len,
                                          str(args.update_mode))
    torch.npu.synchronize()
    second_ms = (time.perf_counter() - start) * 1000.0

    observed_output = second[0]
    observed_lse = second[1]
    expected = ref_update if args.update_mode == "task_update" else ref_capture
    expected_lse = ref_update[1] if args.update_mode == "task_update" else ref_capture[1]
    max_abs_diff = _max_abs_diff(observed_output, expected[0])
    max_lse_abs_diff = _max_abs_diff(observed_lse, expected_lse)
    success = (max_abs_diff <= float(args.atol)
               and max_lse_abs_diff <= float(args.atol))
    return {
        "case": "npugraph_ex_custom_manual_fia",
        "success": bool(success),
        "backend_options": backend_options,
        "update_mode": str(args.update_mode),
        "capture_len": capture_len,
        "update_len": update_len,
        "records_after_first": int(records_after_first),
        "records_after_second": int(len(_CUSTOM_FIA_RECORDS)),
        "first_call_ms": first_ms,
        "second_call_with_update_ms": second_ms,
        "update_detail": update_detail,
        "max_abs_diff": max_abs_diff,
        "max_lse_abs_diff": max_lse_abs_diff,
        "atol": float(args.atol),
        "first_output": _tensor_summary(first[0]),
        "second_output": _tensor_summary(observed_output),
    }


class DualStreamTndFiaModule(nn.Module):

    def __init__(self, config_id: int, layer_count: int) -> None:
        super().__init__()
        self.config_id = int(config_id)
        self.layer_count = int(layer_count)

    def _run_split(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        attn_mask: torch.Tensor,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
        workspace: torch.Tensor,
        split_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer_idx in range(self.layer_count):
            torch.ops.vllm_diag.manual_tnd_fia_out(
                query,
                key,
                value,
                block_table,
                attn_mask,
                output,
                softmax_lse,
                workspace,
                self.config_id,
                int(split_idx),
                int(layer_idx),
            )
        return output, softmax_lse

    def forward(
        self,
        q0: torch.Tensor,
        q1: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bt0: torch.Tensor,
        bt1: torch.Tensor,
        attn_mask: torch.Tensor,
        out0: torch.Tensor,
        out1: torch.Tensor,
        lse0: torch.Tensor,
        lse1: torch.Tensor,
        workspace0: torch.Tensor,
        workspace1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        secondary = torch.npu.Stream()
        fork = torch.npu.Event()
        join = torch.npu.Event()
        fork.record()

        with torch.npu.stream(secondary):
            fork.wait(secondary)
            self._run_split(q1, key, value, bt1, attn_mask, out1, lse1,
                            workspace1, 1)
            join.record()

        self._run_split(q0, key, value, bt0, attn_mask, out0, lse0, workspace0,
                        0)
        join.wait(torch.npu.current_stream())
        return out0, out1


def _make_tnd_dual_inputs(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(f"npu:{int(args.device)}")
    dtype = _dtype(args.dtype)
    torch.manual_seed(int(args.seed))
    num_heads = int(args.num_heads)
    num_kv_heads = int(args.num_kv_heads)
    head_dim = int(args.head_dim)
    block_size = int(args.block_size)
    kv_hidden = num_kv_heads * head_dim
    split0_tokens = int(args.split0_tokens)
    split1_tokens = int(args.split1_tokens)
    split0_reqs = int(args.split0_reqs)
    split1_reqs = int(args.split1_reqs)
    max_kv_len = int(args.kv_len)
    max_blocks_per_req = max(1, math.ceil(max_kv_len / block_size))
    num_blocks = max(8, split0_reqs + split1_reqs + 4)
    scale = 1.0 / math.sqrt(float(head_dim))

    q0 = torch.randn(split0_tokens,
                     num_heads,
                     head_dim,
                     dtype=dtype,
                     device=device)
    q1 = torch.randn(split1_tokens,
                     num_heads,
                     head_dim,
                     dtype=dtype,
                     device=device)
    key = torch.randn(num_blocks, block_size, kv_hidden, dtype=dtype, device=device)
    value = torch.randn_like(key)
    bt0 = torch.arange(split0_reqs * max_blocks_per_req,
                       dtype=torch.int32,
                       device=device).view(split0_reqs, max_blocks_per_req)
    bt1 = torch.arange(split1_reqs * max_blocks_per_req,
                       dtype=torch.int32,
                       device=device).view(split1_reqs, max_blocks_per_req)
    bt1 = bt1 + split0_reqs * max_blocks_per_req
    attn_mask = torch.zeros(2048, 2048, dtype=torch.int8, device=device)

    def q_lens(num_tokens: int, num_reqs: int) -> list[int]:
        if num_reqs <= 1:
            return [int(num_tokens)]
        # Match the failing compact mixed-request pattern: several decode
        # tokens followed by one compact prefill tail.
        prefix = list(range(1, num_reqs))
        prefix_sum = prefix[-1] if prefix else 0
        return prefix + [int(num_tokens)]

    def kv_lens(num_tokens: int, num_reqs: int) -> list[int]:
        if num_reqs <= 1:
            return [max(1, min(max_kv_len, num_tokens))]
        return [max(1, min(max_kv_len, max_kv_len - 1))
                for _ in range(num_reqs - 1)] + [
                    max(1, min(max_kv_len, num_tokens - (num_reqs - 1)))
                ]

    split0_q_lens = q_lens(split0_tokens, split0_reqs)
    split1_q_lens = q_lens(split1_tokens, split1_reqs)
    split0_kv_lens = kv_lens(split0_tokens, split0_reqs)
    split1_kv_lens = kv_lens(split1_tokens, split1_reqs)

    config_id = id(args) & 0x7fffffff
    _TND_FIA_CONFIGS[config_id] = {
        "block_size": block_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "scale": scale,
        "splits": {
            0: {
                "actual_seq_lengths_q": split0_q_lens,
                "actual_seq_lengths_kv": split0_kv_lens,
            },
            1: {
                "actual_seq_lengths_q": split1_q_lens,
                "actual_seq_lengths_kv": split1_kv_lens,
            },
        },
    }

    out0 = torch.empty_like(q0)
    out1 = torch.empty_like(q1)
    lse0 = torch.empty(1, dtype=dtype, device=device)
    lse1 = torch.empty(1, dtype=dtype, device=device)
    workspace0 = _make_tnd_workspace(
        q0,
        key,
        value,
        bt0,
        attn_mask,
        block_size=block_size,
        actual_seq_lengths_q=split0_q_lens,
        actual_seq_lengths_kv=split0_kv_lens,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
    )
    workspace1 = _make_tnd_workspace(
        q1,
        key,
        value,
        bt1,
        attn_mask,
        block_size=block_size,
        actual_seq_lengths_q=split1_q_lens,
        actual_seq_lengths_kv=split1_kv_lens,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
    )
    return {
        "config_id": config_id,
        "q0": q0,
        "q1": q1,
        "key": key,
        "value": value,
        "bt0": bt0,
        "bt1": bt1,
        "attn_mask": attn_mask,
        "out0": out0,
        "out1": out1,
        "lse0": lse0,
        "lse1": lse1,
        "workspace0": workspace0,
        "workspace1": workspace1,
        "config": _TND_FIA_CONFIGS[config_id],
    }


def _record_tnd_updates(update_mode: str) -> list[dict[str, Any]]:
    details = []
    update_stream = torch.npu.Stream()
    with torch.npu.stream(update_stream):
        for record in _CUSTOM_FIA_RECORDS:
            config = _TND_FIA_CONFIGS[int(record["config_id"])]
            split_config = config["splits"][int(record["split_idx"])]
            start = time.perf_counter()
            if update_mode == "task_update":
                torch.npu.graph_task_update_begin(update_stream, record["handle"])
                _fia_tnd_out(
                    record["query"],
                    record["key"],
                    record["value"],
                    record["block_table"],
                    record["attn_mask"],
                    record["output"],
                    record["softmax_lse"],
                    record["workspace"],
                    block_size=int(config["block_size"]),
                    actual_seq_lengths_q=split_config["actual_seq_lengths_q"],
                    actual_seq_lengths_kv=split_config["actual_seq_lengths_kv"],
                    num_kv_heads=int(config["num_kv_heads"]),
                    num_heads=int(config["num_heads"]),
                    scale=float(config["scale"]),
                )
                torch.npu.graph_task_update_end(update_stream)
            record["event"].record(update_stream)
            details.append({
                "split_idx": int(record["split_idx"]),
                "layer_idx": int(record["layer_idx"]),
                "capture_stream_id": record.get("stream_id"),
                "update_stream_id": getattr(update_stream, "stream_id", None),
                "record_ms": (time.perf_counter() - start) * 1000.0,
            })
    return details


def run_npugraph_ex_dual_tnd_fia(args: argparse.Namespace) -> dict[str, Any]:
    _CUSTOM_FIA_RECORDS.clear()
    inputs = _make_tnd_dual_inputs(args)
    module = DualStreamTndFiaModule(inputs["config_id"],
                                    int(args.layer_count)).npu().eval()
    backend_options = {
        "clone_input": False,
        "clone_output": bool(args.clone_output),
        "return_captured_outputs_on_first_run": True,
        "deadlock_check": False,
    }
    compiled = torch.compile(module,
                             backend="npugraph_ex",
                             dynamic=False,
                             fullgraph=True,
                             options=backend_options)

    call_args = (
        inputs["q0"],
        inputs["q1"],
        inputs["key"],
        inputs["value"],
        inputs["bt0"],
        inputs["bt1"],
        inputs["attn_mask"],
        inputs["out0"],
        inputs["out1"],
        inputs["lse0"],
        inputs["lse1"],
        inputs["workspace0"],
        inputs["workspace1"],
    )

    start = time.perf_counter()
    first = compiled(*call_args)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - start) * 1000.0
    records_after_first = len(_CUSTOM_FIA_RECORDS)
    expected_records = int(args.layer_count) * 2

    start = time.perf_counter()
    second = compiled(*call_args)
    update_details = _record_tnd_updates(str(args.update_mode))
    torch.npu.synchronize()
    second_ms = (time.perf_counter() - start) * 1000.0

    return {
        "case": "npugraph_ex_dual_tnd_fia",
        "success": True,
        "backend_options": backend_options,
        "update_mode": str(args.update_mode),
        "layer_count": int(args.layer_count),
        "expected_records": expected_records,
        "records_after_first": int(records_after_first),
        "records_after_second": int(len(_CUSTOM_FIA_RECORDS)),
        "first_call_ms": first_ms,
        "second_call_with_update_ms": second_ms,
        "update_detail_count": len(update_details),
        "update_detail_sample": update_details[:4] + update_details[-4:],
        "split0_output": _tensor_summary(second[0]),
        "split1_output": _tensor_summary(second[1]),
        "workspace0": _tensor_summary(inputs["workspace0"]),
        "workspace1": _tensor_summary(inputs["workspace1"]),
        "config": inputs["config"],
        "first_output": [_tensor_summary(first[0]), _tensor_summary(first[1])],
    }


def run_case(args: argparse.Namespace, case: str) -> dict[str, Any]:
    try:
        if case == "explicit":
            return run_explicit_npugraph(args)
        if case == "npugraph_ex":
            return run_npugraph_ex_fia(args)
        if case == "custom_npugraph_ex":
            return run_npugraph_ex_custom_manual_fia(args)
        if case == "dual_tnd_npugraph_ex":
            return run_npugraph_ex_dual_tnd_fia(args)
        raise ValueError(f"Unsupported case: {case}")
    except Exception as exc:  # noqa: BLE001 - diagnostics should report all failures.
        return {
            "case": case,
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose FIA graph update behavior.")
    parser.add_argument("--case",
                        choices=[
                            "explicit", "npugraph_ex", "custom_npugraph_ex",
                            "dual_tnd_npugraph_ex", "all"
                        ],
                        default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype",
                        choices=["float16", "bfloat16"],
                        default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--split0-tokens", type=int, default=36)
    parser.add_argument("--split1-tokens", type=int, default=32)
    parser.add_argument("--split0-reqs", type=int, default=5)
    parser.add_argument("--split1-reqs", type=int, default=1)
    parser.add_argument("--layer-count", type=int, default=28)
    parser.add_argument("--length", type=int, default=29)
    parser.add_argument("--length-new", type=int, default=100)
    parser.add_argument("--update-mode",
                        choices=["task_update", "event_only"],
                        default="task_update")
    parser.add_argument("--order",
                        choices=["replay-before-update", "update-before-replay"],
                        default="replay-before-update")
    parser.add_argument("--clone-output",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.npu.is_available():
        print(_json({"success": False, "reason": "npu_not_available"}))
        return 2

    torch.npu.set_device(int(args.device))
    cases = ([
        "explicit", "npugraph_ex", "custom_npugraph_ex",
        "dual_tnd_npugraph_ex"
    ]
             if args.case == "all" else [args.case])
    results = [run_case(args, case) for case in cases]
    payload = {
        "success": all(bool(result.get("success")) for result in results),
        "device": int(args.device),
        "torch": str(torch.__version__),
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "results": results,
    }
    print(_json(payload))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
