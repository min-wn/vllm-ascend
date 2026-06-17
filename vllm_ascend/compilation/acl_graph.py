# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import patch

import numpy as np
import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation import monitor as compilation_monitor
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend import inplace_split_debug as split_debug

from ..utils import weak_ref_tensors
from vllm.distributed.device_communicators.pynccl_allocator import \
    set_graph_pool_id

GraphParamKey = int | BatchDescriptor


def _format_dual_stream_attention_key_part(value: Any) -> str:
    try:
        return ",".join(str(int(item)) for item in value)
    except (TypeError, ValueError):
        return ""


def _dual_stream_attention_graph_param_key(
        forward_context: Any, runtime_shape: int,
        desc: BatchDescriptor) -> Optional[BatchDescriptor]:
    dual_metadata = getattr(forward_context,
                            "dual_stream_attention_metadata", None)
    if not isinstance(dual_metadata, list) or len(dual_metadata) != 2:
        return None
    plan = getattr(forward_context, "dual_stream_attention_plan", None)
    if plan is None:
        return None

    graph_tokens = int(getattr(plan, "graph_tokens", runtime_shape)
                       or runtime_shape)
    total_tokens = int(getattr(plan, "total_tokens", graph_tokens)
                       or graph_tokens)
    split_actual = _format_dual_stream_attention_key_part(
        getattr(plan, "split_actual_tokens", ()))
    split_graph = _format_dual_stream_attention_key_part(
        getattr(plan, "split_graph_tokens", ()))
    split_start = _format_dual_stream_attention_key_part(
        getattr(plan, "split_start_tokens", ()))
    metadata_mode = (
        f"total={total_tokens};actual={split_actual};"
        f"graph={split_graph};start={split_start};full={graph_tokens}")
    return BatchDescriptor(
        num_tokens=graph_tokens,
        num_reqs=desc.num_reqs,
        uniform=desc.uniform,
        has_lora=desc.has_lora,
        start_num_tokens=0,
        graph_variant="dual_stream_attention",
        attention_backend="fia",
        capture_metadata_mode=metadata_mode,
    )


def get_graph_param_key(forward_context: Any,
                        runtime_shape: int,
                        *,
                        allow_mtp_offset: bool = False) -> GraphParamKey:
    if getattr(forward_context, "is_mtp_model", False) and not allow_mtp_offset:
        return runtime_shape

    desc = getattr(forward_context, "batch_descriptor", None)
    if not isinstance(desc, BatchDescriptor):
        return runtime_shape
    dual_stream_key = _dual_stream_attention_graph_param_key(
        forward_context, runtime_shape, desc)
    if dual_stream_key is not None:
        return dual_stream_key
    start = int(getattr(desc, "start_num_tokens", 0) or 0)
    has_descriptor_variant = (
        start > 0
        or bool(getattr(desc, "graph_variant", ""))
        or bool(getattr(desc, "attention_backend", ""))
        or bool(getattr(desc, "capture_metadata_mode", ""))
    )
    if has_descriptor_variant:
        return desc
    return runtime_shape


def graph_param_key_info(key: GraphParamKey) -> dict[str, Any]:
    if isinstance(key, BatchDescriptor):
        return {
            "kind": "batch_descriptor",
            "num_tokens": key.num_tokens,
            "num_reqs": key.num_reqs,
            "uniform": key.uniform,
            "has_lora": key.has_lora,
            "start_num_tokens": key.start_num_tokens,
            "graph_variant": getattr(key, "graph_variant", ""),
            "attention_backend": getattr(key, "attention_backend", ""),
            "capture_metadata_mode": getattr(key, "capture_metadata_mode", ""),
        }
    return {
        "kind": "runtime_shape",
        "num_tokens": int(key),
    }


def should_template_fia_seq_lens(forward_context: Any) -> bool:
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    return (getattr(batch_descriptor, "capture_metadata_mode", "") == "template"
            and getattr(batch_descriptor, "attention_backend", "") == "fia")


def _get_fia_key_t(key_tensor: Any, fallback: int) -> int:
    if isinstance(key_tensor, torch.Tensor):
        if key_tensor.ndim > 1:
            return int(key_tensor.shape[1])
        if key_tensor.ndim > 0 and int(fallback) <= 0:
            return int(key_tensor.shape[0])
    return int(fallback)


def _seq_lens_tail(seq_lens: Any) -> Any:
    if seq_lens is None or isinstance(seq_lens, torch.Tensor):
        return None
    try:
        if len(seq_lens) == 0:
            return None
        return int(seq_lens[-1])
    except (TypeError, ValueError):
        return None


def maybe_template_fia_seq_lens(forward_context: Any, seq_lens: Any,
                                target_t: int, *, source: str = "") -> Any:
    if not should_template_fia_seq_lens(forward_context):
        return seq_lens
    if seq_lens is None or isinstance(seq_lens, torch.Tensor):
        if split_debug.is_enabled():
            split_debug.log_event(
                "fia_seq_lens_template",
                {
                    "source": source,
                    "applied": False,
                    "reason": "none_or_tensor",
                    "batch_descriptor": split_debug.batch_descriptor_info(
                        getattr(forward_context, "batch_descriptor", None)),
                    "ubatch_num": getattr(forward_context, "ubatch_num", None),
                    "in_parallel_streams": bool(
                        getattr(forward_context, "in_parallel_streams",
                                False)),
                    "target_t": int(target_t),
                    "seq_lens_type": type(seq_lens).__name__,
                },
                step_id=getattr(forward_context,
                                "split_inplace_debug_step_id", None),
            )
        return seq_lens
    try:
        if len(seq_lens) == 0:
            return seq_lens
    except TypeError:
        return seq_lens

    original_seq_lens = list(seq_lens)
    tail_before = _seq_lens_tail(original_seq_lens)
    templated_seq_lens = list(original_seq_lens)
    # FIA TND requires actualSeqenceLengthKV[-1] to match key/value T.
    templated_seq_lens[-1] = int(target_t)
    if split_debug.is_enabled():
        split_debug.log_event(
            "fia_seq_lens_template",
            {
                "source": source,
                "applied": True,
                "batch_descriptor": split_debug.batch_descriptor_info(
                    getattr(forward_context, "batch_descriptor", None)),
                "ubatch_num": getattr(forward_context, "ubatch_num", None),
                "in_parallel_streams": bool(
                    getattr(forward_context, "in_parallel_streams", False)),
                "target_t": int(target_t),
                "tail_before": tail_before,
                "tail_after": _seq_lens_tail(templated_seq_lens),
                "seq_lens_len": len(templated_seq_lens),
            },
            step_id=getattr(forward_context, "split_inplace_debug_step_id",
                            None),
        )
    return templated_seq_lens


def _get_attention_update_metadata(forward_context: Any, key: Any) -> Any:
    """Return runtime metadata for graph_task_update.

    Macro graphs keep the captured context object stable for no-arg replay, but
    attention graph_task_update should read the current step's Python metadata
    when it is provided by the runner.
    """
    runtime_metadata = getattr(forward_context,
                               "macro_graph_attention_update_metadata", None)
    if runtime_metadata is not None:
        try:
            return runtime_metadata[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(
                "macro_graph_attention_update_metadata is missing layer "
                f"{key!r}") from exc
    return forward_context.attn_metadata[key]


def _get_dual_attention_update_metadata(forward_context: Any,
                                        dual_metadata: Any, split_idx: int,
                                        key: Any) -> Any:
    runtime_metadata = getattr(
        forward_context, "macro_graph_dual_attention_update_metadata", None)
    if runtime_metadata is not None:
        try:
            return runtime_metadata[split_idx][key]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(
                "macro_graph_dual_attention_update_metadata is missing "
                f"split={split_idx}, layer={key!r}") from exc
    return dual_metadata[split_idx][key]


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None
    capture_count: int = 0
    replay_count: int = 0
    fallback_eager_count: int = 0

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None
    input_tensor_infos: Optional[list[dict[str, Any]]] = None
    attn_metadata_addresses: Optional[list[int]] = None
    attn_metadata_tensor_infos: Optional[list[dict[str, Any]]] = None


_ACL_GRAPH_DIAG_ENABLE = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_DIAG", "0") in ("1", "true", "True")
    or os.environ.get("VLLM_ASCEND_SPLIT_DIAG", "0") in ("1", "true", "True")
)
_ACL_GRAPH_DIAG_MAX_LOGS = int(
    os.environ.get("VLLM_ASCEND_ACLGRAPH_DIAG_MAX_LOGS", "600"))
_acl_graph_diag_count = 0
_ACLGRAPH_REPLAY_GLOBAL_SYNC = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_REPLAY_GLOBAL_SYNC", "0")
    in ("1", "true", "True")
)
_ACL_GRAPH_DEBUG_ENABLE = (
    os.environ.get("VLLM_ASCEND_ACL_GRAPH_DEBUG", "0")
    in ("1", "true", "True")
)
_ACL_GRAPH_DEBUG_FILE = os.environ.get(
    "VLLM_ASCEND_ACL_GRAPH_DEBUG_FILE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "acl_graph_debug.log")),
)
_ACL_GRAPH_UPDATE_PARAM_DIAG = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_UPDATE_PARAM_DIAG", "0")
    in ("1", "true", "True")
)
_ACL_GRAPH_FIA_UPDATE_USE_CAPTURED_PARAMS = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_FIA_UPDATE_USE_CAPTURED_PARAMS", "0")
    in ("1", "true", "True")
)
_ACL_GRAPH_FIA_UPDATE_EVENT_ONLY = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_FIA_UPDATE_EVENT_ONLY", "0")
    in ("1", "true", "True")
)


def _append_acl_graph_debug(tag: str, payload: Any) -> None:
    if not _ACL_GRAPH_DEBUG_ENABLE:
        return
    try:
        with open(_ACL_GRAPH_DEBUG_FILE, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write("[{}] {}: {}\n".format(ts, tag, payload))
    except Exception as e:
        logger.warning(
            "Failed to write acl graph debug file %s: %s",
            _ACL_GRAPH_DEBUG_FILE,
            e,
        )


def _safe_tensor_shape(tensor: Any):
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return list(tensor.shape)
    except Exception:
        return None


def _safe_tensor_ptr(tensor: Any):
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return int(tensor.data_ptr())
    except Exception:
        return None


def _safe_tensor_head(tensor: Any, max_items: int = 4):
    # Keep compatibility for older payload keys; intentionally disabled.
    return None


def _safe_tensor_info(tensor: Any) -> Optional[dict[str, Any]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        stride = list(tensor.stride())
    except Exception:
        stride = None
    try:
        storage_offset = int(tensor.storage_offset())
    except Exception:
        storage_offset = None
    return {
        "ptr": _safe_tensor_ptr(tensor),
        "shape": _safe_tensor_shape(tensor),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": stride,
        "storage_offset": storage_offset,
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _safe_sequence_tail(value: Any, max_items: int = 6) -> Any:
    try:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return value.detach().cpu().reshape(-1)[-max_items:].tolist()
        if isinstance(value, (list, tuple)):
            return list(value[-max_items:])
    except Exception:
        return None
    return None


def _object_id_info(value: Any, *, max_items: int = 2) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return {
            "type":
            type(value).__name__,
            "id":
            int(id(value)),
            "len":
            int(len(value)),
            "items": [
                _object_id_info(item, max_items=max_items)
                for item in list(value)[:int(max_items)]
            ],
        }
    return {
        "type": type(value).__name__,
        "id": int(id(value)),
        "repr": repr(value)[:160],
    }


def _resolve_callable_arg_names(runnable: Callable) -> Optional[list[str]]:
    try:
        return list(inspect.signature(runnable).parameters.keys())
    except Exception:
        return None


def _resolve_callable_name(runnable: Callable) -> str:
    if hasattr(runnable, "__qualname__"):
        return str(getattr(runnable, "__qualname__"))
    if hasattr(runnable, "__name__"):
        return str(getattr(runnable, "__name__"))
    return type(runnable).__name__


def _collect_tensor_arg_infos(
    args: tuple[Any, ...],
    arg_names: Optional[list[str]] = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    addresses: list[int] = []
    tensor_infos: list[dict[str, Any]] = []
    for arg_index, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            continue
        ptr = _safe_tensor_ptr(arg)
        if ptr is None:
            # Keep list lengths stable for index-based mismatch diagnostics.
            ptr = -1
        addresses.append(ptr)
        tensor_infos.append({
            "tensor_index": len(addresses) - 1,
            "arg_index": arg_index,
            "arg_name": (
                arg_names[arg_index]
                if arg_names is not None and arg_index < len(arg_names)
                else None
            ),
            "shape": _safe_tensor_shape(arg),
            "dtype": str(arg.dtype),
            "device": str(arg.device),
            "stride": list(arg.stride()),
            "is_contiguous": bool(arg.is_contiguous()),
        })
    return addresses, tensor_infos


def _build_input_address_mismatch(
    expected: list[int],
    got: list[int],
    expected_infos: Optional[list[dict[str, Any]]],
    got_infos: Optional[list[dict[str, Any]]],
    max_items: int = 8,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for idx, (exp, new) in enumerate(zip(expected, got)):
        if exp == new:
            continue
        capture_info = expected_infos[idx] if expected_infos and idx < len(expected_infos) else None
        replay_info = got_infos[idx] if got_infos and idx < len(got_infos) else None
        mismatches.append({
            "tensor_index": idx,
            "arg_index": (
                capture_info.get("arg_index")
                if isinstance(capture_info, dict) and "arg_index" in capture_info
                else (
                    replay_info.get("arg_index")
                    if isinstance(replay_info, dict) and "arg_index" in replay_info
                    else None
                )
            ),
            "expected_ptr": exp,
            "got_ptr": new,
            "capture": capture_info,
            "replay": replay_info,
        })
        if len(mismatches) >= max_items:
            break

    return {
        "expected_len": len(expected),
        "got_len": len(got),
        "length_mismatch": len(expected) != len(got),
        "mismatch_count": sum(1 for exp, new in zip(expected, got) if exp != new),
        "mismatches": mismatches,
    }


def _collect_attn_metadata_tensor_infos(
    attn_metadata: Any,
    *,
    max_tensors: int = 200,
    max_depth: int = 8,
) -> tuple[list[int], list[dict[str, Any]]]:
    addresses: list[int] = []
    tensor_infos: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str, depth: int) -> None:
        if len(addresses) >= max_tensors or depth > max_depth:
            return
        if isinstance(value, torch.Tensor):
            ptr = _safe_tensor_ptr(value)
            if ptr is None:
                ptr = -1
            addresses.append(ptr)
            tensor_infos.append({
                "tensor_index": len(addresses) - 1,
                "path": path,
                "shape": _safe_tensor_shape(value),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": list(value.stride()),
                "is_contiguous": bool(value.is_contiguous()),
                "storage_offset": int(value.storage_offset()),
            })
            return

        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}", depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                visit(child, f"{path}[{idx}]", depth + 1)
            return
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name, None),
                      f"{path}.{field.name}", depth + 1)
            return

        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            for name, child in attrs.items():
                if name.startswith("__"):
                    continue
                visit(child, f"{path}.{name}", depth + 1)

    visit(attn_metadata, "attn_metadata", 0)
    return addresses, tensor_infos


def _should_validate_inplace_input_ptrs(forward_context: Any,
                                        debug_mode: bool) -> bool:
    return debug_mode or bool(
        getattr(forward_context, "validate_inplace_input_ptrs", False))


def _should_validate_inplace_metadata_ptrs(forward_context: Any) -> bool:
    return bool(getattr(forward_context, "validate_inplace_metadata_ptrs",
                        False))


def _is_allowed_inplace_lazy_capture(forward_context: Any,
                                     batch_descriptor: BatchDescriptor,
                                     aclgraph_runtime_mode: CUDAGraphMode
                                     ) -> bool:
    if aclgraph_runtime_mode not in (CUDAGraphMode.FULL,
                                     CUDAGraphMode.PIECEWISE):
        return False
    if not bool(getattr(forward_context, "allow_inplace_lazy_capture",
                        False)):
        return False

    split_mode = getattr(forward_context, "split_inplace_mode", None)
    graph_variant = getattr(batch_descriptor, "graph_variant", "")
    attention_backend = getattr(batch_descriptor, "attention_backend", "")
    capture_metadata_mode = getattr(batch_descriptor,
                                    "capture_metadata_mode", "")
    if (split_mode == "mixed_request_serial"
            and graph_variant == "mixed_request_serial"
            and capture_metadata_mode == "mixed_request_compact"):
        return True
    if (split_mode == "mixed_request_piecewise_attention_parallel"
            and graph_variant == "mixed_request_piecewise_attention_parallel"
            and capture_metadata_mode == "mixed_request_compact"):
        return True

    return (
        int(getattr(batch_descriptor, "start_num_tokens", 0) or 0) > 0
        and split_mode in ("inplace_serial", "inplace_parallel")
        and graph_variant in ("inplace_serial", "inplace_parallel")
        and attention_backend in ("fia", "pa")
    )


def _extract_block_table_from_metadata(metadata: Any):
    metadata_block_table = None
    metadata_block_source = None
    for attr in ("block_table", "block_tables", "block_table_tensor"):
        candidate = getattr(metadata, attr, None)
        if isinstance(candidate, torch.Tensor):
            metadata_block_table = candidate
            metadata_block_source = attr
            break

    if metadata_block_table is None:
        decode_metadata = getattr(metadata, "decode", None)
        for attr in ("block_table", "block_tables", "block_table_tensor"):
            candidate = getattr(decode_metadata, attr, None)
            if isinstance(candidate, torch.Tensor):
                metadata_block_table = candidate
                metadata_block_source = f"decode.{attr}"
                break

    return metadata_block_table, metadata_block_source


def _extract_graph_param_block_table(forward_context: Any, runtime_shape: Any,
                                     in_parallel_streams: bool = False):
    if runtime_shape is None:
        return None, None

    graph_params = get_graph_params(in_parallel_streams)
    if graph_params is None:
        return None, None
    param_key = get_graph_param_key(forward_context, runtime_shape)
    shape_params = graph_params.attn_params.get(param_key)
    if not shape_params:
        return None, None

    first_param = shape_params[0]
    # Prefer FIA slot first, then PA/MLA-like layouts as fallback.
    for idx, source in ((3, "fia.param[3]"), (6, "pa.param[6]"), (10, "mla.param[10]")):
        if len(first_param) > idx and isinstance(first_param[idx], torch.Tensor):
            return first_param[idx], source

    return None, None


def _refresh_block_table_in_place(graph_block_table: Any,
                                  metadata_block_table: Any) -> bool:
    """Refresh captured graph block_table from runtime metadata in-place.

    This is intended for split replay path only. Unsplit path should keep
    original behavior to minimize runtime risk.
    """
    if (not isinstance(graph_block_table, torch.Tensor)
            or not isinstance(metadata_block_table, torch.Tensor)):
        return False

    # No work needed when both already alias the same storage.
    if graph_block_table.data_ptr() == metadata_block_table.data_ptr():
        return False

    try:
        if graph_block_table.ndim == 2 and metadata_block_table.ndim == 2:
            rows = min(graph_block_table.shape[0], metadata_block_table.shape[0])
            cols = min(graph_block_table.shape[1], metadata_block_table.shape[1])
            if rows <= 0 or cols <= 0:
                return False
            graph_block_table[:rows, :cols].copy_(
                metadata_block_table[:rows, :cols], non_blocking=False)
            return True

        # Fallback for non-2D layouts.
        count = min(graph_block_table.numel(), metadata_block_table.numel())
        if count <= 0:
            return False
        graph_block_table.view(-1)[:count].copy_(
            metadata_block_table.view(-1)[:count], non_blocking=False)
        return True
    except Exception:
        return False


def _build_replay_block_table_diag(forward_context: Any, runtime_shape: Any) -> dict[str, Any]:
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not attn_metadata:
        return {}

    first_key = next(iter(attn_metadata), None)
    if first_key is None:
        return {}

    metadata = attn_metadata[first_key]
    metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(metadata)
    in_parallel_streams = bool(
        getattr(forward_context, "in_parallel_streams", False))
    param_key = get_graph_param_key(forward_context, runtime_shape)
    graph_block_table, graph_block_table_source = _extract_graph_param_block_table(
        forward_context, runtime_shape, in_parallel_streams)
    return {
        "first_key": first_key,
        "runtime_shape": runtime_shape,
        "graph_param_key": graph_param_key_info(param_key),
        "meta_block_table_source": metadata_block_source,
        "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
        "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
        "graph_block_table_source": graph_block_table_source,
        "graph_block_table_shape": _safe_tensor_shape(graph_block_table),
        "graph_block_table_ptr": _safe_tensor_ptr(graph_block_table),
    }


def _block_table_copy_diag(graph_block_table: Any,
                           metadata_block_table: Any) -> dict[str, Any]:
    payload = {
        "graph_block_table_shape": _safe_tensor_shape(graph_block_table),
        "graph_block_table_ptr": _safe_tensor_ptr(graph_block_table),
        "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
        "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
        "copy_numel": 0,
        "same_storage": False,
    }
    if (not isinstance(graph_block_table, torch.Tensor)
            or not isinstance(metadata_block_table, torch.Tensor)):
        return payload
    same_storage = graph_block_table.data_ptr() == metadata_block_table.data_ptr()
    payload["same_storage"] = bool(same_storage)
    if same_storage:
        return payload
    if graph_block_table.ndim == 2 and metadata_block_table.ndim == 2:
        rows = min(graph_block_table.shape[0], metadata_block_table.shape[0])
        cols = min(graph_block_table.shape[1], metadata_block_table.shape[1])
        payload["copy_rows"] = int(max(rows, 0))
        payload["copy_cols"] = int(max(cols, 0))
        payload["copy_numel"] = int(max(rows, 0) * max(cols, 0))
        return payload
    payload["copy_numel"] = int(
        min(graph_block_table.numel(), metadata_block_table.numel()))
    return payload


def _log_block_table_refresh_diag(
        *,
        attn_impl: str,
        key: Any,
        forward_context: Any,
        runtime_shape: Any,
        in_parallel_streams: bool,
        metadata_block_source: Optional[str],
        graph_block_table: Any,
        metadata_block_table: Any,
        block_table_refreshed: bool) -> None:
    if not split_debug.is_enabled():
        return
    split_debug.log_event(
        "acl_graph_block_table_refresh",
        {
            "attn_impl": attn_impl,
            "key": str(key),
            "runtime_shape": runtime_shape,
            "graph_param_key": graph_param_key_info(
                get_graph_param_key(forward_context, runtime_shape)),
            "ubatch_num": getattr(forward_context, "ubatch_num", None),
            "in_parallel_streams": bool(in_parallel_streams),
            "batch_descriptor": split_debug.batch_descriptor_info(
                getattr(forward_context, "batch_descriptor", None)),
            "meta_block_table_source": metadata_block_source,
            "block_table_refreshed": bool(block_table_refreshed),
            **_block_table_copy_diag(graph_block_table, metadata_block_table),
        },
        step_id=getattr(forward_context, "split_inplace_debug_step_id", None),
    )


def _maybe_log_acl_graph_diag(tag: str, payload: Any) -> None:
    global _acl_graph_diag_count
    if ((not _ACL_GRAPH_DIAG_ENABLE and envs.VLLM_LOGGING_LEVEL != "DEBUG")
            or _acl_graph_diag_count >= _ACL_GRAPH_DIAG_MAX_LOGS):
        return
    logger.info("%s: %s", tag, payload)
    _acl_graph_diag_count += 1


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 cudagraph_options: Optional[CUDAGraphOptions] = None,
                 device: torch.device = None,):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.device = device

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self.runnable_name = _resolve_callable_name(runnable)
        self.runnable_arg_names = _resolve_callable_arg_names(runnable)

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()
        # Parallel stream MUST use an independent pool so that two graphs
        # replaying concurrently never share intermediate-tensor addresses.
        self.graph_pool_parallel_streams = torch.npu.graph_pool_handle()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}
        self.concrete_aclgraph_entries2: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}
        self.fallback_eager_count = 0

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def has_graph(self,
                  batch_descriptor: Optional[BatchDescriptor],
                  in_parallel_streams: bool = False) -> bool:
        """Return whether a concrete ACL graph has already been captured."""
        if batch_descriptor is None:
            return False
        entries = (self.concrete_aclgraph_entries2
                   if in_parallel_streams else self.concrete_aclgraph_entries)
        entry = entries.get(batch_descriptor)
        return entry is not None and entry.aclgraph is not None

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        split_debug_step_id = getattr(forward_context,
                                      "split_inplace_debug_step_id", None)
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or \
                            aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            self.fallback_eager_count += 1
            current_entries = (self.concrete_aclgraph_entries2
                               if in_parallel_streams
                               else self.concrete_aclgraph_entries)
            entry = current_entries.get(batch_descriptor)
            if entry is not None:
                entry.fallback_eager_count += 1
            if _ACL_GRAPH_DEBUG_ENABLE or split_debug.is_enabled():
                fallback_payload = {
                    "batch_descriptor": str(batch_descriptor),
                    "ubatch_num": getattr(forward_context, "ubatch_num", None),
                    "in_parallel_streams": in_parallel_streams,
                    "entry_id": id(entry) if entry is not None else None,
                    "entry_has_graph": (
                        entry is not None and entry.aclgraph is not None),
                    "runtime_mode": (
                        aclgraph_runtime_mode.name
                        if isinstance(aclgraph_runtime_mode, CUDAGraphMode)
                        else str(aclgraph_runtime_mode)
                    ),
                    "wrapper_runtime_mode": self.runtime_mode.name,
                    "fallback_eager_count": (
                        int(entry.fallback_eager_count)
                        if entry is not None
                        else int(self.fallback_eager_count)),
                    "wrapper_fallback_eager_count":
                    int(self.fallback_eager_count),
                }
                _append_acl_graph_debug("acl_graph_eager_fallback",
                                        fallback_payload)
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "acl_graph_eager_fallback",
                        {
                            **fallback_payload,
                            "batch_descriptor":
                            split_debug.batch_descriptor_info(
                                batch_descriptor),
                        },
                        step_id=split_debug_step_id,
                    )
            return self.runnable(*args, **kwargs)
        current_concrete_aclgraph_entries = self.concrete_aclgraph_entries2 if in_parallel_streams else self.concrete_aclgraph_entries
        is_new_entry = batch_descriptor not in current_concrete_aclgraph_entries
        if is_new_entry:
            # create a new entry for this batch descriptor
            current_concrete_aclgraph_entries[batch_descriptor] = \
                ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = current_concrete_aclgraph_entries[batch_descriptor]
        entry_select_payload = {
            "batch_descriptor": str(batch_descriptor),
            "ubatch_num": getattr(forward_context, "ubatch_num", None),
            "in_parallel_streams": in_parallel_streams,
            "entry_created": is_new_entry,
            "entry_has_graph": entry.aclgraph is not None,
            "entry_id": id(entry),
            "num_tokens": getattr(forward_context, "split_actual_num_tokens",
                                  None),
            "graph_num_tokens": getattr(
                forward_context, "split_graph_num_tokens",
                getattr(batch_descriptor, "num_tokens", None)),
            "start_num_tokens": int(
                getattr(batch_descriptor, "start_num_tokens", 0) or 0),
            "graph_variant": getattr(batch_descriptor, "graph_variant", ""),
            "attention_backend": getattr(batch_descriptor,
                                         "attention_backend", ""),
            "capture_count": int(entry.capture_count),
            "replay_count": int(entry.replay_count),
            "fallback_eager_count": int(entry.fallback_eager_count),
            "runtime_mode": (
                aclgraph_runtime_mode.name
                if isinstance(aclgraph_runtime_mode, CUDAGraphMode)
                else str(aclgraph_runtime_mode)
            ),
        }
        _append_acl_graph_debug("acl_graph_entry_select", entry_select_payload)
        if split_debug.is_enabled():
            split_debug.log_event(
                "acl_graph_entry_select",
                {
                    **entry_select_payload,
                    "batch_descriptor": split_debug.batch_descriptor_info(
                        batch_descriptor),
                },
                step_id=split_debug_step_id,
            )
        selected_pool=(self.graph_pool_parallel_streams
                     if in_parallel_streams else self.graph_pool)
        if entry.aclgraph is None:
            start_num_tokens = int(
                getattr(batch_descriptor, "start_num_tokens", 0) or 0)
            is_inplace_lazy_capture = _is_allowed_inplace_lazy_capture(
                forward_context, batch_descriptor, aclgraph_runtime_mode)
            if start_num_tokens > 0 and not is_inplace_lazy_capture:
                split_debug.log_event(
                    "inplace_lazy_capture_blocked",
                    {
                        "entry_id": id(entry),
                        "batch_descriptor":
                        split_debug.batch_descriptor_info(batch_descriptor),
                        "runtime_mode": (
                            aclgraph_runtime_mode.name
                            if isinstance(aclgraph_runtime_mode,
                                          CUDAGraphMode) else
                            str(aclgraph_runtime_mode)),
                        "allow_inplace_lazy_capture":
                        bool(getattr(forward_context,
                                     "allow_inplace_lazy_capture", False)),
                        "split_inplace_mode":
                        getattr(forward_context, "split_inplace_mode", None),
                    },
                    step_id=split_debug_step_id,
                )
                raise RuntimeError(
                    "Refusing to capture an inplace offset ACL graph without "
                    "allow_inplace_lazy_capture in the forward context: "
                    f"{batch_descriptor!r}")

            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            previous_capture_enabled = (
                compilation_monitor.cudagraph_capturing_enabled)
            if is_inplace_lazy_capture:
                if _ACL_GRAPH_DEBUG_ENABLE:
                    split_debug.log_event(
                        "inplace_lazy_capture_guard",
                        {
                            "phase": "enable",
                            "entry_id": id(entry),
                            "batch_descriptor":
                            split_debug.batch_descriptor_info(
                                batch_descriptor),
                            "previous_capture_enabled":
                            bool(previous_capture_enabled),
                        },
                        step_id=split_debug_step_id,
                    )
                compilation_monitor.set_cudagraph_capturing_enabled(True)
            try:
                validate_cudagraph_capturing_enabled()
            except Exception:
                if is_inplace_lazy_capture:
                    compilation_monitor.set_cudagraph_capturing_enabled(
                        previous_capture_enabled)
                raise
            input_addresses, input_tensor_infos = _collect_tensor_arg_infos(
                args,
                self.runnable_arg_names,
            )
            entry.input_addresses = input_addresses
            entry.input_tensor_infos = input_tensor_infos
            if _should_validate_inplace_metadata_ptrs(forward_context):
                (entry.attn_metadata_addresses,
                 entry.attn_metadata_tensor_infos) = (
                     _collect_attn_metadata_tensor_infos(
                         getattr(forward_context, "attn_metadata", None)))
            if _ACL_GRAPH_DEBUG_ENABLE:
                split_debug.log_event(
                    "acl_graph_capture",
                    {
                        "phase": "pre",
                        "entry_id": id(entry),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            batch_descriptor),
                        "ubatch_num": getattr(forward_context, "ubatch_num",
                                               None),
                        "in_parallel_streams": in_parallel_streams,
                        "runtime_mode": (
                            aclgraph_runtime_mode.name
                            if isinstance(aclgraph_runtime_mode,
                                          CUDAGraphMode) else
                            str(aclgraph_runtime_mode)),
                        "input_tensors": input_tensor_infos,
                    },
                    step_id=split_debug_step_id,
                )
            aclgraph = None
            previous_forward_context_capturing = bool(
                getattr(forward_context, "capturing", False))

            with ExitStack() as stack:
                try:
                    aclgraph = torch.npu.NPUGraph()
                    if self.aclgraph_options.gc_disable:
                        # during every model forward for piecewise aclgraph
                        # mode, we will capture many pieces of aclgraphs
                        # (roughly one per layer). running gc again and again
                        # across layers will make the aclgraph capture very slow.
                        # therefore, we only run gc for the first graph,
                        # and disable gc for the rest of the graphs.
                        stack.enter_context(patch("gc.collect", lambda: None))
                        stack.enter_context(
                            patch("torch.npu.empty_cache", lambda: None))

                    # mind-exploding: carefully manage the reference and memory.
                    forward_context.capturing = True
                    set_graph_pool_id(selected_pool)
                    with torch.npu.graph(aclgraph, pool=selected_pool):
                        # `output` is managed by pytorch's aclgraph pool
                        output = self.runnable(*args, **kwargs)
                        if self.aclgraph_options.weak_ref_output:
                            # by converting it to weak ref,
                            # the original `output` will immediately be released
                            # to save memory. It is only safe to do this for
                            # the last graph in piecewise aclgraph mode, because
                            # the output of the last graph will not be used by
                            # any other acl graph.
                            output = weak_ref_tensors(output)
                finally:
                    forward_context.capturing = (
                        previous_forward_context_capturing)
                    if is_inplace_lazy_capture:
                        compilation_monitor.set_cudagraph_capturing_enabled(
                            previous_capture_enabled)
                        if _ACL_GRAPH_DEBUG_ENABLE:
                            split_debug.log_event(
                                "inplace_lazy_capture_guard",
                                {
                                    "phase": "restore",
                                    "entry_id": id(entry),
                                    "batch_descriptor":
                                    split_debug.batch_descriptor_info(
                                        batch_descriptor),
                                    "capture_enabled":
                                    bool(previous_capture_enabled),
                                },
                                step_id=split_debug_step_id,
                            )

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph
            entry.capture_count += 1

            compilation_counter.num_cudagraph_captured += 1
            if _ACL_GRAPH_DEBUG_ENABLE:
                split_debug.log_event(
                    "acl_graph_capture",
                    {
                        "phase": "post",
                        "entry_id": id(entry),
                        "batch_descriptor":
                        split_debug.batch_descriptor_info(batch_descriptor),
                        "ubatch_num": getattr(forward_context, "ubatch_num",
                                               None),
                        "in_parallel_streams": in_parallel_streams,
                        "runtime_mode": (
                            aclgraph_runtime_mode.name
                            if isinstance(aclgraph_runtime_mode,
                                          CUDAGraphMode) else
                            str(aclgraph_runtime_mode)),
                        "capture_count": int(entry.capture_count),
                        "replay_count": int(entry.replay_count),
                        "fallback_eager_count":
                        int(entry.fallback_eager_count),
                    },
                    step_id=split_debug_step_id,
                )

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if _should_validate_inplace_input_ptrs(forward_context,
                                               self.is_debugging_mode):
            # check if the input addresses are the same
            new_input_addresses, new_input_tensor_infos = _collect_tensor_arg_infos(
                args,
                self.runnable_arg_names,
            )
            if new_input_addresses != entry.input_addresses:
                mismatch_detail = _build_input_address_mismatch(
                    entry.input_addresses,
                    new_input_addresses,
                    entry.input_tensor_infos,
                    new_input_tensor_infos,
                )
                _append_acl_graph_debug(
                    "acl_graph_input_addr_mismatch",
                    {
                        "entry_id": id(entry),
                        "runnable": self.runnable_name,
                        "runnable_arg_names": self.runnable_arg_names,
                        "batch_descriptor": str(batch_descriptor),
                        "ubatch_num": getattr(forward_context, "ubatch_num", None),
                        "in_parallel_streams": in_parallel_streams,
                        **mismatch_detail,
                    },
                )
                raise AssertionError(
                    "Input addresses for aclgraphs are different during replay. "
                    f"mismatch_detail={mismatch_detail}"
                )

        if _should_validate_inplace_metadata_ptrs(forward_context):
            (new_metadata_addresses,
             new_metadata_tensor_infos) = _collect_attn_metadata_tensor_infos(
                 getattr(forward_context, "attn_metadata", None))
            expected_metadata_addresses = entry.attn_metadata_addresses or []
            if new_metadata_addresses != expected_metadata_addresses:
                mismatch_detail = _build_input_address_mismatch(
                    expected_metadata_addresses,
                    new_metadata_addresses,
                    entry.attn_metadata_tensor_infos,
                    new_metadata_tensor_infos,
                )
                split_debug.log_event(
                    "inplace_metadata_ptr_mismatch",
                    {
                        "entry_id": id(entry),
                        "batch_descriptor":
                        split_debug.batch_descriptor_info(batch_descriptor),
                        "ubatch_num":
                        getattr(forward_context, "ubatch_num", None),
                        **mismatch_detail,
                    },
                    step_id=split_debug_step_id,
                )
                raise AssertionError(
                    "Attention metadata addresses for inplace aclgraphs are "
                    "different during replay. "
                    f"mismatch_detail={mismatch_detail}")

        # Do not write normal replay JSONL here. This call is inside
        # self.model(), immediately before the caller updates graph task
        # attention params; synchronous CPU I/O in this window can perturb NPU
        # graph replay/update ordering. Use VLLM_ASCEND_ACL_GRAPH_DEBUG=1 only
        # for low-level diagnosis.
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        runtime_shape = getattr(batch_descriptor, "num_tokens", None)
        graph_param_key = get_graph_param_key(forward_context, runtime_shape)
        if _ACL_GRAPH_DEBUG_ENABLE:
            _append_acl_graph_debug(
                "acl_graph_replay",
                {
                    "entry_id": id(entry),
                    "batch_descriptor": str(batch_descriptor),
                    "ubatch_num": getattr(forward_context, "ubatch_num", None),
                    "in_parallel_streams": in_parallel_streams,
                    "replay_global_sync": bool(_ACLGRAPH_REPLAY_GLOBAL_SYNC
                                               and not in_parallel_streams),
                    "phase": "pre",
                    "runtime_shape": runtime_shape,
                    "graph_param_key": graph_param_key_info(graph_param_key),
                })
        # Keep legacy ordering for the main stream path, but avoid global
        # barriers for split parallel replay so two streams can overlap.
        if _ACLGRAPH_REPLAY_GLOBAL_SYNC and not in_parallel_streams:
            torch.npu.synchronize()
        set_graph_pool_id(selected_pool)
        entry.aclgraph.replay()
        entry.replay_count += 1
        if _ACL_GRAPH_DEBUG_ENABLE:
            replay_post_diag = _build_replay_block_table_diag(
                forward_context, runtime_shape)
            _append_acl_graph_debug(
                "acl_graph_replay",
                {
                    "entry_id": id(entry),
                    "batch_descriptor": str(batch_descriptor),
                    "ubatch_num": getattr(forward_context, "ubatch_num", None),
                    "in_parallel_streams": in_parallel_streams,
                    "phase": "post",
                    "runtime_shape": runtime_shape,
                    "capture_count": int(entry.capture_count),
                    "replay_count": int(entry.replay_count),
                    "fallback_eager_count": int(entry.fallback_eager_count),
                    **replay_post_diag,
                })
        # _maybe_log_acl_graph_diag(
        #     "acl_graph_replay_post",
        #     {
        #         "entry_id": id(entry),
        #         "batch_descriptor": str(batch_descriptor),
        #         "ubatch_num": getattr(forward_context, "ubatch_num", None),
        #         "in_parallel_streams": in_parallel_streams,
        #         "phase": "post",
        #         **replay_post_diag,
        #     },
        # )
        return entry.output


def _update_attn_pa_params(update_stream, forward_context, runtime_shape,
                           refresh_block_table: bool = False,
                           in_parallel_streams: bool = False):
    graph_params = get_graph_params(in_parallel_streams)
    param_key = get_graph_param_key(forward_context, runtime_shape)
    require_graph_param_key(graph_params, param_key, op="_update_attn_pa_params")
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ):
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale,
                block_table,
                seq_lens,
                output,
            ) = param
            metadata = forward_context.attn_metadata[key]
            runtime_metadata = _get_attention_update_metadata(
                forward_context, key)
            seq_lens = getattr(runtime_metadata, "seq_lens",
                               metadata.seq_lens)
            metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(
                runtime_metadata)
            block_table_refreshed = False
            if refresh_block_table:
                block_table_refreshed = _refresh_block_table_in_place(
                    block_table, metadata_block_table)
                _log_block_table_refresh_diag(
                    attn_impl="pa",
                    key=key,
                    forward_context=forward_context,
                    runtime_shape=runtime_shape,
                    in_parallel_streams=in_parallel_streams,
                    metadata_block_source=metadata_block_source,
                    graph_block_table=block_table,
                    metadata_block_table=metadata_block_table,
                    block_table_refreshed=block_table_refreshed,
                )
            # _maybe_log_acl_graph_diag(
            #     "acl_graph_attn_update_diag",
            #     {
            #         "attn_impl": "pa",
            #         "key": key,
            #         "runtime_shape": runtime_shape,
            #         "ubatch_num": getattr(forward_context, "ubatch_num", None),
            #         "seq_lens_shape": _safe_tensor_shape(seq_lens),
            #         "seq_lens_len": len(seq_lens)
            #         if hasattr(seq_lens, "__len__") else None,
            #         "graph_block_table_shape": _safe_tensor_shape(block_table),
            #         "graph_block_table_ptr": _safe_tensor_ptr(block_table),
            #         "meta_block_table_source": metadata_block_source,
            #         "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
            #         "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
            #         "block_table_refreshed": block_table_refreshed,
            #                 },
            # )

            # When using FULL_DECODE_ONLY, there are some rare bugs for FULL_DECODE_ONLY
            # mode with GQA. This is triggered by getting workspace for _npu_paged_attention
            # in torch_npu. On some rare cases, _npu_paged_attention with smaller seq_lens
            # might encounter a bigger workspace, while currently we use max_model_len to
            # calculate max workspace in capturing. So additional get_workspace is added
            # here to avoid such bugs.
            # TODO(Angazenn): we will remove this once _npu_paged_attention is fully
            # replaced by npu_fused_infer_attention_score which does not contain such bugs.
            workspace = torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output)
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(query=query,
                                           key_cache=key_cache,
                                           value_cache=value_cache,
                                           num_kv_heads=num_kv_heads,
                                           num_heads=num_heads,
                                           scale_value=scale,
                                           block_table=block_table,
                                           context_lens=seq_lens,
                                           out=output,
                                           workspace=workspace)
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                            refresh_block_table: bool = False,
                            in_parallel_streams: bool = False):
    graph_params = get_graph_params(in_parallel_streams)
    param_key = get_graph_param_key(forward_context, runtime_shape)
    require_graph_param_key(graph_params, param_key, op="_update_attn_fia_params")
    # For Qwen3-next, since the kv_cache_config has already categorized
    # linear_attn and self_attn, the attn_metadata is first arranged with
    # self_attn followed by linear_attn. Therefore, using zip directly
    # filters out the update operations for linear_attn.
    with torch.npu.stream(update_stream):
        attn_items = list(zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ))
        for layer_idx, (key, param, handle, event) in enumerate(attn_items):
            (query, key_cache, value, block_tables, attn_mask, block_size,
             seq_lens, query_start_loc, num_kv_heads, num_heads, scale,
             attn_output, softmax_lse) = param

            metadata = forward_context.attn_metadata[key]
            use_captured_params = bool(
                _ACL_GRAPH_FIA_UPDATE_USE_CAPTURED_PARAMS
                and getattr(
                    getattr(forward_context, "batch_descriptor", None),
                    "capture_metadata_mode", "") == "mixed_request_compact")
            event_only = bool(
                _ACL_GRAPH_FIA_UPDATE_EVENT_ONLY
                and getattr(
                    getattr(forward_context, "batch_descriptor", None),
                    "capture_metadata_mode", "") == "mixed_request_compact")
            if use_captured_params:
                runtime_metadata = metadata
                actual_seq_lengths_q = query_start_loc
                metadata_block_table = block_tables
                metadata_block_source = "captured_graph_param"
            else:
                runtime_metadata = _get_attention_update_metadata(
                    forward_context, key)
                seq_lens = maybe_template_fia_seq_lens(
                    forward_context,
                    getattr(runtime_metadata, "seq_lens_list",
                            metadata.seq_lens_list),
                    _get_fia_key_t(key_cache, block_size),
                    source=f"acl_graph_update:{key}")
                actual_seq_lengths_q = getattr(runtime_metadata,
                                               "actual_seq_lengths_q",
                                               metadata.actual_seq_lengths_q)
                metadata_block_table, metadata_block_source = (
                    _extract_block_table_from_metadata(runtime_metadata))
            block_table_refreshed = False
            if refresh_block_table and not use_captured_params:
                block_table_refreshed = _refresh_block_table_in_place(
                    block_tables, metadata_block_table)
                _log_block_table_refresh_diag(
                    attn_impl="fia",
                    key=key,
                    forward_context=forward_context,
                    runtime_shape=runtime_shape,
                    in_parallel_streams=in_parallel_streams,
                    metadata_block_source=metadata_block_source,
                    graph_block_table=block_tables,
                    metadata_block_table=metadata_block_table,
                    block_table_refreshed=block_table_refreshed,
                )
            if (_ACL_GRAPH_UPDATE_PARAM_DIAG
                    and layer_idx in (0, len(attn_items) - 1)):
                split_debug.log_event(
                    "acl_graph_fia_update_params",
                    {
                        "key": key,
                        "layer_idx": int(layer_idx),
                        "runtime_shape": runtime_shape,
                        "graph_param_key": graph_param_key_info(param_key),
                        "in_parallel_streams": bool(in_parallel_streams),
                        "handle_id": _object_id_info(handle),
                        "event_id": _object_id_info(event),
                        "query": _safe_tensor_info(query),
                        "key_cache": _safe_tensor_info(key_cache),
                        "value": _safe_tensor_info(value),
                        "block_tables": _safe_tensor_info(block_tables),
                        "metadata_block_table":
                        _safe_tensor_info(metadata_block_table),
                        "attn_mask": _safe_tensor_info(attn_mask),
                        "attn_output": _safe_tensor_info(attn_output),
                        "softmax_lse": _safe_tensor_info(softmax_lse),
                        "workspace": _safe_tensor_info(
                            graph_params.workspaces.get(param_key)),
                        "seq_lens_type": type(seq_lens).__name__,
                        "seq_lens_len": (
                            len(seq_lens)
                            if hasattr(seq_lens, "__len__") else None),
                        "seq_lens_tail": _safe_sequence_tail(seq_lens),
                        "actual_seq_lengths_q_type":
                        type(actual_seq_lengths_q).__name__,
                        "actual_seq_lengths_q_len": (
                            len(actual_seq_lengths_q)
                            if hasattr(actual_seq_lengths_q, "__len__")
                            else None),
                        "actual_seq_lengths_q_tail":
                        _safe_sequence_tail(actual_seq_lengths_q),
                        "block_size": int(block_size),
                        "num_kv_heads": int(num_kv_heads),
                        "num_heads": int(num_heads),
                        "scale": float(scale),
                        "block_table_refreshed": bool(block_table_refreshed),
                        "use_captured_params": bool(use_captured_params),
                        "event_only": bool(event_only),
                    },
                    step_id=getattr(forward_context,
                                    "split_inplace_debug_step_id", None),
                )
            # _maybe_log_acl_graph_diag(
            #     "acl_graph_attn_update_diag",
            #     {
            #         "attn_impl": "fia",
            #         "key": key,
            #         "runtime_shape": runtime_shape,
            #         "ubatch_num": getattr(forward_context, "ubatch_num", None),
            #         "seq_lens_shape": _safe_tensor_shape(seq_lens),
            #         "seq_lens_len": len(seq_lens)
            #         if hasattr(seq_lens, "__len__") else None,
            #         "graph_block_table_shape": _safe_tensor_shape(block_tables),
            #         "graph_block_table_ptr": _safe_tensor_ptr(block_tables),
            #         "meta_block_table_source": metadata_block_source,
            #         "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
            #         "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
            #         "block_table_refreshed": block_table_refreshed,
            #                 },
            # )
            if event_only:
                event.record(update_stream)
                continue

            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value,
                block_table=block_tables,
                atten_mask=attn_mask,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=seq_lens,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=3,
                workspace=graph_params.workspaces.get(param_key),
                out=[attn_output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def _has_dual_stream_attention_metadata(forward_context) -> bool:
    dual_metadata = getattr(forward_context,
                            "dual_stream_attention_metadata", None)
    return isinstance(dual_metadata, list) and len(dual_metadata) == 2


def _same_tensor_ref(left: Any, right: Any) -> bool:
    left_ptr = _safe_tensor_ptr(left)
    right_ptr = _safe_tensor_ptr(right)
    if left_ptr is not None or right_ptr is not None:
        return left_ptr == right_ptr
    return left is right


def _require_same_dual_fia_tensor(name: str, left: Any, right: Any,
                                  layer_key: Any):
    if not _same_tensor_ref(left, right):
        raise RuntimeError(
            "PTA dual-FIA update requires both splits to share "
            f"{name}; got different tensors for layer {layer_key!s}")


def _update_attn_dual_fia_params(update_stream, forward_context,
                                 runtime_shape,
                                 refresh_block_table: bool = True,
                                 in_parallel_streams: bool = False):
    graph_params = get_graph_params(in_parallel_streams)
    param_key = get_graph_param_key(forward_context, runtime_shape)
    require_graph_param_key(graph_params,
                            param_key,
                            op="_update_attn_dual_fia_params")
    dual_metadata = getattr(forward_context,
                            "dual_stream_attention_metadata", None)
    if not isinstance(dual_metadata, list) or len(dual_metadata) != 2:
        raise RuntimeError(
            "dual_stream_attention_metadata must contain exactly two splits")

    with torch.npu.stream(update_stream):
        attn_items = list(zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ))
        for layer_idx, (key, param, handles, events) in enumerate(attn_items):
            if not isinstance(param, tuple) or len(param) < 2:
                raise RuntimeError(
                    "GraphParams entry is not a dual-stream FIA attention "
                    f"entry for layer {key!s}")

            if param[0] == "dual_stream_fia_pta":
                if len(param) != 5:
                    raise RuntimeError(
                        "PTA dual-stream FIA GraphParams entry must contain "
                        f"full query/output and split ranges for layer {key!s}")
                if not hasattr(torch.npu,
                               "dual_fused_infer_attention_score_update"):
                    raise RuntimeError(
                        "PTA dual-FIA update record was captured but "
                        "torch.npu.dual_fused_infer_attention_score_update "
                        "is unavailable")

                _, query, attn_output, split_params, split_ranges = param
                if len(split_params) != 2 or len(split_ranges) != 2:
                    raise RuntimeError(
                        "PTA dual-stream FIA GraphParams entry must contain "
                        f"exactly two splits for layer {key!s}")
                if isinstance(events, (list, tuple)):
                    update_events = tuple(events)
                elif events is None:
                    update_events = ()
                else:
                    update_events = (events,)
                if len(update_events) not in (1, 2):
                    raise RuntimeError(
                        "PTA dual-stream FIA GraphParams entry must contain "
                        f"one shared event or two split events for layer {key!s}")
                if (_ACL_GRAPH_UPDATE_PARAM_DIAG
                        and layer_idx in (0, len(attn_items) - 1)):
                    split_debug.log_event(
                        "acl_graph_dual_fia_update_handles",
                        {
                            "key": key,
                            "layer_idx": int(layer_idx),
                            "runtime_shape": runtime_shape,
                            "graph_param_key": graph_param_key_info(param_key),
                            "in_parallel_streams": bool(in_parallel_streams),
                            "param_kind": "dual_stream_fia_pta",
                            "handle_id": _object_id_info(handles),
                            "event_id": _object_id_info(update_events),
                        },
                        step_id=getattr(forward_context,
                                        "split_inplace_debug_step_id", None),
                    )

                split_update_params = []
                for split_idx, split_param in enumerate(split_params):
                    (query_view, key_cache, value, block_tables, attn_mask,
                     block_size, seq_lens, query_start_loc, num_kv_heads,
                     num_heads, scale, attn_output_view, softmax_lse,
                     workspace_key) = split_param

                    metadata = dual_metadata[split_idx][key]
                    runtime_metadata = _get_dual_attention_update_metadata(
                        forward_context, dual_metadata, split_idx, key)
                    seq_lens = maybe_template_fia_seq_lens(
                        forward_context,
                        getattr(runtime_metadata, "seq_lens_list",
                                metadata.seq_lens_list),
                        _get_fia_key_t(key_cache, block_size),
                        source=f"acl_graph_update_dual:{key}:{split_idx}")
                    actual_seq_lengths_q = getattr(
                        runtime_metadata, "actual_seq_lengths_q",
                        metadata.actual_seq_lengths_q)
                    metadata_block_table, metadata_block_source = (
                        _extract_block_table_from_metadata(runtime_metadata))
                    block_table_refreshed = False
                    if refresh_block_table:
                        block_table_refreshed = _refresh_block_table_in_place(
                            block_tables, metadata_block_table)
                        _log_block_table_refresh_diag(
                            attn_impl=f"dual_fia_pta:{split_idx}",
                            key=key,
                            forward_context=forward_context,
                            runtime_shape=runtime_shape,
                            in_parallel_streams=in_parallel_streams,
                            metadata_block_source=metadata_block_source,
                            graph_block_table=block_tables,
                            metadata_block_table=metadata_block_table,
                            block_table_refreshed=block_table_refreshed,
                        )

                    split_update_params.append({
                        "key_cache": key_cache,
                        "value": value,
                        "block_tables": block_tables,
                        "attn_mask": attn_mask,
                        "block_size": block_size,
                        "seq_lens": seq_lens,
                        "actual_seq_lengths_q": actual_seq_lengths_q,
                        "num_kv_heads": num_kv_heads,
                        "num_heads": num_heads,
                        "scale": scale,
                        "softmax_lse": softmax_lse,
                        "workspace":
                        graph_params.workspaces.get(workspace_key),
                    })

                split0, split1 = split_update_params
                _require_same_dual_fia_tensor("key_cache",
                                              split0["key_cache"],
                                              split1["key_cache"], key)
                _require_same_dual_fia_tensor("value", split0["value"],
                                              split1["value"], key)
                _require_same_dual_fia_tensor("atten_mask",
                                              split0["attn_mask"],
                                              split1["attn_mask"], key)
                if (split0["block_size"] != split1["block_size"]
                        or split0["num_kv_heads"] != split1["num_kv_heads"]
                        or split0["num_heads"] != split1["num_heads"]
                        or split0["scale"] != split1["scale"]):
                    raise RuntimeError(
                        "PTA dual-FIA update requires both splits to share "
                        f"FIA scalar attrs for layer {key!s}")

                split_start_0, split_graph_tokens_0 = split_ranges[0]
                split_start_1, split_graph_tokens_1 = split_ranges[1]
                torch.npu.dual_fused_infer_attention_score_update(
                    update_stream,
                    handles,
                    query,
                    split0["key_cache"],
                    split0["value"],
                    attn_output,
                    block_table_0=split0["block_tables"],
                    block_table_1=split1["block_tables"],
                    actual_seq_lengths_0=split0["actual_seq_lengths_q"],
                    actual_seq_lengths_1=split1["actual_seq_lengths_q"],
                    actual_seq_lengths_kv_0=split0["seq_lens"],
                    actual_seq_lengths_kv_1=split1["seq_lens"],
                    split_start_0=int(split_start_0),
                    split_graph_tokens_0=int(split_graph_tokens_0),
                    split_start_1=int(split_start_1),
                    split_graph_tokens_1=int(split_graph_tokens_1),
                    atten_mask=split0["attn_mask"],
                    workspace_0=split0["workspace"],
                    workspace_1=split1["workspace"],
                    softmax_lse_0=split0["softmax_lse"],
                    softmax_lse_1=split1["softmax_lse"],
                    num_heads=split0["num_heads"],
                    scale=split0["scale"],
                    block_size=split0["block_size"],
                    num_key_value_heads=split0["num_kv_heads"],
                    sparse_mode=3,
                    input_layout="TND",
                    softmax_lse_flag=False,
                )
                for event in update_events:
                    event.record(update_stream)
                continue

            if param[0] != "dual_stream_fia":
                raise RuntimeError(
                    "GraphParams entry is not a dual-stream FIA attention "
                    f"entry for layer {key!s}")
            split_params = param[1]
            if len(split_params) != 2 or len(handles) != 2 or len(events) != 2:
                raise RuntimeError(
                    "dual-stream FIA GraphParams entry must contain exactly "
                    f"two splits for layer {key!s}")
            if (_ACL_GRAPH_UPDATE_PARAM_DIAG
                    and layer_idx in (0, len(attn_items) - 1)):
                split_debug.log_event(
                    "acl_graph_dual_fia_update_handles",
                    {
                        "key": key,
                        "layer_idx": int(layer_idx),
                        "runtime_shape": runtime_shape,
                        "graph_param_key": graph_param_key_info(param_key),
                        "in_parallel_streams": bool(in_parallel_streams),
                        "param_kind": "dual_stream_fia",
                        "handle_id": _object_id_info(handles),
                        "event_id": _object_id_info(events),
                    },
                    step_id=getattr(forward_context,
                                    "split_inplace_debug_step_id", None),
                )

            for split_idx, split_param in enumerate(split_params):
                (query, key_cache, value, block_tables, attn_mask, block_size,
                 seq_lens, query_start_loc, num_kv_heads, num_heads, scale,
                 attn_output, softmax_lse, workspace_key) = split_param

                metadata = dual_metadata[split_idx][key]
                runtime_metadata = _get_dual_attention_update_metadata(
                    forward_context, dual_metadata, split_idx, key)
                seq_lens = maybe_template_fia_seq_lens(
                    forward_context,
                    getattr(runtime_metadata, "seq_lens_list",
                            metadata.seq_lens_list),
                    _get_fia_key_t(key_cache, block_size),
                    source=f"acl_graph_update_dual:{key}:{split_idx}")
                actual_seq_lengths_q = getattr(runtime_metadata,
                                               "actual_seq_lengths_q",
                                               metadata.actual_seq_lengths_q)
                metadata_block_table, metadata_block_source = (
                    _extract_block_table_from_metadata(runtime_metadata))
                block_table_refreshed = False
                if refresh_block_table:
                    block_table_refreshed = _refresh_block_table_in_place(
                        block_tables, metadata_block_table)
                    _log_block_table_refresh_diag(
                        attn_impl=f"dual_fia:{split_idx}",
                        key=key,
                        forward_context=forward_context,
                        runtime_shape=runtime_shape,
                        in_parallel_streams=in_parallel_streams,
                        metadata_block_source=metadata_block_source,
                        graph_block_table=block_tables,
                        metadata_block_table=metadata_block_table,
                        block_table_refreshed=block_table_refreshed,
                    )

                torch.npu.graph_task_update_begin(update_stream,
                                                  handles[split_idx])
                torch_npu.npu_fused_infer_attention_score.out(
                    query=query,
                    key=key_cache,
                    value=value,
                    block_table=block_tables,
                    atten_mask=attn_mask,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=actual_seq_lengths_q,
                    actual_seq_lengths_kv=seq_lens,
                    num_key_value_heads=num_kv_heads,
                    num_heads=num_heads,
                    scale=scale,
                    sparse_mode=3,
                    workspace=graph_params.workspaces.get(workspace_key),
                    out=[attn_output, softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)
                events[split_idx].record(update_stream)


def update_attn_params(update_stream, forward_context, runtime_shape,
                       vllm_config, in_parallel_streams: bool = False):
    if _has_dual_stream_attention_metadata(forward_context):
        _update_attn_dual_fia_params(update_stream, forward_context,
                                     runtime_shape,
                                     in_parallel_streams=in_parallel_streams)
    elif using_paged_attention(runtime_shape, vllm_config, forward_context):
        _update_attn_pa_params(update_stream, forward_context, runtime_shape,
                               in_parallel_streams=in_parallel_streams)
    else:
        _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                                in_parallel_streams=in_parallel_streams)


def update_attn_params_split(update_stream, forward_context,
                             runtime_shape, vllm_config,
                             in_parallel_streams: bool = False):
    """Split-only attn update with block_table in-place refresh enabled."""
    if _has_dual_stream_attention_metadata(forward_context):
        _update_attn_dual_fia_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )
    elif using_paged_attention(runtime_shape, vllm_config, forward_context):
        _update_attn_pa_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )
    else:
        _update_attn_fia_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )


def update_mla_attn_params(update_stream, forward_context, runtime_shape,
                           speculative_config,
                           in_parallel_streams: bool = False):
    if forward_context.is_mtp_model:
        graph_params = get_mtp_graph_params()
        param_key = runtime_shape
    else:
        graph_params = get_graph_params(in_parallel_streams)
        param_key = get_graph_param_key(forward_context, runtime_shape)
    require_graph_param_key(graph_params, param_key, op="update_mla_attn_params")
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ):
            (q_nope, k_nope, q_pe, k_pe, num_heads, num_kv_heads, input_layout,
             spec_attn_mask, sparse_mode, scale, block_table, block_size,
             seq_lens_list, actual_seq_lengths, attn_output,
             softmax_lse) = param
            seq_lens_list = forward_context.attn_metadata[
                key].decode.seq_lens_list
            if speculative_config and speculative_config.method == "mtp" \
                    and not forward_context.is_mtp_model:
                actual_seq_lengths = forward_context.attn_metadata[
                    key].decode.actual_seq_lengths_q
                spec_multiple = speculative_config.num_speculative_tokens + 1
                seq_lens_list = seq_lens_list + [0] * (
                    runtime_shape // spec_multiple - len(seq_lens_list))
                actual_seq_lengths = [
                    spec_multiple * (i + 1)
                    for i in range(runtime_shape // spec_multiple)
                ]
            elif forward_context.is_mtp_model:
                actual_seq_lengths = forward_context.attn_metadata[
                    key].decode.actual_seq_lengths_q
                block_table = forward_context.attn_metadata[
                    key].decode.block_table
                # TODO: This is a hack and should be fixed in the future.
                if speculative_config.disable_padded_drafter_batch:
                    block_table = block_table[:len(actual_seq_lengths)]
                seq_lens_list = seq_lens_list + [0] * (
                    len(actual_seq_lengths) - len(seq_lens_list))
            else:
                seq_lens_list = seq_lens_list + [0] * (runtime_shape -
                                                       len(seq_lens_list))
            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout=input_layout,
                atten_mask=spec_attn_mask,
                sparse_mode=sparse_mode,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=seq_lens_list,
                actual_seq_lengths=actual_seq_lengths,
                workspace=graph_params.workspaces.get(param_key),
                out=[attn_output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_attn_dcp_pcp_params(update_stream, forward_context, runtime_shape,
                               in_parallel_streams: bool = False):
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    graph_params = get_graph_params(in_parallel_streams)
    param_key = get_graph_param_key(forward_context, runtime_shape)
    require_graph_param_key(graph_params, param_key,
                            op="update_attn_dcp_pcp_params")
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ):
            (q_nope, k_nope, value, num_heads, num_kv_heads, scale,
             block_table, block_size, actual_seq_lengths_kv,
             actual_seq_lengths_q, attn_output, softmax_lse, dcp_size,
             pcp_rank, dcp_rank) = param
            attn_metadata = forward_context.attn_metadata[key]
            actual_seq_lengths_kv = attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:,
                                                                                             pcp_rank,
                                                                                             dcp_rank]
            pad_length = runtime_shape - len(actual_seq_lengths_kv)
            if pad_length > 0:
                pad_tensor = np.zeros(pad_length,
                                      dtype=actual_seq_lengths_kv.dtype)
                actual_seq_lengths_kv = np.concatenate(
                    [actual_seq_lengths_kv, pad_tensor])

            actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[:
                                                                      attn_metadata
                                                                      .
                                                                      num_decode_tokens]
            if (runtime_shape - len(actual_seq_lengths_q)):
                actual_seq_lengths_q = actual_seq_lengths_q + [
                    actual_seq_lengths_q[-1]
                ] * (runtime_shape - len(actual_seq_lengths_q))
            if dcp_size > 1:
                num_heads = num_heads * dcp_size

            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                value,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths=actual_seq_lengths_q,
                workspace=graph_params.workspaces.get(param_key),
                out=[attn_output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_mla_attn_dcp_pcp_params(update_stream, forward_context,
                                   runtime_shape,
                                   in_parallel_streams: bool = False):
    graph_params = get_graph_params(in_parallel_streams)
    param_key = get_graph_param_key(forward_context, runtime_shape)
    require_graph_param_key(graph_params, param_key,
                            op="update_mla_attn_dcp_pcp_params")
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ):
            (q_nope, q_pe, k_nope, k_pe, block_table, seq_len, num_heads,
             scale, num_kv_heads, attn_output, softmax_lse) = param

            decode_meta = forward_context.attn_metadata[key].decode
            seq_len = decode_meta.cp_seq_len

            # For pcp + spec decode, we flatten seq_lens
            # to avoid irregular spec_attn_mask shape,
            # so there's no need to divide runtime_shape by spec_multiple
            pad_length = runtime_shape - len(seq_len)
            pad_tensor = torch.zeros(pad_length,
                                     dtype=seq_len.dtype,
                                     device=seq_len.device)
            seq_len = torch.cat([seq_len, pad_tensor], dim=0)

            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.atb.npu_multi_head_latent_attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                block_table,
                seq_len,
                num_heads,
                scale,
                num_kv_heads,
                return_lse=True,
                calc_type="calc_type_ring",
                workspace=graph_params.workspaces.get(param_key),
                output=attn_output,
                lse=softmax_lse)
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


@dataclass
class GraphParams:
    events: dict[GraphParamKey, list[Any]]
    workspaces: dict[GraphParamKey, torch.Tensor | None]
    handles: dict[GraphParamKey, list[Any]]
    attn_params: dict[GraphParamKey, list[tuple]]


def ensure_graph_param_key(graph_params: Optional[GraphParams],
                           key: GraphParamKey) -> None:
    if graph_params is None:
        return
    graph_params.events.setdefault(key, [])
    graph_params.handles.setdefault(key, [])
    graph_params.attn_params.setdefault(key, [])
    graph_params.workspaces.setdefault(key, None)


def require_graph_param_key(graph_params: Optional[GraphParams],
                            key: GraphParamKey,
                            *,
                            op: str) -> None:
    if graph_params is None:
        raise KeyError(f"Missing GraphParams for {op}: {key!r}")
    if key not in graph_params.attn_params:
        raise KeyError(f"Missing GraphParams key for {op}: {key!r}")
    if key not in graph_params.handles or key not in graph_params.events:
        raise KeyError(f"Incomplete GraphParams key for {op}: {key!r}")


_graph_params: Optional[GraphParams] = None
_graph_params_parallel: Optional[GraphParams] = None


def _make_graph_params(aclgraph_capture_sizes: list[int]) -> GraphParams:
    return GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = _make_graph_params(aclgraph_capture_sizes)


def set_graph_params_parallel(aclgraph_capture_sizes: list[int]):
    """Initialize a separate GraphParams for the parallel stream.

    The parallel stream runs concurrently with the main stream, so it needs
    its own GraphParams to avoid races when both streams call
    graph_task_update_begin/end at the same time.
    """
    global _graph_params_parallel
    if _graph_params_parallel is not None:
        raise ValueError("Parallel graph parameters have already been set!")
    _graph_params_parallel = _make_graph_params(aclgraph_capture_sizes)


def update_graph_params_workspaces(key_or_num_tokens: GraphParamKey,
                                   workspace: torch.Tensor,
                                   in_parallel_streams: bool = False):
    global _graph_params, _graph_params_parallel
    target = _graph_params_parallel if in_parallel_streams else _graph_params
    if target is not None:
        ensure_graph_param_key(target, key_or_num_tokens)
        target.workspaces[key_or_num_tokens] = weak_ref_tensors(workspace)


def get_graph_params(in_parallel_streams: bool = False) -> Optional[GraphParams]:
    """Return the GraphParams for the given stream context.

    When *in_parallel_streams* is True, return the parallel-stream params so
    that concurrent graph_task_update calls never share the same handles/events.
    Falls back to the main params if the parallel params have not been
    initialised yet (e.g. during capture warm-up).
    """
    if in_parallel_streams and _graph_params_parallel is not None:
        return _graph_params_parallel
    return _graph_params


_mtp_graph_params: Optional[GraphParams] = None


def set_mtp_graph_params(aclgraph_capture_sizes: list[int]):
    global _mtp_graph_params
    if _mtp_graph_params is not None:
        raise ValueError("MTPGraph parameters have already been set!")
    _mtp_graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
    )


def update_mtp_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _mtp_graph_params
    if _mtp_graph_params is not None:
        _mtp_graph_params.workspaces[num_tokens] = workspace


def get_mtp_graph_params():
    return _mtp_graph_params
