# SPDX-License-Identifier: Apache-2.0

"""JSONL diagnostics for split-batch inplace migration.

The helpers in this module are intentionally cheap when disabled. Callers
should still guard expensive payload construction with is_enabled() when the
payload needs non-trivial work.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from itertools import count
from typing import Any, Optional

import torch
import torch.distributed as dist
from vllm.logger import logger

_DEBUG_ENV = "VLLM_ASCEND_SPLIT_INPLACE_DEBUG"
_DEBUG_FILE_ENV = "VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"
_DEFAULT_DEBUG_FILE = "/tmp/vllm_ascend_inplace_split.jsonl"
_step_counter = count(1)
_current_step_id: Optional[int] = None


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "0") in ("1", "true", "True", "yes", "YES")


def is_enabled() -> bool:
    return _env_truthy(_DEBUG_ENV)


def next_step_id() -> int:
    return next(_step_counter)


def set_current_step_id(step_id: Optional[int]) -> None:
    global _current_step_id
    _current_step_id = step_id


def get_current_step_id() -> Optional[int]:
    return _current_step_id


def _rank() -> Optional[int]:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    raw_rank = os.environ.get("RANK")
    if raw_rank is None:
        return None
    try:
        return int(raw_rank)
    except ValueError:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def log_event(event: str,
              payload: Optional[dict[str, Any]] = None,
              *,
              step_id: Optional[int] = None) -> None:
    if not is_enabled():
        return
    if step_id is None:
        step_id = get_current_step_id()

    record: dict[str, Any] = {
        "event": event,
        "ts_ns": time.time_ns(),
        "rank": _rank(),
        "pid": os.getpid(),
        "step_id": step_id,
    }
    if payload:
        record.update(payload)

    path = os.environ.get(_DEBUG_FILE_ENV, _DEFAULT_DEBUG_FILE)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
    except Exception as exc:
        logger.warning("Failed to write split inplace debug file %s: %s",
                       path, exc)


def tensor_info(tensor: Any) -> Optional[dict[str, Any]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        ptr = int(tensor.data_ptr())
    except Exception:
        ptr = None
    try:
        stride = list(tensor.stride())
    except Exception:
        stride = None
    return {
        "ptr": ptr,
        "shape": list(tensor.shape),
        "ndim": int(tensor.ndim),
        "dtype": str(tensor.dtype),
        "stride": stride,
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
    }


def tensor_view_info(tensor: Any) -> Optional[dict[str, Any]]:
    info = tensor_info(tensor)
    if info is None:
        return None
    info["data_ptr"] = info.pop("ptr")
    return info


def slice_info(value: slice) -> list[Optional[int]]:
    return [value.start, value.stop, value.step]


def batch_descriptor_info(batch_descriptor: Any) -> Any:
    if batch_descriptor is None:
        return None
    if dataclasses.is_dataclass(batch_descriptor):
        try:
            return dataclasses.asdict(batch_descriptor)
        except Exception:
            pass
    info: dict[str, Any] = {}
    for name in ("num_tokens", "num_reqs", "uniform", "has_lora",
                 "start_num_tokens", "graph_variant", "attention_backend",
                 "capture_metadata_mode"):
        if hasattr(batch_descriptor, name):
            info[name] = getattr(batch_descriptor, name)
    return info or str(batch_descriptor)


def split_slices_info(split_slices: Any) -> list[dict[str, Any]]:
    if not split_slices:
        return []
    result = []
    for idx, split_slice in enumerate(split_slices):
        request_slice = getattr(split_slice, "request_slice", slice(0, 0))
        token_slice = getattr(split_slice, "token_slice", slice(0, 0))
        result.append({
            "idx": idx,
            "request_start": request_slice.start,
            "request_stop": request_slice.stop,
            "token_start": token_slice.start,
            "token_stop": token_slice.stop,
            "num_requests": getattr(split_slice, "num_requests", None),
            "num_tokens": getattr(split_slice, "num_tokens", None),
            "padded_num_tokens": getattr(split_slice, "padded_num_tokens",
                                         None),
            "graph_num_tokens": getattr(split_slice, "graph_num_tokens",
                                        getattr(split_slice,
                                                "padded_num_tokens", None)),
            "start_num_tokens": getattr(split_slice, "start_num_tokens",
                                        None),
        })
    return result


def _iter_metadata_objects(attn_metadata: Any):
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            yield from _iter_metadata_objects(value)
        return
    if isinstance(attn_metadata, list):
        for value in attn_metadata:
            yield from _iter_metadata_objects(value)
        return
    if attn_metadata is not None:
        yield attn_metadata


def metadata_tensor_info(attn_metadata: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metadata_obj in _iter_metadata_objects(attn_metadata):
        common = getattr(metadata_obj, "common_attn_metadata", None)
        candidates = (metadata_obj, common) if common is not None else (
            metadata_obj, )
        for candidate in candidates:
            for attr in ("query_start_loc", "seq_lens",
                         "block_table_tensor", "block_tables",
                         "slot_mapping", "positions"):
                if attr in result:
                    continue
                info = tensor_info(getattr(candidate, attr, None))
                if info is not None:
                    result[attr] = info
        if result:
            break
    return result


def common_metadata_tensor_info(common_metadata: Any) -> dict[str, Any]:
    if common_metadata is None:
        return {}
    result: dict[str, Any] = {}
    for attr in ("query_start_loc", "query_start_loc_cpu", "seq_lens",
                 "seq_lens_cpu", "block_table_tensor", "slot_mapping",
                 "num_computed_tokens_cpu", "positions"):
        info = tensor_view_info(getattr(common_metadata, attr, None))
        if info is not None:
            result[attr] = info
    return result
