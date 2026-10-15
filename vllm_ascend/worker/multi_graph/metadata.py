from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import torch

from vllm_ascend.worker.multi_graph.types import SlicePlan


def metadata_values(source: Any, part: SlicePlan) -> dict[str, Any]:
    """Slice standard decode metadata without changing physical KV addresses."""
    start, stop = part.request_start, part.request_stop
    count = part.actual_reqs
    if source.num_prefills or source.num_decodes < stop or source.max_query_len != 1:
        raise ValueError("metadata does not describe uniform decode")
    if source.reshape_cache_event is not None or source.pcp_local_num_input_tokens is not None:
        raise ValueError("metadata contains unsupported stream/context-parallel state")
    if source.qfa_metadata_cache:
        raise ValueError("quantized metadata is unsupported")
    values: dict[str, Any] = {}
    request_fields = {"seq_lens", "seq_lens_cpu", "seq_lens_gpu", "block_tables"}
    boundary_fields = {"query_start_loc", "query_start_loc_gpu"}
    for field in fields(source):
        name = field.name
        value = getattr(source, name)
        if name in request_fields and value is not None:
            values[name] = value[start:stop]
        elif name in boundary_fields and value is not None:
            values[name] = torch.arange(count + 1, device=value.device, dtype=value.dtype)
        elif name == "slot_mapping":
            values[name] = value[part.token_start : part.token_stop]
        elif name == "seq_lens_list":
            values[name] = value[start:stop]
        elif name == "actual_seq_lengths_q":
            values[name] = list(range(1, count + 1))
        elif name in ("num_actual_tokens", "num_decode_tokens", "num_decodes"):
            values[name] = count
        elif name == "qfa_metadata_cache":
            values[name] = {}
        else:
            values[name] = value
    return values


def clone_metadata(source: Any, part: SlicePlan) -> Any:
    values = metadata_values(source, part)
    return replace(
        source, **{name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in values.items()}
    )


def validate_metadata_refresh(target: Any, source: Any, part: SlicePlan) -> dict[str, Any]:
    """Validate the complete update before mutating captured addresses."""
    if type(target) is not type(source):
        raise ValueError("attention metadata schema changed")
    values = metadata_values(source, part)
    for name, value in values.items():
        captured = getattr(target, name)
        if isinstance(captured, torch.Tensor):
            if not isinstance(value, torch.Tensor) or (captured.shape, captured.dtype, captured.device) != (
                value.shape,
                value.dtype,
                value.device,
            ):
                raise ValueError(f"captured metadata tensor layout changed: {name}")
        elif isinstance(captured, list):
            if not isinstance(value, list) or len(captured) != len(value):
                raise ValueError(f"captured metadata list layout changed: {name}")
        elif not isinstance(captured, dict) and captured != value:
            raise ValueError(f"captured metadata scalar changed: {name}")
    return values


def refresh_metadata(target: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        captured = getattr(target, name)
        if isinstance(captured, torch.Tensor):
            captured.copy_(value, non_blocking=True)
        elif isinstance(captured, list):
            captured[:] = value
        elif isinstance(captured, dict):
            captured.clear()
