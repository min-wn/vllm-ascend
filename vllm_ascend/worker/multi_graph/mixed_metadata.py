from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import torch

from vllm_ascend.worker.multi_graph.request_plan import RequestBatch, RequestSlice, slice_query_boundaries


def slice_mixed_metadata(source: Any, batch: RequestBatch, part: RequestSlice, attention_state: Any) -> Any:
    """Build independent request-aligned metadata for a homogeneous run.

    The caller selects a backend-compatible attention state. A shared causal
    mask is preserved; model-specific masks and distributed metadata must be
    handled by their own adapter before using this component.
    """
    boundaries = slice_query_boundaries(batch, part)
    if source.reshape_cache_event is not None or source.pcp_local_num_input_tokens is not None:
        raise ValueError("mixed metadata cannot inherit stream or context-parallel state")
    if source.qfa_metadata_cache:
        raise ValueError("mixed metadata cannot inherit quantization scratch")
    queries = batch.query_lengths[part.start : part.stop]
    lengths = batch.kv_lengths[part.start : part.stop]
    prefills = batch.is_prefill[part.start : part.stop]
    if any(prefills) and not all(prefills):
        raise ValueError("metadata slice must be homogeneous")
    if tuple(source.seq_lens_list[part.start : part.stop]) != lengths:
        raise ValueError("mixed summary and metadata KV lengths disagree")
    values = {}
    for field in fields(source):
        name = field.name
        value = getattr(source, name)
        if name in ("seq_lens", "seq_lens_cpu", "seq_lens_gpu", "block_tables") and value is not None:
            if value.shape[0] < part.stop:
                raise ValueError("metadata request tensor is too short")
            value = value[part.start : part.stop].clone()
        elif name in ("query_start_loc", "query_start_loc_gpu") and value is not None:
            value = value.new_tensor(boundaries)
        elif name == "slot_mapping":
            if value.shape[0] < part.token_stop:
                raise ValueError("metadata slot mapping is too short")
            value = value[part.token_start : part.token_stop].clone()
        elif isinstance(value, torch.Tensor):
            value = value.clone()
        values[name] = value
    values.update(
        seq_lens_list=list(lengths),
        actual_seq_lengths_q=list(boundaries[1:]),
        num_actual_tokens=sum(queries),
        num_decode_tokens=sum(query for query, prefill in zip(queries, prefills) if not prefill),
        num_prefills=sum(prefills),
        num_decodes=len(prefills) - sum(prefills),
        max_query_len=max(queries),
        attn_state=attention_state,
        qfa_metadata_cache={},
    )
    return replace(source, **values)
