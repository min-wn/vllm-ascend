from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from vllm_ascend.worker.multi_graph.manager import MultiGraphManager
from vllm_ascend.worker.multi_graph.request_plan import RequestBatch, mixed_request_slices
from vllm_ascend.worker.multi_graph.types import ExecutionPlan


@dataclass(frozen=True)
class V2PreparedBatch:
    request_ids: tuple[str, ...]
    summary: RequestBatch
    input_ids: torch.Tensor
    positions: torch.Tensor


def normalize_v2_batch(input_batch: Any) -> V2PreparedBatch:
    """Normalize the existing AscendInputBatch CPU request boundaries.

    V2 scheduled token upper bounds and device metadata are not authoritative
    actual lengths. Only exact CPU query boundaries can enter this bridge.
    """
    count = input_batch.num_reqs
    if type(count) is not int or count < 0 or count != len(input_batch.req_ids):
        raise ValueError("V2 request count does not match IDs")
    raw_boundaries = input_batch.query_start_loc_np[: count + 1]
    raw_lengths = input_batch.seq_lens_np[:count]
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in (*raw_boundaries, *raw_lengths)):
        raise ValueError("V2 CPU lengths must be integral")
    boundaries = tuple(int(value) for value in raw_boundaries)
    if len(boundaries) != count + 1 or boundaries[0] != 0 or boundaries[-1] != input_batch.num_tokens:
        raise ValueError("V2 query boundaries do not describe actual tokens")
    queries = tuple(stop - start for start, stop in zip(boundaries, boundaries[1:]))
    lengths = tuple(int(value) for value in raw_lengths)
    prefills = tuple(bool(value) for value in input_batch.is_prefilling_np[:count])
    summary = RequestBatch(queries, lengths, prefills)
    if input_batch.num_draft_tokens:
        raise ValueError("V2 bridge does not support speculative tokens")
    for tensor in (input_batch.input_ids, input_batch.positions):
        if tensor.ndim != 1 or tensor.shape[0] < input_batch.num_tokens:
            raise ValueError("V2 bridge requires flat token buffers")
    ids = tuple(input_batch.req_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("V2 request IDs must be unique")
    return V2PreparedBatch(ids, summary, input_batch.input_ids, input_batch.positions)


class MultiGraphV2Bridge:
    """Execute an externally captured exact plan against normalized V2 inputs.

    Capture and native V2 metadata conversion are supplied by the integration
    owner. This bridge does not reuse V1 capture hooks or enable V2 at startup.
    """

    def __init__(self, manager: MultiGraphManager):
        self.manager = manager

    def run(self, batch: Any, metadata: dict[str, Any], plan: ExecutionPlan) -> torch.Tensor:
        prepared = normalize_v2_batch(batch)
        runs = mixed_request_slices(prepared.summary)
        if len(runs) != 1 or runs[0].workload != "decode" or plan.num_requests != len(prepared.request_ids):
            raise ValueError("V2 exact bridge requires one-token decode for every actual request")
        return self.manager.run(plan, prepared.input_ids, prepared.positions, metadata)
