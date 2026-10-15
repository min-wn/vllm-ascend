from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from vllm_ascend.worker.multi_graph.types import ExecutionPlan


@dataclass(frozen=True)
class RankProposal:
    step: int
    topology: str
    request_digest: str
    lane_sizes: tuple[int, int] | None
    ready: bool
    execution: str = "serial"

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0 or not self.topology or not self.request_digest:
            raise ValueError("distributed proposal requires step, topology and request identity")
        if self.execution not in ("serial", "parallel") or type(self.ready) is not bool:
            raise ValueError("invalid distributed execution proposal")
        if self.lane_sizes is not None and (
            len(self.lane_sizes) != 2 or any(type(size) is not int or size <= 0 for size in self.lane_sizes)
        ):
            raise ValueError("distributed lane sizes must be positive")


def request_digest(req_ids: tuple[str, ...], query_lengths: tuple[int, ...], kv_lengths: tuple[int, ...]) -> str:
    if not len(req_ids) == len(query_lengths) == len(kv_lengths) or len(set(req_ids)) != len(req_ids):
        raise ValueError("distributed request summary must have unique ordered IDs and equal lengths")
    if any(not isinstance(req_id, str) or not req_id for req_id in req_ids) or any(
        type(query) is not int or query <= 0 or type(kv) is not int or kv < query
        for query, kv in zip(query_lengths, kv_lengths)
    ):
        raise ValueError("distributed request identities and lengths are invalid")
    return hashlib.sha256(json.dumps((req_ids, query_lengths, kv_lengths), separators=(",", ":")).encode()).hexdigest()


def agree(proposals: tuple[RankProposal, ...], world_size: int) -> tuple[int, int] | None:
    if world_size <= 0 or len(proposals) != world_size:
        raise ValueError("agreement requires every rank")
    first = proposals[0]
    if any(
        (proposal.step, proposal.topology, proposal.request_digest)
        != (first.step, first.topology, first.request_digest)
        for proposal in proposals
    ):
        raise RuntimeError("rank step/topology/request mismatch; cannot safely enter model collectives")
    if any(not proposal.ready or proposal.lane_sizes is None for proposal in proposals):
        return None
    if any(proposal.execution != "serial" for proposal in proposals):
        return None
    if any(proposal.lane_sizes != first.lane_sizes for proposal in proposals):
        return None
    return first.lane_sizes


class DistributedPlanCoordinator:
    """All ranks agree before submission; TP graph lanes execute in serial order.

    The transport must include every rank and have a finite timeout. Parallel
    lanes cannot share a collective communicator without verified ordering.
    A post-agreement device failure is fatal and must never cause local retry.
    """

    def __init__(self, world_size: int, gather: Callable[[RankProposal], tuple[RankProposal, ...]]):
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.world_size = world_size
        self.gather = gather
        self.next_step = 0
        self.failed = False

    def coordinate(self, proposal: RankProposal, plan: ExecutionPlan | None) -> ExecutionPlan | None:
        if self.failed or proposal.step != self.next_step:
            raise RuntimeError("coordinator failed or step is out of order")
        if plan is not None and (
            tuple(part.actual_reqs for part in plan.slices) != proposal.lane_sizes
            or plan.execution != proposal.execution
        ):
            raise ValueError("local proposal does not match the executable plan")
        if plan is None and proposal.ready:
            raise ValueError("ready rank requires a local plan")
        try:
            agreed = agree(self.gather(proposal), self.world_size)
        except Exception:
            self.failed = True
            raise
        self.next_step += 1
        return plan if agreed is not None else None


def torch_gloo_gather(proposal: RankProposal, group=None) -> tuple[RankProposal, ...]:
    """CPU protocol transport; model HCCL integration requires separate validation."""
    import torch.distributed as distributed

    if distributed.get_backend(group) != "gloo":
        raise ValueError("offline protocol transport requires a Gloo group")
    payloads = [None] * distributed.get_world_size(group)
    distributed.all_gather_object(payloads, asdict(proposal), group=group)
    return tuple(RankProposal(**payload) for payload in payloads)
