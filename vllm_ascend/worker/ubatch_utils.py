# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Iterable, Mapping, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatch_utils import (UBatchSlice, UBatchSlices,
                                         is_second_ubatch_empty,
                                         check_ubatch_thresholds)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import dbo_current_stream
from vllm_ascend.worker.npu_ubatch_wrapper import NPUCoreControlContextManager

logger = init_logger(__name__)


def should_ubatch_across_dp(
        should_ubatch: bool, orig_num_tokens_per_ubatch: int,
        padded_num_tokens_per_ubatch: int, dp_size: int,
        dp_rank: int) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]
    """

    device = current_platform.device_type
    tensor = torch.zeros(3, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0

    from vllm.distributed.parallel_state import get_dp_group
    dist.all_reduce(tensor, group=get_dp_group().device_group)

    result: bool = bool(torch.all(tensor[2] == 1).item())
    if not result:
        return result, None

    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_second_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens):
        logger.debug("Aborting ubatching %s %s", orig_min_num_tokens,
                     padded_max_num_tokens)
        return False, None
    return result, padded_num_tokens_tensor.cpu()


def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return should_ubatch_across_dp(should_ubatch, orig_num_tokens_per_ubatch,
                                   padded_num_tokens_per_ubatch, dp_size,
                                   dp_rank)


def get_dp_padding_ubatch(
        num_tokens_unpadded: int, num_tokens_padded: int,
        should_attempt_ubatching: bool,
        vllm_config: VllmConfig) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        # Early exit.
        tokens_per_ubatch = torch.tensor([num_tokens_padded // 2])
        #return False, None
        return True, tokens_per_ubatch

    # If this DP rank doesn't want to attempt microbatching
    if not should_attempt_ubatching:
        (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
            False, 0, 0, vllm_config)
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    # Round up to the next multiple of two for even divisibility
    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_second_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug(
            "Empty second µbatch detected: unpadded tokens: %s, padded "
            "tokens: %s", num_tokens_unpadded, num_tokens_padded)
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
        should_ubatch, num_tokens_unpadded // 2, num_tokens_per_ubatch,
        vllm_config)
    if not should_ubatch:
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    return should_ubatch, num_tokens_after_padding

def create_ubatch_slices(num_scheduled_tokens: np.ndarray, split_point: int, use_mla: bool) \
    -> UBatchSlices:
    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    if use_mla:
        first_ubatch_token_slice = slice(0, split_point)
        second_ubatch_token_slice = slice(split_point, cu_num_tokens[-1])

        first_ubatch_req_stop = int(
            np.searchsorted(cu_num_tokens, split_point, side="left"))
        second_ubatch_req_start = int(
            np.searchsorted(cu_num_tokens, split_point, side="right") - 1)
        # Determine request slices using exclusive stop semantics
        # First ubatch includes requests whose tokens overlap [0, split_point)
        first_ubatch_req_slice = slice(0, first_ubatch_req_stop)

        # Second ubatch starts at the request that contains the split_point
        # or the request starting exactly at split_point (if on boundary)
        second_ubatch_req_slice = slice(second_ubatch_req_start,
                                        len(cu_num_tokens) - 1)
    else:
        # currently split by requests
        second_ubatch_req_start = int(
            np.searchsorted(cu_num_tokens, split_point, side="right") - 1)
        first_ubatch_req_slice = slice(0, second_ubatch_req_start)
        second_ubatch_req_slice = slice(second_ubatch_req_start,
                                        len(cu_num_tokens) - 1)
        first_ubatch_token_slice = slice(
            0, cu_num_tokens[second_ubatch_req_start])
        second_ubatch_token_slice = slice(
            cu_num_tokens[second_ubatch_req_start], cu_num_tokens[-1])

    return [
        UBatchSlice(first_ubatch_req_slice, first_ubatch_token_slice),
        UBatchSlice(second_ubatch_req_slice, second_ubatch_token_slice)
    ]


def ubatch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    uniform_decode: bool,
    vllm_config: VllmConfig,
    moe_comm_type: Optional[MoECommType],
    use_mla: bool = True,
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to 
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if ubatch_slices is None
    ]

    """
    parallel_config = vllm_config.parallel_config
    # Don't bother with the should_ubatch handshaking unless microbatching
    # is enabled
    if not parallel_config.enable_dbo:
        return (None, None)

    # Check preconditions for microbatching
    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        uniform_decode=uniform_decode,
    )

    # Don't microbatch unless every other DP worker is also microbatching
    should_ubatch, num_tokens_after_padding = get_dp_padding_ubatch(
        num_tokens_unpadded,
        num_tokens_padded,
        should_attempt_ubatching,
        vllm_config,
    )

    if not should_ubatch or moe_comm_type == MoECommType.MC2:
        return (None, None)

    # This doesn't actually pad the ubatch slices. It just initializes the
    # split point to the padded value so that padding can be applied
    # to the second ubatch in pad_out_ubatch_slice after attention
    # metadata creation
    assert num_tokens_after_padding is not None
    token_split_point = int(num_tokens_after_padding[0].item())

    if not use_mla:
        cu_num_tokens = np.zeros(len(num_scheduled_tokens_per_request) + 1,
                                 dtype=np.int32)
        np.cumsum(num_scheduled_tokens_per_request,
                  dtype=np.int32,
                  out=cu_num_tokens[1:])

        split_point = int(
            np.searchsorted(cu_num_tokens, token_split_point, side="right") -
            1)
        imbalance_ratio = (token_split_point -
                           cu_num_tokens[split_point]) / cu_num_tokens[-1]
        if len(num_scheduled_tokens_per_request) == 1 or imbalance_ratio > 0.5:
            return (None, None)

    ubatch_slices = create_ubatch_slices(num_scheduled_tokens_per_request,
                                         token_split_point, use_mla)

    return (ubatch_slices, num_tokens_after_padding)


def create_core_control_context(aic_core: int, aiv_core: int):
    comm_aic_core = aic_core
    comm_aiv_core = aiv_core
    current_stream = dbo_current_stream()

    return NPUCoreControlContextManager(comm_aiv_core=comm_aiv_core,
                                        comm_aic_core=comm_aic_core,
                                        curren_stream=current_stream)


# ==================== Split Batch Support ====================
# Split batch is similar to DBO (ubatch) but with different conditions:
# - DBO: splits batch for overlapping compute/communication
# - Split batch: splits large decode batches for memory/performance optimization
# They never conflict because they are enabled under different conditions.

@dataclass
class SplitBatchSlice:
    """Represents a slice of the batch for split batch execution."""
    request_slice: slice  # Slice of requests
    token_slice: slice    # Slice of tokens
    # Padded token count aligned to the nearest cudagraph capture size.
    # Set by split_batch_split when cudagraph_capture_sizes is provided;
    # equals num_tokens otherwise.
    padded_num_tokens: int = 0
    # Runtime token offset for descriptor-aware inplace graphs. Existing
    # split-batch paths keep the default zero offset.
    start_num_tokens: int = 0
    # Optional request metadata capacity for bucketed graph replay. This is
    # independent of request_slice length so a captured graph can accept
    # runtime fake one-token padding requests without changing real requests.
    request_capacity: int = 0
    # Maximum runtime query length inside this split. Mixed-request macro
    # padding uses it to group token padding into bounded fake requests.
    max_query_len: int = 0

    def __post_init__(self):
        if self.padded_num_tokens == 0:
            self.padded_num_tokens = self.num_tokens
        if self.request_capacity == 0:
            self.request_capacity = self.num_requests
        if self.max_query_len == 0:
            self.max_query_len = max(
                1,
                (self.num_tokens + max(1, self.num_requests) - 1) //
                max(1, self.num_requests))

    @property
    def num_requests(self) -> int:
        return self.request_slice.stop - self.request_slice.start

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start

    @property
    def graph_num_tokens(self) -> int:
        return self.padded_num_tokens

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )


SplitBatchSlices = list[SplitBatchSlice]


INPLACE_SPLIT_DRY_RUN = "inplace_split_dry_run"
NO_SPLIT_EXACT_GRAPH_HIT = "no_split_exact_graph_hit"
NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE = "no_split_above_max_capture_size"
NO_SPLIT_NO_LOWER_CAPTURE_SIZE = "no_split_no_lower_capture_size"
NO_SPLIT_FIRST_NOT_REQUEST_ALIGNED = "no_split_first_not_request_aligned"
NO_SPLIT_SECOND_EMPTY = "no_split_second_empty"
NO_SPLIT_REMAINDER_TOO_LARGE = "no_split_remainder_too_large"
NO_SPLIT_NON_UNIFORM_DECODE = "no_split_non_uniform_decode"
NO_SPLIT_INVALID_QUERY_LEN = "no_split_invalid_query_len"
NO_SPLIT_NO_CAPTURE_SIZES = "no_split_no_capture_sizes"
NO_SPLIT_ATTENTION_BACKEND_MISMATCH = "no_split_attention_backend_mismatch"
NO_SPLIT_NO_OFFSET_CAPTURE_SIZE = "no_split_no_offset_capture_size"
NO_SPLIT_OFFSET_BUCKET_TOO_SMALL = "no_split_offset_bucket_too_small"
NO_SPLIT_OFFSET_PADDING_TOO_LARGE = "no_split_offset_padding_too_large"
NO_SPLIT_OFFSET_GRAPH_EXCEEDS_PADDED_BATCH = (
    "no_split_offset_graph_exceeds_padded_batch")
NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP = (
    "no_split_offset_graph_exceeds_start_cap")
NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE = (
    "no_split_offset_graph_below_min_size")
NO_SPLIT_INVALID_OFFSET_MATCH_POLICY = "no_split_invalid_offset_match_policy"
NO_SPLIT_INVALID_FIRST_TOKENS_POLICY = (
    "no_split_invalid_first_tokens_policy")
NO_SPLIT_MACRO_GRAPH_DISABLED = "no_split_macro_graph_disabled"
NO_SPLIT_MACRO_GRAPH_MISS = "no_split_macro_graph_miss"
NO_SPLIT_MACRO_GRAPH_INVALID_PLAN = "no_split_macro_graph_invalid_plan"
MIXED_REQUEST_SPLIT_DRY_RUN = "mixed_request_split_dry_run"
NO_SPLIT_MIXED_DISABLED = "no_split_mixed_disabled"
NO_SPLIT_MIXED_DECODE_ONLY = "no_split_mixed_decode_only"
NO_SPLIT_MIXED_BELOW_MIN_TOTAL_TOKENS = (
    "no_split_mixed_below_min_total_tokens")
NO_SPLIT_INVALID_TOKEN_ACCOUNTING = "no_split_invalid_token_accounting"
NO_SPLIT_SINGLE_REQUEST_DOMINATES = "no_split_single_request_dominates"
NO_SPLIT_TOO_FEW_REQUEST_BOUNDARIES = "no_split_too_few_request_boundaries"
NO_SPLIT_TOO_FEW_PREFILL_REQUESTS = "no_split_too_few_prefill_requests"
NO_SPLIT_MIN_TOKENS_PER_SPLIT = "no_split_min_tokens_per_split"
NO_SPLIT_GRAPH_BUCKET_MISSING = "no_split_graph_bucket_missing"
NO_SPLIT_PADDING_TOO_LARGE = "no_split_padding_too_large"
NO_SPLIT_MACRO_BUCKET_ACTUAL_TOKENS_TOO_SMALL = (
    "no_split_macro_bucket_actual_tokens_too_small")
NO_SPLIT_BUCKET_REQ_CAPACITY_EXCEEDED = (
    "no_split_bucket_req_capacity_exceeded")
NO_SPLIT_INVALID_MIXED_REQUEST_SPLIT_POLICY = (
    "no_split_invalid_mixed_request_split_policy")


@dataclass(frozen=True)
class InplaceSplitPlan:
    """Dry-run plan for 2-way inplace split execution."""
    split_slices: SplitBatchSlices
    reason: str
    total_num_tokens: int
    padded_num_tokens_without_split: int
    first_tokens: int
    second_tokens: int
    first_reqs: int
    second_reqs: int
    lower_capture_size: int
    remainder_tokens: int
    capture_sizes_considered: list[int]
    first_tokens_policy: str
    offset_match_policy: str
    second_actual_tokens: int
    second_graph_tokens: int
    second_padding_tokens: int
    offset_capture_sizes_considered: list[int]
    offset_min_graph_tokens: int
    offset_max_graph_tokens_by_start: Optional[dict[int, int]]
    offset_allowed_graph_tokens_by_start: Optional[dict[int, list[int]]]
    extra_debug_payload: Optional[dict[str, object]] = None

    def debug_payload(self) -> dict[str, object]:
        payload = {
            "reason": self.reason,
            "dry_run": True,
            "first_tokens": self.first_tokens,
            "second_tokens": self.second_tokens,
            "first_reqs": self.first_reqs,
            "second_reqs": self.second_reqs,
            "total_tokens": self.total_num_tokens,
            "padded_tokens_without_split":
            self.padded_num_tokens_without_split,
            "lower_capture_size": self.lower_capture_size,
            "remainder_tokens": self.remainder_tokens,
            "capture_sizes_considered": self.capture_sizes_considered,
            "first_tokens_policy": self.first_tokens_policy,
            "offset_match_policy": self.offset_match_policy,
            "second_actual_tokens": self.second_actual_tokens,
            "second_graph_tokens": self.second_graph_tokens,
            "second_padding_tokens": self.second_padding_tokens,
            "offset_capture_sizes_considered":
            self.offset_capture_sizes_considered,
            "offset_min_graph_tokens": self.offset_min_graph_tokens,
            "offset_max_graph_tokens_by_start":
            self.offset_max_graph_tokens_by_start,
            "offset_allowed_graph_tokens_by_start":
            self.offset_allowed_graph_tokens_by_start,
            "fallback_to": "no_split",
        }
        if self.extra_debug_payload:
            payload.update(self.extra_debug_payload)
        return payload


def inplace_split_preserves_attention_backend(
    inplace_split_plan: InplaceSplitPlan,
    uses_paged_attention: Callable[[int], bool],
) -> bool:
    """Return whether split graphs keep the unsplit attention backend.

    Some Ascend decode shapes route to paged attention while others route to
    fused infer attention. Inplace split must not silently change that routing,
    because graph capture records backend-specific task params.
    """
    unsplit_uses_pa = uses_paged_attention(
        inplace_split_plan.padded_num_tokens_without_split)
    return all(
        uses_paged_attention(split_slice.graph_num_tokens) == unsplit_uses_pa
        for split_slice in inplace_split_plan.split_slices)


def inplace_split_first_graph_matches_attention_backend(
    inplace_split_plan: InplaceSplitPlan,
    uses_paged_attention: Callable[[int], bool],
) -> bool:
    """Return whether split-0 can safely reuse its ordinary graph backend."""
    if not inplace_split_plan.split_slices:
        return False
    unsplit_uses_pa = uses_paged_attention(
        inplace_split_plan.padded_num_tokens_without_split)
    first_split = inplace_split_plan.split_slices[0]
    return uses_paged_attention(first_split.graph_num_tokens) == unsplit_uses_pa


def select_inplace_attention_backend(
    inplace_split_plan: InplaceSplitPlan,
    uses_paged_attention: Callable[[int], bool],
) -> str:
    return ("pa" if uses_paged_attention(
        inplace_split_plan.padded_num_tokens_without_split) else "fia")


def _ceil_to_capture_size(num_tokens: int,
                          capture_sizes: list[int]) -> Optional[int]:
    for size in capture_sizes:
        if size >= num_tokens:
            return size
    return None


def _normalize_capture_sizes(capture_sizes: Iterable[int],
                             query_len: int = 1) -> list[int]:
    q = int(query_len)
    return sorted({
        int(size)
        for size in capture_sizes
        if int(size) > 0 and (q <= 1 or int(size) % q == 0)
    })


def _normalize_offset_start_caps(
    caps: Optional[Mapping[int, int]],
) -> Optional[dict[int, int]]:
    if caps is None:
        return None
    return {int(start): int(max_tokens) for start, max_tokens in caps.items()}


def _max_graph_tokens_for_start(
    start_num_tokens: int,
    caps: Optional[dict[int, int]],
) -> Optional[int]:
    if not caps:
        return None
    matching_starts = [
        start for start in caps
        if int(start_num_tokens) >= int(start)
    ]
    if not matching_starts:
        return None
    return caps[max(matching_starts)]


def _normalize_offset_start_allowed_sizes(
    allowed_sizes: Optional[Mapping[int, Iterable[int]]],
) -> Optional[dict[int, list[int]]]:
    if allowed_sizes is None:
        return None
    return {
        int(start): sorted({int(size) for size in sizes})
        for start, sizes in allowed_sizes.items()
    }


def _allowed_graph_tokens_for_start(
    start_num_tokens: int,
    allowed_sizes: Optional[dict[int, list[int]]],
) -> Optional[set[int]]:
    if allowed_sizes is None:
        return None
    sizes = allowed_sizes.get(int(start_num_tokens))
    return set(sizes or [])


def _balanced_inplace_split_score(
    plan: InplaceSplitPlan,
) -> tuple[int, int, int, int]:
    """Rank split plans by graph workload balance for parallel replay."""
    return (
        max(plan.first_tokens, plan.second_graph_tokens),
        abs(plan.first_tokens - plan.second_graph_tokens),
        plan.second_padding_tokens,
        -plan.first_tokens,
    )


def create_inplace_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    uniform_decode_query_len: int,
    cudagraph_capture_sizes: Iterable[int],
    inplace_max_remainder_tokens: Optional[int] = None,
    *,
    offset_match_policy: str = "exact",
    offset_capture_sizes: Optional[Iterable[int]] = None,
    offset_min_graph_tokens: int = 1,
    offset_max_padding_tokens: Optional[int] = None,
    offset_max_padding_ratio: Optional[float] = None,
    offset_max_graph_tokens_by_start: Optional[Mapping[int, int]] = None,
    offset_allowed_graph_tokens_by_start: Optional[
        Mapping[int, Iterable[int]]] = None,
    first_tokens_policy: str = "largest_lower",
) -> tuple[Optional[InplaceSplitPlan], str]:
    """Create a dry-run 2-way inplace split plan.

    The default policy uses the largest lower full-graph capture size for the
    first split. The balanced policy evaluates all valid lower capture sizes and
    chooses the one with the most balanced graph workload. The second split
    records start_num_tokens so later phases can dispatch descriptor-aware
    offset graphs. By default the second split uses the real remainder; with
    offset_match_policy="bucket", it uses a padded bucket for graph dispatch
    while token_slice still covers only real tokens.
    """
    q = int(uniform_decode_query_len)
    if q <= 0:
        return None, NO_SPLIT_INVALID_QUERY_LEN
    if offset_match_policy not in ("exact", "bucket"):
        return None, NO_SPLIT_INVALID_OFFSET_MATCH_POLICY
    if first_tokens_policy not in ("largest_lower", "balanced"):
        return None, NO_SPLIT_INVALID_FIRST_TOKENS_POLICY
    offset_min_graph_tokens = max(1, int(offset_min_graph_tokens))

    num_reqs = len(num_scheduled_tokens_per_request)
    if (num_reqs == 0
            or int(np.sum(num_scheduled_tokens_per_request)) !=
            int(total_num_tokens)
            or not np.all(num_scheduled_tokens_per_request == q)):
        return None, NO_SPLIT_NON_UNIFORM_DECODE

    capture_sizes = _normalize_capture_sizes(cudagraph_capture_sizes)
    if not capture_sizes:
        return None, NO_SPLIT_NO_CAPTURE_SIZES

    total_tokens = int(total_num_tokens)
    max_capture_size = capture_sizes[-1]
    if total_tokens > max_capture_size:
        return None, NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE

    padded_without_split = _ceil_to_capture_size(total_tokens, capture_sizes)
    if padded_without_split is None:
        return None, NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE
    if padded_without_split == total_tokens:
        return None, NO_SPLIT_EXACT_GRAPH_HIT

    valid_lower_sizes = [
        size for size in capture_sizes
        if size < total_tokens and size % q == 0
    ]
    if not valid_lower_sizes:
        return None, NO_SPLIT_NO_LOWER_CAPTURE_SIZE

    offset_max_graph_tokens_by_start = _normalize_offset_start_caps(
        offset_max_graph_tokens_by_start)
    offset_allowed_graph_tokens_by_start = (
        _normalize_offset_start_allowed_sizes(
            offset_allowed_graph_tokens_by_start))
    if offset_match_policy == "bucket":
        raw_offset_capture_sizes = (
            offset_capture_sizes
            if offset_capture_sizes is not None else capture_sizes)
        offset_capture_sizes_considered = [
            size for size in _normalize_capture_sizes(
                raw_offset_capture_sizes, q)
            if size >= offset_min_graph_tokens
        ]
        if not offset_capture_sizes_considered:
            return None, NO_SPLIT_NO_OFFSET_CAPTURE_SIZE
    else:
        offset_capture_sizes_considered = capture_sizes

    last_reject_reason = NO_SPLIT_NO_LOWER_CAPTURE_SIZE
    candidate_plans: list[InplaceSplitPlan] = []
    for first_tokens in reversed(valid_lower_sizes):
        second_tokens = total_tokens - first_tokens
        first_reqs = first_tokens // q
        second_reqs = num_reqs - first_reqs

        if first_reqs * q != first_tokens:
            last_reject_reason = NO_SPLIT_FIRST_NOT_REQUEST_ALIGNED
            continue
        if second_tokens <= 0 or first_reqs <= 0 or second_reqs <= 0:
            last_reject_reason = NO_SPLIT_SECOND_EMPTY
            continue
        if second_tokens != second_reqs * q:
            last_reject_reason = NO_SPLIT_FIRST_NOT_REQUEST_ALIGNED
            continue
        if (inplace_max_remainder_tokens is not None
                and second_tokens > int(inplace_max_remainder_tokens)):
            last_reject_reason = NO_SPLIT_REMAINDER_TOO_LARGE
            continue

        second_graph_tokens = second_tokens
        if offset_match_policy == "bucket":
            candidate_offset_sizes = offset_capture_sizes_considered
            allowed_graph_tokens = _allowed_graph_tokens_for_start(
                first_tokens, offset_allowed_graph_tokens_by_start)
            if allowed_graph_tokens is not None:
                candidate_offset_sizes = [
                    size for size in candidate_offset_sizes
                    if size in allowed_graph_tokens
                ]
            max_graph_tokens = _max_graph_tokens_for_start(
                first_tokens, offset_max_graph_tokens_by_start)
            if max_graph_tokens is not None:
                candidate_offset_sizes = [
                    size for size in candidate_offset_sizes
                    if size <= int(max_graph_tokens)
                ]
            if not candidate_offset_sizes:
                last_reject_reason = NO_SPLIT_NO_OFFSET_CAPTURE_SIZE
                continue
            bucketed_second_tokens = _ceil_to_capture_size(
                second_tokens, candidate_offset_sizes)
            if bucketed_second_tokens is None:
                last_reject_reason = NO_SPLIT_OFFSET_BUCKET_TOO_SMALL
                continue
            second_graph_tokens = bucketed_second_tokens

        max_graph_tokens = _max_graph_tokens_for_start(
            first_tokens, offset_max_graph_tokens_by_start)
        allowed_graph_tokens = _allowed_graph_tokens_for_start(
            first_tokens, offset_allowed_graph_tokens_by_start)
        if (allowed_graph_tokens is not None
                and second_graph_tokens not in allowed_graph_tokens):
            last_reject_reason = NO_SPLIT_NO_OFFSET_CAPTURE_SIZE
            continue
        if second_graph_tokens < offset_min_graph_tokens:
            last_reject_reason = NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE
            continue
        if (max_graph_tokens is not None
                and second_graph_tokens > int(max_graph_tokens)):
            last_reject_reason = NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP
            continue

        second_padding_tokens = second_graph_tokens - second_tokens
        if (offset_max_padding_tokens is not None
                and second_padding_tokens > int(offset_max_padding_tokens)):
            last_reject_reason = NO_SPLIT_OFFSET_PADDING_TOO_LARGE
            continue
        if first_tokens + second_graph_tokens > padded_without_split:
            last_reject_reason = NO_SPLIT_OFFSET_GRAPH_EXCEEDS_PADDED_BATCH
            continue

        split_slices = [
            SplitBatchSlice(
                request_slice=slice(0, first_reqs),
                token_slice=slice(0, first_tokens),
                padded_num_tokens=first_tokens,
                start_num_tokens=0,
            ),
            SplitBatchSlice(
                request_slice=slice(first_reqs, num_reqs),
                token_slice=slice(first_tokens, total_tokens),
                padded_num_tokens=second_graph_tokens,
                start_num_tokens=first_tokens,
            ),
        ]

        plan = InplaceSplitPlan(
            split_slices=split_slices,
            reason=INPLACE_SPLIT_DRY_RUN,
            total_num_tokens=total_tokens,
            padded_num_tokens_without_split=padded_without_split,
            first_tokens=first_tokens,
            second_tokens=second_tokens,
            first_reqs=first_reqs,
            second_reqs=second_reqs,
            lower_capture_size=first_tokens,
            remainder_tokens=second_tokens,
            capture_sizes_considered=capture_sizes,
            first_tokens_policy=first_tokens_policy,
            offset_match_policy=offset_match_policy,
            second_actual_tokens=second_tokens,
            second_graph_tokens=second_graph_tokens,
            second_padding_tokens=second_padding_tokens,
            offset_capture_sizes_considered=offset_capture_sizes_considered,
            offset_min_graph_tokens=offset_min_graph_tokens,
            offset_max_graph_tokens_by_start=
            offset_max_graph_tokens_by_start,
            offset_allowed_graph_tokens_by_start=
            offset_allowed_graph_tokens_by_start,
        )
        if first_tokens_policy == "largest_lower":
            return plan, INPLACE_SPLIT_DRY_RUN
        candidate_plans.append(plan)

    if candidate_plans:
        return min(candidate_plans,
                   key=_balanced_inplace_split_score), INPLACE_SPLIT_DRY_RUN

    return None, last_reject_reason


def _mixed_attention_cost(req_tokens: np.ndarray, decode_threshold: int,
                          decode_weight: int,
                          prefill_weight: int) -> int:
    decode_tokens = int(req_tokens[req_tokens <= decode_threshold].sum())
    prefill_tokens = int(req_tokens[req_tokens > decode_threshold].sum())
    return decode_tokens * int(decode_weight) + prefill_tokens * int(
        prefill_weight)


def _macro_padding_req_chunks(padding_tokens: int,
                              req_tokens: np.ndarray) -> int:
    padding_tokens = int(padding_tokens)
    if padding_tokens <= 0:
        return 0
    max_query_len = 1
    if req_tokens.size > 0:
        max_query_len = max(1, int(req_tokens.max()))
    return (padding_tokens + max_query_len - 1) // max_query_len


def _mixed_request_split_score(
    candidate: tuple[InplaceSplitPlan, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    plan, left_cost, right_cost, left_padding, right_padding = candidate
    return (
        max(left_cost, right_cost),
        abs(left_cost - right_cost),
        left_padding + right_padding,
        max(plan.split_slices[0].graph_num_tokens,
            plan.split_slices[1].graph_num_tokens),
        abs(plan.first_tokens - plan.second_tokens),
    )


def _normalize_mixed_graph_bucket_plans(
    graph_bucket_plans: Optional[Iterable[tuple[int, int, int, int]]],
) -> list[tuple[int, int, int, int]]:
    if graph_bucket_plans is None:
        return []

    normalized: set[tuple[int, int, int, int]] = set()
    for plan in graph_bucket_plans:
        if len(plan) != 4:
            continue
        first_graph, second_graph, first_req_cap, second_req_cap = (
            int(plan[0]),
            int(plan[1]),
            int(plan[2]),
            int(plan[3]),
        )
        if (first_graph <= 0 or second_graph <= 0 or first_req_cap <= 0
                or second_req_cap <= 0):
            continue
        normalized.add(
            (first_graph, second_graph, first_req_cap, second_req_cap))
    return sorted(normalized)


def create_mixed_request_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    cudagraph_capture_sizes: Iterable[int],
    *,
    decode_threshold: int,
    min_total_tokens: int = 128,
    min_tokens_per_split: int = 64,
    max_single_request_ratio: float = 0.70,
    max_padding_tokens_per_split: Optional[int] = None,
    max_padding_ratio_per_split: Optional[float] = 0.0,
    min_prefill_reqs_for_prefill_split: int = 2,
    decode_weight: int = 1,
    prefill_weight: int = 4,
    split_policy: str = "balanced_attention",
    graph_bucket_plans: Optional[Iterable[tuple[int, int, int, int]]] = None,
    bucket_padding_ratio_grace_tokens: int = 0,
    bucket_min_actual_tokens_per_split: int = 1,
    debug_info: Optional[dict[str, object]] = None,
) -> tuple[Optional[InplaceSplitPlan], str]:
    """Create a dry-run 2-way request-boundary split plan for mixed batches.

    Unlike uniform decode inplace split, mixed request split uses compact
    per-split buffers. Therefore both split descriptors keep
    start_num_tokens=0 and do not rely on descriptor-aware offset graph keys.
    """
    def _finish(reason: str) -> tuple[None, str]:
        if debug_info is not None:
            debug_info["mixed_planner_stop_reason"] = reason
        return None, reason

    if split_policy != "balanced_attention":
        return _finish(NO_SPLIT_INVALID_MIXED_REQUEST_SPLIT_POLICY)

    tokens = np.asarray(num_scheduled_tokens_per_request, dtype=np.int32)
    total_tokens = int(total_num_tokens)
    if (tokens.ndim != 1 or len(tokens) == 0 or total_tokens <= 0
            or int(tokens.sum()) != total_tokens
            or np.any(tokens <= 0)):
        return _finish(NO_SPLIT_INVALID_TOKEN_ACCOUNTING)

    capture_sizes = _normalize_capture_sizes(cudagraph_capture_sizes)
    bucket_plans = _normalize_mixed_graph_bucket_plans(graph_bucket_plans)
    bucket_padding_ratio_grace_tokens = max(
        0, int(bucket_padding_ratio_grace_tokens))
    bucket_min_actual_tokens_per_split = max(
        1, int(bucket_min_actual_tokens_per_split))
    if debug_info is not None:
        token_values, token_counts = np.unique(tokens, return_counts=True)
        debug_info.update({
            "mixed_planner_total_tokens":
            total_tokens,
            "mixed_planner_num_reqs":
            int(len(tokens)),
            "mixed_planner_request_token_histogram": {
                str(int(token)): int(count)
                for token, count in zip(token_values, token_counts)
            },
            "mixed_planner_request_tokens":
            [int(token) for token in tokens.tolist()],
            "mixed_planner_capture_sizes_considered":
            [int(size) for size in capture_sizes],
            "mixed_planner_bucket_plans_considered": [
                [int(v) for v in plan] for plan in bucket_plans
            ],
            "mixed_planner_bucket_padding_ratio_grace_tokens":
            bucket_padding_ratio_grace_tokens,
            "mixed_planner_bucket_min_actual_tokens_per_split":
            bucket_min_actual_tokens_per_split,
        })
    if not capture_sizes and not bucket_plans:
        return _finish(NO_SPLIT_NO_CAPTURE_SIZES)

    if total_tokens < int(min_total_tokens):
        return _finish(NO_SPLIT_MIXED_BELOW_MIN_TOTAL_TOKENS)

    decode_threshold = int(decode_threshold)
    if decode_threshold < 1:
        return _finish(NO_SPLIT_INVALID_QUERY_LEN)

    decode_mask = tokens <= decode_threshold
    prefill_mask = ~decode_mask
    if not bool(prefill_mask.any()):
        return _finish(NO_SPLIT_MIXED_DECODE_ONLY)

    max_single_request_ratio = float(max_single_request_ratio)
    if max_single_request_ratio <= 0:
        return _finish(NO_SPLIT_SINGLE_REQUEST_DOMINATES)
    if float(int(tokens.max())) / float(total_tokens) > max_single_request_ratio:
        return _finish(NO_SPLIT_SINGLE_REQUEST_DOMINATES)

    num_decode_tokens = int(tokens[decode_mask].sum())
    num_prefill_tokens = int(tokens[prefill_mask].sum())
    num_decode_reqs = int(decode_mask.sum())
    num_prefill_reqs = int(prefill_mask.sum())
    if (num_prefill_reqs < int(min_prefill_reqs_for_prefill_split)
            and num_decode_tokens < int(min_tokens_per_split)):
        return _finish(NO_SPLIT_TOO_FEW_PREFILL_REQUESTS)

    if len(tokens) < 2:
        return _finish(NO_SPLIT_TOO_FEW_REQUEST_BOUNDARIES)

    if capture_sizes:
        padded_without_split = _ceil_to_capture_size(total_tokens,
                                                     capture_sizes)
        if padded_without_split is None:
            padded_without_split = total_tokens
    else:
        padded_without_split = total_tokens

    cu_tokens = np.concatenate(
        [np.array([0], dtype=np.int64),
         np.cumsum(tokens, dtype=np.int64)])
    last_reject_reason = NO_SPLIT_TOO_FEW_REQUEST_BOUNDARIES
    candidates: list[tuple[InplaceSplitPlan, int, int, int, int]] = []
    debug_reject_counts: dict[str, int] = {}
    debug_rejected_candidates: list[dict[str, object]] = []
    debug_actual_candidates: list[tuple[tuple[int, int, int, int, int],
                                        dict[str, object]]] = []

    def _count_reject(reason: str) -> None:
        debug_reject_counts[reason] = debug_reject_counts.get(reason, 0) + 1

    def _append_rejected_candidate(payload: dict[str, object]) -> None:
        if debug_info is None:
            return
        if len(debug_rejected_candidates) < 32:
            debug_rejected_candidates.append(payload)

    for split_req in range(1, len(tokens)):
        left_tokens = int(cu_tokens[split_req])
        right_tokens = total_tokens - left_tokens
        if left_tokens <= 0 or right_tokens <= 0:
            last_reject_reason = NO_SPLIT_TOO_FEW_REQUEST_BOUNDARIES
            _count_reject(last_reject_reason)
            continue
        if (left_tokens < int(min_tokens_per_split)
                or right_tokens < int(min_tokens_per_split)):
            last_reject_reason = NO_SPLIT_MIN_TOKENS_PER_SPLIT
            _count_reject(last_reject_reason)
            continue

        left_req_tokens = tokens[:split_req]
        right_req_tokens = tokens[split_req:]
        right_reqs = len(tokens) - split_req
        left_cost = _mixed_attention_cost(left_req_tokens, decode_threshold,
                                          decode_weight, prefill_weight)
        right_cost = _mixed_attention_cost(right_req_tokens, decode_threshold,
                                           decode_weight, prefill_weight)
        actual_candidate = {
            "split_req_index": int(split_req),
            "actual_tokens": [int(left_tokens), int(right_tokens)],
            "actual_reqs": [int(split_req), int(right_reqs)],
            "attention_cost": [int(left_cost), int(right_cost)],
        }
        debug_actual_candidates.append((
            (
                max(left_cost, right_cost),
                abs(left_cost - right_cost),
                abs(left_tokens - right_tokens),
                int(split_req),
                int(right_reqs),
            ),
            actual_candidate,
        ))

        graph_choices: list[tuple[int, int, Optional[tuple[int, int, int,
                                                          int]]]] = []
        if bucket_plans:
            if (left_tokens < bucket_min_actual_tokens_per_split
                    or right_tokens < bucket_min_actual_tokens_per_split):
                last_reject_reason = (
                    NO_SPLIT_MACRO_BUCKET_ACTUAL_TOKENS_TOO_SMALL)
                _count_reject(last_reject_reason)
                _append_rejected_candidate({
                    **actual_candidate,
                    "reason": last_reject_reason,
                    "bucket_min_actual_tokens_per_split":
                    bucket_min_actual_tokens_per_split,
                })
                continue
            for bucket_plan in bucket_plans:
                left_graph, right_graph, left_req_cap, right_req_cap = (
                    bucket_plan)
                left_padding = int(left_graph) - left_tokens
                right_padding = int(right_graph) - right_tokens
                if left_tokens <= left_graph and right_tokens <= right_graph:
                    graph_choices.append(
                        (left_graph, right_graph, bucket_plan))
            if not graph_choices:
                last_reject_reason = NO_SPLIT_GRAPH_BUCKET_MISSING
                _count_reject(last_reject_reason)
                _append_rejected_candidate({
                    **actual_candidate,
                    "reason": last_reject_reason,
                    "bucket_plans_considered_count": len(bucket_plans),
                })
                continue
        else:
            left_graph_tokens = _ceil_to_capture_size(left_tokens,
                                                      capture_sizes)
            right_graph_tokens = _ceil_to_capture_size(right_tokens,
                                                       capture_sizes)
            if left_graph_tokens is None or right_graph_tokens is None:
                last_reject_reason = NO_SPLIT_GRAPH_BUCKET_MISSING
                _count_reject(last_reject_reason)
                _append_rejected_candidate({
                    **actual_candidate,
                    "reason": last_reject_reason,
                })
                continue
            graph_choices.append(
                (int(left_graph_tokens), int(right_graph_tokens), None))

        for left_graph_tokens, right_graph_tokens, bucket_plan in graph_choices:
            left_padding = int(left_graph_tokens) - left_tokens
            right_padding = int(right_graph_tokens) - right_tokens
            reject_payload = {
                **actual_candidate,
                "graph_tokens":
                [int(left_graph_tokens),
                 int(right_graph_tokens)],
                "padding_tokens": [int(left_padding),
                                   int(right_padding)],
                "padding_ratios": [
                    float(left_padding / float(max(1, left_tokens))),
                    float(right_padding / float(max(1, right_tokens))),
                ],
            }
            if bucket_plan is not None:
                left_padding_reqs = _macro_padding_req_chunks(
                    left_padding, left_req_tokens)
                right_padding_reqs = _macro_padding_req_chunks(
                    right_padding, right_req_tokens)
                left_effective_reqs = split_req + left_padding_reqs
                right_effective_reqs = right_reqs + right_padding_reqs
                reject_payload["bucket_req_caps"] = [
                    int(bucket_plan[2]),
                    int(bucket_plan[3])
                ]
                reject_payload["effective_reqs"] = [
                    int(left_effective_reqs),
                    int(right_effective_reqs)
                ]
                reject_payload["padding_req_chunks"] = [
                    int(left_padding_reqs),
                    int(right_padding_reqs)
                ]
                reject_payload["actual_reqs"] = [
                    int(split_req),
                    int(right_reqs)
                ]
                reject_payload["metadata_pad_reqs"] = [
                    max(0, int(bucket_plan[2]) - int(left_effective_reqs)),
                    max(0, int(bucket_plan[3]) - int(right_effective_reqs)),
                ]
                reject_payload["req_capacity_ok"] = bool(
                    left_effective_reqs <= int(bucket_plan[2])
                    and right_effective_reqs <= int(bucket_plan[3]))
                if not reject_payload["req_capacity_ok"]:
                    last_reject_reason = (
                        NO_SPLIT_BUCKET_REQ_CAPACITY_EXCEEDED)
                    _count_reject(last_reject_reason)
                    reject_payload["reason"] = last_reject_reason
                    _append_rejected_candidate(reject_payload)
                    continue
            if (max_padding_tokens_per_split is not None
                    and (left_padding > int(max_padding_tokens_per_split)
                         or right_padding >
                         int(max_padding_tokens_per_split))):
                last_reject_reason = NO_SPLIT_PADDING_TOO_LARGE
                _count_reject(last_reject_reason)
                reject_payload["reason"] = last_reject_reason
                _append_rejected_candidate(reject_payload)
                continue
            if max_padding_ratio_per_split is not None:
                max_padding_ratio = float(max_padding_ratio_per_split)
                left_ratio_exempt = bool(
                    bucket_plan is not None and left_tokens <=
                    bucket_padding_ratio_grace_tokens)
                right_ratio_exempt = bool(
                    bucket_plan is not None and right_tokens <=
                    bucket_padding_ratio_grace_tokens)
                left_ratio_too_large = (
                    not left_ratio_exempt
                    and left_padding / float(left_tokens) > max_padding_ratio)
                right_ratio_too_large = (
                    not right_ratio_exempt
                    and right_padding / float(right_tokens) > max_padding_ratio)
                if left_ratio_too_large or right_ratio_too_large:
                    last_reject_reason = NO_SPLIT_PADDING_TOO_LARGE
                    _count_reject(last_reject_reason)
                    reject_payload["reason"] = last_reject_reason
                    reject_payload["padding_ratio_exempt"] = [
                        left_ratio_exempt,
                        right_ratio_exempt,
                    ]
                    reject_payload[
                        "bucket_padding_ratio_grace_tokens"] = (
                            bucket_padding_ratio_grace_tokens)
                    _append_rejected_candidate(reject_payload)
                    continue

            split_slices = [
                SplitBatchSlice(
                    request_slice=slice(0, split_req),
                    token_slice=slice(0, left_tokens),
                    padded_num_tokens=int(left_graph_tokens),
                    start_num_tokens=0,
                    request_capacity=(int(bucket_plan[2])
                                      if bucket_plan is not None else 0),
                    max_query_len=max(1, int(left_req_tokens.max())),
                ),
                SplitBatchSlice(
                    request_slice=slice(split_req, len(tokens)),
                    token_slice=slice(left_tokens, total_tokens),
                    padded_num_tokens=int(right_graph_tokens),
                    start_num_tokens=0,
                    request_capacity=(int(bucket_plan[3])
                                      if bucket_plan is not None else 0),
                    max_query_len=max(1, int(right_req_tokens.max())),
                ),
            ]
            extra_debug_payload = {
                "mixed_request_split": True,
                "split_policy": split_policy,
                "split_req_index": split_req,
                "num_decode_reqs": num_decode_reqs,
                "num_prefill_reqs": num_prefill_reqs,
                "num_decode_tokens": num_decode_tokens,
                "num_prefill_tokens": num_prefill_tokens,
                "first_graph_tokens": int(left_graph_tokens),
                "first_padding_tokens": left_padding,
                "first_attention_cost": left_cost,
                "second_attention_cost": right_cost,
                "max_single_request_ratio":
                float(max_single_request_ratio),
                "max_padding_ratio_per_split":
                (None if max_padding_ratio_per_split is None else
                 float(max_padding_ratio_per_split)),
            }
            if bucket_plan is not None:
                extra_debug_payload.update({
                    "macro_bucket_plan": True,
                    "macro_bucket_graph_tokens": [
                        int(left_graph_tokens),
                        int(right_graph_tokens)
                    ],
                    "macro_bucket_req_caps": [
                        int(bucket_plan[2]),
                        int(bucket_plan[3])
                    ],
                    "macro_bucket_effective_reqs": [
                        int(left_effective_reqs),
                        int(right_effective_reqs),
                    ],
                    "macro_bucket_padding_req_chunks": [
                        int(left_padding_reqs),
                        int(right_padding_reqs),
                    ],
                    "macro_bucket_actual_reqs": [
                        int(split_req),
                        int(right_reqs),
                    ],
                    "macro_bucket_metadata_pad_reqs": [
                        max(0,
                            int(bucket_plan[2]) - int(left_effective_reqs)),
                        max(0,
                            int(bucket_plan[3]) - int(right_effective_reqs)),
                    ],
                    "macro_bucket_padding_ratio_exempt": [
                        bool(left_tokens <=
                             bucket_padding_ratio_grace_tokens),
                        bool(right_tokens <=
                             bucket_padding_ratio_grace_tokens),
                    ],
                    "macro_bucket_padding_ratio_grace_tokens":
                    bucket_padding_ratio_grace_tokens,
                    "macro_bucket_min_actual_tokens_per_split":
                    bucket_min_actual_tokens_per_split,
                    "macro_bucket_plans_considered": [
                        list(plan) for plan in bucket_plans
                    ],
                })
            plan = InplaceSplitPlan(
                split_slices=split_slices,
                reason=MIXED_REQUEST_SPLIT_DRY_RUN,
                total_num_tokens=total_tokens,
                padded_num_tokens_without_split=padded_without_split,
                first_tokens=left_tokens,
                second_tokens=right_tokens,
                first_reqs=split_req,
                second_reqs=len(tokens) - split_req,
                lower_capture_size=int(left_graph_tokens),
                remainder_tokens=right_tokens,
                capture_sizes_considered=capture_sizes,
                first_tokens_policy=split_policy,
                offset_match_policy="compact",
                second_actual_tokens=right_tokens,
                second_graph_tokens=int(right_graph_tokens),
                second_padding_tokens=right_padding,
                offset_capture_sizes_considered=[],
                offset_min_graph_tokens=0,
                offset_max_graph_tokens_by_start=None,
                offset_allowed_graph_tokens_by_start=None,
                extra_debug_payload=extra_debug_payload,
            )
            candidates.append(
                (plan, left_cost, right_cost, left_padding, right_padding))

    if candidates:
        best = min(candidates, key=_mixed_request_split_score)[0]
        if debug_info is not None:
            debug_info.update({
                "mixed_planner_stop_reason":
                MIXED_REQUEST_SPLIT_DRY_RUN,
                "mixed_planner_candidate_count":
                len(candidates),
                "mixed_planner_reject_counts":
                debug_reject_counts,
                "mixed_planner_best_actual_candidates": [
                    candidate for _, candidate in sorted(
                        debug_actual_candidates, key=lambda item: item[0])[:8]
                ],
            })
        return best, MIXED_REQUEST_SPLIT_DRY_RUN
    if debug_info is not None:
        debug_info.update({
            "mixed_planner_stop_reason":
            last_reject_reason,
            "mixed_planner_candidate_count":
            0,
            "mixed_planner_reject_counts":
            debug_reject_counts,
            "mixed_planner_best_actual_candidates": [
                candidate for _, candidate in sorted(
                    debug_actual_candidates, key=lambda item: item[0])[:8]
            ],
            "mixed_planner_rejected_candidates":
            debug_rejected_candidates,
        })
    return None, last_reject_reason


def _round_up_to_multiple(value: int, alignment: int) -> int:
    alignment = max(1, int(alignment))
    return ((int(value) + alignment - 1) // alignment) * alignment


def _macro_capture_plan_to_inplace_plan(
    capture_plan,
    *,
    uniform_decode_query_len: int,
    capture_plans_considered: list[int],
    macro_graph_config,
    allow_mixed_request_plan: bool = False,
) -> tuple[Optional[InplaceSplitPlan], str]:
    q = int(uniform_decode_query_len)
    if q <= 0:
        return None, NO_SPLIT_INVALID_QUERY_LEN

    actual_tokens = tuple(int(v) for v in capture_plan.split_actual_tokens)
    graph_tokens = tuple(int(v) for v in capture_plan.split_graph_tokens)
    start_tokens = tuple(int(v) for v in capture_plan.split_start_tokens)
    split_num_reqs = getattr(capture_plan, "split_num_reqs", None)
    if split_num_reqs is not None:
        split_num_reqs = tuple(int(v) for v in split_num_reqs)
    split_req_caps = getattr(capture_plan, "split_req_caps", None)
    if split_req_caps is not None:
        split_req_caps = tuple(int(v) for v in split_req_caps)
    total_tokens = int(capture_plan.total_tokens)
    if len(actual_tokens) != 2 or len(graph_tokens) != 2 \
            or len(start_tokens) != 2:
        return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN
    if split_num_reqs is not None and len(split_num_reqs) != 2:
        return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN
    if split_req_caps is not None and len(split_req_caps) != 2:
        return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN
    if sum(actual_tokens) != total_tokens:
        return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN
    if split_num_reqs is None and any(tokens % q != 0
                                      for tokens in actual_tokens):
        return None, NO_SPLIT_FIRST_NOT_REQUEST_ALIGNED
    compact_start = start_tokens == (0, 0)
    offset_start = start_tokens == (0, actual_tokens[0])
    if not compact_start and not offset_start:
        return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN

    first_actual, second_actual = actual_tokens
    first_graph, second_graph = graph_tokens
    if split_num_reqs is None:
        first_reqs = first_actual // q
        second_reqs = second_actual // q
    else:
        first_reqs, second_reqs = split_num_reqs
        if (not allow_mixed_request_plan
                and (first_actual != first_reqs * q
                     or second_actual != second_reqs * q)):
            return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN
    if first_reqs <= 0 or second_reqs <= 0:
        return None, NO_SPLIT_SECOND_EMPTY
    if split_req_caps is None:
        first_req_cap = first_reqs + max(0, first_graph - first_actual)
        second_req_cap = second_reqs + max(0, second_graph - second_actual)
    else:
        first_req_cap, second_req_cap = split_req_caps
    if first_req_cap < first_reqs or second_req_cap < second_reqs:
        return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN

    split_slices = [
        SplitBatchSlice(
            request_slice=slice(0, first_reqs),
            token_slice=slice(0, first_actual),
            padded_num_tokens=first_graph,
            start_num_tokens=start_tokens[0],
            request_capacity=first_req_cap,
            max_query_len=max(1,
                              (first_actual + first_reqs - 1) // first_reqs),
        ),
        SplitBatchSlice(
            request_slice=slice(first_reqs, first_reqs + second_reqs),
            token_slice=slice(first_actual, total_tokens),
            padded_num_tokens=second_graph,
            start_num_tokens=start_tokens[1],
            request_capacity=second_req_cap,
            max_query_len=max(1,
                              (second_actual + second_reqs - 1) //
                              second_reqs),
        ),
    ]
    graph_total_tokens = first_graph + second_graph
    if graph_total_tokens > total_tokens:
        padded_without_split = graph_total_tokens + int(
            getattr(macro_graph_config, "min_padding_saved_tokens", 0))
    else:
        padded_without_split = total_tokens
    alignment = int(getattr(macro_graph_config, "graph_token_alignment", 1))
    padded_without_split = _round_up_to_multiple(
        padded_without_split, alignment)

    return InplaceSplitPlan(
        split_slices=split_slices,
        reason=INPLACE_SPLIT_DRY_RUN,
        total_num_tokens=total_tokens,
        padded_num_tokens_without_split=padded_without_split,
        first_tokens=first_actual,
        second_tokens=second_actual,
        first_reqs=first_reqs,
        second_reqs=second_reqs,
        lower_capture_size=first_graph,
        remainder_tokens=second_actual,
        capture_sizes_considered=capture_plans_considered,
        first_tokens_policy="macro_cube_balanced",
        offset_match_policy="compact" if compact_start else "bucket",
        second_actual_tokens=second_actual,
        second_graph_tokens=second_graph,
        second_padding_tokens=second_graph - second_actual,
        offset_capture_sizes_considered=[] if compact_start else sorted({
            int(plan.split_graph_tokens[1])
            for plan in getattr(macro_graph_config, "capture_plans", [])
        }),
        offset_min_graph_tokens=0 if compact_start else int(
            getattr(macro_graph_config, "min_split_graph_tokens", 1)),
        offset_max_graph_tokens_by_start=None,
        offset_allowed_graph_tokens_by_start=None,
        extra_debug_payload=({
            "mixed_request_split": True,
            "macro_compact_plan": True,
        } if compact_start else None),
    ), INPLACE_SPLIT_DRY_RUN


def _build_macro_planner_capture_plan(total_num_tokens: int,
                                      macro_graph_config):
    from vllm_ascend.ascend_config import MacroGraphCapturePlan

    alignment = int(getattr(macro_graph_config, "graph_token_alignment", 64))
    min_split_graph_tokens = int(
        getattr(macro_graph_config, "min_split_graph_tokens", 192))
    graph_total_tokens = _round_up_to_multiple(total_num_tokens, alignment)
    half = _round_up_to_multiple((total_num_tokens + 1) // 2, alignment)
    first_graph_tokens = min(max(half, min_split_graph_tokens),
                             graph_total_tokens - min_split_graph_tokens)
    second_graph_tokens = graph_total_tokens - first_graph_tokens
    if first_graph_tokens < min_split_graph_tokens \
            or second_graph_tokens < min_split_graph_tokens:
        return None
    first_actual_tokens = min(first_graph_tokens, total_num_tokens - 1)
    second_actual_tokens = total_num_tokens - first_actual_tokens
    if second_actual_tokens <= 0:
        return None
    if second_actual_tokens > second_graph_tokens:
        return None
    return MacroGraphCapturePlan(
        total_tokens=total_num_tokens,
        split_actual_tokens=(first_actual_tokens, second_actual_tokens),
        split_graph_tokens=(first_graph_tokens, second_graph_tokens),
        split_start_tokens=(0, first_actual_tokens),
    )


def create_macro_inplace_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    uniform_decode_query_len: int,
    macro_graph_config,
) -> tuple[Optional[InplaceSplitPlan], str]:
    """Create a split plan from load-time macro graph capture plans.

    The macro path is intentionally exact-match by default: runtime must hit a
    predeclared macro graph plan and must not trigger lazy capture.
    """
    if macro_graph_config is None or not getattr(
            macro_graph_config, "enabled", False):
        return None, NO_SPLIT_MACRO_GRAPH_DISABLED

    q = int(uniform_decode_query_len)
    if q <= 0:
        return None, NO_SPLIT_INVALID_QUERY_LEN
    num_reqs = len(num_scheduled_tokens_per_request)
    if (num_reqs == 0
            or int(np.sum(num_scheduled_tokens_per_request)) !=
            int(total_num_tokens)
            or not np.all(num_scheduled_tokens_per_request == q)):
        return None, NO_SPLIT_NON_UNIFORM_DECODE

    total_num_tokens = int(total_num_tokens)
    plan_source = getattr(macro_graph_config, "plan_source", "explicit")
    capture_plans = list(getattr(macro_graph_config, "capture_plans", []))
    capture_plans_considered = [
        int(plan.total_tokens) for plan in capture_plans
    ]

    if plan_source == "planner":
        if total_num_tokens not in set(
                getattr(macro_graph_config, "capture_total_tokens", [])):
            return None, NO_SPLIT_MACRO_GRAPH_MISS
        capture_plan = _build_macro_planner_capture_plan(
            total_num_tokens, macro_graph_config)
        if capture_plan is None:
            return None, NO_SPLIT_MACRO_GRAPH_INVALID_PLAN
        capture_plans_considered = list(
            getattr(macro_graph_config, "capture_total_tokens", []))
        return _macro_capture_plan_to_inplace_plan(
            capture_plan,
            uniform_decode_query_len=q,
            capture_plans_considered=capture_plans_considered,
            macro_graph_config=macro_graph_config,
        )

    for capture_plan in capture_plans:
        if int(capture_plan.total_tokens) != total_num_tokens:
            continue
        return _macro_capture_plan_to_inplace_plan(
            capture_plan,
            uniform_decode_query_len=q,
            capture_plans_considered=capture_plans_considered,
            macro_graph_config=macro_graph_config,
        )

    return None, NO_SPLIT_MACRO_GRAPH_MISS


def create_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    num_splits: int,
    custom_split_sizes: Optional[list[int]] = None,
) -> SplitBatchSlices:
    """Create split batch slices by dividing requests evenly.
    
    Unlike DBO which splits by token count, split batch divides by request count
    for uniform decode batches (where each request has 1 token).
    
    Args:
        num_scheduled_tokens_per_request: Array of token counts per request
        num_splits: Number of splits to create
        custom_split_sizes: Optional list of exact split sizes (number of requests per split).
                          If provided, must sum to total number of requests.
        
    Returns:
        List of SplitBatchSlice objects
    """
    num_reqs = len(num_scheduled_tokens_per_request)
    
    # Compute cumulative token counts
    cu_num_tokens = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens_per_request, dtype=np.int32, out=cu_num_tokens[1:])
    
    # Validate custom_split_sizes if provided
    if custom_split_sizes is not None:
        if sum(custom_split_sizes) != num_reqs:
            raise ValueError(
                f"Sum of custom_split_sizes ({sum(custom_split_sizes)}) "
                f"must equal total number of requests ({num_reqs})"
            )
        if len(custom_split_sizes) != num_splits:
            raise ValueError(
                f"Length of custom_split_sizes ({len(custom_split_sizes)}) "
                f"must equal num_splits ({num_splits})"
            )
    
    
    slices = []
    if custom_split_sizes is not None:
        # Use custom split sizes
        start_req = 0
        for split_size in custom_split_sizes:
            end_req = start_req + split_size
            
            if start_req >= num_reqs:
                break
                
            token_start = int(cu_num_tokens[start_req])
            token_end = int(cu_num_tokens[end_req])
            
            slices.append(SplitBatchSlice(
                request_slice=slice(start_req, end_req),
                token_slice=slice(token_start, token_end),
            ))
            
            start_req = end_req
    else:
        # Calculate requests per split (ceil division for even distribution)
        reqs_per_split = (num_reqs + num_splits - 1) // num_splits
        
        for i in range(num_splits):
            start_req = i * reqs_per_split
            end_req = min((i + 1) * reqs_per_split, num_reqs)
            
            if start_req >= num_reqs:
                break
                
            token_start = int(cu_num_tokens[start_req])
            token_end = int(cu_num_tokens[end_req])
            
            slices.append(SplitBatchSlice(
                request_slice=slice(start_req, end_req),
                token_slice=slice(token_start, token_end),
            ))
    
    return slices


def split_batch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    vllm_config: VllmConfig,
    cudagraph_capture_sizes: Optional[set] = None,
    custom_split_sizes: Optional[list[int]] = None,
) -> tuple[Optional[SplitBatchSlices], Optional[int]]:
    """
    Determine if and how to split the batch for split batch execution.
    
    Split batch is designed for:
    - Large batch sizes that exceed min_batch_size_for_split
    - FULL graph mode where each split can use a captured graph
    
    Args:
        num_scheduled_tokens_per_request: Token counts per request
        num_tokens_unpadded: Total tokens without padding
        num_tokens_padded: Total tokens with padding
        vllm_config: vLLM configuration
        cudagraph_capture_sizes: Set of captured graph sizes (for validation)
        custom_split_sizes: Optional list of exact split sizes (number of requests per split).
                          If provided, must sum to total number of requests.
        
    Returns:
        tuple[Optional[SplitBatchSlices], Optional[int]]:
            - split_slices: List of SplitBatchSlice if splitting, None otherwise
            - padded_total_tokens: Total tokens after padding each split
    """
    from vllm_ascend.ascend_config import get_ascend_config
    
    ascend_config = get_ascend_config()
    split_config = ascend_config.split_batch_config
    
    # Check if split batch is enabled
    if not split_config.enabled:
        return (None, None)
    
    num_reqs = len(num_scheduled_tokens_per_request)
    
    # Check minimum batch size threshold
    if num_reqs < split_config.min_batch_size_for_split:
        return (None, None)
    
    num_splits = split_config.num_splits
    
    # Create split slices
    split_slices = create_split_batch_slices(
        num_scheduled_tokens_per_request,
        num_splits,
        custom_split_sizes,
    )
    
    # If we couldn't create valid splits, return None
    if not split_slices or len(split_slices) < 2:
        return (None, None)
    
    # Validate that each split size has a corresponding captured graph
    # (only relevant when cudagraph is enabled)
    if cudagraph_capture_sizes:
        padded_total = 0
        sorted_capture_sizes = sorted(cudagraph_capture_sizes)
        max_capture_size = max(cudagraph_capture_sizes)
        for split_slice in split_slices:
            split_size = split_slice.num_tokens
            # Find the smallest capture size >= split_size
            padded_size = next(
                (cs for cs in sorted_capture_sizes if cs >= split_size),
                max_capture_size,
            )
            split_slice.padded_num_tokens = padded_size
            padded_total += padded_size
        return (split_slices, padded_total)
    
    return (split_slices, num_tokens_padded)
