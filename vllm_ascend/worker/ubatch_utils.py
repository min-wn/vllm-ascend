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

    def __post_init__(self):
        if self.padded_num_tokens == 0:
            self.padded_num_tokens = self.num_tokens

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

    def debug_payload(self) -> dict[str, object]:
        return {
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
