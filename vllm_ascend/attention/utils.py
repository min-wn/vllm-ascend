from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend import inplace_split_debug as split_debug
from vllm_ascend.utils import (AscendDeviceType, get_ascend_config,
                               get_ascend_device_type)


def slice_positions_by_token(positions: torch.Tensor,
                             token_slice: slice) -> torch.Tensor:
    if positions.ndim == 2:
        return positions[:, token_slice]
    return positions[token_slice]


def slice_model_inputs_by_token(
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    inputs_embeds: Optional[torch.Tensor],
    token_slice: slice,
) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    sliced_input_ids = None if input_ids is None else input_ids[token_slice]
    sliced_positions = slice_positions_by_token(positions, token_slice)
    sliced_inputs_embeds = (None if inputs_embeds is None else
                            inputs_embeds[token_slice])
    return sliced_input_ids, sliced_positions, sliced_inputs_embeds


def stabilize_inplace_common_attn_metadata(
    common: "AscendCommonAttentionMetadata",
    *,
    split_idx: int,
    max_num_reqs: int,
    max_num_tokens: int,
    query_start_loc_secondary: Any,
    seq_lens_secondary: Any,
    slot_mapping_secondary: Any,
    block_table_secondary: Any = None,
) -> "AscendCommonAttentionMetadata":
    if split_idx == 0:
        return common

    nreq = int(common.num_reqs)
    ntok = int(common.num_actual_tokens)
    if nreq + 1 > max_num_reqs + 1:
        raise ValueError(
            f"inplace metadata nreq overflow: {nreq} > {max_num_reqs}")
    if ntok > max_num_tokens:
        raise ValueError(
            f"inplace metadata ntok overflow: {ntok} > {max_num_tokens}")
    if (block_table_secondary is not None
            and common.block_table_tensor is not None):
        block_table_width = int(common.block_table_tensor.shape[1])
        if block_table_secondary.gpu.shape[0] < nreq:
            raise ValueError(
                f"inplace metadata block table nreq overflow: {nreq} > "
                f"{block_table_secondary.gpu.shape[0]}")
        if block_table_secondary.gpu.shape[1] < block_table_width:
            raise ValueError(
                "inplace metadata block table width overflow: "
                f"{block_table_width} > "
                f"{block_table_secondary.gpu.shape[1]}")

    query_start_loc_gpu = query_start_loc_secondary.gpu[:nreq + 1]
    query_start_loc_cpu = query_start_loc_secondary.cpu[:nreq + 1]
    seq_lens_gpu = seq_lens_secondary.gpu[:nreq]
    seq_lens_cpu = seq_lens_secondary.cpu[:nreq]
    slot_mapping = slot_mapping_secondary.gpu[:ntok]
    block_table_tensor = common.block_table_tensor
    if block_table_secondary is not None and block_table_tensor is not None:
        block_table_width = int(block_table_tensor.shape[1])
        block_table_tensor = block_table_secondary.gpu[:nreq, :
                                                       block_table_width]

    query_start_loc_gpu.copy_(common.query_start_loc[:nreq + 1])
    query_start_loc_cpu.copy_(common.query_start_loc_cpu[:nreq + 1])
    seq_lens_gpu.copy_(common.seq_lens[:nreq])
    seq_lens_cpu.copy_(common.seq_lens_cpu[:nreq])
    slot_mapping.copy_(common.slot_mapping[:ntok])
    if (block_table_secondary is not None
            and common.block_table_tensor is not None):
        block_table_tensor.copy_(common.block_table_tensor[:nreq])

    return replace(
        common,
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        seq_lens_cpu=seq_lens_cpu,
        slot_mapping=slot_mapping,
        block_table_tensor=block_table_tensor,
    )


def using_paged_attention(runtime_shape: int,
                          vllm_config: VllmConfig,
                          forward_context: Any = None) -> bool:
    forced_backend = getattr(forward_context, "forced_attention_backend", None)
    if forced_backend == "pa":
        return True
    if forced_backend == "fia":
        return False

    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    descriptor_backend = getattr(batch_descriptor, "attention_backend", "")
    if descriptor_backend == "pa":
        return True
    if descriptor_backend == "fia":
        return False

    if vllm_config.speculative_config is not None:
        return False
    if get_ascend_device_type() == AscendDeviceType.A5:
        return False
    from vllm.config.compilation import CUDAGraphMode
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False

    return runtime_shape in get_ascend_config().pa_shape_list


@lru_cache(maxsize=1)
def enable_cp():
    prefill_config = get_current_vllm_config().parallel_config
    return prefill_config.prefill_context_parallel_size > 1 \
                or prefill_config.decode_context_parallel_size > 1


@dataclass
# class AscendCommonLongSequenceMetadata:
class AscendPrefillContextParallelMetadata:
    pcp_allgather_restore_idx: torch.Tensor = None

    cp_kv_recover_idx_for_chunk: torch.Tensor = None

    num_actual_tokens_pcp_padded: Optional[int] = None

    num_computed_tokens_of_pcp_dcp: Optional[list[list[list[int]]]] = None

    q_head_idx_tensor: torch.Tensor = None

    q_tail_idx_tensor: torch.Tensor = None

    kv_with_q_head_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_head_mask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_mask_idx_tensor: torch.Tensor = None

    attn_mask_seqlens: torch.Tensor = None

    head_attn_nomask_seqlens: torch.Tensor = None

    tail_attn_nomask_seqlens: torch.Tensor = None

    q_full_idx: torch.Tensor = None

    pcp_prefill_mask: torch.Tensor = None


@dataclass
class AscendCommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens_cpu: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    seq_lens: torch.Tensor
    """same to seq_lens_cpu, for compatibility with some new attn metadata
    (such as GDN)."""

    num_computed_tokens_cpu: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""

    max_query_len: int
    """Max token number of request in batch"""

    decode_token_per_req: int
    """decode token number per request"""

    block_table_tensor: torch.Tensor

    slot_mapping: torch.Tensor

    actual_seq_lengths_q: list[int]

    positions: torch.Tensor = None

    attn_mask: torch.Tensor = None

    spec_attn_mask: torch.Tensor = None

    attn_state: Any = None

    graph_pad_size: int = -1

    # num_input_tokens refers to total number of tokens including
    # padding tokens. It is used to handle some padding operations.
    num_input_tokens: int = 0

    prefill_context_parallel_metadata: Optional[
        AscendPrefillContextParallelMetadata] = None

    causal: bool = True

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int,
                 num_actual_reqs: int) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[:num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[:num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.
            num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions[:num_actual_tokens],
            attn_mask=self.attn_mask,
            spec_attn_mask=self.spec_attn_mask,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=num_actual_tokens,
            prefill_context_parallel_metadata=self.
            prefill_context_parallel_metadata,
        )


def filter_chunked_req_indices(
    seq_len: torch.Tensor,
    mask_for_non_zero_chunk: Optional[List[bool]],
) -> torch.Tensor:
    """
    filter the reqs which are doing real chunk_prefill.

    Args:
        seq_len: contains multi-req length: [req0_len, req1_len, ...]
        mask_for_non_zero_chunk: [True, False, True, False, ...]
    Returns:
        filtered_indices: the real chunked req's indices
    """
    assert mask_for_non_zero_chunk is not None and len(seq_len) == len(
        mask_for_non_zero_chunk)
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)
    filtered_indices = torch.cat([
        torch.arange(offsets[i], offsets[i] + seq_len[i])
        for i in range(len(mask_for_non_zero_chunk))
        if mask_for_non_zero_chunk[i]
    ])
    return filtered_indices


def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: AscendCommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: List[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def trans_rope_weight(weight, rope_dim):
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    reordered_rope_part = torch.cat(
        (rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, reordered_rope_part), dim=-2).contiguous()


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, (0, r_pad, 0, c_pad))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1],
             block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat,
        (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in 
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return query_start_loc[request_slice.start: request_slice.stop + 1] -\
        query_start_loc[request_slice.start]


def _make_metadata_with_slice(
        ubatch_slice: UBatchSlice,
        attn_metadata: AscendCommonAttentionMetadata,
        max_num_tokens: int = 0) -> AscendCommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to 
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), (
        f"Ubatch slice {ubatch_slice} is empty")

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], \
        "Token slice start outside of first request"
    assert start_locs[last_req] <= last_tok < start_locs[last_req+1], \
        "Token slice end outside of last request"

    # If the "middle" request has tokens in both ubatches, we have to split it.
    # If ubatch_slice is the first ubatch then we will be splitting the last
    # request. If it's the second microbatch, then we will be splitting the
    # first request
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(attn_metadata.query_start_loc,
                                             request_slice)

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, "
        f"got {len(query_start_loc)}")

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped

    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]

    if splits_last_request:
        tokens_skipped = query_start_loc_cpu[-1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens_cpu = seq_lens_cpu.clone()
        seq_lens[-1] -= tokens_skipped
        seq_lens_cpu[-1] -= tokens_skipped

    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[
        request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] -
                            query_start_loc_cpu[:-1])).item())

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    # adapt to Ascend common metadata
    num_input_tokens = token_slice.stop - token_slice.start
    positions = slice_positions_by_token(attn_metadata.positions, token_slice)
    attn_state = attn_metadata.attn_state
    #if attn_metadata.attn_state != AscendAttentionState.ChunkedPrefill:
    attn_mask = attn_metadata.attn_mask

    if len(attn_metadata.actual_seq_lengths_q) > 0:
        actual_seq_lengths_q = list(
            range(attn_metadata.decode_token_per_req, max_num_tokens + 1,
                  attn_metadata.decode_token_per_req))
    else:
        actual_seq_lengths_q = []

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        num_input_tokens=num_input_tokens,
        actual_seq_lengths_q=actual_seq_lengths_q,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        max_query_len=max_query_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        positions=positions,
        attn_mask=attn_mask,
        spec_attn_mask=attn_metadata.spec_attn_mask,
        attn_state=attn_state,
        graph_pad_size=attn_metadata.graph_pad_size,
        decode_token_per_req=attn_metadata.decode_token_per_req,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: AscendCommonAttentionMetadata,
    max_num_tokens: int = 0,
) -> list[AscendCommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the 
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for idx, ubatch_slice in enumerate(ubatch_slices):
        metadata = _make_metadata_with_slice(ubatch_slice,
                                             common_attn_metadata,
                                             max_num_tokens)
        results.append(metadata)
        if split_debug.is_enabled():
            split_debug.log_event(
                "split_metadata",
                {
                    "idx": idx,
                    "source": "split_attn_metadata",
                    "request_slice":
                    split_debug.slice_info(ubatch_slice.request_slice),
                    "token_slice":
                    split_debug.slice_info(ubatch_slice.token_slice),
                    "num_tokens": int(metadata.num_actual_tokens),
                    "padded_num_tokens": int(max_num_tokens),
                    "num_reqs": int(metadata.num_reqs),
                    **split_debug.metadata_tensor_info(metadata),
                },
            )

    return results
