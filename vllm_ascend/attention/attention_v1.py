#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.registry import (AttentionBackendEnum,
                                              register_backend)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         enable_cp, split_decodes_and_prefills,
                                         using_paged_attention)
from vllm_ascend import inplace_split_debug as split_debug
from vllm_ascend.compilation.acl_graph import (ensure_graph_param_key,
                                               _get_fia_key_t,
                                               get_graph_param_key,
                                               get_graph_params,
                                               maybe_template_fia_seq_lens,
                                               update_graph_params_workspaces)
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               weak_ref_tensors)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.attention_cp import \
                AscendAttentionCPImpl
            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.attention_cp import \
                AscendAttentionCPMetadataBuilder
            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadataForPrefill:

    @dataclass
    class AscendPCPMetadata:
        q_head_idx: torch.Tensor = None
        q_tail_idx: torch.Tensor = None
        kv_with_q_head_nomask_idx: torch.Tensor = None
        kv_with_q_head_mask_idx: torch.Tensor = None
        kv_with_q_tail_nomask_idx: torch.Tensor = None
        kv_with_q_tail_mask_idx: torch.Tensor = None
        attn_mask_seqlens: torch.Tensor = None
        head_attn_nomask_seqlens: torch.Tensor = None
        tail_attn_nomask_seqlens: torch.Tensor = None
        q_full_idx: torch.Tensor = None
        pcp_prefill_mask: torch.Tensor = None

    @dataclass
    class ChunkedContextMetadata:
        actual_chunk_seq_lengths: torch.Tensor
        actual_seq_lengths_kv: torch.Tensor
        starts: torch.Tensor
        chunk_seq_mask_filtered_indices: torch.Tensor
        chunked_req_mask: Optional[list[bool]] = None
        local_context_lens_allranks: Optional[list[list[int]]] = None
        cp_kv_recover_idx_for_chunk: Optional[list[int]] = None
        kv_inverse_idx_for_chunk: Optional[list[int]] = None
        batch_chunk_seq_mask: Optional[list[bool]] = None

    """ Prefill Specific Metadata for Ascend"""
    pcp_metadata: Optional[AscendPCPMetadata] = None
    pcp_allgather_restore_idx: Optional[List[int]] = None
    chunked_context: Optional[ChunkedContextMetadata] = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: torch.Tensor = None


@dataclass
class AscendMetadataForDecode:
    """ Decode Specific Metadata for Ascend"""
    num_computed_tokens_of_pcp_dcp: Optional[list[list[list[int]]]] = None
    batch_seq_mask: torch.Tensor = None
    block_tables: torch.Tensor = None


@dataclass
class AscendMetadata:
    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None  # type: ignore
    actual_seq_lengths_q: List[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: Optional[AscendMetadataForPrefill] = None
    # dcp
    decode_meta: Optional[AscendMetadataForDecode] = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""


class AscendAttentionMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.ALWAYS
    # AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0])

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        AscendAttentionMetadataBuilder.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: Optional[nn.Module] = None,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)

        block_table = common_attn_metadata.block_table_tensor
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded if long_seq_metadata else None
        if num_actual_tokens_pcp_padded is None:
            num_actual_tokens_pcp_padded = num_actual_tokens

        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens_pcp_padded]

        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state

        query_start_loc = common_attn_metadata.query_start_loc[:num_reqs + 1]
        if query_start_loc.device != self.device:
            query_start_loc = query_start_loc_cpu.pin_memory().to(
                self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type)
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    _dual_stream_attention_streams: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

    def _get_dual_stream_attention_stream(self, device: torch.device,
                                          stream_idx: int = 1):
        device_key = f"{str(device)}:{int(stream_idx)}"
        stream = type(self)._dual_stream_attention_streams.get(device_key)
        if stream is None:
            stream = torch.npu.Stream(device=device)
            type(self)._dual_stream_attention_streams[device_key] = stream
        return stream

    def _get_dual_stream_attention_metadata(self, layer: AttentionLayer,
                                            forward_context: ForwardContext):
        dual_metadata = getattr(forward_context,
                                "dual_stream_attention_metadata", None)
        if not isinstance(dual_metadata, list) or len(dual_metadata) != 2:
            return None
        layer_name = getattr(layer, "layer_name", None)
        if layer_name is None:
            return None
        try:
            return (dual_metadata[0][layer_name],
                    dual_metadata[1][layer_name])
        except (KeyError, TypeError):
            return None

    def _get_dual_stream_attention_slices(self,
                                          forward_context: ForwardContext):
        slices = getattr(forward_context, "dual_stream_attention_slices", None)
        if not isinstance(slices, list) or len(slices) != 2:
            return None
        return slices

    def _should_use_dual_stream_attention(
            self, layer: AttentionLayer, attn_metadata: AscendMetadata,
            forward_context: ForwardContext) -> bool:
        if self.sliding_window is not None:
            return False
        if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            return False
        if self._get_dual_stream_attention_metadata(layer,
                                                    forward_context) is None:
            return False
        if self._get_dual_stream_attention_slices(forward_context) is None:
            return False
        return True

    def full_graph_fia(self, query: torch.Tensor, key: torch.Tensor,
                       value: torch.Tensor, attn_metadata: AscendMetadata,
                       output: torch.Tensor) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv \
            = self._get_fia_params(key, value, attn_metadata)

        forward_context = get_forward_context()
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        graph_params = get_graph_params(in_parallel_streams)
        param_key = get_graph_param_key(forward_context, num_tokens)
        ensure_graph_param_key(graph_params, param_key)
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        actual_seq_lengths_kv = maybe_template_fia_seq_lens(
            forward_context, actual_seq_lengths_kv,
            _get_fia_key_t(key, block_size), source="attention_full_graph")
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(param_key)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
            )
            update_graph_params_workspaces(param_key,
                                           workspace,
                                           in_parallel_streams=in_parallel_streams)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[param_key].append(event)
        graph_params.attn_params[param_key].append(
            (weak_ref_tensors(query), weak_ref_tensors(key),
             weak_ref_tensors(value), weak_ref_tensors(block_table),
             weak_ref_tensors(attn_metadata.attn_mask), block_size,
             actual_seq_lengths_kv, actual_seq_lengths_q, self.num_kv_heads,
             self.num_heads, self.scale, weak_ref_tensors(output),
             weak_ref_tensors(softmax_lse)))

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[param_key].append(handle)
        return output, num_tokens

    def _dual_stream_fia_workspace_key(self, param_key, split_idx: int):
        return (param_key, "dual_stream_fia", int(split_idx))

    def _run_dual_stream_fia_split(
            self,
            *,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            attn_metadata: AscendMetadata,
            stream: torch.npu.Stream,
            param_key,
            split_idx: int,
            graph_params,
            capturing: bool,
            record_update_event: bool = True):
        key_cache, value_cache, block_size, block_table, actual_seq_lengths_kv \
            = self._get_fia_params(key, value, attn_metadata)
        forward_context = get_forward_context()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        actual_seq_lengths_kv = maybe_template_fia_seq_lens(
            forward_context,
            actual_seq_lengths_kv,
            _get_fia_key_t(key_cache, block_size),
            source=f"attention_dual_stream_full_graph:{split_idx}")
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)

        workspace = None
        workspace_key = self._dual_stream_fia_workspace_key(
            param_key, split_idx)
        if graph_params is not None:
            workspace = graph_params.workspaces.get(workspace_key)
        if workspace is None:
            with torch.npu.stream(stream):
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query=query,
                    key=key_cache,
                    value=value_cache,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    sparse_mode=3,
                    scale=self.scale,
                )
            if graph_params is not None:
                in_parallel_streams = bool(
                    getattr(forward_context, "in_parallel_streams", False))
                update_graph_params_workspaces(
                    workspace_key,
                    workspace,
                    in_parallel_streams=in_parallel_streams)

        event = None
        if capturing and record_update_event:
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)

        with torch.npu.stream(stream):
            if capturing:
                torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value_cache,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            handle = (torch.npu.graph_task_group_end(stream)
                      if capturing else None)

        if not capturing:
            return None, None, None
        param = (
            weak_ref_tensors(query),
            weak_ref_tensors(key_cache),
            weak_ref_tensors(value_cache),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_metadata.attn_mask),
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            workspace_key,
        )
        return param, handle, event

    def forward_dual_stream_fused_infer_attention(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        dual_metadata = self._get_dual_stream_attention_metadata(
            layer, forward_context)
        dual_slices = self._get_dual_stream_attention_slices(forward_context)
        if dual_metadata is None or dual_slices is None:
            raise RuntimeError(
                "dual_stream_attention_metadata is missing from forward "
                "context")

        main_stream = torch_npu.npu.current_stream()
        stream_mode = str(
            getattr(forward_context,
                    "dual_stream_attention_secondary_stream_mode",
                    "dedicated_pair"))
        if stream_mode == "dedicated_pair":
            primary_stream = self._get_dual_stream_attention_stream(
                query.device, 0)
            secondary_stream = self._get_dual_stream_attention_stream(
                query.device, 1)
        else:
            primary_stream = main_stream
            secondary_stream = self._get_dual_stream_attention_stream(
                query.device, 1)
        capturing = bool(getattr(forward_context, "capturing", False))
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        if split_debug.is_enabled():
            split_debug.log_event(
                "dual_stream_attention_streams",
                {
                    "layer_name": getattr(layer, "layer_name", None),
                    "primary_stream": {
                        "repr": repr(primary_stream),
                        "stream_id": getattr(primary_stream, "stream_id", None),
                    },
                    "secondary_stream": {
                        "repr": repr(secondary_stream),
                        "stream_id": getattr(secondary_stream, "stream_id", None),
                    },
                    "current_stream": {
                        "repr": repr(main_stream),
                        "stream_id": getattr(
                            main_stream, "stream_id", None),
                    },
                    "stream_mode": stream_mode,
                    "capturing": capturing,
                    "in_parallel_streams": in_parallel_streams,
                },
                step_id=getattr(forward_context, "split_debug_step_id", None),
            )
        graph_params = get_graph_params(in_parallel_streams) if capturing else None
        param_key = get_graph_param_key(forward_context, int(query.shape[0]))
        if capturing:
            ensure_graph_param_key(graph_params, param_key)

        has_pta_dual_fia_update = (
            hasattr(torch.npu, "make_dual_task_group_handle")
            and hasattr(torch.npu, "dual_fused_infer_attention_score_update"))
        use_shared_update_event = bool(capturing and has_pta_dual_fia_update)
        shared_update_event = None
        if use_shared_update_event:
            shared_update_event = torch.npu.ExternalEvent()
            shared_update_event.wait(main_stream)
            shared_update_event.reset(main_stream)

        fork_event = torch.npu.Event()
        fork_event.record(main_stream)
        if primary_stream is not main_stream:
            primary_stream.wait_event(fork_event)
        secondary_stream.wait_event(fork_event)

        split_params = []
        split_handles = []
        split_events = []
        split_ranges = []
        for split_idx, stream in enumerate((primary_stream, secondary_stream)):
            split_slice = dual_slices[split_idx]
            token_start = int(split_slice.token_slice.start)
            graph_tokens = int(split_slice.graph_num_tokens)
            token_stop = token_start + graph_tokens
            query_view = query[token_start:token_stop]
            output_view = output[token_start:token_stop]
            param, handle, event = self._run_dual_stream_fia_split(
                query=query_view,
                key=key,
                value=value,
                output=output_view,
                attn_metadata=dual_metadata[split_idx],
                stream=stream,
                param_key=param_key,
                split_idx=split_idx,
                graph_params=graph_params,
                capturing=capturing,
                record_update_event=not use_shared_update_event)
            if capturing:
                split_params.append(param)
                split_handles.append(handle)
                if event is not None:
                    split_events.append(event)
                split_ranges.append((token_start, graph_tokens))

        if stream_mode == "dedicated_pair":
            primary_join_event = torch.npu.Event()
            secondary_join_event = torch.npu.Event()
            primary_join_event.record(primary_stream)
            secondary_join_event.record(secondary_stream)
            main_stream.wait_event(primary_join_event)
            main_stream.wait_event(secondary_join_event)
        else:
            join_event = torch.npu.Event()
            join_event.record(secondary_stream)
            main_stream.wait_event(join_event)

        if capturing:
            if has_pta_dual_fia_update:
                graph_params.attn_params[param_key].append(
                    ("dual_stream_fia_pta", weak_ref_tensors(query),
                     weak_ref_tensors(output), tuple(split_params),
                     tuple(split_ranges)))
                graph_params.handles[param_key].append(
                    torch.npu.make_dual_task_group_handle(
                        split_handles[0], split_handles[1]))
                graph_params.events[param_key].append(shared_update_event)
            else:
                graph_params.attn_params[param_key].append(
                    ("dual_stream_fia", tuple(split_params)))
                graph_params.handles[param_key].append(tuple(split_handles))
                graph_params.events[param_key].append(tuple(split_events))
        return output

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ):
        forward_context: ForwardContext = get_forward_context()
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        graph_params = get_graph_params(in_parallel_streams)
        num_tokens = query.shape[0]
        param_key = get_graph_param_key(forward_context, num_tokens)
        ensure_graph_param_key(graph_params, param_key)
        if getattr(forward_context, "capturing", False):
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(param_key)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
                update_graph_params_workspaces(param_key,
                                               weak_ref_tensors(workspace),
                                               in_parallel_streams=in_parallel_streams)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[param_key].append(event)
            graph_params.attn_params[param_key].append((
                weak_ref_tensors(query),
                weak_ref_tensors(self.key_cache),
                weak_ref_tensors(self.value_cache),
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                attn_metadata.block_tables,
                attn_metadata.seq_lens,
                weak_ref_tensors(output),
            ))

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace)
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[param_key].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor,
                        attn_metadata: AscendMetadata):

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
        elif attn_metadata.attn_state == \
                AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        forward_context = get_forward_context()
        batch_descriptor = getattr(forward_context, "batch_descriptor", None)
        if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and getattr(batch_descriptor, "capture_metadata_mode", "")
                == "template"
                and getattr(batch_descriptor, "attention_backend", "") == "fia"):
            actual_seq_lengths_kv = maybe_template_fia_seq_lens(
                forward_context, actual_seq_lengths_kv,
                _get_fia_key_t(key, block_size),
                source="attention_get_fia_params")
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor,
                                   attn_metadata: AscendMetadata,
                                   output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens)

        output = output.view(batch_size, self.num_heads, self.head_size)
        return output

    def forward_fused_infer_attention(self, query: torch.Tensor,
                                      key: torch.Tensor, value: torch.Tensor,
                                      attn_metadata: AscendMetadata,
                                      output: torch.Tensor):
        forward_context: ForwardContext = get_forward_context()
        if getattr(forward_context, "capturing", False):
            self.full_graph_fia(query, key, value, attn_metadata, output)
            return output
        if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and self.sliding_window is not None
                and attn_metadata.seq_lens.shape[0] == query.size(0)):
            return self._forward_fia_slidingwindow(query, attn_metadata,
                                                   output)
        key, value, block_size, block_table, actual_seq_lengths_kv \
            = self._get_fia_params(key, value, attn_metadata)
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        output_view = output[:num_tokens]
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            out=[output_view, softmax_lse],
        )
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        if getattr(forward_context, "capturing", False):
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(query=query,
                                       key_cache=self.key_cache,
                                       value_cache=self.value_cache,
                                       num_kv_heads=self.num_kv_heads,
                                       num_heads=self.num_heads,
                                       scale_value=self.scale,
                                       block_table=attn_metadata.block_tables,
                                       context_lens=attn_metadata.seq_lens,
                                       out=output)
        return output

    def _forward_encoder_attention(self, query: torch.Tensor,
                                   key: torch.Tensor, value: torch.Tensor,
                                   attn_metadata: AscendMetadata,
                                   _: torch.Tensor) -> torch.Tensor:
        assert attn_metadata is not None

        if attn_metadata.causal:
            # use sparse_mode 3 in causal scenario
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                sparse_mode=3,
                atten_mask=attn_metadata.attn_mask,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]
        else:
            # use default sparse_mode 0 in normal scenario, which means no mask works on it
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
    ):

        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            if get_ascend_device_type() == AscendDeviceType.A5:
                # TODO: Once eagle running to here, it may has error because of the 0 dim of slot_mapping.
                # Should check if the 0 dim of slot_mapping must equal to the 0 dim of key.
                # If it's necessary, the slots should be sliced.
                torch_npu.npu_scatter_pa_kv_cache(
                    key=key[:attn_metadata.num_actual_tokens],
                    value=value[:attn_metadata.num_actual_tokens].contiguous(),
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_mapping=slots)
            else:
                torch_npu._npu_reshape_and_cache(
                    key=key[:attn_metadata.num_actual_tokens],
                    value=value[:attn_metadata.num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots[:attn_metadata.num_actual_tokens])
        return key, value

    def forward_impl(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        forward_context: ForwardContext = get_forward_context()
        if self._should_use_dual_stream_attention(layer, attn_metadata,
                                                  forward_context):
            output = self.forward_dual_stream_fused_infer_attention(
                layer, query, key, value, output)
        elif (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and using_paged_attention(num_tokens, self.vllm_config,
                                          forward_context)
                and self.sliding_window is None):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(
                query, key, value, attn_metadata, output)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type not in [
                AttentionType.DECODER, AttentionType.ENCODER_ONLY
        ]:
            raise NotImplementedError("Encoder/Decoder cross-attention "
                                      "is not implemented for "
                                      "PallasAttentionBackendImpl")
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)
        key, value = self.reshape_and_cache(key, value, kv_cache,
                                            attn_metadata)
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(
                query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        output = self.forward_impl(layer, query, key, value, kv_cache,
                                   attn_metadata, output)
        return output
