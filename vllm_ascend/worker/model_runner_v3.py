#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#
import os
import json
import math
import time
import threading
import traceback
import dataclasses
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import regex as re
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx.node import map_arg
from tqdm import tqdm  # type: ignore
from typing_extensions import TypeAlias
from vllm.attention.backends.abstract import AttentionBackend, AttentionType, AttentionMetadata
from vllm.attention.layer import Attention, MLAAttention
from vllm.attention.selector import get_attn_backend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_dcp_group, get_dp_group,
                                             get_pcp_group, get_pp_group,
                                             get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import (BatchDescriptor, DPMetadata,
                                  get_forward_context,
                                  override_forward_context)
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.torch_utils import direct_register_custom_op, get_dtype_size
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput,
                             SamplerOutput,
                             make_empty_encoder_model_runner_output)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.worker.gpu_model_runner import (AsyncGPUModelRunnerOutput,
                                             GPUModelRunner)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds
from vllm.v1.worker.utils import (AttentionGroup, gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         AscendPrefillContextParallelMetadata,
                                         slice_model_inputs_by_token,
                                         using_paged_attention)
# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (ACLGraphWrapper,
                                               _ACL_GRAPH_FIA_UPDATE_USE_CAPTURED_PARAMS,
                                               _extract_block_table_from_metadata,
                                               _get_fia_key_t,
                                               _refresh_block_table_in_place,
                                               ensure_graph_param_key,
                                               get_graph_param_key,
                                               get_graph_params,
                                               graph_param_key_info,
                                               maybe_template_fia_seq_lens,
                                               set_graph_params,
                                               set_graph_params_parallel,
                                               set_mtp_graph_params,
                                               update_attn_dcp_pcp_params,
                                               update_attn_params,
                                               update_attn_params_split,
                                               update_mla_attn_dcp_pcp_params,
                                               update_mla_attn_params)
# yapf: enable
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend import inplace_split_debug as split_debug
from vllm_ascend.ops.rotary_embedding import (
    disable_external_cos_sin_fast_path,
    set_cos_and_sin,
    set_external_cos_sin_fast_path_enabled,
    update_cos_sin,
)
from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.sample.logits_processor import build_logitsprocs
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.utils import (AscendDeviceType, ProfileExecuteDuration,
                               enable_sp, get_ascend_device_type, is_moe_model,
                               lmhead_tp_enable, maybe_trans_nz)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.npu_ubatch_wrapper import (AscendUBatchWrapper,
                                                   AscendUbatchMetadata)
from vllm_ascend.worker.inplace_piecewise_scheduler import (
    InplacePiecewiseSplitInput,
    InplacePiecewiseSplitScheduler,
    capture_piecewise_model_call,
)
from vllm_ascend.worker.ubatch_utils import (INPLACE_SPLIT_DRY_RUN,
                                             InplaceSplitPlan,
                                             MIXED_REQUEST_SPLIT_DRY_RUN,
                                             NO_SPLIT_ATTENTION_BACKEND_MISMATCH,
                                             SplitBatchSlice,
                                             SplitBatchSlices,
                                             _macro_capture_plan_to_inplace_plan,
                                             create_inplace_split_batch_slices,
                                             create_macro_inplace_split_batch_slices,
                                             create_mixed_request_split_batch_slices,
                                             inplace_split_first_graph_matches_attention_backend,
                                             select_inplace_attention_backend,
                                             split_batch_split, ubatch_split)
from vllm_ascend.attention.utils import split_attn_metadata
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType, create_ascend_forward_context, get_mc2_tokens_capacity,
    select_moe_comm_method, set_ascend_forward_context, set_mc2_mask,
    set_mc2_tokens_capacity)

from msprobe.pytorch import AclGraphDumper

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

if get_ascend_device_type() == AscendDeviceType._310P:
    torch_npu.npu.set_compile_mode(jit_compile=False)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = Union[list[AttnMetadataDict],
                                        AttnMetadataDict]


_MACRO_ATTENTION_CONTEXTS: dict[str, list[Any]] = {}


def _macro_attention_context(
        macro_context_key: str, split_idx: int,
        layer_name: str) -> tuple[Any, Any, Attention | MLAAttention,
                                  torch.Tensor]:
    contexts = _MACRO_ATTENTION_CONTEXTS.get(str(macro_context_key))
    if contexts is None:
        raise RuntimeError(
            "Missing torchair macro attention context for key "
            f"{macro_context_key!r}")
    split_idx = int(split_idx)
    if split_idx < 0 or split_idx >= len(contexts):
        raise RuntimeError(
            "Invalid torchair macro attention split index "
            f"{split_idx}; num_contexts={len(contexts)}")
    forward_context = contexts[split_idx]
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, list):
        if split_idx >= len(attn_metadata):
            raise RuntimeError(
                "Torchair macro attention metadata list too short: "
                f"split_idx={split_idx}, len={len(attn_metadata)}")
        attn_metadata = attn_metadata[split_idx]
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    attn_layer: Attention | MLAAttention = (
        forward_context.no_compile_layers[layer_name])
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    return forward_context, attn_metadata, attn_layer, kv_cache


def _macro_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    macro_context_key: str,
    split_idx: int,
    layer_name: str,
) -> torch.Tensor:
    forward_context, attn_metadata, attn_layer, kv_cache = _macro_attention_context(
        macro_context_key, split_idx, layer_name)
    with override_forward_context(forward_context):
        return attn_layer.impl.forward(attn_layer, query, key, value, kv_cache,
                                       attn_metadata)


def _macro_unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    macro_context_key: str,
    split_idx: int,
    layer_name: str,
) -> torch.Tensor:
    del key, value, macro_context_key, split_idx, layer_name
    return torch.empty_like(query).contiguous()


def _macro_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    macro_context_key: str,
    split_idx: int,
    layer_name: str,
    output_scale: Optional[torch.Tensor] = None,
    output_block_scale: Optional[torch.Tensor] = None,
) -> None:
    forward_context, attn_metadata, attn_layer, kv_cache = _macro_attention_context(
        macro_context_key, split_idx, layer_name)
    with override_forward_context(forward_context):
        attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )


def _macro_unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    macro_context_key: str,
    split_idx: int,
    layer_name: str,
    output_scale: Optional[torch.Tensor] = None,
    output_block_scale: Optional[torch.Tensor] = None,
) -> None:
    del (query, key, value, output, macro_context_key, split_idx, layer_name,
         output_scale, output_block_scale)


direct_register_custom_op(
    op_name="macro_unified_attention",
    op_func=_macro_unified_attention,
    fake_impl=_macro_unified_attention_fake,
)
direct_register_custom_op(
    op_name="macro_unified_attention_with_output",
    op_func=_macro_unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=_macro_unified_attention_with_output_fake,
)



@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(
        torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


_SPLIT_METADATA_DEBUG_FILE = os.environ.get(
    "VLLM_ASCEND_SPLIT_METADATA_DEBUG_FILE",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "split_metadata_debug.log")
    ),
)
_SPLIT_METADATA_DEBUG_ENABLED = os.environ.get(
    "VLLM_ASCEND_SPLIT_METADATA_DEBUG", "0") in ("1", "true", "True")

# Perf stats log file: each decode step appends one JSON line with timing info.
# Set VLLM_ASCEND_PERF_STATS_FILE to a path to enable; empty string disables.
_PERF_STATS_FILE = os.environ.get(
    "VLLM_ASCEND_PERF_STATS_FILE",
    "",
)
_SPLIT_MERGE_DUMP = os.environ.get(
    "VLLM_ASCEND_SPLIT_MERGE_DUMP", "1") not in ("0", "false", "False")

def _write_perf_stats(stats: dict) -> None:
    """Append one JSON line to the perf stats file (if enabled)."""
    if not _PERF_STATS_FILE:
        return
    try:
        import json as _json
        with open(_PERF_STATS_FILE, "a", encoding="utf-8") as f:
            f.write(_json.dumps(stats, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to write perf stats to %s: %s",
                       _PERF_STATS_FILE, e)
# Rollback switch for split context rebuild coordinate validation.
# - default "1": use local coordinate slices for rebuilt split context
# - set VLLM_ASCEND_SPLIT_LOCAL_CONTEXT_REBUILD=0 to restore legacy behavior
_SPLIT_LOCAL_CONTEXT_REBUILD = os.environ.get(
    "VLLM_ASCEND_SPLIT_LOCAL_CONTEXT_REBUILD", "1") not in ("0", "false", "False")
_MACRO_GRAPH_EAGER_PREFLIGHT = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_EAGER_PREFLIGHT", "0") in ("1", "true", "True")
_MACRO_GRAPH_BIND_ALLOWLIST_STRICT = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_BIND_ALLOWLIST_STRICT", "0") in ("1", "true",
                                                              "True")
_MACRO_GRAPH_ATTENTION_UPDATE_STRICT = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_ATTENTION_UPDATE_STRICT", "0") in ("1", "true",
                                                                "True")
_MACRO_GRAPH_USE_OPAQUE_ATTENTION = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_USE_OPAQUE_ATTENTION", "0") in ("1", "true",
                                                             "True")
_MACRO_GRAPH_REWRITE_ATTENTION_OPS = (
    os.environ.get("VLLM_ASCEND_MACRO_GRAPH_REWRITE_ATTENTION_OPS", "1")
    not in ("0", "false", "False") or _MACRO_GRAPH_USE_OPAQUE_ATTENTION)
_MACRO_GRAPH_ATTENTION_UPDATE_SYNC = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_ATTENTION_UPDATE_SYNC", "0") not in (
        "0", "false", "False")
_MACRO_GRAPH_ATTENTION_DUAL_UPDATE = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_ATTENTION_DUAL_UPDATE", "0") not in (
        "0", "false", "False")
_MACRO_GRAPH_NPUGRAPH_EX_OPAQUE_ATTENTION_UPDATE = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_NPUGRAPH_EX_OPAQUE_ATTENTION_UPDATE", "0"
) not in ("0", "false", "False")
_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE", "").lower()
if not _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE:
    _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE = (
        "task_update" if os.environ.get(
            "VLLM_ASCEND_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE", "0") in (
                "1", "true", "True") else "off")
if _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE in ("0", "false", "off",
                                                   "none", "disabled"):
    _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE = "off"
elif _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE in ("1", "true", "task",
                                                     "task_update"):
    _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE = "task_update"
elif _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE not in ("event_only", ):
    logger.warning(
        "Unsupported VLLM_ASCEND_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE=%r; "
        "falling back to off",
        _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
    )
    _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE = "off"
_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE = (
    _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE != "off")
_MACRO_GRAPH_ATTENTION_UPDATE_REPLAY_EVENT = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_ATTENTION_UPDATE_REPLAY_EVENT", "1") not in (
        "0", "false", "False")
_MACRO_GRAPH_NPUGRAPH_EX_FORCE_SERIAL_REPLAY = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_NPUGRAPH_EX_FORCE_SERIAL_REPLAY", "0") in (
        "1", "true", "True")
_MACRO_GRAPH_COMPACT_FAKE_SLOT_POLICY = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_COMPACT_FAKE_SLOT_POLICY", "scratch").lower()
_MACRO_GRAPH_COMPACT_FAKE_SEQ_LENS_POLICY = os.environ.get(
    "VLLM_ASCEND_MACRO_GRAPH_COMPACT_FAKE_SEQ_LENS_POLICY", "last").lower()


def _append_split_metadata_debug(tag: str, payload: Any) -> None:
    if not _SPLIT_METADATA_DEBUG_ENABLED:
        return
    try:
        with open(_SPLIT_METADATA_DEBUG_FILE, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write("[{}] {}: {}\n".format(ts, tag, payload))
    except Exception as e:
        logger.warning(
            "Failed to write split metadata debug file %s: %s",
            _SPLIT_METADATA_DEBUG_FILE,
            e,
        )


class MissingMacroGraphError(RuntimeError):
    """Raised when macro_graph_config requires a graph that was not captured."""


class _PlannedMacroGraphEntry:
    """A planned macro graph keyed by a split plan.

    The backend executable is materialized from the first runtime hit so the
    compiled callable can bind the real vLLM metadata tensor addresses.
    Subsequent hits copy current metadata values into those static addresses.
    """

    def __init__(
        self,
        *,
        key: tuple[Any, ...],
        plan: InplaceSplitPlan,
        inplace_attention_backend: str,
        split_req_caps: Optional[tuple[int, int]] = None,
    ) -> None:
        self.key = key
        self.plan = plan
        self.inplace_attention_backend = inplace_attention_backend
        self.split_req_caps = split_req_caps
        self.compiled_callable: Any = None
        self.module: Optional[nn.Module] = None
        self.captured_metadata: Optional[list[AscendUbatchMetadata]] = None
        self.binding_plan: Optional["MacroGraphBindingPlan"] = None
        self.runtime_slot_metadata: Optional[list[AscendUbatchMetadata]] = None
        self.binding_fallback_count: int = 0
        self.macro_attention_tensor_retention: list[torch.Tensor] = []
        self.macro_attention_tensor_retention_seen: set[tuple[Any, ...]] = set()
        self._macro_attention_retention_context_state: list[tuple[
            Any, bool, Any, bool, Any]] = []
        self.macro_attention_update_replay_event: Any = None
        self.macro_attention_update_replay_event_recorded: bool = False
        self.macro_attention_update_in_backend: bool = False
        self.macro_attention_external_update: bool = False
        self.compile_ms: float = 0.0
        self.replay_count: int = 0
        self.backend: str = ""

    @property
    def materialized(self) -> bool:
        return self.compiled_callable is not None


class _MacroNodeRef:

    __slots__ = ("index", )

    def __init__(self, index: int) -> None:
        self.index = int(index)


@dataclass
class MacroGraphTensorBinding:
    name: str
    dst: torch.Tensor
    src_path: tuple[Any, ...]
    max_shape: tuple[int, ...]
    copy_policy: str = "prefix"
    copy_dim: int = 0
    zero_tail: bool = False
    required: bool = True


@dataclass
class MacroGraphScalarBinding:
    name: str
    dst_parent_path: tuple[Any, ...]
    src_parent_path: tuple[Any, ...]
    attr: str
    required: bool = False


@dataclass
class MacroGraphSplitBindingPlan:
    tensor_bindings: list[MacroGraphTensorBinding]
    scalar_bindings: list[MacroGraphScalarBinding]


@dataclass
class MacroGraphBindingPlan:
    split_bindings: list[MacroGraphSplitBindingPlan]


class _TorchairTaggedPiecewiseMacroModule(nn.Module):
    """Static two-split piecewise schedule with torchair tagged events."""

    def __init__(
        self,
        *,
        runtime_calls: list[Any],
        contexts: list[Any],
        event_tag_prefix: str,
        secondary_stream_tag: str,
        validate_no_inner_aclgraph: bool,
        tng: Any,
    ) -> None:
        super().__init__()
        if len(runtime_calls) != 2 or len(contexts) != 2:
            raise RuntimeError(
                "torchair tagged macro graph currently requires exactly "
                f"2 splits, got runtime_calls={len(runtime_calls)}, "
                f"contexts={len(contexts)}")
        handle = runtime_calls[0].handle
        if runtime_calls[1].handle is not handle:
            raise RuntimeError(
                "Cannot build a macro graph from different piecewise handles")
        self.runtime_calls = runtime_calls
        self.contexts = contexts
        self._attention_context_key = str(event_tag_prefix)
        _MACRO_ATTENTION_CONTEXTS[self._attention_context_key] = self.contexts
        self.handle = handle
        self.split_gm = handle.split_gm
        self._nodes = tuple(handle.split_gm.graph.nodes)
        self._node_to_idx = {
            node: idx
            for idx, node in enumerate(self._nodes)
        }
        self._pieces = {
            item.submod_name: item
            for item in handle.piecewise_graphs
        }
        self._initial_envs = (
            self._bind_placeholders(runtime_calls[0]),
            self._bind_placeholders(runtime_calls[1]),
        )
        self._node_specs = tuple(self._build_node_specs())
        self._piece_callables = self._build_piece_callables()
        self.secondary_stream_tag = str(secondary_stream_tag)
        self._main_done_tag = f"{event_tag_prefix}_main_done"
        self._secondary_done_tag = f"{event_tag_prefix}_secondary_done"
        __import__("torchair.scope._scope")
        # Keep event objects alive; forward uses torch.ops.air by tag so FX
        # tracing does not need to reason about torch.npu.Event objects.
        self._main_done_event = tng.ops.npu_create_tagged_event(
            tag=self._main_done_tag)
        self._secondary_done_event = tng.ops.npu_create_tagged_event(
            tag=self._secondary_done_tag)
        if validate_no_inner_aclgraph:
            self._validate_no_inner_aclgraph()

    def _validate_no_inner_aclgraph(self) -> None:
        offenders: list[str] = []
        for item in self.handle.piecewise_graphs:
            name = str(getattr(item, "submod_name", ""))
            if not name:
                continue
            try:
                module = self._fetch_attr(name)
            except Exception:
                continue
            if isinstance(module, ACLGraphWrapper):
                unwrap = getattr(module, "unwrap", None)
                if not callable(unwrap):
                    offenders.append(name)
        if offenders:
            raise RuntimeError(
                "macro_graph_config.validate_no_inner_aclgraph=True "
                "found ACLGraphWrapper pieces that cannot be unwrapped: "
                f"{offenders[:8]}")

    def _bind_placeholders(self, runtime_call: Any) -> list[Any]:
        env: list[Any] = [None] * len(self._nodes)
        args_iter = iter(runtime_call.args)
        kwargs = runtime_call.kwargs
        for idx, node in enumerate(self._nodes):
            if node.op != "placeholder":
                continue
            target = str(node.target)
            if target.startswith("**"):
                env[idx] = kwargs
                continue
            if target.startswith("*"):
                env[idx] = tuple(args_iter)
                continue
            try:
                env[idx] = next(args_iter)
                continue
            except StopIteration:
                pass
            if target in kwargs:
                env[idx] = kwargs[target]
                continue
            if node.args:
                env[idx] = node.args[0]
                continue
            raise RuntimeError(
                "Cannot bind macro graph placeholder "
                f"{target!r}; captured args={len(runtime_call.args)}, "
                f"kwargs={sorted(kwargs)}")
        return env

    def _resolve_arg(self, arg: Any, env: list[Any]) -> Any:
        return map_arg(arg, lambda node: env[self._node_to_idx[node]])

    def _make_arg_spec(self, arg: Any) -> Any:
        return map_arg(arg,
                       lambda node: _MacroNodeRef(self._node_to_idx[node]))

    def _resolve_spec(self, spec: Any, env: list[Any]) -> Any:
        if isinstance(spec, _MacroNodeRef):
            return env[spec.index]
        if isinstance(spec, tuple):
            return tuple(self._resolve_spec(item, env) for item in spec)
        if isinstance(spec, list):
            return [self._resolve_spec(item, env) for item in spec]
        if isinstance(spec, dict):
            return {
                key: self._resolve_spec(value, env)
                for key, value in spec.items()
            }
        return spec

    def _build_node_specs(self) -> list[tuple[str, int, Any, Any, Any]]:
        specs: list[tuple[str, int, Any, Any, Any]] = []
        for idx, node in enumerate(self._nodes):
            if node.op == "placeholder":
                specs.append(("placeholder", idx, None, None, None))
                continue
            if node.op == "output":
                specs.append(
                    ("output", idx, None, self._make_arg_spec(node.args[0]),
                     None))
                continue
            args_spec = self._make_arg_spec(node.args)
            kwargs_spec = self._make_arg_spec(node.kwargs)
            if node.op == "call_module" and str(node.target) in self._pieces:
                piece = self._pieces[str(node.target)]
                kind = ("parallel_piece" if bool(
                    getattr(piece, "is_splitting_graph", False)) else
                        "serial_piece")
                specs.append(
                    (kind, idx, str(node.target), args_spec, kwargs_spec))
                continue
            if node.op == "get_attr":
                specs.append(("get_attr", idx, str(node.target), None, None))
                continue
            if node.op == "call_function":
                specs.append(
                    ("call_function", idx, node.target, args_spec,
                     kwargs_spec))
                continue
            if node.op == "call_method":
                specs.append(
                    ("call_method", idx, str(node.target), args_spec,
                     kwargs_spec))
                continue
            if node.op == "call_module":
                specs.append(
                    ("call_module", idx, str(node.target), args_spec,
                     kwargs_spec))
                continue
            raise RuntimeError(
                "Unsupported macro graph FX node while building specs: "
                f"op={node.op!r}, target={node.target!r}, node={node!r}")
        return specs

    def _fetch_attr(self, target: str) -> Any:
        target_atoms = target.split(".")
        attr_itr = self.split_gm
        for atom in target_atoms:
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def _eval_node(self, node: Any, env: list[Any]) -> Any:
        args = self._resolve_arg(node.args, env)
        kwargs = self._resolve_arg(node.kwargs, env)
        if node.op == "get_attr":
            return self._fetch_attr(str(node.target))
        if node.op == "call_function":
            return node.target(*args, **kwargs)
        if node.op == "call_method":
            self_obj, *args_tail = args
            return getattr(self_obj, str(node.target))(*args_tail, **kwargs)
        if node.op == "call_module":
            module = self._fetch_attr(str(node.target))
            return module(*args, **kwargs)
        raise RuntimeError(
            "Unsupported macro graph FX node: "
            f"op={node.op!r}, target={node.target!r}, node={node!r}")

    def _call_piece_by_target(self, target: str, args_spec: Any,
                              kwargs_spec: Any, env: list[Any],
                              split_idx: int) -> Any:
        args = self._resolve_spec(args_spec, env)
        kwargs = self._resolve_spec(kwargs_spec, env)
        module = self._piece_callables.get((str(target), int(split_idx)))
        if module is None:
            module = self._fetch_attr(str(target))
            unwrap = getattr(module, "unwrap", None)
            if callable(unwrap):
                module = unwrap()
        return module(*args, **kwargs)

    def _op_target_name(self, target: Any) -> str:
        return str(target)

    def _is_vllm_op_target(self, target: Any, op_name: str) -> bool:
        try:
            op = getattr(torch.ops.vllm, op_name)
            if target is op or target is op.default:
                return True
        except Exception:
            pass
        schema = getattr(target, "_schema", None)
        schema_name = getattr(schema, "name", None)
        if schema_name == f"vllm::{op_name}":
            return True
        target_name = self._op_target_name(target)
        return target_name in (f"vllm.{op_name}",
                               f"vllm.{op_name}.default")

    def _inject_macro_attention_args(
        self,
        node: Any,
        *,
        split_idx: int,
        layer_arg_index: int,
    ) -> None:
        args = list(node.args)
        kwargs = dict(node.kwargs)
        if len(args) > layer_arg_index:
            layer_name = args[layer_arg_index]
            args[layer_arg_index:layer_arg_index + 1] = [
                self._attention_context_key,
                int(split_idx),
                layer_name,
            ]
            node.args = tuple(args)
            node.kwargs = kwargs
            return

        if "layer_name" not in kwargs:
            raise RuntimeError(
                "Cannot rewrite macro attention node without layer_name: "
                f"node={node!r}")
        kwargs["macro_context_key"] = self._attention_context_key
        kwargs["split_idx"] = int(split_idx)
        node.args = tuple(args)
        node.kwargs = kwargs

    def _rewrite_piece_graph_attention_ops(self, graph_module: Any,
                                           split_idx: int) -> None:
        if not _MACRO_GRAPH_REWRITE_ATTENTION_OPS:
            return
        graph = getattr(graph_module, "graph", None)
        if graph is None:
            return
        rewritten = 0
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if self._is_vllm_op_target(node.target, "unified_attention"):
                node.target = torch.ops.vllm.macro_unified_attention.default
                self._inject_macro_attention_args(
                    node,
                    split_idx=split_idx,
                    layer_arg_index=3,
                )
                rewritten += 1
                continue
            if self._is_vllm_op_target(node.target,
                                       "unified_attention_with_output"):
                node.target = (
                    torch.ops.vllm.macro_unified_attention_with_output.default)
                self._inject_macro_attention_args(
                    node,
                    split_idx=split_idx,
                    layer_arg_index=4,
                )
                rewritten += 1
        if rewritten:
            graph.lint()
            graph_module.recompile()

    def _resolve_piece_callable(self, target: str, args: Any,
                                split_idx: int) -> Any:
        module = self._fetch_attr(str(target))
        unwrap = getattr(module, "unwrap", None)
        if callable(unwrap):
            module = unwrap()
        raw_graph = getattr(module, "graph", None)
        sym_shape_indices = getattr(module, "sym_shape_indices", None)
        find_range = getattr(module, "_find_range_for_shape", None)
        if callable(raw_graph):
            if sym_shape_indices:
                runtime_shape = args[int(sym_shape_indices[0])]
                if callable(find_range) and find_range(runtime_shape) is None:
                    raise RuntimeError(
                        "Cannot resolve piecewise backend range for macro "
                        "graph "
                        f"target={target!r}, runtime_shape={runtime_shape!r}")
            piece_graph = deepcopy(raw_graph)
            self._rewrite_piece_graph_attention_ops(piece_graph, split_idx)
            self._fold_piece_graph_python_constants(piece_graph)
            return piece_graph
        if hasattr(module, "graph") and callable(module):
            piece_graph = deepcopy(module)
            self._rewrite_piece_graph_attention_ops(piece_graph, split_idx)
            self._fold_piece_graph_python_constants(piece_graph)
            return piece_graph
        return module

    def _fold_piece_graph_python_constants(self, graph_module: Any) -> None:
        if getattr(graph_module, "_vllm_ascend_macro_constants_folded", False):
            return
        graph = getattr(graph_module, "graph", None)
        if graph is None:
            return

        def _contains_fx_node(value: Any) -> bool:
            if isinstance(value, torch.fx.Node):
                return True
            if isinstance(value, (list, tuple)):
                return any(_contains_fx_node(item) for item in value)
            if isinstance(value, dict):
                return any(_contains_fx_node(item) for item in value.values())
            return False

        def _replace_arg(value: Any, old_node: Any, replacement: Any) -> Any:
            return map_arg(value,
                           lambda node: replacement
                           if node is old_node else node)

        folded = 0
        for node in list(graph.nodes):

            def _replace_device_constants(value: Any) -> Any:
                nonlocal folded
                if isinstance(value, torch.device):
                    folded += 1
                    return str(value)
                if isinstance(value, tuple):
                    return tuple(_replace_device_constants(item)
                                 for item in value)
                if isinstance(value, list):
                    return [_replace_device_constants(item) for item in value]
                if isinstance(value, dict):
                    return {
                        key: _replace_device_constants(item)
                        for key, item in value.items()
                    }
                return value

            node.args = _replace_device_constants(node.args)
            node.kwargs = _replace_device_constants(node.kwargs)

        for node in list(graph.nodes):
            if (node.op != "call_function"
                    or not (node.target is torch.device
                            or node.target == torch.device)):
                continue
            if _contains_fx_node(node.args) or _contains_fx_node(node.kwargs):
                replacement = f"npu:{torch.npu.current_device()}"
            else:
                replacement = str(node.target(*node.args, **node.kwargs))
            for user in list(node.users):
                user.args = _replace_arg(user.args, node, replacement)
                user.kwargs = _replace_arg(user.kwargs, node, replacement)
            graph.erase_node(node)
            folded += 1
        if folded:
            graph.lint()
            graph_module.recompile()
        setattr(graph_module, "_vllm_ascend_macro_constants_folded", True)

    def _build_piece_callables(self) -> dict[tuple[str, int], Any]:
        callables: dict[tuple[str, int], Any] = {}
        for kind, _node_idx, target, args_spec, _kwargs_spec in self._node_specs:
            if kind not in ("serial_piece", "parallel_piece"):
                continue
            target = str(target)
            for split_idx, env in enumerate(self._initial_envs):
                key = (target, split_idx)
                if key in callables:
                    continue
                args = self._resolve_spec(args_spec, env)
                callables[key] = self._resolve_piece_callable(
                    target, args, split_idx)
        return callables

    def _eval_spec(self, kind: str, target: Any, args_spec: Any,
                   kwargs_spec: Any, env: list[Any]) -> Any:
        if kind == "get_attr":
            return self._fetch_attr(str(target))
        args = self._resolve_spec(args_spec, env)
        kwargs = self._resolve_spec(kwargs_spec, env)
        if kind == "call_function":
            return target(*args, **kwargs)
        if kind == "call_method":
            self_obj, *args_tail = args
            return getattr(self_obj, str(target))(*args_tail, **kwargs)
        if kind == "call_module":
            module = self._fetch_attr(str(target))
            return module(*args, **kwargs)
        raise RuntimeError(f"Unsupported macro graph spec kind={kind!r}")

    def _record_main_done(self) -> None:
        torch.ops.air.tagged_event_record(self._main_done_tag)

    def _wait_main_done_on_secondary(self) -> None:
        torch.ops.air.tagged_event_wait(self._main_done_tag)

    def _record_secondary_done(self) -> None:
        torch.ops.air.tagged_event_record(self._secondary_done_tag)

    def _wait_secondary_done_on_main(self) -> None:
        torch.ops.air.tagged_event_wait(self._secondary_done_tag)

    def _enter_secondary_stream(self) -> None:
        torch.ops.air.scope_enter(
            ["_user_stream_label", "_user_stream_priority"],
            [self.secondary_stream_tag, "0"],
        )

    def _exit_secondary_stream(self) -> None:
        torch.ops.air.scope_exit()

    def _record_secondary_stream_tree(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            torch.ops.air.record_tagged_stream_(value,
                                                self.secondary_stream_tag)
        elif isinstance(value, IntermediateTensors):
            for tensor in value.tensors.values():
                self._record_secondary_stream_tree(tensor)
        elif isinstance(value, dict):
            for child in value.values():
                self._record_secondary_stream_tree(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._record_secondary_stream_tree(child)
        return value

    def _run_serial_piece(
        self,
        target: str,
        args_spec: Any,
        kwargs_spec: Any,
        env0: list[Any],
        env1: list[Any],
    ) -> tuple[Any, Any]:
        with override_forward_context(self.contexts[0]):
            out0 = self._call_piece_by_target(target, args_spec, kwargs_spec,
                                              env0, 0)
        self._record_main_done()

        self._enter_secondary_stream()
        try:
            self._wait_main_done_on_secondary()
            with override_forward_context(self.contexts[1]):
                out1 = self._call_piece_by_target(target, args_spec,
                                                  kwargs_spec, env1, 1)
            self._record_secondary_stream_tree(out1)
            self._record_secondary_done()
        finally:
            self._exit_secondary_stream()
        self._wait_secondary_done_on_main()
        return out0, out1

    def _run_parallel_piece(
        self,
        target: str,
        args_spec: Any,
        kwargs_spec: Any,
        env0: list[Any],
        env1: list[Any],
    ) -> tuple[Any, Any]:
        with override_forward_context(self.contexts[0]):
            out0 = self._call_piece_by_target(target, args_spec, kwargs_spec,
                                              env0, 0)
        self._record_main_done()

        self._enter_secondary_stream()
        try:
            with override_forward_context(self.contexts[1]):
                out1 = self._call_piece_by_target(target, args_spec,
                                                  kwargs_spec, env1, 1)
            self._record_secondary_stream_tree(out1)
            self._record_secondary_done()
            self._wait_main_done_on_secondary()
        finally:
            self._exit_secondary_stream()
        self._wait_secondary_done_on_main()
        return out0, out1

    def forward(self) -> Any:
        env0 = list(self._initial_envs[0])
        env1 = list(self._initial_envs[1])
        for kind, node_idx, target, args_spec, kwargs_spec in self._node_specs:
            if kind == "placeholder":
                continue
            if kind == "output":
                return (
                    self._resolve_spec(args_spec, env0),
                    self._resolve_spec(args_spec, env1),
                )
            if kind == "parallel_piece":
                out0, out1 = self._run_parallel_piece(
                    target, args_spec, kwargs_spec, env0, env1)
                env0[node_idx] = out0
                env1[node_idx] = out1
                continue
            if kind == "serial_piece":
                out0, out1 = self._run_serial_piece(
                    target, args_spec, kwargs_spec, env0, env1)
                env0[node_idx] = out0
                env1[node_idx] = out1
                continue

            env0[node_idx] = self._eval_spec(kind, target, args_spec,
                                             kwargs_spec, env0)
            self._enter_secondary_stream()
            try:
                env1[node_idx] = self._eval_spec(kind, target, args_spec,
                                                 kwargs_spec, env1)
                self._record_secondary_stream_tree(env1[node_idx])
                self._record_secondary_done()
            finally:
                self._exit_secondary_stream()
            self._wait_secondary_done_on_main()

        raise RuntimeError("Macro graph piecewise FX graph did not return")


def _record_stream_tree_for_npugraph_ex(value: Any, stream: Any) -> None:
    if isinstance(value, torch.Tensor):
        try:
            if value.device.type == "npu":
                value.record_stream(stream)
        except Exception:
            return
        return
    if isinstance(value, IntermediateTensors):
        for tensor in value.tensors.values():
            _record_stream_tree_for_npugraph_ex(tensor, stream)
        return
    if isinstance(value, dict):
        for child in value.values():
            _record_stream_tree_for_npugraph_ex(child, stream)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _record_stream_tree_for_npugraph_ex(child, stream)


class _NpuGraphExPiecewiseMacroModule(_TorchairTaggedPiecewiseMacroModule):
    """Static two-split piecewise schedule for backend='npugraph_ex'."""

    def __init__(
        self,
        *,
        runtime_calls: list[Any],
        contexts: list[Any],
        context_key: str,
        validate_no_inner_aclgraph: bool,
    ) -> None:
        nn.Module.__init__(self)
        if len(runtime_calls) != 2 or len(contexts) != 2:
            raise RuntimeError(
                "npugraph_ex macro graph currently requires exactly "
                f"2 splits, got runtime_calls={len(runtime_calls)}, "
                f"contexts={len(contexts)}")
        handle = runtime_calls[0].handle
        if runtime_calls[1].handle is not handle:
            raise RuntimeError(
                "Cannot build a macro graph from different piecewise handles")
        self.runtime_calls = runtime_calls
        self.contexts = contexts
        self._attention_context_key = str(context_key)
        _MACRO_ATTENTION_CONTEXTS[self._attention_context_key] = self.contexts
        self.handle = handle
        self.split_gm = handle.split_gm
        self._nodes = tuple(handle.split_gm.graph.nodes)
        self._node_to_idx = {
            node: idx
            for idx, node in enumerate(self._nodes)
        }
        self._pieces = {
            item.submod_name: item
            for item in handle.piecewise_graphs
        }
        self._initial_envs = (
            self._bind_placeholders(runtime_calls[0]),
            self._bind_placeholders(runtime_calls[1]),
        )
        self._node_specs = tuple(self._build_node_specs())
        self._piece_callables = self._build_piece_callables()
        if validate_no_inner_aclgraph:
            self._validate_no_inner_aclgraph()

    def _run_serial_piece(
        self,
        target: str,
        args_spec: Any,
        kwargs_spec: Any,
        env0: list[Any],
        env1: list[Any],
    ) -> tuple[Any, Any]:
        with override_forward_context(self.contexts[0]):
            out0 = self._call_piece_by_target(target, args_spec, kwargs_spec,
                                              env0, 0)
        with override_forward_context(self.contexts[1]):
            out1 = self._call_piece_by_target(target, args_spec, kwargs_spec,
                                              env1, 1)
        return out0, out1

    def _run_parallel_piece(
        self,
        target: str,
        args_spec: Any,
        kwargs_spec: Any,
        env0: list[Any],
        env1: list[Any],
    ) -> tuple[Any, Any]:
        if _MACRO_GRAPH_NPUGRAPH_EX_FORCE_SERIAL_REPLAY:
            return self._run_serial_piece(target, args_spec, kwargs_spec, env0,
                                          env1)

        secondary = torch.npu.Stream()
        fork = torch.npu.Event()
        join = torch.npu.Event()
        fork.record()

        with torch.npu.stream(secondary):
            fork.wait(secondary)
            args1 = self._resolve_spec(args_spec, env1)
            kwargs1 = self._resolve_spec(kwargs_spec, env1)
            _record_stream_tree_for_npugraph_ex((args1, kwargs1), secondary)
            with override_forward_context(self.contexts[1]):
                out1 = self._call_piece_by_target(target, args_spec,
                                                  kwargs_spec, env1, 1)
            _record_stream_tree_for_npugraph_ex(out1, secondary)
            join.record()

        with override_forward_context(self.contexts[0]):
            out0 = self._call_piece_by_target(target, args_spec, kwargs_spec,
                                              env0, 0)
        join.wait(torch.npu.current_stream())
        return out0, out1

    def forward(self) -> Any:
        env0 = list(self._initial_envs[0])
        env1 = list(self._initial_envs[1])
        for kind, node_idx, target, args_spec, kwargs_spec in self._node_specs:
            if kind == "placeholder":
                continue
            if kind == "output":
                return (
                    self._resolve_spec(args_spec, env0),
                    self._resolve_spec(args_spec, env1),
                )
            if kind == "parallel_piece":
                out0, out1 = self._run_parallel_piece(
                    target, args_spec, kwargs_spec, env0, env1)
                env0[node_idx] = out0
                env1[node_idx] = out1
                continue
            if kind == "serial_piece":
                out0, out1 = self._run_serial_piece(
                    target, args_spec, kwargs_spec, env0, env1)
                env0[node_idx] = out0
                env1[node_idx] = out1
                continue

            env0[node_idx] = self._eval_spec(kind, target, args_spec,
                                             kwargs_spec, env0)
            env1[node_idx] = self._eval_spec(kind, target, args_spec,
                                             kwargs_spec, env1)

        raise RuntimeError("Macro graph piecewise FX graph did not return")


def _new_macro_graph_bind_detail() -> dict[str, Any]:
    return {
        "tensor_pair_count": 0,
        "tensor_copy_count": 0,
        "tensor_copy_bytes": 0,
        "tensor_copy_full_count": 0,
        "tensor_copy_prefix_count": 0,
        "tensor_copy_same_storage_clone_count": 0,
        "tensor_skip_same_ptr_count": 0,
        "tensor_skip_static_count": 0,
        "tensor_zero_tail_count": 0,
        "metadata_attr_update_count": 0,
        "metadata_attr_skip_same_value_count": 0,
        "metadata_attrs": {},
        "by_category": {},
        "by_field": {},
    }


def _macro_graph_bind_bucket(parent: dict[str, Any],
                             name: str) -> dict[str, Any]:
    if name not in parent:
        parent[name] = {
            "tensor_pair_count": 0,
            "tensor_copy_count": 0,
            "tensor_copy_bytes": 0,
            "tensor_copy_full_count": 0,
            "tensor_copy_prefix_count": 0,
            "tensor_copy_same_storage_clone_count": 0,
            "tensor_skip_same_ptr_count": 0,
            "tensor_skip_static_count": 0,
            "tensor_zero_tail_count": 0,
            "metadata_attr_update_count": 0,
            "metadata_attr_skip_same_value_count": 0,
            "metadata_attrs": {},
        }
    return parent[name]


def _macro_graph_bind_field_name(category: str) -> str:
    if category.startswith("split") and "." in category:
        return category.split(".", 1)[1]
    return category


def _macro_graph_bind_category(detail: dict[str, Any],
                               category: str) -> dict[str, Any]:
    categories = detail.setdefault("by_category", {})
    return _macro_graph_bind_bucket(categories, category)


def _macro_graph_bind_field(detail: dict[str, Any],
                            category: str) -> dict[str, Any]:
    fields = detail.setdefault("by_field", {})
    return _macro_graph_bind_bucket(fields,
                                    _macro_graph_bind_field_name(category))


def _macro_graph_bind_incr(detail: Optional[dict[str, Any]],
                           category: str,
                           key: str,
                           value: int = 1) -> None:
    if detail is None:
        return
    detail[key] = int(detail.get(key, 0)) + int(value)
    category_detail = _macro_graph_bind_category(detail, category)
    category_detail[key] = int(category_detail.get(key, 0)) + int(value)
    field_detail = _macro_graph_bind_field(detail, category)
    field_detail[key] = int(field_detail.get(key, 0)) + int(value)


def _macro_graph_bind_record_attr(detail: Optional[dict[str, Any]],
                                  category: str,
                                  attr: str) -> None:
    if detail is None:
        return
    attrs = detail.setdefault("metadata_attrs", {})
    attrs[attr] = int(attrs.get(attr, 0)) + 1
    category_detail = _macro_graph_bind_category(detail, category)
    category_attrs = category_detail.setdefault("metadata_attrs", {})
    category_attrs[attr] = int(category_attrs.get(attr, 0)) + 1
    field_detail = _macro_graph_bind_field(detail, category)
    field_attrs = field_detail.setdefault("metadata_attrs", {})
    field_attrs[attr] = int(field_attrs.get(attr, 0)) + 1


def _macro_graph_values_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, list) and isinstance(rhs, list):
        return lhs == rhs
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return lhs == rhs
    if isinstance(lhs, (list, tuple)) or isinstance(rhs, (list, tuple)):
        try:
            return list(lhs) == list(rhs)
        except Exception:
            return False
    try:
        return bool(lhs == rhs)
    except Exception:
        return False


def _macro_graph_tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _macro_graph_storage_ptr(tensor: torch.Tensor) -> Optional[int]:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        return None


def _macro_graph_path_get(root: Any, path: tuple[Any, ...]) -> Any:
    item = root
    for part in path:
        if item is None:
            return None
        if isinstance(part, int):
            if not isinstance(item, (list, tuple)) or part >= len(item):
                return None
            item = item[part]
            continue
        if isinstance(item, dict):
            item = item.get(part)
            continue
        item = getattr(item, str(part), None)
    return item


def _macro_graph_copy_tensor_value(
        dst_tensor: torch.Tensor,
        src_tensor: torch.Tensor,
        *,
        binding_name: str,
        copy_policy: str,
        copy_dim: int = 0,
        copy_len: Optional[int] = None,
        zero_tail: bool = False,
        detail: Optional[dict[str, Any]] = None,
        category: str = "allowlist") -> bool:
    if dst_tensor.data_ptr() == src_tensor.data_ptr():
        _macro_graph_bind_incr(detail, category, "tensor_pair_count")
        _macro_graph_bind_incr(detail, category, "tensor_skip_same_ptr_count")
        return False
    if dst_tensor.device != src_tensor.device:
        raise RuntimeError(
            "macro graph binding device mismatch: "
            f"name={binding_name}, dst={dst_tensor.device}, "
            f"src={src_tensor.device}")
    if dst_tensor.dtype != src_tensor.dtype:
        raise RuntimeError(
            "macro graph binding dtype mismatch: "
            f"name={binding_name}, dst={dst_tensor.dtype}, "
            f"src={src_tensor.dtype}")
    if dst_tensor.ndim != src_tensor.ndim:
        raise RuntimeError(
            "macro graph binding ndim mismatch: "
            f"name={binding_name}, dst_shape={tuple(dst_tensor.shape)}, "
            f"src_shape={tuple(src_tensor.shape)}")
    if any(int(dst) < int(src)
           for dst, src in zip(dst_tensor.shape, src_tensor.shape)):
        raise RuntimeError(
            "macro graph binding destination is smaller than source: "
            f"name={binding_name}, dst_shape={tuple(dst_tensor.shape)}, "
            f"src_shape={tuple(src_tensor.shape)}")

    copy_dim = int(copy_dim)
    if copy_dim < 0:
        copy_dim += int(dst_tensor.ndim)
    if copy_dim < 0 or copy_dim >= int(dst_tensor.ndim):
        raise RuntimeError(
            "macro graph binding copy_dim out of range: "
            f"name={binding_name}, copy_dim={copy_dim}, "
            f"ndim={dst_tensor.ndim}")

    src_view = src_tensor
    dst_view = dst_tensor
    prefix_copy = tuple(dst_tensor.shape) != tuple(src_tensor.shape)
    policy_prefix = copy_policy in (
        "prefix",
        "token_prefix",
        "request_prefix",
        "actual_prefix",
    )
    if copy_len is not None:
        copy_len = int(copy_len)
        if copy_len < 0:
            raise RuntimeError(
                "macro graph binding copy_len must be non-negative: "
                f"name={binding_name}, copy_len={copy_len}")
        if copy_len > int(src_tensor.shape[copy_dim]):
            raise RuntimeError(
                "macro graph binding copy_len exceeds source: "
                f"name={binding_name}, copy_len={copy_len}, "
                f"src_shape={tuple(src_tensor.shape)}, copy_dim={copy_dim}")
        if copy_len > int(dst_tensor.shape[copy_dim]):
            raise RuntimeError(
                "macro graph binding copy_len exceeds destination: "
                f"name={binding_name}, copy_len={copy_len}, "
                f"dst_shape={tuple(dst_tensor.shape)}, copy_dim={copy_dim}")
        copy_slice = [slice(None)] * int(src_tensor.ndim)
        copy_slice[copy_dim] = slice(0, copy_len)
        src_view = src_tensor[tuple(copy_slice)]
        dst_view = dst_tensor[tuple(copy_slice)]
        prefix_copy = tuple(dst_tensor.shape) != tuple(src_view.shape)
    elif prefix_copy and policy_prefix:
        dst_view = dst_tensor[tuple(
            slice(0, int(size)) for size in src_tensor.shape)]

    if copy_policy == "full" and tuple(dst_view.shape) != tuple(
            src_tensor.shape):
        raise RuntimeError(
            "macro graph full binding requires exact shape: "
            f"name={binding_name}, dst_shape={tuple(dst_tensor.shape)}, "
            f"src_shape={tuple(src_tensor.shape)}")
    if copy_policy not in (
            "full",
            "prefix",
            "token_prefix",
            "request_prefix",
            "actual_prefix",
            "skip_static",
    ):
        raise RuntimeError(
            "unsupported macro graph tensor binding copy_policy: "
            f"name={binding_name}, copy_policy={copy_policy}")
    if copy_policy == "skip_static":
        _macro_graph_bind_incr(detail, category, "tensor_pair_count")
        _macro_graph_bind_incr(detail, category, "tensor_skip_static_count")
        return False

    _macro_graph_bind_incr(detail, category, "tensor_pair_count")
    copy_src = src_view
    if _macro_graph_storage_ptr(dst_view) == _macro_graph_storage_ptr(
            src_view):
        copy_src = src_view.clone()
        _macro_graph_bind_incr(detail, category,
                               "tensor_copy_same_storage_clone_count")
    dst_view.copy_(copy_src, non_blocking=True)
    if zero_tail and copy_len is not None and copy_len < int(
            dst_tensor.shape[copy_dim]):
        tail_slice = [slice(None)] * int(dst_tensor.ndim)
        tail_slice[copy_dim] = slice(copy_len, int(dst_tensor.shape[copy_dim]))
        dst_tensor[tuple(tail_slice)].fill_(0)
        _macro_graph_bind_incr(detail, category, "tensor_zero_tail_count")
    _macro_graph_bind_incr(detail, category, "tensor_copy_count")
    _macro_graph_bind_incr(detail, category, "tensor_copy_bytes",
                           _macro_graph_tensor_nbytes(src_view))
    _macro_graph_bind_incr(
        detail, category,
        "tensor_copy_prefix_count" if prefix_copy else "tensor_copy_full_count")
    return True


_MACRO_GRAPH_METADATA_CHILD_ATTRS = (
    "prefill",
    "decode_meta",
    "chunked_context",
    "pcp_metadata",
)


def _macro_graph_iter_metadata_leaves(root: Any,
                                      path: tuple[Any, ...] = (),
                                      visited: Optional[set[int]] = None):
    if visited is None:
        visited = set()
    if root is None:
        return
    root_id = id(root)
    if root_id in visited:
        return
    visited.add(root_id)
    if isinstance(root, dict):
        for key, value in root.items():
            yield from _macro_graph_iter_metadata_leaves(value, path + (key, ),
                                                        visited)
        return
    if isinstance(root, (list, tuple)):
        for idx, value in enumerate(root):
            yield from _macro_graph_iter_metadata_leaves(value, path + (idx, ),
                                                        visited)
        return
    yield path, root
    for attr in _MACRO_GRAPH_METADATA_CHILD_ATTRS:
        child = getattr(root, attr, None)
        if child is not None:
            yield from _macro_graph_iter_metadata_leaves(child, path + (attr, ),
                                                        visited)


_MACRO_GRAPH_METADATA_TENSOR_ATTRS = (
    "input_ids",
    "positions",
    "inputs_embeds",
    "query_start_loc",
    "seq_lens",
    "block_tables",
    "block_table",
    "block_table_tensor",
    "slot_mapping",
    "actual_seq_lengths_q",
    "attn_mask",
    "spec_attn_mask",
    "batch_seq_mask",
    "input_positions",
    "sin",
    "cos",
)

_MACRO_GRAPH_METADATA_SCALAR_ATTRS = (
    "seq_lens_list",
    "actual_seq_lengths_q",
    "num_actual_tokens",
    "num_actual_tokens_pcp_padded",
    "num_decode_tokens",
    "num_prefill_tokens",
    "num_decodes",
    "num_prefills",
    "attn_state",
    "max_query_len",
    "max_seq_lens",
    "num_input_tokens",
)


def _macro_graph_tensor_binding_policy(
        attr: str,
        tensor: torch.Tensor) -> tuple[str, int, bool]:
    if attr in ("attn_mask", "spec_attn_mask"):
        return "skip_static", 0, False
    if attr in ("input_ids", "positions", "inputs_embeds", "slot_mapping",
                "input_positions", "sin", "cos"):
        token_dim = 1 if attr in ("positions",
                                  "input_positions") and tensor.ndim == 2 else 0
        return "token_prefix", token_dim, True
    if attr in ("query_start_loc", ):
        return "request_prefix", 0, False
    if attr in ("seq_lens", "block_tables", "block_table",
                "block_table_tensor", "actual_seq_lengths_q",
                "batch_seq_mask"):
        return "request_prefix", 0, True
    return "prefix", 0, False


def _macro_graph_build_tensor_bindings_for_object(
        captured_obj: Any,
        current_obj: Any,
        *,
        root_path: tuple[Any, ...],
        name_prefix: str,
        required: bool = False) -> list[MacroGraphTensorBinding]:
    bindings: list[MacroGraphTensorBinding] = []
    if captured_obj is None:
        return bindings
    for attr in _MACRO_GRAPH_METADATA_TENSOR_ATTRS:
        captured_tensor = getattr(captured_obj, attr, None)
        if not isinstance(captured_tensor, torch.Tensor):
            continue
        current_tensor = getattr(current_obj, attr, None)
        if current_tensor is None:
            if required:
                raise RuntimeError(
                    "missing required macro graph tensor binding source: "
                    f"{name_prefix}.{attr}")
            continue
        if not isinstance(current_tensor, torch.Tensor):
            raise RuntimeError(
                "macro graph tensor binding source is not a tensor: "
                f"{name_prefix}.{attr}, type={type(current_tensor).__name__}")
        copy_policy, copy_dim, zero_tail = _macro_graph_tensor_binding_policy(
            attr, captured_tensor)
        bindings.append(
            MacroGraphTensorBinding(
                name=f"{name_prefix}.{attr}",
                dst=captured_tensor,
                src_path=root_path + (attr, ),
                max_shape=tuple(int(size) for size in captured_tensor.shape),
                copy_policy=copy_policy,
                copy_dim=copy_dim,
                zero_tail=zero_tail,
                required=required,
            ))
    return bindings


def _macro_graph_build_scalar_bindings_for_object(
        captured_obj: Any,
        current_obj: Any,
        *,
        dst_parent_path: tuple[Any, ...],
        src_parent_path: tuple[Any, ...],
        name_prefix: str) -> list[MacroGraphScalarBinding]:
    bindings: list[MacroGraphScalarBinding] = []
    if captured_obj is None:
        return bindings
    for attr in _MACRO_GRAPH_METADATA_SCALAR_ATTRS:
        if not hasattr(captured_obj, attr) or not hasattr(current_obj, attr):
            continue
        value = getattr(captured_obj, attr)
        if isinstance(value, torch.Tensor):
            continue
        bindings.append(
            MacroGraphScalarBinding(
                name=f"{name_prefix}.{attr}",
                dst_parent_path=dst_parent_path,
                src_parent_path=src_parent_path,
                attr=attr,
            ))
    return bindings


def _macro_graph_build_intermediate_tensor_bindings(
        captured: Optional[IntermediateTensors],
        current: Optional[IntermediateTensors],
        *,
        root_path: tuple[Any, ...],
        name_prefix: str) -> list[MacroGraphTensorBinding]:
    bindings: list[MacroGraphTensorBinding] = []
    if captured is None:
        return bindings
    if current is None:
        raise RuntimeError(
            "missing required macro graph intermediate_tensors source: "
            f"{name_prefix}")
    captured_tensors = getattr(captured, "tensors", None)
    current_tensors = getattr(current, "tensors", None)
    if not isinstance(captured_tensors, dict) or not isinstance(
            current_tensors, dict):
        raise RuntimeError(
            "macro graph intermediate_tensors must expose tensor dicts: "
            f"{name_prefix}")
    for name, dst_tensor in captured_tensors.items():
        if not isinstance(dst_tensor, torch.Tensor):
            continue
        if name not in current_tensors:
            raise RuntimeError(
                "missing required macro graph intermediate tensor source: "
                f"{name_prefix}.{name}")
        src_tensor = current_tensors[name]
        if not isinstance(src_tensor, torch.Tensor):
            raise RuntimeError(
                "macro graph intermediate tensor source is not a tensor: "
                f"{name_prefix}.{name}, type={type(src_tensor).__name__}")
        bindings.append(
            MacroGraphTensorBinding(
                name=f"{name_prefix}.{name}",
                dst=dst_tensor,
                src_path=root_path + ("tensors", name),
                max_shape=tuple(int(size) for size in dst_tensor.shape),
                copy_policy="prefix",
                required=True,
            ))
    return bindings


def _macro_graph_clone_runtime_value(value: Any,
                                     *,
                                     max_depth: int = 12,
                                     _depth: int = 0,
                                     _memo: Optional[dict[int, Any]] = None) -> Any:
    if _memo is None:
        _memo = {}
    if _depth > max_depth or value is None:
        return value
    value_id = id(value)
    if value_id in _memo:
        return _memo[value_id]
    if isinstance(value, torch.Tensor):
        cloned = value.clone()
        _memo[value_id] = cloned
        return cloned
    if isinstance(value, IntermediateTensors):
        cloned = IntermediateTensors({
            name: _macro_graph_clone_runtime_value(
                tensor,
                max_depth=max_depth,
                _depth=_depth + 1,
                _memo=_memo,
            )
            for name, tensor in value.tensors.items()
        })
        _memo[value_id] = cloned
        return cloned
    if isinstance(value, dict):
        cloned_dict: dict[Any, Any] = {}
        _memo[value_id] = cloned_dict
        for key, child in value.items():
            cloned_dict[key] = _macro_graph_clone_runtime_value(
                child,
                max_depth=max_depth,
                _depth=_depth + 1,
                _memo=_memo,
            )
        return cloned_dict
    if isinstance(value, list):
        cloned_list: list[Any] = []
        _memo[value_id] = cloned_list
        cloned_list.extend(
            _macro_graph_clone_runtime_value(
                child,
                max_depth=max_depth,
                _depth=_depth + 1,
                _memo=_memo,
            ) for child in value)
        return cloned_list
    if isinstance(value, tuple):
        cloned_tuple = tuple(
            _macro_graph_clone_runtime_value(
                child,
                max_depth=max_depth,
                _depth=_depth + 1,
                _memo=_memo,
            ) for child in value)
        _memo[value_id] = cloned_tuple
        return cloned_tuple
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kwargs = {
            field.name: _macro_graph_clone_runtime_value(
                getattr(value, field.name),
                max_depth=max_depth,
                _depth=_depth + 1,
                _memo=_memo,
            )
            for field in dataclasses.fields(value)
        }
        cloned = dataclasses.replace(value, **kwargs)
        _memo[value_id] = cloned
        return cloned
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and value.__class__.__module__.startswith(
            "vllm_ascend.attention"):
        cloned = copy(value)
        _memo[value_id] = cloned
        for name, child in attrs.items():
            if name.startswith("__"):
                continue
            setattr(
                cloned,
                name,
                _macro_graph_clone_runtime_value(
                    child,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _memo=_memo,
                ),
            )
        return cloned
    return value


def _macro_graph_clone_ubatch_metadata_slots(
        ubatch_metadata: list[AscendUbatchMetadata]
) -> list[AscendUbatchMetadata]:
    cloned_metadata: list[AscendUbatchMetadata] = []
    memo: dict[int, Any] = {}
    for metadata in ubatch_metadata:
        context = copy(metadata.context)
        context.attn_metadata = _macro_graph_clone_runtime_value(
            getattr(metadata.context, "attn_metadata", None),
            _memo=memo,
        )
        context.positions = _macro_graph_clone_runtime_value(
            getattr(metadata.context, "positions", None),
            _memo=memo,
        )
        cloned_metadata.append(
            AscendUbatchMetadata(
                context=context,
                input_ids=_macro_graph_clone_runtime_value(
                    metadata.input_ids,
                    _memo=memo,
                ),
                positions=_macro_graph_clone_runtime_value(
                    metadata.positions,
                    _memo=memo,
                ),
                inputs_embeds=_macro_graph_clone_runtime_value(
                    metadata.inputs_embeds,
                    _memo=memo,
                ),
                intermediate_tensors=_macro_graph_clone_runtime_value(
                    metadata.intermediate_tensors,
                    _memo=memo,
                ),
                num_tokens=metadata.num_tokens,
            ))
    return cloned_metadata


def _copy_tensor_values_for_macro_graph(
        dst: Any,
        src: Any,
        *,
        max_depth: int = 12,
        detail: Optional[dict[str, Any]] = None,
        category: str = "unknown") -> int:
    """Copy tensors from src into dst, preserving dst object identities."""

    copied = 0
    visited: set[tuple[int, int]] = set()

    def _storage_ptr(tensor: torch.Tensor) -> Optional[int]:
        try:
            return int(tensor.untyped_storage().data_ptr())
        except Exception:
            return None

    def _copy_tensor(dst_tensor: torch.Tensor,
                     src_tensor: torch.Tensor,
                     *,
                     prefix_copy: bool = False) -> bool:
        _macro_graph_bind_incr(detail, category, "tensor_pair_count")
        if dst_tensor.data_ptr() == src_tensor.data_ptr():
            _macro_graph_bind_incr(detail, category,
                                   "tensor_skip_same_ptr_count")
            return False
        copy_src = src_tensor
        if _storage_ptr(dst_tensor) == _storage_ptr(src_tensor):
            copy_src = src_tensor.clone()
            _macro_graph_bind_incr(
                detail, category, "tensor_copy_same_storage_clone_count")
        dst_tensor.copy_(copy_src, non_blocking=True)
        _macro_graph_bind_incr(detail, category, "tensor_copy_count")
        _macro_graph_bind_incr(detail, category, "tensor_copy_bytes",
                               _macro_graph_tensor_nbytes(src_tensor))
        _macro_graph_bind_incr(
            detail, category,
            "tensor_copy_prefix_count" if prefix_copy else
            "tensor_copy_full_count")
        return True

    def _copy(dst_item: Any, src_item: Any, depth: int) -> None:
        nonlocal copied
        if depth > max_depth or dst_item is None or src_item is None:
            return
        pair = (id(dst_item), id(src_item))
        if pair in visited:
            return
        visited.add(pair)

        if isinstance(dst_item, torch.Tensor) and isinstance(
                src_item, torch.Tensor):
            if tuple(dst_item.shape) != tuple(src_item.shape):
                if (dst_item.ndim != src_item.ndim
                        or any(int(dst) < int(src) for dst, src in zip(
                            dst_item.shape, src_item.shape))):
                    raise RuntimeError(
                        "macro graph captured tensor is smaller than the "
                        "runtime tensor: "
                        f"captured_shape={tuple(dst_item.shape)}, "
                        f"runtime_shape={tuple(src_item.shape)}")
                if (dst_item.ndim == src_item.ndim
                        and all(int(dst) >= int(src) for dst, src in zip(
                            dst_item.shape, src_item.shape))):
                    dst_view = dst_item[tuple(
                        slice(0, int(size)) for size in src_item.shape)]
                    if _copy_tensor(dst_view, src_item, prefix_copy=True):
                        copied += 1
                return
            if _copy_tensor(dst_item, src_item):
                copied += 1
            return

        if isinstance(dst_item, IntermediateTensors) and isinstance(
                src_item, IntermediateTensors):
            for name, dst_tensor in dst_item.tensors.items():
                if name in src_item.tensors:
                    _copy(dst_tensor, src_item.tensors[name], depth + 1)
            return

        if isinstance(dst_item, dict) and isinstance(src_item, dict):
            for key, dst_child in dst_item.items():
                if key in src_item:
                    _copy(dst_child, src_item[key], depth + 1)
            return

        if isinstance(dst_item, (list, tuple)) and isinstance(
                src_item, (list, tuple)):
            for dst_child, src_child in zip(dst_item, src_item):
                _copy(dst_child, src_child, depth + 1)
            return

        dst_attrs = getattr(dst_item, "__dict__", None)
        src_attrs = getattr(src_item, "__dict__", None)
        if isinstance(dst_attrs, dict) and isinstance(src_attrs, dict):
            for name, dst_child in dst_attrs.items():
                if name.startswith("__") or name not in src_attrs:
                    continue
                _copy(dst_child, src_attrs[name], depth + 1)

    _copy(dst, src, 0)
    return copied


def _iter_macro_attn_metadata_pairs(captured: Any, current: Any):
    if captured is None or current is None:
        return
    if isinstance(captured, dict) and isinstance(current, dict):
        for key, captured_child in captured.items():
            if key in current:
                yield from _iter_macro_attn_metadata_pairs(
                    captured_child, current[key])
        return
    if isinstance(captured, (list, tuple)) and isinstance(current,
                                                         (list, tuple)):
        for captured_child, current_child in zip(captured, current):
            yield from _iter_macro_attn_metadata_pairs(captured_child,
                                                       current_child)
        return
    yield captured, current


def _copy_mixed_request_macro_metadata_values(
        captured: Any,
        current: Any,
        *,
        detail: Optional[dict[str, Any]] = None,
        category: str = "attn_metadata.scalar/list") -> int:
    copied = 0
    metadata_attrs = (
        "seq_lens_list",
        "actual_seq_lengths_q",
        "num_actual_tokens",
        "num_actual_tokens_pcp_padded",
        "num_decode_tokens",
        "num_prefill_tokens",
        "num_decodes",
        "num_prefills",
        "attn_state",
    )
    for captured_meta, current_meta in _iter_macro_attn_metadata_pairs(
            captured, current):
        for attr in metadata_attrs:
            if not hasattr(current_meta, attr):
                continue
            value = getattr(current_meta, attr)
            if isinstance(value, list):
                value = list(value)
            setattr(captured_meta, attr, value)
            _macro_graph_bind_incr(detail, category,
                                   "metadata_attr_update_count")
            _macro_graph_bind_record_attr(detail, category, attr)
            copied += 1
    return copied


def _first_macro_attn_metadata(attn_metadata: Any) -> Any:
    if isinstance(attn_metadata, dict):
        if not attn_metadata:
            return None
        return next(iter(attn_metadata.values()))
    if isinstance(attn_metadata, (list, tuple)):
        if not attn_metadata:
            return None
        return _first_macro_attn_metadata(attn_metadata[0])
    return attn_metadata


def _macro_context_runtime_shape(context: Any) -> int:
    attn_metadata = _first_macro_attn_metadata(
        getattr(context, "attn_metadata", None))
    actual_seq_lengths_q = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if actual_seq_lengths_q is not None and len(actual_seq_lengths_q) > 0:
        return int(actual_seq_lengths_q[-1])
    split_actual = getattr(context, "split_actual_num_tokens", None)
    if split_actual is not None:
        return int(split_actual)
    descriptor = getattr(context, "batch_descriptor", None)
    if descriptor is not None:
        return int(getattr(descriptor, "num_tokens", 0) or 0)
    return 0


def _macro_graph_debug_list_tail(value: Any, max_items: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        try:
            value = value.detach().cpu().tolist()
        except Exception:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        return []
    return list(value[-int(max_items):])


def _macro_graph_debug_tensor_tail(value: Any,
                                   *,
                                   start: int,
                                   stop: int,
                                   max_rows: int = 3,
                                   max_cols: int = 8) -> Any:
    if not isinstance(value, torch.Tensor):
        return None
    start = max(0, int(start))
    stop = max(start, min(int(stop), int(value.shape[0])))
    if stop <= start:
        return []
    sample_start = max(start, stop - int(max_rows))
    try:
        sample = value[sample_start:stop].detach().cpu()
        if sample.ndim > 1 and int(sample.shape[1]) > int(max_cols):
            sample = sample[:, :int(max_cols)]
        return sample.tolist()
    except Exception:
        return None


def _require_torchair_tagged_backend() -> tuple[Any, Any]:
    try:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
    except Exception as exc:
        raise RuntimeError(
            "macro_graph_config.enabled=True requires torchair tagged-event "
            "support. Import torch_npu before torchair and ensure torchair is "
            "installed in the runtime environment.") from exc
    return tng, CompilerConfig


def _require_npugraph_ex_backend() -> None:
    try:
        backends = set(torch._dynamo.list_backends())
    except Exception as exc:
        raise RuntimeError(
            "macro_graph_config.backend='npugraph_ex' requires "
            "torch._dynamo.list_backends() to be available.") from exc
    if "npugraph_ex" not in backends:
        raise RuntimeError(
            "macro_graph_config.backend='npugraph_ex' requires torch-npu "
            "with a registered 'npugraph_ex' torch.compile backend. "
            f"Available torch.compile backends: {sorted(backends)}")


def _clone_attn_metadata_block_tables(attn_metadata: Any) -> Any:
    """Return a copy of attn_metadata with cloned block_tables tensors.

    When capturing ACL graphs for the parallel stream, both _graph_params and
    _graph_params_parallel would otherwise bind the *same* device block_table
    storage (from self.input_batch.block_table).  At runtime the two concurrent
    _refresh_block_table_in_place calls would both write to block_table[:N, :]
    starting from row 0, causing a data race that corrupts KV-cache lookups.

    Cloning gives _graph_params_parallel its own device buffer so the two
    in-place refreshes target distinct memory regions.
    """
    import dataclasses

    def _clone_single(meta: Any) -> Any:
        if meta is None or not dataclasses.is_dataclass(meta):
            return meta
        kwargs: dict = {}
        if getattr(meta, "block_tables", None) is not None:
            kwargs["block_tables"] = meta.block_tables.clone()
        # Also clone sub-metadata that carry their own block_tables.
        for sub_field in ("prefill", "decode_meta"):
            sub = getattr(meta, sub_field, None)
            if sub is not None and dataclasses.is_dataclass(sub):
                sub_kwargs: dict = {}
                if getattr(sub, "block_tables", None) is not None:
                    sub_kwargs["block_tables"] = sub.block_tables.clone()
                if sub_kwargs:
                    kwargs[sub_field] = dataclasses.replace(sub, **sub_kwargs)
        return dataclasses.replace(meta, **kwargs) if kwargs else meta

    if isinstance(attn_metadata, dict):
        return {k: _clone_single(v) for k, v in attn_metadata.items()}
    if isinstance(attn_metadata, list):
        return [
            {k: _clone_single(v) for k, v in d.items()}
            if isinstance(d, dict) else _clone_single(d)
            for d in attn_metadata
        ]
    return _clone_single(attn_metadata)


def _safe_tensor_ptr(tensor: Any) -> Optional[int]:
    if isinstance(tensor, torch.Tensor):
        return int(tensor.data_ptr())
    return None


def _safe_tensor_head(tensor: Any, max_items: int = 3) -> Any:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        if tensor.ndim == 2:
            return tensor[:, :max_items].detach().cpu().tolist()
        return tensor[:max_items].detach().cpu().tolist()
    except Exception:
        return None


def _safe_tensor_shape(tensor: Any) -> Optional[list[int]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return list(tensor.shape)
    except Exception:
        return None


def _macro_tensor_storage_base_ptr(tensor: Any) -> Optional[int]:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        return None


def _macro_tensor_debug_info(tensor: Any) -> Optional[dict[str, Any]]:
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
        "storage_base_ptr": _macro_tensor_storage_base_ptr(tensor),
        "shape": _safe_tensor_shape(tensor),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": stride,
        "storage_offset": storage_offset,
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _macro_same_tensor_ref(left: Any, right: Any) -> bool:
    left_ptr = _safe_tensor_ptr(left)
    right_ptr = _safe_tensor_ptr(right)
    if left_ptr is not None or right_ptr is not None:
        return left_ptr == right_ptr
    return left is right


def _macro_join_dim0_view(first: Any,
                          second: Any) -> tuple[Optional[torch.Tensor], str]:
    if not isinstance(first, torch.Tensor) or not isinstance(second,
                                                             torch.Tensor):
        return None, "non_tensor"
    if first.ndim == 0 or second.ndim == 0:
        return None, "scalar_tensor"
    if first.device != second.device or first.dtype != second.dtype:
        return None, "device_or_dtype_mismatch"
    if first.ndim != second.ndim or tuple(first.shape[1:]) != tuple(
            second.shape[1:]):
        return None, "shape_tail_mismatch"
    try:
        first_stride = tuple(first.stride())
        second_stride = tuple(second.stride())
    except Exception:
        return None, "stride_unavailable"
    if first_stride != second_stride:
        return None, "stride_mismatch"
    base0 = _macro_tensor_storage_base_ptr(first)
    base1 = _macro_tensor_storage_base_ptr(second)
    if base0 is None or base1 is None or base0 != base1:
        return None, "different_storage"
    try:
        offset0 = int(first.storage_offset())
        offset1 = int(second.storage_offset())
    except Exception:
        return None, "storage_offset_unavailable"
    expected_offset1 = offset0 + int(first.shape[0]) * int(first_stride[0])
    if offset1 != expected_offset1:
        return None, "not_contiguous_dim0_slices"
    full_shape = (int(first.shape[0]) + int(second.shape[0]),
                  *tuple(int(v) for v in first.shape[1:]))
    try:
        return first.as_strided(full_shape, first_stride, offset0), "ok"
    except Exception as exc:
        return None, f"as_strided_failed:{type(exc).__name__}"


def _macro_graph_attn_metadata_len(attn_metadata: Any) -> Optional[int]:
    try:
        return int(len(attn_metadata))
    except Exception:
        return None


def _macro_graph_param_key_repr(key: Any) -> str:
    try:
        return json.dumps(graph_param_key_info(key), sort_keys=True)
    except Exception:
        return repr(key)


def _macro_graph_object_id_info(value: Any, *, max_items: int = 2) -> Any:
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
                _macro_graph_object_id_info(item, max_items=max_items)
                for item in list(value)[:int(max_items)]
            ],
        }
    return {
        "type": type(value).__name__,
        "id": int(id(value)),
        "repr": repr(value)[:160],
    }


def _macro_graph_object_id_sample(values: Any, *, max_items: int = 2) -> Any:
    if not isinstance(values, (list, tuple)):
        return None
    if not values:
        return []
    indices = [0]
    if len(values) > 1:
        indices.append(len(values) - 1)
    return [{
        "index": int(index),
        "value": _macro_graph_object_id_info(values[index],
                                             max_items=max_items),
    } for index in indices]


def _safe_context_id(context: Any) -> Optional[int]:
    if context is None:
        return None
    try:
        return id(context)
    except Exception:
        return None


def _build_split_tensor_debug(name: str, tensor: Any) -> dict[str, Any]:
    return {
        f"{name}_ptr": _safe_tensor_ptr(tensor),
        f"{name}_shape": _safe_tensor_shape(tensor),
        f"{name}_head": _safe_tensor_head(tensor),
    }


def _set_split_debug_step(context: Any, step_id: Optional[int]) -> None:
    if context is not None and step_id is not None:
        setattr(context, "split_inplace_debug_step_id", step_id)


def _split_debug_step_from_runner(runner: Any) -> Optional[int]:
    return getattr(runner, "_split_inplace_debug_step_id", None)


def _split_output_tensor_stats(value: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "tensor_count": 0,
        "total_numel": 0,
        "tensors": [],
    }
    tensors: list[dict[str, Any]] = stats["tensors"]

    def _walk(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            stats["tensor_count"] += 1
            try:
                stats["total_numel"] += int(item.numel())
            except Exception:
                pass
            if len(tensors) < 8:
                info = split_debug.tensor_info(item)
                if info is not None:
                    info["path"] = path
                    tensors.append(info)
            return
        if isinstance(item, IntermediateTensors):
            for name, tensor in item.tensors.items():
                _walk(tensor, f"{path}.tensors[{name!s}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                _walk(child, f"{path}[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            for idx, child in enumerate(item):
                _walk(child, f"{path}[{idx}]")

    _walk(value, "root")
    return stats


def _macro_capture_plan_req_caps(
        capture_plan: Any,
        *,
        uniform_decode_query_len: int = 1) -> tuple[int, int]:
    actual_tokens = tuple(
        int(v) for v in getattr(capture_plan, "split_actual_tokens", ()))
    split_graph_tokens = tuple(
        int(v) for v in getattr(capture_plan, "split_graph_tokens", ()))
    if len(actual_tokens) != 2 or len(split_graph_tokens) != 2:
        return (0, 0)
    split_num_reqs = getattr(capture_plan, "split_num_reqs", None)
    if split_num_reqs is not None:
        actual_reqs = tuple(int(v) for v in split_num_reqs)
    else:
        q = max(1, int(uniform_decode_query_len))
        if any(tokens % q != 0 for tokens in actual_tokens):
            actual_reqs = actual_tokens
        else:
            actual_reqs = tuple(int(tokens // q) for tokens in actual_tokens)
    if len(actual_reqs) != 2:
        return (0, 0)
    captured_caps = tuple(
        int(reqs) + max(0, int(graph) - int(actual))
        for reqs, actual, graph in zip(actual_reqs, actual_tokens,
                                       split_graph_tokens))
    explicit_caps = getattr(capture_plan, "split_req_caps", None)
    if explicit_caps is not None:
        caps = tuple(int(v) for v in explicit_caps)
        if len(caps) == 2:
            if any(cap < req for cap, req in zip(caps, actual_reqs)):
                return (0, 0)
            return (int(caps[0]), int(caps[1]))
    return (int(captured_caps[0]), int(captured_caps[1]))


def _inplace_plan_to_execution_slices(
        split_mode: str,
        inplace_split_plan: Optional[InplaceSplitPlan],
        *,
        enable_parallel_streams: bool = False,
) -> tuple[Optional[SplitBatchSlices], Optional[UBatchSlices]]:
    if inplace_split_plan is None:
        return None, None
    if split_mode == "inplace_parallel" and not enable_parallel_streams:
        return None, None
    if split_mode not in ("inplace_serial", "inplace_parallel"):
        return None, None
    split_batch_slices = inplace_split_plan.split_slices
    split_ubatch_slices = [
        UBatchSlice(s.request_slice, s.token_slice)
        for s in split_batch_slices
    ]
    return split_batch_slices, split_ubatch_slices


def _dual_stream_attention_config(split_cfg: Any) -> Any:
    if split_cfg is None:
        return None
    return getattr(split_cfg, "dual_stream_attention_config", None)


def _dual_stream_attention_enabled(split_cfg: Any) -> bool:
    cfg = _dual_stream_attention_config(split_cfg)
    return bool(cfg is not None and getattr(cfg, "enabled", False))


def _dual_stream_attention_actual_q_policy(split_cfg: Any) -> str:
    cfg = _dual_stream_attention_config(split_cfg)
    return str(getattr(cfg, "actual_q_policy", "graph"))


def _find_dual_stream_attention_plan(cfg: Any, num_tokens: int,
                                     *, key: str) -> Any:
    num_tokens = int(num_tokens)
    for plan in getattr(cfg, "capture_plans", []):
        if key == "total" and int(plan.total_tokens) == num_tokens:
            return plan
        if key == "graph" and int(plan.graph_tokens) == num_tokens:
            return plan
    return None


def _dual_stream_attention_plan_to_slices(
        plan: Any,
        query_len: int) -> tuple[SplitBatchSlices, UBatchSlices]:
    query_len = int(query_len)
    if query_len <= 0:
        raise RuntimeError(
            "dual_stream_attention_config requires positive query_len")

    split_batch_slices: SplitBatchSlices = []
    for split_idx in range(2):
        token_start = int(plan.split_start_tokens[split_idx])
        actual_tokens = int(plan.split_actual_tokens[split_idx])
        graph_tokens = int(plan.split_graph_tokens[split_idx])
        if token_start % query_len != 0:
            raise RuntimeError(
                "dual_stream_attention_config split_start_tokens must be "
                "request-aligned")
        if actual_tokens % query_len != 0 or graph_tokens % query_len != 0:
            raise RuntimeError(
                "dual_stream_attention_config split tokens must be "
                "request-aligned")
        request_start = token_start // query_len
        request_stop = request_start + actual_tokens // query_len
        split_batch_slices.append(
            SplitBatchSlice(
                request_slice=slice(request_start, request_stop),
                token_slice=slice(token_start, token_start + actual_tokens),
                padded_num_tokens=graph_tokens,
                start_num_tokens=token_start,
            ))

    ubatch_slices = [
        UBatchSlice(s.request_slice, s.token_slice)
        for s in split_batch_slices
    ]
    return split_batch_slices, ubatch_slices


_INPLACE_SPLIT_MODES = ("inplace_serial", "inplace_parallel")
_NO_SPLIT_MIXED_MACRO_GRAPH_ENABLED = "no_split_mixed_macro_graph_enabled"


def _inplace_split_precheck_reason(
        *,
        split_enabled: bool,
        split_mode: str,
        enable_parallel_streams: bool,
        use_aclgraph: bool,
        num_splits: int,
        uniform_decode: bool,
        enable_dbo: bool,
        with_prefill: bool,
        attn_state: Any,
        has_spec_decode_tokens: bool,
        enable_spec_decode: bool,
        has_lora: bool,
        uses_mrope: bool,
        enable_mrope: bool,
        use_mla: bool,
        pcp_size: int,
        dcp_size: int,
) -> Optional[str]:
    """Return the explicit inplace fallback reason, or None if plannable."""
    if split_mode not in _INPLACE_SPLIT_MODES:
        return "no_split_not_inplace_mode"
    if not split_enabled:
        return "no_split_inplace_disabled"
    if split_mode == "inplace_parallel" and not enable_parallel_streams:
        return "no_split_parallel_streams_disabled"
    if not use_aclgraph:
        return "no_split_no_aclgraph"
    if int(num_splits) != 2:
        return "no_split_num_splits_not_two"
    if enable_dbo:
        return "no_split_dbo_active"
    if has_spec_decode_tokens and not enable_spec_decode:
        return "no_split_spec_decode"
    if not uniform_decode:
        return "no_split_non_uniform_decode"
    allowed_attn_states = {AscendAttentionState.DecodeOnly}
    if has_spec_decode_tokens and enable_spec_decode:
        allowed_attn_states.add(AscendAttentionState.SpecDecoding)
    if (has_spec_decode_tokens and enable_spec_decode
            and attn_state not in allowed_attn_states):
        return "no_split_spec_decode_attn_state"
    if with_prefill or attn_state not in allowed_attn_states:
        return "no_split_prefill_or_mixed"
    if has_lora:
        return "no_split_lora"
    if uses_mrope and not enable_mrope:
        return "no_split_mrope"
    if uses_mrope and split_mode != "inplace_serial":
        return "no_split_mrope_parallel"
    if use_mla:
        return "no_split_mla"
    if int(pcp_size) * int(dcp_size) > 1:
        return "no_split_pcp_or_context_parallel"
    return None


def _mixed_request_split_precheck_reason(
        *,
        split_enabled: bool,
        mixed_enabled: bool,
        split_mode: str,
        enable_parallel_streams: bool,
        replay_policy: str,
        use_aclgraph: bool,
        num_splits: int,
        uniform_decode: bool,
        with_prefill: bool,
        enable_dbo: bool,
        dual_stream_attention_enabled: bool,
        has_spec_decode_tokens: bool,
        has_lora: bool,
        uses_mrope: bool,
        use_mla: bool,
        pcp_size: int,
        dcp_size: int,
        cudagraph_mode: CUDAGraphMode,
) -> Optional[str]:
    """Return why request-level mixed split is not plannable."""
    if not mixed_enabled:
        return "no_split_mixed_disabled"
    if not split_enabled:
        return "no_split_inplace_disabled"
    if split_mode not in _INPLACE_SPLIT_MODES:
        return "no_split_not_inplace_mode"
    if split_mode == "inplace_parallel" and not enable_parallel_streams:
        return "no_split_parallel_streams_disabled"
    if (split_mode == "inplace_parallel"
            and replay_policy != "piecewise_attention_parallel"):
        return "no_split_mixed_not_piecewise_policy"
    if not use_aclgraph:
        return "no_split_no_aclgraph"
    if int(num_splits) != 2:
        return "no_split_num_splits_not_two"
    if uniform_decode or not with_prefill:
        return "no_split_mixed_decode_only"
    if enable_dbo:
        return "no_split_dbo_active"
    if dual_stream_attention_enabled:
        return "no_split_dual_stream_attention_enabled"
    if has_spec_decode_tokens:
        return "no_split_unsupported_spec_decode"
    if has_lora:
        return "no_split_unsupported_lora"
    if uses_mrope:
        return "no_split_unsupported_mrope"
    if use_mla:
        return "no_split_unsupported_mla"
    if int(pcp_size) * int(dcp_size) > 1:
        return "no_split_unsupported_pcp_dcp"
    if cudagraph_mode.mixed_mode() != CUDAGraphMode.PIECEWISE:
        return "no_split_runtime_mode_not_piecewise"
    return None


def _extract_attn_positions(attn_metadata: Any) -> tuple[Optional[int], Any]:
    candidate = attn_metadata
    if isinstance(candidate, dict) and candidate:
        candidate = next(iter(candidate.values()))
    if isinstance(candidate, list) and candidate:
        candidate = candidate[0]

    common_attn_metadata = getattr(candidate, "common_attn_metadata", None)
    if common_attn_metadata is not None:
        candidate = common_attn_metadata

    positions = getattr(candidate, "positions", None)
    return _safe_tensor_ptr(positions), _safe_tensor_head(positions)


def _unwrap_single_tensor_output(output: Any) -> Any:
    while isinstance(output, (list, tuple)) and len(output) == 1:
        inner = output[0]
        if isinstance(inner, (torch.Tensor, list, tuple)):
            output = inner
            continue
        break
    return output




def _iter_attn_metadata_objects(attn_metadata: Any):
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            yield from _iter_attn_metadata_objects(value)
        return
    if isinstance(attn_metadata, list):
        for value in attn_metadata:
            yield from _iter_attn_metadata_objects(value)
        return
    if attn_metadata is not None:
        yield attn_metadata


def _template_fia_seq_lens_list(attn_metadata: Any, target_t: int) -> int:
    updated = 0
    for metadata_obj in _iter_attn_metadata_objects(attn_metadata):
        seq_lens_list = getattr(metadata_obj, "seq_lens_list", None)
        if not isinstance(seq_lens_list, list) or not seq_lens_list:
            continue
        templated_seq_lens = list(seq_lens_list)
        templated_seq_lens[-1] = int(target_t)
        setattr(metadata_obj, "seq_lens_list", templated_seq_lens)
        updated += 1
    return updated


def _get_slot_mapping_from_attn_metadata(
        attn_metadata: Any) -> Optional[torch.Tensor]:
    for metadata_obj in _iter_attn_metadata_objects(attn_metadata):
        slot_mapping = getattr(metadata_obj, "slot_mapping", None)
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping
        common_attn_metadata = getattr(metadata_obj, "common_attn_metadata",
                                       None)
        slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping
    return None


def _set_slot_mapping_for_attn_metadata(attn_metadata: Any,
                                        slot_mapping: torch.Tensor) -> int:
    updated = 0
    for metadata_obj in _iter_attn_metadata_objects(attn_metadata):
        if hasattr(metadata_obj, "slot_mapping"):
            setattr(metadata_obj, "slot_mapping", slot_mapping)
            updated += 1
        common_attn_metadata = getattr(metadata_obj, "common_attn_metadata",
                                       None)
        if (common_attn_metadata is not None
                and hasattr(common_attn_metadata, "slot_mapping")):
            setattr(common_attn_metadata, "slot_mapping", slot_mapping)
            updated += 1
    return updated


def _validate_split_attn_metadata_count(
    tag: str,
    common_attn_metadata_list: Any,
    expected_splits: int,
) -> None:
    actual_len = len(common_attn_metadata_list) if isinstance(common_attn_metadata_list, list) else None
    if _SPLIT_METADATA_DEBUG_ENABLED:
        payload = {
            "tag": tag,
            "expected_splits": expected_splits,
            "actual_type": type(common_attn_metadata_list).__name__,
            "actual_len": actual_len,
        }
        _append_split_metadata_debug("split_attn_metadata_count", payload)

    if (
        not isinstance(common_attn_metadata_list, list)
        or actual_len is None
        or actual_len != expected_splits
    ):
        raise RuntimeError(
            "split_attn_metadata returned unexpected split count: "
            f"expected={expected_splits}, actual_type={type(common_attn_metadata_list).__name__}, "
            f"actual_len={actual_len}, tag={tag}"
        )


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group(
            ).rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        if envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP:
            self.prefetch_stream = torch.npu.Stream(device=device)
        else:
            self.prefetch_stream = None
        self.sampler = AscendSampler()
        self.attn_mask = None
        self.attn_state = None

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        set_external_cos_sin_fast_path_enabled(True)
        self._split_inplace_debug_step_id: Optional[int] = None
        self.weight_prefetch_method = WeightPrefetchMethod(
            self.ascend_config.weight_prefetch_config)
        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config
        self.dump_enable = dump_cfg.enable_dump
        self.debugger = None
        if self.dump_enable:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger
                self.debugger = PrecisionDebugger(dump_cfg.config_path)
            else:
                raise RuntimeError(
                    "Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False


        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_config,
                                  "index_topk")
        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            self.block_size,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None
            and self.model_config.is_mm_prefix_lm)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self._set_up_drafter()

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs,
                        self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs,
                                self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.pcp_allgather_restore_idx = torch.zeros(
            self.max_num_tokens + 2 * self.pcp_size * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device)
        self.cp_kv_recover_idx_for_chunk: List[List[int]] = [
            [] for _ in range(self.pcp_size)
        ]

        self.num_pcp_pads = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        self.pcp_padded_slot_mapping = torch.zeros(
            self.max_num_tokens + 2 * self.pcp_size * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device)
        self.num_actual_tokens_pcp_padded = 0
        if self.speculative_config and self.pcp_size > 1:
            self.input_ids_pcp_full = self._make_buffer(self.max_num_tokens,
                                                        dtype=torch.int32)
            self.query_start_loc_pcp_full = self._make_buffer(
                self.max_num_reqs + 1, dtype=torch.int32)
            self.positions_pcp_full = torch.zeros(self.max_num_tokens,
                                                  dtype=torch.int64,
                                                  device="cpu",
                                                  pin_memory=True)
            self.decode_token_per_req += self.speculative_config.num_speculative_tokens
            self.positions_pcp_full_np = self.positions_pcp_full.numpy()
        self.decode_threshold = 1 + (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        self.dynamic_eplb = self.ascend_config.dynamic_eplb or self.ascend_config.expert_map_record_path
        if self.dynamic_eplb:
            EPLBParamUtils.check_dynamic_eplb(self.ascend_config.dynamic_eplb)
            EPLBParamUtils.check_expert_map_record_path(
                self.ascend_config.expert_map_record_path)
            self.is_eplb_warmuped = False
            self.policy_type = self.ascend_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({
                "expert_map": None,
                "moe_load": None,
                "expert_maps": None
            })
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict,
                                            policy_type=self.policy_type,
                                            enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            ascend_config = get_ascend_config()
            self.eplb_updator = EplbUpdator(ascend_config, self.eplb_loader,
                                            self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens
                if self.vllm_config.speculative_config else 0),
            cp_kv_cache_interleave_size=self.parallel_config.
            cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                  dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        #并行新增输入地址
        self.input_ids_parallel_streams=self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.inputs_embeds_parallel_streams=self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.positions_parallel_streams=self._make_buffer(self.max_num_tokens, dtype=torch.int64)

        self.stream_main = torch.npu.current_stream()
        self.stream_parallel = torch.npu.Stream(device=self.device)
        self._macro_graph_registry: dict[tuple[Any, ...], Any] = {}

        # Performance measurement accumulators.
        # _perf_accum holds running totals across all decode steps so that
        # callers can compute TPOT = total_replay_ms / total_output_tokens.
        # _last_step_perf holds the most recent step's breakdown.
        self._perf_accum: dict = {
            "total_replay_ms": 0.0,
            "total_header_ms": 0.0,
            "total_output_tokens": 0,
            "num_decode_steps": 0,
        }
        self._last_step_perf: dict = {}
        self._t_replay_start: float = 0.0
        self._t_replay_end: float = 0.0
        self._t_header_start: float = 0.0


    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.spec_attn_mask = None
        self.drafter: Optional[Union[NgramProposer, EagleProposer, MtpProposer,
                                     SuffixDecodingProposer]] = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            self.spec_attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask(
            )
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                self.rejection_sampler = RejectionSampler(self.sampler)
            self.actual_seq_lengths_q = list(
                range(self.decode_token_per_req, self.max_num_tokens + 1,
                      self.decode_token_per_req))
        self.discard_request_indices = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method,
                                      self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and self.compilation_config.mode == CompilationMode.VLLM_COMPILE and not self.model_config.enforce_eager

    def _skip_all_reduce_acorss_dp_group(self) -> bool:
        # NOTE: We can skip the all_reduce operation and avoid paading tokens
        # to max_tokens_acrodd_dp in D nodes. In MoE models, we must ensure that
        # num_tokens DOES NOT exceed mc2_tokens_capacity which means that moe_comm_method
        # of each rank is MC2. For dense models, skipping all_reduce is not necessary
        # since collective-communication is not time-consuming since dp_size in dense
        # model deployments is always small and can be overlapped by async scheduling.
        if not is_moe_model(self.vllm_config):
            return False
        if self.compilation_config.cudagraph_capture_sizes:
            potential_max_num_tokens = self.compilation_config.max_cudagraph_capture_size
        else:
            potential_max_num_tokens = self.max_num_reqs * self.uniform_decode_query_len
        # To ensure skipping all_reduce across dp group is valid, we need to ensure that
        # moe_comm_method of each rank is MC2 and recomputation would never happen in D
        # nodes. So here we check whether recompute_scheduler_enable is True.
        return self.is_kv_consumer and self.ascend_config.recompute_scheduler_enable and select_moe_comm_method(
            potential_max_num_tokens,
            self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}

    def _sync_metadata_across_dp(
            self, num_tokens: int, with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill, enable_dbo

        if self._skip_all_reduce_acorss_dp_group():
            num_tokens_after_padding = torch.tensor([num_tokens] *
                                                    self.dp_size,
                                                    device="cpu",
                                                    dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, with_prefill, enable_dbo

        # Sync num_tokens, with_prefill across dp ranks
        num_tokens_tensor = torch.tensor([
            num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)
        ],
                                         dtype=torch.int32,
                                         device="cpu")

        flags_tensor = torch.tensor(
            [int(with_prefill), int(enable_dbo)],
            dtype=torch.int32,
            device="cpu")

        packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])
        # use cpu_group to avoid cpu synchronization issue.
        # it can be overlapped with main moell execution on npu.
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[:-2]
        synced_flags = packed_tensor[-2:]

        max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
        global_with_prefill = bool(synced_flags[0])
        # all the ranks should execute dummy run at the same time
        global_enable_dbo = (synced_flags[1] == get_dp_group().world_size
                             or synced_flags[1] == 0)

        # Create a tensor for num_tokens_after_padding
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill, global_enable_dbo

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model,
                      (ACLGraphWrapper, AscendUBatchWrapper)):
            return self.model.unwrap()
        return self.model

    def _make_attention_mask(self, attn_state) -> torch.Tensor:
        # pcp situation.
        if self.attn_mask_builder is None:
            raise ValueError("Attn mask builder is None")
        # Pooling situation.
        if self.model_config.runner_type == "pooling":
            return self.attn_mask_builder.get_attn_mask(2048, torch.bool)

        if self.vllm_config.model_config.use_mla:
            if self.pcp_size > 1:
                return self.attn_mask_builder.get_pcp_mla_mask(self.dtype)
            # mla prefill
            if attn_state != AscendAttentionState.DecodeOnly:
                return self.attn_mask_builder.get_mla_mask(self.dtype)
        return self.attn_mask_builder.get_splitfuse_attn_mask()

    def generate_kv_idx(self, scheduler_output):
        if not self.pcp_size > 1:
            return
        self.cp_kv_recover_idx_for_chunk = [[] for _ in range(self.pcp_size)]

        for i, req_id in enumerate(self.input_batch.req_ids):
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            is_prefill = self.input_batch.num_computed_tokens_cpu[
                i] < self.input_batch.num_prompt_tokens[i]
            if is_prefill:
                num_cp_padded_scheduled_tokens = cdiv(
                    num_scheduled_tokens,
                    2 * self.pcp_size) * (2 * self.pcp_size)
                full_indices = list(
                    range(self.max_num_tokens * self.pcp_size * self.dcp_size +
                          self.pcp_size * self.dcp_size * self.max_num_reqs))
                chunk_size = num_cp_padded_scheduled_tokens // (2 *
                                                                self.pcp_size)
                num_added_recover_tokens = len(
                    self.cp_kv_recover_idx_for_chunk[0]) * self.pcp_size
                for rank in range(self.pcp_size):
                    self.cp_kv_recover_idx_for_chunk[rank].extend(
                        full_indices[rank * chunk_size +
                                     num_added_recover_tokens:(rank + 1) *
                                     chunk_size + num_added_recover_tokens])
                    self.cp_kv_recover_idx_for_chunk[rank].extend(
                        full_indices[num_cp_padded_scheduled_tokens -
                                     (rank + 1) * chunk_size +
                                     num_added_recover_tokens:
                                     num_cp_padded_scheduled_tokens -
                                     rank * chunk_size +
                                     num_added_recover_tokens])

        cp_kv_recover_idx_for_chunk = torch.from_numpy(
            np.concatenate(
                self.cp_kv_recover_idx_for_chunk)).to(device=self.device)
        cp_kv_recover_idx_for_chunk.copy_(torch.tensor(
            np.array(self.cp_kv_recover_idx_for_chunk).flatten().tolist()),
                                          non_blocking=True)
        self.cp_kv_recover_idx_for_chunk = cp_kv_recover_idx_for_chunk.to(
            torch.float32).argsort().to(torch.int32)

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[PerLayerAttnMetadata, torch.Tensor, np.ndarray, int,
               torch.Tensor, int, torch.Tensor, SpecDecodeMetadata,
               Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], int, Optional[UBatchSlices],
               Optional[SplitBatchSlices], Optional[torch.Tensor],
               Optional[str]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        self._dual_stream_attention_metadata = None
        self._dual_stream_attention_slices = None
        self._dual_stream_attention_plan = None
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)

        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        _, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        positions_np = np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
        )

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        total_num_pcp_pads = 0
        if self.pcp_size > 1:
            if not self.vllm_config.model_config.use_mla:
                self.generate_kv_idx(scheduler_output)
            tokens, position_pcp, pcp_unpad_mask = self._update_tokens_for_pcp(
                tokens)
            num_scheduled_tokens = np.array(tokens, dtype=np.int32)
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
            total_num_pcp_pads = torch.sum(self.num_pcp_pads).item()
        else:
            position_pcp, pcp_unpad_mask = None, None
            self.num_pcp_pads = self.num_pcp_pads[:num_reqs]

        max_num_scheduled_tokens = max(tokens)
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(tokens, dtype=np.int32)
        else:
            num_valid_tokens = np.array([
                num_tokens -
                len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(tokens, req_ids)
            ],
                                        dtype=np.int32)

        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.cudagraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = num_tokens_unpadded
        uniform_decode = \
            (max_num_scheduled_tokens == self.uniform_decode_query_len) and \
            (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
        moe_comm_type = select_moe_comm_method(num_input_tokens,
                                               self.vllm_config)
        ubatch_slices, num_tokens_after_padding = \
            ubatch_split(num_scheduled_tokens,
                         num_tokens_unpadded,
                         num_tokens_padded,
                         uniform_decode=uniform_decode,
                         vllm_config=self.vllm_config,
                         moe_comm_type=moe_comm_type,
                         use_mla = self.model_config.use_mla)
        if ubatch_slices is not None:
            enable_dbo = True
        else:
            enable_dbo = False

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_input_tokens,
                                                     with_prefill, enable_dbo)

        if not enable_dbo:
            ubatch_slices = None

        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        split_mode = (getattr(split_cfg, "mode", "parallel_buffer")
                      if split_cfg is not None else "parallel_buffer")
        dual_stream_attention_enabled = _dual_stream_attention_enabled(
            split_cfg)
        split_enabled = bool(split_cfg is not None
                             and getattr(split_cfg, "enabled", False))
        split_enable_parallel_streams = bool(
            split_cfg is not None
            and getattr(split_cfg, "enable_parallel_streams", False))
        macro_graph_cfg = (getattr(split_cfg, "macro_graph_config", None)
                           if split_cfg is not None else None)
        macro_graph_enabled = bool(
            macro_graph_cfg is not None
            and getattr(macro_graph_cfg, "enabled", False))
        inplace_offset_capture_sizes = None
        if split_cfg is not None:
            inplace_offset_capture_sizes = getattr(
                split_cfg, "inplace_offset_capture_sizes", None)
            if inplace_offset_capture_sizes is None:
                inplace_offset_capture_sizes = getattr(
                    split_cfg, "parallel_capture_sizes", None)

        split_debug_enabled = split_debug.is_enabled()
        if split_debug_enabled:
            self._split_inplace_debug_step_id = split_debug.next_step_id()
            split_debug.set_current_step_id(
                self._split_inplace_debug_step_id)
            split_debug.log_event(
                "split_planner_input",
                {
                    "num_reqs": int(num_reqs),
                    "total_num_scheduled_tokens":
                    int(total_num_scheduled_tokens),
                    "num_tokens_unpadded": int(num_tokens_unpadded),
                    "num_tokens_padded": int(num_tokens_padded),
                    "uniform_decode": bool(uniform_decode),
                    "uniform_decode_query_len":
                    int(self.uniform_decode_query_len),
                    "with_prefill": bool(with_prefill),
                    "attn_state": getattr(attn_state, "name",
                                          str(attn_state)),
                    "enable_dbo": bool(ubatch_slices is not None),
                    "scheduled_spec_decode_tokens_count": len(
                        scheduler_output.scheduled_spec_decode_tokens or {}),
                    "has_lora": bool(
                        self.lora_config and len(
                            self.input_batch.lora_id_to_lora_request) > 0),
                    "uses_mrope": bool(self.uses_mrope),
                    "use_mla": bool(self.model_config.use_mla),
                    "pcp_size": int(self.pcp_size),
                    "dcp_size": int(self.dcp_size),
                    "use_aclgraph": bool(self.use_aclgraph),
                    "cudagraph_capture_sizes": list(
                        self.compilation_config.cudagraph_capture_sizes
                        or []),
                    "parallel_capture_sizes": list(
                        getattr(self, "cudagraph_batch_sizes_parallel", [])
                        or []),
                    "split_enabled": bool(
                        split_cfg is not None
                        and getattr(split_cfg, "enabled", False)),
                    "enable_parallel_streams": bool(
                        split_cfg is not None and getattr(
                            split_cfg, "enable_parallel_streams", False)),
                    "split_mode": (getattr(split_cfg, "mode", None)
                                   if split_cfg is not None else None),
                    "dual_stream_attention_enabled":
                    bool(dual_stream_attention_enabled),
                    "num_splits": (int(getattr(split_cfg, "num_splits", 0))
                                   if split_cfg is not None else None),
                    "min_batch_size_for_split": (
                        int(getattr(split_cfg, "min_batch_size_for_split", 0))
                        if split_cfg is not None else None),
                    "force_split": bool(
                        split_cfg is not None
                        and getattr(split_cfg, "force_split", False)),
                    "enable_inplace_lazy_capture": (
                        bool(getattr(split_cfg,
                                     "enable_inplace_lazy_capture", False))
                        if split_cfg is not None else None),
                    "enable_inplace_spec_decode": (
                        bool(getattr(split_cfg, "enable_inplace_spec_decode",
                                     False))
                        if split_cfg is not None else None),
                    "enable_inplace_mrope": (
                        bool(getattr(split_cfg, "enable_inplace_mrope",
                                     False))
                        if split_cfg is not None else None),
                    "inplace_serial_first": (
                        bool(getattr(split_cfg, "inplace_serial_first",
                                     False))
                        if split_cfg is not None else None),
                    "inplace_split_planner_policy": (
                        getattr(split_cfg, "inplace_split_planner_policy",
                                None) if split_cfg is not None else None),
                    "inplace_max_remainder_tokens": (
                        getattr(split_cfg, "inplace_max_remainder_tokens",
                                None) if split_cfg is not None else None),
                    "inplace_validate_metadata_ptrs": (
                        bool(getattr(split_cfg,
                                     "inplace_validate_metadata_ptrs", False))
                        if split_cfg is not None else None),
                    "inplace_offset_match_policy": (
                        getattr(split_cfg, "inplace_offset_match_policy",
                                None) if split_cfg is not None else None),
                    "inplace_offset_capture_sizes": (
                        list(getattr(split_cfg,
                                     "inplace_offset_capture_sizes", []) or [])
                        if split_cfg is not None else None),
                    "inplace_offset_effective_capture_sizes": (
                        list(inplace_offset_capture_sizes or [])
                        if split_cfg is not None else None),
                    "inplace_offset_min_graph_tokens": (
                        getattr(split_cfg, "inplace_offset_min_graph_tokens",
                                None) if split_cfg is not None else None),
                    "inplace_offset_max_padding_tokens": (
                        getattr(split_cfg,
                                "inplace_offset_max_padding_tokens", None)
                        if split_cfg is not None else None),
                    "inplace_offset_max_padding_ratio": (
                        getattr(split_cfg,
                                "inplace_offset_max_padding_ratio", None)
                        if split_cfg is not None else None),
                    "inplace_offset_max_graph_tokens_by_start": (
                        getattr(
                            split_cfg,
                            "inplace_offset_max_graph_tokens_by_start", None)
                        if split_cfg is not None else None),
                    "inplace_offset_allowed_graph_tokens_by_start": (
                        getattr(
                            split_cfg,
                            "inplace_offset_allowed_graph_tokens_by_start",
                            None) if split_cfg is not None else None),
                    "enable_mixed_request_split": (
                        bool(getattr(split_cfg,
                                     "enable_mixed_request_split", False))
                        if split_cfg is not None else None),
                    "mixed_request_split_policy": (
                        getattr(split_cfg, "mixed_request_split_policy", None)
                        if split_cfg is not None else None),
                    "mixed_request_split_execution_mode": (
                        getattr(split_cfg, "mixed_request_split_execution_mode",
                                None) if split_cfg is not None else None),
                    "mixed_request_min_total_tokens": (
                        getattr(split_cfg, "mixed_request_min_total_tokens",
                                None) if split_cfg is not None else None),
                    "mixed_request_min_tokens_per_split": (
                        getattr(split_cfg,
                                "mixed_request_min_tokens_per_split", None)
                        if split_cfg is not None else None),
                    "mixed_request_max_single_request_ratio": (
                        getattr(split_cfg,
                                "mixed_request_max_single_request_ratio", None)
                        if split_cfg is not None else None),
                    "mixed_request_max_padding_tokens_per_split": (
                        getattr(
                            split_cfg,
                            "mixed_request_max_padding_tokens_per_split",
                            None) if split_cfg is not None else None),
                    "mixed_request_max_padding_ratio_per_split": (
                        getattr(
                            split_cfg,
                            "mixed_request_max_padding_ratio_per_split",
                            None) if split_cfg is not None else None),
                    "mixed_request_min_prefill_reqs_for_prefill_split": (
                        getattr(
                            split_cfg,
                            "mixed_request_min_prefill_reqs_for_prefill_split",
                            None) if split_cfg is not None else None),
                    "mixed_request_decode_weight": (
                        getattr(split_cfg, "mixed_request_decode_weight",
                                None) if split_cfg is not None else None),
                    "mixed_request_prefill_weight": (
                        getattr(split_cfg, "mixed_request_prefill_weight",
                                None) if split_cfg is not None else None),
                    "macro_graph_enabled": macro_graph_enabled,
                    "macro_graph_plan_source": (
                        getattr(macro_graph_cfg, "plan_source", None)
                        if macro_graph_cfg is not None else None),
                    "macro_graph_miss_policy": (
                        getattr(macro_graph_cfg, "miss_policy", None)
                        if macro_graph_cfg is not None else None),
                },
                step_id=self._split_inplace_debug_step_id,
            )
        else:
            self._split_inplace_debug_step_id = None
            split_debug.set_current_step_id(None)

        split_batch_slices: Optional[SplitBatchSlices] = None
        split_ubatch_slices: Optional[UBatchSlices] = None
        inplace_split_plan: Optional[InplaceSplitPlan] = None
        inplace_attention_backend: Optional[str] = None
        split_planner_decision = "no_split_non_uniform"
        split_planner_payload: dict[str, Any] = {}
        custom_split_sizes = None
        # Split batch - compute split slices for large decode batches
        # Split batch and DBO never conflict:
        # - DBO is for overlapping compute/communication
        # - Split batch is for splitting large uniform decode batches
        mixed_request_split_attempted = bool(
            split_cfg is not None
            and getattr(split_cfg, "enable_mixed_request_split", False)
            and split_mode in _INPLACE_SPLIT_MODES
            and not dual_stream_attention_enabled
            and not uniform_decode
            and with_prefill)
        if mixed_request_split_attempted:
            mixed_request_execution_mode = getattr(
                split_cfg, "mixed_request_split_execution_mode", "dry_run")
            replay_policy = getattr(split_cfg, "inplace_parallel_replay_policy",
                                    "full_graph_parallel")
            reason = _mixed_request_split_precheck_reason(
                split_enabled=split_enabled,
                mixed_enabled=bool(
                    getattr(split_cfg, "enable_mixed_request_split", False)),
                split_mode=split_mode,
                enable_parallel_streams=split_enable_parallel_streams,
                replay_policy=replay_policy,
                use_aclgraph=self.use_aclgraph,
                num_splits=(int(getattr(split_cfg, "num_splits", 0))
                            if split_cfg is not None else 0),
                uniform_decode=uniform_decode,
                with_prefill=with_prefill,
                enable_dbo=ubatch_slices is not None,
                dual_stream_attention_enabled=dual_stream_attention_enabled,
                has_spec_decode_tokens=bool(
                    scheduler_output.scheduled_spec_decode_tokens),
                has_lora=bool(
                    self.lora_config and len(
                        self.input_batch.lora_id_to_lora_request) > 0),
                uses_mrope=bool(self.uses_mrope),
                use_mla=bool(self.model_config.use_mla),
                pcp_size=int(self.pcp_size),
                dcp_size=int(self.dcp_size),
                cudagraph_mode=self.compilation_config.cudagraph_mode,
            )
            if reason is None:
                if macro_graph_enabled:
                    scheduled_tokens_np = np.asarray(
                        num_scheduled_tokens, dtype=np.int32)
                    has_decode_tokens = bool(
                        np.any(scheduled_tokens_np <= int(
                            self.decode_threshold)))
                    has_prefill_tokens = bool(
                        np.any(scheduled_tokens_np > int(
                            self.decode_threshold)))
                    if not (has_decode_tokens and has_prefill_tokens):
                        reason = "no_split_mixed_macro_requires_decode_prefill"
                if reason is None:
                    mixed_graph_bucket_plans = None
                    mixed_request_planner_debug: Optional[dict[str, object]] = (
                        {} if split_debug_enabled else None)
                    mixed_min_total_tokens = getattr(
                        split_cfg, "mixed_request_min_total_tokens", 128)
                    mixed_min_tokens_per_split = getattr(
                        split_cfg, "mixed_request_min_tokens_per_split", 64)
                    mixed_max_single_request_ratio = getattr(
                        split_cfg, "mixed_request_max_single_request_ratio",
                        0.70)
                    mixed_min_prefill_reqs_for_prefill_split = getattr(
                        split_cfg,
                        "mixed_request_min_prefill_reqs_for_prefill_split", 2)
                    mixed_bucket_padding_ratio_grace_tokens = 0
                    mixed_bucket_min_actual_tokens_per_split = 1
                    mixed_max_padding_ratio_per_split = getattr(
                        split_cfg,
                        "mixed_request_max_padding_ratio_per_split",
                        0.0)
                    if (macro_graph_enabled
                            and bool(
                                getattr(macro_graph_cfg,
                                        "allow_bucket_match", False))):
                        mixed_graph_bucket_plans = []
                        for capture_plan in getattr(macro_graph_cfg,
                                                    "capture_plans", []):
                            split_graph_tokens = tuple(
                                int(v) for v in getattr(
                                    capture_plan, "split_graph_tokens", ()))
                            split_req_caps = _macro_capture_plan_req_caps(
                                capture_plan,
                                uniform_decode_query_len=(
                                    self.uniform_decode_query_len))
                            if (len(split_graph_tokens) != 2
                                    or len(split_req_caps) != 2
                                    or any(cap <= 0
                                           for cap in split_req_caps)):
                                continue
                            mixed_graph_bucket_plans.append(
                                (split_graph_tokens[0], split_graph_tokens[1],
                                 int(split_req_caps[0]),
                                 int(split_req_caps[1])))
                        mixed_max_padding_ratio_per_split = getattr(
                            macro_graph_cfg, "max_padding_ratio_per_split",
                            mixed_max_padding_ratio_per_split)
                        if bool(
                                getattr(macro_graph_cfg,
                                        "relax_mixed_request_gates", False)):
                            mixed_min_total_tokens = min(
                                int(mixed_min_total_tokens),
                                int(
                                    getattr(macro_graph_cfg,
                                            "bucket_min_total_tokens", 2)))
                            mixed_min_tokens_per_split = min(
                                int(mixed_min_tokens_per_split),
                                int(
                                    getattr(macro_graph_cfg,
                                            "bucket_min_tokens_per_split", 1)))
                            mixed_max_single_request_ratio = max(
                                float(mixed_max_single_request_ratio),
                                float(
                                    getattr(
                                        macro_graph_cfg,
                                        "bucket_max_single_request_ratio", 1.0)))
                            mixed_min_prefill_reqs_for_prefill_split = min(
                                int(mixed_min_prefill_reqs_for_prefill_split),
                                int(
                                    getattr(
                                        macro_graph_cfg,
                                        "bucket_min_prefill_reqs_for_prefill_split",
                                        1)))
                            mixed_bucket_padding_ratio_grace_tokens = int(
                                getattr(
                                    macro_graph_cfg,
                                    "bucket_padding_ratio_grace_tokens", 0))
                            mixed_bucket_min_actual_tokens_per_split = int(
                                getattr(
                                    macro_graph_cfg,
                                    "bucket_min_actual_tokens_per_split", 1))
                        if not bool(
                                getattr(macro_graph_cfg,
                                        "allow_padded_replay", False)):
                            mixed_max_padding_ratio_per_split = 0.0
                            mixed_bucket_padding_ratio_grace_tokens = 0
                            mixed_bucket_min_actual_tokens_per_split = 1
                    inplace_split_plan, reason = (
                        create_mixed_request_split_batch_slices(
                            num_scheduled_tokens,
                            total_num_scheduled_tokens,
                            self.compilation_config.cudagraph_capture_sizes
                            or [],
                            decode_threshold=int(self.decode_threshold),
                            min_total_tokens=mixed_min_total_tokens,
                            min_tokens_per_split=mixed_min_tokens_per_split,
                            max_single_request_ratio=
                            mixed_max_single_request_ratio,
                            max_padding_tokens_per_split=getattr(
                                split_cfg,
                                "mixed_request_max_padding_tokens_per_split",
                                None),
                            max_padding_ratio_per_split=
                            mixed_max_padding_ratio_per_split,
                            min_prefill_reqs_for_prefill_split=
                            mixed_min_prefill_reqs_for_prefill_split,
                            decode_weight=getattr(
                                split_cfg, "mixed_request_decode_weight", 1),
                            prefill_weight=getattr(
                                split_cfg, "mixed_request_prefill_weight", 4),
                            split_policy=getattr(
                                split_cfg, "mixed_request_split_policy",
                                "balanced_attention"),
                            graph_bucket_plans=mixed_graph_bucket_plans,
                            bucket_padding_ratio_grace_tokens=
                            mixed_bucket_padding_ratio_grace_tokens,
                            bucket_min_actual_tokens_per_split=
                            mixed_bucket_min_actual_tokens_per_split,
                            debug_info=mixed_request_planner_debug,
                        ))
                    if mixed_request_planner_debug:
                        split_planner_payload.update(
                            mixed_request_planner_debug)
            split_planner_decision = "no_split"
            split_planner_payload.update({
                "mode": split_mode,
                "reason": reason,
                "dry_run": True,
                "fallback_to": "no_split",
                "mixed_request_split": True,
                "mixed_request_split_execution_mode":
                mixed_request_execution_mode,
            })
            if inplace_split_plan is not None:
                split_planner_decision = MIXED_REQUEST_SPLIT_DRY_RUN
                split_planner_payload.update(
                    inplace_split_plan.debug_payload())
                split_planner_payload["target_batch_descriptors"] = [{
                    "idx": idx,
                    "actual_num_tokens": split_slice.num_tokens,
                    "graph_num_tokens": split_slice.graph_num_tokens,
                    "padding_tokens": (split_slice.graph_num_tokens -
                                       split_slice.num_tokens),
                    "num_reqs": split_slice.num_requests,
                    "request_capacity": getattr(split_slice, "request_capacity",
                                                split_slice.num_requests),
                    "uniform": False,
                    "has_lora": False,
                    "start_num_tokens": split_slice.start_num_tokens,
                } for idx, split_slice in enumerate(
                    inplace_split_plan.split_slices)]
                if (mixed_request_execution_mode == "serial"
                        and split_mode == "inplace_serial"):
                    (split_batch_slices, split_ubatch_slices) = (
                        _inplace_plan_to_execution_slices(
                            split_mode,
                            inplace_split_plan,
                            enable_parallel_streams=
                            split_enable_parallel_streams))
                    if split_batch_slices is not None:
                        inplace_attention_backend = "mixed_request"
                        split_planner_decision = (
                            "mixed_request_split_execute_serial")
                        split_planner_payload["dry_run"] = False
                        split_planner_payload["fallback_to"] = None
                        split_planner_payload[
                            "inplace_attention_backend"] = (
                                inplace_attention_backend)
                elif (mixed_request_execution_mode ==
                      "piecewise_attention_parallel"
                      and split_mode == "inplace_parallel"):
                    (split_batch_slices, split_ubatch_slices) = (
                        _inplace_plan_to_execution_slices(
                            split_mode,
                            inplace_split_plan,
                            enable_parallel_streams=
                            split_enable_parallel_streams))
                    if split_batch_slices is not None:
                        inplace_attention_backend = "mixed_request"
                        split_planner_decision = (
                            "mixed_request_split_execute_"
                            "piecewise_attention_parallel")
                        split_planner_payload["dry_run"] = False
                        split_planner_payload["fallback_to"] = None
                        split_planner_payload[
                            "inplace_attention_backend"] = (
                                inplace_attention_backend)
        elif (split_mode in _INPLACE_SPLIT_MODES
              and not dual_stream_attention_enabled):
            #检查不可split的原因
            reason = _inplace_split_precheck_reason(
                split_enabled=split_enabled,
                split_mode=split_mode,
                enable_parallel_streams=split_enable_parallel_streams,
                use_aclgraph=self.use_aclgraph,
                num_splits=(int(getattr(split_cfg, "num_splits", 0))
                            if split_cfg is not None else 0),
                uniform_decode=uniform_decode,
                enable_dbo=ubatch_slices is not None,
                with_prefill=with_prefill,
                attn_state=attn_state,
                has_spec_decode_tokens=bool(
                    scheduler_output.scheduled_spec_decode_tokens),
                enable_spec_decode=bool(
                    split_cfg is not None and getattr(
                        split_cfg, "enable_inplace_spec_decode", False)),
                has_lora=bool(
                    self.lora_config and len(
                        self.input_batch.lora_id_to_lora_request) > 0),
                uses_mrope=bool(self.uses_mrope),
                enable_mrope=bool(
                    split_cfg is not None and getattr(
                        split_cfg, "enable_inplace_mrope", False)),
                use_mla=bool(self.model_config.use_mla),
                pcp_size=int(self.pcp_size),
                dcp_size=int(self.dcp_size),
            )
            split_planner_decision = (
                "no_split_dbo_active"
                if reason == "no_split_dbo_active" else "no_split")
            if reason is None:
                if int(total_num_scheduled_tokens) < 64:
                    reason = "no_split_inplace_below_64"

                if reason is None:
                    if macro_graph_enabled:
                        inplace_split_plan, reason = (
                            create_macro_inplace_split_batch_slices(
                                num_scheduled_tokens,
                                total_num_scheduled_tokens,
                                self.uniform_decode_query_len,
                                macro_graph_cfg,
                            ))
                    else:
                        inplace_split_plan, reason = (
                            create_inplace_split_batch_slices(
                                num_scheduled_tokens,
                                total_num_scheduled_tokens,
                                self.uniform_decode_query_len,
                                self.compilation_config.
                                cudagraph_capture_sizes or [],
                                getattr(split_cfg,
                                        "inplace_max_remainder_tokens", None),
                                offset_match_policy=getattr(
                                    split_cfg, "inplace_offset_match_policy",
                                    "exact"),
                                offset_capture_sizes=inplace_offset_capture_sizes,
                                offset_min_graph_tokens=getattr(
                                    split_cfg,
                                    "inplace_offset_min_graph_tokens", 1),
                                offset_max_padding_tokens=getattr(
                                    split_cfg,
                                    "inplace_offset_max_padding_tokens", None),
                                offset_max_padding_ratio=getattr(
                                    split_cfg,
                                    "inplace_offset_max_padding_ratio", None),
                                offset_max_graph_tokens_by_start=getattr(
                                    split_cfg,
                                    "inplace_offset_max_graph_tokens_by_start",
                                    None),
                                offset_allowed_graph_tokens_by_start=getattr(
                                    split_cfg,
                                    "inplace_offset_allowed_graph_tokens_by_start",
                                    None),
                                first_tokens_policy=getattr(
                                    split_cfg, "inplace_split_planner_policy",
                                    getattr(
                                        split_cfg,
                                        "inplace_split_first_tokens_policy",
                                        "largest_lower")),
                            ))
                    if (inplace_split_plan is not None
                            and getattr(inplace_split_plan,
                                        "offset_match_policy", "") != "compact"
                            and not inplace_split_first_graph_matches_attention_backend(
                                inplace_split_plan,
                                lambda shape: using_paged_attention(
                                    shape, self.vllm_config),
                            )):
                        reason = NO_SPLIT_ATTENTION_BACKEND_MISMATCH
                        inplace_split_plan = None

            split_planner_payload.update({
                "mode": split_mode,
                "reason": reason,
                "dry_run": True,
                "fallback_to": "no_split",
            })
            if inplace_split_plan is not None:
                if (macro_graph_enabled
                        and getattr(inplace_split_plan,
                                    "offset_match_policy", "") == "compact"):
                    inplace_attention_backend = "mixed_request"
                else:
                    inplace_attention_backend = select_inplace_attention_backend(
                        inplace_split_plan,
                        lambda shape: using_paged_attention(
                            shape, self.vllm_config),
                    )
                split_planner_decision = INPLACE_SPLIT_DRY_RUN
                split_planner_payload.update(
                    inplace_split_plan.debug_payload())
                split_planner_payload[
                    "inplace_attention_backend"] = inplace_attention_backend
                if macro_graph_enabled:
                    split_planner_payload["macro_graph"] = {
                        "enabled": True,
                        "schedule": getattr(macro_graph_cfg, "schedule",
                                            None),
                        "plan_source": getattr(macro_graph_cfg,
                                               "plan_source", None),
                        "miss_policy": getattr(macro_graph_cfg,
                                               "miss_policy", None),
                    }
                split_planner_payload["target_batch_descriptors"] = [{
                    "idx": idx,
                    "actual_num_tokens": split_slice.num_tokens,
                    "graph_num_tokens": split_slice.graph_num_tokens,
                    "padding_tokens": (split_slice.graph_num_tokens -
                                       split_slice.num_tokens),
                    "num_reqs": split_slice.num_requests,
                    "uniform": True,
                    "has_lora": False,
                    "start_num_tokens": split_slice.start_num_tokens,
                } for idx, split_slice in enumerate(
                    inplace_split_plan.split_slices)]
                (split_batch_slices,
                 split_ubatch_slices) = _inplace_plan_to_execution_slices(
                     split_mode,
                     inplace_split_plan,
                     enable_parallel_streams=split_enable_parallel_streams)
                if split_batch_slices is not None:
                    split_planner_decision = "inplace_split_execute"
                    split_planner_payload["dry_run"] = False
                    split_planner_payload["fallback_to"] = None
        elif (not dual_stream_attention_enabled and uniform_decode
              and ubatch_slices is None
              and split_mode == "parallel_buffer"):  # Only split if DBO is not active
            split_planner_decision = "no_split_not_attempted"
            cudagraph_capture_sizes = set(
                self.compilation_config.cudagraph_capture_sizes or []
            ) if self.use_aclgraph else None
            _should_split = True  # whether to call split_batch_split at all
            if cudagraph_capture_sizes and self.use_aclgraph:
                # Find the largest main-stream graph size that fits within
                # num_reqs without padding.  The remainder goes to the parallel
                # stream (which may be padded to its own nearest graph size).
                sorted_main_sizes = sorted(cudagraph_capture_sizes)
                max_main_size = sorted_main_sizes[-1]

                # Helper: ceil num to the nearest graph size in a sorted list.
                def _ceil_to_graph(n: int, sizes: list[int]) -> int:
                    for s in sizes:
                        if s >= n:
                            return s
                    return sizes[-1]  # clamp to max if n exceeds all sizes

                main_reqs = max(
                    (s for s in sorted_main_sizes if s <= num_reqs),
                    default=0,
                )
                parallel_reqs = num_reqs - main_reqs

                # force_split: skip threshold check and always split as
                # (main_reqs + parallel_reqs) whenever there is a non-trivial
                # remainder and main_reqs hits a captured graph.
                _split_cfg = getattr(
                    self.ascend_config, "split_batch_config", None)
                force_split = bool(
                    _split_cfg is not None
                    and getattr(_split_cfg, "force_split", False)
                )
                split_planner_payload.update({
                    "main_reqs": int(main_reqs),
                    "parallel_reqs": int(parallel_reqs),
                    "force_split": bool(force_split),
                })

                if main_reqs == num_reqs:
                    if force_split and len(sorted_main_sizes) >= 2:
                        # Force split even when num_reqs exactly hits a graph.
                        # Use the second-largest graph size <= num_reqs as the
                        # main slice so there is a non-zero parallel remainder.
                        candidates = [s for s in sorted_main_sizes
                                      if s < num_reqs]
                        if candidates:
                            main_reqs = max(candidates)
                            parallel_reqs = num_reqs - main_reqs
                            custom_split_sizes = [main_reqs, parallel_reqs]
                            split_planner_decision = "split"
                        else:
                            # num_reqs == smallest graph size; cannot split.
                            _should_split = False
                            split_planner_decision = "no_split_exact_graph_hit"
                    else:
                        # num_reqs exactly hits a main-stream graph: no split.
                        # Let split_ubatch_slices stay None so the forward pass
                        # goes through _generate_process_reqs_hidden_states.
                        _should_split = False
                        split_planner_decision = "no_split_exact_graph_hit"
                elif main_reqs > 0 and parallel_reqs > 0:
                    if force_split:
                        # Force split: always use (main_reqs + parallel_reqs)
                        # regardless of padding savings or whether num_reqs
                        # exceeds the largest captured graph size.
                        custom_split_sizes = [main_reqs, parallel_reqs]
                        split_planner_decision = "split"
                    elif num_reqs > max_main_size:
                        # num_reqs exceeds all captured sizes; no graph to pad
                        # to, so no split benefit under normal threshold logic.
                        _should_split = False
                        split_planner_decision = "no_split_above_max_capture_size"
                    else:
                        # Evaluate whether splitting saves enough padding.
                        #
                        # padding_saved = padding wasted without split
                        #               - padding the parallel slice still needs
                        #
                        # Without split: num_reqs pads up to the next graph.
                        # With split:    main_reqs hits a graph (0 padding);
                        #                parallel_reqs pads to its nearest graph.
                        threshold = getattr(
                            self.compilation_config,
                            "cudagraph_split_pad_threshold",
                            0,
                        )
                        # Parallel-stream capture sizes (may differ from main).
                        parallel_sizes = sorted(
                            getattr(
                                self, "cudagraph_batch_sizes_parallel", None)
                            or sorted_main_sizes
                        )
                        original_padded = _ceil_to_graph(
                            num_reqs, sorted_main_sizes)
                        original_padding = original_padded - num_reqs
                        remainder_padded = _ceil_to_graph(
                            parallel_reqs, parallel_sizes)
                        remainder_padding = remainder_padded - parallel_reqs
                        padding_saved = original_padding - remainder_padding
                        split_planner_payload.update({
                            "original_padded": int(original_padded),
                            "original_padding": int(original_padding),
                            "remainder_padded": int(remainder_padded),
                            "remainder_padding": int(remainder_padding),
                            "padding_saved": int(padding_saved),
                            "threshold": int(threshold),
                        })
                        if padding_saved > threshold:
                            custom_split_sizes = [main_reqs, parallel_reqs]
                            split_planner_decision = "split"
                        else:
                            # Not worth splitting; let the batch pad normally.
                            _should_split = False
                            split_planner_decision = (
                                "no_split_padding_saving_too_small")
                else:
                    # main_reqs == 0: num_reqs is smaller than all captured
                    # graph sizes; fall through to split_batch_split's own logic.
                    pass
            if _should_split:
                split_batch_slices, _ = split_batch_split(
                    num_scheduled_tokens,
                    num_tokens_unpadded,
                    num_tokens_padded,
                    vllm_config=self.vllm_config,
                    cudagraph_capture_sizes=cudagraph_capture_sizes,
                    custom_split_sizes=custom_split_sizes,
                )
                if split_batch_slices:
                    split_ubatch_slices = [
                        UBatchSlice(s.request_slice, s.token_slice)
                        for s in split_batch_slices
                    ]
                    split_planner_decision = "split"
                elif split_planner_decision == "no_split_not_attempted":
                    split_planner_decision = "no_split_invalid_custom_split"
        elif dual_stream_attention_enabled:
            split_planner_decision = "no_split_dual_stream_attention"
            split_planner_payload.update({
                "mode": split_mode,
                "reason": "dual_stream_attention_enabled",
                "dry_run": True,
                "fallback_to": None,
            })
        elif uniform_decode and ubatch_slices is None:
            split_planner_decision = "no_split_not_inplace_mode"
            split_planner_payload.update({
                "mode": split_mode,
                "reason": "no_split_not_inplace_mode",
            })
        elif uniform_decode and ubatch_slices is not None:
            split_planner_decision = "no_split_dbo_active"
            if split_mode in ("inplace_serial", "inplace_parallel"):
                split_planner_payload.update({
                    "mode": split_mode,
                    "reason": "no_split_dbo_active",
                    "dry_run": True,
                    "fallback_to": "no_split",
                })
        elif split_mode in ("inplace_serial", "inplace_parallel"):
            split_planner_decision = "no_split"
            split_planner_payload.update({
                "mode": split_mode,
                "reason": "no_split_non_uniform_decode",
                "dry_run": True,
                "fallback_to": "no_split",
            })

        if split_debug_enabled:
            if "main_reqs" in locals() and "parallel_reqs" in locals():
                split_planner_payload.update({
                    "main_reqs": int(main_reqs),
                    "parallel_reqs": int(parallel_reqs),
                })
            split_debug.log_event(
                "split_planner_decision",
                {
                    "decision": split_planner_decision,
                    "custom_split_sizes": custom_split_sizes
                    if "custom_split_sizes" in locals() else None,
                    **split_planner_payload,
                },
                step_id=self._split_inplace_debug_step_id,
            )
            if split_batch_slices is not None:
                split_debug.log_event(
                    "split_slices",
                    {
                        "num_splits": len(split_batch_slices),
                        "is_inplace": split_mode in (
                            "inplace_serial", "inplace_parallel"),
                        "execution_mode": split_mode,
                        "dry_run": False,
                        "splits":
                        split_debug.split_slices_info(split_batch_slices),
                    },
                    step_id=self._split_inplace_debug_step_id,
                )
            elif inplace_split_plan is not None:
                split_debug.log_event(
                    "split_slices",
                    {
                        "num_splits":
                        len(inplace_split_plan.split_slices),
                        "is_inplace":
                        True,
                        "execution_mode":
                        split_mode,
                        "dry_run":
                        True,
                        "splits":
                        split_debug.split_slices_info(
                            inplace_split_plan.split_slices),
                    },
                    step_id=self._split_inplace_debug_step_id,
                )


        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        if (split_mode in ("inplace_serial", "inplace_parallel")
                and split_batch_slices is not None):
            num_input_tokens = total_num_scheduled_tokens
        else:
            num_input_tokens = maybe_padded_num_tokens

        dual_stream_attention_plan = self._select_dual_stream_attention_plan(
            total_num_scheduled_tokens=int(total_num_scheduled_tokens),
            graph_num_tokens=int(num_input_tokens),
            uniform_decode=bool(uniform_decode),
            with_prefill=bool(with_prefill),
            ubatch_slices=ubatch_slices,
            has_spec_decode_tokens=bool(
                scheduler_output.scheduled_spec_decode_tokens),
            has_lora=bool(self.lora_config and len(
                self.input_batch.lora_id_to_lora_request) > 0),
            attn_state=attn_state,
        )
        dual_stream_attention_slices = None
        dual_stream_attention_ubatch_slices = None
        if dual_stream_attention_plan is not None:
            (dual_stream_attention_slices,
             dual_stream_attention_ubatch_slices) = (
                 _dual_stream_attention_plan_to_slices(
                     dual_stream_attention_plan,
                     self.uniform_decode_query_len))
            self._dual_stream_attention_plan = dual_stream_attention_plan
            self._dual_stream_attention_slices = dual_stream_attention_slices
            if split_debug_enabled:
                split_debug.log_event(
                    "dual_stream_attention_plan",
                    {
                        "total_tokens":
                        int(dual_stream_attention_plan.total_tokens),
                        "graph_tokens":
                        int(dual_stream_attention_plan.graph_tokens),
                        "split_actual_tokens":
                        list(dual_stream_attention_plan.split_actual_tokens),
                        "split_graph_tokens":
                        list(dual_stream_attention_plan.split_graph_tokens),
                        "split_start_tokens":
                        list(dual_stream_attention_plan.split_start_tokens),
                        "actual_q_policy":
                        _dual_stream_attention_actual_q_policy(split_cfg),
                    },
                    step_id=self._split_inplace_debug_step_id,
                )

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        if self.pcp_size > 1:
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
                   position_pcp[:total_num_scheduled_tokens],
                   out=positions_np)
        else:
            self.positions.np[:total_num_scheduled_tokens] = positions_np

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or
                                                   self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens.copy_to_gpu()

        self.seq_lens.gpu[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens,
                                cu_num_tokens)
        self.positions.cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions.copy_to_gpu()

        # OPTIMIZATION: If split batch is enabled, directly write the second split's
        # data to parallel_streams buffers during preparation, avoiding the copy
        # overhead in _make_split_batch_metadata_parallel_streams.
        # NOTE: inputs_embeds is handled separately in _make_split_batch_metadata_parallel_streams
        # because it's populated after _prepare_inputs returns.
        if split_batch_slices is not None and len(split_batch_slices) > 1:
            if split_mode == "parallel_buffer":
                second_split = split_batch_slices[1]
                second_token_start = second_split.token_slice.start
                second_token_end = second_split.token_slice.stop
                second_num_tokens = second_token_end - second_token_start
                second_padded_tokens = second_split.padded_num_tokens

                # Copy positions for second split to parallel_streams buffer,
                # then zero-pad the tail up to padded_num_tokens.
                if self.positions.gpu.ndim == 2:
                    # M-RoPE case
                    self.positions_parallel_streams.gpu[:, :second_num_tokens].copy_(
                        self.positions.gpu[:, second_token_start:second_token_end])
                    if second_padded_tokens > second_num_tokens:
                        self.positions_parallel_streams.gpu[
                            :, second_num_tokens:second_padded_tokens].fill_(0)
                else:
                    self.positions_parallel_streams.gpu[:second_num_tokens].copy_(
                        self.positions.gpu[second_token_start:second_token_end])
                    if second_padded_tokens > second_num_tokens:
                        self.positions_parallel_streams.gpu[
                            second_num_tokens:second_padded_tokens].fill_(0)

                # Copy input_ids for second split to parallel_streams buffer,
                # then zero-pad the tail up to padded_num_tokens.
                self.input_ids_parallel_streams.gpu[:second_num_tokens].copy_(
                    self.input_ids.gpu[second_token_start:second_token_end])
                if second_padded_tokens > second_num_tokens:
                    self.input_ids_parallel_streams.gpu[
                        second_num_tokens:second_padded_tokens].fill_(0)

                # Zero-pad main stream (split[0]) input_ids and positions beyond
                # actual token count up to its padded capture size.
                first_split = split_batch_slices[0]
                first_num_tokens = first_split.num_tokens
                first_padded_tokens = first_split.padded_num_tokens
                if first_padded_tokens > first_num_tokens:
                    self.input_ids.gpu[first_num_tokens:first_padded_tokens].fill_(0)
                    if self.positions.gpu.ndim == 2:
                        self.positions.gpu[:, first_num_tokens:first_padded_tokens].fill_(0)
                    else:
                        self.positions.gpu[first_num_tokens:first_padded_tokens].fill_(0)

                if split_debug_enabled:
                    first_token_start = first_split.token_slice.start
                    first_token_end = first_split.token_slice.stop
                    split_debug.log_event(
                        "split_input_buffers",
                        {
                            "idx": 0,
                            "path": "original_buffer",
                            "input_ids": split_debug.tensor_info(
                                self.input_ids.gpu[:first_padded_tokens]),
                            "positions": split_debug.tensor_info(
                                self.positions.gpu[:, :first_padded_tokens]
                                if self.positions.gpu.ndim == 2 else
                                self.positions.gpu[:first_padded_tokens]),
                            "inputs_embeds": None,
                            "source_token_start": int(first_token_start),
                            "source_token_stop": int(first_token_end),
                            "num_tokens": int(first_num_tokens),
                            "padded_num_tokens": int(first_padded_tokens),
                        },
                        step_id=self._split_inplace_debug_step_id,
                    )
                    split_debug.log_event(
                        "split_input_buffers",
                        {
                            "idx": 1,
                            "path": "parallel_buffer",
                            "input_ids": split_debug.tensor_info(
                                self.input_ids_parallel_streams.
                                gpu[:second_padded_tokens]),
                            "positions": split_debug.tensor_info(
                                self.positions_parallel_streams.
                                gpu[:, :second_padded_tokens]
                                if self.positions_parallel_streams.gpu.ndim == 2
                                else self.positions_parallel_streams.
                                gpu[:second_padded_tokens]),
                            "inputs_embeds": None,
                            "source_token_start": int(second_token_start),
                            "source_token_stop": int(second_token_end),
                            "num_tokens": int(second_num_tokens),
                            "padded_num_tokens": int(second_padded_tokens),
                        },
                        step_id=self._split_inplace_debug_step_id,
                    )
            elif split_mode in ("inplace_serial", "inplace_parallel"):
                assert inplace_split_plan is not None

        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_mask = self._make_attention_mask(attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp

        attn_metadata: PerLayerAttnMetadata = {}
        dual_stream_attention_metadata: Optional[list[PerLayerAttnMetadata]] = (
            [dict() for _ in range(2)]
            if dual_stream_attention_ubatch_slices is not None else None)
        split_ubatch_slices_for_metadata: Optional[UBatchSlices] = None
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]
            split_ubatch_slices_for_metadata = ubatch_slices
        elif split_ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(split_ubatch_slices))]
            split_ubatch_slices_for_metadata = split_ubatch_slices

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            original_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                np.array(tokens_original, dtype=np.int32))
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
            ):
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)

                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(
                    scheduler_output)

            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the acl graph all the time. The v0
            # engine avoids this by "double compiling" the acl graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the acl graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:total_num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
        positions = self.positions.gpu[:num_input_tokens]
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]

        # type: ignore
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            # If both flashcomm1 and pp are used simultaneously,
            # the shape of the received data and the shape of the space to be copied to will not match,
            # requiring a recalculation of the incoming data's shape.
            tp_size = get_tensor_model_parallel_world_size()
            num_input_tokens_with_flashcomm1 = num_input_tokens
            if enable_sp():
                num_input_tokens_with_flashcomm1 = (num_input_tokens +
                                                    tp_size - 1) // tp_size
                if split_ubatch_slices_for_metadata is not None:
                    # for dbo, we calculate the size of intermediate tensors
                    # later in ubatch_wrapper
                    num_input_tokens_with_dbo = (
                        (ubatch_slices[0].num_tokens + tp_size - 1) // tp_size
                    ) + (
                        (ubatch_slices[1].num_tokens + tp_size - 1) // tp_size)
                    intermediate_tensor_size = next(
                        iter(self.intermediate_tensors.tensors.values())).size(
                            0)
                    if intermediate_tensor_size < num_input_tokens_with_dbo:
                        self.intermediate_tensors = (
                            self.model.make_empty_intermediate_tensors(
                                batch_size=num_input_tokens_with_dbo,
                                dtype=self.dtype,
                                device=self.device))
                    num_input_tokens_with_flashcomm1 = max(
                        num_input_tokens_with_flashcomm1,
                        num_input_tokens_with_dbo)
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[
                    k][:num_input_tokens_with_flashcomm1].copy_(
                        v[:num_input_tokens_with_flashcomm1],
                        non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k:
                v[:num_input_tokens_with_flashcomm1]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            if self.pcp_size * self.dcp_size > 1:
                logits_indices = torch.from_numpy(
                    cu_num_tokens
                ) * self.pcp_size - self.num_pcp_pads[:num_reqs] - 1
                logits_indices = logits_indices.pin_memory().to(
                    self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1:num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens, self.num_pcp_pads[:num_reqs])
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:
                                            num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        self.spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        if self.speculative_config and self.pcp_size > 1:
            self._generate_pcp_mtp_input(
                num_reqs, scheduler_output.total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens)

        long_seq_metadata = self._generate_pcp_metadata(
            total_num_scheduled_tokens)
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            # NOTE: This is strange, why did we use total_num_scheduled_tokens before?
            slot_mapping_size = (total_num_scheduled_tokens
                                 if self.pcp_size == 1 else
                                 total_num_scheduled_tokens * self.pcp_size -
                                 total_num_pcp_pads)
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                blk_table.slot_mapping.gpu[slot_mapping_size:].fill_(0)
                if self.pcp_size > 1:
                    slot_mapping_for_pcp = blk_table.slot_mapping.gpu[:
                                                                      long_seq_metadata
                                                                      .
                                                                      num_actual_tokens_pcp_padded]
                    slot_mapping_for_pcp[slot_mapping_size:].fill_(-1)
                    assert pcp_unpad_mask is not None
                    pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[:
                                                                           pcp_unpad_mask
                                                                           .
                                                                           shape[
                                                                               0]]
                    pcp_padded_slot_mapping.fill_(-1)
                    pcp_padded_slot_mapping[
                        pcp_unpad_mask] = slot_mapping_for_pcp[:
                                                               slot_mapping_size]
                    slot_mapping_for_pcp[:long_seq_metadata.
                                         num_actual_tokens_pcp_padded] = pcp_padded_slot_mapping
                    blk_table.slot_mapping.gpu[:long_seq_metadata.num_actual_tokens_pcp_padded] = \
                        slot_mapping_for_pcp
                slot_mapping = blk_table.slot_mapping.gpu

            # NOTE: This is a temporary hack, now in GPUModelRunner, this prepare_inputs
            # has been split to multiple parts, and there are 3 parts that is related to this
            # `num_reqs`, we'll take `query_start_loc` as an example:
            # 1. self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            # 2. get `num_reqs_padded`, this depends on dispatcher and which is why we have the
            #    following simplified `dispatch` logic here, we try to minimize the impact
            # 3. query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) \
                and (total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs)

            # TODO: We should make this official ASAP. Also note that if we pad here,
            # the builders won’t need to add any extra padding.
            max_decode_tokens = self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
            if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                uniform_decode and self.uniform_decode_query_len <= num_input_tokens <= max_decode_tokens:
                num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
                pad_size = num_reqs_padded - num_reqs
                if pad_size > 0:
                    last_query_loc = self.query_start_loc.np[num_reqs]

                    self.query_start_loc.np[
                        num_reqs + 1:num_reqs_padded + 1] = self.arange_np[
                            1:pad_size +
                            1] * self.uniform_decode_query_len + last_query_loc
                    self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)

                # So we are trying to simulate the behavior of GPUModelRunner's
                # prepare_inputs for uniform decode mode by padding query_start_loc
                num_reqs = num_reqs_padded

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                seq_lens=self.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=slot_mapping_size,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions.gpu,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                max_query_len=max_num_scheduled_tokens,
                decode_token_per_req=self.decode_token_per_req,
                prefill_context_parallel_metadata=long_seq_metadata,
            )

            if self.speculative_config and self.pcp_size > 1:
                # For pcp + spec decode, we flatten block_table
                # to avoid irregular spec_attn_mask shape, e.g.,
                # num_decode_req=2, num_prefill_req=3, num_speculative_tokens=1,
                # ori block_table: # [d0, d1, p0, p1, p2]
                # (num_reqs_d + num_reqs_p, max_num_blocks),
                # flattened block_table: [d0, d0, d1, d1, p0, p1, p2]
                # (num_reqs_d * decode_threshold + num_reqs_p, max_num_blocks),
                ori_query_lens = self.query_start_loc_pcp_full.cpu[1:num_reqs + 1] - \
                    self.query_start_loc_pcp_full.cpu[:num_reqs]
                num_prefill_reqs = (ori_query_lens
                                    > self.decode_threshold).sum().item()
                num_decode_reqs = num_reqs - num_prefill_reqs
                num_decode_reqs_flatten = num_decode_reqs * self.decode_threshold
                blk_table_tensor[
                    num_decode_reqs_flatten:num_decode_reqs_flatten +
                    num_prefill_reqs].copy_(
                        blk_table_tensor[num_decode_reqs:num_decode_reqs +
                                         num_prefill_reqs].clone())
                blk_table_tensor[:num_decode_reqs_flatten].copy_(
                    blk_table_tensor[:num_decode_reqs].repeat_interleave(
                        self.decode_threshold, dim=0))
                common_attn_metadata.block_table_tensor = \
                    blk_table_tensor[:num_decode_reqs_flatten + num_prefill_reqs]

            if self.speculative_config and \
                self.spec_decode_common_attn_metadata is None:
                self.spec_decode_common_attn_metadata = common_attn_metadata
                if self.speculative_config.method in ("eagle", "eagle3") and \
                        self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
                    self.spec_decode_common_attn_metadata = \
                        self.spec_decode_common_attn_metadata.unpadded(
                            total_num_scheduled_tokens, base_num_reqs)

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    if dual_stream_attention_metadata is not None:
                        raise RuntimeError(
                            "dual_stream_attention_config does not support "
                            "GDN/linear attention layers")
                    if use_spec_decode:
                        patch_torch_npu_argsort()
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.
                            num_decode_draft_tokens.cpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)

                    if split_ubatch_slices_for_metadata is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            split_ubatch_slices_for_metadata, common_attn_metadata,
                            self.max_num_tokens)
                        common_attn_metadata_list = (
                            self._stabilize_inplace_common_attn_metadata_list(
                                common_attn_metadata_list,
                                split_mode=split_mode,
                                inplace_split_plan=inplace_split_plan))
                        _validate_split_attn_metadata_count(
                            "decode_gdn",
                            common_attn_metadata_list,
                            len(split_ubatch_slices_for_metadata),
                        )
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            attn_metadata_i = (attn_group.get_metadata_builder(
                                ubatch_id=ubid).build(
                                    common_prefix_len=common_prefix_len,
                                    common_attn_metadata=common_attn_metadata,
                                ))
                            for layer_name in kv_cache_group_spec.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        attn_metadata_i = builder.build(
                            common_prefix_len=common_prefix_len,
                            common_attn_metadata=common_attn_metadata,
                            **extra_attn_metadata_args)

                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i
                elif self.model_config.runner_type == "pooling":
                    if dual_stream_attention_metadata is not None:
                        raise RuntimeError(
                            "dual_stream_attention_config does not support "
                            "pooling model runner")
                    # TODO: support ubatch here
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                else:
                    if split_ubatch_slices_for_metadata is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            split_ubatch_slices_for_metadata, common_attn_metadata,
                            self.max_num_tokens)
                        common_attn_metadata_list = (
                            self._stabilize_inplace_common_attn_metadata_list(
                                common_attn_metadata_list,
                                split_mode=split_mode,
                                inplace_split_plan=inplace_split_plan))
                        _validate_split_attn_metadata_count(
                            "decode_full",
                            common_attn_metadata_list,
                            len(split_ubatch_slices_for_metadata),
                        )
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            attn_metadata_i = (attn_group.get_metadata_builder(
                                ubatch_id=ubid).build(
                                    common_prefix_len=common_prefix_len,
                                    common_attn_metadata=common_attn_metadata,
                                    model=self.get_model()))
                            for layer_name in kv_cache_group_spec.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        attn_metadata_i = builder.build(
                            common_prefix_len=common_prefix_len,
                            common_attn_metadata=common_attn_metadata,
                            model=self.get_model(),
                            **extra_attn_metadata_args)
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i
                    if dual_stream_attention_metadata is not None:
                        assert dual_stream_attention_ubatch_slices is not None
                        assert dual_stream_attention_slices is not None
                        dual_common_attn_metadata_list = split_attn_metadata(
                            dual_stream_attention_ubatch_slices,
                            common_attn_metadata,
                            self.max_num_tokens)
                        dual_common_attn_metadata_list = (
                            self.
                            _stabilize_dual_stream_common_attn_metadata_list(
                                dual_common_attn_metadata_list,
                                dual_stream_attention_slices))
                        _validate_split_attn_metadata_count(
                            "dual_stream_attention",
                            dual_common_attn_metadata_list,
                            len(dual_stream_attention_ubatch_slices),
                        )
                        for ubid, split_common_attn_metadata in enumerate(
                                dual_common_attn_metadata_list):
                            dual_attn_metadata_i = builder.build(
                                common_prefix_len=common_prefix_len,
                                common_attn_metadata=
                                split_common_attn_metadata,
                                model=self.get_model())
                            self._apply_dual_stream_fia_actual_seq_lengths_q(
                                dual_attn_metadata_i,
                                split_common_attn_metadata)
                            for layer_name in attn_group.layer_names:
                                dual_stream_attention_metadata[ubid][
                                    layer_name] = dual_attn_metadata_i

        # update global cos, sin
        update_cos_sin(positions)

        if dual_stream_attention_plan is not None:
            if dual_stream_attention_metadata is None:
                raise RuntimeError(
                    "dual_stream_attention_config selected a plan but did "
                    "not build split attention metadata")
            self._dual_stream_attention_metadata = (
                dual_stream_attention_metadata)

        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens, ubatch_slices,
                split_batch_slices, num_tokens_after_padding,
                inplace_attention_backend)

    def _generate_process_reqs_hidden_states(self, maybe_padded_num_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds):
        assert self.model is not None
        forward_context = get_forward_context()
        # torch.npu.set_stream_limit(self.stream_main, cube_num=20, vector_num=20)
        self._t_replay_start = time.perf_counter()
        with torch.npu.stream(self.stream_main):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **self._init_model_kwargs(maybe_padded_num_tokens))

        if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not self.use_sparse
                ):
            self._update_attn_params_for_wrapper(forward_context, maybe_padded_num_tokens)


        if get_forward_context().sp_enabled and not get_forward_context(
        ).dbo_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        
        return hidden_states
    
    def _update_attn_params_for_wrapper(self, forward_context, num_tokens):
        """Update attention parameters based on the wrapper type.
        
        Handles different wrapper scenarios:
        - Split-batch: handled in execute_model
        - ACLGraphWrapper: attn_metadata is a dict, update directly
        - AscendUBatchWrapper: attn_metadata is a list, update each ubatch separately
        """
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL \
            and not self.use_sparse:
            # TODO: maybe_padded_num_tokens will be removed, use num_input_tokens instead
            if self.vllm_config.model_config.use_mla:
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   num_tokens)
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_params(self.update_stream, forward_context,
                                           num_tokens,
                                           self.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               num_tokens)
                else:
                    update_attn_params(self.update_stream, forward_context,
                                       num_tokens,
                                       self.vllm_config)

    def _ensure_update_streams(self) -> None:
        if not hasattr(self, "update_stream"):
            self.update_stream = torch.npu.Stream()
        if not hasattr(self, "update_stream_main"):
            self.update_stream_main = torch.npu.Stream()
        if not hasattr(self, "update_stream_parallel"):
            self.update_stream_parallel = torch.npu.Stream()

    def _update_attn_params_for_split_ubatch(self, forward_context,
                                             num_tokens: int,
                                             parallel_streams: bool = False) -> None:
        if (forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or forward_context.capturing or self.use_sparse):
            return
        self._ensure_update_streams()
        update_stream = self.update_stream_parallel if parallel_streams else self.update_stream_main
        if self.vllm_config.model_config.use_mla:
            if self.pcp_size * self.dcp_size > 1:
                update_mla_attn_dcp_pcp_params(update_stream,
                                               forward_context,
                                               num_tokens,
                                               in_parallel_streams=parallel_streams)
            else:
                update_mla_attn_params(update_stream, forward_context,
                                       num_tokens,
                                       self.speculative_config,
                                       in_parallel_streams=parallel_streams)
        else:
            if self.pcp_size * self.dcp_size > 1:
                update_attn_dcp_pcp_params(update_stream,
                                           forward_context,
                                           num_tokens,
                                           in_parallel_streams=parallel_streams)
            else:
                update_attn_params_split(update_stream,
                                         forward_context,
                                         num_tokens,
                                         self.vllm_config,
                                         in_parallel_streams=parallel_streams)

    def _select_dual_stream_attention_plan(
            self,
            *,
            total_num_scheduled_tokens: int,
            graph_num_tokens: int,
            uniform_decode: bool,
            with_prefill: bool,
            ubatch_slices: Optional[UBatchSlices],
            has_spec_decode_tokens: bool,
            has_lora: bool,
            attn_state: Any,
            allow_missing_plan: bool = False) -> Any:
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        cfg = _dual_stream_attention_config(split_cfg)
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if with_prefill or attn_state != AscendAttentionState.DecodeOnly:
            return None

        unsupported_reason = None
        if not uniform_decode:
            unsupported_reason = "non_uniform_decode"
        elif ubatch_slices is not None:
            unsupported_reason = "dbo_active"
        elif has_spec_decode_tokens:
            unsupported_reason = "spec_decode"
        elif has_lora:
            unsupported_reason = "lora"
        elif self.uses_mrope:
            unsupported_reason = "mrope"
        elif self.model_config.use_mla:
            unsupported_reason = "mla"
        elif self.pcp_size * self.dcp_size > 1:
            unsupported_reason = "pcp_dcp"
        elif self.vllm_config.parallel_config.tensor_parallel_size != 1:
            unsupported_reason = "tp_not_1"

        if unsupported_reason is not None:
            if getattr(cfg, "miss_policy", "error") == "error":
                raise RuntimeError(
                    "dual_stream_attention_config enabled but unsupported "
                    f"decode case: {unsupported_reason}")
            return None

        plan = _find_dual_stream_attention_plan(
            cfg, total_num_scheduled_tokens, key="total")
        if plan is None and allow_missing_plan:
            plan = _find_dual_stream_attention_plan(
                cfg, graph_num_tokens, key="graph")
        if plan is None:
            if allow_missing_plan:
                return None
            if getattr(cfg, "miss_policy", "error") == "error":
                raise RuntimeError(
                    "dual_stream_attention_config has no capture plan for "
                    "runtime tokens "
                    f"{int(total_num_scheduled_tokens)} or graph tokens "
                    f"{int(graph_num_tokens)}")
            return None
        if int(plan.graph_tokens) > int(graph_num_tokens):
            raise RuntimeError(
                "dual_stream_attention_config plan graph tokens exceed full "
                "graph tokens: "
                f"plan={int(plan.graph_tokens)}, graph={int(graph_num_tokens)}")
        return plan

    def _stabilize_dual_stream_common_attn_metadata_list(
            self,
            common_attn_metadata_list: list[AscendCommonAttentionMetadata],
            split_batch_slices: SplitBatchSlices
    ) -> list[AscendCommonAttentionMetadata]:
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        actual_q_policy = _dual_stream_attention_actual_q_policy(split_cfg)
        stabilized: list[AscendCommonAttentionMetadata] = []
        for split_idx, common_attn_metadata in enumerate(
                common_attn_metadata_list):
            split_slice = split_batch_slices[split_idx]
            if int(split_slice.graph_num_tokens) > int(split_slice.num_tokens):
                actual_seq_lengths_q_tokens = None
                if actual_q_policy == "actual":
                    actual_seq_lengths_q_tokens = int(split_slice.num_tokens)
                common_attn_metadata = (
                    self._pad_inplace_common_attn_metadata_for_graph(
                        common_attn_metadata,
                        split_slice,
                        split_idx=split_idx,
                        fill_padding=True,
                        actual_seq_lengths_q_tokens=
                        actual_seq_lengths_q_tokens,
                        allow_metadata_copy=True))
            stabilized.append(common_attn_metadata)
        return stabilized

    def _apply_dual_stream_fia_actual_seq_lengths_q(
            self, attn_metadata: Any,
            common_attn_metadata: AscendCommonAttentionMetadata) -> None:
        actual_seq_lengths_q = getattr(
            common_attn_metadata, "_dual_stream_fia_actual_seq_lengths_q",
            None)
        if isinstance(actual_seq_lengths_q, list) and actual_seq_lengths_q:
            setattr(attn_metadata, "actual_seq_lengths_q",
                    list(actual_seq_lengths_q))

    def _pad_tensor_first_dim(self, tensor: Any, pad: int) -> Any:
        if not isinstance(tensor, torch.Tensor) or pad <= 0:
            return tensor
        pad_shape = list(tensor.shape)
        if not pad_shape:
            return tensor
        pad_shape[0] = pad
        return torch.cat([tensor, tensor.new_zeros(tuple(pad_shape))], dim=0)

    def _pad_query_start_loc_for_graph(self, tensor: Any, pad_reqs: int,
                                       query_len: int) -> Any:
        if not isinstance(tensor, torch.Tensor) or pad_reqs <= 0:
            return tensor
        if int(query_len) == 0:
            fill = tensor[-1:].expand(int(pad_reqs))
            return torch.cat([tensor, fill], dim=0)
        increments = torch.arange(
            1,
            pad_reqs + 1,
            dtype=tensor.dtype,
            device=tensor.device,
        ) * int(query_len)
        return torch.cat([tensor, tensor[-1] + increments], dim=0)

    def _pad_query_start_loc_with_chunks_for_graph(
            self,
            tensor: Any,
            pad_tokens: int,
            max_chunk_tokens: int) -> Any:
        if not isinstance(tensor, torch.Tensor) or pad_tokens <= 0:
            return tensor
        pad_tokens = int(pad_tokens)
        max_chunk_tokens = max(1, int(max_chunk_tokens))
        chunks: list[int] = []
        remaining = pad_tokens
        while remaining > 0:
            chunk = min(max_chunk_tokens, remaining)
            chunks.append(chunk)
            remaining -= chunk
        increments = torch.tensor(
            chunks,
            dtype=tensor.dtype,
            device=tensor.device,
        ).cumsum(dim=0)
        return torch.cat([tensor, tensor[-1] + increments], dim=0)

    def _compact_padding_req_chunks(self, pad_tokens: int,
                                    max_query_len: int) -> int:
        pad_tokens = int(pad_tokens)
        if pad_tokens <= 0:
            return 0
        max_query_len = max(1, int(max_query_len))
        return (pad_tokens + max_query_len - 1) // max_query_len

    def _expand_tensor_view_for_graph(
            self,
            tensor: Any,
            graph_size: int,
            *,
            name: str,
            dim: int = 0,
            backing_tensor: Optional[torch.Tensor] = None,
            backing_start: int = 0,
            allow_copy: bool = False) -> Any:
        if not isinstance(tensor, torch.Tensor):
            return tensor

        graph_size = int(graph_size)
        dim = int(dim)
        if graph_size <= int(tensor.shape[dim]):
            if dim == 0:
                return tensor[:graph_size]
            if dim == 1:
                return tensor[:, :graph_size]
            raise ValueError(f"Unsupported dim={dim} for {name}")

        shape = list(tensor.shape)
        shape[dim] = graph_size
        try:
            return tensor.as_strided(
                tuple(shape),
                tensor.stride(),
                storage_offset=tensor.storage_offset(),
            )
        except RuntimeError as exc:
            if (isinstance(backing_tensor, torch.Tensor)
                    and int(backing_tensor.shape[dim])
                    >= int(backing_start) + graph_size):
                stop = int(backing_start) + graph_size
                if dim == 0:
                    return backing_tensor[int(backing_start):stop]
                if dim == 1:
                    return backing_tensor[:, int(backing_start):stop]
                raise ValueError(f"Unsupported dim={dim} for {name}")
            if allow_copy:
                expanded = tensor.new_zeros(tuple(shape))
                copy_slice = [slice(None)] * tensor.ndim
                copy_slice[dim] = slice(0, int(tensor.shape[dim]))
                expanded[tuple(copy_slice)].copy_(tensor)
                return expanded
            backing_shape = (None if backing_tensor is None else
                             tuple(backing_tensor.shape))
            raise RuntimeError(
                "Inplace offset graph requires zero-copy backing view for "
                f"{name}: graph_size={graph_size}, dim={dim}, "
                f"tensor_shape={tuple(tensor.shape)}, "
                f"tensor_storage_offset={int(tensor.storage_offset())}, "
                f"backing_start={int(backing_start)}, "
                f"backing_shape={backing_shape}") from exc

    def _fill_tensor_local_tail(self,
                                tensor: Any,
                                start: int,
                                stop: int,
                                *,
                                name: str,
                                dim: int = 0,
                                value: int = 0) -> bool:
        if not isinstance(tensor, torch.Tensor) or stop <= start:
            return False
        start = int(start)
        stop = int(stop)
        dim = int(dim)
        if stop > int(tensor.shape[dim]):
            raise RuntimeError(
                f"Inplace offset padded local tail for {name} exceeds tensor "
                f"shape: tail_stop={stop}, shape={tuple(tensor.shape)}, "
                f"dim={dim}")
        if dim == 0:
            tensor[start:stop].fill_(value)
        elif dim == 1:
            tensor[:, start:stop].fill_(value)
        else:
            raise ValueError(f"Unsupported dim={dim} for {name}")
        return True

    def _copy_last_row_to_tensor_local_tail(self,
                                            tensor: Any,
                                            start: int,
                                            stop: int,
                                            *,
                                            name: str,
                                            dim: int = 0) -> bool:
        if not isinstance(tensor, torch.Tensor) or stop <= start:
            return False
        start = int(start)
        stop = int(stop)
        dim = int(dim)
        if start <= 0:
            raise RuntimeError(
                f"Cannot pad {name} tail from the last real row when there "
                "are no real rows")
        if stop > int(tensor.shape[dim]):
            raise RuntimeError(
                f"Padded local tail for {name} exceeds tensor shape: "
                f"tail_stop={stop}, shape={tuple(tensor.shape)}, dim={dim}")
        if dim == 0:
            tensor[start:stop].copy_(
                tensor[start - 1:start].expand(stop - start,
                                               *tensor.shape[1:]))
        elif dim == 1:
            tensor[:, start:stop].copy_(
                tensor[:, start - 1:start].expand(tensor.shape[0],
                                                  stop - start,
                                                  *tensor.shape[2:]))
        else:
            raise ValueError(f"Unsupported dim={dim} for {name}")
        return True

    def _fill_compact_fake_slot_mapping_tail(
            self,
            common: AscendCommonAttentionMetadata,
            actual_reqs: int,
            actual_tokens: int,
            graph_tokens: int) -> dict[str, Any]:
        slot_mapping = getattr(common, "slot_mapping", None)
        detail: dict[str, Any] = {
            "policy": _MACRO_GRAPH_COMPACT_FAKE_SLOT_POLICY,
            "applied": False,
            "fallback": None,
            "base_slot": None,
            "available_slots": 0,
            "source_req_idx": None,
        }
        if not isinstance(slot_mapping, torch.Tensor) or graph_tokens <= actual_tokens:
            return detail

        policy = _MACRO_GRAPH_COMPACT_FAKE_SLOT_POLICY
        if policy in ("minus_one", "-1"):
            self._fill_tensor_local_tail(
                slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping",
                value=-1)
            detail["applied"] = True
            return detail
        if policy == "zero":
            self._fill_tensor_local_tail(
                slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping",
                value=0)
            detail["applied"] = True
            return detail
        if policy not in ("scratch", "safe_scratch"):
            detail["fallback"] = "unknown_policy_zero"
            self._fill_tensor_local_tail(
                slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping",
                value=0)
            detail["applied"] = True
            return detail

        block_size = int(
            getattr(getattr(self.vllm_config, "cache_config", None),
                    "block_size", 0) or 0)
        if block_size <= 0:
            block_size = 128
        seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(common, "seq_lens", None)
        block_table = getattr(common, "block_table_tensor", None)
        if (not isinstance(seq_lens_cpu, torch.Tensor)
                or not isinstance(block_table, torch.Tensor)
                or actual_reqs <= 0):
            detail["fallback"] = "missing_seq_lens_or_block_table_zero"
            self._fill_tensor_local_tail(
                slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping",
                value=0)
            detail["applied"] = True
            return detail

        tail_len = int(graph_tokens) - int(actual_tokens)
        try:
            seq_lens_host = seq_lens_cpu[:actual_reqs].detach().cpu()
            block_table_host = block_table[:actual_reqs].detach().cpu()
        except Exception:
            detail["fallback"] = "metadata_cpu_copy_failed_zero"
            self._fill_tensor_local_tail(
                slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping",
                value=0)
            detail["applied"] = True
            return detail

        scratch_base_slot: Optional[int] = None
        scratch_available = 0
        scratch_req_idx: Optional[int] = None
        for req_idx in range(int(actual_reqs) - 1, -1, -1):
            seq_len = int(seq_lens_host[req_idx].item())
            if seq_len <= 0:
                continue
            first_free_offset = seq_len % block_size
            if first_free_offset <= 0:
                continue
            block_col = (seq_len - 1) // block_size
            if block_col < 0 or block_col >= int(block_table_host.shape[1]):
                continue
            block_id = int(block_table_host[req_idx, block_col].item())
            if block_id < 0:
                continue
            scratch_base_slot = block_id * block_size + first_free_offset
            scratch_available = block_size - first_free_offset
            scratch_req_idx = req_idx
            break

        if scratch_base_slot is None or scratch_available <= 0:
            detail["fallback"] = "no_allocated_invisible_slot_minus_one"
            self._fill_tensor_local_tail(
                slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping",
                value=-1)
            detail["applied"] = True
            return detail

        values = (
            torch.arange(
                tail_len,
                dtype=slot_mapping.dtype,
                device=slot_mapping.device) % int(scratch_available)
        ) + int(scratch_base_slot)
        slot_mapping[actual_tokens:graph_tokens].copy_(values)
        detail.update({
            "applied": True,
            "base_slot": int(scratch_base_slot),
            "available_slots": int(scratch_available),
            "source_req_idx": int(scratch_req_idx),
        })
        return detail

    def _metadata_positions_for_graph_slice(
            self, positions: Any,
            split_slice: SplitBatchSlice) -> Any:
        if not isinstance(positions, torch.Tensor):
            return positions

        graph_tokens = int(split_slice.graph_num_tokens)
        token_dim = 1 if positions.ndim == 2 else 0
        token_start = int(split_slice.token_slice.start)
        graph_stop = token_start + graph_tokens
        backing = None
        if token_dim == 1:
            backing = getattr(getattr(self, "mrope_positions", None), "gpu",
                              None)
        if backing is None:
            backing = getattr(getattr(self, "positions", None), "gpu", None)

        return self._expand_tensor_view_for_graph(
            positions,
            graph_tokens,
            name="positions",
            dim=token_dim,
            backing_tensor=backing,
            backing_start=token_start,
        )

    def _pad_inplace_common_attn_metadata_for_graph(
            self,
            common: AscendCommonAttentionMetadata,
            split_slice: SplitBatchSlice,
            *,
            split_idx: int,
            fill_padding: bool = True,
            actual_seq_lengths_q_tokens: Optional[int] = None,
            allow_metadata_copy: bool = False,
    ) -> AscendCommonAttentionMetadata:
        actual_tokens = int(split_slice.num_tokens)
        graph_tokens = int(split_slice.graph_num_tokens)
        if graph_tokens < actual_tokens:
            raise RuntimeError(
                "Inplace offset graph metadata has fewer graph tokens than "
                f"actual tokens: graph_tokens={graph_tokens}, "
                f"actual_tokens={actual_tokens}")

        query_len = int(getattr(common, "decode_token_per_req", 0) or
                        getattr(self, "uniform_decode_query_len", 1) or 1)
        if query_len <= 0:
            raise RuntimeError(
                "Invalid inplace offset metadata query length: "
                f"query_len={query_len}")
        if graph_tokens % query_len != 0 or actual_tokens % query_len != 0:
            raise RuntimeError(
                "Inplace offset graph metadata padding requires request-aligned "
                "token counts: "
                f"actual_tokens={actual_tokens}, graph_tokens={graph_tokens}, "
                f"query_len={query_len}")

        graph_reqs = graph_tokens // query_len
        actual_reqs = int(common.num_reqs)
        pad_reqs = graph_reqs - actual_reqs
        if pad_reqs < 0:
            raise RuntimeError(
                "Inplace offset graph metadata has fewer graph requests than "
                f"actual requests: graph_reqs={graph_reqs}, "
                f"actual_reqs={actual_reqs}")
        pad_tokens = graph_tokens - actual_tokens

        request_start = int(split_slice.request_slice.start)
        actual_request_stop = request_start + actual_reqs
        graph_request_stop = request_start + graph_reqs
        token_start = int(split_slice.token_slice.start)
        actual_token_stop = int(split_slice.token_slice.stop)
        graph_token_stop = token_start + graph_tokens

        padded_common = copy(common)
        padded_common.query_start_loc = self._pad_query_start_loc_for_graph(
            common.query_start_loc, pad_reqs, query_len)
        padded_common.query_start_loc_cpu = (
            self._pad_query_start_loc_for_graph(
                common.query_start_loc_cpu, pad_reqs, query_len))

        padded_common.seq_lens = self._expand_tensor_view_for_graph(
            common.seq_lens,
            graph_reqs,
            name="seq_lens",
            allow_copy=allow_metadata_copy)
        padded_common.seq_lens_cpu = self._expand_tensor_view_for_graph(
            common.seq_lens_cpu,
            graph_reqs,
            name="seq_lens_cpu",
            allow_copy=allow_metadata_copy)
        padded_common.num_computed_tokens_cpu = (
            self._expand_tensor_view_for_graph(
                common.num_computed_tokens_cpu,
                graph_reqs,
                name="num_computed_tokens_cpu",
                allow_copy=allow_metadata_copy))
        padded_common.block_table_tensor = self._expand_tensor_view_for_graph(
            common.block_table_tensor,
            graph_reqs,
            name="block_table_tensor",
            allow_copy=allow_metadata_copy)
        padded_common.slot_mapping = self._expand_tensor_view_for_graph(
            common.slot_mapping,
            graph_tokens,
            name="slot_mapping",
            allow_copy=allow_metadata_copy)
        padded_common.positions = self._metadata_positions_for_graph_slice(
            common.positions, split_slice)

        if fill_padding:
            self._fill_tensor_local_tail(
                padded_common.seq_lens,
                actual_reqs,
                graph_reqs,
                name="seq_lens")
            self._fill_tensor_local_tail(
                padded_common.seq_lens_cpu,
                actual_reqs,
                graph_reqs,
                name="seq_lens_cpu")
            self._fill_tensor_local_tail(
                padded_common.num_computed_tokens_cpu,
                actual_reqs,
                graph_reqs,
                name="num_computed_tokens_cpu")
            self._fill_tensor_local_tail(
                padded_common.block_table_tensor,
                actual_reqs,
                graph_reqs,
                name="block_table_tensor")
            self._fill_tensor_local_tail(
                padded_common.slot_mapping,
                actual_tokens,
                graph_tokens,
                name="slot_mapping")
            positions_dim = 1 if (
                isinstance(padded_common.positions, torch.Tensor)
                and padded_common.positions.ndim == 2) else 0
            self._fill_tensor_local_tail(
                padded_common.positions,
                actual_tokens,
                graph_tokens,
                name="positions",
                dim=positions_dim)

        padded_common.num_reqs = graph_reqs
        padded_common.num_actual_tokens = graph_tokens
        padded_common.num_input_tokens = graph_tokens
        padded_common.max_query_len = max(int(common.max_query_len),
                                          query_len)
        effective_q_tokens = graph_tokens
        if actual_seq_lengths_q_tokens is not None:
            effective_q_tokens = int(actual_seq_lengths_q_tokens)
            if effective_q_tokens < 1 or effective_q_tokens > graph_tokens:
                raise RuntimeError(
                    "Invalid dual-stream FIA actual q tokens: "
                    f"actual_seq_lengths_q_tokens={effective_q_tokens}, "
                    f"graph_tokens={graph_tokens}")
            if effective_q_tokens % query_len != 0:
                raise RuntimeError(
                    "Dual-stream FIA actual q tokens must be request-aligned: "
                    f"actual_seq_lengths_q_tokens={effective_q_tokens}, "
                    f"query_len={query_len}")
        padded_common.actual_seq_lengths_q = list(
            range(query_len, effective_q_tokens + 1, query_len))
        setattr(padded_common, "_dual_stream_fia_actual_seq_lengths_q",
                padded_common.actual_seq_lengths_q)
        padded_common.graph_pad_size = graph_reqs

        if split_debug.is_enabled():
            split_debug.log_event(
                "inplace_offset_metadata_normalized",
                {
                    "split_idx": int(split_idx),
                    "request_start": request_start,
                    "actual_request_stop": actual_request_stop,
                    "graph_request_stop": graph_request_stop,
                    "token_start": token_start,
                    "actual_token_stop": actual_token_stop,
                    "graph_token_stop": graph_token_stop,
                    "actual_tokens": actual_tokens,
                    "graph_tokens": graph_tokens,
                    "actual_reqs": actual_reqs,
                    "graph_reqs": graph_reqs,
                    "pad_tokens": pad_tokens,
                    "pad_reqs": pad_reqs,
                    "effective_q_tokens": effective_q_tokens,
                    "actual_seq_lengths_q_len":
                    len(padded_common.actual_seq_lengths_q),
                    "fill_padding": bool(fill_padding),
                    "query_start_loc_allows_new_buffer": True,
                    "metadata_uses_backing_views": not bool(
                        allow_metadata_copy),
                    **split_debug.metadata_tensor_info(padded_common),
                },
                step_id=_split_debug_step_from_runner(self),
            )
        return padded_common

    def _pad_compact_common_attn_metadata_for_graph(
            self,
            common: AscendCommonAttentionMetadata,
            split_slice: SplitBatchSlice,
            *,
            split_idx: int,
            fill_padding: bool = True,
    ) -> AscendCommonAttentionMetadata:
        actual_tokens = int(split_slice.num_tokens)
        graph_tokens = int(split_slice.graph_num_tokens)
        if graph_tokens < actual_tokens:
            raise RuntimeError(
                "Compact mixed request metadata has fewer graph tokens than "
                f"actual tokens: graph_tokens={graph_tokens}, "
                f"actual_tokens={actual_tokens}")
        actual_reqs = int(common.num_reqs)
        request_capacity = int(
            getattr(split_slice, "request_capacity", 0) or 0)
        if graph_tokens == actual_tokens and request_capacity <= actual_reqs:
            return common

        pad_tokens = graph_tokens - actual_tokens
        max_padding_chunk_tokens = max(
            1,
            int(getattr(split_slice, "max_query_len", 0)
                or common.max_query_len),
        )
        padding_req_chunks = self._compact_padding_req_chunks(
            pad_tokens, max_padding_chunk_tokens)
        effective_reqs = actual_reqs + padding_req_chunks
        graph_reqs = max(effective_reqs, request_capacity)
        if graph_reqs < actual_reqs:
            raise RuntimeError(
                "Compact mixed request metadata has fewer graph requests than "
                f"actual requests: graph_reqs={graph_reqs}, "
                f"actual_reqs={actual_reqs}")
        zero_pad_reqs = graph_reqs - effective_reqs

        padded_common = copy(common)
        # Compact mixed padding is not an original-batch offset. Represent the
        # padded tail as a small number of fake requests, each no larger than
        # the captured max query length, so request capacity does not scale
        # linearly with token padding.
        padded_common.query_start_loc = (
            self._pad_query_start_loc_with_chunks_for_graph(
                common.query_start_loc, pad_tokens,
                max_padding_chunk_tokens))
        padded_common.query_start_loc_cpu = (
            self._pad_query_start_loc_with_chunks_for_graph(
                common.query_start_loc_cpu, pad_tokens,
                max_padding_chunk_tokens))
        if zero_pad_reqs > 0:
            padded_common.query_start_loc = self._pad_query_start_loc_for_graph(
                padded_common.query_start_loc, zero_pad_reqs, 0)
            padded_common.query_start_loc_cpu = (
                self._pad_query_start_loc_for_graph(
                    padded_common.query_start_loc_cpu, zero_pad_reqs, 0))

        padded_common.seq_lens = self._expand_tensor_view_for_graph(
            common.seq_lens,
            graph_reqs,
            name="seq_lens",
            allow_copy=True)
        padded_common.seq_lens_cpu = self._expand_tensor_view_for_graph(
            common.seq_lens_cpu,
            graph_reqs,
            name="seq_lens_cpu",
            allow_copy=True)
        padded_common.num_computed_tokens_cpu = (
            self._expand_tensor_view_for_graph(
                common.num_computed_tokens_cpu,
                graph_reqs,
                name="num_computed_tokens_cpu",
                allow_copy=True))
        padded_common.block_table_tensor = self._expand_tensor_view_for_graph(
            common.block_table_tensor,
            graph_reqs,
            name="block_table_tensor",
            allow_copy=True)
        padded_common.slot_mapping = self._expand_tensor_view_for_graph(
            common.slot_mapping,
            graph_tokens,
            name="slot_mapping",
            allow_copy=True)
        positions_dim = 1 if (
            isinstance(common.positions, torch.Tensor)
            and common.positions.ndim == 2) else 0
        padded_common.positions = self._expand_tensor_view_for_graph(
            common.positions,
            graph_tokens,
            name="positions",
            dim=positions_dim,
            allow_copy=True)

        if fill_padding:
            fake_seq_lens_policy = (
                _MACRO_GRAPH_COMPACT_FAKE_SEQ_LENS_POLICY)
            if fake_seq_lens_policy == "zero":
                self._fill_tensor_local_tail(
                    padded_common.seq_lens,
                    actual_reqs,
                    graph_reqs,
                    name="seq_lens")
                self._fill_tensor_local_tail(
                    padded_common.seq_lens_cpu,
                    actual_reqs,
                    graph_reqs,
                    name="seq_lens_cpu")
                self._fill_tensor_local_tail(
                    padded_common.num_computed_tokens_cpu,
                    actual_reqs,
                    graph_reqs,
                    name="num_computed_tokens_cpu")
            else:
                fake_seq_lens_policy = "last"
                self._copy_last_row_to_tensor_local_tail(
                    padded_common.seq_lens,
                    actual_reqs,
                    graph_reqs,
                    name="seq_lens")
                self._copy_last_row_to_tensor_local_tail(
                    padded_common.seq_lens_cpu,
                    actual_reqs,
                    graph_reqs,
                    name="seq_lens_cpu")
                self._copy_last_row_to_tensor_local_tail(
                    padded_common.num_computed_tokens_cpu,
                    actual_reqs,
                    graph_reqs,
                    name="num_computed_tokens_cpu")
            self._copy_last_row_to_tensor_local_tail(
                padded_common.block_table_tensor,
                actual_reqs,
                graph_reqs,
                name="block_table_tensor")
            slot_mapping_padding_detail = (
                self._fill_compact_fake_slot_mapping_tail(
                    padded_common,
                    actual_reqs,
                    actual_tokens,
                    graph_tokens))
            self._fill_tensor_local_tail(
                padded_common.positions,
                actual_tokens,
                graph_tokens,
                name="positions",
                dim=positions_dim)
        else:
            fake_seq_lens_policy = (
                _MACRO_GRAPH_COMPACT_FAKE_SEQ_LENS_POLICY)
            slot_mapping_padding_detail = {
                "policy": _MACRO_GRAPH_COMPACT_FAKE_SLOT_POLICY,
                "applied": False,
                "fallback": "fill_padding_disabled",
            }

        padded_common.num_reqs = graph_reqs
        padded_common.num_actual_tokens = graph_tokens
        padded_common.num_input_tokens = graph_tokens
        padded_common.max_query_len = max(int(common.max_query_len), 1)
        padded_common.actual_seq_lengths_q = (
            padded_common.query_start_loc_cpu[1:].tolist())
        if isinstance(padded_common.query_start_loc_cpu, torch.Tensor):
            padded_common.actual_seq_lengths_q = [
                int(value) for value in padded_common.actual_seq_lengths_q
            ]
        if (isinstance(padded_common.query_start_loc, torch.Tensor)
                and isinstance(common.query_start_loc, torch.Tensor)
                and padded_common.query_start_loc.dtype !=
                common.query_start_loc.dtype):
            padded_common.query_start_loc = padded_common.query_start_loc.to(
                dtype=common.query_start_loc.dtype)
        if (isinstance(padded_common.query_start_loc_cpu, torch.Tensor)
                and isinstance(common.query_start_loc_cpu, torch.Tensor)
                and padded_common.query_start_loc_cpu.dtype !=
                common.query_start_loc_cpu.dtype):
            padded_common.query_start_loc_cpu = (
                padded_common.query_start_loc_cpu.to(
                    dtype=common.query_start_loc_cpu.dtype))
        padded_common.graph_pad_size = graph_reqs

        if split_debug.is_enabled():
            split_debug.log_event(
                "mixed_request_compact_metadata_padded",
                {
                    "split_idx": int(split_idx),
                    "actual_tokens": actual_tokens,
                    "graph_tokens": graph_tokens,
                    "actual_reqs": actual_reqs,
                    "graph_reqs": graph_reqs,
                    "effective_reqs": effective_reqs,
                    "request_capacity": request_capacity,
                    "padding_req_chunks": padding_req_chunks,
                    "max_padding_chunk_tokens": max_padding_chunk_tokens,
                    "zero_pad_reqs": zero_pad_reqs,
                    "pad_tokens": pad_tokens,
                    "fake_seq_lens_policy": fake_seq_lens_policy,
                    "slot_mapping_padding_detail":
                    slot_mapping_padding_detail,
                    "actual_seq_lengths_q_len":
                    len(padded_common.actual_seq_lengths_q),
                    "fill_padding": bool(fill_padding),
                    "seq_lens_tail": _macro_graph_debug_tensor_tail(
                        padded_common.seq_lens,
                        start=actual_reqs,
                        stop=graph_reqs),
                    "seq_lens_cpu_tail": _macro_graph_debug_tensor_tail(
                        padded_common.seq_lens_cpu,
                        start=actual_reqs,
                        stop=graph_reqs),
                    "actual_seq_lengths_q_tail":
                    _macro_graph_debug_list_tail(
                        padded_common.actual_seq_lengths_q),
                    "block_table_tail": _macro_graph_debug_tensor_tail(
                        padded_common.block_table_tensor,
                        start=actual_reqs,
                        stop=graph_reqs),
                    "slot_mapping_tail": _macro_graph_debug_tensor_tail(
                        padded_common.slot_mapping,
                        start=actual_tokens,
                        stop=graph_tokens),
                    **split_debug.metadata_tensor_info(padded_common),
                },
                step_id=_split_debug_step_from_runner(self),
            )
        return padded_common

    def _stabilize_inplace_common_attn_metadata(
            self,
            common: AscendCommonAttentionMetadata,
            *,
            split_idx: int,
            split_slice: Optional[SplitBatchSlice] = None,
            offset_match_policy: str = "",
            fill_padding: bool = True,
    ) -> AscendCommonAttentionMetadata:
        # Experiment: use split metadata directly, including the rebased
        # query_start_loc tensor created by split_attn_metadata().
        if split_slice is not None and split_slice.start_num_tokens > 0:
            common = self._pad_inplace_common_attn_metadata_for_graph(
                common,
                split_slice,
                split_idx=split_idx,
                fill_padding=fill_padding)
        elif (split_slice is not None and offset_match_policy == "compact"
              and (split_slice.graph_num_tokens > split_slice.num_tokens
                   or int(getattr(split_slice, "request_capacity", 0)
                          or 0) > int(getattr(common, "num_reqs", 0) or 0))):
            common = self._pad_compact_common_attn_metadata_for_graph(
                common,
                split_slice,
                split_idx=split_idx,
                fill_padding=fill_padding)
        return common

    def _stabilize_inplace_common_attn_metadata_list(
            self, common_attn_metadata_list: list[AscendCommonAttentionMetadata],
            *, split_mode: str,
            inplace_split_plan: Optional[InplaceSplitPlan]
    ) -> list[AscendCommonAttentionMetadata]:
        if (split_mode not in ("inplace_serial", "inplace_parallel")
                or inplace_split_plan is None):
            return common_attn_metadata_list
        stabilized: list[AscendCommonAttentionMetadata] = []
        for split_idx, common_attn_metadata in enumerate(
                common_attn_metadata_list):
            split_slice = inplace_split_plan.split_slices[split_idx]
            stabilized.append(
                self._stabilize_inplace_common_attn_metadata(
                    common_attn_metadata,
                    split_idx=split_idx,
                    split_slice=split_slice,
                    offset_match_policy=getattr(inplace_split_plan,
                                                "offset_match_policy", "")))
        return stabilized

    def _intermediate_tensor_slice_for_tokens(self,
                                              tokens_slice: slice) -> slice:
        if enable_sp():
            tp_size = get_tensor_model_parallel_world_size()
            start = (tokens_slice.start + tp_size - 1) // tp_size
            if start != 0:
                stop = start + (tokens_slice.stop - tokens_slice.start +
                                tp_size - 1) // tp_size
            else:
                stop = (tokens_slice.stop + tp_size - 1) // tp_size
            return slice(start, stop)
        return tokens_slice

    def _slice_split_batch_inputs(self, tokens_slice: slice, input_ids,
                                  positions, inputs_embeds,
                                  intermediate_tensors):
        (sliced_input_ids, sliced_positions,
         sliced_inputs_embeds) = slice_model_inputs_by_token(
             input_ids, positions, inputs_embeds, tokens_slice)

        if intermediate_tensors is not None:
            tokens_slice = self._intermediate_tensor_slice_for_tokens(
                tokens_slice)
            sliced_intermediate_tensors = intermediate_tensors[
                tokens_slice] if intermediate_tensors else None
        else:
            sliced_intermediate_tensors = None

        return (sliced_input_ids, sliced_positions, sliced_inputs_embeds,
                sliced_intermediate_tensors)

    def _graph_token_slice_for_split(self,
                                     split_slice: SplitBatchSlice) -> slice:
        start = int(split_slice.token_slice.start)
        return slice(start, start + int(split_slice.graph_num_tokens))

    def _padding_tail_slice_for_split(
            self, split_slice: SplitBatchSlice) -> Optional[slice]:
        actual_stop = int(split_slice.token_slice.stop)
        graph_stop = int(split_slice.token_slice.start) + int(
            split_slice.graph_num_tokens)
        if graph_stop <= actual_stop:
            return None
        return slice(actual_stop, graph_stop)

    def _fill_tensor_token_tail(self, tensor: Optional[torch.Tensor],
                                tail_slice: slice, *, name: str,
                                token_dim: int = 0) -> bool:
        if tensor is None:
            return False
        if tail_slice.stop > int(tensor.shape[token_dim]):
            raise RuntimeError(
                f"Inplace offset padded tail for {name} exceeds tensor "
                f"shape: tail_stop={tail_slice.stop}, "
                f"shape={tuple(tensor.shape)}, token_dim={token_dim}")
        if token_dim == 0:
            tensor[tail_slice].fill_(0)
        elif token_dim == 1:
            tensor[:, tail_slice].fill_(0)
        else:
            raise ValueError(f"Unsupported token_dim={token_dim} for {name}")
        return True

    def _maybe_expand_tensor_for_graph_slice(
            self,
            tensor: Optional[torch.Tensor],
            backing_tensor: Optional[torch.Tensor],
            graph_stop: int,
            *,
            name: str,
            token_dim: int = 0) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if graph_stop <= int(tensor.shape[token_dim]):
            return tensor
        if backing_tensor is None:
            raise RuntimeError(
                "Inplace offset graph requires input backing buffer: "
                f"name={name}, graph_stop={int(graph_stop)}, "
                f"tensor_shape={tuple(tensor.shape)}, token_dim={token_dim}, "
                "backing_shape=None")
        if graph_stop > int(backing_tensor.shape[token_dim]):
            raise RuntimeError(
                "Inplace offset graph input backing buffer is too small: "
                f"name={name}, graph_stop={int(graph_stop)}, "
                f"tensor_shape={tuple(tensor.shape)}, token_dim={token_dim}, "
                f"backing_shape={tuple(backing_tensor.shape)}")
        if token_dim == 0:
            return backing_tensor[:graph_stop]
        if token_dim == 1:
            return backing_tensor[:, :graph_stop]
        raise ValueError(f"Unsupported token_dim={token_dim} for {name}")

    def _expand_inplace_inputs_for_graph_slice(
            self,
            graph_stop: int,
            input_ids: Optional[torch.Tensor],
            positions: Optional[torch.Tensor],
            inputs_embeds: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        input_ids_backing = getattr(getattr(self, "input_ids", None), "gpu",
                                    None)
        positions_backing = getattr(getattr(self, "positions", None), "gpu",
                                    None)
        inputs_embeds_backing = getattr(
            getattr(self, "inputs_embeds", None), "gpu", None)

        input_ids = self._maybe_expand_tensor_for_graph_slice(
            input_ids,
            input_ids_backing,
            graph_stop,
            name="input_ids")

        if positions is not None:
            positions_token_dim = 1 if positions.ndim == 2 else 0
            if positions_token_dim == 1:
                mrope_backing = getattr(getattr(self, "mrope_positions", None),
                                        "gpu", None)
                if mrope_backing is not None:
                    positions_backing = mrope_backing
            positions = self._maybe_expand_tensor_for_graph_slice(
                positions,
                positions_backing,
                graph_stop,
                name="positions",
                token_dim=positions_token_dim)

        inputs_embeds = self._maybe_expand_tensor_for_graph_slice(
            inputs_embeds,
            inputs_embeds_backing,
            graph_stop,
            name="inputs_embeds")
        return input_ids, positions, inputs_embeds

    def _fill_inplace_padding_tail(
            self,
            split_slice: SplitBatchSlice,
            input_ids: Optional[torch.Tensor],
            positions: Optional[torch.Tensor],
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            *,
            stream: Optional[Any] = None,
            collect_debug_payload: bool = False,
    ) -> dict[str, Any]:
        tail_slice = self._padding_tail_slice_for_split(split_slice)
        if tail_slice is None:
            if collect_debug_payload:
                return {
                    "tail_filled": False,
                    "padding_tokens": 0,
                    "actual_token_stop": int(split_slice.token_slice.stop),
                    "graph_token_stop": int(split_slice.token_slice.stop),
                }
            return {"tail_filled": False}

        def _fill() -> dict[str, Any]:
            filled_input_ids = self._fill_tensor_token_tail(
                input_ids, tail_slice, name="input_ids")
            filled_positions = False
            if positions is not None:
                positions_token_dim = 1 if positions.ndim == 2 else 0
                filled_positions = self._fill_tensor_token_tail(
                    positions,
                    tail_slice,
                    name="positions",
                    token_dim=positions_token_dim)
            filled_inputs_embeds = self._fill_tensor_token_tail(
                inputs_embeds, tail_slice, name="inputs_embeds")

            filled_intermediate_tensors: list[str] = []
            if intermediate_tensors is not None:
                intermediate_tail = self._intermediate_tensor_slice_for_tokens(
                    tail_slice)
                for name, tensor in intermediate_tensors.tensors.items():
                    filled = self._fill_tensor_token_tail(
                        tensor, intermediate_tail, name=name)
                    if collect_debug_payload and filled:
                        filled_intermediate_tensors.append(str(name))

            if not collect_debug_payload:
                return {"tail_filled": True}
            return {
                "tail_filled": True,
                "padding_tokens":
                int(tail_slice.stop - tail_slice.start),
                "actual_token_stop": int(tail_slice.start),
                "graph_token_stop": int(tail_slice.stop),
                "input_ids": filled_input_ids,
                "positions": filled_positions,
                "inputs_embeds": filled_inputs_embeds,
                "intermediate_tensors": filled_intermediate_tensors,
            }

        if stream is not None:
            with torch.npu.stream(stream):
                payload = _fill()
        else:
            payload = _fill()

        if collect_debug_payload:
            split_debug.log_event(
                "inplace_offset_padding_tail_filled",
                {
                    "token_start": int(split_slice.token_slice.start),
                    "num_tokens": int(split_slice.num_tokens),
                    "graph_num_tokens": int(split_slice.graph_num_tokens),
                    **payload,
                },
                step_id=_split_debug_step_from_runner(self),
            )
        return payload

    def _tokens_slice_for_inplace_execution(
            self, split_slice: SplitBatchSlice) -> slice:
        if split_slice.start_num_tokens > 0:
            return self._graph_token_slice_for_split(split_slice)
        return split_slice.token_slice

    def _context_ubatch_slices_for_inplace(
            self, split_batch_slices: SplitBatchSlices) -> UBatchSlices:
        return [
            UBatchSlice(s.request_slice,
                        self._tokens_slice_for_inplace_execution(s))
            for s in split_batch_slices
        ]

    def _prepare_inplace_split_inputs_for_execution(
            self,
            split_batch_slices: SplitBatchSlices,
            input_ids: Optional[torch.Tensor],
            positions: Optional[torch.Tensor],
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            *,
            stream_for_split: Optional[Any] = None,
            collect_debug_payload: bool = False,
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for idx, split_slice in enumerate(split_batch_slices):
            tokens_slice = self._tokens_slice_for_inplace_execution(
                split_slice)
            (split_input_ids, split_positions, split_inputs_embeds) = (
                self._expand_inplace_inputs_for_graph_slice(
                    int(tokens_slice.stop),
                    input_ids,
                    positions,
                    inputs_embeds))
            stream = stream_for_split(idx) if stream_for_split else None
            padding_tail_payload = self._fill_inplace_padding_tail(
                split_slice,
                split_input_ids,
                split_positions,
                split_inputs_embeds,
                intermediate_tensors,
                stream=stream,
                collect_debug_payload=collect_debug_payload)
            prepared.append({
                "tokens_slice": tokens_slice,
                "input_ids": split_input_ids,
                "positions": split_positions,
                "inputs_embeds": split_inputs_embeds,
                "padding_tail_payload": padding_tail_payload,
            })
        return prepared

    def _copy_compact_token_tensor(self,
                                   source: Optional[torch.Tensor],
                                   target: Optional[torch.Tensor],
                                   token_slice: slice,
                                   graph_tokens: int,
                                   *,
                                   name: str,
                                   token_dim: int = 0) -> Optional[torch.Tensor]:
        if source is None:
            return None
        if target is None:
            raise RuntimeError(
                f"Mixed request split requires compact buffer for {name}")
        actual_tokens = int(token_slice.stop) - int(token_slice.start)
        graph_tokens = int(graph_tokens)
        if graph_tokens < actual_tokens:
            raise RuntimeError(
                "Mixed request split compact buffer graph size is smaller "
                f"than actual tokens for {name}: graph={graph_tokens}, "
                f"actual={actual_tokens}")
        if token_dim == 0:
            compact = target[:graph_tokens]
            compact[:actual_tokens].copy_(
                source[token_slice], non_blocking=True)
            if graph_tokens > actual_tokens:
                compact[actual_tokens:graph_tokens].fill_(0)
            return compact
        if token_dim == 1:
            compact = target[:, :graph_tokens]
            compact[:, :actual_tokens].copy_(
                source[:, token_slice], non_blocking=True)
            if graph_tokens > actual_tokens:
                compact[:, actual_tokens:graph_tokens].fill_(0)
            return compact
        raise ValueError(f"Unsupported token_dim={token_dim} for {name}")

    @staticmethod
    def _shares_tensor_storage(lhs: Optional[torch.Tensor],
                               rhs: Optional[torch.Tensor]) -> bool:
        if not isinstance(lhs, torch.Tensor) or not isinstance(
                rhs, torch.Tensor):
            return False
        try:
            return (lhs.untyped_storage().data_ptr()
                    == rhs.untyped_storage().data_ptr())
        except Exception:
            return lhs.data_ptr() == rhs.data_ptr()

    @classmethod
    def _needs_mixed_request_source_snapshot(
            cls,
            source: Optional[torch.Tensor],
            target: Optional[torch.Tensor],
            split_batch_slices: SplitBatchSlices,
    ) -> bool:
        if not cls._shares_tensor_storage(source, target):
            return False
        final_token_stop = max(int(s.token_slice.stop)
                               for s in split_batch_slices)
        for split_slice in split_batch_slices:
            if (int(split_slice.graph_num_tokens) > int(
                    split_slice.num_tokens)
                    and int(split_slice.token_slice.stop) < final_token_stop):
                return True
        return False

    def _prepare_mixed_request_split_compact_inputs(
            self,
            split_batch_slices: SplitBatchSlices,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> list[dict[str, Any]]:
        if intermediate_tensors is not None:
            raise RuntimeError(
                "mixed request split does not yet support intermediate_tensors"
            )

        main_input_target = getattr(getattr(self, "input_ids", None), "gpu",
                                    input_ids)
        main_positions_target = getattr(getattr(self, "positions", None),
                                        "gpu", positions)
        main_embeds_target = getattr(getattr(self, "inputs_embeds", None),
                                     "gpu", inputs_embeds)
        input_ids_source = input_ids
        positions_source = positions
        inputs_embeds_source = inputs_embeds
        if self._needs_mixed_request_source_snapshot(
                input_ids, main_input_target, split_batch_slices):
            input_ids_source = input_ids.clone() if input_ids is not None else None
        if self._needs_mixed_request_source_snapshot(
                positions, main_positions_target, split_batch_slices):
            positions_source = positions.clone()
        if self._needs_mixed_request_source_snapshot(
                inputs_embeds, main_embeds_target, split_batch_slices):
            inputs_embeds_source = (
                inputs_embeds.clone()
                if inputs_embeds is not None else None)

        prepared: list[dict[str, Any]] = []
        for idx, split_slice in enumerate(split_batch_slices):
            graph_tokens = int(split_slice.graph_num_tokens)
            token_slice = split_slice.token_slice
            if idx == 0:
                input_target = main_input_target
                positions_target = main_positions_target
                embeds_target = main_embeds_target
            else:
                input_target = getattr(
                    getattr(self, "input_ids_parallel_streams", None), "gpu",
                    None)
                positions_target = getattr(
                    getattr(self, "positions_parallel_streams", None), "gpu",
                    None)
                embeds_target = getattr(
                    getattr(self, "inputs_embeds_parallel_streams", None),
                    "gpu", None)

            compact_input_ids = self._copy_compact_token_tensor(
                input_ids_source,
                input_target,
                token_slice,
                graph_tokens,
                name="input_ids")

            positions_token_dim = 1 if positions.ndim == 2 else 0
            compact_positions = self._copy_compact_token_tensor(
                positions_source,
                positions_target,
                token_slice,
                graph_tokens,
                name="positions",
                token_dim=positions_token_dim)

            compact_inputs_embeds = self._copy_compact_token_tensor(
                inputs_embeds_source,
                embeds_target,
                token_slice,
                graph_tokens,
                name="inputs_embeds")

            prepared.append({
                "input_ids": compact_input_ids,
                "positions": compact_positions,
                "inputs_embeds": compact_inputs_embeds,
                "intermediate_tensors": None,
                "local_ubatch_slice": UBatchSlice(
                    slice(0, split_slice.num_requests),
                    slice(0, graph_tokens)),
            })

            if split_debug.is_enabled():
                split_debug.log_event(
                    "mixed_request_compact_input",
                    {
                        "idx": idx,
                        "source_token_start": int(token_slice.start),
                        "source_token_stop": int(token_slice.stop),
                        "num_tokens": int(split_slice.num_tokens),
                        "graph_num_tokens": graph_tokens,
                        "input_ids": split_debug.tensor_view_info(
                            compact_input_ids),
                        "positions": split_debug.tensor_view_info(
                            compact_positions),
                        "inputs_embeds": split_debug.tensor_view_info(
                            compact_inputs_embeds),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

        return prepared

    def _make_split_batch_metadata_parallel_streams(
            self, split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor], positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode) -> list[AscendUbatchMetadata]:

        forward_contexts = []
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata

        for i, split_slice in enumerate(split_batch_slices):
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_num_tokens = split_slice.num_tokens
            ubatch_num_reqs = split_slice.num_requests
            # Use dispatcher to get the correct BatchDescriptor that matches
            # what was captured at graph capture time. This ensures the key
            # used at runtime matches the key stored in concrete_aclgraph_entries
            # (or concrete_aclgraph_entries2 for parallel streams).
            # uniform_decode=True because split batch only runs on uniform decode.
            _, ubatch_batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=split_slice.padded_num_tokens,
                uniform_decode=batch_descriptor.uniform,
                has_lora=batch_descriptor.has_lora,
            )
            if split_debug.is_enabled():
                split_debug.log_event(
                    "split_descriptor",
                    {
                        "idx": i,
                        "dispatch_num_tokens":
                        int(split_slice.padded_num_tokens),
                        "actual_num_tokens": int(ubatch_num_tokens),
                        "runtime_mode": (
                            aclgraph_runtime_mode.name
                            if isinstance(aclgraph_runtime_mode,
                                          CUDAGraphMode) else
                            str(aclgraph_runtime_mode)),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            ubatch_batch_descriptor),
                        "in_parallel_streams": bool(i > 0),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            # For non-first splits, execute create_ascend_forward_context on
            # stream_parallel so that the GPU ops inside it (update_cos_sin,
            # clone) overlap with the stream_main work for split-0, eliminating
            # the bubble that was visible just before set_stream_limit.
            ctx_stream = self.stream_parallel if i > 0 else self.stream_main
            with torch.npu.stream(ctx_stream):
                split_forward_context = create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=split_ubatch_slices,
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=aclgraph_runtime_mode,
                    ubatch_num=i,
                    positions=positions,
                    in_parallel_streams=(i > 0),
                    cos_sin_slot_id=i,
                )
                _set_split_debug_step(split_forward_context,
                                      _split_debug_step_from_runner(self))
                forward_contexts.append(split_forward_context)

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, split_slice in enumerate(split_batch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = self._slice_split_batch_inputs(
                split_slice.token_slice, input_ids, positions, inputs_embeds,
                intermediate_tensors)
            
            # For non-first split, rebind to dedicated parallel buffers.
            # NOTE: input_ids and positions are already copied in _prepare_inputs
            # to avoid copy bubbles between graph replays.
            # inputs_embeds still needs copy here because it's populated after
            # _prepare_inputs returns (in the embedding layer).
            if i > 0:
                num_tokens = split_slice.num_tokens
                padded_tokens = split_slice.padded_num_tokens

                # Rebind input_ids and positions to padded parallel buffers.
                # The tail [num_tokens:padded_tokens] was already zeroed in
                # _prepare_inputs, so the graph sees a full padded-size tensor.
                if sliced_input_ids is not None:
                    sliced_input_ids = self.input_ids_parallel_streams.gpu[:padded_tokens]

                if sliced_positions is not None:
                    if sliced_positions.ndim == 2:
                        sliced_positions = self.positions_parallel_streams.gpu[:, :padded_tokens]
                    else:
                        sliced_positions = self.positions_parallel_streams.gpu[:padded_tokens]

                # Copy inputs_embeds to parallel buffer (not done in _prepare_inputs).
                # Run on stream_parallel so it overlaps with stream_main work.
                if sliced_inputs_embeds is not None:
                    with torch.npu.stream(self.stream_parallel):
                        self.inputs_embeds_parallel_streams.gpu[:num_tokens].copy_(
                            sliced_inputs_embeds, non_blocking=True)
                        if padded_tokens > num_tokens:
                            self.inputs_embeds_parallel_streams.gpu[
                                num_tokens:padded_tokens].fill_(0)
                    sliced_inputs_embeds = self.inputs_embeds_parallel_streams.gpu[:padded_tokens]

            if split_debug.is_enabled():
                split_debug.log_event(
                    "split_input_buffers",
                    {
                        "idx": i,
                        "path": "parallel_buffer" if i > 0 else
                        "original_buffer",
                        "input_ids": split_debug.tensor_info(
                            sliced_input_ids),
                        "positions": split_debug.tensor_info(
                            sliced_positions),
                        "inputs_embeds": split_debug.tensor_info(
                            sliced_inputs_embeds),
                        "source_token_start": split_slice.token_slice.start,
                        "source_token_stop": split_slice.token_slice.stop,
                        "num_tokens": int(split_slice.num_tokens),
                        "padded_num_tokens":
                        int(split_slice.padded_num_tokens),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
                split_debug.log_event(
                    "split_metadata",
                    {
                        "idx": i,
                        "num_tokens": int(split_slice.num_tokens),
                        "padded_num_tokens":
                        int(split_slice.padded_num_tokens),
                        "num_reqs": int(split_slice.num_requests),
                        **split_debug.metadata_tensor_info(
                            forward_contexts[i].attn_metadata),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=forward_contexts[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.padded_num_tokens))

        return ubatch_metadata

    def _make_split_batch_metadata(
            self, split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor], positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode) -> list[AscendUbatchMetadata]:

        forward_contexts = []
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata

        ubatch_cudagraph_mode = CUDAGraphMode.FULL if aclgraph_runtime_mode != CUDAGraphMode.NONE else CUDAGraphMode.NONE

        for i, split_slice in enumerate(split_batch_slices):
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_num_tokens = split_slice.num_tokens
            ubatch_num_reqs = split_slice.num_requests
            # Use dispatcher to get the correct BatchDescriptor matching capture time.
            _, ubatch_batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=split_slice.padded_num_tokens,
                uniform_decode=batch_descriptor.uniform,
                has_lora=batch_descriptor.has_lora,
            )
            if split_debug.is_enabled():
                split_debug.log_event(
                    "split_descriptor",
                    {
                        "idx": i,
                        "dispatch_num_tokens":
                        int(split_slice.padded_num_tokens),
                        "actual_num_tokens": int(ubatch_num_tokens),
                        "runtime_mode": (
                            ubatch_cudagraph_mode.name
                            if isinstance(ubatch_cudagraph_mode,
                                          CUDAGraphMode) else
                            str(ubatch_cudagraph_mode)),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            ubatch_batch_descriptor),
                        "in_parallel_streams": False,
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            split_forward_context = create_ascend_forward_context(
                cur_forward_context,
                attn_metadata=ubatch_attn_metadata,
                vllm_config=self.vllm_config,
                dp_metadata=dp_metadata,
                ubatch_slices=split_ubatch_slices,
                batch_descriptor=ubatch_batch_descriptor,
                cudagraph_runtime_mode=ubatch_cudagraph_mode,
                ubatch_num=i,
                positions=positions,
            )
            if split_debug_enabled:
                _set_split_debug_step(split_forward_context,
                                      _split_debug_step_from_runner(self))
            forward_contexts.append(split_forward_context)

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, split_slice in enumerate(split_batch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = self._slice_split_batch_inputs(
                split_slice.token_slice, input_ids, positions, inputs_embeds,
                intermediate_tensors)

            if split_debug.is_enabled():
                split_debug.log_event(
                    "split_input_buffers",
                    {
                        "idx": i,
                        "path": "original_buffer",
                        "input_ids": split_debug.tensor_info(
                            sliced_input_ids),
                        "positions": split_debug.tensor_info(
                            sliced_positions),
                        "inputs_embeds": split_debug.tensor_info(
                            sliced_inputs_embeds),
                        "source_token_start": split_slice.token_slice.start,
                        "source_token_stop": split_slice.token_slice.stop,
                        "num_tokens": int(split_slice.num_tokens),
                        "padded_num_tokens":
                        int(split_slice.padded_num_tokens),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
                split_debug.log_event(
                    "split_metadata",
                    {
                        "idx": i,
                        "num_tokens": int(split_slice.num_tokens),
                        "padded_num_tokens":
                        int(split_slice.padded_num_tokens),
                        "num_reqs": int(split_slice.num_requests),
                        **split_debug.metadata_tensor_info(
                            forward_contexts[i].attn_metadata),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=forward_contexts[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.padded_num_tokens))

        return ubatch_metadata

    def _make_mixed_request_split_metadata_serial(
            self,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
    ) -> list[AscendUbatchMetadata]:
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata
        prepared_split_inputs = (
            self._prepare_mixed_request_split_compact_inputs(
                split_batch_slices,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors))
        local_ubatch_slices = [
            item["local_ubatch_slice"] for item in prepared_split_inputs
        ]

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for idx, split_slice in enumerate(split_batch_slices):
            if split_slice.start_num_tokens != 0:
                raise RuntimeError(
                    "mixed request split requires compact start_num_tokens=0")

            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and idx < len(
                        attn_metadata):
                    ubatch_attn_metadata = attn_metadata[idx]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=split_slice.graph_num_tokens,
                    uniform_decode=False,
                    has_lora=batch_descriptor.has_lora,
                    disable_full=True,
                    start_num_tokens=0,
                    allow_inplace_lazy_key=False,
                ))
            if ubatch_cudagraph_mode not in (CUDAGraphMode.PIECEWISE,
                                             CUDAGraphMode.NONE):
                raise RuntimeError(
                    "mixed request split serial execution requires "
                    "CUDAGraphMode.PIECEWISE or NONE, got "
                    f"{ubatch_cudagraph_mode}")

            prepared = prepared_split_inputs[idx]
            split_forward_context = create_ascend_forward_context(
                cur_forward_context,
                attn_metadata=ubatch_attn_metadata,
                vllm_config=self.vllm_config,
                dp_metadata=dp_metadata,
                ubatch_slices=local_ubatch_slices,
                batch_descriptor=ubatch_batch_descriptor,
                cudagraph_runtime_mode=ubatch_cudagraph_mode,
                ubatch_num=idx,
                positions=prepared["positions"],
                in_parallel_streams=False,
            )
            _set_split_debug_step(split_forward_context,
                                  _split_debug_step_from_runner(self))
            setattr(split_forward_context, "split_inplace_mode",
                    "mixed_request_serial")
            setattr(split_forward_context, "forced_attention_backend", "")
            setattr(split_forward_context, "allow_inplace_lazy_capture", False)
            setattr(split_forward_context, "split_actual_num_tokens",
                    int(split_slice.num_tokens))
            setattr(split_forward_context, "split_actual_num_reqs",
                    int(split_slice.num_requests))
            setattr(split_forward_context, "split_graph_num_tokens",
                    int(split_slice.graph_num_tokens))
            setattr(split_forward_context, "validate_inplace_input_ptrs",
                    False)
            setattr(split_forward_context, "validate_inplace_metadata_ptrs",
                    False)

            if split_debug.is_enabled():
                split_debug.log_event(
                    "mixed_request_split_descriptor",
                    {
                        "idx": idx,
                        "dispatch_num_tokens":
                        int(split_slice.graph_num_tokens),
                        "actual_num_tokens": int(split_slice.num_tokens),
                        "runtime_mode": ubatch_cudagraph_mode.name,
                        "batch_descriptor":
                        split_debug.batch_descriptor_info(
                            ubatch_batch_descriptor),
                        "metadata": split_debug.metadata_tensor_info(
                            split_forward_context.attn_metadata),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=split_forward_context,
                    input_ids=prepared["input_ids"],
                    positions=prepared["positions"],
                    inputs_embeds=prepared["inputs_embeds"],
                    intermediate_tensors=None,
                    num_tokens=split_slice.graph_num_tokens))

        return ubatch_metadata

    def _make_mixed_request_split_metadata_parallel(
            self,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
    ) -> list[AscendUbatchMetadata]:
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata
        prepared_split_inputs = (
            self._prepare_mixed_request_split_compact_inputs(
                split_batch_slices,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors))
        local_ubatch_slices = [
            item["local_ubatch_slice"] for item in prepared_split_inputs
        ]

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for idx, split_slice in enumerate(split_batch_slices):
            if split_slice.start_num_tokens != 0:
                raise RuntimeError(
                    "mixed request split requires compact start_num_tokens=0")

            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and idx < len(
                        attn_metadata):
                    ubatch_attn_metadata = attn_metadata[idx]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=split_slice.graph_num_tokens,
                    uniform_decode=False,
                    has_lora=batch_descriptor.has_lora,
                    disable_full=True,
                    start_num_tokens=0,
                    allow_inplace_lazy_key=False,
                ))
            if ubatch_cudagraph_mode not in (CUDAGraphMode.PIECEWISE,
                                             CUDAGraphMode.NONE):
                raise RuntimeError(
                    "mixed request split piecewise attention parallel "
                    "requires CUDAGraphMode.PIECEWISE or NONE, got "
                    f"{ubatch_cudagraph_mode}")

            in_parallel_streams = idx > 0
            prepared = prepared_split_inputs[idx]
            ctx_stream = (self.stream_parallel if in_parallel_streams
                          else self.stream_main)
            with torch.npu.stream(ctx_stream):
                split_forward_context = create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=local_ubatch_slices,
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=ubatch_cudagraph_mode,
                    ubatch_num=idx,
                    positions=prepared["positions"],
                    in_parallel_streams=in_parallel_streams,
                    cos_sin_slot_id=idx,
                )
            _set_split_debug_step(split_forward_context,
                                  _split_debug_step_from_runner(self))
            setattr(split_forward_context, "split_inplace_mode",
                    "mixed_request_piecewise_attention_parallel")
            setattr(split_forward_context, "forced_attention_backend", "")
            setattr(split_forward_context, "allow_inplace_lazy_capture", False)
            setattr(split_forward_context, "split_actual_num_tokens",
                    int(split_slice.num_tokens))
            setattr(split_forward_context, "split_actual_num_reqs",
                    int(split_slice.num_requests))
            setattr(split_forward_context, "split_graph_num_tokens",
                    int(split_slice.graph_num_tokens))
            setattr(split_forward_context, "validate_inplace_input_ptrs",
                    False)
            setattr(split_forward_context, "validate_inplace_metadata_ptrs",
                    False)

            if split_debug.is_enabled():
                split_debug.log_event(
                    "mixed_request_split_descriptor",
                    {
                        "idx": idx,
                        "execution":
                        "mixed_request_piecewise_attention_parallel",
                        "stream": ("parallel" if in_parallel_streams
                                   else "main"),
                        "dispatch_num_tokens":
                        int(split_slice.graph_num_tokens),
                        "actual_num_tokens": int(split_slice.num_tokens),
                        "runtime_mode": ubatch_cudagraph_mode.name,
                        "batch_descriptor":
                        split_debug.batch_descriptor_info(
                            ubatch_batch_descriptor),
                        "metadata": split_debug.metadata_tensor_info(
                            split_forward_context.attn_metadata),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=split_forward_context,
                    input_ids=prepared["input_ids"],
                    positions=prepared["positions"],
                    inputs_embeds=prepared["inputs_embeds"],
                    intermediate_tensors=None,
                    num_tokens=split_slice.graph_num_tokens))

        return ubatch_metadata

    def _make_split_batch_metadata_inplace_serial(
            self, split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor], positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str) -> list[AscendUbatchMetadata]:

        forward_contexts = []
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata
        context_ubatch_slices = self._context_ubatch_slices_for_inplace(
            split_batch_slices)
        split_debug_enabled = split_debug.is_enabled()
        prepared_split_inputs = self._prepare_inplace_split_inputs_for_execution(
            split_batch_slices,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            collect_debug_payload=split_debug_enabled)
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        allow_lazy = bool(split_cfg is not None and getattr(
            split_cfg, "enable_inplace_lazy_capture", True))
        force_pa_for_offset = bool(
            split_cfg is not None
            and getattr(split_cfg, "inplace_force_pa_for_offset", False))

        for i, split_slice in enumerate(split_batch_slices):
            split_attention_backend = inplace_attention_backend
            if force_pa_for_offset and split_slice.start_num_tokens > 0:
                split_attention_backend = "pa"
            capture_metadata_mode = (
                "template"
                if split_slice.start_num_tokens > 0
                and split_attention_backend == "fia" else "")
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=split_slice.graph_num_tokens,
                    # Inplace split execution is only reached after the
                    # planner accepted a uniform decode batch. The outer
                    # descriptor can still be conservative/non-uniform after
                    # prefill scheduling, so use the split invariant here.
                    uniform_decode=True,
                    has_lora=batch_descriptor.has_lora,
                    start_num_tokens=split_slice.start_num_tokens,
                    allow_inplace_lazy_key=(
                        allow_lazy and split_slice.start_num_tokens > 0),
                    graph_variant=("inplace_serial"
                                   if split_slice.start_num_tokens > 0 else ""),
                    attention_backend=(split_attention_backend
                                       if split_slice.start_num_tokens > 0
                                       else ""),
                    capture_metadata_mode=capture_metadata_mode,
                ))

            allow_inplace_lazy_capture = bool(
                allow_lazy and split_slice.start_num_tokens > 0
                and ubatch_cudagraph_mode
                in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)
                and getattr(ubatch_batch_descriptor, "graph_variant", "")
                == "inplace_serial"
                and getattr(ubatch_batch_descriptor, "attention_backend", "")
                in ("fia", "pa"))
            templated_fia_seq_lens = 0
            if (split_slice.start_num_tokens > 0
                    and split_attention_backend == "fia"
                    and ubatch_attn_metadata is not None
                    and getattr(ubatch_batch_descriptor,
                                "capture_metadata_mode", "") == "template"):
                templated_fia_seq_lens = _template_fia_seq_lens_list(
                    ubatch_attn_metadata, self.block_size)
            validate_inplace_ptrs = bool(
                split_cfg is not None
                and getattr(split_cfg, "inplace_validate_metadata_ptrs",
                            False)
                and split_slice.start_num_tokens > 0)
            validate_inplace_metadata_ptrs = validate_inplace_ptrs

            if split_debug_enabled:
                split_debug.log_event(
                    "split_descriptor",
                    {
                        "idx": i,
                        "execution": "inplace_serial",
                        "dispatch_num_tokens":
                        int(split_slice.graph_num_tokens),
                        "actual_num_tokens": int(split_slice.num_tokens),
                        "runtime_mode": (
                            ubatch_cudagraph_mode.name
                            if isinstance(ubatch_cudagraph_mode,
                                          CUDAGraphMode) else
                            str(ubatch_cudagraph_mode)),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            ubatch_batch_descriptor),
                        "in_parallel_streams": False,
                        "allow_inplace_lazy_capture":
                        allow_inplace_lazy_capture,
                        "validate_inplace_ptrs": validate_inplace_ptrs,
                        "validate_inplace_metadata_ptrs":
                        validate_inplace_metadata_ptrs,
                        "forced_attention_backend":
                        split_attention_backend,
                        "force_pa_for_offset":
                        force_pa_for_offset,
                        "templated_fia_seq_lens":
                        templated_fia_seq_lens,
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            split_forward_context = create_ascend_forward_context(
                cur_forward_context,
                attn_metadata=ubatch_attn_metadata,
                vllm_config=self.vllm_config,
                dp_metadata=dp_metadata,
                ubatch_slices=context_ubatch_slices,
                batch_descriptor=ubatch_batch_descriptor,
                cudagraph_runtime_mode=ubatch_cudagraph_mode,
                ubatch_num=i,
                positions=prepared_split_inputs[i]["positions"],
                in_parallel_streams=False,
            )
            if split_debug_enabled:
                _set_split_debug_step(split_forward_context,
                                      _split_debug_step_from_runner(self))
            setattr(split_forward_context, "split_inplace_mode",
                    "inplace_serial")
            setattr(split_forward_context, "forced_attention_backend",
                    split_attention_backend)
            setattr(split_forward_context, "allow_inplace_lazy_capture",
                    allow_inplace_lazy_capture)
            setattr(split_forward_context, "split_actual_num_tokens",
                    int(split_slice.num_tokens))
            setattr(split_forward_context, "split_actual_num_reqs",
                    int(split_slice.num_requests))
            setattr(split_forward_context, "split_graph_num_tokens",
                    int(split_slice.graph_num_tokens))
            setattr(split_forward_context, "validate_inplace_input_ptrs",
                    validate_inplace_ptrs)
            setattr(split_forward_context, "validate_inplace_metadata_ptrs",
                    validate_inplace_metadata_ptrs)
            forward_contexts.append(split_forward_context)

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, split_slice in enumerate(split_batch_slices):
            prepared_inputs = prepared_split_inputs[i]
            tokens_slice = prepared_inputs["tokens_slice"]
            padding_tail_payload = prepared_inputs["padding_tail_payload"]
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = self._slice_split_batch_inputs(
                tokens_slice, prepared_inputs["input_ids"],
                prepared_inputs["positions"],
                prepared_inputs["inputs_embeds"],
                intermediate_tensors)

            if split_debug.is_enabled():
                split_debug.log_event(
                    "inplace_serial_execution",
                    {
                        "idx": i,
                        "token_start": int(split_slice.token_slice.start),
                        "token_stop": int(split_slice.token_slice.stop),
                        "graph_token_stop": int(tokens_slice.stop),
                        "start_num_tokens":
                        int(split_slice.start_num_tokens),
                        "num_tokens": int(split_slice.num_tokens),
                        "graph_num_tokens": int(split_slice.graph_num_tokens),
                        "padding_tokens": int(
                            split_slice.graph_num_tokens -
                            split_slice.num_tokens),
                        "tail_filled": bool(
                            padding_tail_payload.get("tail_filled", False)),
                        "input_ids": split_debug.tensor_view_info(
                            sliced_input_ids),
                        "positions": split_debug.tensor_view_info(
                            sliced_positions),
                        "inputs_embeds": split_debug.tensor_view_info(
                            sliced_inputs_embeds),
                        "metadata": split_debug.metadata_tensor_info(
                            forward_contexts[i].attn_metadata),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            forward_contexts[i].batch_descriptor),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=forward_contexts[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.graph_num_tokens))

        return ubatch_metadata

    def _make_split_batch_metadata_inplace_parallel(
            self, split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor], positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str) -> list[AscendUbatchMetadata]:

        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata
        context_ubatch_slices = self._context_ubatch_slices_for_inplace(
            split_batch_slices)
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        split_debug_enabled = split_debug.is_enabled()
        allow_lazy = bool(split_cfg is not None and getattr(
            split_cfg, "enable_inplace_lazy_capture", True))
        force_pa_for_offset = bool(
            split_cfg is not None
            and getattr(split_cfg, "inplace_force_pa_for_offset", False))

        def _stream_for_split(split_idx: int):
            return self.stream_parallel if split_idx > 0 else self.stream_main

        prepared_split_inputs = self._prepare_inplace_split_inputs_for_execution(
            split_batch_slices,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            stream_for_split=_stream_for_split,
            collect_debug_payload=split_debug_enabled)

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, split_slice in enumerate(split_batch_slices):
            in_parallel_streams = i > 0
            split_attention_backend = inplace_attention_backend
            if force_pa_for_offset and split_slice.start_num_tokens > 0:
                split_attention_backend = "pa"
            capture_metadata_mode = (
                "template"
                if split_slice.start_num_tokens > 0
                and split_attention_backend == "fia" else "")
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=split_slice.graph_num_tokens,
                    uniform_decode=True,
                    has_lora=batch_descriptor.has_lora,
                    start_num_tokens=split_slice.start_num_tokens,
                    allow_inplace_lazy_key=(
                        allow_lazy and split_slice.start_num_tokens > 0),
                    graph_variant=("inplace_parallel"
                                   if split_slice.start_num_tokens > 0 else ""),
                    attention_backend=(split_attention_backend
                                       if split_slice.start_num_tokens > 0
                                       else ""),
                    capture_metadata_mode=capture_metadata_mode,
                ))

            allow_inplace_lazy_capture = bool(
                allow_lazy and split_slice.start_num_tokens > 0
                and ubatch_cudagraph_mode
                in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)
                and getattr(ubatch_batch_descriptor, "graph_variant", "")
                == "inplace_parallel"
                and getattr(ubatch_batch_descriptor, "attention_backend", "")
                in ("fia", "pa"))
            templated_fia_seq_lens = 0
            if (split_slice.start_num_tokens > 0
                    and split_attention_backend == "fia"
                    and ubatch_attn_metadata is not None
                    and getattr(ubatch_batch_descriptor,
                                "capture_metadata_mode", "") == "template"):
                templated_fia_seq_lens = _template_fia_seq_lens_list(
                    ubatch_attn_metadata, self.block_size)
            validate_inplace_ptrs = bool(
                split_cfg is not None
                and getattr(split_cfg, "inplace_validate_metadata_ptrs",
                            False)
                and split_slice.start_num_tokens > 0)
            validate_inplace_metadata_ptrs = validate_inplace_ptrs

            if split_debug_enabled:
                split_debug.log_event(
                    "split_descriptor",
                    {
                        "idx": i,
                        "execution": "inplace_parallel",
                        "stream": ("parallel" if in_parallel_streams
                                   else "main"),
                        "dispatch_num_tokens":
                        int(split_slice.graph_num_tokens),
                        "actual_num_tokens": int(split_slice.num_tokens),
                        "runtime_mode": (
                            ubatch_cudagraph_mode.name
                            if isinstance(ubatch_cudagraph_mode,
                                          CUDAGraphMode) else
                            str(ubatch_cudagraph_mode)),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            ubatch_batch_descriptor),
                        "in_parallel_streams": in_parallel_streams,
                        "graph_params_pool": ("parallel"
                                              if in_parallel_streams
                                              else "main"),
                        "graph_entry_pool": ("parallel"
                                             if in_parallel_streams
                                             else "main"),
                        "allow_inplace_lazy_capture":
                        allow_inplace_lazy_capture,
                        "validate_inplace_ptrs": validate_inplace_ptrs,
                        "validate_inplace_metadata_ptrs":
                        validate_inplace_metadata_ptrs,
                        "forced_attention_backend":
                        split_attention_backend,
                        "force_pa_for_offset":
                        force_pa_for_offset,
                        "templated_fia_seq_lens":
                        templated_fia_seq_lens,
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ctx_stream = (self.stream_parallel if in_parallel_streams
                          else self.stream_main)
            with torch.npu.stream(ctx_stream):
                split_forward_context = create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=context_ubatch_slices,
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=ubatch_cudagraph_mode,
                    ubatch_num=i,
                    positions=prepared_split_inputs[i]["positions"],
                    in_parallel_streams=in_parallel_streams,
                    cos_sin_slot_id=i,
                )
            if split_debug_enabled:
                _set_split_debug_step(split_forward_context,
                                      _split_debug_step_from_runner(self))
            setattr(split_forward_context, "split_inplace_mode",
                    "inplace_parallel")
            setattr(split_forward_context, "forced_attention_backend",
                    split_attention_backend)
            setattr(split_forward_context, "allow_inplace_lazy_capture",
                    allow_inplace_lazy_capture)
            setattr(split_forward_context, "split_actual_num_tokens",
                    int(split_slice.num_tokens))
            setattr(split_forward_context, "split_actual_num_reqs",
                    int(split_slice.num_requests))
            setattr(split_forward_context, "split_graph_num_tokens",
                    int(split_slice.graph_num_tokens))
            setattr(split_forward_context, "validate_inplace_input_ptrs",
                    validate_inplace_ptrs)
            setattr(split_forward_context, "validate_inplace_metadata_ptrs",
                    validate_inplace_metadata_ptrs)
            prepared_inputs = prepared_split_inputs[i]
            tokens_slice = prepared_inputs["tokens_slice"]
            padding_tail_payload = prepared_inputs["padding_tail_payload"]
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = self._slice_split_batch_inputs(
                tokens_slice, prepared_inputs["input_ids"],
                prepared_inputs["positions"],
                prepared_inputs["inputs_embeds"],
                intermediate_tensors)

            if split_debug_enabled:
                split_debug.log_event(
                    "inplace_parallel_execution",
                    {
                        "idx": i,
                        "stream": ("parallel" if in_parallel_streams
                                   else "main"),
                        "buffer_source": "original_offset_view",
                        "graph_params_pool": ("parallel"
                                              if in_parallel_streams
                                              else "main"),
                        "graph_entry_pool": ("parallel"
                                             if in_parallel_streams
                                             else "main"),
                        "token_start": int(split_slice.token_slice.start),
                        "token_stop": int(split_slice.token_slice.stop),
                        "graph_token_stop": int(tokens_slice.stop),
                        "start_num_tokens":
                        int(split_slice.start_num_tokens),
                        "num_tokens": int(split_slice.num_tokens),
                        "graph_num_tokens": int(split_slice.graph_num_tokens),
                        "padding_tokens": int(
                            split_slice.graph_num_tokens -
                            split_slice.num_tokens),
                        "tail_filled": bool(
                            padding_tail_payload.get("tail_filled", False)),
                        "input_ids": split_debug.tensor_view_info(
                            sliced_input_ids),
                        "positions": split_debug.tensor_view_info(
                            sliced_positions),
                        "inputs_embeds": split_debug.tensor_view_info(
                            sliced_inputs_embeds),
                        "metadata":
                        split_debug.metadata_tensor_info(
                            split_forward_context.attn_metadata),
                        "batch_descriptor": split_debug.batch_descriptor_info(
                            split_forward_context.batch_descriptor),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=split_forward_context,
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.graph_num_tokens))

        return ubatch_metadata

    def _merge_intermediate_tensors(self, intermediate_tensor_list):
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat(
                [it.tensors[key] for it in intermediate_tensor_list], dim=0)
        return IntermediateTensors(result)

    def _trim_split_output(self, output: Any, num_tokens: int) -> Any:
        """Trim a padded split output to its actual token count.

        When padding is applied to align each split to a cudagraph capture size,
        the model output has shape [padded_N, ...].  This helper slices off the
        padding rows so that _merge_split_outputs concatenates only real tokens.
        """
        if isinstance(output, torch.Tensor):
            return output[:num_tokens]
        if isinstance(output, list):
            return [
                self._trim_split_output(item, num_tokens)
                for item in output
            ]
        if isinstance(output, tuple):
            return tuple(
                self._trim_split_output(item, num_tokens)
                for item in output
            )
        if isinstance(output, IntermediateTensors):
            return IntermediateTensors(
                {k: v[:num_tokens] for k, v in output.tensors.items()}
            )
        # Fallback: return as-is (e.g. None or unknown type)
        return output

    def _clone_split_output(self, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output.clone()
        if isinstance(output, list):
            return [self._clone_split_output(item) for item in output]
        if isinstance(output, tuple):
            return tuple(
                self._clone_split_output(item) for item in output)
        if isinstance(output, IntermediateTensors):
            return IntermediateTensors(
                {k: v.clone() for k, v in output.tensors.items()})
        return output

    def _merge_split_outputs(self, outputs: list[Any]) -> Any:
        if not outputs:
            return None
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(outputs, dim=0)
        if isinstance(first, IntermediateTensors):
            return self._merge_intermediate_tensors(outputs)
        if isinstance(first, (list, tuple)):
            if any(len(output) != len(first) for output in outputs):
                raise RuntimeError(
                    "Cannot merge split outputs with different container "
                    "lengths")
            merged = [
                self._merge_split_outputs([output[idx] for output in outputs])
                for idx in range(len(first))
            ]
            return type(first)(merged)
        return first

    def _has_aclgraph_for_context(self, context: Any) -> bool:
        has_graph = getattr(self.model, "has_graph", None)
        if not callable(has_graph):
            return False
        return bool(
            has_graph(
                getattr(context, "batch_descriptor", None),
                bool(getattr(context, "in_parallel_streams", False)),
            ))

    def _needs_inplace_serial_offset_capture(
            self, metadata: AscendUbatchMetadata) -> bool:
        context = metadata.context
        batch_descriptor = getattr(context, "batch_descriptor", None)
        if batch_descriptor is None:
            return False
        if getattr(context, "cudagraph_runtime_mode",
                   CUDAGraphMode.NONE) != CUDAGraphMode.FULL:
            return False
        if int(getattr(batch_descriptor, "start_num_tokens", 0) or 0) <= 0:
            return False
        if not bool(getattr(context, "allow_inplace_lazy_capture", False)):
            return False
        return not self._has_aclgraph_for_context(context)

    @contextmanager
    def _bind_inplace_parallel_rope_capture_slot(
            self, context: Any, *, parallel_streams: bool):
        """Bind the traced RoPE slot input to split-1 during lazy capture.

        The compiled backbone was initially traced with the slot-0 module
        global as its RoPE input.  ACL graph replay no longer resolves that
        Python global after capture, so rebind it only while capturing the
        offset graph and make that graph retain the split-1 buffer address.
        """
        split_mode = getattr(context, "split_inplace_mode", "")
        if (not parallel_streams or split_mode not in (
                "inplace_parallel",
                "mixed_request_piecewise_attention_parallel",
        )):
            yield
            return

        slot_id = int(getattr(context, "cos_sin_slot_id", 0) or 0)
        if slot_id <= 0:
            yield
            return

        import vllm_ascend.ops.rotary_embedding as rotary_embedding

        cos, sin = rotary_embedding.get_cos_and_sin_slice(slot_id=slot_id)
        if cos is None or sin is None:
            raise RuntimeError(
                "Missing rotary cos/sin buffers for inplace parallel "
                f"capture slot {slot_id}")

        previous_cos = rotary_embedding._cos_slice_slots[0]
        previous_sin = rotary_embedding._sin_slice_slots[0]
        rotary_embedding._cos_slice_slots[0] = cos
        rotary_embedding._sin_slice_slots[0] = sin
        if split_debug.is_enabled():
            split_debug.log_event(
                "inplace_parallel_rope_slot_binding",
                {
                    "phase": "bind",
                    "compiled_slot_id": 0,
                    "capture_slot_id": slot_id,
                    "cos": split_debug.tensor_info(cos),
                    "sin": split_debug.tensor_info(sin),
                },
                step_id=_split_debug_step_from_runner(self),
            )
        try:
            yield
        finally:
            rotary_embedding._cos_slice_slots[0] = previous_cos
            rotary_embedding._sin_slice_slots[0] = previous_sin
            if split_debug.is_enabled():
                split_debug.log_event(
                    "inplace_parallel_rope_slot_binding",
                    {
                        "phase": "restore",
                        "compiled_slot_id": 0,
                        "capture_slot_id": slot_id,
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

    def _run_inplace_serial_offset_capture(
            self,
            metadata: AscendUbatchMetadata,
            split_slice: Any,
            model_kwargs: dict[str, Any],
            *,
            parallel_streams: bool = False,
    ) -> Any:
        """Warm up and capture a missing inplace offset graph on demand.

        The normal GPU/NPU preload path runs eager warmups before graph
        capture.  Split-1 offset graphs are intentionally not pre-captured, so
        this path reproduces that warmup-before-capture shape before the split
        enters the regular replay branch.
        """
        context = metadata.context
        batch_descriptor = getattr(context, "batch_descriptor", None)
        warmups = int(
            getattr(self.compilation_config, "cudagraph_num_of_warmups", 0)
            or 0)
        previous_mode = getattr(context, "cudagraph_runtime_mode",
                                CUDAGraphMode.NONE)
        previous_capturing = bool(getattr(context, "capturing", False))
        step_id = _split_debug_step_from_runner(self)
        target_stream = (self.stream_parallel if parallel_streams
                         else self.stream_main)

        if split_debug.is_enabled():
            split_debug.log_event(
                "inplace_lazy_capture_prepare",
                {
                    "batch_descriptor":
                    split_debug.batch_descriptor_info(batch_descriptor),
                    "warmups": warmups,
                    "num_tokens": int(split_slice.num_tokens),
                    "graph_num_tokens": int(split_slice.graph_num_tokens),
                    "start_num_tokens":
                    int(getattr(batch_descriptor, "start_num_tokens", 0)
                        or 0),
                    "in_parallel_streams": bool(parallel_streams),
                    "stream": "parallel" if parallel_streams else "main",
                },
                step_id=step_id,
            )

        replay_result = None
        try:
            with (
                self._bind_inplace_parallel_rope_capture_slot(
                    context, parallel_streams=parallel_streams),
                torch.npu.stream(target_stream),
            ):
                for warmup_idx in range(warmups):
                    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    context.capturing = False
                    with override_forward_context(context):
                        _ = self.model(
                            input_ids=metadata.input_ids,
                            positions=metadata.positions,
                            inputs_embeds=metadata.inputs_embeds,
                            intermediate_tensors=metadata.intermediate_tensors,
                            **model_kwargs,
                        )
                    target_stream.synchronize()
                    if split_debug.is_enabled():
                        split_debug.log_event(
                            "inplace_lazy_capture_warmup",
                            {
                                "warmup_idx": warmup_idx,
                                "batch_descriptor":
                                split_debug.batch_descriptor_info(
                                    batch_descriptor),
                                "in_parallel_streams": bool(parallel_streams),
                                "stream": ("parallel" if parallel_streams
                                           else "main"),
                            },
                            step_id=step_id,
                        )

                context.cudagraph_runtime_mode = previous_mode
                context.capturing = False
                with override_forward_context(context):
                    _ = self.model(
                        input_ids=metadata.input_ids,
                        positions=metadata.positions,
                        inputs_embeds=metadata.inputs_embeds,
                        intermediate_tensors=metadata.intermediate_tensors,
                        **model_kwargs,
                    )
                target_stream.synchronize()
        finally:
            context.cudagraph_runtime_mode = previous_mode
            context.capturing = previous_capturing

        if not self._has_aclgraph_for_context(context):
            raise RuntimeError(
                "Inplace serial offset graph capture did not create an ACL "
                f"graph entry for {batch_descriptor!r}")

        with torch.npu.stream(target_stream):
            context.cudagraph_runtime_mode = previous_mode
            context.capturing = False
            with override_forward_context(context):
                replay_result = self.model(
                    input_ids=metadata.input_ids,
                    positions=metadata.positions,
                    inputs_embeds=metadata.inputs_embeds,
                    intermediate_tensors=metadata.intermediate_tensors,
                    **model_kwargs,
                )
                if context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    self._update_attn_params_for_split_ubatch(
                        context,
                        split_slice.graph_num_tokens,
                        parallel_streams=parallel_streams)
            target_stream.synchronize()

        if split_debug.is_enabled():
            split_debug.log_event(
                "inplace_lazy_capture_complete",
                {
                    "batch_descriptor":
                    split_debug.batch_descriptor_info(batch_descriptor),
                    "num_tokens": int(split_slice.num_tokens),
                    "graph_num_tokens": int(split_slice.graph_num_tokens),
                    "returned": "replay_after_capture",
                    "in_parallel_streams": bool(parallel_streams),
                    "stream": "parallel" if parallel_streams else "main",
                },
                step_id=step_id,
            )
        return replay_result

    def _run_split_batch_gr0(
            self,
            split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
        ) -> Any:
            """
            执行split-batch，确保每个batch重放时地址一致。

            核心逻辑：
            - 图捕获时，输入地址在 self.input_ids.gpu、self.positions.gpu 等的起始位置
            - 第一个ubatch执行时，数据已经在正确位置
            - 第二个ubatch执行前，将其数据复制到起始位置，然后用起始位置执行
            - 函数结束前恢复被覆盖的起始位置数据
            """
            # Step 1: 为所有split准备元数据
            ubatch_metadata = self._make_split_batch_metadata(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                batch_descriptor,
                aclgraph_runtime_mode,
            )

            results: list[Any] = []
            original_forward_context = get_forward_context()

            # 获取第一个 split 的 token 数量，用于确定固定缓冲区大小
            first_split_num_tokens = split_batch_slices[0].num_tokens

            # 备份可能被覆盖的前缀区域，函数结束时恢复
            backup_input_ids = None
            if input_ids is not None:
                backup_input_ids = self.input_ids.gpu[:first_split_num_tokens].clone()

            if positions.ndim == 2:
                backup_positions = self.positions.gpu[:, :first_split_num_tokens].clone()
            else:
                backup_positions = self.positions.gpu[:first_split_num_tokens].clone()

            backup_inputs_embeds = None
            if inputs_embeds is not None:
                backup_inputs_embeds = self.inputs_embeds[:first_split_num_tokens].clone()

            base_attn_metadata = attn_metadata[0] if isinstance(
                attn_metadata, list) else attn_metadata
            base_slot_mapping = _get_slot_mapping_from_attn_metadata(
                base_attn_metadata)
            slot_mapping_backup_len = 0
            backup_slot_mapping = None
            if base_slot_mapping is not None:
                slot_mapping_backup_len = min(first_split_num_tokens,
                                              int(base_slot_mapping.shape[0]))
                if slot_mapping_backup_len > 0:
                    backup_slot_mapping = base_slot_mapping[
                        :slot_mapping_backup_len].clone()

            try:
                # Step 2: 依次执行每个split
                for slice_idx, split_slice in enumerate(split_batch_slices):
                    metadata = ubatch_metadata[slice_idx]
                    current_num_tokens = split_slice.num_tokens
                    _append_split_metadata_debug(
                        "gr0_loop_enter",
                        {
                            "slice_idx": slice_idx,
                            "token_slice": [split_slice.token_slice.start,
                                            split_slice.token_slice.stop,
                                            split_slice.token_slice.step],
                            "request_slice": [split_slice.request_slice.start,
                                              split_slice.request_slice.stop,
                                              split_slice.request_slice.step],
                            "num_tokens": current_num_tokens,
                            "context_id": _safe_context_id(metadata.context),
                            **_build_split_tensor_debug("input_ids", metadata.input_ids),
                            **_build_split_tensor_debug("positions", metadata.positions),
                        },
                    )

                    if slice_idx == 0:
                        # 第一个ubatch：数据已在正确位置（self.input_ids.gpu起始处）
                        context_attn_positions_ptr, context_attn_positions_head = _extract_attn_positions(
                            getattr(metadata.context, "attn_metadata", None)
                        )
                        with override_forward_context(metadata.context):
                            _append_split_metadata_debug(
                                "gr0_pre_model",
                                {
                                    "slice_idx": slice_idx,
                                    "context_id": _safe_context_id(metadata.context),
                                    "num_tokens": current_num_tokens,
                                },
                            )
                            # Refresh attn params before replay to avoid carrying
                            # stale settings from the previous split.

                            result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                            # [关键] 在模型执行之后更新 attention 参数
                            self._update_attn_params_for_split_ubatch(
                                metadata.context, current_num_tokens)
                            _append_split_metadata_debug(
                                "gr0_post_model",
                                {
                                    "slice_idx": slice_idx,
                                    "context_id": _safe_context_id(metadata.context),
                                    "num_tokens": current_num_tokens,
                                },
                            )
                    else:
                        # 后续ubatch：需要将数据复制到第一个ubatch的位置
                        # [关键修复] 同步等待前一个图完成执行
                        torch.npu.synchronize()

                        # [关键修复] 将当前ubatch的数据复制到起始位置
                        # input_ids
                        if metadata.input_ids is not None:
                            self.input_ids.gpu[:current_num_tokens].copy_(
                                metadata.input_ids, non_blocking=False)

                        # positions
                        if metadata.positions is not None:
                            if metadata.positions.ndim == 2:
                                self.positions.gpu[:, :current_num_tokens].copy_(
                                    metadata.positions, non_blocking=False)
                            else:
                                self.positions.gpu[:current_num_tokens].copy_(
                                    metadata.positions, non_blocking=False)

                        # inputs_embeds
                        if metadata.inputs_embeds is not None:
                            self.inputs_embeds[:current_num_tokens].copy_(
                                metadata.inputs_embeds, non_blocking=False)

                        # [关键修复] 使用起始位置的张量执行（与图捕获时地址一致）
                        metadata.input_ids = (
                            self.input_ids.gpu[:current_num_tokens]
                            if metadata.input_ids is not None else None
                        )
                        if metadata.positions is not None:
                            if metadata.positions.ndim == 2:
                                metadata.positions = self.positions.gpu[:, :current_num_tokens]
                            else:
                                metadata.positions = self.positions.gpu[:current_num_tokens]
                        metadata.inputs_embeds = (
                            self.inputs_embeds[:current_num_tokens]
                            if metadata.inputs_embeds is not None else None
                        )
                        # Rebuild current split forward context to avoid stale context reuse.
                        ubatch_attn_metadata = None
                        if attn_metadata is not None:
                            if isinstance(attn_metadata, list):
                                _append_split_metadata_debug(
                                    "gr0_attn_metadata_select",
                                    {
                                        "slice_idx": slice_idx,
                                        "attn_metadata_type": type(attn_metadata).__name__,
                                        "attn_metadata_len": len(attn_metadata),
                                    },
                                )
                                if slice_idx >= len(attn_metadata):
                                    raise RuntimeError(
                                        "gr0 attn_metadata list too short: "
                                        f"slice_idx={slice_idx}, len={len(attn_metadata)}"
                                    )
                                ubatch_attn_metadata = attn_metadata[slice_idx]
                            else:
                                ubatch_attn_metadata = attn_metadata

                        split_slot_mapping = _get_slot_mapping_from_attn_metadata(
                            ubatch_attn_metadata)
                        if base_slot_mapping is not None and split_slot_mapping is not None:
                            copy_len = min(current_num_tokens,
                                           int(base_slot_mapping.shape[0]),
                                           int(split_slot_mapping.shape[0]))
                            if copy_len != current_num_tokens:
                                raise RuntimeError(
                                    "gr0 slot_mapping length mismatch: "
                                    f"required={current_num_tokens}, "
                                    f"base={int(base_slot_mapping.shape[0])}, "
                                    f"split={int(split_slot_mapping.shape[0])}")
                            base_slot_mapping[:copy_len].copy_(
                                split_slot_mapping[:copy_len],
                                non_blocking=False)
                            relocated_slot_mapping = base_slot_mapping[:copy_len]
                            updated_count = _set_slot_mapping_for_attn_metadata(
                                ubatch_attn_metadata, relocated_slot_mapping)
                            _append_split_metadata_debug(
                                "gr0_slot_mapping_relocated",
                                {
                                    "slice_idx": slice_idx,
                                    "copy_len": copy_len,
                                    "updated_count": updated_count,
                                    **_build_split_tensor_debug(
                                        "relocated_slot_mapping",
                                        relocated_slot_mapping),
                                },
                            )

                        ubatch_batch_descriptor = BatchDescriptor(
                            num_tokens=current_num_tokens,
                            num_reqs=split_slice.num_requests,
                            uniform=batch_descriptor.uniform,
                            has_lora=batch_descriptor.has_lora,
                        )
                        ubatch_cudagraph_mode = (
                            CUDAGraphMode.FULL
                            if aclgraph_runtime_mode != CUDAGraphMode.NONE
                            else CUDAGraphMode.NONE
                        )
                        if _SPLIT_LOCAL_CONTEXT_REBUILD:
                            rebuild_ubatch_slices = [
                                UBatchSlice(
                                    slice(0, split_slice.num_requests),
                                    slice(0, current_num_tokens),
                                )
                            ]
                            rebuild_positions = metadata.positions
                            rebuild_ubatch_num = 0
                        else:
                            rebuild_ubatch_slices = split_ubatch_slices
                            rebuild_positions = positions
                            rebuild_ubatch_num = slice_idx

                        metadata.context = create_ascend_forward_context(
                            original_forward_context,
                            attn_metadata=ubatch_attn_metadata,
                            vllm_config=self.vllm_config,
                            dp_metadata=original_forward_context.dp_metadata,
                            ubatch_slices=rebuild_ubatch_slices,
                            batch_descriptor=ubatch_batch_descriptor,
                            cudagraph_runtime_mode=ubatch_cudagraph_mode,
                            ubatch_num=rebuild_ubatch_num,
                            positions=rebuild_positions,
                        )
                        _set_split_debug_step(
                            metadata.context,
                            _split_debug_step_from_runner(self))
                        torch.npu.synchronize()


                        with override_forward_context(metadata.context):
                            # Refresh attn params before replay to avoid carrying
                            # stale settings from the previous split.
                            result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                            self._update_attn_params_for_split_ubatch(
                                metadata.context, current_num_tokens)
                            _append_split_metadata_debug(
                                "gr0_post_model",
                                {
                                    "slice_idx": slice_idx,
                                    "context_id": _safe_context_id(metadata.context),
                                    "num_tokens": current_num_tokens,
                                },
                            )

                    results.append(self._trim_split_output(
                        result, split_batch_slices[slice_idx].num_tokens))

                with override_forward_context(original_forward_context):
                    result = self._merge_split_outputs(results)

                if (_SPLIT_MERGE_DUMP
                        and not getattr(self, "_split_batch_dumped", False)):
                    dump_path = os.path.join(
                        os.getcwd(), "split_batch_merged_first_result_gg.json")
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(self._to_jsonable(result), f, ensure_ascii=False)
                    self._split_batch_dumped = True

                return result

            finally:
                if backup_input_ids is not None:
                    self.input_ids.gpu[:first_split_num_tokens].copy_(
                        backup_input_ids, non_blocking=False)

                if positions.ndim == 2:
                    self.positions.gpu[:, :first_split_num_tokens].copy_(
                        backup_positions, non_blocking=False)
                else:
                    self.positions.gpu[:first_split_num_tokens].copy_(
                        backup_positions, non_blocking=False)

                if backup_inputs_embeds is not None:
                    self.inputs_embeds[:first_split_num_tokens].copy_(
                        backup_inputs_embeds, non_blocking=False)
                if backup_slot_mapping is not None and base_slot_mapping is not None:
                    base_slot_mapping[:slot_mapping_backup_len].copy_(
                        backup_slot_mapping, non_blocking=False)
                torch.npu.synchronize()


    def _run_mixed_request_split_serial(
            self,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
    ) -> Any:
        ubatch_metadata = self._make_mixed_request_split_metadata_serial(
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            batch_descriptor,
        )

        results: list[Any] = []
        original_forward_context = get_forward_context()
        self._t_replay_start = time.perf_counter()
        try:
            for slice_idx, split_slice in enumerate(split_batch_slices):
                metadata = ubatch_metadata[slice_idx]
                with torch.npu.stream(self.stream_main):
                    with override_forward_context(metadata.context):
                        with disable_external_cos_sin_fast_path():
                            split_result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=
                                metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                self.stream_main.synchronize()
                results.append(
                    self._clone_split_output(
                        self._trim_split_output(split_result,
                                                split_slice.num_tokens)))

            with override_forward_context(original_forward_context):
                return self._merge_split_outputs(results)
        finally:
            self._t_replay_end = time.perf_counter()


    def _run_split_batch_inplace_serial(
            self,
            split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str,
    ) -> Any:
        ubatch_metadata = self._make_split_batch_metadata_inplace_serial(
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            batch_descriptor,
            aclgraph_runtime_mode,
            inplace_attention_backend,
        )

        results: list[Any] = []
        original_forward_context = get_forward_context()
        self._t_replay_start = time.perf_counter()
        try:
            for slice_idx, split_slice in enumerate(split_batch_slices):
                metadata = ubatch_metadata[slice_idx]
                if self._needs_inplace_serial_offset_capture(metadata):
                    split_result = self._run_inplace_serial_offset_capture(
                        metadata,
                        split_slice,
                        model_kwargs,
                    )
                else:
                    if (int(
                            getattr(metadata.context.batch_descriptor,
                                    "start_num_tokens", 0) or 0) > 0
                            and metadata.context.cudagraph_runtime_mode
                            == CUDAGraphMode.FULL
                            and not self._has_aclgraph_for_context(
                                metadata.context)):
                        raise RuntimeError(
                            "Missing inplace serial offset ACL graph before "
                            "normal replay path: "
                            f"{metadata.context.batch_descriptor!r}")
                    with torch.npu.stream(self.stream_main):
                        with override_forward_context(metadata.context):
                            split_result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                            if (metadata.context.cudagraph_runtime_mode
                                    == CUDAGraphMode.FULL):
                                if split_slice.start_num_tokens > 0:
                                    self._update_attn_params_for_split_ubatch(
                                        metadata.context,
                                        split_slice.graph_num_tokens,
                                        parallel_streams=False)
                                else:
                                    self._update_attn_params_for_wrapper(
                                        metadata.context,
                                        split_slice.graph_num_tokens)
                self.stream_main.synchronize()
                results.append(
                    self._clone_split_output(
                        self._trim_split_output(split_result,
                                                split_slice.num_tokens)))

            with override_forward_context(original_forward_context):
                return self._merge_split_outputs(results)
        finally:
            self._t_replay_end = time.perf_counter()


    def _run_split_batch_inplace_parallel_piecewise(
            self,
            split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str,
    ) -> Any:
        if aclgraph_runtime_mode != CUDAGraphMode.PIECEWISE:
            raise RuntimeError(
                "piecewise_attention_parallel requires "
                "CUDAGraphMode.PIECEWISE")
        if len(split_batch_slices) != 2:
            raise RuntimeError(
                "piecewise_attention_parallel currently supports exactly "
                f"2 splits, got {len(split_batch_slices)}")

        if inplace_attention_backend == "mixed_request":
            ubatch_metadata = self._make_mixed_request_split_metadata_parallel(
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                batch_descriptor,
            )
        else:
            ubatch_metadata = self._make_split_batch_metadata_inplace_parallel(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                batch_descriptor,
                aclgraph_runtime_mode,
                inplace_attention_backend,
            )

        original_forward_context = get_forward_context()
        runtime_calls = []
        self._t_replay_start = time.perf_counter()
        try:
            for slice_idx, metadata in enumerate(ubatch_metadata):
                target_stream = (self.stream_parallel if slice_idx > 0
                                 else self.stream_main)
                rotary_context = (
                    disable_external_cos_sin_fast_path()
                    if inplace_attention_backend == "mixed_request"
                    else nullcontext())
                with self._bind_inplace_parallel_rope_capture_slot(
                        metadata.context, parallel_streams=(slice_idx > 0)):
                    with rotary_context:
                        runtime_calls.append(
                            capture_piecewise_model_call(
                                model=self.model,
                                metadata=metadata,
                                model_kwargs=model_kwargs,
                                stream=target_stream,
                            ))

            handle = runtime_calls[0].handle
            if runtime_calls[1].handle is not handle:
                raise RuntimeError(
                    "Captured piecewise split calls use different runtime "
                    "handles")

            scheduler = InplacePiecewiseSplitScheduler(
                handle,
                stream_main=self.stream_main,
                stream_parallel=self.stream_parallel,
                debug_step_id=_split_debug_step_from_runner(self),
                sync_policy=getattr(
                    getattr(self.ascend_config, "split_batch_config", None),
                    "piecewise_scheduler_sync_policy", "event_chain"),
                attention_enqueue_policy=getattr(
                    getattr(self.ascend_config, "split_batch_config", None),
                    "piecewise_attention_enqueue_policy",
                    "persistent_thread"),
            )
            logger.info_once(
                "piecewise_attention_parallel graphs: total=%d, "
                "capturable=%d, attention=%d, sync_policy=%s, "
                "attention_enqueue_policy=%s",
                scheduler.total_pieces,
                scheduler.capturable_pieces,
                scheduler.attention_pieces,
                scheduler.sync_policy,
                scheduler.attention_enqueue_policy,
            )
            if split_debug.is_enabled():
                split_debug.log_event(
                    "piecewise_split_graph_summary",
                    {
                        "piecewise_total_graphs": scheduler.total_pieces,
                        "piecewise_capturable_graphs":
                        scheduler.capturable_pieces,
                        "piecewise_attention_graphs":
                        scheduler.attention_pieces,
                        "piecewise_scheduler_sync_policy":
                        scheduler.sync_policy,
                        "piecewise_attention_enqueue_policy":
                        scheduler.attention_enqueue_policy,
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            split_inputs = []
            for slice_idx, metadata in enumerate(ubatch_metadata):
                split_slice = split_batch_slices[slice_idx]
                split_inputs.append(
                    InplacePiecewiseSplitInput(
                        index=slice_idx,
                        context=metadata.context,
                        input_ids=metadata.input_ids,
                        positions=metadata.positions,
                        inputs_embeds=metadata.inputs_embeds,
                        intermediate_tensors=metadata.intermediate_tensors,
                        model_kwargs=model_kwargs,
                        runtime_call=runtime_calls[slice_idx],
                        num_tokens=split_slice.num_tokens,
                        graph_num_tokens=split_slice.graph_num_tokens,
                        start_num_tokens=split_slice.start_num_tokens,
                    ))

            split_outputs = scheduler.run(split_inputs[0], split_inputs[1])
            self.stream_main.synchronize()
            self.stream_parallel.synchronize()

            merged_results = []
            for idx in range(2):
                output = split_outputs[idx]
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "piecewise_split_output_copy_diag",
                        {
                            "idx": idx,
                            "phase": "pre_trim",
                            "num_tokens":
                            int(split_batch_slices[idx].num_tokens),
                            **_split_output_tensor_stats(output),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                trimmed = self._trim_split_output(
                    output, split_batch_slices[idx].num_tokens)
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "piecewise_split_output_copy_diag",
                        {
                            "idx": idx,
                            "phase": "post_trim_pre_clone",
                            "num_tokens":
                            int(split_batch_slices[idx].num_tokens),
                            **_split_output_tensor_stats(trimmed),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                cloned = self._clone_split_output(trimmed)
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "piecewise_split_output_copy_diag",
                        {
                            "idx": idx,
                            "phase": "post_clone",
                            "num_tokens":
                            int(split_batch_slices[idx].num_tokens),
                            **_split_output_tensor_stats(cloned),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                merged_results.append(cloned)
            with override_forward_context(original_forward_context):
                merged_output = self._merge_split_outputs(merged_results)
            if split_debug.is_enabled():
                split_debug.log_event(
                    "piecewise_split_output_copy_diag",
                    {
                        "idx": None,
                        "phase": "post_merge",
                        **_split_output_tensor_stats(merged_output),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            return merged_output
        finally:
            self._t_replay_end = time.perf_counter()


    def _macro_graph_config(self) -> Any:
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        if split_cfg is None:
            return None
        return getattr(split_cfg, "macro_graph_config", None)

    def _macro_graph_enabled(self) -> bool:
        macro_graph_cfg = self._macro_graph_config()
        return bool(macro_graph_cfg is not None
                    and getattr(macro_graph_cfg, "enabled", False))

    def _macro_graph_key_from_split_slices(
            self,
            split_batch_slices: SplitBatchSlices,
            inplace_attention_backend: str,
            *,
            uniform_decode: bool = True,
    ) -> tuple[Any, ...]:
        split_actual_tokens = tuple(
            int(split_slice.num_tokens) for split_slice in split_batch_slices)
        split_graph_tokens = tuple(
            int(split_slice.graph_num_tokens)
            for split_slice in split_batch_slices)
        split_num_reqs = tuple(
            int(split_slice.num_requests) for split_slice in split_batch_slices)
        split_start_tokens = tuple(
            int(split_slice.start_num_tokens)
            for split_slice in split_batch_slices)
        macro_graph_cfg = self._macro_graph_config()
        return (
            sum(split_actual_tokens),
            bool(uniform_decode),
            split_actual_tokens,
            split_graph_tokens,
            split_num_reqs,
            split_start_tokens,
            getattr(macro_graph_cfg, "schedule", ""),
            str(inplace_attention_backend),
            str(self.dtype),
            int(self.parallel_config.tensor_parallel_size),
            str(getattr(self.model_config, "model", "")),
        )

    def _macro_graph_bucket_entry_for_split_slices(
            self,
            split_batch_slices: SplitBatchSlices,
            inplace_attention_backend: str,
            *,
            uniform_decode: bool = True,
    ) -> tuple[Optional[_PlannedMacroGraphEntry],
               Optional[SplitBatchSlices]]:
        macro_graph_cfg = self._macro_graph_config()
        if not bool(getattr(macro_graph_cfg, "allow_bucket_match", False)):
            return None, None
        if len(split_batch_slices) != 2:
            return None, None

        actual_tokens = tuple(int(s.num_tokens) for s in split_batch_slices)
        runtime_graph_tokens = tuple(
            int(s.graph_num_tokens) for s in split_batch_slices)
        allow_padded_replay = bool(
            getattr(macro_graph_cfg, "allow_padded_replay", False))
        if runtime_graph_tokens != actual_tokens and not allow_padded_replay:
            return None, None
        actual_reqs = tuple(int(s.num_requests) for s in split_batch_slices)
        start_tokens = tuple(
            int(s.start_num_tokens) for s in split_batch_slices)
        min_actual_tokens_per_split = max(
            1,
            int(
                getattr(macro_graph_cfg,
                        "bucket_min_actual_tokens_per_split", 1) or 1))
        if any(actual < min_actual_tokens_per_split
               for actual in actual_tokens):
            return None, None
        max_padding_ratio = getattr(macro_graph_cfg,
                                    "max_padding_ratio_per_split", 0.0)
        padding_ratio_grace_tokens = max(
            0,
            int(
                getattr(macro_graph_cfg,
                        "bucket_padding_ratio_grace_tokens", 0) or 0))
        runtime_requested_bucket = runtime_graph_tokens != actual_tokens

        candidates: list[tuple[int, int, int, _PlannedMacroGraphEntry]] = []
        for entry in self._macro_graph_registry.values():
            if entry.inplace_attention_backend != inplace_attention_backend:
                continue
            if len(entry.plan.split_slices) != len(split_batch_slices):
                continue
            key = entry.key
            if len(key) < 11:
                continue
            if bool(key[1]) != bool(uniform_decode):
                continue
            if str(key[6]) != str(getattr(macro_graph_cfg, "schedule", "")):
                continue
            if str(key[7]) != str(inplace_attention_backend):
                continue
            if str(key[8]) != str(self.dtype):
                continue
            if int(key[9]) != int(self.parallel_config.tensor_parallel_size):
                continue
            if str(key[10]) != str(getattr(self.model_config, "model", "")):
                continue

            graph_tokens = tuple(
                int(s.graph_num_tokens) for s in entry.plan.split_slices)
            graph_reqs = tuple(
                int(s.num_requests) for s in entry.plan.split_slices)
            graph_req_caps = tuple(
                int(v) for v in getattr(entry, "split_req_caps", None)
            ) if getattr(entry, "split_req_caps", None) is not None else graph_reqs
            graph_starts = tuple(
                int(s.start_num_tokens) for s in entry.plan.split_slices)
            if runtime_requested_bucket and graph_tokens != runtime_graph_tokens:
                continue
            if graph_starts != start_tokens:
                continue
            if graph_req_caps != tuple(
                    int(getattr(s, "request_capacity", s.num_requests))
                    for s in entry.plan.split_slices):
                graph_req_caps = tuple(
                    max(int(cap),
                        int(getattr(s, "request_capacity", s.num_requests)))
                    for cap, s in zip(graph_req_caps,
                                      entry.plan.split_slices))
            if any(actual > graph for actual, graph in zip(
                    actual_tokens, graph_tokens)):
                continue
            if any(actual > graph for actual, graph in zip(
                    actual_reqs, graph_req_caps)):
                continue

            padding_tokens = tuple(graph - actual for actual, graph in zip(
                actual_tokens, graph_tokens))
            runtime_max_query_lens = tuple(
                max(1,
                    int(getattr(s, "max_query_len", 0) or 0))
                for s in split_batch_slices)
            padding_req_chunks = tuple(
                self._compact_padding_req_chunks(padding, max_query_len)
                for padding, max_query_len in zip(padding_tokens,
                                                  runtime_max_query_lens))
            effective_reqs = tuple(
                actual + chunks
                for actual, chunks in zip(actual_reqs, padding_req_chunks))
            if any(effective > graph for effective, graph in zip(
                    effective_reqs, graph_req_caps)):
                continue
            if any(padding > 0 for padding in padding_tokens
                   ) and not allow_padded_replay:
                continue
            if max_padding_ratio is not None:
                ratio = float(max_padding_ratio)
                if any(
                    (actual > padding_ratio_grace_tokens
                     and (padding / float(max(1, actual))) > ratio)
                        for padding, actual in zip(padding_tokens,
                                                   actual_tokens)):
                    continue

            graph_mismatch = sum(
                abs(graph - runtime_graph)
                for graph, runtime_graph in zip(graph_tokens,
                                                runtime_graph_tokens))
            candidates.append(
                (graph_mismatch, sum(padding_tokens),
                 sum(graph_req_caps) - sum(effective_reqs), entry))

        if not candidates:
            return None, None

        _, _, _, entry = min(candidates,
                             key=lambda item: (item[0], item[1], item[2]))
        padded_slices: SplitBatchSlices = []
        for runtime_slice, graph_slice in zip(split_batch_slices,
                                              entry.plan.split_slices):
            padded_slices.append(
                SplitBatchSlice(
                    request_slice=runtime_slice.request_slice,
                    token_slice=runtime_slice.token_slice,
                    padded_num_tokens=int(graph_slice.graph_num_tokens),
                    start_num_tokens=int(graph_slice.start_num_tokens),
                    request_capacity=int(
                        getattr(graph_slice, "request_capacity",
                                graph_slice.num_requests)),
                    max_query_len=int(
                        getattr(runtime_slice, "max_query_len", 0)
                        or getattr(graph_slice, "max_query_len", 0) or 0),
                ))
        return entry, padded_slices

    def _macro_graph_capture_inplace_plans(
            self, macro_graph_cfg: Any) -> list[InplaceSplitPlan]:
        q = int(self.uniform_decode_query_len)
        if q <= 0:
            raise RuntimeError(
                "macro graph capture requires a positive "
                f"uniform_decode_query_len, got {q}")

        inplace_plans: list[InplaceSplitPlan] = []
        if getattr(macro_graph_cfg, "plan_source", "explicit") != "planner":
            capture_plans = list(getattr(macro_graph_cfg, "capture_plans", []))
            capture_plans_considered = [
                int(plan.total_tokens) for plan in capture_plans
            ]
            for capture_plan in capture_plans:
                plan, reason = _macro_capture_plan_to_inplace_plan(
                    capture_plan,
                    uniform_decode_query_len=q,
                    capture_plans_considered=capture_plans_considered,
                    macro_graph_config=macro_graph_cfg,
                    allow_mixed_request_plan=True,
                )
                if plan is None:
                    raise RuntimeError(
                        "Failed to build explicit macro graph capture plan "
                        f"for total_tokens={int(capture_plan.total_tokens)}: "
                        f"{reason}")
                inplace_plans.append(plan)
            return inplace_plans

        total_tokens_list = list(
            getattr(macro_graph_cfg, "capture_total_tokens", []))
        for total_tokens in total_tokens_list:
            total_tokens = int(total_tokens)
            if total_tokens % q != 0:
                raise RuntimeError(
                    "macro graph capture plan must be request-aligned: "
                    f"total_tokens={total_tokens}, query_len={q}")
            dummy_tokens = np.full(total_tokens // q, q, dtype=np.int32)
            plan, reason = create_macro_inplace_split_batch_slices(
                dummy_tokens,
                total_tokens,
                q,
                macro_graph_cfg,
            )
            if plan is None:
                raise RuntimeError(
                    "Failed to build macro graph capture plan "
                    f"for total_tokens={total_tokens}: {reason}")
            inplace_plans.append(plan)
        return inplace_plans

    def _build_macro_graph_binding_plan(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata]) -> MacroGraphBindingPlan:
        if entry.captured_metadata is None:
            raise RuntimeError("macro graph entry has no captured metadata")
        if len(entry.captured_metadata) != len(ubatch_metadata):
            raise RuntimeError(
                "macro graph binding split count mismatch: "
                f"captured={len(entry.captured_metadata)}, "
                f"runtime={len(ubatch_metadata)}")

        split_plans: list[MacroGraphSplitBindingPlan] = []
        for split_idx, (captured, current) in enumerate(
                zip(entry.captured_metadata, ubatch_metadata)):
            tensor_bindings: list[MacroGraphTensorBinding] = []
            scalar_bindings: list[MacroGraphScalarBinding] = []
            split_prefix = f"split{split_idx}"
            root_path: tuple[Any, ...] = ()

            for name, dst_tensor, src_path, required in (
                ("input_ids", captured.input_ids, ("input_ids", ),
                 captured.input_ids is not None),
                ("positions", captured.positions, ("positions", ),
                 captured.positions is not None),
                ("inputs_embeds", captured.inputs_embeds, ("inputs_embeds", ),
                 captured.inputs_embeds is not None),
            ):
                if not isinstance(dst_tensor, torch.Tensor):
                    continue
                src_tensor = _macro_graph_path_get(current, src_path)
                if src_tensor is None:
                    if required:
                        raise RuntimeError(
                            "missing required macro graph input binding "
                            f"source: {split_prefix}.{name}")
                    continue
                if not isinstance(src_tensor, torch.Tensor):
                    raise RuntimeError(
                        "macro graph input binding source is not a tensor: "
                        f"{split_prefix}.{name}, "
                        f"type={type(src_tensor).__name__}")
                copy_policy, copy_dim, zero_tail = (
                    _macro_graph_tensor_binding_policy(name, dst_tensor))
                tensor_bindings.append(
                    MacroGraphTensorBinding(
                        name=f"{split_prefix}.{name}",
                        dst=dst_tensor,
                        src_path=src_path,
                        max_shape=tuple(int(size) for size in dst_tensor.shape),
                        copy_policy=copy_policy,
                        copy_dim=copy_dim,
                        zero_tail=zero_tail,
                        required=required,
                    ))

            tensor_bindings.extend(
                _macro_graph_build_intermediate_tensor_bindings(
                    captured.intermediate_tensors,
                    current.intermediate_tensors,
                    root_path=("intermediate_tensors", ),
                    name_prefix=f"{split_prefix}.intermediate_tensors",
                ))

            captured_ctx = captured.context
            current_ctx = current.context
            captured_attn_metadata = getattr(captured_ctx, "attn_metadata",
                                             None)
            current_attn_metadata = getattr(current_ctx, "attn_metadata", None)
            for metadata_path, captured_meta in _macro_graph_iter_metadata_leaves(
                    captured_attn_metadata, ("context", "attn_metadata")):
                current_meta = _macro_graph_path_get(
                    current, metadata_path)
                tensor_bindings.extend(
                    _macro_graph_build_tensor_bindings_for_object(
                        captured_meta,
                        current_meta,
                        root_path=metadata_path,
                        name_prefix=".".join(str(part)
                                             for part in metadata_path),
                    ))
                scalar_bindings.extend(
                    _macro_graph_build_scalar_bindings_for_object(
                        captured_meta,
                        current_meta,
                        dst_parent_path=metadata_path,
                        src_parent_path=metadata_path,
                        name_prefix=".".join(str(part)
                                             for part in metadata_path),
                    ))

            context_positions = getattr(captured_ctx, "positions", None)
            if isinstance(context_positions, torch.Tensor):
                current_context_positions = getattr(current_ctx, "positions",
                                                    None)
                if not isinstance(current_context_positions, torch.Tensor):
                    raise RuntimeError(
                        "missing required macro graph context.positions "
                        f"source for {split_prefix}")
                tensor_bindings.append(
                    MacroGraphTensorBinding(
                        name=f"{split_prefix}.context.positions",
                        dst=context_positions,
                        src_path=("context", "positions"),
                        max_shape=tuple(
                            int(size) for size in context_positions.shape),
                        copy_policy="token_prefix",
                        copy_dim=1 if context_positions.ndim == 2 else 0,
                        zero_tail=True,
                        required=True,
                    ))

            split_plans.append(
                MacroGraphSplitBindingPlan(
                    tensor_bindings=tensor_bindings,
                    scalar_bindings=scalar_bindings,
                ))
        return MacroGraphBindingPlan(split_bindings=split_plans)

    def _macro_graph_binding_copy_len(
            self,
            binding: MacroGraphTensorBinding,
            current: AscendUbatchMetadata,
            src_tensor: torch.Tensor) -> Optional[int]:
        policy = binding.copy_policy
        is_attn_metadata_binding = (
            len(binding.src_path) >= 2
            and binding.src_path[0] == "context"
            and binding.src_path[1] == "attn_metadata")
        if is_attn_metadata_binding and policy in ("token_prefix",
                                                   "request_prefix",
                                                   "actual_prefix"):
            return int(src_tensor.shape[int(binding.copy_dim)])
        if policy == "token_prefix":
            token_len = int(
                getattr(current.context, "split_actual_num_tokens", 0) or 0)
            if token_len <= 0:
                token_len = int(getattr(current, "num_tokens", 0) or 0)
            if token_len <= 0:
                token_len = int(src_tensor.shape[int(binding.copy_dim)])
            return min(token_len, int(src_tensor.shape[int(binding.copy_dim)]))
        if policy == "request_prefix":
            req_len = int(
                getattr(current.context, "split_actual_num_reqs", 0) or 0)
            if req_len <= 0:
                meta = _macro_graph_path_get(
                    current,
                    binding.src_path[:-1],
                )
                req_len = int(getattr(meta, "num_reqs", 0) or 0)
            if binding.name.endswith(".query_start_loc") and req_len > 0:
                req_len += 1
            if req_len <= 0:
                req_len = int(src_tensor.shape[int(binding.copy_dim)])
            return min(req_len, int(src_tensor.shape[int(binding.copy_dim)]))
        if policy == "actual_prefix":
            actual_len = int(
                getattr(current.context, "split_actual_num_tokens", 0) or 0)
            if actual_len <= 0:
                actual_len = int(src_tensor.shape[int(binding.copy_dim)])
            return min(actual_len, int(src_tensor.shape[int(binding.copy_dim)]))
        return None

    def _bind_macro_graph_entry_recursive(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata],
            *,
            collect_detail: bool = False) -> tuple[int, Optional[dict[str, Any]]]:
        if entry.captured_metadata is None:
            return 0, _new_macro_graph_bind_detail() if collect_detail else None
        copied = 0
        detail = _new_macro_graph_bind_detail() if collect_detail else None
        for split_idx, (captured, current) in enumerate(
                zip(entry.captured_metadata, ubatch_metadata)):
            split_prefix = f"split{split_idx}"
            copied += _copy_tensor_values_for_macro_graph(
                captured.input_ids,
                current.input_ids,
                detail=detail,
                category=f"{split_prefix}.input_ids")
            copied += _copy_tensor_values_for_macro_graph(
                captured.positions,
                current.positions,
                detail=detail,
                category=f"{split_prefix}.positions")
            copied += _copy_tensor_values_for_macro_graph(
                captured.inputs_embeds,
                current.inputs_embeds,
                detail=detail,
                category=f"{split_prefix}.inputs_embeds")
            copied += _copy_tensor_values_for_macro_graph(
                captured.intermediate_tensors,
                current.intermediate_tensors,
                detail=detail,
                category=f"{split_prefix}.intermediate_tensors")
            copied += _copy_tensor_values_for_macro_graph(
                getattr(captured.context, "attn_metadata", None),
                getattr(current.context, "attn_metadata", None),
                detail=detail,
                category=f"{split_prefix}.attn_metadata.tensor",
            )
            copied += _copy_tensor_values_for_macro_graph(
                getattr(captured.context, "positions", None),
                getattr(current.context, "positions", None),
                detail=detail,
                category=f"{split_prefix}.context.positions",
            )
            if entry.inplace_attention_backend == "mixed_request":
                copied += _copy_mixed_request_macro_metadata_values(
                    getattr(captured.context, "attn_metadata", None),
                    getattr(current.context, "attn_metadata", None),
                    detail=detail,
                    category=f"{split_prefix}.attn_metadata.scalar/list",
                )
        return copied, detail

    def _bind_macro_graph_entry_allowlist(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata],
            *,
            collect_detail: bool = False) -> tuple[int, Optional[dict[str, Any]]]:
        if entry.binding_plan is None:
            entry.binding_plan = self._build_macro_graph_binding_plan(
                entry, ubatch_metadata)
        plan = entry.binding_plan
        if len(plan.split_bindings) != len(ubatch_metadata):
            raise RuntimeError(
                "macro graph binding plan split count mismatch: "
                f"plan={len(plan.split_bindings)}, "
                f"runtime={len(ubatch_metadata)}")

        copied = 0
        detail = _new_macro_graph_bind_detail() if collect_detail else None
        if detail is not None:
            detail["binding_mode"] = "allowlist"
            detail["binding_plan_tensor_count"] = int(
                sum(len(split.tensor_bindings)
                    for split in plan.split_bindings))
            detail["binding_plan_scalar_count"] = int(
                sum(len(split.scalar_bindings)
                    for split in plan.split_bindings))
            detail["binding_plan_tensor_names"] = [
                binding.name
                for split in plan.split_bindings
                for binding in split.tensor_bindings
            ][:128]
            detail["binding_plan_tensor_policies"] = [
                {
                    "name": binding.name,
                    "copy_policy": binding.copy_policy,
                    "copy_dim": int(binding.copy_dim),
                    "zero_tail": bool(binding.zero_tail),
                }
                for split in plan.split_bindings
                for binding in split.tensor_bindings
            ][:128]
            detail["binding_plan_scalar_names"] = [
                binding.name
                for split in plan.split_bindings
                for binding in split.scalar_bindings
            ][:128]

        for split_idx, (split_plan, current) in enumerate(
                zip(plan.split_bindings, ubatch_metadata)):
            for binding in split_plan.tensor_bindings:
                src_tensor = _macro_graph_path_get(current, binding.src_path)
                if src_tensor is None:
                    if binding.required:
                        raise RuntimeError(
                            "missing required macro graph tensor binding "
                            f"source: {binding.name}")
                    continue
                if not isinstance(src_tensor, torch.Tensor):
                    raise RuntimeError(
                        "macro graph tensor binding source is not a tensor: "
                        f"{binding.name}, type={type(src_tensor).__name__}")
                if _macro_graph_copy_tensor_value(
                        binding.dst,
                        src_tensor,
                        binding_name=binding.name,
                        copy_policy=binding.copy_policy,
                        copy_dim=binding.copy_dim,
                        copy_len=self._macro_graph_binding_copy_len(
                            binding, current, src_tensor),
                        zero_tail=binding.zero_tail,
                        detail=detail,
                        category=binding.name):
                    copied += 1

            captured = entry.captured_metadata[split_idx]
            for binding in split_plan.scalar_bindings:
                dst_parent = _macro_graph_path_get(captured,
                                                   binding.dst_parent_path)
                src_parent = _macro_graph_path_get(current,
                                                   binding.src_parent_path)
                if dst_parent is None or src_parent is None:
                    if binding.required:
                        raise RuntimeError(
                            "missing required macro graph scalar binding "
                            f"parent: {binding.name}")
                    continue
                if not hasattr(src_parent, binding.attr):
                    if binding.required:
                        raise RuntimeError(
                            "missing required macro graph scalar binding "
                            f"source: {binding.name}")
                    continue
                value = getattr(src_parent, binding.attr)
                if isinstance(value, list):
                    value = list(value)
                old_value = getattr(dst_parent, binding.attr, None)
                if _macro_graph_values_equal(old_value, value):
                    _macro_graph_bind_incr(
                        detail, binding.name,
                        "metadata_attr_skip_same_value_count")
                    continue
                setattr(dst_parent, binding.attr, value)
                copied += 1
                _macro_graph_bind_incr(detail, binding.name,
                                       "metadata_attr_update_count")
                _macro_graph_bind_record_attr(detail, binding.name,
                                              binding.attr)
        return copied, detail

    def _bind_macro_graph_entry(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata],
            *,
            collect_detail: bool = False) -> tuple[int, Optional[dict[str, Any]]]:
        try:
            return self._bind_macro_graph_entry_allowlist(
                entry,
                ubatch_metadata,
                collect_detail=collect_detail,
            )
        except Exception as exc:
            if _MACRO_GRAPH_BIND_ALLOWLIST_STRICT:
                raise
            entry.binding_plan = None
            entry.binding_fallback_count += 1
            if entry.binding_fallback_count <= 3:
                logger.warning(
                    "Macro graph allowlist bind failed; falling back to "
                    "recursive bind. count=%d key=%s error=%s: %s",
                    entry.binding_fallback_count,
                    entry.key,
                    type(exc).__name__,
                    exc,
                )
            copied, detail = self._bind_macro_graph_entry_recursive(
                entry,
                ubatch_metadata,
                collect_detail=collect_detail,
            )
            if detail is not None:
                detail["binding_mode"] = "recursive_fallback"
                detail["binding_fallback_reason"] = str(exc)
                detail["binding_fallback_type"] = type(exc).__name__
            return copied, detail

    def _prepare_mixed_request_macro_contexts(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata],
            *,
            opaque_attention_update: bool = False) -> None:
        external_attention_update = bool(
            _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE
            and not opaque_attention_update)
        external_attention_update_mode = (
            _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE
            if external_attention_update else "off")
        entry.macro_attention_external_update = external_attention_update
        for idx, metadata in enumerate(ubatch_metadata):
            context = metadata.context
            descriptor = getattr(context, "batch_descriptor", None)
            has_lora = bool(getattr(descriptor, "has_lora", False))
            split_slice = entry.plan.split_slices[int(idx)]
            num_reqs = int(
                getattr(split_slice, "request_capacity", 0)
                or getattr(split_slice, "num_requests", 0) or 0)
            if num_reqs <= 0:
                first_meta = _first_macro_attn_metadata(
                    getattr(context, "attn_metadata", None))
                num_reqs = int(getattr(first_meta, "num_decodes", 0) or 0)
                num_reqs += int(getattr(first_meta, "num_prefills", 0) or 0)
                if num_reqs <= 0:
                    num_reqs = getattr(descriptor, "num_reqs", None)
            graph_tokens = int(getattr(context, "split_graph_num_tokens",
                                       getattr(metadata, "num_tokens", 0))
                               or getattr(metadata, "num_tokens", 0))
            graph_variant = str(
                getattr(context, "split_inplace_mode",
                        "mixed_request_piecewise_attention_parallel"))
            context.batch_descriptor = BatchDescriptor(
                num_tokens=graph_tokens,
                num_reqs=num_reqs,
                uniform=False,
                has_lora=has_lora,
                start_num_tokens=0,
                graph_variant=graph_variant,
                attention_backend="mixed_request",
                capture_metadata_mode="mixed_request_compact",
            )
            context.capturing = True
            context.macro_graph_opaque_attention_update = bool(
                opaque_attention_update)
            context.macro_graph_external_attention_update = bool(
                external_attention_update)
            context.macro_graph_external_attention_update_mode = (
                external_attention_update_mode)

    def _use_npugraph_ex_macro_opaque_attention_update(
            self, backend: str, inplace_attention_backend: str) -> bool:
        return bool(
            _MACRO_GRAPH_NPUGRAPH_EX_OPAQUE_ATTENTION_UPDATE
            and backend == "npugraph_ex"
            and inplace_attention_backend == "mixed_request"
            and _MACRO_GRAPH_REWRITE_ATTENTION_OPS)

    def _ensure_npugraph_ex_macro_opaque_attention_update_env(self) -> None:
        os.environ["TORCH_NPU_NPUGRAPH_EX_ENABLE_MACRO_OPAQUE_UPDATE"] = "1"
        os.environ.setdefault(
            "TORCH_NPU_NPUGRAPH_EX_MACRO_OPAQUE_UPDATE_BEFORE_REPLAY", "0")

    def _snapshot_mixed_request_macro_graph_params(
            self, ubatch_metadata: list[AscendUbatchMetadata]
    ) -> list[tuple[Any, Any, int, int, int]]:
        snapshots: list[tuple[Any, Any, int, int, int]] = []
        for metadata in ubatch_metadata:
            context = metadata.context
            in_parallel_streams = bool(
                getattr(context, "in_parallel_streams", False))
            graph_params = get_graph_params(in_parallel_streams)
            if graph_params is None:
                continue
            runtime_shape = _macro_context_runtime_shape(context)
            param_key = get_graph_param_key(context, runtime_shape)
            ensure_graph_param_key(graph_params, param_key)
            snapshots.append((
                graph_params,
                param_key,
                len(graph_params.events[param_key]),
                len(graph_params.handles[param_key]),
                len(graph_params.attn_params[param_key]),
            ))
        return snapshots

    def _restore_mixed_request_macro_graph_param_snapshots(
            self, snapshots: list[tuple[Any, Any, int, int, int]]) -> None:
        for graph_params, param_key, events_len, handles_len, params_len in snapshots:
            del graph_params.events[param_key][events_len:]
            del graph_params.handles[param_key][handles_len:]
            del graph_params.attn_params[param_key][params_len:]

    def _begin_macro_attention_tensor_retention(
            self, entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata]) -> None:
        entry.macro_attention_tensor_retention = []
        entry.macro_attention_tensor_retention_seen = set()
        entry._macro_attention_retention_context_state = []
        if entry.inplace_attention_backend != "mixed_request":
            return
        for metadata in ubatch_metadata:
            context = metadata.context
            had_retention = hasattr(
                context, "macro_graph_attention_tensor_retention")
            old_retention = getattr(
                context, "macro_graph_attention_tensor_retention", None)
            had_seen = hasattr(context,
                               "macro_graph_attention_tensor_retention_seen")
            old_seen = getattr(
                context, "macro_graph_attention_tensor_retention_seen", None)
            entry._macro_attention_retention_context_state.append((
                context,
                had_retention,
                old_retention,
                had_seen,
                old_seen,
            ))
            context.macro_graph_attention_tensor_retention = (
                entry.macro_attention_tensor_retention)
            context.macro_graph_attention_tensor_retention_seen = (
                entry.macro_attention_tensor_retention_seen)

    def _end_macro_attention_tensor_retention(
            self, entry: _PlannedMacroGraphEntry) -> None:
        for (context, had_retention, old_retention, had_seen,
             old_seen) in entry._macro_attention_retention_context_state:
            if had_retention:
                context.macro_graph_attention_tensor_retention = old_retention
            elif hasattr(context, "macro_graph_attention_tensor_retention"):
                delattr(context, "macro_graph_attention_tensor_retention")
            if had_seen:
                context.macro_graph_attention_tensor_retention_seen = old_seen
            elif hasattr(context, "macro_graph_attention_tensor_retention_seen"):
                delattr(context, "macro_graph_attention_tensor_retention_seen")
        entry._macro_attention_retention_context_state = []

    def _macro_attention_retention_debug(
            self, entry: _PlannedMacroGraphEntry) -> dict[str, Any]:
        retained = getattr(entry, "macro_attention_tensor_retention", [])
        sample = []
        for tensor in retained[:8]:
            info = split_debug.tensor_info(tensor)
            if info is not None:
                sample.append(info)
        return {
            "retained_tensor_count": int(len(retained)),
            "retained_tensor_sample": sample,
        }

    @contextmanager
    def _macro_graph_runtime_attention_update_metadata(
            self, context: Any,
            runtime_metadata: Optional[AscendUbatchMetadata]):
        if runtime_metadata is None:
            yield "captured_slot"
            return
        runtime_context = runtime_metadata.context
        runtime_attn_metadata = getattr(runtime_context, "attn_metadata", None)
        runtime_dual_metadata = getattr(runtime_context,
                                        "dual_stream_attention_metadata", None)
        state: list[tuple[str, bool, Any]] = []
        for attr, value in (
            ("macro_graph_attention_update_metadata", runtime_attn_metadata),
            ("macro_graph_dual_attention_update_metadata",
             runtime_dual_metadata),
        ):
            state.append((attr, hasattr(context, attr), getattr(context, attr,
                                                                None)))
            if value is None:
                if hasattr(context, attr):
                    delattr(context, attr)
            else:
                setattr(context, attr, value)
        try:
            yield "runtime_metadata"
        finally:
            for attr, had_attr, old_value in state:
                if had_attr:
                    setattr(context, attr, old_value)
                elif hasattr(context, attr):
                    delattr(context, attr)

    def _mixed_request_macro_graph_param_details(
            self, ubatch_metadata: list[AscendUbatchMetadata]
    ) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for idx, metadata in enumerate(ubatch_metadata):
            context = metadata.context
            parallel_streams = bool(
                getattr(context, "in_parallel_streams", False))
            graph_params = get_graph_params(parallel_streams)
            runtime_shape = _macro_context_runtime_shape(context)
            param_key = get_graph_param_key(context, runtime_shape)
            registered = False
            attn_param_count = 0
            handle_count = 0
            event_count = 0
            workspace_present = False
            detail: dict[str, Any] = {
                "split_idx": int(idx),
                "runtime_shape": int(runtime_shape),
                "in_parallel_streams": bool(parallel_streams),
                "graph_param_key": graph_param_key_info(param_key),
                "graph_params_available": graph_params is not None,
            }
            if graph_params is not None:
                registered = param_key in graph_params.attn_params
                attn_param_count = len(
                    graph_params.attn_params.get(param_key, []))
                handle_count = len(graph_params.handles.get(param_key, []))
                event_count = len(graph_params.events.get(param_key, []))
                workspace_present = graph_params.workspaces.get(
                    param_key) is not None
                detail.update({
                    "graph_param_registered": bool(registered),
                    "attn_param_count": int(attn_param_count),
                    "handle_count": int(handle_count),
                    "event_count": int(event_count),
                    "workspace_present": bool(workspace_present),
                    "task_handles_ready": bool(attn_param_count > 0
                                               and handle_count > 0
                                               and event_count > 0),
                })
                if registered:
                    detail.update({
                        "handle_id_sample":
                        _macro_graph_object_id_sample(
                            graph_params.handles.get(param_key, [])),
                        "event_id_sample":
                        _macro_graph_object_id_sample(
                            graph_params.events.get(param_key, [])),
                    })
                if registered and not detail["task_handles_ready"]:
                    detail["handle_status"] = (
                        "workspace_only"
                        if workspace_present else "metadata_only")
                elif detail["task_handles_ready"]:
                    detail["handle_status"] = "ready"
                else:
                    detail["handle_status"] = "missing_key"
            details.append(detail)
        return details

    def _mixed_request_macro_graph_param_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for name, graph_params in (("main", get_graph_params(False)),
                                   ("parallel", get_graph_params(True))):
            if graph_params is None:
                summary.append({"pool": name, "available": False})
                continue
            keys = set(graph_params.attn_params.keys())
            keys.update(graph_params.handles.keys())
            keys.update(graph_params.events.keys())
            for key in keys:
                summary.append({
                    "pool": name,
                    "available": True,
                    "graph_param_key": graph_param_key_info(key),
                    "key_repr": _macro_graph_param_key_repr(key),
                    "attn_param_count":
                    len(graph_params.attn_params.get(key, [])),
                    "handle_count": len(graph_params.handles.get(key, [])),
                    "event_count": len(graph_params.events.get(key, [])),
                    "workspace_present":
                    graph_params.workspaces.get(key) is not None,
                    "task_handles_ready": bool(
                        len(graph_params.attn_params.get(key, [])) > 0
                        and len(graph_params.handles.get(key, [])) > 0
                        and len(graph_params.events.get(key, [])) > 0),
                    "handle_id_sample":
                    _macro_graph_object_id_sample(
                        graph_params.handles.get(key, [])),
                    "event_id_sample":
                    _macro_graph_object_id_sample(
                        graph_params.events.get(key, [])),
                })
        summary.sort(
            key=lambda item:
            (str(item.get("pool", "")), str(item.get("key_repr", ""))))
        return summary

    def _prewarm_mixed_request_macro_fia_workspaces(
            self,
            module: nn.Module,
            ubatch_metadata: list[AscendUbatchMetadata],
    ) -> None:
        context_states: list[tuple[Any, bool, Any, bool, Any]] = []
        for metadata in ubatch_metadata:
            context = metadata.context
            had_capturing = hasattr(context, "capturing")
            old_capturing = getattr(context, "capturing", None)
            had_warmup = hasattr(context, "macro_fia_workspace_warmup")
            old_warmup = getattr(context, "macro_fia_workspace_warmup", None)
            context_states.append((
                context,
                had_capturing,
                old_capturing,
                had_warmup,
                old_warmup,
            ))
            context.capturing = False
            context.macro_fia_workspace_warmup = True

        snapshots = self._snapshot_mixed_request_macro_graph_params(
            ubatch_metadata)
        try:
            with torch.inference_mode():
                module()
            torch.npu.synchronize()
        finally:
            for (context, had_capturing, old_capturing, had_warmup,
                 old_warmup) in context_states:
                if had_capturing:
                    context.capturing = old_capturing
                elif hasattr(context, "capturing"):
                    delattr(context, "capturing")
                if had_warmup:
                    context.macro_fia_workspace_warmup = old_warmup
                elif hasattr(context, "macro_fia_workspace_warmup"):
                    delattr(context, "macro_fia_workspace_warmup")
            self._restore_mixed_request_macro_graph_param_snapshots(snapshots)

    def _mixed_request_macro_attn_metadata_for_key(self, attn_metadata: Any,
                                                   key: Any) -> Any:
        if isinstance(attn_metadata, dict):
            return attn_metadata[key]
        if isinstance(attn_metadata, list):
            return attn_metadata[int(key)]
        return getattr(attn_metadata, key)

    def _build_mixed_request_macro_paired_fia_update_plan(
            self, entry: _PlannedMacroGraphEntry) -> tuple[
                Optional[list[dict[str, Any]]], dict[str, Any]]:
        if not entry.macro_attention_external_update:
            return None, {
                "applied": False,
                "reason": "external_update_disabled",
                "external_update_mode":
                _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
            }
        if (_MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE != "event_only"
                and not hasattr(torch.npu, "graph_task_update_begin")):
            return None, {
                "applied": False,
                "reason": "missing_graph_task_update_api",
                "external_update_mode":
                _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
            }
        if entry.captured_metadata is None or len(entry.captured_metadata) != 2:
            return None, {
                "applied": False,
                "reason": "requires_two_captured_splits",
                "captured_split_count":
                None if entry.captured_metadata is None else
                int(len(entry.captured_metadata)),
            }

        split_records: list[dict[str, Any]] = []
        for split_idx, metadata in enumerate(entry.captured_metadata):
            context = metadata.context
            in_parallel_streams = bool(
                getattr(context, "in_parallel_streams", False))
            graph_params = get_graph_params(in_parallel_streams)
            runtime_shape = _macro_context_runtime_shape(context)
            param_key = get_graph_param_key(context, runtime_shape)
            if graph_params is None:
                return None, {
                    "applied": False,
                    "reason": "missing_graph_params",
                    "split_idx": int(split_idx),
                    "in_parallel_streams": bool(in_parallel_streams),
                }
            params = graph_params.attn_params.get(param_key, [])
            handles = graph_params.handles.get(param_key, [])
            events = graph_params.events.get(param_key, [])
            attn_metadata = getattr(context, "attn_metadata", None)
            if not isinstance(attn_metadata, dict):
                return None, {
                    "applied": False,
                    "reason": "unsupported_attn_metadata_type",
                    "split_idx": int(split_idx),
                    "attn_metadata_type": type(attn_metadata).__name__,
                }
            if not params or not handles or not events:
                return None, {
                    "applied": False,
                    "reason": "missing_graph_task_handles",
                    "split_idx": int(split_idx),
                    "graph_param_key": graph_param_key_info(param_key),
                    "attn_param_count": int(len(params)),
                    "handle_count": int(len(handles)),
                    "event_count": int(len(events)),
                }
            if len(params) != len(handles) or len(params) != len(events):
                return None, {
                    "applied": False,
                    "reason": "graph_task_record_count_mismatch",
                    "split_idx": int(split_idx),
                    "graph_param_key": graph_param_key_info(param_key),
                    "attn_param_count": int(len(params)),
                    "handle_count": int(len(handles)),
                    "event_count": int(len(events)),
                }
            split_records.append({
                "split_idx": int(split_idx),
                "context": context,
                "graph_params": graph_params,
                "param_key": param_key,
                "runtime_shape": int(runtime_shape),
                "in_parallel_streams": bool(in_parallel_streams),
                "attn_items": list(
                    zip(list(attn_metadata.keys()), params, handles, events)),
            })

        if len(split_records[0]["attn_items"]) != len(
                split_records[1]["attn_items"]):
            return None, {
                "applied": False,
                "reason": "split_layer_count_mismatch",
                "split0_layers": int(len(split_records[0]["attn_items"])),
                "split1_layers": int(len(split_records[1]["attn_items"])),
            }
        split0_keys = [item[0] for item in split_records[0]["attn_items"]]
        split1_keys = [item[0] for item in split_records[1]["attn_items"]]
        if split0_keys != split1_keys:
            return None, {
                "applied": False,
                "reason": "split_layer_key_order_mismatch",
                "split0_sample": [str(k) for k in split0_keys[:4]],
                "split1_sample": [str(k) for k in split1_keys[:4]],
            }

        layer_plan: list[dict[str, Any]] = []
        for layer_idx, key in enumerate(split0_keys):
            split_entries = []
            for split_record in split_records:
                _key, param, handle, event = split_record["attn_items"][
                    layer_idx]
                if not isinstance(param, tuple) or len(param) != 13:
                    return None, {
                        "applied": False,
                        "reason": "unsupported_graph_param_layout",
                        "split_idx": int(split_record["split_idx"]),
                        "layer_idx": int(layer_idx),
                        "layer_key": str(key),
                        "param_type": type(param).__name__,
                        "param_len":
                        None if not isinstance(param, tuple) else int(len(param)),
                    }
                split_entries.append({
                    "split_idx": int(split_record["split_idx"]),
                    "context": split_record["context"],
                    "graph_params": split_record["graph_params"],
                    "param_key": split_record["param_key"],
                    "runtime_shape": int(split_record["runtime_shape"]),
                    "in_parallel_streams":
                    bool(split_record["in_parallel_streams"]),
                    "param": param,
                    "handle": handle,
                    "event": event,
                })
            layer_plan.append({
                "layer_idx": int(layer_idx),
                "key": key,
                "splits": split_entries,
            })

        return layer_plan, {
            "applied": True,
            "mode": "paired_single_update_stream",
            "external_update_mode": _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
            "layer_count": int(len(layer_plan)),
            "split0_graph_param_key":
            graph_param_key_info(split_records[0]["param_key"]),
            "split1_graph_param_key":
            graph_param_key_info(split_records[1]["param_key"]),
        }

    def _runtime_mixed_request_macro_attn_metadata(
            self, captured_context: Any,
            runtime_metadata: Optional[AscendUbatchMetadata],
            key: Any) -> Any:
        if runtime_metadata is not None:
            runtime_attn_metadata = getattr(runtime_metadata.context,
                                            "attn_metadata", None)
            return self._mixed_request_macro_attn_metadata_for_key(
                runtime_attn_metadata, key)
        return self._mixed_request_macro_attn_metadata_for_key(
            getattr(captured_context, "attn_metadata", None), key)

    def _update_one_mixed_request_macro_fia_task(
            self,
            *,
            update_stream: torch.npu.Stream,
            layer_idx: int,
            layer_key: Any,
            split_entry: dict[str, Any],
            runtime_metadata: Optional[AscendUbatchMetadata],
    ) -> dict[str, Any]:
        (query, key_cache, value, block_tables, attn_mask, block_size, seq_lens,
         query_start_loc, num_kv_heads, num_heads, scale, attn_output,
         softmax_lse) = split_entry["param"]
        context = split_entry["context"]
        captured_metadata = self._mixed_request_macro_attn_metadata_for_key(
            getattr(context, "attn_metadata", None), layer_key)
        use_captured_params = bool(
            _ACL_GRAPH_FIA_UPDATE_USE_CAPTURED_PARAMS
            and getattr(getattr(context, "batch_descriptor", None),
                        "capture_metadata_mode", "") == "mixed_request_compact")
        external_update_mode = str(
            getattr(context, "macro_graph_external_attention_update_mode",
                    _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE))
        event_only = bool(external_update_mode == "event_only")
        if event_only:
            runtime_attn_metadata = captured_metadata
            actual_seq_lengths_q = query_start_loc
            metadata_block_table = block_tables
            metadata_block_source = "event_only_captured_graph_param"
            block_table_refreshed = False
        elif use_captured_params:
            runtime_attn_metadata = captured_metadata
            actual_seq_lengths_q = query_start_loc
            metadata_block_table = block_tables
            metadata_block_source = "captured_graph_param"
            block_table_refreshed = False
        else:
            runtime_attn_metadata = (
                self._runtime_mixed_request_macro_attn_metadata(
                    context, runtime_metadata, layer_key))
            seq_lens = maybe_template_fia_seq_lens(
                context,
                getattr(runtime_attn_metadata, "seq_lens_list",
                        captured_metadata.seq_lens_list),
                _get_fia_key_t(key_cache, block_size),
                source=(
                    f"macro_graph_update_paired:{layer_key}:"
                    f"{split_entry['split_idx']}"),
            )
            actual_seq_lengths_q = getattr(
                runtime_attn_metadata,
                "actual_seq_lengths_q",
                captured_metadata.actual_seq_lengths_q)
            metadata_block_table, metadata_block_source = (
                _extract_block_table_from_metadata(runtime_attn_metadata))
            block_table_refreshed = _refresh_block_table_in_place(
                block_tables, metadata_block_table)
        workspace = split_entry["graph_params"].workspaces.get(
            split_entry["param_key"])

        if not event_only:
            began = False
            torch.npu.graph_task_update_begin(update_stream,
                                              split_entry["handle"])
            began = True
            try:
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
                    workspace=workspace,
                    out=[attn_output, softmax_lse],
                )
            finally:
                if began:
                    torch.npu.graph_task_update_end(update_stream)
        if split_debug.is_enabled():
            split_debug.log_event(
                "macro_graph_attention_update_event_record",
                {
                    "phase": "before_record",
                    "layer_idx": int(layer_idx),
                    "layer_key": str(layer_key),
                    "split_idx": int(split_entry["split_idx"]),
                    "event_only": bool(event_only),
                    "external_update_mode": external_update_mode,
                    "handle_id":
                    _macro_graph_object_id_info(split_entry["handle"]),
                    "event_id": _macro_graph_object_id_info(
                        split_entry["event"]),
                    "update_stream": {
                        "repr": repr(update_stream),
                        "stream_id": getattr(update_stream, "stream_id", None),
                    },
                },
                step_id=_split_debug_step_from_runner(self),
            )
        split_entry["event"].record(update_stream)
        if split_debug.is_enabled():
            split_debug.log_event(
                "macro_graph_attention_update_event_record",
                {
                    "phase": "after_record",
                    "layer_idx": int(layer_idx),
                    "layer_key": str(layer_key),
                    "split_idx": int(split_entry["split_idx"]),
                    "event_only": bool(event_only),
                    "external_update_mode": external_update_mode,
                    "handle_id":
                    _macro_graph_object_id_info(split_entry["handle"]),
                    "event_id": _macro_graph_object_id_info(
                        split_entry["event"]),
                    "update_stream": {
                        "repr": repr(update_stream),
                        "stream_id": getattr(update_stream, "stream_id", None),
                    },
                },
                step_id=_split_debug_step_from_runner(self),
            )

        return {
            "layer_idx": int(layer_idx),
            "split_idx": int(split_entry["split_idx"]),
            "runtime_shape": int(split_entry["runtime_shape"]),
            "in_parallel_streams": bool(split_entry["in_parallel_streams"]),
            "graph_param_key": graph_param_key_info(split_entry["param_key"]),
            "handle_id": _macro_graph_object_id_info(split_entry["handle"]),
            "event_id": _macro_graph_object_id_info(split_entry["event"]),
            "query": _macro_tensor_debug_info(query),
            "block_tables": _macro_tensor_debug_info(block_tables),
            "metadata_block_table": _macro_tensor_debug_info(
                metadata_block_table),
            "attn_output": _macro_tensor_debug_info(attn_output),
            "workspace": _macro_tensor_debug_info(workspace),
            "block_table_refreshed": bool(block_table_refreshed),
            "metadata_block_source": metadata_block_source,
            "use_captured_params": bool(use_captured_params),
            "event_only": bool(event_only),
            "external_update_mode": external_update_mode,
            "seq_lens_tail": (list(seq_lens[-6:])
                              if isinstance(seq_lens, (list, tuple)) else None),
            "actual_seq_lengths_q_tail":
            (list(actual_seq_lengths_q[-6:]) if isinstance(
                actual_seq_lengths_q, (list, tuple)) else None),
        }

    def _try_update_mixed_request_macro_attention_params_paired(
            self,
            entry: _PlannedMacroGraphEntry,
            runtime_ubatch_metadata: Optional[list[AscendUbatchMetadata]],
    ) -> tuple[bool, dict[str, Any]]:
        plan, detail = self._build_mixed_request_macro_paired_fia_update_plan(
            entry)
        if plan is None:
            return False, detail
        if runtime_ubatch_metadata is not None and len(
                runtime_ubatch_metadata) != 2:
            return False, {
                "applied": False,
                "reason": "runtime_split_count_mismatch",
                "runtime_split_count": int(len(runtime_ubatch_metadata)),
            }

        self._ensure_update_streams()
        update_stream = self.update_stream_main
        layer_samples: list[dict[str, Any]] = []
        update_start = time.perf_counter()
        replay_event_wait: dict[str, Any] = {}
        with torch.npu.stream(update_stream):
            replay_event_wait = (
                self._macro_attention_update_wait_prior_replay(
                    entry, update_stream))
            for layer in plan:
                split_debug_details = []
                for split_entry in layer["splits"]:
                    split_idx = int(split_entry["split_idx"])
                    split_runtime_metadata = (
                        runtime_ubatch_metadata[split_idx]
                        if runtime_ubatch_metadata is not None else None)
                    split_debug_details.append(
                        self._update_one_mixed_request_macro_fia_task(
                            update_stream=update_stream,
                            layer_idx=int(layer["layer_idx"]),
                            layer_key=layer["key"],
                            split_entry=split_entry,
                            runtime_metadata=split_runtime_metadata,
                        ))
                if layer["layer_idx"] in (0, len(plan) - 1):
                    layer_samples.append({
                        "layer_idx": int(layer["layer_idx"]),
                        "key": str(layer["key"]),
                        "splits": split_debug_details,
                    })

        detail.update({
            "applied": True,
            "mode": "paired_single_update_stream",
            "external_update_mode": _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
            "update_stream": {
                "repr": repr(update_stream),
                "stream_id": getattr(update_stream, "stream_id", None),
            },
            "replay_stream": {
                "repr": repr(self.stream_main),
                "stream_id": getattr(self.stream_main, "stream_id", None),
            },
            "replay_event_wait": replay_event_wait,
            "update_ms": (time.perf_counter() - update_start) * 1000.0,
            "layer_samples": layer_samples,
        })
        return True, detail

    def _record_mixed_request_macro_attention_events_only(
            self, entry: _PlannedMacroGraphEntry,
            paired_detail: dict[str, Any]) -> None:
        if entry.captured_metadata is None:
            raise RuntimeError(
                "mixed_request macro graph has no captured metadata for "
                "event-only external attention update")
        self._ensure_update_streams()
        update_stream = self.update_stream_main
        split_details: list[dict[str, Any]] = []
        update_start = time.perf_counter()
        with torch.npu.stream(update_stream):
            replay_event_wait = self._macro_attention_update_wait_prior_replay(
                entry, update_stream)
            for split_idx, metadata in enumerate(entry.captured_metadata):
                context = metadata.context
                in_parallel_streams = bool(
                    getattr(context, "in_parallel_streams", False))
                graph_params = get_graph_params(in_parallel_streams)
                runtime_shape = _macro_context_runtime_shape(context)
                param_key = get_graph_param_key(context, runtime_shape)
                if graph_params is None:
                    split_details.append({
                        "split_idx": int(split_idx),
                        "recorded": False,
                        "reason": "missing_graph_params",
                        "runtime_shape": int(runtime_shape),
                        "in_parallel_streams": bool(in_parallel_streams),
                    })
                    continue
                events = list(graph_params.events.get(param_key, []))
                handles = list(graph_params.handles.get(param_key, []))
                split_detail = {
                    "split_idx": int(split_idx),
                    "runtime_shape": int(runtime_shape),
                    "in_parallel_streams": bool(in_parallel_streams),
                    "graph_param_key": graph_param_key_info(param_key),
                    "event_count": int(len(events)),
                    "handle_count": int(len(handles)),
                    "recorded": bool(events),
                    "event_id_sample": _macro_graph_object_id_sample(events),
                    "handle_id_sample": _macro_graph_object_id_sample(handles),
                }
                for layer_idx, event in enumerate(events):
                    handle = handles[layer_idx] if layer_idx < len(
                        handles) else None
                    for phase in ("before_record", "after_record"):
                        if phase == "after_record":
                            event.record(update_stream)
                        if split_debug.is_enabled():
                            split_debug.log_event(
                                "macro_graph_attention_update_event_record",
                                {
                                    "phase": phase,
                                    "layer_idx": int(layer_idx),
                                    "split_idx": int(split_idx),
                                    "event_only": True,
                                    "external_update_mode": "event_only",
                                    "handle_id":
                                    _macro_graph_object_id_info(handle),
                                    "event_id":
                                    _macro_graph_object_id_info(event),
                                    "update_stream": {
                                        "repr": repr(update_stream),
                                        "stream_id": getattr(
                                            update_stream, "stream_id", None),
                                    },
                                },
                                step_id=_split_debug_step_from_runner(self),
                            )
                split_details.append(split_detail)
        replay_event_record = (
            self._macro_attention_update_record_replay_boundary(entry))
        if split_debug.is_enabled():
            split_debug.log_event(
                "macro_graph_attention_update",
                {
                    "key": repr(entry.key),
                    "replay_count": int(entry.replay_count),
                    "split_count": int(len(entry.captured_metadata)),
                    "total_update_ms":
                    (time.perf_counter() - update_start) * 1000.0,
                    "update_mode": "event_only",
                    "paired_update": paired_detail,
                    "replay_event_wait": replay_event_wait,
                    "replay_event_record": replay_event_record,
                    "update_stream": {
                        "repr": repr(update_stream),
                        "stream_id": getattr(update_stream, "stream_id", None),
                    },
                    "splits": split_details,
                },
                step_id=_split_debug_step_from_runner(self),
            )

    def _update_mixed_request_macro_attention_params(
            self,
            entry: _PlannedMacroGraphEntry,
            runtime_ubatch_metadata: Optional[
                list[AscendUbatchMetadata]] = None) -> None:
        if entry.captured_metadata is None:
            raise RuntimeError(
                "mixed_request macro graph has no captured metadata to "
                "update before replay")
        if (runtime_ubatch_metadata is not None and len(runtime_ubatch_metadata)
                != len(entry.captured_metadata)):
            raise RuntimeError(
                "mixed_request macro attention update split count mismatch: "
                f"captured={len(entry.captured_metadata)}, "
                f"runtime={len(runtime_ubatch_metadata)}")

        update_details: list[dict[str, Any]] = []
        total_update_start = time.perf_counter()

        paired_applied = False
        paired_detail: dict[str, Any] = {}
        try:
            paired_applied, paired_detail = (
                self._try_update_mixed_request_macro_attention_params_paired(
                    entry, runtime_ubatch_metadata))
        except Exception as exc:
            paired_detail = {
                "applied": False,
                "reason": "paired_update_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if _MACRO_GRAPH_ATTENTION_UPDATE_STRICT:
                raise
            logger.warning(
                "mixed_request macro paired FIA update failed; falling back "
                "to split updates. key=%s error=%s: %s",
                entry.key,
                type(exc).__name__,
                exc,
            )
        if split_debug.is_enabled():
            split_debug.log_event(
                "macro_graph_attention_paired_update",
                {
                    "key": repr(entry.key),
                    "replay_count": int(entry.replay_count),
                    **paired_detail,
                },
                step_id=_split_debug_step_from_runner(self),
            )
        if paired_applied:
            replay_event_record = (
                self._macro_attention_update_record_replay_boundary(entry))
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_attention_update",
                    {
                        "key": repr(entry.key),
                        "replay_count": int(entry.replay_count),
                        "split_count": int(len(entry.captured_metadata)),
                        "total_update_ms":
                        (time.perf_counter() - total_update_start) * 1000.0,
                        "update_mode": "paired_single_update_stream",
                        "paired_update": paired_detail,
                        "replay_event_record": replay_event_record,
                        "graph_param_summary":
                        self._mixed_request_macro_graph_param_summary()[:64],
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            return
        if _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE == "event_only":
            self._record_mixed_request_macro_attention_events_only(
                entry, paired_detail)
            return

        graph_details = self._mixed_request_macro_graph_param_details(
            entry.captured_metadata)
        for idx, metadata in enumerate(entry.captured_metadata):
            context = metadata.context
            runtime_metadata = (
                runtime_ubatch_metadata[idx]
                if runtime_ubatch_metadata is not None else None)
            runtime_shape = _macro_context_runtime_shape(context)
            if runtime_shape <= 0:
                raise RuntimeError(
                    "mixed_request macro graph could not determine "
                    f"runtime shape for split {idx}")
            self._ensure_update_streams()
            parallel_streams = bool(idx > 0)
            update_stream = (self.update_stream_parallel
                             if parallel_streams else self.update_stream_main)

            graph_detail = graph_details[idx]
            detail: dict[str, Any] = {
                "split_idx": int(idx),
                "runtime_shape": int(runtime_shape),
                "in_parallel_streams": bool(parallel_streams),
                "runtime_metadata_available": runtime_metadata is not None,
                "slot_context_id": _safe_context_id(context),
                "slot_attn_metadata_len": _macro_graph_attn_metadata_len(
                    getattr(context, "attn_metadata", None)),
                "runtime_attn_metadata_len": (
                    _macro_graph_attn_metadata_len(
                        getattr(runtime_metadata.context, "attn_metadata",
                                None)) if runtime_metadata is not None else None),
            }
            detail.update(graph_detail)

            update_available = bool(
                int(graph_detail.get("attn_param_count", 0) or 0) > 0
                and int(graph_detail.get("handle_count", 0) or 0) > 0
                and int(graph_detail.get("event_count", 0) or 0) > 0)
            detail["update_applied"] = update_available
            if not update_available:
                if not bool(graph_detail.get("graph_params_available", False)):
                    skip_reason = "missing_graph_params"
                elif not bool(
                        graph_detail.get("graph_param_registered", False)):
                    skip_reason = "missing_graph_param_key"
                else:
                    skip_reason = "missing_graph_task_handles"
                detail["skip_reason"] = skip_reason
                if _MACRO_GRAPH_ATTENTION_UPDATE_STRICT:
                    raise RuntimeError(
                        "mixed_request macro attention graph task update "
                        f"is unavailable for split {idx}: {skip_reason}; "
                        f"detail={detail}")
                update_details.append(detail)
                continue

            split_update_start = time.perf_counter()
            with torch.npu.stream(update_stream):
                detail["replay_event_wait"] = (
                    self._macro_attention_update_wait_prior_replay(
                        entry, update_stream))
            with self._macro_graph_runtime_attention_update_metadata(
                    context, runtime_metadata) as update_metadata_source:
                detail["update_metadata_source"] = update_metadata_source
                update_attn_params_split(
                    update_stream,
                    context,
                    runtime_shape,
                    self.vllm_config,
                    in_parallel_streams=parallel_streams,
                )
            detail["update_ms"] = (
                time.perf_counter() - split_update_start) * 1000.0
            update_details.append(detail)

        replay_event_record = (
            self._macro_attention_update_record_replay_boundary(entry))
        if split_debug.is_enabled():
            split_debug.log_event(
                "macro_graph_attention_update",
                {
                    "key": repr(entry.key),
                    "replay_count": int(entry.replay_count),
                    "split_count": int(len(entry.captured_metadata)),
                    "total_update_ms":
                    (time.perf_counter() - total_update_start) * 1000.0,
                    "splits": update_details,
                    "replay_event_record": replay_event_record,
                    "graph_param_summary":
                    self._mixed_request_macro_graph_param_summary()[:64],
                },
                step_id=_split_debug_step_from_runner(self),
            )

    def _macro_attention_update_ensure_replay_event(
            self, entry: _PlannedMacroGraphEntry) -> Optional[Any]:
        if not _MACRO_GRAPH_ATTENTION_UPDATE_REPLAY_EVENT:
            return None
        if entry.macro_attention_update_replay_event is None:
            entry.macro_attention_update_replay_event = torch.npu.Event()
            entry.macro_attention_update_replay_event_recorded = False
        return entry.macro_attention_update_replay_event

    def _macro_attention_update_wait_prior_replay(
            self,
            entry: _PlannedMacroGraphEntry,
            update_stream: torch.npu.Stream) -> dict[str, Any]:
        event = self._macro_attention_update_ensure_replay_event(entry)
        if event is None:
            return {"enabled": False}
        detail = {
            "enabled": True,
            "event_id": _macro_graph_object_id_info(event),
            "update_stream": {
                "repr": repr(update_stream),
                "stream_id": getattr(update_stream, "stream_id", None),
            },
            "waited": False,
        }
        if entry.macro_attention_update_replay_event_recorded:
            update_stream.wait_event(event)
            detail["waited"] = True
        else:
            detail["reason"] = "no_prior_replay_event"
        return detail

    def _macro_attention_update_record_replay_boundary(
            self, entry: _PlannedMacroGraphEntry) -> dict[str, Any]:
        event = self._macro_attention_update_ensure_replay_event(entry)
        if event is None:
            return {"enabled": False}
        replay_stream = self.stream_main
        event.record(replay_stream)
        entry.macro_attention_update_replay_event_recorded = True
        return {
            "enabled": True,
            "event_id": _macro_graph_object_id_info(event),
            "replay_stream": {
                "repr": repr(replay_stream),
                "stream_id": getattr(replay_stream, "stream_id", None),
            },
        }

    def _synchronize_macro_attention_update_boundary(
            self, *, phase: str) -> None:
        if not _MACRO_GRAPH_ATTENTION_UPDATE_SYNC:
            return
        sync_start = time.perf_counter()
        if phase == "before_update":
            self.stream_main.synchronize()
            self.stream_parallel.synchronize()
        elif phase == "after_update":
            self._ensure_update_streams()
            self.update_stream_main.synchronize()
            self.update_stream_parallel.synchronize()
        else:
            raise ValueError(f"Unsupported macro attention sync phase: {phase}")
        if split_debug.is_enabled():
            split_debug.log_event(
                "macro_graph_attention_update_sync",
                {
                    "phase": str(phase),
                    "sync_ms": (time.perf_counter() - sync_start) * 1000.0,
                },
                step_id=_split_debug_step_from_runner(self),
            )

    def _promote_mixed_request_macro_metadata(
            self, ubatch_metadata: list[AscendUbatchMetadata]) -> None:
        seen: set[int] = set()

        def visit(value: Any, device: torch.device) -> None:
            if value is None:
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, device)
                return
            if isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, device)
                return
            obj_id = id(value)
            if obj_id in seen:
                return
            seen.add(obj_id)
            seq_lens = getattr(value, "seq_lens", None)
            if (isinstance(seq_lens, torch.Tensor)
                    and seq_lens.device.type == "cpu"):
                value.seq_lens = seq_lens.to(device, non_blocking=True)

        for metadata in ubatch_metadata:
            device_tensor = metadata.positions
            if device_tensor is None and metadata.input_ids is not None:
                device_tensor = metadata.input_ids
            if not isinstance(device_tensor, torch.Tensor):
                continue
            visit(getattr(metadata.context, "attn_metadata", None),
                  device_tensor.device)

    def _materialize_torchair_macro_graph_entry(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata],
            model_kwargs: dict[str, Any],
    ) -> None:
        macro_graph_cfg = self._macro_graph_config()
        tng, CompilerConfig = _require_torchair_tagged_backend()
        if entry.inplace_attention_backend == "mixed_request":
            self._prepare_mixed_request_macro_contexts(entry, ubatch_metadata)

        runtime_calls = []
        for slice_idx, metadata in enumerate(ubatch_metadata):
            target_stream = (self.stream_parallel if slice_idx > 0
                             else self.stream_main)
            rotary_context = nullcontext()
            with self._bind_inplace_parallel_rope_capture_slot(
                    metadata.context, parallel_streams=(slice_idx > 0)):
                with rotary_context:
                    runtime_calls.append(
                        capture_piecewise_model_call(
                            model=self.model,
                            metadata=metadata,
                            model_kwargs=model_kwargs,
                            stream=target_stream,
                        ))
        if (entry.inplace_attention_backend == "mixed_request"
                and split_debug.is_enabled()):
            split_debug.log_event(
                "macro_graph_attention_params_captured",
                {
                    "key": repr(entry.key),
                    "splits": self._mixed_request_macro_graph_param_details(
                        ubatch_metadata),
                    "graph_param_summary":
                    self._mixed_request_macro_graph_param_summary()[:64],
                },
                step_id=_split_debug_step_from_runner(self),
            )

        event_tag_prefix = (
            f"vllm_macro_{os.getpid()}_{len(self._macro_graph_registry)}_"
            f"{abs(hash(entry.key)) & 0xffffffff:x}")
        module = _TorchairTaggedPiecewiseMacroModule(
            runtime_calls=runtime_calls,
            contexts=[metadata.context for metadata in ubatch_metadata],
            event_tag_prefix=event_tag_prefix,
            secondary_stream_tag="1",
            validate_no_inner_aclgraph=bool(
                getattr(macro_graph_cfg, "validate_no_inner_aclgraph", True)),
            tng=tng,
        )
        if _MACRO_GRAPH_EAGER_PREFLIGHT:
            snapshots = (self._snapshot_mixed_request_macro_graph_params(
                ubatch_metadata)
                         if entry.inplace_attention_backend == "mixed_request"
                         else [])
            try:
                module()
                self.stream_main.synchronize()
                self.stream_parallel.synchronize()
            except Exception:
                logger.error(
                    "Torchair tagged-event macro graph eager preflight "
                    "failed before torch.compile:\n%s",
                    traceback.format_exc(),
                )
                raise
            finally:
                self._restore_mixed_request_macro_graph_param_snapshots(
                    snapshots)
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.enable_output_clone.value = True
        backend = tng.get_npu_backend(compiler_config=config)

        compile_start = time.perf_counter()
        entry.module = module
        entry.backend = "torchair_tagged_event"
        entry.compiled_callable = torch.compile(
            module,
            backend=backend,
            dynamic=False,
            fullgraph=True,
        )
        entry.captured_metadata = ubatch_metadata
        entry.compile_ms = (time.perf_counter() - compile_start) * 1000.0
        logger.info(
            "Materialized torchair tagged-event macro graph: key=%s, "
            "compile_wrapper_ms=%.3f",
            entry.key,
            entry.compile_ms,
        )

    def _materialize_npugraph_ex_macro_graph_entry(
            self,
            entry: _PlannedMacroGraphEntry,
            ubatch_metadata: list[AscendUbatchMetadata],
            model_kwargs: dict[str, Any],
    ) -> None:
        macro_graph_cfg = self._macro_graph_config()
        _require_npugraph_ex_backend()
        backend_opaque_update = self._use_npugraph_ex_macro_opaque_attention_update(
            "npugraph_ex", entry.inplace_attention_backend)
        if backend_opaque_update:
            self._ensure_npugraph_ex_macro_opaque_attention_update_env()
        entry.macro_attention_update_in_backend = bool(backend_opaque_update)
        if entry.inplace_attention_backend == "mixed_request":
            self._prepare_mixed_request_macro_contexts(
                entry,
                ubatch_metadata,
                opaque_attention_update=backend_opaque_update)

        runtime_calls = []
        snapshots = (self._snapshot_mixed_request_macro_graph_params(
            ubatch_metadata)
                     if entry.inplace_attention_backend == "mixed_request"
                     else [])
        try:
            for slice_idx, metadata in enumerate(ubatch_metadata):
                target_stream = (self.stream_parallel if slice_idx > 0
                                 else self.stream_main)
                rotary_context = nullcontext()
                with self._bind_inplace_parallel_rope_capture_slot(
                        metadata.context, parallel_streams=(slice_idx > 0)):
                    with rotary_context:
                        runtime_calls.append(
                            capture_piecewise_model_call(
                                model=self.model,
                                metadata=metadata,
                                model_kwargs=model_kwargs,
                                stream=target_stream,
                            ))
        finally:
            self._restore_mixed_request_macro_graph_param_snapshots(snapshots)

        context_key = (
            f"vllm_macro_npugraph_ex_{os.getpid()}_"
            f"{len(self._macro_graph_registry)}_"
            f"{abs(hash(entry.key)) & 0xffffffff:x}")
        module = _NpuGraphExPiecewiseMacroModule(
            runtime_calls=runtime_calls,
            contexts=[metadata.context for metadata in ubatch_metadata],
            context_key=context_key,
            validate_no_inner_aclgraph=bool(
                getattr(macro_graph_cfg, "validate_no_inner_aclgraph", True)),
        )
        if entry.inplace_attention_backend == "mixed_request":
            try:
                self._prewarm_mixed_request_macro_fia_workspaces(
                    module, ubatch_metadata)
            except Exception:
                logger.error(
                    "npugraph_ex mixed_request macro graph FIA workspace "
                    "prewarm failed before torch.compile:\n%s",
                    traceback.format_exc(),
                )
                raise
        elif _MACRO_GRAPH_EAGER_PREFLIGHT:
            try:
                module()
                torch.npu.synchronize()
            except Exception:
                logger.error(
                    "npugraph_ex macro graph eager preflight failed before "
                    "torch.compile:\n%s",
                    traceback.format_exc(),
                )
                raise

        backend_options = dict(
            getattr(macro_graph_cfg, "backend_options", {}) or {})
        backend_options.setdefault("clone_output", True)
        backend_options.setdefault("return_captured_outputs_on_first_run",
                                   True)
        compile_start = time.perf_counter()
        entry.module = module
        entry.backend = "npugraph_ex"
        entry.compiled_callable = torch.compile(
            module,
            backend="npugraph_ex",
            dynamic=False,
            fullgraph=True,
            options=backend_options,
        )
        entry.captured_metadata = ubatch_metadata
        entry.compile_ms = (time.perf_counter() - compile_start) * 1000.0
        if (entry.inplace_attention_backend == "mixed_request"
                and split_debug.is_enabled()):
            split_debug.log_event(
                "macro_graph_attention_params_materialized",
                {
                    "key": repr(entry.key),
                    "compile_ms": float(entry.compile_ms),
                    "macro_attention_update_in_backend":
                    bool(entry.macro_attention_update_in_backend),
                    "splits": self._mixed_request_macro_graph_param_details(
                        ubatch_metadata),
                    "graph_param_summary":
                    self._mixed_request_macro_graph_param_summary()[:64],
                },
                step_id=_split_debug_step_from_runner(self),
            )
        logger.info(
            "Materialized npugraph_ex macro graph: key=%s, "
            "compile_wrapper_ms=%.3f, backend_options=%s",
            entry.key,
            entry.compile_ms,
            backend_options,
        )

    def _run_split_batch_inplace_parallel_macro_graph(
            self,
            split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str,
    ) -> Any:
        macro_graph_cfg = self._macro_graph_config()
        key = self._macro_graph_key_from_split_slices(
            split_batch_slices,
            inplace_attention_backend,
            uniform_decode=(inplace_attention_backend != "mixed_request"),
        )
        replay_split_batch_slices = split_batch_slices
        replay_key = key

        def fallback_to_piecewise(reason: str,
                                  exc: Optional[BaseException] = None) -> Any:
            miss_policy = getattr(macro_graph_cfg, "miss_policy", "error")
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_miss",
                    {
                        "key": repr(key),
                        "reason": str(reason),
                        "miss_policy": str(miss_policy),
                        "fallback_to": ("piecewise_attention_parallel"
                                        if miss_policy != "error" else None),
                        "error_type": (type(exc).__name__
                                       if exc is not None else None),
                        "error": (str(exc) if exc is not None else None),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            if miss_policy == "error":
                if exc is not None:
                    raise exc
                raise MissingMacroGraphError(
                    "Missing multistream macro graph for split-batch plan "
                    f"key={key!r}; miss_policy={miss_policy!r}. "
                    "Runtime lazy capture is disabled for macro graphs.")
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_fallback",
                    {
                        "key": repr(key),
                        "reason": str(reason),
                        "miss_policy": str(miss_policy),
                        "fallback_to": "piecewise_attention_parallel",
                        "error_type": (type(exc).__name__
                                       if exc is not None else None),
                        "error": (str(exc) if exc is not None else None),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            if exc is not None:
                logger.warning(
                    "Falling back from macro graph to piecewise path "
                    "because %s failed under miss_policy=%s: %s",
                    reason,
                    miss_policy,
                    exc,
                )
            return self._run_split_batch_inplace_parallel_piecewise(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                model_kwargs,
                batch_descriptor,
                CUDAGraphMode.PIECEWISE,
                inplace_attention_backend,
            )

        entry = self._macro_graph_registry.get(key)
        if entry is not None and not entry.materialized:
            replay_split_batch_slices = entry.plan.split_slices
        if entry is None:
            entry, bucket_slices = self._macro_graph_bucket_entry_for_split_slices(
                split_batch_slices,
                inplace_attention_backend,
                uniform_decode=(inplace_attention_backend != "mixed_request"),
            )
            if entry is None or bucket_slices is None:
                return fallback_to_piecewise("missing_macro_graph")
            replay_split_batch_slices = bucket_slices
            replay_key = entry.key
            if split_debug.is_enabled():
                bucket_req_caps = tuple(
                    int(v) for v in getattr(entry, "split_req_caps", None)
                ) if getattr(entry, "split_req_caps", None) is not None else tuple(
                    int(getattr(s, "request_capacity", s.num_requests))
                    for s in entry.plan.split_slices)
                bucket_padding_tokens = [
                    int(replay.graph_num_tokens - runtime.num_tokens)
                    for runtime, replay in zip(split_batch_slices,
                                               replay_split_batch_slices)
                ]
                bucket_padding_req_chunks = [
                    self._compact_padding_req_chunks(
                        padding,
                        max(1,
                            int(getattr(runtime, "max_query_len", 0) or 0)))
                    for padding, runtime in zip(bucket_padding_tokens,
                                                split_batch_slices)
                ]
                bucket_effective_reqs = [
                    int(runtime.num_requests + chunks)
                    for runtime, chunks in zip(split_batch_slices,
                                               bucket_padding_req_chunks)
                ]
                padding_ratio_grace_tokens = max(
                    0,
                    int(
                        getattr(macro_graph_cfg,
                                "bucket_padding_ratio_grace_tokens", 0) or 0))
                split_debug.log_event(
                    "macro_graph_bucket_match",
                    {
                        "runtime_key": repr(key),
                        "graph_key": repr(replay_key),
                        "runtime_actual_tokens": [
                            int(s.num_tokens) for s in split_batch_slices
                        ],
                        "runtime_graph_tokens": [
                            int(s.graph_num_tokens)
                            for s in split_batch_slices
                        ],
                        "bucket_graph_tokens": [
                            int(s.graph_num_tokens)
                            for s in replay_split_batch_slices
                        ],
                        "bucket_padding_tokens": bucket_padding_tokens,
                        "bucket_padding_ratios": [
                            float((replay.graph_num_tokens -
                                   runtime.num_tokens) /
                                  max(1, runtime.num_tokens))
                            for runtime, replay in zip(
                                split_batch_slices,
                                replay_split_batch_slices)
                        ],
                        "bucket_padding_ratio_exempt": [
                            bool(runtime.num_tokens <=
                                 padding_ratio_grace_tokens)
                            for runtime in split_batch_slices
                        ],
                        "bucket_padding_ratio_grace_tokens":
                        padding_ratio_grace_tokens,
                        "allow_padded_replay": bool(
                            getattr(macro_graph_cfg, "allow_padded_replay",
                                    False)),
                        "max_padding_ratio_per_split": (
                            None if getattr(
                                macro_graph_cfg,
                                "max_padding_ratio_per_split", 0.0) is None
                            else float(
                                getattr(macro_graph_cfg,
                                        "max_padding_ratio_per_split", 0.0))
                        ),
                        "runtime_num_reqs": [
                            int(s.num_requests) for s in split_batch_slices
                        ],
                        "bucket_num_reqs": [
                            int(s.num_requests)
                            for s in entry.plan.split_slices
                        ],
                        "bucket_req_caps": [int(v) for v in bucket_req_caps],
                        "bucket_effective_reqs": bucket_effective_reqs,
                        "bucket_padding_req_chunks":
                        bucket_padding_req_chunks,
                        "bucket_metadata_pad_reqs": [
                            max(0,
                                int(req_cap) - int(effective_req))
                            for req_cap, effective_req in zip(
                                bucket_req_caps, bucket_effective_reqs)
                        ],
                        "bucket_req_capacity_ok": all(
                            int(effective_req) <= int(req_cap)
                            for effective_req, req_cap in zip(
                                bucket_effective_reqs, bucket_req_caps)),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

        if inplace_attention_backend == "mixed_request":
            ubatch_metadata = self._make_mixed_request_split_metadata_parallel(
                replay_split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                batch_descriptor,
            )
        else:
            ubatch_metadata = self._make_split_batch_metadata_inplace_parallel(
                split_ubatch_slices,
                replay_split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                batch_descriptor,
                aclgraph_runtime_mode,
                inplace_attention_backend,
            )

        if len(ubatch_metadata) != 2:
            raise RuntimeError(
                "multistream macro graph currently supports exactly "
                f"2 splits, got {len(ubatch_metadata)}")
        backend = getattr(macro_graph_cfg, "backend", "npugraph_ex")
        backend_options = dict(getattr(macro_graph_cfg, "backend_options", {})
                               or {})
        if backend == "npugraph_ex":
            backend_options.setdefault("clone_output", True)
            backend_options.setdefault("return_captured_outputs_on_first_run",
                                       True)
        use_backend_attention_update = (
            entry.macro_attention_update_in_backend if entry.materialized else
            self._use_npugraph_ex_macro_opaque_attention_update(
                backend, inplace_attention_backend))
        if use_backend_attention_update:
            self._ensure_npugraph_ex_macro_opaque_attention_update_env()

        if (backend == "npugraph_ex"
                and inplace_attention_backend == "mixed_request"
                and not use_backend_attention_update
                and not _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE):
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_fallback",
                    {
                        "key": repr(key),
                        "runtime_key": repr(key),
                        "graph_key": repr(replay_key),
                        "reason":
                        "mixed_request_npugraph_ex_external_update_disabled",
                        "fallback_to": "piecewise_attention_parallel",
                        "macro_graph_backend": str(backend),
                        "external_update_mode":
                        _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
                        "macro_attention_update_in_backend":
                        bool(use_backend_attention_update),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            return self._run_split_batch_inplace_parallel_piecewise(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                model_kwargs,
                batch_descriptor,
                CUDAGraphMode.PIECEWISE,
                inplace_attention_backend,
            )

        if inplace_attention_backend == "mixed_request":
            self._promote_mixed_request_macro_metadata(ubatch_metadata)
            self._prepare_mixed_request_macro_contexts(
                entry,
                ubatch_metadata,
                opaque_attention_update=use_backend_attention_update)

        original_forward_context = get_forward_context()
        self._t_replay_start = time.perf_counter()
        try:
            materialized_now = not entry.materialized
            macro_attention_retention_active = False
            copied = 0
            if not entry.materialized:
                slot_metadata = _macro_graph_clone_ubatch_metadata_slots(
                    ubatch_metadata)
                entry.runtime_slot_metadata = slot_metadata
                entry.captured_metadata = slot_metadata
                entry.binding_plan = self._build_macro_graph_binding_plan(
                    entry, ubatch_metadata)
                copied, bind_detail = self._bind_macro_graph_entry(
                    entry,
                    ubatch_metadata,
                    collect_detail=split_debug.is_enabled(),
                )
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "macro_graph_slot_materialize",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            "macro_graph_backend": str(backend),
                            "copied_tensor_count": int(copied),
                            "slot_split_count": int(len(slot_metadata)),
                            **(bind_detail or {}),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                if entry.inplace_attention_backend == "mixed_request":
                    self._begin_macro_attention_tensor_retention(
                        entry, slot_metadata)
                    macro_attention_retention_active = True
                if backend == "npugraph_ex":
                    self._materialize_npugraph_ex_macro_graph_entry(
                        entry, slot_metadata, model_kwargs)
                elif backend == "torchair_tagged_event":
                    self._materialize_torchair_macro_graph_entry(
                        entry, slot_metadata, model_kwargs)
                else:
                    raise RuntimeError(
                        "Unsupported macro_graph_config.backend at replay "
                        f"time: {backend!r}")
            else:
                collect_bind_detail = split_debug.is_enabled()
                copied, bind_detail = self._bind_macro_graph_entry(
                    entry,
                    ubatch_metadata,
                    collect_detail=collect_bind_detail)
                if collect_bind_detail:
                    split_debug.log_event(
                        "macro_graph_bind",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            "macro_graph_backend": str(entry.backend
                                                       or backend),
                            "macro_graph_backend_options": backend_options,
                            "copied_tensor_count": int(copied),
                            "replay_count": int(entry.replay_count),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                    if bind_detail is not None:
                        split_debug.log_event(
                            "macro_graph_bind_detail",
                            {
                                "key": repr(replay_key),
                                "runtime_key": repr(key),
                                "macro_graph_backend": str(entry.backend
                                                           or backend),
                                "macro_graph_backend_options":
                                backend_options,
                                "replay_count": int(entry.replay_count),
                                **bind_detail,
                            },
                            step_id=_split_debug_step_from_runner(self),
                        )

            compiled_call_start = time.perf_counter()
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_replay_boundary",
                    {
                        "key": repr(replay_key),
                        "runtime_key": repr(key),
                        "phase": "before_compiled_callable",
                        "macro_graph_backend": str(entry.backend or backend),
                        "macro_attention_update_in_backend":
                        bool(entry.macro_attention_update_in_backend),
                        "replay_count": int(entry.replay_count),
                        "materialized_now": bool(materialized_now),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            with torch.npu.stream(self.stream_main):
                outputs = entry.compiled_callable()
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_replay_boundary",
                    {
                        "key": repr(replay_key),
                        "runtime_key": repr(key),
                        "phase": "after_compiled_callable",
                        "macro_graph_backend": str(entry.backend or backend),
                        "macro_attention_update_in_backend":
                        bool(entry.macro_attention_update_in_backend),
                        "replay_count": int(entry.replay_count),
                        "materialized_now": bool(materialized_now),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            if macro_attention_retention_active:
                self._end_macro_attention_tensor_retention(entry)
                macro_attention_retention_active = False
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "macro_graph_attention_tensor_retention",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            **self._macro_attention_retention_debug(entry),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
            first_npugraph_ex_call_returns_captured_outputs = bool(
                materialized_now and entry.backend == "npugraph_ex"
                and backend_options.get("return_captured_outputs_on_first_run",
                                        True))
            if (entry.inplace_attention_backend == "mixed_request"
                    and not first_npugraph_ex_call_returns_captured_outputs
                    and entry.macro_attention_update_in_backend):
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "macro_graph_attention_update",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            "replay_count": int(entry.replay_count),
                            "update_mode": "npugraph_ex_backend_opaque",
                            "external_update_skipped": True,
                            "macro_graph_backend_options": backend_options,
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
            elif (entry.inplace_attention_backend == "mixed_request"
                  and not first_npugraph_ex_call_returns_captured_outputs
                  and not entry.macro_attention_external_update):
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "macro_graph_attention_update_skipped",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            "reason": "external_update_disabled",
                            "external_update_mode":
                            _MACRO_GRAPH_EXTERNAL_ATTENTION_UPDATE_MODE,
                            "replay_count": int(entry.replay_count),
                            "materialized_now": bool(materialized_now),
                            "macro_graph_backend_options": backend_options,
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
            elif (entry.inplace_attention_backend == "mixed_request"
                  and not first_npugraph_ex_call_returns_captured_outputs):
                # Match ACLGraphWrapper semantics: graph replay is launched
                # first. The launched graph waits at each captured FIA
                # ExternalEvent; graph_task_update patches the FIA task and
                # records that same event to release execution.
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "macro_graph_replay_boundary",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            "phase": "before_external_attention_update",
                            "replay_count": int(entry.replay_count),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                self._update_mixed_request_macro_attention_params(
                    entry, ubatch_metadata)
                if split_debug.is_enabled():
                    split_debug.log_event(
                        "macro_graph_replay_boundary",
                        {
                            "key": repr(replay_key),
                            "runtime_key": repr(key),
                            "phase": "after_external_attention_update",
                            "replay_count": int(entry.replay_count),
                        },
                        step_id=_split_debug_step_from_runner(self),
                    )
                self._synchronize_macro_attention_update_boundary(
                    phase="after_update")
            elif (entry.inplace_attention_backend == "mixed_request"
                  and first_npugraph_ex_call_returns_captured_outputs
                  and split_debug.is_enabled()):
                split_debug.log_event(
                    "macro_graph_attention_update_skipped",
                    {
                        "key": repr(replay_key),
                        "runtime_key": repr(key),
                        "reason":
                        "npugraph_ex_first_call_returns_captured_outputs",
                        "materialized_now": bool(materialized_now),
                        "macro_graph_backend_options": backend_options,
                    },
                    step_id=_split_debug_step_from_runner(self),
                )
            if not first_npugraph_ex_call_returns_captured_outputs:
                entry.replay_count += 1
            self.stream_main.synchronize()
            compiled_call_ms = (
                time.perf_counter() - compiled_call_start) * 1000.0
            if split_debug.is_enabled():
                split_debug.log_event(
                    "macro_graph_replay",
                    {
                        "key": repr(replay_key),
                        "runtime_key": repr(key),
                        "macro_graph_backend": str(entry.backend or backend),
                        "macro_graph_backend_options": backend_options,
                        "replay_count": int(entry.replay_count),
                        "materialized_now": bool(materialized_now),
                        "copied_tensor_count": int(copied),
                        "compiled_call_ms": float(compiled_call_ms),
                        "compile_ms": float(entry.compile_ms),
                    },
                    step_id=_split_debug_step_from_runner(self),
                )

            merged_results = []
            for idx, output in enumerate(outputs):
                trimmed = self._trim_split_output(
                    output, split_batch_slices[idx].num_tokens)
                merged_results.append(self._clone_split_output(trimmed))
            with override_forward_context(original_forward_context):
                return self._merge_split_outputs(merged_results)
        except Exception as exc:
            logger.warning(
                "Macro graph replay failed; falling back under "
                "miss_policy=%s. key=%s replay_key=%s\n%s",
                getattr(macro_graph_cfg, "miss_policy", "error"),
                key,
                replay_key,
                traceback.format_exc(),
            )
            if 'macro_attention_retention_active' in locals(
            ) and macro_attention_retention_active:
                self._end_macro_attention_tensor_retention(entry)
                macro_attention_retention_active = False
            return fallback_to_piecewise("macro_graph_replay_failed", exc)
        finally:
            if 'macro_attention_retention_active' in locals(
            ) and macro_attention_retention_active:
                self._end_macro_attention_tensor_retention(entry)
            self._t_replay_end = time.perf_counter()


    def _run_split_batch_inplace_parallel(
            self,
            split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str,
    ) -> Any:
        if self._macro_graph_enabled():
            return self._run_split_batch_inplace_parallel_macro_graph(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                model_kwargs,
                batch_descriptor,
                aclgraph_runtime_mode,
                inplace_attention_backend,
            )

        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        replay_policy = getattr(split_cfg, "inplace_parallel_replay_policy",
                                "full_graph_parallel")
        if replay_policy == "piecewise_attention_parallel":
            return self._run_split_batch_inplace_parallel_piecewise(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                model_kwargs,
                batch_descriptor,
                aclgraph_runtime_mode,
                inplace_attention_backend,
            )

        ubatch_metadata = self._make_split_batch_metadata_inplace_parallel(
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            batch_descriptor,
            aclgraph_runtime_mode,
            inplace_attention_backend,
        )

        original_forward_context = get_forward_context()
        num_splits = len(split_batch_slices)
        results: list[Optional[Any]] = [None] * num_splits
        split_errors: list[tuple[int, Exception]] = []
        split_error_lock = threading.Lock()
        self._t_replay_start = time.perf_counter()

        def _run_inplace_parallel_worker(slice_idx: int) -> None:
            try:
                split_slice = split_batch_slices[slice_idx]
                metadata = ubatch_metadata[slice_idx]
                parallel_streams = slice_idx > 0
                target_stream = (self.stream_parallel if parallel_streams
                                 else self.stream_main)

                with torch.inference_mode():
                    if self._needs_inplace_serial_offset_capture(metadata):
                        split_result = self._run_inplace_serial_offset_capture(
                            metadata,
                            split_slice,
                            model_kwargs,
                            parallel_streams=parallel_streams,
                        )
                    else:
                        # if parallel_streams :
                        #     time.sleep(10 / 1000.0)
                        if (int(getattr(
                                metadata.context.batch_descriptor,
                                "start_num_tokens", 0) or 0) > 0
                                and metadata.context.cudagraph_runtime_mode
                                == CUDAGraphMode.FULL
                                and not self._has_aclgraph_for_context(
                                    metadata.context)):
                            raise RuntimeError(
                                "Missing inplace parallel offset ACL graph "
                                "before normal replay path: "
                                f"{metadata.context.batch_descriptor!r}")
                        with torch.npu.stream(target_stream):
                            with override_forward_context(metadata.context):
                                split_result = self.model(
                                    input_ids=metadata.input_ids,
                                    positions=metadata.positions,
                                    inputs_embeds=metadata.inputs_embeds,
                                    intermediate_tensors=
                                    metadata.intermediate_tensors,
                                    **model_kwargs,
                                )
                                if (metadata.context.cudagraph_runtime_mode
                                        == CUDAGraphMode.FULL):
                                    if split_slice.start_num_tokens > 0:
                                        self._update_attn_params_for_split_ubatch(
                                            metadata.context,
                                            split_slice.graph_num_tokens,
                                            parallel_streams=parallel_streams)
                                    else:
                                        self._update_attn_params_for_wrapper(
                                            metadata.context,
                                            split_slice.graph_num_tokens)


                    with torch.npu.stream(target_stream):
                        results[slice_idx] = self._clone_split_output(
                            self._trim_split_output(split_result,
                                                    split_slice.num_tokens))
            except Exception as e:
                with split_error_lock:
                    split_errors.append((slice_idx, e))

        split_workers: list[threading.Thread] = []
        try:
            # torch.npu.set_stream_limit(self.update_stream_main,
            #                            cube_num=10,
            #                            vector_num=20)
            # torch.npu.set_stream_limit(self.update_stream_parallel,
            #                            cube_num=10,
            #                            vector_num=20)
            for slice_idx in range(num_splits):
                worker = threading.Thread(
                    target=_run_inplace_parallel_worker,
                    args=(slice_idx,),
                    name=f"inplace-parallel-replay-{slice_idx}")
                split_workers.append(worker)
                worker.start()

            for worker in split_workers:
                worker.join()

            if split_errors:
                split_errors.sort(key=lambda item: item[0])
                failed_slice_idx, first_error = split_errors[0]
                raise RuntimeError(
                    "inplace parallel replay worker failed at "
                    f"slice_idx={failed_slice_idx}") from first_error

            self.stream_main.synchronize()
            if num_splits > 1:
                self.stream_parallel.synchronize()

            merged_results: list[Any] = [
                result for result in results if result is not None
            ]
            if len(merged_results) != num_splits:
                raise RuntimeError(
                    "Missing inplace parallel split result: "
                    f"expected={num_splits}, got={len(merged_results)}")

            with override_forward_context(original_forward_context):
                return self._merge_split_outputs(merged_results)
        finally:
            self._t_replay_end = time.perf_counter()


    def _run_split_batch_parallel(
        self,
        split_ubatch_slices: UBatchSlices,
        split_batch_slices: SplitBatchSlices,
        attn_metadata: PerLayerAttnMetadata,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
        model_kwargs: dict[str, Any],
        batch_descriptor: BatchDescriptor,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> Any:
        """
        执行split-batch，确保每个batch重放时地址不一致。
        
        核心逻辑：
        - 图捕获时，输入地址在 self.input_ids.gpu、self.positions.gpu 等的起始位置
        - 第一个ubatch执行时，数据已经在正确位置
        - 第二个ubatch执行前，将其数据复制到起始位置，然后用起始位置执行
        """
        return self._run_split_batch_parallel_impl(
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            model_kwargs,
            batch_descriptor,
            aclgraph_runtime_mode,
        )
    
    def _run_split_batch_parallel_impl(
        self,
        split_ubatch_slices: UBatchSlices,
        split_batch_slices: SplitBatchSlices,
        attn_metadata: PerLayerAttnMetadata,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
        model_kwargs: dict[str, Any],
        batch_descriptor: BatchDescriptor,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> Any:
        # Step 1: 为所有split准备元数据
        ubatch_metadata = self._make_split_batch_metadata_parallel_streams(
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            batch_descriptor,
            aclgraph_runtime_mode,
        )
        
        results: list[Any] = []
        original_forward_context = get_forward_context()
        # torch.npu.set_stream_limit(self.stream_main, cube_num=15, vector_num=20)
        # torch.npu.set_stream_limit(self.stream_parallel, cube_num=15, vector_num=20)
        num_splits = len(split_batch_slices)
        results: list[Optional[Any]] = [None] * num_splits
        split_errors: list[tuple[int, Exception]] = []
        split_error_lock = threading.Lock()

        self._t_replay_start = time.perf_counter()
        def _run_split_replay_worker(slice_idx: int) -> None:
            try:
                split_slice = split_batch_slices[slice_idx]
                metadata = ubatch_metadata[slice_idx]
                current_num_tokens = split_slice.num_tokens
                # attn_params are keyed by the padded graph size, not the
                # actual token count.  Use padded_num_tokens as runtime_shape.
                current_padded_num_tokens = split_slice.padded_num_tokens
                parallel_streams = slice_idx > 0
                target_stream = self.stream_parallel if parallel_streams else self.stream_main
                with torch.inference_mode():
                    # if parallel_streams:
                    #     split_attn_metadata = attn_metadata
                    #     if isinstance(attn_metadata, list):
                    #         split_attn_metadata = attn_metadata[slice_idx]

                    #     split_slot_mapping = _get_slot_mapping_from_attn_metadata(
                    #         split_attn_metadata)
                    #     if split_slot_mapping is not None:
                    #         copy_len = min(current_num_tokens,
                    #                         int(split_slot_mapping.shape[0]))
                    #         relocated_slot_mapping = split_slot_mapping[:copy_len]
                    #         updated_count = _set_slot_mapping_for_attn_metadata(
                    #             split_attn_metadata, relocated_slost_mapping)

                    #     ubatch_batch_descriptor = BatchDescriptor(
                    #         num_tokens=current_num_tokens,
                    #         num_reqs=split_slice.num_requests,
                    #         uniform=batch_descriptor.uniform,
                    #         has_lora=batch_descriptor.has_lora,
                    #     )
                    #     if aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                    #         ubatch_batch_descriptor = (
                    #             ubatch_batch_descriptor
                    #             .relax_for_mixed_batch_cudagraphs())
                    #     if _SPLIT_LOCAL_CONTEXT_REBUILD:
                    #         rebuild_ubatch_slices = [
                    #             UBatchSlice(
                    #                 slice(0, split_slice.num_requests),
                    #                 slice(0, current_num_tokens),
                    #             )
                    #         ]
                    #         rebuild_positions = metadata.positions
                    #         rebuild_ubatch_num = 0
                    #     else:
                    #         rebuild_ubatch_slices = split_ubatch_slices
                    #         rebuild_positions = positions
                    #         rebuild_ubatch_num = slice_idx

                    #     metadata.context = create_ascend_forward_context(
                    #         metadata.context,
                    #         attn_metadata=split_attn_metadata,
                    #         vllm_config=self.vllm_config,
                    #         dp_metadata=original_forward_context.dp_metadata,
                    #         ubatch_slices=rebuild_ubatch_slices,
                    #         batch_descriptor=ubatch_batch_descriptor,
                    #         cudagraph_runtime_mode=aclgraph_runtime_mode,
                    #         ubatch_num=rebuild_ubatch_num,
                    #         positions=rebuild_positions,
                    #         in_parallel_streams=True,
                    #         cos_sin_slot_id=slice_idx,
                    #     )

                    with torch.npu.stream(target_stream):
                        with override_forward_context(metadata.context):
                            split_result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )

                            if aclgraph_runtime_mode == CUDAGraphMode.FULL:
                                self._update_attn_params_for_split_ubatch(
                                    metadata.context,
                                    current_padded_num_tokens,
                                    parallel_streams=parallel_streams)
                            results[slice_idx] = self._trim_split_output(
                                split_result,
                                split_batch_slices[slice_idx].num_tokens)
            except Exception as e:
                with split_error_lock:
                    split_errors.append((slice_idx, e))

        split_workers: list[threading.Thread] = []
        for slice_idx in range(num_splits):
            worker = threading.Thread(target=_run_split_replay_worker,
                                        args=(slice_idx,),
                                        name=f"split-replay-{slice_idx}")
            split_workers.append(worker)
            worker.start()

        for worker in split_workers:
            worker.join()

        if split_errors:
            split_errors.sort(key=lambda item: item[0])
            failed_slice_idx, first_error = split_errors[0]
            raise RuntimeError(
                f"split replay worker failed at slice_idx={failed_slice_idx}"
            ) from first_error

        # Wait per stream instead of using a device-wide barrier to preserve
        # overlap between split-0(main) and split-1(parallel) replay.
        logger.debug("[split_batch] synchronizing stream_main")
        self.stream_main.synchronize()
        logger.debug("[split_batch] stream_main synchronized")
        if len(split_batch_slices) > 1:
            logger.debug("[split_batch] synchronizing stream_parallel")
            self.stream_parallel.synchronize()
            logger.debug("[split_batch] stream_parallel synchronized")
        merged_results: list[Any] = [result for result in results
                                        if result is not None]

        logger.debug("[split_batch] merging %d split outputs", len(merged_results))
        with override_forward_context(original_forward_context):
            result = self._merge_split_outputs(merged_results)
        logger.debug("[split_batch] merge done, returning result")

        if (_SPLIT_MERGE_DUMP
                and not getattr(self, "_split_batch_dumped", False)):
            dump_path = os.path.join(
                os.getcwd(), "split_batch_merged_first_result_gg.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(self._to_jsonable(result), f, ensure_ascii=False)
            self._split_batch_dumped = True

        return result

    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
                          num_valid_tokens):
        if np.array_equal(self.seq_lens.np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config and self.speculative_config.method == 'mtp':
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
        return attn_state

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(
                cu_num_scheduled_tokens - num_sampled_tokens,
                num_sampled_tokens)
            logits_indices_pcp += arange
            logits_indices_pcp = torch.from_numpy(
                logits_indices_pcp).pin_memory().to(self.device,
                                                    non_blocking=True)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = (
            torch.from_numpy(cu_num_draft_tokens).pin_memory().to(
                self.device, non_blocking=True))
        cu_num_sampled_tokens = (
            torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(
                self.device, non_blocking=True))
        logits_indices = (torch.from_numpy(logits_indices).pin_memory().to(
            self.device, non_blocking=True))
        target_logits_indices = (
            torch.from_numpy(target_logits_indices).pin_memory().to(
                self.device, non_blocking=True))
        bonus_logits_indices = torch.from_numpy(
            bonus_logits_indices).pin_memory().to(self.device,
                                                  non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        if self.pcp_size > 1:
            logits_indices = logits_indices_pcp
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: PerLayerAttnMetadata,
        aux_hidden_states: torch.Tensor = None,
    ) -> Optional[list[list[int]]]:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        else:
            # TODO: attn_metadata is only used in torchair generate_token_ids, check it
            draft_token_ids = self.drafter.generate_token_ids(
                valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, positions, num_scheduled_tokens,
                hidden_states, aux_hidden_states)
        return draft_token_ids

    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors] | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        # Record the header-overhead start time: everything from execute_model
        # entry up to this point (prepare_inputs, metadata, etc.) is "header".
        self._t_header_start = time.perf_counter()
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                ):
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors, max_query_len, ubatch_slices,
             split_batch_slices, num_tokens_after_padding,
             inplace_attention_backend) = (
                 self._prepare_inputs(scheduler_output, intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        # prevent debugger is None
        need_dump = self.dump_enable and self.debugger is not None
        if need_dump:
            assert self.debugger is not None
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(
                getattr(dbg_cfg, "level",
                        "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        if self.ascend_config.enable_async_exponential != 0:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators)

        split_ubatch_slices = None
        if split_batch_slices is not None:
            split_ubatch_slices = [
                UBatchSlice(s.request_slice, s.token_slice)
                for s in split_batch_slices
            ]

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens
            num_tokens_across_dp = num_tokens_after_padding

        model_kwargs = self._init_model_kwargs(maybe_padded_num_tokens)
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        split_mode = (getattr(split_cfg, "mode", "parallel_buffer")
                      if split_cfg is not None else "parallel_buffer")
        split_enable_parallel_streams = bool(split_cfg is not None
                                 and getattr(split_cfg, "enable_parallel_streams", False))
        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=self.with_prefill,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method,
                    ubatch_slices=(ubatch_slices or split_ubatch_slices),
            ):
                _set_split_debug_step(get_forward_context(),
                                      self._split_inplace_debug_step_id)
                dual_stream_attention_metadata = getattr(
                    self, "_dual_stream_attention_metadata", None)
                if dual_stream_attention_metadata is not None:
                    forward_context = get_forward_context()
                    setattr(forward_context,
                            "dual_stream_attention_metadata",
                            dual_stream_attention_metadata)
                    setattr(forward_context,
                            "dual_stream_attention_slices",
                            getattr(self,
                                    "_dual_stream_attention_slices", None))
                    setattr(forward_context,
                            "dual_stream_attention_plan",
                            getattr(self, "_dual_stream_attention_plan", None))
                    cfg = _dual_stream_attention_config(
                        getattr(self.ascend_config, "split_batch_config",
                                None))
                    setattr(
                        forward_context,
                        "dual_stream_attention_secondary_stream_mode",
                        getattr(cfg, "secondary_stream_mode",
                                "dedicated_pair"))
                self.maybe_setup_kv_connector(scheduler_output)

                if split_ubatch_slices is not None:
                    if split_mode == "inplace_serial":
                        if inplace_attention_backend == "mixed_request":
                            hidden_states = (
                                self._run_mixed_request_split_serial(
                                    split_batch_slices,
                                    attn_metadata,
                                    input_ids,
                                    positions,
                                    intermediate_tensors,
                                    inputs_embeds,
                                    model_kwargs,
                                    batch_descriptor))
                        else:
                            assert inplace_attention_backend in ("fia", "pa")
                            hidden_states = (
                                self._run_split_batch_inplace_serial(
                                    split_ubatch_slices,
                                    split_batch_slices,
                                    attn_metadata,
                                    input_ids,
                                    positions,
                                    intermediate_tensors,
                                    inputs_embeds,
                                    model_kwargs,
                                    batch_descriptor,
                                    aclgraph_runtime_mode,
                                    inplace_attention_backend))
                    elif split_mode == "inplace_parallel":
                        assert inplace_attention_backend in (
                            "fia", "pa", "mixed_request")
                        if not split_enable_parallel_streams:
                            raise RuntimeError(
                                "inplace_parallel execution requires "
                                "split_batch_config.enable_parallel_streams")
                        hidden_states = self._run_split_batch_inplace_parallel(
                            split_ubatch_slices,
                            split_batch_slices,
                            attn_metadata,
                            input_ids,
                            positions,
                            intermediate_tensors,
                            inputs_embeds,
                            model_kwargs,
                            batch_descriptor,
                            aclgraph_runtime_mode,
                            inplace_attention_backend)
                    elif split_enable_parallel_streams:
                        #logger.info("Running split batch with parallel streams, split_cfg=%s", split_cfg)
                        hidden_states = self._run_split_batch_parallel(
                            split_ubatch_slices,
                            split_batch_slices,
                            attn_metadata,
                            input_ids,
                            positions,
                            intermediate_tensors,
                            inputs_embeds,
                            model_kwargs,
                            batch_descriptor,
                            aclgraph_runtime_mode)
                        self._t_replay_end = time.perf_counter()
                    else:
                        #logger.info("Running split batch without parallel streams, split_cfg=%s", split_cfg)
                        self._t_replay_start = time.perf_counter()
                        hidden_states = self._run_split_batch_gr0(
                        split_ubatch_slices,
                        split_batch_slices,
                        attn_metadata,
                        input_ids,
                        positions,
                        intermediate_tensors,
                        inputs_embeds,
                        model_kwargs,
                        batch_descriptor,
                        aclgraph_runtime_mode,
                    )
                else:
                    #logger.info("Running without split batch")
                    hidden_states = self._generate_process_reqs_hidden_states(
                        maybe_padded_num_tokens, input_ids, positions,
                        intermediate_tensors, inputs_embeds)
                    self._t_replay_end = time.perf_counter()
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            # Accumulate perf stats for TPOT / header-overhead tracking.
            # Only count uniform-decode steps (1 output token per request).
            if uniform_decode:
                _header_ms = (self._t_replay_start - self._t_header_start) * 1000.0
                _replay_ms = (self._t_replay_end - self._t_replay_start) * 1000.0
                _n_tokens = self.input_batch.num_reqs
                self._last_step_perf = {
                    "header_ms": _header_ms,
                    "replay_ms": _replay_ms,
                    "batch_size": _n_tokens,
                    "is_split": split_ubatch_slices is not None,
                    "debug_step_id": self._split_inplace_debug_step_id,
                }
                self._perf_accum["total_header_ms"] += _header_ms
                self._perf_accum["total_replay_ms"] += _replay_ms
                self._perf_accum["total_output_tokens"] += _n_tokens
                self._perf_accum["num_decode_steps"] += 1
                _write_perf_stats(self._last_step_perf)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if need_dump:
                        assert self.debugger is not None
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)   
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    pool_output = self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np)
                    if need_dump:
                        assert self.debugger is not None
                        self.debugger.stop()
                        self.debugger.step()
                    return pool_output
                # Sometimes, after the model is compiled through the AOT backend,
                # the model output may become a list containing only one Tensor object.
                hidden_states = _unwrap_single_tensor_output(hidden_states)
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
            )
            self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        need_dump = self.dump_enable and self.debugger is not None
        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with ProfileExecuteDuration().capture_async("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert self.spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with ProfileExecuteDuration().capture_async("Draft"):
            if self.speculative_config:
                use_padded_batch_for_eagle = self.speculative_config and \
                    self.speculative_config.use_eagle() and \
                    not self.speculative_config.disable_padded_drafter_batch
                if use_padded_batch_for_eagle:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch_for_eagle:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        extra_args = ({"kv_connector_output": kv_connector_output})

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            if need_dump:
                assert self.debugger is not None
                self.debugger.stop()
                self.debugger.step()
            return model_runner_output

        if need_dump:
            assert self.debugger is not None
            self.debugger.stop()
            self.debugger.step()
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[:self.input_batch.num_reqs]
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[:len(spec_decode_metadata.logits_indices)]
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        if self.need_accepted_tokens:  # TODO remove this if
            self._update_states_after_model_execute(
                sampler_output.sampled_token_ids)
        return sampler_output

    # TODO: remove this func after eagle_proposer is refactored and
    #  _bookkeeping_sync is moved after propose_draft_token_ids
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
            LogprobsLists | None,
            list[list[int]],
            dict[str, LogprobsTensors | None],
            list[str],
            dict[str, int],
            list[int],
    ]:
        # TODO: implement PR 28597 from vllm
        discard_sampled_tokens_req_indices = \
            self.discard_request_indices.np[:self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    return_cu_num_tokens=logprobs_tensors is not None,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            if self.num_spec_tokens <= 0:
                assert sampled_token_ids.shape[-1] == 1
                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids

            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [
                    -1
                ] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (logprobs_tensors.tolists(cu_num_tokens)
                          if not self.use_async_scheduling
                          and logprobs_tensors is not None else None)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    def _build_dummy_attn_metadata(
        self,
        with_prefill: bool,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
        num_scheduled_tokens: np.ndarray,
        aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        ubatch_slices=None,
    ) -> Optional[PerLayerAttnMetadata]:

        attn_metadata: Optional[PerLayerAttnMetadata] = None

        if force_attention or aclgraph_runtime_mode == CUDAGraphMode.FULL:
            assert with_prefill is False, \
                "Full decode graph only supports uniform batch now."

            attn_metadata = {}
            dual_stream_attention_plan = self._select_dual_stream_attention_plan(
                total_num_scheduled_tokens=int(num_tokens),
                graph_num_tokens=int(num_tokens),
                uniform_decode=bool(max_query_len
                                    == self.uniform_decode_query_len),
                with_prefill=bool(with_prefill),
                ubatch_slices=ubatch_slices,
                has_spec_decode_tokens=bool(self.speculative_config),
                has_lora=bool(self.lora_config),
                attn_state=AscendAttentionState.DecodeOnly,
                allow_missing_plan=True,
            )
            dual_stream_attention_slices = None
            dual_stream_attention_ubatch_slices = None
            dual_stream_attention_metadata: Optional[
                list[PerLayerAttnMetadata]] = None
            if dual_stream_attention_plan is not None:
                (dual_stream_attention_slices,
                 dual_stream_attention_ubatch_slices) = (
                     _dual_stream_attention_plan_to_slices(
                         dual_stream_attention_plan,
                         self.uniform_decode_query_len))
                dual_stream_attention_metadata = [dict() for _ in range(2)]
                self._dual_stream_attention_plan = dual_stream_attention_plan
                self._dual_stream_attention_slices = (
                    dual_stream_attention_slices)
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cu_num_tokens, arange = self._get_cumsum_and_arange(
                num_scheduled_tokens)

            self.query_start_loc.cpu[1:num_reqs +
                                     1] = torch.Tensor(cu_num_tokens)
            self.query_lens = torch.from_numpy(num_scheduled_tokens)
            self.attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask()

            num_computed_tokens_cpu = (
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                block_table_tensor = self.input_batch.block_table[
                    kv_cache_group_id].get_device_tensor()
                slot_mapping = self.input_batch.block_table[
                    kv_cache_group_id].slot_mapping
                self.cp_kv_recover_idx = torch.zeros(self.max_num_tokens,
                                                     dtype=torch.int32,
                                                     device=self.device)
                long_seq_metadata = self._generate_pcp_metadata(num_tokens)
                if long_seq_metadata is not None:
                    pcp_world_size = get_pcp_group().world_size
                    dcp_world_size = get_dcp_group().world_size
                    num_computed_tokens_of_pcp_dcp = [[
                        [0] * dcp_world_size for _ in range(pcp_world_size)
                    ] for _ in range(num_tokens)]
                    long_seq_metadata.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp
                # QUESTION: Why do we separately set query_start_loc for spec in the first place?
                # While in _prepare_inputs we don't?
                if self.speculative_config:
                    self.query_start_loc.gpu[:num_reqs + 1] = torch.tensor(
                        [0] + self.actual_seq_lengths_q[:num_reqs],
                        device=self.device,
                        dtype=torch.int32)
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens_cpu=self.seq_lens.cpu,
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    actual_seq_lengths_q=self.actual_seq_lengths_q,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    positions=self.positions.gpu,
                    attn_mask=self.attn_mask,
                    spec_attn_mask=self.spec_attn_mask,
                    attn_state=self.attn_state,
                    max_query_len=max_query_len,
                    decode_token_per_req=self.decode_token_per_req,
                    prefill_context_parallel_metadata=long_seq_metadata,
                )
                if self.pcp_size > 1:
                    common_attn_metadata.block_table_tensor = \
                        block_table_tensor[:num_reqs * self.decode_threshold]
                attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and \
                        self.speculative_config.method == "mtp":
                    attn_state = AscendAttentionState.SpecDecoding

                common_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    _seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    seq_lens=self.seq_lens.cpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    _num_computed_tokens_cpu=num_computed_tokens_cpu,
                    max_query_len=max_query_len,
                    max_seq_len=seq_lens)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    builder = attn_group.get_metadata_builder()
                    if ubatch_slices is not None:
                        # TODO: check dummy attn construct logic
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata,
                            self.max_num_tokens)
                        _validate_split_attn_metadata_count(
                            "dummy_capture",
                            common_attn_metadata_list,
                            len(ubatch_slices),
                        )
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = (attn_group\
                                               .get_metadata_builder(ubatch_id=ubid)\
                                               .build_for_cudagraph_capture(common_attn_metadata, attn_state, self.get_model()))
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        if isinstance(builder, GDNAttentionMetadataBuilder):
                            if dual_stream_attention_metadata is not None:
                                raise RuntimeError(
                                    "dual_stream_attention_config does not "
                                    "support GDN/linear attention layers")
                            attn_metadata_gdn_attention = builder.build_for_cudagraph_capture(
                                common_metadata)
                        else:
                            attn_metadata_full_attention = builder.build_for_graph_capture(
                                common_attn_metadata, attn_state,
                                self.get_model())
                        for layer_name in kv_cache_group_spec.layer_names:
                            if "linear_attn" in layer_name:
                                attn_metadata[
                                    layer_name] = attn_metadata_gdn_attention
                            else:
                                attn_metadata[
                                    layer_name] = attn_metadata_full_attention
                        if (dual_stream_attention_metadata is not None
                                and not isinstance(
                                    builder, GDNAttentionMetadataBuilder)):
                            assert dual_stream_attention_ubatch_slices is not None
                            assert dual_stream_attention_slices is not None
                            dual_common_attn_metadata_list = split_attn_metadata(
                                dual_stream_attention_ubatch_slices,
                                common_attn_metadata,
                                self.max_num_tokens)
                            dual_common_attn_metadata_list = (
                                self.
                                _stabilize_dual_stream_common_attn_metadata_list(
                                    dual_common_attn_metadata_list,
                                    dual_stream_attention_slices))
                            _validate_split_attn_metadata_count(
                                "dummy_dual_stream_attention",
                                dual_common_attn_metadata_list,
                                len(dual_stream_attention_ubatch_slices),
                            )
                            for ubid, split_common_attn_metadata in enumerate(
                                    dual_common_attn_metadata_list):
                                dual_attn_metadata_i = (
                                    builder.build_for_graph_capture(
                                        split_common_attn_metadata,
                                        attn_state,
                                        self.get_model()))
                                self._apply_dual_stream_fia_actual_seq_lengths_q(
                                    dual_attn_metadata_i,
                                    split_common_attn_metadata)
                                for layer_name in kv_cache_group_spec.layer_names:
                                    if "linear_attn" in layer_name:
                                        continue
                                    dual_stream_attention_metadata[ubid][
                                        layer_name] = dual_attn_metadata_i

            if dual_stream_attention_plan is not None:
                if dual_stream_attention_metadata is None:
                    raise RuntimeError(
                        "dual_stream_attention_config selected a dummy plan "
                        "but did not build split attention metadata")
                self._dual_stream_attention_metadata = (
                    dual_stream_attention_metadata)

        return attn_metadata

    def _generate_dummy_run_hidden_states(self, input_ids, positions,
                                          num_tokens, intermediate_tensors,
                                          inputs_embeds):
        hidden_states = self.model(input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            )
        forward_context = get_forward_context()
        assert forward_context is not None
        model_updates_attn_params_internally = bool(
            getattr(self.model, "updates_attn_params_internally", False))
        if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not forward_context.capturing and not self.use_sparse
                and not model_updates_attn_params_internally):
            if self.vllm_config.model_config.use_mla:
                # FIXME: Try using `auto_dispatch_capture=True`
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   num_tokens)
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    # When using npu_split_wrapper, attn_metadata is a list (one per ubatch)
                    if isinstance(forward_context.attn_metadata, list):
                        for ubatch_attn_metadata in forward_context.attn_metadata:
                            temp_context = type('obj', (object,), {
                                'attn_metadata': ubatch_attn_metadata,
                                'is_mtp_model': forward_context.is_mtp_model,
                                'capturing': forward_context.capturing,
                                'is_mla_model': forward_context.is_mla_model
                            })
                            update_mla_attn_params(self.update_stream, temp_context,
                                                   num_tokens,
                                                   self.vllm_config.speculative_config)
                    else:
                        update_mla_attn_params(self.update_stream, forward_context,
                                               num_tokens,
                                               self.vllm_config.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               num_tokens)
                else:
                    # When using npu_split_wrapper, attn_metadata is a list (one per ubatch)
                    if isinstance(forward_context.attn_metadata, list):
                        for ubatch_attn_metadata in forward_context.attn_metadata:
                            temp_context = type('obj', (object,), {
                                'attn_metadata': ubatch_attn_metadata,
                                'is_mtp_model': forward_context.is_mtp_model,
                                'capturing': forward_context.capturing,
                                'is_mla_model': forward_context.is_mla_model
                            })
                            update_attn_params(self.update_stream, temp_context,
                                               num_tokens,
                                               self.vllm_config)
                    else:
                        update_attn_params(self.update_stream, forward_context,
                                           num_tokens,
                                           self.vllm_config)


        if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
            hidden_states, _ = hidden_states
        else:
            hidden_states = hidden_states
        return hidden_states

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        allow_microbatching: bool = True,
        in_parallel_streams: bool=False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        self._dual_stream_attention_metadata = None
        self._dual_stream_attention_slices = None
        self._dual_stream_attention_plan = None
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        # In multi-DP scenarios, there may be situations where all DP groups are executing dummy runs.
        # If sequence parallelism is enabled, it is essential to ensure that num_tokens is divisible by tp_size.
        if self.use_aclgraph and enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer and not self.is_kv_consumer:
            with_prefill = True

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.seperate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.max_num_reqs
        # TODO: create_mixed_batch should be Fasle in Ascend now
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_reqs = cdiv(num_tokens, 1)
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.max_num_reqs
        # TODO: create_mixed_batch should be Fasle in Ascend now
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                num_reqs = (num_tokens + self.decode_token_per_req -
                            1) // self.decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        # dbo
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())
        ubatch_slices = None
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        dual_stream_attention_enabled = _dual_stream_attention_enabled(
            split_cfg)

        moe_comm_type = select_moe_comm_method(num_tokens, self.vllm_config)
        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, _ = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
                moe_comm_type=moe_comm_type,
            )
         # Split batch - compute split slices for large decode batches (similar to _prepare_inputs)
        # Split batch and DBO never conflict by design
        if (not dual_stream_attention_enabled and uniform_decode
                and ubatch_slices is None):  # Only split if DBO is not active
            cudagraph_capture_sizes = set(
                self.compilation_config.cudagraph_capture_sizes or []
            ) if self.use_aclgraph else None
            ubatch_slices, _ = split_batch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                vllm_config=self.vllm_config,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            )

        # Padding for DP
        # currently, we check the dp scenario that some ranks have tokens
        # but others execute dummy run
        if ubatch_slices is not None:
            enable_dbo = True
        else:
            enable_dbo = False

        (num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_tokens, with_prefill,
                                                     enable_dbo)
        moe_comm_type = select_moe_comm_method(num_tokens, self.vllm_config)
        if not enable_dbo:
            ubatch_slices = None
            

       

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = (batch_descriptor.num_reqs if
                           batch_descriptor.num_reqs is not None else num_reqs)
        if uniform_decode:
            expected_num_reqs_padded = cdiv(num_tokens_padded, max_query_len)
            if num_reqs_padded != expected_num_reqs_padded:
                num_reqs_padded = expected_num_reqs_padded
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        # filter out the valid batch descriptor
        if aclgraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if aclgraph_runtime_mode != CUDAGraphMode.NONE and aclgraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}.")
        else:
            aclgraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            force_attention=force_attention,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens,
                                            num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model:
                input_ids = None
                if in_parallel_streams:
                    inputs_embeds = self.inputs_embeds_parallel_streams.gpu[:num_tokens_padded]
                else:
                    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif self.enable_prompt_embeds:
                input_ids = None
                if in_parallel_streams:
                    inputs_embeds = self.inputs_embeds_parallel_streams.gpu[:num_tokens_padded]
                else:
                    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                if in_parallel_streams:
                    input_ids = self.input_ids_parallel_streams.gpu[:num_tokens_padded]
                else:
                    input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            else:
                if in_parallel_streams:
                    positions = self.positions_parallel_streams.gpu[:num_tokens_padded]
                else:
                    positions = self.positions.gpu[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by tp_size;
                # otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading to incorrect memory estimation and potentially causing OOM.
                actual_tokens = num_tokens
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    actual_tokens = num_tokens // tp_size
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=actual_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k:
                    v[:num_tokens_padded]
                    for k, v in self.intermediate_tensors.items()
                })

            need_dummy_logits = (not is_profile and lmhead_tp_enable())
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                        dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(
                        self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(
                        hidden_states[dummy_indices])

            with set_ascend_forward_context(
                    # When capturing for the parallel stream, clone block_tables
                    # in every per-layer AscendMetadata so that
                    # _graph_params_parallel binds a *different* device address
                    # than _graph_params.  Without this, both graph params point
                    # to the same block_table storage, and the two concurrent
                    # _refresh_block_table_in_place calls at runtime both write
                    # to block_table[:8, :], causing a data race that corrupts
                    # the KV-cache lookup for split-0.
                    _clone_attn_metadata_block_tables(attn_metadata)
                    if in_parallel_streams else attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    in_profile_run=is_profile,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method,
                    in_parallel_streams=in_parallel_streams,
                    ubatch_slices=ubatch_slices,):
                dual_stream_attention_metadata = getattr(
                    self, "_dual_stream_attention_metadata", None)
                if dual_stream_attention_metadata is not None:
                    forward_context = get_forward_context()
                    setattr(forward_context,
                            "dual_stream_attention_metadata",
                            dual_stream_attention_metadata)
                    setattr(forward_context,
                            "dual_stream_attention_slices",
                            getattr(self,
                                    "_dual_stream_attention_slices", None))
                    setattr(forward_context,
                            "dual_stream_attention_plan",
                            getattr(self, "_dual_stream_attention_plan", None))
                    cfg = _dual_stream_attention_config(
                        getattr(self.ascend_config, "split_batch_config",
                                None))
                    setattr(
                        forward_context,
                        "dual_stream_attention_secondary_stream_mode",
                        getattr(cfg, "secondary_stream_mode",
                                "dedicated_pair"))
                with torch.npu.stream(self.stream_parallel if in_parallel_streams else self.stream_main):
                    hidden_states = self._generate_dummy_run_hidden_states(
                        input_ids, positions, num_tokens_padded,
                        intermediate_tensors, inputs_embeds)
                dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile)
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()
            return hidden_states, hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = None

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * self.max_num_reqs
        num_scheduled_tokens_list[
            -1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        # Sometimes, after the model is compiled through the AOT backend,
        # the model output may become a list containing only one Tensor object.
        hidden_states = _unwrap_single_tensor_output(hidden_states)
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states[logit_indices]
            output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and \
            select_moe_comm_method(mc2_tokens_capacity, self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity,
                            with_prefill=True,
                            is_profile=True)
        super().profile_run()

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.dynamic_eplb:
                model_register(self.model, self.model_config)
            if self.drafter:
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
                if self.drafter.name == SpecDcodeType.EAGLE3:
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # Wrap model with the correct runtime wrapper.
        # Priority (mutually exclusive at runtime):
        # - DBO: AscendUBatchWrapper
        # - Split-batch: handled in execute_model
        # - Full graph only: ACLGraphWrapper
        split_enabled = bool(
            getattr(self.ascend_config, "split_batch_config",
                    None) is not None
            and self.ascend_config.split_batch_config.enabled)

        if self.parallel_config.enable_dbo:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.model = AscendUBatchWrapper(self.model, self.vllm_config,
                                                 CUDAGraphMode.FULL,
                                                 self.device)
            else:
                self.model = AscendUBatchWrapper(self.model, self.vllm_config,
                                                 CUDAGraphMode.NONE,
                                                 self.device)
        elif self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.update_stream_main: torch.npu.Stream = torch.npu.Stream()
            self.update_stream_parallel: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL,
                                         device=self.device)


    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = (len(self.attn_groups) > 1)
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any([
            isinstance(attn_group[0].kv_cache_spec, MambaSpec)
            for attn_group in self.attn_groups
        ])

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def _align_memory(self, tensor: torch.Tensor,
                      alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        from vllm.v1.worker.utils import bind_kv_cache
        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)
        return kv_caches

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cahce and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
         """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, Union[torch.Tensor,
                                              Optional[torch.Tensor]]] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # for mamba linear attention
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size,
                                             dtype=torch.int8,
                                             device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned,
                                             dtype=torch.int8,
                                             device=self.device)
                        tensor = self._align_memory(
                            tensor, alignment)[:kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    if self.model_config.use_mla:
                        head_size = self.model_config.hf_text_config.qk_rope_head_dim + \
                            self.model_config.hf_text_config.kv_lora_rank

                    dsa_k_cache_factor = None
                    dsa_k_cache_size = None
                    if not self.model_config.use_mla:
                        # for non-mla model, use FullAttentionSpec
                        k_tensor_split_factor = 2
                        v_tensor_split_factor = 2
                    elif self.use_sparse:
                        # for deepseek v3.2, DSA use FullAttentionSpec
                        # FullAttentionSpec allocate 2 * mla page size bytes,
                        # and we use half of that for k cache in DSA
                        dsa_k_cache_factor = 2
                        k_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.qk_rope_head_dim
                        dsa_k_cache_size = int(kv_cache_tensor.size //
                                               dsa_k_cache_factor)
                    else:
                        # for other deepseek models, use MLAAttentionSpec
                        k_tensor_split_factor = head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = head_size / self.model_config.hf_text_config.qk_rope_head_dim

                    k_tensor_size = int(kv_cache_tensor.size //
                                        k_tensor_split_factor)
                    v_tensor_size = int(kv_cache_tensor.size //
                                        v_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size,
                                dtype=torch.int8,
                                device=self.device)
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        k_tensor = self._align_memory(
                            k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(
                            v_tensor, alignment)[:v_tensor_size]
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None and dsa_k_cache_size is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size + alignment,
                                dtype=torch.int8,
                                device=self.device)
                            dsa_k_cache_tensor = self._align_memory(
                                dsa_k_cache_tensor,
                                alignment)[:dsa_k_cache_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if ("attn" in layer_name_inner
                                and "linear_attn" not in layer_name_inner):
                            kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor) if \
                                not self.use_sparse else (k_tensor, v_tensor, dsa_k_cache_tensor)

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    raw_dsa_k_tensor = None
                    if self.use_sparse:
                        raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        assert raw_dsa_k_tensor is not None
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_block_size"
                               ) and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_block_size()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk, block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    if not self.model_config.use_mla:
                        k_shape = kv_cache_shape[1:]
                        v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.kv_lora_rank
                        ]
                        v_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.qk_rope_head_dim
                        ]
                    k_cache = raw_k_tensor.view(dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(dtype).view(v_shape)
                    if get_ascend_device_type() == AscendDeviceType._310P:
                        k_cache = maybe_trans_nz(k_cache)
                        v_cache = maybe_trans_nz(v_cache)
                    if self.use_sparse and raw_dsa_k_tensor is not None:
                        dsa_k_cache_shape = (num_blocks,
                                             kv_cache_spec.block_size, 1, 128)
                        dsa_k_cache_size = (
                            num_blocks
                        ) * kv_cache_spec.block_size * 128 * dtype.itemsize
                        dsa_k_cache = raw_dsa_k_tensor[:dsa_k_cache_size].view(
                            dtype).view(dsa_k_cache_shape)
                        kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel(
                    ) % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel(
                    ) // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec,
                              EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(
                kv_cache_config.kv_cache_groups):

            if isinstance(kv_cache_group.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_group.kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # the backend.
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                except IndexError:
                    attn_groups = None
                if attn_groups and self.use_hybrid_blocks:
                    # Use the backend's supported block size list
                    backend = attn_groups[0].backend
                    supported_sizes = backend.get_supported_block_size()
                    # If no specific sizes supported, use cache config
                    # block_size
                    kernel_block_size_list = (supported_sizes
                                              if supported_sizes else
                                              [self.cache_config.block_size])
                else:
                    # Fallback to cache config block_size if no backend found
                    kernel_block_size_list = [self.cache_config.block_size]
                kernel_block_sizes.append(kernel_block_size_list)
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                kernel_block_sizes.append([0])
        if block_sizes != [
                self.cache_config.block_size
        ] or kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.model_config.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
                kernel_block_sizes=kernel_block_sizes,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> dict[AttentionGroupKey, list[str]]:
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,
                kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                        layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend,
                                                       layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return {
                attn_backends[k]: v
                for k, v in attn_backend_layers.items()
            }
        def _get_num_attn_metadata_builders() -> int:
            """How many metadata builders we need per AttentionGroup.

            - DBO(ubatch): 2 builders (ubatch_id=0/1).
            - Split-batch: N builders, N = split_config.num_splits.
            - Default: 1 builder.
            """
            if self.parallel_config.enable_dbo:
                return 2

            split_cfg = getattr(self.ascend_config, "split_batch_config", None)
            split_enabled = bool(split_cfg is not None
                                 and getattr(split_cfg, "enabled", False))
            if split_enabled:
                n = int(getattr(split_cfg, "num_splits", 1))
                return max(1, n)

            return 1

        def create_attn_groups(attn_backends_map: dict[AttentionBackend,
                                                       list[str]],
                               kv_cache_group_id: int) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend,
                 kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = [
                    attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        layer_names,
                        self.vllm_config,
                        self.device,
                    ) for _ in range(_get_num_attn_metadata_builders())
                ]
                attn_group = AttentionGroup(attn_backend, layer_names,
                                            kv_cache_spec, kv_cache_group_id,
                                            attn_metadata_builders)
                attn_groups.append(attn_group)
            return attn_groups

        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            attn_backends = get_attn_backends_for_group(  # type: ignore
                kv_cache_group_spec)
            self.attn_groups.append(create_attn_groups(attn_backends, i))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()
            if hasattr(attn_metadata_builder_i,
                       "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = (
                    attn_metadata_builder_i.reorder_batch_threshold)
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != \
                            self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}")
                    else:
                        self.reorder_batch_threshold = reorder_batch_threshold_i  # noqa

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                  AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                # TODO: Support other attention modules, e.g., cross-attention
                # TODO(lucas): move the attention specs into the model layers like
                # the attention backends
                if attn_module.attn_type == AttentionType.DECODER:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                elif attn_module.attn_type in (AttentionType.ENCODER,
                                               AttentionType.ENCODER_ONLY):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")

            elif isinstance(attn_module, MLAAttention):
                if use_mla and not self.use_sparse:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.cache_config.cache_dtype)
                else:
                    # TODO(cmq): This is a hack way to fix deepseek kvcache when
                    # using DSA. Fix the spec in vLLM is a finnal way.
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)

        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            if (self.vllm_config.speculative_config is not None
                    and self.vllm_config.model_config.hf_config.model_type
                    not in ["qwen3_next"]):
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len

            page_size_padded = (
                self.vllm_config.cache_config.mamba_page_size_padded)

            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtypes=mamba_module.get_state_dtype(),
                    block_size=max_model_len,
                    page_size_padded=page_size_padded,
                    mamba_type=mamba_module.mamba_type,
                    num_speculative_blocks=(
                        self.speculative_config.num_speculative_tokens
                        if self.speculative_config else 0),
                )

        return kv_cache_spec

    def initialize_aclgraph_capture(self) -> None:
        min_ag_support = AttentionCGSupport.ALWAYS
        min_ag_builder_name = None

        for attn_group in self._attn_group_iterator():
            builder = attn_group.get_metadata_builder()
            graph_support = None
            if hasattr(builder, 'aclgraph_support'):
                graph_support = builder.aclgraph_support.value
                builder_aclgraph = builder.aclgraph_support
            else:
                graph_support = builder._cudagraph_support.value
                builder_aclgraph = builder._cudagraph_support
            if graph_support < min_ag_support.value:
                min_ag_support = builder_aclgraph
                min_ag_builder_name = builder.__class__.__name__

        # This is an imitation of compilation_config.splitting_ops_contain_attention()
        splitting_ops_contain_attention = (
            self.compilation_config.splitting_ops is not None
            and all(op in self.compilation_config.splitting_ops for op in [
                "vllm.mla_forward",
            ]))

        # Flexible resolve the aclgraph mode
        aclgraph_mode = self.compilation_config.cudagraph_mode
        # check graph for mixed batch is supported
        if aclgraph_mode.mixed_mode() == CUDAGraphMode.FULL \
            and min_ag_support != AttentionCGSupport.ALWAYS:
            msg = (f"ACLGraphMode.{aclgraph_mode.name} is not supported "
                   f"with {min_ag_builder_name} backend (support: "
                   f"{min_ag_support})")
            if min_ag_support == AttentionCGSupport.NEVER:
                # if not supported any full graphs, just raise it.
                msg += "; please try cudagraph_mode=PIECEWISE, and "\
                    "make sure compilation level is piecewise"
                raise ValueError(msg)

            # attempt to resolve the full graph related mode
            if splitting_ops_contain_attention:
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                aclgraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE)
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                aclgraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY)
            logger.warning(msg)

        # double check that we can support full graph if they are requested
        # even after automatic downgrades
        if aclgraph_mode.has_full_cudagraphs() \
            and min_ag_support == AttentionCGSupport.NEVER:
            raise ValueError(f"CUDAGraphMode.{aclgraph_mode.name} is not "
                             f"supported with {min_ag_builder_name} backend ("
                             f"support:{min_ag_support}) "
                             "; please try cudagraph_mode=PIECEWISE, "
                             "and make sure compilation level is piecewise")

        if (aclgraph_mode.decode_mode() == CUDAGraphMode.FULL
                and aclgraph_mode.separate_routine()
                and self.uniform_decode_query_len > 1):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len,
                self.parallel_config.tensor_parallel_size)
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            self.cudagraph_batch_sizes = (capture_sizes
                                          if capture_sizes is not None else [])

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        set_graph_params(self.cudagraph_batch_sizes)
        # Parallel stream needs its own GraphParams to avoid races when both
        # streams call graph_task_update_begin/end concurrently.
        # If split_batch_config.parallel_capture_sizes is set, use those sizes
        # for the parallel-stream graph pool; otherwise fall back to the main
        # capture sizes (legacy behaviour).
        _split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        _parallel_sizes = (
            _split_cfg.parallel_capture_sizes
            if _split_cfg is not None
            and _split_cfg.parallel_capture_sizes is not None
            else self.cudagraph_batch_sizes
        )
        self.cudagraph_batch_sizes_parallel = _parallel_sizes
        set_graph_params_parallel(self.cudagraph_batch_sizes_parallel)
        if self.speculative_config:
            set_mtp_graph_params(self.cudagraph_batch_sizes)

        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len)

    def _capture_aclgraphs(self, compilation_cases: list[int],
                           aclgraph_runtime_mode: CUDAGraphMode,
                           uniform_decode: bool,
                           in_parallel_streams: bool = False
                           ):
        assert aclgraph_runtime_mode != CUDAGraphMode.NONE and \
            aclgraph_runtime_mode in [CUDAGraphMode.FULL,
                                      CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            logger.info(
                "Starting to capture ACL graphs for cases: %s, "
                "mode: %s, uniform_decode: %s", compilation_cases,
                aclgraph_runtime_mode.name, uniform_decode)
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing ACL graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    aclgraph_runtime_mode.name))

        force_attention = (aclgraph_runtime_mode == CUDAGraphMode.FULL)
        # When the kv cache spec is empty, PiecewiseBackend is not initialized, and
        # compilation_case=1 will cause the dynamic shape position to be incorrectly derived.
        if not self.get_kv_cache_spec():
            self._dummy_run(2,
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            force_attention=force_attention,
                            uniform_decode=uniform_decode,
                            in_parallel_streams=in_parallel_streams,)
        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens in compilation_cases:

            allow_microbatching = self.parallel_config.enable_dbo \
                and aclgraph_runtime_mode == CUDAGraphMode.FULL \
                and uniform_decode \
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                self._dummy_run(num_tokens,
                                aclgraph_runtime_mode=None,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode,
                                allow_microbatching=allow_microbatching,
                                in_parallel_streams=in_parallel_streams,
                                # allow_parallel_streams=False
                                )
            #真正捕获的运行
            self._dummy_run(num_tokens,
                            aclgraph_runtime_mode=aclgraph_runtime_mode,
                            force_attention=force_attention,
                            uniform_decode=uniform_decode,
                            allow_microbatching=allow_microbatching,
                            in_parallel_streams=in_parallel_streams,
                            )

    def _capture_multistream_macro_graphs(self) -> None:
        macro_graph_cfg = self._macro_graph_config()
        if macro_graph_cfg is None or not getattr(macro_graph_cfg, "enabled",
                                                  False):
            return

        inplace_plans = self._macro_graph_capture_inplace_plans(
            macro_graph_cfg)
        if len(inplace_plans) > int(
                getattr(macro_graph_cfg, "max_capture_graphs", 16)):
            raise RuntimeError(
                "macro graph planned capture count exceeds "
                "max_capture_graphs: "
                f"{len(inplace_plans)} > "
                f"{getattr(macro_graph_cfg, 'max_capture_graphs', 16)}")

        logger.info(
            "Starting multistream macro graph precapture: schedule=%s, "
            "backend=%s, plan_source=%s, num_plans=%d",
            getattr(macro_graph_cfg, "schedule", None),
            getattr(macro_graph_cfg, "backend", None),
            getattr(macro_graph_cfg, "plan_source", None),
            len(inplace_plans),
        )
        self._macro_graph_registry.clear()
        capture_plans = list(getattr(macro_graph_cfg, "capture_plans", []))
        for plan_idx, plan in enumerate(inplace_plans):
            split_batch_slices = plan.split_slices
            if getattr(plan, "offset_match_policy", "") == "compact":
                inplace_attention_backend = "mixed_request"
            else:
                inplace_attention_backend = select_inplace_attention_backend(
                    plan,
                    lambda shape: using_paged_attention(shape,
                                                        self.vllm_config),
                )
            key = self._macro_graph_key_from_split_slices(
                split_batch_slices,
                inplace_attention_backend,
                uniform_decode=(inplace_attention_backend != "mixed_request"),
            )
            if key in self._macro_graph_registry:
                raise RuntimeError(
                    "Duplicate multistream macro graph key while planning "
                    f"capture: {key!r}")
            self._macro_graph_registry[key] = _PlannedMacroGraphEntry(
                key=key,
                plan=plan,
                inplace_attention_backend=inplace_attention_backend,
                split_req_caps=(
                    _macro_capture_plan_req_caps(
                        capture_plans[plan_idx],
                        uniform_decode_query_len=self.uniform_decode_query_len)
                    if plan_idx < len(capture_plans) else None),
            )
            logger.info(
                "Planned multistream macro graph: backend=%s, key=%s, "
                "actual_tokens=%s, graph_tokens=%s, start_tokens=%s, "
                "req_caps=%s",
                getattr(macro_graph_cfg, "backend", None),
                key,
                tuple(int(s.num_tokens) for s in split_batch_slices),
                tuple(int(s.graph_num_tokens) for s in split_batch_slices),
                tuple(int(s.start_num_tokens) for s in split_batch_slices),
                (self._macro_graph_registry[key].split_req_caps),
            )
        logger.warning(
            "Multistream macro graphs are planned at load time and "
            "materialized with backend=%s on first runtime hit to bind stable "
            "vLLM metadata addresses. Use benchmark warmup steps before "
            "measuring steady-state performance.",
            getattr(macro_graph_cfg, "backend", None),
        )

    def _capture_macro_mixed_piecewise_aclgraphs(self) -> None:
        macro_graph_cfg = self._macro_graph_config()
        if macro_graph_cfg is None or not getattr(macro_graph_cfg, "enabled",
                                                  False):
            return
        inplace_plans = self._macro_graph_capture_inplace_plans(
            macro_graph_cfg)
        main_sizes: set[int] = set()
        parallel_sizes: set[int] = set()
        for plan in inplace_plans:
            if getattr(plan, "offset_match_policy", "") != "compact":
                continue
            main_sizes.add(int(plan.total_num_tokens))
            for idx, split_slice in enumerate(plan.split_slices):
                target = parallel_sizes if idx > 0 else main_sizes
                target.add(int(split_slice.graph_num_tokens))
        # Macro graph replay still needs the normal PIECEWISE graph pool for
        # serving steps that do not hit an exact macro key, including the first
        # prefill-only request and miss_policy="padding" fallback paths.
        base_capture_sizes = (
            self.compilation_config.cudagraph_capture_sizes or [])
        main_sizes.update(int(size) for size in base_capture_sizes
                          if int(size) > 0)
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        if split_cfg is not None and bool(
                getattr(split_cfg, "enable_parallel_streams", False)):
            parallel_capture_sizes = (
                getattr(split_cfg, "parallel_capture_sizes", None)
                or base_capture_sizes)
            parallel_sizes.update(int(size) for size in parallel_capture_sizes
                                  if int(size) > 0)
        if not main_sizes and not parallel_sizes:
            return
        if not self.use_aclgraph:
            raise RuntimeError(
                "mixed_request macro graph requires ACL graph support for "
                "inner PIECEWISE attention graphs")

        self.initialize_aclgraph_capture()
        set_cudagraph_capturing_enabled(True)
        try:
            if main_sizes:
                with graph_capture(device=self.device):
                    self._capture_aclgraphs(
                        compilation_cases=list(reversed(sorted(main_sizes))),
                        aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                        uniform_decode=False,
                        in_parallel_streams=False,
                    )
            if parallel_sizes:
                with graph_capture(device=self.device):
                    self._capture_aclgraphs(
                        compilation_cases=list(
                            reversed(sorted(parallel_sizes))),
                        aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                        uniform_decode=False,
                        in_parallel_streams=True,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

    def _capture_model(self):
        if self._macro_graph_enabled():
            self._capture_macro_mixed_piecewise_aclgraphs()
            self._capture_multistream_macro_graphs()
            return

        if not self.use_aclgraph:
            logger.warning(
                "Skipping ACL graph capture. To turn on ACL graph capture, "
                "ensure `aclraph_mode` was not manually set to `NONE`")
            return
        else:
            self.initialize_aclgraph_capture()

        set_cudagraph_capturing_enabled(True)
        # Trigger ACL graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        aclgraph_mode = self.compilation_config.cudagraph_mode

        # First capture (original cases)
        with graph_capture(device=self.device):
            if aclgraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                aclgraph_runtime_mode = aclgraph_mode.mixed_mode()
                # make sure we capture the largest batch size first
                compilation_cases = list(reversed(self.cudagraph_batch_sizes))

                try:
                    self._capture_aclgraphs(
                        compilation_cases,
                        aclgraph_runtime_mode=aclgraph_runtime_mode,
                        uniform_decode=False)
                except Exception as e:
                    error_msg = str(e)
                    error_code = '0x7020023'
                    pattern = r'retCode=([^,\s\.]+)'
                    match = re.search(pattern, error_msg)
                    if match:
                        retCode = match.group(1)
                    # Determine whether the error message is caused by stream capture failure.
                    if match and retCode == error_code:
                        logger.error(
                            f"ACLgraph sizes capture fail: {type(e).__name__}:\n"
                            "ACLgraph has insufficient available streams to capture the configured number of sizes. "
                            "Please verify both the availability of adequate streams and the appropriateness of the configured size count.\n\n"
                            "Recommended solutions:\n"
                            "1. Manually configure the compilation_config parameter "
                            "with a reduced set of sizes: '{\"cudagraph_capture_sizes\":[size1, size2, size3, ...]}'.\n"
                            "2. Utilize ACLgraph's full graph mode as an alternative to the piece-wise approach.\n\n"
                            f"{str(e)}")
                    raise

            if aclgraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                aclgraph_mode.separate_routine():
                max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    x <= max_num_tokens and x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                self._capture_aclgraphs(
                    compilation_cases=compilation_cases_decode,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True,
                    in_parallel_streams=False)

        # Second capture (parallel-stream graph pool).
        # Uses self.cudagraph_batch_sizes_parallel which is set in
        # initialize_aclgraph_capture() and respects
        # split_batch_config.parallel_capture_sizes when provided.
        if self.ascend_config.split_batch_config.enable_parallel_streams:
            with graph_capture(device=self.device):
                if aclgraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                    aclgraph_runtime_mode = aclgraph_mode.mixed_mode()
                    # make sure we capture the largest batch size first
                    compilation_cases = list(
                        reversed(self.cudagraph_batch_sizes_parallel))

                    try:
                        self._capture_aclgraphs(
                            compilation_cases,
                            aclgraph_runtime_mode=aclgraph_runtime_mode,
                            uniform_decode=False,
                            in_parallel_streams=True)
                    except Exception as e:
                        error_msg = str(e)
                        error_code = '0x7020023'
                        pattern = r'retCode=([^,\s\.]+)'
                        match = re.search(pattern, error_msg)
                        if match:
                            retCode = match.group(1)
                        # Determine whether the error message is caused by stream capture failure.
                        if match and retCode == error_code:
                            logger.error(
                                f"ACLgraph sizes capture fail: {type(e).__name__}:\n"
                                "ACLgraph has insufficient available streams to capture the configured number of sizes. "
                                "Please verify both the availability of adequate streams and the appropriateness of the configured size count.\n\n"
                                "Recommended solutions:\n"
                                "1. Manually configure the compilation_config parameter "
                                "with a reduced set of sizes: '{\"cudagraph_capture_sizes\":[size1, size2, size3, ...]}'.\n"
                                "2. Utilize ACLgraph's full graph mode as an alternative to the piece-wise approach.\n\n"
                                f"{str(e)}")
                        raise
                if aclgraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                aclgraph_mode.separate_routine():
                    max_num_tokens = self.scheduler_config.max_num_seqs * \
                            self.uniform_decode_query_len
                    decode_cudagraph_batch_sizes = [
                        x for x in self.cudagraph_batch_sizes_parallel if
                        x <= max_num_tokens and x >= self.uniform_decode_query_len
                    ]
                    compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                    self._capture_aclgraphs(
                        compilation_cases=compilation_cases_decode,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        uniform_decode=True,
                        in_parallel_streams=True)

        # Disable aclgraph capturing globally, so any unexpected aclgraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may doing lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

    def capture_model(self) -> None:

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]

        self._capture_model()

        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

    def _update_tokens_for_pcp(self, tokens):
        num_reqs = self.input_batch.num_reqs
        self.num_pcp_pads = self.num_pcp_pads[:num_reqs]
        tokens = np.array(tokens, dtype=np.int32)
        num_decode_reqs = sum(
            self.input_batch.num_computed_tokens_cpu[:num_reqs] >=
            self.input_batch.num_prompt_tokens[:num_reqs])
        num_decode_tokens = sum(tokens[:num_decode_reqs])
        num_padded_scheduled_tokens = np.ceil(
            tokens /
            (2 * self.pcp_size)).astype(np.int32) * (2 * self.pcp_size)
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            tokens[:num_decode_reqs] * self.pcp_size)
        self.num_pcp_pads = torch.tensor(num_padded_scheduled_tokens - tokens)
        cu_padded_tokens, pcp_padded_arange = \
            self._get_cumsum_and_arange(num_padded_scheduled_tokens)
        unpad_mask = torch.from_numpy(
            pcp_padded_arange < np.repeat(tokens, num_padded_scheduled_tokens))
        unpad_mask_decode = unpad_mask[:num_decode_tokens * self.pcp_size]
        unpad_mask_decode = unpad_mask_decode.reshape([-1, self.pcp_size])
        unpad_mask_decode[:, 0] = True
        unpad_mask_decode[:, 1:] = False

        pcp_tokens = num_padded_scheduled_tokens // self.pcp_size
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]
        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(pcp_chunk_sizes)
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes,
                                                     pcp_tokens)

        def get_current_rank_positions(cu_tokens, rank):
            positions_start_loc = np.zeros_like(cu_tokens)
            positions_start_loc[1:] = cu_tokens[:-1]
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = positions_start_loc + \
                (2 * self.pcp_size - rank - 1) * pcp_chunk_sizes
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + \
                np.repeat(head_start_loc, pcp_chunk_sizes)
            # Decode reqs do not have tail chunks.
            positions[~pcp_head_chunk_mask] = \
                pcp_chunk_arange[num_decode_tokens:] + \
                np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            return positions

        positions = get_current_rank_positions(
            np.zeros(num_reqs, dtype=np.int32), self.pcp_rank)
        # Decode tokens are duplicate and their positions always be 0.
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                tokens[:num_decode_reqs])[1]

        all_positions = [
            get_current_rank_positions(cu_padded_tokens, rank_i)
            for rank_i in range(self.pcp_size)
        ]
        all_positions_tensor = torch.from_numpy(np.concatenate(all_positions))
        self.pcp_allgather_restore_idx[:all_positions_tensor.shape[0]].copy_(
            all_positions_tensor.float().argsort().long(), non_blocking=True)
        return pcp_tokens, positions, unpad_mask

    def _get_cp_local_seq_lens(
        self,
        seq_lens: torch.Tensor,
        pcp_world_size: int = 1,
        dcp_world_size: int = 1,
        cp_kv_cache_interleave_size: int = 1,
    ) -> torch.Tensor:
        """While using pcp or dcp, kv_cache size stored on each rank may be different,
        use this function to calculate split decode seq_lens of each (p/d)cp rank.
        """
        num_requests = seq_lens.size(0)
        total_world_size = pcp_world_size * dcp_world_size
        seq_lens_tiled = seq_lens.unsqueeze(-1).repeat(1, total_world_size)
        rank_offsets = (torch.arange(total_world_size,
                                     dtype=torch.int32).unsqueeze(0).repeat(
                                         num_requests, 1))
        base = (seq_lens_tiled // cp_kv_cache_interleave_size //
                total_world_size * cp_kv_cache_interleave_size)
        remainder = seq_lens_tiled - base * total_world_size
        remainder = torch.clip(
            remainder - rank_offsets * cp_kv_cache_interleave_size,
            0,
            cp_kv_cache_interleave_size,
        )
        dcp_local_seq_lens = (base + remainder).reshape(
            [-1, pcp_world_size, dcp_world_size])
        return dcp_local_seq_lens

    def _generate_pcp_metadata(self, total_num_scheduled_tokens):
        # In dummy run num_reqs == 0, update it from seq_lens
        num_reqs = self.input_batch.num_reqs or self.query_lens.size(0)
        num_decodes = sum(self.input_batch.num_computed_tokens_cpu[:num_reqs]
                          >= self.input_batch.num_prompt_tokens[:num_reqs])
        num_actual_tokens_pcp_padded = total_num_scheduled_tokens * self.pcp_size
        self.num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded
        long_seq_metadata = None
        if self.pcp_size * self.dcp_size > 1:
            decode_context_lens = self.input_batch.num_tokens[:num_decodes]
            prefill_context_lens = self.input_batch.num_computed_tokens_cpu[
                num_decodes:num_reqs]
            context_lens = np.concatenate(
                [decode_context_lens, prefill_context_lens])
            num_computed_tokens_of_pcp_dcp = torch.zeros(
                [
                    num_reqs * self.decode_threshold, self.pcp_size,
                    self.dcp_size
                ],
                dtype=torch.int32,
            )
            # For pcp + spec decode, we flatten seq_lens
            # to avoid irregular spec_attn_mask shape
            for decode_idx in range(self.decode_threshold):
                num_computed_tokens_of_pcp_dcp[
                    self.decode_threshold - 1 - decode_idx::self.decode_threshold] = \
                    self._get_cp_local_seq_lens(
                        torch.tensor(context_lens),
                        self.pcp_size,
                        self.dcp_size,
                        self.parallel_config.cp_kv_cache_interleave_size,
                    )
            long_seq_metadata = AscendPrefillContextParallelMetadata(
                num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp.
                numpy())
            if self.pcp_size > 1:
                q_head_idx, q_tail_idx = [], []
                kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
                kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
                chunk_seqlens = []
                kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
                q_req_offset = 0
                kv_req_offset = 0
                q_head_chunk_id = self.pcp_rank
                q_tail_chunk_id = self.pcp_size * 2 - 1 - self.pcp_rank
                for i, seq_len in enumerate(self.query_lens):
                    if i < num_decodes:
                        continue
                    chunk_len = seq_len // 2
                    chunk_seqlens.append(chunk_len)
                    q_head_idx.extend(
                        list(range(q_req_offset, q_req_offset + chunk_len)))
                    kv_with_q_head_nomask_idx.extend(
                        list(
                            range(kv_req_offset, kv_req_offset +
                                  chunk_len * q_head_chunk_id)))
                    kv_with_q_head_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_head_chunk_id,
                                kv_req_offset + chunk_len *
                                (q_head_chunk_id + 1))))
                    kv_with_q_head_nomask_seqlens.append(chunk_len *
                                                         q_head_chunk_id)

                    q_tail_idx.extend(
                        list(
                            range(q_req_offset + chunk_len,
                                  q_req_offset + chunk_len * 2)))
                    kv_with_q_tail_nomask_idx.extend(
                        list(
                            range(kv_req_offset, kv_req_offset +
                                  chunk_len * q_tail_chunk_id)))
                    kv_with_q_tail_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                                kv_req_offset + chunk_len *
                                (q_tail_chunk_id + 1))))
                    kv_with_q_tail_nomask_seqlens.append(chunk_len *
                                                         q_tail_chunk_id)

                    q_req_offset += seq_len
                    kv_req_offset += seq_len * self.pcp_size

                # Convert lists to tensors and move to device
                def _list_to_tensor(lst, device, dtype=torch.int32):
                    tensor_npu = torch.zeros(len(lst),
                                             dtype=dtype,
                                             device=device)
                    tensor_npu.copy_(torch.tensor(lst, dtype=dtype),
                                     non_blocking=True)
                    return tensor_npu

                q_head_idx_tensor = _list_to_tensor(q_head_idx, self.device)
                q_tail_idx_tensor = _list_to_tensor(q_tail_idx, self.device)
                self.q_head_idx_tensor = q_head_idx_tensor
                self.q_tail_idx_tensor = q_tail_idx_tensor

                q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
                q_full_idx = q_full_idx.to(torch.float32).argsort().to(
                    torch.int32)
                self.q_full_idx = q_full_idx

                self.kv_idx_names = {
                    'kv_with_q_head_nomask_idx_tensor':
                    kv_with_q_head_nomask_idx,
                    'kv_with_q_head_mask_idx_tensor': kv_with_q_head_mask_idx,
                    'kv_with_q_tail_nomask_idx_tensor':
                    kv_with_q_tail_nomask_idx,
                    'kv_with_q_tail_mask_idx_tensor': kv_with_q_tail_mask_idx
                }
                for key, value in self.kv_idx_names.items():
                    tensor_npu = _list_to_tensor(value, self.device)
                    self.kv_idx_names[key] = tensor_npu

                attn_mask_seqlens = torch.tensor(
                    [chunk_seqlens, chunk_seqlens], dtype=torch.int32)
                head_attn_nomask_seqlens = torch.tensor(
                    [chunk_seqlens, kv_with_q_head_nomask_seqlens],
                    dtype=torch.int32)
                tail_attn_nomask_seqlens = torch.tensor(
                    [chunk_seqlens, kv_with_q_tail_nomask_seqlens],
                    dtype=torch.int32)
                pcp_prefill_mask = self.attn_mask

                self.extra_long_seq_kwargs = {
                    'attn_mask_seqlens': attn_mask_seqlens,
                    'head_attn_nomask_seqlens': head_attn_nomask_seqlens,
                    'tail_attn_nomask_seqlens': tail_attn_nomask_seqlens,
                    'pcp_prefill_mask': pcp_prefill_mask
                }
                long_seq_metadata.pcp_allgather_restore_idx = self.pcp_allgather_restore_idx[:
                                                                                             num_actual_tokens_pcp_padded]
                long_seq_metadata.cp_kv_recover_idx_for_chunk = self.cp_kv_recover_idx_for_chunk
                long_seq_metadata.q_head_idx_tensor = self.q_head_idx_tensor
                long_seq_metadata.q_tail_idx_tensor = self.q_tail_idx_tensor
                long_seq_metadata.q_full_idx = self.q_full_idx
                long_seq_metadata.kv_with_q_head_nomask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_head_nomask_idx_tensor']
                long_seq_metadata.kv_with_q_head_mask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_head_mask_idx_tensor']
                long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_tail_nomask_idx_tensor']
                long_seq_metadata.kv_with_q_tail_mask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_tail_mask_idx_tensor']
                long_seq_metadata.attn_mask_seqlens = self.extra_long_seq_kwargs[
                    'attn_mask_seqlens']
                long_seq_metadata.head_attn_nomask_seqlens = self.extra_long_seq_kwargs[
                    'head_attn_nomask_seqlens']
                long_seq_metadata.tail_attn_nomask_seqlens = self.extra_long_seq_kwargs[
                    'tail_attn_nomask_seqlens']
                long_seq_metadata.pcp_prefill_mask = self.extra_long_seq_kwargs[
                    'pcp_prefill_mask']
            self.long_seq_metadata = long_seq_metadata
        return long_seq_metadata

    def _generate_pcp_mtp_input(
        self,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: dict[str, int],
    ):
        """
        While pcp > 1, model inputs (input_ids, position, etc.) are split across pcp group,
        but mtp need to shift original input_ids before pcp splitting,
        so we record original input_ids here.
        """
        total_num_scheduled_tokens_pcp_full = total_num_scheduled_tokens
        num_scheduled_tokens_pcp_full = np.empty(num_reqs, dtype=np.int32)
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_scheduled_tokens_pcp_full[i] = num_scheduled_tokens[req_id]
        req_indices_pcp_full = np.repeat(self.arange_np[:num_reqs],
                                         num_scheduled_tokens_pcp_full)
        cu_num_tokens_pcp_full = np.cumsum(num_scheduled_tokens_pcp_full)
        self.query_start_loc_pcp_full.np[0] = 0
        self.query_start_loc_pcp_full.np[1:num_reqs +
                                         1] = cu_num_tokens_pcp_full
        self.query_start_loc_pcp_full.np[num_reqs + 1:].fill(-1)
        cumsums_offsets_pcp_full = np.repeat(
            cu_num_tokens_pcp_full - num_scheduled_tokens_pcp_full,
            num_scheduled_tokens_pcp_full)
        arange_pcp_full = self.arange_np[:
                                         total_num_scheduled_tokens_pcp_full] - cumsums_offsets_pcp_full
        positions_pcp_full_np = self.positions_pcp_full_np[:
                                                           total_num_scheduled_tokens_pcp_full]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices_pcp_full],
               arange_pcp_full,
               out=positions_pcp_full_np)
        token_indices_pcp_full = (
            positions_pcp_full_np +
            req_indices_pcp_full * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices_pcp_full),
                           out=self.input_ids_pcp_full.
                           cpu[:total_num_scheduled_tokens_pcp_full])
        self.query_start_loc_pcp_full.copy_to_gpu()
        self.input_ids_pcp_full.gpu[:total_num_scheduled_tokens_pcp_full].copy_(
            self.input_ids_pcp_full.cpu[:total_num_scheduled_tokens_pcp_full],
            non_blocking=True,
        )
    def _to_jsonable(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        # 尝试将对象的属性字典序列化
        if hasattr(obj, "__dict__"):
            return {k: self._to_jsonable(v) for k, v in vars(obj).items()}
        return str(obj)


@contextmanager
def _torch_cuda_wrapper():

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        yield
    except Exception:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
    
