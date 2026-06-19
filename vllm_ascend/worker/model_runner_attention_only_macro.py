#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import time
from typing import Any, Optional

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.ubatch_utils import UBatchSlices

from vllm_ascend.ascend_forward_context import create_ascend_forward_context
from vllm_ascend.worker.attention_only_macro_scheduler import (
    AttentionOnlySplitRuntime,
    capture_piecewise_model_call,
    run_attention_only_split_piecewise,
)
from vllm_ascend.worker.model_runner_v3 import (
    NPUModelRunner,
    PerLayerAttnMetadata,
    _set_split_debug_step,
    _split_debug_step_from_runner,
)
from vllm_ascend.worker.ubatch_utils import SplitBatchSlices


class AttentionOnlyMacroNPUModelRunner(NPUModelRunner):
    """Experimental runner for MatMul-serial, attention-parallel split replay."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._attention_only_macro_warned = False

    def _macro_graph_config(self) -> Any:
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        if split_cfg is None:
            return None
        return getattr(split_cfg, "macro_graph_config", None)

    def _validate_attention_only_macro_request(
            self,
            split_batch_slices: SplitBatchSlices,
            aclgraph_runtime_mode: CUDAGraphMode,
            inplace_attention_backend: str) -> None:
        macro_cfg = self._macro_graph_config()
        if macro_cfg is None or not getattr(macro_cfg, "enabled", False):
            raise RuntimeError(
                "attention_only_macro runner requires "
                "split_batch_config.macro_graph_config.enabled=true")
        split_cfg = self.ascend_config.split_batch_config
        if len(split_batch_slices) != 2:
            raise RuntimeError(
                "attention_only_macro currently supports exactly 2 splits, "
                f"got {len(split_batch_slices)}")
        if aclgraph_runtime_mode != CUDAGraphMode.PIECEWISE:
            raise RuntimeError(
                "attention_only_macro requires PIECEWISE runtime mode, got "
                f"{aclgraph_runtime_mode}")
        if not bool(getattr(split_cfg, "enable_parallel_streams", False)):
            raise RuntimeError(
                "attention_only_macro requires "
                "split_batch_config.enable_parallel_streams=true")
        if bool(getattr(split_cfg, "enable_inplace_lazy_capture", True)):
            raise RuntimeError(
                "attention_only_macro is a no-lazy-capture experiment. Set "
                "split_batch_config.enable_inplace_lazy_capture=false")
        if inplace_attention_backend not in ("fia", "pa"):
            raise RuntimeError(
                "attention_only_macro expected forced attention backend fia "
                f"or pa, got {inplace_attention_backend!r}")
        if self.vllm_config.model_config.use_mla:
            raise RuntimeError(
                "attention_only_macro MVP does not support MLA models yet")
        if self.pcp_size * self.dcp_size > 1:
            raise RuntimeError(
                "attention_only_macro MVP does not support PCP/DCP yet")
        if self.uses_mrope:
            raise RuntimeError(
                "attention_only_macro MVP does not support MROPE yet")

    def _make_attention_only_split_contexts(
            self,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            inplace_attention_backend: str) -> list[Any]:
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata
        context_ubatch_slices = self._context_ubatch_slices_for_inplace(
            split_batch_slices)
        contexts: list[Any] = []
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        force_pa_for_offset = bool(
            split_cfg is not None
            and getattr(split_cfg, "inplace_force_pa_for_offset", False))

        for idx, split_slice in enumerate(split_batch_slices):
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and idx < len(
                        attn_metadata):
                    ubatch_attn_metadata = attn_metadata[idx]
                else:
                    ubatch_attn_metadata = attn_metadata

            split_attention_backend = inplace_attention_backend
            if force_pa_for_offset and split_slice.start_num_tokens > 0:
                split_attention_backend = "pa"

            graph_tokens = int(split_slice.graph_num_tokens)
            if graph_tokens % int(self.uniform_decode_query_len) != 0:
                raise RuntimeError(
                    "attention_only_macro requires request-aligned split "
                    f"tokens: graph_tokens={graph_tokens}, "
                    f"query_len={self.uniform_decode_query_len}")
            ubatch_cudagraph_mode = CUDAGraphMode.PIECEWISE
            ubatch_batch_descriptor = BatchDescriptor(
                num_tokens=graph_tokens,
                num_reqs=graph_tokens // int(self.uniform_decode_query_len),
                uniform=True,
                has_lora=batch_descriptor.has_lora,
                start_num_tokens=int(split_slice.start_num_tokens),
                graph_variant="attention_only_macro",
                attention_backend=split_attention_backend,
                capture_metadata_mode="",
            )

            ctx_stream = (self.stream_parallel if idx > 0
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
                    ubatch_num=idx,
                    positions=positions,
                    in_parallel_streams=(idx > 0),
                    cos_sin_slot_id=idx,
                )
            _set_split_debug_step(split_forward_context,
                                  _split_debug_step_from_runner(self))
            setattr(split_forward_context, "split_inplace_mode",
                    "attention_only_macro")
            setattr(split_forward_context, "forced_attention_backend",
                    split_attention_backend)
            setattr(split_forward_context, "allow_inplace_lazy_capture",
                    False)
            setattr(split_forward_context, "validate_inplace_input_ptrs",
                    False)
            setattr(split_forward_context, "validate_inplace_metadata_ptrs",
                    False)
            contexts.append(split_forward_context)
        return contexts

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
        macro_cfg = self._macro_graph_config()
        if macro_cfg is None or not getattr(macro_cfg, "enabled", False):
            return super()._run_split_batch_inplace_parallel(
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

        if not self._attention_only_macro_warned:
            logger.warning(
                "attention_only_macro MVP runs the piecewise graph with "
                "MatMul/non-attention pieces on the main stream and attention "
                "splitting pieces on two streams. PTA multi-stream macro "
                "graph capture is not implemented in this Python runner yet.")
            self._attention_only_macro_warned = True

        self._validate_attention_only_macro_request(
            split_batch_slices,
            aclgraph_runtime_mode,
            inplace_attention_backend,
        )
        split_contexts = self._make_attention_only_split_contexts(
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            batch_descriptor,
            inplace_attention_backend,
        )

        self._t_replay_start = time.perf_counter()
        try:
            runtime_call = capture_piecewise_model_call(
                self.model,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                model_kwargs=model_kwargs,
            )
            runtime = AttentionOnlySplitRuntime(
                split_batch_slices=split_batch_slices,
                split_contexts=split_contexts,
                stream_main=self.stream_main,
                stream_parallel=self.stream_parallel,
                require_exact_graph_tokens=bool(
                    getattr(macro_cfg, "require_exact_graph_tokens", True)),
            )
            hidden_states = run_attention_only_split_piecewise(
                runtime_call.handle,
                runtime_call.args,
                runtime_call.kwargs,
                runtime,
            )
            self.stream_main.synchronize()
            self.stream_parallel.synchronize()
            return hidden_states
        finally:
            self._t_replay_end = time.perf_counter()
