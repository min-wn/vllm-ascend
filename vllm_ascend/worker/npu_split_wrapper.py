import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.forward_context import (get_forward_context,
                                  override_forward_context,
                                  BatchDescriptor)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm_ascend.utils import enable_sp
from vllm_ascend.ascend_forward_context import create_ascend_forward_context
from vllm_ascend.worker.npu_ubatch_wrapper import AscendUbatchMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ascend_config import SplitBatchConfig
# Import update functions
from vllm_ascend.compilation.acl_graph import (
    update_mla_attn_params,
    update_attn_params,
    update_mla_attn_dcp_pcp_params,
    update_attn_dcp_pcp_params,
)

logger = init_logger(__name__)


# Rollback switch for split context rebuild coordinate validation.
# - default "1": use local coordinate slices for rebuilt split context
# - set VLLM_ASCEND_SPLIT_LOCAL_CONTEXT_REBUILD=0 to restore legacy behavior
_SPLIT_LOCAL_CONTEXT_REBUILD = os.environ.get(
    "VLLM_ASCEND_SPLIT_LOCAL_CONTEXT_REBUILD", "1") not in (
        "0", "false", "False")


def _iter_attn_metadata_objects(attn_metadata: Any):
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            yield value
        return
    if isinstance(attn_metadata, list):
        for value in attn_metadata:
            yield value
        return
    if attn_metadata is not None:
        yield attn_metadata


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
            metadata_obj.slot_mapping = slot_mapping
            updated += 1
        common_attn_metadata = getattr(metadata_obj, "common_attn_metadata",
                                       None)
        if (common_attn_metadata is not None
                and hasattr(common_attn_metadata, "slot_mapping")):
            common_attn_metadata.slot_mapping = slot_mapping
            updated += 1
    return updated


class AscendSplitBatchWrapper:

    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: torch.npu.device,
                 update_stream: Optional[torch.npu.Stream] = None,
                 split_config: Optional[SplitBatchConfig] = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.default_stream = torch.npu.default_stream(device)
        self.parallel_stream = torch.npu.Stream()
        self.device = device
        self.pool = torch.npu.graph_pool_handle()
        self.enable_parallel = getattr(split_config, "enable_parallel_streams", False)
        self.extra_graph_size = getattr(split_config,
                                        "cudagraph_capture_fixed_size", 0)
        self.tag = False
        self._tag_lock = threading.Lock()
        self._parallel_input_ids = None
        self._parallel_positions = None
        self._parallel_inputs_embeds = None
        self._parallel_slot_mapping = None

        # Create update stream for overlapping parameter updates
        self.update_stream = update_stream

        # Use ACLGraphWrapper instead of manual graph management
        self.aclgraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.aclgraph_wrapper = ACLGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _claim_dedicated_pool_capture(self, num_tokens: int) -> bool:
        if not self.enable_parallel or self.aclgraph_wrapper is None:
            return False
        if self.extra_graph_size <= 0 or num_tokens != self.extra_graph_size:
            return False
        with self._tag_lock:
            if self.tag:
                return False
            self.tag = True
            return True

    def _rollback_dedicated_pool_capture(self) -> None:
        with self._tag_lock:
            self.tag = False

    def _update_attn_params_for_ubatch(self, forward_context, num_tokens,
                                       ubatch_id: int):
        """
        Update attention parameters for a specific ubatch.
        This is called on update_stream to overlap with graph replay on default
        stream.
        """
        pcp_size = getattr(self.runnable, "pcp_size", 1)
        dcp_size = getattr(self.runnable, "dcp_size", 1)
        use_mla = self.vllm_config.model_config.use_mla
        speculative_config = self.vllm_config.speculative_config
        update_stream = (self.update_stream
                         if self.update_stream is not None else self.default_stream)
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))

        logger.debug(
            "Updating attn params for ubatch %s on update_stream, "
            "num_tokens=%s, stream_id=%s, in_parallel_streams=%s",
            ubatch_id,
            num_tokens,
            update_stream.stream_id,
            in_parallel_streams,
        )

        if use_mla:
            if pcp_size * dcp_size > 1:
                update_mla_attn_dcp_pcp_params(
                    update_stream, forward_context, num_tokens,
                    in_parallel_streams=in_parallel_streams)
            else:
                update_mla_attn_params(
                    update_stream, forward_context, num_tokens,
                    speculative_config,
                    in_parallel_streams=in_parallel_streams)
        else:
            if pcp_size * dcp_size > 1:
                update_attn_dcp_pcp_params(
                    update_stream, forward_context, num_tokens,
                    in_parallel_streams=in_parallel_streams)
            else:
                # GPU-aligned: block_table capture address matches runtime write
                # address (both offset 0 for micro buffers), so use plain
                # update_attn_params without refresh_block_table=True.
                update_attn_params(
                    update_stream, forward_context, num_tokens,
                    self.vllm_config,
                    in_parallel_streams=in_parallel_streams)
    def _refresh_block_table_for_ubatch(self, forward_context,
                                        num_tokens: int) -> None:
        """Refresh split block_table in-place before graph replay."""
        if (forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or forward_context.capturing):
            return

        pcp_size = getattr(self.runnable, "pcp_size", 1)
        dcp_size = getattr(self.runnable, "dcp_size", 1)

        if self.vllm_config.model_config.use_mla:
            return
        if pcp_size * dcp_size > 1:
            return

        update_stream = (self.update_stream
                         if self.update_stream is not None else self.default_stream)
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        # GPU-aligned: block_table capture address matches runtime write
        # address (both offset 0 for micro buffers), so no in-place
        # refresh needed. Use plain update_attn_params instead.
        update_attn_params(update_stream,
                           forward_context,
                           num_tokens,
                           self.vllm_config,
                           in_parallel_streams=in_parallel_streams)

    def _snapshot_split_output(self, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output.clone()
        if isinstance(output, IntermediateTensors):
            return IntermediateTensors(
                {k: v.clone() for k, v in output.tensors.items()})
        if isinstance(output, tuple):
            return tuple(self._snapshot_split_output(x) for x in output)
        if isinstance(output, list):
            return [self._snapshot_split_output(x) for x in output]
        if isinstance(output, dict):
            return {
                k: self._snapshot_split_output(v)
                for k, v in output.items()
            }
        return output

    def _run_ubatches(self, *args, **kwargs) -> torch.Tensor:
        """
        Split-batch style execution (aligned with model_runner_v3._run_split_batch_gr):
        - Build the first split context explicitly.
        - Reuse metadata object for following splits with in-place context rebuild.
        - Copy subsequent split inputs into a fixed prefix region before replay.
        - Restore overwritten prefix and slot-mapping state at function end.
        """
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        attn_metadata = forward_context.attn_metadata
        original_forward_context = forward_context

        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]

        if not ubatch_slices:
            if self.aclgraph_wrapper is not None:
                use_dedicated_pool = self._claim_dedicated_pool_capture(
                    int(getattr(batch_descriptor, "num_tokens", 0)))
                aclgraph_kwargs = dict(kwargs)
                aclgraph_kwargs["parallel_streams"] = self.enable_parallel
                if use_dedicated_pool:
                    aclgraph_kwargs["graph_pool"] = self.pool
                try:
                    return self.aclgraph_wrapper(*args, **aclgraph_kwargs)
                except Exception:
                    if use_dedicated_pool:
                        self._rollback_dedicated_pool_capture()
                    raise
            return self.runnable(*args, **kwargs)

        first_slice = ubatch_slices[0]
        first_num_tokens = first_slice.token_slice.stop - first_slice.token_slice.start
        first_num_reqs = first_slice.request_slice.stop - first_slice.request_slice.start

        backup_input_ids = None
        if input_ids is not None:
            backup_input_ids = input_ids[:first_num_tokens].clone()

        if positions.ndim == 2:
            backup_positions = positions[:, :first_num_tokens].clone()
        else:
            backup_positions = positions[:first_num_tokens].clone()

        backup_inputs_embeds = None
        if inputs_embeds is not None:
            backup_inputs_embeds = inputs_embeds[:first_num_tokens].clone()

        first_attn_metadata = attn_metadata[0] if isinstance(attn_metadata,
                                                             list) else attn_metadata
        base_slot_mapping = _get_slot_mapping_from_attn_metadata(first_attn_metadata)
        slot_mapping_backup_len = 0
        backup_slot_mapping = None
        if base_slot_mapping is not None:
            slot_mapping_backup_len = min(first_num_tokens,
                                          int(base_slot_mapping.shape[0]))
            if slot_mapping_backup_len > 0:
                backup_slot_mapping = base_slot_mapping[
                    :slot_mapping_backup_len].clone()

        first_input_ids, first_positions, first_inputs_embeds, first_intermediate = \
            self._slice_model_inputs(first_slice.token_slice, input_ids, positions,
                                    inputs_embeds, intermediate_tensors)

        ubatch_cudagraph_mode = CUDAGraphMode.FULL if cudagraph_runtime_mode != CUDAGraphMode.NONE else CUDAGraphMode.NONE
        first_batch_descriptor = BatchDescriptor(
            num_tokens=first_num_tokens,
            num_reqs=first_num_reqs,
            uniform=batch_descriptor.uniform,
            has_lora=batch_descriptor.has_lora,
        )
        first_context = create_ascend_forward_context(
            original_forward_context,
            attn_metadata=first_attn_metadata,
            vllm_config=self.vllm_config,
            dp_metadata=original_forward_context.dp_metadata,
            ubatch_slices=ubatch_slices,
            batch_descriptor=first_batch_descriptor,
            cudagraph_runtime_mode=ubatch_cudagraph_mode,
            ubatch_num=0,
            positions=positions,
        )

        metadata = AscendUbatchMetadata(
            context=first_context,
            input_ids=first_input_ids,
            positions=first_positions,
            inputs_embeds=first_inputs_embeds,
            intermediate_tensors=first_intermediate,
            num_tokens=first_num_tokens,
        )

        results_with_order: list[tuple[int, Any, int]] = []

        try:
            with override_forward_context(None):
                torch.npu.set_device(self.device)

                for ubatch_id, ubatch_slice in enumerate(ubatch_slices):
                    current_num_tokens = ubatch_slice.token_slice.stop - ubatch_slice.token_slice.start
                    current_num_reqs = ubatch_slice.request_slice.stop - ubatch_slice.request_slice.start

                    if ubatch_id > 0:
                        split_attn_metadata = attn_metadata
                        if isinstance(attn_metadata, list):
                            if ubatch_id >= len(attn_metadata):
                                raise RuntimeError(
                                    "split attn_metadata list too short: "
                                    f"ubatch_id={ubatch_id}, len={len(attn_metadata)}"
                                )
                            split_attn_metadata = attn_metadata[ubatch_id]

                        split_slot_mapping = _get_slot_mapping_from_attn_metadata(
                            split_attn_metadata)
                        if base_slot_mapping is not None and split_slot_mapping is not None:
                            copy_len = min(current_num_tokens,
                                           int(base_slot_mapping.shape[0]),
                                           int(split_slot_mapping.shape[0]))
                            if copy_len != current_num_tokens:
                                raise RuntimeError(
                                    "split slot_mapping length mismatch: "
                                    f"required={current_num_tokens}, "
                                    f"base={int(base_slot_mapping.shape[0])}, "
                                    f"split={int(split_slot_mapping.shape[0])}")
                            base_slot_mapping[:copy_len].copy_(
                                split_slot_mapping[:copy_len], non_blocking=False)
                            _set_slot_mapping_for_attn_metadata(
                                split_attn_metadata, base_slot_mapping[:copy_len])

                        cur_input_ids, cur_positions, cur_inputs_embeds, cur_intermediate = \
                            self._slice_model_inputs(
                                ubatch_slice.token_slice,
                                input_ids,
                                positions,
                                inputs_embeds,
                                intermediate_tensors,
                            )

                        torch.npu.synchronize()

                        if cur_input_ids is not None:
                            input_ids[:current_num_tokens].copy_(
                                cur_input_ids, non_blocking=False)

                        if cur_positions is not None:
                            if cur_positions.ndim == 2:
                                positions[:, :current_num_tokens].copy_(
                                    cur_positions, non_blocking=False)
                            else:
                                positions[:current_num_tokens].copy_(
                                    cur_positions, non_blocking=False)

                        if cur_inputs_embeds is not None:
                            inputs_embeds[:current_num_tokens].copy_(
                                cur_inputs_embeds, non_blocking=False)

                        metadata.input_ids = (
                            input_ids[:current_num_tokens]
                            if cur_input_ids is not None else None
                        )
                        if cur_positions is not None:
                            if cur_positions.ndim == 2:
                                metadata.positions = positions[:, :current_num_tokens]
                            else:
                                metadata.positions = positions[:current_num_tokens]
                        else:
                            metadata.positions = None
                        metadata.inputs_embeds = (
                            inputs_embeds[:current_num_tokens]
                            if cur_inputs_embeds is not None else None
                        )
                        metadata.intermediate_tensors = cur_intermediate
                        metadata.num_tokens = current_num_tokens

                        ubatch_batch_descriptor = BatchDescriptor(
                            num_tokens=current_num_tokens,
                            num_reqs=current_num_reqs,
                            uniform=batch_descriptor.uniform,
                            has_lora=batch_descriptor.has_lora,
                        )
                        if _SPLIT_LOCAL_CONTEXT_REBUILD:
                            rebuild_ubatch_slices = [
                                UBatchSlice(
                                    slice(0, current_num_reqs),
                                    slice(0, current_num_tokens),
                                )
                            ]
                            rebuild_positions = (metadata.positions
                                                 if metadata.positions is not None
                                                 else positions)
                            rebuild_ubatch_num = 0
                        else:
                            rebuild_ubatch_slices = ubatch_slices
                            rebuild_positions = positions
                            rebuild_ubatch_num = ubatch_id

                        metadata.context = create_ascend_forward_context(
                            metadata.context,
                            attn_metadata=split_attn_metadata,
                            vllm_config=self.vllm_config,
                            dp_metadata=original_forward_context.dp_metadata,
                            ubatch_slices=rebuild_ubatch_slices,
                            batch_descriptor=ubatch_batch_descriptor,
                            cudagraph_runtime_mode=ubatch_cudagraph_mode,
                            ubatch_num=rebuild_ubatch_num,
                            positions=rebuild_positions,
                        )
                        torch.npu.synchronize()

                    with override_forward_context(metadata.context):
                        self._refresh_block_table_for_ubatch(
                            metadata.context, current_num_tokens)

                        if (metadata.context.cudagraph_runtime_mode !=
                                CUDAGraphMode.NONE
                                and self.aclgraph_wrapper is not None):
                            use_dedicated_pool = self._claim_dedicated_pool_capture(
                                current_num_tokens)
                            try:
                                model_output = self.aclgraph_wrapper(
                                    input_ids=metadata.input_ids,
                                    positions=metadata.positions,
                                    intermediate_tensors=metadata.intermediate_tensors,
                                    inputs_embeds=metadata.inputs_embeds,
                                    parallel_streams=self.enable_parallel,
                                    graph_pool=(self.pool
                                                if use_dedicated_pool else None),
                                )
                            except Exception:
                                if use_dedicated_pool:
                                    self._rollback_dedicated_pool_capture()
                                raise
                        else:
                            model_output = self.runnable(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                intermediate_tensors=metadata.intermediate_tensors,
                                inputs_embeds=metadata.inputs_embeds,
                            )

                        if (metadata.context.cudagraph_runtime_mode ==
                                CUDAGraphMode.FULL
                                and not metadata.context.capturing):
                            self._update_attn_params_for_ubatch(
                                metadata.context, current_num_tokens, ubatch_id)

                    request_start = int(ubatch_slice.request_slice.start)
                    model_output = self._snapshot_split_output(model_output)
                    results_with_order.append(
                        (request_start, model_output,
                         getattr(metadata.context, "pad_size", 0))
                    )

            with override_forward_context(original_forward_context):
                sorted_entries = sorted(results_with_order, key=lambda x: x[0])
                sorted_results = [entry[1] for entry in sorted_entries]
                sorted_pad_sizes = [entry[2] for entry in sorted_entries]
                if get_forward_context().sp_enabled and get_pp_group().is_last_rank:
                    for i in range(len(sorted_results)):
                        sorted_results[i] = tensor_model_parallel_all_gather(
                            sorted_results[i], 0)
                        pad_size = sorted_pad_sizes[i]
                        if pad_size > 0:
                            sorted_results[i] = sorted_results[i][:-pad_size, :]

                if not get_pp_group().is_last_rank:
                    result = self._merge_intermediate_tensors(sorted_results)
                else:
                    result = torch.cat(sorted_results, dim=0)

                get_forward_context().dbo_enabled = False

            return result
        finally:
            if backup_input_ids is not None:
                input_ids[:first_num_tokens].copy_(backup_input_ids,
                                                   non_blocking=False)

            if positions.ndim == 2:
                positions[:, :first_num_tokens].copy_(backup_positions,
                                                      non_blocking=False)
            else:
                positions[:first_num_tokens].copy_(backup_positions,
                                                   non_blocking=False)

            if backup_inputs_embeds is not None:
                inputs_embeds[:first_num_tokens].copy_(backup_inputs_embeds,
                                                       non_blocking=False)

            if backup_slot_mapping is not None and base_slot_mapping is not None:
                base_slot_mapping[:slot_mapping_backup_len].copy_(
                    backup_slot_mapping, non_blocking=False)

            torch.npu.synchronize()
    def _should_use_parallel_ubatches(self, ubatch_slices) -> bool:
        if (not self.enable_parallel or self.extra_graph_size <= 0
                or ubatch_slices is None or len(ubatch_slices) <= 1):
            return False
        return True

    def _ensure_parallel_replay_buffers(self, max_tokens: int, input_ids,
                                        positions, inputs_embeds) -> None:
        if max_tokens <= 0:
            return

        if input_ids is not None:
            if (self._parallel_input_ids is None
                    or self._parallel_input_ids.dtype != input_ids.dtype
                    or self._parallel_input_ids.device != input_ids.device
                    or int(self._parallel_input_ids.shape[0]) < max_tokens):
                self._parallel_input_ids = torch.empty(
                    (max_tokens, ),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )

        if positions is not None:
            if positions.ndim == 2:
                desired_shape = (int(positions.shape[0]), max_tokens)
                need_realloc = (
                    self._parallel_positions is None
                    or self._parallel_positions.dtype != positions.dtype
                    or self._parallel_positions.device != positions.device
                    or self._parallel_positions.ndim != 2
                    or int(self._parallel_positions.shape[0])
                    != desired_shape[0]
                    or int(self._parallel_positions.shape[1])
                    < desired_shape[1])
            else:
                desired_shape = (max_tokens, )
                need_realloc = (
                    self._parallel_positions is None
                    or self._parallel_positions.dtype != positions.dtype
                    or self._parallel_positions.device != positions.device
                    or self._parallel_positions.ndim != 1
                    or int(self._parallel_positions.shape[0])
                    < desired_shape[0])
            if need_realloc:
                self._parallel_positions = torch.empty(
                    desired_shape,
                    dtype=positions.dtype,
                    device=positions.device,
                )

        if inputs_embeds is not None:
            desired_shape = (max_tokens, ) + tuple(inputs_embeds.shape[1:])
            if (self._parallel_inputs_embeds is None
                    or self._parallel_inputs_embeds.dtype != inputs_embeds.dtype
                    or self._parallel_inputs_embeds.device != inputs_embeds.device
                    or tuple(self._parallel_inputs_embeds.shape[1:])
                    != desired_shape[1:]
                    or int(self._parallel_inputs_embeds.shape[0])
                    < desired_shape[0]):
                self._parallel_inputs_embeds = torch.empty(
                    desired_shape,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )

    def _refresh_parallel_slot_mapping(self, attn_metadata: Any,
                                       num_tokens: int):
        split_slot_mapping = _get_slot_mapping_from_attn_metadata(attn_metadata)
        if split_slot_mapping is None:
            return None
        if int(split_slot_mapping.shape[0]) < num_tokens:
            raise RuntimeError(
                "parallel split slot_mapping length mismatch: "
                f"required={num_tokens}, available={int(split_slot_mapping.shape[0])}")

        if (self._parallel_slot_mapping is None
                or self._parallel_slot_mapping.dtype != split_slot_mapping.dtype
                or self._parallel_slot_mapping.device != split_slot_mapping.device
                or self._parallel_slot_mapping.ndim != split_slot_mapping.ndim
                or tuple(self._parallel_slot_mapping.shape[1:])
                != tuple(split_slot_mapping.shape[1:])
                or int(self._parallel_slot_mapping.shape[0])
                < int(split_slot_mapping.shape[0])):
            self._parallel_slot_mapping = torch.empty_like(split_slot_mapping)

        self._parallel_slot_mapping[:num_tokens].copy_(
            split_slot_mapping[:num_tokens], non_blocking=False)
        parallel_slot_view = self._parallel_slot_mapping[:num_tokens]
        _set_slot_mapping_for_attn_metadata(attn_metadata, parallel_slot_view)
        return (attn_metadata, split_slot_mapping)

    def _run_parallel_ubatches(self, *args, **kwargs) -> torch.Tensor:
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        attn_metadata = forward_context.attn_metadata
        original_forward_context = forward_context

        if not self._should_use_parallel_ubatches(ubatch_slices):
            return self._run_ubatches(*args, **kwargs)

        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]

        first_slice = ubatch_slices[0]
        first_num_tokens = first_slice.token_slice.stop - first_slice.token_slice.start
        first_num_reqs = first_slice.request_slice.stop - first_slice.request_slice.start

        small_slices = ubatch_slices[1:]
        max_small_tokens = max(
            int(s.token_slice.stop - s.token_slice.start) for s in small_slices)
        self._ensure_parallel_replay_buffers(max_small_tokens, input_ids,
                                             positions, inputs_embeds)

        first_attn_metadata = attn_metadata[0] if isinstance(attn_metadata,
                                                             list) else attn_metadata
        first_input_ids, first_positions, first_inputs_embeds, first_intermediate = \
            self._slice_model_inputs(first_slice.token_slice, input_ids,
                                     positions, inputs_embeds,
                                     intermediate_tensors)

        ubatch_cudagraph_mode = CUDAGraphMode.FULL if cudagraph_runtime_mode != CUDAGraphMode.NONE else CUDAGraphMode.NONE
        first_batch_descriptor = BatchDescriptor(
            num_tokens=first_num_tokens,
            num_reqs=first_num_reqs,
            uniform=batch_descriptor.uniform,
            has_lora=batch_descriptor.has_lora,
        )
        first_context = create_ascend_forward_context(
            original_forward_context,
            attn_metadata=first_attn_metadata,
            vllm_config=self.vllm_config,
            dp_metadata=original_forward_context.dp_metadata,
            ubatch_slices=ubatch_slices,
            batch_descriptor=first_batch_descriptor,
            cudagraph_runtime_mode=ubatch_cudagraph_mode,
            ubatch_num=0,
            positions=positions,
        )

        results_with_order: list[tuple[int, Any, int]] = []
        slot_restore_entries = []

        try:
            with override_forward_context(None):
                torch.npu.set_device(self.device)

                with torch.npu.stream(self.default_stream):
                    with override_forward_context(first_context):
                        self._refresh_block_table_for_ubatch(
                            first_context, first_num_tokens)

                        if (first_context.cudagraph_runtime_mode !=
                                CUDAGraphMode.NONE
                                and self.aclgraph_wrapper is not None):
                            first_output = self.aclgraph_wrapper(
                                input_ids=first_input_ids,
                                positions=first_positions,
                                intermediate_tensors=first_intermediate,
                                inputs_embeds=first_inputs_embeds,
                                parallel_streams=False,
                            )
                        else:
                            first_output = self.runnable(
                                input_ids=first_input_ids,
                                positions=first_positions,
                                intermediate_tensors=first_intermediate,
                                inputs_embeds=first_inputs_embeds,
                            )

                        if (first_context.cudagraph_runtime_mode ==
                                CUDAGraphMode.FULL
                                and not first_context.capturing):
                            self._update_attn_params_for_ubatch(
                                first_context, first_num_tokens, 0)

                results_with_order.append(
                    (int(first_slice.request_slice.start), first_output,
                     getattr(first_context, "pad_size", 0)))

                for ubatch_id in range(1, len(ubatch_slices)):
                    ubatch_slice = ubatch_slices[ubatch_id]
                    current_num_tokens = (
                        ubatch_slice.token_slice.stop - ubatch_slice.token_slice.start)
                    current_num_reqs = (
                        ubatch_slice.request_slice.stop - ubatch_slice.request_slice.start)

                    split_attn_metadata = attn_metadata
                    if isinstance(attn_metadata, list):
                        split_attn_metadata = attn_metadata[ubatch_id]

                    restore_entry = self._refresh_parallel_slot_mapping(
                        split_attn_metadata, current_num_tokens)
                    if restore_entry is not None:
                        slot_restore_entries.append(restore_entry)

                    (cur_input_ids, cur_positions, cur_inputs_embeds,
                     cur_intermediate) = self._slice_model_inputs(
                         ubatch_slice.token_slice,
                         input_ids,
                         positions,
                         inputs_embeds,
                         intermediate_tensors,
                     )

                    run_input_ids = None
                    if cur_input_ids is not None:
                        self._parallel_input_ids[:current_num_tokens].copy_(
                            cur_input_ids, non_blocking=False)
                        run_input_ids = self._parallel_input_ids[:current_num_tokens]

                    run_positions = None
                    if cur_positions is not None:
                        if cur_positions.ndim == 2:
                            self._parallel_positions[:, :current_num_tokens].copy_(
                                cur_positions, non_blocking=False)
                            run_positions = self._parallel_positions[:,
                                                                    :current_num_tokens]
                        else:
                            self._parallel_positions[:current_num_tokens].copy_(
                                cur_positions, non_blocking=False)
                            run_positions = self._parallel_positions[:current_num_tokens]

                    run_inputs_embeds = None
                    if cur_inputs_embeds is not None:
                        self._parallel_inputs_embeds[:current_num_tokens].copy_(
                            cur_inputs_embeds, non_blocking=False)
                        run_inputs_embeds = self._parallel_inputs_embeds[:current_num_tokens]

                    if _SPLIT_LOCAL_CONTEXT_REBUILD:
                        rebuild_ubatch_slices = [
                            UBatchSlice(
                                slice(0, current_num_reqs),
                                slice(0, current_num_tokens),
                            )
                        ]
                        rebuild_positions = (
                            run_positions if run_positions is not None else positions)
                        rebuild_ubatch_num = 0
                    else:
                        rebuild_ubatch_slices = ubatch_slices
                        rebuild_positions = positions
                        rebuild_ubatch_num = ubatch_id

                    split_batch_descriptor = BatchDescriptor(
                        num_tokens=current_num_tokens,
                        num_reqs=current_num_reqs,
                        uniform=batch_descriptor.uniform,
                        has_lora=batch_descriptor.has_lora,
                    )
                    split_context = create_ascend_forward_context(
                        original_forward_context,
                        attn_metadata=split_attn_metadata,
                        vllm_config=self.vllm_config,
                        dp_metadata=original_forward_context.dp_metadata,
                        ubatch_slices=rebuild_ubatch_slices,
                        batch_descriptor=split_batch_descriptor,
                        cudagraph_runtime_mode=ubatch_cudagraph_mode,
                        ubatch_num=rebuild_ubatch_num,
                        positions=rebuild_positions,
                        in_parallel_streams=True,
                    )

                    with torch.npu.stream(self.parallel_stream):
                        with override_forward_context(split_context):
                            self._refresh_block_table_for_ubatch(
                                split_context, current_num_tokens)

                            if (split_context.cudagraph_runtime_mode !=
                                    CUDAGraphMode.NONE
                                    and self.aclgraph_wrapper is not None):
                                use_dedicated_pool = self._claim_dedicated_pool_capture(
                                    current_num_tokens)
                                try:
                                    split_output = self.aclgraph_wrapper(
                                        input_ids=run_input_ids,
                                        positions=run_positions,
                                        intermediate_tensors=cur_intermediate,
                                        inputs_embeds=run_inputs_embeds,
                                        parallel_streams=True,
                                        graph_pool=(self.pool
                                                    if use_dedicated_pool else None),
                                    )
                                except Exception:
                                    if use_dedicated_pool:
                                        self._rollback_dedicated_pool_capture()
                                    raise
                            else:
                                split_output = self.runnable(
                                    input_ids=run_input_ids,
                                    positions=run_positions,
                                    intermediate_tensors=cur_intermediate,
                                    inputs_embeds=run_inputs_embeds,
                                )

                            if (split_context.cudagraph_runtime_mode ==
                                    CUDAGraphMode.FULL
                                    and not split_context.capturing):
                                self._update_attn_params_for_ubatch(
                                    split_context, current_num_tokens,
                                    ubatch_id)

                    results_with_order.append(
                        (int(ubatch_slice.request_slice.start), split_output,
                         getattr(split_context, "pad_size", 0)))

                self.default_stream.synchronize()
                self.parallel_stream.synchronize()

            with override_forward_context(original_forward_context):
                sorted_entries = sorted(results_with_order, key=lambda x: x[0])
                sorted_results = [
                    self._snapshot_split_output(entry[1])
                    for entry in sorted_entries
                ]
                sorted_pad_sizes = [entry[2] for entry in sorted_entries]
                if get_forward_context().sp_enabled and get_pp_group().is_last_rank:
                    for i in range(len(sorted_results)):
                        sorted_results[i] = tensor_model_parallel_all_gather(
                            sorted_results[i], 0)
                        pad_size = sorted_pad_sizes[i]
                        if pad_size > 0:
                            sorted_results[i] = sorted_results[i][:-pad_size, :]

                if not get_pp_group().is_last_rank:
                    result = self._merge_intermediate_tensors(sorted_results)
                else:
                    result = torch.cat(sorted_results, dim=0)

                get_forward_context().dbo_enabled = False
                return result
        finally:
            for slot_attn_metadata, original_slot_mapping in slot_restore_entries:
                _set_slot_mapping_for_attn_metadata(slot_attn_metadata,
                                                    original_slot_mapping)
            torch.npu.synchronize()

    def _make_ubatch_metadata(
            self, ubatch_slices, attn_metadata, input_ids, positions,
            inputs_embeds, intermediate_tensors, compute_stream, dp_metadata,
            batch_descriptor,
            cudagraph_runtime_mode) -> list[AscendUbatchMetadata]:

        # Create one forward context per ubatch
        forward_contexts = []
        cur_forward_context = get_forward_context()

        for i, ubatch_slice in enumerate(ubatch_slices):
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_num_tokens = ubatch_slice.token_slice.stop - ubatch_slice.token_slice.start
            ubatch_num_reqs = ubatch_slice.request_slice.stop - ubatch_slice.request_slice.start
            ubatch_batch_descriptor = BatchDescriptor(
                num_tokens=ubatch_num_tokens,
                num_reqs=ubatch_num_reqs,
                uniform=batch_descriptor.uniform,
                has_lora=batch_descriptor.has_lora,
            )
            logger.info(
                "[SPLIT_DEBUG] ubatch %s: num_tokens=%s, num_reqs=%s, descriptor=%s",
                i,
                ubatch_num_tokens,
                ubatch_num_reqs,
                ubatch_batch_descriptor,
            )

            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=ubatch_slices,
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    ubatch_num=i,
                    positions=positions,
                ))

        ubatch_ctxs = forward_contexts

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = \
                self._slice_model_inputs(
                    ubatch_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors)
            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop -
                    ubatch_slice.token_slice.start))

        return ubatch_metadata

    def _slice_model_inputs(self, tokens_slice: slice, input_ids, positions,
                            inputs_embeds, intermediate_tensors):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[
            tokens_slice] if inputs_embeds is not None else None
        # consider pp scenario
        if intermediate_tensors is not None:
            # if enable sp, dbo should not split intermediate tensors using token_slice
            # instead, it should calculate the tensor lens after reduce scatter
            if enable_sp():
                tp_size = get_tensor_model_parallel_world_size()
                start = (tokens_slice.start + tp_size - 1) // tp_size
                if start != 0:
                    stop = start + (tokens_slice.stop - tokens_slice.start +
                                    tp_size - 1) // tp_size
                else:
                    stop = (tokens_slice.stop + tp_size - 1) // tp_size
                tokens_slice = slice(start, stop)

            sliced_intermediate_tensors = intermediate_tensors[
                tokens_slice] if intermediate_tensors is not None else None
        else:
            sliced_intermediate_tensors = None

        return (sliced_input_ids, sliced_positions, sliced_inputs_embeds,
                sliced_intermediate_tensors)

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices

        aclgraph_kwargs = dict(kwargs)
        aclgraph_kwargs["parallel_streams"] = self.enable_parallel

        # Keep one-time dedicated-pool capture for fixed-size graph.
        if ubatch_slices is None:
            if self.aclgraph_wrapper is not None:
                return self.aclgraph_wrapper(*args, **aclgraph_kwargs)
            return self.runnable(*args, **kwargs)
        if self._should_use_parallel_ubatches(ubatch_slices):
            return self._run_parallel_ubatches(*args, **aclgraph_kwargs)

        return self._run_ubatches(*args, **aclgraph_kwargs)

    def _merge_intermediate_tensors(self, intermediate_tensor_list):
        assert len(intermediate_tensor_list) > 0
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat(
                [x.tensors[key] for x in intermediate_tensor_list], dim=0)

        res = IntermediateTensors(result)
        return res
