from __future__ import annotations

import hashlib
import importlib.metadata
from collections import Counter
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from vllm.compilation import breakable_cudagraph
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, AscendMetadata
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.worker.multi_graph.buckets import BucketCostTable, KVBucketPolicy
from vllm_ascend.worker.multi_graph.buffers import LaneBuffers
from vllm_ascend.worker.multi_graph.calibration import CalibrationPolicy, OnlineCalibrator
from vllm_ascend.worker.multi_graph.costs import load_cost_table
from vllm_ascend.worker.multi_graph.manager import ManagerState, MultiGraphEntry, MultiGraphManager
from vllm_ascend.worker.multi_graph.npu_backend import NPUExecutionBackend
from vllm_ascend.worker.multi_graph.planner import CostKey, ExactDecodePlanner
from vllm_ascend.worker.multi_graph.support import RuntimeProfile
from vllm_ascend.worker.multi_graph.types import BatchSummary, CaptureSignature, ExecutionPlan, GraphKey, SlicePlan
from vllm_ascend.worker.utils import disable_compilation


class MultiGraphV1Adapter:
    """Experimental exact decode adapter using startup capture and private pools."""

    def __init__(self, runner: Any):
        self.runner = runner
        torch.npu.synchronize()
        self.memory_start_free = torch.npu.mem_get_info()[0]
        self.config = runner.ascend_config.multi_graph_config
        self.planner = ExactDecodePlanner(self.config)
        self.manager = MultiGraphManager(NPUExecutionBackend(), self.config.max_graph_entries)
        self.execution_count = 0
        self.last_plan = None
        self.fallback_counts: Counter[str] = Counter()
        self.cost_table = {}
        self.layers = runner.compilation_config.static_forward_context
        self.validate_runner()
        self.kv_addresses = self.current_kv_addresses()
        identity_config = copy(runner.vllm_config)
        identity_config.additional_config = dict(runner.vllm_config.additional_config)
        identity_config.additional_config.pop("multi_graph_config", None)
        config_hash = hashlib.sha256((identity_config.compute_hash() + self.config.capture_hash()).encode()).hexdigest()
        self.signature = CaptureSignature(
            buffer_generation=0,
            model_config_hash=config_hash,
            device_identity=str(runner.device),
            dtype=str(runner.dtype).removeprefix("torch."),
            attention_backend="AscendAttentionBackend",
            attention_variant="FIA",
            kv_layout="standard-dense",
            block_size=runner.block_size,
            runtime_identity="|".join(
                [torch.npu.get_device_name(runner.device)]
                + [
                    f"{name}={importlib.metadata.version(name)}"
                    for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
                ]
            ),
        )
        if self.config.cost_table_path is not None:
            self.cost_table = load_cost_table(
                self.config.cost_table_path, self.signature, self.config.minimum_cost_samples
            )
        if self.config.kv_bucket_upper_bounds:
            self.cost_table = BucketCostTable(
                self.cost_table,
                KVBucketPolicy(self.config.kv_bucket_upper_bounds, self.config.minimum_bucket_vectors),
            )
        self.compute_streams = (torch.npu.Stream(), torch.npu.Stream())
        self.update_streams = (torch.npu.Stream(), torch.npu.Stream())
        self.calibrator = None
        if self.config.online_calibration:
            self.calibrator = OnlineCalibrator(
                self.signature,
                CalibrationPolicy(
                    self.config.minimum_cost_samples,
                    self.config.calibration_maximum_keys,
                    self.config.calibration_sample_interval,
                ),
            )
        self.profile = RuntimeProfile(
            runner_version="v1",
            device_family=(
                "910C" if get_current_hardware_profile().device_addressing_mode.name == "DUAL_CHIP_CARD" else "910B"
            ),
            model_verified=True,
            parallel_sizes=(1, 1, 1, 1, 1),
            expert_parallel=False,
            configured_graph_mode="FULL_DECODE_ONLY",
            unsupported_features=frozenset(),
        )

    def validate_runner(self) -> None:
        runner = self.runner
        if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.STANDARD:
            raise ValueError("multi-graph currently requires the standard Ascend device adaptor")
        if runner.enable_enpu or breakable_cudagraph.is_breakable_cudagraph_enabled():
            raise ValueError("multi-graph cannot use ENPU or breakable graphs")
        if runner.device_metadata_executor is not None or runner.use_aux_hidden_state_outputs or runner.uses_mrope:
            raise ValueError("multi-graph cannot use device metadata, auxiliary outputs or multi-dimensional RoPE")
        if runner.offload_config.uva.cpu_offload_gb:
            raise ValueError("multi-graph cannot use weight offload/prefetch")
        if len(runner.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("multi-graph requires a single dense KV group")
        for layer in self.layers.values():
            impl = getattr(layer, "impl", None)
            if impl is None:
                continue
            if type(impl) is not AscendAttentionBackendImpl or impl.sliding_window or impl.sinks is not None:
                raise ValueError("multi-graph requires standard dense FIA attention without sliding windows or sinks")
            for sizes in self.config.capture_sizes_per_lane:
                if any(using_paged_attention(size, runner.vllm_config, impl.head_size) for size in sizes):
                    raise ValueError(
                        "multi-graph capture sizes must select FIA; exclude these sizes from pa_shape_list"
                    )

    def current_kv_addresses(self) -> tuple[int, ...]:
        addresses = []
        for layer in self.layers.values():
            cache = getattr(layer, "kv_cache", ())
            if isinstance(cache, torch.Tensor):
                addresses.append(cache.data_ptr())
            else:
                addresses.extend(tensor.data_ptr() for tensor in cache if isinstance(tensor, torch.Tensor))
        return tuple(addresses)

    def capture(self) -> int:
        torch.npu.synchronize()
        before = self.memory_start_free
        try:
            for lane, sizes in enumerate(self.config.capture_sizes_per_lane):
                for size in sizes:
                    self.runner._multi_graph_capture_callback = lambda inputs, lane=lane, size=size: self.capture_entry(
                        inputs, lane, size
                    )
                    self.runner._dummy_run(
                        size,
                        force_attention=True,
                        uniform_decode=True,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        allow_microbatching=False,
                    )
                    torch.npu.synchronize()
                    used = max(0, before - torch.npu.mem_get_info()[0])
                    if used > self.config.graph_memory_budget_bytes:
                        raise RuntimeError("multi-graph capture exceeded its reserved memory budget")
            self.manager.ready()
        except Exception:
            self.manager.startup_failed()
            raise
        finally:
            self.runner._multi_graph_capture_callback = None
        logger.warning("Experimental multi-graph decode captured %d independent entries", len(self.manager.entries))
        return max(0, before - torch.npu.mem_get_info()[0])

    def forward_context(self, buffers: LaneBuffers, size: int, capture: bool):
        return set_ascend_forward_context(
            buffers.metadata,
            self.runner.vllm_config,
            num_tokens=size,
            num_actual_tokens=size,
            aclgraph_runtime_mode=CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=size, num_reqs=size, uniform=True),
            model_instance=self.runner.get_model(),
            skip_compiled=True,
        )

    def capture_entry(self, inputs: dict[str, Any], lane: int, size: int):
        source = get_forward_context().attn_metadata
        if not source or any(type(metadata) is not AscendMetadata for metadata in source.values()):
            raise ValueError("multi-graph requires standard AscendMetadata")
        key = GraphKey(lane, size, size, self.signature)
        part = SlicePlan(0, size, 0, size, key)
        buffers = LaneBuffers.create(inputs["input_ids"], inputs["positions"], source, part)
        for metadata in buffers.metadata.values():
            metadata.slot_mapping.fill_(-1)
            metadata.block_tables.zero_()
        buffers.input_ids.zero_()
        buffers.positions.zero_()
        model = self.runner.get_model()
        compute = self.compute_streams[lane]
        compute.wait_stream(torch.npu.current_stream())
        with disable_compilation(model), torch.npu.stream(compute):
            with self.forward_context(buffers, size, False):
                for _ in range(2):
                    model(input_ids=buffers.input_ids, positions=buffers.positions)
            compute.synchronize()
            pool = torch.npu.graph_pool_handle()
            graph = UpdatableGraph()
            with self.forward_context(buffers, size, True):
                get_forward_context().capturing = True
                try:
                    with torch.npu.graph(graph, pool=pool, stream=compute):
                        output = model(input_ids=buffers.input_ids, positions=buffers.positions)
                finally:
                    get_forward_context().capturing = False
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or not graph.tasks:
            raise ValueError("multi-graph capture requires tensor hidden states and registered FIA update tasks")
        entry = MultiGraphEntry(key, buffers, graph, pool, compute, self.update_streams[lane], output)
        self.manager.backend.resolve(entry)
        self.manager.add_entry(entry)
        return output

    def try_run(self, context: Any, inputs: dict[str, Any], count: int, baseline=None) -> torch.Tensor | None:
        if self.current_kv_addresses() != self.kv_addresses:
            self.manager.invalidate()
            raise RuntimeError("multi-graph KV cache addresses changed; recreate the worker")
        if context.device_metadata_executor is not None or set(inputs) != {
            "input_ids",
            "positions",
            "intermediate_tensors",
            "inputs_embeds",
        }:
            return None
        if inputs["intermediate_tensors"] is not None or inputs["inputs_embeds"] is not None:
            return None
        if inputs["input_ids"] is None or inputs["positions"] is None:
            return None
        metadata = context.attn_metadata
        if not metadata or any(type(value) is not AscendMetadata for value in metadata.values()):
            return None
        first = next(iter(metadata.values()))
        if any(
            value.attn_state != AscendAttentionState.DecodeOnly
            or value.max_query_len != 1
            or value.num_prefills
            or value.num_decodes < count
            for value in metadata.values()
        ):
            return None
        if count != len(self.runner.input_batch.req_ids):
            return None
        lengths = tuple(first.seq_lens_list[:count])
        block_table = self.runner.input_batch.block_table[0].block_table.np
        slots = [
            int(block_table[index, (length - 1) // self.runner.block_size]) * self.runner.block_size
            + (length - 1) % self.runner.block_size
            for index, length in enumerate(lengths)
        ]
        if len(set(slots)) != count:
            return None
        summary = BatchSummary((1,) * count, lengths, full_decode=True)
        plan = self.planner.plan(summary, self.profile, self.signature, self.manager.entries.keys(), self.cost_table)
        calibrator = getattr(self, "calibrator", None)
        if calibrator is not None and not isinstance(plan, ExecutionPlan):
            candidate = ExactDecodePlanner(
                replace(self.config, online_calibration=False, selection_policy="force")
            ).plan(summary, self.profile, self.signature, self.manager.entries.keys())
            if isinstance(candidate, ExecutionPlan):
                if baseline is None:
                    raise RuntimeError("online calibration requires a baseline forward callback")
                key = CostKey(tuple(part.graph_key for part in candidate.slices), candidate.execution, lengths)
                previous_revision = calibrator.revision

                def execute_split():
                    result = self.manager.run(candidate, inputs["input_ids"], inputs["positions"], metadata)
                    self.execution_count += 1
                    self.last_plan = candidate
                    return result

                result = calibrator.dispatch(key, baseline, execute_split, torch.npu.synchronize)
                if calibrator.revision != previous_revision:
                    exact = dict(self.cost_table)
                    exact.update(calibrator.document().cost_table(self.signature, self.config.minimum_cost_samples))
                    self.cost_table = (
                        BucketCostTable(
                            exact,
                            KVBucketPolicy(self.config.kv_bucket_upper_bounds, self.config.minimum_bucket_vectors),
                        )
                        if self.config.kv_bucket_upper_bounds
                        else exact
                    )
                    calibrator.save(Path(self.config.calibration_output_path))
                return result
        if not isinstance(plan, ExecutionPlan):
            self.fallback_counts[plan.reason.value] += 1
            return None
        try:
            output = self.manager.run(plan, inputs["input_ids"], inputs["positions"], metadata)
        except (ValueError, KeyError):
            if self.manager.state != ManagerState.READY:
                raise
            return None
        self.execution_count += 1
        self.last_plan = plan
        return output
