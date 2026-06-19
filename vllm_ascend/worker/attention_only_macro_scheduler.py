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

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.fx as fx
from vllm.compilation.piecewise_runtime import (
    PiecewiseRuntimeArgsCaptured,
    PiecewiseRuntimeCall,
    PiecewiseRuntimeHandle,
    capture_piecewise_runtime_args,
)
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import logger

from vllm_ascend.worker.ubatch_utils import SplitBatchSlice, SplitBatchSlices


def capture_piecewise_model_call(
        model: Any,
        *,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: Optional[torch.Tensor],
        model_kwargs: dict[str, Any]) -> PiecewiseRuntimeCall:
    """Capture the compiled piecewise callable and its real runtime args."""
    with capture_piecewise_runtime_args() as capture:
        try:
            model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        except PiecewiseRuntimeArgsCaptured:
            pass

    if not capture.calls:
        raise RuntimeError(
            "attention_only_macro requires a vLLM piecewise compiled model, "
            "but no PiecewiseRuntimeHandle was captured. Use PIECEWISE "
            "cudagraph mode and VLLM_COMPILE.")
    if len(capture.calls) != 1:
        raise RuntimeError(
            "attention_only_macro expected exactly one piecewise runtime "
            f"call, got {len(capture.calls)}")
    return capture.calls[0]


@dataclass
class AttentionOnlySplitRuntime:
    split_batch_slices: SplitBatchSlices
    split_contexts: list[Any]
    stream_main: torch.npu.Stream
    stream_parallel: torch.npu.Stream
    require_exact_graph_tokens: bool = True

    @property
    def num_splits(self) -> int:
        return len(self.split_batch_slices)

    @property
    def full_num_tokens(self) -> int:
        return int(max(s.token_slice.stop for s in self.split_batch_slices))

    def validate(self, handle: PiecewiseRuntimeHandle) -> None:
        if self.num_splits != 2:
            raise RuntimeError(
                "attention_only_macro currently supports exactly 2 splits, "
                f"got {self.num_splits}")
        if len(self.split_contexts) != self.num_splits:
            raise RuntimeError(
                "attention_only_macro split context count mismatch: "
                f"contexts={len(self.split_contexts)}, "
                f"splits={self.num_splits}")
        if handle.cudagraph_copy_inputs:
            raise RuntimeError(
                "attention_only_macro does not support "
                "cudagraph_copy_inputs because it would rebind symbolic "
                "runtime tensors before split slicing.")
        expected_start = 0
        for idx, split_slice in enumerate(self.split_batch_slices):
            if int(split_slice.token_slice.start) != expected_start:
                raise RuntimeError(
                    "attention_only_macro expects contiguous splits from "
                    f"token 0: idx={idx}, expected_start={expected_start}, "
                    f"actual_start={split_slice.token_slice.start}")
            expected_start = int(split_slice.token_slice.stop)
            if int(split_slice.token_slice.start) != int(
                    split_slice.start_num_tokens):
                raise RuntimeError(
                    "attention_only_macro expects contiguous token-offset "
                    "splits: "
                    f"idx={idx}, token_start={split_slice.token_slice.start}, "
                    f"start_num_tokens={split_slice.start_num_tokens}")
            if (self.require_exact_graph_tokens
                    and int(split_slice.graph_num_tokens)
                    != int(split_slice.num_tokens)):
                raise RuntimeError(
                    "attention_only_macro MVP requires exact graph tokens "
                    "for attention pieces. Disable split padding or extend "
                    "the scheduler with padded intermediate buffers: "
                    f"idx={idx}, num_tokens={split_slice.num_tokens}, "
                    f"graph_num_tokens={split_slice.graph_num_tokens}")


class AttentionOnlySplitInterpreter(fx.Interpreter):
    """Run a piecewise graph with only attention pieces split across streams."""

    def __init__(self, handle: PiecewiseRuntimeHandle,
                 runtime: AttentionOnlySplitRuntime):
        super().__init__(handle.split_gm)
        self.handle = handle
        self.runtime = runtime
        self.piece_by_name = {
            item.submod_name: item
            for item in handle.piecewise_graphs
        }
        self.attention_piece_count = 0

    def call_module(self, target: fx.node.Target, args: tuple[Any, ...],
                    kwargs: dict[str, Any]) -> Any:
        target_name = str(target)
        item = self.piece_by_name.get(target_name)
        submod = self.fetch_attr(target)
        if item is not None and item.is_splitting_graph:
            self.attention_piece_count += 1
            return self._run_attention_piece(submod, args, kwargs)

        with torch.npu.stream(self.runtime.stream_main):
            with override_forward_context(get_forward_context()):
                return submod(*args, **kwargs)

    def _run_attention_piece(self, submod: Any, args: tuple[Any, ...],
                             kwargs: dict[str, Any]) -> Any:
        self.runtime.stream_parallel.wait_stream(self.runtime.stream_main)

        outputs: list[Any] = [None] * self.runtime.num_splits
        errors: list[tuple[int, BaseException]] = []
        error_lock = threading.Lock()

        def _worker(split_idx: int) -> None:
            split_slice = self.runtime.split_batch_slices[split_idx]
            stream = (self.runtime.stream_parallel
                      if split_idx > 0 else self.runtime.stream_main)
            context = self.runtime.split_contexts[split_idx]
            try:
                split_args = self._slice_value(args, split_slice)
                split_kwargs = self._slice_value(kwargs, split_slice)
                with torch.npu.stream(stream):
                    with torch.inference_mode(), override_forward_context(
                            context):
                        outputs[split_idx] = submod(*split_args,
                                                    **split_kwargs)
            except BaseException as exc:
                with error_lock:
                    errors.append((split_idx, exc))

        workers = [
            threading.Thread(target=_worker,
                             args=(split_idx,),
                             name=f"attention-only-split-{split_idx}")
            for split_idx in range(self.runtime.num_splits)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        if errors:
            errors.sort(key=lambda item: item[0])
            split_idx, first_error = errors[0]
            raise RuntimeError(
                "attention_only_macro attention split failed at "
                f"split_idx={split_idx}") from first_error

        self.runtime.stream_main.wait_stream(self.runtime.stream_parallel)
        with torch.npu.stream(self.runtime.stream_main):
            return self._merge_outputs(outputs)

    def _slice_value(self, value: Any, split_slice: SplitBatchSlice) -> Any:
        if isinstance(value, torch.Tensor):
            return self._slice_tensor(value, split_slice)
        if isinstance(value, tuple):
            return tuple(self._slice_value(item, split_slice)
                         for item in value)
        if isinstance(value, list):
            return [self._slice_value(item, split_slice) for item in value]
        if isinstance(value, dict):
            return {
                key: self._slice_value(item, split_slice)
                for key, item in value.items()
            }
        return value

    def _slice_tensor(self, tensor: torch.Tensor,
                      split_slice: SplitBatchSlice) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor
        full_tokens = self.runtime.full_num_tokens
        if int(tensor.shape[0]) != full_tokens:
            return tensor
        start = int(split_slice.token_slice.start)
        stop = start + int(split_slice.graph_num_tokens)
        if stop > int(tensor.shape[0]):
            raise RuntimeError(
                "attention_only_macro split tensor slice exceeds tensor "
                f"shape: start={start}, stop={stop}, "
                f"shape={tuple(tensor.shape)}")
        return tensor[start:stop]

    def _merge_outputs(self, outputs: list[Any]) -> Any:
        if any(output is None for output in outputs):
            raise RuntimeError(
                "attention_only_macro missing split attention output")
        return self._merge_value(outputs)

    def _merge_value(self, values: list[Any]) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            return self._merge_tensor(values)
        if isinstance(first, tuple):
            return tuple(
                self._merge_value([value[idx] for value in values])
                for idx in range(len(first)))
        if isinstance(first, list):
            return [
                self._merge_value([value[idx] for value in values])
                for idx in range(len(first))
            ]
        if isinstance(first, dict):
            return {
                key: self._merge_value([value[key] for value in values])
                for key in first.keys()
            }
        return first

    def _merge_tensor(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        full_tokens = self.runtime.full_num_tokens
        first = tensors[0]
        if first.ndim == 0:
            return first
        expected_first_tokens = int(
            self.runtime.split_batch_slices[0].num_tokens)
        if int(first.shape[0]) != expected_first_tokens:
            raise RuntimeError(
                "attention_only_macro can only merge token-major attention "
                "outputs: "
                f"first_shape={tuple(first.shape)}, "
                f"expected_first_tokens={expected_first_tokens}")
        merged_shape = (full_tokens, ) + tuple(first.shape[1:])
        merged = first.new_empty(merged_shape)
        for tensor, split_slice in zip(tensors, self.runtime.split_batch_slices):
            start = int(split_slice.token_slice.start)
            stop = int(split_slice.token_slice.stop)
            actual_tokens = int(split_slice.num_tokens)
            if int(tensor.shape[0]) < actual_tokens:
                raise RuntimeError(
                    "attention_only_macro split output has too few tokens: "
                    f"shape={tuple(tensor.shape)}, "
                    f"actual_tokens={actual_tokens}")
            merged[start:stop].copy_(tensor[:actual_tokens])
        return merged


def run_attention_only_split_piecewise(
        handle: PiecewiseRuntimeHandle,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        runtime: AttentionOnlySplitRuntime) -> Any:
    runtime.validate(handle)
    interpreter = AttentionOnlySplitInterpreter(handle, runtime)
    interpreter_args = _bind_interpreter_args(handle, args, kwargs)
    result = interpreter.run(*interpreter_args)
    if interpreter.attention_piece_count == 0:
        raise RuntimeError(
            "attention_only_macro did not find any attention splitting "
            "piece. Check compilation_config.splitting_ops.")
    logger.debug(
        "attention_only_macro executed %d attention splitting pieces",
        interpreter.attention_piece_count)
    while (isinstance(result, (list, tuple)) and len(result) == 1
           and isinstance(result[0], (list, tuple, torch.Tensor))):
        result = result[0]
    return result


def _bind_interpreter_args(handle: PiecewiseRuntimeHandle,
                           args: tuple[Any, ...],
                           kwargs: dict[str, Any]) -> tuple[Any, ...]:
    if not kwargs:
        return args

    placeholders = [
        node for node in handle.split_gm.graph.nodes
        if node.op == "placeholder"
    ]
    bound_args: list[Any] = []
    positional_idx = 0
    used_kwargs: set[str] = set()
    for node in placeholders:
        target = str(node.target)
        if target in kwargs:
            bound_args.append(kwargs[target])
            used_kwargs.add(target)
        elif positional_idx < len(args):
            bound_args.append(args[positional_idx])
            positional_idx += 1
        else:
            raise RuntimeError(
                "attention_only_macro cannot bind piecewise placeholder "
                f"{target!r}; args={len(args)}, kwargs={list(kwargs.keys())}")

    if positional_idx != len(args):
        raise RuntimeError(
            "attention_only_macro received extra positional piecewise args: "
            f"used={positional_idx}, total={len(args)}")
    extra_kwargs = set(kwargs) - used_kwargs
    if extra_kwargs:
        raise RuntimeError(
            "attention_only_macro received kwargs that do not match "
            f"piecewise placeholders: {sorted(extra_kwargs)}")
    return tuple(bound_args)
