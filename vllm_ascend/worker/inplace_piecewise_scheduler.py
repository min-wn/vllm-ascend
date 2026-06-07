# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.fx as fx
from torch.fx.node import map_arg

from vllm.compilation.piecewise_runtime import (
    PiecewiseRuntimeArgsCaptured,
    PiecewiseRuntimeCall,
    PiecewiseRuntimeHandle,
    capture_piecewise_runtime_args,
    get_piecewise_runtime_handle,
)
from vllm.forward_context import override_forward_context
from vllm.sequence import IntermediateTensors

from vllm_ascend import inplace_split_debug as split_debug


@dataclass
class InplacePiecewiseSplitInput:
    index: int
    context: Any
    input_ids: Optional[torch.Tensor]
    positions: torch.Tensor
    inputs_embeds: Optional[torch.Tensor]
    intermediate_tensors: Optional[IntermediateTensors]
    model_kwargs: dict[str, Any]
    runtime_call: PiecewiseRuntimeCall
    num_tokens: int
    graph_num_tokens: int
    start_num_tokens: int


@dataclass
class _AttentionTask:
    piece: Any
    node: fx.Node
    env: dict[Any, Any]
    split_input: InplacePiecewiseSplitInput
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    stream_event: Optional[torch.npu.Event] = None
    error: Optional[BaseException] = None


def _count_tensor_leaves(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, IntermediateTensors):
        return sum(_count_tensor_leaves(v) for v in value.tensors.values())
    if isinstance(value, dict):
        return sum(_count_tensor_leaves(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_tensor_leaves(v) for v in value)
    return 0


def _first_tensor_infos(value: Any,
                        *,
                        limit: int = 8) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []

    def _walk(item: Any, path: str) -> None:
        if len(infos) >= limit:
            return
        if isinstance(item, torch.Tensor):
            info = split_debug.tensor_info(item)
            if info is not None:
                info["path"] = path
                infos.append(info)
            return
        if isinstance(item, IntermediateTensors):
            for key, tensor in item.tensors.items():
                _walk(tensor, f"{path}.tensors[{key!s}]")
            return
        if isinstance(item, dict):
            for key, value in item.items():
                _walk(value, f"{path}[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            for idx, value in enumerate(item):
                _walk(value, f"{path}[{idx}]")

    _walk(value, "root")
    return infos


def _runtime_call_debug_payload(
        runtime_call: PiecewiseRuntimeCall) -> dict[str, Any]:
    handle = runtime_call.handle
    return {
        "cudagraph_copy_inputs": bool(handle.cudagraph_copy_inputs),
        "num_input_buffers": len(handle.input_buffers),
        "input_buffers": _first_tensor_infos(handle.input_buffers),
        "num_args": len(runtime_call.args),
        "num_kwargs": len(runtime_call.kwargs),
        "arg_tensor_count": _count_tensor_leaves(runtime_call.args),
        "kwarg_tensor_count": _count_tensor_leaves(runtime_call.kwargs),
        "arg_tensors": _first_tensor_infos(runtime_call.args),
        "kwarg_tensors": _first_tensor_infos(runtime_call.kwargs),
    }


def _looks_like_compiled_model(obj: Any, *,
                               _seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return False
    _seen.add(obj_id)

    if get_piecewise_runtime_handle(obj) is not None:
        return True
    if hasattr(obj, "dynamo_ctx") or hasattr(obj, "_compiled_call_impl"):
        return True

    for attr in ("runnable", "_orig_mod", "module", "model"):
        try:
            child = getattr(obj, attr)
        except Exception:
            continue
        if _looks_like_compiled_model(child, _seen=_seen):
            return True
    return False


def capture_piecewise_model_call(
    *,
    model: Any,
    metadata: Any,
    model_kwargs: dict[str, Any],
    stream: torch.npu.Stream,
) -> PiecewiseRuntimeCall:
    if not _looks_like_compiled_model(model):
        raise RuntimeError(
            "piecewise_attention_parallel requires a torch.compile model "
            "carrying vLLM piecewise runtime metadata")

    with capture_piecewise_runtime_args() as capture:
        try:
            with torch.inference_mode(), torch.npu.stream(stream):
                with override_forward_context(metadata.context):
                    model(
                        input_ids=metadata.input_ids,
                        positions=metadata.positions,
                        inputs_embeds=metadata.inputs_embeds,
                        intermediate_tensors=metadata.intermediate_tensors,
                        **model_kwargs,
                    )
        except PiecewiseRuntimeArgsCaptured:
            pass

    if len(capture.calls) != 1:
        raise RuntimeError(
            "Expected exactly one vLLM piecewise compiled call while "
            f"capturing split args, got {len(capture.calls)}")
    runtime_call = capture.calls[0]
    if split_debug.is_enabled():
        context = getattr(metadata, "context", None)
        split_debug.log_event(
            "piecewise_runtime_call_captured",
            {
                **_runtime_call_debug_payload(runtime_call),
                "batch_descriptor": split_debug.batch_descriptor_info(
                    getattr(context, "batch_descriptor", None)),
                "ubatch_num": getattr(context, "ubatch_num", None),
                "in_parallel_streams": bool(
                    getattr(context, "in_parallel_streams", False)),
            },
            step_id=getattr(context, "split_inplace_debug_step_id", None),
        )
    return runtime_call


class InplacePiecewiseSplitScheduler:

    def __init__(
        self,
        handle: PiecewiseRuntimeHandle,
        *,
        stream_main: torch.npu.Stream,
        stream_parallel: torch.npu.Stream,
        debug_step_id: Optional[int] = None,
        sync_policy: str = "event_chain",
        attention_enqueue_policy: str = "persistent_thread",
    ) -> None:
        if handle.cudagraph_copy_inputs:
            raise RuntimeError(
                "piecewise_attention_parallel does not yet support "
                "cudagraph_copy_inputs=True")
        self.handle = handle
        self.split_gm = handle.split_gm
        self.stream_main = stream_main
        self.stream_parallel = stream_parallel
        self.debug_step_id = debug_step_id
        self.sync_policy = str(sync_policy)
        self.attention_enqueue_policy = str(attention_enqueue_policy)
        valid_sync_policies = ("host_sync", "event_chain")
        if self.sync_policy not in valid_sync_policies:
            raise ValueError(
                "piecewise scheduler sync_policy must be one of "
                f"{valid_sync_policies}, got {self.sync_policy!r}")
        valid_attention_policies = ("per_piece_thread",
                                    "persistent_thread")
        if self.attention_enqueue_policy not in valid_attention_policies:
            raise ValueError(
                "piecewise attention enqueue policy must be one of "
                f"{valid_attention_policies}, got "
                f"{self.attention_enqueue_policy!r}")
        self.pieces = {
            item.submod_name: item
            for item in handle.piecewise_graphs
        }
        self._attention_queues: Optional[
            list[queue.Queue[Optional[_AttentionTask]]]] = None
        self._attention_threads: list[threading.Thread] = []

    @property
    def total_pieces(self) -> int:
        return len(self.handle.piecewise_graphs)

    @property
    def attention_pieces(self) -> int:
        return sum(
            1 for item in self.handle.piecewise_graphs
            if bool(getattr(item, "is_splitting_graph", False)))

    @property
    def capturable_pieces(self) -> int:
        return self.total_pieces - self.attention_pieces

    def run(self, split0: InplacePiecewiseSplitInput,
            split1: InplacePiecewiseSplitInput) -> tuple[Any, Any]:
        if split0.runtime_call.handle is not split1.runtime_call.handle:
            raise RuntimeError(
                "piecewise split inputs were captured from different "
                "compiled runtime handles")
        if split0.runtime_call.handle is not self.handle:
            raise RuntimeError(
                "piecewise scheduler handle does not match captured call "
                "handle")

        env0 = self._bind_placeholders(split0.runtime_call)
        env1 = self._bind_placeholders(split1.runtime_call)

        try:
            for node in self.split_gm.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    return (
                        self._resolve_arg(node.args[0], env0),
                        self._resolve_arg(node.args[0], env1),
                    )
                if node.op == "call_module" and str(node.target) in self.pieces:
                    piece = self.pieces[str(node.target)]
                    if bool(getattr(piece, "is_splitting_graph", False)):
                        out0, out1 = self._run_piece_parallel(
                            piece, node, env0, env1, split0, split1)
                    else:
                        out0, out1 = self._run_piece_serial(
                            piece, node, env0, env1, split0, split1)
                    env0[node] = out0
                    env1[node] = out1
                    continue

                env0[node] = self._eval_node(node, env0)
                env1[node] = self._eval_node(node, env1)
        finally:
            self._shutdown_attention_workers()

        raise RuntimeError("Piecewise split graph did not produce output")

    def _bind_placeholders(self,
                           runtime_call: PiecewiseRuntimeCall) -> dict[Any, Any]:
        env: dict[Any, Any] = {}
        args_iter = iter(runtime_call.args)
        kwargs = runtime_call.kwargs
        for node in self.split_gm.graph.nodes:
            if node.op != "placeholder":
                continue
            target = str(node.target)
            if target.startswith("**"):
                env[node] = kwargs
                continue
            if target.startswith("*"):
                env[node] = tuple(args_iter)
                continue
            try:
                env[node] = next(args_iter)
                continue
            except StopIteration:
                pass
            if target in kwargs:
                env[node] = kwargs[target]
                continue
            if node.args:
                env[node] = node.args[0]
                continue
            raise RuntimeError(
                "Cannot bind piecewise placeholder "
                f"{target!r}; captured args={len(runtime_call.args)}, "
                f"kwargs={sorted(kwargs)}")
        return env

    def _resolve_arg(self, arg: Any, env: dict[Any, Any]) -> Any:
        return map_arg(arg, lambda node: env[node])

    def _fetch_attr(self, target: str) -> Any:
        target_atoms = target.split(".")
        attr_itr = self.split_gm
        for atom in target_atoms:
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def _eval_node(self, node: fx.Node, env: dict[Any, Any]) -> Any:
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
            "Unsupported piecewise FX node: "
            f"op={node.op!r}, target={node.target!r}, node={node!r}")

    def _call_piece(self, node: fx.Node, env: dict[Any, Any]) -> Any:
        args = self._resolve_arg(node.args, env)
        kwargs = self._resolve_arg(node.kwargs, env)
        module = self._fetch_attr(str(node.target))
        return module(*args, **kwargs)

    def _stream_for_split(self, split_idx: int) -> torch.npu.Stream:
        return self.stream_parallel if split_idx > 0 else self.stream_main

    def _piece_has_graph(self, node: fx.Node,
                         split_input: InplacePiecewiseSplitInput
                         ) -> Optional[bool]:
        try:
            module = self._fetch_attr(str(node.target))
        except Exception:
            return None
        has_graph = getattr(module, "has_graph", None)
        if not callable(has_graph):
            return None
        try:
            return bool(
                has_graph(
                    getattr(split_input.context, "batch_descriptor", None),
                    bool(split_input.index > 0),
                ))
        except Exception:
            return None

    def _piece_payload(self, piece: Any,
                       split_input: InplacePiecewiseSplitInput,
                       *,
                       node: Optional[fx.Node] = None) -> dict[str, Any]:
        payload = {
            "piece_name": getattr(piece, "submod_name", None),
            "graph_id": getattr(piece, "graph_id", None),
            "is_splitting_graph": bool(
                getattr(piece, "is_splitting_graph", False)),
            "split_idx": int(split_input.index),
            "num_tokens": int(split_input.num_tokens),
            "graph_num_tokens": int(split_input.graph_num_tokens),
            "start_num_tokens": int(split_input.start_num_tokens),
            "in_parallel_streams": bool(split_input.index > 0),
        }
        if node is not None and split_debug.is_enabled():
            payload["hit_graph_before_call"] = self._piece_has_graph(
                node, split_input)
        return payload

    def _log_event(self, event: str, payload: dict[str, Any]) -> None:
        if split_debug.is_enabled():
            split_debug.log_event(
                event,
                payload,
                step_id=self.debug_step_id,
            )

    def _record_stream_event(self, stream: torch.npu.Stream) -> torch.npu.Event:
        event = torch.npu.Event()
        event.record(stream)
        return event

    def _wait_stream_event(self, stream: torch.npu.Stream,
                           event: Optional[torch.npu.Event]) -> None:
        if event is not None:
            stream.wait_event(event)

    def _cross_wait_parallel_events(
        self,
        event0: Optional[torch.npu.Event],
        event1: Optional[torch.npu.Event],
    ) -> None:
        if self.sync_policy == "host_sync":
            self.stream_main.synchronize()
            self.stream_parallel.synchronize()
            return
        self._wait_stream_event(self.stream_main, event1)
        self._wait_stream_event(self.stream_parallel, event0)

    def _run_piece_serial(self, piece: Any, node: fx.Node,
                          env0: dict[Any, Any], env1: dict[Any, Any],
                          split0: InplacePiecewiseSplitInput,
                          split1: InplacePiecewiseSplitInput) -> tuple[Any, Any]:
        self._log_event("piecewise_split_piece_start",
                        self._piece_payload(piece, split0, node=node))
        with torch.inference_mode(), torch.npu.stream(self.stream_main):
            with override_forward_context(split0.context):
                out0 = self._call_piece(node, env0)
        if self.sync_policy == "host_sync":
            self.stream_main.synchronize()
        else:
            event0 = self._record_stream_event(self.stream_main)
            self._wait_stream_event(self.stream_parallel, event0)
        self._log_event("piecewise_split_piece_end",
                        self._piece_payload(piece, split0))

        self._log_event("piecewise_split_piece_start",
                        self._piece_payload(piece, split1, node=node))
        with torch.inference_mode(), torch.npu.stream(self.stream_parallel):
            with override_forward_context(split1.context):
                out1 = self._call_piece(node, env1)
        if self.sync_policy == "host_sync":
            self.stream_parallel.synchronize()
        else:
            event1 = self._record_stream_event(self.stream_parallel)
            self._wait_stream_event(self.stream_main, event1)
        self._log_event("piecewise_split_piece_end",
                        self._piece_payload(piece, split1))
        return out0, out1

    def _execute_attention_task(self, task: _AttentionTask) -> None:
        try:
            stream = self._stream_for_split(task.split_input.index)
            self._log_event("piecewise_split_piece_start",
                            self._piece_payload(
                                task.piece, task.split_input, node=task.node))
            with torch.inference_mode(), torch.npu.stream(stream):
                with override_forward_context(task.split_input.context):
                    task.result = self._call_piece(task.node, task.env)
                if self.sync_policy == "event_chain":
                    task.stream_event = self._record_stream_event(stream)
            self._log_event("piecewise_split_piece_end",
                            self._piece_payload(task.piece, task.split_input))
        except BaseException as exc:
            task.error = exc
        finally:
            task.done.set()

    def _attention_worker_loop(self, split_idx: int) -> None:
        assert self._attention_queues is not None
        task_queue = self._attention_queues[split_idx]
        while True:
            task = task_queue.get()
            if task is None:
                return
            self._execute_attention_task(task)

    def _ensure_attention_workers(self) -> None:
        if self._attention_queues is not None:
            return
        self._attention_queues = [queue.Queue(), queue.Queue()]
        self._attention_threads = []
        for split_idx, name in (
                (0, "piecewise-attention-main"),
                (1, "piecewise-attention-parallel")):
            thread = threading.Thread(
                target=self._attention_worker_loop,
                args=(split_idx,),
                name=name,
                daemon=True,
            )
            thread.start()
            self._attention_threads.append(thread)

    def _shutdown_attention_workers(self) -> None:
        if self._attention_queues is None:
            return
        for task_queue in self._attention_queues:
            task_queue.put(None)
        for thread in self._attention_threads:
            thread.join()
        self._attention_queues = None
        self._attention_threads = []

    def _run_attention_tasks(self, tasks: list[_AttentionTask]) -> None:
        if self.attention_enqueue_policy == "persistent_thread":
            self._ensure_attention_workers()
            assert self._attention_queues is not None
            for task in tasks:
                self._attention_queues[task.split_input.index].put(task)
        else:
            threads = [
                threading.Thread(
                    target=self._execute_attention_task,
                    args=(task,),
                    name=f"piecewise-attention-split-{task.split_input.index}",
                )
                for task in tasks
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return

        for task in tasks:
            task.done.wait()

    def _run_piece_parallel(
        self,
        piece: Any,
        node: fx.Node,
        env0: dict[Any, Any],
        env1: dict[Any, Any],
        split0: InplacePiecewiseSplitInput,
        split1: InplacePiecewiseSplitInput,
    ) -> tuple[Any, Any]:
        self._log_event(
            "piecewise_split_attention_parallel_start",
            {
                "piece_name": getattr(piece, "submod_name", None),
                "graph_id": getattr(piece, "graph_id", None),
            },
        )

        tasks = [
            _AttentionTask(piece=piece, node=node, env=env0,
                           split_input=split0),
            _AttentionTask(piece=piece, node=node, env=env1,
                           split_input=split1),
        ]
        self._run_attention_tasks(tasks)

        errors = [
            (task.split_input.index, task.error)
            for task in tasks
            if task.error is not None
        ]
        if errors:
            errors.sort(key=lambda item: item[0])
            failed_idx, first_error = errors[0]
            raise RuntimeError(
                "piecewise attention parallel failed at "
                f"split_idx={failed_idx}") from first_error

        self._cross_wait_parallel_events(
            tasks[0].stream_event,
            tasks[1].stream_event,
        )
        self._log_event(
            "piecewise_split_attention_parallel_end",
            {
                "piece_name": getattr(piece, "submod_name", None),
                "graph_id": getattr(piece, "graph_id", None),
            },
        )
        return tasks[0].result, tasks[1].result
