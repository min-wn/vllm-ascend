# SPDX-License-Identifier: Apache-2.0
"""Legacy host-length FIA control for the isolated stage-three experiment.

Every replay performs the real graph-task update for every captured attention
occurrence. Physical Q4 requests remain in the graph; inactive requests retain
their true zero KV lengths. The complete output is exposed without trimming or
postprocessing, so the caller can check padding writes independently.
"""

from dataclasses import dataclass

QUERY_LENGTH = 4
QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 128
UNBOUNDED_WINDOW = 2147483647
WARMUP_ITERATIONS = 2


def validate_lengths(query_lengths, kv_lengths, capacity, maximum_kv_length):
    """Validate CPU metadata without changing zero-length padding semantics."""
    if len(query_lengths) != capacity or len(kv_lengths) != capacity:
        raise ValueError("one query endpoint and true KV length per physical request required")
    expected = [QUERY_LENGTH * (index + 1) for index in range(capacity)]
    if list(query_lengths) != expected or any(type(value) is not int for value in query_lengths):
        raise ValueError("legacy control requires fixed cumulative Q4 endpoints covering the complete query")
    if any(type(value) is not int or not 0 <= value <= maximum_kv_length for value in kv_lengths):
        raise ValueError("true KV lengths must be CPU integers within the allocated page-table extent")
    return list(query_lengths), list(kv_lengths)


@dataclass(frozen=True)
class _LengthProvider:
    occurrence: int

    def resolve(self, context):
        return {
            "actual_seq_lengths": context["query_lengths"],
            "actual_seq_lengths_kv": context["kv_lengths"],
            "block_table": context["block_table"],
        }


def _paged_view(tensor):
    if not tensor.is_contiguous() or tensor.ndim != 4 or tuple(tensor.shape[1:]) != (PAGE_SIZE, KV_HEADS, HEAD_DIM):
        raise ValueError("legacy FIA requires contiguous paged ND [pages,128,2,256] storage")
    return tensor.view(tensor.shape[0], PAGE_SIZE, KV_HEADS * HEAD_DIM)


def _operation_kwargs(layer, block_table, atten_mask, query_lengths, kv_lengths):
    return {
        "query": layer["query"],
        "key": _paged_view(layer["key"]),
        "value": _paged_view(layer["value"]),
        "atten_mask": atten_mask,
        "block_table": block_table,
        "input_layout": "TND",
        "block_size": PAGE_SIZE,
        "actual_seq_lengths": query_lengths,
        "actual_seq_lengths_kv": kv_lengths,
        "num_key_value_heads": KV_HEADS,
        "num_heads": QUERY_HEADS,
        "pre_tokens": UNBOUNDED_WINDOW,
        "next_tokens": UNBOUNDED_WINDOW,
        "scale": HEAD_DIM**-0.5,
        "sparse_mode": 3,
    }


class LegacyGraph:
    """One sequential graph with one true host update per FIA occurrence.

    This object owns outputs, workspaces, graph pool and synchronization events.
    The caller owns the input tensors and must keep their storage stable. Replay
    is asynchronous on ``stream``; ``completion`` includes the update-stream
    join. Input mutation must follow completion of the preceding replay.
    Homogeneous sequential occurrences share temporary workspace, matching the
    original backend's graph-local FIA capture-resource key.
    """

    def __init__(
        self,
        torch,
        layers,
        block_table,
        atten_mask,
        query_lengths,
        kv_lengths,
        *,
        pool=None,
        stream=None,
        update_stream=None,
    ):
        import torch_npu

        from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph

        if not layers or block_table.ndim != 2 or block_table.dtype != torch.int32:
            raise ValueError("nonempty attention layers and an int32 [requests,pages] table required")
        self.torch, self.torch_npu, self.context_source = torch, torch_npu, ContextSource
        self.layers, self.block_table, self.atten_mask = tuple(layers), block_table, atten_mask
        self.capacity = block_table.shape[0]
        self.maximum_kv_length = block_table.shape[1] * PAGE_SIZE
        self.query_lengths, self.kv_lengths = validate_lengths(
            query_lengths, kv_lengths, self.capacity, self.maximum_kv_length
        )
        self._validate_inputs()
        self.pool = torch.npu.graph_pool_handle() if pool is None else pool
        self.stream = torch.npu.Stream() if stream is None else stream
        self.update_stream = torch.npu.Stream() if update_stream is None else update_stream
        self.prepared, self.updated, self.completion = (torch.npu.Event() for _ in range(3))
        self.outputs, self.softmax_lse, self.workspaces, self.kwargs = [], [], [], []
        self.graph = UpdatableGraph()
        self._allocate_operations()
        self._capture()

    def _validate_inputs(self):
        torch = self.torch
        expected_query = (self.capacity * QUERY_LENGTH, QUERY_HEADS, HEAD_DIM)
        device = self.block_table.device
        if device.type != "npu" or self.capacity < 1 or self.maximum_kv_length < 1:
            raise ValueError("a positive fixed NPU request/page capacity is required")
        if self.atten_mask.device != device or self.atten_mask.dtype != torch.bool:
            raise ValueError("the causal mask must be a boolean tensor on the input device")
        key_shape = self.layers[0]["key"].shape
        for layer in self.layers:
            if tuple(layer["query"].shape) != expected_query or not layer["query"].is_contiguous():
                raise ValueError("each query must be contiguous [physical_requests*4,12,256]")
            for name in ("query", "key", "value"):
                if layer[name].dtype != torch.bfloat16 or layer[name].device != device:
                    raise ValueError("all Q/K/V inputs must be BF16 on the input device")
            if layer["key"].shape != layer["value"].shape:
                raise ValueError("key and value must have matching paged shapes")
            if layer["key"].shape != key_shape:
                raise ValueError("shared FIA scratch requires the same paged geometry for every occurrence")

    def _allocate_operations(self):
        torch, torch_npu = self.torch, self.torch_npu
        maximum = _operation_kwargs(
            self.layers[0], self.block_table, self.atten_mask, self.query_lengths, self.kv_lengths
        )
        maximum["actual_seq_lengths_kv"] = [self.maximum_kv_length] * self.capacity
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**maximum)
        self.workspaces.append(workspace)
        for layer in self.layers:
            kwargs = _operation_kwargs(layer, self.block_table, self.atten_mask, self.query_lengths, self.kv_lengths)
            output = torch.empty_like(layer["query"])
            lse = torch.empty(1, dtype=layer["query"].dtype, device=layer["query"].device)
            kwargs.update(workspace=workspace, out=[output, lse])
            self.outputs.append(output)
            self.softmax_lse.append(lse)
            self.kwargs.append(kwargs)

    def _capture(self):
        torch = self.torch
        operation = self.torch_npu.npu_fused_infer_attention_score.out
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            for _ in range(WARMUP_ITERATIONS):
                for kwargs in self.kwargs:
                    operation(**kwargs)
        self.stream.synchronize()
        with torch.npu.graph(self.graph, pool=self.pool, stream=self.stream):
            for index, kwargs in enumerate(self.kwargs):
                self.graph.register_task(operation, kwargs, _LengthProvider(index))
        self.stream.synchronize()
        if len(self.graph.tasks) != len(self.layers):
            raise RuntimeError("each captured FIA occurrence must own one real host-update task")

    @property
    def tasks(self):
        return self.graph.tasks

    def replay(self, query_lengths, kv_lengths, *, prepared=None):
        """Submit replay and all exact-length updates; retain the final join."""
        torch = self.torch
        query_lengths, kv_lengths = validate_lengths(query_lengths, kv_lengths, self.capacity, self.maximum_kv_length)
        context = {
            "query_lengths": query_lengths,
            "kv_lengths": kv_lengths,
            "block_table": self.block_table,
        }
        tasks = self.graph.resolve_tasks(self.context_source(context))
        if prepared is None:
            prepared = self.prepared
            prepared.record(torch.npu.current_stream())
        self.stream.wait_event(prepared)
        self.update_stream.wait_event(prepared)
        with torch.npu.stream(self.stream):
            self.graph.replay()
        self.graph.update(self.update_stream, tasks)
        self.updated.record(self.update_stream)
        self.stream.wait_event(self.updated)
        self.completion.record(self.stream)
        self.query_lengths, self.kv_lengths = query_lengths, kv_lengths
        return self.outputs

    def synchronize(self):
        self.stream.synchronize()

    def workspace_bytes(self):
        return [tensor.numel() * tensor.element_size() for tensor in self.workspaces]
