import multiprocessing
from dataclasses import dataclass

import pytest
import torch
import torch_npu

from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph, register_task
from vllm_ascend.worker.multi_graph.kv_layout import contiguous_dense_kv_views


@dataclass(frozen=True)
class LengthProvider:
    def resolve(self, context):
        return {"actual_seq_lengths_kv": context["lengths"]}


@pytest.mark.parametrize("sizes", [(2, 2), (3, 2)])
@pytest.mark.parametrize("strided_query", [False, True])
def test_two_fia_graphs_independent_pools_and_parameter_updates(sizes, strided_query):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    process = multiprocessing.get_context("spawn").Process(target=exercise_fia_graphs, args=(sizes, strided_query))
    process.start()
    process.join(timeout=180)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("FIA graph replay timed out, possible external-event deadlock")
    assert process.exitcode == 0


def exercise_fia_graphs(sizes, strided_query=False):
    torch.manual_seed(0)
    device = "npu"
    block_size, head_size = 128, 64
    shape = (2, 8, block_size, 1, head_size)
    raw = torch.randn(shape, device=device, dtype=torch.float16)
    key, value = contiguous_dense_kv_views(raw, shape)
    key, value = key.view(8, block_size, head_size), value.view(8, block_size, head_size)
    entries = []
    for lane, size in enumerate(sizes):
        query = torch.randn((size, 1, head_size), device=device, dtype=torch.float16)
        if strided_query:
            query = torch.randn((size, 3, head_size), device=device, dtype=torch.float16)[:, 1:2, :]
            assert not query.is_contiguous()
        output = torch.empty_like(query)
        lse = torch.empty(1, device=device, dtype=torch.float16)
        table = torch.arange(lane * 4, lane * 4 + size, device=device, dtype=torch.int32).reshape(size, 1)
        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "block_table": table,
            "input_layout": "TND",
            "block_size": block_size,
            "actual_seq_lengths": list(range(1, size + 1)),
            "actual_seq_lengths_kv": [1] * size,
            "num_key_value_heads": 1,
            "num_heads": 1,
            "scale": head_size**-0.5,
            "sparse_mode": 0,
            "pre_tokens": block_size,
            "next_tokens": 0,
        }
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
        kwargs.update(workspace=workspace, out=[output, lse])
        compute, update = torch.npu.Stream(), torch.npu.Stream()
        compute.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(compute):
            for _ in range(2):
                torch_npu.npu_fused_infer_attention_score.out(**kwargs)
        compute.synchronize()
        graph = UpdatableGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(graph, pool=pool, stream=compute):
            kwargs["query"] = query.contiguous()
            register_task(torch_npu.npu_fused_infer_attention_score.out, kwargs, LengthProvider())
        entries.append((graph, pool, compute, update, kwargs, output, query))
    assert entries[0][0] is not entries[1][0]
    assert entries[0][1] != entries[1][1]
    assert entries[0][-2].data_ptr() != entries[1][-2].data_ptr()
    for step in range(1, 6):
        for graph, _, compute, update, kwargs, output, query in entries:
            query.add_(0.01)
            lengths = [step * 7 + index for index in range(output.shape[0])]
            prepared = torch.npu.Event()
            prepared.record(torch.npu.current_stream())
            tasks = graph.resolve_tasks(ContextSource({"lengths": lengths}))
            compute.wait_event(prepared)
            update.wait_event(prepared)
            with torch.npu.stream(compute):
                graph.replay()
            graph.update(update, tasks)
        torch.npu.synchronize()
        for _, _, _, _, kwargs, output, query in entries:
            lengths = [step * 7 + index for index in range(output.shape[0])]
            expected = torch.empty_like(output)
            eager_kwargs = {
                **kwargs,
                "query": query,
                "actual_seq_lengths_kv": lengths,
                "out": [expected, kwargs["out"][1]],
            }
            torch_npu.npu_fused_infer_attention_score.out(**eager_kwargs)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
