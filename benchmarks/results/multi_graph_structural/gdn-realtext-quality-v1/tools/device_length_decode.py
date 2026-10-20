"""Qualify the experimental device-length decode kernel on an isolated A2 NPU."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path


def qualify(splits=1):
    import torch

    from vllm_ascend.ops.triton.device_length_decode import device_length_decode

    torch.manual_seed(42)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for dimension in (64, 128):
            query_cpu = torch.randn((2, 4, dimension), dtype=dtype)
            key_cpu = torch.randn((8, 128, 2, dimension), dtype=dtype)
            value_cpu = torch.randn_like(key_cpu)
            query, key, value = (tensor.npu() for tensor in (query_cpu, key_cpu, value_cpu))
            lengths = torch.tensor([256, 256], dtype=torch.int32, device="npu")
            table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device="npu")
            output = torch.empty_like(query)
            workspace = torch.empty((2, 4, splits, dimension + 2), dtype=torch.float32, device="npu")
            for _ in range(3):
                device_length_decode(query, key, value, lengths, table, output, splits=splits, workspace=workspace)
            torch.npu.synchronize()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                device_length_decode(query, key, value, lengths, table, output, splits=splits, workspace=workspace)
            stream.synchronize()
            for actual, pages in (
                ([256, 256], [[0, 1, 2, 3], [4, 5, 6, 7]]),
                ([8, 13], [[0, -1, -1, -1], [4, -1, -1, -1]]),
                ([129, 511], [[4, 5, -1, -1], [0, 1, 2, 3]]),
                ([0, 0], [[-1] * 4, [-1] * 4]),
                ([512, 17], [[3, 2, 1, 0], [7, -1, -1, -1]]),
            ):
                lengths.copy_(torch.tensor(actual, dtype=torch.int32))
                table.copy_(torch.tensor(pages, dtype=torch.int32))
                output.fill_(float("nan"))
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                observed = output.cpu().float()
                reference = torch.zeros_like(query_cpu, dtype=torch.float32)
                for request, length in enumerate(actual):
                    if not length:
                        continue
                    for head in range(4):
                        keys = torch.cat([key_cpu[page, :, head // 2] for page in pages[request] if page >= 0])
                        values = torch.cat([value_cpu[page, :, head // 2] for page in pages[request] if page >= 0])
                        scores = keys[:length].float() @ query_cpu[request, head].float() / dimension**0.5
                        reference[request, head] = scores.softmax(0) @ values[:length].float()
                error = (observed - reference).abs().max().item()
                passed = bool(torch.allclose(observed, reference, rtol=0.02, atol=0.01))
                rows.append(
                    dict(
                        dtype=str(dtype),
                        dimension=dimension,
                        splits=splits,
                        lengths=actual,
                        max_abs_error=error,
                        passed=passed,
                    )
                )
                if not passed:
                    raise AssertionError(rows[-1])
    return rows


def benchmark(batch, length, repeats):
    import torch
    import torch_npu

    from vllm_ascend.ops.triton.device_length_decode import device_length_decode

    heads, kv_heads, dimension, block_size = 14, 2, 64, 128
    width = (length + block_size - 1) // block_size
    query = torch.randn((batch, heads, dimension), dtype=torch.float16, device="npu")
    key = torch.randn((batch * width, block_size, kv_heads, dimension), dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="npu")
    table = torch.arange(batch * width, dtype=torch.int32, device="npu").view(batch, width)
    output = torch.empty_like(query)
    split_output = torch.empty_like(query)
    split_workspace = torch.empty((batch, heads, 4, dimension + 2), dtype=torch.float32, device="npu")
    fia_output = torch.empty_like(query)
    kwargs = dict(
        query=query,
        key=key.view(batch * width, block_size, -1),
        value=value.view(batch * width, block_size, -1),
        block_table=table,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=list(range(1, batch + 1)),
        actual_seq_lengths_kv=[length] * batch,
        num_key_value_heads=kv_heads,
        num_heads=heads,
        scale=dimension**-0.5,
        sparse_mode=0,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    kwargs.update(workspace=workspace, out=[fia_output, torch.empty(1, dtype=query.dtype, device="npu")])

    def prototype():
        device_length_decode(query, key, value, lengths, table, output)

    def fia():
        torch_npu.npu_fused_infer_attention_score.out(**kwargs)

    def split_prototype():
        device_length_decode(query, key, value, lengths, table, split_output, splits=4, workspace=split_workspace)

    for _ in range(3):
        prototype()
        split_prototype()
        fia()
    torch.npu.synchronize()
    if not torch.allclose(output, fia_output, rtol=0.02, atol=0.01):
        raise AssertionError("prototype and FIA outputs differ")
    if not torch.allclose(split_output, fia_output, rtol=0.02, atol=0.01):
        raise AssertionError("split prototype and FIA outputs differ")

    def capture(operation, updatable=False):
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent() if updatable else None
        handle = None
        with torch.npu.graph(graph, stream=stream):
            if event is not None:
                event.wait(stream)
                event.reset(stream)
                torch.npu.graph_task_group_begin(stream)
            operation()
            if event is not None:
                handle = torch.npu.graph_task_group_end(stream)
        stream.synchronize()
        return graph, event, handle

    prototype_graph, _, _ = capture(prototype)
    split_graph, _, _ = capture(split_prototype)
    fixed_graph, _, _ = capture(fia)
    update_graph, event, handle = capture(fia, True)
    update_stream = torch.npu.Stream()

    def replay_update():
        update_graph.replay()
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            fia()
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

    def measure(operation):
        for _ in range(5):
            operation()
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        for _ in range(repeats):
            operation()
        torch.npu.synchronize()
        return (time.perf_counter_ns() - start) / repeats / 1000

    operations = {
        "prototype_replay_us": prototype_graph.replay,
        "split4_prototype_replay_us": split_graph.replay,
        "fia_fixed_length_replay_us": fixed_graph.replay,
        "fia_replay_and_host_update_us": replay_update,
    }
    samples = {name: [] for name in operations}
    for iteration in range(5):
        order = list(operations) if iteration % 2 == 0 else list(reversed(operations))
        for name in order:
            samples[name].append(measure(operations[name]))
    return dict(
        batch=batch,
        kv_length=length,
        repeats=repeats,
        rounds=5,
        samples=samples,
        **{name: statistics.median(values) for name, values in samples.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    result = dict(
        device=torch.npu.get_device_name(),
        versions={name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "triton-ascend")},
        correctness=qualify() + qualify(4),
        measurements=[
            benchmark(batch, length, args.repeats) for batch, length in ((7, 64), (7, 512), (32, 512), (7, 2048))
        ],
        limitations="Single attention microbenchmark; excludes model, scheduler and dynamic metadata preparation.",
        provenance={
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kernel_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "vllm_ascend/ops/triton/device_length_decode.py").read_bytes()
            ).hexdigest(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
