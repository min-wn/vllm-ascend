# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic NZ GEMM screening; never full-model speedup evidence."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def cases(catalog):
    result = [(304,), (152, 152), (128, 176), (64, 240), (400,), (200, 200)]
    if any(n not in catalog or n % 4 for pair in result for n in pair):
        raise ValueError("every row count must already belong to the default Q4 capture catalog")
    return result


def main():
    import torch
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("require a new output path")
    selection = json.loads(args.catalog_report.read_bytes())["selection"]
    shapes = cases(selection["capture_sizes"])
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20261020)
    # Qwen3.5-27B TP2: Down [5120,17408/2], QKVZ [(2*2048+2*6144)/2,5120].
    weights = [
        torch_npu.npu_format_cast(torch.randn((n, k), dtype=torch.bfloat16, device="npu") * 0.01, 29)
        for n, k in ((5120, 8704), (8192, 5120), (48, 5120))
    ]
    source = torch.randn((400, 8704), dtype=torch.bfloat16, device="npu")
    hidden = torch.randn((400, 5120), dtype=torch.bfloat16, device="npu")
    graphs, outputs = {}, {}
    stream = torch.npu.Stream()

    def body(pair, component):
        result, offset = [], 0
        if component in ("down", "dense_chain"):
            for count in pair:
                result.append(torch.nn.functional.linear(source[offset : offset + count], weights[0]))
                offset += count
        if component in ("input", "dense_chain"):
            offset = 0
            for count in pair:
                value = hidden[offset : offset + count]
                result.extend(torch.nn.functional.linear(value, w) for w in weights[1:])
                offset += count
        return result

    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for pair in shapes:
            for component in ("down", "input", "dense_chain"):
                key = (pair, component)
                for _ in range(3):
                    body(*key)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    # Amortize Python submission and events; these are warmed synthetic weights.
                    for _ in range(8):
                        result = body(*key)
                graphs[key], outputs[key] = graph, result
    torch.npu.synchronize()
    for result in outputs.values():
        if not all(torch.isfinite(value).all().item() for value in result):
            raise RuntimeError("nonfinite synthetic projection")
    samples = []
    rng = random.Random(20261020)
    for trial in range(7):
        order = list(graphs)
        rng.shuffle(order)
        for pair, component in order:
            graph = graphs[pair, component]
            with torch.npu.stream(stream):
                graph.replay()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record(stream)
                graph.replay()
                end.record(stream)
            torch.npu.synchronize()
            samples.append(dict(trial=trial, pair=pair, component=component, us=start.elapsed_time(end) * 1000 / 8))
    medians = [
        dict(
            pair=pair,
            component=component,
            us=statistics.median(s["us"] for s in samples if s["pair"] == pair and s["component"] == component),
        )
        for pair, component in graphs
    ]
    report = dict(
        scope="synthetic GEMM screening only; no communication, norm, joins, state, real weights or full-model timing",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        catalog_report_sha256=hashlib.sha256(args.catalog_report.read_bytes()).hexdigest(),
        selection=selection,
        weight_shapes=[list(w.shape) for w in weights],
        weight_formats=[torch_npu.get_npu_format(w) for w in weights],
        repeats_in_graph=8,
        samples=samples,
        medians=medians,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
