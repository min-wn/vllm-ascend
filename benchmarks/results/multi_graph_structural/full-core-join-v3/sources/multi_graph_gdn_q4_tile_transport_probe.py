# SPDX-License-Identifier: Apache-2.0
"""Qualify A2 request-tile handoff, without claiming recurrence/model speedup.

The Vector consumer copies Cube products to a second tensor. This deliberately
small consumer checks protocol, visibility, skipped tiles and repeated capture.
Actual solve work must pass a separate causal gate before real-layer tests.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path


def metadata_values(batch, step):
    if not 3 <= batch <= 128 or step < 0:
        raise ValueError("3..128 requests and a nonnegative step required")
    live = batch - step % 3
    owners = random.Random(20261020 + step).sample(range(batch), live)
    counts = [1 + (owner + step) % 4 for owner in owners]
    return dict(
        live=live,
        owners=owners,
        sequence=[0, *([4] * live), *([0] * (batch - live))],
        ids=[*(owner + 1 for owner in owners for _ in range(4)), *([0] * ((batch - live) * 4))],
        accepted=[*counts, *([1] * (batch - live))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--cores", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.batch <= 128 or not 1 <= args.cores <= 20 or args.repeats < 3:
        parser.error("new output, B3..128, 1..20 Cube cores and at least three repeats required")

    import torch
    import torch_npu
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_gdn_q4_block_probe import allocate_cache_pages

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    batch, device = args.batch, "npu:0"
    module = load_extension("gdn_q4_tile_pipeline", args.extension)
    products_module = load_extension("gdn_q4_products", args.products_extension)
    pages, base = allocate_cache_pages(torch.randn(batch + 1, 24, 128, 128, device=device) * 0.01, 65536)
    before = pages.cpu()
    q, k = (torch.empty(batch * 4, 8, 128, device=device) for _ in range(2))
    sequence = torch.empty(batch + 1, dtype=torch.int32, device=device)
    ids = torch.empty(batch * 4, dtype=torch.int32, device=device)
    accepted = torch.empty(batch, dtype=torch.int32, device=device)
    products, sink, reference = (torch.empty(batch, 24, 128, 8, device=device) for _ in range(3))
    root = torch.npu.current_stream()
    report = dict(
        status="transport_protocol_in_progress",
        scope="FP32 Cube product and Vector copy only; not actual GDN solve or model",
        shape=dict(batch=batch, cube_cores=args.cores),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        checks=[],
        samples=[],
        sources={
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mixed_extension": hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        },
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def prepare(step):
        values = metadata_values(batch, step)
        sequence.copy_(torch.tensor(values["sequence"], dtype=torch.int32))
        ids.copy_(torch.tensor(values["ids"], dtype=torch.int32))
        accepted.copy_(torch.tensor(values["accepted"], dtype=torch.int32))
        torch.manual_seed(20261020 + step)
        for tensor in (q, k):
            tensor.copy_(torch.nn.functional.normalize(torch.randn_like(tensor), dim=-1).bfloat16().float())
        products.fill_(7)
        sink.fill_(7)
        reference.fill_(7)
        products_module.products(base, q, k, sequence, ids, accepted, reference, args.cores)
        torch.npu.synchronize()
        return values

    def operation(streaming):
        module.transport(base, q, k, sequence, ids, accepted, products, sink, args.cores, streaming)

    try:
        prepare(0)
        graphs = {}
        for mode, streaming in (("delayed", False), ("tile_ready", True)):
            for _ in range(2):
                operation(streaming)
            torch.npu.synchronize()
            graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
            stream.wait_stream(root)
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                operation(streaming)
            torch.npu.synchronize()
            graphs[mode] = graph, stream
        for step in range(8):
            values = prepare(step)
            expected = reference.cpu()
            for mode, streaming in (("delayed", False), ("tile_ready", True)):
                for execution in ("eager", "graph", "graph_repeat"):
                    products.fill_(7)
                    sink.fill_(7)
                    if execution == "eager":
                        operation(streaming)
                    else:
                        graph, stream = graphs[mode]
                        stream.wait_stream(root)
                        with torch.npu.stream(stream):
                            graph.replay()
                    torch.npu.synchronize()
                    for field, actual in (("products", products), ("consumer", sink)):
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0, msg=f"{mode}/{execution}/step{step}/{field}"
                        )
                    torch.testing.assert_close(pages.cpu(), before, rtol=0, atol=0)
                    if not torch.isfinite(sink).all().item():
                        raise RuntimeError("nonfinite handoff products")
                    report["checks"].append(
                        dict(
                            step=step,
                            mode=mode,
                            execution=execution,
                            live=values["live"],
                            exact=True,
                            state_and_canaries_unchanged=True,
                        )
                    )
        prepare(0)
        for round_id in range(3):
            for mode in ("delayed", "tile_ready") if round_id % 2 == 0 else ("tile_ready", "delayed"):
                graph, stream = graphs[mode]
                for _ in range(3):
                    with torch.npu.stream(stream):
                        graph.replay()
                    torch.npu.synchronize()
                for _ in range(args.repeats):
                    start, stop = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    wall = time.perf_counter_ns()
                    with torch.npu.stream(stream):
                        start.record(stream)
                        graph.replay()
                        stop.record(stream)
                    torch.npu.synchronize()
                    report["samples"].append(
                        dict(
                            round=round_id,
                            mode=mode,
                            device_us=start.elapsed_time(stop) * 1000,
                            wall_us=(time.perf_counter_ns() - wall) / 1000,
                        )
                    )
        report["median_device_us"] = {
            mode: statistics.median(sample["device_us"] for sample in report["samples"] if sample["mode"] == mode)
            for mode in ("delayed", "tile_ready")
        }
        report["status"] = "qualified_transport_only_actual_solve_not_qualified"
        save()
    except BaseException as error:
        report.update(status="rejected_transport", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
