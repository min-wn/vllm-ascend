# SPDX-License-Identifier: Apache-2.0
"""Causal request-tile GDN solve test with complete accepted-state publication.

Keep original padding, same-consumer padding, mixed delayed consumption and
mixed tile-ready consumption. Include the existing serial/parallel half-bucket
controls. All variants perform the same public deferred accepted-state commit.
No layer projection, convolution, FIA or TP communication is claimed here.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q4_block_probe import allocate_cache_pages
from multi_graph_gdn_q4_log_probe import request_pipeline_operations
from multi_graph_gdn_q4_tile_transport_probe import metadata_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--deferred-extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.batch <= 128 or args.repeats < 3:
        parser.error("new output, B3..128 and at least three samples required")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    batch, device = args.batch, "npu:0"
    tile = load_extension("gdn_q4_tile_pipeline", args.extension)
    deferred = load_extension("gdn_q4_log_deferred", args.deferred_extension)
    producer = load_extension("gdn_q4_products", args.products_extension)
    initial = torch.randn(batch + 1, 24, 128, 128, device=device) * 0.01
    initial[0].zero_()
    pages, base = allocate_cache_pages(initial, 65536)
    logs = torch.zeros(batch + 1, 24, 1056, device=device)
    valid = torch.zeros(batch + 1, 24, 8, dtype=torch.int32, device=device)
    new_logs, new_valid = torch.zeros_like(logs), torch.zeros_like(valid)
    mutable = (pages, logs, valid, new_logs, new_valid)
    q, k = (torch.empty(batch * 4, 8, 128, dtype=torch.bfloat16, device=device) for _ in range(2))
    v = torch.empty(batch * 4, 24, 128, dtype=torch.bfloat16, device=device)
    beta = torch.empty(batch * 4, 24, dtype=torch.bfloat16, device=device)
    gate = torch.empty(batch * 4, 24, device=device)
    output = torch.zeros_like(v)
    products = torch.empty(batch, 24, 128, 8, device=device)
    sequence = torch.empty(batch + 1, dtype=torch.int32, device=device)
    ids = torch.empty(batch * 4, dtype=torch.int32, device=device)
    accepted = torch.ones(batch, dtype=torch.int32, device=device)
    exported = torch.empty(batch + 1, 4, 24, 128, 128, device=device)
    main_stream = torch.npu.current_stream()
    report = dict(
        status="request_tile_solve_in_progress",
        scope="closed compact recurrence including preparation and accepted commit; not model replay",
        batch=batch,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        checks=[],
        samples=[],
        sources={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("probe", Path(__file__)),
                ("tile", args.extension),
                ("deferred", args.deferred_extension),
                ("products", args.products_extension),
            )
        },
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def snapshot():
        return tuple(tensor.clone() for tensor in mutable)

    def restore(saved):
        for tensor, value in zip(mutable, saved, strict=True):
            tensor.copy_(value)
        torch.npu.synchronize()

    def metadata(step):
        values = metadata_values(batch, step)
        for tensor, key in ((sequence, "sequence"), (ids, "ids"), (accepted, "accepted")):
            tensor.copy_(torch.tensor(values[key], dtype=torch.int32))
        torch.manual_seed(20261020 + step)
        for tensor in (q, k):
            tensor.copy_(torch.nn.functional.normalize(torch.randn(tensor.shape, device=device), dim=-1).bfloat16())
        v.normal_(0, 0.1)
        beta.uniform_(0.1, 0.8)
        gate.uniform_(-0.2, -0.01)
        torch.npu.synchronize()
        return values

    def prepare():
        output.zero_()
        return q.float(), k.float()

    def product_range(qf, kf, start, stop):
        producer.products(base, qf, kf, sequence, ids, accepted, products, 20, start, stop)

    def solve_range(start, stop):
        deferred.solve(
            products,
            q,
            k,
            v,
            beta,
            gate,
            base,
            sequence,
            ids,
            accepted,
            output,
            logs,
            valid,
            new_logs,
            new_valid,
            40,
            start,
            stop,
        )

    def commit():
        deferred.commit(base, sequence, ids, accepted, logs, valid, new_logs, new_valid)

    def original_padding():
        qf, kf = prepare()
        product_range(qf, kf, 0, batch)
        solve_range(0, batch)
        commit()

    def same_consumer_padding():
        qf, kf = prepare()
        product_range(qf, kf, 0, batch)
        tile.consume(
            products, q, k, v, beta, gate, base, sequence, ids, accepted, output, logs, valid, new_logs, new_valid, 40
        )
        commit()

    def mixed(streaming):
        qf, kf = prepare()
        tile.solve(
            base,
            qf,
            kf,
            q,
            k,
            v,
            beta,
            gate,
            sequence,
            ids,
            accepted,
            products,
            output,
            logs,
            valid,
            new_logs,
            new_valid,
            20,
            streaming,
        )
        commit()

    halves = request_pipeline_operations(torch, batch, prepare, product_range, solve_range, output)

    def half(mode):
        halves[mode]()
        commit()

    operations = dict(
        original_padding=original_padding,
        same_consumer_padding=same_consumer_padding,
        mixed_delayed=lambda: mixed(False),
        mixed_tile_ready=lambda: mixed(True),
        split_serial=lambda: half("compact_serial"),
        split_parallel=lambda: half("compact_pipeline"),
    )

    def assert_state(expected, label):
        for index, (actual, wanted) in enumerate(zip(mutable, expected, strict=True)):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0, msg=f"{label}/state{index}")
        if not torch.isfinite(output).all().item() or any(not torch.isfinite(t).all().item() for t in mutable):
            raise AssertionError(f"{label}: nonfinite output or physical state")
        if not (pages[:, 24 * 128 * 128 :] == 7).all().item():
            raise AssertionError(f"{label}: page canary changed")

    try:
        metadata(0)
        initial_state = snapshot()
        graphs = {}
        for mode, operation in operations.items():
            restore(initial_state)
            for _ in range(2):
                operation()
            torch.npu.synchronize()
            restore(initial_state)
            graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
            stream.wait_stream(main_stream)
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                operation()
            torch.npu.synchronize()
            graphs[mode] = graph, stream
        restore(initial_state)
        for step in range(8):
            values = metadata(step)
            before = snapshot()
            original_padding()
            torch.npu.synchronize()
            expected, wanted_output = snapshot(), output.clone()
            deferred.export_candidates(base, logs, valid, exported)
            wanted_candidates = exported.clone()
            for mode, operation in operations.items():
                for execution in ("eager", "graph", "graph_repeat"):
                    restore(before)
                    if execution == "eager":
                        operation()
                    else:
                        graph, stream = graphs[mode]
                        stream.wait_stream(main_stream)
                        with torch.npu.stream(stream):
                            graph.replay()
                    torch.npu.synchronize()
                    assert_state(expected, f"{mode}/{execution}/step{step}")
                    torch.testing.assert_close(output, wanted_output, rtol=0, atol=0)
                    deferred.export_candidates(base, logs, valid, exported)
                    torch.testing.assert_close(exported, wanted_candidates, rtol=0, atol=0)
                    if not torch.isfinite(exported).all().item():
                        raise AssertionError("nonfinite logical candidate")
                    for index, (actual, old) in enumerate(zip(mutable, before, strict=True)):
                        torch.testing.assert_close(actual[0], old[0], rtol=0, atol=0)
                        for owner in set(range(batch)) - set(values["owners"]):
                            torch.testing.assert_close(
                                actual[owner + 1],
                                old[owner + 1],
                                rtol=0,
                                atol=0,
                                msg=f"paused owner {owner}, bank {index}",
                            )
                    report["checks"].append(
                        dict(
                            step=step,
                            mode=mode,
                            execution=execution,
                            exact_output_state_candidates=True,
                            paused_null_canaries=True,
                        )
                    )
            restore(expected)
            del before, expected, wanted_output, wanted_candidates
        metadata(0)
        timed_state = snapshot()
        for round_id in range(3):
            order = tuple(operations) if round_id % 2 == 0 else tuple(reversed(operations))
            for mode in order:
                graph, stream = graphs[mode]
                for _ in range(args.repeats):
                    restore(timed_state)
                    stream.wait_stream(main_stream)
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
        medians = {
            mode: statistics.median(sample["device_us"] for sample in report["samples"] if sample["mode"] == mode)
            for mode in operations
        }
        report.update(
            status="qualified_closed_recurrence_not_layer_or_model",
            median_device_us=medians,
            causal_overlap_saving_us=medians["mixed_delayed"] - medians["mixed_tile_ready"],
            saving_vs_best_padding_us=min(medians["original_padding"], medians["same_consumer_padding"])
            - medians["mixed_tile_ready"],
            gate="require approximately 60-100us additional closed-recurrence saving before layer integration",
        )
        save()
    except BaseException as error:
        report.update(status="rejected_request_tile_solve", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
