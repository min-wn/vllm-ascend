# SPDX-License-Identifier: Apache-2.0
"""Matched layer-4 recurrence diagnostic; precomputed consume is not deployable."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension

CORE_MODES = ("products_only", "consume_only", "separate", "delayed", "ready")
CLOSED_MODES = ("closed_delayed", "closed_ready")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(torch, tensor):
    cpu = tensor.detach().cpu().contiguous()
    if cpu.is_floating_point() and not torch.isfinite(cpu.float()).all().item():
        raise AssertionError("nonfinite diagnostic tensor")
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def measure_site(args, torch, tile, producer, deferred, path):
    host = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {name: tensor.to("npu:0") for name, tensor in host.items()}
    q, k, v, beta, gate, base, logs, valid, sequence, ids, accepted = (
        tensors[name]
        for name in ("query", "key", "value", "beta", "gate", "base", "logs", "valid", "sequence", "ids", "accepted")
    )
    batch = accepted.numel()
    products = torch.zeros(batch, 24, 128, 8, dtype=torch.float32, device="npu:0")
    output = torch.zeros_like(v)
    new_logs, new_valid = torch.zeros_like(logs), torch.zeros_like(valid)
    mutable = (base, logs, valid, new_logs, new_valid)
    initial = tuple(t.clone() for t in mutable)
    qf, kf = q.float(), k.float()
    main = torch.npu.current_stream()
    stream = torch.npu.Stream()
    stream.wait_stream(main)

    def restore():
        for target, saved in zip(mutable, initial, strict=True):
            target.copy_(saved)
        output.zero_()
        torch.npu.synchronize()

    def product(q_float=qf, k_float=kf):
        producer.products(base, q_float, k_float, sequence, ids, accepted, products, 20, 0, batch)

    def consume():
        tile.consume(
            products, q, k, v, beta, gate, base, sequence, ids, accepted, output, logs, valid, new_logs, new_valid, 40
        )

    def mixed(streamed, q_float=qf, k_float=kf):
        tile.solve(
            base,
            q_float,
            k_float,
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
            streamed,
        )

    def commit():
        deferred.commit(base, sequence, ids, accepted, logs, valid, new_logs, new_valid)

    def operation(mode):
        if mode.startswith("closed_"):
            output.zero_()
            mixed(mode == "closed_ready", q.float(), k.float())
            commit()
        elif mode == "products_only":
            product()
        elif mode == "consume_only":
            consume()
        elif mode == "separate":
            product()
            consume()
        else:
            mixed(mode == "ready")

    modes = CORE_MODES + CLOSED_MODES if args.phase == "diagnostic" else CLOSED_MODES
    reference = None
    checks = []
    # Check publication and accepted-base state, then close every resident owner.
    owners = base.shape[0] - 1
    terminal_seq = torch.tensor([0, *([4] * owners)], dtype=torch.int32, device="npu:0")
    terminal_ids = torch.arange(1, owners + 1, dtype=torch.int32, device="npu:0").repeat_interleave(4)
    terminal_accepted = torch.tensor([1 + owner % 4 for owner in range(owners)], dtype=torch.int32, device="npu:0")
    empty_logs, empty_valid = torch.zeros_like(logs), torch.zeros_like(valid)
    exported = torch.empty(owners + 1, 4, 24, 128, 128, dtype=torch.float32, device="npu:0")

    def audit(mode):
        restore()
        product()
        operation(mode)
        if not mode.startswith("closed_"):
            commit()
        torch.npu.synchronize()
        deferred.export_candidates(base, logs, valid, exported)
        torch.npu.synchronize()
        record = dict(
            output=tensor_sha(torch, output),
            staged_logs=tensor_sha(torch, new_logs),
            staged_valid=tensor_sha(torch, new_valid),
            accepted_state=[tensor_sha(torch, t) for t in (base, logs, valid)],
            candidates=tensor_sha(torch, exported),
        )
        active = set(ids.cpu().tolist()) - {0}
        for target, saved in zip(mutable, initial, strict=True):
            for owner in set(range(owners + 1)) - active:
                if not torch.equal(target[owner], saved[owner]):
                    raise AssertionError(f"inactive/null owner {owner} changed")
        deferred.commit(base, terminal_seq, terminal_ids, terminal_accepted, logs, valid, empty_logs, empty_valid)
        torch.npu.synchronize()
        if torch.count_nonzero(valid).item() or torch.count_nonzero(logs).item():
            raise AssertionError("terminal debt remains")
        record["terminal_state"] = tensor_sha(torch, base)
        return record

    for mode in modes:
        if mode == "products_only":
            continue
        record = audit(mode)
        if reference is None:
            reference = record
        elif record != reference:
            raise AssertionError(f"{path.name}/{mode}: output, publication, state or terminal mismatch")
        checks.append(dict(mode=mode, exact_publication_accepted_candidates_terminal=True, hashes=record))
    exported = None
    graphs = {}
    for mode in modes:
        restore()
        product()
        for _ in range(3):
            operation(mode)
        torch.npu.synchronize()
        restore()
        product()
        torch.npu.synchronize()
        stream.wait_stream(main)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            operation(mode)
        torch.npu.synchronize()
        graphs[mode] = graph
    samples = []
    for round_id in range(args.rounds):
        order = modes if round_id % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            for index in range(args.repeats):
                restore()
                # Identical untimed producer prelude for every operation. This
                # controls preparation, not the internal cache behavior of P/S.
                product()
                torch.npu.synchronize()
                stream.wait_stream(main)
                begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                wall = time.perf_counter_ns()
                with torch.npu.stream(stream):
                    begin.record(stream)
                    graphs[mode].replay()
                    end.record(stream)
                end.synchronize()
                samples.append(
                    dict(
                        round=round_id,
                        mode=mode,
                        sample=index,
                        device_us=begin.elapsed_time(end) * 1000,
                        wall_us=(time.perf_counter_ns() - wall) / 1000,
                    )
                )
    # Graph execution must preserve the same reference, not merely eager mode.
    for mode in CLOSED_MODES:
        restore()
        product()
        stream.wait_stream(main)
        with torch.npu.stream(stream):
            graphs[mode].replay()
        torch.npu.synchronize()
        actual = dict(
            output=tensor_sha(torch, output),
            staged_logs=tensor_sha(torch, new_logs),
            staged_valid=tensor_sha(torch, new_valid),
            accepted_state=[tensor_sha(torch, t) for t in (base, logs, valid)],
        )
        if any(actual[k] != reference[k] for k in actual):
            raise AssertionError(f"{mode}: graph state differs from eager")
    medians = {
        mode: [
            statistics.median(s["device_us"] for s in samples if s["mode"] == mode and s["round"] == r)
            for r in range(args.rounds)
        ]
        for mode in modes
    }
    live = int((sequence[1:] == 4).sum().item())
    return dict(
        site=str(path),
        site_sha256=sha(path),
        capacity_requests=batch,
        live_requests=live,
        aic=20,
        aiv=40,
        q4=4,
        precision="BF16 inputs/output; FP32 base/logs/products; HF32 off; highest matmul",
        checks=checks,
        samples=samples,
        round_medians_us=medians,
        median_device_us={mode: statistics.median(values) for mode, values in medians.items()},
        input_hashes={name: tensor_sha(torch, tensor) for name, tensor in host.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tile-extension", "products-extension", "deferred-extension"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sites", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("diagnostic", "candidate"), default="diagnostic")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists() or args.rounds < 3 or args.repeats < 8:
        parser.error("new output, >=3 rounds and >=8 samples required")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    tile = load_extension("gdn_q4_tile_pipeline", args.tile_extension)
    producer = load_extension("gdn_q4_products", args.products_extension)
    deferred = load_extension("gdn_q4_log_deferred", args.deferred_extension)
    report = dict(
        status="running",
        phase=args.phase,
        sites=[],
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        sources={
            str(p): sha(p)
            for p in (Path(__file__), args.tile_extension, args.products_extension, args.deferred_extension)
        },
        core_boundary=(
            "only named kernels, preallocated FP32 Q/K; output clear and identical producer prelude outside timer"
        ),
        closed_boundary=(
            "Q/K casts, output clear, mixed solve, accepted commit, graph completion; no layer dense/TP/head"
        ),
        consume_only="precomputed products diagnostic only; its difference from mixed is not pure synchronization cost",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        with torch.inference_mode():
            for path in args.sites:
                report["sites"].append(measure_site(args, torch, tile, producer, deferred, path))
                save()
                print(json.dumps(dict(site=str(path), medians=report["sites"][-1]["median_device_us"])), flush=True)
        report["status"] = "complete_same_algorithm_diagnostic_not_model_timing"
        save()
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
