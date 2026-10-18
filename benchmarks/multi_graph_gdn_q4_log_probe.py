# SPDX-License-Identifier: Apache-2.0
"""Closed Q4 compact-log recurrence qualification, not full model replay.

The timed compact endpoint includes raw Q/K casts, base products, correction,
solve, previous accepted-base commit, new logs and full attention output.
All four logical candidates are exported independently for correctness. An
additional export-inclusive control quantifies a consumer needing dense state.
An optional fixed-half request pipeline separates partition costs from actual
product/state overlap while preserving full bucket capacity and global buffers.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from multi_graph_dynamic_trace import trace_digest
from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q4_block_probe import allocate_cache_pages


def candidate_slot_ids(requests):
    return [1 + request * 4 + token for request in requests for token in range(4)]


def base_slot_ids(requests):
    return [request + 1 for request in requests for _ in range(4)]


def request_pipeline_operations(torch, batch, prepare, product_range, step_range, output):
    """Reuse full buffers with one fixed request split and an explicit final join."""
    boundary = batch // 2
    side_stream = torch.npu.Stream()
    first_products_ready, second_products_ready, state_done = (torch.npu.Event(enable_timing=False) for _ in range(3))

    def serial():
        q_float, k_float = prepare()
        product_range(q_float, k_float, 0, boundary)
        step_range(0, boundary)
        product_range(q_float, k_float, boundary, batch)
        step_range(boundary, batch)
        return output

    def pipeline():
        # Capture runs on its own stream; never retain the setup/default stream.
        root_stream = torch.npu.current_stream()
        q_float, k_float = prepare()
        product_range(q_float, k_float, 0, boundary)
        first_products_ready.record(root_stream)
        with torch.npu.stream(side_stream):
            side_stream.wait_event(first_products_ready)
            step_range(0, boundary)
        product_range(q_float, k_float, boundary, batch)
        second_products_ready.record(root_stream)
        with torch.npu.stream(side_stream):
            side_stream.wait_event(second_products_ready)
            step_range(boundary, batch)
            state_done.record(side_stream)
        root_stream.wait_event(state_done)
        return output

    return dict(compact_serial=serial, compact_pipeline=pipeline)


def main():
    # Keep one setup/qualification/timing transaction so no failed check can
    # leave a success report or a timing-only result without its provenance.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--live", type=int, default=98)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--state-padding", type=int, default=65536)
    parser.add_argument("--request-pipeline", action="store_true", help="fixed-half serial and two-stream controls")
    parser.add_argument("--trace", type=Path, help="fixed request/acceptance manifest for continued correctness")
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.live <= args.batch <= 128 or args.repeats < 3 or args.state_padding < 0:
        parser.error("new output, 3 <= live <= batch <= 128, repeats >= 3 and nonnegative padding required")
    trace_document = json.loads(args.trace.read_text()) if args.trace else None
    if trace_document is not None:
        trace = trace_document["trace"]
        if trace_digest(trace) != trace_document["trace_sha256"] or trace["query"] != 4:
            parser.error("fixed trace digest or query contract mismatch")
        if len(trace["initial_prefix_by_request"]) != args.batch:
            parser.error("trace requires one stable owner per capacity request")
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(19518)
    batch, heads, width, device = args.batch, 24, 128, "npu:0"
    module = load_extension("gdn_q4_log", args.extension)
    products_module = load_extension("gdn_q4_products", args.products_extension)
    initial = torch.randn(batch + 1, heads, width, width, device=device) * 0.01
    dense_initial = torch.cat((initial[:1], initial[1:].repeat_interleave(4, dim=0)))
    official_pages, official_state = allocate_cache_pages(dense_initial, args.state_padding)
    compact_pages, compact_state = allocate_cache_pages(initial, args.state_padding)
    logs = torch.zeros(batch + 1, heads, 1056, device=device)
    valid = torch.zeros(batch + 1, heads, 8, device=device, dtype=torch.int32)
    q, k = (torch.zeros(batch * 4, 8, width, device=device, dtype=torch.bfloat16) for _ in range(2))
    v = torch.zeros(batch * 4, heads, width, device=device, dtype=torch.bfloat16)
    beta = torch.zeros(batch * 4, heads, device=device, dtype=torch.bfloat16)
    gate = torch.zeros(batch * 4, heads, device=device)
    sequence = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    accepted = torch.ones(batch, device=device, dtype=torch.int32)
    official_ids, compact_ids = (torch.zeros(batch * 4, device=device, dtype=torch.int32) for _ in range(2))
    products = torch.zeros(batch, heads, width, 8, device=device)
    output = torch.zeros_like(v)
    exported = torch.empty(batch + 1, 4, heads, width, width, device=device)
    mutable = (official_pages, compact_pages, logs, valid)
    checks, diagnostics, samples = [], [], []

    def snapshot():
        return tuple(tensor.clone() for tensor in mutable)

    def restore(saved):
        for tensor, old in zip(mutable, saved, strict=True):
            tensor.copy_(old)
        torch.npu.synchronize()

    def exact(actual, wanted, stage):
        if not torch.equal(actual, wanted):
            difference = (actual.float() - wanted.float()).abs()
            raise AssertionError(f"{stage}: nonexact repeat/commit, max={difference.max().item()}")

    def finite():
        for tensor in (*mutable, exported, output):
            if not torch.isfinite(tensor).all().item():
                raise AssertionError("nonfinite base/log/output or logical candidate")
        if args.state_padding:
            for pages in (official_pages, compact_pages):
                if not (pages[:, heads * width * width :] == 7).all().item():
                    raise AssertionError("page padding overwritten")

    def export():
        module.export_candidates(compact_state, logs, valid, exported)
        return exported

    def metadata(live, step, record=None):
        torch.manual_seed(19518 + step if record is None else record["input_seed"])
        rng = random.Random(9518 + step)
        requests = rng.sample(range(batch), live) if record is None else record["request_ids"]
        counts = [rng.randint(1, 4) for _ in range(live)] if record is None else record["accepted_from_previous"]
        sequence.zero_()
        sequence[1 : live + 1].fill_(4)
        accepted.fill_(1)
        accepted[:live].copy_(torch.tensor(counts, device=device, dtype=torch.int32))
        for target, values in ((official_ids, candidate_slot_ids(requests)), (compact_ids, base_slot_ids(requests))):
            target.zero_()
            target[: live * 4].copy_(torch.tensor(values, device=device, dtype=torch.int32))
        for tensor in (q, k):
            raw = torch.randn(tensor.shape, device=device)
            tensor.copy_(functional.normalize(raw, p=2, dim=-1).bfloat16())
            tensor[live * 4 :].zero_()
        v.normal_(0, 0.1)
        v[live * 4 :].zero_()
        beta.uniform_(0.1, 0.8)
        gate.uniform_(-0.2, -0.01)
        torch.npu.synchronize()
        return requests, counts

    def official():
        return recurrent_gated_delta_rule(
            q,
            k,
            v,
            official_state,
            g=gate,
            beta=beta,
            scale=width**-0.5,
            actual_seq_lengths=sequence,
            ssm_state_indices=official_ids,
            num_accepted_tokens=accepted,
        )

    def compact():
        output.zero_()
        products_module.products(compact_state, q.float(), k.float(), sequence, compact_ids, accepted, products)
        module.step(products, q, k, v, beta, gate, compact_state, sequence, compact_ids, accepted, output, logs, valid)
        return output

    def compact_with_export():
        compact()
        export()
        return output

    def prepare_request_ranges():
        output.zero_()
        return q.float(), k.float()

    def product_range(q_float, k_float, request_start, request_stop):
        products_module.products(
            compact_state, q_float, k_float, sequence, compact_ids, accepted, products, 20, request_start, request_stop
        )

    def step_range(request_start, request_stop):
        module.step(
            products,
            q,
            k,
            v,
            beta,
            gate,
            compact_state,
            sequence,
            compact_ids,
            accepted,
            output,
            logs,
            valid,
            40,
            request_start,
            request_stop,
        )

    def compare(actual, wanted, stage, step):
        delta = actual.float() - wanted.float()
        dimensions = tuple(range(1, actual.ndim))
        by_request = (delta.square().sum(dimensions) / wanted.float().square().sum(dimensions).clamp_min(1e-30)).sqrt()
        result = dict(
            stage=stage, step=step, max_abs=delta.abs().max().item(), worst_request_nrmse=by_request.max().item()
        )
        if not torch.isfinite(by_request).all().item() or result["worst_request_nrmse"] > 0.005:
            raise AssertionError(f"0.5% cross-algorithm screening gate failed: {result}")
        diagnostics.append(result)

    metadata(args.live, 0)
    graphs, graph_outputs = {}, {}
    operations = dict(official=official, compact=compact, compact_export_all=compact_with_export)
    partition_operations = (
        request_pipeline_operations(torch, batch, prepare_request_ranges, product_range, step_range, output)
        if args.request_pipeline
        else {}
    )
    operations.update(partition_operations)
    saved = snapshot()
    for name, operation in operations.items():
        stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            graph_outputs[name] = operation()
        torch.npu.synchronize()
        restore(saved)
        graphs[name] = (graph, stream)

    records = (
        trace_document["trace"]["steps"]
        if trace_document is not None
        else [dict(batch=live) for live in (args.live, args.live - 1, min(batch, args.live + 1), args.live) * 2]
    )
    for step, record in enumerate(records):
        live = record["batch"]
        requests, counts = metadata(live, step + 1, record if trace_document is not None else None)
        prior_candidates = export().clone()
        torch.npu.synchronize()
        paused = torch.tensor([0] + [request + 1 for request in range(batch) if request not in requests], device=device)
        official_paused = torch.tensor(
            [0] + candidate_slot_ids([request for request in range(batch) if request not in requests]), device=device
        )
        before = snapshot()
        compact_before, canonical_compact_state, canonical_compact_output = None, None, None
        for name in ("official", "compact", *partition_operations):
            if name == "compact":
                compact_before = snapshot()
            elif name in partition_operations:
                restore(compact_before)
            mode_before = snapshot()
            wanted_output = operations[name]().clone()
            torch.npu.synchronize()
            wanted = snapshot()
            restore(mode_before)
            repeated = operations[name]().clone()
            torch.npu.synchronize()
            output_limit = live * 4 if name == "official" else batch * 4
            exact(repeated[:output_limit], wanted_output[:output_limit], f"{name}/eager_output/step{step}")
            for index, (actual, expected) in enumerate(zip(mutable, wanted, strict=True)):
                exact(actual, expected, f"{name}/eager_state{index}/step{step}")
            restore(mode_before)
            graph, stream = graphs[name]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.synchronize()
            exact(graph_outputs[name][:output_limit], wanted_output[:output_limit], f"{name}/replay_output/step{step}")
            for index, (actual, expected) in enumerate(zip(mutable, wanted, strict=True)):
                exact(actual, expected, f"{name}/replay_state{index}/step{step}")
            if name == "compact":
                canonical_compact_state, canonical_compact_output = snapshot(), output.clone()
            elif name in partition_operations:
                exact(output, canonical_compact_output, f"{name}/full_compact_output/step{step}")
                for index, (actual, expected) in enumerate(zip(mutable, canonical_compact_state, strict=True)):
                    exact(actual, expected, f"{name}/full_compact_state{index}/step{step}")
            if name == "official":
                exact(
                    official_state[official_paused],
                    before[0][official_paused, : heads * width * width].view(-1, heads, width, width),
                    "official/null_paused",
                )
            else:
                exact(
                    compact_state[paused],
                    before[1][paused, : heads * width * width].view(-1, heads, width, width),
                    "compact/null_paused",
                )
                exact(logs[paused], before[2][paused], "logs/null_paused")
                exact(valid[paused], before[3][paused], "valid/null_paused")
            checks.append(
                dict(
                    mode=name,
                    step=step,
                    live=live,
                    repeat_eager_exact=True,
                    same_schedule_graph_exact=True,
                    null_paused_exact=True,
                    full_compact_bit_exact=name in partition_operations,
                )
            )
        if partition_operations:
            restore(canonical_compact_state)
            output.copy_(canonical_compact_output)
        export()
        torch.npu.synchronize()
        request_slots = torch.tensor([request + 1 for request in requests], device=device)
        accepted_indices = torch.tensor([count - 1 for count in counts], device=device)
        exact(
            compact_state[request_slots],
            prior_candidates[request_slots, accepted_indices],
            f"accepted_commit/step{step}",
        )
        compare(
            output[: live * 4].view(live, 4, heads, width),
            graph_outputs["official"][: live * 4].view(live, 4, heads, width),
            "attention",
            step,
        )
        for prefix in range(4):
            dense_slots = torch.tensor([1 + request * 4 + prefix for request in requests], device=device)
            compare(exported[request_slots, prefix], official_state[dense_slots], f"candidate{prefix + 1}", step)
        finite()
        checks.append(
            dict(stage="all_candidates_and_exact_accepted_commit", step=step, finite=True, page_canaries=True)
        )

    saved = snapshot()
    expected_output = compact_with_export().clone()
    torch.npu.synchronize()
    expected_state, expected_export = snapshot(), exported.clone()
    restore(saved)
    graph, stream = graphs["compact_export_all"]
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph.replay()
    torch.npu.synchronize()
    exact(output, expected_output, "export_inclusive/replay_output")
    exact(exported, expected_export, "export_inclusive/all_candidates")
    for actual, wanted in zip(mutable, expected_state, strict=True):
        exact(actual, wanted, "export_inclusive/replay_state")
    restore(saved)
    checks.append(dict(stage="export_inclusive_graph", output_state_candidates_exact=True))

    metadata(args.live, 101)
    saved = snapshot()
    torch.npu.reset_peak_memory_stats()
    for round_id in range(3):
        for name in tuple(operations) if round_id % 2 == 0 else tuple(reversed(operations)):
            restore(saved)
            graph, stream = graphs[name]
            stream.wait_stream(torch.npu.current_stream())
            for sample in range(-3, args.repeats):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                started = time.perf_counter()
                with torch.npu.stream(stream):
                    begin.record(stream)
                    graph.replay()
                    end.record(stream)
                end.synchronize()
                if sample >= 0:
                    samples.append(
                        dict(
                            mode=name,
                            round=round_id,
                            sample=sample,
                            device_us=begin.elapsed_time(end) * 1000,
                            wall_us=(time.perf_counter() - started) * 1e6,
                        )
                    )
            export()
            torch.npu.synchronize()
            finite()
    report = dict(
        status=(
            "qualified_q4_log_request_pipeline_not_full_model"
            if args.request_pipeline
            else "qualified_q4_log_recurrence_not_model_or_split"
        ),
        batch=batch,
        trace_sha256=trace_document["trace_sha256"] if trace_document is not None else None,
        trace_scope="request identities/acceptance/input seeds only; no full-model sequence-length execution",
        live=args.live,
        checks=checks,
        diagnostics=diagnostics,
        samples=samples,
        round_medians_us={
            name: [
                statistics.median(
                    row["device_us"] for row in samples if row["mode"] == name and row["round"] == round_id
                )
                for round_id in range(3)
            ]
            for name in operations
        },
        includes=(
            "casts, base products, correction, solve, accepted-base commit, new logs, output and all device completion"
        ),
        dense_export=(
            "all four logical candidates exported for correctness; compact_export_all measures an all-dense consumer"
        ),
        excludes="convolution, real layer/model, FIA host updates, TP communication, trace preparation and dispatch",
        lifecycle="sequential request generations only; production stale/retry/reuse validation not integrated",
        request_pipeline=dict(
            enabled=args.request_pipeline,
            boundary=batch // 2 if args.request_pipeline else None,
            capacity="unchanged full bucket and global input/products/state/log buffers",
            serial="cast full Q/K and clear output once; P_A, S_A, P_B, S_B",
            concurrent="P_A -> (S_A on side stream || P_B on capture stream) -> S_B -> capture-stream join",
            timing="complete range operations, casts, clear, event waits and final join; no full-model claim",
        ),
        state_strides=list(compact_state.stride()),
        memory=dict(
            compact_base_pages_bytes=compact_pages.numel() * compact_pages.element_size(),
            log_bytes=logs.numel() * logs.element_size(),
            valid_bytes=valid.numel() * valid.element_size(),
            official_pages_bytes=official_pages.numel() * official_pages.element_size(),
            export_scratch_bytes=exported.numel() * exported.element_size(),
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),
            scope="both controls, all graphs, exports and correctness snapshots; not incremental model peak",
        ),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        libraries={
            name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for name, path in (("log", args.extension), ("products", args.products_extension))
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(dict(status=report["status"], round_medians_us=report["round_medians_us"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
