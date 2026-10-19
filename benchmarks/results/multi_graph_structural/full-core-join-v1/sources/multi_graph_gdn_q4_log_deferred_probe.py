# SPDX-License-Identifier: Apache-2.0
"""Causal deferred-commit / projection overlap, not a real layer or model result.

Compare identical FP32 compact recurrence and BF16 output projection/residual
work. The concurrent control joins both the state transaction and projection.
No convolution, gating preparation, gated Norm, FIA, TP or model is simulated.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q4_block_probe import allocate_cache_pages
from multi_graph_gdn_q4_log_probe import base_slot_ids


def deferred_operations(torch, prepare, fused, solve, commit, project):
    """Keep commit off the projection dependency, then close both endpoints."""
    commit_stream = torch.npu.Stream()
    solved, committed = (torch.npu.Event(enable_timing=False) for _ in range(2))

    def fused_projection():
        prepare()
        fused()
        return project()

    def serial():
        prepare()
        solve()
        commit()
        return project()

    def overlap():
        root = torch.npu.current_stream()
        prepare()
        solve()
        solved.record(root)
        with torch.npu.stream(commit_stream):
            commit_stream.wait_event(solved)
            commit()
            committed.record(commit_stream)
        result = project()
        root.wait_event(committed)
        return result

    return dict(fused_projection=fused_projection, deferred_serial=serial, deferred_overlap=overlap)


def request_record(batch, live, step):
    """Predeclared local causal trajectory; not the full-model workload."""
    if not 1 <= live <= batch <= 128 or step < 0:
        raise ValueError("valid TP2 request capacity, live count and nonnegative step required")
    rng = random.Random(20261019 + step)
    return rng.sample(range(batch), live), [rng.randint(1, 4) for _ in range(live)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--fused-extension", type=Path, required=True)
    parser.add_argument("--products-extension", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--live", type=int, default=98)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--state-padding", type=int, default=65536)
    parser.add_argument("--weight-format", choices=("ND", "NZ"), default="NZ")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.live <= args.batch <= 128 or args.repeats < 3 or args.state_padding < 0:
        parser.error("new output, 3 <= live <= batch <= 128, repeats >= 3 and nonnegative state padding required")
    import torch
    import torch.nn.functional as functional
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = args.weight_format == "NZ"
    torch.manual_seed(20261019)
    batch, heads, width, device = args.batch, 24, 128, "npu:0"
    deferred = load_extension("gdn_q4_log_deferred", args.extension)
    fused_module = load_extension("gdn_q4_log", args.fused_extension)
    products_module = load_extension("gdn_q4_products", args.products_extension)
    initial = torch.randn(batch + 1, heads, width, width, device=device) * 0.01
    pages, base = allocate_cache_pages(initial, args.state_padding)
    old_logs = torch.zeros(batch + 1, heads, 1056, device=device)
    old_valid = torch.zeros(batch + 1, heads, 8, device=device, dtype=torch.int32)
    new_logs = torch.zeros_like(old_logs)
    new_valid = torch.zeros_like(old_valid)
    mutable = (pages, old_logs, old_valid, new_logs, new_valid)
    q, k = (torch.empty(batch * 4, 8, width, device=device, dtype=torch.bfloat16) for _ in range(2))
    v = torch.empty(batch * 4, heads, width, device=device, dtype=torch.bfloat16)
    beta = torch.empty(batch * 4, heads, device=device, dtype=torch.bfloat16)
    gate = torch.empty(batch * 4, heads, device=device)
    sequence = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    ids = torch.zeros(batch * 4, device=device, dtype=torch.int32)
    accepted = torch.ones(batch, device=device, dtype=torch.int32)
    products = torch.empty(batch, heads, width, 8, device=device)
    output = torch.zeros_like(v)
    # Same local TP2 GDN output-projection dimensions as Qwen3.5-27B.
    # Synthetic weights/inputs, no checkpoint or TP collective is claimed.
    weight = torch.randn(5120, heads * width, dtype=torch.bfloat16, device=device) * 0.01
    if args.weight_format == "NZ":
        weight = torch_npu.npu_format_cast(weight, 29)
        if torch_npu.get_npu_format(weight) != 29:
            raise ValueError("NZ projection conversion did not take effect")
    residual = torch.randn(batch * 4, 5120, device=device, dtype=torch.bfloat16) * 0.1
    exports = [torch.empty(batch + 1, 4, heads, width, width, device=device) for _ in range(2)]
    checks, samples = [], []
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def snapshot():
        return tuple(tensor.clone() for tensor in mutable)

    def restore(saved):
        for tensor, old in zip(mutable, saved, strict=True):
            tensor.copy_(old)
        torch.npu.synchronize()

    def exact(actual, wanted, label):
        if not torch.equal(actual, wanted):
            difference = (actual.float() - wanted.float()).abs()
            raise AssertionError(f"{label}: nonexact, max_abs={difference.max().item()}")

    def exact_state(actual, wanted, label, count=5):
        for index in range(count):
            exact(actual[index], wanted[index], f"{label}/{index}")

    def export(module, destination):
        module.export_candidates(base, old_logs, old_valid, destination)
        return destination

    def finite():
        for tensor in (*mutable, output, *exports):
            if not torch.isfinite(tensor).all().item():
                raise AssertionError("nonfinite compact state, staging bank, output or exported candidate")
        if args.state_padding and not (pages[:, heads * width * width :] == 7).all().item():
            raise AssertionError("page padding overwritten")

    def metadata(live, step):
        requests, counts = request_record(batch, live, step)
        torch.manual_seed(20261019 + step)
        sequence.zero_()
        sequence[1 : live + 1].fill_(4)
        ids.zero_()
        ids[: live * 4].copy_(torch.tensor(base_slot_ids(requests), device=device, dtype=torch.int32))
        accepted.fill_(1)
        accepted[:live].copy_(torch.tensor(counts, device=device, dtype=torch.int32))
        for tensor in (q, k):
            tensor.copy_(functional.normalize(torch.randn(tensor.shape, device=device), dim=-1).bfloat16())
            tensor[live * 4 :].zero_()
        v.normal_(0, 0.1)
        v[live * 4 :].zero_()
        beta.uniform_(0.1, 0.8)
        gate.uniform_(-0.2, -0.01)
        residual.normal_(0, 0.1)
        torch.npu.synchronize()
        return requests, counts

    def prepare():
        output.zero_()
        products_module.products(base, q.float(), k.float(), sequence, ids, accepted, products)

    def fused():
        fused_module.step(products, q, k, v, beta, gate, base, sequence, ids, accepted, output, old_logs, old_valid)

    def solve():
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
            old_logs,
            old_valid,
            new_logs,
            new_valid,
        )

    def commit():
        deferred.commit(base, sequence, ids, accepted, old_logs, old_valid, new_logs, new_valid)

    def project():
        return functional.linear(output.reshape(batch * 4, heads * width), weight) + residual

    operations = deferred_operations(torch, prepare, fused, solve, commit, project)
    metadata(args.live, 0)
    graphs, graph_outputs = {}, {}
    saved = snapshot()
    for name, operation in operations.items():
        stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            graph_outputs[name] = operation()
        torch.npu.synchronize()
        restore(saved)
        graphs[name] = graph, stream

    for step, live in enumerate((args.live, args.live - 1, min(batch, args.live + 1), args.live) * 2):
        requests, counts = metadata(live, step + 1)
        prior_candidates = export(fused_module, exports[0]).clone()
        torch.npu.synchronize()
        before = snapshot()
        paused = torch.tensor([0] + [request + 1 for request in range(batch) if request not in requests], device=device)
        # A solve may publish staging logs and attention, but cannot commit
        # the old base or overwrite the stable old log bank before projection.
        prepare()
        solve()
        torch.npu.synchronize()
        solved_state = snapshot()
        exact_state(solved_state, before, f"solve_readonly_old/step{step}", count=3)
        for index in (3, 4):
            exact(mutable[index][paused], before[index][paused], f"solve_paused_staging/{index}/step{step}")
        restore(before)
        canonical_state = canonical_attention = canonical_projection = canonical_candidates = None
        for name, operation in operations.items():
            restore(before)
            eager_projection = operation().clone()
            torch.npu.synchronize()
            eager_attention, eager_state = output.clone(), snapshot()
            export(deferred if name.startswith("deferred") else fused_module, exports[1])
            eager_candidates = exports[1].clone()
            torch.npu.synchronize()
            restore(before)
            repeated = operation()
            torch.npu.synchronize()
            exact(repeated, eager_projection, f"{name}/repeat_projection/step{step}")
            exact(output, eager_attention, f"{name}/repeat_attention/step{step}")
            exact_state(snapshot(), eager_state, f"{name}/repeat_state/step{step}")
            restore(before)
            graph, stream = graphs[name]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.synchronize()
            exact(graph_outputs[name], eager_projection, f"{name}/replay_projection/step{step}")
            exact(output, eager_attention, f"{name}/replay_attention/step{step}")
            replay_state = snapshot()
            exact_state(replay_state, eager_state, f"{name}/replay_state/step{step}")
            export(deferred if name.startswith("deferred") else fused_module, exports[1])
            exact(exports[1], eager_candidates, f"{name}/replay_all_candidates/step{step}")
            for index in range(5):
                exact(mutable[index][paused], before[index][paused], f"{name}/null_paused/{index}/step{step}")
            slots = torch.tensor([request + 1 for request in requests], device=device)
            prefix = torch.tensor([count - 1 for count in counts], device=device)
            exact(base[slots], prior_candidates[slots, prefix], f"{name}/exact_accepted_commit/step{step}")
            if name == "fused_projection":
                canonical_state, canonical_attention = eager_state, eager_attention
                canonical_projection, canonical_candidates = eager_projection, eager_candidates
            else:
                exact_state(replay_state, canonical_state, f"{name}/fused_stable_state/step{step}", count=3)
                exact(output, canonical_attention, f"{name}/fused_attention/step{step}")
                exact(graph_outputs[name], canonical_projection, f"{name}/fused_projection/step{step}")
                exact(exports[1], canonical_candidates, f"{name}/fused_all_candidates/step{step}")
            if not torch.isfinite(graph_outputs[name]).all().item():
                raise AssertionError("nonfinite projection/residual result")
            finite()
            checks.append(
                dict(
                    mode=name,
                    step=step,
                    live=live,
                    repeat_eager_exact=True,
                    eager_replay_exact=True,
                    all_candidates_exact=True,
                    accepted_commit_exact=True,
                    null_paused_exact=True,
                    stable_state_matches_fused=True,
                    finite=True,
                    page_canaries=True,
                )
            )
        # Continue exactly one canonical history. The next step must consume
        # this generation, rather than replaying each operation's own leftovers.
        restore(canonical_state)
        checks.append(dict(stage="solve_readonly_old_and_paused_staging", step=step, passed=True))

    metadata(args.live, 1001)
    saved = snapshot()
    torch.npu.reset_peak_memory_stats()
    expected_final_state = None
    expected_final_projection = None
    for round_id in range(3):
        for name in tuple(operations) if round_id % 2 == 0 else tuple(reversed(operations)):
            restore(saved)
            graph, stream = graphs[name]
            stream.wait_stream(torch.npu.current_stream())
            for sample in range(-3, args.repeats):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                started = time.perf_counter_ns()
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
                            wall_us=(time.perf_counter_ns() - started) / 1000,
                        )
                    )
            export(deferred, exports[1])
            torch.npu.synchronize()
            finite()
            if expected_final_state is None:
                expected_final_state, expected_final_projection = snapshot(), graph_outputs[name].clone()
            else:
                exact_state(snapshot(), expected_final_state, f"timed_continuation/{name}/{round_id}", count=3)
                exact(graph_outputs[name], expected_final_projection, f"timed_projection/{name}/{round_id}")

    components = {}
    if args.profile_components:
        # Every isolated component is restored from the same prepared history.
        # These intervals are diagnostics and must not be added as a prediction.
        restore(saved)
        prepare()
        solve()
        torch.npu.synchronize()
        component_state = snapshot()
        for name, operation in (
            ("prepare", prepare),
            ("solve", solve),
            ("commit", commit),
            ("projection_residual", project),
        ):
            restore(component_state)
            stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                operation()
            torch.npu.synchronize()
            values = []
            for _ in range(args.repeats):
                restore(component_state)
                stream.wait_stream(torch.npu.current_stream())
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record(stream)
                    graph.replay()
                    end.record(stream)
                end.synchronize()
                values.append(begin.elapsed_time(end) * 1000)
            components[name] = dict(device_us=values, median_us=statistics.median(values))

    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("probe source changed during qualification")
    report = dict(
        status="qualified_deferred_commit_projection_causal_endpoint_not_model",
        scope="synthetic_single_device_TP2_local_shapes; not real GDN layer, full replay or TP2 communication",
        batch=batch,
        live=args.live,
        checks=checks,
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
        components=components,
        component_scope="isolated, state restored before each sample; intervals are not additive",
        projection=dict(
            weight_shape=[5120, 3072],
            input_shape=[batch * 4, 3072],
            dtype="bfloat16",
            weight_format=args.weight_format,
            synthetic_fixed_weights=True,
            residual_add_included=True,
        ),
        dependency=dict(
            fused="products -> fused solve+commit -> projection+residual",
            serial="products -> solve -> commit -> projection+residual",
            overlap="products -> solve -> (commit AIV || projection/residual) -> explicit final state join",
            stable_bank="old logs remain authoritative; active staged logs published by commit",
        ),
        includes=(
            "output clear, raw Q/K casts, base products, solve, accepted base commit, "
            "new log publication, projection, residual and both completion paths"
        ),
        excludes=(
            "input trajectory preparation, conv, gating prep, gated Norm, MLP, FIA host update, "
            "TP collective, model, scheduler"
        ),
        all_candidate_export="outside measured recurrence/projection; exact checks every continued step",
        lifecycle="sequential isolated request generations; no production stale/retry/fallback validation",
        source_sha256=source_sha256,
        libraries={
            name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for name, path in (
                ("deferred", args.extension),
                ("fused", args.fused_extension),
                ("products", args.products_extension),
            )
        },
        memory=dict(
            base_pages_bytes=pages.numel() * pages.element_size(),
            log_banks_bytes=sum(t.numel() * t.element_size() for t in (old_logs, old_valid, new_logs, new_valid)),
            projection_weight_bytes=weight.numel() * weight.element_size(),
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            scope="all graphs, diagnostic exports, snapshots and weights; not model incremental memory",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(report["round_medians_us"]), flush=True)


if __name__ == "__main__":
    main()
