"""Qualify a Q4 block recurrence with selected-state gather and eight-column BMM.

This is one device's recurrence endpoint, not TP2 model replay. The complete
candidate includes gather, grouped-head Q/K packing, FP32 products, the small
triangular solve, all four snapshots at the original cache precision and BF16
attention outputs. FP32 direct products include the required Q/K conversions.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path


def allocate_cache_pages(initial, state_padding):
    """Preserve state precision and expose page gaps as separately checked storage."""
    slots, heads, width_v, width_k = initial.shape
    logical_elements = heads * width_v * width_k
    pages = initial.new_full((slots, logical_elements + state_padding), 7)
    state = pages[:, :logical_elements].view(slots, heads, width_v, width_k)
    state.copy_(initial)
    return pages, state


def product_operands(state, query, key, ids, accepted, *, fold_gqa=False):
    """Gather accepted states and pack raw Q/K at the state operand precision."""
    import torch

    batch, _, key_heads, width = query.view(-1, 4, *query.shape[1:]).shape
    heads = state.shape[1]
    selected = ids.view(batch, 4).gather(1, (accepted - 1).long().view(batch, 1)).view(batch).long()
    matrix_heads = key_heads if fold_gqa else heads
    matrix_rows = width * (heads // key_heads) if fold_gqa else width
    base = state.index_select(0, selected).view(batch * matrix_heads, matrix_rows, width)
    # Raw query stays unscaled. The FP32 recurrence consumer applies scale.
    rhs = torch.cat((key.view(batch, 4, key_heads, width), query.view(batch, 4, key_heads, width)), dim=1)
    rhs = rhs.permute(0, 2, 3, 1)
    if not fold_gqa:
        rhs = rhs.repeat_interleave(heads // key_heads, dim=1)
    return base, rhs.to(state.dtype).contiguous().view(batch * matrix_heads, width, 8)


def direct_products(module, state, query, key, sequence, ids, accepted, target):
    """Conversion belongs to the captured/timed operation, never input setup."""
    module.products(state, query.to(state.dtype), key.to(state.dtype), sequence, ids, accepted, target)
    return target


def profile_components(torch, operations, restore_inputs, repeats):
    """Time isolated captured components; these costs must not be added together."""
    captured, samples = [], []
    for name, operation in operations:
        restore_inputs()
        stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            result = operation()
        torch.npu.synchronize()
        # Retain capture outputs, including both conversion destinations.
        captured.append((name, graph, stream, result))
    for round_id in range(3):
        for name, graph, stream, _ in captured if round_id % 2 == 0 else reversed(captured):
            for sample in range(-3, repeats):
                # Emit-only consumes a frozen product: restore its matching state
                # before every invocation, outside this diagnostic timing scope.
                restore_inputs()
                stream.wait_stream(torch.npu.current_stream())
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
                            component=name,
                            round=round_id,
                            sample=sample,
                            wall_us=(time.perf_counter() - started) * 1e6,
                            device_us=begin.elapsed_time(end) * 1000,
                        )
                    )
    restore_inputs()
    return dict(
        scope="isolated captured components; not additive, not a complete recurrence or model speedup",
        state_restoration="matching state restored before every capture/replay, excluded from component event timing",
        warmup_replays_per_round=3,
        rounds=3,
        samples=samples,
        summary=[
            dict(
                component=name,
                median_device_us=statistics.median(row["device_us"] for row in samples if row["component"] == name),
                median_wall_us=statistics.median(row["wall_us"] for row in samples if row["component"] == name),
                per_round_median_device_us=[
                    statistics.median(
                        row["device_us"] for row in samples if row["component"] == name and row["round"] == round_id
                    )
                    for round_id in range(3)
                ],
                samples=3 * repeats,
            )
            for name, _ in operations
        ],
    )


def main():
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument(
        "--products-extension", type=Path, help="direct selected-cache Cube products instead of gather/BMM"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--live", type=int, default=98)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--state-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--state-padding", type=int, default=0, help="state-dtype canary elements between logical slots"
    )
    parser.add_argument("--fold-gqa", action="store_true", help="stack the three value-head matrices sharing each Q/K")
    parser.add_argument("--recheck-eager", action="store_true", help="diagnose eager determinism before replay checks")
    parser.add_argument("--profile-components", action="store_true", help="isolated non-additive component timings")
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.live <= args.batch <= 128 or args.repeats < 3 or args.state_padding < 0:
        parser.error("new output path, 3 <= live <= batch <= 128 and repeats >= 3 required")
    if args.profile_components and not args.products_extension:
        parser.error("--profile-components requires --products-extension for the direct Cube diagnostic")
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.manual_seed(88107)
    device, batch, heads, key_heads, width = "npu:0", args.batch, 24, 8, 128
    state_dtype = getattr(torch, args.state_dtype)
    spec = importlib.util.spec_from_file_location("gdn_q4_block", args.extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    product_module = None
    if args.products_extension:
        product_spec = importlib.util.spec_from_file_location("gdn_q4_products", args.products_extension)
        product_module = importlib.util.module_from_spec(product_spec)
        product_spec.loader.exec_module(product_module)
    q, k = tuple(torch.empty(batch * 4, key_heads, width, device=device, dtype=torch.bfloat16) for _ in range(2))
    v = torch.empty(batch * 4, heads, width, device=device, dtype=torch.bfloat16)
    beta = torch.empty(batch * 4, heads, device=device, dtype=torch.bfloat16)
    gate = torch.empty(batch * 4, heads, device=device)
    initial = (torch.randn(batch + 1, heads, width, width, device=device) * 0.01).to(state_dtype)
    initial = torch.cat((initial[:1], initial[1:].repeat_interleave(4, dim=0)))
    cache_pages, caches = {}, {}
    for mode in ("official", "block"):
        cache_pages[mode], caches[mode] = allocate_cache_pages(initial, args.state_padding)
    output = torch.zeros_like(v)
    sequence = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    ids = torch.zeros(batch * 4, dtype=torch.int32, device=device)
    accepted = torch.ones(batch, dtype=torch.int32, device=device)
    product_storage = torch.zeros(batch, heads, width, 8, device=device)
    state_tensors = (*caches.values(), output)
    rng = random.Random(88107)
    checks, diagnostics = [], []

    def exact(actual, wanted, label):
        try:
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        except AssertionError as error:
            indices = (actual != wanted).nonzero()[:16].cpu().tolist()
            detail = [
                dict(index=index, actual=actual[tuple(index)].item(), expected=wanted[tuple(index)].item())
                for index in indices
            ]
            report = dict(
                status="failed_same_schedule",
                label=label,
                error=str(error),
                examples=detail,
                checks=checks,
                diagnostics=diagnostics,
                source_sha256=source_sha256,
                extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
            )
            if actual.ndim == 4 and actual.shape[-2:] == (128, 128):
                report["mismatch_by_value_row"] = (actual != wanted).sum(dim=(0, 1, 3)).cpu().tolist()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise

    def metadata(live, variant):
        requests = rng.sample(range(batch), live)
        sequence.copy_(torch.tensor([0] + [4] * live + [0] * (batch - live), dtype=torch.int32, device=device))
        selected = [1 + request * 4 + token for request in requests for token in range(4)]
        ids.copy_(torch.tensor(selected + [0] * ((batch - live) * 4), dtype=torch.int32, device=device))
        accepted.copy_(torch.tensor([1 + (position + variant) % 4 for position in range(batch)], device=device))
        for tensor in (q, k):
            tensor.copy_(functional.normalize(torch.randn_like(tensor, dtype=torch.float32), dim=-1))
        v.copy_(torch.randn_like(v, dtype=torch.float32) * 0.05)
        beta.copy_(torch.sigmoid(torch.randn_like(beta, dtype=torch.float32)))
        gate.copy_(-torch.rand_like(gate) * 0.2)
        torch.npu.synchronize()
        return [0] + [1 + request * 4 + token for request in set(range(batch)) - set(requests) for token in range(4)]

    def bmm_products(state, target):
        base, rhs = product_operands(state, q, k, ids, accepted, fold_gqa=args.fold_gqa)
        destination = target.view(base.shape[0], base.shape[1], 8)
        if state.dtype == torch.float32:
            torch.bmm(base, rhs, out=destination)
        else:
            torch.ops.aten.bmm.dtype_out(base, rhs, torch.float32, out=destination)
        return target

    def products(state):
        if product_module is not None:
            direct_products(product_module, state, q, k, sequence, ids, accepted, product_storage)
        else:
            bmm_products(state, product_storage)
        return product_storage

    def block_op():
        module.emit(products(caches["block"]), q, k, v, beta, gate, caches["block"], sequence, ids, accepted, output)
        return output

    def official_op():
        return recurrent_gated_delta_rule(
            q,
            k,
            v,
            caches["official"],
            g=gate,
            beta=beta,
            scale=width**-0.5,
            actual_seq_lengths=sequence,
            ssm_state_indices=ids,
            num_accepted_tokens=accepted,
        )

    def snapshot():
        return tuple(tensor.clone() for tensor in state_tensors)

    def restore(saved):
        for tensor, before in zip(state_tensors, saved, strict=True):
            tensor.copy_(before)
        torch.npu.synchronize()

    def finite(tensors):
        if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise AssertionError("nonfinite block recurrence result")
        if args.state_padding:
            for pages in cache_pages.values():
                if not (pages[:, heads * width * width :] == 7).all().item():
                    raise AssertionError("recurrence overwrote padding between state pages")

    def metrics(actual, expected, stage, step, live):
        finite((actual, expected))
        delta = actual.float() - expected.float()
        norm = (delta.square().sum() / expected.float().square().sum().clamp_min(1e-30)).sqrt()
        result = dict(stage=stage, step=step, max_abs=delta.abs().max().item(), normalized_rmse=norm.item())
        if stage == "attention_output":
            shaped_delta = delta.view(-1, 4, heads, width)
            shaped_expected = expected.float().view(-1, 4, heads, width)
            by_request = (
                shaped_delta.square().sum(dim=(1, 2, 3)) / shaped_expected.square().sum(dim=(1, 2, 3)).clamp_min(1e-30)
            ).sqrt()
            result["worst_request_normalized_rmse"] = by_request.max().item()
        else:
            # Stable request IDs are recovered from the active destination slots.
            # Prefix metrics below assess each logical candidate independently.
            for prefix in range(4):
                active_slots = ids.view(batch, 4)[:live, prefix].long()
                actual_prefix = caches["block"].index_select(0, active_slots).float()
                expected_prefix = caches["official"].index_select(0, active_slots).float()
                prefix_delta = actual_prefix - expected_prefix
                by_request = (
                    prefix_delta.square().sum(dim=(1, 2, 3))
                    / expected_prefix.square().sum(dim=(1, 2, 3)).clamp_min(1e-30)
                ).sqrt()
                result[f"prefix{prefix + 1}_worst_request_normalized_rmse"] = by_request.max().item()
        diagnostics.append(result)
        # Preregistered diagnostic gate, not a model-quality certification.
        if any(value > 0.005 for name, value in result.items() if "normalized_rmse" in name):
            raise AssertionError(f"block recurrence exceeds 0.5% normalized RMSE gate: {result}")

    graphs, graph_outputs = {}, {}
    operations = (("official", official_op), ("block", block_op))
    metadata(args.live, 0)
    if product_module is not None:
        cache_before = caches["block"].clone()
        expected = bmm_products(caches["block"], torch.empty_like(product_storage))
        observed = products(caches["block"])
        torch.npu.synchronize()
        torch.testing.assert_close(observed[: args.live], expected[: args.live], rtol=1e-4, atol=2e-6)
        exact(caches["block"], cache_before, "direct_products/cache_read_only")
        checks.append(dict(stage="direct_products", compared_to_registered_bmm_fp32=True))
    for mode, operation in operations:
        saved = snapshot()
        stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            graph_outputs[mode] = operation()
        torch.npu.synchronize()
        restore(saved)
        graphs[mode] = (graph, stream)
    for step, live in enumerate((args.live, args.live - 1, min(batch, args.live + 1), args.live) * 2):
        inactive = metadata(live, step)
        inactive = torch.tensor(inactive, dtype=torch.int64, device=device)
        for mode, operation in operations:
            saved = snapshot()
            expected_output = operation().clone()
            torch.npu.synchronize()
            expected = snapshot()
            expected_products = product_storage.clone() if mode == "block" and args.recheck_eager else None
            restore(saved)
            if args.recheck_eager:
                actual_output = operation()
                torch.npu.synchronize()
                exact(actual_output[: live * 4], expected_output[: live * 4], f"{mode}/step{step}/eager_repeat/output")
                if expected_products is not None:
                    exact(product_storage, expected_products, f"{mode}/step{step}/eager_repeat/products")
                for index, (actual, wanted) in enumerate(zip(state_tensors, expected, strict=True)):
                    exact(actual, wanted, f"{mode}/step{step}/eager_repeat/tensor{index}")
                restore(saved)
            graph, stream = graphs[mode]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.synchronize()
            for index, (actual, wanted) in enumerate(zip(state_tensors, expected, strict=True)):
                exact(actual, wanted, f"{mode}/step{step}/graph/tensor{index}")
            exact(graph_outputs[mode][: live * 4], expected_output[: live * 4], f"{mode}/step{step}/graph/output")
            prior = saved[0 if mode == "official" else 1]
            torch.testing.assert_close(
                caches[mode].index_select(0, inactive), prior.index_select(0, inactive), rtol=0, atol=0
            )
            finite((*state_tensors, graph_outputs[mode][: live * 4]))
            checks.append(dict(mode=mode, step=step, live=live, same_schedule_exact=True, null_inactive_exact=True))
        metrics(output[: live * 4], graph_outputs["official"][: live * 4], "attention_output", step, live)
        metrics(caches["block"], caches["official"], "all_four_states_and_inactive", step, live)

    metadata(args.live, 17)
    caches["block"].copy_(caches["official"])
    torch.npu.synchronize()
    saved = snapshot()
    samples = []
    torch.npu.reset_peak_memory_stats()
    for round_id in range(3):
        for mode in tuple(graphs) if round_id % 2 == 0 else tuple(reversed(graphs)):
            restore(saved)
            graph, stream = graphs[mode]
            stream.wait_stream(torch.npu.current_stream())
            for _ in range(3):
                with torch.npu.stream(stream):
                    graph.replay()
                torch.npu.synchronize()
            for sample in range(args.repeats):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                started = time.perf_counter()
                with torch.npu.stream(stream):
                    begin.record(stream)
                    graph.replay()
                    end.record(stream)
                end.synchronize()
                samples.append(
                    dict(
                        mode=mode,
                        round=round_id,
                        sample=sample,
                        wall_us=(time.perf_counter() - started) * 1e6,
                        device_us=begin.elapsed_time(end) * 1000,
                    )
                )
            finite((*state_tensors, graph_outputs[mode][: args.live * 4]))
    report = dict(
        status="qualified_block_recurrence_not_full_model",
        checks=checks,
        diagnostics=diagnostics,
        samples=samples,
        batch=batch,
        state_dtype=args.state_dtype,
        state_precision_preserved=True,
        state_strides=list(caches["block"].stride()),
        state_padding_canary_elements=args.state_padding,
        live=args.live,
        includes=(
            "direct selected-state Cube products (including Q/K casts for FP32 state), solve, four dense states"
            if product_module is not None
            else "selected-state gather, GQA Q/K packing, FP32 BMM, solve, four dense states"
        ),
        excludes="convolution, model layers, TP communication, FIA and serving input preparation",
        source_sha256=source_sha256,
        extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        products_extension_sha256=(
            hashlib.sha256(args.products_extension.read_bytes()).hexdigest() if args.products_extension else None
        ),
        fold_gqa=args.fold_gqa,
        numerical_gate="cross-algorithm <=0.5% NRMSE; same-schedule eager/replay bit exact; no model quality claim",
        memory=dict(
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),
            fixed_product_bytes=product_storage.numel() * product_storage.element_size(),
            gathered_base_bytes=(
                0 if product_module is not None else batch * heads * width * width * initial.element_size()
            ),
            packed_rhs_bytes=(
                0
                if product_module is not None
                else batch * (key_heads if args.fold_gqa else heads) * width * 8 * initial.element_size()
            ),
            direct_qk_cast_bytes=(
                (q.numel() + k.numel()) * initial.element_size()
                if product_module is not None and state_dtype == torch.float32
                else 0
            ),
            scope="process with both graphs and correctness snapshots; not full-model incremental peak",
        ),
    )
    if args.profile_components:
        restore(saved)
        q_preconverted, k_preconverted = q.to(state_dtype), k.to(state_dtype)
        frozen_products = products(caches["block"]).clone()
        torch.npu.synchronize()

        def pure_products():
            product_module.products(
                caches["block"], q_preconverted, k_preconverted, sequence, ids, accepted, product_storage
            )
            return product_storage

        def qk_cast_only():
            return q.to(state_dtype), k.to(state_dtype)

        def products_with_casts():
            return products(caches["block"])

        def emit_only():
            module.emit(frozen_products, q, k, v, beta, gate, caches["block"], sequence, ids, accepted, output)
            return output

        components = [("direct_products_preconverted_qk", pure_products)]
        if state_dtype != q.dtype:
            components.append(("qk_cast_only", qk_cast_only))
        components.extend((("direct_products_including_qk_casts", products_with_casts), ("emit_only", emit_only)))
        report["component_profile"] = profile_components(torch, components, lambda: restore(saved), args.repeats)
        report["component_profile"]["qk_cast_required"] = state_dtype != q.dtype
        report["component_profile"]["emit_input"] = "precomputed products from the restored matching initial state"
        report["component_profile"]["output_clear"] = "no separate clear in the qualified block endpoint"
        finite((*state_tensors, output[: args.live * 4]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("probe source changed during qualification; result is invalid")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], checks=len(checks)), indent=2))


if __name__ == "__main__":
    main()
