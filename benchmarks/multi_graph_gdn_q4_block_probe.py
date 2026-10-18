"""Qualify a Q4 block recurrence with selected-state gather and eight-column BMM.

This is one device's recurrence endpoint, not TP2 model replay. The complete
candidate includes gather, grouped-head Q/K packing, FP32 products, the small
triangular solve, all four BF16 snapshots and BF16 attention outputs.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path


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
    parser.add_argument("--state-padding", type=int, default=0, help="BF16 canary elements between logical state slots")
    parser.add_argument("--fold-gqa", action="store_true", help="stack the three value-head matrices sharing each Q/K")
    parser.add_argument("--recheck-eager", action="store_true", help="diagnose eager determinism before replay checks")
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.live <= args.batch <= 128 or args.repeats < 3 or args.state_padding < 0:
        parser.error("new output path, 3 <= live <= batch <= 128 and repeats >= 3 required")
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.manual_seed(88107)
    device, batch, heads, key_heads, width = "npu:0", args.batch, 24, 8, 128
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
    initial = (torch.randn(batch + 1, heads, width, width, device=device) * 0.01).bfloat16()
    initial = torch.cat((initial[:1], initial[1:].repeat_interleave(4, dim=0)))
    cache_pages = {
        mode: torch.full(
            (len(initial), heads * width * width + args.state_padding), 7, dtype=torch.bfloat16, device=device
        )
        for mode in ("official", "block")
    }
    caches = {
        mode: pages[:, : heads * width * width].view(len(initial), heads, width, width)
        for mode, pages in cache_pages.items()
    }
    for state in caches.values():
        state.copy_(initial)
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
        selected = ids.view(batch, 4).gather(1, (accepted - 1).long().view(batch, 1)).view(batch).long()
        matrix_heads = key_heads if args.fold_gqa else heads
        matrix_rows = width * (heads // key_heads) if args.fold_gqa else width
        base = state.index_select(0, selected).view(batch * matrix_heads, matrix_rows, width)
        # Raw BF16 query stays unscaled here. The FP32 consumer applies scale.
        rhs = torch.cat((k.view(batch, 4, key_heads, width), q.view(batch, 4, key_heads, width)), dim=1)
        rhs = rhs.permute(0, 2, 3, 1)
        if not args.fold_gqa:
            rhs = rhs.repeat_interleave(heads // key_heads, dim=1)
        rhs = rhs.contiguous()
        torch.ops.aten.bmm.dtype_out(
            base,
            rhs.view(batch * matrix_heads, width, 8),
            torch.float32,
            out=target.view(batch * matrix_heads, matrix_rows, 8),
        )
        return target

    def products(state):
        if product_module is not None:
            product_module.products(state, q, k, sequence, ids, accepted, product_storage)
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
        state_strides=list(caches["block"].stride()),
        state_padding_canary_elements=args.state_padding,
        live=args.live,
        includes=(
            "direct selected-state Cube products, solve, four dense states"
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
            gathered_base_bytes=0 if product_module is not None else batch * heads * width * width * 2,
            packed_rhs_bytes=(
                0 if product_module is not None else batch * (key_heads if args.fold_gqa else heads) * width * 8 * 2
            ),
            scope="process with both graphs and correctness snapshots; not full-model incremental peak",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("probe source changed during qualification; result is invalid")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], checks=len(checks)), indent=2))


if __name__ == "__main__":
    main()
