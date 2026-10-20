# SPDX-License-Identifier: Apache-2.0
"""Qualify Q4 GDN initial-state products, including runtime gather and Q/K pack.

This is a causal product probe, not a recurrence or model benchmark. Both
BF16 inputs and the FP32 product output are mandatory for the mixed-precision
candidate. Query columns contain raw BF16 values; downstream FP32 consumers
must apply the query scale. No scaled query is rounded back to BF16 here.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def pack_rhs(query, key, value_heads, columns):
    """Return contiguous [B,Hv,K,8/16], with four K then four raw Q columns."""
    import torch

    if (
        columns not in (8, 16)
        or query.shape != key.shape
        or query.ndim != 4
        or query.shape[1] != 4
        or min(query.shape) < 1
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or query.device != key.device
    ):
        raise ValueError("matching [B,4,Hk,K] query/key and 8/16 product columns required")
    key_heads = query.shape[2]
    if value_heads < 1 or value_heads % key_heads:
        raise ValueError("value heads must be a positive multiple of key heads")
    packed = torch.cat((key, query), dim=1).permute(0, 2, 3, 1)
    packed = packed.repeat_interleave(value_heads // key_heads, dim=1).contiguous()
    if columns == 16:
        packed = torch.nn.functional.pad(packed, (0, 8))
    return packed


def selected_indices(request_ids, accepted):
    """Map stable request IDs to their BF16 accepted candidates; -1 is null."""
    import torch

    return torch.where(request_ids >= 0, 1 + 4 * request_ids + accepted - 1, 0)


def discrepancy(actual, expected):
    """Finite status and errors are separate from exact eager/replay checks."""
    import torch

    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    delta = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
        "exact": bool(torch.equal(actual, expected)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", nargs="+", choices=("dtype_out", "dtype", "fp32"), default=["dtype_out", "dtype"])
    parser.add_argument("--columns", nargs="+", type=int, choices=(8, 16), default=[8, 16])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3 or args.rounds < 2:
        parser.error("new output path, at least three samples and two alternating rounds required")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(1)
    device = torch.device("npu:0")
    batch, value_heads, key_heads, width = 98, 24, 8, 128
    groups = batch * value_heads
    generator = torch.Generator().manual_seed(args.seed)
    cache_cpu = (torch.randn(1 + 4 * batch, value_heads, width, width, generator=generator) * 0.02).bfloat16()
    cache = cache_cpu.to(device)
    cache_version = cache._version
    query = torch.empty(batch, 4, key_heads, width, dtype=torch.bfloat16, device=device)
    key = torch.empty_like(query)
    request_ids = torch.empty(batch, dtype=torch.int64, device=device)
    accepted = torch.empty_like(request_ids)
    selected = torch.empty(batch, value_heads, width, width, dtype=torch.bfloat16, device=device)
    report = {
        "kind": "gdn_q4_initial_state_products_only",
        "shape": {"batch": batch, "value_heads": value_heads, "key_heads": key_heads, "width": width, "groups": groups},
        "query_scale_applied": False,
        "query_scale_required_by_consumer": width**-0.5,
        "seed": args.seed,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": getattr(torch_npu.version, "git_version", None),
        "precision_settings": {
            "allow_hf32_matmul": torch.npu.matmul.allow_hf32,
            "allow_internal_format_option": torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT").decode(),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schemas": {name: str(getattr(torch.ops.aten.bmm, name)._schema) for name in ("dtype", "dtype_out", "out")},
        "includes_endpoint": [
            "accepted-candidate index arithmetic",
            "BF16 cache gather",
            "GQA key/query pack",
            "FP32 product",
        ],
        "excludes": [
            "host input copies",
            "GDN solve",
            "four candidate writes",
            "attention",
            "TP communication",
            "complete model",
        ],
        "qualification": [],
        "checks": [],
        "samples": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def update_inputs(step):
        q_cpu, k_cpu = tuple(
            torch.nn.functional.normalize(
                torch.randn(batch, 4, key_heads, width, generator=generator), dim=-1
            ).bfloat16()
            for _ in range(2)
        )
        ids_cpu = torch.randperm(batch, generator=generator)
        # Changing membership/order and mixed accepted counts reach stable graph pointers.
        live = (batch, batch - 2, batch - 1, batch)[step % 4]
        ids_cpu[live:] = -1
        accepted_cpu = (torch.arange(batch) + step) % 4 + 1
        query.copy_(q_cpu)
        key.copy_(k_cpu)
        request_ids.copy_(ids_cpu)
        accepted.copy_(accepted_cpu)
        torch.npu.synchronize()
        expected_state = cache_cpu.index_select(0, selected_indices(ids_cpu, accepted_cpu))
        return q_cpu, k_cpu, expected_state, live

    cpu_inputs = update_inputs(0)
    candidates = []
    for columns in args.columns:
        rhs_shape = (groups, width, columns)
        lower_rhs = pack_rhs(query, key, value_heads, columns).reshape(rhs_shape)
        lower_base = cache.index_select(0, selected_indices(request_ids, accepted)).reshape(groups, width, width)
        for api in args.api:
            product = torch.empty(rhs_shape, dtype=torch.float32, device=device)

            def multiply(left, right, *, chosen=api, destination=product):
                if chosen == "dtype_out":
                    return torch.ops.aten.bmm.dtype_out(left, right, torch.float32, out=destination)
                if chosen == "dtype":
                    return torch.ops.aten.bmm.dtype(left, right, torch.float32)
                # Cast traffic is inside the operation, including the product-only diagnostic.
                return torch.bmm(left.float(), right.float(), out=destination)

            def endpoint(*, count=columns, operation=multiply):
                torch.index_select(cache, 0, selected_indices(request_ids, accepted), out=selected)
                rhs = pack_rhs(query, key, value_heads, count).reshape(groups, width, count)
                return operation(selected.view(groups, width, width), rhs).view(batch, value_heads, width, count)

            def product_only(*, left=lower_base, right=lower_rhs, operation=multiply, count=columns):
                return operation(left, right).view(batch, value_heads, width, count)

            name = f"{api}_n{columns}"
            try:
                actual = endpoint()
                torch.npu.synchronize()
                if actual.dtype != torch.float32 or not actual.is_contiguous():
                    raise AssertionError("product must be contiguous FP32")
                q_cpu, k_cpu, state_cpu, _ = cpu_inputs
                expected = torch.matmul(state_cpu.float(), pack_rhs(q_cpu, k_cpu, value_heads, columns).float())
                error = discrepancy(actual, expected)
                torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=2e-6)
                if not error["finite"]:
                    raise AssertionError("nonfinite initial products")
                report["qualification"].append({"mode": name, "supported": True, "cpu_fp32_error": error})
            except (RuntimeError, NotImplementedError, AssertionError) as error:
                report["qualification"].append({"mode": name, "supported": False, "error": str(error)})
                save()
                continue
            for scope, operation in (("gather_pack_product", endpoint), ("product_only_diagnostic", product_only)):
                stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
                stream.wait_stream(torch.npu.current_stream())
                try:
                    for _ in range(2):
                        with torch.npu.stream(stream):
                            operation()
                    torch.npu.synchronize()
                    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                        graph_output = operation()
                    torch.npu.synchronize()
                except RuntimeError as error:
                    report["qualification"].append(
                        {"mode": name, "scope": scope, "capture_supported": False, "error": str(error)}
                    )
                    save()
                    continue
                candidates.append(
                    {
                        "mode": name,
                        "scope": scope,
                        "columns": columns,
                        "graph": graph,
                        "stream": stream,
                        "operation": operation,
                        "output": graph_output,
                        "lower_base": lower_base,
                        "lower_rhs": lower_rhs,
                    }
                )
            save()
    if not candidates:
        report["status"] = "no_supported_candidate"
        save()
        return

    for step in range(4):
        q_cpu, k_cpu, state_cpu, live = update_inputs(step)
        for candidate in candidates:
            columns = candidate["columns"]
            if candidate["scope"] == "product_only_diagnostic":
                candidate["lower_base"].copy_(state_cpu.to(device).reshape(groups, width, width))
                candidate["lower_rhs"].copy_(pack_rhs(query, key, value_heads, columns).reshape(groups, width, columns))
                torch.npu.synchronize()
            stream = candidate["stream"]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                eager = candidate["operation"]().clone()
                candidate["graph"].replay()
            stream.synchronize()
            torch.testing.assert_close(candidate["output"], eager, rtol=0, atol=0)
            expected = torch.matmul(state_cpu.float(), pack_rhs(q_cpu, k_cpu, value_heads, columns).float())
            torch.testing.assert_close(candidate["output"].cpu(), expected, rtol=1e-4, atol=2e-6)
            error = discrepancy(candidate["output"], expected)
            if not error["finite"]:
                raise AssertionError("nonfinite dynamic replay product")
            if columns == 16 and candidate["output"][..., 8:].count_nonzero().item():
                raise AssertionError("padded diagnostic columns must stay exactly zero")
            report["checks"].append(
                {
                    "mode": candidate["mode"],
                    "scope": candidate["scope"],
                    "step": step,
                    "live": live,
                    "mixed_accepted": True,
                    "eager_replay_exact": True,
                    "cpu_fp32_error": error,
                }
            )

    torch.testing.assert_close(cache.cpu(), cache_cpu, rtol=0, atol=0)
    if cache._version != cache_version:
        raise AssertionError("read-only product probe mutated the candidate cache")
    report["cache_all_candidates_null_inactive_unchanged"] = True
    torch.npu.reset_peak_memory_stats()
    for round_id in range(args.rounds):
        order = candidates if round_id % 2 == 0 else list(reversed(candidates))
        for candidate in order:
            stream = candidate["stream"]
            stream.wait_stream(torch.npu.current_stream())
            for _ in range(3):
                with torch.npu.stream(stream):
                    candidate["graph"].replay()
                stream.synchronize()
            for sample in range(args.repeats):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start = time.perf_counter()
                with torch.npu.stream(stream):
                    begin.record(stream)
                    candidate["graph"].replay()
                    end.record(stream)
                end.synchronize()
                report["samples"].append(
                    {
                        "mode": candidate["mode"],
                        "scope": candidate["scope"],
                        "round": round_id,
                        "sample": sample,
                        "device_us": begin.elapsed_time(end) * 1000,
                        "wall_us": (time.perf_counter() - start) * 1e6,
                    }
                )
            if not torch.isfinite(candidate["output"]).all().item():
                raise AssertionError("nonfinite timed graph output")
        save()
    report["summary"] = []
    for candidate in candidates:
        rows = [
            row for row in report["samples"] if row["mode"] == candidate["mode"] and row["scope"] == candidate["scope"]
        ]
        values = sorted(row["device_us"] for row in rows)
        report["summary"].append(
            {
                "mode": candidate["mode"],
                "scope": candidate["scope"],
                "samples": len(values),
                "mean_us": statistics.mean(values),
                "median_us": statistics.median(values),
                "p95_us": values[min(len(values) - 1, int(0.95 * len(values)))],
                "p99_us": values[min(len(values) - 1, int(0.99 * len(values)))],
                "per_round_median_us": [
                    statistics.median(row["device_us"] for row in rows if row["round"] == item)
                    for item in range(args.rounds)
                ],
            }
        )
    report["memory"] = {
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
    }
    report["status"] = "product_qualification_passed_not_complete_recurrence"
    save()
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
