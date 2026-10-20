# SPDX-License-Identifier: Apache-2.0
"""Bounded existing-GMM qualification for device counts in a fixed ACLGraph.

One synthetic Down matrix, no model loading or communication. Timing uses only
the ordinary default-capacity B98/Q4 case, not the small correctness cases.
This probes an existing community mechanism; it is not a new dense operator.
"""

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    report = dict(scope=__doc__, status="started", samples=[], checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu

        report["versions"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__)
        report["schemas"] = [str(s) for s in torch._C._jit_get_schemas_for_operator("npu::npu_grouped_matmul")]
        doc = str(torch_npu.npu_grouped_matmul.__doc__)
        report["documentation"] = doc
        report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.inventory_only:
            report["status"] = "inventory_only_no_device_initialization"
            save()
            return

        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        torch.manual_seed(20261020)
        capacity, k, n = 400, 8704, 5120
        # A single 3-D weight is the documented dynamic M grouping interface.
        weight_cpu = torch.randn((1, n, k), dtype=torch.bfloat16) * 0.01
        weight_nd = weight_cpu.to("npu")
        # Detach the 2-D allocation from the parent 3-D tensor: format-casting
        # a view can preserve a 5-D storage descriptor, unsupported by NZ MM.
        weight_nz = torch_npu.npu_format_cast(weight_nd[0].clone(), 29)
        x = torch.randn((capacity, k), dtype=torch.bfloat16, device="npu")
        source = torch.randn_like(x)
        counts_host = torch.empty((1,), dtype=torch.int64, pin_memory=True)
        counts = torch.empty((1,), dtype=torch.int64, device="npu")
        full_count = torch.full_like(counts, capacity)
        row_ids = torch.arange(capacity, device="npu").view(-1, 1)
        report["geometry"] = dict(capacity=capacity, k=k, n=n, live=392, dtype="BF16")

        def gmm(descriptor):
            return torch_npu.npu_grouped_matmul(
                [x],
                [weight_nd.transpose(-1, -2)],
                group_list=descriptor,
                split_item=2,
                group_type=0,
                group_list_type=0,
            )[0]

        counts.fill_(392)
        eager = gmm(counts)
        expected = torch.nn.functional.linear(x, weight_nd[0])
        torch.testing.assert_close(eager[:392], expected[:392], rtol=0.02, atol=0.02)
        report["eager"] = dict(shape=list(eager.shape), passed=True)
        save()
        if tuple(eager.shape) != (capacity, n):
            raise AssertionError("existing GMM did not retain fixed-capacity output")

        torch.nn.functional.linear(x, weight_nz)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graphs, outputs = {}, {}
        for mode in ("matmul_nz", "gmm_full", "gmm_dynamic"):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                outputs[mode] = (
                    torch.nn.functional.linear(x, weight_nz)
                    if mode == "matmul_nz"
                    else gmm(counts if mode == "gmm_dynamic" else full_count)
                )
                if mode == "gmm_dynamic":
                    # GMM leaves inactive output untouched. A complete output
                    # needs a descriptor-driven tail clear inside the graph.
                    outputs[mode].masked_fill_(row_ids >= counts.view(1, 1), 0)
            graphs[mode] = graph
        stream.synchronize()
        pointers = dict(x=x.data_ptr(), counts=counts.data_ptr(), **{m: y.data_ptr() for m, y in outputs.items()})

        # Poison prior output to expose shrink/grow stale-tail errors. The
        # descriptor-driven clear must restore the entire output in the graph.
        for step, live in enumerate((392, 400, 4, 0, 396, 388, 392)):
            source.normal_()
            x.copy_(source)
            counts_host.fill_(live)
            counts.copy_(counts_host, non_blocking=True)
            outputs["gmm_dynamic"].fill_(7)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graphs["gmm_dynamic"].replay()
            stream.synchronize()
            actual = outputs["gmm_dynamic"]
            ref = torch.nn.functional.linear(x, weight_nz)
            if live:
                torch.testing.assert_close(actual[:live], ref[:live], rtol=0.02, atol=0.02)
            if not torch.equal(actual[live:], torch.zeros_like(actual[live:])):
                raise AssertionError("descriptor tail clear left stale values")
            current = dict(x=x.data_ptr(), counts=counts.data_ptr(), **{m: y.data_ptr() for m, y in outputs.items()})
            if current != pointers:
                raise AssertionError("captured addresses changed")
            report["checks"].append(
                dict(step=step, live=live, fixed_addresses=True, active_close=True, tail_zero_exact=True)
            )
            save()

        # All controls pay input staging. Only the dynamic path pays count H2D
        # and a required tail clear; padded bias-free MM already yields zeros.
        source[392:].zero_()
        for trial in range(4):
            modes = tuple(graphs) if trial % 2 == 0 else tuple(reversed(graphs))
            for mode in modes:
                for repeat in range(13):
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    counts_host.fill_(392)
                    begun = time.perf_counter_ns()
                    with torch.npu.stream(stream):
                        start.record(stream)
                        x.copy_(source)
                        if mode == "gmm_dynamic":
                            counts.copy_(counts_host, non_blocking=True)
                        graphs[mode].replay()
                        end.record(stream)
                    end.synchronize()
                    wall = (time.perf_counter_ns() - begun) / 1000
                    if repeat >= 3:
                        report["samples"].append(
                            dict(
                                trial=trial,
                                mode=mode,
                                sample=repeat - 3,
                                device_us=start.elapsed_time(end) * 1000,
                                wall_us=wall,
                            )
                        )
                save()
        for mode, actual in outputs.items():
            ref = torch.nn.functional.linear(source, weight_nz)
            torch.testing.assert_close(actual[:392], ref[:392], rtol=0.02, atol=0.02)
            if not torch.equal(actual[392:], torch.zeros_like(actual[392:])):
                raise AssertionError(f"timed endpoint has stale tail: {mode}")
        report["post_timing_full_output_checked"] = True
        for mode in graphs:
            values = [r["device_us"] for r in report["samples"] if r["mode"] == mode]
            report.setdefault("median_device_us", {})[mode] = statistics.median(values)
        report["status"] = "complete_bounded_community_gmm_probe"
        report["limitations"] = [
            "One synthetic weight and warm-cache endpoint, not a model or service benchmark.",
            "Existing GMM uses ND 3-D weight; matmul uses qualified NZ, so dispatch and layout change together.",
            "No tensor parallel communication, new kernel or dense innovation is established.",
            "Small and zero counts are correctness cases only; timed live count is B98/Q4=392 at default400.",
        ]
        save()
    except Exception:
        report.update(status="failed_bounded_probe", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
