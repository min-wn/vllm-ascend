# SPDX-License-Identifier: Apache-2.0
"""One fixed-graph metadata producer probe, not attention/model performance.

The existing AICPU whole-head producer and an experimental AIV implementation
read the same changing device lengths and export the same 1024-word ABI.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


def run(args, report, save):
    import torch
    import torch_npu
    from aclgraph_descriptor_plan_kernel import launch

    torch.npu.set_device(0)
    torch.ops.load_library(str(args.extension))
    cases = json.loads(args.cases.read_bytes())
    batch = cases["physical_batch"]
    kv = torch.zeros((batch,), device="npu", dtype=torch.int64)
    host = torch.empty((batch,), dtype=torch.int64, pin_memory=True)
    q = torch.arange(4, batch * 4 + 1, 4, device="npu", dtype=torch.int64)
    output = torch.empty((1024,), device="npu", dtype=torch.int32)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    report.update(versions=dict(torch=torch.__version__, torch_npu=torch_npu.__version__), stage="compiling")
    save()

    def native():
        return torch.ops.thesis_fia_sink_stage1._npu_fused_infer_attention_score_v2_sink_metadata(
            12,
            2,
            256,
            256,
            actual_seq_lengths=q,
            actual_seq_lengths_kv=kv,
            batch_size=batch,
            sparse_mode=3,
            input_layout="TND",
            input_layout_kv="BnBsH",
            block_size=128,
            aic_core_num=20,
            aiv_core_num=40,
            batch_invariant=False,
        )

    host.copy_(torch.tensor(cases["cases"][0]["kv_lengths"], dtype=torch.int64))
    kv.copy_(host, non_blocking=True)
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        launch(kv, output)
        native()
    stream.synchronize()
    graphs, outputs = {}, dict(aiv=output)
    for mode in ("aicpu", "aiv"):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            if mode == "aicpu":
                outputs[mode] = native()
            else:
                launch(kv, output)
        graphs[mode] = graph
    stream.synchronize()
    initial = dict(kv=kv.data_ptr(), q=q.data_ptr(), **{m: t.data_ptr() for m, t in outputs.items()})
    report["stage"] = "qualification"
    save()
    for case in cases["cases"]:
        host.copy_(torch.tensor(case["kv_lengths"], dtype=torch.int64))
        with torch.npu.stream(stream):
            kv.copy_(host, non_blocking=True)
            for mode in graphs:
                graphs[mode].replay()
        stream.synchronize()
        observed = {m: t.cpu().tolist() for m, t in outputs.items()}
        report["last_comparison"] = dict(name=case["name"], metadata=observed)
        report["loaded_opapi_paths"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libcust_opapi.so" in line
            }
        )
        save()
        if observed["aiv"] != case["expected_metadata"]:
            differences = [i for i, (a, b) in enumerate(zip(observed["aiv"], case["expected_metadata"])) if a != b]
            raise AssertionError(f"AIV versus CPU oracle: {case['name']}, words={differences[:20]}")
        if observed["aiv"] != observed["aicpu"]:
            differences = [i for i, (a, b) in enumerate(zip(observed["aiv"], observed["aicpu"])) if a != b]
            raise AssertionError(f"AIV versus native whole policy: {case['name']}, words={differences[:20]}")
        current = dict(kv=kv.data_ptr(), q=q.data_ptr(), **{m: t.data_ptr() for m, t in outputs.items()})
        if current != initial or kv.cpu().tolist() != case["kv_lengths"]:
            raise AssertionError("descriptor address/content or metadata address changed")
        report["checks"].append(dict(name=case["name"], all_1024_words_equal=True, descriptor_read_only=True))
        save()

    report["stage"] = "timing_producers_only"
    save()
    for trial in range(4):
        for mode in tuple(graphs) if trial % 2 == 0 else tuple(reversed(graphs)):
            for repeat in range(13):
                values = cases["cases"][repeat % 2]["kv_lengths"]
                torch.npu.synchronize()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begun = time.perf_counter_ns()
                host.copy_(torch.tensor(values, dtype=torch.int64))
                with torch.npu.stream(stream):
                    start.record(stream)
                    kv.copy_(host, non_blocking=True)
                    graphs[mode].replay()
                    end.record(stream)
                end.synchronize()
                if repeat >= 3:
                    report["samples"].append(
                        dict(
                            trial=trial,
                            mode=mode,
                            sample=repeat - 3,
                            device_us=start.elapsed_time(end) * 1000,
                            wall_us=(time.perf_counter_ns() - begun) / 1000,
                        )
                    )
            save()
    for mode in graphs:
        report.setdefault("median_device_us", {})[mode] = statistics.median(
            r["device_us"] for r in report["samples"] if r["mode"] == mode
        )
    report.update(
        status="complete_metadata_only_probe", stage="finished", attention_or_model_performance_measured=False
    )
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opp", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        environment = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(args.opp), TASK_QUEUE_ENABLE="0")
        environment["LD_LIBRARY_PATH"] = str(args.opp / "op_api/lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            env=environment,
            check=True,
            timeout=240,
        )
        return
    report = dict(
        status="started",
        scope=__doc__,
        checks=[],
        samples=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(
            Path(__file__).with_name("aclgraph_descriptor_plan_kernel.py").read_bytes()
        ).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        opapi_sha256=hashlib.sha256((args.opp / "op_api/lib/libcust_opapi.so").read_bytes()).hexdigest(),
        aicpu_sha256=hashlib.sha256(
            (args.opp / "op_impl/cpu/aicpu_kernel/impl/libthesis_stage5_whole_aicpu.so").read_bytes()
        ).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if report["extension_sha256"] != "25a41e47473a3bc9c34a50c053766181c7a764ea496b93d4d3b70180b0ecf76b":
            raise ValueError("extension differs from the qualified stage-two artifact")
        if report["opapi_sha256"] != "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40":
            raise ValueError("OpAPI differs from the qualified common consumer")
        if report["aicpu_sha256"] != "1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90":
            raise ValueError("AICPU baseline differs from the qualified fresh whole-head export")
        run(args, report, save)
    except Exception:
        report.update(status="failed_metadata_only_probe", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
