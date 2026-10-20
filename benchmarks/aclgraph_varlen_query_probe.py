# SPDX-License-Identifier: Apache-2.0
"""Bounded fixed-graph variable-Q budget probe; no new attention kernel/model.

The consumer-only graph uses a freshly generated plan OUTSIDE timing and is
diagnostic, not a deployable algorithm. Closed includes descriptor/table H2D,
native planning, native attention, and the final join. Q/K/V stay resident.
Frozen CPU snapshots represent scheduler-produced inputs; fixture synthesis is
excluded, while pinned-buffer preparation and H2D remain inside the endpoint.
"""

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from aclgraph_varlen_query_trace import PHYSICAL_BATCH, PHYSICAL_TOKENS, Q4_DIGEST, digest, validate_trace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persist(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def experiment(args, report, save):
    import aclgraph_fia_sink_stage2_validate as validation

    torch, cores, handle = validation.initialize_native(args, report, save)
    import torch_npu

    torch.set_num_threads(4)
    torch.manual_seed(20261020)
    frozen = validate_trace(json.loads(args.trace.read_bytes()))
    original = json.loads(args.q4.read_bytes())
    canonical = dict(original["trace"])
    canonical["initial_prefix_by_request"] = {int(k): v for k, v in canonical["initial_prefix_by_request"].items()}
    if digest(canonical) != Q4_DIGEST:
        raise ValueError("Q4 regression source changed")
    trace = frozen["trace"]
    records = [trace["steps"][i] for i in trace["probe_steps"]]
    maximum = max(max(s["seq_lengths"]) for s in original["trace"]["steps"])
    width = (maximum + 127) // 128
    request_count = max(max(s["request_ids"]) for s in original["trace"]["steps"]) + 1
    tensors = dict(
        query_storage=torch.randn((PHYSICAL_TOKENS + 4, 12, 256), device="npu", dtype=torch.bfloat16) * 0.5,
        key=torch.randn((request_count * width + 1, 128, 512), device="npu", dtype=torch.bfloat16) * 0.5,
        value=torch.randn((request_count * width + 1, 128, 512), device="npu", dtype=torch.bfloat16) * 0.5,
        q_lengths=torch.zeros(PHYSICAL_BATCH, device="npu", dtype=torch.int64),
        kv_lengths=torch.zeros(PHYSICAL_BATCH, device="npu", dtype=torch.int64),
        block_table=torch.zeros((PHYSICAL_BATCH, width), device="npu", dtype=torch.int32),
        atten_mask=torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), diagonal=1),
    )
    tensors["key"][0].fill_(13.25)
    tensors["value"][0].fill_(-17.5)
    tensors["query"] = tensors["query_storage"][2:-2]
    host = {
        k: torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
        for k, v in tensors.items()
        if k in ("q_lengths", "kv_lengths", "block_table")
    }
    ops = torch.ops.thesis_fia_sink_stage1

    def fixture(record, q4=False):
        if q4:
            q = [4] * PHYSICAL_BATCH
            kv = record["seq_lengths"] + [0] * (PHYSICAL_BATCH - record["batch"])
        else:
            q, kv = record["q_lengths"], record["kv_lengths"]
        endpoints = torch.tensor(q, dtype=torch.int64).cumsum(0)
        table = torch.zeros((PHYSICAL_BATCH, width), dtype=torch.int32)
        for row, request_id in enumerate(record["request_ids"]):
            count = (kv[row] + 127) // 128
            table[row, :count] = torch.arange(1 + request_id * width, 1 + request_id * width + count, dtype=torch.int32)
        return q, kv, table, endpoints, torch.tensor(kv, dtype=torch.int64)

    fixtures = {
        (q4, record["step"]): fixture(record, q4)
        for q4, selected in ((False, records), (True, original["trace"]["steps"]))
        for record in selected
    }

    def prepare(record, q4=False):
        q, kv, table, endpoints, kv_tensor = fixtures[q4, record["step"]]
        host["q_lengths"].copy_(endpoints)
        host["kv_lengths"].copy_(kv_tensor)
        host["block_table"].copy_(table)
        return q, kv, table

    def upload():
        for name, source in host.items():
            tensors[name].copy_(source, non_blocking=True)

    def planner():
        return ops._npu_fused_infer_attention_score_v2_sink_metadata(
            12,
            2,
            256,
            256,
            actual_seq_lengths=tensors["q_lengths"],
            actual_seq_lengths_kv=tensors["kv_lengths"],
            batch_size=PHYSICAL_BATCH,
            sparse_mode=3,
            input_layout="TND",
            input_layout_kv="BnBsH",
            block_size=128,
            aic_core_num=cores["cube_core_num"],
            aiv_core_num=cores["vector_core_num"],
            batch_invariant=False,
        )

    def attention(meta):
        return ops.npu_fused_infer_attention_score_v2_sink(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            actual_seq_qlen=tensors["q_lengths"],
            actual_seq_kvlen=tensors["kv_lengths"],
            block_table=tensors["block_table"],
            meta_data=meta,
            atten_mask=tensors["atten_mask"],
            num_query_heads=12,
            num_key_value_heads=2,
            softmax_scale=256**-0.5,
            input_layout="TND",
            sparse_mode=3,
            block_size=128,
            return_softmax_lse=False,
        )[0]

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    prepare(records[0])
    with torch.npu.stream(stream):
        upload()
        diagnostic_meta = planner()
        attention(diagnostic_meta)
    stream.synchronize()
    graphs, outputs, plans = {}, {}, {}
    for mode in ("closed", "producer", "consumer", "upload"):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            if mode in ("closed", "producer"):
                plans[mode] = planner()
            if mode in ("closed", "consumer"):
                outputs[mode] = attention(plans["closed"] if mode == "closed" else diagnostic_meta)
            if mode == "upload":
                # H2D is kept outside the graph in every mode. Empty graph is
                # a matched submission/join diagnostic, not a memcpy baseline.
                tensors["q_lengths"].add_(0)
        graphs[mode] = graph
    stream.synchronize()
    input_addresses = {k: v.data_ptr() for k, v in tensors.items()}
    output_addresses = {k: v.data_ptr() for k, v in outputs.items()}
    immutable_before = {
        k: (v.clone() if k == "query_storage" else v[0].clone())
        for k, v in tensors.items()
        if k in ("query_storage", "key", "value")
    }

    def qualify(record, *, q4=False, oracle=False):
        q, kv, table = prepare(record, q4)
        with torch.npu.stream(stream):
            upload()
            outputs["closed"].fill_(31.25)
            graphs["closed"].replay()
        stream.synchronize()
        actual = outputs["closed"].float().cpu()
        if not bool(torch.isfinite(actual).all()):
            raise AssertionError("nonfinite output")
        active = record["batch"] * 4 if q4 else sum(q)
        if torch.count_nonzero(actual[active:]):
            raise AssertionError("physical tail was not overwritten with exact zeros")
        errors = []
        if oracle:
            offset = 0
            selected = {0, record["batch"] // 2, record["batch"] - 1}
            for row, length in enumerate(q):
                if row in selected:
                    count = (kv[row] + 127) // 128
                    pages = table[row, :count].long().to("npu")
                    key = tensors["key"][pages].cpu()
                    value = tensors["value"][pages].cpu()
                    query = tensors["query"][offset : offset + length].cpu()
                    local_table = torch.arange(count, dtype=torch.int32).unsqueeze(0)
                    expected = validation.cpu_reference(
                        torch, query, key, value, local_table, [length], [kv[row]], True
                    )
                    observed = actual[offset : offset + length]
                    torch.testing.assert_close(observed, expected, rtol=validation.RTOL, atol=validation.ATOL)
                    errors.append(
                        dict(
                            row=row,
                            query_length=length,
                            kv_length=kv[row],
                            max_abs_error=float((observed - expected).abs().max()),
                        )
                    )
                offset += length
        if {k: v.data_ptr() for k, v in tensors.items()} != input_addresses or {
            k: v.data_ptr() for k, v in outputs.items()
        } != output_addresses:
            raise AssertionError("capture storage changed")
        for k in host:
            if not torch.equal(tensors[k].cpu(), host[k]):
                raise AssertionError("read-only descriptor/table changed")
        return dict(
            step=record["step"],
            q4=q4,
            finite=True,
            exact_zero_tail=True,
            stable_addresses=True,
            oracle=errors,
            metadata=validation.metadata_summary(plans["closed"].cpu(), cores),
        )

    report.update(
        stage="qualification",
        trace_sha256=frozen["trace_sha256"],
        q4_trace_sha256=Q4_DIGEST,
        geometry=dict(batch=PHYSICAL_BATCH, tokens=PHYSICAL_TOKENS, table_width=width),
        checks=[],
    )
    save()
    for record in records:
        report["checks"].append(qualify(record, oracle=True))
        save()
    # Original Q4 order and true KV lengths; all 128 snapshots are replayed.
    # FP32 reference only on first/final snapshots, finite/tail/address on all.
    report["q4_regression"] = []
    for i, record in enumerate(original["trace"]["steps"]):
        report["q4_regression"].append(qualify(record, q4=True, oracle=i in (0, 127)))
    report["stage"] = "timing"
    report["samples"] = []
    save()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    modes = ("closed", "producer", "consumer", "upload")
    for trial in range(4):
        for mode in modes if trial % 2 == 0 else modes[::-1]:
            for record in records:
                # Pinned-buffer preparation is included in wall timing.
                # Consumer-only plan readiness is excluded and explicitly
                # diagnostic; its CPU construction is still repeated below.
                if mode == "consumer":
                    prepare(record)
                    with torch.npu.stream(stream):
                        upload()
                        diagnostic_meta.copy_(planner())
                    stream.synchronize()
                began = time.perf_counter_ns()
                prepare(record)
                with torch.npu.stream(stream):
                    start.record(stream)
                    upload()
                    graphs[mode].replay()
                    end.record(stream)
                end.synchronize()
                elapsed = (time.perf_counter_ns() - began) / 1000
                report["samples"].append(
                    dict(
                        trial=trial,
                        mode=mode,
                        step=record["step"],
                        wall_us=elapsed,
                        queue_us=start.elapsed_time(end) * 1000,
                    )
                )
        save()
    report["post_timing"] = [qualify(records[-1], oracle=True)]
    for k, before in immutable_before.items():
        after = tensors[k] if k == "query_storage" else tensors[k][0]
        if not torch.equal(after, before):
            raise AssertionError("query guard/input or poisoned page changed")
    report["summary"] = {
        mode: {
            field: statistics.median(s[field] for s in report["samples"] if s["mode"] == mode)
            for field in ("wall_us", "queue_us")
        }
        for mode in modes
    }
    report["stage"] = "profile"
    save()
    report["profiles"] = []
    for index in trace["profile_steps"]:
        record = trace["steps"][index]
        prepare(record)
        with torch.npu.stream(stream):
            upload()
            graphs["closed"].replay()
        stream.synchronize()
        path = args.output_dir / f"profile-step-{index}.json"
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as prof:
            for repeat in range(3):
                with torch.profiler.record_function(f"varlen/closed/step{index}/repeat{repeat}"):
                    with torch.npu.stream(stream):
                        upload()
                        graphs["closed"].replay()
                    stream.synchronize()
        prof.export_chrome_trace(str(path))
        report["profiles"].append(dict(step=index, path=str(path), sha256=sha(path), outside_formal_timing=True))
        save()
    if handle is None:
        raise AssertionError("native handle lost")
    report.update(
        state="completed",
        stage="completed",
        candidate_implemented=False,
        model_measured=False,
        qualifications=(
            "8 variable-Q snapshots and all 128 original Q4 snapshots; sampled FP32 reference, not all-output oracle"
        ),
    )
    save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--opp-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
    if not args.child:
        if args.output_dir.exists():
            raise FileExistsError("use a fresh output directory")
        args.output_dir.mkdir(parents=True)
        environment = dict(
            os.environ, ASCEND_CUSTOM_OPP_PATH=str(args.opp_root), ASCEND_RT_VISIBLE_DEVICES="0", TASK_QUEUE_ENABLE="0"
        )
        environment["LD_LIBRARY_PATH"] = str(args.opapi_lib.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
        return subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--child"], env=environment, timeout=240
        ).returncode
    signal.alarm(230)
    report = dict(
        state="started",
        scripts={Path(p).name: sha(p) for p in (__file__, Path(__file__).with_name("aclgraph_varlen_query_trace.py"))},
        trace_file_sha256=sha(args.trace),
        extension_sha256=sha(args.extension),
        opapi_sha256=sha(args.opapi_lib),
        opp_root=str(args.opp_root),
    )
    target = args.output_dir / "probe.json"

    def save():
        persist(target, report)

    try:
        save()
        experiment(args, report, save)
        print(json.dumps(report["summary"], indent=2))
        return 0
    except BaseException:
        report.update(state="failed", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
