# SPDX-License-Identifier: Apache-2.0
"""Compare one native attention-plan policy per fresh isolated process.

The endpoint includes pinned descriptor fill, required H2D, one native metadata
producer, sequential private attention occurrences and the final join. Four
canonical records keep their default buckets. This is an attention-chain
experiment, not a full-model, full-trace or serving throughput measurement.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import aclgraph_fia_sink_stage3_budget as baseline
from aclgraph_fia_sink_stage3_trace import load_trace_schedule
from aclgraph_fia_sink_stage4_prepare import PreparedDescriptors
from aclgraph_fia_sink_stage4_whole_plan import validate_whole_plan
from aclgraph_fia_sink_stage4_whole_probe import representative_records
from aclgraph_fia_sink_stage5_gate import pointers

REPORT_NAME = "stage6-producer-probe.json"
HASH_CHUNK_BYTES = 64 * 1024**2
OPAPI_SHA256 = "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40"
POLICIES = {
    "upstream": ("libfia_v2_sink_aicpu_kernels.so", "796983827bc6326a250009a2aacc810c44f65545ec2ae56dab517a5ec4e7111b"),
    "whole": ("libthesis_stage5_whole_aicpu.so", "1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90"),
}


def vendor_policy(vendor, policy):
    """Bind the configured AICPU dispatch name to its qualified library hash."""
    library_name, digest = POLICIES[policy]
    config = json.loads((vendor / "op_impl/cpu/config/cust_aicpu_kernel.json").read_text())
    if config["FusedInferAttentionScoreV2SinkMetadata"]["opInfo"]["kernelSo"] != library_name:
        raise ValueError("AICPU dispatcher selects a different policy library")
    library = vendor / "op_impl/cpu/aicpu_kernel/impl" / library_name
    if baseline.isolated.sha256(library) != digest:
        raise ValueError("selected AICPU policy library hash differs from the qualified build")
    opapi = vendor / "op_api/lib/libcust_opapi.so"
    if baseline.isolated.sha256(opapi) != OPAPI_SHA256:
        raise ValueError("attention OpAPI differs from the common qualified consumer")
    return dict(policy=policy, library=str(library), library_sha256=digest, opapi_sha256=OPAPI_SHA256)


def tensor_identity(torch, tensor):
    """Hash every byte using bounded CPU chunks outside all timed intervals."""
    digest = hashlib.sha256()
    row_bytes = tensor.numel() * tensor.element_size() // tensor.shape[0]
    rows_per_chunk = max(1, HASH_CHUNK_BYTES // row_bytes)
    for start in range(0, tensor.shape[0], rows_per_chunk):
        host = tensor[start : start + rows_per_chunk].cpu().contiguous()
        digest.update(host.view(torch.uint8).numpy().tobytes())
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest.hexdigest())


def input_identity(torch, layers):
    """Record bytes, not addresses, so independent policy processes can match."""
    entries = [{name: tensor_identity(torch, layer[name]) for name in ("query", "key", "value")} for layer in layers]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(occurrences=entries, sha256=digest)


def private_storage_ranges(layers):
    """Require private contiguous Q/K/V storage across all occurrences."""
    ranges = []
    for occurrence, layer in enumerate(layers):
        for name in ("query", "key", "value"):
            tensor = layer[name]
            if not tensor.is_contiguous():
                raise ValueError("private Q/K/V identity requires contiguous storage")
            start = tensor.data_ptr()
            ranges.append(
                dict(occurrence=occurrence, name=name, start=start, end=start + tensor.numel() * tensor.element_size())
            )
    ordered = sorted(ranges, key=lambda item: item["start"])
    for previous, current in zip(ordered, ordered[1:]):
        if previous["end"] > current["start"]:
            raise ValueError("private occurrence Q/K/V storage overlaps")
    return ranges


def plan_evidence(metadata, record, policy, cores):
    host = metadata.cpu()
    words = host.tolist()
    result = dict(words=words, sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest())
    result["metadata"] = baseline.validation.metadata_summary(host, cores)
    if policy == "whole":
        result["whole_coverage"] = validate_whole_plan(record, words, cores=cores["cube_core_num"])
    return result


def capture(torch, cores, args, schedule, records, status, save):
    layers, base = baseline.allocate_inputs(torch, schedule, args.occurrences, args.seed)
    status["input_identity"] = input_identity(torch, layers)
    status["private_storage_ranges"] = private_storage_ranges(layers)
    status["mask_identity"] = tensor_identity(torch, base["atten_mask"])
    status["memory"] = [baseline.memory(torch, "private_inputs_allocated")]
    status["metadata_preflight"] = []
    entries, frames, prepared = {}, {}, {}
    stream = torch.npu.Stream()
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = baseline.frame_inputs(base, layers, capacity)
        frames[capacity] = views, tensors
        baseline.prepare(torch, tensors, record)
        sources = PreparedDescriptors(torch, tensors, record, mode="sink")
        torch.npu.synchronize()
        sources.complete()
        metadata = baseline.graph_helpers.metadata_call(torch, tensors, cores)
        torch.npu.synchronize()
        host = metadata.cpu()
        evidence = dict(
            capacity=capacity,
            record=record,
            words=host.tolist(),
            sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest(),
        )
        status["metadata_preflight"].append(evidence)
        save()
        evidence.update(plan_evidence(metadata, record, args.policy, cores))
        evidence["verified_before_attention_capture"] = True
        save()
        # Buckets rotate during replay, so they retain independent graph pools
        # rather than relying on the shared-pool capture-order contract.
        graph = baseline.SinkGraph(torch, "shared", views, tensors, cores, torch.npu.graph_pool_handle(), stream)
        tracked = {**tensors, "metadata": graph.metadata[0]}
        tracked.update({f"query{index}": layer["query"] for index, layer in enumerate(views)})
        tracked.update({f"output{index}": output for index, output in enumerate(graph.outputs)})
        status.setdefault("captured_addresses", {})[str(capacity)] = pointers(tracked)
        status["memory"].append(baseline.memory(torch, f"captured_{capacity}"))
        prepared[capacity], entries[capacity] = sources, graph
    return layers, frames, prepared, entries


def qualify(torch, cores, args, records, layers, frames, prepared, entries, status, save, label):
    """Audit every bucket and occurrence before and after the timed repeats."""
    checks = []
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        entry, sources = entries[capacity], prepared[capacity]
        tracked = {**tensors, "metadata": entry.metadata[0]}
        tracked.update({f"query{index}": layer["query"] for index, layer in enumerate(views)})
        tracked.update({f"output{index}": output for index, output in enumerate(entry.outputs)})
        before = pointers(tracked)
        if before != status["captured_addresses"][str(capacity)]:
            raise AssertionError("input or graph output allocation changed since capture")
        query_bytes = [tensor_identity(torch, layer["query"]) for layer in views]
        q, kv = sources.prepare(record)
        entry.replay(q, kv)
        entry.synchronize()
        torch.npu.synchronize()
        sources.complete()
        outputs = [output.cpu().float() for output in entry.outputs]
        for actual in outputs:
            if not bool(torch.isfinite(actual).all()):
                raise AssertionError("native output is nonfinite")
            if not torch.equal(actual[record["live_tokens"] :], torch.zeros_like(actual[record["live_tokens"] :])):
                raise AssertionError("physical KV0 padding was not written as exact zero")
        if pointers(tracked) != before or [tensor_identity(torch, layer["query"]) for layer in views] != query_bytes:
            raise AssertionError("native execution changed an address or resident query")
        if tensors["q_lengths"].cpu().tolist() != q or tensors["kv_lengths"].cpu().tolist() != kv:
            raise AssertionError("native execution modified exact length inputs")
        table = tensors["block_table"].cpu()
        if table.tolist() != [list(row) for row in record["block_table"]]:
            raise AssertionError("native execution modified the page routing")
        errors = []
        for occurrence, layer in enumerate(layers):
            key = layer["key"].cpu().view(layer["key"].shape[0], 128, -1)
            value = layer["value"].cpu().view(layer["value"].shape[0], 128, -1)
            for row in sorted({0, record["batch"] // 2, record["batch"] - 1}):
                query = views[occurrence]["query"][row * 4 : (row + 1) * 4].cpu()
                expected = baseline.validation.cpu_reference(
                    torch, query, key, value, table[row : row + 1], [4], [kv[row]], True
                )
                actual = outputs[occurrence][row * 4 : (row + 1) * 4]
                torch.testing.assert_close(
                    actual, expected, rtol=baseline.validation.RTOL, atol=baseline.validation.ATOL
                )
                errors.append(
                    dict(occurrence=occurrence, row=row, max_abs_error=float((actual - expected).abs().max()))
                )
        if tensor_identity(torch, tensors["atten_mask"]) != status["mask_identity"]:
            raise AssertionError("native execution modified the causal mask")
        check = dict(
            capacity=capacity,
            step=record["step"],
            private_occurrences=len(views),
            fp32_sampled_rows=errors,
            output_hashes=[tensor_identity(torch, output) for output in entry.outputs],
            finite=True,
            all_outputs_finite=True,
            exact_zero_padding=True,
            stable_addresses=True,
            exact_descriptors_read_only=True,
            resident_queries_unchanged=True,
            plan=plan_evidence(entry.metadata[0], record, args.policy, cores),
        )
        checks.append(check)
        status[label] = checks
        save()
    return checks


def experiment(torch, cores, args, status, save):
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError("native policy comparison requires the qualified 20/40 allocation")
    torch.set_num_threads(4)
    schedule = load_trace_schedule(args.trace, args.selection)
    records = representative_records(schedule, screen=args.screen)
    status.update(
        canonical_digest=schedule["canonical_digest"],
        selected_records=records,
        stage="capture",
        occurrences=args.occurrences,
        policy=args.policy,
        trial=args.trial,
        repeats=args.repeats,
        trace_contract={key: value for key, value in schedule.items() if key != "records"},
    )
    save()
    status["installed_catalog"] = baseline.installed_catalog(schedule, args.trace, args.model)
    status["trace_contract"]["installed_catalog_revalidated"] = True
    layers, frames, prepared, entries = capture(torch, cores, args, schedule, records, status, save)
    status.update(npu_work_executed=True, stage="qualification")
    qualify(torch, cores, args, records, layers, frames, prepared, entries, status, save, "pre_checks")
    for record in records:
        capacity = record["physical_tokens"]
        q, kv = prepared[capacity].prepare(record)
        entries[capacity].replay(q, kv)
        entries[capacity].synchronize()
        torch.npu.synchronize()
        prepared[capacity].complete()
    status["samples"] = []
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for repeat in range(args.repeats):
        offset = (args.trial + repeat) % len(records)
        for record in records[offset:] + records[:offset]:
            capacity = record["physical_tokens"]
            entry = entries[capacity]
            torch.npu.synchronize()
            begin = time.perf_counter_ns()
            q, kv = prepared[capacity].prepare(record)
            ready = time.perf_counter_ns()
            start.record(entry.stream)
            entry.replay(q, kv)
            end.record(entry.stream)
            submitted = time.perf_counter_ns()
            entry.synchronize()
            end.synchronize()
            finish = time.perf_counter_ns()
            prepared[capacity].complete()
            status["samples"].append(
                dict(
                    trial=args.trial,
                    policy=args.policy,
                    repeat=repeat,
                    step=record["step"],
                    batch=record["batch"],
                    capacity=capacity,
                    wall_us=(finish - begin) / 1000,
                    preparation_enqueue_us=(ready - begin) / 1000,
                    submit_join_cpu_us=(submitted - ready) / 1000,
                    queue_span_us=start.elapsed_time(end) * 1000,
                )
            )
        status.update(stage="timing", completed_repeat=repeat)
        save()
    qualify(torch, cores, args, records, layers, frames, prepared, entries, status, save, "post_checks")
    status["post_input_identity"] = input_identity(torch, layers)
    if status["post_input_identity"] != status["input_identity"]:
        raise AssertionError("private Q/K/V bytes changed during native policy experiment")
    status["post_mask_identity"] = tensor_identity(torch, frames[records[0]["physical_tokens"]][1]["atten_mask"])
    if status["post_mask_identity"] != status["mask_identity"]:
        raise AssertionError("causal mask bytes changed during native policy experiment")
    status.update(
        state="completed",
        stage="completed",
        returncode=0,
        performance_measured=True,
        full_model_qualified=False,
        full_trace_measured=False,
        end_to_end_gain_established=False,
        prepared_descriptors={str(capacity): source.summary() for capacity, source in prepared.items()},
        finished_unix=time.time(),
    )
    save()


def child(args):
    target = args.output_dir / REPORT_NAME
    status = json.loads(target.read_text())
    save = lambda: baseline.isolated.persist(target, status)
    try:
        if os.environ.get("TASK_QUEUE_ENABLE") != "0" or os.environ.get("VLLM_BATCH_INVARIANT") != "0":
            raise RuntimeError("fresh child requires queue0/batch-invariant0 before Torch import")
        status.update(child_pid=os.getpid(), state="running")
        save()
        torch, cores, handle = baseline.validation.initialize_native(args, status, save)
        with torch.inference_mode():
            experiment(torch, cores, args, status, save)
        if handle is None:
            raise AssertionError("native library owner lost")
        status["final_native_provenance"] = baseline.final_native_provenance(args.opapi_lib)
        save()
        return 0
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        traceback.print_exc()
        return 1


def stop_child(process):
    """Join the owned child session, escalating only if TERM cannot finish."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def launcher(args):
    isolated = baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only an isolated /workspace root is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    policy = vendor_policy(args.opp_root, args.policy)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build["extension_library"], args.root, file=True)
    if not args.output_dir.is_relative_to(args.root) or args.output_dir.exists():
        raise ValueError("choose a new output directory inside the isolated root")
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / REPORT_NAME
    isolated.persist(
        target,
        dict(
            state="queued",
            started_unix=time.time(),
            policy_provenance=policy,
            scope="four canonical default buckets; sequential attention chain, not full model/trace/serving",
            endpoint=(
                "pinned preparation + required H2D + native shared producer + private attention replay + final join"
            ),
            scripts={p.name: isolated.sha256(p) for p in Path(__file__).parent.glob("aclgraph_fia_sink_stage*.py")},
            extension_sha256=isolated.sha256(args.extension),
            opp_artifacts={
                str(p.relative_to(args.opp_root)): isolated.sha256(p)
                for p in sorted(args.opp_root.rglob("*"))
                if p.is_file()
            },
            performance_measured=False,
            npu_work_executed=False,
        ),
    )
    env = dict(
        os.environ,
        ASCEND_CUSTOM_OPP_PATH=str(args.opp_root),
        ASCEND_RT_VISIBLE_DEVICES="0",
        TASK_QUEUE_ENABLE="0",
        VLLM_BATCH_INVARIANT="0",
    )
    env["LD_LIBRARY_PATH"] = str(args.opapi_lib.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    for name in (
        "root",
        "opp_root",
        "extension",
        "output_dir",
        "trace",
        "selection",
        "policy",
        "trial",
        "repeats",
        "seed",
        "model",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.screen:
        command.append("--screen")
    with (args.output_dir / "stage6-producer-probe.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=args.root,
            env=env,
            start_new_session=True,
        )
        previous_handlers = {}

        def forward_signal(number, _frame):
            stop_child(process)
            raise SystemExit(128 + number)

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[number] = signal.signal(number, forward_signal)
            try:
                rc = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                stop_child(process)
                rc = 124
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
            stop_child(process)
    status = json.loads(target.read_text())
    if status.get("returncode") != rc:
        status.update(state="failed", returncode=rc, error="child ended before final status")
        isolated.persist(target, status)
    print(json.dumps(dict(state=status["state"], returncode=rc, status=str(target))))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "opp-root", "output-dir", "trace", "selection"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--policy", choices=tuple(POLICIES), required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("root", "opp_root", "trace", "selection"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.repeats < 1 or args.timeout < 1 or args.trial < 0:
        parser.error("positive repeats/timeout and nonnegative trial required")
    args.occurrences = 1 if args.screen else 16
    if args.screen:
        args.repeats = 1
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
