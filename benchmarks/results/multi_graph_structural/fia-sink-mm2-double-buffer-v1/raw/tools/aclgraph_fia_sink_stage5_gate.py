# SPDX-License-Identifier: Apache-2.0
"""Qualify native no-FD metadata on one fixed Q4 ACLGraph; no timing claims."""

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

REPORT_NAME = "stage5-native-gate.json"
CAPACITY = 400
AICPU_LIBRARY = "libthesis_stage5_whole_aicpu.so"


def gate_records(schedule):
    selected = [row for row in schedule["records"] if row["physical_tokens"] == CAPACITY]
    if len(selected) < 2 or selected[0]["step"] != 0:
        raise ValueError("the unchanged trace must start at capacity400 and later revisit that bucket")
    page_count = schedule["owner_pages"]["page_count"]
    first, later = (dict(row, page_count=page_count) for row in selected[:2])
    empty = dict(
        later,
        batch=0,
        live_tokens=0,
        kv_lengths=(0,) * later["physical_batch"],
        block_table=tuple((0,) * len(row) for row in later["block_table"]),
        request_ids=(),
    )
    return (("initial", first), ("changed", later), ("zero_kv", empty), ("recover", first))


def pointers(tensors):
    return {
        name: dict(pointer=tensor.data_ptr(), shape=list(tensor.shape), stride=list(tensor.stride()))
        for name, tensor in tensors.items()
    }


def inspect_native_plan(metadata, record, label, args, status, save):
    """Persist the exact payload and its provenance before any ABI assertion."""
    diagnostic = dict(
        label=label,
        record=record,
        metadata_descriptor=pointers({"metadata": metadata})["metadata"],
        metadata_dtype=str(metadata.dtype),
        selected_opapi_library=str(args.opapi_lib),
        extension_sha256=status.get("extension_sha256"),
        staged_opp_artifacts=status.get("opp_artifacts"),
        status="reading_native_payload",
    )
    status.setdefault("native_plan_diagnostics", []).append(diagnostic)
    save()
    try:
        host = metadata.cpu()
        words = host.tolist()
        diagnostic.update(
            words=words,
            word_count=len(words),
            payload_sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest(),
            base_metadata=words[576:586],
            aic_slots=[words[index * 10 : (index + 1) * 10] for index in range(20)],
            status="payload_saved_before_validation",
        )
        save()
        diagnostic["native_provenance"] = baseline.final_native_provenance(args.opapi_lib)
        save()
        check = validate_whole_plan(record, words, cores=20)
        diagnostic.update(status="validated", validation=check)
        save()
        return words, check
    except BaseException:
        diagnostic.update(status="failed", error=traceback.format_exc())
        save()
        raise


def run_gate(torch, cores, args, status, save):
    """Validate metadata before capture, then audit each fixed-address replay."""
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError("this safety gate requires the qualified 910B3 20/40 allocation")
    schedule = load_trace_schedule(args.trace, args.selection)
    records = gate_records(schedule)
    status["installed_catalog"] = baseline.installed_catalog(schedule, args.trace, args.model)
    layers, base = baseline.allocate_inputs(torch, schedule, 1, args.seed)
    views, tensors = baseline.frame_inputs(base, layers, CAPACITY)
    first = records[0][1]
    baseline.prepare(torch, tensors, first)
    prepared = PreparedDescriptors(torch, tensors, first, mode="sink")
    torch.npu.synchronize()
    prepared.complete()
    status.update(stage="native_metadata_preflight", npu_work_executed=True)
    save()
    metadata_preflight = baseline.graph_helpers.metadata_call(torch, tensors, cores)
    torch.npu.synchronize()
    inspect_native_plan(metadata_preflight, first, "before_attention_warmup", args, status, save)
    status.update(stage="attention_capture")
    save()
    graph = baseline.SinkGraph(
        torch, "shared", views, tensors, cores, torch.npu.graph_pool_handle(), torch.npu.Stream()
    )
    tracked = {**tensors, **views[0], "metadata": graph.metadata[0], "output": graph.outputs[0]}
    original_addresses = pointers(tracked)
    query_cpu = views[0]["query"].cpu()
    key_cpu = views[0]["key"].cpu().view(views[0]["key"].shape[0], 128, -1)
    value_cpu = views[0]["value"].cpu().view(views[0]["value"].shape[0], 128, -1)
    checks = []
    status.update(stage="replaying", npu_work_executed=True, canonical_digest=schedule["canonical_digest"])
    save()
    for label, record in records:
        q, kv = prepared.prepare(record)
        graph.replay(q, kv)
        graph.synchronize()
        torch.npu.synchronize()
        prepared.complete()
        plan, plan_check = inspect_native_plan(graph.metadata[0], record, label, args, status, save)
        if pointers(tracked) != original_addresses:
            raise AssertionError("fixed-address graph descriptor or output allocation changed")
        if plan[579] != 0 or plan[581] != 0:
            raise AssertionError("native policy fell back to FD scheduling")
        actual = graph.outputs[0].cpu().float()
        if not bool(torch.isfinite(actual).all()):
            raise AssertionError("native no-FD output is nonfinite")
        padding = actual[record["live_tokens"] :]
        if not torch.equal(padding, torch.zeros_like(padding)):
            raise AssertionError("KV0 physical query output was not written as exact zero")
        if not torch.equal(views[0]["query"].cpu(), query_cpu):
            raise AssertionError("attention modified its resident query")
        if tensors["q_lengths"].cpu().tolist() != q or tensors["kv_lengths"].cpu().tolist() != kv:
            raise AssertionError("native execution modified exact length inputs")
        table_cpu = tensors["block_table"].cpu()
        if table_cpu.tolist() != [list(row) for row in record["block_table"]]:
            raise AssertionError("native execution modified owner page routing")
        errors = []
        if record["batch"]:
            for row in sorted({0, record["batch"] // 2, record["batch"] - 1}):
                expected = baseline.validation.cpu_reference(
                    torch,
                    query_cpu[row * 4 : (row + 1) * 4],
                    key_cpu,
                    value_cpu,
                    table_cpu[row : row + 1],
                    [4],
                    [kv[row]],
                    True,
                )
                result = actual[row * 4 : (row + 1) * 4]
                torch.testing.assert_close(
                    result, expected, rtol=baseline.validation.RTOL, atol=baseline.validation.ATOL
                )
                errors.append(dict(row=row, max_abs_error=float((result - expected).abs().max())))
        checks.append(
            dict(
                label=label,
                canonical_step=record["step"],
                live_requests=record["batch"],
                plan=plan_check,
                base_metadata=plan[576:586],
                native_metadata=baseline.validation.metadata_summary(graph.metadata[0].cpu(), cores),
                fp32_rows=errors,
                finite=True,
                exact_zero_padding=True,
                exact_descriptors_read_only=True,
                stable_addresses=True,
                native_plan_fd_zero=True,
            )
        )
        status.update(checks=checks)
        save()
    status.update(
        state="completed",
        returncode=0,
        stage="completed",
        safety_gate_passed=True,
        descriptors=prepared.summary(),
        addresses=original_addresses,
        native_policy_verified_from_device_metadata=True,
        performance_measured=False,
        finished_unix=time.time(),
    )
    save()


def child(args):
    target = args.output_dir / REPORT_NAME
    status = json.loads(target.read_text())
    save = lambda: baseline.isolated.persist(target, status)
    try:
        if os.environ.get("TASK_QUEUE_ENABLE") != "0" or os.environ.get("VLLM_BATCH_INVARIANT") != "0":
            raise RuntimeError("fresh child requires queue0 and batch-invariant0 before Torch import")
        status.update(child_pid=os.getpid(), stage="initialize")
        save()
        torch, cores, handle = baseline.validation.initialize_native(args, status, save)
        with torch.inference_mode():
            run_gate(torch, cores, args, status, save)
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


def launcher(args):
    isolated = baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only an isolated remote /workspace tree is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    library = list(args.opp_root.rglob(AICPU_LIBRARY))
    if len(library) != 1:
        raise RuntimeError("one isolated uniquely named stage-five AICPU library is required")
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
            scope="one occurrence fixed default capacity400: initial/changed/zero-KV/recover; safety only",
            scripts={p.name: isolated.sha256(p) for p in Path(__file__).parent.glob("aclgraph_fia_sink_stage*.py")},
            extension_sha256=isolated.sha256(args.extension),
            opp_artifacts={
                str(p.relative_to(args.opp_root)): isolated.sha256(p) for p in args.opp_root.rglob("*") if p.is_file()
            },
            performance_measured=False,
            full_trace_measured=False,
            full_model_qualified=False,
            end_to_end_gain_established=False,
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
    for name in ("root", "opp_root", "extension", "output_dir", "trace", "selection", "model", "seed"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    with (args.output_dir / "stage5-native-gate.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=args.root,
            env=env,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    status = json.loads(target.read_text())
    if status.get("returncode") != rc:
        status.update(state="failed", returncode=rc, error="child ended before final status")
        isolated.persist(target, status)
    print(json.dumps(dict(status=str(target), state=status["state"], returncode=rc)))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "opp-root", "output-dir", "trace", "selection"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("root", "opp_root", "trace", "selection"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.timeout < 1:
        parser.error("positive timeout required")
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
