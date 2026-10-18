# SPDX-License-Identifier: Apache-2.0
"""Paired preparation experiment using the qualified stage-three attention graphs.

The four cases share the same resident Q/K/V and two graphs per default bucket.
Only descriptor preparation changes. Every real legacy update is retained.
No model weights, native planner changes or full-model speedup are involved.
"""

import argparse
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

CASES = ("sink_fresh", "sink_reuse", "legacy_fresh", "legacy_reuse")
REPORT_NAME = "stage4-preparation.json"


def prepare_case(torch, case, record, tensors, prepared):
    if case.endswith("reuse"):
        return prepared[case.split("_")[0], record["physical_tokens"]].prepare(record)
    return baseline.prepare(torch, tensors, record, upload_lengths=case.startswith("sink"))


def complete_case(case, record, prepared):
    if case.endswith("reuse"):
        prepared[case.split("_")[0], record["physical_tokens"]].complete()


def case_entry(entries, case, capacity):
    return entries[case.split("_")[0], capacity]


def qualify(torch, entries, frames, prepared, records, layers, cores, progress):
    """Compare both preparation implementations and real FIA, plus FP32 rows."""
    chosen = {}
    for record in records:
        chosen.setdefault(record["physical_tokens"], record)
    chosen[-1] = records[-1]
    key, value = layers[0]["key"].cpu(), layers[0]["value"].cpu()
    checks = []
    for record in chosen.values():
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        outputs = {}
        for case in CASES:
            q, kv = prepare_case(torch, case, record, tensors, prepared)
            entry = case_entry(entries, case, capacity)
            entry.replay(q, kv)
            entry.synchronize()
            torch.npu.synchronize()
            complete_case(case, record, prepared)
            outputs[case] = [item.cpu().float() for item in entry.outputs]
            for item in outputs[case]:
                if not bool(torch.isfinite(item).all()):
                    raise AssertionError(f"nonfinite output in {case}")
                tail = item[record["live_tokens"] :]
                if not torch.equal(tail, torch.zeros_like(tail)):
                    raise AssertionError(f"nonzero Q4/KV0 padding in {case}")
        for case in CASES[1:]:
            for actual, expected in zip(outputs[case], outputs["sink_fresh"], strict=True):
                torch.testing.assert_close(
                    actual, expected, rtol=baseline.validation.RTOL, atol=baseline.validation.ATOL
                )
        errors = []
        for row in sorted({0, record["batch"] // 2, record["batch"] - 1}):
            expected = baseline.validation.cpu_reference(
                torch,
                views[0]["query"][row * 4 : (row + 1) * 4].cpu(),
                key.view(key.shape[0], 128, -1),
                value.view(value.shape[0], 128, -1),
                tensors["block_table"][row : row + 1].cpu(),
                [4],
                [record["kv_lengths"][row]],
                True,
            )
            actual = outputs["sink_reuse"][0][row * 4 : (row + 1) * 4]
            torch.testing.assert_close(actual, expected, rtol=baseline.validation.RTOL, atol=baseline.validation.ATOL)
            errors.append(dict(row=row, max_abs_error=float((actual - expected).abs().max())))
        checks.append(
            dict(
                step=record["step"],
                capacity=capacity,
                all_private_occurrences=len(layers),
                controls_close=True,
                zero_padding_exact=True,
                fp32_sampled_rows=errors,
                metadata=baseline.validation.metadata_summary(entries["sink", capacity].metadata[0].cpu(), cores),
            )
        )
        progress(checks)
    return checks


def capture(torch, cores, schedule, records, args, status, save):
    layers, base = baseline.allocate_inputs(torch, schedule, args.occurrences, args.seed)
    status["npu_work_executed"] = True
    required = sorted({row["physical_tokens"] for row in records}, reverse=True)
    frames = {capacity: baseline.frame_inputs(base, layers, capacity) for capacity in required}
    entries, prepared = {}, {}
    pool, stream = torch.npu.graph_pool_handle(), torch.npu.Stream()
    status["memory"] = [baseline.memory(torch, "private_cache_allocated")]
    for capacity in required:
        record = next(row for row in records if row["physical_tokens"] == capacity)
        views, tensors = frames[capacity]
        q, kv = baseline.prepare(torch, tensors, record)
        torch.npu.synchronize()
        for mode in ("sink", "legacy"):
            status.update(stage="capture", capture_capacity=capacity, capture_mode=mode)
            save()
            if mode == "sink":
                entry = baseline.SinkGraph(torch, "shared", views, tensors, cores, pool, stream)
            else:
                entry = baseline.LegacyGraph(
                    torch, views, tensors["block_table"], tensors["atten_mask"], q, kv, pool=pool, stream=stream
                )
            entries[mode, capacity] = entry
            prepared[mode, capacity] = PreparedDescriptors(torch, tensors, record, mode=mode)
            torch.npu.synchronize()
            prepared[mode, capacity].complete()
            status["memory"].append(baseline.memory(torch, f"captured_{mode}_{capacity}"))
            print(f"CAPTURED {mode} capacity={capacity}", flush=True)
        save()
    return layers, frames, entries, prepared


def measure(torch, records, frames, entries, prepared, args, status, save):
    """Interleave paired cases and rotate their order to limit run-order drift."""
    for record in records:
        capacity = record["physical_tokens"]
        for case in CASES:
            entry = case_entry(entries, case, capacity)
            q, kv = prepare_case(torch, case, record, frames[capacity][1], prepared)
            entry.replay(q, kv)
            entry.synchronize()
            torch.npu.synchronize()
            complete_case(case, record, prepared)
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    status["samples"] = []
    for trial in range(args.trials):
        order = CASES if trial % 2 == 0 else tuple(reversed(CASES))
        for record in records:
            offset = (record["step"] + trial) % len(CASES)
            for case in order[offset:] + order[:offset]:
                capacity = record["physical_tokens"]
                entry = case_entry(entries, case, capacity)
                torch.npu.synchronize()
                begin = time.perf_counter_ns()
                q, kv = prepare_case(torch, case, record, frames[capacity][1], prepared)
                uploaded = time.perf_counter_ns()
                start.record(entry.stream)
                entry.replay(q, kv)
                end.record(entry.stream)
                submitted = time.perf_counter_ns()
                entry.synchronize()
                end.synchronize()
                finish = time.perf_counter_ns()
                complete_case(case, record, prepared)
                status["samples"].append(
                    dict(
                        case=case,
                        trial=trial,
                        step=record["step"],
                        batch=record["batch"],
                        capacity=capacity,
                        wall_us=(finish - begin) / 1000,
                        preparation_enqueue_us=(uploaded - begin) / 1000,
                        submit_join_cpu_us=(submitted - uploaded) / 1000,
                        queue_span_us=start.elapsed_time(end) * 1000,
                    )
                )
        status.update(stage="timing", completed_trial=trial)
        save()


def experiment(torch, cores, args, status, save):
    schedule = load_trace_schedule(args.trace, args.selection)
    schedule["core_limit"] = cores
    records = schedule["records"][: args.screen_steps]
    status.update(
        state="running",
        stage="catalog",
        schedule=schedule,
        trials=args.trials,
        screen_steps=args.screen_steps,
        cases=list(CASES),
        occurrences=args.occurrences,
    )
    save()
    status["installed_catalog"] = baseline.installed_catalog(schedule, args.trace, args.model)
    schedule["installed_catalog_revalidated"] = True
    torch.set_num_threads(4)
    layers, frames, entries, prepared = capture(torch, cores, schedule, records, args, status, save)

    def progress(checks):
        status.update(stage="qualification", checks=checks)
        save()

    status["checks"] = qualify(torch, entries, frames, prepared, records, layers, cores, progress)
    measure(torch, records, frames, entries, prepared, args, status, save)

    def post_progress(checks):
        status.update(stage="post_timing_qualification", post_timing_checks=checks)
        save()

    status["post_timing_checks"] = qualify(torch, entries, frames, prepared, records[-1:], layers, cores, post_progress)
    status["prepared_descriptors"] = {
        f"{mode}_{capacity}": item.summary() for (mode, capacity), item in prepared.items()
    }
    status["legacy_update_tasks_per_graph"] = args.occurrences
    status["memory"].append(baseline.memory(torch, "timing_complete"))
    status["final_native_provenance"] = baseline.final_native_provenance(args.opapi_lib)
    status.update(
        state="completed",
        stage="completed",
        returncode=0,
        performance_measured=True,
        finished_unix=time.time(),
        full_model_qualified=False,
        end_to_end_gain_established=False,
        incremental_native_planner_implemented=False,
    )
    save()


def child(args):
    target = args.output_dir / REPORT_NAME
    status = json.loads(target.read_text())

    def save():
        baseline.isolated.persist(target, status)

    try:
        if os.environ.get("TASK_QUEUE_ENABLE") != "0":
            raise RuntimeError("TASK_QUEUE_ENABLE=0 required before Torch import")
        status.update(state="initializing", child_pid=os.getpid())
        save()
        torch, cores, handle = baseline.validation.initialize_native(args, status, save)
        baseline.validation.validate_core_limit(cores)
        with torch.inference_mode():
            experiment(torch, cores, args, status, save)
        if handle is None:
            raise AssertionError("native library owner lost")
        return 0
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        traceback.print_exc()
        return 1


def launcher(args):
    """Run a bounded fresh process while retaining all native and source evidence."""
    isolated = baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only the isolated remote /workspace tree is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build["extension_library"], args.root, file=True)
    if not args.output_dir.is_relative_to(args.root):
        raise ValueError("output directory must be inside isolated root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / REPORT_NAME
    if target.exists():
        raise RuntimeError("preserve previous run; choose a fresh output directory")
    scripts = list(Path(__file__).parent.glob("aclgraph_fia_sink_stage*.py"))
    isolated.persist(
        target,
        dict(
            state="queued",
            started_unix=time.time(),
            scope="isolated attention chain, four paired preparation controls; not full model/serving",
            endpoint=(
                "current host descriptor fill + required H2D + shared plan or 16 true FIA updates + replay + final join"
            ),
            ordering="paired cases interleaved each step; rotate by step/trial; reverse odd-trial order",
            ordering_strategy="step_rotation_reverse_odd_trial_v1",
            qualification_scope=(
                "first occurrence of each default bucket plus final step; all private outputs, "
                "three occurrence0 FP32 rows"
            ),
            scripts={path.name: isolated.sha256(path) for path in scripts},
            extension_sha256=isolated.sha256(args.extension),
            opp_artifacts={
                str(path.relative_to(args.opp_root)): isolated.sha256(path)
                for path in sorted(args.opp_root.rglob("*"))
                if path.is_file()
            },
            performance_measured=False,
            npu_work_executed=False,
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
    for name in (
        "root",
        "opp_root",
        "extension",
        "output_dir",
        "trace",
        "selection",
        "model",
        "occurrences",
        "screen_steps",
        "trials",
        "seed",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    with (args.output_dir / "stage4-preparation.log").open("xb") as log:
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
        status.update(state="failed", returncode=rc, error="child terminated before final status")
        isolated.persist(target, status)
    print(json.dumps(dict(status=str(target), state=status["state"], returncode=rc)))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "opp-root", "output-dir", "trace", "selection"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--occurrences", type=int, choices=(1, 16), default=16)
    parser.add_argument("--screen-steps", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.opp_root = args.root.resolve(strict=True), args.opp_root.resolve(strict=True)
    args.output_dir, args.trace, args.selection = (
        args.output_dir.resolve(),
        args.trace.resolve(strict=True),
        args.selection.resolve(strict=True),
    )
    if not 1 <= args.screen_steps <= 128 or args.trials < 1 or args.timeout < 1:
        parser.error("positive trials/timeout and screen steps in 1..128 required")
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
