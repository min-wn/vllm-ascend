# SPDX-License-Identifier: Apache-2.0
"""Bounded whole-head attention-plan diagnostic with an unchanged consumer.

Four representative records use only their archived default capture buckets.
Resident private Q/K/V and inner tiling are identical across controls. Oracle
controls exclude required planning and are diagnostics, not dynamic algorithms.
No full-model, serving or incremental-device-planner improvement is established.
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
from aclgraph_fia_sink_stage4_whole_plan import (
    preflight_whole_plan,
    validate_whole_plan,
    whole_plan,
    whole_plan_fast,
)

CASES = ("native_shared", "native_oracle", "whole_cpu", "whole_oracle")
REPRESENTATIVE_CAPACITIES = (256, 400, 496, 512)
REPORT_NAME = "stage4-whole-plan.json"
METADATA_WORDS = 1024


def representative_records(schedule, *, screen=False):
    capacities = (400,) if screen else REPRESENTATIVE_CAPACITIES
    selected = []
    for capacity in capacities:
        matches = [row for row in schedule["records"] if row["physical_tokens"] == capacity]
        if not matches:
            raise ValueError(f"canonical trace has no record at default capacity {capacity}")
        record = matches[0]
        page_count = schedule["owner_pages"]["page_count"]
        page_width = schedule["owner_pages"]["width"]
        if len(record["block_table"]) != record["physical_batch"]:
            raise ValueError("representative record page table does not cover its physical Q4 bucket")
        if any(len(row) != page_width for row in record["block_table"]):
            raise ValueError("representative record page table width differs from the fixed owner reservation")
        for row in record["block_table"]:
            if any(type(page) is not int or not 0 <= page < page_count for page in row):
                raise ValueError("representative page table contains an invalid physical page ID")
        # Carry trace-owned page-count provenance into the CPU plan validator;
        # the native metadata table may use zero only for padding sentinels.
        selected.append(dict(record, page_count=page_count))
    return tuple(selected)


class WholeGraph(baseline.SinkGraph):
    """Capture attention with an independently allocated exact metadata payload."""

    def __init__(self, torch, layers, tensors, cores, pool, stream):
        self.torch, self.mode, self.layers, self.tensors, self.cores = torch, "attention_only", layers, tensors, cores
        self.stream, self.completion = stream, torch.npu.Event()
        self.graph, self.outputs, self.metadata = torch.npu.NPUGraph(), [], []
        self.native_template_owner = baseline.graph_helpers.metadata_call(torch, tensors, cores)
        self.external_metadata = self.native_template_owner.clone()
        torch.npu.synchronize()
        if self.external_metadata.data_ptr() == self.native_template_owner.data_ptr():
            raise AssertionError("whole-plan payload aliases the native template allocation")
        self.template_metadata = self.native_template_owner.cpu().tolist()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            warm = self.operations()
        torch.npu.synchronize()
        with torch.npu.graph(self.graph, pool=pool, stream=stream):
            self.metadata, self.outputs = self.operations()
        torch.npu.synchronize()
        del warm


class WholePlanSource:
    """Retain one pinned plan source through every copy and consumer join."""

    def __init__(self, torch, entry, record, cores):
        self.entry, self.cores = entry, cores["cube_core_num"]
        self.host = torch.empty((METADATA_WORDS,), dtype=torch.int32, device="cpu", pin_memory=True)
        if not self.host.is_pinned():
            raise RuntimeError("whole-plan payload source is not page-locked")
        self.view = self.host.numpy()
        self.host_pointer, self.device_pointer = self.host.data_ptr(), entry.external_metadata.data_ptr()
        self.inflight, self.copy_count = False, 0
        self.preflight = preflight_whole_plan(record, entry.template_metadata, self.cores)
        self.first_validation = validate_whole_plan(record, whole_plan(record, entry.template_metadata, self.cores))

    def prepare(self, record):
        if self.inflight:
            raise RuntimeError("whole-plan source is still in use; join before preparing another plan")
        words = whole_plan_fast(record, self.entry.template_metadata, self.cores, preflight=self.preflight)
        self.view[:] = words
        self.inflight = True
        self.entry.external_metadata.copy_(self.host, non_blocking=True)
        self.copy_count += 1

    def complete(self):
        self.inflight = False

    def summary(self):
        if self.inflight:
            raise AssertionError("whole-plan source retained unfinished work at final reporting")
        return dict(
            host_pointer=self.host_pointer,
            device_pointer=self.device_pointer,
            host_pinned=self.host.is_pinned(),
            host_storage_bytes=self.host.untyped_storage().nbytes(),
            device_storage_bytes=self.entry.external_metadata.untyped_storage().nbytes(),
            native_template_pointer=self.entry.native_template_owner.data_ptr(),
            template_sha256=hashlib.sha256(self.entry.native_template_owner.cpu().numpy().tobytes()).hexdigest(),
            plan_sha256=hashlib.sha256(self.view.tobytes()).hexdigest(),
            stable_host_pointer=self.host.data_ptr() == self.host_pointer,
            stable_device_pointer=self.entry.external_metadata.data_ptr() == self.device_pointer,
            async_plan_copies=self.copy_count,
            first_plan_validation=self.first_validation,
        )


def entry_for(entries, case, capacity):
    return entries["whole" if case.startswith("whole") else case, capacity]


def prepare_case(torch, case, record, entries, prepared, sources, *, oracle_setup=False):
    capacity = record["physical_tokens"]
    q, kv = prepared[capacity].prepare(record)
    if case == "whole_cpu" or case == "whole_oracle" and oracle_setup:
        sources[capacity].prepare(record)
    if case == "native_oracle" and oracle_setup:
        # Descriptors must be current before building the oracle metadata.
        torch.npu.synchronize()
        entry_for(entries, case, capacity).prepare_oracle()
    return q, kv


def complete_case(record, prepared, sources):
    capacity = record["physical_tokens"]
    prepared[capacity].complete()
    sources[capacity].complete()


def oracle_setup(torch, case, record, entries, prepared, sources):
    if case.endswith("oracle"):
        prepare_case(torch, case, record, entries, prepared, sources, oracle_setup=True)
        torch.npu.synchronize()
        complete_case(record, prepared, sources)


def capture(torch, cores, schedule, records, args, status, save):
    layers, base = baseline.allocate_inputs(torch, schedule, args.occurrences, args.seed)
    status["npu_work_executed"] = True
    frames, entries, prepared, sources = {}, {}, {}, {}
    pool, stream = torch.npu.graph_pool_handle(), torch.npu.Stream()
    status["memory"] = [baseline.memory(torch, "private_cache_allocated")]
    for record in reversed(records):
        capacity = record["physical_tokens"]
        views, tensors = baseline.frame_inputs(base, layers, capacity)
        frames[capacity] = views, tensors
        baseline.prepare(torch, tensors, record)
        torch.npu.synchronize()
        prepared[capacity] = PreparedDescriptors(torch, tensors, record, mode="sink")
        torch.npu.synchronize()
        prepared[capacity].complete()
        for name, mode in (("native_shared", "shared"), ("native_oracle", "attention_only"), ("whole", None)):
            status.update(stage="capture", capture_capacity=capacity, capture_mode=name)
            save()
            entry = (
                WholeGraph(torch, views, tensors, cores, pool, stream)
                if mode is None
                else baseline.SinkGraph(torch, mode, views, tensors, cores, pool, stream)
            )
            entries[name, capacity] = entry
            status["memory"].append(baseline.memory(torch, f"captured_{name}_{capacity}"))
            print(f"CAPTURED {name} capacity={capacity}", flush=True)
        sources[capacity] = WholePlanSource(torch, entries["whole", capacity], record, cores)
    return layers, frames, entries, prepared, sources


def qualify(torch, records, layers, frames, entries, prepared, sources, progress):
    """Compare all private outputs and three FP32 rows for each fixed record."""
    key, value = layers[0]["key"].cpu(), layers[0]["value"].cpu()
    checks = []
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        before_queries = [layer["query"].cpu().clone() for layer in views]
        outputs = {}
        for case in CASES:
            oracle_setup(torch, case, record, entries, prepared, sources)
            q, kv = prepare_case(torch, case, record, entries, prepared, sources)
            entry = entry_for(entries, case, capacity)
            entry.replay(q, kv)
            entry.synchronize()
            torch.npu.synchronize()
            complete_case(record, prepared, sources)
            outputs[case] = [item.cpu().float() for item in entry.outputs]
            for output in outputs[case]:
                if not bool(torch.isfinite(output).all()):
                    raise AssertionError(f"nonfinite output for {case}")
                tail = output[record["live_tokens"] :]
                if not torch.equal(tail, torch.zeros_like(tail)):
                    raise AssertionError(f"nonzero Q4/KV0 physical padding for {case}")
        for case in CASES[1:]:
            for actual, expected in zip(outputs[case], outputs["native_shared"], strict=True):
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
            for case in CASES:
                actual = outputs[case][0][row * 4 : (row + 1) * 4]
                torch.testing.assert_close(
                    actual, expected, rtol=baseline.validation.RTOL, atol=baseline.validation.ATOL
                )
                errors.append(dict(case=case, row=row, max_abs_error=float((actual - expected).abs().max())))
        if any(
            not torch.equal(layer["query"].cpu(), before) for layer, before in zip(views, before_queries, strict=True)
        ):
            raise AssertionError("read-only resident query was modified")
        if tensors["q_lengths"].cpu().tolist() != list(record["actual_seq_qlen"]) or tensors[
            "kv_lengths"
        ].cpu().tolist() != list(record["kv_lengths"]):
            raise AssertionError("native execution modified exact length descriptors")
        if tensors["block_table"].cpu().tolist() != [list(row) for row in record["block_table"]]:
            raise AssertionError("native execution modified the owner page routing")
        checks.append(
            dict(
                step=record["step"],
                capacity=capacity,
                controls_close=True,
                private_occurrences=len(layers),
                zero_padding_exact=True,
                resident_queries_unchanged=True,
                exact_descriptors_unchanged=True,
                fp32_sampled_rows=errors,
                whole_plan=validate_whole_plan(
                    record, entries["whole", capacity].external_metadata.cpu().tolist(), cores=20
                ),
            )
        )
        progress(checks)
    return checks


def measure(torch, records, entries, prepared, sources, args, status, save):
    for record in records:
        for case in CASES:
            oracle_setup(torch, case, record, entries, prepared, sources)
            q, kv = prepare_case(torch, case, record, entries, prepared, sources)
            entry = entry_for(entries, case, record["physical_tokens"])
            entry.replay(q, kv)
            entry.synchronize()
            torch.npu.synchronize()
            complete_case(record, prepared, sources)
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    status["samples"] = []
    for trial in range(args.trials):
        order = CASES if trial % 2 == 0 else tuple(reversed(CASES))
        for record in records:
            capacity = record["physical_tokens"]
            for repeat in range(args.repeats):
                offset = (record["step"] + trial + repeat) % len(CASES)
                for case in order[offset:] + order[:offset]:
                    entry = entry_for(entries, case, capacity)
                    oracle_setup(torch, case, record, entries, prepared, sources)
                    torch.npu.synchronize()
                    begin = time.perf_counter_ns()
                    q, kv = prepare_case(torch, case, record, entries, prepared, sources)
                    prepared_at = time.perf_counter_ns()
                    start.record(entry.stream)
                    entry.replay(q, kv)
                    end.record(entry.stream)
                    submitted = time.perf_counter_ns()
                    entry.synchronize()
                    end.synchronize()
                    finish = time.perf_counter_ns()
                    complete_case(record, prepared, sources)
                    status["samples"].append(
                        dict(
                            case=case,
                            trial=trial,
                            repeat=repeat,
                            step=record["step"],
                            batch=record["batch"],
                            capacity=capacity,
                            wall_us=(finish - begin) / 1000,
                            preparation_enqueue_us=(prepared_at - begin) / 1000,
                            submit_join_cpu_us=(submitted - prepared_at) / 1000,
                            queue_span_us=start.elapsed_time(end) * 1000,
                            excluded_oracle_planning=case.endswith("oracle"),
                        )
                    )
        status.update(stage="timing", completed_trial=trial)
        save()


def experiment(torch, cores, args, status, save):
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError("whole-plan diagnostic requires the qualified 20/40 core allocation")
    schedule = load_trace_schedule(args.trace, args.selection)
    records = representative_records(schedule, screen=args.screen)
    status.update(
        state="running",
        stage="catalog",
        schedule=schedule,
        selected_records=[{key: record[key] for key in ("step", "batch", "physical_tokens")} for record in records],
        cases=list(CASES),
        occurrences=args.occurrences,
        trials=args.trials,
        repeats=args.repeats,
        screen=args.screen,
        expected_samples=len(CASES) * len(records) * args.trials * args.repeats,
    )
    save()
    status["installed_catalog"] = baseline.installed_catalog(schedule, args.trace, args.model)
    schedule["installed_catalog_revalidated"] = True
    torch.set_num_threads(4)
    layers, frames, entries, prepared, sources = capture(torch, cores, schedule, records, args, status, save)

    def progress(checks):
        status.update(stage="qualification", checks=checks)
        save()

    status["checks"] = qualify(torch, records, layers, frames, entries, prepared, sources, progress)
    measure(torch, records, entries, prepared, sources, args, status, save)

    def post_progress(checks):
        status.update(stage="post_timing_qualification", post_timing_checks=checks)
        save()

    status["post_timing_checks"] = qualify(
        torch, records[-1:], layers, frames, entries, prepared, sources, post_progress
    )
    status["prepared_descriptors"] = {str(capacity): item.summary() for capacity, item in prepared.items()}
    status["whole_plan_sources"] = {str(capacity): item.summary() for capacity, item in sources.items()}
    status["memory"].append(baseline.memory(torch, "timing_complete"))
    status["final_native_provenance"] = baseline.final_native_provenance(args.opapi_lib)
    if len(status["samples"]) != status["expected_samples"]:
        raise AssertionError("bounded diagnostic omitted planned timing observations")
    status.update(
        state="completed",
        stage="completed",
        returncode=0,
        performance_measured=True,
        finished_unix=time.time(),
        full_trace_measured=False,
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
    """Use a fresh device-zero child and preserve every previous run artifact."""
    isolated = baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only the isolated remote /workspace tree is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build["extension_library"], args.root, file=True)
    if not args.output_dir.is_relative_to(args.root):
        raise ValueError("output directory must be inside the isolated root")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("preserve previous run; choose an empty fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / REPORT_NAME
    isolated.persist(
        target,
        dict(
            state="queued",
            started_unix=time.time(),
            scope="Four fixed canonical records, isolated attention chain; not full trace/model/serving",
            endpoint="current pinned descriptors + required CPU whole plan and H2D or shared native plan + replay/join",
            oracle_diagnostic="Oracle cases exclude fresh planning and its copy; they are not dynamic algorithms.",
            qualification_scope="all private outputs on all four selected records; three occurrence0 FP32 rows",
            scripts={
                path.name: isolated.sha256(path) for path in Path(__file__).parent.glob("aclgraph_fia_sink_stage*.py")
            },
            extension_sha256=isolated.sha256(args.extension),
            opp_artifacts={
                str(path.relative_to(args.opp_root)): isolated.sha256(path)
                for path in sorted(args.opp_root.rglob("*"))
                if path.is_file()
            },
            performance_measured=False,
            npu_work_executed=False,
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
    for name in (
        "root",
        "opp_root",
        "extension",
        "output_dir",
        "trace",
        "selection",
        "model",
        "trials",
        "repeats",
        "seed",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.screen:
        command.append("--screen")
    with (args.output_dir / "stage4-whole-plan.log").open("xb") as log:
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
    parser.add_argument(
        "--screen", action="store_true", help="single occurrence, capacity400, one trial/repeat safety gate"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
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
    if min(args.trials, args.repeats, args.timeout) < 1:
        parser.error("positive trials/repeats/timeout required")
    args.occurrences = 1 if args.screen else 16
    if args.screen:
        args.trials = args.repeats = 1
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
