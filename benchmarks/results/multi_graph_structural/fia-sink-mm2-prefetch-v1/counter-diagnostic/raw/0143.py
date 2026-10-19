# SPDX-License-Identifier: Apache-2.0
"""Bounded current-consumer comparison with separately collected device traces.

Every sample includes persistent pinned descriptor preparation, required H2D,
the real producer or legacy host updates, replay and final synchronization.
Device profiles are diagnostic and never included in the performance samples.
The unchanged default capacities 400 and 512 retain their first trace records.
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

import aclgraph_fia_sink_stage6_probe as six
import aclgraph_fia_sink_stage7_probe as seven

REPORT_NAME = "consumer-probe.json"
MODES = ("upstream", "whole", "legacy")
CAPACITIES = (400, 512)
OCCURRENCES = 16
ORACLE_OCCURRENCES = (0, 7, 15)


def selected_records(schedule):
    """Use only the first original records for the two existing default buckets."""
    original = seven.trace_records(schedule)
    records = [next(record for record in original if record["physical_tokens"] == cap) for cap in CAPACITIES]
    if [record["step"] for record in records] != [0, 29]:
        raise ValueError("the original first-use records for capacities 400 and 512 are required")
    return records


def ordered_records(records, trial, repeat):
    offset = (trial + repeat) % len(records)
    return records[offset:] + records[:offset]


class _AuditedTask:
    """Forward the real task, counting successful native apply calls only."""

    def __init__(self, task, audit):
        self.task, self.audit = task, audit

    def __getattr__(self, name):
        return getattr(self.task, name)

    def apply(self, stream):
        result = self.task.apply(stream)
        self.audit.updated_tasks += 1
        return result


class LegacyAudit:
    """Instrument this graph's true update dispatch without replacing operations."""

    def __init__(self, entry):
        self.entry = entry
        self.original_update = entry.graph.update
        self.update_calls, self.updated_tasks = 0, 0
        self.last_query_lengths, self.last_kv_lengths = None, None
        self.set_enabled(True)

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.entry.graph.update = self.update if enabled else self.original_update

    def snapshot(self):
        return self.update_calls, self.updated_tasks

    def update(self, stream, tasks):
        if len(tasks) != len(self.entry.outputs):
            raise AssertionError("each private occurrence must receive one resolved native update task")
        q = list(tasks[0].kwargs["actual_seq_lengths"])
        kv = list(tasks[0].kwargs["actual_seq_lengths_kv"])
        for task in tasks:
            if task.kwargs["actual_seq_lengths"] != q or task.kwargs["actual_seq_lengths_kv"] != kv:
                raise AssertionError("native legacy update tasks disagree on current true host lengths")
        before = self.updated_tasks
        result = self.original_update(stream, tuple(_AuditedTask(task, self) for task in tasks))
        if self.updated_tasks - before != len(tasks):
            raise AssertionError("native update did not apply every real attention task")
        self.update_calls += 1
        self.last_query_lengths, self.last_kv_lengths = q, kv
        return result

    def evidence(self):
        return dict(
            task_count=len(self.entry.tasks),
            workspace_bytes=self.entry.workspace_bytes(),
            current_query_lengths=self.last_query_lengths,
            current_kv_lengths=self.last_kv_lengths,
            update_calls=self.update_calls,
            updated_tasks=self.updated_tasks,
            instrumentation_enabled=self.enabled,
            counter_semantics="successful real graph.update calls and delegated real task.apply calls",
        )


def update_delta(audit, before):
    if audit is not None and not audit.enabled:
        return dict(legacy_update_calls_delta=None, legacy_updated_tasks_delta=None, legacy_counter_instrumented=False)
    calls, tasks = (0, 0) if audit is None else audit.snapshot()
    delta = dict(legacy_update_calls_delta=calls - before[0], legacy_updated_tasks_delta=tasks - before[1])
    expected = (0, 0) if audit is None else (1, len(audit.entry.outputs))
    if (delta["legacy_update_calls_delta"], delta["legacy_updated_tasks_delta"]) != expected:
        raise AssertionError("replay omitted a required real legacy update")
    return delta


def tracked_tensors(views, tensors, entry, mode):
    tracked = dict(tensors)
    if mode != "legacy":
        tracked["metadata"] = entry.metadata[0]
    tracked.update({f"query{index}": view["query"] for index, view in enumerate(views)})
    tracked.update({f"output{index}": output for index, output in enumerate(entry.outputs)})
    if mode == "legacy":
        tracked.update({f"workspace{index}": value for index, value in enumerate(entry.workspaces)})
    return tracked


def capture(torch, cores, args, schedule, records, status, save):
    """Capture each bucket independently using identical private resident inputs."""
    baseline = six.baseline
    layers, base = baseline.allocate_inputs(torch, schedule, OCCURRENCES, args.seed)
    status.update(
        input_identity=six.input_identity(torch, layers),
        private_storage_ranges=six.private_storage_ranges(layers),
        mask_identity=six.tensor_identity(torch, base["atten_mask"]),
        memory=[baseline.memory(torch, "private_inputs_allocated")],
        metadata_preflight=[],
        captured_addresses={},
        legacy_contract={},
    )
    frames, prepared, entries, audits = {}, {}, {}, {}
    stream = torch.npu.Stream()
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = baseline.frame_inputs(base, layers, capacity)
        frames[capacity] = views, tensors
        q, kv = baseline.prepare(torch, tensors, record, upload_lengths=args.mode != "legacy")
        sources = six.PreparedDescriptors(torch, tensors, record, mode="legacy" if args.mode == "legacy" else "sink")
        torch.npu.synchronize()
        sources.complete()
        pool = torch.npu.graph_pool_handle()
        if args.mode == "legacy":
            entry = baseline.LegacyGraph(
                torch, views, tensors["block_table"], tensors["atten_mask"], q, kv, pool=pool, stream=stream
            )
            audits[capacity] = LegacyAudit(entry)
            status["legacy_contract"][str(capacity)] = audits[capacity].evidence()
            module = sys.modules[type(entry.graph).__module__]
            source = Path(module.__file__).resolve()
            snapshot = args.output_dir / "runtime" / "updatable_graph.py"
            if not snapshot.exists():
                snapshot.parent.mkdir(parents=True)
                snapshot.write_bytes(source.read_bytes())
            status.setdefault("runtime_sources", {})["vllm_ascend/compilation/updatable_graph.py"] = dict(
                path=str(source),
                sha256=baseline.isolated.sha256(source),
                snapshot_path=snapshot.relative_to(args.output_dir).as_posix(),
                snapshot_sha256=baseline.isolated.sha256(snapshot),
            )
            if baseline.isolated.sha256(source) != baseline.isolated.sha256(snapshot):
                raise AssertionError("actual native-update Python source changed since its retained snapshot")
        else:
            metadata = baseline.graph_helpers.metadata_call(torch, tensors, cores)
            torch.npu.synchronize()
            evidence = dict(
                capacity=capacity, record=record, **seven.current_plan(metadata.cpu(), record, args.mode, cores)
            )
            evidence["verified_before_attention_capture"] = True
            status["metadata_preflight"].append(evidence)
            save()
            entry = baseline.SinkGraph(torch, "shared", views, tensors, cores, pool, stream)
            audits[capacity] = None
        status["captured_addresses"][str(capacity)] = six.pointers(tracked_tensors(views, tensors, entry, args.mode))
        status["memory"].append(baseline.memory(torch, f"captured_{capacity}"))
        prepared[capacity], entries[capacity] = sources, entry
    return layers, frames, prepared, entries, audits


def replay(torch, record, source, entry, audit):
    before = (0, 0) if audit is None else audit.snapshot()
    q, kv = source.prepare(record)
    entry.replay(q, kv)
    entry.synchronize()
    torch.npu.synchronize()
    source.complete()
    return q, kv, update_delta(audit, before)


def qualify(torch, cores, args, records, layers, frames, prepared, entries, audits, status, save, label):
    """Check all private outputs and nine ordered-page FP32 samples per bucket."""
    checks = []
    status[label] = checks
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        entry, audit = entries[capacity], audits[capacity]
        if audit is not None:
            audit.set_enabled(True)
        tracked = tracked_tensors(views, tensors, entry, args.mode)
        before = six.pointers(tracked)
        if before != status["captured_addresses"][str(capacity)]:
            raise AssertionError("capture allocations changed before qualification")
        q, kv, delta = replay(torch, record, prepared[capacity], entry, audit)
        check = dict(capacity=capacity, step=record["step"], private_occurrences=len(views), **delta)
        checks.append(check)
        if audit is None:
            # Keep raw complete metadata evidence before any structural assertion.
            host_plan = entry.metadata[0].cpu()
            check["plan"] = dict(
                words=host_plan.tolist(), sha256=hashlib.sha256(host_plan.numpy().tobytes()).hexdigest()
            )
            save()
            check["plan"].update(seven.current_plan(host_plan, record, args.mode, cores))
        else:
            check["legacy"] = audit.evidence()
            if audit.last_query_lengths != q or audit.last_kv_lengths != kv:
                raise AssertionError("legacy update did not consume the current true host lengths")
            check["current_host_lengths_verified"] = True
        host_outputs = [output.cpu() for output in entry.outputs]
        outputs = [output.float() for output in host_outputs]
        if len(outputs) != OCCURRENCES:
            raise AssertionError("all 16 private outputs must be checked")
        for actual in outputs:
            if not bool(torch.isfinite(actual).all()):
                raise AssertionError("native output is nonfinite")
            if not torch.equal(actual[record["live_tokens"] :], torch.zeros_like(actual[record["live_tokens"] :])):
                raise AssertionError("physical KV0 padding is not exact zero")
        if six.pointers(tracked) != before:
            raise AssertionError("native execution changed an input/output/workspace allocation")
        if tensors["block_table"].cpu().tolist() != [list(row) for row in record["block_table"]]:
            raise AssertionError("native execution modified current page routing")
        if audit is None and (tensors["q_lengths"].cpu().tolist() != q or tensors["kv_lengths"].cpu().tolist() != kv):
            raise AssertionError("native consumer modified exact device length inputs")
        errors = []
        for occurrence in ORACLE_OCCURRENCES:
            for row in (0, record["batch"] // 2, record["batch"] - 1):
                query = views[occurrence]["query"][row * 4 : (row + 1) * 4]
                expected, provenance = seven.compact_reference(torch, layers[occurrence], query, record, row)
                actual = outputs[occurrence][row * 4 : (row + 1) * 4]
                torch.testing.assert_close(
                    actual, expected, rtol=six.baseline.validation.RTOL, atol=six.baseline.validation.ATOL
                )
                errors.append(
                    dict(
                        occurrence=occurrence,
                        row=row,
                        max_abs_error=float((actual - expected).abs().max()),
                        **provenance,
                    )
                )
        check.update(
            fp32_sampled_rows=errors,
            output_hashes=[six.tensor_identity(torch, output) for output in host_outputs],
            finite=True,
            all_outputs_finite=True,
            exact_zero_padding=True,
            stable_addresses=True,
            exact_descriptors_read_only=True,
            descriptor_scope=(
                "legacy true host lengths and device block_table"
                if audit is not None
                else "sink true device lengths and device block_table"
            ),
        )
        status.update(stage=label)
        save()
    return checks


def measure(torch, args, records, prepared, entries, audits, status, save):
    status["samples"], status["warmups"] = [], []
    for audit in audits.values():
        if audit is not None:
            audit.set_enabled(False)
    for record in records:
        capacity = record["physical_tokens"]
        for _ in range(args.warmups):
            replay(torch, record, prepared[capacity], entries[capacity], audits[capacity])
        status["warmups"].append(dict(capacity=capacity, replays=args.warmups, outside_timing=True))
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for repeat in range(args.repeats):
        for record in ordered_records(records, args.trial, repeat):
            capacity = record["physical_tokens"]
            entry, audit = entries[capacity], audits[capacity]
            torch.npu.synchronize()
            before = (0, 0) if audit is None else audit.snapshot()
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
                    mode=args.mode,
                    policy=args.mode,
                    repeat=repeat,
                    step=record["step"],
                    batch=record["batch"],
                    capacity=capacity,
                    wall_us=(finish - begin) / 1000,
                    preparation_enqueue_us=(ready - begin) / 1000,
                    submit_join_cpu_us=(submitted - ready) / 1000,
                    queue_span_us=start.elapsed_time(end) * 1000,
                    **update_delta(audit, before),
                )
            )
        status.update(stage="timing", completed_repeat=repeat)
        save()


def profiles(torch, args, records, prepared, entries, audits, status, save):
    """Collect actual prepared replay traces after all formal timing samples."""
    import torch_npu

    status["profiles"] = []
    for record in records:
        capacity = record["physical_tokens"]
        target = args.output_dir / "profiles" / f"cap{capacity}" / "trace.json"
        target.parent.mkdir(parents=True)
        audit = audits[capacity]
        if audit is not None:
            audit.set_enabled(False)
        labels = [f"consumer/{args.mode}/cap{capacity}/repeat{index}" for index in range(args.profile_repeats)]
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target.parent / "counters")),
        ) as profiler:
            for label in labels:
                # Keep preparation DMA outside replay ranges so update DMA can
                # be attributed independently. Formal wall includes this work.
                q, kv = prepared[capacity].prepare(record)
                torch.npu.synchronize()
                with torch.profiler.record_function(label):
                    entries[capacity].replay(q, kv)
                    entries[capacity].synchronize()
                    torch.npu.synchronize()
                prepared[capacity].complete()
        profiler.export_chrome_trace(str(target))
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("profiler did not export the real CPU/NPU trace")
        status["profiles"].append(
            dict(
                capacity=capacity,
                step=record["step"],
                path=target.relative_to(args.output_dir).as_posix(),
                bytes=target.stat().st_size,
                sha256=six.baseline.isolated.sha256(target),
                repeats=args.profile_repeats,
                labels=labels,
                outside_timing=True,
                legacy_update_calls_delta=None if audit is not None else 0,
                legacy_updated_tasks_delta=None if audit is not None else 0,
                legacy_counter_instrumented=False,
                label_scope="native plan/update + replay + final join; prepare/H2D completed before label",
                preparation_in_trace_outside_label=True,
            )
        )
        status.update(stage="profiling")
        save()


def experiment(torch, cores, args, status, save):
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError("consumer comparison requires the qualified 20/40 allocation")
    torch.set_num_threads(4)
    schedule = six.load_trace_schedule(args.trace, args.selection)
    records = selected_records(schedule)
    status.update(
        canonical_digest=schedule["canonical_digest"],
        selected_records=records,
        stage="capture",
        mode=args.mode,
        policy=args.mode,
        trial=args.trial,
        occurrences=OCCURRENCES,
        repeats=args.repeats,
        profile_repeats=args.profile_repeats,
        warmup_replays=args.warmups,
        warmup_count=args.warmups,
        trace_contract={key: value for key, value in schedule.items() if key != "records"},
        independent_graph_pool_per_bucket=True,
        separate_qualification=True,
        timed_samples_individually_qualified=False,
        actual_task_audit_scope="pre/post qualification only; timing and profiles use original graph.update",
    )
    save()
    status["installed_catalog"] = six.baseline.installed_catalog(schedule, args.trace, args.model)
    status["trace_contract"]["installed_catalog_revalidated"] = True
    layers, frames, prepared, entries, audits = capture(torch, cores, args, schedule, records, status, save)
    status["frame_query_identity"] = seven.frame_query_identity(torch, frames)
    for capacity in frames:
        expected = [six.tensor_identity(torch, layer["query"][:capacity]) for layer in layers]
        if status["frame_query_identity"][str(capacity)] != expected:
            raise AssertionError("capture modified a bucket's private query clone")
    status.update(frame_queries_match_base_prefixes=True, npu_work_executed=True)
    qualify(torch, cores, args, records, layers, frames, prepared, entries, audits, status, save, "pre_checks")
    measure(torch, args, records, prepared, entries, audits, status, save)
    profiles(torch, args, records, prepared, entries, audits, status, save)
    qualify(torch, cores, args, records, layers, frames, prepared, entries, audits, status, save, "post_checks")
    status["post_input_identity"] = six.input_identity(torch, layers)
    status["post_frame_query_identity"] = seven.frame_query_identity(torch, frames)
    status["post_mask_identity"] = six.tensor_identity(torch, frames[records[0]["physical_tokens"]][1]["atten_mask"])
    for before, after in (
        ("input_identity", "post_input_identity"),
        ("frame_query_identity", "post_frame_query_identity"),
        ("mask_identity", "post_mask_identity"),
    ):
        if status[before] != status[after]:
            raise AssertionError(f"native experiment changed resident bytes: {before}")
    status.update(
        state="completed",
        stage="completed",
        returncode=0,
        performance_measured=True,
        full_model_qualified=False,
        full_trace_measured=False,
        end_to_end_gain_established=False,
        prepared_descriptors={str(cap): source.summary() for cap, source in prepared.items()},
        legacy_contract={str(cap): audit.evidence() for cap, audit in audits.items() if audit is not None},
        finished_unix=time.time(),
    )
    save()


def child(args):
    target = args.output_dir / REPORT_NAME
    status = json.loads(target.read_text())
    save = lambda: six.baseline.isolated.persist(target, status)
    try:
        if os.environ.get("TASK_QUEUE_ENABLE") != "0" or os.environ.get("VLLM_BATCH_INVARIANT") != "0":
            raise RuntimeError("fresh child requires queue0/batch-invariant0 before Torch import")
        status.update(child_pid=os.getpid(), state="running")
        save()
        torch, cores, handle = six.baseline.validation.initialize_native(args, status, save)
        with torch.inference_mode():
            experiment(torch, cores, args, status, save)
        if handle is None:
            raise AssertionError("native library owner lost")
        status["final_native_provenance"] = six.baseline.final_native_provenance(args.opapi_lib)
        save()
        return 0
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        traceback.print_exc()
        return 1


def launcher(args):
    """Enforce isolated fresh-process ownership and stop its native child on exit."""
    isolated = six.baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only an isolated /workspace root is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    policy = six.vendor_policy(args.opp_root, "upstream" if args.mode == "legacy" else args.mode)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build["extension_library"], args.root, file=True)
    if not args.output_dir.is_relative_to(args.root) or args.output_dir.exists():
        raise ValueError("choose a new output directory inside the isolated root")
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / REPORT_NAME
    scripts = list(Path(__file__).parent.glob("aclgraph_fia_sink_stage*.py")) + [
        Path(__file__),
        Path(__file__).with_name("aclgraph_fia_sink_consumer_suite.py"),
    ]
    isolated.persist(
        target,
        dict(
            state="queued",
            started_unix=time.time(),
            mode=args.mode,
            policy_provenance=policy,
            scope="two original default buckets; 16 sequential private synthetic attention occurrences",
            synthetic_inputs="fixed private Q/K/V; exact lengths and owner pages from original first-use records",
            endpoint="pinned preparation + required H2D + producer or true legacy updates + replay + final join",
            queue_span_semantics="queue span includes host dependency/submission gaps; not pure kernel duration",
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
            full_trace_measured=False,
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
        "mode",
        "trial",
        "repeats",
        "profile_repeats",
        "warmups",
        "seed",
        "model",
        "timeout",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    with (args.output_dir / "consumer-probe.log").open("xb") as log:
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
            six.stop_child(process)
            raise SystemExit(128 + number)

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[number] = signal.signal(number, forward_signal)
            try:
                rc = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                six.stop_child(process)
                rc = 124
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
            six.stop_child(process)
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
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("root", "opp_root", "trace", "selection"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if min(args.repeats, args.profile_repeats, args.warmups, args.timeout) < 1 or args.trial < 0:
        parser.error("positive repeats/profile-repeats/warmups/timeout and nonnegative trial required")
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
