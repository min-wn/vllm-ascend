# SPDX-License-Identifier: Apache-2.0
"""Compare qualified native policies on the unchanged chronological trace.

Fixed synthetic resident Q/K/V serve current exact lengths and owner pages.
Separate complete pre/post qualification passes audit the replay contract;
individual timed samples are not independently correctness checked. This is a
sequential attention-chain experiment, not token generation or serving.
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

REPORT_NAME = "stage7-dynamic-probe.json"
TRACE_STEPS = 128
SCREEN_STEPS = 8
QUERY_LENGTH = 4
SAVE_INTERVAL = 8


def trace_records(schedule, *, screen=False):
    """Keep original records and chronology; screen is only an eight-step prefix."""
    records = schedule["records"]
    if len(records) != TRACE_STEPS or [record["step"] for record in records] != list(range(TRACE_STEPS)):
        raise ValueError("the unchanged chronological 128-step trace is required")
    capacities = {record["physical_tokens"] for record in records}
    if capacities != set(schedule["required_capture_sizes"]) or not capacities.issubset(
        schedule["default_capture_sizes"]
    ):
        raise ValueError("trace capacities must retain the original default catalog")
    selected = records[:SCREEN_STEPS] if screen else records
    return [dict(record, page_count=schedule["owner_pages"]["page_count"]) for record in selected]


def capture_records(records):
    """First original use initializes each independent bucket graph."""
    first = {}
    for record in records:
        first.setdefault(record["physical_tokens"], record)
    return list(first.values())


def transition(record, previous):
    changed = record["physical_tokens"] != previous["physical_tokens"]
    return dict(
        previous_capacity=previous["physical_tokens"],
        capacity_transition=changed,
        transition_class="pass_start" if record["step"] == 0 else "capacity_transition" if changed else "same_capacity",
    )


def frame_query_identity(torch, frames):
    return {
        str(capacity): [six.tensor_identity(torch, view["query"]) for view in views]
        for capacity, (views, _tensors) in frames.items()
    }


def tracked_tensors(views, tensors, entry):
    tracked = {**tensors, "metadata": entry.metadata[0]}
    tracked.update({f"query{index}": layer["query"] for index, layer in enumerate(views)})
    tracked.update({f"output{index}": output for index, output in enumerate(entry.outputs)})
    return tracked


def compact_reference(torch, layer, query, record, row):
    """Gather only real ordered pages, preserving duplicate IDs and partial tails."""
    page_size = six.baseline.validation.PAGE_SIZE
    kv_length = record["kv_lengths"][row]
    if kv_length < 0:
        raise ValueError("current exact KV length must be nonnegative")
    count = (kv_length + page_size - 1) // page_size
    original_ids = list(record["block_table"][row][:count])
    if len(original_ids) != count or any(page < 0 or page >= layer["key"].shape[0] for page in original_ids):
        raise ValueError("current logical page prefix exceeds the real private cache")
    # index_select runs on the cache device before D2H; do not unique or sort.
    indices = torch.tensor(original_ids, dtype=torch.int64, device=layer["key"].device)
    key = layer["key"].index_select(0, indices).cpu().flatten(start_dim=2)
    value = layer["value"].index_select(0, indices).cpu().flatten(start_dim=2)
    local_ids = list(range(count))
    pages = torch.tensor([local_ids], dtype=torch.int32)
    expected = six.baseline.validation.cpu_reference(
        torch, query.cpu(), key, value, pages, [QUERY_LENGTH], [kv_length], True
    )
    provenance = dict(kv_length=kv_length, original_page_ids=original_ids, local_page_ids=local_ids)
    return expected, provenance


def current_plan(metadata, record, policy, cores):
    evidence = six.plan_evidence(metadata, record, policy, cores)
    if len(evidence["words"]) != 1024:
        raise AssertionError("native plan must retain the complete 1024-word ABI")
    base = evidence["metadata"]["base"]
    expected = dict(
        maxS1=QUERY_LENGTH,
        maxS2=max(record["kv_lengths"]),
        actualQDims=record["physical_batch"],
        actualKVDims=record["physical_batch"],
    )
    if any(base[name] != value for name, value in expected.items()):
        raise AssertionError(f"native plan does not describe this step's exact lengths: {base} != {expected}")
    evidence["current_exact_base_verified"] = True
    return evidence


def qualify(torch, cores, args, records, layers, frames, prepared, entries, status, save, label):
    """Replay every original step; rotate independent FP32 oracle occurrences."""
    checks = []
    status[label] = checks
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        entry, sources = entries[capacity], prepared[capacity]
        if len(views) != args.occurrences or len(entry.outputs) != args.occurrences:
            raise AssertionError("every private attention occurrence must retain one output")
        tracked = tracked_tensors(views, tensors, entry)
        before = six.pointers(tracked)
        if before != status["captured_addresses"][str(capacity)]:
            raise AssertionError("input or graph output allocation changed since capture")
        q, kv = sources.prepare(record)
        entry.replay(q, kv)
        entry.synchronize()
        torch.npu.synchronize()
        sources.complete()
        host_plan = entry.metadata[0].cpu()
        occurrence = record["step"] % args.occurrences
        check = dict(
            capacity=capacity,
            step=record["step"],
            batch=record["batch"],
            private_occurrences=len(views),
            oracle_occurrence=occurrence,
            plan=dict(words=host_plan.tolist(), sha256=hashlib.sha256(host_plan.numpy().tobytes()).hexdigest()),
        )
        checks.append(check)
        # Retain raw plan in the failure report even if structural validation fails.
        check["plan"].update(current_plan(host_plan, record, args.policy, cores))
        host_outputs = [output.cpu() for output in entry.outputs]
        outputs = [output.float() for output in host_outputs]
        for actual in outputs:
            if not bool(torch.isfinite(actual).all()):
                raise AssertionError("native output is nonfinite")
            if not torch.equal(actual[record["live_tokens"] :], torch.zeros_like(actual[record["live_tokens"] :])):
                raise AssertionError("physical KV0 padding was not written as exact zero")
        if six.pointers(tracked) != before:
            raise AssertionError("native execution changed an input or output address")
        if tensors["q_lengths"].cpu().tolist() != q or tensors["kv_lengths"].cpu().tolist() != kv:
            raise AssertionError("native execution modified exact length inputs")
        if tensors["block_table"].cpu().tolist() != [list(row) for row in record["block_table"]]:
            raise AssertionError("native execution modified the current page routing")
        errors = []
        for row in sorted({0, record["batch"] // 2, record["batch"] - 1}):
            query = views[occurrence]["query"][row * QUERY_LENGTH : (row + 1) * QUERY_LENGTH]
            expected, provenance = compact_reference(torch, layers[occurrence], query, record, row)
            actual = outputs[occurrence][row * QUERY_LENGTH : (row + 1) * QUERY_LENGTH]
            torch.testing.assert_close(
                actual, expected, rtol=six.baseline.validation.RTOL, atol=six.baseline.validation.ATOL
            )
            errors.append(
                dict(occurrence=occurrence, row=row, max_abs_error=float((actual - expected).abs().max()), **provenance)
            )
        check.update(
            fp32_sampled_rows=errors,
            output_hashes=[six.tensor_identity(torch, output) for output in host_outputs],
            finite=True,
            all_outputs_finite=True,
            exact_zero_padding=True,
            stable_addresses=True,
            exact_descriptors_read_only=True,
        )
        status.update(stage=label, qualification_completed_step=record["step"])
        if len(checks) % SAVE_INTERVAL == 0 or record is records[-1]:
            save()
    return checks


def warm_trace(torch, records, prepared, entries):
    for record in records:
        capacity = record["physical_tokens"]
        q, kv = prepared[capacity].prepare(record)
        entries[capacity].replay(q, kv)
        entries[capacity].synchronize()
        torch.npu.synchronize()
        prepared[capacity].complete()


def measure(torch, args, records, prepared, entries, status, save):
    status["samples"], status["warmup_passes"] = [], []
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for repeat in range(args.repeats):
        warm_trace(torch, records, prepared, entries)
        status["warmup_passes"].append(dict(repeat=repeat, steps=[r["step"] for r in records], completed=True))
        for index, record in enumerate(records):
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
                    **transition(record, records[index - 1]),
                )
            )
        status.update(stage="timing", completed_repeat=repeat)
        save()


def experiment(torch, cores, args, status, save):
    baseline = six.baseline
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError("native policy comparison requires the qualified 20/40 allocation")
    torch.set_num_threads(4)
    schedule = six.load_trace_schedule(args.trace, args.selection)
    records = trace_records(schedule, screen=args.screen)
    first_records = capture_records(records)
    status.update(
        canonical_digest=schedule["canonical_digest"],
        selected_records=records,
        capture_records=first_records,
        stage="capture",
        occurrences=args.occurrences,
        policy=args.policy,
        trial=args.trial,
        repeats=args.repeats,
        screen_prefix=args.screen,
        trace_contract={key: value for key, value in schedule.items() if key != "records"},
        independent_graph_pool_per_bucket=True,
        separate_qualification=True,
        timed_samples_individually_qualified=False,
    )
    save()
    status["installed_catalog"] = baseline.installed_catalog(schedule, args.trace, args.model)
    status["trace_contract"]["installed_catalog_revalidated"] = True
    layers, frames, prepared, entries = six.capture(torch, cores, args, schedule, first_records, status, save)
    status["frame_query_identity"] = frame_query_identity(torch, frames)
    for capacity in frames:
        expected = [six.tensor_identity(torch, layer["query"][:capacity]) for layer in layers]
        if status["frame_query_identity"][str(capacity)] != expected:
            raise AssertionError("attention warmup/capture changed a bucket query clone")
    status["frame_queries_match_base_prefixes"] = True
    status.update(npu_work_executed=True, stage="qualification")
    qualify(torch, cores, args, records, layers, frames, prepared, entries, status, save, "pre_checks")
    measure(torch, args, records, prepared, entries, status, save)
    qualify(torch, cores, args, records, layers, frames, prepared, entries, status, save, "post_checks")
    status["post_input_identity"] = six.input_identity(torch, layers)
    status["post_frame_query_identity"] = frame_query_identity(torch, frames)
    status["post_mask_identity"] = six.tensor_identity(torch, frames[records[0]["physical_tokens"]][1]["atten_mask"])
    for before, after in (
        ("input_identity", "post_input_identity"),
        ("frame_query_identity", "post_frame_query_identity"),
        ("mask_identity", "post_mask_identity"),
    ):
        if status[before] != status[after]:
            raise AssertionError(f"native dynamic experiment changed resident bytes: {before}")
    status.update(
        state="completed",
        stage="completed",
        returncode=0,
        performance_measured=True,
        full_model_qualified=False,
        full_trace_measured=not args.screen,
        end_to_end_gain_established=False,
        prepared_descriptors={str(capacity): source.summary() for capacity, source in prepared.items()},
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
    isolated = six.baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only an isolated /workspace root is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    policy = six.vendor_policy(args.opp_root, args.policy)
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
            scope=(
                "first eight chronological steps only; one sequential attention occurrence; prefix screen"
                if args.screen
                else "original chronological 128-step trace; 16 sequential private attention occurrences"
            ),
            synthetic_inputs="fixed private Q/K/V; current exact lengths and owner pages follow original trace",
            endpoint="pinned preparation + required H2D + native shared producer + attention replay + final join",
            queue_span_semantics="queue span includes host dependency/submission gaps; not pure kernel duration",
            scripts={p.name: isolated.sha256(p) for p in Path(__file__).parent.glob("aclgraph_fia_sink_stage*.py")},
            extension_sha256=isolated.sha256(args.extension),
            opp_artifacts={
                str(p.relative_to(args.opp_root)): isolated.sha256(p)
                for p in sorted(args.opp_root.rglob("*"))
                if p.is_file()
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
        "policy",
        "trial",
        "repeats",
        "seed",
        "model",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.screen:
        command.append("--screen")
    with (args.output_dir / "stage7-dynamic-probe.log").open("xb") as log:
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
    parser.add_argument("--policy", choices=tuple(six.POLICIES), required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--timeout", type=int, default=2400)
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
