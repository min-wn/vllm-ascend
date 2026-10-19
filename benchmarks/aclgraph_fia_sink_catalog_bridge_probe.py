# SPDX-License-Identifier: Apache-2.0
"""Bridge two-bucket and full-catalog capture with identical bounded timings.

Only the first original capacity-400 and capacity-512 records are qualified and
timed. The larger catalog retains all seventeen original first-use graphs but
does not replay its other buckets after capture. Frozen consumer helpers keep
the pinned prepare, native producer, replay and final-join endpoint unchanged.
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

import aclgraph_fia_sink_consumer_probe as consumer

six, seven = consumer.six, consumer.seven
REPORT_NAME = "catalog-bridge-probe.json"
MODES = ("upstream", "whole")
CATALOG_COUNTS = (2, 17)
PREFLIGHT_SOURCE = "executed eager preflight; captured graph producer not yet replayed"
REPLAY_SOURCE = "captured producer replay on selected qualification/timing path"


def catalog_records(schedule, count):
    """Retain Stage7 first-use order without introducing a new capture size."""
    if type(count) is not int or count not in CATALOG_COUNTS:
        raise ValueError("catalog count must be exactly 2 or 17")
    original = seven.trace_records(schedule)
    records = consumer.selected_records(schedule) if count == 2 else seven.capture_records(original)
    if len(records) != count:
        raise ValueError("original first-use capture catalog does not have the requested count")
    return records


def audit_catalog(torch, cores, args, records, frames, entries, status, save, label):
    """Read existing capture state without replaying additional catalog graphs."""
    checks = []
    status[label] = checks
    torch.npu.synchronize()
    for record in records:
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        entry = entries[capacity]
        actual = six.pointers(consumer.tracked_tensors(views, tensors, entry, args.mode))
        if actual != status["captured_addresses"][str(capacity)]:
            raise AssertionError("retained catalog descriptor changed since capture")
        expected = {
            "q_lengths": record["actual_seq_qlen"],
            "kv_lengths": record["kv_lengths"],
            "block_table": record["block_table"],
        }
        for name, value in expected.items():
            if tensors[name].cpu().tolist() != [list(row) if isinstance(row, (list, tuple)) else row for row in value]:
                raise AssertionError(f"retained catalog {name} differs from its exact original record")
        preflight = next(item for item in status["metadata_preflight"] if item["capacity"] == capacity)
        replayed = label == "catalog_post_checks" and capacity in consumer.CAPACITIES
        if replayed:
            plan = seven.current_plan(entry.metadata[0].cpu(), record, args.mode, cores)
            if plan["sha256"] != preflight["sha256"]:
                raise AssertionError("replayed capture metadata differs from the preflight plan")
        else:
            # NPUGraph capture records the native producer without executing it.
            # Unused graph outputs have no initialized semantic plan to read.
            plan = {
                key: value
                for key, value in preflight.items()
                if key not in ("capacity", "record", "verified_before_attention_capture")
            }
        checks.append(
            dict(
                capacity=capacity,
                step=record["step"],
                plan=plan,
                stable_addresses=True,
                exact_descriptors_read_only=True,
                replay_performed=False,
                plan_source=REPLAY_SOURCE if replayed else PREFLIGHT_SOURCE,
                plan_is_captured_graph_output=replayed,
                graph_metadata_semantically_initialized=replayed,
                graph_replays_after_capture=17 if replayed else 0,
            )
        )
        save()
    return checks


def verify_resident_identities(torch, layers, frames, status, *, post=False):
    prefix = "post_" if post else ""
    inputs = six.input_identity(torch, layers)
    if not post and inputs != status["input_identity"]:
        raise AssertionError("catalog capture changed original resident input bytes")
    status[prefix + "input_identity"] = inputs
    status[prefix + "frame_query_identity"] = seven.frame_query_identity(torch, frames)
    first_tensors = next(iter(frames.values()))[1]
    mask = six.tensor_identity(torch, first_tensors["atten_mask"])
    if not post and mask != status["mask_identity"]:
        raise AssertionError("catalog capture changed original causal mask bytes")
    status[prefix + "mask_identity"] = mask
    if not post:
        for capacity in frames:
            expected = [six.tensor_identity(torch, layer["query"][:capacity]) for layer in layers]
            if status["frame_query_identity"][str(capacity)] != expected:
                raise AssertionError("capture modified a retained bucket query clone")
        status["frame_queries_match_base_prefixes"] = True
    else:
        for key in ("input_identity", "frame_query_identity", "mask_identity"):
            if status[key] != status["post_" + key]:
                raise AssertionError(f"bridge changed resident bytes: {key}")


def experiment(torch, cores, args, status, save):
    """Vary retained graph count while reusing the frozen two-record endpoint."""
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError("catalog bridge requires the qualified 20/40 allocation")
    if (args.trial, args.repeats, args.warmups, args.profile_repeats, args.seed) != (0, 10, 5, 0, 1703):
        raise ValueError("the four bridge cells must retain the same fixed timing protocol")
    if args.mode not in MODES:
        raise ValueError("catalog bridge supports only the two native sink policies")
    torch.set_num_threads(4)
    schedule = six.load_trace_schedule(args.trace, args.selection)
    selected = consumer.selected_records(schedule)
    captures = catalog_records(schedule, args.catalog_count)
    status.update(
        canonical_digest=schedule["canonical_digest"],
        selected_records=selected,
        capture_records=captures,
        capture_order=[record["physical_tokens"] for record in captures],
        catalog_count=args.catalog_count,
        stage="capture",
        mode=args.mode,
        policy=args.mode,
        trial=0,
        occurrences=16,
        repeats=10,
        warmup_replays=5,
        warmup_count=5,
        profile_repeats=0,
        profiles=[],
        trace_contract={key: value for key, value in schedule.items() if key != "records"},
        independent_graph_pool_per_bucket=True,
        separate_qualification=True,
        timed_samples_individually_qualified=False,
        additional_catalog_replays_after_capture=0,
        actual_task_audit_scope="sink policies only; frozen consumer timing endpoint unchanged",
    )
    save()
    status["installed_catalog"] = six.baseline.installed_catalog(schedule, args.trace, args.model)
    status["trace_contract"]["installed_catalog_revalidated"] = True
    layers, frames, prepared, entries, audits = consumer.capture(torch, cores, args, schedule, captures, status, save)
    verify_resident_identities(torch, layers, frames, status)
    status.update(npu_work_executed=True, stage="qualification")
    audit_catalog(torch, cores, args, captures, frames, entries, status, save, "catalog_pre_checks")
    consumer.qualify(
        torch, cores, args, selected, layers, frames, prepared, entries, audits, status, save, "pre_checks"
    )
    consumer.measure(torch, args, selected, prepared, entries, audits, status, save)
    consumer.qualify(
        torch, cores, args, selected, layers, frames, prepared, entries, audits, status, save, "post_checks"
    )
    audit_catalog(torch, cores, args, captures, frames, entries, status, save, "catalog_post_checks")
    verify_resident_identities(torch, layers, frames, status, post=True)
    status.update(
        state="completed",
        stage="completed",
        returncode=0,
        performance_measured=True,
        full_model_qualified=False,
        full_trace_measured=False,
        end_to_end_gain_established=False,
        prepared_descriptors={str(cap): source.summary() for cap, source in prepared.items()},
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


def runtime_scripts():
    parent = Path(__file__).parent
    return sorted(
        set(parent.glob("aclgraph_fia_sink_stage*.py"))
        | {
            Path(__file__),
            parent / "aclgraph_fia_sink_consumer_probe.py",
            parent / "aclgraph_fia_sink_catalog_bridge_suite.py",
        }
    )


def launch_child(args, command, env):
    """Own the native process group, including timeout and signal cleanup."""
    with (args.output_dir / "catalog-bridge-probe.log").open("xb") as log:
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
                return process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                six.stop_child(process)
                return 124
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
            six.stop_child(process)


def launcher(args):
    """Retain frozen vendor provenance while owning a new isolated output path."""
    isolated = six.baseline.isolated
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only an isolated /workspace root is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    policy = six.vendor_policy(args.opp_root, args.mode)
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
            mode=args.mode,
            catalog_count=args.catalog_count,
            policy_provenance=policy,
            scope="two original default records timed; two or seventeen first-use graphs retained",
            synthetic_inputs="fixed private Q/K/V; exact lengths and pages from unchanged original first-use records",
            endpoint="frozen consumer prepare + required H2D + real shared producer + replay + final join",
            queue_span_semantics="queue span includes host dependency/submission gaps; not pure kernel duration",
            scripts={path.name: isolated.sha256(path) for path in runtime_scripts()},
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
        "catalog_count",
        "trial",
        "repeats",
        "profile_repeats",
        "warmups",
        "seed",
        "model",
        "timeout",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    rc = launch_child(args, command, env)
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
    parser.add_argument("--catalog-count", type=int, choices=CATALOG_COUNTS, required=True)
    parser.add_argument("--trial", type=int, choices=(0,), default=0)
    parser.add_argument("--repeats", type=int, choices=(10,), default=10)
    parser.add_argument("--profile-repeats", type=int, choices=(0,), default=0)
    parser.add_argument("--warmups", type=int, choices=(5,), default=5)
    parser.add_argument("--seed", type=int, choices=(1703,), default=1703)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("root", "opp_root", "trace", "selection"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
