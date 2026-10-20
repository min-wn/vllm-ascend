# SPDX-License-Identifier: Apache-2.0
"""Compare native producers over chronological descriptor traces in fresh processes."""

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

from aclgraph_fia_sink_stage3_trace import load_trace_schedule
from aclgraph_fia_sink_stage6_suite import owned, process_orders

POLICIES = ("upstream", "whole")
REPORT_NAME = "stage7-suite.json"
PROBE_REPORT_NAME = "stage7-dynamic-probe.json"


def design_for(schedule, args):
    """Fix sample counts and scope before any native child starts."""
    records = schedule["records"][:8] if args.screen else schedule["records"]
    rounds, repeats = (1, 1) if args.screen else (args.rounds, args.repeats)
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("repeats must be an exact integer in 1..10")
    return dict(
        policies=list(POLICIES),
        capacities=sorted({record["physical_tokens"] for record in records}),
        occurrences=1 if args.screen else 16,
        rounds=rounds,
        repeats=repeats,
        step_count=len(records),
        process_orders=process_orders(rounds),
        screen=args.screen,
        seed=args.seed,
        step_order="unchanged chronological canonical prefix"
        if args.screen
        else "unchanged chronological canonical trace",
        qualification_passes=2,
        reference_rows_per_step=3,
        reference_occurrence="step % occurrences",
        warmup_passes_per_repeat=1,
    )


def run_one(command, log_path, timeout, entry, save, *, runner=subprocess.Popen):
    """Own a launcher group and forward cancellation until its child joins."""
    with log_path.open("xb") as log:
        process = runner(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        entry["pid"] = process.pid
        save()

        def stop():
            if process.poll() is not None:
                return
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

        def forward(number, _frame):
            stop()
            raise SystemExit(128 + number)

        previous = {}
        try:
            if os.name == "posix":
                for number in (signal.SIGTERM, signal.SIGINT):
                    previous[number] = signal.signal(number, forward)
            return process.wait(timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            stop()
            raise RuntimeError("probe launcher exceeded its process deadline") from None
        finally:
            stop()
            for number, handler in previous.items():
                signal.signal(number, handler)


def run_suite(args, *, runner=subprocess.Popen):
    """Run all process blocks serially and preserve byte-exact relative reports."""
    root = args.root.resolve(strict=True)
    output = owned(args.output_dir, root, exists=False)
    probe = owned(args.probe, root)
    vendors = {name: owned(getattr(args, name + "_vendor"), root) for name in POLICIES}
    if vendors["upstream"] == vendors["whole"]:
        raise ValueError("policies require distinct isolated vendors")
    if output.exists():
        raise ValueError("choose a fresh output directory to preserve earlier evidence")
    if args.timeout < 1:
        raise ValueError("timeout must be positive")
    schedule = load_trace_schedule(args.trace, args.selection)
    design = design_for(schedule, args)
    output.mkdir(parents=True)
    target = output / REPORT_NAME
    status = dict(
        state="running",
        started_unix=time.time(),
        design=design,
        canonical_digest=schedule["canonical_digest"],
        scope="synthetic attention-chain full chronological descriptor trace; fixed resident Q/K/V",
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        runs=[],
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
    )

    def save():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="ascii")
        temporary.replace(target)

    save()
    try:
        for trial, order in enumerate(design["process_orders"]):
            for policy in order:
                directory = output / f"round-{trial:02d}-{policy}"
                options = dict(
                    root=root,
                    opp_root=vendors[policy],
                    output_dir=directory,
                    trace=args.trace,
                    selection=args.selection,
                    model=args.model,
                    policy=policy,
                    trial=trial,
                    repeats=design["repeats"],
                    seed=args.seed,
                    timeout=args.timeout,
                )
                command = [sys.executable, str(probe)]
                for key, value in options.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                if args.screen:
                    command.append("--screen")
                entry = dict(
                    trial=trial,
                    policy=policy,
                    process_order_index=len(status["runs"]),
                    command=command,
                    launched_unix=time.time(),
                    report_path=(directory / PROBE_REPORT_NAME).relative_to(output).as_posix(),
                )
                status["runs"].append(entry)
                save()
                rc = run_one(
                    command, output / f"round-{trial:02d}-{policy}.log", args.timeout, entry, save, runner=runner
                )
                entry.update(returncode=rc, finished_unix=time.time())
                path = directory / PROBE_REPORT_NAME
                if path.is_file():
                    entry["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                save()
                if rc:
                    raise RuntimeError(f"probe failed: trial={trial}, policy={policy}, returncode={rc}")
                report = json.loads(path.read_text())
                if (
                    report.get("state") != "completed"
                    or type(report.get("returncode")) is not int
                    or report["returncode"] != 0
                    or report.get("policy") != policy
                    or type(report.get("trial")) is not int
                    or report["trial"] != trial
                ):
                    raise RuntimeError("raw report execution or policy/trial identity failed")
                print(f"COMPLETED trial={trial} policy={policy}", flush=True)
        status.update(state="completed", returncode=0, finished_unix=time.time(), full_trace_measured=not args.screen)
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        raise
    save()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "upstream-vendor", "whole-vendor", "output-dir", "trace", "selection"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path(__file__).with_name("aclgraph_fia_sink_stage7_probe.py"))
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        parser.error("run in an existing isolated /workspace experiment")
    for name in ("trace", "selection"):
        setattr(args, name, owned(getattr(args, name), args.root))
    result = run_suite(args)
    print(json.dumps(dict(state=result["state"], output=str(args.output_dir / REPORT_NAME))))


if __name__ == "__main__":
    main()
