# SPDX-License-Identifier: Apache-2.0
"""Run alternating fresh-process native planner comparisons on device zero.

Every probe owns a separate interpreter and vendor selection. This is a
blocked comparison across processes, not per-replay randomization. Preserve
the bundle and all relative raw reports together for independent analysis.
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

POLICIES = ("upstream", "whole")
CAPACITIES = (256, 400, 496, 512)
REPORT_NAME = "stage6-suite.json"
PROBE_REPORT_NAME = "stage6-producer-probe.json"


def process_orders(rounds):
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("rounds must be an exact integer in 1..8")
    return [list(POLICIES if index % 2 == 0 else reversed(POLICIES)) for index in range(rounds)]


def owned(path, root, *, exists=True):
    result = Path(path).resolve(strict=exists)
    if not result.is_relative_to(root):
        raise ValueError("all experiment paths must remain inside the isolated root")
    return result


def run_suite(args, *, runner=subprocess.Popen):
    """Wait for each process to exit before selecting the other vendor."""
    root = args.root.resolve(strict=True)
    output = owned(args.output_dir, root, exists=False)
    probe = owned(args.probe, root)
    vendors = {name: owned(getattr(args, name + "_vendor"), root) for name in POLICIES}
    if vendors["upstream"] == vendors["whole"]:
        raise ValueError("the two policies require distinct isolated vendor directories")
    if output.exists():
        raise ValueError("preserve prior evidence; choose a fresh output directory")
    rounds, repeats = (1, 1) if args.screen else (args.rounds, args.repeats)
    orders = process_orders(rounds)
    if type(repeats) is not int or not 1 <= repeats <= 100 or args.timeout < 1:
        raise ValueError("repeats must be in 1..100 and timeout must be positive")
    output.mkdir(parents=True)
    target = output / REPORT_NAME
    status = dict(
        state="running",
        started_unix=time.time(),
        design=dict(
            policies=list(POLICIES),
            capacities=[400] if args.screen else list(CAPACITIES),
            occurrences=1 if args.screen else 16,
            rounds=rounds,
            repeats=repeats,
            process_orders=orders,
            screen=args.screen,
            seed=args.seed,
        ),
        scope="blocked fresh-process four-record synthetic attention-chain comparison",
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
        for trial, order in enumerate(orders):
            for policy in order:
                directory = output / f"round-{trial:02d}-{policy}"
                command = [sys.executable, str(probe)]
                options = dict(
                    root=root,
                    opp_root=vendors[policy],
                    output_dir=directory,
                    trace=args.trace,
                    selection=args.selection,
                    model=args.model,
                    policy=policy,
                    trial=trial,
                    repeats=repeats,
                    seed=args.seed,
                    timeout=args.timeout,
                )
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
                    report_path=str((directory / PROBE_REPORT_NAME).relative_to(output)),
                )
                status["runs"].append(entry)
                save()
                with (output / f"round-{trial:02d}-{policy}.log").open("xb") as log:
                    process = runner(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    entry["pid"] = process.pid
                    save()
                    try:
                        rc = process.wait(timeout=args.timeout + 30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError("probe launcher exceeded its process deadline") from None
                entry.update(returncode=rc, finished_unix=time.time())
                report_path = directory / PROBE_REPORT_NAME
                if report_path.is_file():
                    entry["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
                save()
                if rc:
                    raise RuntimeError(f"probe failed: trial={trial}, policy={policy}, returncode={rc}")
                report = json.loads(report_path.read_text())
                if report.get("state") != "completed" or type(report.get("returncode")) is not int:
                    raise RuntimeError("probe did not save a completed numeric return status")
                if report["returncode"] != 0 or report.get("policy") != policy or report.get("trial") != trial:
                    raise RuntimeError("probe status has the wrong exit code or policy/trial identity")
                print(f"COMPLETED trial={trial} policy={policy}", flush=True)
        status.update(state="completed", returncode=0, finished_unix=time.time())
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
    parser.add_argument("--probe", type=Path, default=Path(__file__).with_name("aclgraph_fia_sink_stage6_probe.py"))
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=600)
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
