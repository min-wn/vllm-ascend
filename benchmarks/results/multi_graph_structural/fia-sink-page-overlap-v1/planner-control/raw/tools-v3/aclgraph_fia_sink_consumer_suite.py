# SPDX-License-Identifier: Apache-2.0
"""Run a bounded, fresh-process diagnostic of current attention consumers."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from aclgraph_fia_sink_stage3_trace import load_trace_schedule
from aclgraph_fia_sink_stage6_suite import owned
from aclgraph_fia_sink_stage7_suite import run_one

MODES = ("upstream", "whole", "legacy")
CAPACITIES = (400, 512)
REPORT_NAME = "consumer-suite.json"
PROBE_REPORT_NAME = "consumer-probe.json"


def design_for(schedule):
    records = [next(row for row in schedule["records"] if row["physical_tokens"] == cap) for cap in CAPACITIES]
    if [row["step"] for row in records] != [0, 29]:
        raise ValueError("diagnostic must use the original first records at capacities 400 and 512")
    return dict(
        modes=list(MODES),
        capacities=list(CAPACITIES),
        steps=[row["step"] for row in records],
        occurrences=16,
        rounds=2,
        repeats=10,
        warmups=5,
        profile_repeats=3,
        process_orders=[list(MODES), list(reversed(MODES))],
        seed=1703,
        legacy_update_audit=(
            "pre/post qualification only; uninstrumented timing/profile; independent profile task counts"
        ),
        profile_label_scope="replay and final join; preparation completed before each label",
    )


def run_suite(args, *, runner=subprocess.Popen):
    """Preserve all six serial processes and reject partial/misattributed reports."""
    root = args.root.resolve(strict=True)
    output, probe = owned(args.output_dir, root, exists=False), owned(args.probe, root)
    vendors = {name: owned(getattr(args, name + "_vendor"), root) for name in ("upstream", "whole")}
    if vendors["upstream"] == vendors["whole"]:
        raise ValueError("native policies require distinct isolated vendors")
    if output.exists() or args.timeout < 1:
        raise ValueError("choose a fresh output directory and positive timeout")
    schedule = load_trace_schedule(args.trace, args.selection)
    design = design_for(schedule)
    output.mkdir(parents=True)
    target = output / REPORT_NAME
    status = dict(
        state="running",
        started_unix=time.time(),
        design=design,
        canonical_digest=schedule["canonical_digest"],
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        scope="two-record synthetic attention-chain consumer diagnostic; no full-trace/model inference claim",
        runs=[],
        performance_measured=False,
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
            for mode in order:
                directory = output / f"round-{trial:02d}-{mode}"
                options = dict(
                    root=root,
                    opp_root=vendors["whole" if mode == "whole" else "upstream"],
                    output_dir=directory,
                    trace=args.trace,
                    selection=args.selection,
                    model=args.model,
                    mode=mode,
                    trial=trial,
                    repeats=design["repeats"],
                    profile_repeats=design["profile_repeats"],
                    warmups=design["warmups"],
                    seed=design["seed"],
                    timeout=args.timeout,
                )
                command = [sys.executable, str(probe)]
                for key, value in options.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                entry = dict(
                    trial=trial,
                    mode=mode,
                    process_order_index=len(status["runs"]),
                    command=command,
                    launched_unix=time.time(),
                    report_path=(directory / PROBE_REPORT_NAME).relative_to(output).as_posix(),
                )
                status["runs"].append(entry)
                save()
                rc = run_one(
                    command, output / f"round-{trial:02d}-{mode}.log", args.timeout, entry, save, runner=runner
                )
                entry.update(returncode=rc, finished_unix=time.time())
                path = directory / PROBE_REPORT_NAME
                if path.is_file():
                    entry["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                save()
                if rc:
                    raise RuntimeError(f"probe failed: trial={trial}, mode={mode}, returncode={rc}")
                report = json.loads(path.read_text())
                if (
                    report.get("state") != "completed"
                    or type(report.get("returncode")) is not int
                    or report["returncode"] != 0
                    or report.get("mode") != mode
                    or type(report.get("trial")) is not int
                    or report["trial"] != trial
                ):
                    raise RuntimeError("probe completion or mode/trial identity failed")
                print(f"COMPLETED trial={trial} mode={mode}", flush=True)
        status.update(state="completed", returncode=0, finished_unix=time.time(), performance_measured=True)
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
    parser.add_argument("--probe", type=Path, default=Path(__file__).with_name("aclgraph_fia_sink_consumer_probe.py"))
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--timeout", type=int, default=900)
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
