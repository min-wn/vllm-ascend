# SPDX-License-Identifier: Apache-2.0
"""Run four isolated graph-catalog context controls on two original records.

One fresh process per cell is a bounded context screen. It cannot establish a
stable speedup, memory-byte causality, complete-trace or full-model performance.
"""

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

MODES = ("upstream", "whole")
CELLS = (("upstream", 2), ("whole", 2), ("upstream", 17), ("whole", 17))
CAPACITIES = (400, 512)
FIRST_USE_ORDER = (400, 304, 288, 352, 272, 496, 464, 480, 320, 448, 384, 368, 432, 256, 512, 336, 416)
FIRST_USE_STEPS = (0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 18, 21, 24, 29, 37, 38)
REPORT_NAME = "catalog-bridge-suite.json"
PROBE_REPORT_NAME = "catalog-bridge-probe.json"


def design_for(schedule):
    """Fix capture and measurement order before creating any native process."""
    records = schedule["records"]
    if len(records) != 128 or [row["step"] for row in records] != list(range(128)):
        raise ValueError("the original chronological 128-step trace is required")
    first = {}
    for record in records:
        first.setdefault(record["physical_tokens"], record)
    if (
        tuple(first) != FIRST_USE_ORDER
        or tuple(row["step"] for row in first.values()) != FIRST_USE_STEPS
        or set(first) != set(schedule["required_capture_sizes"])
        or not set(first).issubset(schedule["default_capture_sizes"])
        or [first[capacity]["step"] for capacity in CAPACITIES] != [0, 29]
    ):
        raise ValueError("catalogs must retain the original default first-use records and order")
    return dict(
        modes=list(MODES),
        catalog_counts=[2, 17],
        cells=[dict(mode=mode, catalog_count=count, trial=0) for mode, count in CELLS],
        capacities=list(CAPACITIES),
        steps=[0, 29],
        occurrences=16,
        repeats=10,
        warmups=5,
        profile_repeats=0,
        seed=1703,
        capture_orders={"2": list(CAPACITIES), "17": list(FIRST_USE_ORDER)},
        capture_steps={"2": [0, 29], "17": list(FIRST_USE_STEPS)},
        timed_step_order=[[0, 29] if repeat % 2 == 0 else [29, 0] for repeat in range(10)],
        trial_fixed_for_all_cells=0,
        independent_graph_pool_per_bucket=True,
        full_catalog_retained_until_exit=True,
        extra_catalog_replay_before_measurement=False,
        context_screen=True,
        process_replicates_per_cell=1,
    )


def completed_report(report, mode, count):
    """Reject bool-as-int and missing identities instead of accepting a wrong cell."""
    return (
        isinstance(report, dict)
        and report.get("state") == "completed"
        and type(report.get("returncode")) is int
        and report["returncode"] == 0
        and report.get("mode") == mode
        and type(report.get("trial")) is int
        and report["trial"] == 0
        and type(report.get("catalog_count")) is int
        and report["catalog_count"] == count
    )


def run_suite(args, *, runner=subprocess.Popen):
    """Run serial owned cells, preserving failure reports and exact raw hashes.

    The explicit source and report identities distinguish a resident catalog
    control from a dynamic-trace experiment. Cancellation and timeout ownership
    are inherited from the already tested Stage7 launcher implementation.
    """
    root = args.root.resolve(strict=True)
    output, probe = owned(args.output_dir, root, exists=False), owned(args.probe, root)
    vendors = {mode: owned(getattr(args, mode + "_vendor"), root) for mode in MODES}
    if vendors["upstream"] == vendors["whole"]:
        raise ValueError("native policies require distinct isolated vendors")
    if output.exists() or type(args.timeout) is not int or args.timeout < 1:
        raise ValueError("choose a fresh output directory and positive integer timeout")
    if not probe.is_file() or any(not vendor.is_dir() for vendor in vendors.values()):
        raise ValueError("the probe must be a file and both vendors must be directories")
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
        scope="four-cell synthetic attention graph-catalog context screen; one fresh process per cell",
        interpretation="catalog context includes capture history and address placement; memory-byte cause unproven",
        context_screen=True,
        runs=[],
        performance_measured=False,
        stable_performance_gain_established=False,
        catalog_memory_causality_established=False,
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
        for index, (mode, count) in enumerate(CELLS):
            name = f"cell-{index:02d}-{mode}-catalog{count}"
            directory = output / name
            options = dict(
                root=root,
                opp_root=vendors[mode],
                output_dir=directory,
                trace=args.trace,
                selection=args.selection,
                model=args.model,
                mode=mode,
                catalog_count=count,
                trial=0,
                repeats=design["repeats"],
                warmups=design["warmups"],
                seed=design["seed"],
                timeout=args.timeout,
            )
            command = [sys.executable, str(probe)]
            for key, value in options.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            entry = dict(
                mode=mode,
                catalog_count=count,
                trial=0,
                process_order_index=index,
                command=command,
                launched_unix=time.time(),
                report_path=(directory / PROBE_REPORT_NAME).relative_to(output).as_posix(),
            )
            status["runs"].append(entry)
            save()
            rc = run_one(command, output / (name + ".log"), args.timeout, entry, save, runner=runner)
            entry.update(returncode=rc, finished_unix=time.time())
            path = directory / PROBE_REPORT_NAME
            if path.is_file():
                entry["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            save()
            if rc:
                raise RuntimeError(f"probe failed: mode={mode}, catalog_count={count}, returncode={rc}")
            report = json.loads(path.read_text())
            if not completed_report(report, mode, count):
                raise RuntimeError("probe completion or mode/catalog/trial identity failed")
            print(f"COMPLETED mode={mode} catalog_count={count}", flush=True)
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
    parser.add_argument(
        "--probe", type=Path, default=Path(__file__).with_name("aclgraph_fia_sink_catalog_bridge_probe.py")
    )
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--timeout", type=int, default=1800)
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
