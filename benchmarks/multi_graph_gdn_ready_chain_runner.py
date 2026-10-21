# SPDX-License-Identifier: Apache-2.0
"""Run predeclared controls sequentially, each in a fresh TP2 process group."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from multi_graph_gdn_ready_chain_protocol import LOADS, run_plan, workload

EXTENSIONS = {
    "extension": "/workspace/multi-graph-results/q4-log-build-v3-ranges/gdn_q4_log.so",
    "products-extension": "/workspace/multi-graph-results/q4-products-build-v9-ranges/gdn_q4_products.so",
    "deferred-extension": "/workspace/multi-graph-results/q4-log-deferred-build-v1/gdn_q4_log_deferred.so",
    "tile-extension": "/workspace/multi-graph-results/q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so",
}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--profile-load", choices=LOADS)
    parser.add_argument("--profile-steps", type=int, nargs="+")
    args = parser.parse_args()
    if bool(args.profile_load) != bool(args.profile_steps):
        parser.error("profile load and selected steps required together")
    root = args.root
    tools = Path(__file__).resolve().parent
    status_path = root / ("profile-status.json" if args.profile_load else "status.json")
    if status_path.exists():
        raise FileExistsError("retain prior runner evidence; do not duplicate or overwrite")
    trace_path = root / "trace.json"
    plan = (
        run_plan()
        if not args.profile_load
        else [
            dict(load=args.profile_load, mode=mode, pair="profile", order=order)
            for order, mode in enumerate(("delayed", "ready"))
        ]
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(tools) + ":" + os.environ.get("PYTHONPATH", ""),
        ASCEND_RT_VISIBLE_DEVICES=args.devices,
        TASK_QUEUE_ENABLE="0",
        HCCL_DETERMINISTIC="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    protocol = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        plan=plan,
        devices=args.devices,
        process_groups="independent and sequential, no simultaneous timing",
        extensions={
            name: dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
            for name, path in EXTENSIONS.items()
        },
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tools.glob("*.py")},
        workloads={
            load: dict(sha256=workload(trace_path, load)[0].digest, steps=len(workload(trace_path, load)[0].steps))
            for load in LOADS
        },
        environment={
            k: env[k]
            for k in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "TASK_QUEUE_ENABLE",
                "HCCL_DETERMINISTIC",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        scope="fixed-trajectory complete-model execution replay; no sampling/draft/service claims",
        timeout_seconds_per_process_group=1200,
    )
    save(root / ("profile-protocol.json" if args.profile_load else "protocol.json"), protocol)
    status = dict(state="running", runner_pid=os.getpid(), completed=[], current=None)
    save(status_path, status)
    for index, item in enumerate(plan):
        label = f"{item['load']}.pair{item['pair']}.{item['mode']}"
        directory = root / label
        directory.mkdir(exist_ok=False)
        command = [
            "torchrun",
            "--nproc_per_node=2",
            f"--master_port={29824 + index}",
            str(tools / "multi_graph_gdn_ready_chain_probe.py"),
            "--trace",
            str(trace_path),
            "--mode",
            item["mode"],
            "--load",
            item["load"],
            "--output",
            str(directory / "result.json"),
        ]
        for name, path in EXTENSIONS.items():
            command += ["--" + name, path]
        if args.profile_steps:
            command += ["--profile-steps", *map(str, args.profile_steps)]
        save(directory / "command.json", command)
        status["current"] = dict(**item, label=label, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save(status_path, status)
        begin = time.monotonic()
        try:
            with (directory / "run.log").open("w") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            exit_code, failure = process.returncode, None
        except subprocess.TimeoutExpired:
            exit_code, failure = -1, "bounded process group exceeded timeout; inspect children before any new run"
        result = dict(
            **item, label=label, exit_code=exit_code, elapsed_seconds=time.monotonic() - begin, failure=failure
        )
        if exit_code == 0:
            expected = (
                "profile_complete_not_an_unprofiled_timing_sample"
                if args.profile_load
                else "closed_complete_model_replay_pending_cross_process_pair_audit"
            )
            for rank in (0, 1):
                try:
                    report = json.loads((directory / f"result.rank{rank}.json").read_text())
                    if report["status"] != expected:
                        raise ValueError("incomplete report status")
                except (OSError, ValueError, KeyError) as error:
                    result.update(exit_code=-2, failure=str(error))
        status["completed"].append(result)
        if result["exit_code"] != 0:
            status["state"] = "failed_preserve_evidence_no_automatic_retry"
            save(status_path, status)
            return 1
        save(status_path, status)
    status.update(state="complete_pending_analysis", current=None)
    save(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
