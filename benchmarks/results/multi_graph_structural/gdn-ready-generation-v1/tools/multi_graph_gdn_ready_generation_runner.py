# SPDX-License-Identifier: Apache-2.0
"""Bounded sequential independent TP2 groups, auditing each pair before proceeding."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from multi_graph_gdn_aime26_probe import save_json
from multi_graph_gdn_ready_chain_runner import EXTENSIONS
from multi_graph_gdn_ready_generation import check_pair, load_manifest, run_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root, tools = args.root, Path(__file__).resolve().parent
    manifest = load_manifest(root / "workload.json")
    status_path = root / "status.json"
    if status_path.exists():
        raise FileExistsError("do not repeat or overwrite existing generation")
    env = dict(
        os.environ,
        PYTHONPATH=str(tools) + ":" + os.environ.get("PYTHONPATH", ""),
        ASCEND_RT_VISIBLE_DEVICES="0,1",
        TASK_QUEUE_ENABLE="0",
        HCCL_DETERMINISTIC="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    protocol = dict(
        plan=run_plan(),
        workload_sha256=manifest["payload_sha256"],
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tools.glob("*.py")},
        extensions={
            key: dict(path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for key, p in EXTENSIONS.items()
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
        timing="independent sequential TP2 processes; no concurrent timing; one untimed and one timed generation each",
        timeout_seconds_per_process=1200,
        stable_gate="all three generation gains positive; pooled gain >1% and >2x both repeat ranges",
    )
    save_json(root / "protocol.json", protocol)
    status = dict(state="running", completed=[], current=None, pair_checks=[])
    save_json(status_path, status)
    for index, item in enumerate(run_plan()):
        label = f"pair{item['pair']}.{item['mode']}"
        directory = root / label
        directory.mkdir(exist_ok=False)
        command = [
            "torchrun",
            "--nproc_per_node=2",
            f"--master_port={29904 + index}",
            str(tools / "multi_graph_gdn_ready_generation_probe.py"),
            "--manifest",
            str(root / "workload.json"),
            "--mode",
            item["mode"],
            "--output",
            str(directory / "result.json"),
        ]
        for key, path in EXTENSIONS.items():
            command.extend(["--" + key, path])
        save_json(directory / "command.json", command)
        status["current"] = dict(**item, label=label, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_json(status_path, status)
        started = time.monotonic()
        failure = None
        with (directory / "run.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["current"]["process_group"] = process.pid
            save_json(status_path, status)
            try:
                code = process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code, failure = -1, "bounded process-group timeout; no automatic retry"
        status["completed"].append(
            dict(**item, label=label, exit_code=code, elapsed_seconds=time.monotonic() - started, failure=failure)
        )
        if code:
            status["state"] = "failed_preserve_evidence"
            save_json(status_path, status)
            return 1
        if item["order"] == 1:
            try:
                reports = [
                    [
                        json.loads((root / f"pair{item['pair']}.{mode}" / f"result.rank{rank}.json").read_bytes())
                        for rank in (0, 1)
                    ]
                    for mode in ("delayed", "ready")
                ]
                check_pair(*reports)
            except (OSError, ValueError, KeyError) as error:
                status.update(state="correctness_failure_no_performance_comparison", failure=str(error))
                save_json(status_path, status)
                return 1
            status["pair_checks"].append(dict(pair=item["pair"], identical_work_and_closed_state=True))
        save_json(status_path, status)
    status.update(state="complete_pending_analysis", current=None)
    save_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
