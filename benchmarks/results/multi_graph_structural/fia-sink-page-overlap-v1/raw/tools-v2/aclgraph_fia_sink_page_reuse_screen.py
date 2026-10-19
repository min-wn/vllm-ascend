# SPDX-License-Identifier: Apache-2.0
"""Bounded ABBA address-reuse screen in the existing isolated A2 experiment."""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from aclgraph_fia_sink_page_reuse_contract import run as contract
from aclgraph_fia_sink_stage7_suite import run_one

ROOT = Path("/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e")
TOOLS = Path(__file__).resolve().parent
OUT = ROOT / "page-reuse-screen-v1"
CELLS = ("control", "candidate", "candidate", "control")


def main():
    OUT.mkdir(exist_ok=False)
    report_path = OUT / "page-reuse-screen.json"
    builds = {name: ROOT / ("page-reuse-build-" + name + "-v1") for name in ("control", "candidate")}
    status = dict(
        state="running",
        stage="contract",
        runs=[],
        builds=[],
        started_unix=time.time(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in TOOLS.glob("*.py")},
        scope="two synthetic chain records, 16 attention occurrences; not full model or split-exclusive",
    )

    def save():
        report_path.write_text(json.dumps(status, indent=2) + "\n")

    def idle(label):
        info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
        (OUT / ("device-" + label + ".txt")).write_text(info)
        if "No running processes found in NPU 0" not in info:
            raise RuntimeError("NPU0 is occupied")

    save()
    try:
        idle("before")
        native = ROOT / "source/csrc/build/binary/ascend910b/src/fused_infer_attention_score_v2_sink/op_kernel"
        status["contract"] = contract(native, OUT)
        save()
        for name, enabled in (("control", 0), ("candidate", 1)):
            command = [
                sys.executable,
                str(TOOLS / "aclgraph_fia_sink_page_reuse_build.py"),
                "--root",
                str(ROOT),
                "--upstream-vendor",
                str(ROOT / "opp/packages/vendors/thesis_fia_stage2_transformer"),
                "--whole-vendor",
                str(ROOT / "stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3"),
                "--output-dir",
                str(builds[name]),
                "--page-reuse",
                str(enabled),
                "--timeout",
                "1200",
            ]
            entry = dict(name=name, command=command, started_unix=time.time())
            status["builds"].append(entry)
            status["stage"] = "build-" + name
            save()
            with (OUT / ("build-" + name + ".log")).open("xb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1300)
            entry.update(returncode=result.returncode, finished_unix=time.time())
            save()
            if result.returncode:
                raise RuntimeError("isolated build failed")
        binary = "FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673.o"
        control_hash = hashlib.sha256((builds["control"] / "binary" / binary).read_bytes()).hexdigest()
        status["control_sha256"] = control_hash
        if control_hash != "0ef799f0ec513f2fb2c9b0e642379f3123911fdaf9b656f17ab3976c139d1743":
            raise RuntimeError("disabled candidate no longer matches qualified baseline bytes")
        probe = TOOLS / "aclgraph_fia_sink_consumer_probe.py"
        if (
            hashlib.sha256(probe.read_bytes()).hexdigest()
            != "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9"
        ):
            raise RuntimeError("probe byte drift")
        for index, variant in enumerate(CELLS):
            idle("cell" + str(index))
            output = OUT / f"cell-{index:02d}-{variant}-whole"
            options = dict(
                root=ROOT,
                opp_root=builds[variant] / "vendor-whole",
                output_dir=output,
                trace=TOOLS / "structural-dynamic-trace-v1.json",
                selection=TOOLS / "q4-tile-full-replay-v8-queue0.rank0.json",
                mode="whole",
                trial=0,
                repeats=10,
                warmups=5,
                profile_repeats=1,
                seed=1703,
                timeout=900,
            )
            command = [sys.executable, str(probe)]
            for key, value in options.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            entry = dict(
                variant=variant, command=command, report_path=str((output / "consumer-probe.json").relative_to(OUT))
            )
            status["runs"].append(entry)
            status["stage"] = f"cell-{index}"
            save()
            rc = run_one(command, OUT / f"cell-{index:02d}.log", 900, entry, save)
            entry.update(returncode=rc, finished_unix=time.time())
            save()
            if rc:
                raise RuntimeError("probe failed")
            report = json.loads((output / "consumer-probe.json").read_bytes())
            if report["state"] != "completed" or len(report["samples"]) != 20:
                raise RuntimeError("incomplete probe")
        idle("after")
        status.update(state="completed", stage="completed", returncode=0)
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc())
        traceback.print_exc()
    status["finished_unix"] = time.time()
    save()
    return status["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
