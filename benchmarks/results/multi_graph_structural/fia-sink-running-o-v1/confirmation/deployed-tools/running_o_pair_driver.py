"""One reversed whole-policy pair using the unchanged probe and v3 binaries."""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e")
TOOLS = ROOT / "running-o-tools-v3"
sys.path.insert(0, str(TOOLS))
from aclgraph_fia_sink_stage7_suite import run_one

OUT = ROOT / "running-o-confirm-v1"
assert not OUT.exists()
OUT.mkdir()
REPORT = OUT / "running-o-confirm.json"
assert json.loads((ROOT / "running-o-screen-v3/running-o-screen.json").read_bytes())["state"] == "completed"
assert hashlib.sha256((TOOLS / "aclgraph_fia_sink_consumer_probe.py").read_bytes()).hexdigest() == "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9"
status = dict(state="running", started_unix=time.time(), runs=[], builds={},
              design=dict(cells=[["control", "whole"], ["resident", "whole"]],
                          repeats=10, warmups=5, profile_repeats=1, seed=1703, trial=0,
                          capacities=[400, 512], occurrences=16),
              motivation="Reverse preceding whole pair order to check the 1% median observation and submission spike; retain every sample",
              builds_reused_from_v3=True, consumer_probe_unchanged=True,
              performance_measured=False, full_trace_measured=False,
              full_model_qualified=False, end_to_end_gain_established=False)


def save():
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(REPORT)


def idle(label):
    info = subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, text=True, check=True).stdout
    (OUT / ("device-" + label + ".txt")).write_text(info)
    assert "No running processes found in NPU 0" in info


save()
try:
    idle("before")
    for index, variant in enumerate(("control", "resident")):
        build = ROOT / ("running-o-build-" + variant + "-v3")
        raw = (build / "running-o-build.json").read_bytes()
        metadata = json.loads(raw)
        assert metadata["state"] == "completed" and metadata["resident"] == (variant == "resident")
        assert metadata["binary_qualification"]["actual_tiling_keys"] == ["104000000010222223", "104000000010322223"]
        status["builds"][variant] = dict(report_sha256=hashlib.sha256(raw).hexdigest(), artifacts=metadata["artifacts"])
        output = OUT / (f"cell-{index:02d}-{variant}-whole")
        options = dict(root=ROOT, opp_root=build / "vendor-whole", output_dir=output,
                       trace=TOOLS / "structural-dynamic-trace-v1.json",
                       selection=TOOLS / "q4-tile-full-replay-v8-queue0.rank0.json",
                       mode="whole", trial=0, repeats=10, warmups=5, profile_repeats=1, seed=1703, timeout=900)
        command = [sys.executable, str(TOOLS / "aclgraph_fia_sink_consumer_probe.py")]
        for key, value in options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        idle("cell" + str(index))
        entry = dict(variant=variant, policy="whole", process_order_index=index, command=command,
                     launched_unix=time.time(), report_path=(output / "consumer-probe.json").relative_to(OUT).as_posix())
        status["runs"].append(entry)
        save()
        rc = run_one(command, OUT / (f"cell-{index:02d}.log"), 900, entry, save)
        entry.update(returncode=rc, finished_unix=time.time())
        save()
        assert rc == 0
        report = json.loads((output / "consumer-probe.json").read_bytes())
        assert report["state"] == "completed" and len(report["samples"]) == 20
        entry["report_sha256"] = hashlib.sha256((output / "consumer-probe.json").read_bytes()).hexdigest()
        save()
    idle("after")
    status.update(state="completed", returncode=0, performance_measured=True, finished_unix=time.time())
except BaseException:
    status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
    traceback.print_exc()
save()
raise SystemExit(status["returncode"])
