"""Bounded ABBA screen: same-layout original control and MM2 prefetch."""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from aclgraph_fia_sink_mm2_prefetch_contract import run as contract
from aclgraph_fia_sink_stage7_suite import run_one

ROOT = Path("/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e")
TOOLS = Path(__file__).resolve().parent

OUT = ROOT / "mm2-prefetch-screen-v3"
assert not OUT.exists()
OUT.mkdir()
REPORT = OUT / "mm2-prefetch-screen.json"
CELLS = ("control", "candidate", "candidate", "control")
BUILDS = {name: ROOT / ("mm2-prefetch-build-" + name + "-v3") for name in ("control", "candidate")}
status = dict(
    state="running",
    stage="cpu_contract",
    started_unix=time.time(),
    runs=[],
    builds=[],
    design=dict(
        cells=list(CELLS),
        policy="whole",
        trial=0,
        seed=1703,
        warmups=5,
        repeats=10,
        profile_repeats=1,
        capacities=[400, 512],
        occurrences=16,
    ),
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TOOLS.glob("*.py"))},
    scope="Two synthetic attention-chain records; ABBA same-layout prefetch; not full-model or split-exclusive",
    performance_measured=False,
    full_trace_measured=False,
    full_model_qualified=False,
    end_to_end_gain_established=False,
)


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
    native = ROOT / "source/csrc/build/binary/ascend910b/src/fused_infer_attention_score_v2_sink/op_kernel"
    status["cpu_contract"] = contract(native, OUT)
    save()
    for name, enabled in (("control", 0), ("candidate", 1)):
        command = [
            sys.executable,
            str(TOOLS / "aclgraph_fia_sink_mm2_prefetch_build.py"),
            "--root",
            str(ROOT),
            "--upstream-vendor",
            str(ROOT / "opp/packages/vendors/thesis_fia_stage2_transformer"),
            "--whole-vendor",
            str(ROOT / "stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3"),
            "--output-dir",
            str(BUILDS[name]),
            "--prefetch",
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
        assert result.returncode == 0
        metadata = json.loads((BUILDS[name] / "mm2-prefetch-build.json").read_bytes())
        assert metadata["state"] == "completed" and metadata["prefetch"] == bool(enabled)
    assert (
        hashlib.sha256((TOOLS / "aclgraph_fia_sink_consumer_probe.py").read_bytes()).hexdigest()
        == "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9"
    )
    for index, variant in enumerate(CELLS):
        idle("cell" + str(index))
        output = OUT / (f"cell-{index:02d}-{variant}-whole")
        options = dict(
            root=ROOT,
            opp_root=BUILDS[variant] / "vendor-whole",
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
        command = [sys.executable, str(TOOLS / "aclgraph_fia_sink_consumer_probe.py")]
        for key, value in options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        entry = dict(
            variant=variant,
            policy="whole",
            process_order_index=index,
            command=command,
            launched_unix=time.time(),
            report_path=(output / "consumer-probe.json").relative_to(OUT).as_posix(),
        )
        status["runs"].append(entry)
        status["stage"] = "cell-" + str(index)
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
    status.update(
        state="completed", stage="completed", returncode=0, performance_measured=True, finished_unix=time.time()
    )
except BaseException:
    status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
    traceback.print_exc()
save()
raise SystemExit(status["returncode"])
