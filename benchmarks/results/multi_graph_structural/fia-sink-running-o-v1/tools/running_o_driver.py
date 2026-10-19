"""Remote isolated compile and four-cell consumer source-ablation screen."""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e")
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from aclgraph_fia_sink_stage7_suite import run_one

VERSION = int(sys.argv[1])
assert 1 <= VERSION <= 9
OUT = ROOT / f"running-o-screen-v{VERSION}"
assert not OUT.exists()
OUT.mkdir()
REPORT = OUT / "running-o-screen.json"
UPSTREAM = ROOT / "opp/packages/vendors/thesis_fia_stage2_transformer"
WHOLE = ROOT / "stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3"
CELLS = [("control", "upstream"), ("resident", "upstream"), ("resident", "whole"), ("control", "whole")]
status = dict(
    state="running", started_unix=time.time(), stage="build", builds=[], runs=[],
    design=dict(cells=CELLS, trial=0, seed=1703, warmups=5, repeats=10,
                profile_repeats=1, steps=[0,29], capacities=[400,512], occurrences=16),
    scope="One source-ablation screen, one fresh process per cell; not a stable dynamic/full-model or split-over-padding claim",
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TOOLS.glob("*.py"))},
    performance_measured=False, full_model_qualified=False, full_trace_measured=False,
    end_to_end_gain_established=False,
)


def save():
    temp = REPORT.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n")
    temp.replace(REPORT)


def idle(label):
    info = subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, text=True, check=True).stdout
    (OUT / ("device-" + label + ".txt")).write_text(info)
    assert "No running processes found in NPU 0" in info, info


save()
try:
    idle("before")
    for name, enabled in (("control",0),("resident",1)):
        build = ROOT / ("running-o-build-" + name + f"-v{VERSION}")
        command=[sys.executable, str(TOOLS/"aclgraph_fia_sink_running_o_build.py"),
                 "--root",str(ROOT),"--upstream-vendor",str(UPSTREAM),"--whole-vendor",str(WHOLE),
                 "--output-dir",str(build),"--resident",str(enabled),"--timeout","1200",
                 "--tiling-keys","104000000010222223,104000000010322223"]
        entry=dict(name=name, command=command, started_unix=time.time(), status=str(build/"running-o-build.json"))
        status["builds"].append(entry);save()
        with (OUT/("build-"+name+".log")).open("xb") as log:
            result=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1350)
        entry.update(returncode=result.returncode, finished_unix=time.time());save()
        assert result.returncode==0, entry
        report=json.loads((build/"running-o-build.json").read_text())
        assert report["state"]=="completed" and report["returncode"]==0, report
    status["stage"]="probes";save()
    for index,(variant,policy) in enumerate(CELLS):
        idle("cell"+str(index))
        build=ROOT/("running-o-build-"+variant+f"-v{VERSION}")
        output=OUT/(f"cell-{index:02d}-{variant}-{policy}")
        options=dict(root=ROOT, opp_root=build/("vendor-"+policy), output_dir=output,
                     trace=TOOLS/"structural-dynamic-trace-v1.json",
                     selection=TOOLS/"q4-tile-full-replay-v8-queue0.rank0.json",
                     mode=policy,trial=0,repeats=10,warmups=5,profile_repeats=1,seed=1703,timeout=900)
        command=[sys.executable,str(TOOLS/"aclgraph_fia_sink_consumer_probe.py")]
        for key,value in options.items():
            command.extend(["--"+key.replace("_","-"),str(value)])
        entry=dict(variant=variant,policy=policy,process_order_index=index,command=command,
                   launched_unix=time.time(),report_path=(output/"consumer-probe.json").relative_to(OUT).as_posix())
        status["runs"].append(entry);save()
        rc=run_one(command,OUT/(f"cell-{index:02d}.log"),900,entry,save)
        entry.update(returncode=rc,finished_unix=time.time());save()
        assert rc==0, entry
        report=json.loads((output/"consumer-probe.json").read_text())
        assert report["state"]=="completed" and report["returncode"]==0 and report["mode"]==policy
        assert len(report["samples"])==20 and len(report["pre_checks"])==len(report["post_checks"])==2
        entry["report_sha256"]=hashlib.sha256((output/"consumer-probe.json").read_bytes()).hexdigest();save()
    idle("after")
    status.update(state="completed",stage="completed",returncode=0,performance_measured=True,finished_unix=time.time())
except BaseException:
    status.update(state="failed",returncode=1,error=traceback.format_exc(),finished_unix=time.time())
    traceback.print_exc()
save()
raise SystemExit(status["returncode"])
