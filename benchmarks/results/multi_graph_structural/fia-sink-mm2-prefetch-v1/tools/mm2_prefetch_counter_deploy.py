"""Copy the measured tools and enable counters in the profiling context only."""

import base64
import shlex
import subprocess
from pathlib import Path

driver = Path(__file__).with_name("mm2_prefetch_counter_driver.py").read_bytes()
script = (
    "DRIVER="
    + repr(base64.b64encode(driver).decode())
    + "\n"
    + r"""
import base64,difflib,hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
assert json.loads((root/'mm2-prefetch-screen-v3/mm2-prefetch-screen.json').read_bytes())['state']=='completed'
tools=root/'mm2-prefetch-counters-tools-v1'
assert not tools.exists() and not (root/'mm2-prefetch-counters-screen-v1').exists()
shutil.copytree(root/'mm2-prefetch-tools-v3',tools,ignore=shutil.ignore_patterns('__pycache__','*.pyc','deployed-tools.json','launch.log'))
path=tools/'aclgraph_fia_sink_consumer_probe.py'
before=path.read_bytes()
assert hashlib.sha256(before).hexdigest()=='b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9'
anchor='            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]\n'
text=before.decode()
assert text.count(anchor)==1
after=text.replace(anchor,anchor.rstrip()+',\n'+
    '            experimental_config=torch_npu.profiler._ExperimentalConfig(\n'+
    '                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,\n'+
    '                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),\n'+
    '            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target.parent / "counters")),\n')
path.write_text(after)
(tools/'probe-before.py.txt').write_bytes(before)
(tools/'counter-profile-only.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),after.splitlines(True))))
(tools/'mm2_prefetch_counter_driver.py').write_bytes(base64.b64decode(DRIVER))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tools.iterdir() if p.is_file()}
(tools/'diagnostic-sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
with (tools/'launch.log').open('xb') as log:
    child=subprocess.Popen([sys.executable,str(tools/'mm2_prefetch_counter_driver.py')],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(pid=child.pid,status=str(root/'mm2-prefetch-counters-screen-v1/counter-diagnostic.json'))))
"""
)
r = subprocess.run(
    [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "vllm-ascend-030",
        "bash -lc " + shlex.quote("source /etc/profile.d/ascend-multigraph.sh; python -"),
    ],
    input=script.encode(),
    timeout=45,
)
raise SystemExit(r.returncode)
