"""Deploy immutable files into a fresh experiment directory, then launch."""

import base64
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "work/vllm-ascend"
paths = []
for pattern in (
    "aclgraph_fia_sink_stage*.py",
    "aclgraph_fia_sink_consumer*.py",
    "aclgraph_fia_sink_running_o*.py",
    "aclgraph_fia_sink_mm2_transfer*.py",
    "aclgraph_fia_sink_mm2_prefetch*.py",
):
    paths.extend((REPO / "benchmarks").glob(pattern))
paths += [
    REPO / "benchmarks" / name
    for name in (
        "multi_graph_gdn_q4_full_metadata.py",
        "multi_graph_gdn_q4_full_loader.py",
        "multi_graph_dynamic_trace.py",
    )
]
paths += [
    REPO / "benchmarks/results/multi_graph_structural" / name
    for name in (
        "structural-dynamic-trace-v1.json",
        "q4-tile-full-replay-v8-queue0.rank0.json",
    )
]
paths += [
    HERE / "mm2_prefetch_driver.py",
    REPO / "benchmarks/aclgraph_fia_sink_mm2_double_buffer_contract.py",
    REPO / "benchmarks/aclgraph_fia_sink_mm2_double_buffer_patch.py",
]
files = {path.name: base64.b64encode(path.read_bytes()).decode() for path in paths}
script = (
    "FILES="
    + repr(files)
    + "\n"
    + r"""
import base64, hashlib, json, subprocess, sys
from pathlib import Path
root = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
assert json.loads((root/'mm2-transfer-screen-v1/mm2-transfer-screen.json').read_bytes())['state'] == 'completed'
for name in ('tools','screen','build-control','build-wide'):
    assert not (root/('mm2-prefetch-'+name+'-v3')).exists()
tools = root/'mm2-prefetch-tools-v3'
tools.mkdir()
for name, data in FILES.items():
    assert Path(name).name == name
    (tools/name).write_bytes(base64.b64decode(data))
hashes = {name: hashlib.sha256((tools/name).read_bytes()).hexdigest() for name in FILES}
assert hashes['aclgraph_fia_sink_consumer_probe.py'] == (
    'b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9')
(tools/'deployed-tools.json').write_text(json.dumps(hashes, indent=2)+'\n')
with (tools/'launch.log').open('xb') as log:
    child = subprocess.Popen([sys.executable,str(tools/'mm2_prefetch_driver.py')], stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
print(json.dumps(dict(pid=child.pid,status=str(root/'mm2-prefetch-screen-v3/mm2-prefetch-screen.json'))))
"""
)
result = subprocess.run(
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
raise SystemExit(result.returncode)
