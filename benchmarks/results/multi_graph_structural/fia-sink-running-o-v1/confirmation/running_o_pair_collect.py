"""Read-only collection of the completed reverse-order whole pair."""

import base64
import hashlib
import json
import shlex
import subprocess
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ro-confirm-v1"
assert not OUT.exists()
REMOTE = r'''
import base64,hashlib,json,zlib
from pathlib import Path
root=Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
screen=root/'running-o-confirm-v1'
suite=json.loads((screen/'running-o-confirm.json').read_bytes())
assert suite['state'] in ('completed','failed')
paths=[(path,path.relative_to(screen).as_posix()) for path in screen.rglob('*')
       if path.is_file() and path.suffix in ('.json','.log','.txt')]
tools=root/'running-o-confirm-tools-v1'
paths.extend((path,'deployed-tools/'+path.name) for path in tools.iterdir() if path.is_file())
frozen=root/'running-o-tools-v3'
for name in ('deployed-tools.json','aclgraph_fia_sink_consumer_probe.py','aclgraph_fia_sink_stage7_suite.py'):
 paths.append((frozen/name,'frozen-v3/'+name))
rows=[]
for path,name in sorted(paths):
 data=path.read_bytes()
 rows.append(dict(path=name,remote_path=path.relative_to(root).as_posix(),bytes=len(data),
                  sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(zlib.compress(data)).decode()))
print(json.dumps(rows))
'''
command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "vllm-ascend-030",
           "bash -lc " + shlex.quote("source /etc/profile.d/ascend-multigraph.sh; python -")]
result = subprocess.run(command, input=REMOTE.encode(), stdout=subprocess.PIPE, check=True, timeout=90)
rows = json.loads(result.stdout)
manifest = []
OUT.mkdir()
for row in rows:
    data = zlib.decompress(base64.b64decode(row.pop("data")))
    assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
    path = OUT / row["path"]
    assert path.resolve().is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest.append(row)
suite_path = OUT / "running-o-confirm.json"
suite = json.loads(suite_path.read_bytes())
collection = dict(partial=suite["state"] != "completed", controller_interrupted=False,
                  screen_state=suite["state"], suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                  npu_work_executed_by_collector=False, reused_builds_from_v3=True)
data = (json.dumps(collection, indent=2) + "\n").encode("ascii")
(OUT / "collection.json").write_bytes(data)
manifest.append(dict(path="collection.json", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                     generated_by_collector=True))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
print(json.dumps(dict(output=str(OUT), files=len(rows), bytes=sum(row["bytes"] for row in rows))))
