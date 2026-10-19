"""Collect textual counter outputs with short aliases and exact byte hashes."""

import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "mm2-prefetch-counters-evidence-v1"
assert not TARGET.exists()
script = r"""
import hashlib,json,shutil,tarfile
from pathlib import Path
root=Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
screen=root/'mm2-prefetch-counters-screen-v1'
status=json.loads((screen/'counter-diagnostic.json').read_bytes())
assert status['state']=='completed' and status['returncode']==0
output=root/'mm2-prefetch-counters-collection-v1'
assert not output.exists()
output.mkdir()
tools=root/'mm2-prefetch-counters-tools-v1'
paths=[]
for p in screen.rglob('*'):
    if p.is_file() and (p.suffix in ('.csv','.log') or
                       p.name in ('consumer-probe.json','counter-diagnostic.json','trace.json')
                       or p.name.startswith('profiler_info') and p.suffix=='.json'):
        paths.append(p)
for name in ('mm2_prefetch_counter_driver.py','aclgraph_fia_sink_consumer_probe.py','probe-before.py.txt',
             'counter-profile-only.patch','diagnostic-sources.json'):
    paths.append(tools/name)
files=[]
for i,p in enumerate(sorted(paths)):
    assert p.is_file() and not p.is_symlink()
    alias=f'{i:04d}'+p.suffix
    data=p.read_bytes()
    (output/alias).write_bytes(data)
    files.append(dict(source=p.relative_to(root).as_posix(),alias=alias,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
selection = 'Text reports, CSV counters, log and source delta; device raw profiler data remains on remote'
(output/'manifest.json').write_text(json.dumps(dict(files=files,selection=selection),indent=2)+'\n')
archive=root/'mm2-prefetch-counters-collection-v1.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
    t.add(output,arcname='evidence')
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
"""
base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "vllm-ascend-030"]
r = subprocess.run(base + ["python -"], input=script.encode(), capture_output=True, timeout=60)
if r.returncode:
    print(r.stderr.decode(errors="replace"))
    raise SystemExit(r.returncode)
metadata = json.loads(r.stdout)
assert metadata["bytes"] < 30 * 1024 * 1024
archive = HERE / "mm2-prefetch-counters-evidence-v1.tar.gz"
with archive.open("xb") as sink:
    subprocess.run(base + ["cat " + shlex.quote(metadata["path"])], stdout=sink, check=True, timeout=60)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
with tarfile.open(archive, "r:gz") as t:
    members = t.getmembers()
    assert sum(m.size for m in members) < 80 * 1024 * 1024
    for m in members:
        assert not Path(m.name).is_absolute() and ".." not in Path(m.name).parts
        assert Path(m.name).parts[0] == "evidence" and not m.issym() and not m.islnk()
    TARGET.mkdir()
    t.extractall(TARGET, filter="data")
root = TARGET / "evidence"
for item in json.loads((root / "manifest.json").read_bytes())["files"]:
    assert Path(item["alias"]).name == item["alias"]
    data = (root / item["alias"]).read_bytes()
    assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
(TARGET / "collection.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(dict(target=str(TARGET), archive_bytes=metadata["bytes"])))
