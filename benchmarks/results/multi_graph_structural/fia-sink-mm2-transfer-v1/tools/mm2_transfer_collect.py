"""Collect a completed isolated screen with hashes and safe local extraction."""

import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mm2-transfer-evidence-v1"
assert not OUT.exists()
script = r'''
import hashlib, json, shutil, tarfile
from pathlib import Path
root = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
screen = root/'mm2-transfer-screen-v1'
status = json.loads((screen/'mm2-transfer-screen.json').read_bytes())
assert status['state'] == 'completed' and status['returncode'] == 0
out = root/'mm2-transfer-collection-v1'
assert not out.exists()
out.mkdir()
ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
shutil.copytree(screen, out/'screen', ignore=ignore)
shutil.copytree(root/'mm2-transfer-tools-v1', out/'tools', ignore=ignore)
for variant in ('control','wide'):
    build = root/('mm2-transfer-build-'+variant+'-v1')
    target = out/'builds'/variant
    target.mkdir(parents=True)
    for name in ('mm2-transfer-build.json','compile.log','patch.log'):
        shutil.copyfile(build/name, target/name)
    for relative in ('source/mm2-transfer-patch.json','source/mm2-transfer-source.patch'):
        shutil.copyfile(build/relative, target/Path(relative).name)
    native = build/'source/csrc/attention/fused_infer_attention_score_v2_sink'
    for relative in ('op_kernel/fia_block_cube_nonquant_gqa_sink.h','FusedInferAttentionScoreV2Sink.py'):
        shutil.copyfile(native/relative, target/(Path(relative).name+'.txt'))
    shutil.copytree(build/'binary', target/'binary', ignore=ignore)
files = {}
for path in sorted(out.rglob('*')):
    assert not path.is_symlink()
    if path.is_file():
        files[path.relative_to(out).as_posix()] = dict(bytes=path.stat().st_size,
                                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(dict(files=files, excluded_generated=['__pycache__','*.pyc']),indent=2)+'\n')
archive = root/'mm2-transfer-collection-v1.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as bundle:
    bundle.add(out,arcname='evidence')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))))
'''
base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "vllm-ascend-030"]
result = subprocess.run(base + ["python -"], input=script.encode(), capture_output=True, timeout=90)
if result.returncode:
    print(result.stderr.decode(errors="replace"))
    raise SystemExit(result.returncode)
metadata = json.loads(result.stdout)
assert metadata["bytes"] < 100 * 1024 * 1024
archive = HERE / "mm2-transfer-evidence-v1.tar.gz"
assert not archive.exists()
with archive.open("xb") as sink:
    subprocess.run(base + ["cat " + shlex.quote(metadata["archive"])], stdout=sink, check=True, timeout=90)
assert archive.stat().st_size == metadata["bytes"]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
with tarfile.open(archive, "r:gz") as bundle:
    members = bundle.getmembers()
    assert sum(m.size for m in members) < 250 * 1024 * 1024
    for member in members:
        path = Path(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert path.parts[0] == "evidence" and not member.issym() and not member.islnk()
    OUT.mkdir()
    bundle.extractall(OUT, members=members, filter="data")
root = OUT / "evidence"
manifest = json.loads((root / "manifest.json").read_bytes())
for relative, entry in manifest["files"].items():
    path = root / relative
    assert path.is_file() and path.stat().st_size == entry["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
(OUT / "collection.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="ascii")
print(json.dumps(dict(output=str(root), files=len(manifest["files"]), archive_bytes=metadata["bytes"])))
