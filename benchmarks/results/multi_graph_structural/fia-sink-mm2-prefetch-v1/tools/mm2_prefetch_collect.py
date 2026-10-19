"""Collect a completed isolated screen with hashes and safe local extraction."""

import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mm2-prefetch-evidence-v1"
assert not OUT.exists()
script = r"""
import hashlib, json, shutil, tarfile
from pathlib import Path
root = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
screen = root/'mm2-prefetch-screen-v3'
status = json.loads((screen/'mm2-prefetch-screen.json').read_bytes())
assert status['state'] == 'completed' and status['returncode'] == 0
out = root/'mm2-prefetch-collection-v1'
assert not out.exists()
out.mkdir()
ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
shutil.copytree(screen, out/'screen', ignore=ignore)
shutil.copytree(root/'mm2-prefetch-tools-v3', out/'tools', ignore=ignore)
for variant in ('control','candidate'):
    build = root/('mm2-prefetch-build-'+variant+'-v3')
    tag = 'mm2-prefetch'
    target = out/'builds'/variant
    target.mkdir(parents=True)
    for name in (tag+'-build.json','compile.log','patch.log'):
        shutil.copyfile(build/name, target/name)
    for relative in ('source/'+tag+'-patch.json','source/'+tag+'-source.patch'):
        shutil.copyfile(build/relative, target/Path(relative).name)
    native = build/'source/csrc/attention/fused_infer_attention_score_v2_sink'
    for relative in ('op_kernel/fia_block_cube_nonquant_gqa_sink.h','FusedInferAttentionScoreV2Sink.py'):
        shutil.copyfile(native/relative, target/(Path(relative).name+'.txt'))
    shutil.copytree(build/'binary', target/'binary', ignore=ignore)
reference_path = root/'mm2-transfer-build-control-v1/mm2-transfer-build.json'
reference = json.loads(reference_path.read_bytes())
current = json.loads((out/'builds/control/mm2-prefetch-build.json').read_bytes())
binary_name = 'FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673.o'
(out/'control-reference.json').write_text(json.dumps(dict(
    original_build_report_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    old_control_artifacts=reference['artifacts'],
    fresh_control_artifacts=current['artifacts'],
    objects_equal=reference['artifacts'][binary_name]['sha256'] ==
                  current['artifacts'][binary_name]['sha256'],
    scope='Binary identity comparison; if unequal, no claim of byte-identical old control'),indent=2)+'\n')
failures = out/'initial-checks'
failures.mkdir()
for label, source in (('cpu-v1',root/'mm2-prefetch-screen-v1'),
                      ('build-v1',root/'mm2-prefetch-build-control-v1'),
                      ('screen-v2',root/'mm2-prefetch-screen-v2')):
    target = failures/label
    target.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.cpp'):
            shutil.copyfile(path,target/path.name)
files = {}
for path in sorted(out.rglob('*')):
    assert not path.is_symlink()
    if path.is_file():
        files[path.relative_to(out).as_posix()] = dict(bytes=path.stat().st_size,
                                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(out/'manifest.json').write_text(
    json.dumps(dict(files=files, excluded_generated=['__pycache__','*.pyc']),indent=2)+'\n')
archive = root/'mm2-prefetch-collection-v1.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as bundle:
    bundle.add(out,arcname='evidence')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))))
"""
base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "vllm-ascend-030"]
result = subprocess.run(base + ["python -"], input=script.encode(), capture_output=True, timeout=90)
if result.returncode:
    print(result.stderr.decode(errors="replace"))
    raise SystemExit(result.returncode)
metadata = json.loads(result.stdout)
assert metadata["bytes"] < 100 * 1024 * 1024
archive = HERE / "mm2-prefetch-evidence-v1.tar.gz"
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
(OUT / "collection.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
print(json.dumps(dict(output=str(root), files=len(manifest["files"]), archive_bytes=metadata["bytes"])))
