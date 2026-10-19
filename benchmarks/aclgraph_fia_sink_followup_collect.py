# SPDX-License-Identifier: Apache-2.0
"""Collect the completed page-reuse and optimistic overlap screens, preserving bytes."""

import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

REMOTE = r"""
import hashlib,json,tarfile
from pathlib import Path
root=Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
targets={
 'page-screen':root/'page-reuse-screen-v1',
 'overlap-forward':root/'sink-overlap-screen-v1',
 'overlap-reverse':root/'sink-overlap-screen-v2',
}
for name,path in targets.items():
    report=path/('page-reuse-screen.json' if name=='page-screen' else 'overlap.json')
    assert json.loads(report.read_bytes())['state']=='completed'
archive=root/'page-overlap-followup-v1.tar.gz'
assert not archive.exists()
files=[]
for name,path in targets.items():
    files.extend((p,name+'/'+p.relative_to(path).as_posix()) for p in path.rglob('*')
                 if p.is_file() and p.name!='page-contract')
for variant in ('control','candidate'):
    path=root/('page-reuse-build-'+variant+'-v1')
    for relative in ('page-reuse-build.json','compile.log','patch.log',
                     'source/page-reuse-patch.json','source/page-reuse-source.patch'):
        files.append((path/relative,'build-'+variant+'/'+relative))
for name,folder in (('tools-v1','page-reuse-tools-v1'),('tools-v2','sink-overlap-tools-v2')):
    path=root/folder
    for p in path.iterdir():
        if p.is_file() and p.suffix in ('.py','.log','.json'):
            files.append((p,name+'/'+p.name))
manifest={name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p,name in files}
manifest_path=root/'page-overlap-followup-v1-manifest.json'
assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as out:
    for p,name in files: out.add(p,arcname=name,recursive=False)
    out.add(manifest_path,arcname='manifest.json',recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-controls", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    script = REMOTE
    if args.planner_controls:
        begin = script.index("targets={")
        end = script.index("for name,path in targets.items():")
        script = script[:begin] + "targets={'planner-controls':root/'sink-overlap-screen-v3'}\n" + script[end:]
        begin = script.index("for variant in ('control','candidate'):")
        end = script.index("for name,folder in", begin)
        script = script[:begin] + script[end:]
        script = script.replace(
            "(('tools-v1','page-reuse-tools-v1'),('tools-v2','sink-overlap-tools-v2'))",
            "(('tools-v3','sink-overlap-tools-v3'),)",
        )
        script = script.replace("page-overlap-followup-v1", "page-overlap-planner-v1")
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
        capture_output=True,
        check=True,
        timeout=60,
    )
    remote = json.loads(result.stdout)
    archive = args.output / "raw.tar.gz"
    subprocess.run(["scp", "-q", "vllm-ascend-030:" + remote["path"], str(archive)], check=True, timeout=60)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != remote["sha256"]:
        raise ValueError("download hash mismatch")
    raw = args.output / "raw"
    with tarfile.open(archive) as bundle:
        bundle.extractall(raw, filter="data")
    manifest = json.loads((raw / "manifest.json").read_bytes())
    for name, identity in manifest.items():
        path = raw / name
        if (
            path.stat().st_size != identity["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]
        ):
            raise ValueError("raw file identity mismatch: " + name)
    (args.output / "collection.json").write_text(json.dumps(remote, indent=2) + "\n")
    print(json.dumps(remote))


if __name__ == "__main__":
    main()
