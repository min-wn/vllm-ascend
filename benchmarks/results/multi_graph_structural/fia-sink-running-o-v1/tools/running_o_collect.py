"""Collect bounded running-O raw evidence; never launches NPU work or builds.

Each version gets a fresh output directory. Interrupted/failed runs remain
explicitly partial. Native raw profiler dumps, source trees and object files
are inventoried but omitted; generated Chrome traces and textual reports stay.
"""

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath

REMOTE = r'''
import hashlib,io,json,sys,tarfile,time
from pathlib import Path

base=Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e').resolve(strict=True)
screen=base/f'running-o-screen-v{VERSION}'
tools=base/f'running-o-tools-v{VERSION}'
suite_path=screen/'running-o-screen.json'
suite_before=suite_path.read_bytes()
suite=json.loads(suite_before)
interrupted=(screen/'controller-interruption.json').is_file()
if suite.get('state') not in ('completed','failed','interrupted') and not interrupted and not ALLOW_PARTIAL:
    raise ValueError('Active run requires --allow-partial; interrupted runs need controller-interruption.json')
complete=suite.get('state')=='completed' and suite.get('returncode')==0 and not interrupted
files={}
skipped=[]
def add(path,name):
    path=path.resolve(strict=True)
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError('Evidence path escapes isolated experiment')
    if name in files:
        if files[name]!=path:
            raise ValueError('Duplicate evidence destination')
        return
    files[name]=path

def omitted(path,reason):
    skipped.append(dict(remote_path=path.relative_to(base).as_posix(),bytes=path.stat().st_size,reason=reason))

def keep_screen_file(path):
    parts=path.relative_to(screen).parts
    lowered=[part.lower() for part in parts]
    if any(part in ('__pycache__','kernel_meta','framework','device','host','data','analyze') or part.startswith(('device_','host_')) for part in lowered):
        return False
    if path.name in ('trace.json','trace_view.json','kernel_details.csv','op_summary.csv','op_statistic.csv','memory_record.csv','operator_details.csv','api_statistic.csv','step_trace_time.csv','npu_module_mem.csv'):
        return True
    if 'output' in lowered:
        return path.suffix.lower() in ('.json','.csv','.log','.txt')
    return path.suffix.lower() in ('.json','.csv','.log','.txt','.py','.md')

for path in sorted(screen.rglob('*')):
    if path.is_file():
        if keep_screen_file(path):
            add(path,path.relative_to(screen).as_posix())
        else:
            omitted(path,'profiler raw/binary/cache or unselected file type')
if not tools.is_dir():
    raise ValueError('Frozen tools directory is missing')
for path in sorted(tools.iterdir()):
    if path.is_file() and path.suffix.lower() in ('.py','.json','.log','.txt'):
        add(path,'deployed-tools/'+path.name)

build_presence=[]
for variant in ('control','resident'):
    build=base/f'running-o-build-{variant}-v{VERSION}'
    prefix='builds/'+variant+'/'
    present=build.is_dir()
    build_presence.append(dict(variant=variant,present=present,remote_path=str(build)))
    if not present:
        if complete:
            raise ValueError('Completed screen is missing selected build')
        continue
    for name in ('running-o-build.json','patch.log','compile.log','running-o-ub-patch.json','running-o-ub-source.patch'):
        path=build/name
        if path.is_file():
            add(path,prefix+name)
    for name in ('running-o-ub-patch.json','running-o-ub-source.patch'):
        path=build/'source'/name
        if path.is_file():
            add(path,prefix+'patch-evidence/'+name)
    source=build/'source/csrc/attention/fused_infer_attention_score_v2_sink'
    for relative in ('op_kernel/fia_block_vec_nonquant_sink.h','FusedInferAttentionScoreV2Sink.py'):
        path=source/relative
        if path.is_file():
            # Preserve exact bytes; extension identifies a source snapshot only.
            add(path,prefix+'source-snapshots/'+relative+'.txt')
        elif complete:
            raise ValueError('Completed build is missing exact source snapshot')
    binary=build/'binary'
    if binary.is_dir():
        for path in sorted(binary.rglob('*')):
            if path.is_file():
                if path.suffix.lower()=='.json' and 'kernel_meta' not in path.parts:
                    label=hashlib.sha256(path.relative_to(binary).as_posix().encode()).hexdigest()[:12]+'.json'
                    add(path,prefix+'binary-json/'+label)
                else:
                    omitted(path,'selected compile object/binary; hash retained in build report')
    # Gather dispatcher JSONs to prove only metadata kernelSo differs; no .so/.o copy.
    for policy in ('upstream','whole'):
        vendor=build/('vendor-'+policy)
        for relative in ('op_impl/cpu/config/cust_aicpu_kernel.json',
                         'op_impl/ai_core/tbe/kernel/ascend910b/fused_infer_attention_score_v2_sink/FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673.json'):
            path=vendor/relative
            if path.is_file():
                label='dispatch.json' if relative.endswith('cust_aicpu_kernel.json') else 'binary-metadata.json'
                add(path,prefix+'vendor-'+policy+'/'+label)

# The frozen probe explicitly exports these paths. Never guess a profiler dump
# as its replacement: a missing referenced trace invalidates a completed bundle.
for run in suite.get('runs',[]):
    relative=run.get('report_path')
    if not relative:
        continue
    report_path=(screen/relative).resolve(strict=False)
    if not report_path.is_relative_to(screen):
        raise ValueError('Cell report escapes selected screen')
    if not report_path.is_file():
        if complete:
            raise ValueError('Missing completed cell report')
        continue
    report=json.loads(report_path.read_bytes())
    for profile in report.get('profiles',[]):
        path=(report_path.parent/profile['path']).resolve(strict=False)
        if not path.is_relative_to(screen) or not path.is_file():
            if complete:
                raise ValueError('Missing referenced actual Chrome trace')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=profile['sha256']:
            raise ValueError('Referenced Chrome trace byte hash changed')
        add(path,path.relative_to(screen).as_posix())

if suite_path.read_bytes()!=suite_before:
    raise ValueError('Screen status changed during collection; retry after it stops')
collection=dict(version=VERSION,screen_state=suite.get('state'),screen_returncode=suite.get('returncode'),
                controller_interrupted=interrupted,partial=not complete,
                collection_does_not_qualify_results=True,npu_work_executed_by_collector=False,
                collected_unix=time.time(),selected_screen_remote=str(screen),
                frozen_tools_remote=str(tools),build_presence=build_presence,
                suite_sha256=hashlib.sha256(suite_before).hexdigest(),
                omitted_files=skipped,
                profile_path_contract='cell-XX-VARIANT-POLICY/profiles/cap{400,512}/trace.json; optional existing output/trace_view.json and kernel_details.csv retained',
                omission_scope='No full copied source, vendor libraries, AICore objects, profiler raw device/framework dumps or kernel_meta')
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for name,path in sorted(files.items()):
        data=path.read_bytes()
        row=dict(path=name,remote_path=str(path.relative_to(base)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        manifest.append(row)
        info=tarfile.TarInfo(name);info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    data=(json.dumps(collection,indent=2)+'\n').encode()
    manifest.append(dict(path='collection.json',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),generated_by_collector=True))
    info=tarfile.TarInfo('collection.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
    data=(json.dumps(manifest,indent=2)+'\n').encode()
    info=tarfile.TarInfo('manifest.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=range(1, 10), required=True)
    parser.add_argument("--output", type=Path, help="New local evidence directory; defaults to running-o-evidence-vN")
    parser.add_argument("--allow-partial", action="store_true", help="Permit a stopped partial run whose suite still says running")
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parent / f"running-o-evidence-v{args.version}"
    output = output.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing evidence directory")
    payload = f"VERSION={args.version}\nALLOW_PARTIAL={args.allow_partial!r}\n" + REMOTE
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "vllm-ascend-030",
               "bash -lc " + shlex.quote("source /etc/profile.d/ascend-multigraph.sh; python -")]
    # Buffer the bounded compressed archive so failed remote collection cannot
    # leave a directory that looks like complete evidence.
    result = subprocess.run(command, input=payload.encode("utf-8"), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=180)
    if result.returncode:
        raise RuntimeError("Remote read-only collection failed: " + result.stderr.decode("utf-8", errors="replace"))
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts or PureWindowsPath(member.name).drive or "\\" in member.name or member.name in entries:
                raise ValueError("Unsafe or duplicate tar evidence path")
            entries[member.name] = archive.extractfile(member).read()
    manifest = json.loads(entries["manifest.json"])
    if {row["path"] for row in manifest} != set(entries) - {"manifest.json"} or len(manifest) != len(entries) - 1:
        raise ValueError("Manifest does not cover the exact collected file set")
    for item in manifest:
        data = entries[item["path"]]
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Collected evidence byte hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in entries.items():
        path = output / name
        if not path.resolve().is_relative_to(output):
            raise ValueError("Evidence extraction escapes output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    collection = json.loads(entries["collection.json"])
    print(json.dumps(dict(output=str(output), files=len(manifest), bytes=sum(row["bytes"] for row in manifest),
                          manifest_sha256=hashlib.sha256(entries["manifest.json"]).hexdigest(),
                          screen_state=collection["screen_state"], partial=collection["partial"]), indent=2))


if __name__ == "__main__":
    main()
