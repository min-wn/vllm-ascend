"""Read compact experiment progress without starting any new NPU work."""

import shlex
import subprocess

script = r'''
import json
from pathlib import Path
root = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
out = root/'mm2-transfer-screen-v1'
report = out/'mm2-transfer-screen.json'
d = json.loads(report.read_bytes())
result = {k:d.get(k) for k in ('state','stage','returncode','error','cpu_contract','runs')}
for entry in result.get('runs') or []:
    path = out/entry['report_path']
    if path.exists():
        cell = json.loads(path.read_bytes())
        entry['progress'] = {k:cell.get(k) for k in ('state','stage','completed_repeat','returncode','error')}
for name in ('control','wide'):
    path = root/('mm2-transfer-build-'+name+'-v1')/'mm2-transfer-build.json'
    if path.exists():
        build = json.loads(path.read_bytes())
        result['build_'+name] = {k:build.get(k) for k in ('state','stage','returncode','error','binary_qualification')}
if d['state'] == 'failed':
    for name in ('cpu-transfer-contract.log','build-control.log','build-wide.log'):
        p = out/name
        if p.exists(): result[name] = p.read_text(errors='replace')[-4000:]
print(json.dumps(result))
'''
result = subprocess.run([
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "vllm-ascend-030",
    "bash -lc " + shlex.quote("source /etc/profile.d/ascend-multigraph.sh; python -"),
], input=script.encode(), timeout=40)
raise SystemExit(result.returncode)
