"""Compact read-only progress of the three-way screen."""
import json
import shlex
import subprocess

script = r'''
import json
from pathlib import Path
root = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
out = root/'mm2-double-buffer-screen-v1'
d = json.loads((out/'mm2-double-buffer-screen.json').read_bytes())
result = {k:d.get(k) for k in ('state','stage','returncode','error','cpu_contract')}
result['runs'] = []
for entry in d.get('runs', []):
    row = {k:entry.get(k) for k in ('variant','process_order_index','returncode')}
    path = out/entry['report_path']
    if path.exists():
        c = json.loads(path.read_bytes())
        row.update({k:c.get(k) for k in ('state','stage','completed_repeat','error')})
    result['runs'].append(row)
for name in ('control','wide'):
    p = root/('mm2-double-buffer-build-'+name+'-v1')/'mm2-double-buffer-build.json'
    if p.exists():
        b = json.loads(p.read_bytes())
        result['build_'+name] = {k:b.get(k) for k in ('state','stage','returncode','error','binary_qualification')}
if d['state'] == 'failed':
    for p in out.glob('*.log'):
        result[p.name] = p.read_text(errors='replace')[-5000:]
print(json.dumps(result))
'''
result = subprocess.run([
    'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'vllm-ascend-030',
    'bash -lc ' + shlex.quote('source /etc/profile.d/ascend-multigraph.sh; python -'),
], input=script.encode(), timeout=40)
raise SystemExit(result.returncode)
