"""Compact read-only progress of the ABBA screen."""

import shlex
import subprocess

script = r"""
import json
from pathlib import Path
root = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
out = root/'mm2-prefetch-screen-v3'
d = json.loads((out/'mm2-prefetch-screen.json').read_bytes())
result = {k:d.get(k) for k in ('state','stage','returncode','error')}
result['cpu_passed'] = d.get('cpu_contract',{}).get('passed')
result['runs'] = []
for entry in d.get('runs', []):
    row = {k:entry.get(k) for k in ('variant','process_order_index','returncode')}
    path = out/entry['report_path']
    if path.exists():
        c = json.loads(path.read_bytes())
        row.update({k:c.get(k) for k in ('state','stage','completed_repeat','error')})
    result['runs'].append(row)
for name in ('control','candidate'):
    p = root/('mm2-prefetch-build-'+name+'-v3')/'mm2-prefetch-build.json'
    if p.exists():
        b = json.loads(p.read_bytes())
        result['build_'+name] = {k:b.get(k) for k in ('state','stage','returncode','error','binary_qualification')}
if d['state'] == 'failed':
    for p in out.glob('build-*.log'):
        result[p.name] = p.read_text(errors='replace')[-1800:]
print(json.dumps(result))
"""
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
    timeout=40,
)
raise SystemExit(result.returncode)
