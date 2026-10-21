import json
from pathlib import Path

root = Path('/workspace/gdn-ready-runner-compat-v1')
status = json.loads((root / 'native-status.json').read_bytes())
status['ranks'] = []
for rank in (0, 1):
    p = root / f'native.rank{rank}.json'
    if p.exists():
        row = json.loads(p.read_bytes())
        status['ranks'].append({k: row.get(k) for k in ('rank', 'status', 'error', 'effective', 'cache_config')})
        status['ranks'][-1]['steps'] = len(row['steps'])
status['tail'] = (root / 'native.log').read_text(errors='replace').splitlines()[-12:]
print(json.dumps(status, indent=2))
