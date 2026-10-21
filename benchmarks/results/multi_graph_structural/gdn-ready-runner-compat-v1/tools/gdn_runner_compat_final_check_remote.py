import hashlib
import json
from pathlib import Path

root = Path('/workspace/gdn-ready-generation-v1')
previous = json.loads((root / 'kernel-identities.json').read_bytes())
checks = {}
for row in previous.values():
    checks[row['path']] = row['sha256']
    checks.update(row['workspace_dependencies_sha256'])
identities = {
    path: dict(expected=expected, actual=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    for path, expected in checks.items()
}
active = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except OSError:
        continue
    if ('torchrun' in command or 'multi_graph_gdn_ready_runner_probe.py' in command) and not command.startswith(('bash', 'python -')):
        active.append(dict(pid=int(path.name), command=command[:400]))
print(json.dumps(dict(
    schema='gdn-ready-runner-frozen-libraries-check-v1',
    checks=identities,
    all_eight_libraries_unchanged=all(v['expected'] == v['actual'] for v in identities.values()),
    active_model_commands=active,
    native_generation_retried=False,
), indent=2))
