import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/workspace/gdn-ready-runner-compat-v1')
previous = json.loads(Path('/workspace/gdn-ready-generation-v1/pair0.ready/result.rank0.json').read_bytes())
identities = {}
for filename, expected in previous['kernel_identities'].items():
    path = Path(filename)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    identities[filename] = dict(expected=expected, actual=actual, unchanged=expected == actual)
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
hardware = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
(root / 'hardware-after.txt').write_text(hardware.stdout + hardware.stderr)
report = dict(
    schema='gdn-ready-runner-compat-closeout-v1',
    native_status=json.loads((root / 'native-status.json').read_bytes()),
    native_steps={str(rank): len(json.loads((root / f'native.rank{rank}.json').read_bytes())['steps']) for rank in (0, 1)},
    kernel_identities=identities,
    all_frozen_kernels_unchanged=all(row['unchanged'] for row in identities.values()),
    active_model_commands=active,
    npu_smi_exit_code=hardware.returncode,
    metadata_sha256=hashlib.sha256((root / 'metadata-replay.json').read_bytes()).hexdigest(),
    no_model_generation_retried=True,
)
(root / 'closeout.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
