import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

root = Path('/workspace/gdn-ready-runner-compat-v1')
root.mkdir(exist_ok=True)
packages = {}
paths = []
for name in ('vllm', 'vllm_ascend', 'torch', 'torch_npu'):
    location = Path(importlib.util.find_spec(name).origin).resolve()
    packages[name] = dict(path=str(location), version=importlib.metadata.version(name.replace('_', '-')))
    if name not in ('vllm', 'vllm_ascend'):
        continue
    package = location.parent
    patterns = (
        ('v1/worker/*.py', 'v1/spec_decode/*.py', 'v1/core/*.py', 'v1/core/sched/*.py',
         'v1/cudagraph_dispatcher.py', 'config/cache.py', 'v1/engine/core.py',
         'v1/attention/backends/gdn_attn.py', 'model_executor/layers/mamba/*.py',
         'model_executor/layers/mamba/gdn/*.py', 'model_executor/models/qwen3_5.py',
         'model_executor/layers/logits_processor.py', 'config/compilation.py',
         'config/speculative.py', 'v1/kv_cache_interface.py')
        if name == 'vllm' else
        ('worker/*.py', 'ops/gdn*.py', 'attention/attention_v1.py',
         'compilation/*.py', 'patch/worker/patch_*.py', 'platform.py')
    )
    for pattern in patterns:
        paths.extend((p, name + '/' + str(p.relative_to(package))) for p in package.glob(pattern) if p.is_file())

active = []
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        try:
            command = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'torchrun' in command and not command.startswith(('bash', 'python -')):
            active.append(dict(pid=int(p.name), command=command[:300]))

files = {}
with zipfile.ZipFile(root / 'installed-sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p, arc in sorted(set(paths), key=lambda item: item[1]):
        data = p.read_bytes()
        archive.writestr(arc, data)
        files[arc] = dict(source=str(p), sha256=hashlib.sha256(data).hexdigest(), size=len(data))
git = {}
for name in ('vllm', 'vllm_ascend'):
    package = Path(packages[name]['path']).parent
    git[name] = {}
    for label, args in (('head', ['rev-parse', 'HEAD']), ('status', ['status', '--short'])):
        process = subprocess.run(['git', '-C', str(package), *args], capture_output=True, text=True)
        git[name][label] = dict(code=process.returncode, output=process.stdout[:16000])
document = dict(packages=packages, git=git, active=active, files=files,
                archive_sha256=hashlib.sha256((root / 'installed-sources.zip').read_bytes()).hexdigest())
(root / 'inventory.json').write_text(json.dumps(document, indent=2) + '\n')
print(json.dumps({k: document[k] for k in ('packages', 'git', 'active', 'archive_sha256')}, indent=2))
print('source_files', len(files))
