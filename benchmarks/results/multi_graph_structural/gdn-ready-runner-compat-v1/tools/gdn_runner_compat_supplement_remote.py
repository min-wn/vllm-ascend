import hashlib
import importlib.metadata
import importlib.util
import json
import zipfile
from pathlib import Path

root = Path('/workspace/gdn-ready-runner-compat-v1')
paths = [
    (Path('/vllm-workspace/vllm/vllm') / p, 'vllm/' + p)
    for p in ('v1/attention/backends/utils.py', 'forward_context.py',
              'model_executor/models/qwen3_next.py', 'config/model.py')
]
fla = Path(importlib.util.find_spec('fla_npu').origin).parent
for p in fla.rglob('*.py'):
    if 'recurrent' in p.name or 'causal_conv' in p.name:
        paths.append((p, 'fla_npu/' + str(p.relative_to(fla))))
files = {}
with zipfile.ZipFile(root / 'installed-supplement.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p, name in paths:
        if p.is_file():
            data = p.read_bytes()
            archive.writestr(name, data)
            files[name] = dict(source=str(p), sha256=hashlib.sha256(data).hexdigest(), size=len(data))
summary = dict(fla_path=str(fla), files=files,
               archive_sha256=hashlib.sha256((root / 'installed-supplement.zip').read_bytes()).hexdigest())
(root / 'supplement.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
