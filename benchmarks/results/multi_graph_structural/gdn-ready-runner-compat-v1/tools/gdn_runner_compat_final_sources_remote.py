import hashlib
import json
import zipfile
from pathlib import Path

root = Path('/workspace/gdn-ready-runner-compat-v1')
files = {}
with zipfile.ZipFile(root / 'installed-final-sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('utils.py', 'ascend_config.py', 'attention/utils.py'):
        path = Path('/vllm-workspace/vllm-ascend/vllm_ascend') / name
        data = path.read_bytes()
        key = 'vllm_ascend/' + name
        archive.writestr(key, data)
        files[key] = dict(source=str(path), size=len(data), sha256=hashlib.sha256(data).hexdigest())
summary = dict(files=files, archive_sha256=hashlib.sha256((root / 'installed-final-sources.zip').read_bytes()).hexdigest())
(root / 'final-sources.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
