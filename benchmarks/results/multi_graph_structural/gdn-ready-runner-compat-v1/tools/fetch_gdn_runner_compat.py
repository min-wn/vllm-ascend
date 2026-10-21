import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent / 'gdn-runner-compat-download-v1'
root.mkdir(exist_ok=True)
for filename in ('inventory.json', 'installed-sources.zip', 'supplement.json', 'installed-supplement.zip',
                 'final-sources.json', 'installed-final-sources.zip',
                 'native-status.json', 'native-command.json', 'native.log', 'native.rank0.json', 'native.rank1.json',
                 'metadata-replay.json', 'closeout.json', 'hardware-after.txt',
                 'multi_graph_gdn_ready_runner_probe.py', 'multi_graph_gdn_ready_runner_metadata_probe.py'):
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                    'vllm-ascend-030:/workspace/gdn-ready-runner-compat-v1/' + filename,
                    str(root / filename)], check=True)
inventory = json.loads((root / 'inventory.json').read_bytes())
assert hashlib.sha256((root / 'installed-sources.zip').read_bytes()).hexdigest() == inventory['archive_sha256']
with zipfile.ZipFile(root / 'installed-sources.zip') as archive:
    for name in archive.namelist():
        destination = (root / 'sources' / name).resolve()
        assert destination.is_relative_to((root / 'sources').resolve())
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == inventory['files'][name]['sha256']
    archive.extractall(root / 'sources')
supplement = json.loads((root / 'supplement.json').read_bytes())
assert hashlib.sha256((root / 'installed-supplement.zip').read_bytes()).hexdigest() == supplement['archive_sha256']
with zipfile.ZipFile(root / 'installed-supplement.zip') as archive:
    for name in archive.namelist():
        destination = (root / 'sources' / name).resolve()
        assert destination.is_relative_to((root / 'sources').resolve())
        assert hashlib.sha256(archive.read(name)).hexdigest() == supplement['files'][name]['sha256']
    archive.extractall(root / 'sources')
final_sources = json.loads((root / 'final-sources.json').read_bytes())
assert hashlib.sha256((root / 'installed-final-sources.zip').read_bytes()).hexdigest() == final_sources['archive_sha256']
with zipfile.ZipFile(root / 'installed-final-sources.zip') as archive:
    for name in archive.namelist():
        destination = (root / 'sources' / name).resolve()
        assert destination.is_relative_to((root / 'sources').resolve())
        assert hashlib.sha256(archive.read(name)).hexdigest() == final_sources['files'][name]['sha256']
    archive.extractall(root / 'sources')
print(json.dumps(dict(files=len(inventory['files']), sha256_verified=True, path=str(root)), indent=2))
