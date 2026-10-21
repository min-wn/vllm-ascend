import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

root = Path('/workspace/gdn-aime26-q28-bf16-head-control-v1')
pair = Path('/workspace/gdn-aime26-q28-fp32-head-v1')
old = Path('/workspace/gdn-aime26-v1')
sys.path[:0] = [str(root / 'tools'), str(pair / 'tools'), str(old / 'tools'),
                '/workspace/gdn-aime26-q28-divergence-v1/tools']
while not (root / 'compact.generate.exit.json').exists():
    time.sleep(30)
exit_record = json.loads((root / 'compact.generate.exit.json').read_bytes())
identity = json.loads((root / 'identity-before.json').read_bytes())
for filename, expected in identity['frozen_tools'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
for filename, record in json.loads((old / 'runtime-binaries.json').read_bytes()).items():
    path = Path(filename)
    assert path.stat().st_size == record['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], filename
for filename, record in json.loads((old / 'weight-sha256.json').read_bytes()).items():
    actual = (Path('/models/Qwen3.5-27B-original') / filename).stat()
    assert (actual.st_size, actual.st_mtime_ns) == (record['bytes'], record['mtime_ns']), filename
(root / 'identity-after.json').write_text(json.dumps({'frozen_tools_match': True,
    'frozen_binaries_match': True, 'weight_sizes_and_mtime_match': True}, indent=2) + '\n')
if exit_record['returncode'] == 0:
    from multi_graph_gdn_aime26 import load_manifest
    from multi_graph_gdn_aime26_audit import audit_group
    from multi_graph_gdn_aime26_divergence import common_prefix
    from multi_graph_gdn_aime26_fp32_head import real_projection
    from transformers import AutoTokenizer

    manifest = load_manifest(old / 'manifest.json')
    directory = root / 'generation/compact'
    audit = audit_group(manifest, directory / 'compact.shard0.group0.result.json')
    (root / 'lifecycle-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    record = json.loads((directory / 'compact.question27.json').read_bytes())
    for rank in range(2):
        head = json.loads((directory / f'compact.head-audit.rank{rank}.json').read_bytes())
        assert head['calls'] == record['q4_steps'] + 1 and head['fallback_calls'] == 0
        assert head['weight_dtype'] == head['output_dtype'] == head['configured_head_dtype'] == 'torch.bfloat16'
        assert head['allow_hf32'] is False and head['float32_matmul_precision'] == 'highest'
        assert [item['rows'] for item in head['audited_calls']] == [1, 4]
        assert all(real_projection(item['operations'], 'torch.bfloat16') for item in head['audited_calls'])
    records = {'compact_bf16': record}
    for variant in ['official', 'compact']:
        path = pair / 'generation' / variant / (variant + '.question27.json')
        records[variant + '_fp32'] = json.loads(path.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.5-27B-original')
    pairs = [('official_fp32', 'compact_fp32'), ('compact_bf16', 'compact_fp32'),
             ('official_fp32', 'compact_bf16')]
    text = {}
    for left, right in pairs:
        sequences = {key: records[key]['generated_tokens'] for key in [left, right]}
        prefix = common_prefix(*sequences.values())
        row = {'common_output_prefix': prefix, 'variants': {}}
        if prefix < min(map(len, sequences.values())):
            for key, ids in sequences.items():
                row['variants'][key] = {'token': ids[prefix], 'text': tokenizer.decode([ids[prefix]]),
                    'context': tokenizer.decode(ids[max(0, prefix - 20):prefix + 21])}
        text[left + '__' + right] = row
    (root / 'token-text.json').write_text(json.dumps(text, ensure_ascii=False, indent=2) + '\n')
files = sorted(p for p in root.rglob('*') if p.is_file()
    and '__pycache__' not in p.parts and p.suffix != '.gz'
    and p.name not in ['completion.json', 'generation-evidence-identity.json', 'collect-generation.log'])
archive = root / 'generation-evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as stream:
    for path in files:
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
archive_identity = {'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'files': {str(path.relative_to(root)): {'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files}}
(root / 'generation-evidence-identity.json').write_text(json.dumps(archive_identity, indent=2) + '\n')
(root / 'completion.json').write_text(json.dumps({'status': 'control_complete'
    if exit_record['returncode'] == 0 else 'runtime_failure_preserved',
    'exit': exit_record, 'archive_sha256': archive_identity['sha256'], 'ended': time.time()}, indent=2) + '\n')
print('control collection completed', flush=True)
