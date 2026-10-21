"""Finalize saved evidence and user-readable copies without more measurements."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / 'work/vllm-ascend'
DEST = REPO / 'benchmarks/results/multi_graph_structural/gdn-ready-runner-compat-v1'
DOWNLOAD = ROOT / 'work/gdn-runner-compat-download-v1'
DOC = REPO / 'docs/source/developer_guide/Design_Documents/aclgraph_gdn_ready_runner_compat_zh.md'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    shutil.copyfile(DOWNLOAD / 'format-ci-final.log', DEST / 'format-ci.log')
    checks = json.loads((DEST / 'validation.json').read_bytes())
    checks['format_ci'] = json.loads((DOWNLOAD / 'format-ci-final-status.json').read_bytes())
    write_json(DEST / 'validation.json', checks)
    for name in ('archive_gdn_runner_compat.py', 'finalize_gdn_runner_compat.py'):
        shutil.copyfile(ROOT / 'work' / name, DEST / 'tools' / name)
    text = DOC.read_text(encoding='utf-8')
    assert '实际 runner 兼容性核验' in text
    assert '\ufffd' not in text
    (DEST / 'report.md').write_text(text, encoding='utf-8')
    mirrors = []
    for suffix in ('.md', '.txt'):
        path = ROOT / ('GDN_ready_runner兼容性核验_20261021' + suffix)
        path.write_text(text, encoding='utf-8-sig')
        assert path.read_text(encoding='utf-8-sig') == text
        assert path.stat().st_size > 10000
        mirrors.append(dict(path=path.name, bytes=path.stat().st_size, utf8_bom=path.read_bytes().startswith(b'\xef\xbb\xbf')))
    source_paths = [
        'benchmarks/multi_graph_gdn_ready_runner_probe.py',
        'benchmarks/multi_graph_gdn_ready_runner_metadata_probe.py',
        'benchmarks/multi_graph_gdn_ready_runner_compat.py',
        'tests/ut/worker/multi_graph/test_gdn_ready_runner_compat.py',
        'docs/source/developer_guide/Design_Documents/aclgraph_gdn_ready_runner_compat_zh.md',
    ]
    for relative in source_paths:
        path = REPO / relative
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    write_json(DEST / 'source-identities.json', {
        p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in source_paths
    })
    manifest = {}
    for path in sorted(DEST.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            data = path.read_bytes()
            manifest[path.relative_to(DEST).as_posix()] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    write_json(DEST / 'manifest.json', dict(
        schema='gdn-ready-runner-compat-archive-manifest-v1',
        excludes_self=True,
        files=manifest,
        readable_workspace_mirrors=mirrors,
        source_sha_scope='local checkout bytes; executed remote probe bytes separately under executed/',
    ))
    print(json.dumps(dict(files=len(manifest), mirrors=mirrors, formal_checks=checks['format_ci']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
