"""Archive the completed runner compatibility audit, without running a model."""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
REPO = WORKSPACE / 'work/vllm-ascend'
DOWNLOAD = WORKSPACE / 'work/gdn-runner-compat-download-v1'
DEST = REPO / 'benchmarks/results/multi_graph_structural/gdn-ready-runner-compat-v1'
DOC = REPO / 'docs/source/developer_guide/Design_Documents/aclgraph_gdn_ready_runner_compat_zh.md'
PYTHON = sys.executable
BENCHMARKS = [
    'benchmarks/multi_graph_gdn_ready_runner_compat.py',
    'benchmarks/multi_graph_gdn_ready_runner_metadata_probe.py',
    'benchmarks/multi_graph_gdn_ready_runner_probe.py',
]
TEST = 'tests/ut/worker/multi_graph/test_gdn_ready_runner_compat.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def run_check(name, args, *, cwd=WORKSPACE):
    result = subprocess.run(args, cwd=cwd, capture_output=True, encoding='utf-8', errors='replace')
    data = result.stdout + result.stderr
    (DEST / (name + '.log')).write_text(data, encoding='utf-8')
    return dict(command=args, cwd=str(cwd), exit_code=result.returncode)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    raw_names = [
        'inventory.json', 'installed-sources.zip',
        'supplement.json', 'installed-supplement.zip',
        'final-sources.json', 'installed-final-sources.zip',
        'native-status.json', 'native-command.json', 'native.log',
        'native.rank0.json', 'native.rank1.json', 'metadata-replay.json',
        'closeout.json', 'hardware-after.txt', 'final-kernel-check.json',
    ]
    for name in raw_names:
        shutil.copyfile(DOWNLOAD / name, DEST / name)
    executed = DEST / 'executed'
    executed.mkdir(exist_ok=True)
    for name in ('multi_graph_gdn_ready_runner_probe.py', 'multi_graph_gdn_ready_runner_metadata_probe.py'):
        shutil.copyfile(DOWNLOAD / name, executed / name)
    tools = DEST / 'tools'
    tools.mkdir(exist_ok=True)
    for name in (
        'gdn_runner_compat_inventory_remote.py', 'gdn_runner_compat_supplement_remote.py',
        'gdn_runner_compat_final_sources_remote.py', 'gdn_runner_compat_launch_remote.py',
        'gdn_runner_compat_status_remote.py', 'gdn_runner_compat_closeout_remote.py',
        'gdn_runner_compat_final_check_remote.py', 'fetch_gdn_runner_compat.py',
        'archive_gdn_runner_compat.py',
    ):
        shutil.copyfile(WORKSPACE / 'work' / name, tools / name)

    verified_sources = {}
    for manifest_name, zip_name in (
        ('inventory.json', 'installed-sources.zip'),
        ('supplement.json', 'installed-supplement.zip'),
        ('final-sources.json', 'installed-final-sources.zip'),
    ):
        info = json.loads((DEST / manifest_name).read_bytes())
        archive_path = DEST / zip_name
        assert sha(archive_path.read_bytes()) == info['archive_sha256']
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == set(info['files'])
            for name, row in info['files'].items():
                data = archive.read(name)
                assert sha(data) == row['sha256']
                assert data == (DOWNLOAD / 'sources' / name).read_bytes()
                if name in verified_sources:
                    assert verified_sources[name] == row['sha256']
                verified_sources[name] = row['sha256']

    excerpt_specs = {
        'vllm/v1/core/sched/scheduler.py': [(650, 670), (808, 828), (1040, 1063), (1260, 1275)],
        'vllm_ascend/worker/model_runner_v1.py': [(1738, 1779), (2720, 2726), (3445, 3472)],
        'vllm_ascend/ops/gdn_attn_builder.py': [(62, 105), (705, 790)],
        'vllm_ascend/ops/gdn.py': [(556, 582), (639, 690), (709, 728)],
        'vllm/v1/attention/backends/utils.py': [(1140, 1178)],
        'vllm/v1/worker/gpu_model_runner.py': [(1214, 1277), (1461, 1481), (1590, 1620)],
        'vllm/v1/worker/mamba_utils.py': [(1385, 1410), (1433, 1455), (1540, 1565)],
        'vllm/v1/core/kv_cache_coordinator.py': [(94, 111), (170, 236)],
        'vllm/v1/core/single_type_kv_cache_manager.py': [(1713, 1749)],
        'vllm_ascend/utils.py': [(958, 965)],
        'vllm_ascend/ascend_config.py': [(1304, 1320)],
    }
    excerpts = {}
    for name, ranges in excerpt_specs.items():
        lines = (DOWNLOAD / 'sources' / name).read_text(encoding='utf-8').splitlines()
        excerpts[name] = dict(
            sha256=verified_sources[name],
            excerpts=[dict(start=start, end=end, lines=lines[start - 1:end]) for start, end in ranges],
        )
    write_json(DEST / 'source-excerpts.json', excerpts)

    ledger = REPO / 'benchmarks/results/multi_graph_structural/gdn-ready-generation-v1/pair0.ready/result.rank0.json'
    command = [
        PYTHON, '-X', 'utf8', str(REPO / BENCHMARKS[0]),
        '--ledger', str(ledger), '--metadata', str(DEST / 'metadata-replay.json'),
        '--root', str(DEST),
    ]
    checks = dict(analysis=run_check('analysis-recompute', command))
    checks['tests'] = run_check('targeted-tests', [
        PYTHON, '-X', 'utf8', '-m', 'pytest', '-q', str(REPO / TEST), '--noconftest',
    ])
    files = [str(REPO / path) for path in [*BENCHMARKS, TEST]]
    checks['ruff'] = run_check('ruff-check', [PYTHON, '-X', 'utf8', '-m', 'ruff', 'check', *files])
    checks['format'] = run_check('ruff-format', [PYTHON, '-X', 'utf8', '-m', 'ruff', 'format', '--check', *files])
    checks['syntax'] = run_check('compile', [PYTHON, '-X', 'utf8', '-m', 'compileall', '-q', *files])
    if (DOWNLOAD / 'format-ci-final-status.json').exists():
        checks['format_ci'] = json.loads((DOWNLOAD / 'format-ci-final-status.json').read_bytes())
        shutil.copyfile(DOWNLOAD / 'format-ci-final.log', DEST / 'format-ci.log')
    else:
        checks['format_ci'] = run_check('format-ci', ['bash', 'format.sh', 'ci'], cwd=REPO)
    write_json(DEST / 'validation.json', checks)
    assert all(row['exit_code'] == 0 for key, row in checks.items() if key != 'format_ci')
    result = json.loads((DEST / 'analysis.json').read_bytes())
    final = json.loads((DEST / 'final-kernel-check.json').read_bytes())
    assert final['all_eight_libraries_unchanged'] and not final['active_model_commands']
    assert result['metadata_sha256'] == json.loads((DEST / 'closeout.json').read_bytes())['metadata_sha256']
    assert result['native_generation_measured'] is False
    assert result['ledger_inventory']['steps'] == 239
    assert result['ledger_inventory']['real_proposed_candidates'] == 2691
    assert result['installed_metadata_replay']['graph_modes'] == {'FULL': 2, 'PIECEWISE': 237}
    assert all(len(json.loads((DEST / f'native.rank{rank}.json').read_bytes())['steps']) == 0 for rank in (0, 1))
    (DEST / '.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof,-blank-at-eol\n', encoding='utf-8')
    write_json(DEST / 'audit.json', dict(
        schema='gdn-ready-runner-compat-local-audit-v1',
        raw_download_bytes_preserved=all((DOWNLOAD / name).read_bytes() == (DEST / name).read_bytes() for name in raw_names),
        unique_installed_source_files_verified=len(verified_sources),
        all_eight_frozen_libraries_unchanged=True,
        no_active_model_commands_at_remote_closeout=True,
        one_native_attempt_zero_generation_steps=True,
        no_model_executed_in_metadata_replay=True,
        ledger_and_metadata_sha_checked=True,
        quality_failures_preserved=True,
        decision=result['decision'],
    ))
    print(json.dumps(dict(archive=str(DEST), checks=checks, source_files=len(verified_sources)), indent=2))


if __name__ == '__main__':
    main()
