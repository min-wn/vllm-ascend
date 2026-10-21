import json
import os
import subprocess
import time
from pathlib import Path

root = Path('/workspace/gdn-aime26-q28-bf16-head-control-v1')
pair = Path('/workspace/gdn-aime26-q28-fp32-head-v1')
command = ['torchrun', '--nproc_per_node=2', '--master_port=29845',
    str(root / 'tools/multi_graph_gdn_aime26_fp32_head.py'), 'generate',
    '--variant', 'compact', '--head-dtype', 'bfloat16', '--site', str(pair / 'verified-site'),
    '--output', str(root / 'generation/compact')]
environment = os.environ.copy()
environment.update(ASCEND_RT_VISIBLE_DEVICES='2,3', TASK_QUEUE_ENABLE='0', HCCL_DETERMINISTIC='true',
    PYTHONPATH=str(root / 'tools') + ':' + str(pair / 'tools')
    + ':/workspace/gdn-aime26-q28-divergence-v1/tools:/workspace/gdn-aime26-v1/tools:'
    + environment.get('PYTHONPATH', ''))
(root / 'compact.generate.launch.json').write_text(json.dumps({'command': command, 'devices': '2,3',
    'pid': os.getpid(), 'started': time.time(), 'batch': 1, 'query': 4,
    'hf_overrides': {'head_dtype': 'bfloat16'}, 'independent_continuous_generation': True,
    'performance_measurement': False}, indent=2) + '\n')
result = subprocess.run(command, env=environment)
(root / 'compact.generate.exit.json').write_text(json.dumps({'returncode': result.returncode,
    'ended': time.time()}, indent=2) + '\n')
