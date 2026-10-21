import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path('/workspace/gdn-ready-runner-compat-v1')
status_path = root / 'native-status.json'
if status_path.exists():
    raise FileExistsError('one bounded native probe only; no retry')
status = dict(state='starting', timeout_seconds=900)
status_path.write_text(json.dumps(status, indent=2) + '\n')
env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='0,1', TASK_QUEUE_ENABLE='0',
           HCCL_DETERMINISTIC='true', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
command = ['torchrun', '--nproc_per_node=2', '--master_port=29931',
           str(root / 'multi_graph_gdn_ready_runner_probe.py'), '--root', str(root),
           '--manifest', '/workspace/gdn-ready-generation-v1/workload.json']
(root / 'native-command.json').write_text(json.dumps(dict(command=command, env={k: env[k] for k in
    ('ASCEND_RT_VISIBLE_DEVICES', 'TASK_QUEUE_ENABLE', 'HCCL_DETERMINISTIC', 'OMP_NUM_THREADS',
     'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}), indent=2) + '\n')
started = time.monotonic()
with (root / 'native.log').open('w') as log:
    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status.update(state='running', pid=child.pid)
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        code = child.wait(timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        code = -1
status.update(state='complete' if code == 0 else 'failed_no_retry', exit_code=code,
              elapsed_seconds=time.monotonic() - started)
status_path.write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2))
