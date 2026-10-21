import json, os, subprocess, sys, time
from pathlib import Path
root=Path('/workspace/gdn-ready-consumer-v1')
deadline=time.monotonic()+1800
while True:
    status=json.loads((root/'extract-status.json').read_text())
    if any(s.get('exit_code',0) != 0 for s in status): raise RuntimeError('extraction failed')
    if len(status)==2 and all(s.get('exit_code')==0 for s in status): break
    if time.monotonic()>deadline: raise TimeoutError('extraction wait')
    time.sleep(15)
env=dict(os.environ, TASK_QUEUE_ENABLE='0', HCCL_DETERMINISTIC='true', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', ASCEND_RT_VISIBLE_DEVICES='0', PYTHONPATH='/workspace/gdn-ready-chain-v1/tools:'+os.environ.get('PYTHONPATH',''))
rows=[]
for rank in (0,1):
    command=[sys.executable,str(root/'probe.py'),'--phase','diagnostic','--sites',
        str(root/f'sites/original/original.step7.rank{rank}.pt'),
        str(root/f'sites/original/original.step31.rank{rank}.pt'),
        str(root/f'sites/low/low.step7.rank{rank}.pt'),
        '--output',str(root/f'diagnostic.rank{rank}.json'),
        '--tile-extension','/workspace/multi-graph-results/q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so',
        '--products-extension','/workspace/multi-graph-results/q4-products-build-v9-ranges/gdn_q4_products.so',
        '--deferred-extension','/workspace/multi-graph-results/q4-log-deferred-build-v1/gdn_q4_log_deferred.so']
    p=subprocess.Popen(command,env=env)
    rows.append(dict(rank=rank,pid=p.pid,state='running',command=command))
    (root/'diagnostic-status.json').write_text(json.dumps(rows,indent=2)+'\n')
    code=p.wait()
    rows[-1].update(state='finished',exit_code=code)
    (root/'diagnostic-status.json').write_text(json.dumps(rows,indent=2)+'\n')
    if code: sys.exit(code)
