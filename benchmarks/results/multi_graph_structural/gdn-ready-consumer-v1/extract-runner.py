import json, os, subprocess, sys
from pathlib import Path
root = Path('/workspace/gdn-ready-consumer-v1')
extensions = dict(extension='/workspace/multi-graph-results/q4-log-build-v3-ranges/gdn_q4_log.so',
    products_extension='/workspace/multi-graph-results/q4-products-build-v9-ranges/gdn_q4_products.so',
    deferred_extension='/workspace/multi-graph-results/q4-log-deferred-build-v1/gdn_q4_log_deferred.so',
    tile_extension='/workspace/multi-graph-results/q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so')
env = dict(os.environ, TASK_QUEUE_ENABLE='0', HCCL_DETERMINISTIC='true', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', ASCEND_RT_VISIBLE_DEVICES='0,1', PYTHONPATH='/workspace/gdn-ready-chain-v1/tools:' + os.environ.get('PYTHONPATH',''))
status = []
for load in ('original','low'):
    destination = root / 'sites' / load
    destination.mkdir(parents=True, exist_ok=False)
    command = ['torchrun','--nproc_per_node=2','--master_port=29861',str(root/'extract.py'),
        '--trace','/workspace/gdn-ready-chain-v1/trace.json','--load',load,'--output',str(destination/'result.json')]
    for name,path in extensions.items(): command += ['--'+name.replace('_','-'),path]
    (destination/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (destination/'worker.log').open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.append(dict(load=load,pid=process.pid,state='running'))
        (root/'extract-status.json').write_text(json.dumps(status,indent=2)+'\n')
        code = process.wait(timeout=900)
    status[-1].update(state='finished',exit_code=code)
    (root/'extract-status.json').write_text(json.dumps(status,indent=2)+'\n')
    if code: sys.exit(code)
