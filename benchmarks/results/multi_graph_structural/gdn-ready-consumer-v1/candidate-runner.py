import hashlib, json, os, subprocess, sys, time
from pathlib import Path
root=Path('/workspace/gdn-ready-consumer-v1')
candidate=root/'candidate-build/gdn_q4_tile_pipeline.so'
deadline=time.monotonic()+900
while not (candidate.parent/'source.sha256').exists():
    if time.monotonic()>deadline: raise TimeoutError('candidate build')
    time.sleep(10)
for row in json.loads((root/'diagnostic-status.json').read_bytes()):
    if row.get('exit_code') != 0: raise RuntimeError('diagnostic incomplete or failed')
env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='0', TASK_QUEUE_ENABLE='0', HCCL_DETERMINISTIC='true',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    PYTHONPATH='/workspace/gdn-ready-chain-v1/tools:'+os.environ.get('PYTHONPATH',''))
sites=[str(root/f'sites/{load}/{load}.step{step}.rank{rank}.pt') for rank in (0,1)
       for load,step in (('original',7),('original',31),('low',7))]
libraries={}
for tag,extension in (('old',Path('/workspace/multi-graph-results/q4-tile-build-v3-balanced/gdn_q4_tile_pipeline.so')),('new',candidate)):
    ldd=subprocess.run(['ldd',str(extension)],capture_output=True,text=True,check=True).stdout
    related=[Path(line.split('=>',1)[1].strip().split()[0]) for line in ldd.splitlines()
             if 'gdn_q4_tile_pipeline_kernels' in line and '=>' in line]
    libraries[tag]=dict(extension=str(extension),extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        ldd=ldd,kernels={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in related})
(root/'candidate-binary-identity.json').write_text(json.dumps(libraries,indent=2)+'\n')
rows=[]
for pair in range(3):
    for tag in (('old','new') if pair%2==0 else ('new','old')):
        extension=Path(libraries[tag]['extension'])
        command=[sys.executable,str(root/'probe.py'),'--phase','candidate','--sites',*sites,
            '--output',str(root/f'pair{pair}.{tag}.json'),'--tile-extension',str(extension),
            '--products-extension','/workspace/multi-graph-results/q4-products-build-v9-ranges/gdn_q4_products.so',
            '--deferred-extension','/workspace/multi-graph-results/q4-log-deferred-build-v1/gdn_q4_log_deferred.so']
        with (root/f'pair{pair}.{tag}.log').open('w') as log:
            p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            rows.append(dict(pair=pair,tag=tag,pid=p.pid,state='running',command=command))
            (root/'candidate-status.json').write_text(json.dumps(rows,indent=2)+'\n')
            code=p.wait(timeout=600)
        rows[-1].update(state='finished',exit_code=code)
        (root/'candidate-status.json').write_text(json.dumps(rows,indent=2)+'\n')
        if code: sys.exit(code)
subprocess.run([sys.executable,str(root/'analyze.py'),'--root',str(root),'--output',str(root/'candidate-analysis.json')],check=True)
