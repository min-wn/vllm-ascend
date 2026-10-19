"""Two diagnostic processes; no performance inference from their single sample."""
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e')
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from aclgraph_fia_sink_stage7_suite import run_one

OUT = ROOT / 'mm2-prefetch-counters-screen-v1'
assert not OUT.exists()
OUT.mkdir()
REPORT = OUT / 'counter-diagnostic.json'
status = dict(state='running', runs=[], started_unix=time.time(),
              timing_samples_are_diagnostic=True, performance_gain_measured=False,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in TOOLS.glob('*.py')},
              metric='PipeUtilization', profile_level=1, policy='whole', capacities=[400,512])


def save():
    REPORT.write_text(json.dumps(status,indent=2)+'\n')


save()
try:
    for index,variant in enumerate(('control','candidate')):
        info=subprocess.run(['npu-smi','info'],stdout=subprocess.PIPE,text=True,check=True).stdout
        (OUT/f'device-{index}.txt').write_text(info)
        assert 'No running processes found in NPU 0' in info
        output=OUT/f'cell-{index}-{variant}'
        options=dict(root=ROOT,opp_root=ROOT/f'mm2-prefetch-build-{variant}-v3/vendor-whole',
                     output_dir=output,trace=TOOLS/'structural-dynamic-trace-v1.json',
                     selection=TOOLS/'q4-tile-full-replay-v8-queue0.rank0.json',mode='whole',trial=0,
                     repeats=1,warmups=1,profile_repeats=1,seed=1703,timeout=900)
        command=[sys.executable,str(TOOLS/'aclgraph_fia_sink_consumer_probe.py')]
        for key,value in options.items():
            command.extend(['--'+key.replace('_','-'),str(value)])
        entry=dict(variant=variant,launched_unix=time.time(),command=command)
        status['runs'].append(entry)
        status['stage']=variant
        save()
        rc=run_one(command,OUT/f'cell-{index}.log',900,entry,save)
        entry.update(returncode=rc,finished_unix=time.time())
        save()
        assert rc==0
        result=json.loads((output/'consumer-probe.json').read_bytes())
        assert result['state']=='completed'
        entry['profile_files']=[p.relative_to(OUT).as_posix() for p in output.rglob('*.csv')]
    status.update(state='completed',returncode=0,finished_unix=time.time())
except BaseException:
    status.update(state='failed',returncode=1,error=traceback.format_exc(),finished_unix=time.time())
    traceback.print_exc()
save()
raise SystemExit(status['returncode'])
