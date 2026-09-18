"""Launch only the three agreed experiments after all preflights finish."""
import datetime
import json
import os
import subprocess
from common import *

def main():
    assert not (REPORT/'launch_receipt.json').exists()
    tests=json.loads((REPORT/'tests_summary.json').read_text());assert tests['status']=='passed'
    assert json.loads((REPORT/'gradient_probe.json').read_text())['optimizer_updates']==0
    for arm in (*G_ARMS,S_ARM):
        m=json.loads((RUNS/f'{arm}-s1-smoke'/'manifest.json').read_text())
        assert m['status']=='completed' and m['step']==2 and m['nonfinite_count']==0
    records=[]
    for gpu,arm in [(0,'A2-G0'),(1,'A2-G1'),(2,S_ARM)]:
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert memory<1000,f'GPU {gpu} has another allocation; do not stop it'
        run=RUNS/f'{arm}-s1'
        assert not run.exists()
        logfile=REPORT/'runtime'/f'{arm}-s1.log'
        args=['/home/korea_sdv01/cv2env/bin/python','-u',str(HERE/'train_next.py'),
              '--arm',arm,'--gpu',str(gpu),'--run-dir',str(run)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with logfile.open('xb') as stream:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        records.append({'arm':arm,'gpu':gpu,'pid':process.pid,'command':args,'run':str(run),
          'log':str(logfile),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    value={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'runs':records,'GPU3':'diagnostics, no extra training arm','GPU4_7':'excluded'}
    (REPORT/'launch_receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2),flush=True)

if __name__=='__main__':main()
