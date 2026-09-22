"""Launch exactly the two authorized experiments on GPU0/1; preserve receipts."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess

ROOT=Path('/NHNHOME/data/sukim/adcl')
REPORT=ROOT/'reports/a2_motion_fresh_20260919'
RUNS=ROOT/'work_dirs/a2_motion_fresh_20260919'
PYTHON='/home/<B200-USER>/cv2env/bin/python'
ARMS=((0,'A2-C2F-MOTION'),(1,'A2-FRESH-NUIM'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    assert json.loads((REPORT/'preflight.json').read_text())['status']=='passed'
    phase='smoke' if args.smoke else 'main'
    if not args.smoke:
        assert json.loads((REPORT/'smoke_summary.json').read_text())['status']=='passed'
    receipt=REPORT/f'launch_{phase}.json'
    if receipt.exists():raise RuntimeError('Receipt exists; inspect processes before another launch')
    runtime=REPORT/'runtime';runtime.mkdir(exist_ok=True)
    candidates=[]
    for gpu,arm in ARMS:
        run=RUNS/f"{arm}-s1{'-smoke' if args.smoke else ''}"
        log=runtime/f'{run.name}.log'
        if run.exists() or log.exists():raise RuntimeError('Existing run/log: '+str(run))
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True).strip())
        if used!=0:raise RuntimeError(f'GPU{gpu} is in use ({used} MiB)')
        cmd=[PYTHON,'-u',str(Path(__file__).with_name('train_experiment.py')),
             '--arm',arm,'--gpu',str(gpu),'--run-dir',str(run)]
        if args.smoke:cmd+=['--smoke']
        candidates.append(dict(arm=arm,gpu=gpu,run_dir=str(run),log=str(log),command=cmd,
                               updates=2 if args.smoke else 20554))
    state={'phase':phase,'status':'launching','source_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[]}
    with receipt.open('x') as f:json.dump(state,f,indent=2);f.write('\n')
    for row in candidates:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row['gpu']),OPENBLAS_NUM_THREADS='1',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with open(row['log'],'xb') as log:
            proc=subprocess.Popen(row['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
        row['pid']=proc.pid;state['runs'].append(row)
        temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(receipt)
    state['status']='spawned; verify actual updates before calling healthy'
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(receipt)
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
