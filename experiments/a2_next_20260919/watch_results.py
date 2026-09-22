"""Finish bounded diagnostics/reports when the three agreed runs terminate.

Never starts training, changes budgets, launches FULL, or uploads submissions.
Only GPU3 is used, when idle, for fixed read-only inference probes.
"""
import datetime
import json
import os
import subprocess
import time
import traceback
from common import *

PYTHON='/home/<B200-USER>/cv2env/bin/python'

def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)

def publish(phase):
    # Existing session authorization covers experiment records and GitHub mirror.
    # Do not absorb anyone else's staged changes or perform resets/conflict edits.
    if command(['git','diff','--cached','--name-only']).strip():
        raise RuntimeError('Index contains other staged work; results saved, automatic publishing stopped')
    names=(['G_terminal_results.json','G_artifact_index.json','G_results.csv','G_RESULTS_KO.md'] if phase=='G'
        else ['terminal_results.json','run_artifact_index.json','results.csv','RESULTS_KO.md'])
    arms=['PARENT','BASE',*G_ARMS] if phase=='G' else [S_ARM]
    names += [f'posttrain_probe_{a}.json' for a in arms]
    paths=[str((REPORT/n).relative_to(ROOT)) for n in names]
    command(['git','add','--',*paths]);command(['git','diff','--cached','--check'])
    if not command(['git','diff','--cached','--name-only']).strip():return {'already_published':True}
    print(command(['git','commit','-m',f'Record A2 {phase} terminal comparison and fixed probes']),flush=True)
    work=command(['git','rev-parse','HEAD']).strip()
    mirror=Path('/home/<B200-USER>/edrive_mirror')
    assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
    if command(['git','status','--porcelain'],mirror).strip():raise RuntimeError('Mirror is not clean; publishing stopped')
    print(command(['git','fetch',str(ROOT),work],mirror),flush=True)
    print(command(['git','cherry-pick',work],mirror),flush=True)
    print(command(['git','push','github','HEAD:refs/heads/motiondrive-v2-20260910'],mirror),flush=True)
    mirrored=command(['git','rev-parse','HEAD'],mirror).strip()
    assert command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]==mirrored
    return {'work_commit':work,'mirror_commit':mirrored}

def probe(arm):
    target=REPORT/f'posttrain_probe_{arm}.json'
    if target.exists():return
    while int(command(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())>=1000:
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    log=REPORT/'runtime'/f'posttrain_probe_{arm}.log'
    with log.open('xb') as stream:
        subprocess.run([PYTHON,str(HERE/'posttrain_probe.py'),'--arm',arm],cwd=ROOT,env=env,
          stdout=stream,stderr=subprocess.STDOUT,check=True)

def completed(arm):
    run=RUNS/f'{arm}-s1';m=json.loads((run/'manifest.json').read_text())
    if m['status']=='completed':return (run/'experiment.json').exists()
    proc=Path(f"/proc/{m['pid']}/stat")
    if not proc.exists() or proc.read_text().split()[2]=='Z':
        raise RuntimeError(f"{arm} process ended with manifest status {m['status']}; inspect its log")
    return False

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Watcher itself must not allocate a GPU'
    state={'pid':os.getpid(),'status':'running','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'training_started_by_watcher':False,'FULL_or_submission_actions':False,'phases':{}}
    path=REPORT/'watcher_status.json'
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        probe('PARENT');probe('BASE')
        while not all(completed(a) for a in G_ARMS):time.sleep(30)
        for a in G_ARMS:probe(a)
        print(command([PYTHON,str(HERE/'review_results.py'),'--scope','G']),flush=True)
        state['phases']['G']=publish('G');save()
        while not completed(S_ARM):time.sleep(30)
        probe(S_ARM)
        print(command([PYTHON,str(HERE/'review_results.py'),'--scope','all']),flush=True)
        state['phases']['all']=publish('all')
        state['status']='completed';state['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    except Exception:
        state['status']='failed';state['error']=traceback.format_exc();save();raise

if __name__=='__main__':main()
