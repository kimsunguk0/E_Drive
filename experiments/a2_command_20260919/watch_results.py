"""Complete one command trial: fixed inference, result report and authorized mirror commit."""
import datetime
import json
import os
import subprocess
import time
import traceback
from command_common import *

PYTHON='/home/korea_sdv01/cv2env/bin/python'

def call(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)

def alive(pid):
    stat=Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().split()[2]!='Z'

def publish():
    if call(['git','diff','--cached','--name-only']).strip():
        raise RuntimeError('Other staged work exists; reports saved, automatic commit stopped')
    names=['terminal_results.json','run_artifact_index.json','results.csv','RESULTS_KO.md','command_counterfactual.json']
    paths=[str((REPORT/name).relative_to(ROOT)) for name in names]
    # This bounded publisher stages only generated result files, never source/credentials.
    for name in names:
        body=(REPORT/name).read_text()
        assert 'PRIVATE KEY' not in body and 'ghp_' not in body
    call(['git','add','--',*paths]);call(['git','diff','--cached','--check'])
    if not call(['git','diff','--cached','--name-only']).strip():
        return {'already_published':True}
    print(call(['git','commit','-m','Record terminal A2 semantic command comparison']),flush=True)
    work=call(['git','rev-parse','HEAD']).strip()
    mirror=Path('/home/korea_sdv01/edrive_mirror')
    assert call(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
    if call(['git','status','--porcelain'],mirror).strip():
        raise RuntimeError('Mirror is not clean; no reset or automatic conflict resolution')
    print(call(['git','fetch',str(ROOT),work],mirror),flush=True)
    print(call(['git','cherry-pick',work],mirror),flush=True)
    print(call(['git','push','github','HEAD:refs/heads/motiondrive-v2-20260910'],mirror),flush=True)
    mirrored=call(['git','rev-parse','HEAD'],mirror).strip()
    assert call(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]==mirrored
    return dict(work_commit=work,mirror_commit=mirrored)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    state=dict(pid=os.getpid(),status='waiting',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        starts_training=False,starts_FULL=False,uploads_submission=False)
    path=REPORT/'watcher_status.json'
    def save():
        path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        receipt=json.loads((REPORT/'launch_receipt.json').read_text());run=Path(receipt['run_dir'])
        while True:
            manifest=run/'manifest.json'
            if manifest.exists():
                value=json.loads(manifest.read_text())
                if value['status']=='completed' and (run/'experiment.json').exists():
                    break
            if not alive(receipt['pid']):
                raise RuntimeError('Training process exited without a completed artifact set')
            time.sleep(30)
        state['status']='terminal_report';save()
        print(call([PYTHON,str(HERE/'review_results.py')]),flush=True)
        # Report is already persisted; bounded idle wait cannot delay its availability.
        state['status']='waiting_GPU3_for_fixed_inference';save()
        deadline=time.monotonic()+3600
        while int(call(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())>=1000:
            if time.monotonic()>deadline:
                raise RuntimeError('GPU3 remained occupied for 1h; terminal report is saved; no job interrupted')
            time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (REPORT/'runtime'/'command_probe.log').open('xb') as log:
            subprocess.run([PYTHON,str(HERE/'probe_command.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(call([PYTHON,str(HERE/'review_results.py')]),flush=True)
        state['publication']=publish()
        state.update(status='completed',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save();raise

if __name__=='__main__':
    main()
