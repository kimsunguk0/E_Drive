"""Launch only the agreed command run on idle GPU0; preserve existing jobs."""
import datetime
import json
import os
import subprocess
from command_common import *

PYTHON='/home/korea_sdv01/cv2env/bin/python'

def main():
    assert json.loads((REPORT/'tests_summary.json').read_text())['status']=='passed'
    assert json.loads((REPORT/'smoke_summary.json').read_text())['status']=='passed'
    assert not (REPORT/'launch_receipt.json').exists()
    used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<1000,'GPU0 busy; never stop another process'
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip(),'Commit tracked source edits first'
    runtime=REPORT/'runtime';runtime.mkdir(exist_ok=True)
    run=RUNS/f'{ARM}-s1'
    assert not run.exists()
    args=[PYTHON,str(HERE/'train_command.py'),'--gpu','0','--run-dir',str(run)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (runtime/'train.log').open('xb') as log:
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=proc.pid,physical_gpu=0,arm=ARM,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_dir=str(run),argv=args,updates=20554,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        control_reused=str(CONTROL),new_training_jobs=1)
    (REPORT/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    main()
