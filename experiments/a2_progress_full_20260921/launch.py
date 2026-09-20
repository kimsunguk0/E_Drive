"""Launch one FULL fit on an idle authorized GPU; preserve exact receipts."""
from pathlib import Path
import argparse,datetime,json,os,subprocess
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_progress_full_20260921'
RUNS=ROOT/'work_dirs/a2_progress_full_20260921';ARM='A2-H4-PROGRESS-FULL'
PYTHON='/home/korea_sdv01/cv2env/bin/python'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=range(4),default=0)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    assert json.loads((REPORT/'input_policy.json').read_text())['rows']==101520
    assert json.loads((ROOT/'reports/a2_progress_h4_20260920/preflight.json').read_text())['status']=='passed'
    if not args.smoke:assert json.loads((REPORT/'smoke_summary.json').read_text())['status']=='passed'
    phase='smoke' if args.smoke else 'main';run=RUNS/f"{ARM}-s1{'-smoke' if args.smoke else ''}"
    receipt=REPORT/f'launch_{phase}.json';log=REPORT/f'runtime/{run.name}.log'
    assert not run.exists() and not receipt.exists() and not log.exists()
    used=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used==0,(args.gpu,used)
    cmd=[PYTHON,'-u',str(Path(__file__).with_name('train_full.py')),'--gpu',str(args.gpu),'--run-dir',str(run)]
    if args.smoke:cmd.append('--smoke')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    r={'phase':phase,'physical_gpu':args.gpu,'arm':ARM,'run_dir':str(run),'log':str(log),'command':cmd,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'updates':2 if args.smoke else 24931,'status':'launching'}
    with receipt.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    with log.open('xb') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
    r.update(pid=p.pid,status='spawned; verify actual updates');receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
