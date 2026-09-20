"""Launch only the selected preflighted arm on an idle authorized GPU."""
from pathlib import Path
import argparse,datetime,json,os,subprocess
ROOT=Path('/NHNHOME/data/sukim/adcl')
REPORT=ROOT/'reports/a2_progress_h4_20260920'
RUNS=ROOT/'work_dirs/a2_progress_h4_20260920'
PYTHON='/home/korea_sdv01/cv2env/bin/python'
ARMS=('A2-H4-DIRECT','A2-H4-PROGRESS')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True);args=ap.parse_args()
    phase='smoke' if args.smoke else 'main'
    assert json.loads((REPORT/'preflight.json').read_text())['status']=='passed'
    if not args.smoke:assert json.loads((REPORT/'smoke_summary.json').read_text())['status']=='passed'
    run=RUNS/f"{args.arm}-s1{'-smoke' if args.smoke else ''}"
    receipt=REPORT/f'launch_{args.arm}_{phase}.json'
    runtime=REPORT/'runtime';runtime.mkdir(parents=True,exist_ok=True);log=runtime/f'{run.name}.log'
    assert not run.exists() and not receipt.exists() and not log.exists()
    used=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used:raise RuntimeError(f'GPU{args.gpu} already using {used} MiB')
    cmd=[PYTHON,'-u',str(Path(__file__).with_name('train_progress.py')),'--arm',args.arm,'--gpu',str(args.gpu),'--run-dir',str(run)]
    if args.smoke:cmd.append('--smoke')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    state=dict(phase=phase,arm=args.arm,physical_gpu=args.gpu,run_dir=str(run),log=str(log),command=cmd,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),updates=2 if args.smoke else 20554)
    with receipt.open('x') as stream:json.dump(state,stream,indent=2);stream.write('\n')
    with log.open('xb') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
    state.update(pid=proc.pid,status='spawned; verify actual updates')
    receipt.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
if __name__=='__main__':main()
