"""Separate preflight from explicit main launch; preserve failed logs/runs."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/a2_capacity_dynamics_20260922';RUNS=ROOT/'work_dirs/a2_capacity_dynamics_20260922'
PY='/home/<B200-USER>/cv2env/bin/python'
ARMS=('C-CTRL','C-R101','C-DECSPLIT','C-AGENT')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def start(gpu,arm,phase,args):
    log=REPORT/'runtime'/f'{arm}_{phase}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with log.open('x') as f:
        p=subprocess.Popen([PY,*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return p,{'gpu':gpu,'arm':arm,'pid':p.pid,'log':str(log)}
def checked(jobs):
    errors=[]
    for p,record in jobs:
        code=p.wait()
        if code:errors.append(dict(record,returncode=code))
    if errors:raise RuntimeError(str(errors))
def preflight(state,path):
    cache=ROOT/'data/etri/motiondrive_v2/a2_agent_future_20260922/manifest.json'
    assert json.loads(cache.read_text())['status']=='complete'
    for arm in ARMS:assert json.loads((REPORT/f'{arm}_initial_checks.json').read_text())['status']=='passed'
    jobs=[start(i,a,'smoke',[HERE/'train_capacity.py','--arm',a,'--gpu',i,'--run-dir',RUNS/f'{a}-s1-smoke','--smoke']) for i,a in enumerate(ARMS)]
    state.update(status='five_update_smoke',jobs=[r for _,r in jobs]);atomic(path,state);checked(jobs)
    for mode in ('reload','export'):
        state['status']=mode;atomic(path,state)
        checked([start(i,a,mode,[HERE/'verify_capacity.py','--arm',a,'--mode',mode]) for i,a in enumerate(ARMS)])
    source=None;streams={};checks={}
    for arm in ARMS:
        d=json.loads((RUNS/f'{arm}-s1-smoke/experiment.json').read_text())
        for p,h in d['source'].items():assert sha(ROOT/p)==h,p
        if source is None:source=d['source']
        assert source==d['source']
        checks[arm]={}
        for phase in ('initial_checks','reload','export'):
            p=REPORT/f'{arm}_{phase}.json';assert json.loads(p.read_text())['status']=='passed'
            checks[arm][phase]={'path':str(p),'sha256':sha(p)}
        streams[arm]=json.loads((RUNS/f'{arm}-s1-smoke/manifest.json').read_text())['stream_audit']
    for key in ('microcalls','rows_sha256','baseline_augmentation_sha256','detail_pixels_sha256'):
        assert len({s[key] for s in streams.values()})==1,(key,streams)
    atomic(REPORT/'smoke_and_reload.json',{'status':'passed','source':source,'checks':checks,
        'actual_smoke_updates':5,'streams':streams,'same_baseline_and_sample_stream':True,
        'detail_input_identical':True,'main_restarts_from_unchanged_DEV_parent':True,
        'smoke_weights_or_optimizer_used':False})
    state.update(status='preflight_complete_main_not_started',completed_utc=now());atomic(path,state)
def main_training(state,path):
    checks=json.loads((REPORT/'smoke_and_reload.json').read_text());assert checks['status']=='passed'
    for p,h in checks['source'].items():assert sha(ROOT/p)==h,p
    cost=json.loads((REPORT/'RTX4090_cost.json').read_text());assert cost['status']=='passed'
    assert all(r['flops']<7053e9 and max(t['p95_ms'] for t in r['timing'])<100 for r in cost['models'])
    jobs=[start(i,a,'main',[HERE/'train_capacity.py','--arm',a,'--gpu',i,'--run-dir',RUNS/f'{a}-s1']) for i,a in enumerate(ARMS)]
    state.update(status='training',main_training_started=True,jobs=[r for _,r in jobs],main_started_utc=now());atomic(path,state)
    with (REPORT/'runtime/collector.log').open('x') as log:
        collector=subprocess.Popen([PY,str(HERE/'collect_capacity.py')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state['collector_pid']=collector.pid;atomic(path,state)
    checked(jobs)
    if collector.wait()!=0:raise RuntimeError('Collector failed')
    state.update(status='DEV_completed',completed_utc=now());atomic(path,state)
def initial(state,path):
    jobs=[start(i,a,'initial',[HERE/'verify_capacity.py','--arm',a,'--mode','initial']) for i,a in enumerate(ARMS)]
    state.update(status='initial_checks',jobs=[r for _,r in jobs]);atomic(path,state);checked(jobs)
    state.update(status='initial_complete',completed_utc=now());atomic(path,state)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('initial','preflight','main'),required=True);args=ap.parse_args()
    path=REPORT/'runtime'/f'{args.phase}_orchestrator.json'
    if path.exists():raise ValueError('Refuse orchestration record overwrite')
    state={'phase':args.phase,'status':'starting','pid':os.getpid(),'started_utc':now(),'main_training_started':False}
    atomic(path,state)
    try:{'initial':initial,'preflight':preflight,'main':main_training}[args.phase](state,path)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(path,state);raise
if __name__=='__main__':main()
