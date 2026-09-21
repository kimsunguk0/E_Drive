"""Start matched DEV runs only after production smoke and export checks pass."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/a2_splitread_lanegeom_20260921';RUNS=ROOT/'work_dirs/a2_splitread_lanegeom_20260921'
PY='/home/korea_sdv01/cv2env/bin/python'
ARMS=('P-CTRL-NEXT','P-SPLITREAD','P-LANE-GEOM')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def start(gpu,arm,phase,args):
    log=REPORT/'runtime'/f'{arm}_{phase}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with log.open('x') as f:
        p=subprocess.Popen([PY,*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return p,{'gpu':gpu,'arm':arm,'pid':p.pid,'log':str(log)}
def checked(jobs):
    for process,record in jobs:
        if process.wait()!=0:raise RuntimeError('Failed: '+str(record))
def main():
    path=REPORT/'runtime/orchestrator.json'
    state={'status':'waiting_for_initial_checks_and_cache','pid':os.getpid(),'started_utc':now(),'main_training_started':False}
    atomic(path,state);began=time.monotonic()
    try:
        while True:
            initial=[REPORT/f'{a}_initial_checks.json' for a in ARMS]
            cache=ROOT/'data/etri/motiondrive_v2/a2_lane_geometry_20260921_v3/manifest.json'
            if all(p.exists() for p in initial) and cache.exists() and json.loads(cache.read_text())['status']=='complete':break
            for log in (REPORT/'runtime').glob('*_initial.log'):
                if 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Initial check failed: '+str(log))
            if time.monotonic()-began>2400:raise TimeoutError('Initial preparation incomplete')
            time.sleep(5)
        for p in initial:assert json.loads(p.read_text())['status']=='passed'
        checked([start(3,'geometry','cache_check',[HERE/'verify_next.py','--mode','geometry'])])
        state['status']='five_update_smoke';atomic(path,state)
        jobs=[start(i,a,'smoke',[HERE/'train_next.py','--arm',a,'--gpu',i,'--run-dir',RUNS/f'{a}-s1-smoke','--smoke']) for i,a in enumerate(ARMS)]
        state['smoke_jobs']=[r for _,r in jobs];atomic(path,state);checked(jobs)
        for mode in ('reload','export'):
            state['status']=mode;atomic(path,state)
            checked([start(i,a,mode,[HERE/'verify_next.py','--arm',a,'--mode',mode]) for i,a in enumerate(ARMS)])
        source=None;streams=[];checks={}
        for arm in ARMS:
            d=json.loads((RUNS/f'{arm}-s1-smoke/experiment.json').read_text())
            for p,h in d['source'].items():assert sha(ROOT/p)==h,p
            if source is None:source=d['source']
            assert source==d['source']
            checks[arm]={}
            for phase in ('initial_checks','reload','export'):
                p=REPORT/f'{arm}_{phase}.json';assert json.loads(p.read_text())['status']=='passed'
                checks[arm][phase]={'path':str(p),'sha256':sha(p)}
            streams.append(json.loads((RUNS/f'{arm}-s1-smoke/manifest.json').read_text())['stream_audit'])
        assert all(s==streams[0] for s in streams)
        atomic(REPORT/'smoke_and_reload.json',{'status':'passed','source':source,'checks':checks,
            'actual_smoke_updates':5,'same_sample_and_augmentation_stream':streams[0],
            'geometry_cache_check':json.loads((REPORT/'geometry_cache_checks.json').read_text()),
            'main_restarts_from_unchanged_DEV_CTRL_parent':True,'smoke_weights_or_optimizer_used':False})
        jobs=[start(i,a,'main',[HERE/'train_next.py','--arm',a,'--gpu',i,'--run-dir',RUNS/f'{a}-s1']) for i,a in enumerate(ARMS)]
        state.update(status='training',main_training_started=True,jobs=[r for _,r in jobs],main_started_utc=now())
        atomic(path,state)
        with (REPORT/'runtime/collector.log').open('x') as log:
            collector=subprocess.Popen([PY,str(HERE/'collect_next.py')],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['collector_pid']=collector.pid;atomic(path,state)
        checked(jobs)
        if collector.wait()!=0:raise RuntimeError('Collector failed')
        state.update(status='DEV_completed',completed_utc=now());atomic(path,state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(path,state);raise

if __name__=='__main__':main()
