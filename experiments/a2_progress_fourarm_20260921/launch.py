"""Gate actual five-update smoke + new-process export, then start four DEV arms."""
from pathlib import Path
import hashlib,json,os,subprocess,time,datetime,traceback
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/a2_progress_fourarm_20260921';RUNS=ROOT/'work_dirs/a2_progress_fourarm_20260921'
PY='/home/korea_sdv01/cv2env/bin/python'
ARMS=('P-CTRL','P-VECTOR','P-FINE','P-SHARED768')
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def start(i,arm,phase,argv):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    log=REPORT/'runtime'/f'{arm}_{phase}.log'
    if log.exists():raise FileExistsError(log)
    with log.open('w') as f:p=subprocess.Popen([PY,*argv],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return p,{'arm':arm,'gpu':i,'pid':p.pid,'log':str(log)}
def main():
    status=REPORT/'runtime/orchestrator.json'
    state={'status':'waiting_for_smokes','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic(status,state);begin=time.monotonic()
    try:
        while True:
            complete=[]
            for arm in ARMS:
                p=RUNS/f'{arm}-s1-smoke/manifest.json'
                if not p.exists():complete.append(False);continue
                m=json.loads(p.read_text())
                if m['status'] in ('failed','stopped'):raise RuntimeError(arm+' smoke '+m['status'])
                complete.append(m['status']=='completed' and (p.parent/'experiment.json').exists())
            if all(complete):break
            if time.monotonic()-begin>900:raise TimeoutError('Smoke did not finish')
            time.sleep(5)
        for mode in ('reload','export'):
            state['status']=mode;atomic(status,state)
            jobs=[start(i,a,mode,[str(HERE/'verify.py'),'--arm',a,'--mode',mode]) for i,a in enumerate(ARMS)]
            for p,job in jobs:
                if p.wait()!=0:raise RuntimeError(str(job))
        streams=[];source=None;records={}
        for arm in ARMS:
            d=json.loads((RUNS/f'{arm}-s1-smoke/experiment.json').read_text())
            for p,h in d['source'].items():
                if sha(ROOT/p)!=h:raise RuntimeError('Source changed after smoke: '+p)
            if source is None:source=d['source']
            assert source==d['source']
            records[arm]={}
            for phase in ('initial_checks','reload','export'):
                p=REPORT/f'{arm}_{phase}.json';v=json.loads(p.read_text());assert v['status']=='passed'
                records[arm][phase]={'path':str(p),'sha256':sha(p)}
            streams.append(json.loads((RUNS/f'{arm}-s1-smoke/manifest.json').read_text())['stream_audit'])
        assert all(v==streams[0] for v in streams)
        receipt={'status':'passed','actual_smoke_updates_per_arm':5,'fresh_main_restarts_from_DEV_parent':True,
            'source':source,'checks':records,'same_smoke_sample_and_augmentation_stream':streams[0],
            'reused_reference_tests':['reports/a2_pro_review_20260921/VECTOR_INTEGRATION.json',
                'reports/a2_pro_review_20260921/VECTOR_REFERENCE_REPLAY.json',
                'reports/a2_pro_review_20260921/FINE_READ_PAIRED_REVIEW.json'],
            'construction_and_loss_preparation_notes':[
                'Initial import-only attempt used a colliding train.py name; renamed before GPU forwards.',
                'First five-update smoke preserved; added the common fixed train-probe callback and reran from parent.',
                'No smoke optimizer state is used by the main runs.']}
        atomic(REPORT/'smoke_and_reload.json',receipt)
        # These processes only start after all four concrete checks have passed.
        jobs=[start(i,a,'main',[str(HERE/'train_fourarm.py'),'--arm',a,'--gpu',str(i),
            '--run-dir',str(RUNS/f'{a}-s1')]) for i,a in enumerate(ARMS)]
        state.update(status='training',jobs=[j for _,j in jobs],main_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic(status,state)
        log=REPORT/'runtime/collector.log'
        with log.open('w') as f:
            collector=subprocess.Popen([PY,str(HERE/'collect.py')],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state['collector_pid']=collector.pid;atomic(status,state)
        for p,job in jobs:
            if p.wait()!=0:raise RuntimeError('Training failed: '+str(job))
        if collector.wait()!=0:raise RuntimeError('Result collector failed')
        state.update(status='DEV_completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic(status,state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(status,state);raise
if __name__=='__main__':main()
