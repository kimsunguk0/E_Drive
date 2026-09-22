"""Watch one authorized FULL run, validate terminal weights and prepare one ZIP."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time,traceback
from bundle import ROOT,build,archive,sha

HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/a2_progress_full_20260921'
RUN=ROOT/'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1'
PACKAGE=ROOT/'work_dirs/a2_progress_full_20260921/package'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def command(argv,cwd=ROOT):return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT)
def read(path):return json.loads(Path(path).read_text())

def terminal_summary():
    manifest=read(RUN/'manifest.json');protocol=read(RUN/'experiment.json')
    assert manifest['status']=='completed' and manifest['step']==24931 and manifest['nonfinite_count']==0
    assert manifest['initial_model_state_sha256']==protocol['expected_initial_model_state_sha256']
    assert protocol['train_data']['rows']==101520 and protocol['train_data']['scenes']==376
    for path,checksum in protocol['source'].items():assert sha(ROOT/path)==checksum,('Training source drift',path)
    evals=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines() if line.strip()]
    evals=[r for r in evals if r['kind']=='eval'];assert evals[-1]['step']==24931
    record={'status':'completed','step':24931,'nonfinite_count':0,'train_rows':101520,'unique_scenes':376,
        'checkpoint':str(RUN/'ckpt_step24931.pth'),'checkpoint_sha256':sha(RUN/'ckpt_step24931.pth'),
        'manifest_sha256':sha(RUN/'manifest.json'),'training_source_commit':protocol['source_commit'],
        'initial_state_sha256':manifest['initial_model_state_sha256'],
        'elapsed_seconds':manifest['elapsed_seconds'],'evaluation_role':'in-fit diagnostic, not unseen DEV accuracy',
        'in_fit_evaluations':[{'step':r['step'],'PREFIX':r['official_d3'],'rows':r['n']} for r in evals],
        'selection':'fixed terminal, no choice by in-fit score','selected_DEV_PREFIX':.15117885989032362,
        'official_server_score':None,'FULL_to_DEV_artifacts_imported':False}
    atomic(REPORT/'training_complete.json',record);return record

def wait_gpu(gpu,deadline):
    while time.monotonic()<deadline:
        used=int(command(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())
        if used<256:return
        time.sleep(30)
    raise TimeoutError('GPU still occupied; no process was interrupted')

def gpu_run(args,gpu,root=ROOT):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',ADCL_CODE_ROOT=str(root))
    subprocess.run([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,check=True)

def portable_check(reproduction,gpu):
    import numpy as np
    checkpoint=reproduction/'artifacts/model.pth';clips=reproduction/'artifacts/raw_clips'
    old=REPORT/'reproduction_original_fp32.json';new=REPORT/'reproduction_portable_fp32.json'
    common=['--checkpoint',checkpoint,'--clips-root',clips,'--precision','fp32','--require-full']
    gpu_run([HERE/'infer_full.py',*common,'--output',old],gpu)
    gpu_run([reproduction/'code/experiments/a2_progress_full_20260921/infer_full.py',*common,'--output',new],gpu,reproduction/'code')
    a,b=read(old),read(new);assert a['checkpoint_sha256']==b['checkpoint_sha256']
    assert a['predictions'].keys()==b['predictions'].keys()
    diff=max(float(np.max(abs(np.asarray(a['predictions'][k])-np.asarray(b['predictions'][k])))) for k in a['predictions'])
    assert diff==0,diff
    assert all(Path(p).is_relative_to(reproduction/'code') for p in b['imported_source_files'].values())
    result={'status':'passed','actual_FULL_weights':True,'clips':len(a['predictions']),'FP32_max_abs_m':diff,
        'imports_resolved_inside_bundle':True,'checkpoint_sha256':a['checkpoint_sha256'],
        'scope':'B200 original versus portable sources; not RTX4090 latency'}
    atomic(REPORT/'reproduction_validation.json',result)
    import shutil
    shutil.copy2(new,reproduction/'artifacts/reference_B200_fp32.json')
    return result

def publish():
    result={}
    try:
        assert not command(['git','diff','--cached','--name-only']).strip(),'Concurrent staged files'
        names=['training_complete.json','raw_parity.json','flops.json','inference_validation_summary.json',
            'source_bundle.json','reproduction_validation.json','PACKAGE_KO.md','completion.json']
        names += [n for n in ('LOCAL_DELIVERY.json','clean_container_validation.json') if (REPORT/n).exists()]
        paths=[str((REPORT/n).relative_to(ROOT)) for n in names]
        command(['git','add','--','HANDOVER.md',*paths]);command(['git','diff','--cached','--check'])
        command(['git','commit','-m','Record completed H4-PROGRESS FULL and verified submission package'])
        work=command(['git','rev-parse','HEAD']).strip();result['work_commit']=work
        mirror=Path('/home/<B200-USER>/edrive_mirror')
        assert not command(['git','status','--porcelain'],mirror).strip(),'Concurrent mirror changes'
        assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
        command(['git','fetch','github','motiondrive-v2-20260910'],mirror)
        command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
        command(['git','fetch',str(ROOT),work],mirror);command(['git','cherry-pick',work],mirror)
        result['mirror_commit']=command(['git','rev-parse','HEAD'],mirror).strip()
        command(['git','push','github','HEAD:motiondrive-v2-20260910'],mirror)
        result['remote_head']=command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]
        assert result['remote_head']==result['mirror_commit'];result['status']='pushed_and_verified'
    except BaseException:result.update(status='publication_incomplete',error=traceback.format_exc())
    atomic(REPORT/'publication_receipt.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=range(4),default=0);ap.add_argument('--publish',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    assert read(REPORT/'smoke_summary.json')['status']=='passed'
    assert read(REPORT/'deploy_preflight/summary.json')['status']=='passed'
    assert read(REPORT/'portable_preflight.json')['status']=='passed'
    receipt=read(REPORT/'launch_main.json');assert receipt['updates']==24931
    assert receipt['physical_gpu']==a.gpu
    if a.check_only:
        assert (RUN/'manifest.json').exists() and not PACKAGE.exists()
        print('FINISH PIPELINE CONFIGURATION PASSED');return
    state={'status':'watching_training','pid':os.getpid(),'started_utc':now(),'gpu':a.gpu,'official_upload_performed':False}
    status=REPORT/'runtime/finish_status.json';assert not status.exists(),'Existing watcher state'
    atomic(status,state);deadline=time.monotonic()+8*3600
    try:
        while True:
            manifest=read(RUN/'manifest.json')
            state.update(training_status=manifest['status'],step=manifest.get('step'),last_checked_utc=now());atomic(status,state)
            if manifest['status']=='completed' and (RUN/'experiment.json').exists():break
            if manifest['status'] in ('failed','stopped'):raise RuntimeError(manifest['status'])
            proc=Path(f"/proc/{receipt['pid']}/stat")
            if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Training exited without completion')
            if time.monotonic()>deadline:raise TimeoutError('8h deadline; training untouched')
            time.sleep(30)
        training=terminal_summary();wait_gpu(a.gpu,deadline)
        state.update(status='validating_and_inferring',last_checked_utc=now());atomic(status,state)
        gpu_run([HERE/'deploy_full.py'],a.gpu)
        reproduction=PACKAGE/'reproduction';bundle=build(reproduction,RUN/'ckpt_step24931.pth')
        portable_check(reproduction,a.gpu);atomic(REPORT/'source_bundle.json',bundle)
        manifest=read(PACKAGE/'manifest.json');assert manifest['checks']['clips_present']==1125 and not manifest['checks']['failures']
        notes=f'''# A2-H4-PROGRESS-FULL-s1 제출 준비 완료

DEV 최고 단일 후보 PREFIX 0.151178860의 레시피를 전체 376 scene / 101,520행에서 24,931 update 학습했다. 공개 초기값에서 학습했으며 DEV terminal에 추가 학습한 결과가 아니다. 완료된 FULL의 로컬 평가는 학습 내 진단이고 공식 서버 점수는 아직 없다.

제출 파일: `{PACKAGE}/submission.zip`

ZIP 안에는 submission.json 하나만 있다. 1,125개 clip의 absolute XY 6×2와 정수 __flops__를 검수했다. Raw 8 fixture 입력/출력 일치, 전체 test 추론, clip isolation, 동일 FULL 가중치의 portable source 재현을 통과했다. FLOPs는 730,044,861,120이다.

Checkpoint SHA256: {training['checkpoint_sha256']}
Submission ZIP SHA256: {manifest['submission_zip_sha256']}

Status는 RGB를 실제 소비하는 5시점 [-10,-5,-2,-1,0]으로 만들고 공통 scene query에만 넣는다. Motion/state/history에는 제공 status를 직접 넣지 않는다. 이는 구현 경계 설명이며 개별 운영국 승인 주장이 아니다.

`reproduction/`에는 가중치, portable code, Dockerfile, train fixture 2개와 B200 FP32 기준 출력이 있다. 공식 ZIP에는 이 자료를 넣지 않았다. PC의 NVIDIA 드라이버/NVML 불일치 때문에 RTX4090 시간은 미측정이며, clean-container CPU 정합 검사는 GPU latency 검증이 아니다.

사용자 PC의 Downloads/A2-H4-PROGRESS-FULL-s1_submission_20260921로 자동 복사를 연결했다. 복사 완료 여부는 PC의 LOCAL_DELIVERY.json을 확인한다. 공식 업로드는 실행하지 않았으며 이 작업으로 제출 횟수를 사용하지 않았다.
'''
        (REPORT/'PACKAGE_KO.md').write_text(notes);(PACKAGE/'README_KO.md').write_text(notes)
        artifact=archive(PACKAGE,PACKAGE.parent/'A2-H4-PROGRESS-FULL-s1_submission_20260921.tar.gz')
        completion={'status':'ready_for_one_submission','completed_utc':now(),'checkpoint_sha256':training['checkpoint_sha256'],
            'package':str(PACKAGE),'submission_zip_sha256':manifest['submission_zip_sha256'],
            'submission_json_sha256':manifest['submission_json_sha256'],'clips':1125,'archive':artifact,
            'official_upload_performed':False,'official_server_score':None,'RTX4090_ms':None}
        atomic(REPORT/'completion.json',completion)
        handover=ROOT/'HANDOVER.md';old=handover.read_text();first,rest=old.split('\n',1)
        intro='\n**최신 FULL 완료: '+now()+'.** 사용자 요청에 따라 **A2-H4-PROGRESS-FULL-s1** 24,931 update를 마쳤고, raw parity·1,125 clip·portable source 검증 후 제출 ZIP을 확보했다. 공식 업로드/서버 채점은 아직 없다. FULL의 로컬 점수는 in-fit이며 DEV 성능으로 사용하지 않는다. [제출 파일과 검증](reports/a2_progress_full_20260921/PACKAGE_KO.md).\n'
        handover.write_text(first+'\n'+intro+rest)
        state.update(status='ready_for_one_submission',completed_utc=now(),archive=artifact);atomic(status,state)
        # The separate PC watcher starts downloading when completion.json appears.
        # A disconnected PC must not invalidate the verified remote submission.
        delivery_deadline=time.monotonic()+600
        while time.monotonic()<delivery_deadline:
            if all((REPORT/n).exists() for n in ('LOCAL_DELIVERY.json','clean_container_validation.json')):
                delivery=read(REPORT/'LOCAL_DELIVERY.json');validation=read(REPORT/'clean_container_validation.json')
                assert delivery['checks']['submission_zip_sha256']==completion['submission_zip_sha256']
                assert validation['checkpoint_sha256']==completion['checkpoint_sha256']
                completion.update(local_delivery_status=delivery['status'],local_container_status=validation['status'])
                atomic(REPORT/'completion.json',completion);break
            time.sleep(10)
        if a.publish:state['publication']=publish();atomic(status,state)
    except BaseException:
        state.update(status='pipeline_failed',error=traceback.format_exc(),last_checked_utc=now());atomic(status,state);raise

if __name__=='__main__':main()
