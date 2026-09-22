"""After fixed DEV terminals: publish evidence, transfer one eligible recipe, package.

Does not upload to the challenge, combine models, sweep or extend a schedule.
"""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DEV=ROOT/'reports/a2_progress_fourarm_20260921'
OUT=ROOT/'reports/a2_progress_fourarm_full_20260921'
RUNS=ROOT/'work_dirs/a2_progress_fourarm_full_20260921'
PY='/home/<B200-USER>/cv2env/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
def command(args,cwd=ROOT):return subprocess.check_output([str(a) for a in args],cwd=cwd,text=True,stderr=subprocess.STDOUT)
def gpu(args,name,root=ROOT):
    log=OUT/'runtime'/f'{name}.log';assert not log.exists()
    with log.open('w') as f:
        subprocess.run([PY,*map(str,args)],cwd=root,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',ADCL_CODE_ROOT=str(root)),
            stdout=f,stderr=subprocess.STDOUT,check=True)
def publish(message):
    receipt={'message':message,'created_utc':now()}
    try:
        if command(['git','diff','--cached','--name-only']).strip():raise RuntimeError('Concurrent git staging; preserve it')
        paths=[]
        for folder in (DEV,OUT):
            if not folder.exists():continue
            for p in folder.iterdir():
                if not p.is_file() or p.suffix not in ('.md','.json','.csv'):continue
                if (p.name.startswith('P-') and p.name.endswith('_step0.json')) or 'before_probe_hook' in p.name or p.name in ('predictions.json','inference_validation.json'):continue
                if p.name.startswith('publication_'):continue
                paths.append(str(p.relative_to(ROOT)))
        command(['git','add','--',*paths,'HANDOVER.md'])
        command(['git','-c','core.whitespace=-blank-at-eol,-blank-at-eof','diff','--cached','--check'])
        if command(['git','diff','--cached','--name-only']).strip():command(['git','commit','-m',message])
        work=command(['git','rev-parse','HEAD']).strip();receipt['work_commit']=work
        mirror=Path('/home/<B200-USER>/edrive_mirror')
        if command(['git','status','--porcelain'],mirror).strip():raise RuntimeError('Concurrent mirror changes')
        assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
        command(['git','fetch','github','motiondrive-v2-20260910'],mirror)
        command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
        command(['git','fetch',ROOT,work],mirror);command(['git','cherry-pick',work],mirror)
        receipt['mirror_commit']=command(['git','rev-parse','HEAD'],mirror).strip()
        command(['git','push','github','HEAD:motiondrive-v2-20260910'],mirror)
        receipt['remote_head']=command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]
        assert receipt['remote_head']==receipt['mirror_commit'];receipt['status']='pushed'
    except BaseException:receipt.update(status='publication_incomplete',error=traceback.format_exc())
    atomic(OUT/('publication_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),receipt)
    return receipt
def handover(line):
    p=ROOT/'HANDOVER.md';s=p.read_text();title,rest=s.split('\n',1)
    p.write_text(title+'\n\n'+line+'\n'+rest)
def wait_free_gpu0():
    begin=time.monotonic()
    while True:
        used=int(command(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())
        if used<100:return
        if time.monotonic()-begin>3600:raise TimeoutError('GPU0 occupied; other jobs not touched')
        time.sleep(20)
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'runtime').mkdir(exist_ok=True)
    state={'status':'waiting_for_DEV_terminals','pid':os.getpid(),'started_utc':now(),
        'upload_automated':False,'FULL_started':False}
    status=OUT/'runtime/pipeline_status.json';atomic(status,state);begin=time.monotonic()
    try:
        while True:
            p=DEV/'runtime/orchestrator.json';v=read(p)
            if v['status']=='failed':raise RuntimeError('DEV pipeline failed; no FULL')
            if v['status']=='DEV_completed':break
            if time.monotonic()-begin>6*3600:raise TimeoutError('DEV timeout')
            time.sleep(20)
        decision=read(DEV/'decision.json');arm=decision['candidate_for_FULL']
        handover('**2026-09-21 네 arm DEV 완료:** CONTROL/VECTOR/FINE/SHARED768의 같은 부모·3,426 update 대조가 끝났다. '
            +f"최저 terminal은 {decision['lowest_terminal_arm']} / {decision['primary']['PREFIX']:.9f}. "
            +(f'부모보다 낮은 {arm} 한 종류만 별도 FULL stage2로 이전한다.' if arm else '부모보다 좋아진 terminal이 없어 새 FULL은 실행하지 않는다.')
            +' [점수·조건별 결과](reports/a2_progress_fourarm_20260921/RESULTS_KO.md). 서버 환산 또는 새 제출 결과가 아니다.')
        state['DEV_publication']=publish('Record four-arm PROGRESS terminal comparison and FULL transfer decision')
        if arm is None:
            state.update(status='completed_no_new_FULL',completed_utc=now());atomic(status,state);return
        assert decision['primary']['PREFIX']<decision['parent_PREFIX']
        state.update(status='FULL_smoke',selected_arm=arm);atomic(status,state)
        wait_free_gpu0()
        gpu([HERE/'train_full_stage2.py','--arm',arm,'--gpu','0','--smoke'],'full_smoke')
        smoke=RUNS/f'{arm}-FULL-STAGE2-s1-smoke'
        m=read(smoke/'manifest.json');assert m['status']=='completed' and m['step']==5 and m['nonfinite_count']==0
        cp=smoke/'ckpt_step5.pth'
        # Raw adapter parity and whole graph cost use this concrete FULL smoke.
        check=OUT/'smoke';check.mkdir()
        gpu([HERE/'deploy_fourarm.py','--checkpoint',cp,'--report-dir',check,
             '--package-dir',RUNS/'unused_smoke_package','--label','smoke','--preflight-only'],'full_raw_smoke')
        portable=RUNS/'smoke_reproduction'
        command([PY,HERE/'bundle_fourarm.py','--destination',portable,'--checkpoint',cp])
        for name,code in [('repository',ROOT),('portable',portable/'code')]:
            gpu([code/'experiments/a2_progress_fourarm_20260921/infer_fourarm.py','--checkpoint',cp,
                 '--clips-root',portable/'artifacts/raw_clips','--output',check/f'{name}.json'],f'full_smoke_{name}',code)
        a=read(check/'repository.json');b=read(check/'portable.json')
        assert a['predictions']==b['predictions'] and a['input_tensor_sha256']==b['input_tensor_sha256']
        spec=read(smoke/'experiment.json')
        atomic(OUT/'smoke_and_export.json',{'status':'passed','arm':arm,'updates':5,
            'new_process_portable_predictions_exact':True,'raw_parity':read(check/'raw_parity.json'),
            'source':spec['source'],'main_restarts_from_submitted_FULL_parent':True})
        state.update(status='FULL_training',FULL_started=True,FULL_started_utc=now());atomic(status,state)
        handover(f'**2026-09-21 FULL stage2 시작:** DEV에서 선택한 {arm} 레시피 하나를 제출 FULL terminal의 복사본에4,156 update 적용한다. '
            'DEV 가중치를 FULL에 복사하지 않고 선택된 변경·레시피만 이전한다. Fresh AdamW·warmup100·고정 terminal. '
            '기존 서버0.133684828 제출물은 보존되며 업로드는 자동화하지 않는다.')
        gpu([HERE/'train_full_stage2.py','--arm',arm,'--gpu','0'],'full_main')
        run=RUNS/f'{arm}-FULL-STAGE2-s1';m=read(run/'manifest.json')
        assert m['status']=='completed' and m['step']==4156 and m['nonfinite_count']==0
        checkpoint=run/'ckpt_step4156.pth';state.update(status='FULL_packaging',checkpoint=str(checkpoint));atomic(status,state)
        package=RUNS/'package'
        gpu([HERE/'deploy_fourarm.py','--checkpoint',checkpoint,'--report-dir',OUT,
             '--package-dir',package,'--label',f'{arm}-FULL-STAGE2-s1'],'full_deploy')
        reproduction=package/'reproduction'
        command([PY,HERE/'bundle_fourarm.py','--destination',reproduction,'--checkpoint',checkpoint])
        for name,code in [('repository',ROOT),('portable',reproduction/'code')]:
            gpu([code/'experiments/a2_progress_fourarm_20260921/infer_fourarm.py','--checkpoint',checkpoint,
                '--clips-root',reproduction/'artifacts/raw_clips','--output',OUT/f'terminal_{name}.json','--require-full'],f'terminal_{name}',code)
        a=read(OUT/'terminal_repository.json');b=read(OUT/'terminal_portable.json')
        assert a['predictions']==b['predictions'] and a['input_tensor_sha256']==b['input_tensor_sha256']
        receipt={'status':'B200_package_complete','arm':arm,'checkpoint':str(checkpoint),
            'checkpoint_sha256':sha(checkpoint),'package':str(package),
            'submission_zip_sha256':sha(package/'submission.zip'),
            'portable_exact':True,'clips':1125,'flops':read(OUT/'flops.json')['flops'],
            'FULL_local_eval_role':'in-fit only; fixed terminal, no checkpoint selection',
            'RTX4090_validation':'pending separate target-device check',
            'official_upload_performed':False,'completed_utc':now()}
        atomic(OUT/'completion_receipt.json',receipt)
        registry=read(DEV/'candidate_registry.json');registry['new_FULL']=receipt;atomic(DEV/'candidate_registry.json',registry)
        (OUT/'RESULTS_KO.md').write_text('# 선택된 단일 모델 FULL stage2\n\n'+
            f'{arm},4,156 update 완료. 원래 FULL은 보존했다. Raw8clip parity/1,125clip/strict portable export 완료.\n\n'+
            f"FLOPs {receipt['flops']/1e9:.6f}G. RTX4090의 실제 최종 가중치 검사와 로컬 전달은 별도 상태를 따른다. 공식 업로드 없음.\n")
        handover(f'**2026-09-21 FULL stage2 완료:** {arm} /4,156update, raw parity·1,125clip·portable 재현과 ZIP 확보. '
            '로컬 V0는 in-fit이며 새로운 서버 성능은 아직 없다. [제출 준비 상태](reports/a2_progress_fourarm_full_20260921/RESULTS_KO.md).')
        state['FULL_publication']=publish('Record selected single-model FULL stage2 and verified submission package')
        state.update(status='B200_package_complete',completed_utc=now(),receipt=receipt);atomic(status,state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(status,state);raise
if __name__=='__main__':main()
