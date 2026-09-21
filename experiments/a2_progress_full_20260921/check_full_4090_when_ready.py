"""Wait for the authorized FULL delivery, then verify its unchanged graph on RTX4090."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shlex,subprocess,tarfile,time,traceback

B200_ROOT='/NHNHOME/data/sukim/adcl'
B200_REPORT=B200_ROOT+'/reports/a2_progress_full_20260921'
TARGET_ROOT='/home/chi/adcl_h4_progress_20260921'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()

def validate(reference,actual,timing,require_full=True):
    assert reference['checkpoint_sha256']==actual['checkpoint_sha256']==timing['checkpoint_sha256']
    if require_full:assert actual['arm']==timing['arm']=='A2-H4-PROGRESS-FULL' and actual['step']==timing['step']==24931
    assert reference['input_tensor_sha256']==actual['input_tensor_sha256']==timing['input_tensor_sha256']
    assert reference['predictions'].keys()==actual['predictions'].keys()==timing['predictions'].keys()
    delta=max(abs(x-y) for k in reference['predictions'] for a,b in zip(reference['predictions'][k],actual['predictions'][k]) for x,y in zip(a,b))
    assert delta<=.001,delta
    assert all(p.startswith('/workspace/code/') for p in actual['imported_source_files'].values())
    assert len(timing['timing'])==len(actual['predictions'])==2
    assert all(t['warmup']==30 and t['repeats']==200 for t in timing['timing'])
    return {'status':'passed','actual_FULL_terminal_weights':require_full,'checkpoint_sha256':actual['checkpoint_sha256'],
        'GPU':actual['device'],'torch':actual['torch'],'batch_size':1,'precision':'BF16 with FP32 planner',
        'all_inputs_bitwise_equal_to_B200':True,'FP32_max_abs_B200_vs_4090_m':delta,'FP32_tolerance_m':.001,
        'timing':timing['timing'],'largest_clip_median_ms':max(t['median_ms'] for t in timing['timing']),
        'largest_clip_p95_ms':max(t['p95_ms'] for t in timing['timing']),
        'timing_scope':timing['timing_scope'],'official_upload_performed':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--delivery',type=Path,required=True);ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--b200-key',required=True);a=ap.parse_args()
    assert not a.state.exists();state={'status':'waiting_for_FULL_delivery','pid':os.getpid(),'started_utc':now(),'official_upload_performed':False};atomic(a.state,state)
    opts=['-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new']
    target=['ssh',*opts,'chi@192.168.10.102'];scp=['scp','-q',*opts]
    bopts=['-i',a.b200_key,'-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null']
    bssh=['ssh',*bopts,'-p','42101','korea_sdv01@59.150.32.1']
    bscp=['scp','-q',*bopts,'-P','42101']
    def target_cmd(argv,timeout=600):return subprocess.check_output([*target,shlex.join(map(str,argv))],text=True,stderr=subprocess.STDOUT,timeout=timeout)
    deadline=time.monotonic()+8*3600
    try:
        while not (a.delivery/'LOCAL_DELIVERY.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('FULL delivery deadline')
            state['last_checked_utc']=now();atomic(a.state,state);time.sleep(30)
        delivery=read(a.delivery/'LOCAL_DELIVERY.json');assert delivery['status']=='delivered_and_checked'
        reproduction=a.delivery/'reproduction';checkpoint=sha(reproduction/'artifacts/model.pth')
        assert checkpoint==delivery['clean_container']['checkpoint_sha256']
        output=a.delivery/'RTX4090';output.mkdir();bundle=output/'reproduction.tar.gz'
        with tarfile.open(bundle,'w:gz') as tar:tar.add(reproduction,arcname='reproduction')
        stage=TARGET_ROOT+'/full_'+checkpoint[:12]
        target_cmd(['mkdir',stage])
        subprocess.run([*scp,str(bundle),'chi@192.168.10.102:'+stage+'/reproduction.tar.gz'],check=True,timeout=600)
        assert target_cmd(['sha256sum',stage+'/reproduction.tar.gz']).split()[0]==sha(bundle)
        target_cmd(['tar','-xzf',stage+'/reproduction.tar.gz','-C',stage]);target_cmd(['mkdir',stage+'/output'])
        # Do not interrupt another job. Only the explicitly supplied 4090 is used.
        idle_deadline=time.monotonic()+1800
        while True:
            fields=target_cmd(['nvidia-smi','-i','0','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits']).strip().split(',')
            assert 'RTX 4090' in fields[0]
            if int(fields[1])<256 and int(fields[2])<15:break
            if time.monotonic()>idle_deadline:raise TimeoutError('4090 busy; other processes untouched')
            time.sleep(30)
        state.update(status='validating_FULL_on_RTX4090',checkpoint_sha256=checkpoint,last_checked_utc=now());atomic(a.state,state)
        image_id=target_cmd(['docker','image','inspect','--format','{{.Id}}','md-v2:4090']).strip()
        base=['docker','run','--rm','--network','none','--gpus','device=0','--cpus','4','--user','1000:1000','-w','/workspace/code',
            '-e','ADCL_CODE_ROOT=/workspace/code','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=4',
            '-v',stage+'/reproduction/code:/workspace/code:ro','-v',stage+'/reproduction/artifacts:/workspace/artifacts:ro',
            '-v',stage+'/output:/workspace/output','--entrypoint','python3','md-v2:4090',
            '/workspace/code/experiments/a2_progress_full_20260921/infer_full.py','--checkpoint','/workspace/artifacts/model.pth',
            '--clips-root','/workspace/artifacts/raw_clips','--require-full','--expected-device','RTX 4090']
        target_cmd([*base,'--precision','fp32','--output','/workspace/output/FP32.json'])
        target_cmd([*base,'--precision','bf16','--timing','--warmup','30','--repeats','200','--output','/workspace/output/BF16_timing.json'])
        for name in ('FP32.json','BF16_timing.json'):
            subprocess.run([*scp,'chi@192.168.10.102:'+stage+'/output/'+name,str(output/name)],check=True,timeout=60)
        result=validate(read(reproduction/'artifacts/reference_B200_fp32.json'),read(output/'FP32.json'),read(output/'BF16_timing.json'))
        result.update(host='chi@192.168.10.102',docker_image='md-v2:4090',docker_image_id=image_id,completed_utc=now(),remote_artifacts=stage)
        atomic(a.delivery/'RTX4090_FULL_validation.json',result)
        with (a.delivery/'README_KO.md').open('a') as f:f.write('\n추가 완료: 동일 FULL terminal의 RTX4090 전체 B1 BF16 forward '+str(round(result['largest_clip_median_ms'],3))+'ms (두 train fixture 중 큰 median). 전처리 제외. 상세: RTX4090_FULL_validation.json. 공식 업로드 아님.\n')
        destination='korea_sdv01@59.150.32.1:'+B200_REPORT+'/RTX4090_FULL_validation.json'
        subprocess.run([*bscp,str(a.delivery/'RTX4090_FULL_validation.json'),destination+'.tmp'],check=True,timeout=60)
        code='from pathlib import Path; p=Path('+repr(B200_REPORT)+'); (p/"RTX4090_FULL_validation.json.tmp").replace(p/"RTX4090_FULL_validation.json")'
        subprocess.run([*bssh,'python3 -c '+shlex.quote(code)],check=True,timeout=60)
        # The original collector publishes first; serialize follow-up publication.
        publish_deadline=time.monotonic()+1200
        while True:
            code='import json; from pathlib import Path; p=Path('+repr(B200_REPORT+'/publication_receipt.json')+'); print(json.dumps(json.loads(p.read_text()) if p.exists() else {}))'
            pub=json.loads(subprocess.check_output([*bssh,'python3 -c '+shlex.quote(code)],text=True,stderr=subprocess.DEVNULL,timeout=30))
            if pub.get('status')=='pushed_and_verified':break
            if pub.get('status')=='publication_incomplete':raise RuntimeError('Original publication needs repair; FULL 4090 result preserved')
            if time.monotonic()>publish_deadline:raise TimeoutError('Publication still pending; 4090 result preserved')
            time.sleep(15)
        script=B200_ROOT+'/experiments/a2_progress_full_20260921/publish_4090_result.py'
        subprocess.run([*bssh,shlex.join(['python3',script])],check=True,timeout=180)
        state.update(status='completed',result=result,completed_utc=now());atomic(a.state,state)
    except BaseException:
        state.update(status='4090_followup_failed',error=traceback.format_exc(),last_checked_utc=now());atomic(a.state,state);raise

if __name__=='__main__':main()
