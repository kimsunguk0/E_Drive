"""Deliver the completed B200 package to Downloads and verify clean-container inference."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,shlex,subprocess,tarfile,time,traceback,zipfile

REMOTE='/NHNHOME/data/sukim/adcl'
REPORT=REMOTE+'/reports/a2_progress_full_20260921'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify_package(folder,receipt):
    assert sha(folder/'submission.zip')==receipt['submission_zip_sha256']
    assert sha(folder/'submission.json')==receipt['submission_json_sha256']
    with zipfile.ZipFile(folder/'submission.zip') as z:
        assert z.namelist()==['submission.json']
        data=json.loads(z.read('submission.json'))
        assert data==json.loads((folder/'submission.json').read_text())
    assert len(data)==1126 and data.pop('__flops__')==730044861120
    for value in data.values():
        assert len(value)==6 and all(len(point)==2 and all(math.isfinite(x) for x in point) for point in value)
    return {'clips':1125,'all_finite':True,'shape':'6x2','zip_members':['submission.json'],
        'submission_zip_sha256':receipt['submission_zip_sha256']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--user',required=True)
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--key',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--image',default='adcl-a2-full:20260919');a=ap.parse_args()
    assert not a.output.exists(),'Preserve existing delivery'
    common=['-i',a.key,'-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null']
    ssh=['ssh',*common,'-p',str(a.port),a.user+'@'+a.host]
    scp=['scp','-q',*common,'-P',str(a.port)]
    def remote_read():
        code="import pathlib,json; root=pathlib.Path("+repr(REPORT)+"); print(json.dumps({n:json.loads((root/n).read_text()) if (root/n).exists() else None for n in ['completion.json','runtime/finish_status.json','publication_receipt.json']}))"
        return json.loads(subprocess.check_output([*ssh,'python3 -c '+shlex.quote(code)],text=True,timeout=30,stderr=subprocess.DEVNULL))
    state={'status':'waiting_for_FULL_package','pid':os.getpid(),'started_utc':now(),'destination':str(a.output),'official_upload_performed':False}
    assert not a.state.exists(),'Existing delivery watcher';atomic(a.state,state)
    deadline=time.monotonic()+8*3600;connection_errors=0
    try:
        while True:
            try:r=remote_read();connection_errors=0
            except (subprocess.SubprocessError,OSError):
                connection_errors+=1
                if connection_errors>20:raise
                time.sleep(30);continue
            state.update(last_checked_utc=now(),remote_status=(r['runtime/finish_status.json'] or {}).get('status'))
            atomic(a.state,state)
            receipt=r['completion.json']
            if receipt and receipt['status']=='ready_for_one_submission':break
            if state['remote_status']=='pipeline_failed':raise RuntimeError(r['runtime/finish_status.json']['error'])
            if time.monotonic()>deadline:raise TimeoutError('8h delivery deadline')
            time.sleep(30)
        staging=a.output.parent/('.'+a.output.name+'.staging_'+str(os.getpid()));staging.mkdir(parents=True)
        archive=staging/'package.tar.gz'
        subprocess.run([*scp,a.user+'@'+a.host+':'+receipt['archive']['path'],str(archive)],check=True,timeout=600)
        assert sha(archive)==receipt['archive']['sha256']
        with tarfile.open(archive,'r:gz') as tar:
            for m in tar.getmembers():
                p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and not m.issym() and not m.islnk()
                assert p.parts[0]=='package'
            tar.extractall(staging,filter='data')
        package=staging/'package';checks=verify_package(package,receipt)
        state.update(status='checking_clean_container',checks=checks);atomic(a.state,state)
        reproduction=package/'reproduction';output=reproduction/'output';output.mkdir()
        image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',a.image],text=True).strip()
        command=['docker','run','--rm','--network','none','--cpus','4','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','ADCL_CODE_ROOT=/workspace/code','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=4',
            '-v',str(reproduction/'code')+':/workspace/code:ro','-v',str(reproduction/'artifacts')+':/workspace/artifacts:ro',
            '-v',str(output)+':/workspace/output','--entrypoint','python',a.image,
            '/workspace/code/experiments/a2_progress_full_20260921/infer_full.py',
            '--checkpoint','/workspace/artifacts/model.pth','--clips-root','/workspace/artifacts/raw_clips',
            '--output','/workspace/output/CPU_fp32.json','--require-full','--cpu','--precision','fp32']
        subprocess.run(command,check=True,timeout=900)
        actual=json.loads((output/'CPU_fp32.json').read_text());ref=json.loads((reproduction/'artifacts/reference_B200_fp32.json').read_text())
        assert actual['checkpoint_sha256']==receipt['checkpoint_sha256']==ref['checkpoint_sha256']
        assert actual['predictions'].keys()==ref['predictions'].keys()
        assert actual['input_tensor_sha256']==ref['input_tensor_sha256'],'Raw preprocessing differs'
        gap=max(abs(x-y) for k in ref['predictions'] for aa,bb in zip(actual['predictions'][k],ref['predictions'][k]) for x,y in zip(aa,bb))
        assert gap<=.001,('CPU versus B200 FP32 difference exceeds 1mm',gap)
        assert all(p.startswith('/workspace/code/') for p in actual['imported_source_files'].values())
        validation={'status':'passed','scope':'two train fixture clips in isolated CPU Docker, not RTX4090 timing',
            'actual_FULL_weights':True,'checkpoint_sha256':receipt['checkpoint_sha256'],
            'max_abs_CPU_vs_B200_FP32_m':gap,'tolerance_m':.001,'image':a.image,'image_id':image_id,
            'input_tensors_bitwise_equal_to_B200':True,
            'GPU_latency_measured':False,'local_GPU_issue':'NVML driver/library version mismatch'}
        atomic(package/'clean_container_validation.json',validation)
        delivery={'status':'delivered_and_checked','completed_utc':now(),'destination':str(a.output),'checks':checks,
            'clean_container':validation,'remote_archive':receipt['archive'],'official_upload_performed':False}
        atomic(package/'LOCAL_DELIVERY.json',delivery)
        assert not a.output.exists();package.rename(a.output)
        archive.unlink();staging.rmdir()
        state.update(delivery);atomic(a.state,state)
        # Preserve the local validation alongside remote reports. No other files or jobs are changed.
        for name in ('LOCAL_DELIVERY.json','clean_container_validation.json'):
            subprocess.run([*scp,str(a.output/name),a.user+'@'+a.host+':'+REPORT+'/'+name],check=True,timeout=60)
        print(json.dumps(delivery,indent=2),flush=True)
    except BaseException:
        state.update(status='delivery_failed',error=traceback.format_exc(),last_checked_utc=now());atomic(a.state,state);raise

if __name__=='__main__':main()
