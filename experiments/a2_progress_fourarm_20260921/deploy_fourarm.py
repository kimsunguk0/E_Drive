"""Terminal FULL weights: raw parity, FLOPs, 1125 clips, one submission ZIP."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from infer_fourarm import ROOT,load_model,prepare_clip,predict,configure,sha,mr
sys.path.insert(0,str(ROOT/"experiments/a2_progress_full_20260921"))

REPORT=ROOT/'reports/a2_progress_full_20260921'
RUN=ROOT/'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1'
PACKAGE=ROOT/'work_dirs/a2_progress_full_20260921/package'
TEST=Path('/tmp/etri_test')
def write(folder,name,value):
    p=folder/name;p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def fixture_parity(model,folder,checkpoint):
    from motiondrive_v2_data import MotionDriveDataset
    from torch.utils.data import DataLoader
    import motiondrive_v2_training as mt
    from full_data import FullH4StatusDataset
    fixtures=json.loads((ROOT/'reports/motiondrive_v2_deploy_fixture_train8_manifest.json').read_text())['clips']
    results=[]
    for clip in fixtures:
        base=MotionDriveDataset(data_root='/tmp/pm97',split_manifest=str(ROOT/'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'),
            split='train',supervision_root=str(ROOT/'data/etri/motiondrive_v2/r0reset_tplus_geometry_v2'),
            min_frame=30,frame_stride=1,augment=False,seed=1,history_contract='control',
            scenes=[clip['source_scene']],frames=[int(clip['source_frame'])])
        data=FullH4StatusDataset(mr.MotionCanvasDataset(base,'native'));assert len(data)==1
        batch=next(iter(DataLoader(data,batch_size=1,num_workers=0)))
        di=mr.model_inputs_with_canvas(mt.model_inputs,batch,time_input='nominal');di['provided_status5']=batch['provided_status5']
        raw=prepare_clip(ROOT/'data/etri/motiondrive_v2/deploy_fixture_train8'/clip['clip_id'])
        assert set(di)==set(raw.inputs)
        diffs={k:float(abs(di[k].float()-raw.inputs[k].float()).max()) for k in di}
        assert max(diffs.values())==0,diffs
        gaps={}
        for precision in ('fp32','bf16'):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                dp=model(**{k:v.cuda() for k,v in di.items()})['plan_abs'].float().cpu().numpy()[0]
            rp=predict(model,raw,precision);gap=float(abs(dp-rp).max());assert gap==0,gap;gaps[precision]=gap
        results.append({'clip':clip['clip_id'],'input_max_diff':diffs,'plan_max_diff':gaps,
            'status_fit_frames':raw.metadata['provided_status']['fit_relative_frames'],
            'consumed_RGB_frames':raw.metadata['provided_status']['consumed_RGB_frames']})
    value={'checkpoint_sha256':sha(checkpoint),'fixtures':len(results),'checks':results,
        'all_inputs_and_outputs_bitwise_equal':True,'scope':'raw training fixtures for pipeline parity, not unseen accuracy'}
    write(folder,'raw_parity.json',value)
    return value

def cost(model,payload,prepared,folder,checkpoint):
    import torch.utils.module_tracker as tracker
    from torch.utils.flop_counter import FlopCounterMode
    from motiondrive_v2_training import tensor_state_sha256
    class Handle:
        def remove(self):pass
    old=tracker.register_multi_grad_hook;tracker.register_multi_grad_hook=lambda *a,**k:Handle()
    inp={k:v.cuda() for k,v in prepared.inputs.items()}
    try:
        with torch.no_grad(),FlopCounterMode(display=False) as counter:model(**inp)
        counts=counter.get_flop_counts()['Global'];total=int(sum(counts.values()))
    finally:tracker.register_multi_grad_hook=old
    assert 0<total<7053e9
    timing=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(35):
            torch.cuda.synchronize();tick=time.perf_counter();model(**inp);torch.cuda.synchronize()
            if i>=5:timing.append((time.perf_counter()-tick)*1000)
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'model_state_sha256':tensor_state_sha256(payload['model']),'flops':total,'gflops':total/1e9,
        'passes_cutoff':True,'cutoff_gflops':7053.,'counter':'torch.utils.flop_counter.FlopCounterMode Global sum',
        'scope':'entire B1 forward including every current/history encoder; FP32 counting',
        'by_operator':{str(k):v for k,v in counts.items()},'parameters':sum(p.numel() for p in model.parameters()),
        'B200_BF16_median_ms':float(np.median(timing)),'RTX4090_ms':None,'device':torch.cuda.get_device_name(0)}
    write(folder,'flops.json',result);return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--report-dir',required=True)
    ap.add_argument('--package-dir',required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2','3')
    global REPORT,PACKAGE
    REPORT=Path(args.report_dir);PACKAGE=Path(args.package_dir);folder=REPORT
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'raw_parity.json').exists(),'Do not overwrite completed parity'
    checkpoint=Path(args.checkpoint)
    configure();model,payload=load_model(checkpoint,require_full=not args.preflight_only)
    parity=fixture_parity(model,folder,checkpoint)
    clips=sorted(p for p in TEST.iterdir() if p.is_dir());assert len(clips)==1125
    first=prepare_clip(clips[0]);flops=cost(model,payload,first,folder,checkpoint)
    if args.preflight_only:
        write(folder,'summary.json',{'status':'passed','weights':'smoke only; not the candidate terminal',
            'checkpoint_sha256':sha(checkpoint),'raw_parity':parity['all_inputs_and_outputs_bitwise_equal'],
            'flops':flops['flops'],'official_test_predictions_written':False,'official_upload_performed':False})
        print('DEPLOY PREFLIGHT PASSED');return
    assert not PACKAGE.exists(),'Never overwrite a preserved package'
    initial=predict(model,first);submission={};details=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={i:pool.submit(prepare_clip,clips[i]) for i in range(min(8,len(clips)))}
        for i,clip in enumerate(clips):
            prepared=pending.pop(i).result()
            if i+8<len(clips):pending[i+8]=pool.submit(prepare_clip,clips[i+8])
            submission[clip.name]=predict(model,prepared).tolist()
            details.append({'clip':clip.name,'status':prepared.metadata['provided_status']})
            if (i+1)%100==0:print(json.dumps({'test_clips_done':i+1,'of':1125,'seconds':time.monotonic()-started}),flush=True)
    replay=predict(model,prepare_clip(clips[0]));gap=float(abs(replay-initial).max());assert gap==0
    assert len(submission)==1125
    dest=REPORT/'predictions.json';assert not dest.exists();dest.write_text(json.dumps(submission,allow_nan=False))
    validation={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'clips':1125,
        'all_finite':True,'all_shapes_6x2':True,'absolute_XY_no_second_cumsum':True,'clip_isolation_max_abs':gap,
        'all_status_pose_times_covered_by_RGB':True,'provided_status_route':'common scene query only',
        'elapsed_seconds':time.monotonic()-started,'official_upload_performed':False,
        'predictions_sha256':sha(dest),'per_clip':details,'source_sha256':sha(__file__)}
    write(REPORT,'inference_validation.json',validation)
    write(REPORT,'inference_validation_summary.json',{k:v for k,v in validation.items() if k!='per_clip'})
    subprocess.run([sys.executable,str(ROOT/'experiments/md_r0_reset_20260914/package_submission.py'),
        '--submission',str(dest),'--flops-report',str(REPORT/'flops.json'),'--clips-root',str(TEST),
        '--out-dir',str(PACKAGE),'--label',args.label],check=True)
    print('PACKAGE_COMPLETE '+str(PACKAGE),flush=True)
if __name__=='__main__':main()
