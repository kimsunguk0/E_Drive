"""Already-trained A2 FULL: raw parity, complete inference, cost and packaging."""
import os, sys, json, time, subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/NHNHOME/data/sukim/adcl')
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/md_r0_reset_20260914',
          ROOT/'experiments/md_a2_nominal_mh4_20260918',ROOT/'experiments/md_a2_deploy_status_20260918'):
    sys.path.insert(0,str(p))
from a2_model import A2NominalModel
from nominal_data import NominalStatusDataset,sha
from nominal_status import status5_from_clip_records
from models.motiondrive_v2 import MotionDriveV2Config
from models import motiondrive_v2_inputs as adapter
import matching_resolution as mr
import mr_deploy
import motiondrive_v2_training as mt
import train_motiondrive_v2 as trainer

CHECKPOINT=ROOT/'work_dirs/md_a2_nominal_mh4_20260918/A2-FULL-NOM-s1/ckpt_step24931.pth'
EXPECTED='aabdb2dca24491b46fd2e56e66d57fe0742e17a5b4c63c29f3199137cf8ff297'
REPORT=ROOT/'reports/a2_full_submission_20260919'
PACKAGE=ROOT/'work_dirs/a2_full_submission_20260919/package'
TEST=Path('/tmp/etri_test')


def prepare_clip(clip):
    import pyarrow.parquet as pq
    clip=Path(clip)
    calibration=pq.read_table(clip/'calibration.parquet',columns=list(adapter.CALIBRATION_COLUMNS)).to_pylist()
    poses=pq.read_table(clip/'ego_pose.parquet',columns=list(adapter.POSE_COLUMNS)).to_pylist()
    prepared=mr_deploy.prepare_mr_clip_from_records(calibration,poses,
        lambda camera,frame:(clip/camera/f'frame_{frame}.jpg').read_bytes(),detail='native')
    status,info=status5_from_clip_records(poses)
    prepared.inputs['provided_status5']=torch.from_numpy(status)[None]
    prepared.metadata['a2_status']={'producer':'nominal_status.status5_from_clip_records',
        'role':'provided status INPUT, only shared scene query',**info}
    return prepared


def load_full():
    assert sha(CHECKPOINT)==EXPECTED
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=A2NominalModel(MotionDriveV2Config(**payload['manifest']['model_config']),arm='A2-FULL-NOM')
    mr.rebuild_correlation_fuse(model,4);model.load_state_dict(payload['model'],strict=True)
    return model.cuda().eval(),payload


def write(name,value):
    (REPORT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def predict(model,prepared,precision='bf16'):
    inputs={k:v.cuda() for k,v in prepared.inputs.items()}
    with torch.no_grad(),trainer.autocast(torch.device('cuda:0'),precision):
        plan=model(**inputs)['plan_abs'].float().cpu().numpy()[0]
    assert plan.shape==(6,2) and np.isfinite(plan).all()
    return plan


def fixture_parity(model):
    from motiondrive_v2_data import MotionDriveDataset
    from torch.utils.data import DataLoader
    fixtures=json.loads((ROOT/'reports/motiondrive_v2_deploy_fixture_train8_manifest.json').read_text())['clips']
    result=[]
    for clip in fixtures:
        base=MotionDriveDataset(data_root='/tmp/pm97',split_manifest=str(ROOT/'data/etri/motiondrive_v2/grouped_split_rawtime.json'),
            split=clip['source_split'],supervision_root=str(ROOT/'data/etri/motiondrive_v2/train_tune_geometry_v2'),
            min_frame=30,frame_stride=1,augment=False,seed=0,history_contract='control',
            scenes=[clip['source_scene']],frames=[int(clip['source_frame'])])
        data=NominalStatusDataset(mr.MotionCanvasDataset(base,'native'),allowed_splits=('train','tune','val'))
        assert len(data)==1
        batch=next(iter(DataLoader(data,batch_size=1,num_workers=0)))
        di=mr.model_inputs_with_canvas(mt.model_inputs,batch,time_input='nominal')
        di['provided_status5']=batch['provided_status5']
        raw=prepare_clip(ROOT/'data/etri/motiondrive_v2/deploy_fixture_train8'/clip['clip_id'])
        diffs={k:float((di[k].float()-raw.inputs[k].float()).abs().max()) for k in di}
        assert all(v==0 for k,v in diffs.items() if k!='provided_status5'),diffs
        assert diffs['provided_status5']<2e-5,diffs
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            dp=model(**{k:v.cuda() for k,v in di.items()})['plan_abs'].float().cpu().numpy()[0]
        rp=predict(model,raw);gap=float(abs(dp-rp).max())
        assert gap<5e-4,gap
        result.append(dict(clip=clip['clip_id'],input_max_diff=diffs,plan_max_abs_xy=gap))
    value={'checkpoint_sha256':EXPECTED,'fixtures':len(result),'checks':result,
        'max_plan_abs_xy':max(r['plan_max_abs_xy'] for r in result),
        'scope':'train-sourced raw fixtures; pipeline parity, not accuracy validation'}
    write('raw_parity.json',value);print('PARITY '+json.dumps(value),flush=True)


def count_flops(model,payload,prepared,clip):
    import torch.utils.module_tracker as tracker
    from torch.utils.flop_counter import FlopCounterMode
    class Handle:
        def remove(self):pass
    old=tracker.register_multi_grad_hook
    tracker.register_multi_grad_hook=lambda *a,**k:Handle()
    inputs={k:v.cuda() for k,v in prepared.inputs.items()}
    try:
        with torch.no_grad():model(**inputs)
        with torch.no_grad(),FlopCounterMode(display=False,depth=3) as counter:model(**inputs)
        counts=counter.get_flop_counts().get('Global',{})
    finally:tracker.register_multi_grad_hook=old
    total=int(sum(counts.values()))
    value={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':EXPECTED,
        'model_state_sha256':mt.tensor_state_sha256(payload['model']),
        'flops':total,'gflops':total/1e9,'passes_cutoff':total<=7053e9,'cutoff_gflops':7053.,
        'counter':'official-style torch.utils.flop_counter.FlopCounterMode Global sum',
        'scope':'one complete A2 forward, including every current/history image encoding; no division',
        'measured_clip':clip,'precision':'FP32 counting','parameters':sum(p.numel() for p in model.parameters()),
        'input_shapes':{k:list(v.shape) for k,v in inputs.items()},
        'by_operator':{str(k):v for k,v in counts.items()},'teacher_in_forward':False}
    write('flops.json',value);print('FLOPS '+json.dumps(value),flush=True)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0','1','2','3')
    torch.set_num_threads(4);trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not (PACKAGE/'submission.zip').exists(),'Do not overwrite a preserved package'
    model,payload=load_full();fixture_parity(model)
    clips=sorted(p for p in TEST.iterdir() if p.is_dir());assert len(clips)==1125
    first=prepare_clip(clips[0]);count_flops(model,payload,first,clips[0].name)
    # Archive exact raw-input B1 tensors for repeatable target-device timing.
    torch.save(first.inputs,REPORT/'first_clip_inputs.pt')
    initial=predict(model,first);submission={};details=[];started=time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    # Bound prefetch to eight clips; GPU remains serial B1 and has no cross-clip memory.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={i:pool.submit(prepare_clip,clips[i]) for i in range(min(8,len(clips)))}
        for i,clip in enumerate(clips):
            prepared=pending.pop(i).result()
            if i+8<len(clips):pending[i+8]=pool.submit(prepare_clip,clips[i+8])
            plan=predict(model,prepared);submission[clip.name]=plan.tolist()
            details.append(dict(clip=clip.name,provided_status5=prepared.inputs['provided_status5'][0].tolist(),
                status_fit=prepared.metadata['a2_status']))
            if (i+1)%100==0:print('INFER '+json.dumps({'done':i+1,'total':1125,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    replay=predict(model,prepare_clip(clips[0]));gap=float(abs(replay-initial).max());assert gap==0
    assert len(submission)==1125
    destination=REPORT/'predictions.json';destination.write_text(json.dumps(submission,allow_nan=False))
    validation={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':EXPECTED,'clips':1125,
        'all_finite':True,'all_shapes_6x2':True,'absolute_XY_no_second_cumsum':True,
        'clip_isolation_max_abs':gap,'provided_status':'nominal causal input; query-only',
        'no_future_pose_in_status':True,'full_weights_not_used_in_DEV':True,
        'elapsed_wall_seconds':time.perf_counter()-started,'uploaded':False,
        'source_git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':sha(__file__),'predictions_sha256':sha(destination),'per_clip':details}
    write('inference_validation.json',validation)
    subprocess.run([sys.executable,str(ROOT/'experiments/md_r0_reset_20260914/package_submission.py'),
        '--submission',str(destination),'--flops-report',str(REPORT/'flops.json'),
        '--clips-root',str(TEST),'--out-dir',str(PACKAGE),'--label','A2-FULL-NOM-s1'],check=True)
    print('COMPLETE '+str(PACKAGE),flush=True)

if __name__=='__main__':main()
