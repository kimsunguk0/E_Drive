"""Fixed-weight readout interventions and an unaugmented random train probe.

No optimizer, label injection, state/scene gate, fitting, or submission.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT/'experiments/a2_motion_fresh_20260919'))
import train_experiment as base
from nominal_data import NominalStatusDataset, sha
from motiondrive_v2_training import model_inputs, to_device, tensor_state_sha256
from terminal_review import metrics

REPORT = ROOT/'reports/a2_error_diagnosis_20260920'
W = np.array([11,11,5,5,2,2], np.float64)/36
VARIANTS = ('ON','STATE_OFF','HISTORY_OFF','BOTH_OFF','MOTION_OFF')


def evaluate(model, data, stored, scope, arm):
    values = {k:[] for k in VARIANTS}
    rows, sessions, gt, gt_state, valid_state, state, history = [], [], [], [], [], [], []
    replay_max = 0.
    loader = DataLoader(data,batch_size=8,num_workers=8,pin_memory=True,shuffle=False)
    start = time.monotonic()
    with torch.inference_mode():
        for i,raw in enumerate(loader):
            batch = to_device(raw,torch.device('cuda:0'))
            x = base.mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal')
            x['provided_status5'] = batch['provided_status5']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out = model(**x)
                s,m,st,hi = (out[k] for k in ('scene_features','motion_features','state_hat','history_hat'))
                replay = model.plan_from_features(s,m,st,hi)
                replay_max = max(replay_max,float((replay-out['plan_abs']).abs().max()))
                assert torch.equal(replay,out['plan_abs'])
                outputs = {'ON':out['plan_abs'],
                    'STATE_OFF':model.plan_from_features(s,m,torch.zeros_like(st),hi),
                    'HISTORY_OFF':model.plan_from_features(s,m,st,torch.zeros_like(hi)),
                    'BOTH_OFF':model.plan_from_features(s,m,torch.zeros_like(st),torch.zeros_like(hi)),
                    'MOTION_OFF':model.plan_from_features(s,torch.zeros_like(m),st,hi)}
            for name,p in outputs.items():
                assert torch.isfinite(p).all()
                values[name].append(p.float().cpu().numpy())
            rows.extend(int(v) for v in raw['row'])
            sessions.extend(raw['session_id'])
            gt.append(raw['gt_plan'].numpy())
            gt_state.append(raw['state_target'].numpy())
            valid_state.append(raw['state_valid'].numpy())
            state.append(st.float().cpu().numpy())
            history.append(hi.float().cpu().numpy())
            if (i+1)%50==0:
                print(json.dumps(dict(arm=arm,scope=scope,rows=len(rows),elapsed_s=time.monotonic()-start)),flush=True)
    p = {k:np.concatenate(v).astype(np.float64) for k,v in values.items()}
    gt,gs,vs,ps,ph = [np.concatenate(v) for v in (gt,gt_state,valid_state,state,history)]
    gt=gt.astype(np.float64)
    maxdisp = np.linalg.norm(gt,axis=-1).max(1)
    buckets=np.where(~vs[:,5].astype(bool),'invalid_stop',np.where(gs[:,5]<.5,'nonstop',np.where(maxdisp<=.2,'steady','depart')))
    dg=np.diff(np.concatenate([np.zeros((len(gt),1,2)),gt],1),axis=1)
    mask=np.linalg.norm(dg,axis=-1)>.05
    stats={k:metrics(v,gt,buckets,mask)[0] for k,v in p.items()}
    score={k:np.linalg.norm(v-gt,axis=-1)@W for k,v in p.items()}
    session=np.array(sessions)
    distinct=sorted(set(sessions))
    indices=[np.flatnonzero(session==s) for s in distinct]
    counts=np.array([len(i) for i in indices])
    draws=np.random.default_rng(0).integers(0,len(distinct),(20000,len(distinct)))
    for k in VARIANTS[1:]:
        delta=score[k]-score['ON']
        sums=np.array([delta[i].sum() for i in indices])
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        stats[k]['minus_ON']={'PREFIX':float(delta.mean()),'session_CI95':np.quantile(boot,[.025,.975]).tolist(),
            'plan_PREFIX_distance':float((np.linalg.norm(p[k]-p['ON'],axis=-1)@W).mean())}
    stored_delta=None
    if stored:
        old=json.loads(stored.read_text())['records']
        assert rows==[r['row'] for r in old]
        assert np.array_equal(gt,np.array([r['gt_abs_xy'] for r in old]))
        original=np.array([r['pred_abs_xy'] for r in old])
        stored_delta=float(abs(p['ON']-original).max())
        assert stored_delta<5e-4
    dest=REPORT/f'{arm}_{scope}.npz'
    np.savez_compressed(dest,row=np.array(rows),session=session,gt=gt,gt_state=gs,gt_state_valid=vs,
        state=ps,history=ph,bucket=buckets,**p)
    return dict(n=len(rows),sessions=len(distinct),models=stats,source_prediction_parity_max_abs_m=stored_delta,
        same_forward_planner_parity_max_abs_m=replay_max,row_sha256=hashlib.sha256(np.array(rows,dtype='<i8').tobytes()).hexdigest(),
        arrays_path=str(dest),arrays_sha256=sha(dest),elapsed_seconds=time.monotonic()-start)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=('QREFINE','FRESH'),required=True)
    ap.add_argument('--gpu',type=int,choices=range(4),required=True)
    args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    REPORT.mkdir(parents=True,exist_ok=True)
    dest=REPORT/f'probe_{args.arm}.json'
    assert not dest.exists()
    torch.set_num_threads(4)
    base.trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    run=base.CONTROL if args.arm=='QREFINE' else base.RUNS/'A2-FRESH-NUIM-s1'
    ckpt=run/'ckpt_step20554.pth'
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    config=base.MotionDriveV2Config(**payload['manifest']['model_config'])
    model=(base.SceneExtensionModel(config,arm=base.QREFINE) if args.arm=='QREFINE'
           else base.ExperimentModel(config,arm=base.FRESH))
    base.mr.rebuild_correlation_fuse(model,4)
    model.load_state_dict(payload['model'],strict=True)
    del payload
    before=tensor_state_sha256(model.state_dict())
    assert model.planner.config.state_on is True
    model.cuda().eval().requires_grad_(False)
    train,tune=base.nominal.raw_datasets(False,1)
    # Uniform random selection before examining any model error or GT subgroup.
    index=np.sort(np.random.default_rng(20260920).choice(len(train),size=1024,replace=False))
    train=Subset(NominalStatusDataset(base.mr.MotionCanvasDataset(train,'native')),index.tolist())
    tune=NominalStatusDataset(base.mr.MotionCanvasDataset(tune,'native'))
    result=dict(arm=args.arm,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),model_state_sha256=before,
        source_sha256=sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=args.gpu,optimizer_updates=0,train_probe_seed=20260920,train_probe_size=1024,
        train_probe_selection='uniform without replacement; no augmentation; same indices for both checkpoints',
        variants=list(VARIANTS),provided_status_pose_goal_unchanged=True,full_lineage_used=False,
        limitation='Frozen-weight OOD interventions. Zero readout still leaves biased projection token. Not a trained removal ablation, deployable improvement, or hidden-test estimate.')
    result['V0']=evaluate(model,tune,run/'final_eval.json','V0',args.arm)
    result['train_probe']=evaluate(model,train,None,'train_probe',args.arm)
    after=tensor_state_sha256(model.state_dict())
    assert before==after and model.planner.config.state_on is True
    result['parameter_buffer_hash_unchanged']=True
    result['original_flag_unchanged']=True
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':str(dest),'arm':args.arm,'V0':result['V0']['models'],'train':result['train_probe']['models']}),flush=True)


if __name__=='__main__':
    main()
