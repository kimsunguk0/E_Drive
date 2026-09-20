"""Bounded frozen diagnostics. No optimizer or deployable intervention policy."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0,str(ROOT/'experiments/a2_motion_fresh_20260919'))
import train_experiment as base
from nominal_data import NominalStatusDataset, sha
from motiondrive_v2_training import model_inputs,to_device,tensor_state_sha256
from terminal_review import metrics
from models.motiondrive_v2.planner import DirectTrajectoryPlanner
sys.path.insert(0,str(ROOT/'experiments/a2_temporal_read_20260920'))
from temporal_model import TemporalReadModel

REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'
CONT=ROOT/'work_dirs/a2_fresh_continue_20260920/A2-FRESH-CONT-s1'
TEMP=ROOT/'work_dirs/a2_temporal_read_20260920/A2-TEMPORAL-READ-s1'
RUNS={'CONT_SELECTED':(CONT,5710),'CONT_TERMINAL':(CONT,6852),'TEMPORAL':(TEMP,20554)}
W=np.array([11,11,5,5,2,2],np.float64)/36

def planner(model,out,st=None,hi=None,motion=None,bypass=False):
    s,m,p,h=(out[k] for k in ('scene_features','motion_features','state_hat','history_hat'))
    st=p if st is None else st;hi=h if hi is None else hi;m=m if motion is None else motion
    if bypass:
        return DirectTrajectoryPlanner.forward(model.planner,s,m,st,hi)
    args=[s,m,st,hi]
    if 'motion_pair_features' in out:args.append(out['motion_pair_features'])
    return model.plan_from_features(*args)

def evaluate(model,data,stored,scope,arm):
    values={}; rows=[];sessions=[];gt=[];gs=[];vs=[];state=[];history=[];provided=[]
    invariance={'replay':0.,'status_to_motion':0.,'status_to_state':0.,'status_to_history':0.,'native_history_to_scene':0.}
    start=time.monotonic()
    loader=DataLoader(data,batch_size=8,num_workers=8,pin_memory=True,shuffle=False)
    with torch.inference_mode():
        for i,raw in enumerate(loader):
            batch=to_device(raw,torch.device('cuda:0'))
            x=base.mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal')
            x['provided_status5']=batch['provided_status5']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(**x); outputs={'ON':out['plan_abs']}
                replay=planner(model,out)
                invariance['replay']=max(invariance['replay'],float((replay-out['plan_abs']).abs().max()))
                assert torch.equal(replay,out['plan_abs'])
                if scope=='V0' and arm!='CONT_TERMINAL':
                    outputs['NUMERIC_OFF']=planner(model,out,st=torch.zeros_like(out['state_hat']),hi=torch.zeros_like(out['history_hat']))
                    outputs['POOLED_MOTION_OFF']=planner(model,out,motion=torch.zeros_like(out['motion_features']))
                    outputs['ALL_CONTINUOUS_MOTION_OFF']=planner(model,out,motion=torch.zeros_like(out['motion_features']),bypass=True)
                    if arm=='TEMPORAL':outputs['NEW_READ_OFF']=planner(model,out,bypass=True)
                    # Recompute only the native motion observation. Scene history stays intact.
                    xx=dict(x);xx['motion_history']=x['motion_current'][:,None].expand_as(x['motion_history']).contiguous()
                    changed=model(**xx)
                    outputs['NATIVE_HISTORY_STILL']=changed['plan_abs']
                    invariance['native_history_to_scene']=max(invariance['native_history_to_scene'],float((out['scene_features']-changed['scene_features']).abs().max()))
                    for sign,name in [(-1.,'STATUS_VX_MINUS_0P1'),(1.,'STATUS_VX_PLUS_0P1')]:
                        xx=dict(x);status=x['provided_status5'].clone();status[:,0]+=sign*.1;xx['provided_status5']=status
                        changed=model(**xx);outputs[name]=changed['plan_abs']
                        for label,key in [('motion','motion_features'),('state','state_hat'),('history','history_hat')]:
                            invariance['status_to_'+label]=max(invariance['status_to_'+label],float((changed[key]-out[key]).abs().max()))
            for name,p in outputs.items():
                assert torch.isfinite(p).all();values.setdefault(name,[]).append(p.float().cpu().numpy())
            rows.extend(int(v) for v in raw['row']);sessions.extend(raw['session_id'])
            for dest,key in [(gt,'gt_plan'),(gs,'state_target'),(vs,'state_valid'),(provided,'provided_status5')]:dest.append(raw[key].numpy())
            state.append(out['state_hat'].float().cpu().numpy());history.append(out['history_hat'].float().cpu().numpy())
            if (i+1)%50==0:print(json.dumps(dict(arm=arm,scope=scope,rows=len(rows),elapsed_s=time.monotonic()-start)),flush=True)
    p={k:np.concatenate(v).astype(np.float64) for k,v in values.items()}
    gt,gs,vs,ps,ph,provided=[np.concatenate(v) for v in (gt,gs,vs,state,history,provided)]
    gt=gt.astype(np.float64);session=np.array(sessions)
    bucket=np.where(~vs[:,5].astype(bool),'invalid_stop',np.where(gs[:,5]<.5,'nonstop',np.where(np.linalg.norm(gt,axis=-1).max(1)<=.2,'steady','depart')))
    dg=np.diff(np.concatenate([np.zeros((len(gt),1,2)),gt],1),axis=1);mask=np.linalg.norm(dg,axis=-1)>.05
    stats={k:metrics(v,gt,bucket,mask)[0] for k,v in p.items()}
    score={k:np.linalg.norm(v-gt,axis=-1)@W for k,v in p.items()}
    groups=[np.flatnonzero(session==s) for s in sorted(set(sessions))];counts=np.array([len(i) for i in groups])
    draws=np.random.default_rng(0).integers(0,len(groups),(20000,len(groups)))
    for k in p:
        if k=='ON':continue
        delta=score[k]-score['ON'];sums=np.array([delta[i].sum() for i in groups]);boot=sums[draws].sum(1)/counts[draws].sum(1)
        stats[k]['minus_ON']={'PREFIX':float(delta.mean()),'session_CI95':np.quantile(boot,[.025,.975]).tolist(),'plan_PREFIX_distance':float((np.linalg.norm(p[k]-p['ON'],axis=-1)@W).mean())}
    stored_delta=None
    if stored:
        old=json.loads(stored.read_text())['records'];assert rows==[r['row'] for r in old]
        assert np.array_equal(gt,np.array([r['gt_abs_xy'] for r in old]));stored_delta=float(abs(p['ON']-np.array([r['pred_abs_xy'] for r in old])).max());assert stored_delta<5e-4
    assert max(invariance.values())==0,invariance
    dest=REPORT/f'{arm}_{scope}.npz'
    np.savez_compressed(dest,row=np.array(rows),session=session,gt=gt,gt_state=gs,gt_state_valid=vs,state=ps,history=ph,provided_status5=provided,bucket=bucket,**p)
    return dict(n=len(rows),models=stats,invariance_max_abs=invariance,source_prediction_parity_max_abs_m=stored_delta,row_sha256=hashlib.sha256(np.array(rows,dtype='<i8').tobytes()).hexdigest(),arrays_path=str(dest),arrays_sha256=sha(dest),elapsed_seconds=time.monotonic()-start)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=RUNS,required=True);ap.add_argument('--gpu',type=int,choices=range(4),required=True);args=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    REPORT.mkdir(parents=True,exist_ok=True);dest=REPORT/f'probe_{args.arm}.json';assert not dest.exists()
    torch.set_num_threads(4);base.trainer.seed_all(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    run,step=RUNS[args.arm];ckpt=run/f'ckpt_step{step}.pth';payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    config=base.MotionDriveV2Config(**payload['manifest']['model_config'])
    model=TemporalReadModel(config) if args.arm=='TEMPORAL' else base.ExperimentModel(config,arm=base.FRESH)
    base.mr.rebuild_correlation_fuse(model,4);model.load_state_dict(payload['model'],strict=True);del payload
    before=tensor_state_sha256(model.state_dict());assert model.planner.config.state_on is True
    model.cuda().eval().requires_grad_(False)
    train,tune=base.nominal.raw_datasets(False,1)
    index=np.sort(np.random.default_rng(20260920).choice(len(train),1024,replace=False))
    train=Subset(NominalStatusDataset(base.mr.MotionCanvasDataset(train,'native')),index.tolist());tune=NominalStatusDataset(base.mr.MotionCanvasDataset(tune,'native'))
    result=dict(arm=args.arm,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(Path(__file__)),model_state_sha256=before,physical_gpu=args.gpu,optimizer_updates=0,train_probe_seed=20260920,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),limitations=['Frozen OOD interventions; not trained ablations or submission candidates.','Train cohort is identical to earlier probe, but its distribution differs from V0.','Status perturbation measures local sensitivity, not a compliant numerical planner bypass.'])
    result['V0']=evaluate(model,tune,run/f'predictions_step{step}.json','V0',args.arm)
    result['train_probe']=evaluate(model,train,None,'train_probe',args.arm)
    assert before==tensor_state_sha256(model.state_dict());result['parameter_buffer_hash_unchanged']=True
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'completed':str(dest),'V0':result['V0']['models']['ON']['PREFIX'],'train':result['train_probe']['models']['ON']['PREFIX']}),flush=True)

if __name__=='__main__':main()
