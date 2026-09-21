"""Actual checkpoint parity, boundaries, nonzero detail gradients and strict fresh-process export."""
from pathlib import Path
import argparse,copy,gc,json,sys
import numpy as np
import torch
from torch.utils.data import default_collate,DataLoader,Subset
from train_native import *
from native_model import NativeDetailModel,load_export,manifest_for,SIGNATURE
from next_model import NextModel
from models.motiondrive_v2.config import MotionDriveV2Config
from motiondrive_v2_training import to_device
from models.motiondrive_v2.scene_encoder import pixel_to_normalized_grid

KEYS=('plan_abs','scene_features','motion_features','motion_pair_features','state_hat','history_hat','occ_logits','lane_logits')
ROWS=(0,30000,60000)

def datasets(arm):
    tr,va=nominal.raw_datasets(False,1)
    def wrap(ds):return NativeDataset(H4StatusDataset(mr.MotionCanvasDataset(ds,'native')),arm)
    return wrap(tr),wrap(va)

BASE_INPUTS=trainer.model_inputs
def inputs(batch,**kw):
    x=mr.model_inputs_with_canvas(BASE_INPUTS,batch,time_input='nominal',nominal_history_seconds=(.1,.2,.5,1.))
    x['provided_status5']=batch['provided_status5'];x['high_images']=batch[KEY];return x

def diff(a,b,keys=KEYS):return {k:float((a[k].float()-b[k].float()).abs().max()) for k in keys}
def forward(model,x):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**x)

def initial(arm):
    assert sha(PARENT)==PARENT_SHA
    cp=torch.load(PARENT,map_location='cpu',weights_only=False);config=MotionDriveV2Config(**cp['manifest']['model_config'])
    trainer.seed_all(1);base=NextModel(copy.deepcopy(config),execution_config=arm_config(arm)['parent_next_execution_config'],include_training_aux=False)
    mr.rebuild_correlation_fuse(base,4);rng=torch.get_rng_state().clone()
    trainer.seed_all(1);model=NativeDetailModel(copy.deepcopy(config),execution_config=arm_config(arm))
    mr.rebuild_correlation_fuse(model,4);assert torch.equal(rng,torch.get_rng_state())
    base.load_state_dict(cp['model'],strict=True);extras=load_parent(model,cp['model'])
    tr,va=datasets(arm);raw=default_collate([tr[i] for i in ROWS]);x=inputs(to_device(raw,torch.device('cuda')))
    xx={k:v for k,v in x.items() if k!='high_images'};base.cuda().eval();model.cuda().eval()
    result={'arm':arm,'parent_sha256':PARENT_SHA,'parameters':sum(p.numel() for p in model.parameters()),
        'CPU_RNG_matches_parent':True,'initial_parity':{},'new_keys':extras}
    for precision in ('fp32','bf16'):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=base(**xx);b=model(**x)
        delta=diff(a,b);assert max(delta.values())<1e-4,delta;result['initial_parity'][precision]=delta
    del base,a,b;gc.collect();torch.cuda.empty_cache()
    # Exercise information routes with a NONZERO detail branch, not just zero-init.
    with torch.no_grad():model.native_projection.weight.copy_(torch.eye(128,device='cuda')[:,:,None,None]*.01)
    out=forward(model,x);result['boundary']={}
    for key,amount in [('provided_status5',5.),('goal_xy',10.)]:
        changed=dict(x);changed[key]=changed[key]+amount
        delta=diff(out,forward(model,changed),('motion_features','motion_pair_features','state_hat','history_hat'))
        assert max(delta.values())==0,delta;result['boundary'][key]=delta
    if arm.startswith('S-'):
        captures={'occ':[],'lane':[],'planner':[]}
        hooks=[model.scene_encoder.occ_head.register_forward_pre_hook(lambda m,a:captures['occ'].append(a[0])),
               model.scene_encoder.lane_head.register_forward_pre_hook(lambda m,a:captures['lane'].append(a[0])),
               model.planner.register_forward_pre_hook(lambda m,a:captures['planner'].append(a[0]))]
        try:shared=forward(model,x)
        finally:
            for h in hooks:h.remove()
        a,b,c=captures['occ'][-1],captures['lane'][-1],captures['planner'][-1]
        assert a.data_ptr()==b.data_ptr()==c.data_ptr()==shared['scene_features'].data_ptr()
        result['same_final_scene_all_consumers']=True
        del captures,shared,a,b,c
    with torch.no_grad():model.native_projection.weight.zero_()
    # Exact augmentation involution and normalization coordinates across canvases.
    sample=tr[0];flip=wrap_flip(mr.wrap_flip_item(flip_api.flip_item),arm);twice=flip(flip(sample,768,384),768,384)
    roundtrip={}
    for k,v in sample.items():
        if not isinstance(v,torch.Tensor):continue
        same=torch.equal(v,twice[k])
        delta=float((v.double()-twice[k].double()).abs().max()) if v.numel() else 0.
        roundtrip[k]={'exact':same,'max_abs':delta}
        # Existing float32 camera reflection applies a subtract twice. Its
        # round-trip can differ by one ULP; all images/status/targets stay exact.
        if k=='lidar2img':
            assert torch.allclose(v,twice[k],atol=6.2e-5,rtol=0),(k,delta)
        else:assert same,(k,delta)
    result['flip_roundtrip_by_tensor']=roundtrip
    uv=torch.tensor([[4.2,7.9],[342.4,235.6]],dtype=torch.float64)
    uhi=(uv+.5)*1.5-.5
    assert torch.allclose(pixel_to_normalized_grid(uv,(432,768)),pixel_to_normalized_grid(uhi,(648,1152)),atol=1e-12,rtol=0)
    result['flip_involution']=True;result['normalized_geometry_consistent']=True
    stage=RUNS/'verification';stage.mkdir(parents=True,exist_ok=True)
    expected=forward(model,x)
    torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},'manifest':manifest_for(model,cp['manifest']),'step':0},stage/f'{arm}_initial.pth')
    torch.save({k:expected[k].cpu() for k in KEYS},stage/f'{arm}_expected.pth')
    torch.save({k:v.cpu() for k,v in x.items()},stage/f'{arm}_inputs.pth')
    del expected,out,x,xx,raw;gc.collect();torch.cuda.empty_cache()
    before=tensor_state_sha256(model.state_dict());old=trainer.model_inputs;trainer.model_inputs=inputs
    try:
        r,rr=trainer.evaluate(model,DataLoader(va,batch_size=8,num_workers=8),torch.device('cuda'),'bf16',
            time_input='nominal',detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
        atomic(REPORT/f'{arm}_step0.json',{'report':r,'records':rr})
        idx=np.random.default_rng(20260921).choice(len(tr),256,replace=False).tolist()
        pr,records=trainer.evaluate(model,DataLoader(Subset(tr,idx),batch_size=8,num_workers=4),torch.device('cuda'),'bf16',
            time_input='nominal',detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
        atomic(REPORT/f'{arm}_train_probe_step0.json',{'report':pr,'records':records,'indices':idx})
    finally:trainer.model_inputs=old
    assert before==tensor_state_sha256(model.state_dict())
    assert abs(r['official_d3']-.14739373370951964)<1e-6,r['official_d3']
    result.update(status='passed',step0_PREFIX=r['official_d3'],step0_n=r['n'],initial_state_sha256=before)
    atomic(REPORT/f'{arm}_initial_checks.json',result);print(json.dumps(result),flush=True)

def reload(arm):
    stage=RUNS/'verification';x=torch.load(stage/f'{arm}_inputs.pth',map_location='cuda',weights_only=True)
    model,cp=load_export(stage/f'{arm}_initial.pth','cuda');expected=torch.load(stage/f'{arm}_expected.pth',map_location='cuda',weights_only=True)
    initial_diff=diff(forward(model,x),expected);assert max(initial_diff.values())<1e-5,initial_diff
    initial=cp['model'];del model,expected;gc.collect();torch.cuda.empty_cache()
    model,cp=load_export(RUNS/f'{arm}-s1-smoke/ckpt_step5.pth','cuda')
    manifest=json.loads((RUNS/f'{arm}-s1-smoke/manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['step']==5 and manifest['nonfinite_count']==0
    names=['native_projection.weight']
    if arm.startswith('M-'):names+=['native_motion.projections.0.weight','native_motion.correlation_fuse.0.weight']
    assert all(not torch.equal(initial[n],cp['model'][n]) for n in names)
    torch.save({k:cp[k] for k in ('model','manifest','step')},stage/f'{arm}_student.pth')
    actual=forward(model,x);torch.save({k:actual[k].cpu() for k in KEYS},stage/f'{arm}_student_expected.pth')
    bad=copy.deepcopy(cp);other=arm[0]+('-NATIVE' if arm.endswith('LOW') else '-LOW')
    bad['manifest']['native_execution_config']=arm_config(other);torch.save(bad,stage/f'{arm}_wrong.pth')
    try:load_export(stage/f'{arm}_wrong.pth')
    except ValueError:pass
    else:raise AssertionError('Wrong detail contract accepted')
    atomic(REPORT/f'{arm}_reload.json',{'status':'passed','initial_fresh_process_diff':initial_diff,
        'new_branch_weights_changed':names,'wrong_graph_rejected':True,'stream':manifest['stream_audit']})

def export(arm):
    stage=RUNS/'verification';x=torch.load(stage/f'{arm}_inputs.pth',map_location='cuda',weights_only=True)
    model,_=load_export(stage/f'{arm}_student.pth','cuda')
    expected=torch.load(stage/f'{arm}_student_expected.pth',map_location='cuda',weights_only=True)
    delta=diff(forward(model,x),expected);assert max(delta.values())<1e-5,delta
    atomic(REPORT/f'{arm}_export.json',{'status':'passed','strict_fresh_process_diff':delta})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('initial','reload','export'),required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    {'initial':initial,'reload':reload,'export':export}[a.mode](a.arm)
