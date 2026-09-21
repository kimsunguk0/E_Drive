"""Actual DEV factory parity, changed-route gradients, step0 and fresh-process reload."""
import argparse,copy,gc,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate,DataLoader,Subset
from train_fourarm import *
from arm_model import load_export,SIGNATURE
from models.motiondrive_v2.config import MotionDriveV2Config
from progress_model import H4ProgressModel,PROGRESS
from motiondrive_v2_training import (model_inputs,to_device,compute_loss,LossWeights,
                                    build_loss_normalizers,set_training_mode)

KEYS=('plan_abs','scene_features','occ_logits','lane_logits','motion_features',
      'motion_pair_features','state_hat','history_hat')
def inputs(batch,**kwargs):
    x=mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal')
    x['provided_status5']=batch['provided_status5'];return x
def forward(model,x):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**x)
def differences(a,b,keys=KEYS):return {k:float((a[k].float()-b[k].float()).abs().max()) for k in keys}
def datasets():
    tr,va=nominal.raw_datasets(False,1)
    return tuple(H4StatusDataset(mr.MotionCanvasDataset(x,'native')) for x in (tr,va))
def manifest(model):
    pm=json.loads((PARENT_RUN/'manifest.json').read_text())
    return {'model_config':model.config.to_dict(),'execution_config':model.execution_config,
            'split_sha256':pm['split_sha256'],'parent_sha256':PARENT_SHA}

def initial(arm):
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Parent bytes changed')
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    conf=MotionDriveV2Config(**cp['manifest']['model_config'])
    trainer.seed_all(1);ref=H4ProgressModel(copy.deepcopy(conf),arm=PROGRESS)
    mr.rebuild_correlation_fuse(ref,4);rng_ref=torch.get_rng_state().clone()
    ref.load_state_dict(cp['model'],strict=True)
    trainer.seed_all(1);model=FourArmModel(copy.deepcopy(conf),execution_config=arm_config(arm))
    mr.rebuild_correlation_fuse(model,4);assert torch.equal(rng_ref,torch.get_rng_state())
    extras=load_dev_parent(model,cp['model']);del cp
    tr,va=datasets();raw=default_collate([tr[0],tr[17]])
    batch=to_device(raw,torch.device('cuda'));x=inputs(batch)
    model.cuda().eval();ref.cuda().eval()
    result={'arm':arm,'parent_sha256':PARENT_SHA,'parent_tensors_exact':True,
            'constructor_RNG_matches_parent':True,'extra_keys':extras,'initial_parity':{}}
    for precision in ('fp32','bf16'):
        xx=dict(x)
        if arm=='P-SHARED768':xx['history_images']=x['motion_history']
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=ref(**xx);b=model(**x)
        diff=differences(a,b);result['initial_parity'][precision]=diff
        assert max(diff.values())==0,diff
        del a,b
    result['parity_reference']='separate_768_history' if arm=='P-SHARED768' else 'frozen_DEV_parent_384_history'
    if arm=='P-SHARED768':
        a=forward(ref,x);b=forward(model,x)
        result['legitimate_change_vs_old384']=differences(a,b);del a,b
    del ref;gc.collect();torch.cuda.empty_cache()
    result['input_boundary']={}
    out=forward(model,x)
    for key,delta in [('provided_status5',5.),('goal_xy',10.)]:
        changed=dict(x);changed[key]=changed[key]+delta
        other=forward(model,changed)
        diff=differences(out,other,('motion_features','motion_pair_features','state_hat','history_hat'))
        assert max(diff.values())==0
        result['input_boundary'][key]=diff
        del other
    # Save expected output from this process; another process reconstructs graph.
    stage=RUNS/'verification';stage.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'manifest':manifest(model),'step':0},stage/f'{arm}_initial.pth')
    torch.save({k:out[k].cpu() for k in KEYS},stage/f'{arm}_expected.pth');del out
    model.cuda().eval()
    # Loss routing identity on actual outputs: objective replacement, not stacking.
    out=forward(model,x);w=LossWeights(occupancy=.2,lane=.2,motion=.2,uncertainty=True)
    norm=to_device(build_loss_normalizers(raw),torch.device('cuda'))
    ll,lp=length_wrapper(compute_loss,.25)(out,batch,w,normalizers=norm)
    vl,vp=vector_wrapper(compute_loss,.25)(out,batch,w,normalizers=norm)
    expected=.25*(vp['plan_interval_vector']-lp['plan_interval_length'])
    assert torch.allclose(vl-ll,expected,atol=2e-6,rtol=2e-6)
    result['vector_replaces_length_total_identity_error']=float(abs(vl-ll-expected))
    del out,ll,vl,lp,vp
    # Check same shared scene consumers and autograd paths with actual DEV input.
    seen={};handles=[model.scene_encoder.occ_head.register_forward_pre_hook(lambda m,a:seen.update(occ=a[0])),
        model.scene_encoder.lane_head.register_forward_pre_hook(lambda m,a:seen.update(lane=a[0])),
        model.planner.register_forward_pre_hook(lambda m,a:seen.update(planner=a[0]))]
    if arm=='P-SHARED768':
        handles.append(model.scene_encoder.register_forward_pre_hook(lambda m,a:seen.update(history=a[1])))
    set_training_mode(model,'fixed')
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**x)
    assert seen['occ'] is seen['lane']
    assert torch.equal(seen['occ'].flatten(2).transpose(1,2).float(),seen['planner'].float())
    if arm=='P-SHARED768':
        scene_loss=out['scene_features'].float().square().mean()
        grads=torch.autograd.grad(scene_loss,seen['history'],retain_graph=True,allow_unused=False)
        result['scene_gradient_to_shared_native_FPN']=[float(v.norm()) for v in grads]
        assert all(v>0 for v in result['scene_gradient_to_shared_native_FPN'])
    loss,_=(vector_wrapper if arm=='P-VECTOR' else length_wrapper)(compute_loss,.25)(out,batch,w,normalizers=norm)
    loss.backward()
    result['initial_backward_finite']=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert result['initial_backward_finite']
    for h in handles:h.remove()
    model.zero_grad(set_to_none=True);del out,loss,seen
    # No update above. Step0 evaluates all 1,998 rows using the real execution graph.
    before=tensor_state_sha256(model.state_dict());original_inputs=trainer.model_inputs
    trainer.model_inputs=inputs
    try:
        report,records=trainer.evaluate(model,DataLoader(va,batch_size=8,num_workers=8),torch.device('cuda'),
            'bf16',time_input='nominal',detailed_records=True)
        report.update(step=0,parent_step=20554)
        atomic(REPORT/f'{arm}_step0.json',{'report':report,'records':records})
        idx=np.random.default_rng(20260921).choice(len(tr),256,replace=False).tolist()
        pr,rr=trainer.evaluate(model,DataLoader(Subset(tr,idx),batch_size=8,num_workers=4),torch.device('cuda'),
            'bf16',time_input='nominal',detailed_records=True)
        atomic(REPORT/f'{arm}_train_probe_step0.json',{'report':pr,'records':rr,'indices':idx})
    finally:trainer.model_inputs=original_inputs
    assert before==tensor_state_sha256(model.state_dict())
    result.update(status='passed',step0_PREFIX=report['official_d3'],step0_n=report['n'],
        parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=before)
    atomic(REPORT/f'{arm}_initial_checks.json',result)
    print(json.dumps(result),flush=True)

def reload(arm):
    tr,_=datasets();x=inputs(to_device(default_collate([tr[0],tr[17]]),torch.device('cuda')))
    stage=RUNS/'verification';model,p=load_export(stage/f'{arm}_initial.pth','cuda')
    expected=torch.load(stage/f'{arm}_expected.pth',weights_only=True,map_location='cuda')
    actual=forward(model,x);diff=differences(actual,expected);assert max(diff.values())==0,diff
    del expected,actual,model
    path=RUNS/f'{arm}-s1-smoke/ckpt_step5.pth'
    model,p=load_export(path,'cuda');actual=forward(model,x)
    assert actual['plan_abs'].shape==(2,6,2) and torch.isfinite(actual['plan_abs']).all()
    run=path.parent;m=json.loads((run/'manifest.json').read_text());d=json.loads((run/'experiment.json').read_text())
    assert m['status']=='completed' and m['step']==5 and m['nonfinite_count']==0
    if arm=='P-FINE':
        grad=json.loads((run/'gradient_audit.json').read_text())
        assert grad[0]['unclipped_grad_norms']['planner.fine_read.attention.out_proj.weight']>0
        assert any(v['unclipped_grad_norms']['planner.fine_read.attention.in_proj_weight']>0 for v in grad[1:])
    # Strict student-only export and a third fresh process compare smoke output.
    torch.save({'model':p['model'],'manifest':p['manifest'],'step':5},stage/f'{arm}_smoke_export.pth')
    torch.save({k:actual[k].cpu() for k in KEYS},stage/f'{arm}_smoke_expected.pth')
    # Wrong same-key configuration is caught by the persistent config digest.
    bad=copy.deepcopy(p);other='P-SHARED768' if arm!='P-SHARED768' else 'P-CTRL'
    bad['manifest']['execution_config']=arm_config(other)
    bad_path=stage/f'{arm}_bad_config.pth';torch.save(bad,bad_path)
    try:load_export(bad_path)
    except ValueError:pass
    else:raise AssertionError('Wrong explicit graph accepted')
    atomic(REPORT/f'{arm}_reload.json',{'status':'passed','initial_fresh_process_maxdiff':diff,
        'smoke_checkpoint_strict_loaded':True,'wrong_config_rejected':True,'smoke_stream':m['stream_audit'],
        'initial_hash_matches_smoke':d['expected_initial_model_state_sha256']==p['manifest']['initial_model_state_sha256']})

def export_check(arm):
    tr,_=datasets();x=inputs(to_device(default_collate([tr[0],tr[17]]),torch.device('cuda')))
    stage=RUNS/'verification';model,_=load_export(stage/f'{arm}_smoke_export.pth','cuda')
    expected=torch.load(stage/f'{arm}_smoke_expected.pth',map_location='cuda',weights_only=True)
    diff=differences(forward(model,x),expected);assert max(diff.values())==0
    atomic(REPORT/f'{arm}_export.json',{'status':'passed','fresh_process_smoke_export_maxdiff':diff})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--mode',choices=('initial','reload','export'),required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    {'initial':initial,'reload':reload,'export':export_check}[args.mode](args.arm)
if __name__=='__main__':main()
