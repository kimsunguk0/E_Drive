"""Geometry, initializer, gradient, A2 isolation and whole-forward cost checks."""
import copy
import gc
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from train_experiment import *
from nominal_data import NominalStatusDataset
from motion_model import coarse_correspondence, sample_neighborhood
from motiondrive_v2_training import model_inputs, to_device, LossWeights, compute_loss, build_loss_normalizers

KEYS = ('scene_features','occ_logits','lane_logits','motion_features','state_hat','history_hat','plan_abs')


def inputs(batch):
    out=mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal')
    out['provided_status5']=batch['provided_status5']
    return out


def from_upstream(fine):
    common=torch.load(OLD_INIT,map_location='cpu',weights_only=False)
    trainer.seed_all(1)
    config=MotionDriveV2Config(**trainer.initialization_configuration(common['manifest'],
        goal_on=1,state_on=1,explicit_arch='resnet50',cross_cell_goal_mode='zero',history_contract='control'))
    config.motion_input_mode='low_feature'
    model=ExperimentModel(config,arm=FINE) if fine else SceneExtensionModel(config,arm=QREFINE)
    mr.rebuild_correlation_fuse(model,4)
    state={k:v for k,v in common['model'].items() if not k.startswith('motion_encoder.correlation_fuse.0.')}
    missing=model.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys
    retained={k:v for k,v in model.state_dict().items() if not k.startswith(FINE_PREFIX)}
    assert tensor_state_sha256(retained)==CONTROL_INITIAL_SHA
    return model.cuda().eval()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    REPORT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    result={'status':'running','physical_gpu':2,'preflight_weights_used_for_training':False}
    # Exact translated one-hot correspondence: history(x+2,y-1) = current(x,y).
    h,w=9,11
    cur=torch.eye(h*w,device='cuda').T.reshape(1,h*w,h,w)
    past=torch.zeros_like(cur)
    past[:,:,:-1,2:]=cur[:,:,1:,:-2]
    flow,_=coarse_correspondence(cur,past,radius=3,temperature=.005)
    expected=flow.new_tensor([2.,-1.]).reshape(1,2,1,1)
    err=float((flow[:,:,1:,:-2]-expected).abs().max())
    assert err<1e-5,err
    reflected,_=coarse_correspondence(cur.flip(-1),past.flip(-1),radius=3,temperature=.005)
    reflected[:,0]*=-1
    assert torch.allclose(reflected.flip(-1),flow,atol=2e-5,rtol=0)
    ramp=(torch.arange(w,device='cuda')[None,:]+10*torch.arange(h,device='cuda')[:,None]).float()[None,None]
    dx=torch.zeros(1,2,h,w,device='cuda');dx[:,0]=1
    sampled,valid=sample_neighborhood(ramp,dx,radius=0)
    assert abs(sampled[0,0,0,3,4].item()-35)<1e-5
    assert not valid[0,0,:,-1].any() and not sampled[0,0,0,:,-1].count_nonzero()
    variable=dx.clone();variable[:,0]=torch.arange(w,device='cuda')[None,None,:]*.1
    sampled,_=sample_neighborhood(ramp,variable,radius=1)
    assert abs(sampled[0,0,5,3,4].item()-35.4)<1e-4 # center dx=.4 plus local +1
    result['geometry']={'translation_max_error_cells':err,'flip_flow_covariance':True,
                        'ramp_positive_dx':35.,'spatially_varying_warp_ramp':35.4,'out_of_bounds_zero':True}
    del cur,past,flow,reflected,ramp,sampled,valid,variable,dx
    _,tune=nominal.raw_datasets(False,1)
    data=NominalStatusDataset(mr.MotionCanvasDataset(tune,'native'))
    raw=default_collate([data[0]])
    batch=to_device(raw,torch.device('cuda:0')); x=inputs(batch)
    base=from_upstream(False);fine=from_upstream(True)
    result['zero_init_parity']={}
    for precision in ('fp32','bf16'):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=base(**x);b=fine(**x)
        difference={k:float((a[k].float()-b[k].float()).abs().max()) for k in KEYS}
        assert max(difference.values())==0,difference
        result['zero_init_parity'][precision]=difference
        del a,b
    result['initial_shared_sha256']=CONTROL_INITIAL_SHA
    # Real planning loss must open the new branch and subsequently its descriptor.
    module=fine.motion_encoder.fine_match
    def plan_loss(out):
        weights=out['plan_abs'].new_tensor([11,11,5,5,2,2])/36
        return (torch.linalg.vector_norm(out['plan_abs']-batch['gt_plan'],dim=-1)*weights).sum(-1).mean()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=fine(**x);loss=plan_loss(output)
    loss.backward()
    first=float(module.output.weight.grad.norm())
    assert first>0 and torch.isfinite(module.output.weight.grad).all()
    with torch.no_grad(): module.output.weight.add_(module.output.weight.grad,alpha=-1e-3)
    fine.zero_grad(set_to_none=True);del output,loss
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=fine(**x);loss=plan_loss(output)
    loss.backward()
    grads={name:float(p.grad.norm()) if p.grad is not None else None
           for name,p in fine.named_parameters() if name in (
               FINE_PREFIX+'descriptor.weight',FINE_PREFIX+'refine.0.weight',
               'motion_encoder.projections.1.weight','backbone_fpn.layer1.0.conv1.weight')}
    assert all(v is not None and v>0 for v in grads.values()),grads
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in fine.parameters())
    result['gradients']={'initial_output_norm':first,'after_disposable_output_step':grads}
    fine.zero_grad(set_to_none=True);del output,loss
    seen={}
    hooks=[fine.scene_encoder.occ_head.register_forward_pre_hook(lambda m,args:seen.update(occ=args[0])),
           fine.scene_encoder.lane_head.register_forward_pre_hook(lambda m,args:seen.update(lane=args[0])),
           fine.planner.register_forward_pre_hook(lambda m,args:seen.update(planner=args[0]))]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        normal=fine(**x)
    assert seen['occ'] is seen['lane']
    assert torch.equal(seen['occ'].flatten(2).transpose(1,2).float(),seen['planner'].float())
    for hdl in hooks: hdl.remove()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        changed=fine(**dict(x,provided_status5=x['provided_status5']+5,goal_xy=x['goal_xy']+10))
    invariant={k:float((normal[k]-changed[k]).abs().max()) for k in ('motion_features','state_hat','history_hat')}
    assert max(invariant.values())==0
    result['A2_isolation']=invariant
    result['shared_scene_consumers_same_tensor']=True
    del normal,changed,seen
    from motiondrive_v2_flip_augment import flip_item
    flip=mr.wrap_flip_item(flip_item)
    item=data[0];twice=flip(flip(item,768,384),768,384)
    for k,v in item.items():
        if isinstance(v,torch.Tensor):
            assert torch.allclose(v,twice[k],atol=1e-4,rtol=0) if v.is_floating_point() else torch.equal(v,twice[k]),k
    result['existing_input_double_flip']=True
    # Fresh model loads only the verified public trunk; all nontrunk state was
    # recorded before/after the load. Reconstruction is strict and output finite.
    result['fresh_initializer']=ensure_fresh_initializer()
    common=torch.load(FRESH_INIT,map_location='cpu',weights_only=False)
    fresh=ExperimentModel(MotionDriveV2Config(**common['manifest']['model_config']),arm=FRESH)
    mr.rebuild_correlation_fuse(fresh,4);fresh.load_state_dict(common['model'],strict=True)
    assert tensor_state_sha256(fresh.state_dict())==result['fresh_initializer']['model_state_sha256']
    fresh=fresh.cuda().eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=fresh(**x)['plan_abs']
    assert pred.shape==(1,6,2) and torch.isfinite(pred).all()
    result['fresh_initial_plan_abs_max_m']=float(pred.abs().max())
    del fresh,common,pred
    gc.collect();torch.cuda.empty_cache()
    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.module_tracker as mt
    class NoHandle:
        def remove(self): pass
    old=mt.register_multi_grad_hook;mt.register_multi_grad_hook=lambda *a,**k:NoHandle()
    result['whole_forward_cost']={}
    module.collect_stats=True
    try:
        for name,model in [('QREFINE',base),('C2F',fine)]:
            with torch.no_grad(),FlopCounterMode(display=False) as count:
                model(**x)
            flops=int(sum(count.get_flop_counts().get('Global',{}).values()))
            times=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(12):
                    torch.cuda.synchronize();tick=time.monotonic();model(**x);torch.cuda.synchronize()
                    if i>=3:times.append(1000*(time.monotonic()-tick))
            result['whole_forward_cost'][name]={'official_counter_flops':flops,
                'B200_B1_forward_median_ms':float(np.median(times)),
                'parameters':sum(p.numel() for p in model.parameters()),'RTX4090_ms':None}
            assert flops<7053e9
    finally:mt.register_multi_grad_hook=old
    result['fine_trace']=module.last_stats
    result.update(status='passed',elapsed_seconds=time.monotonic()-start,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (REPORT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
