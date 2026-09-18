"""Changed-path contracts for learned shared-scene sampling, on actual inputs."""
import json
import os
import time
import numpy as np
from torch.utils.data import default_collate
from common import *
from learned_sample import LearnedSampleModel, shifted_grid, OFFSET_PREFIX
from models.motiondrive_v2.scene_encoder import project_scene_points, pixel_to_normalized_grid, masked_softmax

CHECK=('scene_features','occ_logits','lane_logits','plan_abs','motion_features','state_hat','history_hat')
def diffs(a,b,keys=CHECK):return {k:float((a[k].float()-b[k].float()).abs().max()) for k in keys}

def make_model(sample,common):
    trainer.seed_all(1)
    cfg=MotionDriveV2Config(**trainer.initialization_configuration(common['manifest'],
        goal_on=1,state_on=1,explicit_arch='resnet50',cross_cell_goal_mode='zero',history_contract='control'))
    cfg.motion_input_mode='low_feature'
    model=LearnedSampleModel(cfg) if sample else A2NominalModel(cfg,arm='A2-BASE-NOM')
    mr.rebuild_correlation_fuse(model,4)
    state={k:v for k,v in common['model'].items() if not k.startswith('motion_encoder.correlation_fuse.0.')}
    missing=model.load_state_dict(state,strict=False);assert not missing.unexpected_keys
    shared={k:v for k,v in model.state_dict().items() if not k.startswith(OFFSET_PREFIX)}
    assert tensor_state_sha256(shared)==nominal.BASE_INITIAL_SHA
    return model.cuda().eval()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();result={'status':'running','physical_gpu':3,'optimizer_candidate_steps':0}
    _,tune=nominal.raw_datasets(False,1)
    data=NominalStatusDataset(mr.MotionCanvasDataset(tune,'native'))
    raw=default_collate([data[i] for i in range(2)])
    batch=to_device(raw,torch.device('cuda:0'));x=inputs(batch)
    common=torch.load(legacy.INITIALIZER,map_location='cpu',weights_only=False)
    base=make_model(False,common);sample=make_model(True,common);del common
    result['shared_initial_sha256']=nominal.BASE_INITIAL_SHA
    result['zero_offset_parity']={}
    for mode,amp in [('FP32',False),('BF16',True)]:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            a=base(**x);b=sample(**x)
        value=diffs(a,b);assert max(value.values())<5e-4
        result['zero_offset_parity'][mode]=value
        del a,b
    enc=sample.scene_encoder
    # A +1 feature-cell displacement has the expected align_corners=False effect.
    feature=(torch.arange(12,device='cuda')[None,:]+10*torch.arange(8,device='cuda')[:,None]).float()[None,None,None]
    uv=torch.tensor([4.,3.],device='cuda').reshape(1,1,1,1,2)
    grid=pixel_to_normalized_grid(uv,(8,12));valid=torch.ones((1,1,1,1),dtype=torch.bool,device='cuda')
    offset=torch.tensor([1.,0.],device='cuda').reshape(1,1,1,1,2)
    moved,mask=shifted_grid(grid,offset,(8,12),valid,(8,12))
    got=enc._sample(feature,moved).item();assert abs(got-35.)<1e-5 and bool(mask.all())
    reflected=grid.clone();reflected[...,0]*=-1
    flipped_offset=offset.clone();flipped_offset[...,0]*=-1
    reflected_moved,_=shifted_grid(reflected,flipped_offset,(8,12),valid,(8,12))
    mirror=enc._sample(feature.flip(-1),reflected_moved).item();assert abs(mirror-got)<1e-5
    # Invalid sources never become visible, and out-of-image offsets are masked.
    invalid=torch.zeros_like(valid)
    _,invalid_mask=shifted_grid(grid,offset,(8,12),invalid,(8,12));assert not bool(invalid_mask.any())
    outside_grid=pixel_to_normalized_grid(torch.tensor([11.,3.],device='cuda').reshape_as(grid),(8,12))
    _,outside_mask=shifted_grid(outside_grid,offset,(8,12),valid,(8,12));assert not bool(outside_mask.any())
    scores=torch.randn(2,5,device='cuda',requires_grad=True)
    attention=masked_softmax(scores,torch.zeros_like(scores,dtype=torch.bool))
    assert not bool(attention.count_nonzero());attention.sum().backward();assert not bool(scores.grad.count_nonzero())
    result['ramp']={'positive_x_one_cell_value':got,'mirrored_value':mirror,'initial_invalid_stays_invalid':True,'outbound_mask':True,'all_invalid_attention_zero':True}
    # Actual calibration/pose/flip consistency, retaining the old 1px fringe policy.
    from motiondrive_v2_flip_augment import flip_item
    flip=mr.wrap_flip_item(flip_item)
    item=data[0];flipped=flip(item,768,384);twice=flip(flipped,768,384)
    for k,v in item.items():
        if isinstance(v,torch.Tensor):
            assert torch.allclose(v,twice[k],atol=1e-4,rtol=0) if v.is_floating_point() else torch.equal(v,twice[k]),k
    fx=inputs(to_device(default_collate([flipped]),torch.device('cuda:0')))
    matrices,_,_=enc._view_geometry(x['lidar2img'][:1],x['history_transforms'][:1],x['time_offsets'][:1])
    fm,_,_=enc._view_geometry(fx['lidar2img'],fx['history_transforms'],fx['time_offsets'])
    points=enc.points;refl=points.clone();refl[...,1]*=-1
    a,av=project_scene_points(points,matrices[:,[0,2,1,3,5,4,6,7,8,9]],(432,768))
    b,bv=project_scene_points(refl,fm,(432,768));mask=av&bv
    mirror_error=max(float((a[...,0][mask]+b[...,0][mask]).abs().max()),float((a[...,1][mask]-b[...,1][mask]).abs().max()))
    assert mirror_error<3e-5
    result['projection_flip_max_abs']=mirror_error
    # Gradient opening uses only a disposable output-layer SGD step.
    module=enc.offset_predictor;weight=x['images'].new_tensor([11,11,5,5,2,2])/36
    def loss(out):return (torch.linalg.vector_norm(out['plan_abs']-batch['gt_plan'],dim=-1)*weight).sum(-1).mean()
    with torch.autocast('cuda',dtype=torch.bfloat16):out=sample(**x);value=loss(out)
    value.backward()
    result['offset_output_initial_gradient']=float(module.network[-1].weight.grad.norm())
    assert result['offset_output_initial_gradient']>0 and torch.isfinite(module.network[-1].weight.grad).all()
    torch.optim.SGD(module.network[-1].parameters(),lr=1e-3).step()
    sample.zero_grad(set_to_none=True);del out,value
    with torch.autocast('cuda',dtype=torch.bfloat16):out=sample(**x);value=loss(out)
    value.backward()
    result['offset_internal_gradient_after_output_step']=float(module.network[0].weight.grad.norm())
    result['image_value_projection_gradient']=float(enc.value_proj[0].weight.grad.norm())
    assert result['offset_internal_gradient_after_output_step']>0
    assert result['image_value_projection_gradient']>0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in sample.parameters())
    sample.zero_grad(set_to_none=True);del out,value
    # A2 isolation after the offset module has a nonzero output.
    offset_records=[]
    hook=module.register_forward_hook(lambda m,args,out:offset_records.append(out.detach().cpu()))
    seen={}
    occ_hook=enc.occ_head.register_forward_pre_hook(lambda m,args:seen.update(occ=args[0]))
    lane_hook=enc.lane_head.register_forward_pre_hook(lambda m,args:seen.update(lane=args[0]))
    planner_hook=sample.planner.register_forward_pre_hook(lambda m,args:seen.update(planner=args[0]))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):normal=sample(**x)
    original_offsets=offset_records[:];offset_records.clear()
    assert seen['occ'] is seen['lane']
    assert torch.equal(seen['occ'].flatten(2).transpose(1,2).float(),seen['planner'].float())
    assert torch.equal(normal['scene_features'].float(),seen['planner'].float())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        changed=sample(**dict(x,provided_status5=x['provided_status5']+5,goal_xy=x['goal_xy']+10))
    assert len(original_offsets)==len(offset_records) and all(torch.equal(a,b) for a,b in zip(original_offsets,offset_records))
    isolation=diffs(normal,changed,('motion_features','state_hat','history_hat'))
    assert max(isolation.values())==0
    result.update(status_goal_offset_forward_invariant=True,motion_isolation=isolation,shared_consumers_same_tensor=True)
    for h in [hook,occ_hook,lane_hook,planner_hook]:h.remove()
    del normal,changed,original_offsets,offset_records,seen
    # Complete single-scene forward: real B1 shapes, shared official counter.
    tiny={k:v[:1] for k,v in x.items()}
    enc.collect_sampling_stats=True;enc.sampling_stats=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):sample(**tiny)
    result['offset_trace']=enc.sampling_stats;enc.collect_sampling_stats=False
    result['cost']={}
    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.module_tracker as mt
    class NoHandle:
        def remove(self):pass
    old=mt.register_multi_grad_hook;mt.register_multi_grad_hook=lambda *a,**k:NoHandle()
    try:
        for name,model in [('BASE',base),('SAMPLE',sample)]:
            with torch.no_grad(),FlopCounterMode(display=False) as counter:model(**tiny)
            flops=sum(counter.get_flop_counts().get('Global',{}).values())
            times=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(14):
                    torch.cuda.synchronize();t=time.monotonic();model(**tiny);torch.cuda.synchronize()
                    if i>=4:times.append(1000*(time.monotonic()-t))
            result['cost'][name]={'official_counter_flops':int(flops),'B200_B1_forward_ms_median':float(np.median(times)),
                'parameters':sum(p.numel() for p in model.parameters()),'RTX4090_forward_ms':None}
    finally:mt.register_multi_grad_hook=old
    assert result['cost']['SAMPLE']['official_counter_flops']<7053e9
    result.update(status='passed',elapsed_seconds=time.monotonic()-start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        temporary_gradient_step='disposable output layer only; no preflight weights saved or used for training',
        range='Per-axis radius is 2 * image_dimension / actual_feature_dimension; coarse history has 14 rows, so its vertical radius differs from the nominal stride16 value.',
        cost_scope='whole top-level B1 forward, no frame division; B200 timing is not RTX4090 latency')
    (REPORT/'tests_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
