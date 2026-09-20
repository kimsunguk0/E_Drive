"""Parity, gradient, temporal memory boundary and actual whole-forward cost."""
import copy
import gc
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from train_temporal import *
from nominal_data import NominalStatusDataset
from motiondrive_v2_training import (model_inputs, to_device, LossWeights, compute_loss,
    build_loss_normalizers, set_training_mode)
from train_motiondrive_v2 import slice_batch
from length_auxiliary import wrap_compute_loss

KEYS = ('scene_features','occ_logits','lane_logits','motion_features','state_hat',
        'history_hat','state_logvar','history_logvar','plan_abs')


def inputs(batch):
    value=mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal')
    value['provided_status5']=batch['provided_status5']
    return value


def plan_loss(output,batch):
    w=output['plan_abs'].new_tensor([11,11,5,5,2,2])/36
    return (torch.linalg.vector_norm(output['plan_abs']-batch['gt_plan'],dim=-1)*w).sum(-1).mean()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not (REPORT/'preflight.json').exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    result={'status':'running','physical_gpu':0,'checked_weights_used_for_training':False}
    source=base.ensure_fresh_initializer()
    payload=torch.load(base.FRESH_INIT,map_location='cpu',weights_only=False)
    config=base.MotionDriveV2Config(**payload['manifest']['model_config'])
    trainer.seed_all(1)
    control=base.ExperimentModel(copy.deepcopy(config),arm=base.FRESH)
    mr.rebuild_correlation_fuse(control,4)
    old_rng=torch.get_rng_state().clone()
    control.load_state_dict(payload['model'],strict=True)
    trainer.seed_all(1)
    model=TemporalReadModel(copy.deepcopy(config))
    mr.rebuild_correlation_fuse(model,4)
    assert torch.equal(old_rng,torch.get_rng_state())
    result['constructor_rng_matches_control']=True
    extras=load_fresh_parent(model,payload['model'])
    shared={k:v for k,v in model.state_dict().items() if not k.startswith(NEW_PREFIX)}
    assert tensor_state_sha256(shared)==source['model_state_sha256']
    assert all(torch.equal(control.state_dict()[k],v) for k,v in shared.items())
    result['control_initial_state_sha256']=source['model_state_sha256']
    result['new_initial_state_sha256']=tensor_state_sha256(model.state_dict())
    result['new_keys']=extras
    result['all_parent_tensors_identical']=True
    result['new_parameter_names_and_optimizer_lr']={n:5e-5 for n,_ in model.named_parameters() if n.startswith(NEW_PREFIX)}
    assert result['new_parameter_names_and_optimizer_lr']
    control.cuda().eval();model.cuda().eval()
    train,_=nominal.raw_datasets(False,1)
    data=NominalStatusDataset(mr.MotionCanvasDataset(train,'native'))
    raw=default_collate([data[0],data[17]])
    batch=to_device(raw,torch.device('cuda:0'));x=inputs(batch)
    result['train_fixture_rows']=raw['row'].tolist()
    loss_fn=wrap_compute_loss(compute_loss,.25)
    result['zero_init_parity']={}
    for precision in ('fp32','bf16'):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=control(**x);b=model(**x)
        difference={k:float((a[k].float()-b[k].float()).abs().max()) for k in KEYS}
        assert max(difference.values())==0,difference
        la,_=loss_fn(a,batch,LossWeights());lb,_=loss_fn(b,batch,LossWeights())
        assert torch.equal(la,lb)
        result['zero_init_parity'][precision]={'outputs':difference,'total_loss_abs_diff':float(abs(la-lb))}
        assert b['motion_pair_features'].shape==(2,4,192,128)
        del a,b,la,lb

    # Real planning backward opens the zero projection. Inner attention and the
    # image path must get finite gradients once that projection is opened.
    set_training_mode(model,'fixed')
    module=model.planner.temporal_read
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**x)
    plan_loss(out,batch).backward()
    first=float(module.attention.out_proj.weight.grad.norm())
    assert first>0 and torch.isfinite(module.attention.out_proj.weight.grad).all()
    assert module.attention.in_proj_weight.grad.norm()==0
    with torch.no_grad():
        module.attention.out_proj.weight.add_(module.attention.out_proj.weight.grad,alpha=-1e-3)
    model.zero_grad(set_to_none=True);del out
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**x)
    plan_loss(out,batch).backward()
    wanted=(NEW_PREFIX+'attention.in_proj_weight',NEW_PREFIX+'query_norm.weight',
            NEW_PREFIX+'memory_norm.weight','motion_encoder.correlation_fuse.0.weight',
            'backbone_fpn.layer1.0.conv1.weight')
    gradients={name:float(p.grad.norm()) if p.grad is not None else None
               for name,p in model.named_parameters() if name in wanted}
    assert len(gradients)==len(wanted) and all(v is not None and v>0 for v in gradients.values()),gradients
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    result['planning_gradients']={'zero_init_output_norm':first,'after_disposable_output_step':gradients}
    model.zero_grad(set_to_none=True);del out

    # Isolate the new reader so gradients through the OLD motion branch cannot
    # masquerade as evidence that the additional memory connection works.
    q=torch.randn(2,6,128,device='cuda',requires_grad=True)
    memory=torch.randn(2,4,192,128,device='cuda',requires_grad=True)
    read=module(q,memory)
    objective=(read*torch.randn_like(read)).sum()
    gq,gm=torch.autograd.grad(objective,(q,memory))
    per_time=gm.square().sum((0,2,3)).sqrt()
    assert torch.isfinite(gm).all() and (per_time>0).all() and gq.norm()>0
    result['isolated_new_read_gradients']={'per_history_norm':per_time.tolist(),'query_norm':float(gq.norm())}
    del q,memory,read,objective,gq,gm

    model.eval()
    # Exercise a NONZERO status query condition; equality cannot pass merely
    # because its zero-initialized condition has not learned yet.
    with torch.no_grad():
        model.shared_status_query_fusion.status_mlp[-1].weight.fill_(.01)
    seen={}
    hooks=[model.scene_encoder.occ_head.register_forward_pre_hook(lambda m,a:seen.update(occ=a[0])),
           model.scene_encoder.lane_head.register_forward_pre_hook(lambda m,a:seen.update(lane=a[0])),
           model.planner.register_forward_pre_hook(lambda m,a:seen.update(planner=a[0]))]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):normal=model(**x)
    assert seen['occ'] is seen['lane']
    assert torch.equal(seen['occ'].flatten(2).transpose(1,2).float(),seen['planner'].float())
    for handle in hooks:handle.remove()
    result['same_shared_scene_consumers']=True
    result['input_isolation']={}
    for name,changed in [('provided_status',dict(x,provided_status5=x['provided_status5']+5)),
                         ('goal',dict(x,goal_xy=x['goal_xy']+10))]:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):other=model(**changed)
        difference={k:float((normal[k]-other[k]).abs().max()) for k in
                    ('motion_pair_features','motion_features','state_hat','history_hat')}
        assert max(difference.values())==0,difference
        result['input_isolation'][name]=dict(difference,scene_max_change=float((normal['scene_features']-other['scene_features']).abs().max()))
        del other
    assert result['input_isolation']['provided_status']['scene_max_change']>0
    del normal,seen
    assert model._provided_status_context is None

    # Unchanged actual multi-task loss uses the WHOLE effective batch counts.
    with torch.no_grad():out=model(**x)
    normalizers=to_device(build_loss_normalizers(raw),torch.device('cuda:0'))
    full,parts=loss_fn(out,batch,LossWeights(),normalizers=normalizers)
    accumulated=sum(loss_fn({k:(v[i:i+1] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==2 else v)
        for k,v in out.items()},slice_batch(batch,i,i+1),LossWeights(),normalizers=normalizers)[0]
        for i in range(2))
    assert torch.allclose(full,accumulated,atol=2e-5,rtol=2e-6)
    result['effective_batch_loss']={'whole':float(full),'sum_micro':float(accumulated),
                                  'absolute_difference':float(abs(full-accumulated))}
    del out,full,accumulated,parts

    # The inputs/flip producer are identical to FRESH. Reuse their implementation.
    from motiondrive_v2_flip_augment import flip_item
    flip=mr.wrap_flip_item(flip_item)
    item=data[0];twice=flip(flip(item,768,384),768,384)
    for k,v in item.items():
        if isinstance(v,torch.Tensor):
            assert torch.allclose(v,twice[k],atol=1e-4,rtol=0) if v.is_floating_point() else torch.equal(v,twice[k]),k
    result['existing_double_flip']=True

    # Strict export/reconstruction after the disposable gradient check.
    clone=TemporalReadModel(copy.deepcopy(config));mr.rebuild_correlation_fuse(clone,4)
    clone.load_state_dict(model.state_dict(),strict=True);clone.cuda().eval()
    one=inputs(to_device(slice_batch(raw,0,1),torch.device('cuda:0')))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model(**one)['plan_abs'];b=clone(**one)['plan_abs']
    assert torch.equal(a,b)
    result['strict_reload_B1_plan_max_diff']=float(abs(a-b).max())
    del clone,a,b,payload,shared
    gc.collect();torch.cuda.empty_cache()

    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.module_tracker as mt
    class NoHandle:
        def remove(self):pass
    old=mt.register_multi_grad_hook
    mt.register_multi_grad_hook=lambda *a,**k:NoHandle()
    result['whole_forward_cost']={}
    try:
        for name,network in [('FRESH',control),('TEMPORAL_READ',model)]:
            with torch.no_grad(),FlopCounterMode(display=False) as count:network(**one)
            flops=int(sum(count.get_flop_counts().get('Global',{}).values()))
            times=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(12):
                    torch.cuda.synchronize();tick=time.monotonic();network(**one);torch.cuda.synchronize()
                    if i>=3:times.append(1000*(time.monotonic()-tick))
            result['whole_forward_cost'][name]={'official_counter_flops':flops,
                'parameters':sum(p.numel() for p in network.parameters()),
                'B200_B1_forward_median_ms':float(np.median(times)), 'RTX4090_ms':None}
            assert flops<7053e9
    finally:
        mt.register_multi_grad_hook=old
    result.update(status='passed',elapsed_seconds=time.monotonic()-started,
        source_sha256=sha(Path(__file__)),model_source_sha256=sha(HERE/'temporal_model.py'),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),new_training_started=False)
    (REPORT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
