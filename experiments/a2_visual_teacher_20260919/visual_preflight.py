"""Train-only coefficient/cost probe and focused distillation contract tests."""
import json, math, time, os, copy, hashlib
import numpy as np
from torch.utils.data import DataLoader
from visual_teacher import *
from common import (trainer, load_parent, training_data, inputs, PARENT,
    LossWeights, build_loss_normalizers, to_device, separated_loss)
from motiondrive_v2_training import set_training_mode


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0','1','2','3')
    torch.set_num_threads(4); trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    parent, checkpoint = load_parent()
    before=tensor_state_sha256(parent.state_dict())
    weights=LossWeights(**checkpoint['manifest']['loss_weights'])
    model=VisualStudent(MotionDriveV2Config(**checkpoint['manifest']['model_config']))
    mr.rebuild_correlation_fuse(model,4)
    load_student_parent(model,checkpoint['model'])
    model.cuda(); parent.cuda().eval(); model.eval(); model.capture_in_eval=True
    teacher=load_teacher(); teacher_sha=tensor_state_sha256(teacher.state_dict())
    data=training_data()
    loader=DataLoader(data,batch_size=16,shuffle=True,num_workers=8,pin_memory=True,
        generator=torch.Generator().manual_seed(1),worker_init_fn=trainer.worker_seed)
    iterator=iter(loader); raw=next(iterator)
    b=to_device(trainer.slice_batch(raw,0,1),torch.device('cuda:0'))
    inp=inputs(b)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        original=parent(**inp); new=model(**inp)
        changed=dict(inp)
        changed['provided_status5']=inp['provided_status5']+torch.tensor([2.,.2,.3,.4,.1],device='cuda')
        for key in changed:
            if 'goal' in key: changed[key]=changed[key]*.5
        intervention=model(**changed)
    common_keys=[k for k in original if torch.is_tensor(original[k])]
    parity={k:float((original[k].float()-new[k].float()).abs().max()) for k in common_keys}
    assert max(parity.values())==0,parity
    invariant_keys=['_visual_feature']+[k for k in new if ('motion' in k or 'state' in k or 'history' in k) and torch.is_tensor(new[k])]
    invariance={k:float((new[k].float()-intervention[k].float()).abs().max()) for k in invariant_keys}
    assert max(invariance.values())==0,invariance
    del parent,original,new,intervention,checkpoint,b,inp
    # Pixel-centre mapping: synthetic ramp must equal the intended feature coordinates.
    grid,mask=spatial_grid('cuda'); y,x=torch.meshgrid(torch.arange(27,device='cuda'),torch.arange(48,device='cuda'),indexing='ij')
    ramp=(x+100*y).float()[None,None]
    sampled=F.grid_sample(ramp,grid[None],align_corners=False)[0,0]
    want=((grid[...,0]+1)*48/2-.5)+100*((grid[...,1]+1)*27/2-.5)
    ramp_error=float((sampled[mask]-want[mask]).abs().max()); assert ramp_error<.001
    # Full/microbatch normalization including partial invalid and NaN targets.
    generator=torch.Generator(device='cuda').manual_seed(17)
    f=torch.randn((12,128,27,48),device='cuda',generator=generator,requires_grad=True)
    target=torch.randn((2,6,768,27,48),device='cuda',generator=generator)
    valid=mask[None,None].expand(2,6,-1,-1).clone();valid[1,2,:5]=False
    target[1,2,:,:5]=float('nan'); count=valid.sum()
    lf=visual_loss(f,model.visual_projector,target,valid,count)
    gf=torch.autograd.grad(lf,(f,model.visual_projector.weight))
    lm=sum(visual_loss(f[i*6:(i+1)*6],model.visual_projector,target[i:i+1],valid[i:i+1],count) for i in range(2))
    gm=torch.autograd.grad(lm,(f,model.visual_projector.weight))
    norm_test=dict(loss_max_diff=float((lf-lm).abs()),gradient_max_diff=max(float((a-b).abs().max()) for a,b in zip(gf,gm)))
    assert norm_test['loss_max_diff']<2e-6 and norm_test['gradient_max_diff']<2e-6
    zero=visual_loss(f,model.visual_projector,target,torch.zeros_like(valid),0)
    assert float(zero)==0 and all(not bool(g.count_nonzero()) for g in torch.autograd.grad(zero,(f,model.visual_projector.weight)))
    del f,target,valid,lf,lm,gf,gm,zero
    set_training_mode(model,'fixed'); model.capture_in_eval=False
    params=[p for n,p in model.named_parameters() if n.startswith('backbone_fpn.')]
    # Rule fixed before any gradient observations or new V0 evaluation.
    rule='lambda = clamp(0.10 / median(unweighted_visual/main_gradient_norm), 0.005, 0.25); 4 fixed train batches'
    rows=[];all_seconds=[]
    for i in range(4):
        if i:raw=next(iterator)
        torch.cuda.synchronize(); started=time.perf_counter()
        targets,valid=teacher_targets(teacher,raw['images'])
        teacher_seconds=time.perf_counter()-started
        count=valid.sum()
        normalizers=to_device(build_loss_normalizers(raw),torch.device('cuda:0'))
        grads={k:[None]*len(params) for k in ('main','visual')};values={'main':0.,'visual':0.}
        projector_norm=0.
        for lo in (0,8):
            batch=to_device(trainer.slice_batch(raw,lo,lo+8),torch.device('cuda:0'))
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs(batch))
            objectives,parts,gap=separated_loss(out,batch,weights,normalizers)
            vis=visual_loss(out['_visual_feature'],model.visual_projector,targets[lo:lo+8],valid[lo:lo+8],count)
            for key,loss in (('main',objectives['M']),('visual',vis)):
                gs=torch.autograd.grad(loss,params+([model.visual_projector.weight] if key=='visual' else []),
                    retain_graph=key=='main',allow_unused=True)
                if key=='visual':projector_norm+=float(gs[-1].float().square().sum())
                for j,g in enumerate(gs[:len(params)]):
                    if g is not None:grads[key][j]=g.detach().float() if grads[key][j] is None else grads[key][j]+g.detach().float()
                values[key]+=float(loss.detach())
            del out,objectives,parts,vis,loss,gs,batch
        norms={k:math.sqrt(sum(float(g.double().square().sum()) for g in gs if g is not None)) for k,gs in grads.items()}
        assert all(math.isfinite(v) and v>0 for v in norms.values()) and projector_norm>0
        torch.cuda.synchronize();seconds=time.perf_counter()-started;all_seconds.append(seconds)
        rows.append(dict(batch=i,rows=raw['row'].tolist(),loss=values,norms=norms,
            visual_main_ratio=norms['visual']/norms['main'],teacher_seconds=teacher_seconds,
            dual_gradient_seconds=seconds,valid_count=int(count),projector_gradient_sq=projector_norm))
        print('GRAD '+json.dumps(rows[-1]),flush=True)
        del targets,valid,grads
    ratio=float(np.median([r['visual_main_ratio'] for r in rows])); coefficient=min(.25,max(.005,.1/ratio))
    assert before==tensor_state_sha256(shared_state(model))
    assert teacher_sha==tensor_state_sha256(teacher.state_dict())
    assert all(p.grad is None for p in teacher.parameters())
    tests=dict(zero_visual_forward_parity=parity,input_boundary=invariance,
        shared_parent_state_unchanged=before,teacher_state_unchanged=teacher_sha,
        ramp_max_abs=ramp_error,full_micro=norm_test,all_invalid_zero=True,
        teacher_requires_grad=False,student_and_projection_nonzero_finite_gradient=True,
        student_export_strict_equivalence='parent strict load of identical shared tensors; repeated after training')
    report=dict(rule=rule,lambda_vis=coefficient,median_ratio=ratio,batches=rows,tests=tests,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),teacher=teacher_manifest(),
        selection_used='4 train effective batches only; no V0 selection',optimizer_updates=0)
    REPORT.mkdir(exist_ok=True,parents=True)
    (REPORT/'visual_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    (REPORT/'teacher_manifest.json').write_text(json.dumps(teacher_manifest(),indent=2)+'\n')
    print('PASS '+json.dumps({'lambda_vis':coefficient,'median_ratio':ratio,'teacher_seconds':np.median([r['teacher_seconds'] for r in rows])}),flush=True)

if __name__=='__main__':main()
