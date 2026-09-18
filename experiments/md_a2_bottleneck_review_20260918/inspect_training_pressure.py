"""Read-only gradient and planner-attention audit of the frozen A2 BASE.

Uses 32 train rows from distinct scenes. No optimizer or validation gradients.
"""
from pathlib import Path
import datetime
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0,str(ROOT/'experiments/md_a2_bottleneck_review_20260918'))
from diagnose_inference import nominal, mr, inputs, to_device, A2NominalModel, NominalStatusDataset, MotionDriveV2Config, REPORT
from motiondrive_v2_training import compute_loss, LossWeights, tensor_state_sha256
sys.path.insert(0,str(ROOT/'experiments/md_r0_reset_20260914'))
from length_auxiliary import length_loss


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    run=ROOT/'work_dirs/md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1'
    manifest=json.loads((run/'manifest.json').read_text())
    model=A2NominalModel(MotionDriveV2Config(**manifest['model_config']),arm='A2-BASE-NOM')
    mr.rebuild_correlation_fuse(model,4)
    checkpoint=torch.load(run/'ckpt_step20554.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    initial=tensor_state_sha256(model.state_dict())
    model.eval().cuda()
    train,_=nominal.raw_datasets(False,1)
    scene=np.asarray(train.scene_names[train.rows])
    rng=np.random.default_rng(20260918)
    chosen=rng.choice(sorted(set(scene.tolist())),32,replace=False)
    indices=[int(rng.choice(np.flatnonzero(scene==s))) for s in chosen]
    data=NominalStatusDataset(mr.MotionCanvasDataset(train,'native'))
    loader=DataLoader(Subset(data,indices),batch_size=4,num_workers=4,shuffle=False)
    params=[p for p in model.backbone_fpn.parameters() if p.requires_grad]
    weights=LossWeights(**manifest['loss_weights'])
    rows=[];attention=[]
    hooks=[]
    for layer_id,layer in enumerate(model.planner.decoder.layers):
        def pre(module,args,kwargs):
            kwargs=dict(kwargs,need_weights=True,average_attn_weights=False)
            return args,kwargs
        def after(module,args,output,layer_id=layer_id):
            weight=output[1].detach().float()
            assert weight.shape[-1]==3072+192+1
            masses=torch.stack([weight[...,:3072].sum(-1),weight[...,3072:3264].sum(-1),weight[...,-1]],-1)
            attention.append({'layer':layer_id,'mass_scene_motion_state':masses.mean((0,1,2)).cpu().tolist()})
        hooks.append(layer.multihead_attn.register_forward_pre_hook(pre,with_kwargs=True))
        hooks.append(layer.multihead_attn.register_forward_hook(after))
    # Attention diagnostics are separate from the gradient audit's ordinary forward.
    first=next(iter(loader));x=inputs(to_device(first,torch.device('cuda:0')))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): inspected=model(**x)['plan_abs']
    for h in hooks:h.remove()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): ordinary=model(**x)['plan_abs']
    attention_parity=float((inspected-ordinary).abs().max())
    assert attention_parity < 1e-3
    del first,x,inspected,ordinary
    for batch_id,raw in enumerate(loader):
        batch=to_device(raw,torch.device('cuda:0'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**inputs(batch))
            _,parts=compute_loss(output,batch,weights)
            losses={'PREFIX':weights.plan*parts['plan_d3'],
                    'LENGTH':.25*length_loss(output,batch),
                    'PERCEPTION':weights.occupancy*parts['occ_bce']+weights.lane*parts['lane_bce'],
                    'MOTION':weights.motion*parts['motion']}
        grad={}
        for i,(key,loss) in enumerate(losses.items()):
            value=torch.autograd.grad(loss,params,retain_graph=i<len(losses)-1,allow_unused=True)
            grad[key]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten().float() for p,g in zip(params,value)])
        gp=grad['PREFIX'];pn=gp.norm()
        r={'batch':batch_id,'rows':raw['row'].tolist(),'weighted_loss':{k:float(v.detach()) for k,v in losses.items()},
           'gradient':{}}
        for name,g in grad.items():
            gn=g.norm();dot=torch.dot(gp,g)
            r['gradient'][name]={'norm':float(gn),'norm_ratio_to_PREFIX':float(gn/pn),
                                'cosine_with_PREFIX':float(dot/(gn*pn).clamp_min(1e-20))}
        aux=grad['LENGTH']+grad['PERCEPTION']+grad['MOTION']
        r['auxiliary_combined']={'norm_ratio_to_PREFIX':float(aux.norm()/pn),
            'cosine_with_PREFIX':float(torch.dot(gp,aux)/(pn*aux.norm()).clamp_min(1e-20))}
        rows.append(r);print(json.dumps(r),flush=True)
        del grad,output,losses,parts,batch,gp,g,aux
    final=tensor_state_sha256(model.cpu().state_dict());assert initial==final
    def summary(name):
        a=[r['gradient'][name] for r in rows]
        return {'median_norm_ratio_to_PREFIX':float(np.median([x['norm_ratio_to_PREFIX'] for x in a])),
                'median_cosine_with_PREFIX':float(np.median([x['cosine_with_PREFIX'] for x in a])),
                'opposing_batches':sum(x['cosine_with_PREFIX']<0 for x in a)}
    result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gpu=3,training_performed=False,optimizer_steps=0,state_sha256_unchanged=initial,
        sample_count=32,scene_count=32,probe='train split only; no flip or photometric augmentation; fixed BN',
        parameter_subset='all shared backbone_fpn trainable parameters',weights=manifest['loss_weights'],length_weight=.25,
        summary={k:summary(k) for k in rows[0]['gradient']},batches=rows,
        first_four_train_rows_attention=attention,attention_forward_parity_max_abs=attention_parity,
        caveats=['Raw gradient cosines do not include Adam preconditioning or prove validation improvement.',
                 'Small scene-stratified train probe, not a population estimate.',
                 'Attention mass alone is not causal contribution; value magnitude and later layers also matter.'])
    REPORT.mkdir(exist_ok=True,parents=True)
    (REPORT/'training_pressure.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps({'summary':result['summary'],'attention':attention,'state_unchanged':True}),flush=True)


if __name__=='__main__':main()
