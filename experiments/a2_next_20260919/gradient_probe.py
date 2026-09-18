"""32 fixed training effective batches, shared-parameter gradients, zero updates."""
import csv
import datetime
import json
import math
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import *

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    torch.set_num_threads(4)
    trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    model, checkpoint = load_parent()
    before=tensor_state_sha256(model.state_dict())
    weights=LossWeights(**checkpoint['manifest']['loss_weights'])
    del checkpoint
    model.cuda()
    trainer.set_training_mode(model,'fixed')
    data=training_data()
    loader=DataLoader(data,batch_size=16,shuffle=True,num_workers=8,pin_memory=True,
        generator=torch.Generator().manual_seed(1),worker_init_fn=trainer.worker_seed)
    groups={
      'backbone_fpn':[(n,p) for n,p in model.named_parameters() if n.startswith('backbone_fpn.')],
      'scene_shared':[(n,p) for n,p in model.named_parameters() if n.startswith('scene_encoder.')
          and not n.startswith(('scene_encoder.occ_head.','scene_encoder.lane_head.'))],
      'status_query':[(n,p) for n,p in model.named_parameters() if n.startswith('shared_status_query_fusion.')]}
    named=[];bounds={}
    for g,items in groups.items():bounds[g]=(len(named),len(named)+len(items));named+=items
    assert len({id(p) for n,p in named})==len(named)
    params=[p for n,p in named]; rows=[];decomposition_max=0.
    for bi,raw in enumerate(loader):
        if bi==32:break
        normalizers=to_device(build_loss_normalizers(raw),torch.device('cuda:0'))
        grads={k:[None]*len(params) for k in ('P','M','A')}
        none={k:[True]*len(params) for k in grads};loss_values={k:0. for k in grads}
        for begin in range(0,16,8):
            batch=to_device(trainer.slice_batch(raw,begin,begin+8),torch.device('cuda:0'))
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs(batch))
            objectives,parts,gap=separated_loss(out,batch,weights,normalizers)
            decomposition_max=max(decomposition_max,float(gap))
            for i,(k,l) in enumerate(objectives.items()):
                values=torch.autograd.grad(l,params,allow_unused=True,retain_graph=i<2)
                loss_values[k]+=float(l.detach())
                for j,v in enumerate(values):
                    if v is not None:
                        none[k][j]=False
                        if grads[k][j] is None:grads[k][j]=v.detach().float()
                        else:grads[k][j].add_(v.detach().float())
            del out,parts,objectives,batch,values,v,l
        entry={'batch':bi,'rows':raw['row'].tolist(),'losses':loss_values,'groups':{}}
        for group,(lo,hi) in bounds.items():
            norms={};zero={};missing={}
            for k in grads:
                present=grads[k][lo:hi]
                sq=sum((g.double().square().sum() for g in present if g is not None),torch.zeros((),device='cuda',dtype=torch.float64))
                norms[k]=math.sqrt(float(sq))
                missing[k]=sum(none[k][lo:hi])
                zero[k]=sum(g is not None and not bool(g.count_nonzero()) for g in present)
            def dot(a,b):
                return float(sum((ga.double().mul(gb.double()).sum() for ga,gb in zip(grads[a][lo:hi],grads[b][lo:hi]) if ga is not None and gb is not None),torch.zeros((),device='cuda',dtype=torch.float64)))
            ma,pa=dot('M','A'),dot('P','A')
            entry['groups'][group]={'norm_prefix':norms['P'],'norm_main':norms['M'],'norm_aux':norms['A'],
              'cos_main_aux':ma/(norms['M']*norms['A']) if norms['M']*norms['A'] else None,
              'cos_prefix_aux':pa/(norms['P']*norms['A']) if norms['P']*norms['A'] else None,
              'aux_main_norm_ratio':norms['A']/norms['M'] if norms['M'] else None,
              'dot_main_total':norms['M']**2+ma,'none_parameter_tensors':missing,'zero_parameter_tensors':zero,
              'parameter_tensors':hi-lo}
        rows.append(entry);print(json.dumps(entry),flush=True)
        del grads
    after=tensor_state_sha256(model.cpu().state_dict());assert before==after
    summary={}
    for g in groups:
        vals=[r['groups'][g] for r in rows]
        ratios=[v['aux_main_norm_ratio'] for v in vals if v['aux_main_norm_ratio'] is not None]
        cos=[v['cos_main_aux'] for v in vals if v['cos_main_aux'] is not None]
        summary[g]={'ratio_median':float(np.median(ratios)),'ratio_p90':float(np.percentile(ratios,90)),
                    'cosine_median':float(np.median(cos)),'negative_cosine_batches':sum(c<0 for c in cos),'defined_batches':len(cos)}
    result={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'checkpoint':str(PARENT),'checkpoint_sha256':sha(PARENT),'unchanged_model_state_sha256':before,
      'batch_count':32,'effective_batch':16,'microbatch':8,'train_rows':512,
      'normalization':'full-effective-batch mask counts, additive microbatch gradients',
      'augmentation':'same train seed1/epoch0 photometric and flip policy','optimizer_updates':0,
      'decomposition_max_abs':decomposition_max,'summary':summary,'batches':rows,
      'caveat':'Raw pre-clip gradients, no AdamW preconditioning; not causal or generalization proof.'}
    REPORT.mkdir(exist_ok=True,parents=True)
    (REPORT/'gradient_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    with (REPORT/'gradient_probe_by_batch.csv').open('w') as f:
        fields=['batch','group','norm_prefix','norm_main','norm_aux','cos_main_aux','cos_prefix_aux','aux_main_norm_ratio','dot_main_total']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:
            for g,v in r['groups'].items():w.writerow(dict(batch=r['batch'],group=g,**{k:v[k] for k in fields[2:]}))
    print('RESULT '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
