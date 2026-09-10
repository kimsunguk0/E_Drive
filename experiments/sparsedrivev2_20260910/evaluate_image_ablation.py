"""Frozen-checkpoint image ablation, complete tune population, no training.

Original / zeros after normalization / deterministic cross-scene within-batch
image shuffle. Calibration and causal status stay attached to recipient rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .public_model import PublicSparseDriveV2
except ImportError:
    from public_model import PublicSparseDriveV2


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def snapshot_data(run):
    path=run/'source/experiments/sparsedrivev2_20260910/data.py'
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('ablation_frozen_data',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module,path


def mixed_scene_batches(dataset,batch_size):
    scene_names=dataset.scenarios[dataset.scen_idx[dataset.rows]]
    groups={name:list(np.flatnonzero(scene_names==name)) for name in sorted(set(scene_names.tolist()))}
    order=[]
    for pos in range(max(map(len,groups.values()))):
        order.extend(indices[pos] for indices in groups.values() if pos<len(indices))
    assert sorted(order)==list(range(len(dataset)))
    batches=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
    if len(batches[-1])==1:
        batches[-2].extend(batches.pop())
    for batch in batches:
        names=scene_names[batch]
        if len(set(names.tolist()))!=len(batch):
            raise ValueError('Cannot construct strictly different-scene within-batch shuffle')
    return batches


def inputs(batch):
    return {k:batch[k] for k in ('images','lidar2img','image_hw','status')}


def transfer(batch):
    return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def d3(candidate,gt):
    return (torch.linalg.vector_norm(candidate.float()-gt.float(),dim=-1)*
            candidate.new_tensor([11,11,5,5,2,2]).float()/36).sum(-1)


def verify_bank(model,out):
    bank=model._trajectory_head.traj_vocab.flatten(0,1)
    assert torch.equal(bank[out['candidate_ids'],:6,:2],out['candidate_xy'])
    assert torch.equal(bank[out['selected_candidate_id'],:6,:2],out['trajectory'])
    assert out['candidate_valid'].any(-1).all()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--checkpoint',default='last.pth')
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--latency-samples',type=int,default=50)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='4':
        raise RuntimeError('This bounded evaluation is allocated physical GPU 4 only')
    torch.set_num_threads(4)
    run=Path(args.run).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text())
    settings=manifest['arguments']
    frozen_data,data_path=snapshot_data(run)
    source=Path(__file__).with_name('public_model.py')
    assert sha(source)==manifest['source_sha256']['experiments/sparsedrivev2_20260910/public_model.py']
    assert sha(data_path)==manifest['source_sha256']['experiments/sparsedrivev2_20260910/data.py']
    assert sha(settings['bank'])==manifest['bank_sha256']
    dataset=frozen_data.PlanDataset(base=settings['base'],split_manifest=settings['split_manifest'],
        split=settings['eval_split'],status_mode=settings['status_mode'],rows_file=settings.get('eval_rows'),
        stride=5,augment=False,limit=settings.get('eval_limit',0))
    provenance=dataset.provenance()
    assert provenance['rows_sha256']==manifest['validation']['rows_sha256']
    assert len(dataset)==1998 and settings['eval_split']=='tune'
    checkpoint_path=Path(args.checkpoint)
    if not checkpoint_path.is_absolute():checkpoint_path=run/checkpoint_path
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model,coverage=PublicSparseDriveV2.from_public_checkpoint(settings['checkpoint'],bank_path=settings['bank'])
    for name in ('path_vocab','vel_vocab','traj_vocab','traj_mask'):
        assert torch.equal(getattr(model._trajectory_head,name),saved['model']['_trajectory_head.'+name])
    model.load_state_dict(saved['model'],strict=True)
    model.cuda().eval()
    precision=settings['precision']
    batches=mixed_scene_batches(dataset,args.batch)
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=args.workers,pin_memory=True)
    records={name:[] for name in ('original','zero_normalized','shuffle_different_scene')}
    mappings=[]
    started=time.perf_counter()
    for batch_index,batch in enumerate(loader):
        x=transfer(batch)
        count=len(batch['row'])
        shift=torch.arange(count,device='cuda').roll(-1)
        sender=np.arange(count).reshape(-1)[np.roll(np.arange(count),-1)]
        for i,j in enumerate(sender):
            assert int(batch['row'][i])!=int(batch['row'][j]) and batch['scenario'][i]!=batch['scenario'][j]
            mappings.append({'recipient_row':int(batch['row'][i]),'image_row':int(batch['row'][j]),
                'recipient_scene':batch['scenario'][i],'image_scene':batch['scenario'][j],'batch':batch_index})
        for name in records:
            model_x=inputs(x)
            if name=='zero_normalized':model_x['images']=torch.zeros_like(x['images'])
            elif name=='shuffle_different_scene':model_x['images']=x['images'][shift]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                out=model(**model_x)
            verify_bank(model,out)
            error=d3(out['trajectory'],x['gt_plan'])
            costs=d3(out['candidate_xy'],x['gt_plan'][:,None])
            oracle=costs.masked_fill(~out['candidate_valid'],torch.inf).amin(-1)
            pred=out['trajectory'].cpu().numpy();ids=out['selected_candidate_id'].cpu().numpy()
            for i in range(count):
                records[name].append({'row':int(batch['row'][i]),'session':batch['session'][i],
                    'd3':float(error[i]),'oracle':float(oracle[i]),'pred':pred[i], 'candidate_id':int(ids[i])})
        if batch_index%25==0:print(json.dumps({'batches':batch_index+1,'total_batches':len(batches),'seconds':time.perf_counter()-started}),flush=True)
    summary={}
    for name,record in records.items():
        record.sort(key=lambda r:r['row'])
        arrays={key:np.asarray([r[key] for r in record]) for key in ('row','session','d3','oracle','pred','candidate_id')}
        np.savez_compressed(output/(name+'.npz'),**arrays)
        summary[name]={'n':len(record),'d3':float(arrays['d3'].mean()),'shortlist_oracle':float(arrays['oracle'].mean()),
            'selection_regret':float((arrays['d3']-arrays['oracle']).mean()),
            'session_d3':{s:float(arrays['d3'][arrays['session']==s].mean()) for s in sorted(set(arrays['session'].tolist()))}}
        if name!='original':
            original=records['original']
            summary[name]['changed_selected_row_fraction']=float(np.mean([r['candidate_id']!=o['candidate_id'] for r,o in zip(record,original)]))
            summary[name]['d3_minus_original']=summary[name]['d3']-summary['original']['d3']
    json_write(output/'replacement_mapping.json',mappings)
    # Exact source-run reproduction comparison keyed by immutable cache row.
    prior_path=run/f"eval_{saved['step']:06d}.npz"
    with np.load(prior_path,allow_pickle=False) as z:
        lookup={int(r):i for i,r in enumerate(z['rows'])}
        prior_indices=[lookup[r['row']] for r in records['original']]
        prior_pred=z['pred'][prior_indices]
        reproduced=np.stack([r['pred'] for r in records['original']])
        reproduction={'prior_eval':str(prior_path),'prior_d3':float(z['d3'].mean()),
            'reproduced_minus_prior_d3':summary['original']['d3']-float(z['d3'].mean()),
            'prediction_max_abs_difference':float(np.max(np.abs(prior_pred-reproduced))),
            'prediction_rows_bitwise_equal_fraction':float(np.all(prior_pred==reproduced,axis=(1,2)).mean()),
            'note':'Mixed-scene batches change batching; a BF16 rank tie can change rows. Report actual differences.'}
    # End-to-end current-cache pipeline, batch one, no worker overlap.
    chosen=np.linspace(0,len(dataset)-1,args.latency_samples,dtype=int)
    milliseconds=[]
    for i in [-1]*5+chosen.tolist():
        torch.cuda.synchronize()
        start=time.perf_counter()
        sample=dataset[0 if i<0 else i]
        model_x={k:sample[k].unsqueeze(0).cuda() for k in ('images','lidar2img','image_hw','status')}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            model(**model_x)
        torch.cuda.synchronize()
        if i>=0:milliseconds.append((time.perf_counter()-start)*1000)
    latency={'samples':len(milliseconds),'batch':1,'mean_ms':float(np.mean(milliseconds)),
        'p50_ms':float(np.quantile(milliseconds,.5)),'p90_ms':float(np.quantile(milliseconds,.9)),
        'p95_ms':float(np.quantile(milliseconds,.95)),'max_ms':float(np.max(milliseconds)),
        'included':'cached 3-camera JPEG read/decode, PIL resize, RGB normalization, tensors, H2D, model, CUDA sync',
        'excluded':'raw-image undistortion/crop/cache creation, checkpoint loading, challenge deployment overhead',
        'hardware':'B200 physical GPU4; not official 4090 timing','sample_rows':dataset.rows[chosen].tolist()}
    report={'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),'checkpoint_step':saved['step'],
        'checkpoint_load_weights_only':True,'public_model_sha256':sha(source),'frozen_data_sha256':sha(data_path),
        'script_sha256':sha(__file__),'bank_sha256':manifest['bank_sha256'],'physical_gpu':4,'precision':precision,
        'validation':provenance,'n':len(dataset),'batch_size':args.batch,
        'shuffle':'Scene round-robin batch order; within each batch rotate sender index by one; all donor scenes differ.',
        'status_and_calibration':'Recipient unchanged for every arm; no goal input, no retraining.',
        'exact_fixed_bank_row_identity_all_candidates_and_selected':True,
        'metrics':summary,'reproduction':reproduction,'current_cache_pipeline_latency':latency,
        'elapsed_seconds':time.perf_counter()-started}
    json_write(output/'result.json',report)
    print(json.dumps({'metrics':summary,'reproduction':reproduction,'latency':latency},indent=2),flush=True)


if __name__=='__main__':main()
