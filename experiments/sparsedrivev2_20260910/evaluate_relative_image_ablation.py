"""Original vs fixed different-scene image mapping, trained live CE head unchanged."""
from __future__ import annotations
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from .relative_artifact import load_relative_artifact,file_sha,require
    from .relative_selector import CandidateRelativeSelector
    from .evaluate_checkpoint import inspect_checkpoint,recorded_runtime,load_verified_model,build_dataset,verify_bank_output,TUNE_SHA,rows_sha,WEIGHTS
except ImportError:
    from relative_artifact import load_relative_artifact,file_sha,require
    from relative_selector import CandidateRelativeSelector
    from evaluate_checkpoint import inspect_checkpoint,recorded_runtime,load_verified_model,build_dataset,verify_bank_output,TUNE_SHA,rows_sha,WEIGHTS


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--head',required=True);p.add_argument('--head-sha256',required=True)
    p.add_argument('--mapping',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Allocated physical GPU0 only')
    torch.set_num_threads(4);directory=Path(a.output);directory.mkdir(parents=True,exist_ok=False)
    started=time.time()
    head,cache,receipt=load_relative_artifact(a.head,expected_sha256=a.head_sha256)
    plan=inspect_checkpoint(cache['checkpoint']['path'],bank_path=cache['bank']['path'],population='tune')
    require(plan.receipt['checkpoint_sha256']==receipt['base_checkpoint_sha256'],'Wrong frozen image base')
    mapping=json.loads(Path(a.mapping).read_text())
    require(len(mapping)==1998,'Expected original full mapping')
    with recorded_runtime(plan) as runtime:
        base,_=load_verified_model(plan,runtime)
        model=CandidateRelativeSelector(base,base_goal_mode=plan.goal_mode,freeze_base=True)
        model.relative_head.load_state_dict(head.state_dict(),strict=True)
        model.cuda().eval()
        dataset_plan=copy.copy(plan);dataset_plan.goal_mode='selection'
        data=build_dataset(dataset_plan,runtime)
        row_to_index={int(row):i for i,row in enumerate(data.rows)}
        by_batch={}
        for record in mapping:
            require(record['recipient_row'] in row_to_index and record['image_row'] in row_to_index,'Mapping left tune population')
            require(record['recipient_row']!=record['image_row'] and record['recipient_scene']!=record['image_scene'],'Mapping must replace with another scene')
            by_batch.setdefault(record['batch'],[]).append(record)
        require(set(r['recipient_row'] for r in mapping)==set(row_to_index),'Mapping recipient rows incomplete or duplicated')
        batches=[];donor_positions=[]
        for batch_id,records in sorted(by_batch.items()):
            rows=[record['recipient_row'] for record in records]
            require(len(set(rows))==len(rows),'Duplicate recipient inside batch')
            positions={row:i for i,row in enumerate(rows)}
            require(all(r['image_row'] in positions for r in records),'Donor not in original batch')
            batches.append([row_to_index[row] for row in rows])
            donor_positions.append([positions[r['image_row']] for r in records])
        loader=DataLoader(data,batch_sampler=batches,num_workers=4,pin_memory=True)
        record_lists={name:[] for name in ('original','shuffle_different_scene')}
        for number,batch in enumerate(loader):
            x={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            donor=torch.tensor(donor_positions[number],device='cuda')
            for i,j in enumerate(donor_positions[number]):
                rec=by_batch[sorted(by_batch)[number]][i]
                require(batch['scenario'][i]==rec['recipient_scene'] and batch['scenario'][j]==rec['image_scene'],'Mapping scene identity changed')
            for name in record_lists:
                inputs=runtime.data.model_inputs(x,goal_selection=True)
                if name=='shuffle_different_scene':inputs['images']=x['images'][donor]
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs)
                verify_bank_output(model,out,runtime)
                pred=out['trajectory'].float().cpu().numpy()
                gt=x['gt_plan'].float().cpu().numpy()
                delta=pred.astype(np.float64)-gt.astype(np.float64)
                d3=(np.linalg.norm(delta,axis=-1)*WEIGHTS).sum(-1)
                cost=runtime.data.d3(out['candidate_xy'],x['gt_plan'][:,None].expand_as(out['candidate_xy']))
                oracle=cost.masked_fill(~out['candidate_valid'],torch.inf).amin(-1).cpu().numpy()
                ids=out['selected_candidate_id'].cpu().numpy()
                for i in range(len(pred)):
                    record_lists[name].append({'row':int(batch['row'][i]),'pred':pred[i],
                        'candidate_id':int(ids[i]),'d3':float(d3[i]),'oracle':float(oracle[i]),'session':batch['session'][i]})
            if number%50==0:print(json.dumps({'batch':number,'seconds':time.time()-started}),flush=True)
        summary={};arrays={}
        for name,records in record_lists.items():
            records.sort(key=lambda r:r['row'])
            arrays[name]={key:np.asarray([r[key] for r in records]) for key in ('row','pred','candidate_id','d3','oracle','session')}
            values=arrays[name];require(rows_sha(values['row'])==TUNE_SHA,'Ablation row population changed')
            np.savez_compressed(directory/(name+'.npz'),**values)
            summary[name]={'n':len(records),'d3':float(values['d3'].mean()),'oracle':float(values['oracle'].mean()),
                'selection_regret':float((values['d3']-values['oracle']).mean()),
                'session_d3':{s:float(values['d3'][values['session']==s].mean()) for s in set(values['session'])}}
        reference=Path(a.head).parent/f"eval_{receipt['relative_step']:06d}.npz"
        with np.load(reference,allow_pickle=False) as z:
            require(np.array_equal(arrays['original']['row'],z['rows']),'Reference row mismatch')
            require(np.array_equal(arrays['original']['pred'],z['pred']),'Original mixed-batch live predictions changed')
            require(np.array_equal(arrays['original']['candidate_id'],z['candidate_id']),'Original mixed-batch IDs changed')
            require(np.array_equal(arrays['original']['d3'],z['d3']),'Original D3 changed')
        summary['shuffle_different_scene']['d3_minus_original']=summary['shuffle_different_scene']['d3']-summary['original']['d3']
        summary['shuffle_different_scene']['changed_selected_row_fraction']=float(np.mean(arrays['original']['candidate_id']!=arrays['shuffle_different_scene']['candidate_id']))
        result={'status':'completed','relative':receipt,'base':plan.receipt,'physical_gpu':0,
            'mapping_sha256':file_sha(a.mapping),'mapping_path':str(Path(a.mapping).resolve()),
            'same_mapping_as_pilot500':True,'recipient_calibration_status_goal_and_gt_unchanged':True,
            'image_transform':'Only substitute three normalized current images from fixed different-scene donor',
            'original_reference_xy_ids_d3_bitwise_exact':True,'all_bank_rows_exact':True,
            'metrics':summary,'elapsed_seconds':time.time()-started,'script_sha256':file_sha(__file__)}
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
