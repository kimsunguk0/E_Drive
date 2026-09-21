"""Frozen FULL diagnostic on all 375 provided U_TURN rows of DEV train310.

These are in-fit training rows, not held-out or official-test accuracy.
No optimizer, future/test label reconstruction, or submission generation.
"""
from pathlib import Path
import os,sys,json,hashlib,time,datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0,str(ROOT/'experiments/a2_progress_full_20260921'))
from infer_full import load_model,configure,sha,mr
from full_data import FullH4StatusDataset
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import model_inputs,to_device,tensor_state_sha256
OUT=ROOT/'reports/a2_full_error_audit_20260921'
CKPT=ROOT/'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth'
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'UTURN_train_probe.json';assert not target.exists()
    configure();model,payload=load_model(CKPT,True);model.requires_grad_(False)
    before=tensor_state_sha256(model.state_dict())
    command=ROOT/'data/etri/motiondrive_v2/a2_command_20260919/provided_commands.npz'
    with np.load(command,allow_pickle=False) as z:wanted=z['row'][z['command']==5].copy()
    base=MotionDriveDataset(data_root='/tmp/pm97',split_manifest=str(ROOT/'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json'),
        split='train',supervision_root=str(ROOT/'data/etri/motiondrive_v2/r0reset_tplus_geometry_v2'),
        min_frame=30,frame_stride=1,augment=False,seed=1,history_contract='control')
    base.rows=base.rows[np.isin(base.rows,wanted)]
    assert len(base.rows)==375 and np.array_equal(np.sort(base.rows),np.sort(wanted))
    data=FullH4StatusDataset(mr.MotionCanvasDataset(base,'native'))
    records=[];start=time.monotonic()
    with torch.inference_mode():
        for i,raw in enumerate(DataLoader(data,batch_size=8,shuffle=False,num_workers=4,pin_memory=True)):
            batch=to_device(raw,torch.device('cuda:0'))
            x=mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal');x['provided_status5']=batch['provided_status5']
            with torch.autocast('cuda',dtype=torch.bfloat16):y=model(**x)
            pred=y['plan_abs'].float().cpu().numpy();gt=raw['gt_plan'].numpy();gs=raw['state_target'].numpy();vs=raw['state_valid'].numpy()
            assert np.isfinite(pred).all() and raw['plan_valid'].all()
            for j,row in enumerate(raw['row']):
                bucket='nonstop' if gs[j,5]<.5 else 'steady' if np.linalg.norm(gt[j],axis=-1).max()<=.2 else 'depart'
                records.append({'row':int(row),'session':raw['session_id'][j],'scenario':raw['scenario'][j],'frame':int(raw['frame'][j]),
                    'pred_abs_xy':pred[j].tolist(),'gt_abs_xy':gt[j].tolist(),'gt_state':gs[j].tolist(),'gt_state_valid':vs[j].tolist(),
                    'bucket':bucket,'provided_command':'U_TURN'})
            if (i+1)%15==0:print(json.dumps({'rows':len(records),'seconds':time.monotonic()-start}),flush=True)
    after=tensor_state_sha256(model.state_dict());assert before==after
    p=np.array([r['pred_abs_xy'] for r in records]);g=np.array([r['gt_abs_xy'] for r in records]);d=np.linalg.norm(p-g,axis=-1)
    result={'report':{'n':375,'official_d3':float((d@np.array([11,11,5,5,2,2])/36).mean()),'step':24931},
        'scope':'FULL in-fit U-turn diagnostic; all 375 command U_TURN rows in train310, no error-based selection',
        'checkpoint_sha256':sha(CKPT),'command_cache_sha256':sha(command),'model_parameter_buffer_hash_unchanged':True,
        'row_sha256':hashlib.sha256(np.array([r['row'] for r in records],dtype='<i8').tobytes()).hexdigest(),
        'script_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'batch':8,'precision':'BF16 with FP32 planner','seconds':time.monotonic()-start,'optimizer_updates':0,
        'source_command_is_current_provided_label_not_derived_from_GT':True,'records':records}
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
