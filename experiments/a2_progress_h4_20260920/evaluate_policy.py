"""Frozen old candidates, only the provided scene-query status producer changes."""
from pathlib import Path
import argparse,json,os,sys,hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path('/NHNHOME/data/sukim/adcl');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/a2_motion_fresh_20260919'))
import train_experiment as base
from motiondrive_v2_training import model_inputs,to_device,tensor_state_sha256
sys.path.insert(0,str(ROOT/'experiments/a2_temporal_read_20260920'))
from temporal_model import TemporalReadModel
sys.path.insert(0,str(HERE))
from h4_data import H4StatusDataset,sha
from terminal_review import metrics
REPORT=ROOT/'reports/a2_progress_h4_20260920'
RUNS={'CONT':('a2_fresh_continue_20260920/A2-FRESH-CONT-s1',5710),'TEMPORAL':('a2_temporal_read_20260920/A2-TEMPORAL-READ-s1',20554),'QREFINE':('md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1',20554)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='+',choices=RUNS,required=True);ap.add_argument('--gpu',required=True);args=ap.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu
    torch.set_num_threads(4);base.trainer.seed_all(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    _,tune=base.nominal.raw_datasets(False,1);data=H4StatusDataset(base.mr.MotionCanvasDataset(tune,'native'));w=np.array([11,11,5,5,2,2])/36
    for arm in args.arms:
        suffix,step=RUNS[arm];run=ROOT/'work_dirs'/suffix;ckpt=run/f'ckpt_step{step}.pth';dest=REPORT/f'policy_{arm}.json';assert not dest.exists()
        payload=torch.load(ckpt,map_location='cpu',weights_only=False);config=base.MotionDriveV2Config(**payload['manifest']['model_config'])
        model=TemporalReadModel(config) if arm=='TEMPORAL' else (base.SceneExtensionModel(config,arm=base.QREFINE) if arm=='QREFINE' else base.ExperimentModel(config,arm=base.FRESH))
        base.mr.rebuild_correlation_fuse(model,4);model.load_state_dict(payload['model'],strict=True);before=tensor_state_sha256(model.state_dict());del payload;model.cuda().eval().requires_grad_(False)
        oldpath=run/f'predictions_step{step}.json'
        if not oldpath.exists():oldpath=run/'final_eval.json'
        old=json.loads(oldpath.read_text());values=[];rows=[];status=[]
        with torch.inference_mode():
            for raw in DataLoader(data,batch_size=8,num_workers=8,pin_memory=True,shuffle=False):
                batch=to_device(raw,torch.device('cuda:0'));x=base.mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal');x['provided_status5']=batch['provided_status5']
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**x)
                values.append(out['plan_abs'].float().cpu().numpy());rows.extend(int(v) for v in raw['row']);status.append(raw['provided_status5'].numpy())
        pred=np.concatenate(values).astype(np.float64);gt=np.array([r['gt_abs_xy'] for r in old['records']]);prior=np.array([r['pred_abs_xy'] for r in old['records']]);assert rows==[r['row'] for r in old['records']]
        bucket=np.array([r['bucket'] for r in old['records']]);dg=np.diff(np.concatenate([np.zeros((len(gt),1,2)),gt],1),axis=1);mask=np.linalg.norm(dg,axis=-1)>.05
        stat,_=metrics(pred,gt,bucket,mask);assert before==tensor_state_sha256(model.state_dict()) and np.isfinite(pred).all()
        arraypath=REPORT/f'policy_{arm}.npz';np.savez_compressed(arraypath,row=np.array(rows),gt=gt,pred=pred,prior=prior,bucket=bucket,session=np.array([r['session'] for r in old['records']]),provided_status5=np.concatenate(status))
        result={'arm':arm,'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'old_PREFIX':float((np.linalg.norm(prior-gt,axis=-1)@w).mean()),'new':stat,'plan_PREFIX_change':float((np.linalg.norm(pred-prior,axis=-1)@w).mean()),'state_hash_unchanged':True,'model_state_sha256':before,'arrays':str(arraypath),'arrays_sha256':sha(arraypath),'optimizer_updates':0,'interpretation':'Frozen input policy change; not new training or official submission.'}
        dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'arm':arm,'old':result['old_PREFIX'],'new':stat['PREFIX']}),flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
