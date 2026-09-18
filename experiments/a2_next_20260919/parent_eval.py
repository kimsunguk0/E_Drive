"""One shared step-zero replay for both G arms, without fitting."""
import json
import os
import time
from torch.utils.data import DataLoader
from common import *

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    torch.set_num_threads(4);trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    model,checkpoint=load_parent();del checkpoint
    model.cuda().eval()
    _,tune=nominal.raw_datasets(False,1)
    data=NominalStatusDataset(mr.MotionCanvasDataset(tune,'native'))
    loader=DataLoader(data,batch_size=8,num_workers=8,pin_memory=True,shuffle=False)
    records=[];start=time.monotonic()
    with torch.inference_mode():
        for raw in loader:
            x=inputs(to_device(raw,torch.device('cuda:0')))
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(**x)['plan_abs'].cpu().float()
            for i in range(len(p)):
                records.append({'row':int(raw['row'][i]),'pred_abs_xy':p[i].tolist(),'gt_abs_xy':raw['gt_plan'][i].tolist()})
    old=json.loads((PARENT.parent/'final_eval.json').read_text())
    assert [r['row'] for r in records]==[r['row'] for r in old['records']]
    import numpy as np
    pred=np.asarray([r['pred_abs_xy'] for r in records]);ref=np.asarray([r['pred_abs_xy'] for r in old['records']])
    gt=np.asarray([r['gt_abs_xy'] for r in records]);w=np.array([11,11,5,5,2,2])/36
    score=float((np.linalg.norm(pred-gt,axis=-1)@w).mean());parity=float(abs(pred-ref).max())
    assert parity<5e-4 and abs(score-old['report']['official_d3'])<1e-6
    value=dict(step=0,official_d3=score,checkpoint=str(PARENT),checkpoint_sha256=sha(PARENT),
        row_count=len(records),replay_max_abs=parity,elapsed_seconds=time.monotonic()-start,
        role='same weights/inputs for G0 and G1; no optimizer updates')
    (REPORT/'parent_initial_eval.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
