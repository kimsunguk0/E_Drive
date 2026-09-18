"""Fixed train probe and sampling statistics, after a candidate is complete."""
import argparse
import json
import os
import numpy as np
from torch.utils.data import DataLoader,Subset
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('PARENT','BASE',*G_ARMS,S_ARM),required=True)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    torch.set_num_threads(4);trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    if a.arm=='PARENT':path=PARENT
    elif a.arm=='BASE':path=BASE_CONTROL/'ckpt_step20554.pth'
    else:
        run=RUNS/f'{a.arm}-s1';m=json.loads((run/'manifest.json').read_text())
        assert m['status']=='completed' and m['step']==(20554 if a.arm==S_ARM else 3426)
        path=run/f"ckpt_step{m['step']}.pth"
    common=torch.load(path,map_location='cpu',weights_only=False)
    cfg=MotionDriveV2Config(**common['manifest']['model_config'])
    if a.arm==S_ARM:
        from learned_sample import LearnedSampleModel
        model=LearnedSampleModel(cfg)
    elif a.arm=='BASE':model=A2NominalModel(cfg,arm='A2-BASE-NOM')
    else:model=SceneExtensionModel(cfg,arm=QREFINE)
    mr.rebuild_correlation_fuse(model,4);model.load_state_dict(common['model'],strict=True);del common
    model.eval().cuda()
    train,tune=nominal.raw_datasets(False,1)
    chosen=[row for b in json.loads((REPORT/'gradient_probe.json').read_text())['batches'] for row in b['rows']]
    lookup={int(row):i for i,row in enumerate(train.rows)}
    indices=[lookup[row] for row in chosen];assert len(indices)==512
    datasets={'train':Subset(NominalStatusDataset(mr.MotionCanvasDataset(train,'native')),indices)}
    if a.arm==S_ARM:datasets['V0']=NominalStatusDataset(mr.MotionCanvasDataset(tune,'native'))
    result={'arm':a.arm,'checkpoint':str(path),'checkpoint_sha256':sha(path),
      'train_probe_definition':'same 512 rows as gradient probe, no augmentation, no fitting; selected by train sampler before results',
      'inference_only':True,'datasets':{}}
    w=np.array([11,11,5,5,2,2],np.float64)/36
    for label,data in datasets.items():
        loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=8,pin_memory=True)
        pred=[];gt=[];row=[]
        if a.arm==S_ARM:model.scene_encoder.collect_sampling_stats=True;model.scene_encoder.sampling_stats=[]
        with torch.inference_mode():
            for raw in loader:
                batch=to_device(raw,torch.device('cuda:0'))
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs(batch))
                pred.append(out['plan_abs'].cpu().numpy());gt.append(raw['gt_plan'].numpy());row.extend(raw['row'].tolist())
        score=np.linalg.norm(np.concatenate(pred).astype(np.float64)-np.concatenate(gt),axis=-1)@w
        values={'n':len(score),'PREFIX':float(score.mean()),'rows_sha256':legacy.rows_sha(row)}
        if a.arm==S_ARM:
            traces=model.scene_encoder.sampling_stats;groups={}
            for t in traces:
                key=f"level{t['level']}_{'current' if t['views']==6 else 'history'}"
                g=groups.setdefault(key,dict(initial_valid=0,lost_valid=0,components=0,sum_abs_cells=0.,max_abs_cells=0.,saturated_components=0,feature_hw=t['feature_hw']))
                for k in ['initial_valid','lost_valid','components','saturated_components']:g[k]+=t[k]
                g['sum_abs_cells']+=t['mean_abs_cells']*t['components'];g['max_abs_cells']=max(g['max_abs_cells'],t['max_abs_cells'])
            for g in groups.values():
                g['mean_abs_cells']=g.pop('sum_abs_cells')/max(1,g['components'])
                g['outbound_fraction']=g['lost_valid']/max(1,g['initial_valid'])
                g['saturation_fraction']=g['saturated_components']/max(1,g['components'])
            values['sampling']=groups
        result['datasets'][label]=values
    (REPORT/f'posttrain_probe_{a.arm}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
