"""Fixed terminal inference: supplied command vs all-LANE_KEEP, without fitting."""
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from command_common import *
from command_data import CommandDataset, LABELS, KEY
from command_model import CommandA2Model

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    torch.set_num_threads(4)
    trainer.seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    run = RUNS/f'{ARM}-s1'
    checkpoint = run/'ckpt_step20554.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = CommandA2Model(MotionDriveV2Config(**payload['manifest']['model_config']))
    mr.rebuild_correlation_fuse(model,4)
    model.load_state_dict(payload['model'],strict=True)
    model.cuda().eval()
    _,tune = nominal.raw_datasets(False,1)
    data = CommandDataset(NominalStatusDataset(mr.MotionCanvasDataset(tune,'native')))
    loader = DataLoader(data,batch_size=8,shuffle=False,num_workers=8,pin_memory=True)
    records = json.loads((run/'final_eval.json').read_text())['records']
    pred,clamped,truth,rows = [],[],[],[]
    max_isolation = {'motion_features':0., 'state_hat':0., 'history_hat':0.}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for raw in loader:
            batch = to_device(raw,torch.device('cuda:0'))
            x = inputs(batch)
            out = model(**x)
            keep = torch.zeros_like(x[KEY]);keep[:,0]=1.
            cf = model(**dict(x,provided_command=keep))
            for k in max_isolation:
                max_isolation[k] = max(max_isolation[k],float((out[k].float()-cf[k].float()).abs().max()))
            pred.append(out['plan_abs'].float().cpu().numpy())
            clamped.append(cf['plan_abs'].float().cpu().numpy())
            truth.append(batch['gt_plan'].float().cpu().numpy())
            rows.extend(raw['row'].tolist())
    assert max(max_isolation.values()) == 0.
    p,c,g = (np.concatenate(v).astype(np.float64) for v in (pred,clamped,truth))
    assert rows == [r['row'] for r in records]
    assert np.array_equal(g,np.asarray([r['gt_abs_xy'] for r in records]))
    registered = np.asarray([r['pred_abs_xy'] for r in records])
    replay_delta = float(abs(p-registered).max())
    assert replay_delta < 1e-5, replay_delta
    w = np.array([11,11,5,5,2,2])/36
    supplied = np.linalg.norm(p-g,axis=-1)@w
    allkeep = np.linalg.norm(c-g,axis=-1)@w
    displacement = np.linalg.norm(p-c,axis=-1)@w
    groups = {}
    for i,label in enumerate(LABELS):
        mask = data.command_ids == i
        groups[label] = dict(n=int(mask.sum()), supplied_PREFIX=float(supplied[mask].mean()) if mask.any() else None,
            all_LANE_KEEP_PREFIX=float(allkeep[mask].mean()) if mask.any() else None,
            prediction_change_PREFIX=float(displacement[mask].mean()) if mask.any() else None)
    artifact = run/'command_counterfactual.npz'
    np.savez_compressed(artifact,row=np.asarray(rows),supplied_pred=p,all_keep_pred=c)
    result = dict(checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)), rows=len(rows),
        actual_command_PREFIX=float(supplied.mean()), all_LANE_KEEP_PREFIX=float(allkeep.mean()),
        prediction_change_PREFIX=float(displacement.mean()),semantic_groups=groups,
        motion_state_history_forward_max_delta=max_isolation, replay_max_abs_xy_m=replay_delta,
        command_weight_norm=float(model.shared_command_query.weight.norm()),
        learned_column_norms=model.shared_command_query.weight.norm(dim=0).tolist(),
        artifact=dict(path=str(artifact),bytes=artifact.stat().st_size,sha256=sha(artifact)),
        warning='All-LANE_KEEP is an inference intervention on one trained model, not an independently trained control or organizer approval test.')
    (REPORT/'command_counterfactual.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
