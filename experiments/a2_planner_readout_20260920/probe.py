"""Passive attention summaries for the frozen FRESH-CONT selected checkpoint."""
from pathlib import Path
import sys,json,hashlib,datetime,os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0,str(ROOT/'experiments/a2_motion_fresh_20260919'))
import train_experiment as base
from nominal_data import NominalStatusDataset,sha
from motiondrive_v2_training import model_inputs,to_device,tensor_state_sha256

OUT=ROOT/'reports/a2_planner_readout_20260920'
CHECKPOINT=ROOT/'work_dirs/a2_fresh_continue_20260920/A2-FRESH-CONT-s1/ckpt_step5710.pth'
EXPECTED='073dc47df742b49a93a53ff7cb4bb3f3af7ea7a174a998dc6126597e184696cd'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'probe.json').exists()
    assert sha(CHECKPOINT)==EXPECTED
    torch.set_num_threads(4);base.trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=base.ExperimentModel(base.MotionDriveV2Config(**payload['manifest']['model_config']),arm=base.FRESH)
    base.mr.rebuild_correlation_fuse(model,4);model.load_state_dict(payload['model'],strict=True)
    before=tensor_state_sha256(model.state_dict());del payload
    model.cuda().eval().requires_grad_(False)
    _,raw=base.nominal.raw_datasets(False,1)
    data=NominalStatusDataset(base.mr.MotionCanvasDataset(raw,'native'))
    loader=DataLoader(data,batch_size=8,num_workers=8,pin_memory=True,shuffle=False)
    accum={};temporal=[];handles=[]
    n_scene=int(np.prod(model.config.grid_size));n_motion=int(np.prod(model.config.motion_grid))
    boundaries=[(0,n_scene),(n_scene,n_scene+n_motion),(n_scene+n_motion,n_scene+n_motion+1)]
    def hook_for(name):
        accum[name]={'mass':[],'projected_value_norm':[]}
        def hook(module,args,kwargs):
            q,k,v=args[:3]
            assert kwargs.get('attn_mask') is None and kwargs.get('key_padding_mask') is None
            assert not kwargs.get('is_causal',False) and not module.add_zero_attn
            assert module.bias_k is None and module.bias_v is None
            b,l,c=q.shape;h=module.num_heads;d=c//h
            assert k.shape[1]==n_scene+n_motion+1
            weight=module.in_proj_weight.float();bias=module.in_proj_bias.float()
            qh=F.linear(q.float(),weight[:c],bias[:c]).reshape(b,l,h,d).transpose(1,2)
            kh=F.linear(k.float(),weight[c:2*c],bias[c:2*c]).reshape(b,-1,h,d).transpose(1,2)
            vh=F.linear(v.float(),weight[2*c:],bias[2*c:]).reshape(b,-1,h,d).transpose(1,2)
            alpha=(qh@kh.transpose(-1,-2)/d**.5).softmax(-1)
            mass=torch.stack([alpha[...,a:z].sum(-1) for a,z in boundaries],-1)
            norms=[]
            for a,z in boundaries:
                component=(alpha[...,a:z]@vh[:,:,a:z]).transpose(1,2).reshape(b,l,c)
                projected=F.linear(component,module.out_proj.weight.float(),None)
                norms.append(projected.norm(dim=-1))
            accum[name]['mass'].append(mass.cpu().numpy())
            accum[name]['projected_value_norm'].append(torch.stack(norms,-1).cpu().numpy())
            # No return value: original attention computation and output are untouched.
        return hook
    def time_hook(module,args,out):
        temporal.append(out.float().softmax(1).squeeze(-1).cpu().numpy())
    for i,layer in enumerate(model.planner.decoder.layers):
        handles.append(layer.multihead_attn.register_forward_pre_hook(hook_for(f'layer{i}'),with_kwargs=True))
    handles.append(model.motion_encoder.time_attention.register_forward_hook(time_hook))
    preds=[];gts=[];rows=[]
    try:
        with torch.inference_mode():
            for i,raw in enumerate(loader):
                batch=to_device(raw,torch.device('cuda:0'))
                x=base.mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal');x['provided_status5']=batch['provided_status5']
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**x)
                preds.append(out['plan_abs'].cpu().numpy());gts.append(raw['gt_plan'].numpy());rows.extend(raw['row'].tolist())
                if (i+1)%100==0:print('rows',len(rows),flush=True)
    finally:
        for handle in handles:handle.remove()
    assert tensor_state_sha256(model.state_dict())==before
    source=CHECKPOINT.parent/'predictions_step5710.json';ref=json.loads(source.read_text())['records']
    pred=np.concatenate(preds).astype(np.float64);gt=np.concatenate(gts).astype(np.float64)
    assert rows==[x['row'] for x in ref] and np.array_equal(gt,np.array([x['gt_abs_xy'] for x in ref]))
    delta=float(abs(pred-np.array([x['pred_abs_xy'] for x in ref])).max());assert delta<5e-4
    result={}
    for name,values in accum.items():
        mass=np.concatenate(values['mass']);norms=np.concatenate(values['projected_value_norm'])
        assert np.allclose(mass.sum(-1),1.,atol=2e-6)
        result[name]=dict(mean_mass_scene_motion_state=mass.mean((0,1,2)).tolist(),
            mass_by_head_and_waypoint=mass.mean(0).tolist(),
            mean_projected_value_L2_scene_motion_state=norms.mean((0,1)).tolist())
    temporal=np.concatenate(temporal)
    value=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=str(CHECKPOINT),checkpoint_sha256=EXPECTED,source_sha256=sha(Path(__file__)),
        source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        rows=len(rows),row_sha256=hashlib.sha256(np.array(rows,dtype='<i8').tobytes()).hexdigest(),
        model_parameters_buffers_unchanged=True,new_training=False,provided_inputs_unchanged=True,
        replay_max_abs_m=delta,PREFIX=float((np.linalg.norm(pred-gt,axis=-1)@np.array([11,11,5,5,2,2])/36).mean()),
        token_counts=dict(scene=n_scene,motion=n_motion,state=1,prepool_motion=4*n_motion),
        planner=result,temporal_seconds=[.1,.2,.5,1.],temporal_weight_mean=temporal.mean((0,2)).tolist(),
        temporal_weight_argmax_fraction=[float((temporal.argmax(1)==i).mean()) for i in range(4)],
        limitations=['Attention weights and component norms are descriptive, not causal feature importance.',
            'Component norms do not sum to the norm of their sum; output bias and residual paths are not attributed.',
            'Time pooling can encode temporal statistics; compression alone does not prove loss of acceleration information.',
            'Selected checkpoint and reused DEV; no new architecture performance claim.'])
    (OUT/'probe.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='planner'}));print({k:v['mean_mass_scene_motion_state'] for k,v in result.items()})


if __name__=='__main__':main()
