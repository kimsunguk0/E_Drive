"""Real-input contract, mathematical boundary cases, gradients and full cost."""
import copy,gc,json,math,os,sys,time
import numpy as np
import torch
from torch.utils.data import default_collate
from train_progress import *
from progress_model import compose_progress
from h4_status import FRAME_OFFSETS,TIMES,status_from_five_poses,status_from_clip_records
from motiondrive_v2_training import (model_inputs,to_device,LossWeights,compute_loss,build_loss_normalizers,set_training_mode)
from train_motiondrive_v2 import slice_batch
from length_auxiliary import wrap_compute_loss

KEYS=('scene_features','occ_logits','lane_logits','motion_features','motion_pair_features',
      'state_hat','history_hat','state_logvar','history_logvar')
def inputs(batch):
    value=mr.model_inputs_with_canvas(model_inputs,batch,time_input='nominal')
    value['provided_status5']=batch['provided_status5'];return value
def plan_loss(out,batch):
    w=out['plan_abs'].new_tensor([11,11,5,5,2,2])/36
    return (torch.linalg.vector_norm(out['plan_abs']-batch['gt_plan'],dim=-1)*w).sum(-1).mean()

def producer_checks(data):
    poses=np.broadcast_to(np.eye(4),(5,4,4)).copy()
    poses[:,0,3]=7*TIMES+.5*.4*TIMES**2
    poses[:,1,3]=-.3*TIMES+.5*.2*TIMES**2
    got,_=status_from_five_poses(poses)
    np.testing.assert_allclose(got,[7,-.3,.4,.2,0],atol=1e-6)
    from scipy.spatial.transform import Rotation
    poses[:,:3,:3]=Rotation.from_euler('z',.2*TIMES).as_matrix()
    got,_=status_from_five_poses(poses)
    np.testing.assert_allclose(got,[7,-.3,.4,.2,.2],atol=1e-6)
    reflect=np.diag([1,-1,1,1]);flipped,_=status_from_five_poses(reflect@poses@reflect)
    np.testing.assert_allclose(flipped,got*np.array([1,-1,1,-1,-1]),atol=1e-6)

    import pyarrow.parquet as pq
    for path in (ROOT/'scripts',ROOT/'experiments/md_r0_reset_20260914'):
        sys.path.insert(0,str(path))
    from models import motiondrive_v2_inputs as adapter
    import mr_deploy
    fixtures=json.loads((ROOT/'reports/motiondrive_v2_deploy_fixture_train8_manifest.json').read_text())['clips']
    checks=[]
    for fixture in fixtures:
        clip=ROOT/'data/etri/motiondrive_v2/deploy_fixture_train8'/fixture['clip_id']
        poses=pq.read_table(clip/'ego_pose.parquet',columns=list(adapter.POSE_COLUMNS)).to_pylist()
        status,diag=status_from_clip_records(poses)
        poisoned=copy.deepcopy(poses)
        for row in poisoned:
            if row['frame'] not in FRAME_OFFSETS:
                for key in ('x','y','z','roll','pitch','yaw'):row[key]=float('nan')
        np.testing.assert_array_equal(status,status_from_clip_records(poisoned)[0])
        for change in ('missing','duplicate'):
            bad=([r for r in poses if r['frame']!=-5] if change=='missing' else
                 poses+[next(r for r in poses if r['frame']==-5)])
            try:status_from_clip_records(bad)
            except ValueError:pass
            else:raise AssertionError(change)
        calibration=pq.read_table(clip/'calibration.parquet',columns=list(adapter.CALIBRATION_COLUMNS)).to_pylist()
        calls=[]
        def getter(camera,frame):
            calls.append((camera,int(frame)));return (clip/camera/f'frame_{frame}.jpg').read_bytes()
        prepared=mr_deploy.prepare_mr_clip_from_records(calibration,poses,getter,detail='native')
        image_frames=sorted(set(f for _,f in calls))
        assert set(FRAME_OFFSETS)<=set(image_frames)
        # Fixture manifest maps the raw serialized clip to the canonical row.
        names=data.arr['scenarios'].astype(str)[data.arr['scen_idx']]
        matched=np.flatnonzero((names==fixture['source_scene']) & (data.arr['frame']==fixture['source_frame']))
        assert len(matched)==1
        row=int(matched[0]);idx=int(np.flatnonzero(data.rows==row)[0])
        np.testing.assert_allclose(status,data.values[idx],atol=1e-6,rtol=0)
        checks.append({'clip':fixture['clip_id'],'row':row,'status_max_diff':float(abs(status-data.values[idx]).max()),
            'fit_frames':diag['fit_relative_frames'],'consumed_image_frames':image_frames,
            'image_getter_calls':len(calls),'unused_future_pose_poison_invariant':True})
    return {'synthetic_quadratic_yaw_reflection':True,'raw_fixture_checks':checks,
        'required_missing_duplicate_rejected':True,'supervision_changed':False}

def composition_checks():
    units=torch.tensor([LENGTH_SCALE,1.,LENGTH_BIAS])
    def latent(length,angle):
        return torch.stack([torch.log(torch.expm1(length/LENGTH_SCALE))-LENGTH_BIAS,angle],-1)
    length=torch.full((1,6),2.);heading=torch.zeros_like(length)
    straight=compose_progress(latent(length,heading),units)
    assert torch.allclose(straight[0,:,0],torch.arange(1,7)*2.,atol=2e-6) and straight[...,1].count_nonzero()==0
    angle=torch.tensor([[0,.1,.2,.4,.7,1.]])
    curved=compose_progress(latent(length,angle),units)
    reflected=compose_progress(latent(length,-angle),units)
    assert torch.allclose(reflected,curved*torch.tensor([1,-1]),atol=1e-6)
    reverse=compose_progress(latent(length,heading+math.pi),units)
    assert torch.allclose(reverse,-straight,atol=2e-6)
    near_stop=compose_progress(latent(torch.full_like(length,1e-7),heading),units)
    assert near_stop.abs().max()<1e-6
    start=compose_progress(latent(torch.tensor([[1e-7,1e-7,.1,.4,1.,2.]]),heading),units)
    assert start[0,-1,0]>3
    raw=latent(length,angle).requires_grad_();p=compose_progress(raw,units)
    w=torch.tensor([11,11,5,5,2,2])/36
    target=p.detach()+torch.tensor([.3,.2]);loss=(torch.linalg.vector_norm(p-target,dim=-1)*w).sum()
    loss.backward();norms=raw.grad.square().sum((0,1)).sqrt()
    assert torch.isfinite(raw.grad).all() and (norms>0).all()
    return {'straight_curve_reflection_reverse_nearstop_departure':True,'raw_channel_gradient_norms':norms.tolist(),
        'zero_length_is_limit_not_exact':True,'no_heading_clamp':True,'output_already_absolute':True}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert not (REPORT/'preflight.json').exists()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
    result={'status':'running','physical_gpu':0,'checked_weights_used_for_training':False}
    payload=torch.load(base.FRESH_INIT,map_location='cpu',weights_only=False)
    config=base.MotionDriveV2Config(**payload['manifest']['model_config'])
    models={};states={};rng=[]
    for arm in ARMS:
        trainer.seed_all(1);m=H4ProgressModel(copy.deepcopy(config),arm=arm);mr.rebuild_correlation_fuse(m,4)
        rng.append(torch.get_rng_state().clone());load_initializer(m,payload['model'])
        states[arm]=tensor_state_sha256(m.state_dict());models[arm]=m
    direct,progress=models[DIRECT],models[PROGRESS]
    assert torch.equal(*rng)
    assert set(progress.state_dict())-set(direct.state_dict())=={TAG}
    assert all(torch.equal(v,progress.state_dict()[k]) for k,v in direct.state_dict().items())
    result['initial_states']=states;result['all_common_initial_tensors_identical']=True
    result['constructor_rng_identical']=True
    trainer.seed_all(1);original=reference.TemporalReadModel(copy.deepcopy(config));mr.rebuild_correlation_fuse(original,4)
    reference.load_fresh_parent(original,payload['model'])
    assert tensor_state_sha256(original.state_dict())==states[DIRECT]
    del original
    train,_=nominal.raw_datasets(False,1);data=H4StatusDataset(mr.MotionCanvasDataset(train,'native'))
    result['producer']=producer_checks(data);result['composition']=composition_checks()
    raw=default_collate([data[0],data[17]]);batch=to_device(raw,torch.device('cuda:0'));x=inputs(batch)
    result['fixture_rows']=raw['row'].tolist();loss_fn=wrap_compute_loss(compute_loss,.25)
    direct.cuda().eval();progress.cuda().eval()
    result['common_features']={}
    for precision in ('fp32','bf16'):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=direct(**x);b=progress(**x)
        difference={k:float((a[k].float()-b[k].float()).abs().max()) for k in KEYS}
        assert max(difference.values())==0,difference
        result['common_features'][precision]=difference
        assert torch.isfinite(b['plan_abs']).all() and b['plan_abs'].shape==(2,6,2)
        result['initial_plan_difference_expected']=float(abs(a['plan_abs']-b['plan_abs']).max())
        del a,b
    set_training_mode(progress,'fixed')
    with torch.autocast('cuda',dtype=torch.bfloat16):out=progress(**x)
    plan_loss(out,batch).backward()
    final_linear=[m for m in progress.planner.xy_head.modules() if isinstance(m,torch.nn.Linear)][-1]
    norms=final_linear.weight.grad.square().sum(1).sqrt()
    assert (norms>0).all() and torch.isfinite(norms).all()
    wanted=('motion_encoder.correlation_fuse.0.weight','backbone_fpn.layer1.0.conv1.weight',
        'planner.temporal_read.attention.out_proj.weight')
    gradients={n:float(p.grad.norm()) if p.grad is not None else None for n,p in progress.named_parameters() if n in wanted}
    assert len(gradients)==len(wanted) and all(v is not None and v>0 for v in gradients.values()),gradients
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in progress.parameters())
    result['PREFIX_gradients']={'length_heading_output_rows':norms.tolist(),'shared_visual_path':gradients}
    progress.zero_grad(set_to_none=True);del out;progress.eval()
    with torch.no_grad():progress.shared_status_query_fusion.status_mlp[-1].weight.fill_(.01)
    seen={}
    hooks=[progress.scene_encoder.occ_head.register_forward_pre_hook(lambda m,a:seen.update(occ=a[0])),
        progress.scene_encoder.lane_head.register_forward_pre_hook(lambda m,a:seen.update(lane=a[0])),
        progress.planner.register_forward_pre_hook(lambda m,a:seen.update(planner=a[0]))]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):normal=progress(**x)
    assert seen['occ'] is seen['lane'] and torch.equal(seen['occ'].flatten(2).transpose(1,2).float(),seen['planner'].float())
    for hook in hooks:hook.remove()
    result['shared_consumers_identical']=True;result['input_isolation']={}
    for name,changed in [('provided_status',dict(x,provided_status5=x['provided_status5']+5)),('goal',dict(x,goal_xy=x['goal_xy']+10))]:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):other=progress(**changed)
        diff={k:float(abs(normal[k]-other[k]).max()) for k in ('motion_pair_features','motion_features','state_hat','history_hat')}
        assert max(diff.values())==0
        result['input_isolation'][name]=dict(diff,scene_max_change=float(abs(normal['scene_features']-other['scene_features']).max()))
        del other
    assert result['input_isolation']['provided_status']['scene_max_change']>0
    del normal,seen
    with torch.no_grad():out=progress(**x)
    normalizers=to_device(build_loss_normalizers(raw),torch.device('cuda:0'))
    full,parts=loss_fn(out,batch,LossWeights(),normalizers=normalizers)
    accumulated=sum(loss_fn({k:(v[i:i+1] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==2 else v)
        for k,v in out.items()},slice_batch(batch,i,i+1),LossWeights(),normalizers=normalizers)[0] for i in range(2))
    assert torch.allclose(full,accumulated,atol=2e-5,rtol=2e-6)
    result['full_effective_batch_loss']={'whole':float(full),'micro_sum':float(accumulated),'absolute_difference':float(abs(full-accumulated))}
    del out,full,parts,accumulated
    from motiondrive_v2_flip_augment import flip_item
    flip=mr.wrap_flip_item(flip_item);item=data[0];twice=flip(flip(item,768,384),768,384)
    for k,v in item.items():
        if isinstance(v,torch.Tensor):
            assert torch.allclose(v,twice[k],atol=1e-4,rtol=0) if v.is_floating_point() else torch.equal(v,twice[k]),k
    result['double_flip']=True
    clone=H4ProgressModel(copy.deepcopy(config),arm=PROGRESS);mr.rebuild_correlation_fuse(clone,4)
    clone.load_state_dict(progress.state_dict(),strict=True);clone.cuda().eval()
    one=inputs(to_device(slice_batch(raw,0,1),torch.device('cuda:0')))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=progress(**one)['plan_abs'];b=clone(**one)['plan_abs']
    assert torch.equal(a,b);result['strict_reload_max_diff']=float(abs(a-b).max())
    try:direct.load_state_dict(progress.state_dict(),strict=True)
    except RuntimeError:result['wrong_graph_strict_load_rejected']=True
    else:raise AssertionError('Wrong graph accepted')
    # load_state_dict can copy matching keys before raising; reconstruct original.
    direct.cpu();load_initializer(direct,payload['model']);direct.cuda()
    del clone,a,b,payload;gc.collect();torch.cuda.empty_cache()
    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.module_tracker as mt
    class NoHandle:
        def remove(self):pass
    old=mt.register_multi_grad_hook;mt.register_multi_grad_hook=lambda *a,**k:NoHandle()
    result['whole_forward_cost']={}
    try:
        for arm,network in models.items():
            with torch.no_grad(),FlopCounterMode(display=False) as count:network(**one)
            flops=int(sum(count.get_flop_counts().get('Global',{}).values()));times=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(12):
                    torch.cuda.synchronize();tick=time.monotonic();network(**one);torch.cuda.synchronize()
                    if i>=3:times.append(1000*(time.monotonic()-tick))
            result['whole_forward_cost'][arm]={'official_counter_flops':flops,'parameters':sum(p.numel() for p in network.parameters()),
                'B200_B1_median_ms':float(np.median(times)),'RTX4090_ms':None}
            assert flops<7053e9
    finally:mt.register_multi_grad_hook=old
    result.update(status='passed',elapsed_seconds=time.monotonic()-start,
        source_sha256={p.name:sha(p) for p in (HERE/'progress_model.py',HERE/'h4_data.py',HERE/'h4_status.py',Path(__file__))},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (REPORT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
