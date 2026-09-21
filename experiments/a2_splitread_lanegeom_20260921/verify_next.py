"""Production factory, fixed step0 DEV, five-update and strict student export checks."""
from pathlib import Path
import argparse,copy,gc,json,os,sys
import numpy as np
import torch
from torch.utils.data import default_collate,DataLoader,Subset
from train_next import *
from next_model import NextModel,load_export,manifest_for,export_student,SIGNATURE
from arm_model import FourArmModel as ParentModel,arm_config as parent_config
from models.motiondrive_v2.config import MotionDriveV2Config
from motiondrive_v2_training import to_device,tensor_state_sha256
from verify import datasets,inputs,differences,KEYS

def forward(model,x):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**x)

def initial(arm):
    assert sha(PARENT)==PARENT_SHA
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    config=MotionDriveV2Config(**cp['manifest']['model_config'])
    trainer.seed_all(1);base=ParentModel(copy.deepcopy(config),execution_config=parent_config('P-CTRL'))
    mr.rebuild_correlation_fuse(base,4);rng=torch.get_rng_state().clone()
    trainer.seed_all(1);model=NextModel(copy.deepcopy(config),execution_config=arm_config(arm))
    mr.rebuild_correlation_fuse(model,4);assert torch.equal(rng,torch.get_rng_state())
    base.load_state_dict(cp['model'],strict=True);mapping=load_parent(model,cp['model'])
    tr,va=datasets();rows=[0,30000,60000]
    raw=default_collate([tr[i] for i in rows]);batch=to_device(raw,torch.device('cuda'));x=inputs(batch)
    model.cuda().eval();base.cuda().eval()
    result={'arm':arm,'status':'running','parent_sha256':PARENT_SHA,'mapping':mapping,'CPU_RNG_matches_control':True,
        'parameters':sum(p.numel() for p in model.parameters()),'initial_parity':{}}
    for precision in ('fp32','bf16'):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=base(**x);b=model(**x)
        diff=differences(a,b);assert diff['plan_abs']<1e-4 and max(diff[k] for k in KEYS if k!='plan_abs')==0,diff
        result['initial_parity'][precision]=diff
    del base,a,b;gc.collect();torch.cuda.empty_cache()
    out=forward(model,x);result['boundary']={}
    for key,value in (('provided_status5',5.),('goal_xy',10.)):
        xx=dict(x);xx[key]=xx[key]+value
        changed=forward(model,xx)
        diff=differences(out,changed,('motion_features','motion_pair_features','state_hat','history_hat'))
        assert max(diff.values())==0;result['boundary'][key]=diff
    stage=RUNS/'verification';stage.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'manifest':manifest_for(model,cp['manifest']),'step':0},stage/f'{arm}_initial.pth')
    torch.save({k:out[k].cpu() for k in KEYS},stage/f'{arm}_expected.pth')
    del out,changed;model.cuda().eval()
    before=tensor_state_sha256(model.state_dict());old_inputs=trainer.model_inputs
    trainer.model_inputs=inputs
    try:
        r,records=trainer.evaluate(model,DataLoader(va,batch_size=8,num_workers=8),torch.device('cuda'),'bf16',
            time_input='nominal',detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
        r.update(step=0,parent_step=3426)
        atomic(REPORT/f'{arm}_step0.json',{'report':r,'records':records})
        idx=np.random.default_rng(20260921).choice(len(tr),256,replace=False).tolist()
        pr,rr=trainer.evaluate(model,DataLoader(Subset(tr,idx),batch_size=8,num_workers=4),torch.device('cuda'),'bf16',
            time_input='nominal',detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
        atomic(REPORT/f'{arm}_train_probe_step0.json',{'report':pr,'records':rr,'indices':idx})
    finally:trainer.model_inputs=old_inputs
    assert before==tensor_state_sha256(model.state_dict())
    result.update(status='passed',step0_PREFIX=r['official_d3'],step0_n=r['n'],initial_state_sha256=before)
    atomic(REPORT/f'{arm}_initial_checks.json',result);print(json.dumps(result),flush=True)

def geometry_cache_check():
    from geometry_data import CACHE,GeometryDataset,wrap_geometry_flip,geometry_batch
    from design import geometry_targets
    from build_scene_supervision_v2 import read_scene_metadata,transform_points
    from motiondrive_v2_data import grid_centers
    tr,_=datasets();wrapped=GeometryDataset(tr);checks=[]
    for index in (0,30000,60000):
        item=wrapped[index];scene=item['scenario'];frame=int(item['frame'])
        meta=json.loads((tr.supervision_root/(scene+'.json')).read_text())
        path,=[Path(p) for p in meta['sources'] if p.endswith('/annotation/map.parquet')]
        frames,_,poses,_,lines,_=read_scene_metadata(path.parent.parent)
        pos={int(f):i for i,f in enumerate(frames)}[frame]
        local=[transform_points(line,np.linalg.inv(poses[pos])) for line in lines]
        expected=geometry_targets(local,grid_centers(),item['lane_valid'].numpy()[0])
        flip=wrap_geometry_flip(mr.wrap_flip_item(flip_api.flip_item))
        mirrored=flip(item,768,384);twice=flip(mirrored,768,384)
        for key in ('offset','axis'):
            assert np.array_equal(item['geo_'+key].numpy(),expected[key].transpose(2,0,1))
            assert np.array_equal(item['geo_'+key+'_valid'].numpy(),expected[key+'_valid'])
            wanted=torch.flip(item['geo_'+key],[-1]);wanted[1]*=-1
            assert torch.equal(wanted,mirrored['geo_'+key])
            assert torch.equal(item['geo_'+key],twice['geo_'+key])
            assert torch.equal(item['geo_'+key+'_valid'],twice['geo_'+key+'_valid'])
        checks.append({'index':index,'cache_matches_generator':True,'flip_and_double_flip_exact':True})
    manifest=json.loads((CACHE/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['rows']==83700 and manifest['scenes']==310
    atomic(REPORT/'geometry_cache_checks.json',{'status':'passed','checks':checks,'manifest_sha256':sha(CACHE/'manifest.json')})

def reload(arm):
    stage=RUNS/'verification';tr,_=datasets();x=inputs(to_device(default_collate([tr[i] for i in (0,30000,60000)]),torch.device('cuda')))
    model,p=load_export(stage/f'{arm}_initial.pth','cuda')
    expected=torch.load(stage/f'{arm}_expected.pth',weights_only=True,map_location='cuda')
    diff=differences(forward(model,x),expected);assert max(diff.values())==0,diff
    initial_state=p['model'];del model,expected;gc.collect();torch.cuda.empty_cache()
    run=RUNS/f'{arm}-s1-smoke'
    model,p=load_export(run/'ckpt_step5.pth','cuda');actual=forward(model,x)
    m=json.loads((run/'manifest.json').read_text())
    assert m['status']=='completed' and m['step']==5 and m['nonfinite_count']==0
    if arm=='P-LANE-GEOM':
        for name in ('lane_geometry_head.net.0.weight','lane_geometry_head.net.2.weight'):
            assert not torch.equal(initial_state[name],p['model'][name]),name
    if arm=='P-SPLITREAD':
        for branch in ('length','heading'):
            name=f'planner.{branch}_read.attention.in_proj_weight'
            assert not torch.equal(initial_state[name],p['model'][name]),name
    torch.save(export_student(p),stage/f'{arm}_student.pth')
    torch.save({k:actual[k].cpu() for k in KEYS},stage/f'{arm}_student_expected.pth')
    bad=copy.deepcopy(p);bad['manifest']['next_execution_config']=arm_config('P-CTRL-NEXT' if arm!='P-CTRL-NEXT' else 'P-SPLITREAD')
    torch.save(bad,stage/f'{arm}_wrong.pth')
    try:load_export(stage/f'{arm}_wrong.pth')
    except ValueError:pass
    else:raise AssertionError('Wrong graph accepted')
    atomic(REPORT/f'{arm}_reload.json',{'status':'passed','initial_new_process_maxdiff':diff,
        'five_updates_complete':True,'new_branch_weights_changed':True,'wrong_graph_rejected':True,
        'sample_stream':m['stream_audit'],'optimizer_effective_groups':m['optimizer_effective_groups']})

def export_check(arm):
    stage=RUNS/'verification';tr,_=datasets();x=inputs(to_device(default_collate([tr[i] for i in (0,30000,60000)]),torch.device('cuda')))
    model,p=load_export(stage/f'{arm}_student.pth','cuda')
    assert not hasattr(model,'lane_geometry_head')
    expected=torch.load(stage/f'{arm}_student_expected.pth',weights_only=True,map_location='cuda')
    diff=differences(forward(model,x),expected);assert max(diff.values())==0,diff
    atomic(REPORT/f'{arm}_export.json',{'status':'passed','student_export_new_process_maxdiff':diff,'training_aux_removed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,default=ARMS[0]);p.add_argument('--mode',choices=('initial','reload','export','geometry'),required=True)
    args=p.parse_args();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    if args.mode=='geometry':geometry_cache_check()
    else:{'initial':initial,'reload':reload,'export':export_check}[args.mode](args.arm)
