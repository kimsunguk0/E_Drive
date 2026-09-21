"""Actual parent parity, dependency boundaries, new-branch learning and export."""
from pathlib import Path
import argparse,copy,gc,json,sys
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate,Subset
from train_capacity import *
from capacity_model import CapacityModel,load_export,export_student,manifest_for
from native_model import load_export as load_native
from models.motiondrive_v2.config import MotionDriveV2Config
from motiondrive_v2_training import to_device,set_training_mode

KEYS=('plan_abs','scene_features','motion_features','motion_pair_features','state_hat','history_hat','occ_logits','lane_logits')
STAGE=RUNS/'verification';ROWS=(0,30000,60000)
BASE_INPUTS=trainer.model_inputs
def inputs(batch,**kw):
    x=mr.model_inputs_with_canvas(BASE_INPUTS,batch,time_input='nominal',nominal_history_seconds=(.1,.2,.5,1.))
    x['provided_status5']=batch['provided_status5'];x['high_images']=batch[KEY];return x

def datasets(arm):
    tr,va=nominal.raw_datasets(False,1)
    def wrap(ds):
        if arm=='C-AGENT':ds=agent_data.AgentDataset(ds)
        return NativeDataset(H4StatusDataset(mr.MotionCanvasDataset(ds,'native')),'M-NATIVE')
    return wrap(tr),wrap(va)

def forward(model,x):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**x)

def delta(a,b,keys=KEYS):return {k:float((a[k].float()-b[k].float()).abs().max()) for k in keys}

def initial(arm):
    assert sha(PARENT)==PARENT_SHA
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    trainer.seed_all(1)
    parent,_=load_native(PARENT,'cuda');parent.eval()
    trainer.seed_all(1)
    model=CapacityModel(MotionDriveV2Config(**cp['manifest']['model_config']),execution_config=arm_config(arm))
    mapping=load_parent(model,cp['model']);model.cuda().eval()
    tr,va=datasets(arm);raw=default_collate([tr[i] for i in ROWS]);batch=to_device(raw,torch.device('cuda'));x=inputs(batch)
    result=dict(arm=arm,parent_sha256=PARENT_SHA,mapping=mapping,parity={},parameters=sum(p.numel() for p in model.parameters()))
    for precision in ('fp32','bf16'):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            a=parent(**x);b=model(**x)
        diff=delta(a,b);assert max(diff.values())<2e-4,diff;result['parity'][precision]=diff
    del parent,a,b;gc.collect();torch.cuda.empty_cache()
    out=forward(model,x);result['boundary']={}
    for name,change in [('provided_status5',5.),('goal_xy',10.)]:
        xx=dict(x);xx[name]=x[name]+change
        dd=delta(out,forward(model,xx),('motion_features','motion_pair_features','state_hat','history_hat'))
        assert max(dd.values())==0,dd;result['boundary'][name]=dd
    # Every consumer still receives the same common scene tensor.
    captures={'occ':[],'lane':[],'planner':[]}
    hooks=[model.scene_encoder.occ_head.register_forward_pre_hook(lambda m,a:captures['occ'].append(a[0])),
           model.scene_encoder.lane_head.register_forward_pre_hook(lambda m,a:captures['lane'].append(a[0])),
           model.planner.register_forward_pre_hook(lambda m,a:captures['planner'].append(a[0]))]
    try:forward(model,x)
    finally:
        for h in hooks:h.remove()
    occ,lane,plan=(captures[k][-1] for k in ('occ','lane','planner'))
    assert occ.data_ptr()==lane.data_ptr()==plan.data_ptr();result['same_shared_scene_consumers']=True
    if arm=='C-R101':
        assert len(model.backbone_fpn.layer3)==23
        assert all(model.backbone_fpn.layer3[i].bn3.weight.count_nonzero()==0 for i in range(6,23))
    if arm=='C-DECSPLIT':
        assert model.planner.length_queries.data_ptr()!=model.planner.heading_queries.data_ptr()
        assert model.planner.length_decoder is not model.planner.heading_decoder
    if arm=='C-AGENT':
        flip=agent_data.wrap_flip(mr.wrap_flip_item(flip_api.flip_item))
        item=tr[0];twice=flip(flip(item,768,384),768,384)
        assert all(torch.equal(item[k],twice[k]) for k in ('agent_centers','agent_delta','agent_valid'))
        # Compare one effective batch with uneven microbatches, including empty masks.
        gen=torch.Generator().manual_seed(73)
        field=torch.randn(3,12,64,48,generator=gen).cuda().requires_grad_();norm=agent_data.normalizers(batch)
        loss=agent_data.agent_loss(field,batch,norm);grad=torch.autograd.grad(loss,field)[0]
        total=0
        for lo,hi in ((0,1),(1,3)):
            bb={k:v[lo:hi] for k,v in batch.items() if isinstance(v,torch.Tensor)}
            total=total+agent_data.agent_loss(field[lo:hi],bb,norm)
        grad2=torch.autograd.grad(total,field)[0]
        assert torch.allclose(loss,total,atol=1e-6,rtol=1e-6) and torch.allclose(grad,grad2,atol=1e-6,rtol=1e-6)
        invalid=dict(batch,agent_valid=torch.zeros_like(batch['agent_valid']),agent_delta=torch.full_like(batch['agent_delta'],float('nan')))
        empty=agent_data.agent_loss(field,invalid,agent_data.normalizers(invalid));assert empty==0 and torch.isfinite(empty)
        # A true annotation point is used only by loss sampling, not the model forward.
        set_training_mode(model,'fixed')
        with torch.autocast('cuda',dtype=torch.bfloat16):aa=model(**x)
        av=agent_data.agent_loss(aa['agent_future_field'],batch,norm)
        params=dict(model.named_parameters())
        names=['agent_future_head.2.weight','scene_encoder.key_proj.0.weight']
        gg=torch.autograd.grad(av,[params[k] for k in names],allow_unused=True)
        result['agent_checks']={'loss':float(av),'valid_points':int(batch['agent_valid'].sum()),'flip':True,
            'full_micro_loss_delta':float((loss-total).abs()),'gradient_delta':float((grad-grad2).abs().max()),
            'initial_head_grad_norm':float(gg[0].norm()),'initial_scene_gradient_zero_expected':float(gg[1].norm())}
        assert result['agent_checks']['initial_head_grad_norm']>0
        model.eval();del aa,gg,av,field,grad,grad2,total,loss
    STAGE.mkdir(parents=True,exist_ok=True)
    torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},'manifest':manifest_for(model,cp['manifest']),'step':0},STAGE/f'{arm}_initial.pth')
    expected=forward(model,x)
    torch.save({k:expected[k].cpu() for k in KEYS},STAGE/f'{arm}_expected.pth')
    torch.save({k:v.cpu() for k,v in x.items()},STAGE/f'{arm}_inputs.pth')
    del out,expected,x,batch,raw,captures,occ,lane,plan;gc.collect();torch.cuda.empty_cache()
    before=tensor_state_sha256(model.state_dict());old=trainer.model_inputs;trainer.model_inputs=inputs
    try:
        r,rr=trainer.evaluate(model,DataLoader(va,batch_size=8,num_workers=8),torch.device('cuda'),'bf16',
            time_input='nominal',detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
        atomic(RUNS/f'{arm}-step0.json',{'report':r,'records':rr})
        idx=np.random.default_rng(20260921).choice(len(tr),256,replace=False).tolist()
        pr,recs=trainer.evaluate(model,DataLoader(Subset(tr,idx),batch_size=8,num_workers=4),torch.device('cuda'),'bf16',
            time_input='nominal',detailed_records=True,nominal_history_seconds=model.config.nominal_history_seconds)
        atomic(RUNS/f'{arm}-train-step0.json',{'report':pr,'records':recs,'indices':idx})
    finally:trainer.model_inputs=old
    assert before==tensor_state_sha256(model.state_dict())
    assert abs(r['official_d3']-.14459766188205064)<1e-5,r['official_d3']
    result.update(status='passed',PREFIX=r['official_d3'],n=r['n'],initial_state_sha256=before)
    atomic(REPORT/f'{arm}_initial_checks.json',result);print(json.dumps(result),flush=True)

def reload(arm):
    x=torch.load(STAGE/f'{arm}_inputs.pth',map_location='cuda',weights_only=True)
    initial,cp=load_export(STAGE/f'{arm}_initial.pth','cuda')
    expected=torch.load(STAGE/f'{arm}_expected.pth',map_location='cuda',weights_only=True)
    dd=delta(forward(initial,x),expected);assert max(dd.values())<2e-4,dd
    before=cp['model'];del initial,expected;gc.collect();torch.cuda.empty_cache()
    model,cp=load_export(RUNS/f'{arm}-s1-smoke/ckpt_step5.pth','cuda')
    manifest=json.loads((RUNS/f'{arm}-s1-smoke/manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['step']==5 and not manifest['nonfinite_count']
    names={'C-CTRL':['planner.length_head.3.weight'],
        'C-R101':['backbone_fpn.layer3.6.bn3.weight','backbone_fpn.layer3.6.conv1.weight','backbone_fpn.layer3.22.bn3.weight'],
        'C-DECSPLIT':['planner.length_queries','planner.heading_queries'],
        'C-AGENT':['agent_future_head.0.weight','agent_future_head.2.weight']}[arm]
    assert all(not torch.equal(before[n],cp['model'][n]) for n in names),names
    student=export_student(cp);torch.save(student,STAGE/f'{arm}_student.pth')
    expected=forward(model,x);torch.save({k:expected[k].cpu() for k in KEYS},STAGE/f'{arm}_student_expected.pth')
    if arm=='C-DECSPLIT':assert not torch.equal(cp['model']['planner.length_queries'],cp['model']['planner.heading_queries'])
    if arm=='C-AGENT':
        ga=json.loads((RUNS/f'{arm}-s1-smoke/gradient_audit.json').read_text())
        assert ga[-1]['unclipped_grad_norms']['agent_future_head.0.weight']>0
    atomic(REPORT/f'{arm}_reload.json',dict(status='passed',fresh_process_initial_diff=dd,updated=names,
        strict_student_export=True,stream=manifest['stream_audit']))

def export(arm):
    x=torch.load(STAGE/f'{arm}_inputs.pth',map_location='cuda',weights_only=True)
    model,_=load_export(STAGE/f'{arm}_student.pth','cuda')
    expected=torch.load(STAGE/f'{arm}_student_expected.pth',map_location='cuda',weights_only=True)
    dd=delta(forward(model,x),expected);assert max(dd.values())<2e-4,dd
    assert not hasattr(model,'agent_future_head')
    atomic(REPORT/f'{arm}_export.json',dict(status='passed',strict_student_diff=dd))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('initial','reload','export'),required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    {'initial':initial,'reload':reload,'export':export}[a.mode](a.arm)
