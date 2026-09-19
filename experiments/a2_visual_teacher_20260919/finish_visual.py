"""Terminal comparison and strict teacher-free student export."""
import json,os,csv
import numpy as np
from visual_teacher import *
from common import PARENT,nominal,trainer,mr,NominalStatusDataset,inputs,to_device
from terminal_review import metrics
from torch.utils.data import DataLoader

RUN=RUNS/'A2-VIS-TEACHER-s1-r2'
CONTROL=ROOT/'work_dirs/a2_next_20260919/A2-G0-s1'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0','1','2','3')
    torch.set_num_threads(4);trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    assert json.loads((RUN/'manifest.json').read_text())['status']=='completed'
    assert json.loads((RUN/'teacher_unchanged.json').read_text())['unchanged']
    def logged(path):
        return {x['step']:x for x in map(json.loads,(path/'metrics.jsonl').read_text().splitlines()) if x['kind']=='train'}
    a,b=logged(RUN),logged(CONTROL)
    assert set(a)==set(b)
    assert all(a[k]['sample_order_sha256']==b[k]['sample_order_sha256'] and a[k]['lr']==b[k]['lr'] for k in a)
    receipt=json.loads((REPORT/'control_reuse.json').read_text())
    last_logged=max(a)
    receipt['complete_row_stream_hash_check']={'all_logged_steps_equal':True,'n_steps':len(a),
        'last_logged_step':last_logged,'last_logged_sha256':a[last_logged]['sample_order_sha256'],
        'scope':'All logged rolling hashes match through step3400. Last26 updates have the same deterministic epoch0 sampler policy, but no separate terminal rolling hash was written by the existing trainer.'}
    (REPORT/'control_reuse.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths={'PARENT':PARENT.parent/'final_eval.json','CONTROL':CONTROL/'final_eval.json',
        'VIS-1142':RUN/'predictions_step1142.json','VIS-2284':RUN/'predictions_step2284.json',
        'VIS-TERMINAL':RUN/'final_eval.json'}
    evaluations={k:json.loads(p.read_text()) for k,p in paths.items()}
    records={k:v['records'] for k,v in evaluations.items()}
    base=records['PARENT'];gt=np.asarray([r['gt_abs_xy'] for r in base]);rows=[r['row'] for r in base]
    bucket=np.asarray([r['bucket'] for r in base]);sessions=np.asarray([r['session'] for r in base])
    dg=np.diff(np.concatenate([np.zeros((len(gt),1,2)),gt],1),axis=1);mask=np.linalg.norm(dg,axis=-1)>.05
    stats={};scores={}
    for name,rr in records.items():
        assert rows==[r['row'] for r in rr] and np.array_equal(gt,np.array([r['gt_abs_xy'] for r in rr]))
        predicted=np.asarray([r['pred_abs_xy'] for r in rr])
        stats[name],scores[name]=metrics(predicted,gt,bucket,mask)
        dp=np.diff(np.concatenate([np.zeros((len(gt),1,2)),predicted],1),axis=1)
        length_error=np.linalg.norm(dp,axis=-1)-np.linalg.norm(dg,axis=-1)
        stats[name]['interval_length_MAE_m']=np.abs(length_error).mean(0).tolist()
        stats[name]['interval_length_signed_bias_m']=length_error.mean(0).tolist()
        stats[name]['progress_groups_existing_producer']=nominal.diagnostics(rr)['groups']
        stats[name]['auxiliary']={k:evaluations[name]['report'][k] for k in
            ['occ_iou','lane_iou','state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset']}
    unique=sorted(set(sessions));ix=[np.flatnonzero(sessions==s) for s in unique]
    draws=np.random.default_rng(0).integers(0,len(unique),(20000,len(unique)))
    def compare(left,right):
        delta=scores[left]-scores[right];sums=np.array([delta[i].sum() for i in ix]);counts=np.array([len(i) for i in ix])
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        return {'PREFIX_delta':float(delta.mean()),'session_CI95':np.quantile(boot,[.025,.975]).tolist(),
            'sessions_improved':int((sums<0).sum()),'group_contribution_delta':{g:float(delta[bucket==g].sum()/len(delta)) for g in sorted(set(bucket))}}
    comparisons={f'{x}-minus-{y}':compare(x,y) for x,y in [('VIS-TERMINAL','CONTROL'),('VIS-TERMINAL','PARENT'),('CONTROL','PARENT')]}
    checkpoint=RUN/'ckpt_step3426.pth';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=MotionDriveV2Config(**payload['manifest']['model_config'])
    training=VisualStudent(cfg);mr.rebuild_correlation_fuse(training,4)
    initial_projector=training.visual_projector.weight.detach().clone()
    training.load_state_dict(payload['model'],strict=True)
    parent_weights=torch.load(PARENT,map_location='cpu',weights_only=False)['model']
    assert {k:tuple(v.shape) for k,v in shared_state(training).items()}=={k:tuple(v.shape) for k,v in parent_weights.items()}
    layer_keys=[k for k in payload['model'] if k.startswith('backbone_fpn.out3.') and payload['model'][k].is_floating_point()]
    count=sum(payload['model'][k].numel() for k in layer_keys)
    layer_update=(sum(float((payload['model'][k].double()-parent_weights[k].double()).square().sum()) for k in layer_keys)/count)**.5
    projector_update=float((training.visual_projector.weight.detach()-initial_projector).square().mean().sqrt())
    assert layer_update>0 and projector_update>0
    del parent_weights
    clean=shared_state(training)
    student=SceneExtensionModel(cfg,arm=QREFINE);mr.rebuild_correlation_fuse(student,4)
    student.load_state_dict(clean,strict=True)
    training.cuda().eval();student.cuda().eval()
    _,tune=nominal.raw_datasets(False,1)
    data=NominalStatusDataset(mr.MotionCanvasDataset(tune,'native'))
    raw=next(iter(DataLoader(data,batch_size=1,num_workers=0)))
    x=inputs(to_device(raw,torch.device('cuda:0')))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=training(**x);pure=student(**x)
    gap=float((out['plan_abs'].float()-pure['plan_abs'].float()).abs().max());assert gap==0
    assert not any(k.startswith(PROJECTOR_PREFIX) for k in student.state_dict())
    export=RUN/'student_step3426.pth'
    exported={'model':{k:v.cpu() for k,v in clean.items()},'manifest':payload['manifest'],
        'step':3426,'student_export':{'training_checkpoint_sha256':sha(checkpoint),
            'model_state_sha256':tensor_state_sha256(clean),'strict_load':True,'B1_max_abs_XY':gap,
            'inference_graph':'A2-QREFINE-NOM','excluded':['teacher','visual_projector'],'DEV_only':True}}
    torch.save(exported,export)
    # Saving/loading the actual exported artifact is part of the strict export check.
    reread=torch.load(export,map_location='cpu',weights_only=False)
    student.load_state_dict(reread['model'],strict=True)
    protocol=json.loads((RUN/'experiment.json').read_text())
    results={'models':stats,'comparisons':comparisons,'primary':'VIS-TERMINAL vs CONTROL; separately vs PARENT',
        'intermediates':'planned V0 checkpoints; selecting one reuses V0',
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'student_export':str(export),'student_export_sha256':sha(export),'export_checks':exported['student_export'],
        'lambda_vis':protocol['lambda_vis'],'updates':3426,'source':protocol['source'],
        'teacher_frozen':True,'full_lineage_used':False,'row_stream_matches_G0':True,
        'inference_cost_contract':'Exported graph and tensor shapes match QREFINE parent. No teacher/projector in forward. A2 FULL single-head FLOPs are not assigned to this QREFINE student.',
        'validation_limit':'Repeated DEV use; bootstrap conditional on the same 11 sessions',
        'training_first_visual_loss':a[1]['visual_cosine'],'training_last_logged_visual_loss':a[last_logged]['visual_cosine'],
        'last_logged_train_step':last_logged,
        'parameter_updates':{'student_upper_FPN_RMS_from_parent':layer_update,
            'projector_RMS_from_private_initialization':projector_update,
            'scope':'Actual parameter changes; student updates include all existing losses. Visual-only gradient path was tested separately in preflight.'},
        'run_elapsed_seconds':json.loads((RUN/'manifest.json').read_text())['elapsed_seconds'],
        'coefficient_did_not_use_V0':True}
    (REPORT/'visual_results.json').write_text(json.dumps(results,indent=2)+'\n')
    fields=['run','PREFIX','L2_1s','L2_2s','L2_3s','nonstop_PREFIX','nonstop_contribution','first2s_PREFIX_contribution']
    with (REPORT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for name,m in stats.items():
            writer.writerow({'run':name,**{k:m[k] for k in fields[1:5]},
                'nonstop_PREFIX':m['groups']['nonstop']['PREFIX'],
                'nonstop_contribution':m['groups']['nonstop']['overall_contribution'],
                'first2s_PREFIX_contribution':m['first2s_PREFIX_contribution']})
    print('RESULT '+json.dumps({'scores':{k:v['PREFIX'] for k,v in stats.items()},'comparisons':comparisons,'export':str(export)}),flush=True)

if __name__=='__main__':main()
