"""Matched near-FULL continuation, with/without confident short-pair dense motion supervision."""
import argparse,traceback
from flow_common import *

def audit_pixels(raw,digest):
    for key in ('images','history_images',mr.MOTION_CURRENT_KEY,mr.MOTION_HISTORY_KEY,'provided_status5'):
        x=raw[key]
        if x.ndim>=4:x=x[...,::37,::41]
        digest.update(x.float().contiguous().numpy().tobytes())

def grad_norm(parameters):
    values=[p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.

def update(model,head,capture,teacher,raw,optimizer,step,epoch,d,audit=False):
    set_training_mode(model,'fixed');head.train()
    for g in optimizer.param_groups:g['lr']=g['base_lr']*lr_factor(step,d['updates'])
    optimizer.zero_grad(set_to_none=True)
    target=mask=pairs=None;info={};teacher_seconds=0.
    if teacher is not None:
        began=time.monotonic();target,mask,pairs,info=teacher_batch(teacher,raw,epoch)
        teacher_seconds=time.monotonic()-began
    denom=mask.sum()*2 if mask is not None else None
    normalizers=to_device(build_loss_normalizers(raw),torch.device('cuda'))
    coefficient=d['extra_loss_lambda']*min(1.,(step+1)/d['lambda_ramp_steps'])
    parts={}
    before={n:p.detach().clone() for n,p in model.named_parameters() if
            n.startswith('motion_encoder.correlation_fuse.0.')} if audit else {}
    for start in range(0,len(raw['images']),8):
        batch=to_device(trainer.slice_batch(raw,start,start+8),torch.device('cuda'))
        output=forward(model,batch);loss,items,_=objective(output,batch,normalizers)
        if teacher is not None:
            visual=flow_loss(head,capture.maps,target[start:start+8],mask[start:start+8],pairs[start:start+8],denom)
            loss=loss+coefficient*visual
            items=dict(items,flow_raw=visual,flow_weighted=coefficient*visual)
        if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward()
        for k,v in dict(items,optimized_total=loss).items():parts[k]=parts.get(k,0.)+float(v.detach())
        capture.clear();del batch,output,loss,items
    motion=[p for n,p in model.named_parameters() if n.startswith('motion_encoder.correlation_fuse.')]
    motion_norm=grad_norm(motion) if audit else None
    head_norm=grad_norm(head.parameters()) if audit else None
    params=list(model.parameters())+list(head.parameters())
    global_norm=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True))
    optimizer.step()
    change=(math.sqrt(sum(float((p.detach().double()-before[n].double()).square().sum())
                       for n,p in model.named_parameters() if n in before)) if audit else None)
    return dict(losses=parts,gradient_norm=global_norm,motion_fuse_gradient_norm=motion_norm,
                flow_readout_gradient_norm=head_norm,motion_fuse_update_norm=change,
                teacher_seconds=teacher_seconds,flow_coefficient=coefficient,
                lr={g['name']:g['lr'] for g in optimizer.param_groups},**info)

def smoke(args,model,cp,head,capture,teacher,train,val,d):
    out=FLOW_REPORT/'smoke'/args.arm;assert not out.exists();out.mkdir(parents=True)
    fixture=to_device(default_collate([val[i] for i in (0,800)]),torch.device('cuda'))
    model.eval()
    with torch.no_grad():initial=forward(model,fixture)
    fixture_inputs=inputs(fixture,time_input='nominal',nominal_history_seconds=model.config.nominal_history_seconds)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        altered=model(**dict(fixture_inputs,provided_status5=fixture_inputs['provided_status5']+.1,goal_xy=fixture_inputs['goal_xy']+1.))
    invariant={k:float((initial[k]-altered[k]).abs().max()) for k in
               ('motion_features','motion_pair_features','state_hat','history_hat')}
    assert max(invariant.values())==0.,invariant
    capture.clear();del initial,altered
    buffers={n:t.detach().clone() for n,t in model.named_buffers()}
    teacher_before=tensor_state_sha256(teacher.state_dict()) if teacher is not None else None
    optimizer=optim(model,head);train.set_epoch(AUGMENT_EPOCH_OFFSET)
    rows=[];digest=hashlib.sha256();pixels=hashlib.sha256()
    began=time.monotonic()
    for step,raw in enumerate(loader(train,train=True)):
        if step==100:break
        start=time.monotonic()
        digest.update(np.asarray(raw['row'],dtype='<i8').tobytes());audit_pixels(raw,pixels)
        r=update(model,head,capture,teacher,raw,optimizer,step,AUGMENT_EPOCH_OFFSET,d,step<3 or step==99)
        r.update(step=step+1,seconds=time.monotonic()-start);rows.append(r)
        if step<3 or (step+1)%20==0:print('FLOW_SMOKE '+json.dumps(r),flush=True)
    assert all(torch.equal(t,buffers[n]) for n,t in model.named_buffers())
    if teacher is not None:
        assert teacher_before==tensor_state_sha256(teacher.state_dict())
        assert any(r['flow_readout_gradient_norm'] and r['flow_readout_gradient_norm']>0 for r in rows)
    assert any(r['motion_fuse_update_norm'] and r['motion_fuse_update_norm']>0 for r in rows)
    model.eval()
    with torch.no_grad():expected=forward(model,fixture)
    saved={k:expected[k].detach().cpu() for k in ('plan_abs','scene_features','motion_features','state_hat','history_hat')}
    capture.clear()
    torch.save(dict(inputs={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in fixture_inputs.items()},expected=saved),out/'fixture.pth')
    save_student(out/'smoke.pth',model,head,optimizer,cp,d,100,AUGMENT_EPOCH_OFFSET)
    result=dict(status='passed',arm=args.arm,steps=100,teacher_unchanged=True,BN_buffers_unchanged=True,
        input_boundary=invariant,seconds_per_update_median=float(np.median([r['seconds'] for r in rows[5:]])),
        estimated_training_hours=float(np.median([r['seconds'] for r in rows[5:]])*UPDATES/3600),
        sample_order_sha256=digest.hexdigest(),sampled_input_sha256=pixels.hexdigest(),
        elapsed_seconds=time.monotonic()-began,updates=rows,source=d['source'])
    atomic(out/'checks.json',result);atomic(out/'protocol.json',d)
    print('FLOW_SMOKE_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('source','updates')}),flush=True)

def replay(args):
    out=FLOW_REPORT/'smoke'/args.arm
    cp=torch.load(out/'smoke.pth',map_location='cpu',weights_only=False)
    model=construct(cp,'cuda').eval()
    assert set(model.state_dict())==set(cp['model']) and not any('flow_readout' in k for k in cp['model'])
    fixture=torch.load(out/'fixture.pth',map_location='cuda',weights_only=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(**fixture['inputs'])
    differences={k:float((pred[k]-v).abs().max()) for k,v in fixture['expected'].items()}
    assert max(differences.values())<2e-5,differences
    atomic(out/'replay.json',dict(status='passed',teacher_and_head_absent=True,strict_student_reload=True,differences=differences))
    print(json.dumps(differences))

def train_main(args,model,cp,head,capture,teacher,train,val,probe,idx,d):
    check=json.loads((FLOW_REPORT/'smoke'/args.arm/'checks.json').read_text())
    replayed=json.loads((FLOW_REPORT/'smoke'/args.arm/'replay.json').read_text())
    assert check['status']==replayed['status']=='passed'
    for path,h in check['source'].items():assert sha(ROOT/path)==h,path
    for other in FLOW_ARMS:
        c=json.loads((FLOW_REPORT/'smoke'/other/'checks.json').read_text())
        assert c['sample_order_sha256']==check['sample_order_sha256']
        assert c['sampled_input_sha256']==check['sampled_input_sha256']
    run=FLOW_RUNS/(args.arm+'-s1');assert not run.exists();run.mkdir(parents=True)
    atomic(run/'experiment.json',d);atomic(FLOW_REPORT/f'protocol_{args.arm}_s1.json',d)
    optimizer=optim(model,head);step=0;epoch=0;samples=0;curve=[]
    digest=hashlib.sha256();pixels=hashlib.sha256();started=time.monotonic();train_seconds=0.
    state=dict(status='running',pid=os.getpid(),arm=args.arm,physical_gpu=args.gpu,updates=UPDATES,nonfinite_count=0,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_sha256=PARENT_SHA,
               source_commit=d['source_commit'],evaluation_is_in_fit=False)
    def status():
        state.update(step=step,samples=samples,epoch=epoch,elapsed_seconds=time.monotonic()-started,
                     train_seconds=train_seconds,seconds_per_update=train_seconds/step if step else None,
                     sample_order_sha256=digest.hexdigest(),sampled_input_sha256=pixels.hexdigest(),curve=curve)
        atomic(run/'manifest.json',state)
    def evaluation():
        capture.clear();r,rr=evaluate(model,val);pr,pp=evaluate(model,probe);capture.clear()
        assert r['n']==1998 and rows_sha([x['row'] for x in rr])==d['tune_data']['rows_sha256']
        if step==0:assert abs(r['official_d3']-PARENT_PREFIX)<1e-6,r['official_d3']
        r.update(step=step,evaluation_is_in_fit=False)
        atomic(run/f'predictions_step{step}.json',dict(report=r,records=rr))
        atomic(run/f'train_probe_step{step}.json',dict(report=pr,records=pp,indices=idx))
        atomic(run/f'diagnostics_step{step}.json',nominal.diagnostics(rr))
        row=dict(step=step,PREFIX=r['official_d3'],train_probe_PREFIX=pr['official_d3'],elapsed_seconds=time.monotonic()-started)
        curve.append(row);atomic(FLOW_REPORT/f'curve_{args.arm}.json',curve)
        if step:save_student(run/f'ckpt_step{step}.pth',model,head,optimizer,cp,d,step,epoch+AUGMENT_EPOCH_OFFSET)
        status();print('FLOW_EVAL '+json.dumps(row),flush=True)
    try:
        status();evaluation();stream=loader(train,train=True)
        with (run/'metrics.jsonl').open('w',buffering=1) as log:
            while step<UPDATES:
                train.set_epoch(epoch+AUGMENT_EPOCH_OFFSET)
                for raw in stream:
                    if step>=UPDATES:break
                    before=time.monotonic();digest.update(np.asarray(raw['row'],dtype='<i8').tobytes())
                    if step<5 or (step+1)%100==0:audit_pixels(raw,pixels)
                    rec=update(model,head,capture,teacher,raw,optimizer,step,epoch+AUGMENT_EPOCH_OFFSET,d,
                               step<3 or (step+1)%100==0)
                    train_seconds+=time.monotonic()-before;step+=1;samples+=len(raw['images'])
                    rec.update(kind='train',step=step,epoch=epoch,augmentation_epoch=epoch+AUGMENT_EPOCH_OFFSET,
                        elapsed_seconds=time.monotonic()-started,train_seconds=train_seconds,
                        sample_order_sha256=digest.hexdigest(),sampled_input_sha256=pixels.hexdigest())
                    log.write(json.dumps(rec,allow_nan=False)+'\n')
                    if step<=3 or step%100==0:status();print(json.dumps(rec),flush=True)
                    if step in EVAL_STEPS:evaluation()
                epoch+=1
        state['status']='completed';status()
        atomic(run/'result.json',dict(status='completed',terminal=curve[-1],
            best_on_reused_V0=min(curve,key=lambda r:r['PREFIX']),parent_PREFIX=PARENT_PREFIX,
            primary='fixed terminal',automatic_followup=False))
    except BaseException as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        if isinstance(e,FloatingPointError):state['nonfinite_count']+=1
        status();raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=FLOW_ARMS,required=True)
    ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--mode',choices=['smoke','replay','train'],required=True)
    ap.add_argument('--gpu-override-reason',
                    help='run on a GPU other than the planned one; the planned GPU is '
                         'recorded either way so the allocation stays auditable')
    args=ap.parse_args()
    if args.gpu!=FLOW_ARMS[args.arm]:
        assert args.gpu_override_reason,(
            'planned GPU for %s is %d; pass --gpu-override-reason to use %d'
            %(args.arm,FLOW_ARMS[args.arm],args.gpu))
        print('GPU_OVERRIDE '+json.dumps(dict(arm=args.arm,planned=FLOW_ARMS[args.arm],
              actual=args.gpu,reason=args.gpu_override_reason)),flush=True)
    # flow_common.setup pins the GPU to the planned pair, and its hash is pinned
    # by calibration.json, so editing it there would invalidate the calibration.
    # The same work is done here instead, minus that one assert; everything the
    # setup actually guarantees -- the visible device, threads, seed and the
    # cudnn/TF32 determinism flags -- is reproduced exactly.
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    torch.set_num_threads(4);trainer.seed_all(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    if args.mode=='replay':replay(args);return
    for p,h in json.loads((FLOW_REPORT/'calibration.json').read_text())['source'].items():assert sha(ROOT/p)==h,p
    model,cp=get_model();head=FlowReadout().cuda();capture=Capture(model)
    with preserve_rng():teacher=FrozenFlowTeacher() if args.arm=='F-FLOW' else None
    train,val,probe,idx=datasets();d=protocol(args.arm,args.gpu,model,train,val)
    try:
        if args.mode=='smoke':smoke(args,model,cp,head,capture,teacher,train,val,d)
        else:train_main(args,model,cp,head,capture,teacher,train,val,probe,idx,d)
    finally:capture.remove()

if __name__=='__main__':main()
