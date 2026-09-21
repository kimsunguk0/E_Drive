"""RTX4090 frozen baseline and UNTRAINED input-size cost prototypes.

Upsizing prepared tensors measures cost only; it does not add image information.
No prototype predictions are evaluated as accuracy or exported for submission.
"""
from pathlib import Path
import argparse,sys,os,json,datetime,hashlib,gc
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(os.environ['ADCL_CODE_ROOT'])
sys.path.insert(0,str(ROOT/'experiments/a2_progress_full_20260921'))
from infer_full import load_model,configure,prepare_clip,sha,mr

def resized(t,hw):
    if t.shape[-2:]==hw:return t
    shape=t.shape
    return F.interpolate(t.reshape(-1,*shape[-3:]),hw,mode='bilinear',align_corners=False,antialias=True).reshape(*shape[:-2],*hw)

def inputs_for(prepared,current_hw,history_hw,motion_hw):
    x={k:v.clone() for k,v in prepared.inputs.items()}
    h,w=x['images'].shape[-2:];sy,sx=current_hw[0]/h,current_hw[1]/w
    x['images']=resized(x['images'],current_hw)
    x['history_images']=resized(x['motion_history'],history_hw)
    x['motion_current']=resized(x['motion_current'],motion_hw)
    x['motion_history']=resized(x['motion_history'],motion_hw)
    # Pixel-center geometry for this cost prototype. The original baseline
    # must retain every input tensor exactly, including its history resize.
    if (sy,sx)!=(1.,1.):
        a=x['lidar2img'].clone()
        x['lidar2img'][...,0,:]=sx*a[...,0,:]+((sx-1)/2)*a[...,2,:]
        x['lidar2img'][...,1,:]=sy*a[...,1,:]+((sy-1)/2)*a[...,2,:]
    return {k:v.cuda() for k,v in x.items()}

def forward(model,x):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**x)

def measure(model,x,repeats=100):
    for _ in range(30):forward(model,x)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();y=forward(model,x);end.record();end.synchronize()
        assert torch.isfinite(y['plan_abs']).all()
        times.append(start.elapsed_time(end));del y
    return {'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
        'warmup':30,'repeats':repeats,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}

def flops(model,x):
    import torch.utils.module_tracker as tracker
    from torch.utils.flop_counter import FlopCounterMode
    class Handle:
        def remove(self):pass
    old=tracker.register_multi_grad_hook;tracker.register_multi_grad_hook=lambda *a,**k:Handle()
    try:
        with torch.no_grad(),FlopCounterMode(display=False) as counter:model(**x)
        return int(sum(counter.get_flop_counts()['Global'].values()))
    finally:tracker.register_multi_grad_hook=old

def profile_baseline(model,x):
    records={};handles=[];starts={};calls={};shapes=[];capture=[True]
    for label,module in [('backbone',model.backbone_fpn),('scene',model.scene_encoder),('motion',model.motion_encoder),('planner',model.planner)]:
        def pre(mod,args,name=label):
            event=torch.cuda.Event(enable_timing=True);event.record();starts[name]=event
        def post(mod,args,out,name=label):
            stop=torch.cuda.Event(enable_timing=True);stop.record();index=calls.get(name,0);calls[name]=index+1
            key=f'{name}_{index}';records.setdefault(key,[]).append((starts[name],stop))
            if capture[0] and name=='backbone':shapes.append({'input':list(args[0].shape),'levels':[list(z.shape) for z in out[0]],'p4':list(out[1].shape)})
        handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    try:
        for _ in range(30):calls.clear();forward(model,x);capture[0]=False
        torch.cuda.synchronize()
    finally:
        for h in handles:h.remove()
    return {'component_median_ms':{k:float(np.median([a.elapsed_time(b) for a,b in v])) for k,v in records.items()},
        'backbone_calls':shapes,'scope':'Separate instrumented pass; component medians are not additive whole-forward timing.'}

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--clips-root',required=True);args=a.parse_args()
    target=Path(args.output);assert not target.exists();configure();assert 'RTX 4090' in torch.cuda.get_device_name(0)
    clips=[prepare_clip(p) for p in sorted(Path(args.clips_root).iterdir()) if p.is_dir()];assert len(clips)==2
    cases=[('BASE',(432,768),(216,384),(432,768),4),
        ('SCENE_HISTORY_768',(432,768),(432,768),(432,768),4),
        ('CURRENT_SCENE_1152',(648,1152),(216,384),(432,768),4),
        ('MOTION_1152_R6',(432,768),(216,384),(648,1152),6),
        ('ALL_1152_R6',(648,1152),(324,576),(648,1152),6),
        ('ALL_1536_R8',(864,1536),(432,768),(864,1536),8)]
    result={'scope':'COST ONLY: untrained resolution/radius prototypes; no accuracy claim, new learning, or submission',
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'checkpoint_sha256':sha(args.checkpoint),
        'script_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'precision':'B1 BF16 with FP32 planner','timing_scope':'Whole model including all image backbone passes; excludes preprocessing',
        'counter':'torch.utils.flop_counter.FlopCounterMode FP32 Global sum; same method as submitted model',
        'input_limitation':'Prototype tensors resized from baseline prepared inputs. Real accuracy experiments require original RGB decoding at target resolution.',
        'cases':{}}
    for name,ch,hh,mh,radius in cases:
        model,payload=load_model(args.checkpoint,True)
        if radius!=4:mr.rebuild_correlation_fuse(model,radius);model.cuda().eval()
        values=[]
        for index,prepared in enumerate(clips):
            x=inputs_for(prepared,ch,hh,mh)
            if name=='BASE':x={k:v.cuda() for k,v in prepared.inputs.items()}
            values.append({'clip_index':index,**measure(model,x)})
            if index==0:
                counted=flops(model,x)
                if name=='BASE':
                    assert counted==730044861120,(name,counted)
                    result['baseline_profile']=profile_baseline(model,x)
                shapes={k:list(v.shape) for k,v in x.items()}
            del x
        result['cases'][name]={'image_hw':ch,'scene_history_hw':hh,'motion_hw':mh,'correlation_radius':radius,
            'fuse_reinitialized_cost_prototype':radius!=4,'gflops':counted/1e9,'timing':values,
            'max_clip_median_ms':max(v['median_ms'] for v in values),'max_clip_p95_ms':max(v['p95_ms'] for v in values),
            'max_peak_allocated_bytes':max(v['peak_allocated_bytes'] for v in values),'input_shapes':shapes}
        print(json.dumps({'case':name,**{k:result['cases'][name][k] for k in ('gflops','max_clip_median_ms','max_clip_p95_ms','max_peak_allocated_bytes')}}),flush=True)
        target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');del model,payload;gc.collect();torch.cuda.empty_cache()
    result['completed']=True;target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
