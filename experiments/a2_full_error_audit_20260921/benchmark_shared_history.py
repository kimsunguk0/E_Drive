"""Cost/parity prototype: reuse unconditioned 768 front-history FPN in scene.

Parity is against the NEW high-history computation, not submitted baseline.
No learning or accuracy claim; status stays downstream in scene query only.
"""
from pathlib import Path
import argparse,json,sys,os,types,datetime
import torch
ROOT=Path(os.environ['ADCL_CODE_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_inputs import load_model,configure,prepare_clip,sha,forward,measure,flops

def shared(self,images,history_images,lidar2img,history_transforms,time_offsets,goal_xy,motion_current=None,motion_history=None):
    b,t=time_offsets.shape
    current,current_p4=self.backbone_fpn(images.flatten(0,1))
    current=tuple(v.reshape(b,6,*v.shape[1:]) for v in current)
    current_p4=current_p4.reshape(b,6,*current_p4.shape[1:])
    levels,_=self.backbone_fpn(torch.cat((motion_current[:,None],motion_history),1).flatten(0,1))
    levels=tuple(v.reshape(b,t+1,*v.shape[1:]) for v in levels)
    history=tuple(v[:,1:] for v in levels)
    motion=self.motion_encoder(tuple(v[:,0] for v in levels),history,time_offsets)
    scene=self.scene_encoder(current,history,current_p4,lidar2img,history_transforms,time_offsets,goal_xy,images.shape[-2:])
    return {**scene,**motion}

def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--clips-root',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    out=Path(args.output);assert not out.exists();configure();model,payload=load_model(args.checkpoint,True)
    original=model.forward_parts;values=[];checks=[]
    for clip in sorted(Path(args.clips_root).iterdir()):
        if not clip.is_dir():continue
        p=prepare_clip(clip);x={k:v.cuda() for k,v in p.inputs.items()};x['history_images']=x['motion_history']
        assert torch.equal(x['images'][:,0],x['motion_current'])
        model.forward_parts=original
        with torch.no_grad():ref=model(**x)
        reference={k:ref[k].clone() for k in ('plan_abs','scene_features','motion_features','state_hat','history_hat')};del ref
        model.forward_parts=types.MethodType(shared,model)
        with torch.no_grad():actual=model(**x)
        deltas={k:float((reference[k]-actual[k]).abs().max()) for k in reference}
        assert max(deltas.values())<1e-6,deltas
        del actual,reference
        checks.append({'clip':clip.name,'FP32_max_abs_delta_vs_separate_HIGH_HISTORY':deltas})
        values.append({'clip':clip.name,**measure(model,x)})
        if len(values)==1:counted=flops(model,x)
        del x
    result={'scope':'Cost/parity only, new 768 scene-history prototype; not accuracy or submitted-baseline parity',
        'checkpoint_sha256':sha(args.checkpoint),'script_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'unconditioned_FPN_shared_before_scene_status_query':True,'motion_state_history_same_as_original_input_path':True,
        'gflops':counted/1e9,'timing':values,'max_clip_median_ms':max(v['median_ms'] for v in values),
        'max_clip_p95_ms':max(v['p95_ms'] for v in values),'parity':checks,
        'precision':'B1 BF16 with FP32 planner; FP32 parity','timing_scope':'whole forward, preprocessing excluded',
        'accuracy_not_evaluated':True,'completed':True}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
