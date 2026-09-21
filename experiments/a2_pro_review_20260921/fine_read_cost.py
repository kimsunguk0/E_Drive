"""Untrained FINE-READ review prototype: parity, gradient and RTX4090 cost.

This is not a production model/factory and has no training or submission path.
Uses the already-computed pre-pool visual correlation map exactly once.
"""
from pathlib import Path
import argparse,sys,os,json,datetime,hashlib
import torch
from torch import nn
from torch.nn import functional as F

class FineRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm=nn.LayerNorm(128);self.memory_norm=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
        gy,gx=torch.meshgrid(torch.linspace(-1,1,24),torch.linspace(-1,1,32),indexing='ij')
        self.register_buffer('positions',torch.stack((gx,gy),-1).reshape(-1,2))
    def forward(self,decoded,memory):
        assert decoded.shape[1:]==(6,128) and memory.shape[1:]==(3072,128)
        with torch.autocast(device_type=decoded.device.type,enabled=False):
            q=self.query_norm(decoded.float());m=self.memory_norm(memory.float())
            y,_=self.attention(q,m,m,need_weights=False)
            return y

def synthetic_gradient_check():
    torch.manual_seed(21);head=FineRead();query=torch.randn(1,6,128)
    values=[]
    for stage in (0,1):
        head.zero_grad(set_to_none=True)
        if stage:
            with torch.no_grad():head.attention.out_proj.weight.copy_(torch.eye(128)*.001)
        memory=torch.randn(1,3072,128,requires_grad=True)
        y=query+head(query,memory);loss=y.square().mean();loss.backward()
        values.append({'stage':stage,'output_weight_grad_norm':float(head.attention.out_proj.weight.grad.norm()),
            'attention_in_proj_grad_norm':float(head.attention.in_proj_weight.grad.norm()),
            'memory_grad_norm':float(memory.grad.norm()),'max_feature_change':float((y-query).detach().abs().max())})
    assert values[0]['output_weight_grad_norm']>0 and values[0]['memory_grad_norm']==0 and values[0]['max_feature_change']==0
    assert values[1]['attention_in_proj_grad_norm']>0 and values[1]['memory_grad_norm']>0
    return values

def main():
    a=argparse.ArgumentParser();a.add_argument('--helper-dir',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--clips-root',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    sys.path.insert(0,args.helper_dir)
    from benchmark_inputs import configure,load_model,prepare_clip,measure,flops,sha
    out=Path(args.output);assert not out.exists();configure();assert 'RTX 4090' in torch.cuda.get_device_name(0)
    gradient=synthetic_gradient_check();model,payload=load_model(args.checkpoint,True)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20260921);model.planner.add_module('fine_read_review',FineRead().cuda())
    capture={};enabled=[False];input_time=[None];shapes=[]
    def save_map(module,args,output):
        if not enabled[0]:return
        assert 'map' not in capture;capture['map']=output
    def add_fine(module,args):
        if not enabled[0]:return None
        fused=capture.pop('map');b,t=input_time[0].shape;assert t==4
        fine=F.adaptive_avg_pool2d(fused,(24,32)).flatten(2).transpose(1,2).reshape(b,t,768,128)
        dt=input_time[0].clamp_min(1e-3);time_feature=model.motion_encoder.time_embed(torch.stack((dt,dt.log()),-1))
        fine=fine+model.motion_encoder.position(model.planner.fine_read_review.positions)[None,None]+time_feature[:,:,None]
        memory=fine.flatten(1,2)
        if not shapes:shapes.append({'pre_pool':list(fused.shape),'fine_memory':list(memory.shape),'decoded':list(args[0].shape)})
        return (args[0]+model.planner.fine_read_review(args[0],memory),)
    handles=[model.motion_encoder.correlation_fuse.register_forward_hook(save_map),model.planner.xy_head.register_forward_pre_hook(add_fine)]
    timing=[];control_timing=[];checks=[];flop_values={}
    try:
        for i,clip in enumerate(sorted(p for p in Path(args.clips_root).iterdir() if p.is_dir())):
            prepared=prepare_clip(clip);x={k:v.cuda() for k,v in prepared.inputs.items()};input_time[0]=x['time_offsets']
            for precision in ('fp32','bf16'):
                enabled[0]=False
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):ref=model(**x)
                enabled[0]=True
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):actual=model(**x)
                delta={k:float((ref[k].float()-actual[k].float()).abs().max()) for k in ('plan_abs','scene_features','motion_features','state_hat','history_hat')}
                assert max(delta.values())==0,delta
                checks.append({'clip':clip.name,'precision':precision,'max_abs_delta_vs_submitted':delta});del ref,actual
            if i==0:
                enabled[0]=False;flop_values['baseline']=flops(model,x);assert flop_values['baseline']==730044861120
                enabled[0]=True;flop_values['fine']=flops(model,x)
            # Same-session paired costs; alternate order over the two fixtures.
            for active in ((False,True) if i%2==0 else (True,False)):
                enabled[0]=active
                (timing if active else control_timing).append({'clip':clip.name,**measure(model,x)})
            assert not capture
            del x
    finally:
        for h in handles:h.remove()
    result={'scope':'UNTRAINED fine-read review prototype, no accuracy measurement or optimizer updates',
        'checkpoint_sha256':sha(args.checkpoint),'script_sha256':sha(__file__),'helper_sha256':sha(Path(args.helper_dir)/'benchmark_inputs.py'),
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'precision':'B1 BF16 with FP32 planner','timing_scope':'Whole forward including extra fine read, excludes preprocessing',
        'raw_input_contract_unchanged':True,'backbone_passes_unchanged':True,'fine_input':'pre-pool image correlation map + existing visual XY and frame-time embeddings, no supplied pose/status/goal',
        'queries':'Existing decoded interval queries after coarse temporal read; indirect scene goal/status influence remains',
        'inference_teacher_or_second_checkpoint':False,'runtime_shapes':shapes,'parity':checks,'synthetic_gradient':gradient,
        'extra_parameters':sum(p.numel() for p in model.planner.fine_read_review.parameters()),'flops':flop_values,
        'timing':timing,'paired_control_timing':control_timing,'timing_order':'baseline/fine on fixture0, fine/baseline on fixture1',
        'max_clip_median_ms':max(v['median_ms'] for v in timing),'max_clip_p95_ms':max(v['p95_ms'] for v in timing),'completed':True}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
