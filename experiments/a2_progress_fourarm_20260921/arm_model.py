"""Explicit, strict-loadable single-model graphs for the four-arm DEV screen."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'scripts', ROOT/'experiments/a2_progress_h4_20260920'):
    sys.path.insert(0, str(path))
from progress_model import H4ProgressModel, ProgressHeadingPlanner, PROGRESS, compose_progress
from models.motiondrive_v2.config import MotionDriveV2Config
from fine_motion import FineMotionEncoder

ARMS = ('P-CTRL', 'P-VECTOR', 'P-FINE', 'P-SHARED768')
SIGNATURE = 'execution_signature'

def arm_config(arm):
    if arm not in ARMS:
        raise ValueError('Unknown explicit arm: '+str(arm))
    fine = arm == 'P-FINE'
    return dict(schema_version=1, arm=arm,
        architecture_id='a2_h4_progress_fine_v1' if fine else 'a2_h4_progress_v1',
        fine_read_enabled=fine, fine_grid=[24,32] if fine else None,
        fine_read_position='after_coarse_read_before_progress_head' if fine else None,
        scene_history_feature_source='shared_raw_fpn768' if arm == 'P-SHARED768' else 'separate_low384',
        interval_auxiliary='vector' if arm == 'P-VECTOR' else 'length', auxiliary_lambda=.25,
        status_policy='five_RGB_consumed_pose_times_nominal_frame_dt',
        history_frame_offsets=[1,2,5,10], image_hw=[432,768], motion_hw=[432,768],
        precision='bf16_images_fp32_planner_output_loss', output='absolute_xy_6x2')

def validate_spec(spec):
    if not isinstance(spec,dict) or spec != arm_config(spec.get('arm')):
        raise ValueError('Missing, inconsistent or unsupported explicit execution configuration')
    return copy.deepcopy(spec)

def signature(spec):
    raw=json.dumps(validate_spec(spec),sort_keys=True,separators=(',',':')).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()),dtype=torch.uint8)

class FineRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm=nn.LayerNorm(128)
        self.memory_norm=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        gy,gx=torch.meshgrid(torch.linspace(-1,1,24),torch.linspace(-1,1,32),indexing='ij')
        self.register_buffer('positions',torch.stack((gx,gy),-1).reshape(-1,2))

    def forward(self,decoded,memory):
        if decoded.shape[1:]!=(6,128) or memory.shape!=(decoded.shape[0],3072,128):
            raise ValueError('Wrong FINE query/memory shape')
        with torch.autocast(device_type=decoded.device.type,enabled=False):
            q=self.query_norm(decoded.float());m=self.memory_norm(memory.float())
            return self.attention(q,m,m,need_weights=False)[0]

class FineProgressPlanner(ProgressHeadingPlanner):
    def __init__(self,config):
        super().__init__(config)
        self.fine_read=FineRead()

    def forward(self,scene_features,motion_features,predicted_state,predicted_history,
                motion_pair_features,fine_memory):
        b=scene_features.shape[0]
        with torch.autocast(device_type=scene_features.device.type,enabled=False):
            scene=scene_features.float()+self.scene_position(self.scene_xy.float())[None]
            motion=motion_features.float()+self.motion_type.float()
            state=predicted_state.float()/self.state_scale
            history=(predicted_history.float()/self.history_scale).flatten(1)
            numeric=torch.cat([state,history],-1)
            if not self.config.state_on:numeric=torch.zeros_like(numeric)
            memory=torch.cat([scene,motion,self.state_projection(numeric)[:,None]],1)
            queries=self.waypoint_queries.float()[None].expand(b,-1,-1)
            decoded=self.decoder(queries,memory)
            decoded=decoded+self.temporal_read(decoded,motion_pair_features)
            decoded=decoded+self.fine_read(decoded,fine_memory)
            return compose_progress(self.xy_head(decoded.float()),self.progress_units)

class FourArmModel(H4ProgressModel):
    def __init__(self,config,*,execution_config):
        spec=validate_spec(execution_config)
        super().__init__(config,arm=PROGRESS)
        self.execution_config=spec
        self.register_buffer(SIGNATURE,signature(spec))
        if spec['fine_read_enabled']:
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(20260921)
                motion=FineMotionEncoder(config)
                planner=FineProgressPlanner(config)
            motion.load_state_dict(self.motion_encoder.state_dict(),strict=True)
            merged=planner.state_dict();merged.update(self.planner.state_dict())
            planner.load_state_dict(merged,strict=True)
            self.motion_encoder,self.planner=motion,planner
            assert sum(p.numel() for p in self.planner.fine_read.parameters())==66560

    def forward_parts(self,images,history_images,lidar2img,history_transforms,
                      time_offsets,goal_xy,motion_current=None,motion_history=None):
        if self.execution_config['scene_history_feature_source']=='separate_low384':
            return super().forward_parts(images,history_images,lidar2img,history_transforms,
                                         time_offsets,goal_xy,motion_current,motion_history)
        if motion_current is None or motion_history is None:
            raise ValueError('SHARED768 requires the unchanged raw native motion images')
        b,t=time_offsets.shape
        current,p4=self.backbone_fpn(images.flatten(0,1))
        current=tuple(v.reshape(b,6,*v.shape[1:]) for v in current)
        p4=p4.reshape(b,6,*p4.shape[1:])
        levels,_=self.backbone_fpn(torch.cat((motion_current[:,None],motion_history),1).flatten(0,1))
        levels=tuple(v.reshape(b,t+1,*v.shape[1:]) for v in levels)
        history=tuple(v[:,1:] for v in levels)
        motion=self.motion_encoder(tuple(v[:,0] for v in levels),history,time_offsets)
        scene=self.scene_encoder(current,history,p4,lidar2img,history_transforms,
                                 time_offsets,goal_xy,images.shape[-2:])
        return {**scene,**motion}

    def forward(self,images,history_images,lidar2img,history_transforms,time_offsets,
                goal_xy,motion_current=None,motion_history=None,provided_status5=None):
        if provided_status5 is None:raise ValueError('Missing A2 scene-query status')
        if self._provided_status_context is not None or self._decoded_capture:
            raise RuntimeError('Reentrant or stale A2 forward')
        self._provided_status_context=provided_status5
        try:
            parts=self.forward_parts(images,history_images,lidar2img,history_transforms,
                                     time_offsets,goal_xy,motion_current,motion_history)
            args=[parts[k] for k in ('scene_features','motion_features','state_hat',
                                     'history_hat','motion_pair_features')]
            if self.execution_config['fine_read_enabled']:
                fused=parts.pop('motion_pair_map');b,t=time_offsets.shape
                if fused.shape!=(b*t,128,54,96):raise ValueError('FINE must read the native pre-pool map')
                with torch.autocast(device_type=fused.device.type,enabled=False):
                    fine=F.adaptive_avg_pool2d(fused,(24,32)).flatten(2).transpose(1,2).reshape(b,t,768,128)
                    dt=time_offsets.clamp_min(1e-3)
                    time_feature=self.motion_encoder.time_embed(torch.stack((dt,dt.log()),-1))
                    fine=fine+self.motion_encoder.position(self.planner.fine_read.positions)[None,None]+time_feature[:,:,None]
                args.append(fine.flatten(1,2))
            return {**parts,'plan_abs':self.planner(*args)}
        finally:
            self._provided_status_context=None
            self._decoded_capture.clear()

def load_dev_parent(model,state):
    merged=model.state_dict()
    extras={k for k in merged if k==SIGNATURE or k.startswith('planner.fine_read.')}
    if set(merged)-extras!=set(state):raise ValueError('Unexpected DEV parent keys')
    for k,v in state.items():
        if merged[k].shape!=v.shape:raise ValueError('Parent tensor shape differs: '+k)
    merged.update(state);model.load_state_dict(merged,strict=True)
    if not all(torch.equal(model.state_dict()[k],v) for k,v in state.items()):
        raise ValueError('A parent tensor changed during loading')
    return sorted(extras)

def load_export(path,device='cpu'):
    """Reconstruct from explicit config, reject even same-key wrong feature sources."""
    payload=torch.load(path,map_location='cpu',weights_only=False)
    manifest=payload['manifest'];spec=validate_spec(manifest['execution_config'])
    actual=payload['model'].get(SIGNATURE)
    if actual is None or not torch.equal(actual.cpu(),signature(spec)):
        raise ValueError('Checkpoint/config signature mismatch')
    model=FourArmModel(MotionDriveV2Config(**manifest['model_config']),execution_config=spec)
    # The parent uses radius4. Its declared config already retains that radius.
    if model.config.correlation_radius!=4:raise ValueError('Unexpected correlation geometry')
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload
