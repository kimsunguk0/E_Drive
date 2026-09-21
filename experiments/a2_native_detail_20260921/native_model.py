"""Native RGB detail: function-preserving additions to the frozen SplitRead graph contract."""
from pathlib import Path
import copy,hashlib,json,sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'experiments/a2_splitread_lanegeom_20260921',ROOT/'experiments/a2_temporal_read_20260920'):
    sys.path.insert(0,str(p))
from next_model import NextModel,arm_config as parent_spec
from models.motiondrive_v2.config import MotionDriveV2Config
from models.motiondrive_v2.motion_encoder import local_correlation

ARMS=('M-LOW','M-NATIVE','S-LOW','S-NATIVE')
SIGNATURE='native_detail_signature'
PREFIX=('native_motion.','native_projection.')

def arm_config(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {'schema':1,'arm':arm,'family':'motion' if arm.startswith('M-') else 'scene',
            'detail':'native' if arm.endswith('NATIVE') else 'low',
            'parent_next_execution_config':parent_spec('P-SPLITREAD'),
            'baseline_preserved':True,'high_wh':[1152,648],'low_bottleneck_wh':[768,432],
            'scene_projection_basis':'existing 768 pixel calibration -> normalized same-FOV coordinates',
            'motion_radius':6,'merge':'prepool pair-map' if arm.startswith('M-') else 'shared raster before all consumers',
            'zero_output_projection':True,'new_raw_status_pose_goal_route':False,'output':'absolute_xy_6x2'}

def signature(spec):
    if spec!=arm_config(spec.get('arm')):raise ValueError('Unknown native detail contract')
    return torch.tensor(list(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).digest()),dtype=torch.uint8)

class NativePairMap(nn.Module):
    def __init__(self,channels=128,cm=32):
        super().__init__();self.radius=6
        self.projections=nn.ModuleList([nn.Conv2d(channels,cm,1,bias=False) for _ in range(2)])
        bins=(2*self.radius+1)**2
        self.correlation_fuse=nn.Sequential(nn.Conv2d(2*(bins+2*cm),channels,3,padding=1),
            nn.GroupNorm(1,channels),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(1,channels),nn.GELU())

    def forward(self,levels):
        b,t1=levels[0].shape[:2];t=t1-1;ref_hw=levels[0].shape[-2:];features=[]
        for i,level in enumerate(levels):
            current,history=level[:,0],level[:,1:];h,w=history.shape[-2:]
            cur=self.projections[i](current)
            cur=cur[:,None].expand(-1,t,-1,-1,-1).reshape(b*t,-1,h,w)
            past=self.projections[i](history.flatten(0,1))
            with torch.autocast(device_type=cur.device.type,enabled=False):
                corr=local_correlation(cur,past,self.radius)
            pair=torch.cat((corr.to(cur.dtype),cur,past),1)
            features.append(F.interpolate(pair,ref_hw,mode='bilinear',align_corners=False))
        return self.correlation_fuse(torch.cat(features,1))

class NativeDetailModel(NextModel):
    def __init__(self,config,*,execution_config):
        signature(execution_config)
        super().__init__(config,execution_config=execution_config['parent_next_execution_config'],include_training_aux=False)
        self.native_execution_config=copy.deepcopy(execution_config)
        self.register_buffer(SIGNATURE,signature(execution_config))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(202609213)
            if execution_config['family']=='motion':self.native_motion=NativePairMap(config.channels,config.correlation_channels)
            self.native_projection=nn.Conv2d(config.channels,config.channels,1,bias=False)
            nn.init.zeros_(self.native_projection.weight)
        self._high_context=None
        self.last_detail_rms=None

    def forward(self,*args,high_images=None,**kwargs):
        if high_images is None or self._high_context is not None:raise ValueError('Missing/reentrant detail images')
        count=5 if self.native_execution_config['family']=='motion' else 6
        if high_images.ndim!=5 or high_images.shape[1:]!=(count,3,648,1152):raise ValueError('Wrong detail image shape')
        self._high_context=high_images
        try:return super().forward(*args,**kwargs)
        finally:self._high_context=None

    def forward_parts(self,images,history_images,lidar2img,history_transforms,time_offsets,goal_xy,
                      motion_current=None,motion_history=None):
        high=self._high_context
        if high is None:raise RuntimeError('Detail branch must enter through forward')
        b,n=high.shape[:2]
        levels,p4=self.backbone_fpn(high.flatten(0,1))
        levels=tuple(v.reshape(b,n,*v.shape[1:]) for v in levels)
        if self.native_execution_config['family']=='motion':
            native=self.native_motion(levels)
            def add_detail(module,args,base):
                aligned=F.interpolate(native,base.shape[-2:],mode='bilinear',align_corners=False)
                delta=self.native_projection(aligned)
                return base+delta.to(base.dtype)
            handle=self.motion_encoder.correlation_fuse.register_forward_hook(add_detail)
            try:
                return super().forward_parts(images,history_images,lidar2img,history_transforms,time_offsets,goal_xy,
                                              motion_current,motion_history)
            finally:handle.remove()
        # Capture the actual unchanged low-resolution scene history, without a
        # second history backbone pass or any scene -> motion feedback path.
        captures=[]
        handle=self.scene_encoder.register_forward_pre_hook(lambda module,args:captures.append(args))
        try:
            parts=super().forward_parts(images,history_images,lidar2img,history_transforms,time_offsets,goal_xy,
                                       motion_current,motion_history)
        finally:handle.remove()
        if len(captures)!=1:raise RuntimeError('Expected one baseline scene call')
        old=captures.pop();p4=p4.reshape(b,n,*p4.shape[1:])
        # Normalized coordinates describe the same cropped FOV at every feature
        # resolution, just as in the existing low-resolution history sampler.
        extra=self.scene_encoder(levels,old[1],p4,*old[3:])
        hi_raster=extra['scene_features'].transpose(1,2).reshape(b,self.config.channels,*self.config.grid_size)
        base=parts['scene_features'].transpose(1,2).reshape_as(hi_raster)
        raster=base+self.native_projection(hi_raster).to(base.dtype)
        parts['scene_features']=raster.flatten(2).transpose(1,2)
        parts['occ_logits']=self.scene_encoder.occ_head(raster).float()
        parts['lane_logits']=self.scene_encoder.lane_head(raster).float()
        return parts

def load_parent(model,state):
    merged=model.state_dict();extras={k for k in merged if k==SIGNATURE or k.startswith(PREFIX)}
    if set(merged)-extras!=set(state):raise ValueError('Unexpected parent keys')
    for k,v in state.items():
        if merged[k].shape!=v.shape:raise ValueError('Parent tensor shape mismatch: '+k)
        merged[k]=v
    # Reuse meaningful visual descriptors and all shape-compatible fuse layers.
    # The radius6 first convolution is a fresh layer, identical in matched arms.
    for k in sorted(extras):
        if k.startswith('native_motion.'):
            src=k.replace('native_motion.','motion_encoder.',1)
            if src in state and state[src].shape==merged[k].shape:merged[k]=state[src].clone()
    model.load_state_dict(merged,strict=True)
    if not all(torch.equal(model.state_dict()[k],v) for k,v in state.items()):raise ValueError('Parent tensor changed')
    if model.native_projection.weight.count_nonzero():raise ValueError('Nonzero initial detail projection')
    return sorted(extras)

def manifest_for(model,parent_manifest):
    return dict(model_config=model.config.to_dict(),execution_config=model.execution_config,
        next_execution_config=model.next_execution_config,native_execution_config=model.native_execution_config,
        training_aux_in_state=False,split_sha256=parent_manifest['split_sha256'])

def load_export(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False);m=payload['manifest'];spec=m['native_execution_config']
    if not torch.equal(payload['model'][SIGNATURE].cpu(),signature(spec)):raise ValueError('Wrong detail graph/config')
    model=NativeDetailModel(MotionDriveV2Config(**m['model_config']),execution_config=spec)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload
