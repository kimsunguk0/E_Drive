"""Matched image-based planners: absolute XY or six positive lengths/headings."""
from pathlib import Path
import math,sys
import torch
from torch.nn import functional as F

ROOT=Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0,str(ROOT/'experiments/a2_temporal_read_20260920'))
from temporal_model import TemporalReadModel,TemporalReadPlanner,ARM as TEMPORAL_ARM,load_fresh_parent

DIRECT='A2-H4-DIRECT'
PROGRESS='A2-H4-PROGRESS'
ARMS=(DIRECT,PROGRESS)
LENGTH_SCALE=5.0 # meters per learned normalized interval length
INITIAL_LENGTH=.25 # neural head zero input gives a quarter-meter interval
LENGTH_BIAS=math.log(math.expm1(INITIAL_LENGTH/LENGTH_SCALE))
TAG='planner.progress_units'

def compose_progress(raw,units):
    """Two neural outputs per interval; no state extrapolation or postprocess."""
    with torch.autocast(device_type=raw.device.type,enabled=False):
        raw=raw.float();units=units.float()
        length=F.softplus(raw[...,0]+units[2])*units[0]
        heading=raw[...,1]*units[1] # unbounded angle; permits reversing and full turns
        direction=torch.stack([heading.cos(),heading.sin()],-1)
        return (length[...,None]*direction).cumsum(1)

class ProgressHeadingPlanner(TemporalReadPlanner):
    def __init__(self,config):
        super().__init__(config)
        # Persistent difference ensures strict load catches wrong inference graph.
        self.register_buffer('progress_units',torch.tensor([LENGTH_SCALE,1.,LENGTH_BIAS],dtype=torch.float32))
    def forward(self,scene_features,motion_features,predicted_state,predicted_history,motion_pair_features):
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
            raw=self.xy_head(decoded.float())
            # Return absolute XY; serving must NOT cumsum a second time.
            return compose_progress(raw,self.progress_units)

class H4ProgressModel(TemporalReadModel):
    VALID_ARMS={**TemporalReadModel.VALID_ARMS,DIRECT:0,PROGRESS:0}
    def __init__(self,config,*,arm):
        if arm not in ARMS:raise ValueError(arm)
        super().__init__(config,arm=TEMPORAL_ARM);self.h4_arm=arm
        if arm==PROGRESS:
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(2026092021);planner=ProgressHeadingPlanner(config)
            state=planner.state_dict();state.update(self.planner.state_dict());planner.load_state_dict(state,strict=True);self.planner=planner

def load_initializer(model,parent):
    state=model.state_dict()
    extras={k for k in state if k.startswith('planner.temporal_read.') or k==TAG}
    if set(state)-extras!=set(parent):raise ValueError('Unexpected parent/child state keys')
    state.update(parent);model.load_state_dict(state,strict=True)
    if not all(torch.equal(model.state_dict()[k],v) for k,v in parent.items()):raise ValueError('Parent tensors differ')
    return sorted(extras)
