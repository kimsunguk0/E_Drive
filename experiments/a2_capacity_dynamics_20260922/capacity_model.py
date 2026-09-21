"""Three isolated A2 changes on one completed DEV parent, with explicit export."""
from pathlib import Path
import copy, hashlib, json, sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'experiments/a2_native_detail_20260921'):
    sys.path.insert(0,str(p))
from native_model import NativeDetailModel,arm_config as native_spec
from models.motiondrive_v2.config import MotionDriveV2Config

ARMS=('C-CTRL','C-R101','C-DECSPLIT','C-AGENT')
SIGNATURE='capacity_dynamics_signature'
NEW_BLOCK_PREFIX=tuple(f'backbone_fpn.layer3.{i}.' for i in range(6,23))

def arm_config(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(schema=1,arm=arm,parent_native_execution_config=native_spec('M-NATIVE'),
        actual_backbone='resnet101_grown_3_4_23_3' if arm=='C-R101' else 'resnet50_3_4_6_3',
        initializer='copy all existing layers; append 17 copies of last stage3 block with bn3 affine zero' if arm=='C-R101' else 'completed M-NATIVE',
        split_queries_and_decoder=arm=='C-DECSPLIT',agent_future_auxiliary=arm=='C-AGENT',
        agent_output='dense6x2 current-ego displacement / 10m; GT sampling only in loss',
        input_boundary='existing A2 query-only status/goal; unchanged image-only motion/state',
        output='absolute_xy_6x2',new_raw_input=False)

def signature(spec):
    if spec!=arm_config(spec.get('arm')):raise ValueError('Unknown capacity/dynamics graph')
    return torch.tensor(list(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).digest()),dtype=torch.uint8)

class FullSplitPlanner(nn.Module):
    def __init__(self,parent):
        super().__init__();self.config=copy.deepcopy(parent.config)
        for name,module in parent.named_children():
            if name!='decoder':self.add_module(name,copy.deepcopy(module))
        for name,param in parent.named_parameters(recurse=False):
            if name!='waypoint_queries':self.register_parameter(name,nn.Parameter(param.detach().clone()))
        for name,value in parent.named_buffers(recurse=False):
            self.register_buffer(name,value.detach().clone(),persistent=name not in parent._non_persistent_buffers_set)
        for branch in ('length','heading'):
            self.register_parameter(branch+'_queries',nn.Parameter(parent.waypoint_queries.detach().clone()))
            self.add_module(branch+'_decoder',copy.deepcopy(parent.decoder))

    def forward(self,scene_features,motion_features,predicted_state,predicted_history,motion_pair_features):
        from progress_model import compose_progress
        b=scene_features.shape[0]
        with torch.autocast(device_type=scene_features.device.type,enabled=False):
            scene=scene_features.float()+self.scene_position(self.scene_xy.float())[None]
            motion=motion_features.float()+self.motion_type.float()
            numeric=torch.cat([predicted_state.float()/self.state_scale,
                               (predicted_history.float()/self.history_scale).flatten(1)],-1)
            if not self.config.state_on:numeric=torch.zeros_like(numeric)
            memory=torch.cat([scene,motion,self.state_projection(numeric)[:,None]],1)
            raw=[]
            for branch in ('length','heading'):
                q=getattr(self,branch+'_queries').float()[None].expand(b,-1,-1)
                d=getattr(self,branch+'_decoder')(q,memory)
                read=getattr(self,branch+'_read')(d,motion_pair_features)
                raw.append(getattr(self,branch+'_head')(d+read))
            return compose_progress(torch.cat(raw,-1),self.progress_units)

class CapacityModel(NativeDetailModel):
    def __init__(self,config,*,execution_config,include_training_aux=True):
        signature(execution_config)
        super().__init__(config,execution_config=execution_config['parent_native_execution_config'])
        self.capacity_execution_config=copy.deepcopy(execution_config)
        self.register_buffer(SIGNATURE,signature(execution_config))
        arm=execution_config['arm']
        if arm=='C-R101':
            assert len(self.backbone_fpn.layer3)==6
            for i in range(6,23):
                block=copy.deepcopy(self.backbone_fpn.layer3[-1])
                nn.init.zeros_(block.bn3.weight);nn.init.zeros_(block.bn3.bias)
                self.backbone_fpn.layer3.add_module(str(i),block)
            self.backbone_fpn.arch='resnet101_grown'
        if arm=='C-DECSPLIT':self.planner=FullSplitPlanner(self.planner)
        if arm=='C-AGENT' and include_training_aux:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(2026092201)
                self.agent_future_head=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GELU(),nn.Conv2d(128,12,1))
                nn.init.zeros_(self.agent_future_head[-1].weight);nn.init.zeros_(self.agent_future_head[-1].bias)

    def forward(self,*args,**kwargs):
        out=super().forward(*args,**kwargs)
        if self.training and self.capacity_execution_config['agent_future_auxiliary']:
            if not hasattr(self,'agent_future_head'):raise ValueError('Inference-only export cannot train')
            # No label, target center, raw pose/status or goal enters this head.
            raster=out['scene_features'].transpose(1,2).reshape(-1,128,64,48)
            out['agent_future_field']=self.agent_future_head(raster).float()
        return out

def load_parent(model,state):
    target=model.state_dict();mapped={};arm=model.capacity_execution_config['arm']
    for k,v in state.items():
        if arm=='C-DECSPLIT' and k.startswith('planner.decoder.'):
            for b in ('length','heading'):mapped[k.replace('planner.decoder.',f'planner.{b}_decoder.',1)]=v
        elif arm=='C-DECSPLIT' and k=='planner.waypoint_queries':
            for b in ('length','heading'):mapped[f'planner.{b}_queries']=v
        else:mapped[k]=v
    if arm=='C-R101':
        # Use the actual learned parent block, not constructor-random weights.
        for k in target:
            if k.startswith(NEW_BLOCK_PREFIX):
                suffix=k.split('.',3)[3];src='backbone_fpn.layer3.5.'+suffix
                mapped[k]=torch.zeros_like(state[src]) if suffix in ('bn3.weight','bn3.bias') else state[src].clone()
    extras=set(target)-set(mapped)
    expected={SIGNATURE}|{k for k in target if k.startswith('agent_future_head.')}
    assert extras==expected and not(set(mapped)-set(target)),(extras,expected)
    for k,v in mapped.items():assert target[k].shape==v.shape,k
    target.update(mapped);model.load_state_dict(target,strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in mapped.items())
    return {'mapped_parent_tensors_equal':True,'mapped_count':len(mapped),'new_keys':sorted(extras)}

def manifest_for(model,parent):
    return dict(model_config=model.config.to_dict(),execution_config=model.execution_config,
        next_execution_config=model.next_execution_config,native_execution_config=model.native_execution_config,
        capacity_execution_config=model.capacity_execution_config,
        training_aux_in_state=hasattr(model,'agent_future_head'),split_sha256=parent['split_sha256'])

def load_export(path,device='cpu'):
    cp=torch.load(path,map_location='cpu',weights_only=False);m=cp['manifest'];spec=m['capacity_execution_config']
    if not torch.equal(cp['model'][SIGNATURE],signature(spec)):raise ValueError('Wrong inference graph signature')
    model=CapacityModel(MotionDriveV2Config(**m['model_config']),execution_config=spec,include_training_aux=m['training_aux_in_state'])
    model.load_state_dict(cp['model'],strict=True)
    return model.to(device).eval(),cp

def export_student(payload):
    out={k:copy.deepcopy(payload[k]) for k in ('manifest','step')}
    out['model']={k:v for k,v in payload['model'].items() if not k.startswith('agent_future_head.')}
    out['manifest']['training_aux_in_state']=False
    return out
