"""Three explicit single-model graphs, with strict student-only export."""
from pathlib import Path
import copy,hashlib,json,sys
import torch

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'experiments/a2_progress_fourarm_20260921',ROOT/'experiments/a2_design_preflight_20260921'):
    sys.path.insert(0,str(p))
from arm_model import FourArmModel,arm_config as old_config
from design import SplitReadPlanner,LaneGeometryHead
from models.motiondrive_v2.config import MotionDriveV2Config

ARMS=('P-CTRL-NEXT','P-SPLITREAD','P-LANE-GEOM')
SIGNATURE='next_execution_signature'

def arm_config(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {'schema_version':1,'arm':arm,'base_execution_config':old_config('P-CTRL'),
        'split_read':arm=='P-SPLITREAD','shared_decoder':True,'decoded_skip_retained':True,
        'lane_geometry_auxiliary':arm=='P-LANE-GEOM','lane_geometry_in_planner':False,
        'extra_status_pose_goal_inputs':False,'interval_auxiliary':'length','auxiliary_lambda':.25,
        'output':'absolute_xy_6x2'}

def signature(spec):
    if spec!=arm_config(spec.get('arm')):raise ValueError('Unknown execution spec')
    return torch.tensor(list(hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).digest()),dtype=torch.uint8)

class NextModel(FourArmModel):
    def __init__(self,config,*,execution_config,include_training_aux=True):
        signature(execution_config)
        super().__init__(config,execution_config=execution_config['base_execution_config'])
        self.next_execution_config=copy.deepcopy(execution_config)
        self.register_buffer(SIGNATURE,signature(execution_config))
        if execution_config['split_read']:
            self.planner=SplitReadPlanner(self.planner)
        if execution_config['lane_geometry_auxiliary'] and include_training_aux:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(202609212)
                self.lane_geometry_head=LaneGeometryHead()

    def forward(self,*args,**kwargs):
        output=super().forward(*args,**kwargs)
        if self.training and self.next_execution_config['lane_geometry_auxiliary']:
            if not hasattr(self,'lane_geometry_head'):raise RuntimeError('Inference-only student cannot train')
            raster=output['scene_features'].transpose(1,2).reshape(-1,128,64,48)
            output['lane_geometry_prediction']=self.lane_geometry_head(raster)
        return output

def load_parent(model,state):
    merged=model.state_dict();mapped={};seen=set()
    for name,value in state.items():
        if model.next_execution_config['split_read'] and name.startswith('planner.temporal_read.'):
            for branch in ('length','heading'):
                mapped[name.replace('planner.temporal_read.','planner.'+branch+'_read.',1)]=value
        elif model.next_execution_config['split_read'] and name.startswith('planner.xy_head.'):
            for channel,branch in enumerate(('length','heading')):
                mapped[name.replace('planner.xy_head.','planner.'+branch+'_head.',1)]=value[channel:channel+1] if name.startswith('planner.xy_head.3.') else value
        else:mapped[name]=value
        seen.add(name)
    extras=set(merged)-set(mapped)
    expected={SIGNATURE}|{k for k in merged if k.startswith('lane_geometry_head.')}
    if extras!=expected or set(mapped)-set(merged):raise ValueError('Parent mapping mismatch')
    for name,value in mapped.items():
        if merged[name].shape!=value.shape:raise ValueError('Shape changed: '+name)
    merged.update(mapped);model.load_state_dict(merged,strict=True)
    actual=model.state_dict()
    assert all(torch.equal(actual[k],v) for k,v in mapped.items())
    return {'mapped_parent_tensors':len(mapped),'all_mapped_tensors_equal':True,'new_keys':sorted(extras)}

def manifest_for(model,parent_manifest):
    return {'model_config':model.config.to_dict(),'execution_config':model.execution_config,
        'next_execution_config':model.next_execution_config,
        'training_aux_in_state':hasattr(model,'lane_geometry_head'),
        'split_sha256':parent_manifest['split_sha256']}

def load_export(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    manifest=payload['manifest'];spec=manifest['next_execution_config']
    if not torch.equal(payload['model'][SIGNATURE].cpu(),signature(spec)):
        raise ValueError('Wrong new execution configuration')
    model=NextModel(MotionDriveV2Config(**manifest['model_config']),execution_config=spec,
        include_training_aux=manifest['training_aux_in_state'])
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload

def export_student(payload):
    out={k:copy.deepcopy(payload[k]) for k in ('manifest','step')}
    out['model']={k:v for k,v in payload['model'].items() if not k.startswith('lane_geometry_head.')}
    out['manifest']['training_aux_in_state']=False
    return out
