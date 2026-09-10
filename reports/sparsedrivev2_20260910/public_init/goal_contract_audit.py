"""Read-only CPU audit of the already-provided +50 goal input contract."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

BASE=Path('/NHNHOME/data/sukim/adcl')
WT=BASE/'experiment_worktrees/sparsedrivev2_20260910'
PUBLIC=WT/'reports/sparsedrivev2_20260910/public_init'
MANIFEST=WT/'reports/sparsedrivev2_20260910/split_audit/primary_manifest.json'
META=Path('/tmp/pm97/data/etri/meta_train')
EGO=Path('/tmp/pm97/data/etri/ego_cache.npz')
sys.path.insert(0,str(BASE))
from models.motiondrive_v2_inputs import pose_geometry

manifest=json.loads(MANIFEST.read_text())
with np.load(EGO,allow_pickle=False) as z:
    names=z['scenarios'].astype(str)[z['scen_idx']]
    mask=np.isin(names,manifest['splits']['tune'])&(z['frame']>=30)&(z['frame']%5==0)
    rows=np.flatnonzero(mask);scenes=names[rows];frames=z['frame'][rows];goals=z['goal'][rows]
errors=[];fixture=None
for scene in sorted(manifest['splits']['tune']):
    root=META/scene
    ts=pd.read_parquet(root/'meta/timestamps.parquet')
    ep=pd.read_parquet(root/'annotation/ego_pose.parquet')
    table=ts[['timestamp','frame_id']].merge(ep,on='timestamp',how='left',validate='one_to_one').sort_values('frame_id')
    xyz=table[['x','y','z']].to_numpy(np.float64)
    rpy=table[['roll','pitch','yaw']].to_numpy(np.float64)
    rot=Rotation.from_euler('xyz',rpy).as_matrix()
    lookup={int(f):i for i,f in enumerate(table['frame_id'])}
    for j in np.flatnonzero(scenes==scene):
        now=lookup[int(frames[j])];future=lookup[int(frames[j])+50]
        expected=((xyz[future]-xyz[now])@rot[now])[:2]
        errors.append(float(np.abs(expected-goals[j]).max()))
        if fixture is None:
            records=[]
            for rel in list(range(-30,1))+[50]:
                idx=lookup[int(frames[j])+rel]
                r=dict(zip(('x','y','z'),xyz[idx].tolist()))
                r.update(dict(zip(('roll','pitch','yaw'),rpy[idx].tolist())))
                r['frame']=rel
                if rel==50:r.update(roll=float('nan'),pitch=float('nan'),yaw=float('nan'))
                records.append(r)
            _,adapted=pose_geometry(records)
            assert np.max(np.abs(adapted-goals[j]))<2e-5
            fixture={'row':int(rows[j]),'scene':scene,'frame':int(frames[j]),
                'raw_test_shaped_adapter_goal_max_abs_error':float(np.max(np.abs(adapted-goals[j]))),
                'future_orientation_nan_is_unused':True,
                'input_frames':list(range(-30,1))+[50]}
assert len(rows)==1998 and max(errors)<2e-5
prior=json.loads((PUBLIC/'geometry_plane_audit.json').read_text())
report={'passed':True,'device':'cpu',
 'train':{'rows':prior['train_rows'],'scenes':prior['train_scenes'],
          'max_abs_metres':prior['cached_goal_vs_raw_plus50_xyz_max_abs_metres']},
 'tune':{'rows':len(rows),'scenes':len(set(scenes.tolist())),'max_abs_metres':max(errors)},
 'fixture':fixture,
 'formula':'goal_xy = (R_current.T @ (XYZ_frame_plus_50 - XYZ_current))[:2]',
 'needed_data':['current XYZ','current roll/pitch/yaw','provided future +50 XYZ'],
 'not_used':['future orientation','intermediate future XYZ','future status','new trajectory generation'],
 'same_as_raw_serving_adapter':str(BASE/'models/motiondrive_v2_inputs.py'),
 'raw_serving_adapter_sha256':hashlib.sha256((BASE/'models/motiondrive_v2_inputs.py').read_bytes()).hexdigest(),
 'ego_cache_sha256':hashlib.sha256(EGO.read_bytes()).hexdigest(),
 'split_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
 'scope':'Checks equivalence of cached training/tune goal to the already-provided +50 input semantics; does not assert blanket official approval of a model.'}
(PUBLIC/'goal_contract_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
