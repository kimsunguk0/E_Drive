"""CPU/read-only geometry audit; only the frozen train partition is analyzed."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

BASE = Path('/NHNHOME/data/sukim/adcl')
WT = BASE/'experiment_worktrees/sparsedrivev2_20260910'
OUT = WT/'reports/sparsedrivev2_20260910/public_init/geometry_plane_audit.json'
MANIFEST = WT/'reports/sparsedrivev2_20260910/split_audit/primary_manifest.json'
CAL = BASE/'data/etri/motiondrive_v2/train_tune_geometry_v2/calibration.npz'
META = Path('/tmp/pm97/data/etri/meta_train')
CAMS = ('camera_front_left','camera_front','camera_front_right')
INDICES = [2,0,1]
FIX_HEIGHT = np.array([0.,-.25,-.5,.25,.5])

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantile(x):
    x=np.asarray(x).reshape(-1)
    return dict(zip(('min','p50','p90','p95','p99','max'),np.quantile(x,[0,.5,.9,.95,.99,1]).tolist()))

manifest=json.loads(MANIFEST.read_text())
allowed=manifest['splits']['train']
with np.load('/tmp/pm97/data/etri/ego_cache.npz',allow_pickle=False) as z:
    names=z['scenarios'].astype(str)[z['scen_idx']]
    selected=np.isin(names,allowed)&(z['frame']>=30)
    rows=np.flatnonzero(selected)
    frames=z['frame'][rows]
    scenes=names[rows]
    gt=z['fut'][rows]
    cached_goal=z['goal'][rows]
with np.load(CAL,allow_pickle=False) as z: p=z['lidar2img'][INDICES].astype(np.float64)
p[:,0,:]*=512/768;p[:,1,:]*=256/432
all_xyz=np.zeros((len(rows),8,3),np.float64)
all_goals=np.zeros((len(rows),2),np.float64)
max_gt_error=0.
for scene in sorted(allowed):
    ids=np.flatnonzero(scenes==scene)
    root=META/scene
    ts=pd.read_parquet(root/'meta/timestamps.parquet')
    ep=pd.read_parquet(root/'annotation/ego_pose.parquet')
    joined=ts[['timestamp','frame_id']].merge(ep,on='timestamp',how='left',validate='one_to_one').sort_values('frame_id')
    xyz=joined[['x','y','z']].to_numpy(np.float64)
    rot=Rotation.from_euler('xyz',joined[['roll','pitch','yaw']].to_numpy(np.float64)).as_matrix()
    lookup={int(f):i for i,f in enumerate(joined['frame_id'])}
    now=np.array([lookup[int(f)] for f in frames[ids]])
    future=np.array([[lookup[int(f)+dt] for dt in range(5,41,5)] for f in frames[ids]])
    local=np.einsum('bti,bij->btj',xyz[future]-xyz[now,None],rot[now])
    all_xyz[ids]=local
    goal_index=np.array([lookup[int(f)+50] for f in frames[ids]])
    all_goals[ids]=np.einsum('bi,bij->bj',xyz[goal_index]-xyz[now],rot[now])[:,:2]
    max_gt_error=max(max_gt_error,float(np.abs(local[:,:6,:2]-gt[ids]).max()))
assert max_gt_error<2e-5
goal_error=float(np.max(np.abs(all_goals-cached_goal)))
assert goal_error<2e-5

def project(points):
    homogeneous=np.concatenate([points,np.ones((*points.shape[:-1],1))],-1)
    q=np.einsum('cij,btj->bcti',p,homogeneous)
    uv=q[...,:2]/q[...,2:3].clip(1e-5)
    visible=(q[...,2]>1e-5)&(uv[...,0]>0)&(uv[...,0]<512)&(uv[...,1]>0)&(uv[...,1]<256)
    return uv,visible

uv_true,vis_true=project(all_xyz)
plane=all_xyz.copy();plane[...,2]=0
uv_plane,vis_plane=project(plane)
valid=vis_true&vis_plane
pixel=np.linalg.norm(uv_true-uv_plane,axis=-1)
height_errors=[]
for height in FIX_HEIGHT:
    level=plane.copy();level[...,2]=height
    u,v=project(level)
    height_errors.append(np.where(v,np.linalg.norm(u-uv_true,axis=-1),np.inf))
best_error=np.min(height_errors,axis=0)

scene=sorted(allowed)[0]
cal_rows={r['camera_name']:r for r in pq.read_table(META/scene/'calibration/calibration.parquet').to_pylist()}
cache_meta=json.loads((BASE/'cache/etri_768/cache_meta.json').read_text())[scene]
reconstructed=[]
for camera in CAMS:
    r=cal_rows[camera];T=np.eye(4)
    T[:3,:3]=Rotation.from_euler('xyz',r['euler'],degrees=True).as_matrix()
    T[:3,3]=r['translation']
    K=np.eye(4);K[:3,:3]=cache_meta[camera]['K_cache']
    q=K@np.linalg.inv(T);q[0]*=512/768;q[1]*=256/432
    reconstructed.append(q)

report={
 'train_rows':len(rows),'train_scenes':len(allowed),'calibration_sha256':sha(CAL),
 'manifest_sha256':sha(MANIFEST),'raw_calibration_example':str(META/scene/'calibration/calibration.parquet'),
 'camera_to_ego_translation_metres':{c:cal_rows[c]['translation'] for c in CAMS},
 'camera_to_ego_euler_degrees':{c:cal_rows[c]['euler'] for c in CAMS},
 'reconstructed_projection_max_abs_error_at_512x256':float(np.max(np.abs(np.stack(reconstructed)-p))),
 'gt_xy_full_se3_parity_max_abs_metres':max_gt_error,
 'cached_goal_vs_raw_plus50_xyz_max_abs_metres':goal_error,
 'goal_input_contract':'Only provided future +50 XYZ and current XYZ/RPY are required; neither future orientation nor intermediate future poses enter this formula.',
 'dfa_ground_height':0.,'dfa_fixed_z_levels':FIX_HEIGHT.tolist(),
 'dfa_learned_offsets':'XY only; two offsets per base point per fixed Z; no learned Z',
 'future_ego_local_z_metres_all_8steps':quantile(all_xyz[...,2]),
 'absolute_future_ego_local_z_metres_3s':quantile(np.abs(all_xyz[:,:6,2])),
 'absolute_future_ego_local_z_metres_4s':quantile(np.abs(all_xyz[...,2])),
 'absolute_z_over_half_m_fraction_3s':float((np.abs(all_xyz[:,:6,2])>.5).mean()),
 'absolute_z_over_half_m_fraction_4s':float((np.abs(all_xyz[...,2])>.5).mean()),
 'projection_vs_true_future_ego_z':{
  'interpretation':'Train-only diagnostic at actual future ego XY; not predicted path accuracy or proof of physical road height.',
  'z0_error_pixels_both_visible_allcams_3s':quantile(pixel[:,:,:6][valid[:,:,:6]]),
  'z0_error_pixels_both_visible_allcams_4s':quantile(pixel[valid]),
  'best_of_five_z_error_pixels_truevisible_4s':quantile(best_error[vis_true&np.isfinite(best_error)]),
 },
 'origin_contract':'Both GT and camera projection use current ego XYZ/RPY origin. No rear-axle/lidar height shift is inserted by these code paths.',
 'physical_limit':'The inspected code does not independently identify the physical ego reference point or certify road surface equals z=0. Fixed height levels approximate local geometry.',
 'source_references':{
  'public_ground_z':'third_party/SparseDriveV2/navsim/agents/sparsedrive/blocks.py:SparsePoint3DKeyPointsGenerator and ground_height=0 in DFA',
  'public_levels':'third_party/SparseDriveV2/navsim/agents/sparsedrive/sparsedrive_config.py:fix_height and num_learnable_pts',
  'adapter':'experiments/sparsedrivev2_20260910/public_model.py:SparsePoint3DKeyPointsGenerator',
  'calibration':'scripts/derive_motiondrive_v2_geometry.py:camera_projection',
  'raw_adapter':'models/motiondrive_v2_inputs.py:build_camera_geometry and pose_geometry',
  'native_gt':'scripts/build_scene_supervision_v2.py:build_scene full-pose cache parity',
 }}
OUT.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
