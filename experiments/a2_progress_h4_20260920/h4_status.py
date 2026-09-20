"""Status query input from exactly the five RGB-consumed H4/current poses."""
import numpy as np
from scipy.spatial.transform import Rotation

FRAME_OFFSETS=(-10,-5,-2,-1,0)
TIMES=np.array(FRAME_OFFSETS,np.float64)*.1
DESIGN=np.stack([np.ones_like(TIMES),TIMES,.5*TIMES**2],-1)
PINV=np.linalg.pinv(DESIGN)
assert np.linalg.matrix_rank(DESIGN)==3 and np.linalg.cond(DESIGN)<100

def status_from_five_poses(poses):
    """[...,5,4,4], chronological fixed support. No other pose may be passed."""
    poses=np.asarray(poses,np.float64)
    if poses.shape[-3:]!=(5,4,4) or not np.isfinite(poses).all():raise ValueError('Exactly five finite full poses required')
    relative=np.linalg.inv(poses[...,-1,:,:])[...,None,:,:]@poses
    xy=relative[...,:2,3]
    coef=np.einsum('ij,...jd->...id',PINV,xy)
    yaw=np.unwrap(np.arctan2(relative[...,1,0],relative[...,0,0]),axis=-1)
    yaw_coef=np.einsum('ij,...j->...i',PINV,yaw)
    status=np.concatenate([coef[...,1,:],coef[...,2,:],yaw_coef[...,1,None]],-1).astype(np.float32)
    fit=np.einsum('ij,...jd->...id',DESIGN,coef)
    rms=np.sqrt(np.mean((xy-fit)**2,axis=(-2,-1)))
    if not np.isfinite(status).all() or np.any(rms>=.25):raise ValueError('Invalid fixed-support causal fit')
    return status,rms

def status_from_clip_records(records,current_frame=0):
    wanted={current_frame+f for f in FRAME_OFFSETS};selected={}
    for row in records:
        f=row['frame']
        if isinstance(f,(bool,np.bool_)) or not np.isfinite(f) or int(f)!=f:raise ValueError('Integer pose frame required')
        f=int(f)
        if f not in wanted:continue # Do not read XYZ/RPY of unused or future records.
        if f in selected:raise ValueError('Duplicate required pose')
        selected[f]=row
    if set(selected)!=wanted:raise ValueError('Missing H4/current pose')
    ordered=[selected[current_frame+f] for f in FRAME_OFFSETS]
    xyz=np.array([[r[k] for k in ('x','y','z')] for r in ordered],np.float64)
    rpy=np.array([[r[k] for k in ('roll','pitch','yaw')] for r in ordered],np.float64)
    poses=np.broadcast_to(np.eye(4),(5,4,4)).copy();poses[:,:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();poses[:,:3,3]=xyz
    status,rms=status_from_five_poses(poses)
    return status,{'fit_relative_frames':list(FRAME_OFFSETS),'fit_count':5,'nominal_seconds':TIMES.tolist(),'design_condition':float(np.linalg.cond(DESIGN)),'position_fit_rms_m':float(rms),'all_fit_times_have_consumed_RGB':True}
