"""Causal nominal-time status from allowed past/current pose rows only.

This producer returns an INPUT. It never changes state/history supervision.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

def fit_status5(poses, times, now):
    poses=np.asarray(poses,np.float64);times=np.asarray(times,np.float64);now=int(now)
    fit_ids=np.where((times<=times[now]) & (times>=times[now]-1.001))[0]
    fit_ids=fit_ids[fit_ids<=now]
    if len(fit_ids)<6:
        raise ValueError("At least six past/current poses are required for status")
    relative=np.linalg.inv(poses[now])@poses[fit_ids]
    t=times[fit_ids]-times[now]
    design=np.column_stack((np.ones_like(t),t,.5*t*t))
    gaps=np.diff(times[fit_ids])
    if not (np.all(gaps>0) and np.max(gaps)<=.151 and np.linalg.cond(design)<100):
        raise ValueError("Invalid causal fit timing/conditioning")
    position=np.linalg.lstsq(design,relative[:,:2,3],rcond=None)[0]
    angle=np.unwrap(np.arctan2(relative[:,1,0],relative[:,0,0]))
    angular=np.linalg.lstsq(design,angle,rcond=None)[0]
    status=np.asarray([*position[1],*position[2],angular[1]],np.float32)
    residual=relative[:,:2,3]-design@position
    rms=float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(status).all() or rms>=.25:
        raise ValueError("Invalid causal pose fit")
    return status, {"fit_count":int(len(fit_ids)),"fit_relative_times":t.tolist(),
                    "position_fit_rms_m":rms}

def status5_from_clip_records(records, current_frame=0):
    selected={}
    for row in records:
        frame=row["frame"]
        if isinstance(frame,(bool,np.bool_)) or not np.isfinite(frame) or int(frame)!=frame:
            raise ValueError("Pose frame must be an integer")
        frame=int(frame)
        # Do not access any XYZ/RPY field of a future row, including goal +50.
        if not current_frame-30<=frame<=current_frame:
            continue
        if frame in selected:
            raise ValueError("Duplicate causal pose frame")
        selected[frame]=row
    frames=np.asarray(sorted(selected),np.int64)
    if len(frames)==0 or frames[-1]!=current_frame:
        raise ValueError("Current pose is absent")
    xyz=np.asarray([[selected[int(f)][k] for k in ("x","y","z")] for f in frames],np.float64)
    rpy=np.asarray([[selected[int(f)][k] for k in ("roll","pitch","yaw")] for f in frames],np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
        raise ValueError("Nonfinite causal pose")
    poses=np.broadcast_to(np.eye(4),(len(frames),4,4)).copy()
    poses[:,:3,:3]=Rotation.from_euler("xyz",rpy).as_matrix()
    poses[:,:3,3]=xyz
    # Preserve frame gaps; never use the row ordinal as a timestamp.
    times=(frames-current_frame).astype(np.float64)*.1
    status,diagnostics=fit_status5(poses,times,len(frames)-1)
    diagnostics.update(time_policy="frame_difference_times_0.1_seconds",
                       frame_count=int(len(frames)),max_input_frame=int(frames[-1]))
    return status,diagnostics
