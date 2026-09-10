"""Horizontal-flip augmentation for the six-camera MotionDrive input.

The rig has complete mirror pairs (0-0, 1-2, 3-3, 4-5) so a y -> -y mirror of the
ego frame maps the sensor set onto itself. Every geometric quantity must be
mirrored together; missing one produces a silently wrong label. Applied only in
training - the labels carry a real right-hand bias, so inference stays upright.
"""
import numpy as np, torch
CAM_MIRROR = [0, 2, 1, 3, 5, 4]
def _S4(t):                       # ego-frame mirror about y = 0
    s = torch.eye(4, dtype=t.dtype); s[1, 1] = -1.0
    return s
def flip_item(it, width_full, width_hist):
    o = dict(it)
    img = it["images"]                                    # (6,3,H,W)
    o["images"] = torch.flip(img, dims=[-1])[CAM_MIRROR]
    o["history_images"] = torch.flip(it["history_images"], dims=[-1])
    L = it["lidar2img"].clone()                           # (6,4,4)
    S = _S4(L)
    Fx = torch.eye(3, dtype=L.dtype); Fx[0, 0] = -1.0; Fx[0, 2] = float(width_full - 1)
    new = L.clone()
    for i in range(len(L)):
        m = L[i].clone()
        m[:3, :] = Fx @ (L[i][:3, :] @ S)
        new[CAM_MIRROR[i]] = m
    o["lidar2img"] = new
    T = it["history_transforms"].clone()                  # (4,4,4) current -> past
    o["history_transforms"] = torch.stack([S @ T[k] @ S for k in range(len(T))])
    for k in ("goal_xy", "gt_plan"):
        v = it[k].clone(); v[..., 1] = -v[..., 1]; o[k] = v
    h = it["history_target"].clone()                      # (4,4) dx,dy,sin,cos
    h[:, 1] = -h[:, 1]; h[:, 2] = -h[:, 2]; o["history_target"] = h
    s = it["state_target"].clone()                        # vx,vy,ax,ay,yawrate,stop
    s[1] = -s[1]; s[3] = -s[3]; s[4] = -s[4]; o["state_target"] = s
    if "provided_status5" in it:
        p = it["provided_status5"].clone()
        p[1] = -p[1]; p[3] = -p[3]; p[4] = -p[4]; o["provided_status5"] = p
    for k in ("occ_target", "lane_target", "occ_valid", "lane_valid"):
        if k in it: o[k] = torch.flip(it[k], dims=[-1])   # (1,64,48): y is the last axis
    return o
class FlipAugmented(torch.utils.data.Dataset):
    """Deterministic per (index, epoch), matching the repo's augmentation policy."""
    def __init__(s, base, width_full, width_hist, p=0.5, seed=0):
        s.base, s.wf, s.wh, s.p, s.seed, s.epoch = base, width_full, width_hist, p, seed, 0
        for a in ("rows", "arr", "scene_names"):
            if hasattr(base, a): setattr(s, a, getattr(base, a))
    def __len__(s): return len(s.base)
    def set_epoch(s, e):
        s.epoch = e
        if hasattr(s.base, "set_epoch"): s.base.set_epoch(e)
    def __getattr__(s, n):
        if n in {"base", "wf", "wh", "p", "seed", "epoch"}: raise AttributeError(n)
        return getattr(s.base, n)
    def __getitem__(s, i):
        it = s.base[i]
        r = np.random.RandomState((s.seed * 1000003 + s.epoch * 7919 + i) % (2**31 - 1))
        return flip_item(it, s.wf, s.wh) if r.rand() < s.p else it
