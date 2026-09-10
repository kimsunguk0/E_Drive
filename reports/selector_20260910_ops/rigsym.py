"""Is the six-camera rig left-right symmetric? This gates flip augmentation/TTA.

A mirror of the ego frame (y -> -y) maps camera i to some camera j. If for every
i such a j exists with a matching mirrored projection matrix, horizontal flip is
an exact transform and both flip-TTA and flip augmentation are available.
"""
import sys, json
from pathlib import Path
import numpy as np
R = Path("/NHNHOME/data/sukim/adcl"); sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"scripts"))
from motiondrive_v2_data import MotionDriveDataset
ds = MotionDriveDataset(data_root="/tmp/pm97",
    split_manifest=str(R/"data/etri/motiondrive_v2/grouped_split_rawtime.json"), split="tune",
    supervision_root=str(R/"data/etri/motiondrive_v2/train_tune_geometry_v2"), min_frame=30,
    frame_stride=5, max_samples=0, augment=False, seed=0, history_contract="control",
    history_overlay_root=None, expected_history_overlay_sha256=None)
it = ds[0]; L = it["lidar2img"].numpy().astype(np.float64)   # (6,4,4)
H, Wd = it["images"].shape[-2:]
print("image %dx%d, cameras %d" % (H, Wd, len(L)))
# mirror of the ego frame about y=0
S = np.diag([1., -1., 1., 1.])
# mirror of the image about its vertical centre line: u -> (W-1) - u
Fx = np.array([[-1., 0., Wd-1.], [0., 1., 0.], [0., 0., 1.]])
print("\ncamera principal directions (unit ray through image centre, ego frame):")
for i, M in enumerate(L):
    A = M[:3, :3]
    c = np.linalg.solve(A, np.array([Wd/2, H/2, 1.0]) - M[:3, 3]/1.0) if abs(np.linalg.det(A)) > 1e-9 else np.zeros(3)
    n = c/max(np.linalg.norm(c), 1e-9)
    print("   cam %d  dir (%+.3f, %+.3f, %+.3f)  yaw %+7.1f deg" % (
        i, n[0], n[1], n[2], np.degrees(np.arctan2(n[1], n[0]))))
print("\nbest mirrored partner for each camera (lower = better match):")
ok = True
for i, M in enumerate(L):
    Mi = Fx @ M[:3, :] @ S            # mirror image u, mirror ego y
    best = None
    for j, Mj in enumerate(L):
        a, b = Mi/np.abs(Mi).max(), Mj[:3, :]/np.abs(Mj[:3, :]).max()
        r = float(np.abs(a-b).max())
        if best is None or r < best[0]: best = (r, j)
    flag = "OK" if best[0] < 0.05 else ("close" if best[0] < 0.25 else "NO MATCH")
    if best[0] >= 0.05: ok = False
    print("   cam %d -> cam %d   max rel diff %.4f   %s" % (i, best[1], best[0], flag))
print("\nVERDICT: %s" % ("rig is left-right symmetric - flip is an exact transform"
                         if ok else "rig is NOT exactly symmetric - flip would be approximate"))
print("\nsanity: is the trajectory label itself mirror-symmetric in distribution?")
ego = np.load("/tmp/pm97/data/etri/ego_cache.npz")
y3 = ego["fut"][:, 5, 1]
print("   fut y at 3 s: mean %+.3f  median %+.3f  |skew| %.3f" % (
    y3.mean(), np.median(y3), abs(float(((y3-y3.mean())**3).mean()/max(y3.std(), 1e-9)**3))))
g = ego["goal"][:, 1]
print("   goal y      : mean %+.3f  median %+.3f" % (g.mean(), np.median(g)))
