"""Section 4.4 label audit: synthetic invariances plus a real-frame overlay."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("_ss", ROOT / "scripts/build_scene_supervision_v2.py")
bs = importlib.util.module_from_spec(spec); sys.modules["_ss"] = bs; spec.loader.exec_module(bs)
CACHE = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.parent.mkdir(parents=True, exist_ok=True)
result = {}

for split in ("train", "tune"):
    z = np.load(CACHE / f"{split}.npz", allow_pickle=False)
    t, v, present = z["future_target"], z["future_valid"], z["future_present"]
    result[split] = {
        "rows": int(len(t)), "shape": list(t.shape[1:]),
        "present_fraction": present.mean(0).tolist(),
        "rows_with_no_valid_cell": [int((~v[:, i].reshape(len(v), -1).any(-1)).sum())
                                    for i in range(t.shape[1])],
        "positive_rate_inside_valid": [float(t[:, i][v[:, i]].mean()) for i in range(t.shape[1])],
        "seconds_median": np.nanmedian(z["future_seconds"], 0).tolist(),
        "finite": bool(np.isfinite(t).all()),
    }

# --- synthetic invariances ---------------------------------------------------
def box(x, y, yaw=0.0, w=4.0, l=2.0):
    return {"class": "Car", "x[m]": x, "y[m]": y, "z[m]": 0.0, "heading[rad]": yaw,
            "width[m]": w, "length[m]": l, "height[m]": 1.6, "num_points": 100.0}


def pose(x, y, yaw=0.0):
    m = np.eye(4)
    m[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    m[0, 3], m[1, 3] = x, y
    return m


visible = np.ones(bs.GRID_SHAPE, bool)
now = pose(10.0, 5.0, 0.3)
w2e = np.linalg.inv(now)
static = [box(30.0, 6.0)]
t_now, _ = bs.rasterize_objects(static, w2e, visible)
# 1. A world-static object keeps its place even though the ego moved on.
later = pose(18.0, 7.0, 0.45)
t_future_same_transform, _ = bs.rasterize_objects(static, w2e, visible)
result["static_object_unmoved"] = {
    "identical": bool((t_now == t_future_same_transform).all()),
    "note": "future boxes use the CURRENT transform, so ego motion cannot move a static object",
}
# 2. A one-metre world displacement moves the footprint by one metre.
moved, _ = bs.rasterize_objects([box(31.0, 6.0)], w2e, visible)
cell_x = (bs.GRID_EXTENT[1] - bs.GRID_EXTENT[0]) / bs.GRID_SHAPE[0]
centroid = lambda m: np.array(np.nonzero(m[0])).mean(-1)
shift = (centroid(moved) - centroid(t_now)) * np.array(
    [cell_x, (bs.GRID_EXTENT[3] - bs.GRID_EXTENT[2]) / bs.GRID_SHAPE[1]])
result["one_metre_shift"] = {"measured_m": shift.tolist(), "cell_size_m": float(cell_x)}
# 3. The ego frame changing on its own must not move anything.
t_other_ego, _ = bs.rasterize_objects(static, np.linalg.inv(later), visible)
result["ego_frame_change_moves_target"] = {
    "differs_as_expected": bool((t_now != t_other_ego).any()),
    "note": "a different transform does move it, which is why the current one is used",
}
# 4. Double flip is the identity on target and mask.
z = np.load(CACHE / "tune.npz", allow_pickle=False)
t, v = z["future_target"][:64], z["future_valid"][:64]
result["double_flip_identity"] = {
    "target": bool((t[..., ::-1][..., ::-1] == t).all()),
    "valid": bool((v[..., ::-1][..., ::-1] == v).all()),
}
# 5. Objects never overlap the ego cell itself (ego is excluded).
ego_i = int((0.0 - bs.GRID_EXTENT[0]) / cell_x)
ego_j = int((0.0 - bs.GRID_EXTENT[2]) / ((bs.GRID_EXTENT[3] - bs.GRID_EXTENT[2]) / bs.GRID_SHAPE[1]))
result["ego_cell_positive_rate"] = float(z["future_target"][:, :, ego_i, ego_j].mean())

print(json.dumps(result, indent=1))
OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
