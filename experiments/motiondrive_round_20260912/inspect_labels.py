"""Task 2 label pre-check: can a future footprint target be built at all?"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/NHNHOME/data/sukim/adcl")
META = BASE / "data/etri/meta_train"
SUP = json.loads((BASE / "data/etri/motiondrive_v2/train_tune_geometry_v2"
                  / "supervision_manifest.json").read_text())
split = json.loads((BASE / "data/etri/motiondrive_v2/grouped_split_rawtime.json").read_text())
train_scenes = sorted(split["splits"]["train"])
rng = np.random.default_rng(0)
sample = [train_scenes[i] for i in rng.choice(len(train_scenes), 24, replace=False)]

contract = {"grid_shape": SUP["grid_shape"], "grid_extent": SUP["grid_extent"],
            "grid_axis_order": SUP["grid_axis_order"],
            "object_dimensions_note": SUP["annotation_contract"]["object_dimensions"],
            "occupancy_support_note": SUP["annotation_contract"]["support"],
            "scenes_inspected": sample}

frames_total = present_5 = present_10 = 0
dt5, dt10 = [], []
classes, id_stats, moving = Counter(), [], []
coord_checks = []
for scene in sample:
    path = META / scene / "annotation" / "object.parquet"
    if not path.exists():
        continue
    obj = pd.read_parquet(path)
    classes.update(obj["class"].value_counts().to_dict())
    stamps = np.sort(obj["timestamp"].unique())
    by_stamp = {t: g for t, g in obj.groupby("timestamp")}
    n = len(stamps)
    tracks = obj[obj["class"] != "ego"].groupby("obj_id").size()
    id_stats.append((len(tracks), float(tracks.mean()) if len(tracks) else 0.0))
    for i in range(30, n - 10):
        frames_total += 1
        for lag, bucket, seen in ((5, dt5, "5"), (10, dt10, "10")):
            j = i + lag
            if j >= n:
                continue
            g = by_stamp.get(stamps[j])
            if g is None:
                continue
            bucket.append((stamps[j] - stamps[i]) / 1000.0)
            if (g["class"] != "ego").any():
                if lag == 5:
                    present_5 += 1
                else:
                    present_10 += 1
    # Does a world-frame object stay put once expressed in a fixed ego frame?
    i, j = 40, 50
    a, b = by_stamp.get(stamps[i]), by_stamp.get(stamps[j])
    if a is None or b is None:
        continue
    ea = a[a["class"] == "ego"]
    if not len(ea):
        continue
    e = ea.iloc[0]
    yaw, c, s = float(e["heading[rad]"]), None, None
    c, s = np.cos(yaw), np.sin(yaw)

    def to_ego(g):
        dx = g["x[m]"].to_numpy() - float(e["x[m]"])
        dy = g["y[m]"].to_numpy() - float(e["y[m]"])
        return np.stack([dx * c + dy * s, -dx * s + dy * c], -1)

    common = set(a[a["class"] != "ego"]["obj_id"]) & set(b[b["class"] != "ego"]["obj_id"])
    if not common:
        continue
    pa = {int(r.obj_id): p for r, p in zip(a[a["class"] != "ego"].itertuples(),
                                           to_ego(a[a["class"] != "ego"]))}
    pb = {int(r.obj_id): p for r, p in zip(b[b["class"] != "ego"].itertuples(),
                                           to_ego(b[b["class"] != "ego"]))}
    speeds, shifts = [], []
    for oid in common:
        speeds.append(float(np.hypot(*a[a["obj_id"] == oid][["vx", "vy"]].to_numpy()[0])))
        shifts.append(float(np.linalg.norm(pb[oid] - pa[oid])))
    speeds, shifts = np.array(speeds), np.array(shifts)
    still = speeds < 0.2
    if still.any():
        coord_checks.append(float(np.median(shifts[still])))
    moving.append(float((speeds > 0.5).mean()))

contract.update({
    "frames_examined": frames_total,
    "future_object_presence_rate": {"+5": present_5 / max(frames_total, 1),
                                    "+10": present_10 / max(frames_total, 1)},
    "actual_seconds_per_lag": {"+5": {"median": float(np.median(dt5)), "p1": float(np.percentile(dt5, 1)),
                                      "p99": float(np.percentile(dt5, 99))},
                               "+10": {"median": float(np.median(dt10)), "p1": float(np.percentile(dt10, 1)),
                                       "p99": float(np.percentile(dt10, 99))}},
    "classes": dict(classes),
    "tracks_per_scene_median": float(np.median([t for t, _ in id_stats])),
    "track_length_frames_median": float(np.median([m for _, m in id_stats])),
    "moving_object_fraction_median": float(np.median(moving)),
    "world_frame_check_static_object_shift_m": {
        "median_over_scenes": float(np.median(coord_checks)) if coord_checks else None,
        "note": ("Objects with |v| < 0.2 m/s re-expressed in one fixed ego frame move this "
                 "far over 10 frames. Near zero means the raw coordinates are world-fixed "
                 "and the ego row supplies the pose."),
    },
})
print(json.dumps(contract, indent=1))
Path(sys.argv[1]).write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
