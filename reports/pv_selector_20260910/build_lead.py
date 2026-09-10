"""Ground-truth lead-vehicle features per evaluated row.

This is an ORACLE probe: the features come from the annotation, not from
images.  Its purpose is to price the information before any head is built to
estimate it, the same way the status accuracy curve priced ego velocity.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/NHNHOME/data/sukim/adcl")
META = BASE / "data/etri/meta_train"
CODEX = BASE / "experiment_worktrees/sparsedrivev2_20260910"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

ego_cache = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=False)
scen_idx = ego_cache["scen_idx"]
speed = ego_cache["speed"]
split = json.loads((BASE / "data/etri/motiondrive_v2/grouped_split_rawtime.json").read_text())
scenarios = np.array(split["scenarios"]) if "scenarios" in split else None

CORRIDOR = 3.0        # metres of half-width for "in my lane"
AHEAD_MAX = 80.0
FEATURES = ["present", "distance", "lateral", "lead_speed", "closing", "inv_ttc",
            "nearest_any", "count_ahead"]


def scene_table(scene):
    path = META / scene / "annotation" / "object.parquet"
    if not path.exists():
        return None
    obj = pd.read_parquet(path)
    stamps = np.sort(obj["timestamp"].unique())
    return obj, {t: i for i, t in enumerate(stamps)}, stamps


def features_for(scene, frames, ego_speed):
    out = np.zeros((len(frames), len(FEATURES)), dtype=np.float32)
    table = scene_table(scene)
    if table is None:
        return out, 0
    obj, _, stamps = table
    grouped = {t: g for t, g in obj.groupby("timestamp")}
    hits = 0
    for i, (frame, v_ego) in enumerate(zip(frames, ego_speed)):
        if frame >= len(stamps):
            continue
        g = grouped.get(stamps[frame])
        if g is None:
            continue
        ego = g[g["class"] == "ego"]
        others = g[g["class"] != "ego"]
        if not len(ego) or not len(others):
            continue
        e = ego.iloc[0]
        yaw = float(e["heading[rad]"])
        c, s = np.cos(yaw), np.sin(yaw)
        dx = others["x[m]"].to_numpy() - float(e["x[m]"])
        dy = others["y[m]"].to_numpy() - float(e["y[m]"])
        fwd = dx * c + dy * s
        lat = -dx * s + dy * c
        vf = others["vx"].to_numpy() * c + others["vy"].to_numpy() * s
        ahead = (fwd > 0) & (fwd < AHEAD_MAX)
        if ahead.any():
            out[i, 6] = fwd[ahead].min() / 50.0
            out[i, 7] = min((fwd < 30.0).sum() & 0xFF, 20) / 10.0
        lane = ahead & (np.abs(lat) < CORRIDOR)
        if not lane.any():
            continue
        j = np.argmin(np.where(lane, fwd, np.inf))
        distance, closing = float(fwd[j]), float(v_ego) - float(vf[j])
        out[i, 0] = 1.0
        out[i, 1] = distance / 50.0
        out[i, 2] = float(lat[j]) / 10.0
        out[i, 3] = float(vf[j]) / 20.0
        out[i, 4] = closing / 20.0
        out[i, 5] = np.clip(closing / max(distance, 1.0), -1.0, 2.0)
        hits += 1
    return out, hits


for name, cache in (("train", CODEX / "cache/c_refine_20260910/c_p20v64_train_full_v1/train"),
                    ("tune", CODEX / "cache/c_refine_20260910/c_p20v64_tune_full_v1/tune")):
    manifest = json.loads((cache / "manifest.json").read_text())
    rows = np.load(cache / "rows.npy", allow_pickle=False)
    frames = np.load(cache / "frame.npy", allow_pickle=False)
    scenes = np.array(manifest["dataset_provenance"]["scenes"])
    names = scenes[np.searchsorted(np.sort(np.unique(scen_idx[rows])), scen_idx[rows])] \
        if len(scenes) == len(np.unique(scen_idx[rows])) else None
    if names is None:
        raise SystemExit("scene mapping mismatch for " + name)
    feats = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
    total = 0
    for scene in np.unique(names):
        m = names == scene
        f, hits = features_for(scene, frames[m], speed[rows[m]])
        feats[m] = f
        total += hits
    np.savez(OUT / ("lead_%s.npz" % name), rows=rows.astype(np.int64), features=feats,
             names=np.array(FEATURES))
    print("%-6s rows %6d  lead present %5.1f%%  mean distance %.1f m  mean closing %.2f m/s"
          % (name, len(rows), 100 * feats[:, 0].mean(),
             50 * feats[feats[:, 0] > 0, 1].mean() if (feats[:, 0] > 0).any() else 0,
             20 * feats[feats[:, 0] > 0, 4].mean() if (feats[:, 0] > 0).any() else 0), flush=True)
