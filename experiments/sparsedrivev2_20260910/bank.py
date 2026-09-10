#!/usr/bin/env python3
"""Train-only fixed-metre path x absolute interval-speed anchors for ETRI.

Three separate commands enforce the fitting/evaluation boundary: ``extract``
selects one partition, ``fit`` accepts a train-only extraction, and ``oracle``
verifies a frozen bank before opening tune targets. Neither status nor goal is
an input to composition. Oracle scores are privileged coverage diagnostics,
not model performance. Invalid extrapolated candidates cannot win the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float32) / 36
DT = .5
SCHEMA = "etri_fixed_metre_absolute_velocity_v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def npz_new(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def sample_path(xy, stations):
    """Arc-length sample without endpoint padding/extrapolation.

    The path starts at the origin, can reverse, and can contain exact stops.
    Invalid trailing samples are zero only as storage; callers must use mask.
    """
    xy = np.asarray(xy, np.float64)
    stations = np.asarray(stations, np.float64)
    require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy), "bad XY path")
    require(np.isfinite(xy).all(), "nonfinite path")
    distance = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=-1).cumsum()]
    keep = np.r_[True, np.diff(distance) > 1e-9]
    distance, xy = distance[keep], xy[keep]
    valid = stations <= distance[-1] + 1e-7
    out = np.zeros((len(stations), 2), np.float32)
    if len(distance) > 1:
        for coordinate in range(2):
            out[valid, coordinate] = np.interp(stations[valid], distance, xy[:, coordinate])
    return out, valid, float(distance[-1])


def speeds(future):
    future = np.asarray(future)
    return np.linalg.norm(np.diff(np.concatenate([np.zeros_like(future[:, :1]), future], axis=1), axis=1), axis=-1) / DT


def compose(path_xy, velocity, allow_extrapolation=False):
    """Compose every pair in metres, using each centre's actual arc length.

    Unlike normalized-progress banks, velocity does not rescale path geometry.
    A path's KMeans centre may have slightly shorter arc length than its nominal
    50 stations. Recompute that arc length, as the public V2 code does.
    """
    p, v, t = len(path_xy), len(velocity), velocity.shape[1]
    out = np.zeros((p, v, t, 2), np.float32)
    mask = np.zeros((p, v, t), bool)
    lengths = np.zeros(p, np.float32)
    target = np.cumsum(velocity.astype(np.float64) * DT, axis=1)
    require((velocity >= 0).all(), "absolute interval speeds must be nonnegative")
    for i, path in enumerate(path_xy):
        padded = np.concatenate([np.zeros((1, 2)), path.astype(np.float64)])
        d = np.r_[0., np.linalg.norm(np.diff(padded, axis=0), axis=-1).cumsum()]
        keep = np.r_[True, np.diff(d) > 1e-9]
        d, padded = d[keep], padded[keep]
        lengths[i] = d[-1]
        valid = target <= d[-1] + 1e-7
        mask[i] = valid
        for c in range(2):
            values = np.interp(target, d, padded[:, c])
            if allow_extrapolation and len(d) > 1:
                slope = (padded[-1, c] - padded[-2, c]) / (d[-1] - d[-2])
                values = np.where(valid, values, padded[-1, c] + (target - d[-1]) * slope)
            out[i, :, :, c] = values
    return out, mask, lengths


def add_headings(path_xy, velocity, traj_xy):
    """Public decoder-compatible XY-heading anchors derived from XY centres.

    Heading is unwrapped before interpolation, then wrapped back to [-pi, pi).
    It is auxiliary sampling metadata; external ETRI output uses XY only.
    """
    padded = np.concatenate([np.zeros_like(path_xy[:, :1]), path_xy], axis=1)
    delta = np.diff(padded.astype(np.float64), axis=1)
    heading = np.unwrap(np.arctan2(delta[:, :, 1], delta[:, :, 0]), axis=1)
    path_xyz = np.concatenate([path_xy, (((heading + np.pi) % (2*np.pi)) - np.pi)[:, :, None]], axis=-1).astype(np.float32)
    trajectory_heading = np.zeros(traj_xy.shape[:-1], np.float32)
    target_distance = np.cumsum(velocity.astype(np.float64)*DT, axis=1)
    for i in range(len(path_xy)):
        d = np.r_[0., np.linalg.norm(delta[i], axis=-1).cumsum()]
        h = np.r_[0., heading[i]]
        keep = np.r_[True, np.diff(d) > 1e-9]
        h_at_s = np.interp(target_distance, d[keep], h[keep])
        trajectory_heading[i] = ((h_at_s + np.pi) % (2*np.pi)) - np.pi
    return path_xyz, np.concatenate([traj_xy, trajectory_heading[..., None]], axis=-1)


def extract(args):
    import pandas as pd
    from scipy.spatial.transform import Rotation

    output = Path(args.output)
    require(not output.exists(), f"output exists: {output}")
    split = json.loads(Path(args.split).read_text())
    allowed = set(split["splits"][args.partition])
    if args.scenes_json:
        subset = json.loads(Path(args.scenes_json).read_text())
        if isinstance(subset, dict):
            subset = subset["scenes"]
        require(set(subset) <= allowed, "scene subset contains nonpartition scenes")
        allowed = set(subset)
    with np.load(args.ego, allow_pickle=False) as z:
        names, scen_idx, frames = z["scenarios"], z["scen_idx"], z["frame"]
        selection = np.isin(names[scen_idx], list(allowed)) & (frames >= args.min_frame) & (frames % args.stride == 0)
        rows = np.flatnonzero(selection).astype("<i8")
        # No other partition's label values are selected or analyzed.
        future6 = z["fut"][rows].astype(np.float32)
        selected_frames, selected_scenes = frames[rows].astype(np.int32), names[scen_idx[rows]]
    require(len(rows) > 0, "no selected rows")
    with np.load(args.ego5, allow_pickle=False) as z:
        require(np.array_equal(z["scenarios"], names), "ego5 scene order mismatch")
        require(np.array_equal(z["scen_idx"][rows], scen_idx[rows]), "ego5 scene index mismatch")
        require(np.array_equal(z["frame"][rows], frames[rows]), "ego5 frame mismatch")
        require(z["mask5"][rows, :8].all(), "incomplete 4sec velocity targets")
        future8 = z["fut5"][rows, :8].astype(np.float32)
    require(np.array_equal(future8[:, :6], future6), "fut6/fut5 coordinate mismatch")
    stations = np.arange(1, args.path_points + 1, dtype=np.float64) * args.spacing
    paths = np.zeros((len(rows), len(stations), 2), np.float32)
    masks = np.zeros((len(rows), len(stations)), bool)
    raw_lengths = np.zeros(len(rows), np.float32)
    remaining_frames = np.zeros(len(rows), np.int32)
    source_records = []
    reverse_rows = 0
    gt_max_error = 0.
    for s, scene in enumerate(sorted(allowed)):
        ids = np.flatnonzero(selected_scenes == scene)
        if not len(ids):
            continue
        root = Path(args.metadata) / str(scene)
        ts_path, pose_path = root / "meta/timestamps.parquet", root / "annotation/ego_pose.parquet"
        ts, ep = pd.read_parquet(ts_path), pd.read_parquet(pose_path)
        table = ts[["timestamp", "frame_id"]].merge(ep, on="timestamp", how="left", validate="one_to_one").sort_values("frame_id")
        xyz = table[["x", "y", "z"]].to_numpy(np.float64)
        rpy = table[["roll", "pitch", "yaw"]].to_numpy(np.float64)
        require(np.isfinite(xyz).all() and np.isfinite(rpy).all(), f"invalid pose {scene}")
        rot = Rotation.from_euler("xyz", rpy).as_matrix()
        frame_ids = table["frame_id"].to_numpy(np.int64)
        lookup = {int(frame): i for i, frame in enumerate(frame_ids)}
        require(len(lookup) == len(frame_ids), "repeated raw frame")
        timestamps = table["timestamp"].to_numpy(np.float64)
        require(np.all(np.diff(timestamps) > 0), "nonmonotonic timestamps")
        for j in ids:
            frame = int(selected_frames[j])
            now = lookup[frame]
            local = ((xyz[now:] - xyz[now]) @ rot[now])[:, :2]
            target_ids = [lookup[frame + dt] - now for dt in (5, 10, 15, 20, 25, 30)]
            error = float(np.abs(local[target_ids] - future6[j]).max())
            gt_max_error = max(gt_max_error, error)
            require(error < 2e-5, f"native/GT convention mismatch {scene}/{frame}: {error}")
            paths[j], masks[j], raw_lengths[j] = sample_path(local, stations)
            remaining_frames[j] = len(local) - 1
            reverse_rows += int(np.any(np.diff(local[:target_ids[-1] + 1, 0]) < -.01))
        source_records.append({"scene": str(scene), "timestamp_sha256": sha(ts_path), "pose_sha256": sha(pose_path),
                               "raw_frame_min": int(frame_ids[0]), "raw_frame_max": int(frame_ids[-1]),
                               "timestamp_min": float(timestamps[0]), "timestamp_max": float(timestamps[-1])})
        if (s + 1) % 25 == 0:
            print(f"extract {args.partition} {s+1}/{len(allowed)} scenes", flush=True)
    vel8 = speeds(future8).astype(np.float32)
    metadata = {
        "schema": SCHEMA, "partition": args.partition, "row_count": len(rows), "scene_count": len(set(selected_scenes)),
        "scenes": sorted(set(map(str, selected_scenes))), "rows_sha256": arr_sha(rows),
        "split_sha256": sha(args.split), "ego_sha256": sha(args.ego), "ego5_sha256": sha(args.ego5),
        "code_sha256": SOURCE_SHA256, "source": source_records,
        "stride": args.stride, "min_frame": args.min_frame, "nominal_dt": DT,
        "path_spacing_metres": args.spacing, "path_points": args.path_points,
        "native_future_scope": "remaining frames in the same selected scene, never cross scene/session",
        "coordinate": "full SE3 pose inverse; x forward, y left; exact cache fut6 parity",
        "native_gt6_max_abs": gt_max_error, "full_path_rows": int(masks.all(1).sum()),
        "full_path_fraction": float(masks.all(1).mean()), "any_reverse_x_native_3sec_rows": reverse_rows,
        "zero_speed8_rows": int((vel8.max(1) < 1e-7).sum()),
        "distance3sec_quantiles": np.quantile((vel8[:, :6]*DT).sum(1), [0,.1,.5,.9,.99,1]).tolist(),
        "native_remaining_length_quantiles": np.quantile(raw_lengths, [0,.1,.5,.9,.99,1]).tolist(),
        "path_eligible_speed0_mean": float(vel8[masks.all(1), 0].mean()),
        "all_speed0_mean": float(vel8[:, 0].mean()),
    }
    npz_new(output, rows=rows, frame=selected_frames, scene=selected_scenes,
            path_xy=paths, path_mask=masks, path_stations=stations.astype(np.float32),
            velocity8=vel8, velocity=vel8[:, :6], gt_xy=future6,
            native_future_length=raw_lengths, native_future_frames=remaining_frames,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    metadata["artifact_sha256"] = sha(output)
    json_new(str(output) + ".json", metadata)
    print(json.dumps({k: metadata[k] for k in ("partition", "row_count", "rows_sha256", "full_path_fraction", "native_gt6_max_abs", "artifact_sha256")}), flush=True)


def verified_extraction(path, partition):
    manifest = json.loads(Path(str(path) + ".json").read_text())
    require(manifest["partition"] == partition, f"expected {partition} extraction")
    require(sha(path) == manifest["artifact_sha256"], "extraction artifact changed")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    require(arr_sha(data["rows"]) == manifest["rows_sha256"], "extraction row hash mismatch")
    return data, manifest


def fit(args):
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits

    require(not Path(args.output).exists(), f"output exists: {args.output}")
    train, train_manifest = verified_extraction(args.train, "train")
    full = train["path_mask"].all(1)
    path_features = train["path_xy"][full].reshape(int(full.sum()), -1)
    moving = train["velocity8"].max(1) > 1e-7
    moving_velocity = train["velocity8"][moving]
    velocity_mode = getattr(args, "velocity_mode", "absolute8")
    if velocity_mode == "progress6":
        velocity_features = np.cumsum(moving_velocity[:, :6] * DT, axis=1) * np.sqrt(WEIGHTS)
    else:
        velocity_features = moving_velocity
    require(len(path_features) >= args.paths, "insufficient complete physical paths")
    require(len(velocity_features) >= args.velocities - 1, "insufficient velocity profiles")
    kmeans_kwargs = dict(n_init=3, batch_size=4096, max_iter=args.max_iter, max_no_improvement=25,
                         reassignment_ratio=0., tol=1e-5, random_state=args.seed, compute_labels=True)
    started = time.time()
    reused_path_manifest = None
    with threadpool_limits(limits=args.threads):
        if getattr(args, "path_bank", None):
            reused_path_manifest = json.loads(Path(str(args.path_bank)+".json").read_text())
            require(sha(args.path_bank) == reused_path_manifest["bank_sha256"], "reused path bank changed")
            require(reused_path_manifest["train_rows_sha256"] == train_manifest["rows_sha256"], "reused path train rows mismatch")
            with np.load(args.path_bank, allow_pickle=False) as z:
                path_xy = z["path_xy"].copy()
                path_support = z["path_support"].copy()
                require(np.array_equal(z["path_stations"], train["path_stations"]), "reused path stations mismatch")
            require(len(path_xy) == args.paths, "requested path count differs from reused bank")
            print(f"reuse {args.paths} frozen paths, SHA {reused_path_manifest['bank_sha256']}", flush=True)
        else:
            print(f"fit paths {len(path_features)} x {path_features.shape[1]} -> {args.paths}", flush=True)
            pk = MiniBatchKMeans(n_clusters=args.paths, **kmeans_kwargs).fit(path_features)
            path_xy = pk.cluster_centers_.reshape(args.paths, train["path_xy"].shape[1], 2).astype(np.float32)
            path_support = np.bincount(pk.labels_, minlength=args.paths)
        print(f"fit velocity {len(velocity_features)} x {velocity_features.shape[1]} -> {args.velocities-1} + exact stop ({velocity_mode})", flush=True)
        vk = MiniBatchKMeans(n_clusters=args.velocities - 1, **kmeans_kwargs).fit(velocity_features)
    if velocity_mode == "progress6":
        progress_centres = vk.cluster_centers_ / np.sqrt(WEIGHTS)
        v6 = np.diff(np.c_[np.zeros(len(progress_centres)), progress_centres], axis=1) / DT
        require(v6.min() > -1e-5, "nonmonotone progress centre")
        tail = np.zeros((args.velocities-1, 2), np.float64)
        support = np.bincount(vk.labels_, minlength=args.velocities-1)
        np.add.at(tail, vk.labels_, moving_velocity[:, 6:8])
        tail /= np.maximum(support, 1)[:, None]
        for empty in np.flatnonzero(support == 0):
            nearest = np.square(velocity_features - vk.cluster_centers_[empty]).sum(1).argmin()
            tail[empty] = moving_velocity[nearest, 6:8]
        centres8 = np.concatenate([np.maximum(v6, 0), tail], axis=1).astype(np.float32)
    else:
        centres8 = np.maximum(vk.cluster_centers_, 0).astype(np.float32)
    velocity8 = np.concatenate([np.zeros((1, 8), np.float32), centres8])
    # Store full 8 for the public embedding, but completed candidates are exactly
    # 6 positions. Extrapolated storage is explicitly masked out of selection.
    traj8, valid8, path_length = compose(path_xy, velocity8, allow_extrapolation=True)
    path_xyz, traj_xyz8 = add_headings(path_xy, velocity8, traj8)
    candidate_valid = valid8[:, :, :6].all(2)
    require(candidate_valid[:, 0].all() and np.count_nonzero(traj8[:, 0]) == 0, "stop contract broken")
    require(np.isfinite(traj8).all(), "nonfinite composed bank")
    metadata = {
        "schema": SCHEMA, "partition": "train", "paths": args.paths, "velocities": args.velocities,
        "candidate_count": args.paths * args.velocities, "valid_candidate_count": int(candidate_valid.sum()),
        "unique_stop_rows": 1, "cartesian_stop_duplicates": args.paths,
        "train_rows_sha256": train_manifest["rows_sha256"], "train_artifact_sha256": train_manifest["artifact_sha256"],
        "train_scenes": train_manifest["scenes"], "split_sha256": train_manifest["split_sha256"],
        "code_sha256": SOURCE_SHA256, "fit_seed": args.seed, "kmeans": kmeans_kwargs,
        "path_fit_rows": int(full.sum()), "velocity_fit_rows": int(moving.sum()),
        "path_stations_metres": train["path_stations"].tolist(),
        "fit_seconds": time.time() - started, "path_fit": "unscaled fixed-metre XY MiniBatchKMeans centres; complete native paths only",
        "velocity_fit": "unscaled absolute speed 8 half-second intervals MiniBatchKMeans centres; exact zero reserved",
        "velocity_mode": velocity_mode,
        "reused_path_bank_sha256": reused_path_manifest["bank_sha256"] if reused_path_manifest else None,
        "path_xy_sha256": arr_sha(path_xy),
        "composition": "s=cumsum(v*0.5); linear interpolation on physical arc length of path centre; no runtime coordinate refinement",
        "invalid": "extrapolated coordinates stored with traj_mask false; candidate_valid requires all first6 true; selection must mask invalid rows",
        "status_goal_dependency": "none for anchors or composition; later ranking of completed rows is separate",
        "official_weights": WEIGHTS.tolist(), "nominal_dt": DT,
        "path_length_quantiles": np.quantile(path_length, [0,.1,.5,.9,1]).tolist(),
        "path_inertia": reused_path_manifest["path_inertia"] if reused_path_manifest else float(pk.inertia_),
        "velocity_inertia": float(vk.inertia_),
        "path_iterations": 0 if reused_path_manifest else int(pk.n_iter_), "velocity_iterations": int(vk.n_iter_),
    }
    if velocity_mode == "progress6":
        metadata["velocity_fit"] = "KMeans first6 cumulative progress * sqrt(D3_time_weights); derive v0:6 by differences; v6:8 mean of actual train profiles assigned to same cluster (nearest train for empty cluster); exact zero reserved"
    npz_new(args.output, path_xy=path_xy, path_xyz=path_xyz, path_mask=np.ones(path_xy.shape[:2], bool),
            path_stations=train["path_stations"], path_length=path_length,
            velocity=velocity8[:, :6], velocity8=velocity8,
            traj_xy=traj8[:, :, :6], traj_xy6=traj8[:, :, :6], traj_xy8=traj8, traj_xyz8=traj_xyz8,
            traj_mask=valid8[:, :, :6], mask8=valid8, candidate_valid=candidate_valid,
            path_support=path_support,
            velocity_support=np.r_[(~moving).sum(), np.bincount(vk.labels_, minlength=args.velocities-1)],
            train_rows=train["rows"], train_rows_sha256=np.asarray(train_manifest["rows_sha256"]),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    metadata["bank_sha256"] = sha(args.output)
    json_new(str(args.output) + ".json", metadata)
    print(json.dumps(metadata), flush=True)


def summarize_scores(values, gt_distance):
    result = {"mean": float(np.mean(values)), "median": float(np.median(values)),
              "p90": float(np.quantile(values, .9)), "p99": float(np.quantile(values, .99))}
    bins = [(0., .1), (.1, 5.), (5., 15.), (15., 30.), (30., 50.), (50., float("inf"))]
    result["distance3sec_bins"] = {}
    for lo, hi in bins:
        selected = (gt_distance >= lo) & (gt_distance < hi)
        if selected.any():
            result["distance3sec_bins"][f"{lo:g}-{hi:g}m"] = {"n": int(selected.sum()), "mean": float(values[selected].mean())}
    return result


def oracle(args):
    import torch
    # Freeze/verify the bank before loading any tune target.
    bm = json.loads(Path(str(args.bank) + ".json").read_text())
    require(sha(args.bank) == bm["bank_sha256"], "frozen bank SHA mismatch")
    require(bm["partition"] == "train", "bank did not fit train")
    with np.load(args.bank, allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    tune, tm = verified_extraction(args.tune, "tune")
    require(tm["split_sha256"] == bm["split_sha256"], "bank/tune split mismatch")
    require(not set(tm["scenes"]) & set(bm["train_scenes"]), "bank/tune scene overlap")
    require(not np.intersect1d(bank["train_rows"], tune["rows"]).size, "bank/tune row overlap")
    if "path_stations" in bank:
        require(np.array_equal(bank["path_stations"], tune["path_stations"]), "bank/tune path stations mismatch")
    require(not Path(args.output).exists(), f"output exists: {args.output}")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    p, v = bank["traj_xy"].shape[:2]
    modes = {"path1_velocity_all": (1, v), "path_all_velocity1": (p, 1), "coarse1x1": (1, 1),
             "coarse20x10": (min(20,p), min(10,v)), "coarse20x20": (min(20,p), min(20,v)),
             "coarse64x16": (min(64,p), min(16,v)), "coarse128x64": (min(128,p), min(64,v))}
    n = len(tune["rows"])
    all_scores = {"full_bank": np.empty(n, np.float32), **{k: np.empty(n, np.float32) for k in modes}}
    winner = np.empty(n, np.int32)
    point_errors = np.empty((n, 6), np.float32)
    path_rank_of_winner, velocity_rank_of_winner = np.empty(n, np.int32), np.empty(n, np.int32)
    candidate = torch.as_tensor(bank["traj_xy"].reshape(-1, 6, 2), device=device)
    candidate_valid = torch.as_tensor(bank["candidate_valid"].reshape(-1), device=device)
    candidate_pid = torch.arange(p, device=device).repeat_interleave(v)
    candidate_vid = torch.arange(v, device=device).repeat(p)
    weight = torch.as_tensor(WEIGHTS, device=device)
    path_centres = torch.as_tensor(bank["path_xy"], device=device)
    vel_centres = torch.as_tensor(bank["velocity8"], device=device)
    coarse_mode = getattr(args, "coarse_mode", "native")
    coarse_source = None
    if coarse_mode == "d3":
        from losses import coarse_costs
        coarse_source = {"losses_sha256": sha(Path(__file__).with_name("losses.py")),
                         "data_sha256": sha(Path(__file__).with_name("data.py"))}
    coarse_path_min = np.empty(n, np.float32)
    coarse_velocity_min = np.empty(n, np.float32)
    started = time.time()
    for start in range(0, n, args.rows_chunk):
        end = min(start + args.rows_chunk, n)
        target = torch.as_tensor(tune["gt_xy"][start:end], device=device)
        if coarse_mode == "d3":
            path_cost, vel_cost = coarse_costs(path_centres, vel_centres, target)
        else:
            target_path = torch.as_tensor(tune["path_xy"][start:end], device=device)
            target_mask = torch.as_tensor(tune["path_mask"][start:end], device=device)
            path_error = ((target_path[:, None] - path_centres[None]) ** 2).sum(-1)
            path_cost = (path_error * target_mask[:, None]).sum(-1) / target_mask.sum(-1).clamp_min(1)[:, None]
            vel_cost = (torch.as_tensor(tune["velocity8"][start:end], device=device)[:, None] - vel_centres[None]).abs().mean(-1)
        coarse_path_min[start:end] = path_cost.amin(-1).cpu().numpy()
        coarse_velocity_min[start:end] = vel_cost.amin(-1).cpu().numpy()
        path_order = torch.argsort(path_cost, dim=-1, stable=True)
        vel_order = torch.argsort(vel_cost, dim=-1, stable=True)
        path_ranks, vel_ranks = path_order.argsort(-1), vel_order.argsort(-1)
        best = torch.full((end-start,), float("inf"), device=device)
        best_id = torch.zeros(end-start, dtype=torch.long, device=device)
        coarse_best = {name: best.clone() for name in modes}
        for c0 in range(0, len(candidate), args.candidate_chunk):
            c1 = min(c0 + args.candidate_chunk, len(candidate))
            delta = candidate[None, c0:c1] - target[:, None]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            d3 = (distance * weight).sum(-1)
            d3[:, ~candidate_valid[c0:c1]] = float("inf")
            values, positions = d3.min(-1)
            update = values < best
            best_id = torch.where(update, positions + c0, best_id)
            best = torch.minimum(best, values)
            pr = path_ranks[:, candidate_pid[c0:c1]]
            vr = vel_ranks[:, candidate_vid[c0:c1]]
            for name, (kp, kv) in modes.items():
                selected = (pr < kp) & (vr < kv)
                restricted = d3.masked_fill(~selected, float("inf")).amin(-1)
                coarse_best[name] = torch.minimum(coarse_best[name], restricted)
        require(torch.isfinite(best).all().item(), "no valid full-bank candidate")
        all_scores["full_bank"][start:end] = best.cpu().numpy()
        for name in modes:
            all_scores[name][start:end] = coarse_best[name].cpu().numpy()
        winner[start:end] = best_id.cpu().numpy()
        point_errors[start:end] = torch.linalg.vector_norm(candidate[best_id] - target, dim=-1).cpu().numpy()
        local_rows = torch.arange(end-start, device=device)
        path_rank_of_winner[start:end] = path_ranks[local_rows, best_id // v].cpu().numpy()
        velocity_rank_of_winner[start:end] = vel_ranks[local_rows, best_id % v].cpu().numpy()
        if start == 0 or end == n or (start // args.rows_chunk) % 10 == 0:
            print(f"oracle {end}/{n}; full mean={all_scores['full_bank'][:end].mean():.6f}; elapsed={time.time()-started:.1f}s", flush=True)
    gt_distance = (tune["velocity"] * DT).sum(1)
    report = {"schema": SCHEMA, "bank_sha256": bm["bank_sha256"], "tune_artifact_sha256": tm["artifact_sha256"],
              "tune_rows_sha256": tm["rows_sha256"], "train_rows_sha256": bm["train_rows_sha256"],
              "row_count": n, "paths": p, "velocities": v, "seconds": time.time()-started,
              "device": str(device), "dtype": "float32", "tf32": False,
              "metric": "mean(sum_t([11,11,5,5,2,2]/36 * Euclidean(pred_t-GT_t)))",
              "oracle_warning": "every score uses held-out GT for candidate choice; this is coverage, never deployable model performance",
              "coarse_cost": "path masked native sampled-station mean squared XY; velocity mean absolute difference of 8 absolute speeds; shortlist uses GT",
              "coarse_mode": coarse_mode, "coarse_source": coarse_source,
              "coarse_path_min_mean": float(coarse_path_min.mean()), "coarse_velocity_min_mean": float(coarse_velocity_min.mean()),
              "invalid_candidate_policy": "all first6 positions must lie inside centre's observed arc length; no extrapolated candidate admitted",
              "metrics": {}, "full_bank_point_error_means": point_errors.mean(0).tolist(),
              "full_bank_pair_error_means": point_errors.reshape(n, 3, 2).mean((0,2)).tolist(),
              "code_sha256": SOURCE_SHA256}
    if coarse_mode == "d3":
        report["coarse_cost"] = "root losses.coarse_costs: path evaluated at GT6 cumulative progress under D3; velocity GT6-vocabulary cumulative progress weighted absolute error; both metres"
    for name, scores in all_scores.items():
        finite = np.isfinite(scores)
        values = summarize_scores(scores[finite], gt_distance[finite]) if finite.any() else {}
        values["rows_with_valid_candidate"] = int(finite.sum())
        values["rows_without_valid_candidate"] = int((~finite).sum())
        values["mean_all_rows"] = float(scores.mean()) if finite.all() else None
        if name in modes:
            kp, kv = modes[name]
            values["full_winner_recall"] = float(((path_rank_of_winner < kp) & (velocity_rank_of_winner < kv)).mean())
        report["metrics"][name] = values
    require(np.allclose((point_errors * WEIGHTS).sum(1), all_scores["full_bank"], atol=1e-6), "D3 point parity")
    npz_new(args.output, rows=tune["rows"], scene=tune["scene"], frame=tune["frame"],
            winner=winner, winner_path_id=winner // v, winner_velocity_id=winner % v,
            point_errors=point_errors, gt_distance3sec=gt_distance,
            path_rank_of_winner=path_rank_of_winner, velocity_rank_of_winner=velocity_rank_of_winner,
            coarse_path_min=coarse_path_min, coarse_velocity_min=coarse_velocity_min,
            **{f"d3_{name}": value for name, value in all_scores.items()})
    report["artifact_sha256"] = sha(args.output)
    json_new(str(args.output) + ".json", report)
    print(json.dumps(report), flush=True)


def diagnose(args):
    """CPU decomposition of native sampling and vocabulary coverage.

    These privileged candidate replacements are NOT additive error floors.
    A quantized velocity can sometimes compensate a quantized path's error.
    """
    bm = json.loads(Path(str(args.bank) + ".json").read_text())
    require(sha(args.bank) == bm["bank_sha256"], "frozen bank SHA mismatch")
    with np.load(args.bank, allow_pickle=False) as z:
        velocity8 = z["velocity8"]
    tune, tm = verified_extraction(args.tune, "tune")
    train, trm = verified_extraction(args.train, "train")
    require(trm["rows_sha256"] == bm["train_rows_sha256"], "diagnostic train rows differ from bank")
    require(not set(tm["scenes"]) & set(trm["scenes"]), "train/tune scene overlap")
    require(not Path(args.output).exists(), "diagnostic output exists")
    n = len(tune["rows"])
    exact_score = np.full(n, np.nan, np.float32)
    velocity_bank_score = np.zeros(n, np.float32)
    velocity_bank_id = np.zeros(n, np.int32)
    exact_valid = np.zeros(n, bool)
    started = time.time()
    for i in range(n):
        path = tune["path_xy"][i, tune["path_mask"][i]]
        if not len(path):
            path = np.zeros((1, 2), np.float32)
        velocities = np.concatenate([tune["velocity8"][i:i+1], velocity8], axis=0)
        trajectories, valid, _ = compose(path[None], velocities)
        valid6 = valid[0, :, :6].all(-1)
        error = np.linalg.norm(trajectories[0, :, :6] - tune["gt_xy"][i], axis=-1)
        d3 = (error * WEIGHTS).sum(-1)
        d3[~valid6] = np.inf
        exact_valid[i] = valid6[0]
        if valid6[0]:
            exact_score[i] = d3[0]
        velocity_bank_id[i] = int(d3[1:].argmin())
        velocity_bank_score[i] = d3[1:].min()
    require(np.isfinite(velocity_bank_score).all(), "exact-stop diagnostic must remain valid")
    distance = (tune["velocity"] * DT).sum(1)
    eligible = train["path_mask"].all(1)
    train_distance = (train["velocity"] * DT).sum(1)
    endpoint_angle = np.abs(np.arctan2(train["gt_xy"][:, -1, 1], train["gt_xy"][:, -1, 0])) * 180 / np.pi
    audit = {}
    for name, selected in {"all_train": np.ones(len(eligible), bool), "full100m_eligible": eligible, "incomplete100m_excluded": ~eligible}.items():
        moving = selected & (train_distance > 1.)
        audit[name] = {
            "rows": int(selected.sum()), "v0_mean": float(train["velocity"][selected, 0].mean()),
            "distance3sec_mean": float(train_distance[selected].mean()),
            "distance3sec_quantiles": np.quantile(train_distance[selected], [0,.1,.5,.9,1]).tolist(),
            "moving_gt3_endpoint_direction_gt10deg_fraction": float((endpoint_angle[moving] > 10).mean()),
            "moving_gt3_endpoint_direction_gt30deg_fraction": float((endpoint_angle[moving] > 30).mean()),
        }
    report = {
        "schema": SCHEMA, "bank_sha256": bm["bank_sha256"], "code_sha256": SOURCE_SHA256,
        "train_rows_sha256": trm["rows_sha256"], "tune_rows_sha256": tm["rows_sha256"],
        "seconds": time.time()-started, "train_eligibility_audit": audit,
        "warning": "privileged GT path/speed replacement diagnostics; different candidate families, not additive causal error floors",
        "native_sampling": "same extraction stations as bank, 2m for primary100m; physical arc length recomputed on sampled polygon",
        "gt_native_path_and_gt_velocity": {
            "valid_rows": int(exact_valid.sum()), "unrepresented_short_native_path_rows": int((~exact_valid).sum()),
            **summarize_scores(exact_score[exact_valid], distance[exact_valid])},
        "gt_native_path_and_bank_velocity_oracle": summarize_scores(velocity_bank_score, distance),
        "short_native_policy": "only actually supported sampled stations used; no endpoint padding; zero velocity always valid",
    }
    npz_new(args.output, rows=tune["rows"], exact_native_gtv_d3=exact_score, exact_native_gtv_valid=exact_valid,
            gt_native_bankv_d3=velocity_bank_score, gt_native_bankv_id=velocity_bank_id)
    report["artifact_sha256"] = sha(args.output)
    json_new(str(args.output) + ".json", report)
    print(json.dumps(report), flush=True)


def summarize(args):
    """Aggregate exact same-row oracle comparisons and session bootstrap CIs."""
    split = json.loads(Path(args.split).read_text())
    root = Path(args.reports)
    entries, arrays = {}, {}
    for p in sorted(root.glob("*_oracle*.npz.json")):
        meta = json.loads(p.read_text())
        artifact = Path(str(p)[:-5])
        require(sha(artifact) == meta["artifact_sha256"], "oracle report artifact changed")
        with np.load(artifact, allow_pickle=False) as z:
            arrays[p.stem[:-4]] = {k: z[k] for k in ("rows", "scene", "d3_full_bank", "point_errors")}
        entries[p.stem[:-4]] = meta
    require(entries, "no oracle reports")
    baseline_name = "p1024_v256_native100m_v8_oracle_d3coarse"
    require(baseline_name in entries, "missing predeclared V256 comparison baseline")
    baseline = arrays[baseline_name]
    session = np.asarray([split["scene_to_session"][str(scene)] for scene in baseline["scene"]])
    unique, inverse = np.unique(session, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(unique), (args.bootstrap, len(unique)))
    totals = counts[draws].sum(1)
    out = {"schema": SCHEMA, "code_sha256": SOURCE_SHA256, "split_sha256": sha(args.split),
           "row_count": len(session), "session_count": len(unique), "rows_sha256": arr_sha(baseline["rows"]),
           "bootstrap": {"replicates": args.bootstrap, "seed": args.seed,
                         "method": "resample whole tune sessions; pooled row-weighted means; paired common draws"},
           "baseline": baseline_name, "warning": "tune model-selection diagnostics; oracle uses GT choice; CIs do not certify deployment performance or final136",
           "results": {}}
    for name, meta in entries.items():
        a = arrays[name]
        require(np.array_equal(a["rows"], baseline["rows"]), "comparison rows differ")
        require(np.array_equal(a["scene"], baseline["scene"]), "comparison scene order differs")
        require(np.allclose((a["point_errors"] * WEIGHTS).sum(1), a["d3_full_bank"], atol=1e-6), "saved D3 mismatch")
        score = a["d3_full_bank"].astype(np.float64)
        sums = np.bincount(inverse, weights=score)
        boots = sums[draws].sum(1) / totals
        delta = score - baseline["d3_full_bank"].astype(np.float64)
        delta_sums = np.bincount(inverse, weights=delta)
        delta_boots = delta_sums[draws].sum(1) / totals
        out["results"][name] = {
            "bank_sha256": meta["bank_sha256"], "oracle_artifact_sha256": meta["artifact_sha256"],
            "paths": meta["paths"], "velocities": meta["velocities"], "coarse_mode": meta.get("coarse_mode", "native"),
            "full_oracle_mean": float(score.mean()), "full_oracle_ci95": np.quantile(boots, [.025,.975]).tolist(),
            "delta_vs_v256": float(delta.mean()), "paired_delta_ci95": np.quantile(delta_boots, [.025,.975]).tolist(),
            "headroom_to_0_15": .15 - float(score.mean()),
            "gt_coarse1x1": meta["metrics"]["coarse1x1"]["mean_all_rows"],
            "gt_coarse20x10": meta["metrics"]["coarse20x10"]["mean_all_rows"],
            "pair_error_means": a["point_errors"].reshape(len(score), 3, 2).mean((0,2)).tolist(),
            "distance3sec_bins": meta["metrics"]["full_bank"]["distance3sec_bins"],
        }
    json_new(args.output, out)
    print(json.dumps({k:{a:b for a,b in v.items() if a in ("full_oracle_mean", "paired_delta_ci95", "headroom_to_0_15")}
                      for k,v in out["results"].items()}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--partition", choices=("train", "tune"), required=True)
    e.add_argument("--split", required=True); e.add_argument("--ego", required=True)
    e.add_argument("--ego5", required=True); e.add_argument("--metadata", required=True)
    e.add_argument("--scenes-json"); e.add_argument("--min-frame", type=int, default=30)
    e.add_argument("--stride", type=int, default=1); e.add_argument("--path-points", type=int, default=50)
    e.add_argument("--spacing", type=float, default=1.); e.add_argument("--output", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--train", required=True); f.add_argument("--paths", type=int, default=256)
    f.add_argument("--velocities", type=int, default=128); f.add_argument("--seed", type=int, default=20260910)
    f.add_argument("--path-bank"); f.add_argument("--velocity-mode", choices=("absolute8", "progress6"), default="absolute8")
    f.add_argument("--max-iter", type=int, default=150); f.add_argument("--threads", type=int, default=4)
    f.add_argument("--output", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--bank", required=True); o.add_argument("--tune", required=True)
    o.add_argument("--device", default="cpu"); o.add_argument("--rows-chunk", type=int, default=64)
    o.add_argument("--coarse-mode", choices=("native", "d3"), default="native")
    o.add_argument("--candidate-chunk", type=int, default=4096); o.add_argument("--threads", type=int, default=4)
    o.add_argument("--output", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--bank", required=True); d.add_argument("--train", required=True)
    d.add_argument("--tune", required=True); d.add_argument("--output", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--reports", required=True); s.add_argument("--split", required=True)
    s.add_argument("--bootstrap", type=int, default=20000); s.add_argument("--seed", type=int, default=20260910)
    s.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"extract": extract, "fit": fit, "oracle": oracle, "diagnose": diagnose, "summarize": summarize}[args.command](args)
