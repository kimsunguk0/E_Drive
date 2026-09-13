"""Task 1 driver: re-aggregate the pinned parent evaluation and compare to the
numbers reported on 2026-09-11, keeping both definitions side by side."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reaggregate_errors import (NEAR_STATIONARY_M, T, W, along_cross, arclength,
                                bucket_table, concentration, load_records,
                                progress_scale_oracle, row_errors)

SOURCE = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


pred, gt, meta, report = load_records(SOURCE)
point_l2, d3, ade = row_errors(pred, gt)
n = len(d3)
result = {"source": str(SOURCE), "source_sha256": sha(SOURCE), "rows": int(n),
          "sessions": int(len(np.unique(meta["session"]))),
          "reported_official_d3": report.get("official_d3"),
          "recomputed_official_d3": float(d3.mean()),
          "reported_eval_batch": "see parent manifest: 4",
          "max_abs_row_delta_vs_stored_d3": float(np.abs(d3 - meta["reported_d3"]).max())}

# --- per waypoint -----------------------------------------------------------
result["per_waypoint"] = [{
    "t_s": float(t), "weight": float(W[i]),
    "l2_mean": float(point_l2[:, i].mean()),
    "l2_p50": float(np.percentile(point_l2[:, i], 50)),
    "l2_p90": float(np.percentile(point_l2[:, i], 90)),
    "l2_p95": float(np.percentile(point_l2[:, i], 95)),
    "observed_share_of_d3": float(W[i] * point_l2[:, i].mean() / d3.mean()),
} for i, t in enumerate(T)]
result["weight_sums"] = {"first_1s": float(W[:2].sum()), "first_2s": float(W[:4].sum())}

# --- along / cross with validity kept ---------------------------------------
along, cross, tvalid = along_cross(pred, gt)
wl2 = point_l2 * W
directed = float((wl2 * tvalid).sum() / wl2.sum())
result["along_cross"] = {
    "valid_segment_fraction": float(tvalid.mean()),
    "share_of_weighted_error_on_valid_segments": directed,
    "share_of_weighted_error_without_direction": 1.0 - directed,
    "undirected_segments": int((~tvalid).sum()),
    "weighted_abs_along_valid_only": float((np.abs(along) * W).sum(-1).mean()),
    "weighted_abs_cross_valid_only": float((np.abs(cross) * W).sum(-1).mean()),
    "along_share_of_squared_error_valid_only": float(
        ((along ** 2) * W)[tvalid].sum() / (((along ** 2 + cross ** 2) * W)[tvalid].sum())),
    "weighted_signed_along_mean": float((along * W).sum(-1).mean()),
    "weighted_signed_along_median": float(np.median((along * W).sum(-1))),
    "rows_with_positive_signed_along": float(((along * W).sum(-1) > 0).mean()),
    "per_waypoint": [{"t_s": float(t), "valid_fraction": float(tvalid[:, i].mean()),
                      "along_mean": float(along[tvalid[:, i], i].mean()),
                      "along_median": float(np.median(along[tvalid[:, i], i])),
                      "abs_cross_mean": float(np.abs(cross[tvalid[:, i], i]).mean())}
                     for i, t in enumerate(T)],
}
signed = (along * W).sum(-1)

# --- path length, moving rows only ------------------------------------------
progress = arclength(gt)[:, -1]
moving = progress > NEAR_STATIONARY_M
lp, lg = arclength(pred)[:, -1], progress
result["path_length"] = {
    "moving_rows": int(moving.sum()),
    "pred_mean_m": float(lp[moving].mean()), "gt_mean_m": float(lg[moving].mean()),
    "median_ratio": float(np.median(lp[moving] / lg[moving])),
    "fraction_longer": float((lp[moving] > lg[moving]).mean()),
    "note": "ratios are undefined on near-stationary rows and are excluded here",
}

# --- buckets ----------------------------------------------------------------
speed = np.linalg.norm(meta["gt_state"][:, :2], axis=-1)
accel = meta["gt_state"][:, 2]
legacy_bearing = np.abs(np.arctan2(gt[:, -1, 1], np.maximum(gt[:, -1, 0], 1e-6)))
bearing = np.abs(np.arctan2(gt[:, -1, 1], gt[:, -1, 0]))
groups = {
    "speed_lt_0.5": speed < .5, "speed_0.5_5": (speed >= .5) & (speed < 5),
    "speed_5_12": (speed >= 5) & (speed < 12), "speed_12_18": (speed >= 12) & (speed < 18),
    "speed_ge_18": speed >= 18,
    "braking_lt_-0.5": accel < -.5, "steady_pm0.5": np.abs(accel) <= .5,
    "accelerating_gt_0.5": accel > .5,
}
result["buckets_gt_state"] = bucket_table(d3, groups)
for label, mask in groups.items():
    if mask.any():
        result["buckets_gt_state"][label]["weighted_signed_along"] = float(signed[mask].mean())

bearing_groups = {
    "moving_straight_lt5deg": moving & (bearing < np.deg2rad(5)),
    "moving_bend_5_15deg": moving & (bearing >= np.deg2rad(5)) & (bearing < np.deg2rad(15)),
    "moving_turn_ge15deg": moving & (bearing >= np.deg2rad(15)),
    "near_stationary_le_1m": ~moving,
}
result["buckets_endpoint_bearing"] = bucket_table(d3, bearing_groups)
for label, mask in bearing_groups.items():
    if mask.any():
        result["buckets_endpoint_bearing"][label].update(
            weighted_abs_cross=float((np.abs(cross) * W).sum(-1)[mask].mean()),
            weighted_signed_along=float(signed[mask].mean()))
legacy_groups = {
    "legacy_straight_lt5deg": legacy_bearing < np.deg2rad(5),
    "legacy_bend_5_15deg": (legacy_bearing >= np.deg2rad(5)) & (legacy_bearing < np.deg2rad(15)),
    "legacy_turn_ge15deg": legacy_bearing >= np.deg2rad(15),
}
result["legacy_endpoint_bearing_group"] = bucket_table(d3, legacy_groups)
result["bearing_definition_changed_rows"] = int(
    ((legacy_bearing >= np.deg2rad(15)) != (bearing >= np.deg2rad(15))).sum())
result["yaw_note"] = "NOT_AVAILABLE: the artifact stores no future yaw; these are endpoint bearings"

# --- concentration and sessions ---------------------------------------------
result["concentration"] = concentration(d3)
sessions = sorted(np.unique(meta["session"]))
per_session = {s: float(d3[meta["session"] == s].mean()) for s in sessions}
result["session"] = {"min": min(per_session.values()),
                     "median": float(np.median(list(per_session.values()))),
                     "max": max(per_session.values()),
                     "worst": sorted(per_session.items(), key=lambda kv: -kv[1])[:3]}
with open(OUT / "session_d3.csv", "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(["session", "rows", "mean_d3"])
    for s in sessions:
        m = meta["session"] == s
        writer.writerow([s, int(m.sum()), f"{d3[m].mean():.6f}"])

# --- oracles, both grids, reproduced separately ------------------------------
result["progress_scale_oracle"] = {}
for name, grid in (("legacy_decomp_0.5_1.6_111", np.linspace(0.5, 1.6, 111)),
                   ("legacy_decomp2_0.3_2.0_171", np.linspace(0.3, 2.0, 171))):
    o = progress_scale_oracle(pred, gt, grid)
    result["progress_scale_oracle"][name] = {
        "mean_d3": float(o["d3"].mean()), "baseline_mean_d3": float(d3.mean()),
        "scale_median": float(np.median(o["scale"])),
        "scale_p10_p90": [float(np.percentile(o["scale"], 10)), float(np.percentile(o["scale"], 90))],
        "rows_at_scale_one": float(np.isclose(o["scale"], 1.0).mean()),
        "mean_clamped_waypoint_fraction": float(o["clamped_fraction"].mean()),
        "rows_with_any_clamping": float((o["clamped_fraction"] > 0).mean()),
        "degenerate_polylines": int(o["degenerate_polyline"].sum()),
        "non_monotone_arclength": int((~o["monotone_arclength"]).sum()),
        "scale_count": o["scale_count"], "scale_range": o["scale_range"],
    }
    np.savez_compressed(OUT / f"oracle_{name}.npz", **{k: v for k, v in o.items()
                                                       if isinstance(v, np.ndarray)})
result["progress_scale_oracle"]["interpretation"] = (
    "Oracle over one operation whose scale is chosen with the GT. Not a reachable "
    "score, not a pure shape error, and not a post-hoc corrector.")

np.savez_compressed(OUT / "rows.npz", row=meta["row"], session=meta["session"],
                    scenario=meta["scenario"], frame=meta["frame"], d3=d3,
                    point_l2=point_l2, ade123=ade, along=along, cross=cross,
                    tangent_valid=tvalid, gt_progress_m=progress,
                    endpoint_bearing_rad=bearing, legacy_bearing_rad=legacy_bearing,
                    gt_speed=speed, gt_accel=accel)
(OUT / "task1_reaggregated.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: result[k] for k in ("rows", "sessions", "reported_official_d3",
                                         "recomputed_official_d3",
                                         "max_abs_row_delta_vs_stored_d3")}, indent=1))
print("along/cross:", json.dumps({k: v for k, v in result["along_cross"].items()
                                  if not isinstance(v, list)}, indent=1))
print("oracles:", json.dumps({k: {kk: vv for kk, vv in v.items()
                                  if kk in ("mean_d3", "scale_median", "rows_at_scale_one",
                                            "rows_with_any_clamping")}
                              for k, v in result["progress_scale_oracle"].items()
                              if isinstance(v, dict)}, indent=1))
