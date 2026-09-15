#!/usr/bin/env python3
"""E1-EXP 0.2256 diagnosis: where the remaining error sits.

Re-aggregation only.  No training, no H, no split change, no post-processing of
the model output.  Every number comes from predictions already stored by the
evaluation runs, plus one history dump taken under the same conditions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "reports/md_exp_diagnosis_20260915"
WORK = ROOT / "work_dirs/md_r0_reset_20260914"
EXP_EVAL = WORK / "E1-EXP/final_eval.json"
R0_EVAL = ROOT / "work_dirs/motiondrive_v2/q10_q10_flip50_s0/final_eval.json"
PROBE_EVAL = WORK / "evals/E1-EXP_probe_step20554/evaluation.json"
HISTORY_NPZ = OUT / "exp_v0_history.npz"
CANDIDATE = ROOT / "reports/md_r0_reset_20260914/candidate_registry.json"

W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36.0
DT = 0.5
TIMES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# The corrected report flagged segments under 1 cm as direction-noise; this is a
# diagnostic threshold, not a sensor-noise guarantee.
DIRECTION_MIN_M = 0.01
# Fixed a-priori subset boundaries, chosen before looking at the split results.
VX_GOOD_MS = 0.5
HISTORY_GOOD_M = 0.25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["session"], records[i]["scenario"], int(records[i]["frame"])))
    records = [records[i] for i in order]
    out = {
        "row": np.asarray([int(r["row"]) for r in records]),
        "session": np.asarray([r["session"] for r in records]),
        "scenario": np.asarray([r["scenario"] for r in records]),
        "frame": np.asarray([int(r["frame"]) for r in records]),
        "pred": np.asarray([r["pred_abs_xy"] for r in records], dtype=np.float64),
        "gt": np.asarray([r["gt_abs_xy"] for r in records], dtype=np.float64),
        "d3_stored": np.asarray([r["d3"] for r in records], dtype=np.float64),
    }
    if "pred_state" in records[0]:
        out["pred_state"] = np.asarray([r["pred_state"] for r in records], dtype=np.float64)
        out["gt_state"] = np.asarray([r["gt_state"] for r in records], dtype=np.float64)
        out["state_valid"] = np.asarray([r["gt_state_valid"] for r in records], dtype=bool)
    return out


def segments(p):
    """Chord displacement per 0.5 s interval, starting from the ego origin."""
    start = np.concatenate([np.zeros((len(p), 1, 2)), p[:, :-1]], axis=1)
    dp = p - start
    ell = np.linalg.norm(dp, axis=-1)
    return dp, ell


def point_error_budget(exp, r0):
    e_exp = np.linalg.norm(exp["pred"] - exp["gt"], axis=-1)
    e_r0 = np.linalg.norm(r0["pred"] - r0["gt"], axis=-1)
    d3_exp, d3_r0 = e_exp @ W, e_r0 @ W
    rows = []
    for k in range(6):
        contribution = W[k] * e_exp[:, k].mean()
        rows.append({
            "t_s": TIMES[k], "weight": W[k],
            "exp_mean_l2_m": e_exp[:, k].mean(),
            "exp_p90_l2_m": np.percentile(e_exp[:, k], 90),
            "exp_p95_l2_m": np.percentile(e_exp[:, k], 95),
            "exp_d3_contribution": contribution,
            "exp_share_of_d3": contribution / d3_exp.mean(),
            "r0_mean_l2_m": e_r0[:, k].mean(),
            "r0_d3_contribution": W[k] * e_r0[:, k].mean(),
            "improvement_mean_l2_m": e_exp[:, k].mean() - e_r0[:, k].mean(),
            "improvement_d3_contribution": W[k] * (e_exp[:, k].mean() - e_r0[:, k].mean()),
        })
    check = {
        "sum_of_contributions": float(sum(r["exp_d3_contribution"] for r in rows)),
        "mean_d3": float(d3_exp.mean()),
        "max_abs_diff": abs(float(sum(r["exp_d3_contribution"] for r in rows)) - float(d3_exp.mean())),
        "stored_d3_max_row_diff": float(np.abs(d3_exp - exp["d3_stored"]).max()),
        "exp_plain_ade6_m": float(e_exp.mean()),
        "r0_plain_ade6_m": float(e_r0.mean()),
        "exp_official_d3": float(d3_exp.mean()),
        "r0_official_d3": float(d3_r0.mean()),
    }
    return rows, check, e_exp, d3_exp


def progress_and_direction(exp):
    dp_p, ell_p = segments(exp["pred"])
    dp_g, ell_g = segments(exp["gt"])
    defined = ell_g > DIRECTION_MIN_M
    with np.errstate(invalid="ignore", divide="ignore"):
        u_g = np.where(defined[..., None], dp_g / np.maximum(ell_g, 1e-12)[..., None], np.nan)
    # along/cross of the per-segment displacement error, on GT-defined directions
    err = dp_p - dp_g
    along = np.where(defined, (err * np.nan_to_num(u_g)).sum(-1), np.nan)
    cross = np.where(defined, err[..., 0] * np.nan_to_num(u_g)[..., 1]
                     - err[..., 1] * np.nan_to_num(u_g)[..., 0], np.nan)
    heading_err = np.full_like(along, np.nan)
    both = defined & (ell_p > DIRECTION_MIN_M)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_p = np.where(both[..., None], dp_p / np.maximum(ell_p, 1e-12)[..., None], np.nan)
    dot = np.clip((np.nan_to_num(u_p) * np.nan_to_num(u_g)).sum(-1), -1, 1)
    heading_err = np.where(both, np.degrees(np.arccos(dot)), np.nan)

    rows = []
    for k in range(6):
        d = defined[:, k]
        rows.append({
            "t_s": TIMES[k],
            "segment_index": k,
            "gt_chord_mean_m": ell_g[:, k].mean(),
            "pred_chord_mean_m": ell_p[:, k].mean(),
            "chord_error_mean_m": (ell_p[:, k] - ell_g[:, k]).mean(),
            "chord_abs_error_mean_m": np.abs(ell_p[:, k] - ell_g[:, k]).mean(),
            "chord_abs_error_p95_m": np.percentile(np.abs(ell_p[:, k] - ell_g[:, k]), 95),
            "vbar_gt_mean_ms": (ell_g[:, k] / DT).mean(),
            "vbar_error_mean_ms": ((ell_p[:, k] - ell_g[:, k]) / DT).mean(),
            "vbar_abs_error_mean_ms": (np.abs(ell_p[:, k] - ell_g[:, k]) / DT).mean(),
            "direction_defined_fraction": float(d.mean()),
            "along_abs_mean_m": float(np.nanmean(np.abs(along[:, k]))) if d.any() else np.nan,
            "cross_abs_mean_m": float(np.nanmean(np.abs(cross[:, k]))) if d.any() else np.nan,
            "heading_error_mean_deg": float(np.nanmean(heading_err[:, k])),
            "heading_error_p95_deg": float(np.nanpercentile(heading_err[:, k], 95)),
        })
    cumulative = {
        "cumulative_gt_chord_mean_m": float(ell_g.sum(1).mean()),
        "cumulative_pred_chord_mean_m": float(ell_p.sum(1).mean()),
        "cumulative_chord_error_mean_m": float((ell_p.sum(1) - ell_g.sum(1)).mean()),
        "cumulative_chord_abs_error_mean_m": float(np.abs(ell_p.sum(1) - ell_g.sum(1)).mean()),
        "rows_with_all_segments_direction_defined": int(defined.all(1).sum()),
        "rows_total": int(len(ell_g)),
        "note": ("chord displacement per 0.5 s interval, not continuous arc length; "
                 "vbar is a discrete segment-average speed proxy, not the state head's "
                 "instantaneous vx"),
        "along_cross_caveat": ("along and cross are squared-error style components of the "
                               "segment displacement error on GT-defined directions; "
                               "|along| + |cross| is NOT an additive decomposition of D3"),
    }
    dvbar_p = np.diff(ell_p / DT, axis=1) / DT
    dvbar_g = np.diff(ell_g / DT, axis=1) / DT
    return rows, cumulative, ell_p, ell_g, dvbar_p, dvbar_g, defined, along, cross, heading_err


def regimes(exp, ell_g):
    """Mutually exclusive regimes from the GT future trajectory and state labels."""
    total = ell_g.sum(1)
    vbar = ell_g / DT
    change = vbar[:, -1] - vbar[:, 0]
    dv = np.diff(vbar, axis=1)
    has_decel = (dv < -0.5).any(1)
    has_accel = (dv > 0.5).any(1)
    label = np.full(len(total), "constant", dtype=object)
    label[(change <= -1.0) & ~has_accel] = "decelerating"
    label[(change >= 1.0) & ~has_decel] = "accelerating"
    label[has_decel & has_accel] = "transition"
    label[(total > 1.0) & (vbar[:, 0] < 1.0)] = "departing"
    label[total <= 1.0] = "stop_hold"
    return label


def regime_budget(label, d3, e, ell_p, ell_g, heading_err, sessions):
    total_d3 = d3.sum()
    rows = []
    for name in ["stop_hold", "departing", "constant", "decelerating", "accelerating", "transition"]:
        mask = label == name
        if not mask.any():
            continue
        rows.append({
            "regime": name, "rows": int(mask.sum()),
            "row_share": float(mask.mean()),
            "sessions": int(len(set(sessions[mask].tolist()))),
            "mean_d3_m": float(d3[mask].mean()),
            "share_of_total_d3": float(d3[mask].sum() / total_d3),
            **{f"l2_t{TIMES[k]:g}_m": float(e[mask, k].mean()) for k in range(6)},
            "chord_error_mean_m": float((ell_p[mask] - ell_g[mask]).sum(1).mean()),
            "chord_abs_error_mean_m": float(np.abs(ell_p[mask] - ell_g[mask]).sum(1).mean()),
            "heading_error_mean_deg": float(np.nanmean(heading_err[mask])),
        })
    return rows


def head_vs_plan(exp, ell_p, ell_g, dvbar_p, dvbar_g, d3, history):
    vx_err = exp["pred_state"][:, 0] - exp["gt_state"][:, 0]
    vx_valid = exp["state_valid"][:, 0]
    first_pred = exp["pred"][:, 0] / DT
    first_gt = exp["gt"][:, 0] / DT
    first_speed_err = np.linalg.norm(first_pred, axis=-1) - np.linalg.norm(first_gt, axis=-1)
    vbar0_err = (ell_p[:, 0] - ell_g[:, 0]) / DT

    result = {
        "state_vx_mae_ms": float(np.abs(vx_err[vx_valid]).mean()),
        "state_vx_bias_ms": float(vx_err[vx_valid].mean()),
        "state_vx_valid_rows": int(vx_valid.sum()),
        "first_segment_mean_speed_error_ms": float(first_speed_err.mean()),
        "first_segment_mean_abs_speed_error_ms": float(np.abs(first_speed_err).mean()),
        "vbar0_error_mean_ms": float(vbar0_err.mean()),
        "vbar0_abs_error_mean_ms": float(np.abs(vbar0_err).mean()),
        "corr_state_vx_error_with_vbar0_error": float(
            np.corrcoef(vx_err[vx_valid], vbar0_err[vx_valid])[0, 1]),
        "dvbar_abs_error_mean_ms2": [float(np.abs(dvbar_p[:, k] - dvbar_g[:, k]).mean())
                                     for k in range(dvbar_p.shape[1])],
        "definitions": {
            "first_segment_mean_speed": ("gt_p[0]/0.5 is the mean velocity over the first "
                                         "0.5 s, not the instantaneous velocity at t0"),
            "state_label": ("vx comes from a causal quadratic fit over <=1.001 s of past pose, "
                            "so it is a fitted quantity, not a sensor reading"),
            "correlation": ("a correlation between head error and plan error is diagnostic "
                            "evidence, not causal proof; the planner also reads scene and "
                            "motion features directly"),
        },
    }
    if history is not None:
        order = np.argsort(history["row"])
        lookup = {int(r): i for i, r in enumerate(history["row"][order])}
        index = np.asarray([lookup[int(r)] for r in exp["row"]])
        hat = history["history_hat"][order][index]
        target = history["history_target"][order][index]
        valid = history["history_valid"][order][index]
        position_err = np.linalg.norm(hat[..., :2] - target[..., :2], axis=-1)
        yaw_hat = np.arctan2(hat[..., 2], hat[..., 3])
        yaw_gt = np.arctan2(target[..., 2], target[..., 3])
        yaw_err = np.degrees(np.abs(np.arctan2(np.sin(yaw_hat - yaw_gt), np.cos(yaw_hat - yaw_gt))))
        ok = valid[..., :2].all(-1)
        offsets = history["history_frame_offsets"].tolist()
        seconds = history["nominal_history_seconds"].tolist()
        result["history"] = {
            "frame_offsets": offsets, "nominal_seconds": seconds,
            "position_mae_m_by_offset": [float(position_err[:, j][ok[:, j]].mean())
                                         for j in range(position_err.shape[1])],
            "yaw_mae_deg_by_offset": [float(yaw_err[:, j][ok[:, j]].mean())
                                      for j in range(yaw_err.shape[1])],
            "valid_fraction_by_offset": [float(ok[:, j].mean()) for j in range(ok.shape[1])],
            "target_definition": "[dx, dy, sin(yaw), cos(yaw)] of each past pose in the current ego frame",
        }
        history_score = position_err[:, -1]
    else:
        result["history"] = "NOT_AVAILABLE"
        history_score = None

    subsets = {}
    good_vx = vx_valid & (np.abs(vx_err) <= VX_GOOD_MS)
    subsets["state_vx_within_0.5ms"] = good_vx
    subsets["state_vx_beyond_0.5ms"] = vx_valid & ~good_vx
    if history_score is not None:
        good_h = history_score <= HISTORY_GOOD_M
        subsets["history_1s_within_0.25m"] = good_h
        subsets["history_1s_beyond_0.25m"] = ~good_h
    result["subset_d3"] = {
        name: {"rows": int(mask.sum()), "mean_d3_m": float(d3[mask].mean()),
               "share_of_total_d3": float(d3[mask].sum() / d3.sum())}
        for name, mask in subsets.items() if mask.any()}
    result["subset_boundaries"] = {"vx_good_ms": VX_GOOD_MS, "history_good_m": HISTORY_GOOD_M,
                                   "fixed_before_looking_at_the_split": True}
    return result


def gt_factor_substitution(exp, ell_p, ell_g, defined):
    """Section 6: swap GT segment lengths or GT directions, on the moving subset."""
    subset = defined.all(1) & (np.linalg.norm(exp["pred"][:, 0], axis=-1) > DIRECTION_MIN_M)
    if not subset.any():
        return {"status": "NO_ROWS"}
    dp_p, _ = segments(exp["pred"])
    dp_g, _ = segments(exp["gt"])
    u_p = dp_p / np.maximum(np.linalg.norm(dp_p, axis=-1), 1e-12)[..., None]
    u_g = dp_g / np.maximum(np.linalg.norm(dp_g, axis=-1), 1e-12)[..., None]
    q = np.cumsum(ell_g[..., None] * u_p, axis=1)          # GT lengths, predicted directions
    r = np.cumsum(ell_p[..., None] * u_g, axis=1)          # predicted lengths, GT directions
    check = np.cumsum(ell_g[..., None] * u_g, axis=1)      # must reconstruct GT
    def d3_of(x):
        return (np.linalg.norm(x[subset] - exp["gt"][subset], axis=-1) @ W).mean()
    baseline = d3_of(exp["pred"])
    return {
        "subset_rows": int(subset.sum()), "subset_share": float(subset.mean()),
        "baseline_d3_on_subset": float(baseline),
        "gt_lengths_predicted_directions_d3": float(d3_of(q)),
        "predicted_lengths_gt_directions_d3": float(d3_of(r)),
        "gt_lengths_gt_directions_reconstruction_max_error_m": float(
            np.abs(check[subset] - exp["gt"][subset]).max()),
        "rows_worsened_by_gt_lengths": int((np.linalg.norm(q[subset] - exp["gt"][subset], axis=-1) @ W
                                            > np.linalg.norm(exp["pred"][subset] - exp["gt"][subset], axis=-1) @ W).sum()),
        "rows_worsened_by_gt_directions": int((np.linalg.norm(r[subset] - exp["gt"][subset], axis=-1) @ W
                                               > np.linalg.norm(exp["pred"][subset] - exp["gt"][subset], axis=-1) @ W).sum()),
        "caveats": [
            "this is a segment length and direction sensitivity probe, not a physical retiming of a fixed path",
            "the two improvements are not additive and error cancellation is removed, so some rows get worse",
            "it is not a reachable score and not a percentage attribution of cause",
            "rows whose GT segments fall below the direction threshold are excluded, not given a fallback direction",
        ],
    }


def pick_clips(exp, ell_p, ell_g, d3, label, heading_err):
    chord_err = np.abs(ell_p[:, 0] - ell_g[:, 0])
    late_err = np.abs(ell_p[:, 3:] - ell_g[:, 3:]).sum(1)
    picks, used = {}, set()

    def take(order, n, name):
        chosen = []
        for i in order:
            key = (exp["session"][i], exp["scenario"][i])
            if i in used:
                continue
            if sum(1 for c in chosen if (exp["session"][c], exp["scenario"][c]) == key) >= 2:
                continue
            chosen.append(int(i))
            used.add(int(i))
            if len(chosen) == n:
                break
        picks[name] = [{"row": int(exp["row"][i]), "session": str(exp["session"][i]),
                        "scenario": str(exp["scenario"][i]), "frame": int(exp["frame"][i]),
                        "d3": float(d3[i]), "regime": str(label[i]),
                        "first_chord_error_m": float(ell_p[i, 0] - ell_g[i, 0]),
                        "late_chord_abs_error_m": float(late_err[i]),
                        "heading_error_mean_deg": float(np.nanmean(heading_err[i]))}
                       for i in chosen]

    take(np.argsort(-chord_err), 8, "first_segment_progress")
    take(np.argsort(-np.where(chord_err < np.percentile(chord_err, 50), late_err, -1)), 8,
         "late_acceleration")
    direction_or_stop = np.where(np.isin(label, ["stop_hold", "departing"]), d3,
                                 np.nan_to_num(np.nanmean(heading_err, axis=1)) / 100.0 * d3)
    take(np.argsort(-direction_or_stop), 8, "stop_depart_direction")
    picks["note"] = ("automatically selected worst cases for cause finding; this is not a "
                     "frequency-representative sample")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    exp = load(EXP_EVAL)
    r0 = load(R0_EVAL)
    if not np.array_equal(exp["row"], r0["row"]):
        raise SystemExit("EXP and R0 are not scored on the same rows")
    probe = load(PROBE_EVAL)
    history = dict(np.load(HISTORY_NPZ, allow_pickle=False)) if HISTORY_NPZ.exists() else None

    budget, check, e_exp, d3 = point_error_budget(exp, r0)
    prog, cumulative, ell_p, ell_g, dvbar_p, dvbar_g, defined, along, cross, heading = \
        progress_and_direction(exp)
    label = regimes(exp, ell_g)
    regime_rows = regime_budget(label, d3, e_exp, ell_p, ell_g, heading, exp["session"])
    heads = head_vs_plan(exp, ell_p, ell_g, dvbar_p, dvbar_g, d3, history)
    substitution = gt_factor_substitution(exp, ell_p, ell_g, defined)
    clips = pick_clips(exp, ell_p, ell_g, d3, label, heading)

    # the same point budget on the fixed T0 probe rows, for contrast
    e_probe = np.linalg.norm(probe["pred"] - probe["gt"], axis=-1)
    probe_budget = [{"t_s": TIMES[k], "mean_l2_m": float(e_probe[:, k].mean()),
                     "d3_contribution": float(W[k] * e_probe[:, k].mean())} for k in range(6)]

    candidate = json.loads(CANDIDATE.read_text())
    identity = {
        "schema_version": 1,
        "purpose": "limited diagnosis of the E1-EXP candidate; no training, no H, no new score",
        "target": {
            "run": "E1-EXP seed0 short terminal (step 20,554)",
            "checkpoint": candidate["checkpoint_path"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "model_state_sha256": candidate["model_state_sha256"],
            "reported_v0_d3": candidate["results"]["v0_b4_official_d3"],
            "runtime_source_git_sha": candidate["training"]["runtime_source_git_sha"],
        },
        "rows": {
            "v0": {"n": int(len(exp["row"])), "sessions": int(len(set(exp["session"].tolist()))),
                   "source": str(EXP_EVAL), "sha256": sha(EXP_EVAL)},
            "t0_probe": {"n": int(len(probe["row"])),
                         "sessions": int(len(set(probe["session"].tolist()))),
                         "source": str(PROBE_EVAL), "sha256": sha(PROBE_EVAL)},
            "r0_comparison": {"n": int(len(r0["row"])), "source": str(R0_EVAL), "sha256": sha(R0_EVAL)},
            "identical_rows_exp_vs_r0": True,
        },
        "history_dump": ({"source": str(HISTORY_NPZ), "sha256": sha(HISTORY_NPZ),
                          "note": "one extra evaluation pass under the same conditions; "
                                  "the stored records carry plan and state but not history"}
                         if history is not None else "NOT_AVAILABLE"),
        "latency_as_reported": {
            "value_ms": 16.272,
            "measured_here": True,
            "conditions": ("RTX 4090, batch 1, bf16, one top-level forward per clip that "
                           "internally consumes the four past frames; model forward only, "
                           "input preparation excluded; warmup 30 and 200 repetitions"),
            "artifact": "reports/md_r0_reset_20260914/rtx4090_forward_cost.json",
        },
        "not_available": [
            "command aligned to V0 rows was not joined in this pass; the training side carries "
            "a per-frame series and the test side a single reduced row, and no verified join "
            "exists yet (see command_inventory.json)",
        ],
        "not_copied_from_r0": [
            "the 77.47 percent along-track share and the 0.1255 resampling oracle belong to R0 "
            "and are not reused here; the oracle also included endpoint clamping and is not a "
            "timing/shape split",
        ],
    }

    (OUT / "identity.json").write_text(json.dumps(identity, indent=1, sort_keys=True) + "\n")

    def write_csv(name, rows):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("point_error_budget.csv", budget)
    write_csv("progress_and_direction.csv", prog)
    write_csv("regime_error_budget.csv", regime_rows)
    write_csv("head_vs_plan.csv", [{"metric": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                                   for k, v in heads.items()])

    summary = {
        "point_error_budget": {"rows": budget, "checks": check, "t0_probe": probe_budget},
        "progress_and_direction": {"per_segment": prog, "cumulative": cumulative},
        "regime_error_budget": {"rows": regime_rows,
                                "mutually_exclusive": True,
                                "classification_uses_future_gt": True,
                                "note": ("regimes are labelled offline from the GT future "
                                         "trajectory; that labelling never reaches a model input")},
        "head_vs_plan": heads,
        "gt_factor_substitution": substitution,
        "review_clips": clips,
    }
    (OUT / "diagnosis.json").write_text(json.dumps(summary, indent=1, sort_keys=True,
                                                   default=float) + "\n")
    print(json.dumps({
        "d3": check["exp_official_d3"], "plain_ade6": check["exp_plain_ade6_m"],
        "contribution_check_max_diff": check["max_abs_diff"],
        "share_of_d3_by_timestep": [round(r["exp_share_of_d3"], 4) for r in budget],
        "improvement_vs_r0_by_timestep": [round(r["improvement_d3_contribution"], 5) for r in budget],
        "chord_error_mean_by_segment": [round(r["chord_error_mean_m"], 4) for r in prog],
        "regimes": [{k: r[k] for k in ("regime", "rows", "mean_d3_m", "share_of_total_d3")}
                    for r in regime_rows],
        "substitution": {k: substitution[k] for k in substitution if k != "caveats"},
    }, indent=1, default=float))


if __name__ == "__main__":
    main()
