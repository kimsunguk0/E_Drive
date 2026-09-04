#!/usr/bin/env python
"""Phase F — constant-velocity (CV) and constant-turn-rate-velocity (CTRV) ego baselines.

These answer "how much does perception actually add over a dumb ego prior?".
Everything is evaluated with the SAME protocol the repo uses, so the numbers are
directly comparable to SparseDrive's L2.

Conventions, all read out of tools/data_converter/nuscenes_converter.py:364-410 —
none of this is guessed:

  gt_ego_fut_trajs : (6, 2) PER-STEP DELTAS in the LIDAR frame at t=0.
                     Absolute positions = cumsum. Horizon 3 s at 2 Hz.
                     Built from 7 absolute poses then diffed (line 394).
  gt_ego_fut_masks : (6,)  0 for steps padded past the end of a scene.
                     The repo evaluator SKIPS a sample unless mask.all(), so we do too.
  lidar frame      : +x right, +y forward (ego motion is +y).
  gt_ego_fut_cmd   : right / left / straight, from the 3 s x-offset (±2 m).
  ego_status       : [0:3] accel (incl. gravity), [3:6] rotation_rate rad/s,
                     [6:9] velocity m/s in ego frame, [9] steering angle.
                     **All ten are 0 when the scene has no CAN bus data**
                     (`except: ego_status = [0]*10`, line 436), so CAN cannot be
                     the primary velocity source — it silently means "stopped".

Velocity estimation therefore prefers finite-differencing the ego pose of the
PREVIOUS keyframe (always available, and it is the same quantity the GT is built
from), and only falls back to CAN. Both are reported so the disagreement is visible.

Usage:
  python ego_baselines.py --pkl <nuscenes_infos_val.pkl> --dataroot <nuscenes root> \
                          --out <baselines.md>
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from challenge_metrics import evaluate_trajectories, format_table  # noqa: E402

DT = 0.5          # 2 Hz keyframes
N_STEPS = 6       # 3 s horizon


def integrate_cv(vx: float, vy: float) -> np.ndarray:
    """Constant velocity: straight line, no yaw change. Returns (6,2) absolute."""
    t = np.arange(1, N_STEPS + 1) * DT
    return np.stack([vx * t, vy * t], axis=1)


def integrate_ctrv(v: float, yaw_rate: float) -> np.ndarray:
    """Constant turn rate + constant speed, starting heading = +y (lidar forward).

    Heading is measured from +y toward -x for a positive (counter-clockwise) yaw
    rate, matching the lidar frame's right-handed z-up convention.
    Closed form for |w| > eps, straight-line limit otherwise.
    """
    t = np.arange(1, N_STEPS + 1) * DT
    if abs(yaw_rate) < 1e-6:
        return np.stack([np.zeros_like(t), v * t], axis=1)
    th = yaw_rate * t
    # arc starting at origin heading +y, turning by th
    x = -(v / yaw_rate) * (1.0 - np.cos(th))
    y = (v / yaw_rate) * np.sin(th)
    return np.stack([x, y], axis=1)


def ego_pose_velocity(nusc, info) -> tuple[float, float] | None:
    """Velocity in the CURRENT lidar frame from the previous keyframe's ego pose.

    Returns None at the first sample of a scene (no previous keyframe).
    """
    from pyquaternion import Quaternion

    sample = nusc.get("sample", info["token"])
    if sample["prev"] == "":
        return None
    prev = nusc.get("sample", sample["prev"])

    def lidar_pose(s):
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        e2g_t = np.array(pose["translation"])
        e2g_r = Quaternion(pose["rotation"])
        s2e_t = np.array(cs["translation"])
        s2e_r = Quaternion(cs["rotation"])
        # lidar origin in global
        p = e2g_r.rotation_matrix @ s2e_t + e2g_t
        R = e2g_r.rotation_matrix @ s2e_r.rotation_matrix
        return p, R, s["timestamp"]

    p_cur, R_cur, t_cur = lidar_pose(sample)
    p_prev, _, t_prev = lidar_pose(prev)
    dt = (t_cur - t_prev) / 1e6
    if dt <= 0:
        return None
    # displacement of the previous origin as seen from the current lidar frame
    d_global = p_cur - p_prev
    d_local = R_cur.T @ d_global
    return float(d_local[0] / dt), float(d_local[1] / dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--dataroot", default="/tmp/pm97/data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        infos = pickle.load(f)["infos"]
    if args.limit:
        infos = infos[: args.limit]
    print(f"pkl: {args.pkl}  ({len(infos)} samples)")

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    gt, cv_pose, cv_can, ctrv, kept = [], [], [], [], 0
    n_no_prev = n_can_zero = n_masked_out = 0
    v_pose_all, v_can_all = [], []

    for info in infos:
        mask = np.asarray(info["gt_ego_fut_masks"], dtype=bool)
        # Same rule as the repo evaluator: incomplete GT -> sample excluded entirely.
        if not mask.all():
            n_masked_out += 1
            continue

        es = np.asarray(info["ego_status"], dtype=np.float64)
        can_zero = not np.any(es)
        if can_zero:
            n_can_zero += 1

        pv = ego_pose_velocity(nusc, info)
        if pv is None:
            n_no_prev += 1
            # first frame of a scene: fall back to CAN, else assume stopped
            vx, vy = (0.0, float(es[6])) if not can_zero else (0.0, 0.0)
        else:
            vx, vy = pv

        # CAN longitudinal velocity; ego +x forward in the CAN frame maps to lidar +y
        v_can = float(es[6])
        yaw_rate = float(es[5])

        gt.append(np.asarray(info["gt_ego_fut_trajs"], dtype=np.float64).cumsum(0))
        cv_pose.append(integrate_cv(vx, vy))
        cv_can.append(integrate_cv(0.0, v_can))
        ctrv.append(integrate_ctrv(np.hypot(vx, vy) if pv else v_can, yaw_rate))
        v_pose_all.append(np.hypot(vx, vy))
        v_can_all.append(v_can)
        kept += 1

    gt = np.stack(gt)
    res = {
        "CV (ego-pose diff)": evaluate_trajectories(np.stack(cv_pose), gt),
        "CV (CAN vel)": evaluate_trajectories(np.stack(cv_can), gt),
        "CTRV (pose speed + CAN yaw-rate)": evaluate_trajectories(np.stack(ctrv), gt),
    }

    vp, vc = np.array(v_pose_all), np.array(v_can_all)
    hdr = (
        f"# Ego prior baselines (Phase F)\n\n"
        f"- pkl: `{args.pkl}`\n"
        f"- samples in pkl: **{len(infos)}**, evaluated: **{kept}**, "
        f"excluded by incomplete GT mask: **{n_masked_out}** "
        f"(same exclusion rule as the repo evaluator)\n"
        f"- first-frame-of-scene (no previous keyframe): {n_no_prev}\n"
        f"- **samples with all-zero `ego_status` (no CAN bus): {n_can_zero}** "
        f"— this is why ego-pose differencing is the primary velocity source; "
        f"CAN reads as 'stopped' for these.\n"
        f"- speed agreement: mean |pose − CAN| = {np.abs(vp - vc).mean():.3f} m/s, "
        f"max = {np.abs(vp - vc).max():.3f} m/s, corr = {np.corrcoef(vp, vc)[0,1]:.4f}\n"
        f"- horizon 3 s @ 2 Hz (6 steps), lidar frame (+y forward)\n\n"
        f"> Two L2 conventions are reported for every row. They are NOT the same "
        f"number — see `src/challenge_metrics.py`. SparseDrive's paper 0.61 is the "
        f"UniAD-style `avg`.\n\n"
    )
    body = "".join(format_table(k, v) + "\n" for k, v in res.items())

    tab = ("| baseline | L2 1s | L2 2s | L2 3s | **avg (UniAD)** | **mean (challenge)** |\n"
           "|---|---|---|---|---|---|\n")
    for k, v in res.items():
        u, c = v["uniad"], v["challenge"]
        tab += (f"| {k} | {u['L2_1s']:.4f} | {u['L2_2s']:.4f} | {u['L2_3s']:.4f} "
                f"| **{u['L2_avg']:.4f}** | **{c['L2_mean']:.4f}** |\n")
    tab += ("| _SparseDrive paper (reference)_ | 0.29 | 0.58 | 0.96 | **0.61** | _n/a_ |\n"
            "| _DiffusionDrive paper (reference)_ | 0.27 | 0.54 | 0.90 | **0.57** | _n/a_ |\n")

    out = hdr + tab + "\n" + body
    print("\n" + tab)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
