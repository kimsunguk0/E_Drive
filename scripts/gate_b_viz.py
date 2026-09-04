#!/usr/bin/env python
"""Gate B — prove the info pkl's coordinate frames are right, visually.

A pkl that loads fine and has the right shapes can still be geometrically wrong
(swapped w/l, center-vs-bottom z, missing inverse on sensor2lidar). That kind of
bug does not crash training; it just quietly caps accuracy. So this projects real
GT 3D boxes onto all 6 camera images and draws a BEV view, and a human looks at it.

Conventions read out of tools/data_converter/nuscenes_converter.py:300-334 —
not guessed:
  gt_boxes[:, 0:3] = b.center          -> box CENTER in the LIDAR frame
  gt_boxes[:, 3:6] = b.wlh[[1, 0, 2]]  -> [x_size, y_size, z_size] == [l, w, h]
  gt_boxes[:, 6]   = yaw about lidar z
  cams[c]['sensor2lidar_rotation'/'_translation'] map CAMERA -> LIDAR, i.e.
      p_lidar = R @ p_cam + t
  so projection needs the inverse:
      p_cam = R.T @ (p_lidar - t)

Ego future trajectory is stored as PER-STEP DELTAS; cumsum gives absolute
positions (same as the evaluator's `gt_ego_fut_trajs.cumsum(dim=-2)`).

Usage:
  python gate_b_viz.py --pkl <infos.pkl> --out <dir> [--n 2] [--index 0]
"""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# 12 edges of a cuboid given the corner ordering in box_corners()
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # bottom face
         (4, 5), (5, 6), (6, 7), (7, 4),      # top face
         (0, 4), (1, 5), (2, 6), (3, 7)]      # verticals

COLORS = {
    "car": "#4E79A7", "truck": "#F28E2B", "bus": "#E15759",
    "trailer": "#76B7B2", "construction_vehicle": "#59A14F",
    "pedestrian": "#EDC948", "motorcycle": "#B07AA1",
    "bicycle": "#FF9DA7", "traffic_cone": "#9C755F", "barrier": "#BAB0AC",
}


def box_corners(box: np.ndarray) -> np.ndarray:
    """(7,) [cx,cy,cz,xs,ys,zs,yaw] -> (8,3) corners in the LIDAR frame.

    Bottom face first (z-), then top (z+), each counter-clockwise starting at
    (+x,+y). Center-based, matching b.center from the converter.
    """
    cx, cy, cz, xs, ys, zs, yaw = box[:7]
    x, y, z = xs / 2.0, ys / 2.0, zs / 2.0
    local = np.array([
        [+x, +y, -z], [+x, -y, -z], [-x, -y, -z], [-x, +y, -z],
        [+x, +y, +z], [+x, -y, +z], [-x, -y, +z], [-x, +y, +z],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + np.array([cx, cy, cz])


def lidar_to_cam(pts_lidar: np.ndarray, cam: dict) -> np.ndarray:
    """sensor2lidar maps cam->lidar, so invert it to go lidar->cam."""
    R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
    return (pts_lidar - t) @ R          # == (R.T @ (p - t).T).T


def project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    uvw = pts_cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def resolve(path: str, data_root: str) -> str:
    """pkl stores './data/nuscenes/...' relative to the repo; remap to a real root."""
    p = path.replace("\\", "/")
    for marker in ("/samples/", "/sweeps/"):
        if marker in p:
            return os.path.join(data_root, p.split(marker, 1)[0].split("/")[-1]
                                if False else marker.strip("/") + "/" + p.split(marker, 1)[1])
    return p


def draw_cameras(info: dict, data_root: str, out_png: str, min_dist: float = 1.0) -> int:
    boxes, names = info["gt_boxes"], info["gt_names"]
    fig, axes = plt.subplots(2, 3, figsize=(24, 9))
    drawn_total = 0

    for ax, cname in zip(axes.ravel(), CAM_ORDER):
        cam = info["cams"][cname]
        img_path = resolve(cam["data_path"], data_root)
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            ax.set_title(f"{cname}\nMISSING {img_path}", fontsize=8, color="red")
            ax.axis("off")
            continue
        W, H = img.size
        ax.imshow(img)
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)

        n_drawn = 0
        for b, nm in zip(boxes, names):
            c_lidar = box_corners(b)
            c_cam = lidar_to_cam(c_lidar, cam)
            if (c_cam[:, 2] < min_dist).any():      # any corner behind/at the lens
                continue
            uv = project(c_cam, K)
            if uv[:, 0].max() < 0 or uv[:, 0].min() > W or \
               uv[:, 1].max() < 0 or uv[:, 1].min() > H:
                continue
            col = COLORS.get(str(nm), "#FFFFFF")
            for i, j in EDGES:
                ax.plot(uv[[i, j], 0], uv[[i, j], 1], color=col, linewidth=1.2)
            n_drawn += 1
        drawn_total += n_drawn
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"{cname}   {n_drawn} boxes", fontsize=10)

    fig.suptitle(f"GT 3D boxes projected onto 6 cameras — token {info['token'][:12]}  "
                 f"({len(boxes)} annotated, {drawn_total} visible)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return drawn_total


def draw_bev(info: dict, out_png: str, rng: float = 60.0) -> None:
    boxes, names = info["gt_boxes"], info["gt_names"]
    fig, ax = plt.subplots(figsize=(10, 10))

    for b, nm in zip(boxes, names):
        c = box_corners(b)[:4, :2]              # bottom face footprint
        ax.add_patch(MplPolygon(c, closed=True, fill=True, alpha=0.45,
                                facecolor=COLORS.get(str(nm), "#888888"),
                                edgecolor="black", linewidth=0.6))
        # heading tick so a wrong yaw is obvious
        cx, cy, yaw, xs = b[0], b[1], b[6], b[3]
        ax.plot([cx, cx + np.cos(yaw) * xs * 0.7],
                [cy, cy + np.sin(yaw) * xs * 0.7], color="black", linewidth=0.8)

    # ego at origin, 4.08 x 1.73 m (nuScenes ego footprint)
    ego = box_corners(np.array([0, 0, 0, 4.084, 1.730, 1.562, 0.0]))[:4, :2]
    ax.add_patch(MplPolygon(ego, closed=True, fill=True, facecolor="red",
                            edgecolor="darkred", alpha=0.9, label="ego"))

    # ego future trajectory: stored as deltas -> cumsum
    fut = np.asarray(info["gt_ego_fut_trajs"], dtype=np.float64).cumsum(0)
    m = np.asarray(info["gt_ego_fut_masks"], dtype=bool)
    ax.plot(np.r_[0, fut[:, 0]], np.r_[0, fut[:, 1]], "-o", color="lime",
            markersize=5, linewidth=2, label="ego GT future (cumsum, 3 s)")
    if not m.all():
        ax.plot(fut[~m, 0], fut[~m, 1], "x", color="red", markersize=10,
                label="invalid step")

    cmd = np.asarray(info["gt_ego_fut_cmd"]).argmax()
    speed = float(np.asarray(info["ego_status"])[6])
    ax.set_xlim(-rng / 2, rng / 2); ax.set_ylim(-rng / 2, rng)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("lidar x (m)  →right"); ax.set_ylabel("lidar y (m)  →forward")
    ax.set_title(f"BEV — token {info['token'][:12]}   {len(boxes)} boxes   "
                 f"cmd={['right','left','straight'][cmd] if cmd < 3 else cmd}   "
                 f"ego_status[6]={speed:.2f} m/s\n"
                 f"first GT step {fut[0,1]:.2f} m / 0.5 s = {fut[0,1]/0.5:.2f} m/s")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="/tmp/pm97/data/nuscenes")
    ap.add_argument("--n", type=int, default=2, help="how many samples to render")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    infos, meta = d["infos"], d["metadata"]
    print(f"pkl        : {args.pkl}")
    print(f"metadata   : {meta}")
    print(f"n samples  : {len(infos)}")
    print(f"info keys  : {len(infos[0])} -> {sorted(infos[0].keys())}")

    tag = os.path.basename(args.pkl).replace(".pkl", "")
    for k in range(args.n):
        idx = args.index + k * max(1, len(infos) // max(args.n, 1))
        idx = min(idx, len(infos) - 1)
        info = infos[idx]
        cam_png = os.path.join(args.out, f"{tag}_{idx:06d}_cams.jpg")
        bev_png = os.path.join(args.out, f"{tag}_{idx:06d}_bev.jpg")
        n_vis = draw_cameras(info, args.data_root, cam_png)
        draw_bev(info, bev_png)
        fut = np.asarray(info["gt_ego_fut_trajs"]).cumsum(0)
        print(f"\n[{idx}] token={info['token'][:12]} boxes={len(info['gt_boxes'])} "
              f"visible_in_cams={n_vis}")
        print(f"     ego_status[6]={info['ego_status'][6]:.3f} m/s   "
              f"first_step/0.5s={fut[0,1]/0.5:.3f} m/s   "
              f"3s_displacement={np.linalg.norm(fut[-1]):.2f} m")
        print(f"     -> {cam_png}")
        print(f"     -> {bev_png}")

    print("\nRendered. INSPECT THE IMAGES — boxes must sit on the objects and the "
          "BEV heading ticks must point along vehicle motion.")


if __name__ == "__main__":
    main()
