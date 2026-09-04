#!/usr/bin/env python
"""게이트 9 — 캐시 이미지 위 재투영 시각 검증.

게이트 3(`etri_geom_gate.py`)은 `lidar2img` **행렬**이 두 경로에서 같음을 확인했다
(최대차 2.842e-14). 그건 "캐시 로더가 원본 로더와 같다"는 뜻일 뿐이고, **원본 로더
자체가 맞는지는 말해주지 않는다.** 픽셀 위에 실제로 얹어봐야 안다.

과거에 밟은 함정 두 개를 그대로 재현 검사한다:
  * 박스 축 미교환 -> 박스가 90도 돌아 그려짐. 전방 길이 = `width[m]` 이다.
  * 차선 비인접 점 연결 -> 카메라 뒤 점을 필터링한 뒤 폴리라인 끝점을 관통하는 직선이
    생긴다. **구간별로** 그려야 한다.

주행 상황을 골라 뽑는다: 직선 / 좌회전 / 우회전 / 정지 / 고속 / 교차로(곡률).
rear_wide(fisheye)를 반드시 포함한다.

    python scripts/etri_viz_align.py --out logs/viz_align
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
CAMS = ["camera_front", "camera_front_left", "camera_front_right",
        "camera_rear_left", "camera_rear_right", "camera_rear_wide"]
COLOR = {"Car": (0, 200, 255), "Pedestrian": (255, 80, 255),
         "Cyclist": (120, 255, 120)}


def project(pts, l2i, W, H):
    """(N,3) lidar -> (N,2) 픽셀 + 유효 마스크. 카메라 뒤(z<=0)는 무효."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ np.asarray(l2i).T
    z = p[:, 2:3]
    ok = z[:, 0] > 0.1
    uv = np.zeros((len(pts), 2))
    uv[ok] = p[ok, :2] / z[ok]
    ok &= (uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H)
    return uv, ok


def box_corners(b):
    """ETRI gt_boxes 한 줄 -> 8개 코너.

    `(x, y, z, w, l, h, yaw)`이다. 두 가지를 조심한다.

    1. yaw에 이미 -pi/2가 들어 있다 (`etri_vad_converter.py:186`
       `wrap_angle(-yaw_ego - np.pi/2)`). 그래서 로컬 x축에 **w**를 놓으면 회전 후
       w가 세계 y(횡)로, l이 세계 x(종)로 간다 -- 이게 맞다.
    2. **z는 박스 상단이다.** 컨버터가 `xyz[:,2] + wlh[:,2]/2 - ego_z`로 저장한다
       (`etri_vad_converter.py:180-183`). 하단으로 착각하면 박스가 차량 위로 한 칸
       떠서 그려진다 -- 실제로 그렇게 그려졌다. 그래서 z에서 h만큼 **내려간다.**
       검산: gt_boxes[0] z=1.54, h=1.62 -> 하단 -0.08 ~= 노면(ego_z를 뺐으므로 0).
    """
    x, y, z, w, l, h, yaw = b[:7]
    dx, dy = w / 2, l / 2
    c = np.array([[+dx, +dy, -h], [+dx, -dy, -h], [-dx, -dy, -h], [-dx, +dy, -h],
                  [+dx, +dy, 0], [+dx, -dy, 0], [-dx, -dy, 0], [-dx, +dy, 0]],
                 float)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return c @ R.T + np.array([x, y, z])


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw(img, info, l2i, lanes, traj, cam_i):
    H, W = img.shape[:2]
    out = img.copy()
    # ---- 차선: 구간별로 그린다 ----
    for pl in lanes:
        if len(pl) < 2:
            continue
        p3 = np.c_[pl[:, 0], pl[:, 1], np.zeros(len(pl))]
        uv, ok = project(p3, l2i, W, H)
        for a, b in zip(range(len(pl) - 1), range(1, len(pl))):
            if ok[a] and ok[b]:      # 양 끝이 모두 유효할 때만 -- 관통 직선 방지
                cv2.line(out, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                         (90, 220, 90), 2)
    # ---- 박스 ----
    for b, name in zip(info["gt_boxes"], info["gt_names"]):
        cn = box_corners(b)
        uv, ok = project(cn, l2i, W, H)
        col = COLOR.get(str(name), (200, 200, 200))
        for a, c in EDGES:
            if ok[a] and ok[c]:
                cv2.line(out, tuple(uv[a].astype(int)), tuple(uv[c].astype(int)),
                         col, 2)
    # ---- ego 미래 궤적 (누적) ----
    t3 = np.c_[traj[:, 0], traj[:, 1], np.zeros(len(traj))]
    uv, ok = project(np.vstack([[[0, 0, 0.0]], t3]), l2i, W, H)
    for a in range(len(uv) - 1):
        if ok[a] and ok[a + 1]:
            cv2.line(out, tuple(uv[a].astype(int)), tuple(uv[a + 1].astype(int)),
                     (60, 60, 255), 3)
    for a in range(len(uv)):
        if ok[a]:
            cv2.circle(out, tuple(uv[a].astype(int)), 4, (60, 60, 255), -1)
    cv2.putText(out, CAMS[cam_i], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/viz_align")
    ap.add_argument("--cache-root", default="/tmp/pm97/cache/etri_768")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(CACHE, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    vi = sp["val_idx"]
    scen_all = np.array([str(x) for x in d["scenarios"]])
    scen = scen_all[d["scen_idx"][vi]]
    frame = d["frame"][vi].astype(int)
    speed = d["speed"][vi]
    vcmd = d["vad_cmd"][vi].astype(int)
    goal = np.linalg.norm(d["goal"][vi][:, :2], axis=1)

    # 주행 상황 선정 -- 각 조건의 첫 앵커
    picks = []
    for name, m in (("직선_고속", (vcmd == 2) & (speed >= 15)),
                    ("직선_중속", (vcmd == 2) & (speed >= 5) & (speed < 10)),
                    ("우회전", vcmd == 0),
                    ("좌회전", vcmd == 1),
                    ("정지", (speed < 0.5) & (goal < 1.0)),
                    ("저속_크립", (speed >= 0.5) & (speed < 3))):
        w = np.where(m)[0]
        if len(w):
            picks.append((name, int(w[len(w) // 2])))
    print(f"선정 {len(picks)}개 상황:")
    for n, k in picks:
        print(f"  {n:12s} {scen[k]} f{frame[k]:3d}  speed {speed[k]:5.2f} "
              f"cmd {vcmd[k]} goal {goal[k]:6.2f}")

    _pk = pickle.load(open(ANN, "rb"))
    infos = _pk["infos"]
    map_lanes = _pk["metadata"].get("map_lanes", {})
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j
              for j, i in enumerate(infos)}
    meta = None
    cm_path = os.path.join(args.cache_root, "cache_meta.json")
    if os.path.exists(cm_path):
        import json
        meta = json.load(open(cm_path))

    n_ok = 0
    for name, k in picks:
        s, f = scen[k], int(frame[k])
        info = infos[key2ds[(s, f)]]
        traj = np.cumsum(info["gt_ego_fut_trajs"].reshape(6, 2), axis=0)
        lanes = [np.asarray(x, float) for x in map_lanes.get(s, [])]
        tiles = []
        for ci, cam in enumerate(CAMS):
            p = os.path.join(args.cache_root, s, cam, f"{f:08d}.jpg")
            if not os.path.exists(p):
                print(f"  !! 없음 {p}")
                continue
            img = cv2.imread(p)
            # nuscenes_vad_dataset.py:1333-1342 와 **동일하게** 만든다.
            # 전치가 두 번 들어가므로 눈으로 옮기면 틀린다.
            ci_ = info["cams"][cam]
            lidar2cam_r = np.linalg.inv(np.asarray(ci_["sensor2lidar_rotation"]))
            lidar2cam_t = np.asarray(ci_["sensor2lidar_translation"]) @ lidar2cam_r.T
            l2c = np.eye(4)
            l2c[:3, :3] = lidar2cam_r.T
            l2c[3, :3] = -lidar2cam_t
            intr = np.asarray(ci_["cam_intrinsic"])
            viewpad = np.eye(4)
            viewpad[:intr.shape[0], :intr.shape[1]] = intr
            l2i = viewpad @ l2c.T
            # 캐시는 crop -> scale 을 이미 거쳤다. CachedImageGeometry 와 같은 합성행렬.
            if meta and s in meta and cam in meta[s]:
                mm = meta[s][cam]
                sc, (ox, oy) = mm["scale"], mm["crop"][:2]
                M = np.eye(4)
                M[0, 0] = M[1, 1] = sc
                M[0, 2] = -ox * sc
                M[1, 2] = -oy * sc
                l2i = M @ l2i
            tiles.append(draw(img, info, l2i, lanes, traj, ci))
        if len(tiles) == 6:
            top = np.hstack(tiles[:3])
            bot = np.hstack(tiles[3:])
            grid = np.vstack([top, bot])
            grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2))
            op = os.path.join(args.out, f"{name}_{s}_f{f:03d}.jpg")
            cv2.imwrite(op, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  저장 {op}")
            n_ok += 1
    print(f"\n{n_ok}/{len(picks)} 생성. 눈으로 확인할 것:")
    print("  * 박스가 차량에 붙는가 (90도 돌아가 있지 않은가 -- 전방 길이 = width[m])")
    print("  * ego 궤적(빨강)이 자차 진행 방향 노면에 놓이는가")
    print("  * rear_wide(fisheye)에서도 정합되는가")
    return 0


if __name__ == "__main__":
    sys.exit(main())
