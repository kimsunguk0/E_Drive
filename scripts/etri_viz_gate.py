#!/usr/bin/env python
"""Phase E 게이트 — undistort 전/후 이미지에 GT 박스·차선을 투영해 사람이 확인한다.

게이트 판정 기준
    주최측 파이프라인이 쓰는 **새 내부파라미터(new_K, 왜곡 없음)**로 투영한 결과를
    (a) 원본 이미지와 (b) undistort 이미지 양쪽에 겹친다.
    -> (b)에서만 정합해야 정상이다. (a)에서 맞으면 undistort를 안 하고 있다는 뜻이고,
       (b)에서도 안 맞으면 외부파라미터/박스 규약이 틀렸다는 뜻이다.
    fisheye인 `camera_rear_wide`를 **반드시 포함**한다 -- 일반 모델과 fisheye 모델은
    API가 다르고(`cv2.fisheye.*`) 섞으면 조용히 어긋난다.

재현하는 파이프라인 (주최측과 동일)
    1. undistort: initUndistortRectifyMap(raw_K, dist, None, new_K) + remap
                  fisheye면 cv2.fisheye.initUndistortRectifyMap(raw_K, dist[:4], I, new_K)
                  new_K = getOptimalNewCameraMatrix(alpha=0) / fisheye는
                  estimateNewCameraMatrixForUndistortRectify(balance=0)
                  (etri_vad_converter.py:42-48, transform_3d.py:23-35)
    2. crop 1920x1080: ox=(W-1920)//2, oy=0 (front_left/front_right/... 는 위 유지)
                       나머지는 oy=H-1080. new_K를 -ox,-oy 만큼 shift
                       (etri_vad_dataset.py:143-149, transform_3d.py:57-71)

박스 규약 (etri_vad_converter.py:173-186) -- 여기서 틀리면 박스가 90도 돌아간다
    * parquet의 `length[m]`/`width[m]`가 nuScenes의 w/l 슬롯에 **뒤바뀌어** 들어간다
    * `z[m]`은 **바닥 기준**이다. 중심은 z + height/2
    * 기준 프레임은 `ego_pose.parquet`가 아니라 **object.parquet의 class=='ego' 행**
      (`ego_anno`)이고, 회전은 **yaw만** 쓴다 (roll/pitch 무시)
    * 차선은 반대로 `ego_pose`(full rpy) 기준이다 -- 두 기준은 시나리오 후반에
      xy 1.4 m / z 7.4 m까지 벌어지는 별개 오도메트리다. 의도된 설계인지 불명.

    python scripts/etri_viz_gate.py --n-scenarios 3 --frames 60 240
"""
import argparse
import io
import os
import subprocess
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation

META = "/tmp/pm97/data/etri/meta_train"
TARS = "/home/pm97/workspace/sukim/adcl/data/challenge/train"
OUT = "/home/pm97/workspace/sukim/adcl/logs/viz/etri"

CAM_NAMES = ["camera_front", "camera_front_right", "camera_front_left",
             "camera_rear_wide", "camera_rear_left", "camera_rear_right"]
CROP = (1920, 1080)
CROP_KEEP_TOP = ("camera_front_left", "camera_front_right",
                 "camera_rear_left", "camera_rear_right")
CLS_COLOR = {"Car": (60, 220, 60), "Pedestrian": (60, 160, 255),
             "Cyclist": (255, 200, 60)}
FRAME_OFFSET = 50


def col(t, n):
    return np.asarray(t.column(n).to_pylist())


def euler_deg(e):
    return Rotation.from_euler("xyz", e, degrees=True).as_matrix()


def euler_rad(e):
    return Rotation.from_euler("xyz", e, degrees=False).as_matrix()


def yaw2d(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s], [s, c]])


def new_intrinsic(K, dist, size, fisheye):
    if fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist[:4], size, np.eye(3), balance=0.0)
    nk, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha=0)
    return nk


def undistort(img, K, dist, nk, fisheye):
    h, w = img.shape[:2]
    if fisheye:
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist[:4], np.eye(3), nk, (w, h), cv2.CV_32FC1)
    else:
        m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, nk, (w, h),
                                             cv2.CV_32FC1)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def crop_of(name, w, h):
    ox = (w - CROP[0]) // 2
    oy = 0 if name in CROP_KEEP_TOP else h - CROP[1]
    return ox, oy


def read_jpg(tar, member):
    r = subprocess.run(["tar", "-xOf", tar, member], capture_output=True)
    if r.returncode:
        return None
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(r.stdout))),
                        cv2.COLOR_RGB2BGR)


def project(pts_ego, R_c2e, t_c2e, K):
    """ego -> camera -> pixel. R_c2e는 cam->ego 이므로 R^T (p - t)."""
    pc = (pts_ego - t_c2e) @ R_c2e
    z = pc[:, 2]
    uv = (K @ pc.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def box_corners(cx, cy, z_bot, L, W, H, yaw):
    """ego 프레임 8코너. z_bot이 바닥, 길이 L은 heading 방향."""
    dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * L / 2
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * W / 2
    dz = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * H
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([cx + c * dx - s * dy, cy + s * dx + c * dy, z_bot + dz], 1)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box(img, uv, z, color):
    if (z <= 0.5).any():
        return False                      # 카메라 뒤/너무 가까운 박스는 그리지 않는다
    h, w = img.shape[:2]
    if not ((uv[:, 0] > -w) & (uv[:, 0] < 2 * w)
            & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)).all():
        return False
    p = uv.astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
    cv2.line(img, tuple(p[0]), tuple(p[2]), color, 1, cv2.LINE_AA)  # 앞면 대각
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=3)
    ap.add_argument("--frames", type=int, nargs="+", default=[60, 240])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    scen = sorted(os.listdir(META))
    rng = np.random.default_rng(args.seed)
    pick = [scen[i] for i in rng.choice(len(scen), args.n_scenarios, replace=False)]
    print("시나리오:", pick)

    summary = []
    for s in pick:
        d = os.path.join(META, s)
        tar = os.path.join(TARS, s + ".tar")
        cal = pq.read_table(f"{d}/calibration/calibration.parquet").to_pydict()
        cams = {}
        for i, n in enumerate(cal["camera_name"]):
            K = np.asarray(cal["K"][i], np.float64).reshape(3, 3)
            dist = np.asarray(cal["distortion"][i], np.float64)
            fe = bool(cal["is_fisheye"][i])
            size = (int(cal["image_width"][i]), int(cal["image_height"][i]))
            cams[n] = dict(K=K, dist=dist, fisheye=fe, size=size,
                           nk=new_intrinsic(K, dist, size, fe),
                           R=euler_deg(np.asarray(cal["euler"][i], np.float64)),
                           t=np.asarray(cal["translation"][i], np.float64))

        ts = pq.read_table(f"{d}/meta/timestamps.parquet")
        fid = col(ts, "frame_id").astype(int)
        tstamp = col(ts, "timestamp")
        f2t = dict(zip(fid, tstamp))
        ob = pq.read_table(f"{d}/annotation/object.parquet")
        ob_t = col(ob, "timestamp")
        ob_c = col(ob, "class").astype(str)
        ob_x, ob_y, ob_z = col(ob, "x[m]"), col(ob, "y[m]"), col(ob, "z[m]")
        ob_h = col(ob, "heading[rad]")
        ob_L, ob_W, ob_H = col(ob, "length[m]"), col(ob, "width[m]"), col(ob, "height[m]")

        ep = pq.read_table(f"{d}/annotation/ego_pose.parquet")
        ep_t = col(ep, "timestamp")
        ep_xyz = np.stack([col(ep, k) for k in "xyz"], 1)
        ep_rpy = np.stack([col(ep, k) for k in ("roll", "pitch", "yaw")], 1)
        ept = {t: i for i, t in enumerate(ep_t)}
        lanes = [np.asarray([np.asarray(p, np.float64) for p in pts])
                 for pts in pq.read_table(f"{d}/annotation/map.parquet")
                 .column("points").to_pylist()]

        for f in args.frames:
            t0 = f2t[f]
            m = ob_t == t0
            ego = m & (ob_c == "ego")
            if not ego.any():
                continue
            ex, ey, ez = ob_x[ego][0], ob_y[ego][0], ob_z[ego][0]
            eh = ob_h[ego][0]
            to_ego = yaw2d(-eh)
            objm = m & (ob_c != "ego")
            # 박스: ego_anno 기준, yaw만. z는 바닥 기준 그대로 유지한다
            bxy = (np.stack([ob_x[objm], ob_y[objm]], 1) - [ex, ey]) @ to_ego.T
            bz = ob_z[objm] - ez
            byaw = (ob_h[objm] - eh + np.pi) % (2 * np.pi) - np.pi
            bcls = ob_c[objm]
            # 차선: ego_pose 기준, full rpy (컨버터 vectormap_pipeline과 동일)
            ei = ept[t0]
            Rg = euler_rad(ep_rpy[ei])
            lanes_ego = [(ln[:, :3] - ep_xyz[ei]) @ Rg for ln in lanes]

            for cam in CAM_NAMES:
                c = cams[cam]
                raw = read_jpg(tar, f"{s}/{cam}/{f:08d}.jpg")
                if raw is None:
                    continue
                und = undistort(raw, c["K"], c["dist"], c["nk"], c["fisheye"])
                ox, oy = crop_of(cam, c["size"][0], c["size"][1])
                shift = np.eye(3)
                shift[0, 2], shift[1, 2] = -ox, -oy
                Kc = shift @ c["nk"]
                a = raw[oy:oy + CROP[1], ox:ox + CROP[0]].copy()
                b = und[oy:oy + CROP[1], ox:ox + CROP[0]].copy()

                drawn = 0
                for i in range(len(bxy)):
                    # 축 주의: ETRI의 컬럼명은 물리 의미와 뒤바뀌어 있다.
                    # Car 중앙값이 length[m]=2.04 / width[m]=4.65 -- 즉
                    # width[m]이 진행방향 길이다. 30 시나리오 198,947건으로 확인.
                    # 처음엔 컬럼명 그대로 써서 박스가 90도 돌아갔다.
                    cor = box_corners(bxy[i, 0], bxy[i, 1], bz[i],
                                      ob_W[objm][i], ob_L[objm][i],
                                      ob_H[objm][i], byaw[i])
                    uv, z = project(cor, c["R"], c["t"], Kc)
                    if np.isnan(uv).any():
                        continue
                    col_ = CLS_COLOR.get(str(bcls[i]), (200, 200, 200))
                    ok = draw_box(a, uv, z, col_)
                    draw_box(b, uv, z, col_)
                    drawn += int(ok)
                for ln in lanes_ego:
                    # 근거리만 그린다. 55개 차선을 거리 제한 없이 그리면 교차로 차선과
                    # 수백 m 전방 차선이 소실점 주변에 뭉쳐서 정합 판정이 불가능해진다.
                    if not ((np.abs(ln[:, 0]) < 70) & (np.abs(ln[:, 1]) < 25)
                            & (np.abs(ln[:, 2]) < 3)).any():
                        continue
                    uv, z = project(ln, c["R"], c["t"], Kc)
                    inrange = ((np.abs(ln[:, 0]) < 70) & (np.abs(ln[:, 1]) < 25)
                               & (np.abs(ln[:, 2]) < 3))
                    v = (z > 0.5) & np.isfinite(uv).all(1) & inrange
                    # 세그먼트 단위로만 잇는다. 앞서 유효점만 모아 연속으로 이으면
                    # 카메라 뒤 구간을 건너뛰며 폴리라인 양 끝을 관통하는 직선이
                    # 그려진다 -- 첫 판에서 실제로 그 artifact가 나왔다.
                    for j in range(len(ln) - 1):
                        if not (v[j] and v[j + 1]):
                            continue
                        p0 = tuple(uv[j].astype(int))
                        p1 = tuple(uv[j + 1].astype(int))
                        if max(abs(p0[0]), abs(p1[0])) > 8000 or \
                           max(abs(p0[1]), abs(p1[1])) > 8000:
                            continue
                        cv2.line(a, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.line(b, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)

                tag = f"{cam}  fisheye={c['fisheye']}  boxes={drawn}"
                for im, lab in ((a, "(a) RAW  misaligned expected"),
                                (b, "(b) UNDISTORTED  aligned expected")):
                    cv2.putText(im, f"{s} f{f}  {tag}  {lab}", (20, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
                    cv2.putText(im, f"{s} f{f}  {tag}  {lab}", (20, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                pair = np.concatenate([a, b], axis=0)
                pair = cv2.resize(pair, (a.shape[1] // 2, pair.shape[0] // 2))
                fn = f"{OUT}/{s}_f{f:03d}_{cam}.jpg"
                cv2.imwrite(fn, pair, [cv2.IMWRITE_JPEG_QUALITY, 92])
                summary.append((s, f, cam, c["fisheye"], drawn, fn))
                print(f"  {os.path.basename(fn)}  박스 {drawn}", flush=True)

    print(f"\n{len(summary)}장 저장 -> {OUT}")
    nz = [r for r in summary if r[4] == 0]
    if nz:
        print(f"박스 0개인 조합 {len(nz)}건 (뒤쪽 카메라는 정상일 수 있음):")
        for r in nz[:10]:
            print("   ", r[2], r[0], r[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
