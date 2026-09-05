#!/usr/bin/env python
"""투영 기하 육안 검증: 캐시 이미지 위에 GT 궤적 + 에이전트 박스를 그린다.

sparse scorer 가 waypoint 에서 읽는 픽셀이 실제로 그 지면 위치인지 확인한다.
가시성 비율만으로는 검증되지 않는 부분이다.
"""
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
from sparse_scoredrive import build_cached_lidar2img, CAMERA_ORDER  # noqa: E402
import sparse_common as C  # noqa: E402

OUT = "/tmp/proj_check.png"


def proj(M, pts_xyz):
    """M [4,4], pts [N,3] -> uv [N,2], depth [N]"""
    p = np.concatenate([pts_xyz, np.ones((len(pts_xyz), 1))], 1)
    q = p @ M.T
    d = q[:, 2]
    uv = q[:, :2] / np.clip(d, 1e-6, None)[:, None]
    return uv, d


def main():
    arr = C.load_arrays()
    infos = pickle.load(open(
        "/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl", "rb"))["infos"]
    # 에이전트가 많고 자차가 움직이는 프레임 고르기
    pick = None
    for info in infos:
        if len(info.get("gt_boxes", [])) >= 8 and int(info["frame_idx"]) >= 60:
            key = (info["scene_token"], int(info["frame_idx"]))
            r = None
            for rr in range(len(arr["frame"])):
                if (arr["scenarios"][arr["scen_idx"][rr]], int(arr["frame"][rr])) == key:
                    r = rr
                    break
            if r is not None and np.linalg.norm(arr["fut"][r][-1]) > 8:
                pick = (info, r)
                break
    info, row = pick
    scen = info["scene_token"]
    frame = int(info["frame_idx"])
    print("scene", scen, "frame", frame, "agents", len(info["gt_boxes"]))

    l2i_new = build_cached_lidar2img(info)                       # cam_intrinsic
    info_raw = dict(info)
    cams_raw = {k: dict(v) for k, v in info["cams"].items()}
    for k in cams_raw:
        cams_raw[k]["cam_intrinsic"] = cams_raw[k]["cam_intrinsic_raw"]
    info_raw["cams"] = cams_raw
    l2i_raw = build_cached_lidar2img(info_raw)                   # cam_intrinsic_raw
    same = np.abs(l2i_new - l2i_raw).max()
    print("cam_intrinsic vs raw 로 만든 lidar2img 최대차:", same)

    gt = arr["fut"][row].astype(np.float64)                      # [6,2] abs
    gb = np.asarray(info["gt_boxes"], np.float64)
    tiles = []
    for ci, cam in enumerate(("camera_front", "camera_front_left", "camera_front_right")):
        k = CAMERA_ORDER.index(cam)
        p = os.path.join(C.CACHE, scen, cam, f"{frame:08d}.jpg")
        im = Image.open(p).convert("RGB")
        for M, color, tag in ((l2i_new[k], (0, 255, 0), "new_K"),
                              (l2i_raw[k], (255, 0, 255), "raw_K")):
            img = im.copy()
            d = ImageDraw.Draw(img)
            # GT 궤적 (지면 z=0)
            pts = np.concatenate([gt, np.zeros((6, 1))], 1)
            uv, dep = proj(M, pts)
            for i in range(6):
                if dep[i] > 0.1:
                    x, y = uv[i]
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=color, width=3)
                    d.text((x + 8, y - 6), str(i), fill=color)
            # 에이전트 박스 하단 중심 (z = box_z - h/2)
            cen = np.stack([gb[:, 0], gb[:, 1], gb[:, 2] - gb[:, 5] / 2], 1)
            uv2, dep2 = proj(M, cen)
            for i in range(len(gb)):
                if dep2[i] > 0.1:
                    x, y = uv2[i]
                    rr = max(4, 400.0 / max(dep2[i], 1.0))
                    d.rectangle([x - rr, y - rr * 1.5, x + rr, y], outline=(255, 0, 0), width=2)
            d.text((6, 6), f"{cam} {tag}", fill=(255, 255, 0))
            tiles.append(img)
    W, H = tiles[0].size
    sheet = Image.new("RGB", (W * 2, H * 3))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * W, (i // 2) * H))
    sheet = sheet.resize((W * 2 // 2, H * 3 // 2))
    sheet.save(OUT)
    print("saved", OUT, sheet.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
