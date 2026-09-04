#!/usr/bin/env python
"""train 이미지 오프라인 캐시: undistort -> crop -> scale 0.4 -> 768x432 JPEG.

왜 캐시하나: 주최측 로더는 매 샘플마다 1920x1536 6장에 remap을 건다. 학습 내내 CPU가
그것만 하게 되고 GPU가 굶는다. 기하 변환은 카메라 캘리브레이션에만 의존하고 학습 중
바뀌지 않으므로 한 번만 해두면 된다.

**크기가 768x432이지 768x448이 아니다.** 파이프라인은
    crop 1920x1080 -> RandomScaleImageMultiViewImage(0.4) -> 768x432
    -> PadMultiViewImage(size_divisor=32) -> 768x448
이다. 448은 **패딩 후 모델 입력**이고, 패딩은 로더가 하도록 남겨둔다. 캐시에 패딩을
구워 넣으면 로더가 한 번 더 패딩하거나 패딩 영역을 실제 픽셀로 착각한다.

이중 undistort 금지
-------------------
캐시를 쓸 때 로더에서 **반드시 빼야 하는 단계**:
    UndistortMultiViewImage        (이미 적용됨)
    CropMultiViewImage             (이미 적용됨 -- 단 lidar2img shift는 대신 해줘야 함)
    RandomScaleImageMultiViewImage (이미 적용됨 -- 단 lidar2img scale도 대신)
남겨야 하는 단계: NormalizeMultiviewImage, PadMultiViewImage, 그 이후 전부.

기하 보정은 이미지에만 적용됐고 `lidar2img`에는 적용되지 않았으므로, 캐시 로더는
    K_cache = diag(0.4, 0.4, 1) @ shift(-ox, -oy) @ new_K
를 써야 한다. 이 행렬을 시나리오·카메라별로 `cache_meta.json`에 함께 저장한다 --
로더가 재계산하면 두 경로가 갈릴 수 있고, 갈려도 손실은 정상으로 보인다.

    python scripts/etri_build_cache.py                    # train 전량 (tiny용 0.4)
    python scripts/etri_build_cache.py --limit 4 --jobs 4 # 소규모 확인

base(1536x864) + 2Hz 서브셋
---------------------------
`VAD_etri_base.py`는 `scales=[0.8]`이라 0.4 캐시를 쓸 수 없다. 그리고
`etri_vad_dataset.py`의 `sample_interval=5` 때문에 앵커를 2 Hz(5의 배수)로 잡으면
학습이 읽는 프레임도 전부 5의 배수가 된다 -- 이미지가 20%만 필요하다 (tar 1개 실측 20.0%).

    # train: 376x60x6 = 135,360장, 약 40 GB
    python scripts/etri_build_cache.py --split train --scale 0.8 --frame-mod 5 \
        --out /tmp/pm97/cache/etri_1536_2hz
    # test: 1125x7x6 = 47,250장, 약 14 GB (frame_0,-5,...,-30)
    python scripts/etri_build_cache.py --split test  --scale 0.8 --frame-mod 5 \
        --out /tmp/pm97/cache/etri_1536_test

`--frame-mod`를 쓰면 로더 쪽에서 **pkl도 같은 규칙으로 걸러야 하고
`sample_interval`을 1로 내려야 한다.** 안 내리면 큐가 원본 100프레임(10초) 뒤를
보게 되는데, 손실은 정상으로 보이고 결과만 나빠진다.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pyarrow.parquet as pq

CHALLENGE = "/home/pm97/workspace/sukim/adcl/data/challenge"
OUT = "/tmp/pm97/cache/etri_768"
CROP = (1920, 1080)
QUALITY = 95
CROP_KEEP_TOP = ("camera_front_left", "camera_front_right",
                 "camera_rear_left", "camera_rear_right")

# train 파일명 = 00000064.jpg (프레임 번호), test = frame_-15.jpg / frame_0.jpg.
# 두 규칙 모두 정수 프레임 인덱스를 뽑아 --frame-mod로 나눈다.
def frame_index(fn):
    stem = fn[:-4]
    if stem.startswith("frame_"):
        stem = stem[6:]
    try:
        return int(stem)
    except ValueError:
        return None


def calib_path(split, scen):
    """train은 meta_train/<scen>/calibration/, test는 meta_test/<clip>/ 직하."""
    if split == "train":
        return (f"/tmp/pm97/data/etri/meta_train/{scen}"
                f"/calibration/calibration.parquet")
    return f"/tmp/pm97/data/etri/meta_test/{scen}/calibration.parquet"


def new_intrinsic(K, dist, size, fisheye):
    """주최측 컨버터 undistorted_intrinsic과 동일 (etri_vad_converter.py:42-48)."""
    if fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist[:4], size, np.eye(3), balance=0.0)
    nk, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha=0)
    return nk


def build_maps(K, dist, nk, size, fisheye):
    # CV_16SC2: float32 맵의 절반 메모리, remap도 더 빠르다. 6캠 × 워커 수만큼
    # 상주하므로 float32면 워커당 142 MB가 된다.
    if fisheye:
        return cv2.fisheye.initUndistortRectifyMap(
            K, dist[:4], np.eye(3), nk, size, cv2.CV_16SC2)
    return cv2.initUndistortRectifyMap(K, dist, None, nk, size, cv2.CV_16SC2)


def crop_of(name, w, h):
    return (w - CROP[0]) // 2, (0 if name in CROP_KEEP_TOP else h - CROP[1])


def do_scenario(args):
    tar_path, out_root, scale, fmod, split = args
    cv2.setNumThreads(1)          # 워커 안에서 OpenCV가 스레드를 또 까는 것을 막는다
    scen = os.path.basename(tar_path)[:-4]
    t0 = time.time()
    n_img = 0
    nbytes = 0
    meta = {}

    # 스트리밍 모드(r|)는 되돌아갈 수 없어 tar 안의 calibration을 먼저 못 읽는다.
    # 대신 이미 뽑아둔 parquet를 쓴다 (Phase A 산출물).
    cal = pq.read_table(calib_path(split, scen)).to_pydict()
    maps, Kc = {}, {}
    for i, name in enumerate(cal["camera_name"]):
        K = np.asarray(cal["K"][i], np.float64).reshape(3, 3)
        dist = np.asarray(cal["distortion"][i], np.float64)
        fe = bool(cal["is_fisheye"][i])
        size = (int(cal["image_width"][i]), int(cal["image_height"][i]))
        nk = new_intrinsic(K, dist, size, fe)
        maps[name] = build_maps(K, dist, nk, size, fe)
        ox, oy = crop_of(name, size[0], size[1])
        shift = np.eye(3)
        shift[0, 2], shift[1, 2] = -ox, -oy
        sc = np.diag([scale, scale, 1.0])
        Kc[name] = (sc @ shift @ nk).tolist()
        meta[name] = dict(crop=[ox, oy, CROP[0], CROP[1]], scale=scale,
                          out_size=[int(CROP[0] * scale), int(CROP[1] * scale)],
                          is_fisheye=fe, new_K=nk.tolist(), K_cache=Kc[name])
        for d in (os.path.join(out_root, scen, name),):
            os.makedirs(d, exist_ok=True)

    tw, th = int(CROP[0] * scale), int(CROP[1] * scale)
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.name.endswith(".jpg"):
                continue
            parts = m.name.split("/")
            if len(parts) < 3:
                continue
            cam, fn = parts[-2], parts[-1]
            if cam not in maps:
                continue
            if fmod > 1:
                fi = frame_index(fn)
                # 프레임 번호를 못 읽으면 건너뛰지 말고 남긴다. 조용히 빠지는 것이
                # 조용히 어긋나는 것보다 낫다 -- 개수 검증에서 바로 걸린다.
                if fi is not None and fi % fmod != 0:
                    continue
            dst = os.path.join(out_root, scen, cam, fn)
            if os.path.exists(dst):
                n_img += 1
                continue
            buf = tf.extractfile(m).read()
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            und = cv2.remap(img, maps[cam][0], maps[cam][1], cv2.INTER_LINEAR)
            ox, oy = crop_of(cam, img.shape[1], img.shape[0])
            cropped = und[oy:oy + CROP[1], ox:ox + CROP[0]]
            small = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", small,
                                   [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if not ok:
                continue
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(enc.tobytes())
            os.replace(tmp, dst)
            n_img += 1
            nbytes += len(enc)

    return scen, n_img, nbytes, time.time() - t0, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--scale", type=float, default=0.4,
                    help="tiny=0.4 (768x432), base=0.8 (1536x864)")
    ap.add_argument("--frame-mod", type=int, default=0,
                    help="이 값의 배수 프레임만 굽는다. 5 = 2 Hz (20%%)")
    args = ap.parse_args()
    # META를 모듈 상수로 두면 --out을 줘도 기본 경로에 쓰려다 죽는다.
    meta_path = os.path.join(args.out, "cache_meta.json")

    tar_dir = os.path.join(CHALLENGE, args.split)
    tars = sorted(os.path.join(tar_dir, f) for f in os.listdir(tar_dir)
                  if f.endswith(".tar"))
    if args.limit:
        tars = tars[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    tw, th = int(CROP[0] * args.scale), int(CROP[1] * args.scale)
    print(f"[{args.split}] 시나리오 {len(tars)}개 -> {args.out}  "
          f"jobs {args.jobs}", flush=True)
    print(f"출력 {tw}x{th} JPEG q{QUALITY}  scale {args.scale}  "
          f"frame-mod {args.frame_mod or 1}  "
          f"(size_divisor 패딩은 로더의 PadMultiViewImage가 만든다)", flush=True)

    t0 = time.time()
    all_meta, tot_img, tot_bytes = {}, 0, 0
    jobargs = [(t, args.out, args.scale, args.frame_mod, args.split)
               for t in tars]
    with Pool(args.jobs) as p:
        for i, (scen, n, nb, dt, meta) in enumerate(
                p.imap_unordered(do_scenario, jobargs), 1):
            all_meta[scen] = meta
            tot_img += n
            tot_bytes += nb
            if i % 10 == 0 or i == len(tars):
                el = time.time() - t0
                rate = tot_img / max(el, 1e-9)
                eta = (len(tars) - i) * (el / i)
                print(f"  {i:3d}/{len(tars)}  이미지 {tot_img:,}  "
                      f"{tot_bytes/2**30:.1f} GiB  {rate:.0f} img/s  "
                      f"ETA {eta/60:.0f}분", flush=True)
                json.dump(all_meta, open(meta_path, "w"))

    json.dump(all_meta, open(meta_path, "w"))
    el = time.time() - t0
    print(f"\n완료 {tot_img:,}장  {tot_bytes/2**30:.1f} GiB  "
          f"{el/60:.1f}분  {tot_img/el:.0f} img/s")
    print(f"메타 {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
