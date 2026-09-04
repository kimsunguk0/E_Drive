#!/usr/bin/env python
"""게이트 3 — 캐시 로더의 기하 등가성. 원본 파이프라인 vs `CachedImageGeometry`.

왜 필요한가
----------
캐시는 이미 `undistort -> crop(1920x1080) -> scale`을 거쳤으므로 로더에서
`UndistortMultiViewImage` / `CropMultiViewImage` / `RandomScaleImageMultiViewImage`를
빼야 한다. 그런데 그 셋은 이미지뿐 아니라 **`lidar2img`·`cam_intrinsic`도 함께 고친다.**
그냥 빼면 행렬이 원본 해상도 기준으로 남아 **투영만 조용히 어긋난다** -- 학습 손실은
정상으로 보이고 성능만 나빠지는 종류의 버그다.

같은 시나리오·같은 프레임에 대해 두 파이프라인을 각각 통과시켜 `lidar2img`를 비교한다.
합성 행렬은
    lidar2img_cached = scale(s) @ shift(-ox, -oy) @ lidar2img_undistorted
이고 undistort 자체는 행렬을 바꾸지 않는다(new_K가 이미 `cam_intrinsic`에 있다).

    python scripts/etri_geom_gate.py            # scale 0.4 (tiny)
    python scripts/etri_geom_gate.py --scale 0.8 --orig-config <base config>
"""
import argparse
import importlib
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
CFGDIR = os.path.join(REPO, "projects", "configs", "VAD")
ANN_ORIG = "/tmp/pm97/data/etri/pkl_orig/vad_etri_infos_temporal_train.pkl"
ANN_CACHE = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"


def build(cfg_path, ann, plugin_done=[False]):
    from mmcv import Config
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, "plugin_dir") and not plugin_done[0]:
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
        plugin_done[0] = True
    from mmdet3d.datasets import build_dataset
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ann
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    return build_dataset(cfg.data.test)


def l2i(sample):
    """test 파이프라인은 MultiScaleFlipAug3D로 한 겹 더 감싼다."""
    m = sample["img_metas"]
    while isinstance(m, list):
        m = m[0]
    m = getattr(m, "data", m)
    while isinstance(m, list):
        m = m[0]
    return np.asarray(m["lidar2img"]), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-config", default="VAD_etri_tiny.py")
    ap.add_argument("--cached-config", default="VAD_etri_tvad_bootstrap.py")
    ap.add_argument("--n", type=int, default=6, help="비교할 프레임 수")
    args = ap.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    import pickle
    if not os.path.exists(ANN_ORIG):
        print(f"원본 비교용 pkl 없음: {ANN_ORIG}")
        print("(원본 20 시나리오 추출본이 필요하다 -- logs/phaseG_report.md §2)")
        return 1
    orig_scens = sorted({i["scene_token"]
                         for i in pickle.load(open(ANN_ORIG, "rb"))["infos"]})
    print(f"원본 추출본 시나리오 {len(orig_scens)}개, 앞 {args.n}프레임씩 비교")

    ds_o = build(os.path.join(CFGDIR, args.orig_config), ANN_ORIG)
    ds_c = build(os.path.join(CFGDIR, args.cached_config), ANN_CACHE)
    io = {(i["scene_token"], int(i["frame_idx"])): j
          for j, i in enumerate(pickle.load(open(ANN_ORIG, "rb"))["infos"])}
    ic = {(i["scene_token"], int(i["frame_idx"])): j
          for j, i in enumerate(pickle.load(open(ANN_CACHE, "rb"))["infos"])}

    worst = 0.0
    rows = []
    for s in orig_scens[:3]:
        for f in range(0, args.n * 20, 20):
            if (s, f) not in io or (s, f) not in ic:
                continue
            Lo, mo = l2i(ds_o[io[(s, f)]])
            Lc, mc = l2i(ds_c[ic[(s, f)]])
            assert Lo.shape == Lc.shape, (Lo.shape, Lc.shape)
            d = np.abs(Lo - Lc).max()
            worst = max(worst, d)
            rows.append((s, f, d, mo.get("img_shape", [None])[0],
                         mc.get("img_shape", [None])[0]))

    print(f"\n{'시나리오':<18}{'frame':>6}{'lidar2img 최대차':>18}"
          f"{'원본 shape':>16}{'캐시 shape':>16}")
    for s, f, d, so, sc in rows:
        print(f"{s:<18}{f:>6}{d:>18.3e}   {str(so):>13} {str(sc):>13}")
    print(f"\n전체 최대 절대차 {worst:.3e}")
    ok = worst < 1e-9
    print(f"판정: {'PASS (< 1e-9)' if ok else 'FAIL -- 투영이 어긋난다'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
