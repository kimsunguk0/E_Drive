#!/usr/bin/env python
"""같은 (시나리오, 프레임)에 대해 **train 파이프라인 vs test 파이프라인**의
모델 입력을 원소 단위로 비교한다.

왜
--
epoch 16 tvad_metric_overfit:
    train 경로  0.30   (model.train / model.eval 무관)
    test 경로   3.53   (full streaming) / 4.99 (깊이 2 = 학습과 동일 깊이)
누적 깊이를 학습과 같게 맞춰도 4.99다. 그러면 남은 변수는 **입력 그 자체**뿐이다.

게이트 3은 `orig test pipeline` vs `cached test pipeline`의 `lidar2img`를 봤다.
**train 파이프라인은 test 파이프라인과 한 번도 대조된 적이 없다.**

    python scripts/etri_pipeline_diff.py
"""
import argparse
import importlib
import os
import pickle
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"


def unwrap(x):
    while isinstance(x, list):
        x = x[0]
    x = getattr(x, "data", x)
    while isinstance(x, list):
        x = x[0]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset

    print("=== 파이프라인 구성 ===")
    for tag, pl in (("train", cfg.data.train.pipeline),
                    ("test ", cfg.data.test.pipeline)):
        print(f"  {tag}: " + " -> ".join(
            p["type"] + (f"({p.get('scale')})" if "scale" in p else "") for p in pl))

    cfg.data.train.ann_file = SUB
    ds_tr = build_dataset(cfg.data.train)
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = SUB
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds_te = build_dataset(cfg.data.test)

    infos = pickle.load(open(SUB, "rb"))["infos"]
    # 학습 큐가 성립하는 프레임(>=15)이고 filter_empty_gt를 통과하는 것을 고른다
    cands = [j for j, i in enumerate(infos)
             if int(i["frame_idx"]) >= 20 and int(i["frame_idx"]) % 5 == 0]
    picked = []
    for j in cands:
        d = ds_tr.prepare_train_data(j)   # None이면 대체 대상이므로 건너뛴다
        if d is not None:
            picked.append((j, d))
        if len(picked) >= args.n:
            break

    for j, dtr in picked:
        s, f = infos[j]["scene_token"], int(infos[j]["frame_idx"])
        dte = ds_te[j]
        mtr = dtr["img_metas"].data            # {0:…, 1:…, 2:current}
        cur = mtr[max(mtr.keys())]
        mte = unwrap(dte["img_metas"])
        itr = dtr["img"].data                  # [Q, N, C, H, W]
        ite = unwrap(dte["img"])               # [N, C, H, W]
        print(f"\n===== {s} frame {f}  (train 큐 {itr.shape[0]}프레임) =====")
        print(f"  scene/frame 일치   train {cur.get('scene_token')}/{cur.get('frame_idx')}"
              f"   test {mte.get('scene_token')}/{mte.get('frame_idx')}")
        print(f"  img shape          train {tuple(itr.shape)}   test {tuple(ite.shape)}")
        a = itr[-1].double()
        b = ite.double()
        if a.shape == b.shape:
            print(f"  img 최대절대차     {float((a - b).abs().max()):.3e}"
                  f"   (train |x| 평균 {float(a.abs().mean()):.4f}, "
                  f"test {float(b.abs().mean()):.4f})")
        else:
            print(f"  !! img shape 불일치 {tuple(a.shape)} vs {tuple(b.shape)}")
        for k in ("img_shape", "pad_shape", "scale_factor", "flip", "pcd_horizontal_flip"):
            print(f"  {k:<18} train {str(cur.get(k))[:44]:<46} test {str(mte.get(k))[:44]}")
        L1 = np.asarray(cur["lidar2img"]).astype(np.float64)
        L2 = np.asarray(mte["lidar2img"]).astype(np.float64)
        print(f"  lidar2img 최대차   {np.abs(L1 - L2).max():.3e}   shape {L1.shape}")
        c1 = np.asarray(cur["can_bus"], dtype=np.float64)
        c2 = np.asarray(mte["can_bus"], dtype=np.float64)
        print(f"  can_bus 최대차     {np.abs(c1 - c2).max():.3e}")
        print(f"    train can_bus[:3] {np.round(c1[:3],4)}  [-2:] {np.round(c1[-2:],4)}")
        print(f"    test  can_bus[:3] {np.round(c2[:3],4)}  [-2:] {np.round(c2[-2:],4)}")
        print(f"  prev_bev 플래그     train {[mtr[i].get('prev_bev') for i in sorted(mtr)]}"
              f"   test {mte.get('prev_bev')}")
        for k in sorted(mtr):
            cb = np.asarray(mtr[k]["can_bus"], dtype=np.float64)
            print(f"    큐[{k}] frame {mtr[k].get('frame_idx')}  "
                  f"can_bus[:2] {np.round(cb[:2],4)}  [-1] {cb[-1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
