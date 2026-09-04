#!/usr/bin/env python
"""ego-motion shift의 구조 검증 — goal의 T6에 대응하는 것 (형님 §3).

구분하려는 것
------------
    (a) shift가 **영상 content의 sampling 위치만** 바꾸는가
    (b) shift가 learned positional basis와 결합해 **장면 정보 없이도** trajectory를 만드는가

방법 (T6와 같은 논리)
    S1  visual feature를 상수로 고정 + shift 3종     -> 출력이 같으면 (a)
    S2  visual feature를 실데이터 + shift 3종        -> 달라야 테스트가 유효 (대조군)
    S3  visual feature를 상수로 고정 + shift 고정    -> S1의 기준선

`can_bus[0], can_bus[1]`(0.5초 변위)과 `can_bus[-2], can_bus[-1]`(각도)을 바꾼다.
그게 `get_bev_features`의 shift/rotate 계산에 들어가는 전부다.

주의: 이 테스트가 완전 불변이어야 합법이라는 뜻이 **아니다**. 결과를 알고 있어야
코드 심사에서 정확히 설명할 수 있다는 것이 목적이다:
  "ego motion은 영상 feature의 시공간 정렬·선택에만 쓰이고, 독립적인 trajectory
   value를 만드는 learned state branch는 제거했다"
가 어디까지 성립하는지 수치로 못 박는다.

    python scripts/etri_shift_structcheck.py --ckpt .../epoch_14.pth
"""
import argparse
import importlib
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"

# (delta_x, delta_y, patch_angle_rad, angle_delta_deg)
SHIFTS = [
    ("정지        ", 0.00, 0.00, 0.0, 0.0),
    ("직진 4m/0.5s", 4.00, 0.00, 0.0, 0.0),
    ("좌회전      ", 3.00, 1.50, 0.0, 8.0),
    ("우회전 고속  ", 7.00, -1.00, 0.0, -6.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_deploymatch_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8, help="앵커 수")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import pickle
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ANN
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    k2 = {(i["scene_token"], int(i["frame_idx"])): j
          for j, i in enumerate(ds.data_infos)}
    want = sorted({i["scene_token"] for i in
                   pickle.load(open(SUB, "rb"))["infos"]})
    keys = [(s, f) for s in want[:4] for f in (60, 120, 180, 240)][:args.n]

    torch.manual_seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}")
    tr = model.pts_bbox_head.transformer
    print(f"use_can_bus={tr.use_can_bus}  use_shift={tr.use_shift}  "
          f"rotate_prev_bev={tr.rotate_prev_bev}")
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    # visual feature를 상수로 만드는 hook (extract_feat 반환값을 덮어쓴다)
    cls = type(model)
    orig_ef = cls.extract_feat
    state = {"const": False, "shift": None}

    def ef(self, img, img_metas=None, len_queue=None):
        out = orig_ef(self, img, img_metas=img_metas, len_queue=len_queue)
        if state["const"]:
            out = [torch.full_like(x, 0.37) for x in out]
        return out
    cls.extract_feat = ef

    tcls = type(tr)
    orig_gbf = tcls.get_bev_features

    def gbf(self, *a, **k):
        if state["shift"] is not None:
            dx, dy, ang_rad, dang = state["shift"]
            for m in k.get("img_metas", []):
                cb = np.asarray(m["can_bus"])
                cb[0], cb[1] = dx, dy
                cb[-2] = ang_rad
                cb[-1] = dang
        return orig_gbf(self, *a, **k)
    tcls.get_bev_features = gbf

    def predict(s, f, const, shift, depth=6):
        state["const"] = const
        model.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                 "prev_pos": 0, "prev_angle": 0}
        state["shift"] = shift
        for k in range(depth, 0, -1):
            if (s, f - k * 5) not in k2:
                continue
            with torch.no_grad():
                mm(return_loss=False, rescale=True,
                   **collate([ds[k2[(s, f - k * 5)]]], samples_per_gpu=1))
        with torch.no_grad():
            out = mm(return_loss=False, rescale=True,
                     **collate([ds[k2[(s, f)]]], samples_per_gpu=1))
        state["const"] = False
        state["shift"] = None
        return out[0]["pts_bbox"]["ego_fut_preds"].double().numpy()

    try:
        print(f"\n앵커 {len(keys)}개, 깊이 6")
        print(f"\n=== S1: visual feature 상수 (0.37) + shift 변경 ===")
        print(f"{'shift 조건':<14}{'3초 |p| p50':>13}{'vs 정지 최대차':>16}")
        base = None
        s1 = {}
        for lab, dx, dy, ar, da in SHIFTS:
            P = np.stack([predict(s, f, True, (dx, dy, ar, da)) for s, f in keys])
            s1[lab] = P
            if base is None:
                base = P
            m = np.abs(P - base).max()
            cum = np.cumsum(P[:, 2], axis=1)[:, -1]
            print(f"{lab:<14}{np.median(np.linalg.norm(cum, axis=-1)):>13.3f}{m:>16.3e}")
        d1 = max(np.abs(s1[l] - base).max() for l, *_ in SHIFTS)

        print(f"\n=== S2: visual feature 실데이터 + shift 변경 (대조군) ===")
        print(f"{'shift 조건':<14}{'3초 |p| p50':>13}{'vs 정지 최대차':>16}")
        base2 = None
        s2 = {}
        for lab, dx, dy, ar, da in SHIFTS:
            P = np.stack([predict(s, f, False, (dx, dy, ar, da)) for s, f in keys])
            s2[lab] = P
            if base2 is None:
                base2 = P
            m = np.abs(P - base2).max()
            cum = np.cumsum(P[:, 2], axis=1)[:, -1]
            print(f"{lab:<14}{np.median(np.linalg.norm(cum, axis=-1)):>13.3f}{m:>16.3e}")
        d2 = max(np.abs(s2[l] - base2).max() for l, *_ in SHIFTS)

        print(f"\n=== S3: shift 고정 + visual 상수 vs 실데이터 ===")
        Pc = s1["정지        "]
        Pr = s2["정지        "]
        d3 = np.abs(Pc - Pr).max()
        print(f"  최대차 {d3:.3e}   (영상 content가 출력을 바꾸는 크기)")
    finally:
        cls.extract_feat = orig_ef
        tcls.get_bev_features = orig_gbf

    print("\n=== 판정 ===")
    print(f"  S1 상수영상 + shift 변화  최대차 {d1:.3e}")
    print(f"  S2 실영상   + shift 변화  최대차 {d2:.3e}   (대조군, 유효성)")
    print(f"  S3 shift 고정 + 영상 변화 최대차 {d3:.3e}")
    if d1 < 1e-9:
        print("\n  (a) shift는 영상 content의 sampling 위치만 바꾼다.")
        print("      장면 정보 없이는 trajectory를 만들지 못한다 -- goal의 T6와 같은 성질.")
    else:
        print("\n  (b) shift가 learned positional basis와 결합해 장면 정보 없이도")
        print(f"      출력을 바꾼다 (최대 {d1:.3f} m).")
        print(f"      영상 content의 기여 {d3:.3f} m 와 비교해 크기를 함께 보고해야 한다.")
        print(f"      비율 shift/영상 = {d1/max(d3,1e-12):.3f}")
    print("\n  * 이 값이 0이어야 합법인 것은 아니다. ego motion은 과거 프레임 영상이")
    print("    존재하는 구간의 정보이므로 4항 위반이 아니다. 다만 '영상 정렬용'이라는")
    print("    설명이 어디까지 성립하는지는 이 수치로 말해야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
