#!/usr/bin/env python
"""`prev_bev=None`이면 can_bus가 출력에 **전혀** 영향을 주지 않는가.

왜 필요한가
----------
`current-only` ablation은 앵커마다 `prev_bev`를 버린다. 그런데 `forward_test`는
`prev_bev is None`일 때 `can_bus[:3]=0, can_bus[-1]=0`으로 만든다(VAD.py:310-311).
그래서 나는 "current-only의 +8.29에는 ego 변위 제거가 섞여 있다"고 보고했다.

코드상 반론이 있다. `use_can_bus=False`, `ego_his_encoder=None`,
`ego_lcf_feat_idx=None`이므로 can_bus의 위치·회전은 **prev_bev를 정렬할 때만** 쓰인다.
정렬할 prev_bev가 없으면 can_bus를 어떻게 만져도 출력이 같아야 한다.

    A  prev_bev=None,  can_bus 원본
    B  prev_bev=None,  can_bus[:3]=0, can_bus[-1]=0     (forward_test가 하는 것)
    C  prev_bev=None,  can_bus 전체를 난수로
    D  prev_bev 있음,   can_bus 원본 vs 난수             <- 민감도 대조군

기대: max|A-B| = max|A-C| = 0. 그러면 current-only는 순수 temporal BEV 제거 효과다.
D에서 차이가 나야 테스트가 무력하지 않다는 근거가 된다.

    python scripts/etri_canbus_nopath.py --ckpt .../epoch_8.pth
"""
import argparse
import copy
import importlib
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
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
          for j, i in enumerate(ds.data_infos)}          # dataset 순서!
    want = sorted({i["scene_token"] for i in
                   pickle.load(open(SUB, "rb"))["infos"]})
    keys = [(s, f) for s in want[:4] for f in range(30, 300, 5)][:args.n]
    print(f"앵커 {len(keys)}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}  "
          f"unexpected {len(r.unexpected_keys)}")
    print(f"use_can_bus={model.pts_bbox_head.transformer.use_can_bus}  "
          f"use_shift={model.pts_bbox_head.transformer.use_shift}  "
          f"rotate_prev_bev={model.pts_bbox_head.transformer.rotate_prev_bev}")
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    rng = np.random.default_rng(0)

    # can_bus를 **모델이 실제로 읽는 지점**에서 덮어쓴다. forward_test가
    # prev_bev=None일 때 can_bus를 0으로 만드는 분기(VAD.py:310-311)를 우회하려면
    # simple_test를 직접 부르는 게 아니라 여기를 잡는 것이 맞다
    # (simple_test는 gt_bboxes_3d를 [0][0]로 인덱싱해서 DataContainer를 못 받는다).
    tcls = type(model.pts_bbox_head.transformer)
    orig_gbf = tcls.get_bev_features
    state = {"mode": "zero", "saved": None}

    def gbf(self, *a, **k):
        for m in k.get("img_metas", []):
            cb = np.asarray(m["can_bus"])
            if state["mode"] == "keep" and state["saved"] is not None:
                cb[:] = state["saved"]
            elif state["mode"] == "rand":
                cb[:] = rng.normal(0, 10, cb.shape)
            # "zero"는 forward_test가 이미 한 것을 그대로 둔다
        return orig_gbf(self, *a, **k)
    tcls.get_bev_features = gbf

    def predict(s, f, mode, depth=0):
        model.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                 "prev_pos": 0, "prev_angle": 0}
        state["mode"] = "zero"      # warmup은 손대지 않는다
        for k in range(depth, 0, -1):
            if (s, f - k * 5) not in k2:
                continue
            with torch.no_grad():
                mm(return_loss=False, rescale=True,
                   **collate([ds[k2[(s, f - k * 5)]]], samples_per_gpu=1))
        data = collate([ds[k2[(s, f)]]], samples_per_gpu=1)
        # 파이프라인이 준 원본 can_bus (forward_test가 아직 건드리지 않은 값)
        state["saved"] = np.asarray(
            data["img_metas"][0].data[0][0]["can_bus"], dtype=float).copy()
        state["mode"] = mode
        with torch.no_grad():
            out = mm(return_loss=False, rescale=True, **data)
        state["mode"] = "zero"
        return out[0]["pts_bbox"]["ego_fut_preds"].double().numpy()

    def sweep(mode, depth):
        return np.stack([predict(s, f, mode, depth) for s, f in keys])

    try:
        print("\n=== prev_bev = None (current-only 조건) ===")
        A = sweep("keep", 0)
        B = sweep("zero", 0)
        C = sweep("rand", 0)
        print(f"  max|A(원본) - B(0 처리)|   {np.abs(A - B).max():.3e}")
        print(f"  max|A(원본) - C(난수)|     {np.abs(A - C).max():.3e}")

        print("\n=== prev_bev 있음 (깊이 6) -- 민감도 대조군 ===")
        D1 = sweep("keep", 6)
        D2 = sweep("rand", 6)
        print(f"  max|원본 - 난수|           {np.abs(D1 - D2).max():.3e}")
    finally:
        tcls.get_bev_features = orig_gbf

    ok = np.abs(A - B).max() < 1e-9 and np.abs(A - C).max() < 1e-9
    ctrl = np.abs(D1 - D2).max() > 1e-6
    print("\n=== 판정 ===")
    print(f"  prev_bev 없으면 can_bus 무영향: {'PASS' if ok else 'FAIL'}")
    print(f"  민감도 대조군(테스트 유효성) : {'PASS' if ctrl else 'FAIL -- 테스트가 무력'}")
    if ok and ctrl:
        print("  ==> current-only의 악화는 **순수 temporal BEV 제거 효과**다.")
        print("      can_bus 제거가 섞였다는 내 앞선 보고는 틀렸다.")
        print("      그리고 can_bus에는 prev_bev 정렬 외의 경로가 없다(컴플라이언스 근거).")
    elif not ok:
        print("  ==> can_bus에 prev_bev 정렬 외의 경로가 남아 있다. 찾아야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
