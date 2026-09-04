#!/usr/bin/env python
"""한 앵커에 대해 VADHead가 **실제로 받는 입력과 내는 출력**을 두 경로에서 덤프해 비교한다.

여기까지의 이분법 결과 (같은 prev_bev 주입, 같은 프레임, 둘 다 eval 모드):
    1 학습 head + 학습 prev_bev   가중 L2 0.3662   3초 |p| 23.59  (GT 22.99)
    2 배포 head + 학습 prev_bev            3.2609             5.21
    3 배포 head + 배포 prev_bev            5.8850            36.09
코드상 head 호출은 두 경로가 **문자 그대로 동일**하다.
    forward_pts_train : self.pts_bbox_head(pts_feats, img_metas, prev_bev, ego_his_trajs=…, ego_lcf_feat=…)
    simple_test_pts   : self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, ego_his_trajs=…, ego_lcf_feat=…)
그러면 입력 텐서 중 무엇이 다르다. 추측을 멈추고 전수 비교한다.

3개 mode 출력을 전부 찍는다. 붕괴(|p| 5 m)가 **학습 안 된 mode를 골랐기 때문**인지
출력 자체가 다른지 구분하기 위해서다.
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
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame", type=int, default=100)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg.data.train.ann_file = SUB
    cfg.data.train.temporal_shuffle = False
    ds_tr = build_dataset(cfg.data.train)
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ANN
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds_te = build_dataset(cfg.data.test)
    infos_tr = pickle.load(open(SUB, "rb"))["infos"]
    infos_te = pickle.load(open(ANN, "rb"))["infos"]
    k2 = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(infos_te)}
    j = [n for n, i in enumerate(infos_tr) if int(i["frame_idx"]) == args.frame][0]
    s, f = infos_tr[j]["scene_token"], int(infos_tr[j]["frame_idx"])
    print(f"앵커 {s} frame {f}   train idx {j}   test idx {k2[(s, f)]}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                        test_cfg=cfg.get("test_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    head = model.pts_bbox_head
    cls_h = type(head)
    orig_fwd = cls_h.forward
    log = {}
    tag = ["?"]

    def fwd(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, **kw):
        out = orig_fwd(self, mlvl_feats, img_metas, prev_bev=prev_bev,
                       only_bev=only_bev, **kw)
        if only_bev:
            return out
        cb = np.asarray(img_metas[0]["can_bus"], dtype=float)
        log[tag[0]] = dict(
            n_lvl=len(mlvl_feats),
            feat_shape=tuple(mlvl_feats[0].shape),
            feat_sum=float(mlvl_feats[0].double().sum()),
            feat_absmean=float(mlvl_feats[0].double().abs().mean()),
            prev_none=prev_bev is None,
            prev_shape=None if prev_bev is None else tuple(prev_bev.shape),
            prev_sum=None if prev_bev is None else float(prev_bev.double().sum()),
            can_bus=cb.copy(),
            n_meta=len(img_metas),
            ego=out["ego_fut_preds"].detach().double().cpu().numpy().copy(),
        )
        return out
    cls_h.forward = fwd

    cls_m = type(model)
    orig_hist = cls_m.obtain_history_bev
    stash = {}

    def hist(self, iq, iml):
        o = orig_hist(self, iq, iml)
        self.eval()
        stash["bev"] = o.detach().clone()
        return o
    cls_m.obtain_history_bev = hist

    try:
        tag[0] = "train"
        with torch.no_grad():
            mm(**collate([ds_tr[int(j)]], samples_per_gpu=1))
        bev_train = stash["bev"]

        tag[0] = "test_inject"
        model.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                 "prev_pos": 0, "prev_angle": 0}
        with torch.no_grad():
            mm(return_loss=False, rescale=True,
               **collate([ds_te[k2[(s, f - 5)]]], samples_per_gpu=1))
        model.prev_frame_info["prev_bev"] = bev_train
        model.prev_frame_info["scene_token"] = s
        d_te = collate([ds_te[k2[(s, f)]]], samples_per_gpu=1)
        with torch.no_grad():
            mm(return_loss=False, rescale=True, **d_te)
    finally:
        cls_h.forward = orig_fwd
        cls_m.obtain_history_bev = orig_hist

    a, b = log["train"], log["test_inject"]
    print(f"\n{'항목':<20}{'train':>26}{'test(주입)':>26}{'같음':>6}")
    for k in ("n_lvl", "feat_shape", "n_meta", "prev_none", "prev_shape"):
        print(f"{k:<20}{str(a[k]):>26}{str(b[k]):>26}"
              f"{'O' if a[k] == b[k] else 'X':>6}")
    for k in ("feat_sum", "feat_absmean", "prev_sum"):
        av, bv = a[k], b[k]
        same = (av is None and bv is None) or (
            av is not None and bv is not None and abs(av - bv) < 1e-6 * max(1, abs(av)))
        print(f"{k:<20}{av if av is None else f'{av:26.6f}'}"
              f"{bv if bv is None else f'{bv:26.6f}'}{'O' if same else 'X':>6}")
    print(f"\ncan_bus 18차원  (train / test)  최대차 {np.abs(a['can_bus']-b['can_bus']).max():.3e}")
    for i in range(18):
        d = abs(a["can_bus"][i] - b["can_bus"][i])
        mark = "  <== 다름" if d > 1e-9 else ""
        print(f"  [{i:2d}]  {a['can_bus'][i]:>14.6f}  {b['can_bus'][i]:>14.6f}{mark}")

    print("\nego_fut_preds 3 mode 각각의 누적 3초 |p| (m)")
    print(f"{'mode':>6}{'train':>12}{'test(주입)':>14}")
    for m in range(a["ego"].shape[1] if a["ego"].ndim == 4 else a["ego"].shape[0]):
        pa = np.cumsum(a["ego"].reshape(-1, a["ego"].shape[-3], 6, 2)[0, m], 0)[-1]
        pb = np.cumsum(b["ego"].reshape(-1, b["ego"].shape[-3], 6, 2)[0, m], 0)[-1]
        print(f"{m:>6}{np.linalg.norm(pa):>12.3f}{np.linalg.norm(pb):>14.3f}")
    print(f"\nego_fut_preds 전체 최대절대차 {np.abs(a['ego']-b['ego']).max():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
