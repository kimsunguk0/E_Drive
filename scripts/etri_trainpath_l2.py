#!/usr/bin/env python
"""학습 경로의 챌린지 L2를 앵커별·시나리오별로 전수 기록한다.

두 측정이 어긋났다.
    etri_train_eval_gap.py      ds[j],                 deepcopy 없음  -> 0.303 (480앵커)
    etri_trainpath_ablation.py  prepare_train_data(j),  deepcopy 함    -> 5.150 (앞 160개)
차이는 (a) 샘플 획득 방법, (b) deepcopy, (c) 부분집합 셋뿐이다. 세 가지를 한 프로세스에서
같은 앵커로 교차 측정해 어느 것이 원인인지 못 박는다.

    python scripts/etri_trainpath_l2.py --ckpt .../epoch_16.pth
"""
import argparse
import copy
import importlib
import os
import pickle
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
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
    ds = build_dataset(cfg.data.train)
    infos = pickle.load(open(SUB, "rb"))["infos"]
    anc = [j for j, i in enumerate(infos) if int(i["frame_idx"]) % 5 == 0]
    scen = np.array([infos[j]["scene_token"] for j in anc])
    print(f"앵커 {len(anc)}  시나리오 {len(set(scen))}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}  "
          f"unexpected {len(r.unexpected_keys)}")
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    from projects.mmdet3d_plugin.VAD.plan_metric_loss import PlanChallengeL2Loss
    grab = {}
    orig = PlanChallengeL2Loss.forward

    def spy(self, pred_delta, gt_delta, cmd_onehot, masks=None):
        B = pred_delta.shape[0]
        ci = cmd_onehot.reshape(B, -1).argmax(dim=-1)
        b = torch.arange(B, device=pred_delta.device)
        grab["p"] = pred_delta[b, ci].detach().float().cpu().numpy()
        grab["g"] = (gt_delta[b, ci] if gt_delta.dim() == 4
                     else gt_delta).detach().float().cpu().numpy()
        return orig(self, pred_delta, gt_delta, cmd_onehot, masks)
    PlanChallengeL2Loss.forward = spy

    def one(data):
        with torch.no_grad():
            out = model(**data)
        p = np.cumsum(grab["p"][0], 0)
        g = np.cumsum(grab["g"][0], 0)
        return (float(out["loss_plan_metric"]),
                float((np.sqrt(((p - g) ** 2).sum(-1)) * W).sum()),
                np.linalg.norm(p[-1]), np.linalg.norm(g[-1]))

    variants = {
        "A: ds[j], deepcopy 없음": lambda j: collate([ds[int(j)]], samples_per_gpu=1),
        "B: ds[j], deepcopy":      lambda j: copy.deepcopy(
            collate([ds[int(j)]], samples_per_gpu=1)),
    }
    res = {}
    try:
        for name, mk in variants.items():
            L = np.zeros(len(anc)); PM = np.zeros(len(anc)); GM = np.zeros(len(anc))
            for n, j in enumerate(anc):
                L[n], _, PM[n], GM[n] = one(mk(j))
                if (n + 1) % 160 == 0:
                    print(f"    {name}  {n+1}/{len(anc)}", flush=True)
            res[name] = (L, PM, GM)
            print(f"  {name:<26} 가중 L2 {L.mean():.4f}   "
                  f"3초 |p| p50 {np.median(PM):.3f}   GT {np.median(GM):.3f}", flush=True)
    finally:
        PlanChallengeL2Loss.forward = orig

    L = res["A: ds[j], deepcopy 없음"][0]
    print(f"\n=== 시나리오별 (A) ===")
    print(f"{'시나리오':<20}{'앵커':>5}{'가중 L2':>10}{'최악 앵커':>10}")
    for s in sorted(set(scen)):
        m = scen == s
        print(f"{s:<20}{int(m.sum()):>5}{L[m].mean():>10.4f}{L[m].max():>10.4f}")
    print(f"\n분위: p50 {np.median(L):.3f}  p90 {np.percentile(L,90):.3f}  "
          f"p99 {np.percentile(L,99):.3f}  max {L.max():.3f}")
    print(f"L2>2.0 앵커 {int((L>2).sum())}/{len(L)}   L2>1.0 {int((L>1).sum())}/{len(L)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
