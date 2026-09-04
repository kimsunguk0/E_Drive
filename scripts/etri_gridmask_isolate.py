#!/usr/bin/env python
"""학습 경로 0.32의 정체 — 현재 프레임이 train 모드로 도는 것 때문인가.

왜 이걸 마지막에 봐야 하는가
--------------------------
`VAD.obtain_history_bev`는 마지막에 **`self.train()`** 을 호출한다(VAD.py:200).
그래서 `forward_train`에서 현재 프레임은 우리가 `model.eval()`을 걸어놨든 말든
**항상 train 모드**로 처리된다 -- GridMask ON, dropout ON.
반면 배포(`forward_test`)는 eval 모드다: GridMask OFF, dropout OFF.

내가 앞서 "BN/dropout 아님(train 0.3149 vs eval 0.3029)"이라고 판정한 건 틀렸다.
두 측정 모두 현재 프레임이 train 모드였다. 격리가 안 된 측정이었다.

여기서는 `obtain_history_bev`의 꼬리 `self.train()`을 `self.eval()`로 바꿔
**현재 프레임까지 eval 모드**로 만든 뒤 같은 앵커를 다시 잰다.

    A  원본           현재 프레임 train 모드 (GridMask ON,  dropout ON)
    B  패치           현재 프레임 eval  모드 (GridMask OFF, dropout OFF)
    C  B + GridMask   use_grid_mask=False 로 학습·평가 모두 OFF

B가 무너지면 원인은 **GridMask/dropout 학습-배포 불일치**다.

    python scripts/etri_gridmask_isolate.py --ckpt .../epoch_16.pth
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
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=240)
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
    idxs = [j for j, i in enumerate(infos) if int(i["frame_idx"]) % 5 == 0][:args.n]
    print(f"앵커 {len(idxs)}")

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
    print(f"use_grid_mask = {getattr(model, 'use_grid_mask', None)}")
    mm = MMDataParallel(model.cuda(0), device_ids=[0])

    from projects.mmdet3d_plugin.VAD.plan_metric_loss import PlanChallengeL2Loss
    grab = {}
    ol = PlanChallengeL2Loss.forward

    def spy(self, pred_delta, gt_delta, cmd_onehot, masks=None):
        B = pred_delta.shape[0]
        ci = cmd_onehot.reshape(B, -1).argmax(dim=-1)
        b = torch.arange(B, device=pred_delta.device)
        grab["p"] = pred_delta[b, ci].detach().float().cpu().numpy()
        grab["g"] = (gt_delta[b, ci] if gt_delta.dim() == 4
                     else gt_delta).detach().float().cpu().numpy()
        return ol(self, pred_delta, gt_delta, cmd_onehot, masks)
    PlanChallengeL2Loss.forward = spy

    # obtain_history_bev의 꼬리 self.train()을 바꿀 수 있게 감싼다
    cls = type(model)
    orig_hist = cls.obtain_history_bev

    def hist_eval(self, imgs_queue, img_metas_list):
        out = orig_hist(self, imgs_queue, img_metas_list)
        self.eval()          # 원본은 여기서 self.train() 이었다
        return out

    def sweep(label):
        L, PM, GM = [], [], []
        for j in idxs:
            data = collate([ds[int(j)]], samples_per_gpu=1)
            with torch.no_grad():
                out = mm(**data)
            L.append(float(out["loss_plan_metric"]))
            p = np.cumsum(grab["p"][0], 0)
            g = np.cumsum(grab["g"][0], 0)
            PM.append(np.linalg.norm(p[-1]))
            GM.append(np.linalg.norm(g[-1]))
        print(f"{label:<44}{np.mean(L):>10.4f}{np.median(PM):>13.3f}"
              f"{np.median(GM):>12.3f}", flush=True)
        return np.mean(L)

    print(f"\n{'조건':<44}{'가중 L2':>10}{'3초 |p| p50':>13}{'GT |p| p50':>12}")
    try:
        mm.train()
        a = sweep("A 원본 (현재프레임 train: GridMask+dropout ON)")

        cls.obtain_history_bev = hist_eval
        mm.eval()
        b = sweep("B 현재프레임 eval (GridMask+dropout OFF)")

        model.use_grid_mask = False
        c = sweep("C B + use_grid_mask=False")
    finally:
        cls.obtain_history_bev = orig_hist
        PlanChallengeL2Loss.forward = ol

    print("\n참조: streaming evaluator(깊이 59) 3.5335 / 깊이 6 3.5371")
    print("\n=== 판정 ===")
    if b > a * 2:
        print(f"  현재프레임을 eval로 바꾸자 {a:.3f} -> {b:.3f} 로 무너졌다.")
        print("  ==> 원인 확정: **GridMask/dropout 학습-배포 불일치**.")
        print(f"      GridMask까지 끄면 {c:.3f} (그 안에서 GridMask 몫을 분리).")
    else:
        print(f"  eval 모드에서도 {b:.3f} 로 낮다 ==> train/eval 모드가 원인이 아니다.")
        print("      남은 후보는 prev_bev 생성 경로(only_bev vs simple_test)뿐이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
