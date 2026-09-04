#!/usr/bin/env python
"""학습 경로에서 무엇이 0.30을 만드는가 — 영상인가 can_bus(ego 변위)인가.

배경
----
epoch 16 tvad_metric_overfit, 같은 8개 시나리오·같은 앵커·같은 metric:
    학습 경로 (queue)                 0.30
    streaming evaluator               3.53
    evaluator + 영상 전부 0            3.65   (영상 기여 0.12뿐)
    evaluator + prev_bev 리셋          8.26   (단, forward_test가 can_bus까지 0으로
                                              만들므로 ego 변위 제거가 섞여 있다)
영상·lidar2img 텐서는 두 경로에서 **완전히 동일**(최대차 0.000e+00)함을
scripts/etri_pipeline_diff.py 로 확인했다. 남은 입력은 can_bus뿐이다.

`use_can_bus=False`로 learned embedding은 막았지만 `use_shift=True`,
`rotate_prev_bev=True`가 여전히 `can_bus[0], can_bus[1], can_bus[-2], can_bus[-1]`을
읽는다. 그건 0.5초 ego 변위 = **속도**다.

    image-zero  : 현재+과거 큐 영상 전부 0
    canbus-zero : can_bus[:3]=0, can_bus[-1]=0  -> shift/rotate 무력화
    both        : 둘 다

0.30이 can_bus 제거로 무너지면 이 overfit은 **영상 기억이 아니라 ego 변위 기억**이다.

    python scripts/etri_trainpath_ablation.py --ckpt .../epoch_16.pth
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
MODES = ("none", "image-zero", "canbus-zero", "both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=160, help="샘플 수")
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
    # frame%5==0 이고 filter_empty_gt를 통과하는(= 대체되지 않는) 인덱스만 쓴다.
    idxs = [j for j, i in enumerate(infos)
            if int(i["frame_idx"]) % 5 == 0 and int(i["frame_idx"]) >= 20]

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
    model.eval()          # dropout/GridMask 끔 (0.3029로 이미 확인)

    # ★ 샘플을 미리 만들어 모아두면 안 된다. `get_data_info`가
    #   `input_dict['can_bus'] = info['can_bus']` 로 **data_infos의 배열을 그대로 참조**하고
    #   `union2one`이 그 배열을 in-place로 깎는다. N개를 먼저 빌드하면 나중 빌드가
    #   앞 샘플의 can_bus를 덮어써 전부 오염된다(실측: 0.317 -> 5.15로 왜곡됐다).
    #   반드시 **빌드 직후 곧바로 forward** 한다.
    idxs = idxs[:args.n]
    print(f"앵커 {len(idxs)} (frame%5==0, frame>=20)")

    def ablate(data, mode):
        if mode in ("image-zero", "both"):
            data["img"].data[0].zero_()
        if mode in ("canbus-zero", "both"):
            for meta in data["img_metas"].data[0]:      # {0:…,1:…,2:…}
                for k in meta:
                    cb = np.asarray(meta[k]["can_bus"])
                    cb[:3] = 0.0
                    cb[-1] = 0.0
        return data

    print(f"\n{'조건':<14}{'가중 챌린지 L2':>16}{'3초 |p| p50':>13}{'GT |p| p50':>12}")
    from projects.mmdet3d_plugin.VAD.plan_metric_loss import PlanChallengeL2Loss
    grab = {}
    orig = PlanChallengeL2Loss.forward

    def spy(self, pred_delta, gt_delta, cmd_onehot, masks=None):
        B = cmd_onehot.shape[0]
        M = cmd_onehot.reshape(B, -1).shape[1]
        ci = cmd_onehot.reshape(B, M).argmax(dim=-1)
        b = torch.arange(B, device=pred_delta.device)
        grab["p"] = pred_delta[b, ci].detach().float().cpu().numpy()
        grab["g"] = (gt_delta[b, ci] if gt_delta.dim() == 4
                     else gt_delta).detach().float().cpu().numpy()
        return orig(self, pred_delta, gt_delta, cmd_onehot, masks)
    PlanChallengeL2Loss.forward = spy

    try:
        for mode in MODES:
            L = []
            pm, gm = [], []
            for j in idxs:
                d = ablate(collate([ds[int(j)]], samples_per_gpu=1), mode)
                with torch.no_grad():
                    out = model(**d)
                L.append(float(out["loss_plan_metric"]))
                pm.append(np.linalg.norm(np.cumsum(grab["p"][0], 0)[-1]))
                gm.append(np.linalg.norm(np.cumsum(grab["g"][0], 0)[-1]))
            print(f"{mode:<14}{np.mean(L):>16.4f}{np.median(pm):>13.3f}"
                  f"{np.median(gm):>12.3f}", flush=True)
    finally:
        PlanChallengeL2Loss.forward = orig

    print("\n참조: streaming evaluator 3.53 / evaluator+영상0 3.65")
    return 0


if __name__ == "__main__":
    sys.exit(main())
