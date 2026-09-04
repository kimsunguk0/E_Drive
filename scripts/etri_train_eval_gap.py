#!/usr/bin/env python
"""train loss와 실제 evaluator의 괴리를 분해한다.

증상
----
tvad_metric_overfit epoch 16:
    학습 로그 loss_plan_metric  0.3162   (= 가중 챌린지 L2, loss_weight=1.0)
    evaluator  8-scene 480앵커  3.5335
같은 데이터·같은 metric인데 11배다. 그리고 evaluator 값은 epoch 2부터 3.4~3.6에
**갇혀 있다** -- 학습 loss만 내려갔다.

`scripts/test_plan_metric_loss.py`가 loss==metric 등가를 이미 8/8로 증명했으므로
metric 변환 문제가 아니다. **예측 자체가 두 경로에서 다르다.**

두 경로의 차이는 셋뿐이다.
    (1) model.train() vs model.eval()          -- BN running stats, dropout
    (2) train pipeline vs test pipeline        -- 전처리/증강
    (3) prev_bev 만드는 방법
          train: obtain_history_bev(queue 3프레임 = -1.5/-1.0/-0.5s, 매 샘플 새로)
          eval : streaming (직전 앵커의 bev를 이어받아 시나리오 전체를 누적)

여기서는 (1)을 격리한다. **train 경로를 그대로 두고 eval()/train()만 바꿔** 같은
앵커의 챌린지 L2를 잰다. 그리고 그 예측을 evaluator가 저장한 pred와 앵커별로
직접 비교한다.

    (a) train경로 + train()   ~ 학습 로그와 같아야 한다
    (b) train경로 + eval()    -- 여기서 무너지면 원인은 BN
    (c) evaluator (test경로 + eval() + streaming) = 3.5335 (기존 측정)

    python scripts/etri_train_eval_gap.py --ckpt .../epoch_16.pth
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
ADCL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
SUB = "/tmp/pm97/data/etri/pkl/overfit8.pkl"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
W = np.array([11.0, 11.0, 5.0, 5.0, 2.0, 2.0]) / 36.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_metric_overfit.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-pred", default="/tmp/pm97/eval_overfit/ep16_pred.npy")
    ap.add_argument("--n", type=int, default=0, help="앵커 N개만 (0=전체 480)")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt)
    args.eval_pred = os.path.abspath(args.eval_pred) if args.eval_pred else ""
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

    # ---- 학습과 완전히 같은 dataset (train pipeline, overfit8 ann) ----
    cfg.data.train.ann_file = SUB
    ds = build_dataset(cfg.data.train)
    infos = pickle.load(open(SUB, "rb"))["infos"]
    print(f"train dataset {type(ds).__name__} {len(ds):,}  infos {len(infos):,}")

    # evaluator가 쓴 앵커와 **같은 격자**: frame % 5 == 0
    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    want = sorted({i["scene_token"] for i in infos})
    cidx = np.where(np.isin(scen_all, want) & (frame_all % 5 == 0))[0]
    cidx = cidx[np.lexsort((frame_all[cidx], scen_all[cidx]))]
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(infos)}
    keys = [(scen_all[c], int(frame_all[c])) for c in cidx]
    dsi = [key2ds[k] for k in keys]
    if args.n:
        keys, dsi, cidx = keys[:args.n], dsi[:args.n], cidx[:args.n]
    print(f"앵커 {len(keys)}  (frame%5==0, evaluator와 동일 격자)")
    gt_cum = d["fut"][cidx].astype(np.float64)          # 누적 GT [N,6,2]

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

    # ---- loss 호출을 가로채 예측 증분을 그대로 뽑는다 ----
    from projects.mmdet3d_plugin.VAD.plan_metric_loss import PlanChallengeL2Loss
    grab = {}
    orig = PlanChallengeL2Loss.forward

    def spy(self, pred_delta, gt_delta, cmd_onehot, masks=None):
        B, M = cmd_onehot.reshape(cmd_onehot.shape[0], -1).shape
        ci = cmd_onehot.reshape(B, M).argmax(dim=-1)
        b = torch.arange(B, device=pred_delta.device)
        grab["pred"] = pred_delta[b, ci].detach().float().cpu().numpy()
        grab["gt"] = (gt_delta[b, ci] if gt_delta.dim() == 4
                      else gt_delta).detach().float().cpu().numpy()
        grab["cmd"] = ci.detach().cpu().numpy()
        return orig(self, pred_delta, gt_delta, cmd_onehot, masks)
    PlanChallengeL2Loss.forward = spy

    def sweep(mode):
        getattr(model, mode)()
        P = np.zeros((len(keys), 6, 2))
        G = np.zeros((len(keys), 6, 2))
        C = np.zeros(len(keys), dtype=int)
        L = np.zeros(len(keys))
        for n, j in enumerate(dsi):
            data = collate([ds[int(j)]], samples_per_gpu=1)
            with torch.no_grad():
                out = model(**data)
            L[n] = float(out["loss_plan_metric"])
            P[n] = np.cumsum(grab["pred"][0], axis=0)
            G[n] = np.cumsum(grab["gt"][0], axis=0)
            C[n] = grab["cmd"][0]
            if (n + 1) % 120 == 0:
                print(f"    {n+1}/{len(keys)}", flush=True)
        return P, G, C, L

    def wl2(P, G):
        return float((np.sqrt(((P - G) ** 2).sum(-1)) * W).sum(-1).mean())

    print("\n(a) train경로 + model.train()")
    Pa, Ga, Ca, La = sweep("train")
    print(f"    loss_plan_metric 평균 {La.mean():.4f}   재계산 가중 L2 {wl2(Pa, Ga):.4f}")
    print("\n(b) train경로 + model.eval()")
    Pb, Gb, Cb, Lb = sweep("eval")
    print(f"    loss_plan_metric 평균 {Lb.mean():.4f}   재계산 가중 L2 {wl2(Pb, Gb):.4f}")

    PlanChallengeL2Loss.forward = orig

    print("\n=== GT 정합 ===")
    print(f"  train GT(cumsum) vs ego_cache GT  최대차 {np.abs(Ga - gt_cum).max():.3e}")
    print(f"  train/eval 모드 간 GT 동일        {np.array_equal(Ga, Gb)}")
    print(f"  cmd 동일                          {np.array_equal(Ca, Cb)}")
    print(f"  cmd 분포 {np.bincount(Ca, minlength=3)}")
    print(f"  ego_cache 기준 가중 L2  (a) {wl2(Pa, gt_cum):.4f}   (b) {wl2(Pb, gt_cum):.4f}")

    if args.eval_pred and os.path.exists(args.eval_pred):
        Pe = np.load(args.eval_pred)
        if Pe.shape[0] == len(keys):
            print("\n=== (c) evaluator 예측과의 직접 비교 ===")
            print(f"  evaluator 가중 L2                 {wl2(Pe, gt_cum):.4f}")
            print(f"  |P_eval - P_train경로| 3초 성분 p50 "
                  f"{np.median(np.linalg.norm(Pe[:, -1] - Pb[:, -1], axis=-1)):.4f} m")
            for t, lab in ((1, "1초"), (3, "2초"), (5, "3초")):
                print(f"    {lab}: train경로|p| p50 {np.median(np.linalg.norm(Pb[:,t],axis=-1)):7.3f}"
                      f"   evaluator {np.median(np.linalg.norm(Pe[:,t],axis=-1)):7.3f}"
                      f"   GT {np.median(np.linalg.norm(gt_cum[:,t],axis=-1)):7.3f}")
        else:
            print(f"\n  (evaluator pred shape {Pe.shape} != 앵커 {len(keys)} -- 비교 생략)")

    print("\n=== 판정 ===")
    a, b = wl2(Pa, gt_cum), wl2(Pb, gt_cum)
    if b > a * 2:
        print(f"  eval() 만으로 {a:.3f} -> {b:.3f} 붕괴  ==> 원인은 BN/train-mode 의존")
    elif abs(b - 3.53) < 0.5:
        print(f"  train경로도 eval()에서 {b:.3f}  ==> BN 문제. evaluator와 일치")
    else:
        print(f"  train경로는 eval()에서도 {b:.3f}로 낮다  ==> 원인은 전처리 또는 "
              f"prev_bev 구성(streaming vs queue)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
