#!/usr/bin/env python
"""⑤-C 플래토 원인 진단.

핵심 질문: 학습된 sparse ranker 가 이미지를 실제로 쓰는가, 아니면 후보 빈도
사전분포(정적 순위)만 학습했는가?

비교 기준:
  (a) STATIC  : train 에서 각 anchor 가 D3-argmin 인 빈도 로그 (이미지 무관, 프레임 불변)
  (b) RANDOM  : 무작위 logits
  (c) MODEL   : 학습된 체크포인트
  (d) SHUFFLE : 프레임끼리 이미지를 뒤섞어 넣은 모델 (C5 negative control 과 동형)
  (e) ZERO    : 이미지 0
"""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402


def show(name, lg, T):
    r = C.eval_logits(lg, T["D3gt"], T["goal_xy"], T["cand_end5"],
                      T["anchor_dist"], T["nms_tau"], T["weight"],
                      lambdas=(0.0, 0.1))
    top1 = r["top1"]; o3 = r["oracle3"]; o12 = r["oracle12"]
    sl = r["shortlist_oracle12"]; rz = r["realized"]["0.1"]
    print("  %-32s top1=%.4f o@3=%.4f o@12=%.4f slO@12=%.4f real=%.4f"
          % (name, top1, o3, o12, sl, rz), flush=True)
    return r


def run_logits_transform(model, arr, rows, l2i, device, mode, batch=16):
    """mode: normal | shuffle | zero"""
    from torch.utils.data import DataLoader
    ds = C.SparseFrameDataset(arr, rows)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=6,
                    pin_memory=True)
    model.eval()
    out = []
    l2i = l2i.to(device)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(device, non_blocking=True)
            if mode == "shuffle":
                perm = torch.randperm(img.shape[0], generator=g).to(img.device)
                img = img[perm]
            elif mode == "zero":
                img = torch.zeros_like(img)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model.forward_logits(img, l2ib)
            out.append(lg.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    split = C.make_split(arr, all_frames=True)
    tune = split["tune_rows"]
    T = C.precompute_targets(arr, tune, bank)
    l2i = torch.from_numpy(C.build_global_lidar2img())
    print("=== tuneval n=%d (30 scene, frame>=30, plain weight) ===" % len(tune))

    # (a) static frequency prior
    tr = split["train_rows"]
    An = bank["anchors_abs"].astype(np.float64)
    gt3 = arr["fut"][tr].astype(np.float64)
    cnt = np.zeros(An.shape[0])
    for s in range(0, len(tr), 4000):
        G = gt3[s:s + 4000]
        D = (np.linalg.norm(G[:, None] - An[None], axis=-1) * C.CW3).sum(-1)
        np.add.at(cnt, D.argmin(1), 1)
    prior = np.log(cnt + 1e-3)
    show("(a) STATIC freq prior", np.tile(prior, (len(tune), 1)), T)
    print("      (상위 12 anchor 가 train argmin 의 %.1f%% 차지)"
          % (100 * np.sort(cnt)[::-1][:12].sum() / cnt.sum()))

    # (b) random
    rng = np.random.default_rng(0)
    show("(b) RANDOM logits", rng.standard_normal((len(tune), An.shape[0])), T)

    # (c)(d)(e) model
    for run in ["d3_s0", "d3_s1", "d3aux5_s0"]:
        p = os.path.join(A, "work_dirs", "sc_" + run, "best.pth")
        if not os.path.isfile(p):
            continue
        ck = torch.load(p, map_location="cpu")
        fn = bool(ck.get("args", {}).get("feature_norm", 0))
        m = TrainableSparseScoreDrive(C.BANK_A0, feature_norm=fn).to(dev)
        m.load_state_dict(ck["model"])
        print("--- %s best step=%s ---" % (run, ck.get("step")))
        lgn = run_logits_transform(m, arr, tune, l2i, dev, "normal")
        show("(c) MODEL normal", lgn, T)
        lgs = run_logits_transform(m, arr, tune, l2i, dev, "shuffle")
        show("(d) MODEL image-SHUFFLE", lgs, T)
        lgz = run_logits_transform(m, arr, tune, l2i, dev, "zero")
        show("(e) MODEL image-ZERO", lgz, T)
        # 이미지가 순위를 바꾸는 정도
        a1 = lgn.argmax(1); a2 = lgs.argmax(1)
        print("      top-1 이 shuffle 로 바뀐 비율: %.1f%%" % (100 * (a1 != a2).mean()))
        print("      logits 프레임간 std(mean over K): normal=%.4f shuffle=%.4f"
              % (lgn.std(0).mean(), lgs.std(0).mean()))
        print("      top-1 후보 몇 종류만 쓰는가: unique=%d / %d frames"
              % (len(np.unique(a1)), len(a1)))
        del m
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
