#!/usr/bin/env python
"""⑤-C 실패 체크포인트 원인분해 (재학습 없음).

branch ablation × image-shuffle × (학습에 쓴 시나리오 / 미학습 시나리오).

branch:  local  = path_score 만 (경로 투영지점 sparse 샘플)
         global = g_term 만 (전체 이미지 평균 feature -> 후보 프로파일 내적)
         both   = 현재 모델
image:   normal      = 정상 대응
         same_scene  = 같은 시나리오의 다른 frame 이미지로 교체
         cross_scene = 다른 시나리오의 frame 이미지로 교체
split:   train  = 학습에 쓴 시나리오(모델이 본 프레임)
         tune   = tuneval 30 scene(미학습)

판정 규칙 (user spec):
  - train 에서 normal ≈ same_scene  -> 프레임 내용이 아니라 시나리오 정체성 암기
  - global-only > local-only        -> global branch 가 주된 과적합 통로
  - local-only 도 매우 나쁨          -> ImageNet feature 가 투영지점 주행가능성을 표현 못함
"""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402

N_ROWS = 1620          # tuneval 과 동일 크기로 맞춘다
LAM = (0.0, 0.1)


def components(model, arr, img_rows, l2i, device, batch=16):
    """img_rows 순서대로 (path_score, g_term) [N,K] 반환."""
    from torch.utils.data import DataLoader
    dl = DataLoader(C.SparseFrameDataset(arr, img_rows), batch_size=batch,
                    shuffle=False, num_workers=6, pin_memory=True)
    model.eval()
    P, G = [], []
    l2i = l2i.to(device)
    with torch.no_grad():
        for b in dl:
            img = b["img"].to(device, non_blocking=True)
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lv, p4 = model.encode_images(img)
                ev = img.abs().amax(dim=(1, 2, 3, 4)) > 0
                ps, gt, evm = model.score_components(
                    lv, p4, l2ib, (img.shape[-2], img.shape[-1]), ev)
            P.append((ps * evm).float().cpu().numpy())
            G.append((gt * evm).float().cpu().numpy())
    return np.concatenate(P, 0), np.concatenate(G, 0)


def shuffled_rows(arr, gt_rows, mode, rng):
    """gt_rows 와 같은 길이의 이미지 출처 row 배열."""
    if mode == "normal":
        return gt_rows.copy()
    scen = arr["scen_idx"]
    # 시나리오별 전체 프레임 row 목록 (이미지는 300장 모두 존재)
    scenes = np.unique(scen[gt_rows])
    by = {int(s): np.where(scen == s)[0] for s in scenes}
    out = np.empty_like(gt_rows)
    for i, r in enumerate(gt_rows):
        s = int(scen[r])
        if mode == "same_scene":
            pool = by[s]
            pick = int(rng.choice(pool))
            while pick == int(r) and len(pool) > 1:
                pick = int(rng.choice(pool))
        else:  # cross_scene
            other = scenes[scenes != s]
            pool = by[int(rng.choice(other))]
            pick = int(rng.choice(pool))
        out[i] = pick
    return out


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    split = C.make_split(arr, all_frames=True)
    l2i = torch.from_numpy(C.build_global_lidar2img())
    rng = np.random.default_rng(0)

    # 평가 row: tune = tuneval 전부 / train = 학습 시나리오 main frame 에서 동일 개수 표본
    tune_rows = split["tune_rows"]
    tr_all = split["train_rows"]
    main = tr_all[(arr["frame"][tr_all] >= 30) & (arr["frame"][tr_all] % 5 == 0)]
    train_rows = np.sort(rng.choice(main, size=min(N_ROWS, len(main)), replace=False))
    sets = {"train": train_rows, "tune": tune_rows}
    tgts = {k: C.precompute_targets(arr, v, bank) for k, v in sets.items()}
    print("행 수: train=%d (학습에 쓴 시나리오) / tune=%d (미학습)"
          % (len(train_rows), len(tune_rows)), flush=True)

    for run in ["d3_s0", "d3_s1"]:
        p = os.path.join(A, "work_dirs", "sc_" + run, "best.pth")
        if not os.path.isfile(p):
            continue
        ck = torch.load(p, map_location="cpu")
        m = TrainableSparseScoreDrive(
            C.BANK_A0,
            feature_norm=bool(ck.get("args", {}).get("feature_norm", 0)),
            logit_norm=bool(ck.get("args", {}).get("logit_norm", 0))).to(dev)
        m.load_state_dict(ck["model"])
        print("\n" + "=" * 108)
        print("### %s  best step=%s" % (run, ck.get("step")))
        print("=" * 108)
        print("%-6s %-12s %-8s %8s %8s %9s %9s" %
              ("split", "image", "branch", "top1", "o@12", "slO@12", "real.1"))

        # 성분 동일성 검사: both 재결합 == forward_logits
        chk = np.sort(train_rows)[:16]
        ps, gt = components(m, arr, chk, l2i, dev)
        lg_ref = C.run_logits(m, arr, chk, l2i, dev, batch=16, num_workers=2)
        sc = float(m.logit_scale.exp().clamp(max=100.0)) if m.logit_norm else 1.0
        recon = sc * (ps + gt)
        print("[성분 재결합 검사] max|recon - forward_logits| = %.3e" %
              np.abs(recon - lg_ref).max(), flush=True)

        for sname, rows in sets.items():
            T = tgts[sname]
            for mode in ["normal", "same_scene", "cross_scene"]:
                img_rows = shuffled_rows(arr, rows, mode, np.random.default_rng(1))
                ps, gt = components(m, arr, img_rows, l2i, dev)
                variants = {"both": ps + gt, "local": ps, "global": gt}
                for bname, lg in variants.items():
                    r = C.eval_logits(lg.astype(np.float64), T["D3gt"],
                                      T["goal_xy"], T["cand_end5"],
                                      T["anchor_dist"], T["nms_tau"],
                                      T["weight"], lambdas=LAM)
                    print("%-6s %-12s %-8s %8.4f %8.4f %9.4f %9.4f" %
                          (sname, mode, bname, r["top1"], r["oracle12"],
                           r["shortlist_oracle12"], r["realized"]["0.1"]),
                          flush=True)
        del m
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
