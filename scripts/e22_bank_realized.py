#!/usr/bin/env python
"""⑥-E: 새 계층 bank 가 실제로 realized 를 낮추는가 (scorer 재학습 없이).

파이프라인:
  1) 모델이 1024 parent 를 점수화 -> image-only shortlist 로 상위 P 개 parent
  2) 각 parent 를 자기 child 로 전개 (고정 결정적 맵, 이미지·goal 무관)
  3) learned row selector 가 그 중 **행 하나**를 고른다

coverage(oracle) 가 아니라 **realized** 를 본다. child 순위 매기는 scorer 가 아직
없으므로 이 값은 "child 선택을 selector 에게만 맡겼을 때"의 값이다.
child fine-scorer 를 만들면 더 좋아질 여지가 남는다.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402
from train_row_selector import Selector  # noqa: E402

W6 = np.array([11., 11., 5., 5., 2., 2.]); W6 /= W6.sum()


def d3_pair(a, b, chunk=256):
    out = np.empty((len(a), len(b)), np.float32)
    for s in range(0, len(a), chunk):
        d = np.linalg.norm(a[s:s + chunk, None] - b[None], axis=-1)
        out[s:s + chunk] = (d * W6).sum(-1)
    return out


def feats(abs5, lg, bfk, rprofk, rmean, goal, cmd):
    """[n,M,10,2] 후보 + [n,M] logit -> selector 특징. bank 무관하게 좌표에서 직접."""
    n, M = lg.shape
    e5 = abs5[:, :, 9]
    gd = np.linalg.norm(e5 - goal[:, None], axis=-1)
    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)      # noqa: E731
    rank = lambda x: np.argsort(np.argsort(x, 1), 1) / max(M - 1., 1.)  # noqa: E731
    top = np.argmax(lg, 1)
    div = np.linalg.norm(e5 - np.take_along_axis(e5, top[:, None, None], 1), axis=-1)
    pdev = np.abs(rprofk - rmean[None, None]).mean(-1)
    oh = np.zeros((n, M, 3))
    oh[np.arange(n), :, np.clip(cmd, 0, 2)] = 1.0
    return np.concatenate([
        bfk, gd[..., None] / 10.0, nz(gd)[..., None], rank(gd)[..., None],
        (e5[..., 0] - goal[:, None, 0])[..., None] / 10.0,
        (e5[..., 1] - goal[:, None, 1])[..., None] / 10.0,
        nz(lg)[..., None], rank(lg)[..., None],
        (lg - lg.mean(1, keepdims=True))[..., None],
        div[..., None] / 10.0, nz(div)[..., None],
        pdev[..., None] * 10.0, nz(pdev)[..., None], oh], -1).astype(np.float32)


def static_feats(c5):
    z = np.zeros_like(c5[:, :1, :])
    st = np.linalg.norm(np.diff(np.concatenate([z, c5], 1), axis=1), axis=-1)
    S5 = st.sum(-1); S3 = st[:, :6].sum(-1)
    r = np.cumsum(st[:, :6], -1) / np.clip(S3, 1e-3, None)[:, None]
    d = np.diff(np.concatenate([np.zeros((len(c5), 1, 2)), c5], 1), axis=1)
    hd = np.arctan2(d[..., 1], d[..., 0]); dh = np.diff(np.unwrap(hd, axis=1), axis=1)
    bf = np.column_stack([S3 / 30., S5 / 50., c5[:, 5, 0] / 30., c5[:, 5, 1] / 10.,
                          c5[:, 9, 0] / 50., c5[:, 9, 1] / 20., hd[:, 5], hd[:, 9],
                          dh.sum(1), np.abs(dh).sum(1), np.abs(dh).max(1), *r[:, :5].T])
    return bf.astype(np.float32), r.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/etri/bank_hier_K7680.npz")
    ap.add_argument("--ckpt", default="work_dirs/sc_u_ctrl/best.pth")
    ap.add_argument("--sel", default="work_dirs/row_selector_vis.pth")
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(dev)
    arr = C.load_arrays()
    a0 = np.load(C.BANK_A0, allow_pickle=False)
    hb = np.load(os.path.join(A, args.bank))
    kids = hb["candidate_xy_abs_5s"].astype(np.float64)
    kpar = hb["parent_id"]
    bf_k, rp_k = static_feats(kids)
    a0c5 = a0["candidate_xy_abs_5s"].astype(np.float64)
    bf_a, rp_a = static_feats(a0c5)
    rmean = rp_a.mean(0)

    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    ar = ck["args"]; nh = int(ar.get("n_hist", 3))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(ar.get("logit_norm", 1)), n_hist=nh,
        score_weight=ar.get("score_weight", "uniform")).to(dev)
    m.load_state_dict(ck["model"]); m.fuse_mul = bool(ar.get("fuse_mul", 0)); m.eval()
    sck = torch.load(os.path.join(A, args.sel), map_location="cpu")
    sel = Selector(sck["feat_dim"]).to(dev); sel.load_state_dict(sck["model"]); sel.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)
    lg = C.run_logits(m, arr, rows, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=24, amp=True, n_hist=nh).astype(np.float64)
    V3 = arr["fut"][rows].astype(np.float64)
    goal = arr["fut5"][rows][:, 9].astype(np.float64)
    cmd = arr["vad_cmd"][rows].astype(np.int64)
    ad, tau = a0["anchor_dist"].astype(np.float64), float(a0["nms_tau"])
    D_a0 = d3_pair(V3, a0["anchors_abs"].astype(np.float64))
    D_k = d3_pair(V3, kids[:, :6])
    wm = lambda x: float(np.average(x, weights=w))                     # noqa: E731
    n = len(rows)
    print(f"bank={os.path.basename(args.bank)} K={len(kids)}  ckpt={os.path.basename(os.path.dirname(args.ckpt))}"
          f"  val38 n={n}", flush=True)

    # 기준: 현행 A0 12행 파이프라인
    S12 = np.stack([C._shortlist(lg[i], ad, tau) for i in range(n)])
    X = feats(a0c5[S12], np.take_along_axis(lg, S12, 1), bf_a[S12], rp_a[S12],
              rmean, goal, cmd)
    with torch.no_grad():
        pick = sel(torch.from_numpy(X).to(dev)).cpu().numpy().argmax(1)
    base = wm(np.take_along_axis(np.take_along_axis(D_a0, S12, 1),
                                 pick[:, None], 1)[:, 0])
    base_orc = wm(np.take_along_axis(D_a0, S12, 1).min(1))
    print(f"  [기준] A0 12행: shortlist oracle {base_orc:.4f}  realized {base:.4f}\n")

    # child logit 을 세 가지로 준다:
    #   inherit = 부모 값 상속 (child 간 시각 구분 0, 하한)
    #   oracle  = 참 D3 로 매긴 순위 (완벽한 fine-scorer, 상한)
    #   random  = 무작위 (귀무가설)
    print(f"{'parent P':>9} {'cap':>5} {'유효N':>7} {'slOracle':>10} "
          f"{'inherit':>9} {'oracle':>9} {'random':>9}")
    res = {"base": base, "base_oracle": base_orc, "K": int(len(kids))}
    rng = np.random.default_rng(0)
    for P, cap in ((12, 2), (12, 4), (8, 4), (12, 8)):
        R = {k: np.empty(n) for k in ("inherit", "oracle", "random")}
        O = np.empty(n); NN = []
        for i in range(n):
            par = S12[i][:P]
            idx = np.where(np.isin(kpar, par))[0]
            if cap:
                keep = []
                for p in par:
                    c = idx[kpar[idx] == p]
                    keep.append(c[np.argsort(-hb["support"][c])[:cap]])
                idx = np.concatenate(keep)
            NN.append(len(idx))
            O[i] = D_k[i, idx].min()
            base_lg = np.take_along_axis(lg[i], kpar[idx], 0)
            src = {"inherit": base_lg,
                   "oracle": -D_k[i, idx].astype(np.float64),
                   "random": rng.normal(size=len(idx))}
            for k, v in src.items():
                Xi = feats(kids[idx][None], v[None], bf_k[idx][None], rp_k[idx][None],
                           rmean, goal[i:i + 1], cmd[i:i + 1])
                with torch.no_grad():
                    s = sel(torch.from_numpy(Xi).to(dev)).cpu().numpy()[0]
                R[k][i] = D_k[i, idx[int(s.argmax())]]
        res[f"P{P}_cap{cap}"] = {"shortlist_oracle": wm(O), "N": float(np.mean(NN)),
                                 **{f"realized_{k}": wm(v) for k, v in R.items()}}
        print(f"{P:>9} {cap:>5} {np.mean(NN):>7.0f} {wm(O):>10.4f} "
              f"{wm(R['inherit']):>9.4f} {wm(R['oracle']):>9.4f} {wm(R['random']):>9.4f}",
              flush=True)
    json.dump(res, open(os.path.join(A, "logs/e22_bank_realized.json"), "w"),
              indent=1, default=float)
    print("\n읽는 법: oracle 열은 **완벽한 child fine-scorer** 를 가정한 상한이다.")
    print("        그 값이 기준(A0 12행)보다 낮지 않으면 fine-scorer 를 만들 이유가 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
