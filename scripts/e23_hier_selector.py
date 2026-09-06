#!/usr/bin/env python
"""⑥-F: 확장 후보 집합으로 selector 를 **재학습**한다.

⑥-E 는 A0 12행으로 학습된 selector 를 24~83행에 적용해 분포 이탈이었다.
여기서는 계층 bank 의 확장 후보로 학습 데이터를 다시 만들고 selector 를 재학습해
"bank 확장이 정말 무용한가"를 공정하게 검정한다.

파이프라인은 ⑥-E 와 동일: 모델이 1024 parent 점수화 -> 상위 P parent ->
각 parent 의 지지도 상위 cap child 로 전개 -> selector 가 행 하나 선택.
child logit 은 부모 상속(하한) 과 GT(상한) 두 가지로 각각 학습/평가한다.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402
from train_row_selector import Selector  # noqa: E402
from e22_bank_realized import d3_pair, feats, static_feats  # noqa: E402


SIGMAS = (0.0, 0.05, 0.10, 0.20, 0.40)


def _spearman(a, b):
    """행 단위 순위상관 평균."""
    ra = np.argsort(np.argsort(a, 1), 1).astype(np.float64)
    rb = np.argsort(np.argsort(b, 1), 1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)) + 1e-9
    return float((num / den).mean())


def build_split(m, arr, split, nh, dev, kids, kpar, sup, bf_k, rp_k, rmean,
                a0, P, cap, stride=1):
    if split == "val":
        r = arr["val_idx"]; sb = (arr["frame"][r] >= 30) & C.history_available(arr, r, nh)
        r = r[sb]; w = arr["val_weight"][sb].astype(np.float64)
    else:
        s = C.make_split(arr, all_frames=True)
        if split == "tune":
            r = s["tune_rows"]
        else:
            scen = arr["scen_idx"]
            ts = np.unique(scen[s["train_rows"]])
            r = np.where(np.isin(scen, ts) & (arr["frame"] >= 30))[0]
        r = r[C.history_available(arr, r, nh)]
        w = np.ones(len(r))
    r, w = r[::stride], w[::stride]
    lg = C.run_logits(m, arr, r, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=24, amp=True, n_hist=nh).astype(np.float64)
    V3 = arr["fut"][r].astype(np.float64)
    goal = arr["fut5"][r][:, 9].astype(np.float64)
    cmd = arr["vad_cmd"][r].astype(np.int64)
    ad, tau = a0["anchor_dist"].astype(np.float64), float(a0["nms_tau"])
    D_k = d3_pair(V3, kids[:, :6])
    n = len(r)
    IDX = np.empty((n, P * cap), np.int64)
    for i in range(n):
        par = C._shortlist(lg[i], ad, tau)[:P]
        keep = []
        for p in par:
            c = np.where(kpar == p)[0]
            c = c[np.argsort(-sup[c])[:cap]]
            if len(c) < cap:                       # child 부족하면 반복해 채운다
                c = np.resize(c, cap) if len(c) else np.zeros(cap, np.int64)
            keep.append(c)
        IDX[i] = np.concatenate(keep)[:P * cap]
    D = np.take_along_axis(D_k, IDX, 1).astype(np.float32)
    plg = np.take_along_axis(lg, kpar[IDX], 1)
    # child fine-scorer 를 흉내낸다. sigma=0 은 정답 누설이므로 상한이 아니다.
    # sigma 를 키우면 순위상관이 떨어지고, 그 지점의 realized 가 실제 요구 사양이 된다.
    rng = np.random.default_rng(0)
    out, corr = {}, {}
    D64 = D.astype(np.float64)
    out["inherit"] = feats(kids[IDX], plg, bf_k[IDX], rp_k[IDX], rmean, goal, cmd)
    corr["inherit"] = _spearman(plg, -D64)
    for sig in SIGMAS:
        L = -(D64 + rng.normal(0, sig, D64.shape))
        out[f"fine{sig}"] = feats(kids[IDX], L, bf_k[IDX], rp_k[IDX], rmean, goal, cmd)
        corr[f"fine{sig}"] = _spearman(L, -D64)
    return dict(X=out, D3=D, w=w, IDX=IDX, D_k=D_k, corr=corr)


def train_sel(Xtr, Dtr, Xtu, Dtu, wtu, dev, epochs=25, bs=512, lr=1e-3, tau=0.05, seed=0):
    torch.manual_seed(seed)
    m = Selector(Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    Xt = torch.from_numpy(Xtr); Dt = torch.from_numpy(Dtr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(Xt) // bs))
    best = (1e9, None)
    for ep in range(epochs):
        m.train()
        pm = torch.randperm(len(Xt))
        for i in range(0, len(pm), bs):
            b = pm[i:i + bs]
            x, dd = Xt[b].to(dev), Dt[b].to(dev)
            s = m(x)
            t = Fn.softmax(-dd / tau, 1); lp = Fn.log_softmax(s, 1)
            loss = -(t * lp).sum(1).mean() + (lp.exp() * dd).sum(1).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            sc = m(torch.from_numpy(Xtu).to(dev)).cpu().numpy()
        rv = float(np.average(np.take_along_axis(Dtu, sc.argmax(1)[:, None], 1)[:, 0],
                              weights=wtu))
        if rv < best[0]:
            best = (rv, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1]); m.eval()
    return m, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/etri/bank_hier_K7680.npz")
    ap.add_argument("--ckpt", default="work_dirs/sc_u_ctrl/best.pth")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--stride", type=int, default=3, help="train 행 서브샘플")
    ap.add_argument("--configs", default="12x2,12x4,8x4")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(dev)
    arr = C.load_arrays()
    a0 = np.load(C.BANK_A0, allow_pickle=False)
    hb = np.load(os.path.join(A, args.bank))
    kids = hb["candidate_xy_abs_5s"].astype(np.float64)
    kpar = hb["parent_id"]; sup = hb["support"]
    bf_k, rp_k = static_feats(kids)
    _, rp_a = static_feats(a0["candidate_xy_abs_5s"].astype(np.float64))
    rmean = rp_a.mean(0)
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    ar = ck["args"]; nh = int(ar.get("n_hist", 3))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(ar.get("logit_norm", 1)), n_hist=nh,
        score_weight=ar.get("score_weight", "uniform")).to(dev)
    m.load_state_dict(ck["model"]); m.fuse_mul = bool(ar.get("fuse_mul", 0)); m.eval()

    print(f"bank={os.path.basename(args.bank)}  ckpt={os.path.basename(os.path.dirname(args.ckpt))}"
          f"  train stride={args.stride}\n", flush=True)
    hdr = "  ".join(f"fine{s}" for s in SIGMAS)
    print(f"{'구성':>8} {'N':>4} {'slOracle':>9} {'inherit':>9}  {hdr}")
    res = {}
    for cfg in args.configs.split(","):
        P, cap = (int(x) for x in cfg.split("x"))
        d = {sp: build_split(m, arr, sp, nh, dev, kids, kpar, sup, bf_k, rp_k,
                             rmean, a0, P, cap, args.stride if sp == "train" else 1)
             for sp in ("train", "tune", "val")}
        orc = float(np.average(d["val"]["D3"].min(1), weights=d["val"]["w"]))
        row = {}
        for src in ["inherit"] + [f"fine{x}" for x in SIGMAS]:
            sel, tu = train_sel(d["train"]["X"][src], d["train"]["D3"],
                                d["tune"]["X"][src], d["tune"]["D3"], d["tune"]["w"], dev)
            with torch.no_grad():
                sc = sel(torch.from_numpy(d["val"]["X"][src]).to(dev)).cpu().numpy()
            row[src] = float(np.average(
                np.take_along_axis(d["val"]["D3"], sc.argmax(1)[:, None], 1)[:, 0],
                weights=d["val"]["w"]))
        res[cfg] = dict(N=P * cap, shortlist_oracle=orc, corr=d["val"]["corr"], **row)
        cells = "  ".join(f"{row[f'fine{x}']:>6.4f}" for x in SIGMAS)
        print(f"{cfg:>8} {P*cap:>4} {orc:>9.4f} {row['inherit']:>9.4f}  {cells}", flush=True)
        cc = "  ".join(f"{d['val']['corr'][f'fine{x}']:>6.3f}" for x in SIGMAS)
        print(f"{'  (순위상관)':>8} {'':>4} {'':>9} {d['val']['corr']['inherit']:>9.3f}  {cc}",
              flush=True)
    json.dump(res, open(os.path.join(A, "logs/e23_hier_selector.json"), "w"),
              indent=1, default=float)
    print("\n기준(A0 12행 + 전용 selector) val38 realized = 0.2522")
    return 0


if __name__ == "__main__":
    sys.exit(main())
