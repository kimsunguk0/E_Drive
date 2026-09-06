#!/usr/bin/env python
"""⑤-O: learned row selector 의 scenario 5-fold CV.

감사 지적: mini8/tune30/val38 이 모두 반복 사용돼 독립 검증셋이 없다.
train330(= train 300 + tune 30) 을 시나리오 단위 5-fold 로 나눠
fold 별로 학습/평가하고 mean±std 를 낸다. val38 은 건드리지 않는다.

이 수치가 val38 단발값보다 정직한 일반화 추정이다.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_row_selector import (Selector, bank_features, build, kin_feats,  # noqa: E402
                                realized, rule_selector)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--w-exp", type=float, default=1.0)
    ap.add_argument("--kin", type=int, default=0)
    ap.add_argument("--kin-sig-v", type=float, default=0.0)
    ap.add_argument("--kin-sig-a", type=float, default=0.0)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(A, "logs/e17_selector_cv.json"))
    ap.add_argument("--prefix", default="selector")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    bank = np.load(C.BANK_A0, allow_pickle=False)
    bf, end5, rprof = bank_features(bank)
    rmean = rprof.mean(0)

    parts = [build(os.path.join(A, f"data/etri/{args.prefix}_{s}.npz"), bf, end5, rprof, rmean)
             for s in ("train", "tune")]
    if args.kin:
        arr = C.load_arrays()
        c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
        for p in parts:
            K = kin_feats(arr, p["rows"], c5, p["S"], args.kin_sig_v, args.kin_sig_a, 0)
            p["X"] = np.concatenate([p["X"], K], -1)
    D = {k: np.concatenate([p[k] for p in parts], 0)
         for k in ("X", "D3", "w", "gd", "lg", "scen")}
    scen = D["scen"]
    uniq = np.unique(scen)
    print(f"시나리오 {len(uniq)}개, 행 {len(D['X'])}, feature {D['X'].shape[-1]}"
          f"  kin={args.kin} sig_v={args.kin_sig_v}", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(uniq)
    folds = np.array_split(order, args.folds)
    res = {"fold": [], "rule": [], "learned": [], "oracle": []}
    for fi, held in enumerate(folds):
        te = np.isin(scen, held)
        tr = ~te
        d_te = {k: D[k][te] for k in ("X", "D3", "w", "gd", "lg")}
        rule = min((realized(rule_selector(d_te, lam), d_te["D3"], d_te["w"])
                    for lam in (0.0, 0.1, 0.2)))
        orc = float(np.average(d_te["D3"].min(1), weights=d_te["w"]))
        torch.manual_seed(fi)
        m = Selector(D["X"].shape[-1]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
        Xtr = torch.from_numpy(D["X"][tr]); Dtr = torch.from_numpy(D["D3"][tr])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs * max(1, len(Xtr) // args.batch))
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), args.batch):
                b = perm[i:i + args.batch]
                x, dd = Xtr[b].to(dev), Dtr[b].to(dev)
                s = m(x)
                tgt = F.softmax(-dd / args.tau, dim=1)
                p = F.log_softmax(s, dim=1)
                loss = -(tgt * p).sum(1).mean() + args.w_exp * (p.exp() * dd).sum(1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            sc = m(torch.from_numpy(d_te["X"]).to(dev)).cpu().numpy()
        lr_ = realized(sc, d_te["D3"], d_te["w"])
        res["fold"].append(int(fi)); res["rule"].append(rule)
        res["learned"].append(lr_); res["oracle"].append(orc)
        print(f"  fold{fi}: 시나리오{len(held):3d} 행{te.sum():6d}  oracle {orc:.4f}  "
              f"규칙 {rule:.4f}  learned {lr_:.4f}  이득 {rule-lr_:+.4f}", flush=True)
    r = np.array(res["rule"]); l = np.array(res["learned"]); o = np.array(res["oracle"])
    print(f"\n=== 5-fold CV (시나리오 단위, val38 미사용) ===")
    print(f"  oracle   {o.mean():.4f} ± {o.std():.4f}")
    print(f"  규칙     {r.mean():.4f} ± {r.std():.4f}")
    print(f"  learned  {l.mean():.4f} ± {l.std():.4f}")
    print(f"  이득     {(r-l).mean():+.4f} ± {(r-l).std():.4f}   "
          f"(전 fold 개선: {bool((r>l).all())})")
    res["summary"] = dict(oracle=o.mean(), rule=r.mean(), learned=l.mean(),
                          gain=(r - l).mean(), gain_std=(r - l).std(),
                          all_improved=bool((r > l).all()))
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
