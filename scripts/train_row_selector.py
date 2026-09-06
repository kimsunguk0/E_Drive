#!/usr/bin/env python
"""⑤-E7-b: learned row selector.

후보 12행은 이미 완성돼 있고 selector 는 그 중 **행 index 하나**만 고른다.
따라서 후보 좌표는 bitwise 불변이고 goal counterfactual 도 자동 성립한다.
입력으로 goal/cmd/visual logit/후보 기하만 쓴다. ego status(speed/acc)는 쓰지 않는다.

학습 = train scene, 하이퍼/조기중단 = tune scene, val38 은 마지막 1회만 본다.
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

CW3 = C.CW3


def arclen_steps(xy):
    z = np.zeros_like(xy[..., :1, :])
    return np.linalg.norm(np.diff(np.concatenate([z, xy], -2), axis=-2), axis=-1)


def bank_features(bank):
    """후보별 정적 기하 특징 [K,F0] 과 5초 endpoint [K,2]."""
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)      # [K,10,2]
    st = arclen_steps(c5)
    S5 = st.sum(-1)
    S3 = st[:, :6].sum(-1)
    r = np.cumsum(st[:, :6], -1) / np.clip(S3, 1e-3, None)[:, None]   # [K,6]
    d = np.diff(np.concatenate([np.zeros((len(c5), 1, 2)), c5], 1), axis=1)
    hd = np.arctan2(d[..., 1], d[..., 0])                    # [K,10]
    dh = np.diff(np.unwrap(hd, axis=1), axis=1)              # [K,9] 곡률 대용
    f = np.column_stack([
        S3 / 30.0, S5 / 50.0,
        c5[:, 5, 0] / 30.0, c5[:, 5, 1] / 10.0,              # 3초 위치
        c5[:, 9, 0] / 50.0, c5[:, 9, 1] / 20.0,              # 5초 위치
        hd[:, 5], hd[:, 9],                                  # 3s/5s heading
        dh.sum(1), np.abs(dh).sum(1), np.abs(dh).max(1),     # 총/절대/최대 곡률
        r[:, :5],                                            # timing profile (r6==1)
    ])
    return f.astype(np.float32), c5[:, 9].astype(np.float32), r.astype(np.float32)


def kin_feats(arr, rows, c5, S, sig_v=0.0, sig_a=0.0, seed=0):
    """과거 자차 운동학 -> 후보별 근미래 위치 정합도.
    ⑤-M/N. ego status 이므로 규정 확인 전까지 실험용이며, sig_* 로 VO 오차를 흉내낸다."""
    now = C.HIS_NOW
    his = arr["his"][rows].astype(np.float64)
    v1 = np.linalg.norm(his[:, now] - his[:, now - 5], axis=1) / 0.5
    v0 = np.linalg.norm(his[:, now - 5] - his[:, now - 10], axis=1) / 0.5
    a = (v1 - v0) / 0.5
    if sig_v or sig_a:
        r = np.random.default_rng(seed)
        v1 = v1 + r.normal(0, sig_v, len(v1))
        a = a + r.normal(0, sig_a, len(a))
    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)   # noqa: E731
    out = []
    for t, ti in ((1.0, 1), (2.0, 3), (3.0, 5)):
        p = np.maximum(v1, 0) * t + 0.5 * a * t * t
        d = np.abs(c5[:, ti, 0][S] - p[:, None])
        out += [d[..., None] / 5.0, nz(d)[..., None]]
    return np.concatenate(out, -1).astype(np.float32)


def build(npz, bf, end5, rprof, rmean):
    z = np.load(npz)
    S = z["shortlist"].astype(np.int64)                      # [N,12]
    N, M = S.shape
    lg = z["logit"].astype(np.float64)
    g = z["goal"].astype(np.float64)
    e5 = end5[S]                                             # [N,12,2]
    gd = np.linalg.norm(e5 - g[:, None], axis=-1)            # [N,12]
    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)   # noqa: E731
    rank = lambda x: np.argsort(np.argsort(x, 1), 1) / (M - 1.0)    # noqa: E731
    top = np.argmax(lg, 1)
    div = np.linalg.norm(e5 - np.take_along_axis(
        e5, top[:, None, None], 1), axis=-1)                 # top-1 과의 거리
    pdev = np.abs(rprof[S] - rmean[None, None]).mean(-1)     # 고정 prior
    cmd = z["cmd"].astype(np.int64)
    oh = np.zeros((N, M, 3), np.float64)
    oh[np.arange(N), :, np.clip(cmd, 0, 2)] = 1.0
    X = np.concatenate([
        bf[S],                                               # 정적 기하 16
        gd[..., None] / 10.0, nz(gd)[..., None], rank(gd)[..., None],
        (e5[..., 0] - g[:, None, 0])[..., None] / 10.0,
        (e5[..., 1] - g[:, None, 1])[..., None] / 10.0,
        nz(lg)[..., None], rank(lg)[..., None],
        (lg - lg.mean(1, keepdims=True))[..., None],
        div[..., None] / 10.0, nz(div)[..., None],
        pdev[..., None] * 10.0, nz(pdev)[..., None],
        oh,
    ], -1).astype(np.float32)
    return dict(X=X, D3=z["D3"].astype(np.float32), w=z["weight"].astype(np.float64),
                gd=gd, lg=lg, scen=z["scen"], S=S, rows=z["rows"])


class Selector(nn.Module):
    def __init__(self, f, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, h), nn.GELU(), nn.LayerNorm(h),
                                 nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
                                 nn.Linear(h, 1))

    def forward(self, x):                      # [B,12,F] -> [B,12]
        return self.net(x).squeeze(-1)


def realized(score, D3, w):
    idx = score.argmax(1)
    return float(np.average(np.take_along_axis(D3, idx[:, None], 1)[:, 0], weights=w))


def rule_selector(d, lam):
    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)   # noqa: E731
    return -(nz(d["gd"]) + lam * nz(d["lg"].max(1, keepdims=True) - d["lg"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--w-exp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kin", type=int, default=0,
                    help="1이면 자차 과거 운동학 특징 추가 (ego status, 규정 확인 필요)")
    ap.add_argument("--kin-sig-v", type=float, default=0.0)
    ap.add_argument("--kin-sig-a", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(A, "logs/e7_selector.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(f"cuda:{args.gpu}")
    bank = np.load(C.BANK_A0, allow_pickle=False)
    bf, end5, rprof = bank_features(bank)
    rmean = rprof.mean(0)

    D = {s: build(os.path.join(A, f"data/etri/selector_{s}.npz"), bf, end5, rprof, rmean)
         for s in ("train", "tune", "val")}
    if args.kin:
        arr = C.load_arrays()
        c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
        for sp in D:
            K = kin_feats(arr, D[sp]["rows"], c5, D[sp]["S"],
                          args.kin_sig_v, args.kin_sig_a, args.seed)
            D[sp]["X"] = np.concatenate([D[sp]["X"], K], -1)
        print(f"운동학 특징 +{K.shape[-1]}  (sig_v={args.kin_sig_v} sig_a={args.kin_sig_a})",
              flush=True)
    F0 = D["train"]["X"].shape[-1]
    print(f"feature dim {F0}   train {len(D['train']['X'])} / tune {len(D['tune']['X'])}"
          f" / val {len(D['val']['X'])}", flush=True)

    print("\n=== 기준선 (규칙 selector) ===")
    base = {}
    for sp in ("tune", "val"):
        d = D[sp]
        r = {lam: realized(rule_selector(d, lam), d["D3"], d["w"]) for lam in (0.0, 0.1, 0.2, 0.3)}
        base[sp] = r
        orc = float(np.average(d["D3"].min(1), weights=d["w"]))
        print(f"  {sp:5s} oracle {orc:.4f}   " +
              "  ".join(f"lam{k}={v:.4f}" for k, v in r.items()), flush=True)
    lam_fix = min(base["tune"], key=lambda k: base["tune"][k])
    print(f"  -> tune 고정 lambda={lam_fix} (val {base['val'][lam_fix]:.4f})", flush=True)

    m = Selector(F0).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
    Xtr = torch.from_numpy(D["train"]["X"])
    Dtr = torch.from_numpy(D["train"]["D3"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(Xtr) // args.batch))
    best = (1e9, None)
    print("\n=== 학습 ===", flush=True)
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
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
            tot += float(loss) * len(b)
        m.eval()
        with torch.no_grad():
            sc = m(torch.from_numpy(D["tune"]["X"]).to(dev)).cpu().numpy()
        rt = realized(sc, D["tune"]["D3"], D["tune"]["w"])
        if rt < best[0]:
            best = (rt, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
            mark = " *"
        else:
            mark = ""
        if ep % 4 == 0 or mark:
            print(f"  ep{ep:3d} loss={tot/len(perm):.4f} tune_realized={rt:.4f}{mark}", flush=True)

    m.load_state_dict(best[1])
    m.eval()
    res = {"baseline": base, "lam_fixed": lam_fix, "tune_best": best[0]}
    print("\n=== val38 (tune 최적 ckpt 1회 적용) ===")
    with torch.no_grad():
        sv = m(torch.from_numpy(D["val"]["X"]).to(dev)).cpu().numpy()
    d = D["val"]
    rv = realized(sv, d["D3"], d["w"])
    orc = float(np.average(d["D3"].min(1), weights=d["w"]))
    res["val_learned"] = rv
    res["val_oracle"] = orc
    res["val_rule"] = base["val"][lam_fix]
    print(f"  shortlist oracle      {orc:.4f}")
    print(f"  규칙 selector(lam{lam_fix})  {base['val'][lam_fix]:.4f}")
    print(f"  learned selector      {rv:.4f}   이득 {base['val'][lam_fix]-rv:+.4f}")
    print(f"  regret  {base['val'][lam_fix]-orc:.4f} -> {rv-orc:.4f}")

    print("\n=== 음성대조 ===")
    rng = np.random.default_rng(0)
    Xs = d["X"].copy()
    vis = [F0 - 8, F0 - 7, F0 - 6]          # nz(lg), rank(lg), lg-mean
    Xs[:, :, vis] = Xs[rng.permutation(len(Xs))][:, :, vis]
    with torch.no_grad():
        rs = realized(m(torch.from_numpy(Xs).to(dev)).cpu().numpy(), d["D3"], d["w"])
    Xz = d["X"].copy(); Xz[:, :, vis] = 0.0
    with torch.no_grad():
        rz = realized(m(torch.from_numpy(Xz).to(dev)).cpu().numpy(), d["D3"], d["w"])
    res["val_visual_shuffle"] = rs
    res["val_visual_zero"] = rz
    print(f"  visual 셔플 {rs:.4f}   visual 0 {rz:.4f}   (정상 {rv:.4f})")
    sel = sv.argmax(1)
    res["match_oracle"] = float(np.average(
        (sel == d["D3"].argmin(1)).astype(float), weights=d["w"]))
    print(f"  선택==oracle {100*res['match_oracle']:.1f}%")
    torch.save({"model": best[1], "args": vars(args), "feat_dim": F0},
               os.path.join(A, "work_dirs/row_selector.pth"))
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
