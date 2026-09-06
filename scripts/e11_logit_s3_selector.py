#!/usr/bin/env python
"""⑤-J: logit 유래 S3 추정(MAE 1.29m)을 selector 에 넣으면 이득이 있는가.

⑤-E3 은 전용 head(2.45m)로 실패했다. ⑤-I 에서 full-K logit 분포가 1.29m 로
1.9배 정확함이 확인됐다. 요구 정밀도 ~0.74m 이므로 여전히 미달이지만 직접 잰다.
mu 는 tune 에서 고정하고 val38 은 1회만 본다.
"""
import argparse, json, os, sys
import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402


def steps(xy):
    z = np.zeros_like(xy[..., :1, :])
    return np.linalg.norm(np.diff(np.concatenate([z, xy], -2), axis=-2), axis=-1)


def run(m, arr, bank, dev, split, nh):
    if split == "val":
        r = arr["val_idx"]
        sub = (arr["frame"][r] >= 30) & C.history_available(arr, r, nh)
        r, w = r[sub], arr["val_weight"][sub].astype(np.float64)
    else:
        s = C.make_split(arr)
        r = s["tune_rows"]
        sub = C.history_available(arr, r, nh)
        r, w = r[sub], None
    lg = C.run_logits(m, arr, r, torch.from_numpy(C.build_global_lidar2img()),
                      dev, batch=24, amp=True, n_hist=nh).astype(np.float64)
    T = C.precompute_targets(arr, r, bank, weight=w)
    return r, lg, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_sp_ctrl/best.pth")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(A, "logs/e11_logit_s3_sel.json"))
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(dev)
    arr = C.load_arrays(); bank = np.load(C.BANK_A0, allow_pickle=False)
    c5 = bank["candidate_xy_abs_5s"].astype(np.float64)
    cS3 = steps(c5[:, :6]).sum(-1)
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu"); a = ck["args"]
    nh = int(a.get("n_hist", 3))
    m = TrainableSparseScoreDrive(C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)),
                                  n_hist=nh, speed_head=bool(a.get("w_speed", 0) > 0),
                                  speed_gamma=0.0).to(dev)
    m.load_state_dict(ck["model"]); m.fuse_mul = bool(a.get("fuse_mul", 0)); m.eval()

    LAMS = (0.0, 0.1, 0.2)
    MUS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)
    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)          # noqa: E731
    store = {}
    for split in ("tune", "val"):
        rows, lg, T = run(m, arr, bank, dev, split, nh)
        w = T["weight"]
        e = np.exp((lg - lg.max(1, keepdims=True)) / 0.5)
        s3est = (e @ cS3) / e.sum(1)                                   # ⑤-I 최적
        gS3 = steps(arr["fut"][rows].astype(np.float64)).sum(-1)
        mae = float(np.average(np.abs(s3est - gS3), weights=w))
        grid = {}
        R = {}
        orc = np.empty(len(rows))
        for n in range(len(rows)):
            S = C._shortlist(lg[n], T["anchor_dist"], T["nms_tau"])
            orc[n] = T["D3gt"][n][S].min()
            gcn = nz(np.linalg.norm(T["cand_end5"][S] - T["goal_xy"][n], axis=1))
            vcn = nz(lg[n].max() - lg[n][S])
            sd = np.abs(cS3[S] - s3est[n])
            sdn = nz(sd) if sd.max() - sd.min() > 1e-9 else np.zeros_like(sd)
            for lam in LAMS:
                for mu in MUS:
                    R.setdefault((lam, mu), np.empty(len(rows)))[n] = \
                        T["D3gt"][n][S[int(np.argmin(gcn + lam * vcn + mu * sdn))]]
        for k, v in R.items():
            grid[k] = float(np.average(v, weights=w))
        store[split] = dict(grid=grid, oracle=float(np.average(orc, weights=w)),
                            mae=mae, n=len(rows))
        print(f"[{split}] n={len(rows)}  S3추정 MAE {mae:.3f}m  oracle {store[split]['oracle']:.4f}",
              flush=True)
        for lam in LAMS:
            print("   lam%.1f: " % lam + "  ".join(
                f"mu{mu}={grid[(lam,mu)]:.4f}" for mu in MUS), flush=True)

    tg = store["tune"]["grid"]
    lam0, mu0 = min(tg, key=lambda k: tg[k])
    vg = store["val"]["grid"]
    base = vg[(lam0, 0.0)]
    got = vg[(lam0, mu0)]
    print(f"\n=== val38 (tune 고정 lam={lam0} mu={mu0}) ===")
    print(f"  shortlist oracle {store['val']['oracle']:.4f}")
    print(f"  mu=0 기준        {base:.4f}")
    print(f"  logit-S3 적용    {got:.4f}   이득 {base-got:+.4f}")
    best_val = min(vg.values())
    print(f"  (참고) val 격자 최소 {best_val:.4f} @ {min(vg, key=lambda k: vg[k])}")
    json.dump({k: {"grid": {f"{a_}|{b_}": v for (a_, b_), v in d["grid"].items()},
                   "oracle": d["oracle"], "mae": d["mae"], "n": d["n"]}
               for k, d in store.items()} | {"fixed": [lam0, mu0]},
              open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
