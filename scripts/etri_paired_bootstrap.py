#!/usr/bin/env python
"""두 예측 파일(A/B)의 시나리오 단위 페어드 부트스트랩.

`etri_vad_eval.py`가 저장한 `*_pred.npy` 두 개를 받아 같은 앵커에서 비교한다.
앵커 순서는 두 파일이 동일하다(둘 다 val split `sp['val_idx']` 순서 = ValSet.vi).

  usage:
    etri_paired_bootstrap.py A.npy B.npy --name-a goal --name-b no-goal

38개 시나리오뿐이므로 앵커 단위 부트스트랩은 CI를 과소평가한다. 시나리오를
복원추출하고, 뽑힌 시나리오의 앵커 전체를 한 덩어리로 재계산한다(=클러스터
부트스트랩). 가중 L2는 시나리오별 평균이 아니라 재표집된 앵커 풀 전체에서
다시 계산해야 test 정합 가중치 W가 올바르게 반영된다.
"""
import argparse
import os
import sys

import numpy as np

ADCL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
sys.path.insert(0, os.path.join(ADCL, "scripts"))
sys.path.insert(0, os.path.join(ADCL, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_a")
    ap.add_argument("pred_b")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-frame", type=int, default=30,
                    help="배포충실 조건. 0이면 전체 앵커.")
    args = ap.parse_args()

    from etri_table import ValSet, wl2, STOP, SPEED_LO, SPEED_HI, SPEED_HI2

    v = ValSet()
    gt, W = v.gt, v.w
    ci = v.vi
    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])
    scen = scen_all[d["scen_idx"][ci]]
    frame = d["frame"][ci].astype(int)
    speed = d["speed"][ci].astype(np.float64)
    vcmd = d["vad_cmd"][ci].astype(int)

    pa = np.load(args.pred_a).astype(np.float64)
    pb = np.load(args.pred_b).astype(np.float64)
    assert pa.shape == pb.shape == gt.shape, (pa.shape, pb.shape, gt.shape)

    keep = np.ones(len(ci), bool)
    if args.min_frame > 0:
        keep &= frame >= args.min_frame
    # NaN 예측은 양쪽 모두에서 제외해야 페어드가 성립한다.
    finite = np.isfinite(pa).all((1, 2)) & np.isfinite(pb).all((1, 2))
    if (~finite & keep).any():
        print(f"경고: 비유한 예측 {(~finite & keep).sum()}개 제외")
    keep &= finite

    cols = {
        "가중": (None, True),
        "무가중": (np.ones(len(ci), bool), False),
        "right": (vcmd == 0, False),
        "이동": (speed >= STOP, False),
        "3–8m/s": ((speed >= SPEED_LO) & (speed < SPEED_HI), False),
        "15+m/s": (speed >= SPEED_HI2, False),
    }

    scens = np.unique(scen[keep])
    rows_by_scen = {s: np.where(keep & (scen == s))[0] for s in scens}
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(scens), size=(args.n_boot, len(scens)))

    print(f"앵커 {int(keep.sum())}   시나리오 {len(scens)}   "
          f"부트스트랩 {args.n_boot}회 (시나리오 복원추출)")
    print(f"조건: frame >= {args.min_frame}\n")
    hdr = (f"{'열':<10} {args.name_a:>9} {args.name_b:>9} {'Δ(B−A)':>9} "
           f"{'상대':>8}  {'시나리오 95% CI':>22}  {'A우세':>7}  판정")
    print(hdr)
    print("-" * len(hdr))

    for cname, (cm, weighted) in cols.items():
        m = keep if cm is None else (keep & cm)
        if m.sum() == 0:
            continue
        w = W if weighted else None
        la = wl2(pa[m], gt[m], w[m] if w is not None else None)
        lb = wl2(pb[m], gt[m], w[m] if w is not None else None)

        # 시나리오별 행 인덱스를 미리 마스크와 교차시켜 둔다.
        per = {s: r[m[r]] for s, r in rows_by_scen.items()}
        order = list(scens)
        deltas = np.empty(args.n_boot)
        for b in range(args.n_boot):
            sel = np.concatenate([per[order[j]] for j in draws[b]])
            if len(sel) == 0:
                deltas[b] = np.nan
                continue
            ww = W[sel] if weighted else None
            deltas[b] = wl2(pb[sel], gt[sel], ww) - wl2(pa[sel], gt[sel], ww)
        deltas = deltas[np.isfinite(deltas)]
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        # 시나리오 단위 일치율: 각 시나리오에서 A가 더 좋은가.
        # 분모는 해당 마스크에 앵커가 있는 시나리오만이다. 38로 나누면
        # 앵커가 없는 시나리오가 전부 '패배'로 집계돼 일치율이 과소평가된다.
        wins = n_scen = 0
        for s in order:
            r = per[s]
            if len(r) == 0:
                continue
            n_scen += 1
            ww = W[r] if weighted else None
            wins += wl2(pa[r], gt[r], ww) < wl2(pb[r], gt[r], ww)
        frac = wins / max(n_scen, 1)

        verdict = "A 우세" if lo > 0 else ("B 우세" if hi < 0 else "구분 못 함")
        print(f"{cname:<10} {la:9.4f} {lb:9.4f} {lb - la:+9.4f} "
              f"{(lb - la) / lb * 100:+7.1f}%  [{lo:+7.4f}, {hi:+7.4f}]  "
              f"{frac:5.1%}/{n_scen:<2d}  {verdict}")

    print("\nΔ = B − A 이므로 양수면 A(=%s)가 좋다. CI 하한 > 0 이어야 유의." % args.name_a)
    print("'A우세'는 38개 시나리오 각각에서 A가 이긴 비율이다.")


if __name__ == "__main__":
    main()
