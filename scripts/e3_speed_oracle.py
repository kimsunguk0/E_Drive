#!/usr/bin/env python
"""⑤-E3 진단: 속도(진행량)를 알면 realized 가 얼마나 내려가는가 — 오라클 상한.

주의: GT 진행량을 selector 에 넣는 **진단용 오라클**이다. 제출 모델이 아니다.
실제 구현은 §4.5 대로 영상에서 진행량을 회귀해 모델 내부에서 logits 를 다듬고,
외부 selector 는 기존과 동일하게 둔다.

측정:
  A. 현재 selector            J = gcn + λ·vcn
  B. + GT 3초 진행량          J = gcn + μ·|S3(cand) - S3(GT)| (정규화)
  C. GT 진행량만              J = |S3 diff|
  D. shortlist oracle (상한)
"""
import argparse
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402

DUMP = os.path.join(A, "logs/dump_sparse_c")


def arclen(xy_abs):
    """[...,T,2] 절대좌표 -> 누적 이동거리(마지막 값)."""
    z = np.zeros_like(xy_abs[..., :1, :])
    inc = np.diff(np.concatenate([z, xy_abs], axis=-2), axis=-2)
    return np.linalg.norm(inc, axis=-1).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sc_e1_gt")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    z = np.load(os.path.join(DUMP, f"logits_{args.tag}_{args.split}.npz"),
                allow_pickle=True)
    lg = z["logits"].astype(np.float64)
    rows = z["rows"]
    w = z["weight"].astype(np.float64)
    T = C.precompute_targets(arr, rows, bank, weight=w)
    Dg, ce, gx = T["D3gt"], T["cand_end5"], T["goal_xy"]
    ad, tau = T["anchor_dist"], T["nms_tau"]

    cand5 = bank["candidate_xy_abs_5s"].astype(np.float64)      # [K,10,2]
    cS3 = arclen(cand5[:, :6])                                  # [K] 후보 3초 진행량
    gS3 = arclen(arr["fut"][rows].astype(np.float64))           # [N] GT 3초 진행량

    wm = lambda a: float(np.average(a, weights=w))              # noqa: E731
    N = len(lg)
    res = {k: np.empty(N) for k in
           ("cur", "gt_mu05", "gt_mu1", "gt_mu2", "gt_only", "orc")}
    for n in range(N):
        S = C._shortlist(lg[n], ad, tau)
        d = Dg[n][S]
        res["orc"][n] = d.min()
        gc = np.linalg.norm(ce[S] - gx[n], axis=1)
        vc = lg[n].max() - lg[n][S]
        nz = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-9)  # noqa: E731
        gcn, vcn = nz(gc), nz(vc)
        sc = np.abs(cS3[S] - gS3[n])                            # 진행량 불일치
        scn = nz(sc)
        res["cur"][n] = Dg[n][S[int(np.argmin(gcn + 0.2 * vcn))]]
        res["gt_mu05"][n] = Dg[n][S[int(np.argmin(gcn + 0.5 * scn))]]
        res["gt_mu1"][n] = Dg[n][S[int(np.argmin(gcn + 1.0 * scn))]]
        res["gt_mu2"][n] = Dg[n][S[int(np.argmin(gcn + 2.0 * scn))]]
        res["gt_only"][n] = Dg[n][S[int(np.argmin(scn))]]

    print(f"=== ⑤-E3 속도 오라클 ({args.tag}, {args.split}, n={N}) ===")
    print(f"  D. shortlist oracle (도달 상한)          {wm(res['orc']):.4f}")
    print(f"  A. 현재 selector (goal + 0.2*visual)     {wm(res['cur']):.4f}")
    print(f"  B. goal + 0.5*GT진행량                   {wm(res['gt_mu05']):.4f}")
    print(f"  B. goal + 1.0*GT진행량                   {wm(res['gt_mu1']):.4f}")
    print(f"  B. goal + 2.0*GT진행량                   {wm(res['gt_mu2']):.4f}")
    print(f"  C. GT진행량만                            {wm(res['gt_only']):.4f}")
    best = min(("gt_mu05", "gt_mu1", "gt_mu2", "gt_only"), key=lambda k: wm(res[k]))
    gain = wm(res["cur"]) - wm(res[best])
    left = wm(res[best]) - wm(res["orc"])
    print(f"\n  속도 오라클 최대 이득 : {gain:+.4f}  ({wm(res['cur']):.4f} -> {wm(res[best]):.4f})")
    print(f"  남는 regret           : {left:.4f}  (oracle {wm(res['orc']):.4f} 까지)")
    print(f"  regret 감소율         : {100*gain/(wm(res['cur'])-wm(res['orc'])):.1f}%")

    # 버킷별
    print("\n  버킷별 (현재 -> 속도 오라클):")
    for k, m in T["buckets"].items():
        m = np.asarray(m, bool)
        if m.sum() == 0:
            continue
        a1 = float(np.average(res["cur"][m], weights=w[m]))
        a2 = float(np.average(res[best][m], weights=w[m]))
        print(f"    {k:8s} n={m.sum():4d}  {a1:.4f} -> {a2:.4f}  ({a2-a1:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
