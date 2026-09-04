#!/usr/bin/env python
"""부록6 항목1 — 15+ m/s 구간 진단. f235가 개별 사례인가 구간 현상인가.

판정 기준
  * 구간 현상 = 15+ m/s 부분집합 전체에서 승자가 goal-등가속보다 나쁘다
  * 개별 사례 = 부분집합 평균은 승자가 낫고 f235만 튄다

구간 현상이면 잔차 크기를 진단한다. 변형 B는 `pred = 등가속 prior + 잔차`이므로
속도별 |잔차| 분포를 보면 "고속에서 잔차가 과하게 움직이는가"를 직접 알 수 있다.
과하면 잔차에 L2 정규화 λ를 걸어 prior에 붙여둔다.

    python scripts/etri_phase1_speed.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import waypoint_weights            # noqa: E402
from etri_phase1_extras import L2, goal_accel_np, load    # noqa: E402
from etri_table import COLUMNS, ValSet, fmt               # noqa: E402

CKPT = "/tmp/pm97/ckpt/etri_priornet"
OUT = "/home/pm97/workspace/sukim/adcl/logs/phase1_speed.json"
BINS = [0, 0.5, 3, 8, 15, np.inf]
W = waypoint_weights()


def lab(i):
    hi = "+" if BINS[i + 1] == np.inf else f"–{BINS[i+1]:g}"
    return f"{BINS[i]:g}{hi} m/s"


def main():
    d = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=True)
    sp = np.load("/tmp/pm97/data/etri/val_clips.npz", allow_pickle=True)
    v = ValSet()
    win = "transformer_B+mirror-glitch"
    pw, cfg = load(win, d, sp, v)
    ga = goal_accel_np(v.vel, v.goal)
    b = np.clip(np.digitize(v.speed, BINS) - 1, 0, len(BINS) - 2)

    print("=== 속도 bin별: 승자 vs goal-등가속 ===")
    print(f"{'구간':<12}{'n':>6}{'승자':>10}{'등가속':>10}{'차(승자-등가속)':>16}")
    rows = []
    for i in range(len(BINS) - 1):
        m = b == i
        if not m.sum():
            continue
        a, g = L2(pw[m], v.gt[m]), L2(ga[m], v.gt[m])
        rows.append({"bin": lab(i), "n": int(m.sum()),
                     "winner": round(a, 5), "goal_accel": round(g, 5),
                     "diff": round(a - g, 5)})
        print(f"{lab(i):<12}{m.sum():>6}{a:>10.4f}{g:>10.4f}{a-g:>16.4f}")
    fast = b == len(BINS) - 2
    verdict = "구간 현상" if rows[-1]["diff"] > 0 else "개별 사례"
    print(f"\n판정: 15+ m/s에서 차 {rows[-1]['diff']:+.4f} -> **{verdict}**")

    # f235 위치
    k = np.where((v.scen == "20260210-104708") & (v.frame == 235))[0]
    per_w = np.sqrt(((pw - v.gt) ** 2).sum(-1)) @ W
    per_g = np.sqrt(((ga - v.gt) ** 2).sum(-1)) @ W
    if len(k):
        k = int(k[0])
        pw_f, pg_f = per_w[fast], per_g[fast]
        pct = float((pw_f < per_w[k]).mean() * 100)
        print(f"  f235: 승자 {per_w[k]:.4f} / 등가속 {per_g[k]:.4f}  "
              f"-> 15+ 부분집합에서 상위 {100-pct:.1f}% (백분위 {pct:.1f})")
        print(f"  15+ 부분집합 승자 L2  p50 {np.percentile(pw_f,50):.4f}  "
              f"p90 {np.percentile(pw_f,90):.4f}  max {pw_f.max():.4f}")
        print(f"  15+ 부분집합에서 승자>등가속인 앵커: "
              f"{int((pw_f > pg_f).sum())} / {int(fast.sum())} "
              f"({(pw_f > pg_f).mean()*100:.1f}%)")

    # 잔차 크기 = 승자 - 등가속 prior (변형 B의 학습 잔차)
    print("\n=== 변형 B의 학습 잔차 |Δ| 분포 (3초 지점) ===")
    dres = np.linalg.norm(pw[:, -1] - ga[:, -1], axis=1)
    print(f"{'구간':<12}{'n':>6}{'|Δ| p50':>10}{'p90':>9}{'max':>9}{'GT변위 p50':>12}")
    resrows = []
    for i in range(len(BINS) - 1):
        m = b == i
        if not m.sum():
            continue
        gtd = np.linalg.norm(v.gt[m][:, -1], axis=1)
        resrows.append({"bin": lab(i), "n": int(m.sum()),
                        "res_p50": round(float(np.percentile(dres[m], 50)), 4),
                        "res_p90": round(float(np.percentile(dres[m], 90)), 4),
                        "res_max": round(float(dres[m].max()), 3),
                        "gt_disp_p50": round(float(np.percentile(gtd, 50)), 2)})
        r = resrows[-1]
        print(f"{lab(i):<12}{m.sum():>6}{r['res_p50']:>10.4f}{r['res_p90']:>9.4f}"
              f"{r['res_max']:>9.3f}{r['gt_disp_p50']:>12.2f}")
    json.dump({"speed_bins": rows, "residual": resrows, "verdict": verdict,
               "f235": {"winner": round(float(per_w[k]), 5),
                        "goal_accel": round(float(per_g[k]), 5)} if isinstance(k, int) else None},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\n저장 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
