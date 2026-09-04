#!/usr/bin/env python
"""0-7 — 챌린지 L2를 세 독립 구현으로 대조.

  (1) 우리 것        src/challenge_metrics.py  l2_challenge
  (2) 주최측 것      ETRI 레포의 metric_stp3.PlanningMetric.compute_L2 를 **직접 import**
                     해서 VAD.py:640-651 과 같은 방식으로 k=1,2,3 호출 후 평균
  (3) SparseDrive 것 planning_eval.py:160-168 의 절차를 그대로 재현
                     (샘플 평균 -> 누적평균 -> idx 1,3,5 평균)

세 개가 같아야 한다. (2)와 (3)은 평균 순서가 다르므로(샘플별 누적 후 평균 vs 샘플평균 후
누적) 수학적으로는 같지만 구현 경로가 달라 부호·인덱스 실수를 서로 잡아준다.

토이 입력을 여러 종류 쓴다: 랜덤, 정지(전부 0), 직선 등속, 커브, 그리고 3탄에서
혼동을 일으켰던 실제 stage2 표 행. 마지막 것이 이 대조의 핵심이다 -- 그 값을 순시로
보느냐 누적으로 보느냐가 0.3725와 0.5939를 갈랐다.

    venv_dev/bin/python scripts/etri_metric_crosscheck.py
"""
import os
import sys

import numpy as np
import torch

REPO = "/home/pm97/workspace/sukim/adcl/src/etri_vad/ETRI_E2E_Driving_Challenge"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/pm97/workspace/sukim/adcl/src")

from challenge_metrics import l2_challenge, l2_from_per_step  # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # 헤드리스
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import \
    PlanningMetric                                             # noqa: E402


def ours(pred, gt):
    return l2_challenge(pred, gt)["L2_avg"]


def organizer(pred, gt):
    """VAD.py:640-651 과 동일: 샘플별로 k=1,2,3 누적 ADE를 구해 평균, 그 뒤 샘플 평균."""
    pm = PlanningMetric()
    p = torch.tensor(np.asarray(pred, np.float64))
    g = torch.tensor(np.asarray(gt, np.float64))
    per_sample = []
    for i in range(p.shape[0]):
        vals = []
        for k in range(3):
            cur = (k + 1) * 2
            vals.append(pm.compute_L2(p[i, :cur], g[i, :cur]))
        per_sample.append(float(np.mean(vals)))
    return float(np.mean(per_sample))


def sparsedrive(pred, gt):
    """planning_eval.py:160-168 과 동일: 샘플평균 -> 누적평균 -> idx 1,3,5 평균."""
    p = np.asarray(pred, np.float64)[..., :2]
    g = np.asarray(gt, np.float64)[..., :2]
    per_step = np.sqrt(((p - g) ** 2).sum(-1)).mean(0)
    cum = [float(np.mean(per_step[: i + 1])) for i in range(len(per_step))]
    return float(np.mean([cum[1], cum[3], cum[5]]))


def case_random(n=64, seed=0):
    rng = np.random.default_rng(seed)
    gt = np.cumsum(rng.normal(0, 1.2, (n, 6, 2)), axis=1)
    return gt + rng.normal(0, 0.35, gt.shape), gt


def case_zero(n=16):
    gt = np.cumsum(np.tile(np.array([[3.0, 0.0]]), (6, 1))[None], axis=1)
    gt = np.repeat(gt, n, 0)
    return np.zeros_like(gt), gt


def case_straight(n=8):
    t = np.arange(1, 7)[:, None] * 0.5
    gt = np.repeat((t * np.array([[12.0, 0.0]]))[None], n, 0)
    pred = np.repeat((t * np.array([[11.4, 0.0]]))[None], n, 0)
    return pred, gt


def case_curve(n=8):
    t = np.arange(1, 7) * 0.5
    r, w = 20.0, 0.25
    gt = np.stack([r * np.sin(w * t), r * (1 - np.cos(w * t))], 1)
    pred = np.stack([r * np.sin(w * t * 1.03), r * (1 - np.cos(w * t * 0.97))], 1)
    return np.repeat(pred[None], n, 0), np.repeat(gt[None], n, 0)


def case_stage2_row():
    """3탄에서 혼동을 일으켰던 실제 표 행.

    per_step = 그 6개 숫자로 두고, 세 구현이 모두 0.3725를 내야 한다.
    표 행 자체가 이미 누적이었으므로 0.5939는 mean(row[1],row[3],row[5])로만 나온다.
    두 값이 어떻게 갈리는지 이 케이스로 고정한다.
    """
    per = np.array([0.1840, 0.2883, 0.4141, 0.5629, 0.7351, 0.9306])
    # per_step을 그대로 만족하는 1샘플 예측/GT: x축으로만 오차를 준다
    gt = np.cumsum(np.tile(np.array([[6.0, 0.0]]), (6, 1)), axis=0)[None]
    pred = gt.copy()
    pred[0, :, 0] += per
    return pred, gt, per


def main():
    print("=" * 78)
    print("0-7  챌린지 L2 3중 대조")
    print("=" * 78)
    print(f"  {'케이스':<22} {'우리':>10} {'주최측':>10} {'SparseDrive':>12} {'최대차':>10}")
    fails = 0
    cases = [("random n=64", *case_random()),
             ("정지 출력", *case_zero()),
             ("직선 등속", *case_straight()),
             ("커브", *case_curve()),
             ("random seed=7", *case_random(32, 7))]
    for name, pred, gt in cases:
        a, b, c = ours(pred, gt), organizer(pred, gt), sparsedrive(pred, gt)
        spread = max(a, b, c) - min(a, b, c)
        ok = spread < 1e-9
        fails += (not ok)
        print(f"  {name:<22} {a:>10.6f} {b:>10.6f} {c:>12.6f} {spread:>10.2e}"
              f"  {'' if ok else '<-- 불일치'}")

    pred, gt, per = case_stage2_row()
    a, b, c = ours(pred, gt), organizer(pred, gt), sparsedrive(pred, gt)
    spread = max(a, b, c) - min(a, b, c)
    fails += (spread >= 1e-9)
    print(f"  {'stage2 행(순시취급)':<22} {a:>10.6f} {b:>10.6f} {c:>12.6f} "
          f"{spread:>10.2e}{'' if spread < 1e-9 else '  <-- 불일치'}")

    print()
    print("  같은 6개 숫자의 두 해석:")
    print(f"    순시값으로 보면 (세 구현 모두)          {a:.6f}")
    print(f"    이미 누적인 표 행으로 보면 mean(1,3,5)  "
          f"{np.mean([per[1], per[3], per[5]]):.6f}")
    print(f"    → 실제 stage2 avg 0.5939 는 후자. 3탄에서 전자로 오해했던 값이 0.3725.")

    print()
    print(f"  결과: {'전부 일치' if fails == 0 else f'{fails}건 불일치'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
