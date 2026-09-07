#!/usr/bin/env python
"""리더보드 상위권이 무엇으로 만들어졌는지 추정한다.

공식 지표(질문3 답변): 1s=앞 2점 평균, 2s=앞 4점 평균, 3s=앞 6점 평균, 최종=셋의 평균.
전개하면 waypoint 가중 [11,11,5,5,2,2]/36 과 정확히 같다(검증 포함).

지금 **금지된** 방식들의 점수를 재서 리더보드 1위 0.0924 가 어디서 나오는지 본다.
- 등속/등가속 외삽 (공지1 3-2 금지)
- goal 직선 보간 (공지1 3-1 금지: 영상 없이 목표점만으로 궤적 생성)
- 둘의 결합
이 값들이 0.09 대면 상위권은 코드 심사에서 탈락 대상이라는 뜻이다.
"""
import os
import sys

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402


def official_l2(pred, gt, w):
    """질문3 답변 그대로 2단계 계산. pred/gt [N,6,2]."""
    e = np.linalg.norm(pred - gt, axis=-1)                 # [N,6]
    l1 = e[:, :2].mean(1); l2 = e[:, :4].mean(1); l3 = e[:, :6].mean(1)
    per = (l1 + l2 + l3) / 3.0
    return float(np.average(per, weights=w)), (
        float(np.average(l1, weights=w)), float(np.average(l2, weights=w)),
        float(np.average(l3, weights=w)))


def main():
    arr = C.load_arrays()
    rows = arr["val_idx"]
    sub = arr["frame"][rows] >= 30
    rows = rows[sub]
    w = arr["val_weight"][sub].astype(np.float64)
    G = arr["fut"][rows].astype(np.float64)                # [N,6,2] 0.5s 간격 3초
    g5 = arr["fut5"][rows][:, 9].astype(np.float64)        # 5초 목표점
    his = arr["his"][rows].astype(np.float64)              # [N,31,2] 0.1s 간격
    yaw = arr["his_yaw"][rows].astype(np.float64)
    now = C.HIS_NOW
    T = np.arange(1, 7) * 0.5                              # 0.5..3.0s

    # 가중 등가성 검증
    W6 = np.array([11., 11., 5., 5., 2., 2.]) / 36.
    e = np.linalg.norm(G - G * 0.9, axis=-1)
    a, _ = official_l2(G * 0.9, G, w)
    b = float(np.average((e * W6).sum(1), weights=w))
    print(f"지표 등가성 검증: 2단계 {a:.8f} vs 가중 [11,11,5,5,2,2]/36 {b:.8f} "
          f"-> {'일치' if abs(a-b) < 1e-9 else '불일치'}\n")

    print(f"val38 n={len(rows)}   (리더보드 1위 0.0924106 / 3위 0.1375893)\n")
    print(f"{'방식':>40} {'L2 avg':>9} {'1s':>7} {'2s':>7} {'3s':>7}")
    res = {}

    def rep(name, pred):
        v, (x1, x2, x3) = official_l2(pred, G, w)
        res[name] = v
        print(f"{name:>40} {v:>9.4f} {x1:>7.4f} {x2:>7.4f} {x3:>7.4f}")

    rep("정지 (전부 0)", np.zeros_like(G))
    # 등속: 과거 구간별 속도
    for back, bn in ((5, "0.5s"), (10, "1.0s"), (2, "0.2s")):
        v = (his[:, now] - his[:, now - back]) / (0.1 * back)      # [N,2] m/s
        rep(f"등속 외삽 (과거 {bn} 속도)", v[:, None] * T[None, :, None])
    # 등가속
    v1 = (his[:, now] - his[:, now - 5]) / 0.5
    v0 = (his[:, now - 5] - his[:, now - 10]) / 0.5
    acc = (v1 - v0) / 0.5
    rep("등가속 외삽 (과거 1.0s)",
        v1[:, None] * T[None, :, None] + 0.5 * acc[:, None] * (T ** 2)[None, :, None])
    # yaw rate 포함 (곡선 등속)
    om = (yaw[:, now] - yaw[:, now - 5]) / 0.5
    sp = np.linalg.norm(v1, axis=1)
    th = om[:, None] * T[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.abs(om) > 1e-3, sp / np.where(np.abs(om) > 1e-3, om, 1.0), 0.0)
    xx = np.where(np.abs(om)[:, None] > 1e-3, R[:, None] * np.sin(th),
                  sp[:, None] * T[None, :])
    yy = np.where(np.abs(om)[:, None] > 1e-3, R[:, None] * (1 - np.cos(th)), 0.0)
    rep("곡선 등속 (속도+yaw rate)", np.stack([xx, yy], -1))
    # goal 직선 보간
    rep("goal 직선 보간 (5s 목표점)", g5[:, None] * (T / 5.0)[None, :, None])
    # goal + 과거 속도 프로파일 (진행량은 운동학, 방향은 goal)
    d = np.linalg.norm(g5, axis=1, keepdims=True)
    u = g5 / np.clip(d, 1e-6, None)
    s_kin = (np.linalg.norm(v1, axis=1)[:, None] * T[None, :]
             + 0.5 * ((v1 * acc).sum(1) / np.clip(np.linalg.norm(v1, axis=1), 1e-6, None))[:, None] * (T ** 2)[None, :])
    rep("goal 방향 x 등가속 진행량", u[:, None, :] * s_kin[..., None])
    # goal 방향 + 실제 GT 진행량(상한 참고)
    st = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], 1), axis=1), axis=-1)
    s_gt = np.cumsum(st, 1)
    rep("[상한참고] goal 방향 x GT 진행량", u[:, None, :] * s_gt[..., None])

    print(f"\n우리 최선 image-only (A0+learned selector) val38 = 0.2522")
    best = min(res, key=lambda k: res[k])
    print(f"금지 방식 중 최선: {best} = {res[best]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
