#!/usr/bin/env python
"""제출 JSON 위생 검사 + 궤적 시각화.

주최측 규격 (INTEL.md Q8):
    {"__flops__": int, "<clip_hash>": [[x1,y1],...,[x6,y6]], ...}
    6x2 누적 waypoint, 0.5/1.0/1.5/2.0/2.5/3.0초, 현재 프레임 원점, x=전방 y=좌.
    누락·형식 오류 clip은 [0,0]으로 평가되고 **불이익은 참가자 책임**이다.

검사 항목
---------
형식   1125 clip 전부 존재 / 6x2 / 유한값 / 숫자형
물리   3초 이동거리, 속도·가속도 범위, 후진량
정합   goal 방향과의 각도차, |p@3s|/|goal| 비율 (등속이면 3/5)
       command(좌/우/직진)와 횡변위 부호 일치
대조   val 예측 분포와 비교 -- test 쪽만 이상하면 파이프라인을 의심한다

    python scripts/etri_submit_check.py --json /tmp/pm97/submit/x.json --plot out.png
"""
import argparse
import json
import os
import sys

import numpy as np

ADCL = "/home/pm97/workspace/sukim/adcl"
sys.path.insert(0, os.path.join(ADCL, "scripts"))
sys.path.insert(0, os.path.join(ADCL, "src"))
EGO_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"
EGO_TRAIN = "/tmp/pm97/data/etri/ego_cache.npz"
DT = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--plot", default="")
    ap.add_argument("--n-plot", type=int, default=24)
    ap.add_argument("--val-pred",
                    default="/tmp/pm97/eval_b200/fullval_v2a_goal_ep2_pred.npy")
    args = ap.parse_args()

    sub = json.load(open(args.json))
    flops = sub.pop("__flops__", None)
    r = np.load(EGO_TEST, allow_pickle=True)
    clips = [str(c) for c in r["clips"]]
    goal, speed, cmd = r["goal"], r["speed"], r["vad_cmd"]
    gi = {c: n for n, c in enumerate(clips)}

    fail = []
    print("=" * 64)
    print("1. 형식")
    print("=" * 64)
    missing = [c for c in clips if c not in sub]
    extra = [k for k in sub if k not in gi]
    print(f"  clip {len(sub)}개 (기대 {len(clips)})  누락 {len(missing)}  잉여 {len(extra)}")
    if missing:
        fail.append(f"누락 clip {len(missing)}개: {missing[:3]}")
    if extra:
        fail.append(f"잉여 키 {len(extra)}개: {extra[:3]}")

    bad_shape = [k for k, v in sub.items() if np.asarray(v, dtype=object).shape != (6, 2)]
    print(f"  6x2 아닌 clip {len(bad_shape)}")
    if bad_shape:
        fail.append(f"shape 오류 {len(bad_shape)}개: {bad_shape[:3]}")

    order = [c for c in clips if c in sub]
    P = np.array([sub[c] for c in order], dtype=np.float64)
    nfin = int((~np.isfinite(P)).any(axis=(1, 2)).sum())
    print(f"  비유한(NaN/Inf) clip {nfin}")
    if nfin:
        fail.append(f"비유한 값 {nfin}개")
    print(f"  __flops__ = {flops if flops is not None else '없음 (1차 컷오프 위험)'}")
    if flops is None:
        fail.append("__flops__ 없음 — 7053 GFLOPs 컷오프 판정 불가")

    idx = np.array([gi[c] for c in order])
    g, sp, cm = goal[idx], speed[idx], cmd[idx]
    end = P[:, -1]
    dist = np.linalg.norm(end, axis=1)
    step = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P], 1),
                                  axis=1), axis=2)          # (N,6) 구간 이동거리
    v = step / DT
    a = np.diff(v, axis=1) / DT

    print("\n" + "=" * 64)
    print("2. 물리")
    print("=" * 64)
    q = lambda x, p: np.percentile(x, p)
    print(f"  3초 이동거리 |p|  p10 {q(dist,10):6.2f}  p50 {q(dist,50):6.2f}  "
          f"p90 {q(dist,90):6.2f}  max {dist.max():6.2f} m")
    print(f"  구간속도 v        p50 {q(v,50):6.2f}  p99 {q(v,99):6.2f}  "
          f"max {v.max():6.2f} m/s  (36 m/s=130 km/h)")
    print(f"  구간가속 a        p1  {q(a,1):6.2f}  p99 {q(a,99):6.2f}  "
          f"|max| {np.abs(a).max():6.2f} m/s^2  (|9| 이상은 비현실)")
    back = P[:, -1, 0] < -0.05
    print(f"  후진(x@3s<-0.05)  {int(back.sum())}/{len(P)} = {back.mean():.1%}"
          f"   그중 정지(<0.5m/s) {int((back & (sp < 0.5)).sum())}")
    if v.max() > 40:
        fail.append(f"비현실 속도 {v.max():.1f} m/s")
    if np.abs(a).max() > 15:
        fail.append(f"비현실 가속 {np.abs(a).max():.1f} m/s^2")

    print("\n" + "=" * 64)
    print("3. 입력 정합 (goal / command)")
    print("=" * 64)
    gn = np.linalg.norm(g, axis=1)
    mv = gn > 2.0                      # goal이 의미 있는 클립만
    ratio = dist[mv] / gn[mv]
    print(f"  |p@3s|/|goal|     p10 {q(ratio,10):.3f}  p50 {q(ratio,50):.3f}  "
          f"p90 {q(ratio,90):.3f}   (등속 기대 0.600)  n={int(mv.sum())}")
    ang = np.degrees(np.abs(np.arctan2(end[mv, 1], end[mv, 0])
                            - np.arctan2(g[mv, 1], g[mv, 0])))
    ang = np.minimum(ang, 360 - ang)
    print(f"  goal 방향 각도차   p50 {q(ang,50):5.1f}°  p90 {q(ang,90):5.1f}°  "
          f">45° 인 clip {int((ang>45).sum())}")
    for c, name, sign in [(0, "우회전", -1), (1, "좌회전", +1), (2, "직진", 0)]:
        m = (cm == c) & mv
        if not m.sum():
            continue
        y = end[m, 1]
        ok = (np.sign(y) == sign).mean() if sign else (np.abs(y) < 3).mean()
        print(f"  cmd={c} {name:4s} n={int(m.sum()):4d}  y@3s p50 {np.median(y):+6.2f} m"
              f"   기대부호 일치 {ok:.1%}")

    print("\n" + "=" * 64)
    print("4. val 예측 분포와 대조")
    print("=" * 64)
    if os.path.isfile(args.val_pred):
        from etri_table import ValSet
        vs = ValSet()
        d = np.load(EGO_TRAIN, allow_pickle=True)
        fr = d["frame"][vs.vi].astype(int)
        vp = np.load(args.val_pred)[fr >= 30]
        vd = np.linalg.norm(vp[:, -1], axis=1)
        print(f"  {'':10s} {'p10':>8} {'p50':>8} {'p90':>8}")
        print(f"  {'test |p|':10s} {q(dist,10):8.2f} {q(dist,50):8.2f} {q(dist,90):8.2f}")
        print(f"  {'val  |p|':10s} {q(vd,10):8.2f} {q(vd,50):8.2f} {q(vd,90):8.2f}")
        print("  * test는 정지 클립 비중이 커서 p10/p50이 낮은 것이 정상이다.")
    else:
        print("  val 예측 없음 — 건너뜀")

    print("\n" + "=" * 64)
    print("판정: " + ("통과 (치명 문제 없음)" if not fail else "문제 발견"))
    for f in fail:
        print("  !! " + f)
    print("=" * 64)

    if args.plot:
        plot(P, g, sp, cm, order, args.plot, args.n_plot)
    return 1 if fail else 0


def plot(P, g, sp, cm, names, out, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    # 속도대·command를 골고루 뽑아 한쪽만 보고 안심하는 것을 막는다
    pick = []
    for lo, hi in [(0, 0.5), (0.5, 3), (3, 8), (8, 15), (15, 99)]:
        cand = np.where((sp >= lo) & (sp < hi))[0]
        if len(cand):
            pick += list(rng.choice(cand, min(len(cand), n // 5), replace=False))
    pick = pick[:n]
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    for ax, i in zip(np.ravel(axes), pick):
        t = np.vstack([[0, 0], P[i]])
        ax.plot(t[:, 1], t[:, 0], "-o", ms=2.5, lw=1.4, color="#1f77b4")
        ax.plot(g[i, 1], g[i, 0], "*", ms=11, color="#d62728")
        ax.plot(0, 0, "s", ms=4, color="k")
        lim = max(6, np.abs(np.r_[t.ravel(), g[i]]).max() * 1.15)
        ax.set_xlim(lim, -lim); ax.set_ylim(-lim * 0.25, lim)
        ax.set_title(f"{names[i][:8]}\n{sp[i]:.1f}m/s cmd{int(cm[i])}", fontsize=7)
        ax.tick_params(labelsize=5); ax.grid(alpha=.3)
    for ax in np.ravel(axes)[len(pick):]:
        ax.axis("off")
    fig.suptitle("blue = predicted 3s path (0.5s steps)   red star = goal (5s)   black square = now   x-axis: left(+)/right(-),  y-axis: forward", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\n그림 저장 {out}  ({len(pick)}개 클립, 속도대별 균등 표집)")


if __name__ == "__main__":
    sys.exit(main())
