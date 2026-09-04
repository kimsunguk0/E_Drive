#!/usr/bin/env python
"""Phase 0 — 잔차·노이즈 해부 + 사전 검증. 이미지 불필요, ego/parquet만.

항목 (4탄 Phase 0)
  1. prior-MLP 오차 분해: vad_cmd×이동상태 / 속도 bin / goal거리 bin / worst 20 clip
  2. **GT 노이즈 바닥** — 이 대회의 이론적 최저점. 단독 보고 대상
  3. deep-stop 임계 민감도 그리드 (최종값은 mini-val에서 선택)
  4. test U_TURN 12 clip이 관측 기준 deep-stop인가
  5. test +50행의 rpy(yaw) 실재 확인
  6. train vad_cmd·meta 프레임 분포 -> right 표본 충분성
  7. 우리 metric을 주최측 metric_stp3와 toy 입력으로 3중 대조

노이즈 바닥을 어떻게 재는가
---------------------------
GT 자체가 오도메트리 추정치라 완벽하지 않다. 같은 물리적 지점을 서로 다른 프레임에서
관측하면 값이 다르고, 그 불일치는 어떤 모델도 줄일 수 없다.

    frame t   에서 본 +3.0초 지점  (fut[t][5],  t+30 프레임)
    frame t+5 에서 본 +2.5초 지점  (fut[t+5][4], 역시 t+30 프레임)

둘은 **같은 프레임의 자차 위치**를 서로 다른 원점에서 표현한 것이다. t+5 기준 좌표를
t 기준으로 되돌리면 이론상 정확히 같아야 한다. 남는 차이가 오도메트리 자기불일치이고,
이것이 곧 채점 지표에서 뺄 수 없는 바닥이다.

주의: 이 값은 '바닥의 추정치'다. 두 관측이 상관된 오차를 공유하면 과소추정, 원점 변환
자체에 오차가 있으면 과대추정 쪽으로 편향된다. 그래서 상한/하한을 함께 낸다.

    python scripts/etri_phase0.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import (l2_challenge, l2_from_per_step,  # noqa: E402
                              waypoint_weights)

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
CACHE_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
PERSIST = "/home/pm97/workspace/sukim/adcl"
OUT = os.path.join(PERSIST, "logs", "etri_phase0.json")

NWP, DT_WP, T_GOAL = 6, 0.5, 5.0
TS = np.arange(1, NWP + 1) * DT_WP
STOP = 0.5
VCMD = ["right", "left", "straight"]
SPEED_BINS = [0, 3, 8, 15, np.inf]
GOAL_BINS = [0, 1, 10, 30, 50, 70, np.inf]


def wl2(pred, gt, w=None):
    if w is None:
        return l2_challenge(pred, gt)["L2_avg"]
    d = np.sqrt(((pred[..., :2] - gt[..., :2]) ** 2).sum(-1))
    if w.sum() <= 0:
        return float("nan")
    return l2_from_per_step((d * (w / w.sum())[:, None]).sum(0))["L2_avg"]


def binlab(edges, unit=""):
    out = []
    for i in range(len(edges) - 1):
        hi = "+" if edges[i + 1] == np.inf else f"–{edges[i+1]:g}"
        out.append(f"{edges[i]:g}{hi}{unit}")
    return out


def binof(x, edges):
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)


# ------------------------------------------------------------ 0-2 노이즈 바닥
def noise_floor(d):
    """fut[t][5] 와 fut[t+5][4] 는 같은 프레임(t+30)의 자차 위치다.

    fut는 각 앵커의 ego 프레임 기준 누적 좌표이므로, t+5 기준 좌표를 t 기준으로
    되돌려야 비교할 수 있다. 변환은 his를 쓴다 -- his[t][30]=0이고
    t 기준에서 t+5의 위치는 fut[t][0] (= +5프레임 = 0.5초 지점)이다.
    회전은 두 프레임의 heading 차이인데 ego_cache에 heading이 없으므로
    his 궤적의 접선으로 근사한다... 대신 더 깨끗한 방법을 쓴다:
    **거리 불변량**만 비교한다. 원점·회전에 무관한 양이므로 변환 오차가 섞이지 않는다.

        r_t   = ||fut[t][5]  - fut[t][0]||     t+5 -> t+30 구간의 직선거리
        r_t5  = ||fut[t+5][4]||                t+5 -> t+30 구간의 직선거리

    같은 물리 구간의 길이를 두 관측이 각각 말한 값이고, 차이는 순수 오도메트리
    불일치다. 이것을 3초 waypoint 오차 단위로 환산해 바닥으로 삼는다.
    """
    si, fr = d["scen_idx"], d["frame"]
    fut = d["fut"].astype(np.float64)
    spd = d["speed"].astype(np.float64)
    # t와 t+5가 같은 시나리오 안에 있어야 한다
    idx = np.arange(len(fr))
    nxt = np.full(len(fr), -1, np.int64)
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(si, fr))}
    for i, (a, b) in enumerate(zip(si, fr)):
        j = key.get((int(a), int(b) + 5), -1)
        nxt[i] = j
    ok = nxt >= 0
    a, b = idx[ok], nxt[ok]

    r_t = np.linalg.norm(fut[a][:, 5, :] - fut[a][:, 0, :], axis=1)
    r_t5 = np.linalg.norm(fut[b][:, 4, :], axis=1)
    diff = np.abs(r_t - r_t5)
    mov = spd[a] >= STOP

    # 상한 추정: 두 관측 오차가 독립이라면 각 관측의 오차는 diff/sqrt(2)
    res = {
        "method": "같은 물리구간(t+5 -> t+30)의 직선거리를 t와 t+5 두 앵커에서 각각 계산해 비교. "
                  "원점·회전 변환을 타지 않는 불변량이라 변환 오차가 섞이지 않는다.",
        "n_pairs": int(ok.sum()),
        "all": {"p50": round(float(np.percentile(diff, 50)), 5),
                "mean": round(float(diff.mean()), 5),
                "p95": round(float(np.percentile(diff, 95)), 5)},
        "moving_only": {"n": int(mov.sum()),
                        "p50": round(float(np.percentile(diff[mov], 50)), 5),
                        "mean": round(float(diff[mov].mean()), 5),
                        "p95": round(float(np.percentile(diff[mov], 95)), 5)},
        "per_observation_estimate_mean": round(float(diff.mean() / np.sqrt(2)), 5),
    }
    # 3초 지점 오차가 이 크기라고 보고, waypoint 가중으로 챌린지 L2 단위 환산.
    # 오차가 시간에 비례해 커진다고 가정하면 j번째 waypoint 오차 = e3 * (j+1)/6.
    e3 = diff.mean() / np.sqrt(2)
    per = e3 * (np.arange(1, NWP + 1) / NWP)
    res["challenge_L2_floor_estimate"] = round(
        float(l2_from_per_step(per)["L2_avg"]), 5)
    e3m = diff[mov].mean() / np.sqrt(2)
    res["challenge_L2_floor_moving"] = round(
        float(l2_from_per_step(e3m * (np.arange(1, NWP + 1) / NWP))["L2_avg"]), 5)
    return res


def dynamics(d):
    """가속·jerk 규모 — GT 궤적이 물리적으로 매끄러운지."""
    acc = np.linalg.norm(d["acc"].astype(np.float64), axis=1)
    spd = d["speed"].astype(np.float64)
    mov = spd >= STOP
    # jerk: 같은 시나리오 내 인접 앵커의 가속 차분 / dt(0.1s * 1프레임)
    si, fr = d["scen_idx"], d["frame"]
    a2 = d["acc"].astype(np.float64)
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(si, fr))}
    j = []
    for i, (x, y) in enumerate(zip(si, fr)):
        k = key.get((int(x), int(y) + 1), -1)
        if k >= 0:
            j.append(np.linalg.norm(a2[k] - a2[i]) / 0.1)
    j = np.asarray(j)
    return {"accel_p50": round(float(np.percentile(acc, 50)), 4),
            "accel_p95": round(float(np.percentile(acc, 95)), 4),
            "accel_p95_moving": round(float(np.percentile(acc[mov], 95)), 4),
            "accel_max": round(float(acc.max()), 3),
            "jerk_p50": round(float(np.percentile(j, 50)), 4),
            "jerk_p95": round(float(np.percentile(j, 95)), 4),
            "jerk_max": round(float(j.max()), 2),
            "note": "가속 p95가 3 m/s^2 부근이면 정상 주행. jerk 꼬리가 크면 오도메트리 노이즈"}


# ------------------------------------------------------------ 0-3 임계 민감도
def deepstop_grid(d, t, vi, gt):
    def past_speeds(his):
        return np.linalg.norm(np.diff(his.astype(np.float64), axis=1), axis=2) / 0.1

    ps_t = past_speeds(t["his"])
    gd_t = np.linalg.norm(t["goal"].astype(np.float64), axis=1)
    ps_v = past_speeds(d["his"][vi])
    gd_v = np.linalg.norm(d["goal"][vi].astype(np.float64), axis=1)
    rows = []
    for v in (0.3, 0.5, 0.7):
        for g in (0.5, 1.0, 2.0):
            mt = (ps_t < v).all(1) & (gd_t < g)
            mv = (ps_v < v).all(1) & (gd_v < g)
            z = np.zeros((int(mv.sum()), NWP, 2))
            rows.append({
                "v": v, "goal": g,
                "test_n": int(mt.sum()),
                "test_ratio": round(float(mt.mean()), 5),
                "val_n": int(mv.sum()),
                "zero_L2_on_val_subset": (round(wl2(z, gt[mv]), 6)
                                          if mv.sum() else None),
                "val_gt_disp3s_p95": (round(float(np.percentile(
                    np.linalg.norm(gt[mv][:, -1, :], axis=1), 95)), 5)
                    if mv.sum() else None)})
    return rows


def main():
    d = np.load(CACHE, allow_pickle=True)
    t = np.load(CACHE_TEST, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    vi, mvi = sp["val_idx"], sp["minival_idx"]
    w = sp["val_weight"].astype(np.float64)
    labels = list(d["meta_labels"])

    gt = d["fut"][vi].astype(np.float64)
    goal = d["goal"][vi].astype(np.float64)
    vel = d["vel"][vi].astype(np.float64)
    spd = d["speed"][vi].astype(np.float64)
    vcm = d["vad_cmd"][vi].astype(int)
    mtm = d["meta"][vi].astype(int)
    mov = spd >= STOP

    out = {}

    # ---- 0-2 노이즈 바닥 (최우선)
    out["noise_floor"] = noise_floor(d)
    out["gt_dynamics"] = dynamics(d)
    print("=" * 66)
    print("[0-2] GT 노이즈 바닥 — 이 대회의 이론적 최저점")
    print("=" * 66)
    nf = out["noise_floor"]
    print(f"  방법: {nf['method']}")
    print(f"  쌍 {nf['n_pairs']:,}개")
    print(f"  구간거리 불일치  전체 p50 {nf['all']['p50']:.4f}  mean {nf['all']['mean']:.4f}"
          f"  p95 {nf['all']['p95']:.4f} m")
    print(f"                   이동만 p50 {nf['moving_only']['p50']:.4f}"
          f"  mean {nf['moving_only']['mean']:.4f}  p95 {nf['moving_only']['p95']:.4f} m")
    print(f"  관측 1회당 오차 추정 (독립 가정, /sqrt2) : {nf['per_observation_estimate_mean']:.4f} m")
    print(f"  --> 챌린지 L2 바닥 추정   전체 {nf['challenge_L2_floor_estimate']:.4f}"
          f"   이동만 {nf['challenge_L2_floor_moving']:.4f}")
    dy = out["gt_dynamics"]
    print(f"  가속 p50 {dy['accel_p50']} p95 {dy['accel_p95']} max {dy['accel_max']} m/s^2")
    print(f"  jerk  p50 {dy['jerk_p50']} p95 {dy['jerk_p95']} max {dy['jerk_max']} m/s^3")
    print()

    # ---- 0-3 임계 민감도
    out["deepstop_grid"] = deepstop_grid(d, t, vi, gt)
    print("[0-3] deep-stop 임계 민감도")
    print(f"  {'v':>4} {'goal':>5} {'test_n':>7} {'test%':>7} {'val_n':>6} {'zeroL2':>9}")
    for r in out["deepstop_grid"]:
        print(f"  {r['v']:>4} {r['goal']:>5} {r['test_n']:>7} "
              f"{r['test_ratio']*100:>6.2f}% {r['val_n']:>6} "
              f"{(r['zero_L2_on_val_subset'] if r['zero_L2_on_val_subset'] is not None else float('nan')):>9.6f}")
    print()

    # ---- 0-4 test U_TURN이 deep-stop인가
    ps_t = np.linalg.norm(np.diff(t["his"].astype(np.float64), axis=1), axis=2) / 0.1
    gd_t = np.linalg.norm(t["goal"].astype(np.float64), axis=1)
    ds = (ps_t < STOP).all(1) & (gd_t < 1.0)
    ut = t["meta"].astype(int) == labels.index("U_TURN")
    out["test_uturn"] = {
        "n_uturn": int(ut.sum()),
        "n_uturn_deepstop": int((ut & ds).sum()),
        "all_deepstop": bool(ut.sum() and (ut & ds).all()),
        "uturn_goal_dist": [round(float(x), 4) for x in gd_t[ut]],
        "uturn_max_past_speed": [round(float(x), 4) for x in ps_t[ut].max(1)],
    }
    print(f"[0-4] test U_TURN {out['test_uturn']['n_uturn']}개 중 deep-stop "
          f"{out['test_uturn']['n_uturn_deepstop']}개  "
          f"전부 deep-stop: {out['test_uturn']['all_deepstop']}")
    print(f"      goal 거리: {out['test_uturn']['uturn_goal_dist']}")
    print(f"      과거 최대속도: {out['test_uturn']['uturn_max_past_speed']}")
    print()

    # ---- 0-6 right 표본 충분성
    tr_idx = sp["train_idx"]
    vc_tr = d["vad_cmd"][tr_idx].astype(int)
    mt_tr = d["meta"][tr_idx].astype(int)
    out["train_balance"] = {
        "n_train_anchors": int(len(tr_idx)),
        "vad_cmd": {VCMD[i]: int((vc_tr == i).sum()) for i in range(3)},
        "vad_cmd_ratio": {VCMD[i]: round(float((vc_tr == i).mean()), 5) for i in range(3)},
        "meta": {labels[i]: int((mt_tr == i).sum()) for i in range(len(labels))},
        "test_vad_ratio": {VCMD[i]: round(float((t["vad_cmd"] == i).mean()), 5)
                           for i in range(3)},
    }
    b = out["train_balance"]
    print(f"[0-6] train 앵커 {b['n_train_anchors']:,}  "
          f"right {b['vad_cmd']['right']:,} ({b['vad_cmd_ratio']['right']*100:.2f}%)  "
          f"left {b['vad_cmd']['left']:,} ({b['vad_cmd_ratio']['left']*100:.2f}%)")
    print(f"      test 비율 right {b['test_vad_ratio']['right']*100:.2f}% "
          f"left {b['test_vad_ratio']['left']*100:.2f}%")
    print(f"      → 미러 증강으로 left 표본을 right로 뒤집으면 right "
          f"{b['vad_cmd']['right']+b['vad_cmd']['left']:,}까지 확보 가능")
    print()

    # ---- 0-1 prior 오차 분해 (goal-등가속 기준. MLP는 Phase 1에서 갱신)
    a = 2.0 * (goal - vel * T_GOAL) / T_GOAL ** 2
    tt = TS[None, :, None]
    p = vel[:, None, :] * tt + 0.5 * a[:, None, :] * tt ** 2
    dec = {"overall_w": round(wl2(p, gt, w), 5),
           "overall_unw": round(wl2(p, gt), 5)}
    for ci, cn in enumerate(VCMD):
        for mv2, mn in ((True, "이동"), (False, "정지")):
            m = (vcm == ci) & (mov == mv2)
            dec[f"{cn}/{mn}"] = round(wl2(p[m], gt[m], w[m]), 5) if m.sum() else None
            dec[f"n_{cn}/{mn}"] = int(m.sum())
        m = vcm == ci
        dec[f"{cn}/전체"] = round(wl2(p[m], gt[m], w[m]), 5) if m.sum() else None
    sb, gb = binof(spd, SPEED_BINS), binof(np.linalg.norm(goal, axis=1), GOAL_BINS)
    dec["speed_bins"] = {}
    for i, lab in enumerate(binlab(SPEED_BINS, " m/s")):
        m = sb == i
        dec["speed_bins"][lab] = {"n": int(m.sum()),
                                  "L2_w": round(wl2(p[m], gt[m], w[m]), 5) if m.sum() else None}
    dec["goal_bins"] = {}
    for i, lab in enumerate(binlab(GOAL_BINS, " m")):
        m = gb == i
        dec["goal_bins"][lab] = {"n": int(m.sum()),
                                 "L2_w": round(wl2(p[m], gt[m], w[m]), 5) if m.sum() else None}
    # worst 20
    per = np.sqrt(((p - gt) ** 2).sum(-1))
    wgt = waypoint_weights()
    score = per @ wgt
    order = np.argsort(-score)[:20]
    dec["worst20"] = [{"val_row": int(vi[k]),
                       "scenario": str(d["scenarios"][d["scen_idx"][vi[k]]]),
                       "frame": int(d["frame"][vi[k]]),
                       "L2": round(float(score[k]), 4),
                       "speed": round(float(spd[k]), 3),
                       "goal_dist": round(float(np.linalg.norm(goal[k])), 2),
                       "vad_cmd": VCMD[vcm[k]],
                       "meta": labels[mtm[k]] if mtm[k] >= 0 else "?"}
                      for k in order]
    out["prior_decomposition_goalaccel"] = dec
    print("[0-1] goal-등가속 오차 분해")
    print(f"      전체 가중 {dec['overall_w']}  무가중 {dec['overall_unw']}")
    for lab, v in dec["speed_bins"].items():
        print(f"      속도 {lab:>10}  n={v['n']:>5}  L2 {v['L2_w']}")
    print(f"      worst20 상위 3: " + ", ".join(
        f"{x['scenario']}/f{x['frame']} L2={x['L2']} ({x['vad_cmd']},{x['meta']})"
        for x in dec["worst20"][:3]))
    print()

    # ---- 0-5 test +50행 rpy
    out["test_goal_yaw"] = {
        "note": "ego_cache_test에는 yaw를 담지 않았으므로 parquet를 직접 확인",
    }

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"저장 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
