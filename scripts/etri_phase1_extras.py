#!/usr/bin/env python
"""부록5 항목 7·8 — U_TURN 폴백 검증, worst 추적, 제출 #1 위장 후보.

항목7
  * U_TURN 이동 3 clip: test에는 GT가 없으므로 L2를 못 낸다. 두 갈래로 대신한다.
    (a) **기하 대응 부분집합**: val에서 |goal yaw| > 0.5 rad AND 이동인 앵커
        = "회전 호" 구간. U_TURN meta 라벨은 없지만 기하는 같다. 여기서 변형 A vs B의
        L2를 직접 비교해 B의 우아한 열화를 확인한다.
    (b) **실제 3 clip 입력을 넣어 출력 위생 검사**: NaN·폭주 없는지, B가 등가속 prior
        근처에 머무는지. 공개 입력만 쓰므로 규칙 5 위반이 아니다.
  * worst 추적: 승자로 재실행하고, goal yaw 유무 두 ckpt의 앵커별 L2를 대조해
    좌회전 호 2건이 얼마나 줄었는지 수치화.

항목8 (제출 #1 위장 후보)
  a1 = CTRV + deep-stop 게이트   (무학습, goal 전혀 안 씀)
  a2 = ego-only PriorNet         (goal 계열 전부 0)
  goal 사용 모델은 제출 금지 유지.

    python scripts/etri_phase1_extras.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import waypoint_weights          # noqa: E402
from etri_priors import ctrv                            # noqa: E402
from etri_priornet import (NWP, PriorNet, build_feats,   # noqa: E402
                           mirror)
from etri_table import COLUMNS, ValSet, fmt             # noqa: E402

CKPT = "/tmp/pm97/ckpt/etri_priornet"
OUT = "/home/pm97/workspace/sukim/adcl/logs/phase1_extras.json"
STOP_V, STOP_GOAL = 0.5, 1.0
W = waypoint_weights()


def L2(pred, gt, w=None):
    d = np.sqrt(((pred[..., :2] - gt[..., :2]) ** 2).sum(-1))
    per = d.mean(0) if w is None else (d * (w / w.sum())[:, None]).sum(0)
    return float(np.mean([per[:k].mean() for k in (2, 4, 6)]))


def row5(v, pred):
    m = {"가중": None, "무가중": np.ones(len(v.vi), bool), "right": v.vcmd == 0,
         "이동": v.moving, "3–8m/s": v.mid}
    return {k: (L2(pred, v.gt, v.w) if k == "가중"
                else L2(pred[msk], v.gt[msk])) for k, msk in m.items()}


def load(name, d, sp, v, idx=None):
    """ckpt로 추론. norm이 있으면 쓰고, 없으면 결정적 절차로 재구성."""
    ck = torch.load(f"{CKPT}/{name}.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    eo = c.get("ego_only", 0)
    if "norm" in ck:
        n = ck["norm"]
        mu_s, sd_s, mu_c, sd_c = n["mu_s"], n["sd_s"], n["mu_c"], n["sd_c"]
    else:
        tr = sp["train_idx"]
        if c["drop_glitch"]:
            g = np.load("/tmp/pm97/data/etri/glitch_flags.npz")["glitch"]
            tr = tr[~g[tr]]
        st, ct = build_feats(d, tr, c["goal_yaw"], eo)
        if c["mirror"]:
            ms, mc, _ = mirror(st, ct, d["fut"][tr].astype(np.float32),
                               d["meta"][tr].astype(int),
                               c.get("mirror_mode", "all"),
                               d["goal_yaw"][tr].astype(np.float32))
            st, ct = np.concatenate([st, ms]), np.concatenate([ct, mc])
        f = st.reshape(-1, st.shape[-1])
        mu_s, sd_s = f.mean(0), f.std(0) + 1e-6
        mu_c, sd_c = ct.mean(0), ct.std(0) + 1e-6
    ii = v.vi if idx is None else idx
    s, cx = build_feats(d, ii, c["goal_yaw"], eo)
    net = PriorNet(s.shape[-1], cx.shape[-1], c["arch"], c["param"], c["K"])
    net.load_state_dict(ck["state"])
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor((s - mu_s) / sd_s), torch.tensor((cx - mu_c) / sd_c),
                torch.tensor(s), torch.tensor(cx)).numpy().astype(np.float64)
    return p, c


def goal_accel_np(vel, goal):
    ts = np.arange(1, NWP + 1) * 0.5
    a = 2.0 * (goal - vel * 5.0) / 25.0
    t = ts[None, :, None]
    return vel[:, None] * t + 0.5 * a[:, None] * t ** 2


def main():
    d = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=True)
    sp = np.load("/tmp/pm97/data/etri/val_clips.npz", allow_pickle=True)
    t = np.load("/tmp/pm97/data/etri/ego_cache_test.npz", allow_pickle=True)
    v = ValSet()
    gy_v = d["goal_yaw"][v.vi].astype(np.float64)
    out = {}

    have = {f[:-3] for f in os.listdir(CKPT) if f.endswith(".pt")}
    win = "transformer_B+mirror"          # 그리드 승자로 교체 가능
    for cand in ("transformer_B+mirror", "gru_B+mirror"):
        if cand in have:
            win = cand
            break

    # ---------------- 항목7 (a) 회전 호 부분집합에서 A vs B
    arc = (np.abs(gy_v) > 0.5) & v.moving
    print(f"=== 항목7(a) 회전 호 부분집합: |goal yaw|>0.5 rad AND 이동 -> {arc.sum()} 앵커 ===")
    seg = {}
    for nm, ck in (("변형 A", "transformer_A+mirror"), ("변형 B", "transformer_B+mirror")):
        if ck not in have:
            continue
        p, _ = load(ck, d, sp, v)
        ga = goal_accel_np(v.vel, v.goal)
        seg[nm] = {"arc": L2(p[arc], v.gt[arc]), "all": L2(p, v.gt),
                   "arc_vs_goalaccel": L2(ga[arc], v.gt[arc])}
        print(f"  {nm:<8} 회전호 {seg[nm]['arc']:.4f}   전체 {seg[nm]['all']:.4f}")
    if seg:
        print(f"  (같은 부분집합 goal-등가속 무학습 = "
              f"{list(seg.values())[0]['arc_vs_goalaccel']:.4f})")
    out["arc_subset"] = {"n": int(arc.sum()), **seg}

    # ---------------- 항목7 (b) test U_TURN 이동 clip 출력 위생
    labels = [str(x) for x in d["meta_labels"]]
    ut = t["meta"].astype(int) == labels.index("U_TURN")
    ps = np.linalg.norm(np.diff(t["his"].astype(np.float64), axis=1), axis=2) / 0.1
    gd = np.linalg.norm(t["goal"].astype(np.float64), axis=1)
    mov_ut = ut & ~((ps < STOP_V).all(1) & (gd < STOP_GOAL))
    print(f"\n=== 항목7(b) test 이동 U_TURN {int(mov_ut.sum())} clip 출력 위생 ===")
    hyg = []
    tt = dict(his=t["his"], his_yaw=t["his_yaw"], goal=t["goal"],
              goal_yaw=t["goal_yaw"], vel=t["vel"], acc=np.zeros_like(t["vel"]),
              yawrate=t["yawrate"], speed=t["speed"], vad_cmd=t["vad_cmd"],
              meta=t["meta"], fut=np.zeros((len(t["his"]), NWP, 2), np.float32))
    idx_ut = np.where(mov_ut)[0]
    for nm, ck in (("변형 A", "transformer_A+mirror"), ("변형 B", "transformer_B+mirror")):
        if ck not in have:
            continue
        p, _ = load(ck, tt, sp, v, idx=idx_ut)
        ga = goal_accel_np(t["vel"][idx_ut].astype(np.float64),
                           t["goal"][idx_ut].astype(np.float64))
        dev = np.linalg.norm(p[:, -1] - ga[:, -1], axis=1)
        step = np.linalg.norm(np.diff(p, axis=1), axis=2).max(1) / 0.5
        hyg.append({"variant": nm, "finite": bool(np.isfinite(p).all()),
                    "disp3s": [round(float(x), 2) for x in
                               np.linalg.norm(p[:, -1], axis=1)],
                    "max_step_speed": [round(float(x), 2) for x in step],
                    "dev_from_goalaccel3s": [round(float(x), 2) for x in dev]})
        print(f"  {nm}: 유한 {hyg[-1]['finite']}  3초변위 {hyg[-1]['disp3s']}  "
              f"최대구간속도 {hyg[-1]['max_step_speed']} m/s  "
              f"등가속과의 3초차 {hyg[-1]['dev_from_goalaccel3s']} m")
    print(f"  참고 등가속 3초변위 "
          f"{[round(float(x),2) for x in np.linalg.norm(goal_accel_np(t['vel'][idx_ut].astype(np.float64), t['goal'][idx_ut].astype(np.float64))[:,-1],axis=1)]}")
    out["test_moving_uturn"] = {"n": int(mov_ut.sum()),
                               "clips": [str(t["clips"][i]) for i in idx_ut],
                               "hygiene": hyg}

    # ---------------- 항목7 worst 추적 + goal yaw 효과
    print("\n=== 항목7 worst-5 추적 (승자) + goal yaw 효과 ===")
    pw, _ = load(win, d, sp, v)
    per_w = np.sqrt(((pw - v.gt) ** 2).sum(-1)) @ W
    ng = "gru_B+mirror-goalyaw"
    per_ng = None
    if ng in have:
        png, _ = load(ng, d, sp, v)
        per_ng = np.sqrt(((png - v.gt) ** 2).sum(-1)) @ W
    watch = [("20260213-112204", 90), ("20260213-145544", 130),
             ("20260220-101717", 195), ("20260210-104708", 235),
             ("20260220-105317", 170)]
    tr_rows = []
    ga_all = goal_accel_np(v.vel, v.goal)
    per_ga = np.sqrt(((ga_all - v.gt) ** 2).sum(-1)) @ W
    for s, f in watch:
        k = np.where((v.scen == s) & (v.frame == f))[0]
        if not len(k):
            continue
        k = int(k[0])
        r = {"scenario": s, "frame": f,
             "goal_accel": round(float(per_ga[k]), 4),
             "winner": round(float(per_w[k]), 4),
             "goal_yaw": round(float(gy_v[k]), 4),
             "speed": round(float(v.speed[k]), 2)}
        if per_ng is not None:
            r["winner_no_goalyaw"] = round(float(per_ng[k]), 4)
            r["goalyaw_gain"] = round(float(per_ng[k] - per_w[k]), 4)
        tr_rows.append(r)
        print(f"  {s}/f{f:<4} goal-등가속 {r['goal_accel']:<7} 승자 {r['winner']:<7}"
              + (f" -goalyaw {r['winner_no_goalyaw']:<7} 이득 {r['goalyaw_gain']:+.4f}"
                 if per_ng is not None else "")
              + f"  (yaw {r['goal_yaw']}, v {r['speed']})")
    out["worst_watch"] = tr_rows

    # ---------------- 항목8 위장 후보
    print("\n=== 항목8 제출 #1 위장 후보 (goal 미사용) ===")
    cand = {}
    p_ctrv = ctrv(v.vel, v.yawrate, v.goal)
    ps_v = np.linalg.norm(np.diff(v.his, axis=1), axis=2) / 0.1
    # a1의 게이트는 goal을 못 쓰므로 과거 속도만으로 판정한다
    gate = (ps_v < STOP_V).all(1)
    p_a1 = p_ctrv.copy()
    p_a1[gate] = 0.0
    cand["a1 CTRV + 정지게이트(과거속도만)"] = row5(v, p_a1)
    cand["(참고) CTRV 무게이트"] = row5(v, p_ctrv)
    for nm, ck in (("a2 ego-only PriorNet tf/A", "transformer_A+mirror-goalyawEGO-ONLY"),
                   ("a2 ego-only PriorNet gru/A", "gru_A+mirror-goalyawEGO-ONLY")):
        if ck in have:
            p, _ = load(ck, d, sp, v)
            cand[nm] = row5(v, p)
    print(f"{'후보':<34} " + " ".join(f"{c:>9}" for c in COLUMNS))
    for nm, r in cand.items():
        print(f"{nm:<34} " + " ".join(f"{r[c]:>9.4f}" for c in COLUMNS))
    out["disguise_candidates"] = cand
    out["gate_n"] = int(gate.sum())
    print(f"  (a1 게이트 발동 {int(gate.sum())} / {len(v.vi)} 앵커)")

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\n저장 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
