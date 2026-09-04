#!/usr/bin/env python
"""표준 비교 표 생성 (4탄 규칙 4 + 부록3 항목 2·6).

최상단 = 라벨 모호성 프록시 고정 행. 그 아래 prior 5종. Phase 1 이후 모델 행이 같은
코드로 추가된다. 열은 `etri_table.COLUMNS` 하나뿐이므로 표가 갈릴 수 없다.

worst-N은 **시나리오 중복을 제거**한다(항목 6). 같은 시나리오의 인접 프레임이 상위를
독점하면 진단 가치가 없다 -- 3탄에서 실제로 20건 중 상위 3건이 두 시나리오였다.

    python scripts/etri_report_table.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import waypoint_weights          # noqa: E402
from etri_table import (COLUMNS, ValSet, fmt, render,   # noqa: E402
                        residual)
from etri_priors import PRIORS                          # noqa: E402

OUT_MD = "/home/pm97/workspace/sukim/adcl/logs/baselines_etri.md"
OUT_JSON = "/home/pm97/workspace/sukim/adcl/logs/baselines_etri.json"
GLITCH = "/tmp/pm97/data/etri/glitch_flags.npz"


def worst_n(v, pred, n=20, per_scenario=1):
    per = np.sqrt(((pred - v.gt) ** 2).sum(-1)) @ waypoint_weights()
    order = np.argsort(-per)
    seen, out = {}, []
    for k in order:
        s = v.scen[k]
        if seen.get(s, 0) >= per_scenario:
            continue
        seen[s] = seen.get(s, 0) + 1
        out.append({"scenario": str(s), "frame": int(v.frame[k]),
                    "L2": round(float(per[k]), 4),
                    "speed": round(float(v.speed[k]), 3),
                    "goal_dist": round(float(np.linalg.norm(v.goal[k])), 2),
                    "goal_yaw": round(float(v.d["goal_yaw"][v.vi[k]]), 4),
                    "vad_cmd": ["right", "left", "straight"][v.vcmd[k]],
                    "meta": v.labels[v.meta[k]] if v.meta[k] >= 0 else "?"})
        if len(out) >= n:
            break
    return out


def main():
    v = ValSet()
    rows = [("**라벨 모호성 프록시** (hd_ego_pose 대체GT)", v.label_proxy_row())]
    res = {"label_proxy": rows[0][1]}
    preds = {}
    for name, fn in PRIORS:
        p = fn(v.vel, v.yawrate, v.goal)
        preds[name] = p
        r = v.row(p)
        rows.append((name, r))
        res[name] = r

    g = np.load(GLITCH)["glitch"]
    gv = g[v.vi]
    note = (
        "**라벨 모호성 프록시**는 `ego_pose`(채점 원천)와 `hd_ego_pose`(지도정합)로 만든 "
        "두 GT의 불일치다. **노이즈 바닥이 아니다** — `ego_pose` 단독 오차는 미지이므로 "
        "상한성 지표로만 읽는다. 모델과 동일한 앵커·가중·열로 계산했다.\n\n"
        "`가중` 열만 test 정합 가중 평균이고 나머지는 해당 부분집합의 무가중 평균이다. "
        "`가중`이 리더보드 예측기, `right`가 1차 지표다.\n\n"
        f"val 앵커 중 라벨 글리치 {int(gv.sum())}건 ({gv.mean()*100:.2f}%) 포함 상태의 값이다.\n")

    md = ["# ETRI 챌린지 — 표준 비교 표\n\n",
          f"- val {len(v.vi):,} 앵커 (홀드아웃 38 시나리오, 2Hz) / "
          f"mini-val {len(v.sp['minival_idx'])} 앵커 (8 시나리오, affine·임계 적합 전용)\n",
          "- 지표: `src/challenge_metrics.py` 챌린지 L2 (1·2·3초 누적 ADE 평균, "
          "가중 [11,11,5,5,2,2]/36). 3중 대조 통과.\n",
          "- 참고: VAD tiny 리더보드 0.561 / 우리 SparseDrive stage2(nuScenes) 0.5939 — 같은 단위\n\n",
          render(rows, note=note)]

    # 글리치 제외 시 표 (A/B용)
    keep = ~gv
    rows_ng = []
    for name, r in rows:
        pass
    sub = []
    alt = np.load("/tmp/pm97/data/etri/label_proxy_val.npz")["fut_hd"].astype(np.float64)
    for name, p in [("라벨 모호성 프록시", alt)] + [(n, preds[n]) for n, _ in PRIORS]:
        d = np.sqrt(((p[keep] - v.gt[keep]) ** 2).sum(-1))
        from challenge_metrics import l2_from_per_step
        ww = v.w[keep]
        sub.append((name, {
            "가중": float(l2_from_per_step((d * (ww / ww.sum())[:, None]).sum(0))["L2_avg"]),
            "무가중": float(l2_from_per_step(d.mean(0))["L2_avg"])}))
    md.append("\n## 글리치 민감도 (진단용, **정책 아님**)\n\n"
              "확정 정책: **val은 글리치 포함 유지** — val이 test를 닮아야 한다. 제외는 train에서만 하고 Phase 1에서 플래그 A/B로 비교한다 (부록4 항목 2). 아래는 val 글리치 16건이 지표를 얼마나 흔드는지 보는 민감도일 뿐이다.\n\n")
    md.append("| | 가중(포함) | 가중(제외) | 차 |\n|---|---|---|---|\n")
    for (name, s), (_, r) in zip(sub, rows):
        md.append(f"| {name} | {fmt(r['가중'])} | {fmt(s['가중'])} | "
                  f"{fmt(s['가중'] - r['가중'])} |\n")
    res["glitch_excluded"] = {n: {k: round(x, 5) for k, x in s.items()}
                              for n, s in sub}

    # worst-N (시나리오 1건씩)
    wn = worst_n(v, preds["goal-보간 등가속"], n=20, per_scenario=1)
    res["worst20_goalaccel_dedup"] = wn
    md.append("\n## worst-20 (goal-등가속, 시나리오 중복 제거)\n\n")
    md.append("| 시나리오 | frame | L2 | speed | goal거리 | goal yaw | vad_cmd | meta |\n")
    md.append("|" + "---|" * 8 + "\n")
    for r in wn:
        md.append(f"| {r['scenario']} | {r['frame']} | {r['L2']} | {r['speed']} | "
                  f"{r['goal_dist']} | {r['goal_yaw']} | {r['vad_cmd']} | {r['meta']} |\n")

    open(OUT_MD, "w").write("".join(md))
    json.dump(res, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)

    # 콘솔
    print(f"{'':<34} " + " ".join(f"{c:>9}" for c in COLUMNS))
    for name, r in rows:
        n2 = name.replace("**", "")[:33]
        print(f"{n2:<34} " + " ".join(f"{fmt(r[c]):>9}" for c in COLUMNS))
    print(f"\n표본 — " + ", ".join(f"{c} {rows[0][1]['n_'+c]:,}" for c in COLUMNS))
    print(f"\n프록시 대비 잔여:")
    print(f"{'':<34} " + " ".join(f"{c:>9}" for c in COLUMNS))
    for name, r in rows[1:]:
        res = residual(r, rows[0][1])
        print(f"{name[:33]:<34} " + " ".join(f"{fmt(res[c]):>9}" for c in COLUMNS))
    print(f"\n글리치 민감도 (진단용, 정책 아님 — val은 포함 유지):")
    for (name, s), (_, r) in zip(sub, rows):
        print(f"  {name[:32]:<34} {r['가중']:.5f} -> {s['가중']:.5f} "
              f"({s['가중']-r['가중']:+.5f})")
    print(f"\nworst20 상위 5 (시나리오 중복 제거):")
    for r in wn[:5]:
        print(f"  {r['scenario']}/f{r['frame']:<4} L2 {r['L2']:<7} "
              f"v {r['speed']:<6} goal {r['goal_dist']:<6} yaw {r['goal_yaw']:<8} "
              f"{r['vad_cmd']}/{r['meta']}")
    print(f"\n저장 {OUT_MD}\n     {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
