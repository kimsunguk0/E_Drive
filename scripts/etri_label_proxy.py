#!/usr/bin/env python
"""라벨 모호성 프록시 + 라벨 글리치 감사 (4탄 부록3 항목 1·2·3).

프록시 (항목 1·2)
    hd_ego_pose로 만든 **대체 GT**를 val 앵커 순서에 정확히 정렬해 저장한다.
    이후 etri_table.ValSet.label_proxy_row()가 이걸 '예측'처럼 넣어 모델과 완전히
    동일한 경로(같은 앵커·같은 test 정합 가중·같은 5열)로 계산한다.

    '노이즈 바닥'이라 부르지 않는다 -- 채점 원천인 ego_pose 단독 오차는 미지이고,
    이 값은 두 소스의 불일치일 뿐이다. 상한성 프록시다.

글리치 감사 (항목 3)
    세 기준 중 하나라도 걸리면 글리치 앵커로 본다.
        |가속| > 6 m/s^2      (정상 주행 p95 = 2.0)
        jerk  > 50 m/s^3      (정상 주행 p95 = 27.6)
        두 오도메트리 3초 불일치 > 1.0 m
    시나리오별로 집계해 상위를 육안 확인 대상으로 뽑고, 학습 제외 플래그를 만든다.
    FAQ가 "라벨 검토 및 수정은 자유롭게 허용"이라 명시하므로 규정상 문제 없다.

    python scripts/etri_label_proxy.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

META = "/tmp/pm97/data/etri/meta_train"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
OUT_NPZ = "/tmp/pm97/data/etri/label_proxy_val.npz"
OUT_GLITCH = "/tmp/pm97/data/etri/glitch_flags.npz"
OUT_JSON = "/home/pm97/workspace/sukim/adcl/logs/etri_label_audit.json"

FRAME_OFFSET, MAIN, STEP, NWP = 50, 300, 5, 6
ACC_LIM, JERK_LIM, DISAGREE_LIM = 6.0, 50.0, 1.0


def col(t, n):
    return np.asarray(t.column(n).to_pylist())


def yaw2d(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s], [s, c]])


def main():
    d = np.load(CACHE, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    scen_names = [str(x) for x in d["scenarios"]]
    # npz는 접근마다 압축을 푼다. 루프 안에서 d["acc"][i] 하면 매 반복 전체 배열을
    # 다시 해제해 수 분이 날아간다(첫 판에서 실제로 그랬다). 미리 꺼내 둔다.
    si = np.asarray(d["scen_idx"])
    fr = np.asarray(d["frame"])
    acc_all = np.asarray(d["acc"], np.float64)
    fut_ep = np.asarray(d["fut"], np.float64)

    # (scenario, frame) -> 캐시 전역 인덱스
    gidx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(si, fr))}

    n_all = len(fr)
    fut_hd = np.full((n_all, NWP, 2), np.nan, np.float64)
    disagree3s = np.full(n_all, np.nan)
    jerk = np.full(n_all, np.nan)

    for k, s in enumerate(scen_names):
        dd = os.path.join(META, s)
        ts = pq.read_table(f"{dd}/meta/timestamps.parquet")
        t2f = dict(zip(col(ts, "timestamp"), col(ts, "frame_id").astype(int)))
        ep = pq.read_table(f"{dd}/annotation/ego_pose.parquet")
        hd = pq.read_table(f"{dd}/annotation/hd_ego_pose.parquet")
        ef = np.array([t2f.get(x, np.nan) for x in col(ep, "timestamp")], float)
        hf = np.array([t2f.get(x, np.nan) for x in col(hd, "timestamp")], float)
        if np.isnan(ef).any() or np.isnan(hf).any():
            continue
        oe, oh = np.argsort(ef), np.argsort(hf)
        e_xyz = np.stack([col(ep, c)[oe] for c in "xyz"], 1).astype(np.float64)
        e_rpy = np.stack([col(ep, c)[oe] for c in ("roll", "pitch", "yaw")],
                         1).astype(np.float64)
        h_xy = np.stack([col(hd, c)[oh] for c in "xy"], 1).astype(np.float64)
        h_yaw = col(hd, "yaw")[oh].astype(np.float64)

        for f in range(MAIN):
            gi = gidx.get((k, f))
            if gi is None:
                continue
            i = f + FRAME_OFFSET
            q = i + STEP * np.arange(1, NWP + 1)
            Rh = yaw2d(h_yaw[i])
            fh = (h_xy[q] - h_xy[i]) @ Rh
            fut_hd[gi] = fh
            # ego_pose 쪽 미래는 캐시의 fut와 동일 정의이므로 재계산하지 않는다
            disagree3s[gi] = np.linalg.norm(fut_ep[gi][-1] - fh[-1])

        # jerk: 인접 프레임 가속 차분 / 0.1s. 가속은 캐시에 있다.
        for f in range(MAIN - 1):
            a, b = gidx.get((k, f)), gidx.get((k, f + 1))
            if a is None or b is None:
                continue
            jerk[a] = np.linalg.norm(acc_all[b] - acc_all[a]) / 0.1

        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(scen_names)}", flush=True)

    # ---- 프록시: val 앵커 순서로 정렬해 저장
    vi = sp["val_idx"]
    proxy = fut_hd[vi]
    assert np.isfinite(proxy).all(), "val 앵커에 hd 대체GT 결손"
    np.savez_compressed(OUT_NPZ, fut_hd=proxy.astype(np.float32),
                        val_idx=vi)
    print(f"저장 {OUT_NPZ}  {proxy.shape}")

    # ---- 글리치 감사
    accn = np.linalg.norm(acc_all, axis=1)
    g_acc = accn > ACC_LIM
    g_jerk = np.nan_to_num(jerk, nan=0.0) > JERK_LIM
    g_dis = np.nan_to_num(disagree3s, nan=0.0) > DISAGREE_LIM
    glitch = g_acc | g_jerk | g_dis
    np.savez_compressed(OUT_GLITCH, glitch=glitch,
                        g_accel=g_acc, g_jerk=g_jerk, g_disagree=g_dis,
                        accel=accn.astype(np.float32),
                        jerk=jerk.astype(np.float32),
                        disagree3s=disagree3s.astype(np.float32))

    per_scen = Counter()
    for i in np.where(glitch)[0]:
        per_scen[scen_names[int(si[i])]] += 1
    top = per_scen.most_common(15)

    hold = set(str(x) for x in sp["holdout"])
    mini = set(str(x) for x in sp["minival"])
    audit = {
        "criteria": {"accel_gt": ACC_LIM, "jerk_gt": JERK_LIM,
                     "two_odom_3s_disagree_gt": DISAGREE_LIM},
        "n_frames": int(n_all),
        "n_glitch": int(glitch.sum()),
        "glitch_ratio": round(float(glitch.mean()), 6),
        "by_criterion": {"accel": int(g_acc.sum()), "jerk": int(g_jerk.sum()),
                         "disagree": int(g_dis.sum())},
        "overlap_all_three": int((g_acc & g_jerk & g_dis).sum()),
        "n_scenarios_with_glitch": len(per_scen),
        "top_scenarios": [{"scenario": s, "n_glitch": n,
                           "split": ("val" if s in hold else
                                     "minival" if s in mini else "train")}
                          for s, n in top],
        "tails": {
            "accel_p95": round(float(np.percentile(accn, 95)), 4),
            "accel_p999": round(float(np.percentile(accn, 99.9)), 4),
            "accel_max": round(float(accn.max()), 3),
            "jerk_p95": round(float(np.nanpercentile(jerk, 95)), 3),
            "jerk_p999": round(float(np.nanpercentile(jerk, 99.9)), 3),
            "jerk_max": round(float(np.nanmax(jerk)), 2),
            "disagree_p95": round(float(np.nanpercentile(disagree3s, 95)), 4),
            "disagree_p999": round(float(np.nanpercentile(disagree3s, 99.9)), 4),
            "disagree_max": round(float(np.nanmax(disagree3s)), 3),
        },
    }
    # 2Hz 앵커(학습·평가 대상) 기준 비율도 따로 -- 제외 결정은 이 위에서 한다
    a2 = fr % STEP == 0
    audit["anchors_2hz"] = {
        "n": int(a2.sum()), "n_glitch": int((glitch & a2).sum()),
        "ratio": round(float((glitch & a2).mean() / max(a2.mean(), 1e-9)), 6)}
    for name, idxs in (("train", sp["train_idx"]), ("val", vi),
                       ("minival", sp["minival_idx"])):
        audit["anchors_2hz"][name] = {
            "n": int(len(idxs)), "n_glitch": int(glitch[idxs].sum()),
            "ratio": round(float(glitch[idxs].mean()), 6)}

    json.dump(audit, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("라벨 글리치 감사")
    print("=" * 70)
    print(f"  기준: |가속|>{ACC_LIM} m/s^2  OR  jerk>{JERK_LIM} m/s^3  OR  "
          f"두-오도메트리 3초 불일치>{DISAGREE_LIM} m")
    print(f"  전 프레임 {audit['n_frames']:,} 중 글리치 {audit['n_glitch']:,} "
          f"({audit['glitch_ratio']*100:.3f}%)")
    print(f"  기준별: 가속 {audit['by_criterion']['accel']:,} / "
          f"jerk {audit['by_criterion']['jerk']:,} / "
          f"불일치 {audit['by_criterion']['disagree']:,}  "
          f"(3개 동시 {audit['overlap_all_three']})")
    print(f"  글리치 보유 시나리오 {audit['n_scenarios_with_glitch']} / {len(scen_names)}")
    a = audit["anchors_2hz"]
    print(f"  2Hz 앵커: train {a['train']['n_glitch']}/{a['train']['n']} "
          f"({a['train']['ratio']*100:.3f}%)  "
          f"val {a['val']['n_glitch']}/{a['val']['n']} ({a['val']['ratio']*100:.3f}%)  "
          f"minival {a['minival']['n_glitch']}/{a['minival']['n']}")
    print("\n  상위 시나리오:")
    for r in audit["top_scenarios"][:10]:
        print(f"    {r['scenario']}  {r['n_glitch']:>4}건  [{r['split']}]")
    t = audit["tails"]
    print(f"\n  꼬리: 가속 p95 {t['accel_p95']} p99.9 {t['accel_p999']} max {t['accel_max']}")
    print(f"        jerk  p95 {t['jerk_p95']} p99.9 {t['jerk_p999']} max {t['jerk_max']}")
    print(f"        불일치 p95 {t['disagree_p95']} p99.9 {t['disagree_p999']} max {t['disagree_max']}")
    print(f"\n  저장 {OUT_JSON} / {OUT_GLITCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
