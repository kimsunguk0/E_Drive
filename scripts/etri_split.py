#!/usr/bin/env python
"""로컬 val 분할 (부록2 규칙) + test 정합 가중치 산출.

규칙
----
1. **U_TURN 유일 시나리오는 train 고정.** train 376개 중 시나리오 단위 U_TURN이 단 1개다
   (부록 B.5). 이걸 val로 보내면 train에 U_TURN이 0개가 되고, train에 두면 val에 0개가
   된다. 후자를 택한다 -- 학습 가능성을 지키는 쪽이다. **val은 U_TURN을 못 재며, test의
   1.07%(12 clip)는 로컬에서 검증 불가**임을 명시한다. (test에서 U_TURN은 사실상 정지
   구간이라 -- 목표거리 중앙값 0.006 m -- 위험도는 낮다고 판단하지만, 근거 없는 낙관은
   아니고 부록 B.6의 실측에 근거한다.)

2. **홀드아웃 38개는 층화 추출.** 층 키 = (TURN_RIGHT 프레임 보유 여부,
   vad_cmd right 프레임 비율 3분위, vad_cmd left 프레임 비율 3분위).
   무작위 38개를 뽑으면 TURN_RIGHT가 val에 0개일 확률이 무시할 수 없다 -- train 376개
   중 TURN_RIGHT를 첫 프레임으로 갖는 시나리오가 11개뿐이다. right/left 비율까지 층에
   넣는 이유는, 이 열이 앞으로 모든 모델 비교의 1차 지표(부록2 항목 3)이므로 val의
   right 표본이 train과 체계적으로 다르면 지표 자체가 못 쓰게 된다.

3. **val 지표 2종 병기.**
   (a) 무가중 -- val clip 전체 단순 평균.
   (b) **test 정합 가중** -- test 공개 입력의 결합분포
       (vad_cmd × 현재속도 구간 × +50 목표거리 구간)에 val clip을 매칭한 가중 평균.
       **(b)가 리더보드 예측기다.** (a)는 val 자체의 난이도를 보는 값일 뿐이다.

   결합분포를 쓰는 이유: vad_cmd만 맞추면 부족하다. 부록 B.6에서 LANE_KEEP 목표거리
   중앙값이 train 52.6 m vs test 40.8 m로 어긋났다 -- test clip은 train 프레임의 균등
   표본이 아니다. 속도·목표거리까지 넣어야 리더보드와 상관이 붙는다.

    python scripts/etri_split.py
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
CACHE_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"
PERSIST = "/home/pm97/workspace/sukim/adcl"
HOLDOUT_TXT = os.path.join(PERSIST, "logs", "etri_holdout_38.txt")
OUT_NPZ = "/tmp/pm97/data/etri/val_clips.npz"
OUT_JSON = os.path.join(PERSIST, "logs", "etri_split_report.json")

N_HOLDOUT = 38
N_MINIVAL = 8
SEED = 42
VAL_STRIDE = 5                    # 2Hz 서브샘플 (원 데이터 10Hz)
# 구간 경계. 속도는 정지(0.5)와 도심/간선 구분점을 넣고, 목표거리는 '거의 정지'(1 m)와
# 저속·중속·고속 주행을 가르는 지점을 넣었다. test 분포(부록 B.6)를 보고 정한 값이다.
SPEED_BINS = [0.0, 0.5, 2.0, 5.0, 10.0, 15.0, np.inf]
GOAL_BINS = [0.0, 1.0, 10.0, 30.0, 50.0, 70.0, np.inf]
VCMD = ["right", "left", "straight"]


def binof(x, edges):
    return int(np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-cap", type=float, default=20.0,
                    help="가중치 상한. 희소 셀 하나가 지표를 지배하는 것을 막는다")
    args = ap.parse_args()

    d = np.load(CACHE, allow_pickle=True)
    t = np.load(CACHE_TEST, allow_pickle=True)
    scen = list(d["scenarios"])
    meta_labels = list(d["meta_labels"])
    U = meta_labels.index("U_TURN")
    TR = meta_labels.index("TURN_RIGHT")

    si, vc, mt = d["scen_idx"], d["vad_cmd"], d["meta"]
    n_s = len(scen)

    # ---- 시나리오별 층화 특징
    has_tr = np.zeros(n_s, bool)
    has_ut = np.zeros(n_s, bool)
    r_ratio = np.zeros(n_s)
    l_ratio = np.zeros(n_s)
    for k in range(n_s):
        m = si == k
        has_tr[k] = (mt[m] == TR).any()
        has_ut[k] = (mt[m] == U).any()
        r_ratio[k] = float((vc[m] == 0).mean())
        l_ratio[k] = float((vc[m] == 1).mean())

    forced_train = np.where(has_ut)[0]
    pool = np.array([k for k in range(n_s) if k not in set(forced_train.tolist())])

    def tercile(v, idx):
        q = np.quantile(v[idx], [1 / 3, 2 / 3])
        return np.digitize(v[idx], q)

    rt = tercile(r_ratio, pool)
    lt = tercile(l_ratio, pool)
    keys = [(int(has_tr[k]), int(rt[i]), int(lt[i])) for i, k in enumerate(pool)]

    # 층별 비례 배분. 나머지 자리는 층 크기 큰 순으로 채운다.
    rng = np.random.default_rng(SEED)
    strata = {}
    for i, k in enumerate(pool):
        strata.setdefault(keys[i], []).append(int(k))
    total = len(pool)
    alloc, frac = {}, {}
    for key, members in strata.items():
        exact = N_HOLDOUT * len(members) / total
        alloc[key] = int(np.floor(exact))
        frac[key] = exact - alloc[key]
    while sum(alloc.values()) < N_HOLDOUT:
        key = max(frac, key=lambda kk: (frac[kk], len(strata[kk])))
        if alloc[key] < len(strata[key]):
            alloc[key] += 1
        frac[key] = -1
    holdout = []
    for key in sorted(strata):
        m = sorted(strata[key])
        pick = rng.choice(m, size=min(alloc[key], len(m)), replace=False)
        holdout.extend(int(x) for x in pick)
    holdout = sorted(holdout)
    assert len(holdout) == N_HOLDOUT, len(holdout)
    hold_set = set(holdout)
    rest = [k for k in range(n_s) if k not in hold_set]

    # ---- mini-val: 캘리브레이션(affine 6x2)·deep-stop 임계를 '적합'하는 전용 집합
    #
    # 왜 학습셋에서 떼는가: 본 val에서 적합하고 본 val로 평가하면 누수다. 그렇다고
    # 학습 데이터로 적합하면 모델이 이미 맞춘 잔차라서 캘리브레이션 계수가 과소추정된다.
    # 둘 다 피하려면 **학습에서 빠지고 본 val과도 겹치지 않는 제3의 집합**이어야 한다.
    # 그래서 train 쪽에서 떼고, 학습 인덱스에서 영구 제외한다.
    #
    # 크기: 3탄은 "mini-val 1개"였는데 4탄이 여기에 6x2 affine(12 파라미터)과 임계
    # 그리드 적합까지 얹었다. 1 시나리오(2Hz 60 앵커)로 12 파라미터를 맞추면 계수가
    # 노이즈를 따라간다. 그래서 8개(480 앵커)로 늘렸다. U_TURN 시나리오는 후보에서
    # 제외해 train에 남긴다.
    mv_pool = [k for k in rest if not has_ut[k]]
    mv_keys = {}
    for k in mv_pool:
        i = list(pool).index(k)
        mv_keys.setdefault(keys[i], []).append(k)
    rng_mv = np.random.default_rng(SEED)
    minival = []
    for key in sorted(mv_keys):
        share = N_MINIVAL * len(mv_keys[key]) / len(mv_pool)
        n = int(round(share))
        if n:
            minival.extend(int(x) for x in rng_mv.choice(
                mv_keys[key], size=min(n, len(mv_keys[key])), replace=False))
    minival = sorted(set(minival))[:N_MINIVAL]
    while len(minival) < N_MINIVAL:                 # 반올림 부족분 보충
        cand = [k for k in mv_pool if k not in minival]
        minival.append(int(rng_mv.choice(cand)))
        minival = sorted(set(minival))
    mv_set = set(minival)
    train_scen = [k for k in rest if k not in mv_set]

    # ---- val clip: 홀드아웃 시나리오의 2Hz 앵커
    val_mask = np.isin(si, holdout) & (d["frame"] % VAL_STRIDE == 0)
    val_idx = np.where(val_mask)[0]
    trn_mask = np.isin(si, train_scen) & (d["frame"] % VAL_STRIDE == 0)
    trn_idx = np.where(trn_mask)[0]
    mv_mask = np.isin(si, minival) & (d["frame"] % VAL_STRIDE == 0)
    mv_idx = np.where(mv_mask)[0]
    assert not (set(holdout) & mv_set), "val과 mini-val이 겹친다"
    assert not (set(train_scen) & mv_set), "train과 mini-val이 겹친다"

    # ---- test 정합 가중치
    gd_t = np.linalg.norm(t["goal"], axis=1)
    cell_t = Counter((int(a), binof(b, SPEED_BINS), binof(c, GOAL_BINS))
                     for a, b, c in zip(t["vad_cmd"], t["speed"], gd_t))
    gd_v = np.linalg.norm(d["goal"][val_idx], axis=1)
    cells_v = [(int(a), binof(b, SPEED_BINS), binof(c, GOAL_BINS))
               for a, b, c in zip(vc[val_idx], d["speed"][val_idx], gd_v)]
    cell_v = Counter(cells_v)

    n_t, n_v = sum(cell_t.values()), len(val_idx)
    w = np.zeros(n_v)
    for i, c in enumerate(cells_v):
        if cell_t.get(c, 0) == 0:
            w[i] = 0.0                       # test에 없는 상황은 리더보드에 영향 없음
        else:
            w[i] = (cell_t[c] / n_t) / (cell_v[c] / n_v)
    w = np.minimum(w, args.weight_cap)
    if w.sum() > 0:
        w = w * (n_v / w.sum())              # 평균 1로 정규화 (읽기 쉽게)

    covered_t = sum(v for c, v in cell_t.items() if cell_v.get(c, 0) > 0)
    report = {
        "seed": SEED, "n_holdout": N_HOLDOUT, "val_stride": VAL_STRIDE,
        "forced_train_reason": "U_TURN 시나리오가 train 전체에 1개뿐 -> train 고정, val 커버리지 포기",
        "forced_train_scenarios": [scen[k] for k in forced_train],
        "holdout_scenarios": [scen[k] for k in holdout],
        "minival_scenarios": [scen[k] for k in minival],
        "minival_purpose": "affine 6x2 · deep-stop 임계 적합 전용. 학습에서 영구 제외",
        "n_train_scenarios": len(train_scen),
        "n_minival_scenarios": len(minival),
        "n_minival_anchors": int(len(mv_idx)),
        "n_val_clips": int(n_v), "n_train_anchors": int(len(trn_idx)),
        "strata_alloc": {str(k): int(v) for k, v in sorted(alloc.items())},
        "val_has_turn_right_scenarios": int(sum(has_tr[k] for k in holdout)),
        "train_has_turn_right_scenarios": int(sum(has_tr[k] for k in train_scen)),
        "vad_ratio_check": {
            "train_right": round(float((vc[trn_mask] == 0).mean()), 5),
            "val_right": round(float((vc[val_mask] == 0).mean()), 5),
            "train_left": round(float((vc[trn_mask] == 1).mean()), 5),
            "val_left": round(float((vc[val_mask] == 1).mean()), 5),
            "test_right": round(float((t["vad_cmd"] == 0).mean()), 5),
            "test_left": round(float((t["vad_cmd"] == 1).mean()), 5),
        },
        "val_meta_counts": {meta_labels[i]: int((mt[val_mask] == i).sum())
                            for i in range(len(meta_labels))},
        "weight": {
            "cap": args.weight_cap,
            "n_test_cells": len(cell_t), "n_val_cells": len(cell_v),
            "test_mass_covered_by_val": round(covered_t / n_t, 5),
            "val_clips_with_zero_weight": int((w == 0).sum()),
            "w_min": round(float(w[w > 0].min()), 4) if (w > 0).any() else 0.0,
            "w_max": round(float(w.max()), 4),
            "effective_n": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
        },
    }

    os.makedirs(os.path.dirname(HOLDOUT_TXT), exist_ok=True)
    with open(HOLDOUT_TXT, "w") as f:
        f.write("# 로컬 val 홀드아웃 38 시나리오 (seed=42, 층화). 이 목록을 고정한다.\n")
        f.write(f"# U_TURN 고정 train: {', '.join(scen[k] for k in forced_train)}\n")
        for k in holdout:
            f.write(scen[k] + "\n")
    with open(HOLDOUT_TXT.replace("holdout_38", "minival_8"), "w") as f:
        f.write("# mini-val 8 시나리오 (seed=42, 층화). affine·임계 적합 전용.\n")
        f.write("# 학습에서 영구 제외. 최종 전체 재학습에서도 제외 유지.\n")
        for k in minival:
            f.write(scen[k] + "\n")
    np.savez_compressed(OUT_NPZ, val_idx=val_idx, train_idx=trn_idx,
                        minival_idx=mv_idx,
                        val_weight=w.astype(np.float32),
                        holdout=np.array([scen[k] for k in holdout]),
                        minival=np.array([scen[k] for k in minival]))
    json.dump(report, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
