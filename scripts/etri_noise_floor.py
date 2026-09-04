#!/usr/bin/env python
"""GT 노이즈 바닥 — 두 독립 오도메트리 소스의 불일치로 추정.

첫 시도가 왜 틀렸나 (기록)
--------------------------
`fut[t][5]`(t 앵커가 본 +3.0초)와 `fut[t+5][4]`(t+5 앵커가 본 +2.5초)를 비교했다.
같은 물리 프레임(t+30)을 두 번 관측한 것이라고 봤는데, 아니다. **둘 다 같은
`ego_pose` 배열의 같은 행을 읽는다.** 원점만 다르게 표현한 것이므로 거리 불변량은
정확히 같아야 하고, 실제로 0.0006 m가 나왔다 -- 35 m 값에 대해 1.7e-5, 딱 float32
반올림이다. 오도메트리 노이즈가 아니라 저장 정밀도를 측정했다.

이 데이터에는 프레임당 자차 pose 추정치가 하나뿐이므로, 자기일관성으로는 노이즈를
드러낼 수 없다. 독립적인 두 소스가 필요하다.

올바른 측정
-----------
데이터에 오도메트리가 **둘** 있다:
    ego_pose.parquet      원시 오도메트리 (x y z roll pitch yaw) -- 채점 GT의 원천
    hd_ego_pose.parquet   지도 정합 오도메트리 (x y yaw)

같은 3초 구간의 **상대 궤적**을 두 소스로 각각 만들어 챌린지 L2로 비교한다.
절대 드리프트가 아니라 상대 궤적을 보는 이유: 채점은 현재 프레임 원점 기준이므로
장기 드리프트는 상쇄되고, 3초 창 안의 불확실성만 지표에 남는다.

이 값의 해석
    * GT를 어느 소스로 만드느냐에 따라 정답이 이만큼 달라진다 = 라벨 자체의 흔들림.
    * 어떤 모델도 두 소스 중 하나만 맞힐 수 있으므로, 이 값이 **모델이 줄일 수 없는
      성분의 규모**를 알려준다. 단 정확한 하한은 아니다 -- 채점은 ego_pose 기준이고,
      hd 쪽 오차가 더 클 수도 있다. 그래서 '규모의 지표'로만 쓴다.

    python scripts/etri_noise_floor.py
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import l2_challenge, l2_from_per_step  # noqa: E402

META = "/tmp/pm97/data/etri/meta_train"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
OUT = "/home/pm97/workspace/sukim/adcl/logs/etri_noise_floor.json"

FRAME_OFFSET, MAIN, STEP, NWP = 50, 300, 5, 6
STOP = 0.5


def col(t, n):
    return np.asarray(t.column(n).to_pylist())


def yaw2d(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s], [s, c]])


def main():
    scen = sorted(os.listdir(META))
    d = np.load(CACHE, allow_pickle=True)
    sp = np.load(SPLIT, allow_pickle=True)
    val_scen = set(str(x) for x in sp["holdout"])

    diffs, speeds, in_val = [], [], []
    per_step_acc = np.zeros(NWP)
    n_acc = 0
    dropped = 0

    for s in scen:
        dd = os.path.join(META, s)
        ts = pq.read_table(f"{dd}/meta/timestamps.parquet")
        fid = col(ts, "frame_id").astype(int)
        tst = col(ts, "timestamp")
        t2f = dict(zip(tst, fid))

        ep = pq.read_table(f"{dd}/annotation/ego_pose.parquet")
        ef = np.array([t2f.get(x, np.nan) for x in col(ep, "timestamp")], float)
        hd = pq.read_table(f"{dd}/annotation/hd_ego_pose.parquet")
        hf = np.array([t2f.get(x, np.nan) for x in col(hd, "timestamp")], float)
        if np.isnan(ef).any() or np.isnan(hf).any():
            dropped += 1
            continue

        oe, oh = np.argsort(ef), np.argsort(hf)
        e_xyz = np.stack([col(ep, k)[oe] for k in "xyz"], 1).astype(np.float64)
        e_rpy = np.stack([col(ep, k)[oe] for k in ("roll", "pitch", "yaw")],
                         1).astype(np.float64)
        h_xy = np.stack([col(hd, k)[oh] for k in "xy"], 1).astype(np.float64)
        h_yaw = col(hd, "yaw")[oh].astype(np.float64)
        # 정렬 후 index i == frame_id + FRAME_OFFSET (Phase A에서 결손 0 확인)

        for f in range(0, MAIN, STEP):          # 2Hz 앵커
            i = f + FRAME_OFFSET
            q = i + STEP * np.arange(1, NWP + 1)
            Re = Rotation.from_euler("xyz", e_rpy[i], degrees=False).as_matrix()
            fut_e = ((e_xyz[q] - e_xyz[i]) @ Re)[:, :2]
            Rh = yaw2d(h_yaw[i])
            fut_h = (h_xy[q] - h_xy[i]) @ Rh
            per = np.sqrt(((fut_e - fut_h) ** 2).sum(-1))
            per_step_acc += per
            n_acc += 1
            diffs.append(per)
            v = np.linalg.norm((e_xyz[i + 1] - e_xyz[i - 1]) / 0.2 @ Re)
            speeds.append(v)
            in_val.append(s in val_scen)

    diffs = np.asarray(diffs)                    # (N, 6) 시점별 불일치
    speeds = np.asarray(speeds)
    in_val = np.asarray(in_val)
    mov = speeds >= STOP

    def block(mask, label):
        if mask.sum() == 0:
            return None
        per = diffs[mask].mean(0)
        return {"label": label, "n": int(mask.sum()),
                "per_step_mean": [round(float(x), 5) for x in per],
                "challenge_L2": round(float(l2_from_per_step(per)["L2_avg"]), 5),
                "disp3s_p50": round(float(np.percentile(diffs[mask][:, -1], 50)), 5),
                "disp3s_p95": round(float(np.percentile(diffs[mask][:, -1], 95)), 5),
                "disp3s_max": round(float(diffs[mask][:, -1].max()), 4)}

    out = {
        "method": "ego_pose(원시)와 hd_ego_pose(지도정합)로 같은 3초 상대궤적을 각각 만들어 "
                  "챌린지 L2로 비교. 상대궤적이므로 장기 드리프트는 상쇄된다.",
        "caveat": "정확한 하한이 아니다. 채점은 ego_pose 기준이고 hd 쪽 오차가 더 클 수도 있다. "
                  "'모델이 줄일 수 없는 성분의 규모' 지표로만 쓴다.",
        "n_scenarios": len(scen) - dropped, "dropped": dropped,
        "all": block(np.ones(len(diffs), bool), "전체"),
        "moving": block(mov, "이동 |v|>=0.5"),
        "stopped": block(~mov, "정지"),
        "val_scenarios_only": block(in_val, "val 38 시나리오만"),
    }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)

    print("=" * 70)
    print("GT 노이즈 바닥 — 두 오도메트리 소스의 3초 상대궤적 불일치")
    print("=" * 70)
    print(f"  {out['method']}")
    print(f"  시나리오 {out['n_scenarios']}개 (제외 {out['dropped']}), 앵커 {len(diffs):,}\n")
    print(f"  {'구분':<22} {'n':>7} {'챌린지L2':>10} {'3초 p50':>9} {'3초 p95':>9} {'3초 max':>9}")
    for k in ("all", "moving", "stopped", "val_scenarios_only"):
        b = out[k]
        if b:
            print(f"  {b['label']:<22} {b['n']:>7,} {b['challenge_L2']:>10.5f} "
                  f"{b['disp3s_p50']:>9.4f} {b['disp3s_p95']:>9.4f} {b['disp3s_max']:>9.3f}")
    print(f"\n  시점별 평균 불일치(전체): "
          f"{[f'{x:.4f}' for x in out['all']['per_step_mean']]}")
    print(f"\n  저장 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
