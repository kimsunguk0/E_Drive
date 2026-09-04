#!/usr/bin/env python
"""§4.4 진짜 10-waypoint 5초 label 생성. 기존 etri_ego_cache.py build_train 과
정확히 동일한 순서/변환(rel)으로 iterate 하고, fut 를 6→10 step 으로 확장한다.

산출 ego_cache_5s.npz (ego_cache.npz 와 동일 행 순서):
    fut5   (N,10,2)  미래 +5..+50 상대 누적 위치 = 진짜 5초 GT
    mask5  (N,10)    각 waypoint pose 인덱스 in-bounds 여부 (1 valid)
    scen_idx, frame  (정렬 확인용, ego_cache 와 bitwise 일치해야 함)

검증 gate (스크립트 끝):
    fut5[:,:6]  == ego_cache.fut    bitwise   (순서·변환 동일 증명)
    fut5[:,9]   == ego_cache.goal   bitwise   (+50프레임 = +5.0s endpoint 일치)
    scen_idx/frame == ego_cache      bitwise
"""
import os
import sys
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

META_TRAIN = "/tmp/pm97/data/etri/meta_train"
OUT = "/tmp/pm97/data/etri/ego_cache_5s.npz"
REF = "/tmp/pm97/data/etri/ego_cache.npz"
FRAME_OFFSET = 50
MAIN_FRAMES = 300
TRAJ_STEP = 5
FUT_TS_5S = 10
DT = 0.1


def col(t, n):
    return np.asarray(t.column(n).to_pylist())


def rot(rpy):
    return Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()


def rel(xyz, rpy, o, q):
    return ((xyz[q] - xyz[o]) @ rot(rpy[o]))[..., :2]


def main():
    scen = sorted(os.listdir(META_TRAIN))
    n = len(scen) * MAIN_FRAMES
    fut5 = np.zeros((n, FUT_TS_5S, 2), np.float32)
    mask5 = np.zeros((n, FUT_TS_5S), np.float32)
    sidx = np.zeros(n, np.int32)
    frame = np.zeros(n, np.int16)

    k = 0
    for si, s in enumerate(scen):
        d = os.path.join(META_TRAIN, s)
        ts = pq.read_table(f"{d}/meta/timestamps.parquet")
        fid = col(ts, "frame_id").astype(int)
        tstamp = col(ts, "timestamp")
        ep = pq.read_table(f"{d}/annotation/ego_pose.parquet")
        tmap = dict(zip(tstamp, fid))
        ep_f = np.asarray([tmap[t] for t in col(ep, "timestamp")])
        order = np.argsort(ep_f)
        xyz = np.stack([col(ep, c)[order] for c in "xyz"], 1).astype(np.float64)
        rpy = np.stack([col(ep, c)[order] for c in ("roll", "pitch", "yaw")],
                       1).astype(np.float64)
        npose = xyz.shape[0]
        for f in range(MAIN_FRAMES):
            i = f + FRAME_OFFSET
            fut_ids = (f + TRAJ_STEP * np.arange(1, FUT_TS_5S + 1)) + FRAME_OFFSET
            valid = fut_ids < npose
            ids_c = np.clip(fut_ids, 0, npose - 1)
            fut5[k] = rel(xyz, rpy, i, ids_c)
            fut5[k][~valid] = 0.0
            mask5[k] = valid.astype(np.float32)
            sidx[k] = si
            frame[k] = f
            k += 1
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(scen)}", flush=True)

    np.savez_compressed(OUT, fut5=fut5, mask5=mask5, scen_idx=sidx, frame=frame,
                        scenarios=np.array(scen))
    print(f"저장 {OUT}  프레임 {k:,}")

    # ---- 검증 gate ----
    r = np.load(REF, allow_pickle=True)
    ok_order = np.array_equal(sidx, r["scen_idx"]) and np.array_equal(frame, r["frame"])
    ok_fut = np.array_equal(fut5[:, :6], r["fut"])
    # goal 은 valid 한 마지막 waypoint 인 행에서만 비교
    vgoal = mask5[:, 9] > 0
    ok_goal = np.array_equal(fut5[vgoal, 9], r["goal"][vgoal])
    print(f"GATE order(scen/frame)={ok_order}  fut[:6]==ego_cache.fut={ok_fut}  "
          f"fut5[:,9]==goal (valid only, n={int(vgoal.sum())})={ok_goal}")
    inv = int((mask5[:, 9] == 0).sum())
    print(f"5s tail invalid frames (near clip end): {inv:,} / {n:,} "
          f"({100*inv/n:.2f}%);  full-5s-valid frames = {n-inv:,}")
    print("ALL_GATES_PASS" if (ok_order and ok_fut and ok_goal) else "GATE_FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
