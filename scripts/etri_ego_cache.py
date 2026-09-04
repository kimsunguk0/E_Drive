#!/usr/bin/env python
"""train 376 시나리오 × 300 프레임의 ego 파생량을 npz 한 덩어리로 캐시.

왜 캐시하나: 이후 Phase C(분할·가중치)와 Phase D(prior 4종 + MLP)가 같은 파생량을
반복해서 필요로 한다. parquet를 매번 열면 실험 한 번마다 수 분이 날아가고, 더 나쁜 것은
좌표 변환을 여러 스크립트에 복붙하면서 부호가 어긋나는 것이다. 변환은 여기 한 곳에만
둔다.

좌표 변환은 주최측 컨버터와 동일 (etri_vad_converter.py:227):
    (p[q] - p[origin]) @ R(rpy[origin])          R = scipy 'xyz' 오일러(라디안)

산출 (프레임당):
    his   (31,2)  과거 -30..0 상대 위치 (누적, ego@f 기준)
    fut   (6,2)   미래 +5..+30 상대 위치 (누적) = GT
    goal  (2,)    +50 상대 위치
    speed, vel(2), accel(2), yawrate
    vad_cmd  0=right 1=left 2=straight   (컨버터 규칙: 미래 누적 횡변위 ±2 m)
    meta     0..5  주최측 6종 command

vad_cmd를 GT 미래에서 계산하는 이유: test의 `vad_cmd`도 주최측이 비공개 GT 미래에
같은 규칙을 적용해 만든 값이다. val을 test와 동형으로 만들려면 같은 방식이어야 한다.

    python scripts/etri_ego_cache.py
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

META_TRAIN = "/tmp/pm97/data/etri/meta_train"
META_TEST = "/tmp/pm97/data/etri/meta_test"
OUT = "/tmp/pm97/data/etri/ego_cache.npz"
OUT_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"

FRAME_OFFSET = 50
MAIN_FRAMES = 300
TRAJ_STEP = 5
FUT_TS = 6
HIS_FRAMES = 30
GOAL_FRAME = 50
DT = 0.1
META_LABELS = ["LANE_KEEP", "TURN_LEFT", "TURN_RIGHT",
               "LANE_CHANGE_L", "LANE_CHANGE_R", "U_TURN"]
META_IDX = {m: i for i, m in enumerate(META_LABELS)}


def col(t, n):
    return np.asarray(t.column(n).to_pylist())


def rot(rpy):
    return Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rel(xyz, rpy, o, q):
    """(p[q] - p[o]) @ R(rpy[o]) 의 xy. q는 스칼라 또는 배열 인덱스."""
    return ((xyz[q] - xyz[o]) @ rot(rpy[o]))[..., :2]


def vad_cmd_of(fut_cum):
    """컨버터와 동일: 미래 누적 횡변위(마지막 y)의 부호·크기로 3분류."""
    lat = fut_cum[-1, 1]
    if lat <= -2:
        return 0                      # right
    if lat >= 2:
        return 1                      # left
    return 2                          # straight


def build_train():
    scen = sorted(os.listdir(META_TRAIN))
    n = len(scen) * MAIN_FRAMES
    his = np.zeros((n, HIS_FRAMES + 1, 2), np.float32)
    his_yaw = np.zeros((n, HIS_FRAMES + 1), np.float32)
    fut = np.zeros((n, FUT_TS, 2), np.float32)
    goal = np.zeros((n, 2), np.float32)
    goal_yaw = np.zeros(n, np.float32)
    vel = np.zeros((n, 2), np.float32)
    acc = np.zeros((n, 2), np.float32)
    yawrate = np.zeros(n, np.float32)
    speed = np.zeros(n, np.float32)
    vcmd = np.zeros(n, np.int8)
    meta = np.full(n, -1, np.int8)
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
        # 정렬 후 index i == frame_id + FRAME_OFFSET (B.4에서 결손 0 확인됨)
        cm = pq.read_table(f"{d}/meta/command.parquet")
        cmap = dict(zip(col(cm, "timestamp"), col(cm, "command")))
        fr2ts = dict(zip(fid, tstamp))
        # timestamps.parquet의 timestamp는 ms 단위. dt는 컨버터와 같은 방식으로.
        tsec = np.zeros(len(fid) + 0)
        tsort = np.argsort(fid)
        tms = tstamp[tsort]

        for f in range(MAIN_FRAMES):
            i = f + FRAME_OFFSET
            his_ids = np.arange(f - HIS_FRAMES, f + 1) + FRAME_OFFSET
            fut_ids = (f + TRAJ_STEP * np.arange(1, FUT_TS + 1)) + FRAME_OFFSET
            his[k] = rel(xyz, rpy, i, his_ids)
            fut[k] = rel(xyz, rpy, i, fut_ids)
            gi = f + GOAL_FRAME + FRAME_OFFSET
            goal[k] = rel(xyz, rpy, i, gi)
            # 상대 yaw: 절대 yaw 차이를 [-pi, pi]로 감는다. 회전행렬을 거치지 않으므로
            # roll/pitch가 큰 구간에서는 근사지만, 차량 yaw는 이 근사로 충분하다.
            his_yaw[k] = wrap(rpy[his_ids, 2] - rpy[i, 2])
            goal_yaw[k] = wrap(rpy[gi, 2] - rpy[i, 2])
            dt = (tms[i + 1] - tms[i - 1]) / 1e3
            R = rot(rpy[i])
            v = (xyz[i + 1] - xyz[i - 1]) / dt @ R
            a = (xyz[i + 1] - 2 * xyz[i] + xyz[i - 1]) / (dt / 2) ** 2 @ R
            w = ((rpy[i + 1] - rpy[i - 1] + np.pi) % (2 * np.pi) - np.pi) / dt
            vel[k] = v[:2]
            acc[k] = a[:2]
            yawrate[k] = w[2]
            speed[k] = np.linalg.norm(v[:2])
            vcmd[k] = vad_cmd_of(fut[k])
            meta[k] = META_IDX.get(cmap.get(fr2ts[f], ""), -1)
            sidx[k] = si
            frame[k] = f
            k += 1
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(scen)}", flush=True)

    np.savez_compressed(
        OUT, scenarios=np.array(scen), his=his, his_yaw=his_yaw, fut=fut,
        goal=goal, goal_yaw=goal_yaw, vel=vel,
        acc=acc, yawrate=yawrate, speed=speed, vad_cmd=vcmd, meta=meta,
        scen_idx=sidx, frame=frame, meta_labels=np.array(META_LABELS))
    print(f"저장 {OUT}  프레임 {k:,}")


def build_test():
    clips = sorted(os.listdir(META_TEST))
    n = len(clips)
    his = np.zeros((n, HIS_FRAMES + 1, 2), np.float32)
    his_yaw = np.zeros((n, HIS_FRAMES + 1), np.float32)
    goal = np.zeros((n, 2), np.float32)
    goal_yaw = np.zeros(n, np.float32)
    vel = np.zeros((n, 2), np.float32)
    yawrate = np.zeros(n, np.float32)
    speed = np.zeros(n, np.float32)
    vcmd = np.zeros(n, np.int8)
    meta = np.full(n, -1, np.int8)

    for ci, c in enumerate(clips):
        d = os.path.join(META_TEST, c)
        ep = pq.read_table(f"{d}/ego_pose.parquet")
        fr = col(ep, "frame").astype(int)
        order = np.argsort(fr)
        fr_s = fr[order]
        xyz = np.stack([col(ep, k)[order] for k in "xyz"], 1).astype(np.float64)
        rpy = np.stack([col(ep, k)[order] for k in ("roll", "pitch", "yaw")],
                       1).astype(np.float64)
        idx = {f: i for i, f in enumerate(fr_s)}
        i0 = idx[0]
        his_ids = np.array([idx[f] for f in range(-HIS_FRAMES, 1)])
        his[ci] = rel(xyz, rpy, i0, his_ids)
        goal[ci] = rel(xyz, rpy, i0, idx[GOAL_FRAME])
        his_yaw[ci] = wrap(rpy[his_ids, 2] - rpy[i0, 2])
        goal_yaw[ci] = wrap(rpy[idx[GOAL_FRAME], 2] - rpy[i0, 2])
        # test는 frame +1이 없다 -> 후방차분 (컨버터 local_motion의 has_prev 분기)
        R = rot(rpy[i0])
        v = (xyz[i0] - xyz[idx[-1]]) / DT @ R
        w = ((rpy[i0] - rpy[idx[-1]] + np.pi) % (2 * np.pi) - np.pi) / DT
        vel[ci] = v[:2]
        yawrate[ci] = w[2]
        speed[ci] = np.linalg.norm(v[:2])
        cm = pq.read_table(f"{d}/command.parquet")
        vc = list(cm.column("vad_cmd").to_pylist()[0])
        # 제공된 one-hot을 그대로 신뢰한다. 우리 규칙으로 재계산할 GT 미래가 없다.
        vcmd[ci] = int(np.argmax(vc))
        meta[ci] = META_IDX.get(cm.column("command").to_pylist()[0], -1)

    np.savez_compressed(OUT_TEST, clips=np.array(clips), his=his,
                        his_yaw=his_yaw, goal_yaw=goal_yaw, goal=goal,
                        vel=vel, yawrate=yawrate, speed=speed, vad_cmd=vcmd,
                        meta=meta, meta_labels=np.array(META_LABELS))
    print(f"저장 {OUT_TEST}  clip {n:,}")


if __name__ == "__main__":
    print("test 캐시...", flush=True)
    build_test()
    print("train 캐시...", flush=True)
    build_train()
    sys.exit(0)
