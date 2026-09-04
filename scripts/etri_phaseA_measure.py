#!/usr/bin/env python
"""Phase A -- ETRI 챌린지 데이터 구조·스키마 전수 실측.

INTEL.md는 코드만 읽고 쓴 정적 분석이라 '미검증' 항목이 남아 있었다. 이 스크립트는
그 항목을 실제 데이터로 전수 측정한다. 이미지 픽셀은 거의 건드리지 않고 parquet만
읽으므로 (train 493 MB, test 23 MB) 몇 분 안에 끝난다.

전수인 이유: 샘플 몇 개로 "6캠 중 rear_wide만 fisheye"를 확인하면, 예외가 한 시나리오만
있어도 컨버터가 런타임에 죽는다. 스키마 가정은 전수로 깨거나 확정해야 한다.

좌표 변환은 주최측 컨버터와 **동일 규칙**을 쓴다 (etri_vad_converter.py:227
ego_positions_in_frame). 우리가 따로 구현하면 부호·전치 하나로 결론이 뒤집히므로
(positions - origin) @ R 형태를 그대로 옮겼다. R은 scipy 'xyz' 오일러(라디안).

    python scripts/etri_phaseA_measure.py
    python scripts/etri_phaseA_measure.py --goal-sample 40   # goal 통계 시나리오 수
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

META_TRAIN = "/tmp/pm97/data/etri/meta_train"
META_TEST = "/tmp/pm97/data/etri/meta_test"
PERSIST = "/home/pm97/workspace/sukim/adcl"
OUT_JSON = os.path.join(PERSIST, "logs", "etri_phaseA.json")

# 주최측 컨버터 상수 (etri_vad_converter.py:20-26)
FRAME_OFFSET = 50
MAIN_FRAMES = 300
MIN_FRAME = -FRAME_OFFSET
MAX_FRAME = FRAME_OFFSET + MAIN_FRAMES - 1      # 349
TRAJ_STEP = 5
FUT_TS = 6
HIS_TS = 2
GOAL_FRAME = 50                                  # test가 제공하는 +50 목표점
CAM_NAMES = ["camera_front", "camera_front_right", "camera_front_left",
             "camera_rear_wide", "camera_rear_left", "camera_rear_right"]
DT = 0.1


def col(t, name):
    return t.column(name).to_pylist()


def read(path):
    return pq.read_table(path)


def euler_to_matrix(rpy):
    return Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()


def positions_in_frame(xyz, rpy, origin_idx, query_idx):
    """컨버터와 동일: (p - origin) @ R(origin). world -> ego(origin) 변환."""
    R = euler_to_matrix(rpy[origin_idx])
    return (xyz[query_idx] - xyz[origin_idx]) @ R


# ---------------------------------------------------------------- A-1,2,3,4,7
def measure_train(goal_sample):
    scen = sorted(os.listdir(META_TRAIN))
    res = {"n_scenarios": len(scen)}

    calib_bad, fisheye_map = [], Counter()
    kdims, ddims, sizes = Counter(), Counter(), Counter()
    no_numpoints, ego_gap, frame_range = [], [], Counter()
    cmd_frames = Counter()               # A-6 train: 프레임별
    cmd_per_scen = Counter()             # 시나리오 대표값(첫 프레임)
    obj_classes = Counter()
    rows_ego, rows_ts, rows_cmd = Counter(), Counter(), Counter()
    goal_rows = []
    fut_invalid = 0
    total_frames = 0

    rng = np.random.default_rng(42)
    goal_pick = set(rng.choice(len(scen), min(goal_sample, len(scen)),
                               replace=False).tolist())

    for si, s in enumerate(scen):
        d = os.path.join(META_TRAIN, s)

        # --- A-1 calibration 전수
        c = read(f"{d}/calibration/calibration.parquet").to_pydict()
        names = c["camera_name"]
        if sorted(names) != sorted(CAM_NAMES):
            calib_bad.append((s, "camera set", sorted(names)))
        for i, n in enumerate(names):
            kdims[len(c["K"][i])] += 1
            ddims[(n if n == "camera_rear_wide" else "others",
                   len(c["distortion"][i]))] += 1
            sizes[(c["image_width"][i], c["image_height"][i])] += 1
            if c["is_fisheye"][i]:
                fisheye_map[n] += 1

        # --- A-3 num_points
        obj = read(f"{d}/annotation/object.parquet")
        if "num_points" not in obj.schema.names:
            no_numpoints.append(s)
        obj_classes.update(col(obj, "class"))

        # --- A-4 frame_id 범위, ego_pose 연속성
        ts = read(f"{d}/meta/timestamps.parquet")
        fid = np.asarray(col(ts, "frame_id"))
        frame_range[(int(fid.min()), int(fid.max()), len(fid))] += 1
        rows_ts[ts.num_rows] += 1

        ep = read(f"{d}/annotation/ego_pose.parquet")
        rows_ego[ep.num_rows] += 1
        # ego_pose는 frame_id 컬럼이 없다 -- timestamp로 join해야 한다
        # (컨버터 etri_vad_converter.py:88과 동일).
        tmap = {t: f for t, f in zip(col(ts, "timestamp"), col(ts, "frame_id"))}
        ep_f = np.asarray([tmap.get(t, np.nan) for t in col(ep, "timestamp")],
                          dtype=float)
        if np.isnan(ep_f).any():
            ego_gap.append((s, "timestamp join 실패", int(np.isnan(ep_f).sum())))
        else:
            got = set(ep_f.astype(int).tolist())
            want = set(range(MIN_FRAME, MAX_FRAME + 1))
            if got != want:
                ego_gap.append((s, "frame 결손",
                                sorted(want - got)[:5], len(want - got)))

        rows_cmd[read(f"{d}/meta/command.parquet").num_rows] += 1
        cm = read(f"{d}/meta/command.parquet")
        cmt = {t: v for t, v in zip(col(cm, "timestamp"), col(cm, "command"))}
        # A-6: train은 프레임별. 메인 구간(0..299)만 센다 -- 학습에 쓰이는 구간.
        main_ts = [t for t, f in zip(col(ts, "timestamp"), col(ts, "frame_id"))
                   if 0 <= f < MAIN_FRAMES]
        vals = [cmt.get(t) for t in main_ts]
        cmd_frames.update([v for v in vals if v is not None])
        if vals:
            cmd_per_scen[vals[0]] += 1
        total_frames += len(main_ts)

        # --- A-4 fut_valid, A-goal
        if np.isnan(ep_f).any():
            continue
        order = np.argsort(ep_f)
        xyz = np.stack([np.asarray(col(ep, k))[order] for k in "xyz"], axis=1)
        rpy = np.stack([np.asarray(col(ep, k))[order]
                        for k in ("roll", "pitch", "yaw")], axis=1)
        # 정렬 후 index i == frame_id + FRAME_OFFSET
        for f in range(MAIN_FRAMES):
            if f + FUT_TS * TRAJ_STEP > MAX_FRAME:
                fut_invalid += 1

        if si in goal_pick:
            for f in range(0, MAIN_FRAMES, 10):
                i = f + FRAME_OFFSET
                gi = f + GOAL_FRAME + FRAME_OFFSET
                if gi > MAX_FRAME + FRAME_OFFSET:
                    continue
                goal = positions_in_frame(xyz, rpy, i, gi)[:2]
                v = (xyz[i + 1] - xyz[i - 1]) / (2 * DT) @ euler_to_matrix(rpy[i])
                speed = float(np.linalg.norm(v[:2]))
                cv = np.array([speed * GOAL_FRAME * DT, 0.0])
                goal_rows.append(dict(
                    scenario=s, frame=f, gx=float(goal[0]), gy=float(goal[1]),
                    dist=float(np.linalg.norm(goal)), speed=speed,
                    cv_resid=float(np.linalg.norm(goal - cv)),
                    cmd=cmt.get(main_ts[f]) if f < len(main_ts) else None))

    res.update(
        calib_anomalies=calib_bad,
        fisheye_cameras=dict(fisheye_map),
        K_dims=dict(kdims),
        dist_dims={f"{k[0]}:{k[1]}": v for k, v in ddims.items()},
        image_sizes={f"{w}x{h}": n for (w, h), n in sizes.items()},
        object_missing_num_points=no_numpoints,
        object_class_counts=dict(obj_classes),
        frame_id_ranges={f"{a}..{b} (n={n})": c
                         for (a, b, n), c in frame_range.items()},
        ego_pose_row_counts=dict(rows_ego),
        timestamps_row_counts=dict(rows_ts),
        command_row_counts=dict(rows_cmd),
        ego_pose_gaps=ego_gap,
        fut_invalid_frames=fut_invalid,
        main_frames_total=total_frames,
        command_dist_frames=dict(cmd_frames),
        command_dist_scenarios=dict(cmd_per_scen),
    )
    return res, goal_rows


# ------------------------------------------------------------------ A-5,6 test
def measure_test():
    clips = sorted(os.listdir(META_TEST))
    res = {"n_clips": len(clips)}
    have_goal, frames_sets = 0, Counter()
    cmd, vad = Counter(), Counter()
    kdims, ddims, sizes, fisheye = Counter(), Counter(), Counter(), Counter()
    goal_rows, bad = [], []

    for t in clips:
        d = os.path.join(META_TEST, t)
        ep = read(f"{d}/ego_pose.parquet")
        fr = sorted(col(ep, "frame"))
        frames_sets[tuple(fr)] += 1
        if GOAL_FRAME in fr:
            have_goal += 1

        cm = read(f"{d}/command.parquet")
        cmd[col(cm, "command")[0]] += 1
        v = col(cm, "vad_cmd")[0]
        vad[tuple(v)] += 1

        c = read(f"{d}/calibration.parquet").to_pydict()
        for i, n in enumerate(c["camera_name"]):
            kdims[len(c["K"][i])] += 1
            ddims[(n if n == "camera_rear_wide" else "others",
                   len(c["distortion"][i]))] += 1
            sizes[(c["image_width"][i], c["image_height"][i])] += 1
            if c["is_fisheye"][i]:
                fisheye[n] += 1

        # goal 통계: origin = frame 0
        idx = {f: i for i, f in enumerate(col(ep, "frame"))}
        if 0 not in idx or GOAL_FRAME not in idx or -1 not in idx:
            bad.append(t)
            continue
        xyz = np.stack([np.asarray(col(ep, k)) for k in "xyz"], axis=1)
        rpy = np.stack([np.asarray(col(ep, k))
                        for k in ("roll", "pitch", "yaw")], axis=1)
        i0, ig, im = idx[0], idx[GOAL_FRAME], idx[-1]
        goal = positions_in_frame(xyz, rpy, i0, ig)[:2]
        # test는 frame +1이 없으므로 후방차분으로 속도를 잡는다
        # (컨버터 local_motion의 elif has_prev 분기와 동일).
        v3 = (xyz[i0] - xyz[im]) / DT @ euler_to_matrix(rpy[i0])
        speed = float(np.linalg.norm(v3[:2]))
        cvp = np.array([speed * GOAL_FRAME * DT, 0.0])
        goal_rows.append(dict(clip=t, gx=float(goal[0]), gy=float(goal[1]),
                              dist=float(np.linalg.norm(goal)), speed=speed,
                              cv_resid=float(np.linalg.norm(goal - cvp)),
                              cmd=col(cm, "command")[0]))

    res.update(
        clips_with_goal_frame=have_goal,
        frame_layouts={f"n={len(k)} min={min(k)} max={max(k)}": v
                       for k, v in frames_sets.items()},
        command_dist=dict(cmd),
        vad_cmd_dist={str(list(k)): v for k, v in vad.items()},
        K_dims=dict(kdims),
        dist_dims={f"{k[0]}:{k[1]}": v for k, v in ddims.items()},
        image_sizes={f"{w}x{h}": n for (w, h), n in sizes.items()},
        fisheye_cameras=dict(fisheye),
        malformed=bad,
    )
    return res, goal_rows


def stats(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {"n": int(len(a)), "mean": round(float(a.mean()), 3),
            "std": round(float(a.std()), 3), "p05": round(float(np.percentile(a, 5)), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-sample", type=int, default=40)
    args = ap.parse_args()

    print("train 측정 중...", flush=True)
    tr, tr_goal = measure_train(args.goal_sample)
    print("test 측정 중...", flush=True)
    te, te_goal = measure_test()

    out = {"train": tr, "test": te,
           "goal_train": {k: stats(tr_goal, k)
                          for k in ("dist", "gx", "gy", "speed", "cv_resid")},
           "goal_test": {k: stats(te_goal, k)
                         for k in ("dist", "gx", "gy", "speed", "cv_resid")}}
    # command별 goal 분해 -- prior 설계에 직접 쓰인다
    for tag, rows in (("goal_train_by_cmd", tr_goal), ("goal_test_by_cmd", te_goal)):
        by = defaultdict(list)
        for r in rows:
            by[r["cmd"] or "?"].append(r)
        out[tag] = {c: {"n": len(v), "dist": stats(v, "dist")["p50"],
                        "gy_p05": stats(v, "gy")["p05"],
                        "gy_p50": stats(v, "gy")["p50"],
                        "gy_p95": stats(v, "gy")["p95"],
                        "cv_resid_p50": stats(v, "cv_resid")["p50"]}
                    for c, v in sorted(by.items())}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n저장: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
