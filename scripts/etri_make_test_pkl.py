#!/usr/bin/env python
"""test 1125 clip → VAD info pkl (제출용).

공식 컨버터 `tools/data_converter/etri_vad_converter.py`는 **train 전용**이다
(`MAIN_FRAMES=300`, `annotation/` 하위를 읽는다). test clip은 구조가 다르다:

    meta_test/<clip>/calibration.parquet   6 카메라 K/distortion/euler/translation
    meta_test/<clip>/command.parquet       command(문자열) + vad_cmd(3,) one-hot
    meta_test/<clip>/ego_pose.parquet      frame -30..0 (10Hz) + **frame 50 = 5초 goal**

그래서 이 스크립트가 test 전용 pkl을 만든다. 좌표·행렬 규약은 컨버터 함수를
**그대로 import** 해서 쓴다(복붙하면 부호가 갈리고, 갈려도 손실은 정상으로 보인다).

앵커 프레임
-----------
0.5초 간격 7프레임 `[-30,-25,-20,-15,-10,-5,0]` = queue 7, 학습 cadence와 동일.
768 캐시(`etri_build_cache.py --split test --scale 0.4 --frame-mod 5`)가 정확히
이 7프레임만 갖고 있다. 제출 스크립트는 7프레임을 순차로 흘려 `prev_bev`를 쌓고
**마지막(frame 0) 출력만** 채택한다.

좌표계
------
test `ego_pose.parquet`는 이미 **frame 0 기준 상대 좌표**다(frame 0 = 0,0,0).
따라서 여기서 만드는 "ego2global"은 clip 로컬 좌표다. VAD는 큐 프레임 사이의
**상대 변위**만 쓰고(`use_shift`) clip마다 stream을 reset하므로 등가다.

GT 필드
-------
`nuscenes_vad_dataset.get_data_info`가 test_mode에서도 `get_ann_info`를 무조건
호출한다(원본에서 test_mode 가드가 주석 처리돼 있다: "now we load gt in test_mode
for evaluating"). 그래서 GT 키가 **존재는 해야** 한다. test에는 정답이 없으므로
agent 0개짜리 빈 배열로 채운다. 빈 GT 경로는 이미 가드가 들어가 있다.

모델이 실제로 읽지 않는 것
--------------------------
config 실측: `use_can_bus=False`, `ego_lcf_feat_idx=None`, `ego_his_encoder=None`.
→ 속도/가속도/yawrate/과거궤적이 모델에 들어가지 않는다. test는 frame 0 다음
프레임이 없어 컨버터의 중앙차분(index±1)을 쓸 수 없는데, 쓰이지 않으므로
무해하다. 그래도 값은 후방차분으로 채워 둔다(형식 유지).

    python scripts/etri_make_test_pkl.py --out /tmp/pm97/data/etri/pkl/etri_test1125_goal.pkl
"""
import argparse
import os
import os.path as osp
import pickle
import sys

import numpy as np
import pandas as pd

REPO = "/home/pm97/workspace/sukim/adcl/worktree/c0t_v2/ETRI_E2E_Driving_Challenge"
sys.path.insert(0, osp.join(REPO, "tools", "data_converter"))
from etri_vad_converter import (  # noqa: E402
    CAM_NAMES, euler_to_matrix, matrix_to_quat_wxyz, undistorted_intrinsic)

META_TEST = "/tmp/pm97/data/etri/meta_test"
CACHE = "/tmp/pm97/cache/etri_768_test"
EGO_CACHE_TEST = "/tmp/pm97/data/etri/ego_cache_test.npz"
FRAMES = [-30, -25, -20, -15, -10, -5, 0]   # 0.5 s cadence, queue 7
GOAL_FRAME = 50
FUT_TS, HIS_TS, TRAJ_STEP = 6, 2, 5
# clip 간 timestamp가 겹치면 mmdet3d의 timestamp 정렬이 clip을 뒤섞는다.
# clip마다 100초씩 띄우고 프레임은 0.5초 간격으로 둔다(단위 ms).
TS_BASE, TS_CLIP_GAP, TS_FRAME_GAP = 1_700_000_000_000.0, 100_000.0, 500.0


def load_cams(clip):
    calib = pd.read_parquet(osp.join(META_TEST, clip, "calibration.parquet"))
    calib = calib.set_index("camera_name")
    out = {}
    for cam in CAM_NAMES:
        row = calib.loc[cam]
        K = np.array(row["K"], dtype=np.float64).reshape(3, 3)
        dist = np.array(row["distortion"], dtype=np.float64)
        fisheye = bool(row["is_fisheye"])
        size = (int(row["image_width"]), int(row["image_height"]))
        c2e_R = euler_to_matrix(np.array(row["euler"], dtype=np.float64),
                                degrees=True)
        c2e_t = np.array(row["translation"], dtype=np.float64)
        out[cam] = dict(
            type=cam,
            sensor2ego_translation=c2e_t.tolist(),
            sensor2ego_rotation=matrix_to_quat_wxyz(c2e_R),
            sensor2lidar_rotation=c2e_R,
            sensor2lidar_translation=c2e_t,
            cam_intrinsic=undistorted_intrinsic(K, dist, size, fisheye),
            cam_intrinsic_raw=K,
            distortion=dist,
            is_fisheye=fisheye,
            image_width=size[0],
            image_height=size[1],
        )
    return out


def empty_gt(n=0):
    """agent 0개짜리 GT 자리채움. 키가 없으면 get_ann_info가 죽는다."""
    return dict(
        gt_boxes=np.zeros((n, 7), dtype=np.float64),
        gt_names=np.array([], dtype="<U3"),
        gt_velocity=np.zeros((n, 2), dtype=np.float64),
        num_lidar_pts=np.zeros(n, dtype=np.int64),
        num_radar_pts=np.zeros(n, dtype=np.int64),
        valid_flag=np.zeros(n, dtype=bool),
        gt_agent_fut_trajs=np.zeros((n, FUT_TS * 2), dtype=np.float32),
        gt_agent_fut_masks=np.zeros((n, FUT_TS), dtype=np.float32),
        gt_agent_fut_yaw=np.zeros((n, FUT_TS), dtype=np.float32),
        gt_agent_fut_goal=np.zeros(n, dtype=np.float32),
        gt_agent_lcf_feat=np.zeros((n, 9), dtype=np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ego-length", type=float, default=4.635)
    ap.add_argument("--ego-width", type=float, default=1.890)
    args = ap.parse_args()

    clips = sorted(d for d in os.listdir(META_TEST)
                   if osp.isdir(osp.join(META_TEST, d)))
    if args.limit:
        clips = clips[:args.limit]
    print(f"test clip {len(clips)}개")

    ref = np.load(EGO_CACHE_TEST, allow_pickle=True)
    ref_idx = {str(c): n for n, c in enumerate(ref["clips"])}
    ref_goal, ref_cmd = ref["goal"], ref["vad_cmd"]

    infos = []
    goal_err = []
    cmd_mismatch = 0
    for ci, clip in enumerate(clips):
        cams_base = load_cams(clip)
        pose = pd.read_parquet(osp.join(META_TEST, clip, "ego_pose.parquet"))
        pose = pose.sort_values("frame")
        fr = pose["frame"].to_numpy()
        xyz = pose[["x", "y", "z"]].to_numpy(dtype=np.float64)
        rpy = pose[["roll", "pitch", "yaw"]].to_numpy(dtype=np.float64)
        row_of = {int(f): n for n, f in enumerate(fr)}
        assert GOAL_FRAME in row_of, f"{clip}: frame {GOAL_FRAME} 없음"

        cmd = pd.read_parquet(osp.join(META_TEST, clip, "command.parquet"))
        vad_cmd = np.asarray(cmd["vad_cmd"].iloc[0], dtype=np.float32).reshape(3)

        if clip in ref_idx and int(vad_cmd.argmax()) != int(ref_cmd[ref_idx[clip]]):
            cmd_mismatch += 1

        for k, f in enumerate(FRAMES):
            r = row_of[f]
            R = euler_to_matrix(rpy[r])
            origin = xyz[r]

            def to_ego(frame_id):
                """frame_id 위치를 프레임 f의 ego 좌표로. 컨버터와 동일한 식."""
                return (xyz[row_of[frame_id]] - origin) @ R

            goal3 = to_ego(GOAL_FRAME)
            goal_yaw = float(np.arctan2(
                *(np.array([np.sin(rpy[row_of[GOAL_FRAME]][2] - rpy[r][2]),
                            np.cos(rpy[row_of[GOAL_FRAME]][2] - rpy[r][2])])[::-1])))

            his_ids = [max(f + TRAJ_STEP * j, int(fr.min()))
                       for j in range(-HIS_TS, 1)]
            his = np.stack([to_ego(h)[:2] for h in his_ids])
            his_trajs = np.diff(his, axis=0).astype(np.float32)

            # 속도/가속도: 모델이 안 읽는다(위 docstring). frame 0은 앞 프레임이
            # 없으므로 후방차분을 쓴다. 형식 유지 목적.
            i0 = r
            i1 = min(r + 1, row_of[0])
            im1 = max(r - 1, 0)
            dt = max((i1 - im1), 1) * 0.1
            vel = (xyz[i1] - xyz[im1]) / dt @ R
            acc = (xyz[i1] - 2 * xyz[i0] + xyz[im1]) / (dt / 2) ** 2 @ R
            yawrate = (rpy[i1] - rpy[im1]) / dt

            can_bus = np.zeros(18)
            can_bus[:3] = origin
            can_bus[3:7] = matrix_to_quat_wxyz(R)
            can_bus[7:10] = acc
            can_bus[10:13] = yawrate
            can_bus[13:16] = vel

            token = f"{clip}_{k:02d}"
            cams = {}
            for cam, base in cams_base.items():
                c = dict(base)
                c["data_path"] = osp.join(args.cache, clip, cam,
                                          f"frame_{f}.jpg")
                c["sample_data_token"] = f"{token}_{cam}"
                c["ego2global_translation"] = origin.tolist()
                c["ego2global_rotation"] = matrix_to_quat_wxyz(R)
                c["timestamp"] = TS_BASE + ci * TS_CLIP_GAP + k * TS_FRAME_GAP
                cams[cam] = c

            info = dict(
                token=token,
                scene_token=clip,
                map_location=clip,
                frame_idx=k,
                prev=f"{clip}_{k-1:02d}" if k > 0 else "",
                next=f"{clip}_{k+1:02d}" if k < len(FRAMES) - 1 else "",
                timestamp=TS_BASE + ci * TS_CLIP_GAP + k * TS_FRAME_GAP,
                lidar_path="",
                sweeps=[],
                cams=cams,
                ego2global_translation=origin.tolist(),
                ego2global_rotation=matrix_to_quat_wxyz(R),
                # ego == lidar 규약(컨버터와 동일: sensor2lidar = sensor2ego)
                lidar2ego_translation=[0.0, 0.0, 0.0],
                lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
                can_bus=can_bus,
                gt_ego_his_trajs=his_trajs,
                gt_ego_fut_trajs=np.zeros((FUT_TS, 2), dtype=np.float32),
                gt_ego_fut_masks=np.zeros(FUT_TS, dtype=np.float32),
                gt_ego_fut_cmd=vad_cmd,
                gt_ego_lcf_feat=np.array(
                    [vel[0], vel[1], acc[0], acc[1], yawrate[2],
                     args.ego_length, args.ego_width,
                     np.linalg.norm(vel[:2]), 0.0], dtype=np.float32),
                gt_ego_fut_goal=goal3[:2].astype(np.float32),
                gt_ego_fut_goal_yaw=np.float32(goal_yaw),
                ego_goal_valid=True,
                fut_valid_flag=False,   # test에는 GT 미래가 없다
            )
            info.update(empty_gt())
            infos.append(info)

            if f == 0 and clip in ref_idx:
                goal_err.append(np.abs(goal3[:2] - ref_goal[ref_idx[clip]]).max())

        if (ci + 1) % 200 == 0:
            print(f"  {ci+1}/{len(clips)}")

    # --- 교차검증: 독립적으로 만든 ego_cache_test 와 일치해야 한다 ---
    goal_err = np.array(goal_err)
    print(f"\ngoal 교차검증 (frame 0, ego_cache_test 대비) "
          f"n={len(goal_err)}  max|Δ|={goal_err.max():.6f} m  "
          f"p99={np.percentile(goal_err, 99):.6f} m")
    assert goal_err.max() < 1e-3, "goal 좌표 규약이 어긋났다"
    print(f"vad_cmd 불일치 {cmd_mismatch}건 (0이어야 한다)")
    assert cmd_mismatch == 0

    # --- 캐시 이미지 존재 확인 ---
    miss = [c["data_path"] for i in infos[:50] for c in i["cams"].values()
            if not osp.isfile(c["data_path"])]
    assert not miss, f"캐시 이미지 없음 예: {miss[:3]}"

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fp:
        pickle.dump(dict(infos=infos,
                         metadata=dict(version="etri-v1.0", map_lanes={},
                                       split="test", n_scenarios=len(clips),
                                       frame_stride=TRAJ_STEP)), fp)
    print(f"\n저장 {args.out}")
    print(f"  infos {len(infos):,}  = clip {len(clips)} × frame {len(FRAMES)}")
    print(f"  크기 {osp.getsize(args.out)/1e6:.1f} MB")
    print("  map_lanes 비움 — test에는 HD map이 없고, prepare_test_data 는 "
          "is_vis_on_test 가 False면 vectormap_pipeline 을 타지 않는다.")


if __name__ == "__main__":
    main()
