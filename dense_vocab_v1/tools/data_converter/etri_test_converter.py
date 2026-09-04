import argparse
import os
from os import path as osp

import cv2
import mmcv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

CAM_NAMES = [
    'camera_front',
    'camera_front_right',
    'camera_front_left',
    'camera_rear_wide',
    'camera_rear_left',
    'camera_rear_right',
]
HIS_FRAMES = 30
STREAM_STRIDE = 5
STREAM_FRAMES = list(range(-HIS_FRAMES, 1, STREAM_STRIDE))
TRAJ_STEP = 5
HIS_TS = 2
FUT_TS = 6
DT = 0.1
EGO_LENGTH = 4.635
EGO_WIDTH = 1.890


def euler_to_matrix(euler, degrees=False):
    return Rotation.from_euler('xyz', euler, degrees=degrees).as_matrix()


def matrix_to_quat_wxyz(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return [w, x, y, z]


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def undistorted_intrinsic(intrinsic, distortion, image_size, is_fisheye):
    if is_fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intrinsic, distortion[:4], image_size, np.eye(3), balance=0.0)
    new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, distortion, image_size, alpha=0)
    return new_intrinsic


def load_camera_infos(clip_dir):
    calib = pd.read_parquet(osp.join(clip_dir, 'calibration.parquet'))
    calib = calib.set_index('camera_name')
    cam_infos = {}
    for cam_name in CAM_NAMES:
        row = calib.loc[cam_name]
        intrinsic = np.array(row['K'], dtype=np.float64).reshape(3, 3)
        distortion = np.array(row['distortion'], dtype=np.float64)
        is_fisheye = bool(row['is_fisheye'])
        image_size = (int(row['image_width']), int(row['image_height']))
        rotation = euler_to_matrix(
            np.array(row['euler'], dtype=np.float64), degrees=True)
        translation = np.array(row['translation'], dtype=np.float64)
        cam_infos[cam_name] = dict(
            type=cam_name,
            sensor2ego_translation=translation.tolist(),
            sensor2ego_rotation=matrix_to_quat_wxyz(rotation),
            sensor2lidar_rotation=rotation,
            sensor2lidar_translation=translation,
            cam_intrinsic=undistorted_intrinsic(
                intrinsic, distortion, image_size, is_fisheye),
            cam_intrinsic_raw=intrinsic,
            distortion=distortion,
            is_fisheye=is_fisheye,
            image_width=image_size[0],
            image_height=image_size[1],
        )
    return cam_infos


def load_ego_pose(clip_dir):
    ego = pd.read_parquet(osp.join(clip_dir, 'ego_pose.parquet'))
    ego = ego.set_index('frame')
    xyz, rpy = {}, {}
    for frame, row in ego.iterrows():
        xyz[int(frame)] = np.array([row['x'], row['y'], row['z']])
        rpy[int(frame)] = np.array([row['roll'], row['pitch'], row['yaw']])
    return xyz, rpy


def positions_in_frame(xyz, rpy, origin, query_frames):
    rotation = euler_to_matrix(rpy[origin])
    return np.array([(xyz[q] - xyz[origin]) @ rotation for q in query_frames])


def local_motion(xyz, rpy, frame):
    rotation = euler_to_matrix(rpy[frame])
    has_prev = (frame - 1) in xyz
    has_next = (frame + 1) in xyz
    if has_prev and has_next:
        dt = 2 * DT
        velocity = (xyz[frame + 1] - xyz[frame - 1]) / dt @ rotation
        accel = (xyz[frame + 1] - 2 * xyz[frame] +
                 xyz[frame - 1]) / DT ** 2 @ rotation
        rotation_rate = wrap_angle(rpy[frame + 1] - rpy[frame - 1]) / dt
    elif has_prev:
        velocity = (xyz[frame] - xyz[frame - 1]) / DT @ rotation
        accel = np.zeros(3)
        rotation_rate = wrap_angle(rpy[frame] - rpy[frame - 1]) / DT
    else:
        velocity = (xyz[frame + 1] - xyz[frame]) / DT @ rotation
        accel = np.zeros(3)
        rotation_rate = wrap_angle(rpy[frame + 1] - rpy[frame]) / DT
    return velocity, accel, rotation_rate


def ego_annotations(xyz, rpy, command):
    his_ids = [TRAJ_STEP * k for k in range(-HIS_TS, 1)]
    his = positions_in_frame(xyz, rpy, 0, his_ids)
    his_trajs = np.diff(his[:, :2], axis=0).astype(np.float32)
    velocity, accel, rotation_rate = local_motion(xyz, rpy, 0)
    ego_lcf_feat = np.array([
        velocity[0], velocity[1], accel[0], accel[1], rotation_rate[2],
        EGO_LENGTH, EGO_WIDTH, np.linalg.norm(velocity[:2]), 0.0
    ], dtype=np.float32)
    return dict(
        gt_ego_his_trajs=his_trajs,
        gt_ego_fut_trajs=np.zeros((FUT_TS, 2), dtype=np.float32),
        gt_ego_fut_masks=np.zeros(FUT_TS, dtype=np.float32),
        gt_ego_fut_cmd=command.astype(np.float32),
        gt_ego_lcf_feat=ego_lcf_feat,
    )


def empty_agent_annotations():
    return dict(
        gt_boxes=np.zeros((0, 7)),
        gt_names=np.zeros(0, dtype='<U16'),
        gt_velocity=np.zeros((0, 2)),
        num_lidar_pts=np.zeros(0, dtype=np.int64),
        num_radar_pts=np.zeros(0, dtype=np.int64),
        valid_flag=np.zeros(0, dtype=bool),
        gt_agent_fut_trajs=np.zeros((0, FUT_TS * 2), dtype=np.float32),
        gt_agent_fut_masks=np.zeros((0, FUT_TS), dtype=np.float32),
        gt_agent_lcf_feat=np.zeros((0, 9), dtype=np.float32),
        gt_agent_fut_yaw=np.zeros((0, FUT_TS), dtype=np.float32),
        gt_agent_fut_goal=np.zeros(0, dtype=np.float32),
    )


def convert_clip(clip_dir, clip_token):
    cam_infos = load_camera_infos(clip_dir)
    xyz, rpy = load_ego_pose(clip_dir)
    command = np.array(
        pd.read_parquet(osp.join(clip_dir, 'command.parquet')).iloc[0]['vad_cmd'])
    ego_anns = ego_annotations(xyz, rpy, command)

    infos = []
    for order, frame in enumerate(STREAM_FRAMES):
        rotation = euler_to_matrix(rpy[frame])
        quat = matrix_to_quat_wxyz(rotation)
        translation = xyz[frame].tolist()
        timestamp = int(round((frame + HIS_FRAMES) * DT * 1e3))
        velocity, accel, rotation_rate = local_motion(xyz, rpy, frame)

        can_bus = np.zeros(18)
        can_bus[:3] = xyz[frame]
        can_bus[3:7] = quat
        can_bus[7:10] = accel
        can_bus[10:13] = rotation_rate
        can_bus[13:16] = velocity

        cams = {}
        for cam_name in CAM_NAMES:
            cam_info = dict(cam_infos[cam_name])
            cam_info.update(
                data_path=osp.join(clip_dir, cam_name, f'frame_{frame}.jpg'),
                sample_data_token=f'{clip_token}_{frame}_{cam_name}',
                ego2global_translation=translation,
                ego2global_rotation=quat,
                timestamp=timestamp,
            )
            cams[cam_name] = cam_info

        info = dict(
            lidar_path='',
            token=f'{clip_token}_{frame}',
            prev=f'{clip_token}_{STREAM_FRAMES[order - 1]}' if order > 0 else '',
            next=f'{clip_token}_{STREAM_FRAMES[order + 1]}'
            if order < len(STREAM_FRAMES) - 1 else '',
            can_bus=can_bus,
            frame_idx=order,
            sweeps=[],
            cams=cams,
            scene_token=clip_token,
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
            ego2global_translation=translation,
            ego2global_rotation=quat,
            timestamp=timestamp,
            fut_valid_flag=True,
            map_location=clip_token,
        )
        info.update(empty_agent_annotations())
        info.update(ego_anns)
        infos.append(info)
    return infos


def create_test_infos(test_root, out_dir, info_prefix):
    clip_tokens = sorted(
        name for name in os.listdir(test_root)
        if osp.isfile(osp.join(test_root, name, 'ego_pose.parquet')))
    if not clip_tokens:
        raise FileNotFoundError(f'no clip found under {test_root}')
    print(f'{len(clip_tokens)} clips found')

    infos = []
    for clip_token in mmcv.track_iter_progress(clip_tokens):
        infos.extend(convert_clip(osp.join(test_root, clip_token), clip_token))

    print(f'test samples: {len(infos)}')
    mmcv.mkdir_or_exist(out_dir)
    info_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_test.pkl')
    mmcv.dump(
        dict(infos=infos, metadata=dict(version='etri-test-v1.0')), info_path)
    print(f'wrote {info_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='ETRI challenge test converter')
    parser.add_argument('--test-root', type=str, default='./data/etri/test')
    parser.add_argument('--out-dir', type=str, default='./data/etri')
    parser.add_argument('--extra-tag', type=str, default='vad_etri')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_test_infos(args.test_root, args.out_dir, args.extra_tag)
