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
CLASS_NAMES = ('Car', 'Pedestrian', 'Cyclist')
FRAME_OFFSET = 50
MAIN_FRAMES = 300
MIN_FRAME = -FRAME_OFFSET
MAX_FRAME = FRAME_OFFSET + MAIN_FRAMES - 1
TRAJ_STEP = 5
FUT_TS = 6
HIS_TS = 2


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def euler_to_matrix(euler, degrees=False):
    return Rotation.from_euler('xyz', euler, degrees=degrees).as_matrix()


def matrix_to_quat_wxyz(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return [w, x, y, z]


def undistorted_intrinsic(intrinsic, distortion, image_size, is_fisheye):
    if is_fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intrinsic, distortion[:4], image_size, np.eye(3), balance=0.0)
    new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, distortion, image_size, alpha=0)
    return new_intrinsic


def load_camera_infos(scenario_dir):
    calib = pd.read_parquet(
        osp.join(scenario_dir, 'calibration', 'calibration.parquet'))
    calib = calib.set_index('camera_name')
    cam_infos = {}
    for cam_name in CAM_NAMES:
        row = calib.loc[cam_name]
        intrinsic = np.array(row['K'], dtype=np.float64).reshape(3, 3)
        distortion = np.array(row['distortion'], dtype=np.float64)
        is_fisheye = bool(row['is_fisheye'])
        image_size = (int(row['image_width']), int(row['image_height']))
        cam2ego_rotation = euler_to_matrix(
            np.array(row['euler'], dtype=np.float64), degrees=True)
        cam2ego_translation = np.array(row['translation'], dtype=np.float64)
        cam_infos[cam_name] = dict(
            type=cam_name,
            sensor2ego_translation=cam2ego_translation.tolist(),
            sensor2ego_rotation=matrix_to_quat_wxyz(cam2ego_rotation),
            sensor2lidar_rotation=cam2ego_rotation,
            sensor2lidar_translation=cam2ego_translation,
            cam_intrinsic=undistorted_intrinsic(
                intrinsic, distortion, image_size, is_fisheye),
            cam_intrinsic_raw=intrinsic,
            distortion=distortion,
            is_fisheye=is_fisheye,
            image_width=image_size[0],
            image_height=image_size[1],
        )
    return cam_infos


def load_scenario(scenario_dir):
    frames = pd.read_parquet(
        osp.join(scenario_dir, 'meta', 'timestamps.parquet'))
    frames = frames[['timestamp', 'frame_id']]
    ego_pose = pd.read_parquet(
        osp.join(scenario_dir, 'annotation', 'ego_pose.parquet'))
    ego_pose = ego_pose.merge(frames, on='timestamp').sort_values('frame_id')
    objects = pd.read_parquet(
        osp.join(scenario_dir, 'annotation', 'object.parquet'))
    objects = objects.merge(frames, on='timestamp').sort_values('frame_id')
    if 'num_points' not in objects.columns:
        raise ValueError(
            f'{scenario_dir}: object.parquet has no num_points column')

    timestamps = np.full(MAX_FRAME - MIN_FRAME + 1, np.nan)
    timestamps[frames['frame_id'].to_numpy() + FRAME_OFFSET] = \
        frames['timestamp'].to_numpy()

    ego_pose_xyz = np.full((len(timestamps), 3), np.nan)
    ego_pose_rpy = np.full((len(timestamps), 3), np.nan)
    pose_index = ego_pose['frame_id'].to_numpy() + FRAME_OFFSET
    ego_pose_xyz[pose_index] = ego_pose[['x', 'y', 'z']].to_numpy()
    ego_pose_rpy[pose_index] = ego_pose[['roll', 'pitch', 'yaw']].to_numpy()

    ego_rows = objects[objects['class'] == 'ego']
    ego_anno = np.full((len(timestamps), 4), np.nan)
    ego_index = ego_rows['frame_id'].to_numpy() + FRAME_OFFSET
    ego_anno[ego_index] = ego_rows[
        ['x[m]', 'y[m]', 'z[m]', 'heading[rad]']].to_numpy()

    object_rows = objects[objects['class'] != 'ego'].copy()
    object_rows['obj_id'] = object_rows['obj_id'].astype(np.int64)
    frame_objects = {
        frame_id: group
        for frame_id, group in object_rows.groupby('frame_id')
    }
    tracks = {
        obj_id: dict(
            zip(group['frame_id'],
                group[['x[m]', 'y[m]', 'heading[rad]']].to_numpy()))
        for obj_id, group in object_rows.groupby('obj_id')
    }
    return dict(
        timestamps=timestamps,
        ego_pose_xyz=ego_pose_xyz,
        ego_pose_rpy=ego_pose_rpy,
        ego_anno=ego_anno,
        frame_objects=frame_objects,
        tracks=tracks,
    )


def yaw_rotation_2d(yaw):
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([[cos, -sin], [sin, cos]])


def track_velocity(data, obj_id, frame_id):
    track = data['tracks'][obj_id]
    prev_id = frame_id - 1 if frame_id - 1 in track else frame_id
    next_id = frame_id + 1 if frame_id + 1 in track else frame_id
    if prev_id == next_id:
        return np.zeros(2)
    dt = (data['timestamps'][next_id + FRAME_OFFSET] -
          data['timestamps'][prev_id + FRAME_OFFSET]) / 1e3
    return (track[next_id][:2] - track[prev_id][:2]) / dt


def agent_annotations(data, frame_id):
    group = data['frame_objects'].get(frame_id)
    num_box = 0 if group is None else len(group)
    anns = dict(
        gt_boxes=np.zeros((num_box, 7)),
        gt_names=np.zeros(num_box, dtype='<U16'),
        gt_velocity=np.zeros((num_box, 2)),
        num_lidar_pts=np.zeros(num_box, dtype=np.int64),
        num_radar_pts=np.zeros(num_box, dtype=np.int64),
        valid_flag=np.zeros(num_box, dtype=bool),
        gt_agent_fut_trajs=np.zeros((num_box, FUT_TS * 2), dtype=np.float32),
        gt_agent_fut_masks=np.zeros((num_box, FUT_TS), dtype=np.float32),
        gt_agent_lcf_feat=np.zeros((num_box, 9), dtype=np.float32),
        gt_agent_fut_yaw=np.zeros((num_box, FUT_TS), dtype=np.float32),
        gt_agent_fut_goal=np.zeros(num_box, dtype=np.float32),
    )
    if num_box == 0:
        return anns

    ego_x, ego_y, ego_z, ego_heading = data['ego_anno'][frame_id +
                                                        FRAME_OFFSET]
    to_ego = yaw_rotation_2d(-ego_heading)

    xyz = group[['x[m]', 'y[m]', 'z[m]']].to_numpy()
    heading = group['heading[rad]'].to_numpy()
    wlh = group[['length[m]', 'width[m]', 'height[m]']].to_numpy()
    obj_ids = group['obj_id'].to_numpy()
    names = group['class'].to_numpy().astype(str)
    num_points = group['num_points'].to_numpy().astype(np.int64)

    centers = np.column_stack([
        (xyz[:, :2] - [ego_x, ego_y]) @ to_ego.T,
        xyz[:, 2] + wlh[:, 2] / 2 - ego_z,
    ])
    yaw_ego = wrap_angle(heading - ego_heading)
    anns['gt_boxes'] = np.column_stack(
        [centers, wlh, wrap_angle(-yaw_ego - np.pi / 2)])
    anns['gt_names'] = names
    anns['num_lidar_pts'] = num_points
    anns['valid_flag'] = num_points > 0

    fut_trajs = np.zeros((num_box, FUT_TS, 2), dtype=np.float32)
    for i, obj_id in enumerate(obj_ids):
        track = data['tracks'][obj_id]
        velocity = to_ego @ track_velocity(data, obj_id, frame_id)
        anns['gt_velocity'][i] = velocity

        label = CLASS_NAMES.index(names[i]) if names[i] in CLASS_NAMES else -1
        anns['gt_agent_lcf_feat'][i] = [
            centers[i, 0], centers[i, 1], yaw_ego[i], velocity[0],
            velocity[1], wlh[i, 0], wlh[i, 1], wlh[i, 2], label
        ]

        prev_xy, prev_heading = xyz[i, :2], heading[i]
        for j in range(FUT_TS):
            fut_id = frame_id + (j + 1) * TRAJ_STEP
            if fut_id not in track:
                break
            fut_x, fut_y, fut_heading = track[fut_id]
            fut_trajs[i, j] = to_ego @ ([fut_x, fut_y] - prev_xy)
            anns['gt_agent_fut_yaw'][i, j] = wrap_angle(fut_heading -
                                                        prev_heading)
            anns['gt_agent_fut_masks'][i, j] = 1
            prev_xy, prev_heading = np.array([fut_x, fut_y]), fut_heading

        fut_coords = np.cumsum(fut_trajs[i], axis=0)
        coord_diff = fut_coords[-1] - fut_coords[0]
        if coord_diff.max() < 1.0:
            anns['gt_agent_fut_goal'][i] = 9
        else:
            motion_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
            anns['gt_agent_fut_goal'][i] = motion_yaw // (np.pi / 4)

    anns['gt_agent_fut_trajs'] = fut_trajs.reshape(num_box, FUT_TS * 2)
    return anns


def ego_positions_in_frame(data, frame_id, frame_ids):
    rotation = euler_to_matrix(data['ego_pose_rpy'][frame_id + FRAME_OFFSET])
    origin = data['ego_pose_xyz'][frame_id + FRAME_OFFSET]
    clipped = np.clip(np.asarray(frame_ids), MIN_FRAME, MAX_FRAME)
    positions = data['ego_pose_xyz'][clipped + FRAME_OFFSET]
    return (positions - origin) @ rotation, clipped == np.asarray(frame_ids)


def ego_annotations(data, frame_id, ego_length, ego_width):
    his_ids = frame_id + TRAJ_STEP * np.arange(-HIS_TS, 1)
    his_trajs, _ = ego_positions_in_frame(data, frame_id, his_ids)
    his_trajs = np.diff(his_trajs[:, :2], axis=0)

    fut_ids = frame_id + TRAJ_STEP * np.arange(FUT_TS + 1)
    fut_trajs, fut_in_range = ego_positions_in_frame(data, frame_id, fut_ids)
    fut_trajs = np.diff(fut_trajs[:, :2], axis=0)
    fut_masks = fut_in_range[1:]
    fut_valid_flag = bool(fut_in_range.all())

    lateral = fut_trajs.sum(axis=0)[1]
    if lateral <= -2:
        command = np.array([1, 0, 0])
    elif lateral >= 2:
        command = np.array([0, 1, 0])
    else:
        command = np.array([0, 0, 1])

    index = frame_id + FRAME_OFFSET
    rotation = euler_to_matrix(data['ego_pose_rpy'][index])
    dt = (data['timestamps'][index + 1] - data['timestamps'][index - 1]) / 1e3
    velocity = (data['ego_pose_xyz'][index + 1] -
                data['ego_pose_xyz'][index - 1]) / dt @ rotation
    accel = (data['ego_pose_xyz'][index + 1] - 2 * data['ego_pose_xyz'][index]
             + data['ego_pose_xyz'][index - 1]) / (dt / 2)**2 @ rotation
    rotation_rate = wrap_angle(data['ego_pose_rpy'][index + 1] -
                               data['ego_pose_rpy'][index - 1]) / dt

    can_bus = np.zeros(18)
    can_bus[:3] = data['ego_pose_xyz'][index]
    can_bus[3:7] = matrix_to_quat_wxyz(rotation)
    can_bus[7:10] = accel
    can_bus[10:13] = rotation_rate
    can_bus[13:16] = velocity

    ego_lcf_feat = np.array([
        velocity[0], velocity[1], accel[0], accel[1], rotation_rate[2],
        ego_length, ego_width,
        np.linalg.norm(velocity[:2]), 0.0
    ])
    return dict(
        gt_ego_his_trajs=his_trajs.astype(np.float32),
        gt_ego_fut_trajs=fut_trajs.astype(np.float32),
        gt_ego_fut_masks=fut_masks.astype(np.float32),
        gt_ego_fut_cmd=command.astype(np.float32),
        gt_ego_lcf_feat=ego_lcf_feat.astype(np.float32),
    ), can_bus, fut_valid_flag


def load_lanes(scenario_dir):
    lanes = pd.read_parquet(
        osp.join(scenario_dir, 'annotation', 'map.parquet'))
    return [
        np.asarray([np.asarray(pt, dtype=np.float64) for pt in points])
        for points in lanes['points']
    ]


def convert_scenario(scenario_dir, ego_length, ego_width):
    scenario = osp.basename(osp.normpath(scenario_dir))
    data = load_scenario(scenario_dir)
    cam_infos = load_camera_infos(scenario_dir)
    lanes = load_lanes(scenario_dir)

    infos = []
    for frame_id in range(MAIN_FRAMES):
        index = frame_id + FRAME_OFFSET
        timestamp = data['timestamps'][index]
        ego2global_rotation = matrix_to_quat_wxyz(
            euler_to_matrix(data['ego_pose_rpy'][index]))
        ego2global_translation = data['ego_pose_xyz'][index].tolist()

        cams = {}
        for cam_name in CAM_NAMES:
            cam_info = dict(cam_infos[cam_name])
            cam_info.update(
                data_path=osp.join(scenario_dir, cam_name,
                                   f'{frame_id:08d}.jpg'),
                sample_data_token=f'{scenario}_{frame_id:08d}_{cam_name}',
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                timestamp=timestamp,
            )
            cams[cam_name] = cam_info

        ego_anns, can_bus, fut_valid_flag = ego_annotations(
            data, frame_id, ego_length, ego_width)

        info = dict(
            lidar_path='',
            token=f'{scenario}_{frame_id:08d}',
            prev=f'{scenario}_{frame_id - 1:08d}' if frame_id > 0 else '',
            next=f'{scenario}_{frame_id + 1:08d}'
            if frame_id < MAIN_FRAMES - 1 else '',
            can_bus=can_bus,
            frame_idx=frame_id,
            sweeps=[],
            cams=cams,
            scene_token=scenario,
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            timestamp=timestamp,
            fut_valid_flag=fut_valid_flag,
            map_location=scenario,
        )
        info.update(agent_annotations(data, frame_id))
        info.update(ego_anns)
        infos.append(info)
    return infos, lanes


def create_etri_infos(root_path, out_dir, info_prefix, ego_length, ego_width):
    scenarios = sorted(
        name for name in os.listdir(root_path)
        if osp.isfile(
            osp.join(root_path, name, 'annotation', 'object.parquet')))
    if not scenarios:
        raise FileNotFoundError(f'no scenario found under {root_path}')
    print(f'{len(scenarios)} scenarios found')

    infos = []
    map_lanes = {}
    for scenario in mmcv.track_iter_progress(scenarios):
        scenario_infos, lanes = convert_scenario(
            osp.join(root_path, scenario), ego_length, ego_width)
        infos.extend(scenario_infos)
        map_lanes[scenario] = lanes

    print(f'train sample: {len(infos)}')
    mmcv.mkdir_or_exist(out_dir)
    info_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')
    mmcv.dump(
        dict(infos=infos,
             metadata=dict(version='etri-v1.0', map_lanes=map_lanes)),
        info_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='ETRI challenge data converter')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/etri',
        help='root path of the scenario directories')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/etri',
        help='output directory of the info pkl')
    parser.add_argument('--extra-tag', type=str, default='vad_etri')
    parser.add_argument('--ego-length', type=float, default=4.635)
    parser.add_argument('--ego-width', type=float, default=1.890)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_etri_infos(args.root_path, args.out_dir, args.extra_tag,
                      args.ego_length, args.ego_width)
