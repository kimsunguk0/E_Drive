import os
import json
import copy
import random

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from shapely.geometry import LineString, box

from .nuscenes_vad_dataset import (
    VADCustomNuScenesDataset,
    LiDARInstanceLines,
    v1CustomDetectionConfig,
)


@DATASETS.register_module()
class VADCustomETRIDataset(VADCustomNuScenesDataset):
    """ETRI challenge dataset for VAD.
    """
    MAPCLASSES = ('lane',)

    def __init__(
        self,
        queue_length=4,
        bev_size=(200, 200),
        overlap_test=False,
        with_attr=True,
        fut_ts=6,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        map_classes=None,
        map_ann_file=None,
        map_fixed_ptsnum_per_line=-1,
        map_eval_use_same_gt_sample_num_flag=False,
        padding_value=-10000,
        use_pkl_result=False,
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        crop_size=(1920, 1080),
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right'),
        sample_interval=5,
        map_pts_dim=2,
        temporal_shuffle=True,
        min_frame_idx=0,
        anchor_stride=1,
        *args,
        **kwargs
    ):
        # `load_annotations`/`_set_group_flag`가 super().__init__ 안에서 돌기 때문에
        # 앵커 제한 파라미터는 그 전에 세워야 한다.
        self.min_frame_idx = int(min_frame_idx)
        self.anchor_stride = int(anchor_stride)
        if self.anchor_stride < 1:
            raise ValueError('anchor_stride must be >= 1')
        NuScenesDataset.__init__(self, *args, **kwargs)
        self.crop_size = crop_size
        self.crop_keep_top = crop_keep_top
        self.sample_interval = sample_interval
        self.queue_length = queue_length
        self.temporal_shuffle = temporal_shuffle
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.with_attr = with_attr
        self.fut_ts = fut_ts
        self.use_pkl_result = use_pkl_result

        self.custom_eval_version = custom_eval_version
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(this_dir, '%s.json' % self.custom_eval_version)
        assert os.path.exists(cfg_path), \
            'Requested unknown configuration {}'.format(self.custom_eval_version)
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        self.custom_eval_detection_configs = v1CustomDetectionConfig.deserialize(data)

        # Map lanes are loaded from the info PKL metadata.  The separate file
        # used by the upstream evaluator is only a generated GT cache.
        if map_ann_file is None:
            ann_file = kwargs.get('ann_file')
            if ann_file is not None:
                stem, _ = os.path.splitext(ann_file)
                map_ann_file = stem + '_map_gt.json'
        self.map_ann_file = map_ann_file
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = map_fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = map_eval_use_same_gt_sample_num_flag
        self.map_lanes = self.metadata.get('map_lanes', {})
        self.map_pts_dim = map_pts_dim
        self.is_vis_on_test = True

    def load_annotations(self, ann_file):
        """`min_frame_idx` 미만 프레임을 **학습 앵커에서만** 제외한다.

        왜 pkl에서 지우지 않는가 (중요)
        ------------------------------
        `prepare_train_data`는 과거 이력을 **전역 인덱스 산술**(`index - k*sample_interval`)
        로 찾는다. pkl에서 frame 0~29를 지우면 frame 30이 시나리오의 첫 프레임이 되고,
        똑같은 큐 중복 문제가 frame 30~59로 그대로 옮겨간다. 그래서 `data_infos`는
        300 프레임을 온전히 유지하고 **앵커 목록만** 줄인다.

        측정 근거 (scripts/etri_queue_audit.py)
            frame f 의 큐 = {f-5k : k=0..6, f-5k>=0} + 부족분은 직전 프레임 복제.
            복제 쌍은 ego pose가 같아 can_bus delta/shift가 정확히 0 -->
            '정지 이력 + 움직이는 현재'라는 모순 샘플. frame 0~29 = 전체의 10%.
            no-goal ep3 실측: frame>=30 L2 0.4345 vs frame<30 L2 2.6111.
        """
        data_infos = super().load_annotations(ann_file)
        if getattr(self, 'test_mode', False) or self.min_frame_idx <= 0:
            self.anchor_indices = None
            return data_infos
        self.anchor_indices = [
            i for i, info in enumerate(data_infos)
            if int(info['frame_idx']) >= self.min_frame_idx
            and int(info['frame_idx']) % self.anchor_stride == 0]
        assert len(self.anchor_indices) > 0, \
            f'min_frame_idx={self.min_frame_idx}가 앵커를 전부 없앴다'
        print(f'[VADCustomETRIDataset] 학습 앵커 {len(self.anchor_indices):,} / '
              f'{len(data_infos):,} (frame_idx >= {self.min_frame_idx}, '
              f'anchor_stride={self.anchor_stride}). '
              f'이력 조회용으로 data_infos는 전체를 유지한다.')
        return data_infos

    def __len__(self):
        if getattr(self, 'anchor_indices', None) is not None:
            return len(self.anchor_indices)
        return len(self.data_infos)

    def prepare_train_data(self, index):
        # 앵커 공간 -> 전역 data_infos 공간. `_rand_another`도 flag 길이(=len(self))를
        # 쓰므로 앵커 공간 인덱스를 돌려주고, 여기서 한 번만 매핑하면 일관된다.
        if getattr(self, 'anchor_indices', None) is not None:
            index = self.anchor_indices[index]
        data_queue = []
        prev_indexs_list = list(
            range(index - self.queue_length * self.sample_interval, index,
                  self.sample_interval))
        if self.temporal_shuffle:
            # 원본(BEVFormer 유래): 후보 queue_length개를 섞고 1개를 버린다. 그래서 큐
            # 내부 간격이 0.5초/1.0초로 **가변**이 되고, `union2one`이 계산하는
            # can_bus delta도 그 가변 간격을 따른다.
            #
            # 이게 배포와 어긋난다. 제출 경로(tools/etri_test_submit.py)는 clip마다
            # prev_frame_info를 리셋하고 7프레임을 **고정 0.5초 간격**으로 흘린다.
            # 즉 배포 누적 깊이는 항상 6, 간격은 항상 0.5초다.
            #
            # 실측(epoch 16, 180앵커): 깊이를 바꾸면 예측 궤적의 3초 크기가
            #   K=1  22.27 m / K=2  26.16 / K=3  27.07 / K=6  27.70   (GT 21.38)
            # 로 **깊이에 따라 부풀어 오른다.** 모델이 누적된 prev_bev의 shift 정렬에서
            # 속도를 읽기 때문이다. 학습 깊이(2)와 배포 깊이(6)가 다르면 스케일이
            # 체계적으로 어긋나고, 학습 loss 0.32 vs 배포 3.5~5.3의 괴리가 된다.
            random.shuffle(prev_indexs_list)
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        else:
            # 배포와 동일: 가장 최근 queue_length-1 프레임을 고정 간격으로 쓴다.
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and \
                ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or
                 (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = self.vectormap_pipeline(example, input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()) and \
                        (example is None or ~(example['map_gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        if input_dict is None:
            return None
        info = self.data_infos[index]
        input_dict['timestamp'] = info['timestamp'] / 1e3
        # ego 5초 goal (C0-T). scripts/etri_inject_goal.py 가 pkl에 넣는다.
        # 주최측이 test clip의 ego_pose.parquet frame 50 으로 주는 **입력**이다.
        # 없으면 0으로 둔다 -- goal 없는 pkl로도 config가 돌아야 한다.
        if 'gt_ego_fut_goal' in info:
            input_dict['ego_fut_goal'] = np.asarray(
                info['gt_ego_fut_goal'], dtype=np.float32).reshape(2)
        else:
            input_dict['ego_fut_goal'] = np.zeros(2, dtype=np.float32)
        if self.modality['use_camera']:
            cams = list(info['cams'].items())
            input_dict['cam_intrinsic_raw'] = [c['cam_intrinsic_raw'] for _, c in cams]
            input_dict['cam_intrinsic_undist'] = [c['cam_intrinsic'] for _, c in cams]
            input_dict['distortion'] = [c['distortion'] for _, c in cams]
            input_dict['is_fisheye'] = [c['is_fisheye'] for _, c in cams]
            crop_box = []
            for name, c in cams:
                ox = (c['image_width'] - self.crop_size[0]) // 2
                if name in self.crop_keep_top:
                    oy = 0
                else:
                    oy = c['image_height'] - self.crop_size[1]
                crop_box.append((ox, oy, self.crop_size[0], self.crop_size[1]))
            input_dict['crop_box'] = crop_box
        return input_dict

    def vectormap_pipeline(self, example, input_dict):
        scene_token = input_dict['scene_token']
        lanes = self.map_lanes.get(scene_token, [])

        rotation = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        translation = np.array(input_dict['ego2global_translation'])
        patch = box(self.pc_range[0], self.pc_range[1],
                    self.pc_range[3], self.pc_range[4])

        instances = []
        for lane in lanes:
            pts_ego = (lane[:, :3] - translation) @ rotation
            line = LineString(pts_ego[:, :self.map_pts_dim])
            cropped = line.intersection(patch)
            if cropped.is_empty:
                continue
            if cropped.geom_type == 'MultiLineString':
                cropped = max(cropped.geoms, key=lambda g: g.length)
            if cropped.geom_type != 'LineString' or cropped.length == 0:
                continue
            instances.append(cropped)

        gt_labels = [0] * len(instances)
        gt_instance = LiDARInstanceLines(
            instances, sample_dist=1, num_samples=250, padding=False,
            fixed_num=self.fixed_num, padding_value=self.padding_value,
            patch_size=self.patch_size)

        example['map_gt_labels_3d'] = DC(to_tensor(gt_labels), cpu_only=False)
        example['map_gt_bboxes_3d'] = DC(gt_instance, cpu_only=True)
        return example
