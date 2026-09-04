# anchor 조건 스코어러: 각 후보가 자기 웨이포인트의 BEV 를 grid_sample 로 직접
# 조회한다. 전역 요약만 쓰던 ImageOnlyVocabularyDecoder 가 4 epoch 에서
# top6_l2 0.655 / top-1 3.75 로 정체한 것에 대한 대응이다 (사전 oracle 0.095).
# pc_range x 확장판: [-60, 60] (기존 ±30). bev_w_ 100 -> 200 이므로
# x 셀 크기는 0.6 m 로 **동일**하고 커버리지만 2배다. y 는 손대지 않았다
# (횡변위가 작아 확장이 낭비 -- GT |dy| p50 이 1 m 미만).
# 근거: val GT 3초 지점의 58.2%가 기존 ±30 BEV 밖이었다.
# 규정 준수 계보: goal/cmd/ego status 를 planner 에 넣지 않는다.
# 영상 증거만으로 고정 사전(K=1024)에 점수를 매기고, top-M 중 선택은
# 모델 밖의 고정 규칙이 한다 (projects/mmdet3d_plugin/VAD/vocab_decoder.py 참조).
point_cloud_range = [-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
dataset_type = 'VADCustomETRIDataset'
data_root = '/tmp/pm97/data/etri/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CachedImageGeometry', scale=0.4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='CustomObjectRangeFilter',
        point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
    dict(
        type='CustomObjectNameFilter',
        classes=['Car', 'Pedestrian', 'Cyclist']),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='CustomDefaultFormatBundle3D',
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        with_ego=True),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
            'ego_fut_goal', 'gt_attr_labels'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CachedImageGeometry', scale=0.4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='CustomObjectRangeFilter',
        point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
    dict(
        type='CustomObjectNameFilter',
        classes=['Car', 'Pedestrian', 'Cyclist']),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='CustomDefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist'],
                with_label=False,
                with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                    'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'ego_fut_goal',
                    'gt_attr_labels'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='VADCustomETRIDataset',
        data_root='/tmp/pm97/data/etri/',
        ann_file='/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='CachedImageGeometry', scale=0.4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='CustomDefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist'],
                with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                    'ego_lcf_feat', 'ego_fut_goal', 'gt_attr_labels'
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        bev_size=(100, 200),
        pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
        queue_length=7,
        map_classes=['lane'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right',
                       'camera_rear_wide'),
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        temporal_shuffle=False,
        filter_empty_gt=False,
        min_frame_idx=30,
        anchor_stride=5),
    val=dict(
        type='VADCustomETRIDataset',
        ann_file='/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='CachedImageGeometry', scale=0.4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='CustomDefaultFormatBundle3D',
                        class_names=['Car', 'Pedestrian', 'Cyclist'],
                        with_label=False,
                        with_ego=True),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'ego_fut_goal', 'gt_attr_labels'
                        ])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='/tmp/pm97/data/etri/',
        pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
        bev_size=(100, 200),
        samples_per_gpu=1,
        map_classes=['lane'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right',
                       'camera_rear_wide'),
        use_pkl_result=True,
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    test=dict(
        type='VADCustomETRIDataset',
        data_root='/tmp/pm97/data/etri/',
        ann_file='/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='CachedImageGeometry', scale=0.4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='CustomDefaultFormatBundle3D',
                        class_names=['Car', 'Pedestrian', 'Cyclist'],
                        with_label=False,
                        with_ego=True),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'ego_fut_goal', 'gt_attr_labels'
                        ])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
        bev_size=(100, 200),
        samples_per_gpu=1,
        map_classes=['lane'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right',
                       'camera_rear_wide'),
        use_pkl_result=True,
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=999,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='CachedImageGeometry', scale=0.4),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=True),
        dict(
            type='CustomObjectRangeFilter',
            point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
        dict(
            type='CustomObjectNameFilter',
            classes=['Car', 'Pedestrian', 'Cyclist']),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1920, 1080),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='CustomDefaultFormatBundle3D',
                    class_names=['Car', 'Pedestrian', 'Cyclist'],
                    with_label=False,
                    with_ego=True),
                dict(
                    type='CustomCollect3D',
                    keys=[
                        'gt_bboxes_3d', 'gt_labels_3d', 'img',
                        'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                        'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                        'gt_attr_labels'
                    ])
            ])
    ],
    metric='bbox',
    map_metric='chamfer')
checkpoint_config = dict(interval=1, max_keep_ckpts=4, by_epoch=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/anchorvocab_k1024_wide_b200'
load_from = '/NHNHOME/data/sukim/adcl/checkpoints/anchorvocab_k1024_wide/epoch_4.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.15, 0.15, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_classes = 3
map_classes = ['lane']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = 1
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 200
queue_length = 7
total_epochs = 4
crop_keep_top = ('camera_front_left', 'camera_front_right', 'camera_rear_left',
                 'camera_rear_right', 'camera_rear_wide')
model = dict(
    type='VAD',
    freeze_except_goal_decoder=False,   # BEV 격자가 바뀌어 bev_embedding 재초기화 -> 동결 금지
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(
        img='/NHNHOME/data/sukim/adcl/ckpt/backbones/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=60,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=100,
        bev_w=200,
        num_query=300,
        num_classes=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=100,
        map_num_classes=1,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=100,
            map_num_pts_per_vec=20,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=False,
            rotate_center=[50, 50],
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            expose_image_evidence=True),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-65, -20, -10.0, 65, 20, 10.0],
            pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
            max_num=100,
            voxel_size=[0.15, 0.15, 8],
            num_classes=3),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-65, -20, -65, -20, 65, 20, 65, 20],
            pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 8],
            num_classes=1),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            # bev_h_=100 (y), bev_w_=200 (x). col <-> w 이므로 200 이어야 한다.
            # 100 으로 두면 인덱스 199 에서 embedding out-of-range ->
            # CUDA device-side assert (실측: positional_encoding 에서 터짐).
            row_num_embed=100,
            col_num_embed=200),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.0),
        loss_plan_reg=dict(type='L1Loss', loss_weight=0.0),
        loss_plan_bound=dict(
            type='PlanMapBoundLoss',
            loss_weight=0.0,
            dis_thresh=1.0,
            lane_bound_cls_idx=0,
            point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
        loss_plan_col=dict(
            type='PlanCollisionLoss',
            loss_weight=0.0,
            x_dis_thresh=3.0,
            y_dis_thresh=1.5,
            point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.0),
        loss_plan_metric=dict(type='PlanChallengeL2Loss', loss_weight=0.0),
        goal_decoder=dict(
            type='AnchorGroundedVocabularyDecoder',
            pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
            bev_h=100,
            bev_w=200,
            use_global=True,
            anchors_path='/NHNHOME/data/sukim/adcl/h200_latest/artifacts/dense_vocab/etri_train330_vocab_k1024.npy',
            n_head=8,
            loss_weight=1.0,
            target_temperature=0.1,
            top_m=6),
        aux_loss_scale=0.0),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 8],
            point_cloud_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]),
            map_assigner=dict(
                type='MapHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                pc_range=[-60.0, -15.0, -2.0, 60.0, 15.0, 2.0]))),
    expose_image_evidence=True)
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=0.01)
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type='NanSkipOptimizerHook',
    max_consecutive=20)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=4)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
OVERFIT = '/tmp/pm97/data/etri/pkl/overfit8.pkl'
GOAL_OVERFIT = '/tmp/pm97/data/etri/pkl/overfit8_goal.pkl'
_ego_keys = [
    'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
    'ego_lcf_feat', 'ego_fut_goal', 'gt_attr_labels'
]
TRAIN330 = '/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl'
VAL38 = '/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl'
gpu_ids = range(0, 1)
