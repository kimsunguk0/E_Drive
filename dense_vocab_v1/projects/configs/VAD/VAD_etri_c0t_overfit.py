# C0-T — goal-conditioned temporal VAD. **실제 C0-T의 시작점.**
#
# 이전까지의 tvad_* 는 goal 입력이 아예 없는 no-goal temporal VAD였다. 게이트 11은
# 그 모델로 닫혔다 (epoch 8: 6.5447 -> 0.8537, ≤1.0 시나리오 6/8, 최악 1.2520).
# 그게 증명한 것은 "영상->BEV->planner 배관이 살아 있고 challenge-aligned loss를 주면
# 작은 데이터셋을 학습할 수 있다"까지다.
#
# goal은 주최측이 주는 입력이다 -- test clip의 `ego_pose.parquet`가
#     frame -30 … -1, 0, **50**
# 이고 frame 50 이 5초 goal이다 (1125 clip 전부 동일, 실측).
# train 쪽은 `scripts/etri_inject_goal.py`가 `ego_cache.npz['goal']`을 pkl에 넣는다.
# 정합 확인: |goal| / |3초 누적| p50 = 1.665 ≈ 5/3 (등속 가정과 일치).
#
# 컴플라이언스 (8/18 공지 3-1항: goal은 참조 지시 정보, 궤적 도출 계산의 근거 불가)
#   `GoalWaypointDecoder`가 구조로 강제한다.
#       goal + command -> waypoint query  (어느 시각 증거를 읽을지만 결정)
#       temporal BEV   -> key / value     (궤적 값의 유일한 출처)
#       출력 = Linear(attn_out), query residual을 출력에 더하지 않는다
#   금지 설계(goal->MLP->traj, concat->큰 MLP, goal prior + image residual,
#   query residual -> output)는 코드에 존재하지 않는다.
#   `scripts/etri_goal_pathcheck.py`가 불변식을 실증한다.
#
# §9: overfit checkpoint를 초기값으로 쓰지 않는다. canonical bootstrap에서 시작한다.
_base_ = ['./VAD_etri_tvad_metric_overfit.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
GOAL_OVERFIT = '/tmp/pm97/data/etri/pkl/overfit8_goal.pkl'

# ego_fut_goal 을 collect 키에 추가해야 모델까지 도달한다.
_ego_keys = ['ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
             'ego_lcf_feat', 'ego_fut_goal', 'gt_attr_labels']
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CachedImageGeometry', scale=0.4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'] + _ego_keys),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CachedImageGeometry', scale=0.4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PadMultiViewImage', size_divisor=32),
             dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                  with_label=False, with_ego=True),
             dict(type='CustomCollect3D',
                  keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag']
                       + _ego_keys)])
]

data = dict(
    train=dict(ann_file=GOAL_OVERFIT, pipeline=train_pipeline,
               temporal_shuffle=False, filter_empty_gt=False),
    val=dict(ann_file=GOAL_OVERFIT, pipeline=test_pipeline),
    test=dict(ann_file=GOAL_OVERFIT, pipeline=test_pipeline),
)

model = dict(
    pts_bbox_head=dict(
        goal_decoder=dict(n_layers=2, n_head=8, x_scale=60.0, y_scale=30.0),
    ))

total_epochs = 8
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=total_epochs)

load_from = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
