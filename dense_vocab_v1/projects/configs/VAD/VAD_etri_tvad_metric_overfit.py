# planning-isolated 결정적 overfit — "경우 B/C" 규명용.
#
# 8-scene 12ep 결과가 이 config를 필요하게 만들었다:
#   * train 챌린지 L2  6.5764(init) -> 3.3654(ep4) -> 2.9109(ep8) -> 2.7727(ep12)
#     = 초기값의 42%. "25% 이하"에 미달하고 곡선이 평탄해졌다.
#   * 오차의 **95%가 종방향** (|종| 2.745 vs |횡| 0.152). 방향은 맞추고 거리를 못 맞춘다.
#   * 감속 구간에서 오차 폭증 (20260210-150618: speed 8->0, L2 1->16).
#   * planning이 전체 loss의 **1.9%** (plan_* 0.073 / total 3.80).
#   * 종방향 증분오차 부호가 6스텝 모두 같은 앵커 75.6%,
#     누적종오차/증분종오차합 p50 = 1.000 -> 증분 loss가 누적 지표를 대표하지 못한다.
#
# 그래서 두 가지를 동시에 바꾼다.
#   (1) planning을 loss의 지배항으로 만든다 (det/map은 query ranking 유지용 최소치).
#   (2) **누적 planning loss를 켠다** -- 채점 단위와 정렬 (VAD_head.py 패치).
_base_ = ['./VAD_etri_overfit8.py']

# 배관 검증 단계에서는 확률성을 없앤다. augmentation은 full training용이다.
# PhotoMetricDistortionMultiViewImage 를 뺀 것 외에는 c0t 와 동일하다.
# _base_ 상속 config에서는 부모의 변수를 참조할 수 없다. 파이프라인이 쓰는 값만 재정의한다
# (VAD_etri_tiny.py / VAD_etri_tvad_bootstrap.py 와 동일한 값이어야 한다).
point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CachedImageGeometry', scale=0.4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))

model = dict(
    pts_bbox_head=dict(
        # (2) **채점 함수와 동일한** planning loss.
        #     PlanChallengeL2Loss: GT command mode 먼저 slice -> cumsum ->
        #     waypoint별 Euclidean -> [11,11,5,5,2,2]/36 가중.
        #     scripts/test_plan_metric_loss.py 8/8 통과 (l2_challenge 대비 <2e-6).
        #
        #     베이스라인 loss_plan_reg 는 세 곳에서 채점과 어긋났다:
        #       증분에 걸림 / 3-mode 희석(실측 정확히 3.0000) / 성분별 L1(Euclidean 아님)
        loss_plan_metric=dict(type='PlanChallengeL2Loss', loss_weight=1.0),
        #     증분 L1은 **0**으로 둔다. 이 실험의 질문은 "채점 함수와 같은 loss만으로
        #     planner가 8개 시나리오를 외울 수 있는가" 하나뿐이다. 0.1(전체의 0.6%)이라도
        #     남기면 결과가 좋아졌을 때 누적 metric 덕인지 증분 보조 덕인지 섞인다.
        loss_plan_reg=dict(type='L1Loss', loss_weight=0.0),
        #     col/dir 은 챌린지 점수에 안 들어간다. 지금 목적은 planner가 실제 L2를
        #     외울 수 있는지 확인하는 것이므로 끈다.
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.0, dis_thresh=1.0,
                             lane_bound_cls_idx=0,
                             point_cloud_range=point_cloud_range),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=0.0,
                           x_dis_thresh=3.0, y_dis_thresh=1.5,
                           point_cloud_range=point_cloud_range),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.0),
        # (1) det/map은 최소치만 -- planning을 지배항으로.
        #
        # **주의**: loss_cls weight만 바꾸면 `The classification weight for loss and
        # matcher should be exactly the same` assert에 걸린다. 그리고 matcher 비용을
        # 따로 바꾸면 **어떤 예측이 어떤 GT에 매칭되는지**가 달라져 다른 개입이 된다.
        # Hungarian은 비용행렬을 **균등 배율**하면 해가 불변이므로, loss weight와
        # matching cost를 **같은 배율(x0.05)로 함께** 줄인다. 매칭은 그대로, loss만 1/20.
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=0.1),          # 2.0 x0.05
        loss_bbox=dict(type='L1Loss', loss_weight=0.0125),   # 0.25 x0.05
        loss_traj=dict(type='L1Loss', loss_weight=0.01),     # 0.2 x0.05
        loss_traj_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                           alpha=0.25, loss_weight=0.01),    # 0.2 x0.05
        loss_map_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=0.1),      # 2.0 x0.05
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=0.05),   # 1.0 x0.05
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.0),
    ),
    train_cfg=dict(pts=dict(
        assigner=dict(cls_cost=dict(weight=0.1),        # 2.0  x0.05
                      reg_cost=dict(weight=0.0125)),    # 0.25 x0.05
        map_assigner=dict(cls_cost=dict(weight=0.1),    # 2.0  x0.05
                          pts_cost=dict(weight=0.05)),  # 1.0  x0.05
    )))

# lr을 낮추지 않는다 (외우는 것이 목적). epoch만 늘려 수렴을 본다.
total_epochs = 16
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=total_epochs)

load_from = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
