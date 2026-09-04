# C0-T v2a — goal-conditioned, **현재 프레임 영상 content만이 value**인 planner.
# 330 시나리오 staged pilot.
#
# 계보
# ----
#   v1 (VAD_etri_c0t_overfit_v3.py)  positional shortcut 적발 -> 제출 제외, negative control
#   v2a (여기)                        V = 현재 영상 content 전용. K에는 기하/위치/ego shift/
#                                     temporal state를 허용하되 **값이 아니다**.
#
# 구조 게이트 (배포 해상도 GPU 실측, scripts/c0t_v2a_structural_gates_gpu.py)
#   T6   BEV를 공간적 상수로              0.000e+00
#   T6'  영상 0 + goal 변경               0.000e+00
#   T6"  content V 제거 + K 유지          0.000e+00
#   shift  V=0 + ego shift 변경           0.000e+00
#   T6''' 영상 0 + goal/cmd/shift 임의 -> |pred_xy| 자체가 0   (scripts/etri_t6ppp_gate.py)
#   T8   실제 evidence 교체               non-zero (경로 살아 있음)
# v1은 T6은 통과하지만 T6'(2.938)/T6"(3.742)에서 탈락한다 -- 게이트가 실제로 판별한다.
#
# 초기값 — no-goal full330 epoch 3 visual trunk
# --------------------------------------------
#   0.4345 deploy-faithful L2, 영상 게이트 3종 모두 scenario-bootstrap CI 하한 > 0
#   (matched clip shuffle +5.1848 / BEV-content shuffle +1.5018 / chan_mean +10.6295,
#    97~100% 시나리오 동일 방향). 즉 trajectory prior만 배운 trunk가 아니다.
#   `pts_bbox_head.ego_fut_decoder.*`(기존 planning head)는 제거했다.
#   scripts/etri_make_trunk_ckpt.py, sha256 947bc007...
#
# 학습 앵커 frame_idx >= 30
# ------------------------
#   scripts/etri_queue_audit.py 실측: queue 7 / interval 5 에서
#       frame 0~4   큐 distinct 1  (현재 프레임 7장 복제)  span 0.00 s
#       frame 5k    큐 distinct k+1
#       frame >=30  큐 distinct 7                          span 3.00 s
#   부족분은 `data_queue.insert(0, deepcopy(example))`로 **직전 프레임 복제**가 되고,
#   복제 쌍은 ego pose가 같아 union2one의 can_bus delta/shift가 정확히 0이다.
#   즉 frame<30 은 "정지 이력 3초 + 움직이는 현재"라는 모순 샘플이다 (전체의 10.00%).
#   타 시나리오 프레임은 34,545회 조회되지만 scene_token 검사가 막아 **채택 0회** (누수 없음).
#   no-goal ep3 실측 L2: frame>=30 0.4345 vs frame<30 2.6111.
#   test clip은 완전한 과거 3초를 주므로 학습에 넣을 이유가 없다.
#   ** pkl에서 지우지 않는다 ** -- 이력 조회용으로 frame 0~29는 data_infos에 남긴다.
#   지우면 frame 30이 시나리오 첫 프레임이 되어 같은 문제가 frame 30~59로 옮겨간다.
_base_ = ['./VAD_etri_c0t_v2a_overfit.py']

TRAIN330 = '/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl'
VAL38 = '/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl'
queue_length = 7

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=TRAIN330,
        queue_length=queue_length,
        temporal_shuffle=False,
        filter_empty_gt=False,
        min_frame_idx=30),          # 89,100 앵커 (99,000 - 9,900)
    val=dict(ann_file=VAL38),
    test=dict(ann_file=VAL38),
)

# staged pilot의 LR 그룹 (형님 지시)
#   goal/content planner   2e-4    brand-new random init. 먼저 content를 읽게 만든다
#   나머지 head / BEV      2e-5    lr_mult 0.1
#   image backbone / neck  0       freeze. content tap(`img_neck` 출력)을 고정해 두고
#                                  새 decoder가 그 고정된 content를 읽는 법을 먼저 배운다
# mmcv `DefaultOptimizerConstructor.add_params`는 custom_keys를 **길이 내림차순**으로
# 보고 첫 일치에서 break한다. 그래서 키 길이가 곧 우선순위다:
#   deformable_attention.content_value_proj (39) > pts_bbox_head.goal_decoder (26)
#   > pts_bbox_head (13) > img_backbone (12) > img_neck (8)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'deformable_attention.content_value_proj': dict(lr_mult=1.0),
        'pts_bbox_head.goal_decoder': dict(lr_mult=1.0),
        'pts_bbox_head': dict(lr_mult=0.1),
        'img_backbone': dict(lr_mult=0.0),
        'img_neck': dict(lr_mult=0.0),
    }),
    weight_decay=0.01)
# non-finite loss/grad가 나오면 그 스텝만 버린다. AMP GradScaler와 같은 표준 방식.
#
# 왜 필요한가: 1차 발사가 iter 621에서 죽었고, 직접 원인은 한 스텝 전의 NaN gradient가
# **전체 모델**을 오염시킨 것이었다. AdamW는 lr=0에서도 `addcdiv_`를 실행하고
# IEEE754에서 0*NaN=NaN이며, `clip_grad_norm_`은 total_norm이 non-finite면 모든 grad에
# 그 값을 곱한다. 즉 `lr_mult=0`은 NaN 격리 수단이 아니다.
#
# divergence를 숨기지 않는다: 건너뛴 횟수를 로그에 남기고 20회 연속이면 assert로 멈춘다.
optimizer_config = dict(type='NanSkipOptimizerHook',
                        grad_clip=dict(max_norm=35, norm_type=2),
                        max_consecutive=20)
lr_config = dict(policy='CosineAnnealing', warmup='linear',
                 warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)

# 89,100 / batch 4 = 22,275 iter/epoch. cosine은 4 epoch에 맞춘다 (no-goal control과 동일).
# 게이트를 통과하지 못하면 0.5 epoch에서 세운다 -- 그래도 스케줄은 재현 가능하다.
total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# 0.25 epoch = 5,569 iter. 1,857 x 3 = 5,571 이므로 이 간격이면 0.25 epoch 배수마다
# 대략 정확히 걸리고 **동시에 30분마다 저장**된다.
# 첫 판(interval=5569)은 iter ~600 크래시에서 아무것도 남기지 못했다 -- 1.6시간 손실.
checkpoint_config = dict(interval=1857, by_epoch=False, max_keep_ckpts=24)
evaluation = dict(interval=total_epochs)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

load_from = '/tmp/pm97/ckpt/etri_tvad_visual_trunk_v1_ep3.pth'
