# C0-T overfit, **matcher 부채 정리판** (형님 §8).
#
# 이전 계보는 planning을 지배항으로 만들려고 config에서 det/map loss weight와
# **matcher cost를 함께** ×0.05 했다. mmdet이 `loss_cls`와 `cls_cost`를 같게
# assert하기 때문이다. 균등 배율이면 Hungarian 해는 불변이고 그것도 실측했지만
# (실제 비용행렬 72개 assignment 불일치 0/72, 비균등 교란 시 7/72),
# matcher를 건드린 상태로 남기는 건 방법론적 부채다.
#
# 이 config는 그걸 없앤다.
#   matcher cost      원본 (cls 2.0, reg 0.25, map cls 2.0, pts 1.0)
#   det/map loss weight 원본 (cls 2.0, bbox 0.25, traj 0.2, map cls 2.0, pts 1.0)
#   aux_loss_scale=0.05  -> assignment와 loss 계산이 **끝난 뒤** 반환값만 스케일
#   planning loss     ×1.0 그대로
# 이러면 matching이 upstream VAD와 완전히 동일하고 gradient 비중만 planning 중심이다.
#
# 검증: scripts/etri_auxscale_equiv.py 가 이전 config와 loss dict를 대조한다.
_base_ = ['./VAD_etri_c0t_overfit.py']

model = dict(
    pts_bbox_head=dict(
        aux_loss_scale=0.05,
        # ---- 원본 weight 복원 ----
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                           alpha=0.25, loss_weight=0.2),
        loss_map_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=2.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        # map_dir 은 원래도 0이었다 (기하가 우리 lane 표현과 안 맞는다)
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.0),
    ),
    train_cfg=dict(pts=dict(
        # ---- matcher cost 원본 복원 ----
        assigner=dict(cls_cost=dict(weight=2.0),
                      reg_cost=dict(weight=0.25)),
        map_assigner=dict(cls_cost=dict(weight=2.0),
                          pts_cost=dict(weight=1.0)),
    )))
