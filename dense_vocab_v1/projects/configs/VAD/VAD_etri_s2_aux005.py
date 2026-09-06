# ⑥-A Stage 2: open-trunk end-to-end (설계서 §9.3). champion 에서 이어서.
#
# champion(pa2_softceexp/epoch_2)은 freeze_except_goal_decoder=True 로 trunk 를 얼린 채
# goal_decoder 만 2 epoch 학습한 것이다. 설계 §9.3 의 Stage 2(가장 좋은 ranker 하나만
# image encoder 까지 연다)는 지금까지 한 번도 실행되지 않았다.
#
# 계층 LR (§9.3 표를 dense VAD 에 대응):
#   goal_decoder            2e-4  (lr_mult 1.0)   - 새 scorer
#   img_neck (FPN)          2e-5  (lr_mult 0.1)
#   img_backbone            1e-5  (lr_mult 0.05)  - config 에서 이미 frozen_stages=1,
#                                                   norm_eval=True (stem+layer1 동결, BN 고정)
#   나머지 pts_bbox_head    2e-5  (lr_mult 0.1)   - BEV transformer, det/map head
#
# trunk 를 열면 detection/map/motion 보조 loss 가 다시 살아난다. 이것이 표현을
# 일반화시키는 신호이며(⑤-C 에서 sparse 가 갖지 못했던 것) Stage 2 의 핵심 이점이다.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']

model = dict(
    freeze_except_goal_decoder=False,
    pts_bbox_head=dict(aux_loss_scale=0.05,
                       goal_decoder=dict(loss_mode='softce_exp')))

optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.05),
        'img_neck': dict(lr_mult=0.1),
        'pts_bbox_head.transformer': dict(lr_mult=0.1),
        'pts_bbox_head.goal_decoder': dict(lr_mult=1.0),
    }))

# champion 에서 이어서 연다(ep4 원본이 아니라).
load_from = '/NHNHOME/data/sukim/adcl/work_dirs/pa2_softceexp/epoch_2.pth'

total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=3)
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/s2_aux005'
