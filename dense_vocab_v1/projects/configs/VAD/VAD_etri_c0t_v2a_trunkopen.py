# v2a + trunk 개방 — H200 런에서 쓰지 않은 여력을 시험한다.
#
# H200 런은 `img_backbone`/`img_neck`을 **처음부터 끝까지 lr 0으로 동결**했다.
# 원래 staged 계획은 "content path가 살아 있으면 BEV 마지막 부분을 풀고, backbone은
# 더 나중에 푼다"였는데, content path는 확실히 살아 있다 (2.25 epoch에서
# evidence_chan_mean 배율 28.8×, 100% 시나리오). 그런데 trunk는 한 번도 안 풀었다.
#
# 왜 mid-run 해제가 아니라 처음부터인가
# ----------------------------------
# mmcv `LrUpdaterHook`이 매 iteration `param_group['lr']`을 `base_lr`에서 다시 쓴다.
# 그래서 mid-run 해제는 `base_lr` 자체를 바꾸는 custom hook이 필요하고, 게다가
# mid-epoch resume가 그 epoch의 나머지를 버리는 문제(§5-3)까지 얽힌다.
# 여기서는 **단일 변수**로 만든다: trunk 동결 vs trunk 저속 학습.
# 이득이 확인되면 그 다음 판에서 제대로 staging 한다.
#
# LR 그룹 (goal 판 대비 backbone/neck만 다르다)
#   deformable_attention.content_value_proj  1.0   -> 2e-4
#   pts_bbox_head.goal_decoder              1.0   -> 2e-4
#   pts_bbox_head                           0.1   -> 2e-5
#   img_neck                                0.1   -> 2e-5   (goal 판은 0.0)
#   img_backbone                            0.05  -> 1e-5   (goal 판은 0.0)
#
# backbone을 neck보다 더 낮게 두는 이유: 사전학습 ImageNet 표현이 가장 깨지기 쉽고,
# no-goal full330 런도 backbone은 lr_mult 0.1로만 돌렸다.
#
# 그 외 전부 goal 판과 동일: 같은 trunk 초기값, 같은 frame>=30 89,100 앵커,
# 같은 queue 7 / 고정 cadence / filter_empty_gt=False, 같은 4 epoch cosine,
# 같은 seed 0 --deterministic, 같은 NanSkipOptimizerHook.
_base_ = ['./VAD_etri_c0t_v2a_full330.py']

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        '_delete_': True,
        'deformable_attention.content_value_proj': dict(lr_mult=1.0),
        'pts_bbox_head.goal_decoder': dict(lr_mult=1.0),
        'pts_bbox_head': dict(lr_mult=0.1),
        'img_neck': dict(lr_mult=0.1),
        'img_backbone': dict(lr_mult=0.05),
    }),
    weight_decay=0.01)
