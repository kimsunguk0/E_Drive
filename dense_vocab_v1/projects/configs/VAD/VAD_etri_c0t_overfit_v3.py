# ##############################################################################
# !! 제출 후보 제외. 이 config로 학습한 모델은 `c0t_positional_shortcut_v1` 이다.
#    goal이 영상 대신 temporal BEV의 learned positional structure를 주소처럼 써서
#    궤적을 만든다. val에서 image-zero(1.1361)가 full(1.1749)보다 **좋다**.
#    negative control 전용. full-data 학습 금지.
#    상세: /tmp/pm97/ckpt/negctrl/c0t_positional_shortcut_v1.md
# ##############################################################################
# 우선순위 2 — C0-T A/B, **deploymatch 설정에 정합**시킨 판.
#
# v2는 queue 3 / 8 epoch으로 돌아 deploymatch(queue 7 / 16 epoch)와 비교가 안 됐다.
# 실측 handicap: v2 최선 0.7919 vs deploymatch 최선 0.4220.
# 여기서는 temporal 조건과 학습 예산을 deploymatch와 동일하게 맞추고 goal만 더한다.
#     queue_length 7 / temporal_shuffle False / filter_empty_gt False / 16 epoch
#     matcher는 §8 청산판(원본 cost + aux_loss_scale) 유지
# 대조군은 VAD_etri_c0t_nogoal_control_v3.py (goal_decoder만 제거).
_base_ = ['./VAD_etri_c0t_overfit_v2.py']

queue_length = 7
data = dict(train=dict(queue_length=queue_length))

total_epochs = 16
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=total_epochs)
