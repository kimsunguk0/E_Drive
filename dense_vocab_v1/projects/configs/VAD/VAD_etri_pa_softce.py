# Phase A ranking-loss tournament — loss_mode='softce'
# (ETRI_SCOREDRIVE_DESIGN.md §8.5, §9.2). Stage-1 frozen-trunk:
# epoch_4 trunk 고정, goal_decoder(scorer) 만 학습. 유일 변수 = loss_mode.
# lr/seed/batch 불변. 판끼리 동일 seed/sample order/update 수로 비교.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']

model = dict(
    freeze_except_goal_decoder=True,
    pts_bbox_head=dict(
        goal_decoder=dict(loss_mode='softce')))

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/pa_softce'
