# Phase A r2 — pos_rank_exp 재현성(다른 seed). 0.3865 재현 확인.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']
model = dict(freeze_except_goal_decoder=True,
    pts_bbox_head=dict(goal_decoder=dict(loss_mode='pos_rank_exp')))
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/pa2_posrankexp_s1'
