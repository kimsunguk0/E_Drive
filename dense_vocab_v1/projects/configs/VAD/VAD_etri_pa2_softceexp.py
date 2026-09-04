# Phase A r2 — softCE + L_exp. positive-set(L_pos)의 실제 기여 분리.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']
model = dict(freeze_except_goal_decoder=True,
    pts_bbox_head=dict(goal_decoder=dict(loss_mode='softce_exp')))
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/pa2_softceexp'
