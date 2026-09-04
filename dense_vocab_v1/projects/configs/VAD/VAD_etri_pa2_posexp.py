# Phase A r2 — L_pos + L_exp (rank=0). rank 순효과 분리용 최유력 후보.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']
model = dict(freeze_except_goal_decoder=True,
    pts_bbox_head=dict(goal_decoder=dict(loss_mode='pos_exp')))
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/pa2_posexp'
