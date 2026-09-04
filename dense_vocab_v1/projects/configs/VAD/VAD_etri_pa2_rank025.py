# Phase A r2 — L_pos + 0.25*L_rank + L_exp. rank 가중치 과대 여부 확인.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']
model = dict(freeze_except_goal_decoder=True,
    pts_bbox_head=dict(goal_decoder=dict(loss_mode='pos_rank_exp', w_rank=0.25)))
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/pa2_rank025'
