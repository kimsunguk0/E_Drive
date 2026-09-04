# Phase C — image-only velocity/stop residual. base(champion)+trunk freeze, residual만 학습.
# zero-init -> 시작 시 champion 재현. goal/status 미입력.
_base_ = ['./VAD_etri_anchorvocab_k1024_wide_b200.py']
model = dict(
    freeze_except_goal_decoder=True,
    train_only_goal_residual=True,
    pts_bbox_head=dict(goal_decoder=dict(
        loss_mode='softce_exp', residual=True, res_beta=1.0,
        use_timing=True, use_stop=False, timing_tau=0.1, timing_k=16,
        stop_thresh=1.5, stop_weight=1.0)))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(interval=1114, by_epoch=False, max_keep_ckpts=8)  # ~0.25 epoch
load_from = '/NHNHOME/data/sukim/adcl/work_dirs/pa2_softceexp/epoch_2.pth'
work_dir = '/NHNHOME/data/sukim/adcl/work_dirs/pc_res_timing'
