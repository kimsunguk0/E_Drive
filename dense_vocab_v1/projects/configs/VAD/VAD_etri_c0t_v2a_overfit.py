"""C0-T v2a: current-frame, camera-content-only trajectory values."""
_base_ = ['./VAD_etri_c0t_overfit_v3.py']

queue_length = 7

model = dict(
    # Builds the content-only FPN stream and passes it separately from the
    # ordinary BEV trunk.  Off by default in model code.
    expose_image_evidence=True,
    pts_bbox_head=dict(
        aux_loss_scale=0.05,
        goal_decoder=dict(
            _delete_=True,
            type='ImageEvidenceGoalWaypointDecoder',
            n_head=8,
            x_scale=60.0,
            y_scale=30.0),
        transformer=dict(expose_image_evidence=True)))

data = dict(train=dict(
    ann_file='/tmp/pm97/data/etri/pkl/overfit8_goal.pkl',
    queue_length=queue_length,
    temporal_shuffle=False,
    filter_empty_gt=False))

load_from = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
