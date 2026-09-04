# 게이트 11 — 8-scene overfit. 좌표·loader·loss를 잡는 진단용이지 성능 실험이 아니다.
#
# 8-scene도 외우지 못하면 full run을 돌리지 않는다. 그 경우 아키텍처 문제가 아니라
# 거의 항상 좌표·loader·loss 문제다.
#
# 통과 조건 (절대값 하나보다 이것들을 본다)
#   * train planning loss가 계속 감소, 초기값 대비 최소 수 배
#   * 8-scene train metric이 val보다 충분히 낮음
#   * 좌/우회전 출력 방향이 맞음
#   * image-zero / clip-shuffle 시 출력이 크게 변함
#   * backbone까지 planning gradient 도달
#   * can_bus 불변성 (use_can_bus=False)
_base_ = ['./VAD_etri_tvad_bootstrap.py']

OVERFIT = '/tmp/pm97/data/etri/pkl/overfit8.pkl'   # train 330개 중 8개, 2,400 샘플

data = dict(
    samples_per_gpu=4,          # H200 실측 16.8 GB (여유 ~50 GB)
    workers_per_gpu=4,
    train=dict(ann_file=OVERFIT),
    # val/test도 같은 8개로 둔다 -- 여기서 보는 건 "외우는가"이고, 일반화는
    # scripts/etri_vad_eval.py 가 진짜 val 38개로 따로 잰다.
    val=dict(ann_file=OVERFIT),
    test=dict(ann_file=OVERFIT),
)

# 2,400 샘플 / bs 4 = 600 iter/epoch.
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=total_epochs)

# 외우는 것이 목적이므로 lr을 낮추지 않는다. batch 4는 원본(8)의 절반이지만
# 이건 진단이라 절대규칙 7의 대상이 아니다 -- 본 학습에서는 batch 8을 맞춘다.
optimizer = dict(type='AdamW', lr=2e-4,
                 paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
                 weight_decay=0.01)
lr_config = dict(policy='CosineAnnealing', warmup='linear',
                 warmup_iters=100, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)

checkpoint_config = dict(interval=4)
log_config = dict(interval=25, hooks=[dict(type='TextLoggerHook')])

# 축 리맵된 nuScenes 가중치에서 출발한다 (scripts/etri_remap_ckpt.py --parts ego,traj,ref,reg,map).
# BEV 격자 순열은 **제외**한다 -- 실측 7.6847 -> 7.7106 으로 오히려 나빴다 (seed 노이즈 0.004의 6배).
# 학습된 bev_embedding 내용이 격자 인덱스로 생성되는 bev_pos 와 짝지어져 있어서,
# 내용만 옮기면 그 짝이 깨진다.
load_from = '/tmp/pm97/ckpt/VAD_tiny_etri_axes.pth'
