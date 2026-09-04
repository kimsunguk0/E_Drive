# no-goal tvad_bootstrap **full-data control** — 330 시나리오 전체.
#
# 왜 이걸 먼저 돌리는가
# --------------------
# C0-T v1은 positional shortcut이 적발되어 full-data 후보에서 제외됐다
# (val image-zero 1.1361 < full 1.1749). 그러나 no-goal 계보는 그 우회로가 없고
# 8-scene에서 강한 통과(deploymatch 0.4220, 8/8)를 했으므로 전체 학습을 시작해도 된다.
# 이 모델은 최종 A/B의 **필수 control**이다.
#
# 확정된 recipe (P1 3-way 분해 + §7 + §8 결과)
#   queue_length 7        학습 이력 3.0초 = 배포 clip과 동일. P1에서 단독 효과 40% 확인
#                         (q3fixed 0.6987 -> deploymatch 0.4220). 수렴 안정성도 queue 7만
#                         ep14까지 단조 개선하고 queue 3은 ep8 이후 2배 되돌아간다.
#   temporal_shuffle F    큐 간격을 배포와 같은 고정 0.5초로
#   filter_empty_gt F     4.4% 회복. 0-GT 프레임 loss 검증 PASS, log(0) 잠재 크래시 수정
#   aux_loss_scale 0.05   matcher/loss weight는 upstream 원본, assignment 후에만 스케일
#   10 Hz 앵커            I/O 무료(RAM 1.5TB, 캐시 62GB), 0.1초 변위 0.8~1.3m는 목표
#                         정밀도 0.2~0.5m의 2~6배라 중복이 아니다. 과적합이 지배적
#                         실패 모드였으므로 반복을 줄이는 쪽이 맞다.
#   canonical bootstrap   8-scene overfit ckpt를 초기값으로 쓰지 않는다 (§9)
#
# 데이터: etri_train330_goal.pkl (99,000 / 330 시나리오). val 38 + 나머지 8 제외 확인.
#         goal 필드는 들어 있지만 이 계보는 읽지 않는다 (collect 키에 없음).
_base_ = ['./VAD_etri_c0t_nogoal_control_v3.py']

TRAIN330 = '/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl'
VAL38 = '/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl'

data = dict(
    train=dict(ann_file=TRAIN330),
    val=dict(ann_file=VAL38),
    test=dict(ann_file=VAL38),
)

# 24,750 iter/epoch @ 0.89 s = 6.12 h/epoch.
# 4 epoch = 99,000 iter ~= 24.5 h. 8-scene run(9,600 iter)의 10배 optimization,
# 41배 데이터. cosine이 4 epoch에 맞춰 완주하도록 total_epochs로 고정한다.
total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)   # 6시간마다 사용 가능한 ckpt
evaluation = dict(interval=total_epochs)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

load_from = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
