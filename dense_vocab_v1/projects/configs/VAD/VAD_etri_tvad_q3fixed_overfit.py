# 우선순위 1 — **queue 7 단독 효과**를 격리한다 (형님 §7).
#
# tvad_deploymatch(0.4220)와 tvad_metric_overfit(0.8537) 사이에는 세 변경이 섞여 있다.
#     queue_length      3 -> 7          학습 이력 1.0초 -> 3.0초
#     temporal_shuffle  True -> False   큐 간격 가변 -> 고정 0.5초
#     filter_empty_gt   True -> False   4.4% 회복 (한 시나리오 21.7%)
# 뒤 두 개만 적용하고 queue는 3으로 둔다. 그러면
#     이 config vs deploymatch = queue 7 의 단독 효과
#     이 config vs metric_overfit = shuffle+filter 의 합산 효과
# 로 분해된다. loss/matcher/batch/lr/epoch은 metric_overfit 계보 그대로 유지한다
# (deploymatch도 같은 계보이므로 비교가 성립한다).
_base_ = ['./VAD_etri_tvad_metric_overfit.py']

# queue_length는 부모의 3을 그대로 쓴다 (명시하지 않는다).
data = dict(train=dict(temporal_shuffle=False, filter_empty_gt=False))

load_from = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
