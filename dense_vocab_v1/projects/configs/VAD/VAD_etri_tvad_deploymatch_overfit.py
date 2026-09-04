# planning-isolated overfit, **배포 정합판**.
#
# 왜 새 config가 필요한가
# ----------------------
# VAD_etri_tvad_metric_overfit.py 는 학습 loss를 0.3162까지 내렸는데 실제 evaluator는
# epoch 2부터 3.4~3.6에 갇혔다. 11배 괴리를 끝까지 추적한 결과 원인은 **학습과 배포의
# temporal 조건 불일치** 하나였다.
#
#   배제된 것 (모두 실측)
#     metric 변환      단위테스트 8/8 (scripts/test_plan_metric_loss.py)
#     GT 무효          gt_ego_fut_masks 합 6, 2400/2400
#     BN/dropout       train() 0.3149 vs eval() 0.3029
#     전처리           img 최대차 0.000e+00, lidar2img 0.000e+00
#                      (scripts/etri_pipeline_diff.py)
#     GT 누수          ego_fut_trajs/ego_his_trajs/ego_lcf_feat 모두 head forward 미사용
#     streaming shift  60앵커 전부 참값 대비 비율 1.000 (scripts/stream_seq)
#
#   확정된 원인
#     학습: prepare_train_data가 후보 3개를 shuffle 후 1개를 버려 큐 간격이
#           0.5초/1.0초로 **가변**, 누적 깊이 **2**.
#     배포: tools/etri_test_submit.py 가 clip마다 prev_frame_info를 리셋하고
#           7프레임을 **고정 0.5초**로 흘린다 -> 누적 깊이 **항상 6**.
#     그리고 예측 궤적의 크기가 깊이에 따라 부풀어 오른다 (실측 3초 |p| p50):
#           K=1 22.27 / K=2 26.16 / K=3 27.07 / K=6 27.70   (GT 21.38)
#     모델이 누적된 prev_bev의 shift 정렬에서 속도를 읽기 때문이다.
#     (can_bus 변위를 0으로 만들면 궤적이 3초 2.19 m로 붕괴 -- 영상 제거는 0.85까지만
#      악화. scripts/etri_trainpath_ablation.py)
#
# 그래서 세 가지를 바꾼다.
#   (1) queue_length 3 -> 7      : 학습 누적 깊이 6 = 배포 깊이 6
#   (2) temporal_shuffle=False   : 큐 간격을 배포와 같은 고정 0.5초로
#   (3) filter_empty_gt=False    : range 안에 박스가 없는 프레임이 _rand_another로
#                                  **조용히 다른 랜덤 프레임으로 대체**되던 것을 막는다.
#                                  480앵커 중 21개(4.4%), 시나리오별로는
#                                  20260219-105126 21.7% / 20260213-135828 13.3%.
#                                  그 프레임들은 학습된 적이 없는데 평가에는 들어갔다.
#
# 평가는 반드시 `--depth 6`으로 한다 (배포와 동일). 깊이 0(시나리오 전체 연속)은
# 배포보다 깊어서 숫자가 달라진다.
_base_ = ['./VAD_etri_tvad_metric_overfit.py']

queue_length = 7          # 현재 1 + 과거 6 = 배포 clip(7프레임)과 동일

# batch / lr / epoch / scheduler는 이전 run(VAD_etri_tvad_metric_overfit)과 **동일하게**
# 둔다. 그래야 3.53 -> ? 의 변화가 temporal 조건 하나에서 왔다고 말할 수 있다.
# bs=2 queue=7 실측 0.555 s/iter, 메모리 8.9 GB -> bs=4도 여유가 충분하다.
data = dict(
    train=dict(
        queue_length=queue_length,
        temporal_shuffle=False,
        filter_empty_gt=False,
    ),
)

load_from = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
