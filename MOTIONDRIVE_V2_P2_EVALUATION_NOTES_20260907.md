# P2 평가 준비와 경쟁 목표 — 2026-09-07 20:15 KST

이 문서는 P2 학습 도중 작성한 **사후 진단 준비**다. 학습 전 사전등록 문서가 아니다.
기존 P2의 LAST3000 주판정, 학습 조건, 데이터 분할은 변경하지 않는다.

## 현재 경쟁 목표의 공개 근거

공식 리더보드의 `E2E Driving` 탭, `팀 최고점`을 선택한 공개 화면을
2026-09-07 20:04 KST에 확인했다. 로그인·비공개 API·제출은 사용하지 않았다.

| 공개 순위 | 팀 | 표시 점수 |
|---|---|---|
| 1 | Neural Drive | 0.0924 |
| 2 | E2Ego | 0.1305 |
| 3 | spilab | 0.1376 |

출처: https://dxchallenge.ai.kr/competitions/5/leaderboard

원 공개 화면 관측은 `reports/public_e2e_leaderboard_20260907_2004.json`에 보존한다.
우리의 반복 사용 tune37 D3와 공개 test 점수는 동일 평가 모집단이 아니며, 공개 점수를
원 L2 또는 최종 규정 심사 통과 점수라고 단정하지 않는다. 따라서 둘의 비율로 예상
순위를 계산하지 않는다. 현재 V2 결과로 1위권 경쟁력이 입증됐다는 주장도 하지 않는다.

20:06 KST의 공개 개요에는 **참가 신청 마감**만 확인됐다. 이는 제출 마감 또는
대회 종료를 뜻하지 않는다. 제출 마감 날짜는 이 조회로 확인하지 못했다.
`reports/public_competition_overview_20260907_2006.json` 참조.

## 평가에서 추가로 분리할 질문

1. C1/T1로 고친 입력 계약에서 실제 D3가 얼마인지 확인한다. C0가 더 좋아도
   배포 계약을 되돌리지 않는다. P1의 잘못된 기하로 학습된 가중치에서 시작한
   적응 실험이라는 한계도 유지한다.
2. 영상에서 예측한 vx/vy/ax/ay/yaw의 MAE, RMSE, bias, Pearson을 계산한다.
   stop은 raw logit에 sigmoid를 적용한 Brier와 AUROC로 평가하며 임계값을 고르지 않는다.
3. GT 현재 yaw ↔ GT 미래 시간 2차 계수, 예측 yaw ↔ GT yaw,
   예측 yaw ↔ 예측/GT 미래 계수는 서로 다른 질문으로 보고한다.
   GT끼리 상관이 높다는 이유로 영상 상태 추정이 정확하다고 주장하지 않는다.
4. 전체·동일 11세션·사전 고정 GT bucket을 각각 집계한다. 반복 tune 분석은
   독립 holdout 검증이 아니며 frame들을 독립 실험처럼 취급하지 않는다.

새 CPU 도구 `scripts/analyze_motiondrive_v2_motion_predictions.py`는 기존 평가기의
`--include-motion-predictions`를 켠 **동일 full forward** 보고서만 받는다.
SHA, 조건별 row 순서, GT/mask 불변, 출력 계약, 좌표로 재계산한 D3를 검사한다.
GT history는 현재 보고서에 없으므로 history 정확도를 만들어 내지 않고
`gt_available=false`로 명시한다. 미래 경로로 GT history를 대체하지 않는다.

출력은 집계만 포함하며, 경로 보정·운동학적 미래 생성·새 forward·추가 GT 조회·
최종 val 접근을 하지 않는다. 이전 P1 보고서는 neural state/history 누락으로
명시적으로 거절된다. 실제 P2 진단 수치는 아직 없다.

## 다음 처방의 조건 — 아직 구현/발사하지 않음

현재 planner에는 scene 3072개, raw motion 192개, 추정 state/history 1개의
memory token이 있다. 이는 구조상의 개수이지 실제 attention 비율이 아니다.
S0에도 raw motion 경로는 남아 있으므로 P1의 S 효과를 전체 history의 효과로
해석하지 않는다.

영상 상태 추정은 쓸 만하지만 계획과의 연결이 약하다는 증거가 생기면,
같은 체크포인트에서 영상 trunk와 motion estimator를 동결하고 planner query에
영상 유래 state/history를 전달하는 작은 zero-init adapter를 통제 실험할 수 있다.
대조군도 동일 adapter/파라미터 수를 가지되 추가 state 입력만 0으로 한다.
양쪽 모두 기존 scene/motion feature와의 연속 신경망 경로를 유지한다.

반대로 상태 추정부터 부정확하다면 adapter를 먼저 늘릴 근거가 약하다.
영상 대응 표현 개선을 우선 검토한다. RAFT 구성요소의 3090 비용만 측정되어 있을 뿐
통합 정확도·통합 latency·사전학습 가중치 사용 적합성은 아직 검증되지 않았다.

어느 경우든 실제 제공 goal/status를 planner query로 넣거나 v*t/a*t²로 미래를
외부 생성하지 않는다. 추가 설계의 성공 기준은 paired D3 개선이며, 예측 곡선의
분산 증가나 학습 loss 감소 자체를 성공으로 보지 않는다.

## 20:38 KST: 공유 GPU 평가 실행 조건 고정

`scripts/evaluate_motiondrive_v2_shared.py`는 기존 모델/학습기/평가기를 수정하지 않는
P2 LAST3000 전용 보호 실행기다. 기존 evaluator의 동일 full forward를 호출한다.

- 네 팔 모두 실제 학습 child/supervisor 종료0·PID 부재를 root가 먼저 확인한다.
  실행기도 선택 팔의 원 launch·plan·완료 manifest·LAST3000·공통 초기값 SHA를 확인한다.
- arm으로 GPU0–3/C0·C1/raw·nominal을 고정한다. batch4, bf16, tune1998,
  normal/image_shuffle/repeat_current/reverse_history와 motion 기록은 변경하지 않는다.
- allocator cap12000MiB, reserve8192MiB, 시작 전 free20192MiB 이상.
  물리 GPU의 UUID 순서를 고정하고 child의 실제 CUDA UUID를 대조한다.
- parent는5초 간격으로 여유를 관측한다. 부족/조회실패 시 자신이 만든 child만
  종료한다. parent 예외에서도 실제 child 종료를 회수한다. 다른 PID나 process group에
  신호를 보내지 않는다. pressure 후 rc0도 실패다. OOM 방지의 절대 보장은 아니다.
- 네 평가의 SOURCE commit을 같게 고정하고, 평가 중 소스/계보가 바뀌면 거절한다.
  기존 출력 파일/원 protocol/기록은 덮어쓰지 않는다. 실제 child exit와 출력 SHA를
  확인한 후 P2 분석기에 전달한다. normal과 학습 LAST 평가의 정확 일치는 별도 확인한다.

전용 CPU56 tests와 독립 리뷰가 완료됐다. 실제 평가 실행은 아직 하지 않았다.
운동 진단에는 고정0 예측 대비 MAE와 GT/예측 분산을 추가한다. 이는 사후 설명용으로,
GT 기준을 planner에 넣거나 검증값으로 보정식을 학습하는 것이 아니다.
