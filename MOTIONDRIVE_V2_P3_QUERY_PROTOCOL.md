# P3 — 영상 추정 상태의 planner-query 연결 대조

2026-09-07. P2 결과를 본 뒤 정한 탐색 실험이며 P2의 사전등록을 바꾸지 않는다.
모델 학습 전 이 문서와 실행 코드를 같은 B200 commit에 고정한다.

## 질문과 근거

정상 종료한 C1/T1 LAST3000의 실제 tune1998 full forward에서 영상 추정 yaw와
GT yaw의 Pearson은0.94666, 영상 추정 yaw와 GT 미래 y의 시간2차 계수의 상관은
0.81617이다. 그러나 계획의 해당 계수와 GT의 상관은-0.31459이고 표준편차 비율은
0.07504다(분산 비율0.00563). `reports/p2_c1t1_last3000_motion_analysis.json` 참조.

이는 일부 유용한 영상 추정 신호가 있다는 근거다. 현재 planner의 상태 토큰1개가
원인이라는 인과 증거 또는 회수 가능한 L2 개선량은 아니다. vx MAE0.964m/s,
ax MAE0.284m/s²(고정0 대조0.297)라는 추정 오차도 있다. 물리 상태로 궤적을
대체하거나 수식으로 외삽하지 않는다.

실험 질문: **같은 추가 파라미터·planner 학습에서 영상 추정 상태를 query에도
전달하면 control보다 공식 D3가 개선되는가?**

## 두 팔

| 항목 | control | state |
|---|---|---|
| run | p3_query_control_s0 | p3_query_state_s0 |
| 물리 GPU | 4 | 5 |
| 새 adapter 입력 | 정규화된 영상 추정22차원 ×0 | 동일22차원 ×1 |
| 기존 scene/raw motion/state 토큰 | 모두 유지 | 모두 유지 |

MLP는6개 query에 공유되는150→128→128이며35,840 parameters다.
`q' = q + MLP(concat(q,z22))`, 마지막 Linear의 W/b만0 초기화한다.
control도 동일 MLP를 실행하며 query-only 적응은 가능하다. stop은 원 raw logit이다.
기존 FP32 planner, ABS XY, 출력 단위(10,5), G1/S1 및 low_feature 설정은 유지한다.
제공 goal/status/history/pose를 query에 넣지 않고, 여러 task가 공유하는 연속
영상 특징 경로를 유지한다. Q10의 영상 유래 상태 허용과 goal-query 금지를 구분한다.
이는 규정 준수 설계이지 운영국의 개별 제출 승인이라는 뜻은 아니다.

## 초기화와 동결

원본은 실제 정상 종료한 C1/T1 LAST3000이며 SHA는
`5cd98d28157abf169f0f2378c3d0ab29d43f4d6382e9c03ee611e7fc5d864c20`다.
원본 모델을 strict load하고 원 tensor를 그대로 보존하면서 정확히 adapter4개 tensor만
명시적으로 추가한다. 새 architecture는 `motiondrive_v2_image_state_query_v1`이다.
기존 checkpoint에 strict=False를 적용하거나 원 exporter를 우회하지 않는다.

두 팔의 공통 adapter seed는0이다. initial 전체 tensor SHA가 같아야 한다.
실제 train8에서 legacy/control/state의 full forward를 비교하고, 각 팔 초기 full
tune D3도 원값0.4454620049779748과 정확히 같아야 학습한다. train8 검사는
초기 함수 동등성 검사이지 성능·일반화 평가가 아니다.

backbone/FPN/scene/perception/motion/state/history estimator를 모두 동결한다.
기존 planner 전체와 adapter만 학습한다. 동결 부위는 항상 eval, feature 계산은
no_grad이며 inference_mode feature를 학습 planner에 주지 않는다. 모든 batch에서
영상 인코딩을 다시 실행하며 학습·평가 feature cache를 만들지 않는다.
동결 parameter/BN buffer의 시작·종료 SHA와 gradient 부재를 검증한다.

## 고정 학습 조건

- geometry_v2/nominal, train203(54,810행), tune37(1,998행·11세션), frame≥30.
- train stride1/tune stride5, 동일 row/epoch photometric augmentation과 sampler seed0.
- AdamW 새 optimizer,1000steps,LR5e-5,warmup50+cosine,wd.01,clip5.
- logical batch16, microbatch2×8, full-batch label/mask 분모 보존.
- bf16 encoders/FP32 planner, 고정 cuDNN 결정성 및 matmul TF32 off.
- 기존 D3+.2occ+.2lane+.2motion(uncertainty 포함). 동결 auxiliary는 상수항이며
  학습 가능한 gradient는 D3에서만 나온다. total/NLL 감소를 성능 이득으로 쓰지 않는다.
- eval step0/250/500/750/1000. **LAST1000이 주판정**, BEST는 별도 참고다.
- 각 eval의 기존 per-frame D3/scene/session/frame 기록을 보존한다.
- GPU4/5 단일 UUID→cuda:0, 시작 cap12000MiB+reserve8192MiB, 부모5초 감시와
  매 microbatch 여유 검사. 타 작업/타 PID/process group/6·7번은 사용·중지하지 않는다.
- 실패/OOM/nonfinite/SIGSEGV는 실제 그대로 기록한다. completed manifest가
  actual OS exit를 대신하지 않는다. 원본·기존 run·실패 record는 덮어쓰지 않는다.

## 판정과 다음 단계

주 지표는 동일 tune1998의 공식 D3([11,11,5,5,2,2]/36 시간 가중, frame 동일 가중).
11개 세션을 복원추출하는 paired cluster bootstrap10,000회(seed20260907)로
state−control 차이의95% 구간을 계산한다. 같은 세션의 모든 프레임을 함께 뽑고,
추출된 프레임 수로 가중한다. 프레임 독립 bootstrap이나 proxy 가중치는 쓰지 않는다.

탐색 진전 기준은 state−control≤-0.01이며95% 구간 상단도0 미만일 때 seed 복제로
이어가는 것이다. 구간이0을 포함하면 미확정, 두 팔이 같이 좋아지면 추가 학습 효과다.
단일 seed·반복 tune 한계는 유지한다. curve 분산 증가만으로 채택하지 않는다.
후속 실제 full-forward 평가에서 종/횡·시간별·stop/turn 교란 반응을 다시 확인하고,
긍정적인 경우 다른 seed 및 실제3090 전체 forward 비용을 검증한다.
최종 val/test 및 공식 제출은 이 screening에 사용하지 않는다.

P2 원4판의 clean-exit gate 실패는 그대로 남으며 이 새 P3로 소급해서 바꾸지 않는다.
또한 오류 기하 P1에서 warm-start한 가중치의 한계가 남는다. 올바른 기하를 공개
nuImages R50 초기값부터 학습하는 별도 대조는 아직 실행한 것으로 주장하지 않는다.
