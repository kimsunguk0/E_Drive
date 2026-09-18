# MR → A2/A3: 변경 내용과 규정 근거

2026-09-18, 완료 checkpoint 및 OPEN_ISSUE.md 원문을 다시 확인하고 입력 교체 검사를 실행했다.
새 학습이나 제출은 수행하지 않았다.

## 판단

Q6의 과거 pose에서 현재 status 계산 허용, Q7의 여러 task 공통 특징을 향상하는 간접 활용
허용을 근거로 설계한 구조다. planner에 raw status를 concatenate하거나 단순 embedding으로
직결하지 않는다. 그러나 이 구조에 대한 운영국 개별 승인은 없고, Q9에 따라 최종 판정은
제출 코드 심사다. 영상 교체 검사는 영상 의존성의 증거이며 규정 통과 증명은 아니다.

A3는 provided status로 조건화한 motion 특징에서 state/history도 예측하므로, 그 출력을
Q10의 '영상으로부터 직접 추론한 상태'와 동일시하면 안 된다. 실제 경로는 다음과 같다.

    past/current pose → current status5
                       ├→ shared scene query
                       └→ scene / motion / global FPN channel gates
    current/history images → ResNet/FPN → conditioned image features
                                         ├→ occupancy / lane
                                         ├→ state / history
                                         └→ planner (scene, motion, state, history) → direct XY

채널 gate는 ResNet trunk 이후 FPN 출력에 적용된다. 이미지 픽셀 입력을 바꾼 것이 아니며,
모든 branch에 하나의 동일한 gate를 쓰는 것도 아니다. scene/motion/global용 gate가 따로 있다.
특히 motion gate에서 state/history를 거쳐 planner로 가는 경로까지 포함해 Q7의 허용 근거를
설명해야 한다. 함수 signature에 raw status가 없다는 이유만으로 승인을 주장할 수 없다.

## 기존 제출 대비 변경

| 항목 | 등록 MR | A2-DIRECT | A3-DIRECT |
|---|---|---|---|
| pose-derived status5 입력 | 없음 | shared scene query | shared scene query + FPN gates |
| motion/state 분기에 제공 status 영향 | 없음 | 없음 | 있음 |
| 최종 출력 | direct XY | direct XY | direct XY |
| V0 PREFIX | 0.191002 | 0.164281 | 0.149289 |

기존 제출의 서버 점수는 0.197988이고, 위 비교는 모두 동일한 V0의 점수다.
새 backbone 또는 SparseDrive checkpoint로 교체한 결과가 아니다. 같은 MR 이전 초기값,
train310/83,700행, tune37/1,998행, seed1, 20,554 optimizer update, 유효 batch16,
같은 LR/loss/flip 조건으로 공동 학습했다. registered MR의 microbatch2/eval batch4는
새 실험에서 microbatch8/eval batch8로 바뀌었다. A2와 A3는 이 실행 조건까지 동일하다.

현재 0.149289 모델에는 progress residual, dv+da factorization, P×V selector가 없다.
최초 factorized arm은 별도로 종료됐고, A3-DIRECT의 coefficient_count는 0이다.

## status 출처 재검증

학습 loader는 supervision 파일의 state_target[:5]를 provided_status5로 재사용한다.
이름만 보고 미래 GT인지 판정하지 않고 원본 timestamps/ego_pose에서 다시 계산했다.
37개 scene, 1,998행 전체에 대해 현재까지의 31개 pose만 남기고 미래 pose를 제외한 상태로
causal fitting을 재실행했다. 입력으로 쓴 다섯 status와 최대 절대오차는 모두 정확히 0이었다.
실제 derivative fit은 마지막 약 1초를 사용한다. 테스트 제출 adapter에서도 같은 계산을
제공된 pose/timestamp만으로 수행해야 한다; 이번 검사는 제출 adapter 인증은 아니다.

결과: causal_status_replay_20260918.json.

## 완료 모델의 입력 교체 검사

동일 V0 1,998행에서 이미지 교체 시 receiver의 pose/status/goal/calibration/time/GT를 유지했다.
교체 영상은 같은 frame index의 다른 session에서 결정론적으로 골랐고 현재 6-camera,
scene history, native motion current/history를 모두 함께 교체했다.

| 조건 | A2 PREFIX | A3 PREFIX |
|---|---:|---:|
| 정상 입력 | 0.164281 | 0.149289 |
| 영상 전체를 다른 session으로 교체 | 1.082452 | 0.894936 |
| 영상 전체를 정규화 공간의 단색으로 교체 | 2.418113 | 2.118935 |
| status만 다른 session 값으로 교체 | 0.391047 | 0.718751 |

정상 입력의 예측 좌표는 저장된 terminal prediction과 최대 절대오차 0으로 재현됐다.
A3는 영상과 제공 status 양쪽 모두에 민감하다. 이 검사는 입력 불일치에 대한 민감도를
측정한 것으로, 영상·status의 인과적 기여 비율이나 규정 통과를 수치로 증명하지 않는다.

A3 vx MAE는 정상 0.080202m/s, 영상 교체 0.603835, 단색 1.815908,
status 교체 5.011298이었다. 따라서 정상 vx MAE를 image-only speed recovery로 해석할 수 없다.
A2는 status를 scene query에만 쓰므로 status 교체 후 motion/state 출력은 그대로였다.

결과 파일:
- A2-DIRECT_terminal_input_audit_20260918.json
- A3-DIRECT_terminal_input_audit_20260918.json

## 왜 좋아졌는가에 대한 해석

직접 확인한 것은 status-conditioned scene query의 공동 학습이 -0.026721,
여기에 FPN status conditioning을 추가한 A3가 다시 -0.014992 개선했다는 것이다.
이는 기존 이미지 운동 특징이 불확실하게 표현하던 자차 운동 정보를 허용된 공통 특징
단계에서 보조한 효과라는 해석과 맞는다. 구체적인 내부 경로별 기여는 이 비교만으로
분리되지 않으며, 독립 seed 재현도 아직 없다.

규정 원문 위치: OPEN_ISSUE.md Q6, Q7, Q8, Q9, Q10.
구현: experiments/md_shared_dynamics_20260917/factorized_model.py:210,
train_shared_dynamics.py:52, models/motiondrive_v2/shared_status_query.py:19.
