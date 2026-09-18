# A2 정체 구간 재검토 — 2026-09-18

## 결론과 작업 범위

SIDE/MH4/QREFINE의 작은 효과가 A2의 성능 상한을 입증하지는 않는다.
다만 같은 학습 목표 아래 scene aggregation을 확대하는 것을 다음 주력으로 반복할 근거는 약해졌다.
다음 첫 대조는 **A2의 후반 planning 중심 학습: motion 보조 loss의 비중 조정**을 권한다.
이는 새로 확인한 작은 train gradient 검사에서 나온 가설이며, 아직 성능 개선 결과가 아니다.
더 큰 표현 변경으로는 기존 raw-image motion matching의 세밀도를 높이는 방향을 후속 후보로 둔다.

사용자의 “향상시킬 방법” 요청에 대해 기존 기록/코드를 검토하고 GPU 0–3에서 읽기 전용 진단을 수행했다.
새 학습, optimizer update, 가중치 변경, 공식 제출, production source 수정은 하지 않았다.
GPU 4–7에는 작업을 실행하지 않았다. 이 문서의 학습안은 제안이며 시작된 run이 아니다.

## 기존 결과가 말하는 범위

같은 V0 1,998행의 PREFIX는 기존 A2의 배포형 nominal 평가 0.164455,
BASE-NOM 0.165511, MH4 0.164495, SIDE-SCENE 0.172406, QREFINE 0.164252다.
QREFINE은 기존 A2 대비 0.000204(0.124%)만 낮고, 일반 주행은 0.168442→0.168445로 개선되지 않았다.
11-session bootstrap 구간도 0을 포함한다. 작은 수치 차이를 새 구조의 확실한 이득으로 간주하지 않는다.
이 숫자는 로컬 DEV이며 공식 제출 점수와 동일시하지 않는다.
현재 확인된 공식 MR FULL은 0.18596892793122946이고 A2 FULL 공식 점수는 아직 없다.
A2 FULL의 V0 0.088934는 학습에 포함된 데이터의 진단값이다.

기존 MR에서는 영상 대응 cost-volume의 읽는 방식을 바꾼 native-motion 계열이
LEN 대비 약 0.032–0.033 개선됐고, 폭 확대·ADJ·단순 LONG은 뚜렷한 추가 이득이 없었다.
고해상도 입력만의 이득과 이 대응 표현의 이득은 구별해야 한다.
MR residual 실험은 train/V0 보정량 분포 차이와 일반 주행에서 매우 작은 실제 correction 이득을 보였다.
따라서 oracle 수치나 pooled-feature MLP 하나를 근거로 residual 재시도를 주력으로 되돌리지 않는다.

## 이번 실측 1: 추론 정밀도 변경은 빠른 개선책이 아니었다

A2-BASE-NOM terminal, 동일 V0/GT/row/입력을 사용했다. 값이 작을수록 좋다.

| 계산 | PREFIX |
|---|---:|
| 기존 BF16 재현 | 0.165510641 |
| BF16 + FP32 geometry sampler, 기존 sampler 출력 dtype 유지 | 0.165531119 |
| FP16 추론 | 0.165748653 |
| FP32 추론, TF32 비활성화 | 0.165744149 |

scene sampling에서 FP32 투영 좌표를 BF16 feature dtype으로 바꾸는 것을 확인했다.
첫 15,872개 유효 좌표의 반올림 차이는 768×432 기준 평균 x/y 0.253/0.151 pixel이었다.
그러나 이를 추론에서 고쳐도 PREFIX가 개선되지 않았다. 실제 정밀도 손실과 성능 병목은 다른 주장이다.
BF16로 학습한 가중치에 대한 개입이므로 FP32 재학습의 결과까지 판정한 것은 아니다.
현재 근거로는 정밀도 재학습을 다음 우선순위로 올리지 않는다.

## 이번 실측 2: SIDE의 평균 경로를 추론에서 제거해도 개선되지 않았다

scene은 learned attention 이외에 모든 유효 source의 무가중 평균을 base skip과 query 초기값으로 사용한다.
따라서 side 영상을 추가하면 attention이 새 영상을 무시하더라도 기존 함수가 보존되지 않는다.
이는 비교를 해석할 때 중요한 구현 사실이다. SIDE 실패의 원인이 평균 경로라고 확정할 수는 없다.

| SIDE terminal 가중치 고정 개입 | PREFIX |
|---|---:|
| 원래 SIDE 재현 | 0.172406031 |
| 추가 side 영상을 비활성화 | 0.219972303 |
| attention에는 14개 source 유지, 평균/query 초기값에는 원래 10개만 사용 | 0.520714443 |

학습된 SIDE는 추가 관측 및 기존 혼합 방식에 적응했다. 이를 추론에서 빼는 것으로 복구되지 않는다.
강한 A2를 보존하고 zero-init 잔차로 side evidence를 추가하는 새 학습은 별도 가설이지만,
이번 결과를 그 방식의 성능 근거로 사용할 수 없다. 지금은 다시 SIDE부터 돌리지 않는다.

## 이번 실측 3: motion 보조 loss의 backbone gradient가 작지 않았다

A2-BASE-NOM terminal에서 train의 서로 다른 32개 scene에서 1행씩 고정 seed로 뽑았다.
4행씩 8 batch, augmentation 없음, fixed BN/eval, shared backbone/FPN 전체 trainable parameter에 대해 측정했다.
V0 gradient나 optimizer는 사용하지 않았다. 모델 state SHA-256은 전후 동일하다.
아래는 실제 학습 가중치를 적용한 gradient norm의 PREFIX 대비 비율과 방향이다.

| 항 | norm 비율 중앙값 | PREFIX와 cosine 중앙값 | 음의 cosine batch |
|---|---:|---:|---:|
| LEN × 0.25 | 0.0973 | 0.6452 | 0/8 |
| occupancy/lane, 각각 × 0.2 | 0.1053 | 0.0752 | 3/8 |
| motion × 0.2 (state/history/stop) | 0.8523 | 0.0591 | 4/8 |

**가설:** 이미 표현을 학습한 후반에도 과거/현재 상태를 맞히는 보조 목표가
planning과 비슷한 크기로 backbone을 계속 갱신한다. 후반에는 이 비중을 낮추고
영상 motion feature를 최종 미래 경로 오차에 더 직접 맞추는 것이 도움이 될 수 있다.
이것은 “motion 정보가 필요 없다”거나 “보조 loss가 원인으로 확정됐다”는 뜻이 아니다.
32개 train 장면의 작은 검사이며 raw gradient는 Adam preconditioning/clip/일반화 효과를 설명하지 못한다.
부호 충돌 자체는 multi-task 학습에서 생길 수 있고 규제 효과가 유익할 수도 있다.
음수 uncertainty NLL의 scalar 크기만 보고 이 판단을 한 것도 아니다.

첫 4개 train 행의 planner attention에서 motion token의 질량은 두 layer에서 약 9.4%/12.0%였다.
scene/motion/state token 수만으로 motion이 무시된다고 주장할 근거도 부족하다.
Attention 질량은 value 크기와 후단 변환을 반영하지 않아 인과 기여도가 아니다.

## 다음 첫 실험: 동일 조건의 짧은 planning 중심 2단계 학습

첫 공통 초기값은 A2-BASE-NOM step20,554를 사용한다. nominal 입력 계보가 같고,
가장 복잡하지 않은 A2의 query-only 입력 경계를 보존하기 위한 선택이다.
이전 A2 및 QREFINE terminal도 별도의 실제 이전 기준으로 함께 보고한다.

| 항목 | S2-CONTROL | S2-MOTION-LOW |
|---|---|---|
| 초기 model state | 같은 BASE-NOM terminal | 동일 |
| 추가 budget | 2,000 update | 동일 |
| optimizer/sampler | 새 optimizer, 같은 seed와 같은 시작 위치 | 동일 |
| LR 제안 | backbone 1e-6, 나머지 1e-5; warmup 100 후 같은 cosine | 동일 |
| motion auxiliary weight | 0.2 유지 | 0.02 |
| PREFIX / LEN / occ / lane | 1 / 0.25 / 0.2 / 0.2 | 동일 |
| B16·fixed BN·flip·데이터·입력 정책 | 기존 DEV | 동일 |

0.02와 LR/budget은 실험 시작을 위한 고정 선택이지 측정된 최적값이 아니다.
motion encoder/state/history 경로를 삭제하거나 raw status를 planner에 연결하지 않는다.
제공 status는 기존 shared scene query에만, pose는 기존 영상 정렬에, goal은 기존 scene 조건에 사용한다.
occ/lane의 공통 인지 학습도 유지한다. 이 변경은 A3 입력 경로를 재도입하지 않는다.

**resume 주의:** 기존 trainer는 in-epoch sampler의 정확한 복원을 약속하지 않고
`resume_sample_order_exact=False`를 기록한다. 예전 중간 checkpoint를 이어 학습한 한 arm을
원래 terminal과 비교해 완전히 같은 sample-order control이라고 부르면 안 된다.
두 arm을 같은 terminal에서 같은 fresh sampler/optimizer로 시작하고 row hash를 비교해야 한다.
따라서 “기존 terminal 대비 개선”과 “새 control 대비 loss 변경 효과”를 별도로 판정한다.

판정은 고정 2,000 update의 final PREFIX, 일반 주행 PREFIX, 첫 2초 점수 기여,
session별 paired delta를 함께 사용한다. 작은 V0 차이를 새 최고 성능으로 과장하지 않는다.
명확한 추가 이득이 없으면 이 가설을 닫고 loss 계수 여러 개를 무기한 탐색하지 않는다.
같은 terminal에서 출발하는 control이 필요하므로 기존 LONG 결과로 이 실험을 대체하지 않는다.
LONG은 시작부터 더 긴 cosine schedule로 학습한 다른 설정이었다.
현재 제안은 추가학습량과 후반의 학습 목표 비중을 분리한다.

## 그다음 표현 변경 후보: front motion matching을 더 세밀하게

큰 구조 변경이 필요하다면 attention head를 더 늘리기보다,
현재 raw-image motion branch가 pooling 전에 만드는 대응 특징의 해상도/변위 표현을 바꾸는 편을 검토한다.
현재 motion branch는 stride 8/16 두 level의 local correlation을 만들고, fuse 후 12×16으로 줄인다.
공통 ResNet/FPN 구현에는 stride 4 `use_p1` 기능이 있지만 MotionDrive motion encoder의 두-level 전제,
초기화, 메모리, 검색 범위를 함께 바꿔야 하므로 옵션 하나 켜면 끝나는 구현은 아니다.
기존 stride 8에서 radius 4라면 stride 4에서 같은 입력 pixel 범위를 보려면 radius 8이 필요하다.
coarse-to-fine matching이나 세밀한 branch의 잔차 추가는 후보이지 이번에 검증된 개선책이 아니다.

이 방향의 근거는 MR에서 대응 표현 변경이 실제로 유효했다는 기록과 현재 구조다.
옛 C selector의 `use_p1` 주석에 나온 후보 판별률을 A2 planning 성능 근거로 가져오지 않는다.
정확한 v0 숫자 복원을 선행조건으로 두지 않고 연속 motion feature를 최종 PREFIX와 연결한다.
기존 backbone은 이미 nuImages detection 공개 초기값을 사용했다.
따라서 “지금 처음 주행 도메인 pretraining을 쓰자”는 설명도 사실과 다르다.

## 병행 확보와 증거의 한계

A2 FULL의 raw-input 배포 정합성 및 제출 패키징은 별도로 진행할 가치가 있다.
이는 성능 개선 실험을 그만두자는 뜻이 아니며 FULL 서버 성능을 예측할 근거도 아직 없다.
후반 곡선이 평평한 것도 cosine LR가 0으로 내려간 상황이라 표현 능력의 상한 증거로 쓰지 않는다.
이번에는 새로운 성능 향상을 얻은 것이 아니라, 일부 즉시 수정 가설을 기각하고 다음 대조를 좁혔다.
0.15 또는 0.13 도달을 이번 작은 gradient 검사로 약속하지 않는다.

## 재현과 기록

- `experiments/md_a2_bottleneck_review_20260918/diagnose_inference.py`
- `experiments/md_a2_bottleneck_review_20260918/inspect_training_pressure.py`
- 이 폴더의 `precision.json`, `fp32.json`, `side.json`, `training_pressure.json`
- `side_base_intervention_source.txt`: in-memory 개입의 정확한 생성 소스
- `artifact_index.json`: 사용 checkpoint, script, local prediction, 로그의 경로/hash

진단 실행은 저장 checkpoint만 읽었고 production 코드는 변경하지 않았다.
초기 inference 시도에는 동적으로 생성한 함수의 indentation 오류가 있었고 이를 수정했다.
다음 시도는 full1998 forward 후 너무 엄격한 기존 예측 replay 최대 오차 1e-4 검사에서 중단됐다.
최종 replay 최대 좌표 차이는 BASE 0.000127m, SIDE 0.000104m이며 PREFIX 차이는 각각
-5.2e-9/+1.6e-8다. 최종 guard는 max<=5e-4와 score 차이<1e-6을 모두 요구했다.
재시도 로그도 remote에 보존하며 이를 학습 실패나 모델 수치 불안정으로 기록하지 않는다.

원래 실행 명령은 repo root에서 해당 GPU를 명시했다:

```bash
CUDA_VISIBLE_DEVICES=0 /home/korea_sdv01/cv2env/bin/python experiments/md_a2_bottleneck_review_20260918/diagnose_inference.py --kind precision --gpu 0
CUDA_VISIBLE_DEVICES=1 /home/korea_sdv01/cv2env/bin/python experiments/md_a2_bottleneck_review_20260918/diagnose_inference.py --kind side --gpu 1
CUDA_VISIBLE_DEVICES=2 /home/korea_sdv01/cv2env/bin/python experiments/md_a2_bottleneck_review_20260918/diagnose_inference.py --kind fp32 --gpu 2
CUDA_VISIBLE_DEVICES=3 /home/korea_sdv01/cv2env/bin/python experiments/md_a2_bottleneck_review_20260918/inspect_training_pressure.py
```

기존 근거:
`reports/md_a2_scene_extensions_20260918/TERMINAL_REVIEW_KO.md`,
`reports/md_exp_diagnosis_20260915/MR_ROUND2_RESULTS_KO.md`,
`MR_ROUND3_RESULTS_KO.md`,
`reports/md_progress_residual_20260917/DIRECTION_RECHECK_KO.md`,
`reports/md_shared_dynamics_20260917/RESULTS_20260918_KO.md`.
