# C 후보 보존 및 장면 특징 선택기 검증 — 2026-09-10

사용자 요청: C 약 D3 0.314에서 속도 후보 조기 탈락을 개선하고 후보별 256차원 장면 특징을 이용한 최종 선택기를 검증한다. 0.1대 성능을 확인하면 기존 약 0.584 epoch 이후로 전체 모델을 이어 학습한다. 정확한 영상 v0 복원은 필수 조건으로 삼지 않는다.

## 고정 조건

- 원 TRAIN203 54,810행 / TUNE37 1,998행을 유지한다. 새 분할을 만들지 않는다.
- C terminal 체크포인트 `temporal_c_common_s0_v1/last.pth` SHA256 `b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff`를 보존한다.
- public SparseDriveV2, train-only P1024×V1024 bank, 원 이미지/과거 프레임 설정을 유지한다.
- 실제 과거 pose 기반 상태는 공통 시간 영상 인지의 query 조건으로만 사용한다. planner와 기존 relative selector의 직접 상태 입력은 0이다. goal은 완성된 후보의 최종 선택에만 사용한다. 신규 head는 원시 상태를 받지 않는다.
- GPU 0, 1, 4만 사용한다. 다른 GPU의 별도 작업을 중단하거나 변경하지 않는다.
- 원 체크포인트/소스/결과는 수정하지 않고 새 파일 및 실행 디렉터리로 기록한다.

## 순서 및 판정

1. 고정 C에서 V1024→64→10의 단계별 oracle 손실을 분리한다. P는 최종 유지 P20으로 고정하고, V10/V64/all1024의 oracle을 같은 forward 뒤에 계산한다. GT는 이 진단에만 사용하며 candidate generator/selector의 입력에 넣지 않는다.
2. 필요 범위에서 실제 forward의 velocity_filter를 (64,32), (64,64), (128,32), (128,64)로 비교한다. 후보 수 변경에 따른 fine attention 및 점수 변화도 실제로 평가한다. 모든 V1024를 fine transformer에 투입하는 것은 기본 실행 계획에 포함하지 않는다. 비용과 oracle을 보고 실제 학습할 한 구성을 결정하고 근거를 기록한다.
3. 후보별 마지막 fine trajectory MLP 직전 256차원 특징을 추출한다. C가 고정된 동안 TRAIN/TUNE의 이미지 augmentation 없는 특징을 저장한다. 토큰·점수·후보 ID와 GT cost는 별도 배열로 보존하고, head inference 입력은 명시적으로 제한한다. 이는 추가로 backbone을 학습한 epoch가 아니다.
4. 동일 구조의 real-token / zero-token 최종 residual head를 비교한다. 초기 head 파라미터, 배치 순서, loss, 학습률, 횟수를 일치시킨다. 신규 마지막 선형층은 0으로 초기화해 학습 전 원 C 결정을 재현한다. 최초 계획: seed 0, batch 128, 4,000 step, AdamW lr 1e-3, wd .01, warmup 100 + cosine, D3 soft-label CE temperature .1, 500 step 간격 고정 평가. 원 C와 head-only 학습 예산을 구분한다.
5. 0.1대(우선 D3 < .20, 원 희망값 .15도 별도 표기)에 도달한 구성이 있으면 명시적 C/신규 head 가중치 복원과 새 LR 일정으로 전체 모델을 추가 학습한다. 기존 trainer의 steps만 늘리면 초기 public 가중치에서 재시작하므로 그 방식은 사용하지 않는다. optimizer 복원 또는 새 구성의 초기화 정책을 기록한다.
6. 부족하면 oracle, selector regret, 전체 TRAIN/TUNE의 학습 추이를 보고 병목에 대응한다. 단순히 실험 횟수를 늘려 TUNE 최저값을 찾는 것을 성과로 보고하지 않는다.
7. 최종 구성은 실제 모델 B1 평가, 원본 입력 경로, goal/상태 경계, bank row 보존 및 전체 forward 비용을 검증한다. 캐시 head 평가 시간은 전체 모델 latency가 아니다.

## 기존 수치와 해석 제한

- C B8 D3 .313322872, 현재 P20×V10 oracle .164611728.
- 같은 최종 P20에 모든 V1024를 붙인 GT oracle .067896052. 이는 현재 200개 후보의 결과나 실제 추론 점수가 아니다.
- C B1 D3 .313933327. B1/B8의 미세한 수치 차이가 후보 선택을 바꾸므로 최종 B1을 별도 측정한다.
- 과거 confirmation12의 32 scene은 현재 TRAIN203에 포함되어 이미 학습되었다. 이를 미사용 holdout으로 취급하지 않는다. reserve136은 그대로 둔다.
- TUNE는 반복 탐색에 사용되며 이번 실험도 독립 blind validation이 아니다. 단일 seed 비교의 한계를 명시한다.
- C는 사용자와 합의한 공통 인지 상태 조건 구조를 유지한다. 규정 구조에 맞춘 구현 판단과 주최 측의 최종 승인을 구분한다.

## 후보 수 결정 — head 학습 전

원 C B8 TUNE 전체를 그대로 재현한 단계별 진단에서 P20×V10 oracle .1646117276, P20×stage1 V64 oracle .0714597765, P20×allV1024 oracle .0678960515였다. 속도 pruning으로 생긴 oracle 손실의 96.315%가 마지막 64→10에서 생겼다. 이에 첫 1024→64는 유지한다.

실제 최종 V32 forward의 oracle은 .0895087854, 기존 head의 실제 D3는 .3239029684였다. 후보 확장만으로 점수는 개선되지 않았다. P20은 원형과 모든 행에서 동일했다. V32 모델 B8 p50은 약102.9ms, 원형은97.9ms였고 B1은 측정 변동 안에 있었다.

최종 head 학습 후보는 V64, 즉 **P20×V64=1,280개**로 정한다. V32보다 oracle 여유가 약 .018 있어 .15 목표에 유리하며 B200 메모리 안에서 다룰 수 있다. V64 전체 forward의 비용과 실제 D3는 별도 진단한다. (128,32)/(128,64)는 첫 pruning의 손실이 .00356으로 작아 이 단계에서는 실행하지 않는다. 후보 구성은 head 학습 전에 고정하며 GT로 샘플별 후보 수를 정하지 않는다.
