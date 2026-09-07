# MotionDrive V2 구현·실행 기록 — 2026-09-07

작업 서버: B200 `/NHNHOME/data/sukim/adcl`.
기준 설계: `ETRI_MOTIONDRIVE_V2_EXECUTION_SPEC.md`, 최신 `OPEN_ISSUE.md`.

## 현재 결론

모델·데이터·학습·평가·감사·3090 측정 코드를 구현했고, 실제 데이터 파일럿을 실행했다.
현재 단계는 **P0 공통 인지/영상 상태 사전학습 및 작은 표본의 경로 피팅 검증**이다.
P1 G×S 네 판은 아직 실행하지 않았다. 1위권 정확도나 일반화 성공을 입증한 상태가 아니다.

소스 커밋:

- `bd38aa8`: 새 모델과 데이터/학습/감사 코드 구현.
- `fcab96e`: 실제 timestamp 세션 분리, 체크포인트 설정 복원, 감사 회귀 테스트 보강.

기존 dense/sparse 챔피언, 기존 실험 및 사용자 작업 디렉터리는 수정·삭제하지 않았다.
GPU 4–7은 사용하지 않았으며 GPU 6의 기존 작업을 중지하지 않았다.

## 구현한 모델

입력은 현재 6 camera × 768×432, 과거 전방 4장 × 384×216이다.
과거는 frame offset 1/2/5/10이며 실제 timestamp 간격도 전달한다.

1. 공개 nuImages 계열 R50 trunk를 로드하고 128-channel compact FPN은 새로 학습한다.
   실제 로드 결과 trunk tensor 318개, missing/unexpected 0. 기존 ETRI full330 가중치는 사용하지 않는다.
2. 64×48 spatial scene에서 실제 image key/value attention을 수행한다.
   goal은 이 공통 scene의 attention condition에만 들어간다.
   같은 scene raster에 현재 객체 footprint와 실제 lane polyline head를 연결했다.
3. goal/pose 정렬 전 영상 correspondence에서 motion feature 및 history/state를 예측한다.
   제공된 상태 정답은 이 forward 경로에 들어가지 않는다.
4. planner는 연속적인 scene/motion feature와 예측 상태를 읽고 FP32 6-waypoint absolute XY를 출력한다.
   planner query에 goal을 직접 넣거나 외부 goal selector로 좌표를 보정하지 않는다.
5. G/S OFF에서도 파라미터 수는 동일하다. S OFF는 예측 상태 전달만 마스킹하며
   raw motion feature 및 상태 auxiliary는 유지한다. 초기 5초 tail loss는 0이다.

## 데이터와 새로 발견한 정정

정식 공통 P0 경로:

```text
data/etri/motiondrive_v2/grouped_split_rawtime.json
data/etri/motiondrive_v2/train_tune_rawtime/
```

| 분리 | 시나리오 | 실제 시간 세션 | 학습용 라벨 프레임 |
|---|---:|---:|---:|
| train | 203 | 72 | 54,810 |
| tune | 37 | 11 | 9,990 |
| val 격리 | 136 | 31 | 이번에 생성하지 않음 |

기존 val38은 val 격리 집합의 보고 전용 부분집합이다. 시간 그룹이 지리적 route 분리를
보증하지 않으며 이미 살펴본 historical validation을 untouched라고 부르지 않는다.

초기 파일명 시간 파싱에는 5자리 시각 표기/문자열 정렬 문제가 있었다.
원본 frame0 timestamp를 시작, frame299 timestamp + median dt를 끝으로 사용하고,
다음 시작까지 60초 이내인 main clip을 같은 세션으로 정정했다. 미래 frame349는 끝 시각에 쓰지 않는다.
기존 그룹의 과병합 4개와 분절 1개를 바로잡았다. 실제 세션의 train/tune/val 교차는 0이어서
**203/37/136 시나리오 membership은 재추첨하지 않고 그대로 보존**했다.

전체 64,800프레임 검증:

- 원본 full SE(3)에서 재계산한 future와 기존 GT 최대 차이 `2.33e-10m`.
- goal 최대 차이 `3.815e-6m`, state/history 유효율 100%.
- 필요한 camera/frame 이미지 391,200개 존재, 누락 0.
- 기존 calibration과 bitwise 동일. 실제 loader, row/frame 및 provenance 검사 통과.
- 5개 train scene의 median 간격은 약 0.10624초다. nominal 10Hz로 강제하지 않는다.
  history/state에는 실제 시간간격을 쓰되 기존 공식/cache future 라벨은 재표본화하지 않는다.
  따라서 모든 frame+50을 물리적으로 정확한 5.000초라고 주장하지 않는다.
- 과거 train330+val38 밖의 8개는 기존 minival이다. 새 train에 포함된 4개도 유효성 검사를 통과했다.

기존 pilot 및 초기 manifest/cache를 보존했다. 새 supervision의 NPZ는 동일 target bytes의
hardlink이고, 새 provenance 보고서에 원본과 재연결 이유를 기록했다. hardlink 파일은 이후에도
직접 수정하지 않는 불변 아티팩트로 취급한다.

객체 target은 **annotation이 있는 유효 객체 footprint**다. 음성은 보수적 가시/support 영역에서의
annotation completeness 가정이며 물리적 free-space 인증이 아니다. unknown cell을 음성으로
강제하지 않는다. 실제 map.parquet lane polyline을 사용하며 좌표계가 다른 hd_map은 혼용하지 않는다.

근거:

- `reports/motiondrive_v2_grouped_split_rawtime.json`
- `reports/motiondrive_v2_train_tune_data_validation.json`
- 서버 `data/etri/motiondrive_v2/rawtime_split_audit.json`

## 실행 검증

B200에서 모델/데이터/학습/감사 테스트 **35개 통과**.
검사에는 GT 입력 whitelist, 정확한 cumulative ADE 가중치, invisible cell 처리,
raw motion의 goal/정렬 불변성, FP32 head 실제 입력 dtype, 실제 shared feature gradient,
checkpoint G/S 복원 및 NumPy RNG를 포함한 checkpoint 로드가 들어간다.

resume/eval에서 Python 설정인 G/S가 state_dict에 없다는 문제를 수정했다.
training evaluator는 저장된 설정을 복원하며 모순된 명시 옵션을 거부한다.
감사 도구의 의도적 G/S 변경은 명시적 ablation으로 원래값/적용값을 기록한다.
부분 trunk 초기화는 실패 처리하고, worker epoch가 갱신되지 않는 augmentation 문제도 수정했다.

### 3090 실제 입력 전체 forward

실제 `20260112-105434/frame30`, 현재 6장 + 과거 전방 4장.
batch 1, R50, BF16 backbone, FP32 최종 heads, warmup 20, 반복 50.
CUDA event와 `torch.cuda.synchronize()` 및 wall-time을 사용했다.
모든 영상 encoder, motion, scene, state, occupancy/lane, planner를 포함한다.
파일 읽기/모델 밖 전처리/H2D는 제외하며 사전 image feature cache를 쓰지 않는다.

| 구성 | CUDA median | p95 | p99 | 실제 입력 graph 감사 |
|---|---:|---:|---:|---:|
| G1S1 | 31.7722ms | 31.8218ms | 31.8836ms | 42/42 PASS |
| G0S0 | 31.4301ms | 31.4843ms | 31.5280ms | 43/43 PASS |

Peak allocated 402.63MiB, reserved 498MiB. PyTorch 2.7.1+cu128 / torchvision 0.22.1.
실제 calibration에서 기하적으로 가시한 scene cell이 존재함을 확인했다.
이는 기하 정합의 모든 시각 검증이나 인지 품질을 보증하지 않는다.

**Random weights의 실제-input 개발 실측**이다. 학습 완료 checkpoint/최종 wrapper/공식 4090
시간의 보증이 아니며, 3090→4090 스펙 비율 환산으로 공식 시간을 확정하지 않는다.
감사 PASS도 운영국의 최종 구조 승인과는 구분한다.

근거: `reports/{latency,audit}_3090_r50_real_{g1s1,g0s0}.json`.

### 완료한 200-step smoke 파일럿

학습 2개 인접 scene에서 64프레임, 다른 tune scene 12프레임만 사용한 작은 시험이다.
두 판 모두 public trunk에서 시작해 200 step 완료, 유한 gradient 및 체크포인트 저장을 확인했다.

auxiliary 전용 판의 작은 tune probe:

| 지표 | 초기 | step 200 |
|---|---:|---:|
| 객체 footprint IoU | 0.0192 | 0.2573 |
| lane IoU | 0.1995 | 0.3391 |
| vx MAE, m/s | 11.844 | 2.142 |
| 1초 근처 history 위치 MAE, m | 9.881 | 2.128 |

경로까지 연결한 joint smoke의 tune D3는 13.649→7.807m, 같은 64프레임 train 재평가는 7.302m다.
**아직 경로 피팅 자체가 덜 된 상태**이며, 이 결과를 수상권 성능이나 일반화 증거로 포장하지 않는다.
auxiliary loss의 음수는 normalized Gaussian NLL의 log-variance 항에서 가능하며 NaN을 의미하지 않는다.
손실 숫자 외에 실제 물리단위 오차와 IoU를 함께 평가한다.

## 현재 시작한 P0 작업

아래는 실행 시작 기록이며 실시간 진행률은 각 manifest/metrics.jsonl을 읽어야 한다.
두 작업 모두 코드 `fcab96e`, rawtime split SHA
`f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`를 사용한다.

| GPU | run | 목적 | 설정 |
|---|---|---|---|
| 0 | `p0_common_rawtime_s0` | 공통 인지/영상 상태 초기값 학습 | train54,810, tune1,998(stride5), batch16, 2,000 steps |
| 1 | `p0_joint_fit_rawtime_s0` | 3초 planner의 작은 표본 피팅 확인 | train16, tune12, batch8, 1,500 steps |

경로는 `work_dirs/motiondrive_v2/<run>/`, 로그는 `logs/motiondrive_v2/<run>.log`.
시작 명령/PID는 `logs/motiondrive_v2/p0_rawtime_launch.json`에 기록했다.
파일럿 가중치를 공통 초기값으로 재사용하지 않고 새 public 초기값에서 시작했다.
AdamW backbone LR 1e-5/new head LR 1e-4, weight decay .01, BF16, clip 5.
common은 250 step마다 전체 tune37의 stride5 표본을 평가한다. val은 checkpoint 선택에 쓰지 않는다.

## 다음 판정과 실행 순서

1. 공통 P0의 실제 history/state 오차, footprint/lane 품질과 finite gradient를 확인한다.
   상수 상태 baseline, temporal shuffle과도 대조해야 영상에서 motion을 읽는다고 주장할 수 있다.
   현재 사전학습 best 선택은 history 위치 MAE이므로 인지 품질을 별도로 확인해야 한다.
2. 작은 표본의 3초 경로가 충분히 fitting되는지 확인한다. underfit 상태에서 일반화 실패로 결론내지 않는다.
3. 적합한 공통 P0 checkpoint **하나의 SHA를 고정**하고 GPU0–3에 G0S0/G1S0/G0S1/G1S1을 복제한다.
   모든 팔은 같은 split, seed, 영상, 초기값, 스케줄을 사용한다. 아직 이 네 판을 자동 실행하도록 예약하지 않았다.
4. 튜닝 지표는 proxy sample weight 없는 공식 cumulative D3이다.
   과거 full330 모델 수치나 다른 split/proxy 가중 수치와 직접 동률 비교하지 않는다.
5. 유망한 조합만 seed 복제와 실제 checkpoint 3090 전체 forward 재측정을 한다.
   최종 val과 공식 환경으로 넘어가기 전에 규정 정보 흐름을 다시 점검한다.

현재까지는 검증 가능한 구현과 실행 시작을 완료한 것이다. 최종 정확도, 주력 모델 채택,
리더보드 순위는 아직 판정하지 않았다.
