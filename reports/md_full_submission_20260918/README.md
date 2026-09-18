# MR-NATIVE-FULL-s1 제출 후보 — 2026-09-18

완료된 MR FULL terminal 가중치를 기존 MR 제출 경로에 연결한다. 이 작업에서는
재학습, 추가 fine-tuning, 후보 선택, 서버 업로드를 하지 않았다.

**사용자 제출·채점 확인: FULL 공식 L2_avg 0.18596892793122946.**
기존 MR 0.19798776670488366 대비 6.0705% 개선됐다. 원문 응답과 설정 비교는
SERVER_RESULT_20260918_KO.md 및 server_comparison_20260918.json에 기록했다.
공식 test 1,125/1,125, 누락·잉여·NaN/Inf 0, clip 재실행 차이 0 검증은 그대로다.
completion.json은 업로드 전 파일 제작 시점의 기록이다.

## 제출 파일

- 실제 업로드 파일: `submission/submission.zip`
- ZIP 내부: `submission.json` 하나
- JSON: 공식 test clip 1,125개와 정수 `__flops__` 한 개
- 각 clip: 현재 ego 좌표계의 누적 절대 XY 6개, 0.5초 간격으로 3초까지
- 모델 출력에 cumsum, 위치 보정, 상태 기반 후처리를 추가하지 않는다.
- `completion.json`에 최종 파일 SHA256, 준비 완료 여부, 미업로드 상태를 기록한다.

## 가중치와 학습 계보

| 항목 | 값 |
|---|---|
| 모델 | MotionDrive V2, MR native 768×432, correlation radius 4 / C32 |
| 학습 데이터 | 전체 376 unique scenes / 101,520 rows |
| 선택 checkpoint | 사전에 정한 terminal step 24,931 |
| 추가 학습 | 0 update |
| 원본 | `work_dirs/md_progress_residual_20260917/MR-NATIVE-FULL-s1/ckpt_step24931.pth` |
| 별도 보존본 | `work_dirs/md_full_submission_20260918/preserved/ckpt_step24931.pth` |
| checkpoint SHA256 | `ce569da5d3cfe70f89f042faa2a5f6449e30fccbd469f2e689e26445accf2b87` |
| model tensors SHA256 | `740881daf0412117b486f044c3831ef43b76de1e0c344af9bf5396b86f716cf4` |

Checkpoint 해시는 이전 `records_20260918/experiment_index.json`에 보존한 FULL terminal
해시와 같다. 원본과 별도 보존본의 파일 해시도 대조한다. 보존본은 쓰기 권한을 제거했다.
학습 설정과 완료 기록은 `training_experiment.json`, `training_manifest.json`에 복사했다.

**이 모델의 V0 0.097245는 학습에 포함된 행의 in-fit 진단이다.** Held-out 또는 서버
성능으로 해석하지 않는다. 기존 DEV MR의 서버 0.197988을 이 FULL 모델의 점수로
기재하지도 않는다. 사용자가 전달한 FULL 공식 점수는 **0.18596892793122946**이다.

## 제공 status가 없는 실제 추론 경로

현재 6개 카메라와 전방 과거 4장(frame -1, -2, -5, -10)을 사용한다.

- Motion: 정렬 전 현재·과거 영상 특징과 고정 시간 간격 → motion feature 및 예측 state/history.
- Scene: 영상 특징 + calibration + pose 기반 과거 영상 정렬 + goal 조건 → 공통 scene feature.
- Planner: scene feature + motion feature + 영상에서 예측한 state/history → direct XY.

제공 pose에서 속도·가속도·yaw rate를 계산해 입력하는 경로는 없다.
A2/A3의 shared status query·channel gate, progress residual도 없다.
`state_on=true`는 영상에서 예측한 상태를 사용한다는 뜻이며, 제공 status를 사용한다는 뜻이 아니다.
Pose 정렬과 goal의 공통 scene 조건은 기존 제출 경로대로 사용한다.

실제 model 입력 key는 다음 8개로 제한하고 공식 test 각 clip에서 검사한다.

```
images, history_images, lidar2img, history_transforms,
time_offsets, goal_xy, motion_current, motion_history
```

공식 test 추론은 raw JPEG와 calibration/ego_pose parquet를 사용한다.
Command, 제공 status, training supervision, 정답 미래 궤적은 추론 입력으로 사용하지 않는다.
Goal로 제공된 +50 frame의 위치는 사용하며 미래 자세는 사용하지 않는다.
이는 구현의 입력 경로 기록이며, 운영국의 별도 구조 승인을 의미하지 않는다.

## 검증 기록

| 검사 | 기록 |
|---|---|
| FULL 가중치의 raw/cache 입력·출력 parity | `raw_b1_fixture8.json`: 8개 fixture 모두 bitwise 일치, 좌표 차이 0 |
| 기존 채점 제출 경로 재현 | `previous_submission_replay.json`: 첫 공식 clip의 기존 MR 예측과 정확히 일치 |
| 공식 counter의 FULL forward FLOPs | `flops.json`: 729,815,613,824 = 729.816G, cutoff 7,053G 통과 |
| 전체 clip 추론과 clip 간 상태 분리 | `official_test_MR-NATIVE-FULL-s1.validation.json` |
| 공식 clip 집합·6×2·유한값·ZIP 구조 | `submission/manifest.json`, `completion.json` |
| 실행 명령과 소요 시간 | `execution.json`, 각 stage의 `.log` |
| 코드 fingerprint / 환경 | `source_manifest.json`, `environment.json` |

이번 작업에서 RTX4090 시간을 새로 측정하지 않았다. 실행 로그의 벽시계 시간은
파일 읽기·영상 전처리를 포함하며 공식 model-forward latency로 해석하지 않는다.

## 재현

B200 저장소 루트에서 기존 cv2 환경으로 실행한다. 기본 출력 폴더가 이미 완성돼 있으면
덮어쓰기를 거부하므로 재현할 때는 별도 `--out-dir`을 지정한다.

```bash
CUDA_VISIBLE_DEVICES=0 ~/cv2env/bin/python \
  experiments/md_full_submission_20260918/prepare_mr_full_submission.py \
  --gpu 0 --clips-root /tmp/etri_test \
  --out-dir reports/md_full_submission_replay
```

준비 스크립트는 완료 checkpoint를 검사·보존한 뒤 raw parity, FLOPs, 기존 제출 재현,
FULL 전체 test 추론, ZIP 포장을 수행한다. GPU 0~3 중 하나만 지정할 수 있다.
서버 업로드 기능은 없으므로 제출 횟수를 소모하지 않는다.
