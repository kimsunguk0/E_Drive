# A2 BASE/MH4 판정 및 SIDE-SCENE·QREFINE 독립 비교

## 본 학습 시작 — 2026-09-18 18:18 KST

두 run은 시작됐고 18:20 KST snapshot에서 SIDE150 / QREFINE200 update가 정상 진행됐다.
GPU1/PID3341915와 GPU2/PID3341916이며 source hash·초기 tensor·BASE 데이터 순서가 일치한다.
첫 평가 예상은 오늘18:50~18:52, 최종 비교는21:25~21:40 전후다. 초기 속도 기준 추정이다.
구현 commit은 a572e233288347b03aff9a13d2b1150819103a16, 미러 구현은165f760이다.
정확한 시각과 이후 상태는 아래 launch receipt 및 실제 metrics를 우선한다.

## 판단

동일 초기값·학습 예산의 BASE/MH4 terminal 결과는 다음과 같다.

| 모델 | 학습 update | V0 PREFIX |
|---|---:|---:|
| A2-BASE-NOM-s1 | 20,554 | 0.1655106489655904 |
| A2-MH4-NOM-s1 | 20,554 | 0.16449543014956494 |
| 이전 A2 terminal, 배포형 nominal 입력으로 재평가 | 기존 완료 계보 | 0.1644553140831734 |

MH4의 matched BASE 대비 이득은 0.00101521881602545, 약 0.6134%다.
11개 session 중 9개에서 개선됐지만 이전 A2의 동일 입력 평가보다 0.000040116066 높다.
새로운 최고점이나 0.15대로 향하는 큰 돌파로 보지 않는다. 이번 단일 seed의 작은 차이를
구조의 확정적 우월성으로 일반화하지 않는다. MH4 추가학습/FULL보다 다른 관측·조회 축을 먼저 본다.

BASE→MH4에서 first2s 점수 기여는 0.123746884→0.123055088,
nonstop 평균은 0.169590813→0.168698109다. 같은 GT tangent mask의 종방향 절대오차는
0.144607966→0.143339287, 횡방향은 0.061086022→0.061164626이다.
주된 오차가 크게 해소됐다는 근거가 없다. 감가속 타이밍을 근본 원인으로 확정하지도 않는다.
원본: `../md_a2_nominal_mh4_20260918/dev_terminal_summary.json` 및 `dev_terminal_records/`.

## 실행할 두 독립 arm

사용자는 SIDE-SCENE와 별도로 QREFINE도 독립 1회 실행하라고 명시했다.

| GPU | Run | 바꾸는 것 | 유지하는 것 |
|---|---|---|---|
| 1 | A2-SIDE-SCENE-NOM-s1 | FL/FR의 -0.1/-0.5초 영상 4장을 shared scene에 추가 | single-head, native front H4 motion |
| 2 | A2-QREFINE-NOM-s1 | 첫 영상 조회 결과로 query를 갱신한 뒤 두 번째 조회 | BASE의 영상·projection·motion 입력 |

두 run은 각각 r0_init_tplus.pth에서 새 optimizer로 시작한다. BASE terminal이나 FULL 가중치에서
이어 학습하지 않는다. 공통 초기 tensor SHA는
`116f67a476b5ab519ec4384d6e15d5f5e36e83d1c89812f733b056ba5874593f`다.
train310 83,700행 / V0 1,998행, seed1, 20,554 update, 3,426 update마다 평가,
effective batch16/microbatch8, BF16, fixed BN, backbone LR5e-6/나머지5e-5,
warmup200, flip0.5, LEN0.25와 기존 loss를 유지한다.

SIDE는 CONTROL H4 `(1,2,5,10)`의 pose index `0,2`를 쓴다. Camera ID는
left2/right1이며 각 카메라 calibration·과거 transform·시간 metadata를 사용한다.
같은 카메라의 현재/과거 색상 증강을 일치시키고 flip에서 좌우 카메라도 교환한다.
추가 영상은 shared scene에만 들어가 occupancy·lane·planner가 함께 읽는다.
기존 image-only motion/state/history 입력은 그대로다. 공동 backbone 학습까지 동결한다는 뜻은 아니다.

QREFINE은 `S0=Read(q0,K,V)`, `q1=q0+MLP(q0,S0)`,
`S=S0+W·Read(q1,K,V)`로 구현했다. W는 bias 없이 0 초기화하고 query MLP는 일반 초기화한다.
따라서 처음 BASE 함수를 보존하면서 첫 출력층 update 뒤 query MLP도 학습한다.
새 parameter는 28,768개다. 첫 시각 조회가 두 번째 query에 실제 영향을 주며,
같은 attention을 동일 query로 반복 호출한 구조가 아니다.

입력 경계는 A2를 유지한다. Provided status는 scene query만 조건화하고,
새 query/status를 scene value나 motion/state/history에 직접 더하지 않는다.
Goal은 기존 scene query에서 계속 간접 사용하므로 goal-free 모델이라고 부르지 않는다.
이는 새 구조에 대한 별도 운영국 승인을 받았다는 의미가 아니다.

Command/회전 전용 학습, MH4와의 결합, progress residual, A3 status gate는 이번 두 run에 없다.
회전은 동일 평가의 진단 열로 보고, 주 판단은 전체 PREFIX·first2s·nonstop·종/횡 오차로 한다.

## 실행 전 확인

`preflight.json` 및 `smoke_summary.json`이 원본이다.

- 177,872개의 필요한 측면 과거 이미지 파일이 존재한다.
- 실제 입력 FP32/BF16에서 QREFINE 0 초기화 및 SIDE 비활성화 출력이 BASE와 정확히 같다.
- 완료된 BASE checkpoint의 저장 예측 재현 차이도 0이다.
- 측면 네 입력 모두 최종 planning loss의 gradient를 받는다.
- QREFINE은 첫 step에서 출력층, 출력층 1회 update 후 query MLP에 유한한 gradient가 들어온다.
- Status 또는 추가 SIDE 이미지 교환 시 같은 forward의 motion/state/history 변화는 0이다.
- 투영 mirror 좌표 차이 최대 5.96e-7. 기존 `[0,W)` mask의 경계 1픽셀 차이는 별도 확인했다.
  이 비교 중 기존 BASE의 경계 처리 함수를 바꾸지 않았다.
- 두 arm 모두 2-step+전체 V0 smoke 완료, nonfinite 0, sample order 일치.
  QREFINE 첫 step의 모든 loss 항은 BASE와 정확히 같다.
- Smoke PREFIX 0.57249(SIDE)/0.47751(QREFINE)는 실행 확인용이다. 성능 판정에는 사용하지 않는다.

## FULL 후보 및 후속 판정

A2-FULL-NOM-s1은 101,520행/376 scene, 24,931 update를 완료했다.
Checkpoint는 `work_dirs/md_a2_nominal_mh4_20260918/A2-FULL-NOM-s1/ckpt_step24931.pth`다.
최종 진단 PREFIX 0.08893408501920225는 **학습 포함 V0의 in-fit 값**이며 제출 점수가 아니다.
완료 manifest, 원본 metric, checkpoint hash는
`../md_a2_nominal_mh4_20260918/full_terminal_summary.json` 및 `full_terminal_records/`에 기록한다.
A2 제출 adapter의 실제 raw-input 재현 인증/패키징은 별도 남은 작업이다.

공식 서버에서 확인된 MR FULL 점수는 0.18596892793122946이다. A2 FULL 서버 점수는 미측정이다.
DEV에는 FULL 가중치·teacher·feature cache를 가져오지 않는다.

첫 3,426 update 평가는 진행 확인이며 최종 비교는 같은 20,554 update에서 한다.
원래 BASE 0.165510648966와 이전 A2 nominal 0.164455314083을 함께 보고한다.
두 새 arm이 실제로 좋아지기 전에는 새 FULL이나 결합 모델을 자동으로 시작하지 않는다.

본 학습 PID·시각은 `launch_receipt.json`, 실행 건강 상태와 속도는 `launch_health_snapshot.json`을 본다.
stdout은 `runtime/<ARM>-s1.log`, 체크포인트·예측은
`work_dirs/md_a2_scene_extensions_20260918/<ARM>-s1/`에 저장한다.
