# MotionDrive V2 진행 기록 — 2026-09-07

## 15:48 KST: P0 보완 대조 시작

사용자 요청에 따라 장기 목표를 활성화하고 데이터/모델/독립감사 에이전트를 운영한다.
B200 `/NHNHOME/data/sukim/adcl`, GPU0–3만 사용한다. GPU6 기존 작업은 건드리지 않는다.
제출이나 외부 유료 자원 사용은 별도 승인 없이 하지 않는다.

코드·사전 판정 기준 커밋: `92fbdc6` (B200 테스트 86개 통과).
계획: `configs/motiondrive_v2/p0_repair_pairs_r1_s0.json`.
원본 실행 명세: `logs/motiondrive_v2/launch_p0_repair_pairs_r1_s0.json`.

| GPU | run | 단일 비교 변수 | 단계 |
|---|---|---|---|
| 0 | p0_tail_unit1_s0 | XY 출력 단위 (1,1), 대조 | joint 500 step, train16 fitting |
| 1 | p0_tail_unit10x5_s0 | XY 출력 단위 (10,5), 초기 함수 보존 | 동일 |
| 2 | p0_motion_highfeature_s0 | 현재 고해상도 feature 축소 | pretrain 1000 step, 전체 train/tune |
| 3 | p0_motion_lowfeature_s0 | 현재 영상부터 저해상도로 맞춰 추출 | 동일 |

학습 중간값을 최종 채택 결과로 읽지 않는다. 비교 기준은 별도 사전 프로토콜에 고정했다.
이 네 판은 P1의 G×S 실험이 아니다. P0 전제조건 보완을 먼저 판정한다.

## 이전 두 판의 종료 결과

- `p0_common_rawtime_s0`: 2000 step 완료, 사전 선택 지표인 전체 tune history MAE로 best step750.
- `p0_joint_fit_rawtime_s0`: 1500 step 완료, 미니 train16 평가 D3 0.5222.
  첫 2초에 비해 마지막 3초 waypoint 오차가 크다. query collapse나 gradient 단절은 확인되지 않았다.
- 공통 사전학습은 lane/객체 raster와 영상 조건부 속도를 학습했지만,
  정상 history가 현재 영상 반복/반전보다 실질적으로 우수하다는 증거는 아직 약하다.
- 같은 영상에 ±8px 수평 이동을 준 correspondence 진단에서는 feature 해상도 정합이
  올바른 이동 검출률 약13%→약96%를 만들었다. 실제 주행 속도·planning 이득의 증거는 아니다.

전체 감사 원본:
`reports/motiondrive_v2_p0_aux_audit.json`,
`reports/motiondrive_v2_p0_fit_audit.json`,
`reports/motiondrive_v2_correspondence_probe.json`.

## 실제 3090 측정의 범위

학습된 common best750, G0S0/legacy 원설정, 실제 영상 batch1:
CUDA median 32.1296ms, p95 32.1691ms, p99 32.1926ms, wall median 32.1484ms.
warmup20/repeats50, 전체 encoder+motion+scene+heads+planner 포함,
CUDA event와 synchronize 사용. 전처리/H2D 제외. 감사 43/43 통과.

이 숫자는 새 high_feature/low_feature 추가 current encoder 모드나 최종 G1S1의 측정값이 아니다.
채택 구조와 학습 체크포인트에서 다시 전체 forward를 측정한다. 4090 실측값이라고 환산하지 않는다.
원본: `reports/latency_3090_r50_p0_common.json`, `reports/audit_3090_r50_p0_common.json`.

## 아직 달성하지 않은 것

V2의 grouped holdout planning 경쟁력, P1 G×S 순효과, 복수 seed 재현성,
채택 모델의 전체 이미지 교란/출처 감사와 latency, 기존 제출 후보 대비 우위는 미확인이다.
P0 인지 점수나 작은 fitting 결과로 1위 가능성을 확정하지 않는다.

## 16:09 KST: 보완 결과 및 BN 정책 대조 시작

앞선 네 판은 모두 종료했다. 결과/후속 사전 기준 커밋은 `38ee977`이다.

| LAST500 정상 train16 | 단위(1,1) | 단위(10,5) |
|---|---:|---:|
| D3 | .28081 | .21418 |
| 3초 L2 | 1.96714 | .80705 |
| 첫2초 정규화 D3 | .16313 | .16296 |

출력 단위 변경은 초기 실제 좌표를1.526e-5m 이내로 보존했고, 앞부분 악화 없이 tail을 개선했다.
대조군 자체의 .52216→.28081은 추가 일정의 효과이며, scale의 순효과와 구분한다.
정상 D3<=.15는 두 판 모두 미달이다. train-BN .12012로 기준을 바꾸지 않는다.

Motion LAST1000 고정 tune370에서 low-high vx MAE 차이-.1480m/s,
11세션 paired CI[-.3082,+.0184], history 차이-.0316m, CI[-.1045,+.0414]다.
full tune1998 vx는1.3682→1.2702, history는.6182→.6025다.
시간반전 민감도의 일부 증거는 있지만 현재영상 반복 대조는 CI가0을 포함한다.
확정적 정상 개선이나 정확한 속도 관측 해결로 선언하지 않는다.

train16 영상만으로 shared BN을 표준 재보정했을 때 .21418→.21892로 개선되지 않았다.
가중치/원 checkpoint는 그대로다. 통계를 재평균하면 해결된다는 가설은 지지되지 않는다.
이후에는 **학습 중 BN 정책만** adaptive/fixed로 대조한다. 모든 가중치는 계속 학습한다.

| GPU | 현재 run | 시작점 | 일정 |
|---|---|---|---|
| 0 | p0_tail_bn_adaptive_s0 | unit10x5 LAST500 | 같은 train16, 500 step |
| 1 | p0_tail_bn_fixed_s0 | 동일 | 동일 |
| 2 | p0_motion_bn_adaptive_s0 | low_feature LAST1000 | 전체 train/tune, 1000 step |
| 3 | p0_motion_bn_fixed_s0 | 동일 | 동일 |

프로토콜 `MOTIONDRIVE_V2_P0_BN_PROTOCOL.md`에 후속 판정 기준을 사전 고정했다.
실제 초기 state SHA와 데이터 순서는 각 쌍 안에서 일치했다. P1 G×S는 아직 시작하지 않았다.

## P0 low_feature 실제 3090 재측정

새로운 low_feature LAST1000 원설정(G0S0/scale1,1)에서 추가 current front encoder까지 포함:
CUDA median32.256ms/p95 32.292ms/p99 32.308ms, wall median32.282ms,
VRAM alloc403.0/reserved500MiB, warmup20/repeats50, 감사43/43 통과.
이는 최종 G1S1 정확도·심사 승인 결과가 아니다.

배포 bundle은 optimizer/RNG를 제외하고 모든 config/출처 SHA를 보존하여
317,745,437→106,193,135bytes로 줄였다. 기본 config로 복원해 scale이나 mode를 잃지 않는다.
원본 보고서는 `reports/*p0_motion_lowfeature_last1000.json`에 보존했다.

## 16:38 KST: P0 BN 결과 및 P1 실행 준비

BN 대조 네 판의 계산이 끝났다. 작은 train16의 정상 eval D3는 adaptive .19161,
fixed .04360이며 fixed만 사전 fitting gate를 통과했다. 3초 L2는.31013m다.
train-BN 모드로 평가 기준을 바꾸지 않았다. 이 결과는 작은 train fitting이며 일반화 주장이 아니다.

전체 tune1998에서 fixed는 vx MAE1.21150→1.03759m/s,
history MAE.56234→.48646m를 보였다. footprint IoU 약2%p 하락/lane IoU 약1.3%p 상승도 기록한다.
고정 tune370의 11세션 paired CI는 vx/history 개선에서0을 제외했다.
저해상도 feature 정합 + fixed BN statistics를 P1 공통 개발 설정으로 선택했다.

adaptive motion run은 checkpoint 저장·평가 완료 후 native SIGSEGV 문구와 종료 지연이 있었다.
원인은 확정하지 않았다. 프로세스는 개별 중지 시도 전 이미 사라졌으므로 실제 signal을 보내지 않았다.
두 LAST checkpoint 무결성/모든 tensor 유한성을 검사하고 별도 전체 tune replay에서
보고된 모든 지표를 정확히 재현했다. 학습 결과 유효성과 정상 종료 여부는 구분한다.
후속 P1에는 실제 OS 종료를 별도 기록하는 supervisor를 사용한다. native 문제를 해결했다고 주장하지 않는다.

원본은 `reports/p0_tail_bn_pair_comparison.json`,
`reports/p0_bn_motion_pair_comparison.json`,
`reports/p0_bn_motion_checkpoint_integrity.json`,
`reports/p0_bn_motion_independent_replay.json`이다.

P1은 planning 미학습 공통 BN-fixed best1000에서 G×S 네 판을 같은6000step으로 시작한다.
LAST6000 2×2 비교를 주표, 팔별 tune-BEST를 보조표로 사전 고정했다.
정확한 계보·손익·실행/평가 조건은 `MOTIONDRIVE_V2_P1_PROTOCOL.md`에 고정했다.
이 문단 작성 시 P1은 아직 미실행이며 최종val136과 과거val38을 열지 않았다.

## 16:45 KST: P1 네 팔 발사

소스/계획 커밋 `fdd309d8ef23a6ea35131636551c51e51aa12b89`, B200194 tests 통과.
실행 manifest: `logs/motiondrive_v2/launch_p1_gs_r1_s0.json`.
실제 시작은16:45:17 KST, GPU0–3에 G0S0/G1S0/G0S1/G1S1 순서다.
supervisor PID1921374–1921377, 실제 trainer PID1921378–1921381.

공통 초기 tensor state SHA `30324570835ad3895322c596c1666ee983b46dd8f331be7b8d9e972dc9de86c0`,
parameter26,409,112개, train/eval row SHA가 네 팔 모두 같다.
step1–210의 공통22개 로그에서 누적 샘플 순서 SHA가 모두 일치했다.
모델 설정 차이는 G/S뿐이다. 영상에서 추론한 motion/history/state auxiliary는 모두 유지한다.
초기 정상 tune D3는 약13.1m이며 planning 미학습 초기값이므로 성공/실패 판정 값이 아니다.

GPU6 기존 작업은 보존했다. 시작 이후 GPU1/2에 짧게 보인 별도 프로세스는
추가 신원 조회 전에 종료되어 소유/작업을 확인하지 못했다. 어떤 signal도 보내지 않았다.
16:47경에는 GPU0–3 각각 우리 trainer만 남았고 VRAM은 각약43,844MiB(42.8GiB)였다.
이 일시적 관찰은 실패 원인으로 단정하지 않으며 최종 latency는 별도3090 단독 측정한다.

실험 진행 중이므로 P1 효과·경쟁력·규정 최종 적합성을 아직 선언하지 않는다.

## P1 step1000 G1S1: 학습 중 3090 독립 감사

실제 읽힌 checkpoint 객체의 step==1000과 G1S1/low_feature/scale=(10,5)를
bundle 공개 전에 검사했다. 계속 갱신되는 B200 LAST 원파일은 수정하지 않았다.
원본 SHA `70e1d3ea`로 시작하며 전체 SHA는 export 보고서에 있다.
완전 config를 보존한 bundle SHA는
`87b91d8ceb449ada60a3d362778656815a6180dd386c92c06cabbb9572129319`다.

3090 실영상 batch1/원config/no override, warmup20·repeats50:
CUDA median32.23058ms/p95 32.25929/p99 32.29329, wall median32.25712ms,
allocated403/reserved500MiB. 현재/과거 영상 encoder·추가 저해상도 current front·
motion·공통 scene·인지 head·planner를 모두 포함하며 feature cache는 없다.
모델 밖 전처리/H2D는 제외했다. 4090 실측 또는 최종 제출 wrapper 수치가 아니다.

감사42/42 통과: G1에서는 G0 전용 goal-OFF 전모델 불변성 검사를 적용하지 않는다.
goal 변경 시 raw motion/history/state bitwise 불변, 고정 feature planner replay 일치,
scene/motion에서 실제 planner gradient 연결을 확인했다. 이 검사는 full holdout에서
영상의 실질적 기여를 입증하거나 운영국의 최종 코드 심사를 대신하지 않는다.

보고서: `reports/audit_3090_r50_p1_g1s1_step1000.json`,
`reports/latency_3090_r50_p1_g1s1_step1000.json`,
`reports/export_motiondrive_v2_p1_g1s1_step1000.json`.
측정 컨테이너 종료와3090 GPU 해제를 확인했다. B200 학습은 계속 진행 중이다.

## 17:28 KST: 배포 정합 오류 발견, P1은 원 조건으로 보존

P1 네 판은 약5310/6000step 진행 중이다. G1S1의 중간 BEST D3는.367913이다.
LAST6000 주 분석을 기다리며 이 값을 최종 성능으로 선언하지 않는다.

실제 train 원본 JPEG 대조로 rear_wide의182.4px 투영 오차를 확인했다.
캐시는 bottom crop인데 V2 canonical projection은 top crop이었다.
별도 수정 geometry edition을 준비하며 현재 학습의 파일/조건은 바꾸지 않는다.
공식 테스트에는 원 timestamp가 없으므로 nominal time-input 평가도 분리한다.
자세한 출처와 변경 제한은 `MOTIONDRIVE_V2_DEPLOYMENT_CONTRACT_AUDIT.md`에 기록했다.

3090의 immutable train8/2세션, step1000 G1S1/B4×2 loss-gradient 진단에서는
가중 plan gradient norm127.7/136.8, motion14.9/11.4였다. 따라서 이 두 배치에서는
“NLL motion이 전체 gradient norm을 지배한다”는 가설이 지지되지 않았다.
history logvar의93.75%는 loss clamp −6 아래였고, state는50%였다.
작은 state-projection gradient나 일부 음의 shared-gradient cosine만으로
status 무효 또는 Adam 실제 update 크기를 단정하지 않는다. 파라미터/BN/.grad는 불변이었다.
음수 Gaussian NLL(상수 생략)은 그 자체로 오류가 아니다.
원본: `reports/p1_g1s1_step1000_train8_loss_gradients_3090*.json`.

## 17:38 KST: P1 LAST6000 독립 재평가와 요인효과 확인

네 trainer 모두 실제 OS 종료0/completed_cleanly, PID 부재·GPU 해제 확인.
별도 raw-time planning 평가의 전체 tune1998 D3는 원 학습 로그와 네 팔 모두 exact 일치했다.
비유한 값0, 공통 초기 tensor/데이터 노출·sample-order 계보 검사도 통과했다.

| G: 공통 영상특징 goal | S: 영상추론 상태 전달 | LAST6000 D3 | BEST 보조 D3 |
|---|---|---:|---:|
| OFF | OFF | .747844 | .735553 |
| ON | OFF | .381652 | .381381 |
| OFF | ON | .785360 | .769845 |
| ON | ON | .372043 | .367913 |

11 rawtime 세션 paired bootstrap10000, ON−OFF의 frame가중 효과:

- G 효과(S OFF): −.366192, CI95%[−.45756,−.25604].
- G 효과(S ON): −.413317, CI95%[−.51226,−.28359].
- S 효과(G OFF): +.037516, CI95%[+.01063,+.06825].
- S 효과(G ON): −.009609, CI95%[−.01592,−.00333].

G 효과는 크며 S는 G ON에서만 소폭 개선했다. S의 평균개선 .01m 사전 screening 기준은
.000391m 차이로 미달이며, CI가0을 제외한다고 기준을 바꾸지 않는다.
단일seed·반복tune·기존 rear 기하 오류 조건의 결과다. 기존val38 점수와 직접 비교하거나
1등 성능·구조 상한으로 선언하지 않는다. 다음은 같은 G1S1 LAST의 nominal/영상 교란 진단이다.

원본: `reports/p1_last6000_raw_gs_analysis.json`,
`reports/p1_g*s*_last6000_raw_planning*.json`.

## 17:40 KST: 독립 geometry edition 생성 완료

`data/etri/motiondrive_v2/train_tune_geometry_v2`에 train203+tune37/240scene/64800행을
독립 inode로 복사했다. 14종 모든 배열·scene JSON byte SHA와 기존 원본 SHA 보존을 확인했다.
감독 visibility3064셀은 XOR0이며 모든 occ/lane valid가 불변이다.
다른5개 카메라는 bitwise 보존, rear 행렬과 전역 provenance만 수정했다.

새 canonical SHA `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961`,
새 manifest SHA `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93`.
schema2는 source PKL과 실제 canonical SHA를 구분한다.
preflight의 최초 float64 곱셈순서 차이1.42e-14는 실제 캐시 builder와 같은 순서로 맞췄고
허용오차를 넓히지 않았다. 본 생성 전 전240scene 공통성을 다시 검증했다.

검증 원본: `reports/motiondrive_v2_geometry_preflight.json`,
`reports/motiondrive_v2_geometry_created.json`. 실제 dataset 입력 parity를 추가 검사한다.
P2는 같은 P1 G1S1 LAST6000에서 C(기하)×T(시간입력)의 추가3000step 통제 수정을 준비한다.
새 정책의 trainer/loader 테스트와 P1 nominal 진단이 끝나기 전에는 발사하지 않는다.
실행 계획은 `MOTIONDRIVE_V2_P2_REPAIR_PROTOCOL.md`에 별도로 고정했다.

## P1 배포 시간 입력 및 영상 기여 진단 완료

같은 G1S1 LAST6000/원 geometry/전체 tune1998의 nominal 네 조건은 실제 OS 종료0이다.

| 조건 | D3 | nominal 정상 대비 |
|---|---:|---:|
| raw 정상(원 주표) | .372043182 | 비교 기준 별도 |
| nominal 정상 | .372059222 | — |
| 다른 scene의 같은 frame 영상으로 교체 | 2.183526 | +1.811467 |
| 과거 영상을 현재 front 반복으로 교체 | .395318 | +.023259 |
| 과거 영상 순서 반전 | .373696 | +.001637 |

nominal−raw는+.00001604m, CI[+.00000200,+.00002842]로 실제 시간 차이의 영향은 작다.
shuffle/repeat의11세션 frame가중 paired CI는 각각[+1.23792,+2.33517],
[+.01741,+.03312]다. reverse의 주 CI[−.000105,+.005135]는0을 포함한다.
세션 동일가중 보조 추정은 reverse+.00360, CI[+.00136,+.00599]로 주 추정과 구분한다.
따라서 영상 의존은 강하고 history 사용도 있지만, 정확한 순서 민감도까지 해결했다고
주장하지 않는다. 영상 교란은 geometry와 불일치를 만들므로 규정 승인의 자동 증명도 아니다.
원본은 `reports/p1_g1s1_last6000_nominal_comparisons.json` 및 조건별 보고서에 있다.

실제 old/new dataset10샘플(train8+tune2) 비교에서도 rear 투영행렬 외 **모든 반환값**의
bitwise 일치를 확인했다. 영상100개 출처 SHA와 schema2 canonical SHA 검사도 통과했다.
수정된 loader/trainer 및 관련 B200 CPU tests179개가 통과했다.
GPU0–3 평가 작업은 모두 해제됐고 GPU6 기존 작업은 그대로다.

## 17:52 KST: P2 자원 대기, CPU 배포 검증 계속

`547fcdc`에 P1 원본 평가/geometry 근거, `0cb3a29`에 입력정책·보호검사/P2계획,
`77cf862`에 다른 작업을 보존하는 분리 실행 계획을 커밋했다.
하지만 별도 alpamayo 벤치마크가GPU0–3을 다시 점유하여 원4판·분리3판 모두
launcher preflight에서 안전하게 거절됐다. **P2 학습은0개 실행**이다.
실제 새 데이터/코드는 준비됐지만 자원 대기 상태이며 이를 학습 시작으로 보고하지 않는다.
다른 project의 coordinator/worker를 중지하지 않고, GPU4–7로 범위를 넓히지 않았다.

기다리는 동안 GT-free raw-clip 입력 adapter와 train8 test-shaped fixture를 작성했다.
OpenCV가 있는 기존 `env/venv`에서 관련30 CPU tests 통과. 학습용 systemPython은
OpenCV가 없으므로 그 환경을 변경하지 않았다. 실제 학습환경의 기존 immutable train8
tensor export를 reference로 사용해 Python/PIL 환경 차이까지 별도 대조한다.
fixture clip에는 JPEG10개/calibration/과거31행+제공goal1행 pose만 두며 timestamp,
status, 미래 중간 궤적·object/map 라벨은 넣지 않는다.

## 18:12 KST: 실제 raw-clip 배포 입력 parity 완료

기존 systemPython이 저장한 immutable train8 tensor export(SHA `4fb192ff…`)를
다른 환경에서 다시 디코딩하지 않고 기준으로 사용했다. 기준의 calibration과 time만
명시적으로 geometry_v2/nominal로 교체했으며 영상·history·goal·GT는 보존했다.
기존 `env/venv`의 OpenCV4.8.1/Pillow12.2.0에서 원본 JPEG와 공식형 pose/calibration
파일로 다시 만든 결과, **6종 입력 전부8/8 bitwise 동일, 최대 절대오차0**이다.
재구성 캐시 JPEG80/80개 SHA도 기존 export 출처와 일치했다. 행렬 허용오차로
차이를 숨긴 결과가 아니며 모든 행렬과 goal도 실제 bitwise 동일했다.

실제 비교 전 B200 CPU tests55개 통과. 비교 자체는 CUDA 초기화·model forward 없이
끝났고 원본 reference/라벨/fixture96파일/코드/계약 SHA가 전후 불변이었다.
raw fixture는 train4scene×frame30/180뿐이며 test/final-val을 읽지 않았다.
이8개 입력의 정합 증거이지 전체 clip 일반화·최종 모델 출력·운영국 승인 증거는 아니다.

원본: `reports/motiondrive_v2_deploy_input_parity_train8.json`,
SHA `5f60c0f6dbaaadb0774a2596f1c1f2e2be920b25f17bdd9e166ff323c7c00fa7`.
raw fixture 계보: `reports/motiondrive_v2_deploy_fixture_train8_manifest.json`.
배포 adapter는 `models/motiondrive_v2_inputs.py`이며 아직 제출 실행기/JSON writer는 아니다.

18:10 확인에도 GPU0–3은 별도 alpamayo 작업이 점유 중이었다. P2 실행0개이며
타 작업 중지/추가 GPU 사용 없이 대기한다. 다음 단계는 C×T 통제 학습과
그 C1/T1 checkpoint의 계약 보존 export·전체 forward 검증이다.

계약 정의를 OpenCV/Torch를 import하지 않는 `models/motiondrive_v2_input_contract.py`로
분리해 학습환경의 export에서도 같은 계약을 사용할 수 있게 했다. 값/픽셀 처리 변경은
없으며 분리 후 actual train8 parity를 새 보고서로 재실행해6종 모두 bitwise를 재확인했다.
원 보고서를 덮어쓰지 않았다: `motiondrive_v2_deploy_input_parity_train8_contract_replay.json`.

## P1 nominal 저장 결과의 사후 오차 분석

추가 영상/라벨을 열지 않고 기존 tune1998 결과만 분석했다. 전체D3는.372059다.
현재 GT 속도<.2m/s인123개는 계속정지99(D3.3284),3초 내.5m초과 이동22(1.2066),
경계2(.8026)로 분리됐다. 22개는5세션이며 '출발 지연 시점'을 직접 검출한 분류가 아니다.
이22개에서 .5초 GT 평균 변위.032m 대비 예측.642m로 초기 이동량이 과대했다.
그러나 정지 전체가 D3합의8.16%뿐이므로 나머지 예측을 그대로 둘 때 정지 오차를
전부 없애도 개선 상한은 약.0304m다. 정지만으로 전체 격차를 설명하지 않는다.

공식 시간가중 MAE는 ego-x .2835m/ego-y .1427m지만3초 MAE는 x .4901m/y .5969m다.
두 MAE는 L2의 가산분해가 아니며 현재 ego축은 각 시점 경로 접선축과 다르다.
사후 OLS [t,t²]는 오차 제곱에너지99.67%를 설명했으나 예측 자체99.99975%,
GT 자체99.99981%도 같은 저차 함수로 설명된다. 이는 속도·가속도 원인 규명이나
99.67% 회수 가능성을 뜻하지 않는다. 공식 가중D3와도 다른 비가중SSE 분석이다.

원본: `reports/p1_g1s1_last6000_nominal_error_structure.json`.
독립 리뷰에서11 CPU tests와 보고서 전체 재계산 동일성을 확인했다.
반복 tune11세션·old geometry의 기술적 관찰이다. 현재 보고서의 상태값은 모두 유효하나
분석기는 부분 유효 상태의 null/object dtype을 아직 지원하지 않으므로 후속 확장 시
float 변환 회귀 검사를 추가해야 한다. 이 제한은 현재1998행 결과에는 영향을 주지 않는다.

## 배포 export 및 stateless 추론 도우미 검증

기존 generic inference export는 그대로 유지했다. 새 명시적
`--deployment-contract geometry-v2-nominal --expected-checkpoint-sha256 ...` 모드만
선정 checkpoint, 원 run의 완료 manifest, 초기 가중치·split·C1 calibration/supervision
파일의 SHA와 nominal 정책을 검증하고 `input_contract`/`deployment_provenance`를 저장한다.
서버 밖으로 옮겨져 원 계보가 확인되지 않는 파일이나 P1 raw/C0는 새 계약으로 승격하지 않는다.
이 검증은 정확도·latency·운영국 승인 또는 OS 정상 종료를 인증하지 않는다.

`models/motiondrive_v2_serving.py`는 명시적인 bundle SHA와 내부 계약 정합을 검사하고
weights_only=True/완전 config/strict weights로 로드한다. 원 B200 파일을 다른 서버에서
재열람하는 대신 export 때 검증한 묶음의 SHA와 선언 정합을 검사한다.
모델에는6개 입력만 주고 전체 forward1회를 실행한다. 영상유래state/motion/perception을
생략하는 별도 지름길은 없으며 `[1,6,2]` FP32 `plan_abs`를 순수 slice/list로 변환한다.
절대좌표에 cumsum·clamp·보정·resampling을 적용하지 않는다.

독립 코드 리뷰에서 발견된 provenance 중복선언 모순 통과와 index 없는 `cuda`의
장치 불일치를 수정했다. C1/T1 SHA·완료step 모순은 거절하고 `cpu` 또는 명시적
`cuda:N`만 받는다. 체크포인트 parameter dtype의 묵시적 변환도 거절한다.
export+serving CPU tests는 로컬/B200 모두122개 통과했다. OpenCV 없는 systemPython에서
검증했으며 실제 배포 checkpoint export·GPU forward·공식 제출은 아직 하지 않았다.
CLI 전체 제출 파일 생성기는 아직 없고 실제 C1/T1 checkpoint와3090 검증이 남아 있다.

## 18:36 KST: P2 분석 검증 완료, 유휴 시 실행 대기

P2 전용 C×T 분석기의 mock CPU tests47개가 로컬/B200에서 통과했다. 고정 atol1e-5의
좌표→지표 정합 검사만 기존 P1 nominal1998행에 따로 적용해 전부 통과했다.
이는 P2 결과 분석이 아니며 **P2 학습은 여전히0개 실행**이다.

18:18–18:33의 추가15분 read-only 감시에서도 GPU0–3 전체 유휴는 관측되지 않았다.
마지막 메모리154784/156162/154784/160296MiB, alpamayo coordinator1963557 및
children1963560–1963563은 생존했다. 타 작업에는 signal을 보내지 않았다.

검증 코드를 커밋한 단일 HEAD/원4판 plan SHA로60분 한정 대기 작업을 넘긴다.
GPU0–3 네 장 모두 메모리<1000MiB·compute PID 없음·기존 coordinator 없음이
연속 두 번 확인되고, pinned HEAD 및 tracked-clean 조건이 유지될 때만 기존
supervised launcher를 통해 원4판을 시작한다. 코드/계획이 바뀌거나 다른 gate가
실패하면 자동 우회하지 않고 보고한다. GPU4–7 또는 타 작업 중지는 허용하지 않는다.
대기 시간 제한이 끝났다는 사실은 학습 완료·목표 달성·영구 장애를 뜻하지 않는다.

## 19:26 KST: 3090 입력·flow 비용 근거 반영, P2는 자원 대기

3090 기존 런타임에서도 실제 raw train8 입력6종48개가 학습 export와 모두 bitwise
일치했고 Q95 JPEG80개 SHA도 재현됐다. 이 검증은 CPU 전용이며 최종 모델 속도가 아니다.
별도 RAFT-small FP32 비용 측정은384×216에서1pair/4pair,갱신4/8/12의6조건을 실행했다.
4pair CUDA median은9.55/14.89/20.22ms, 전체 forward·warmup20·반복50·synchronize 조건이다.
모델의 영상 대응을 강화할 후보로 검토할 여지는 있지만, 전체 V2 Tinfer·정확도·대회
사용 승인으로 해석하지 않는다. 자세한 출처/라이선스 한계는
`MOTIONDRIVE_V2_FLOW_FEASIBILITY_20260907.md`에 기록했다.

기존 P1 tune1998에 대한 시간 형태 분석을 재현 가능한 CPU 분석기로 보존했다.
예측 y2차 계수 표준편차가 GT의4.48%라는 관찰은 원인/회수량을 증명하지 않는다.
실제 image-derived state의 정확도와 planner 사용 문제를 분리하기 위해 평가기에
opt-in `--include-motion-predictions`를 추가했다. 같은 forward 출력의 FP32
state/history를 추가 저장할 뿐 기본 off의 점수·records·forward/RNG 경로는 보존한다.
local planning-eval+P2 tests101개, flow 비용 도우미6개, 시간형태+기존오차25개를 확인했다.
B200 system Python에서도 GPU를 비활성화한 동일132개 CPU 테스트가 통과했다.
이 옵션으로 실제 P2 평가를 수행한 적은 아직 없다.

Git 반영을 위해19:24:36에 **우리 전용 감시만** 종료했고 발사 권한을 회수했다.
P2 발사0개, 해당 run/log/supervisor/launch 아티팩트 없음, 직전 HEAD cab53b1
tracked-clean을 확인했다. 별도 alpamayo coordinator1963557/children1963560–1963563은
중지하지 않았으며 GPU4–7도 사용하지 않았다. 학습기/모델/기하데이터/원4판 계획/초기
가중치는 보존한다. 새 HEAD의 동일성 검증 후 재감시하며, 자원 대기를 결과로 보고하지 않는다.

## 19:55 KST 추가: GPU 공유 허용 및 실제 P2 발사

사용자 공동사용 허용에 따라 idle-only 대기 cell273을 종료했다. GPU0–3만 유지하고
기존 다른 프로젝트 작업에는 신호를 보내지 않았다. 기존 batch16 메모리 관찰43844MiB가
여유 공간보다 커 microbatch2×8로 logical batch16을 유지했고, full-batch label/mask
분모 보존을 검증했다. B200 구현 `c6845fb`, CPU276 tests PASS.

GPU2의 2-step canary actual exit0, peak allocated5077/reserved5668MiB,
최저 관측free19884MiB로 통과했다. allocator12000MiB cap와 reserve8192MiB,
trainer microbatch 전 검사 및 supervisor5초 own-child-only 중단장치를 활성화했다.

P2 실제 trainer PID: GPU0 C0T0=2008118, GPU1 C1T0=2008121,
GPU2 C0T1=2007362, GPU3 C1T1=2008122. 공통 source c6845fb, 동일 initial tensor,
train54810/tune1998/3000steps. 19:55 관측step80/80/190/80, 네 팔 pressure_event 없음.
C0 초기 D3는 P1 raw/nominal값0.372043/0.372059를 정확히 재현했고,
C1 초기 D3는0.98955/0.98963으로 악화되어 geometry 적응 결과를 기다려야 한다.
이는 완료 결과가 아니다. `MOTIONDRIVE_V2_SHARED_GPU_20260907.md` 및 startup JSON 참조.

## 20:28 KST: 실제 C1/T1 canary의 3090 배포 연결 검증

진행 중 P2 네 팔의 학습 소스/조건은 그대로다. 별도로 이미 정상 종료한
`p2_shared_memory_probe_s0` step2를 **포장·입력·forward 연결 검사에만** 사용했다.
기존 strict exporter는 검사 완화 없이 B200 CPU에서 통과했다. 원 checkpoint SHA
`ba53c9dd5a84db04297cd3e6fb065be081ef8c1a3c179afdb76852b5be044703`,
새 geometry-v2/nominal bundle SHA
`43c349ebabbc1cff182255e19ff590c6ed513af572eaf8bdc286d5d6b50870ba`다.

3090의 기존 runtime을 읽기 전용으로 사용해 raw train8 입력→strict bundle load→
전체 forward→ABS JSON 경로를 실제 실행했다. 8클립×6종 입력48개가 reference와
bitwise 동일했고, raw/reference 출력8쌍과 A/다른클립/A 재실행 결과도 최대 차이0m였다.
총17 complete forward 각각 현재6+저해상도5 이미지 인코딩, 주요7출력 유한성,
최종 FP32좌표 및 ABS 직렬화 무변형을 확인했다. model state/자료/source 전후불변이다.

실제 Docker/SSH rc0, PID72989 부재, GPU130MiB/no compute, Docker 종료를 독립 확인했다.
이 과정에서 다른 작업을 중지하거나 패키지를 설치하지 않았다. 관련 CPU131tests가
로컬과 B200 기존 `env/venv/bin/python`에서 통과했다. B200 system Python은 cv2가
없어 최초 collection에 실패했고, 코드를 우회하지 않고 이미 설치된 환경으로 검증했다.
원 결과 `reports/motiondrive_v2_canary_deployment_smoke_3090.json`,
실행 증거는 같은 이름의 `.execution.json`, CPU export 증거는
`reports/motiondrive_v2_canary_strict_export.json`이다.

**이 체크포인트는 2-step 메모리 canary이지 최종 제출 후보가 아니다.** 이번 결과는
일반화·대회 승인·최종 Tinfer 또는 실제 P2 LAST3000의 배포 검증을 대신하지 않는다.
최종 C1/T1 학습 결과와 실제 해당 체크포인트의 검증은 여전히 남아 있다.
