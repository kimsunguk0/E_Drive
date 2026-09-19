# 2026-09-17–19 MotionDrive 최근 작업 통합 보고서

이 문서는 최근 실험의 현재 판정과 근거를 모은 인수인계 자료다.
2026-09-19 12시대 KST에 실제 manifest, 저장 prediction, checkpoint, Git 기록을 읽어 정리했다.
새 학습·분할 변경·모델 수정·공식 업로드는 수행하지 않았다.
정리 전 시간순 기록은 [고정 HANDOVER](https://github.com/kimsunguk0/E_Drive/blob/a8a6ee9ff6fee2597fce991f4300670d8d3f41ae/HANDOVER.md)에 보존돼 있다.
현재 작업 지침은 [루트 HANDOVER](../../HANDOVER.md)를 따른다.

## 1. 현재 판단

공식 제출 기준은 status-free MR FULL **0.185968928**이다.
현재 A2 계열의 고정 terminal 기준은 QREFINE **0.164251770**이며 공식 서버 점수는 없다.
G1 step 2,284 **0.163882485**는 반복 사용한 V0에서 선택한 중간 checkpoint로 별도 보존한다.
Command **0.164884294**는 BASE 대비 작은 이득을 보였지만 QREFINE을 넘지 못했다.
공통 scene의 표현·학습 비중·command 변경이 일반 주행/첫 2초 오류를 크게 낮추지는 못했다.

A2 FULL은 이미 학습을 완료했다. 0.088934는 학습에 사용한 V0의 진단값으로,
DEV 또는 서버 성능과 비교하면 안 된다. 배포 정합성·패키징은 남아 있다.

## 2. 확인 범위와 재현 정보

- 주요 완료 학습 14개의 manifest·terminal prediction·checkpoint·SHA를 확인했다.
- 14개 모두 `status=completed`, `nonfinite_count=0`이며, 저장 좌표에서 PREFIX를 독립 재계산했다.
- G1 중간 checkpoint와 prediction의 SHA를 기존 선택 기록과 대조했다.
- 9/17 residual·status warmup·direct·smoke 등의 24개 단계는 기존 원시 색인을 연결했다.
  이 24개에는 주요 표와 겹치는 FULL/direct 단계가 있으므로 14+24개의 독립 학습이라고 세지 않는다.
- 과거 MR ADJ/W64/LONG까지 포함하는 최근 작업 commit 49개의 미러 대응을 기록했다.
  매핑 근거는 유일한 동일 제목 또는 기존 명시 매핑이며, 작업/미러 전체 트리 동일성을 주장하지 않는다.
- G/S·command watcher의 완료 상태와 기록 복구를 보존했다.
- 확인 시 GPU 0–3 메모리 사용량/사용률은 0이었다. 이후 사용할 때 다시 확인한다.

[run_registry.json](run_registry.json)은 각 run의 scope, step, seed, 입력 정책, source commit,
초기값, data-row SHA, checkpoint/prediction/manifest/metrics 경로·크기·SHA를 담는다.
[results.csv](results.csv)는 `evaluation_scope` 열을 포함하며, FULL과 DEV가 섞인 정렬표로 사용하지 않는다.
`PREFIX_reported`는 기존 producer 값, `PREFIX_recomputed_float64`는 저장 좌표의 재계산 값이다.
수십억분의 몇 수준 차이는 계산 dtype에 따른 것이며 다른 모델 결과가 아니다.

## 3. 공식 제출과 FULL

|제출|L2_1s|L2_2s|L2_3s|PREFIX|출처|
|---|---:|---:|---:|---:|---|
|MR-NATIVE-s1|0.113672391|0.196343930|0.283946978|0.197987767|사용자 전달 공식 응답|
|MR-NATIVE-FULL-s1|0.105356792|0.184546318|0.268003674|0.185968928|사용자 전달 공식 응답|

같은 status-free MR 계열의 FULL 전환으로 절대 0.012018839, 상대 6.0705% 개선됐다.
train 310→376 unique scene, 83,700→101,520행, 20,554→24,931 update이며 microbatch도 2→8로 달라졌다.
따라서 데이터량 하나만 바꾼 순수 대조라고 설명하지 않는다.

MR FULL 보존 checkpoint SHA:
`ce569da5d3cfe70f89f042faa2a5f6449e30fccbd469f2e689e26445accf2b87`.
제출 ZIP SHA:
`dc7cd52a01db5e96a810b0397b81c113da11c72bb54455152f0e3a1f5d4ca428`.
둘 다 실제 파일에서 재확인했다. raw/cache B1 8개 fixture bitwise parity 및 1,125 clip 검사는 기존 기록을 따른다.

A2 FULL은 같은 nominal 입력 계열의 single-head 모델이며 24,931 update를 완료했다.
checkpoint SHA:
`aabdb2dca24491b46fd2e56e66d57fe0742e17a5b4c63c29f3199137cf8ff297`.
그 raw-input adapter 인증과 공식 점수는 저장된 기록 기준으로 없다.

출처: [MR FULL 공식 응답·설정 대조](../md_full_submission_20260918/server_comparison_20260918.json),
[MR FULL 서버 보고](../md_full_submission_20260918/SERVER_RESULT_20260918_KO.md),
[A2 FULL 완료 기록](../md_a2_nominal_mh4_20260918/full_terminal_summary.json).

## 4. 실험의 진행과 현재 판정

|축|실제로 한 일|관측과 판단|근거|
|---|---|---|---|
|초기 MR 개선|native matching, W64, ADJ, LONG, seed 조합|native matching 이득 이후 단순 확대/장기 일정은 큰 새 이득을 주지 못함. 전반적 한계 증명은 아님|[MR round3](../md_exp_diagnosis_20260915/MR_ROUND3_RESULTS_KO.md)|
|Progress residual|FRONT/SIDE, scalar/두 계수, AUX, denoise, status warmup|일반 주행 개선이 작아 당시 주력 종료. 일부 단계는 미완주/미평가|[24단계 색인](../md_shared_dynamics_20260917/records_20260918/experiment_index.json)|
|A2/A3 direct 공동 학습|MR upstream에서 각각 20,554 update|A2 real-time DEV 0.164281, A3 0.149289. A3는 입력 경계 쟁점으로 이후 제출 방향에서 제외|[당시 결과](../md_shared_dynamics_20260917/RESULTS_20260918_KO.md)|
|배포 status 정합|동일 A2 checkpoint의 실측→nominal 교체|0.164281→0.164455. 입력 producer 차이를 해소했지만 이번 V0의 주요 병목은 아니었음|[producer 검증](../md_a2_deploy_status_20260918/REVIEW_AND_P0_KO.md)|
|BASE/MH4|같은 nominal 학습에서 single-head/4-head|0.165511→0.164495. 기존 A2 nominal-eval 0.164455보다 높음|[terminal 기록](../md_a2_nominal_mh4_20260918/dev_terminal_summary.json)|
|SIDE-SCENE|측면 과거를 motion 대신 shared scene에 편입|0.172406으로 악화. 이번 설정의 FULL/단순 연장 제외|[SIDE/QREFINE 판정](../md_a2_scene_extensions_20260918/TERMINAL_REVIEW_KO.md)|
|QREFINE|첫 read의 영상 정보를 사용해 두 번째 query 형성|0.164252. 이전 A2 대비 이득 0.000204에 그침|[동일 판정](../md_a2_scene_extensions_20260918/TERMINAL_REVIEW_KO.md)|
|Gradient 진단과 G0/G1|32 train batch probe 후 보조 묶음 ×1/×0.25 matched continuation|G1 terminal은 G0보다 낮지만 부모보다 높음. 중간 G1만 별도 보존|[G/S 판정](../a2_next_20260919/DECISION_KO.md)|
|Learned sampling|source별 영상+고정 metadata 기반 ±2 feature-cell offset|0.164667. BASE 소폭 개선, QREFINE 미달|[G/S 결과](../a2_next_20260919/RESULTS_KO.md)|
|Command|실제 제공 의미 command6종 → A2 shared scene query|0.164884. 회전 도움은 관측됐고 전체 이득은 작음|[Command 판정](../a2_command_20260919/DECISION_KO.md)|

과거 문서 끝의 A3 우선 실행, command 보류, motion 보조만 ×0.1의 2,000-step 계획은
이후 사용자 지시와 실제 실행으로 대체됐다. 현재 자동 실행 목록으로 읽지 않는다.

## 5. 정확한 A2 결과표

|Run|단계 update|PREFIX|일반 주행 PREFIX|
|---|---:|---:|---:|
|A2-BASE-NOM-s1|20,554|0.165510649|0.169590813|
|A2-MH4-NOM-s1|20,554|0.164495430|0.168698109|
|A2-SIDE-SCENE-NOM-s1|20,554|0.172406018|0.174362595|
|A2-QREFINE-NOM-s1|20,554|0.164251770|0.168445113|
|A2-G0-s1|3,426|0.165084069|0.168719514|
|A2-G1-s1|3,426|0.164740804|0.168372241|
|A2-LEARNED-SAMPLE-s1|20,554|0.164667226|0.169427009|
|A2-COMMAND-NOM-s1|20,554|0.164884294|0.169268853|

G0/G1은 같은 QREFINE terminal에서 fresh optimizer/동일 sample stream으로 시작했다.
Backbone/head LR 1e-6/1e-5, warmup 100, 추가 3,426 update이고,
두 arm의 차이는 occupancy/lane/motion 기존 보조 묶음 계수 1.0 대 0.25다.
LEN 0.25는 그대로다. G1−G0가 보조 비중 대조이며, 부모 대비는 추가 학습까지 포함한 비교다.

G1 step 2,284의 선택된 PREFIX는 **0.163882485**, parent 대비 **−0.000369282**,
session 95%CI **[−0.002074331,+0.000533726]**이다.
일반 주행은 **0.168651014**, 부모 **0.168445113**보다 높다.
3개 예정 평가 중 선택했으며 선택 편향 보정이나 독립 검증은 하지 않았다.
manifest의 best 필드만으로 이 후보를 찾지 말고 [선택된 checkpoint 기록](../a2_next_20260919/G1_selected_step2284_review.json)을 따른다.

G1−G0 **−0.000343264**, sampling−BASE **−0.000843423**, command−BASE **−0.000626354**.
이 세 비교의 session bootstrap CI는 모두 0을 포함한다.
G0/G1은 69개, sampling/BASE와 command/BASE는 412개 로그의 row SHA가 일치했다.
반복 사용한 DEV와 단일 training seed의 한계를 포함해 해석한다.

## 6. 진단으로 확인한 것과 확인하지 못한 것

배포 nominal status는 frame 차이×0.1초로 생성하고 supervision은 실측 timestamp 정의를 유지한다.
1,125 test clip의 timestamp 부재와 causal producer 유효성을 확인했다.
기존 A2의 status 교체에 따른 PREFIX 차이는 +0.000174336이었다.
이 검사는 전체 A2 raw-input adapter 인증과 별개다.

G 진단에서 backbone의 auxiliary/main gradient norm 중앙값은 약1.0183,
cosine 중앙값은0.0108, 32 batch 중15개가 음수였다.
공유 scene의 norm 비율은 약0.0404였다. Probe 전후 parameter/buffer hash는 같았다.
이 값으로 충돌을 유일한 원인이라고 확정하지 않았고, 후속 G1의 train 개선은 V0 terminal 개선으로 이어지지 않았다.

Learned sampling의 offset은 학습됐지만 평균 이동은0.031–0.078 feature cell,
saturation0, 신규 invalid 비율<0.059%였다. 낮은 offset만으로 더 큰 범위가 답이라고 결론 내리지 않는다.

같은 GT tangent mask에서 command의 종방향 절대오차는0.144608→0.144064,
횡방향은0.061086→0.061114였다. 두 투영 오차는 합해서 PREFIX가 되지 않는다.
첫2초 score 기여 감소는0.000367140, 일반 주행 개선은0.19%에 그쳤다.
공통 진행량·시간변화 성분 진단은 미래 구간 길이에 기반하며 실제 현재 v0 오차와 다르다.
따라서 ‘가감속 타이밍 하나가 원인’, ‘정확한 v0만 맞히면 해결’이라는 인과 결론은 보류한다.

## 7. Command의 의미와 경계

현재 프레임의 raw command.parquet와 timestamps.parquet를 exact join한85,698행이
기존 의미 command cache와 전부 일치했다. Test1,125개도 raw label parser로 읽었다.
추가 projection은192파라미터, zero-init이며 기존 모델의 RNG를 소비하지 않는다.
초기 실제 영상 FP32/BF16 출력이 BASE와 같았고, 첫 batch loss/row stream도 같았다.

완료 모델에서 올바른 command를 모두 LANE_KEEP으로 바꾸면 PREFIX0.164884291→0.165816821.
좌회전은0.206247798→0.225718570, 우회전은0.284718121→0.313267695로 악화한다.
이 개입은 같은 모델이 지시를 활용한다는 근거이며 독립 학습 control을 대신하지 않는다.
motion/state/history는 command 교체 시 차이0, terminal replay XY 차이도0이다.

V0 의미 TURN_LEFT39행/1session, TURN_RIGHT42행/3sessions, U_TURN0행.
GT3초 횡변위로 나눈 left41/right161행은 다른 정의이며 굽은 도로·차선 변경을 포함한다.
nonstop1,875행 역시 직진만을 뜻하지 않는다.

이번 구조는 status/goal/command를 공통 scene query 조건으로 사용하고,
같은 scene을 occupancy/lane/planner가 읽는다. 영상 기반 motion/state/history 분기는 별도다.
detach나 planner 함수 signature만으로 입력 정보가 사라진다고 설명하지 않는다.
운영국 원문은 [OPEN_ISSUE](../../OPEN_ISSUE.md) Q1/Q7/Q8/Q10 및 공지2다.
이 기록은 개별 구조의 심사 승인을 대신하지 않는다.

## 8. 비용·배포·계보

|대상|공식 counter FLOPs|기타|
|---|---:|---|
|MR 공식 제출|729,815,613,824|두 제출 동일|
|BASE A2 preflight|729,815,616,192|같은 top-level B1 계산|
|Command A2 preflight|729,815,616,576|추가384 FLOPs, B200 약18.88ms|
|Learned sampling|약733.166G|B200 약28ms|

도구·graph가 다른 FLOPs 수치를 섞지 않는다. B200 시간과 RTX4090 모델 forward 시간도 구분한다.
서버 응답의 `elapsed_ms`는 사용자의 지시에 따라 모델 latency 판단에서 제외한다.
새 graph의 4090 비용은 미측정이다. A2 FULL raw adapter/패키징 완료로 읽지 않는다.

FULL은 유효376 scene/101,520행이며 24,931 update는 성공한 약3.93회 노출을 이전한 레시피다.
항상 최적인 학습량으로 증명된 값은 아니다. FULL weights/teacher/cache를 DEV로 가져오지 않았다.
H29에 historical_val9가 포함되므로 FULL385 계산은 폐기한다.
OOF producer는 old203/54,810행 학습을 끝냈지만 new107 예측·분석은 아직 남아 있다.

## 9. 기록 상태와 정정 사항

- A3-FP-VA의 raw manifest는 `running`이지만 프로세스가 없고 마지막 train 로그는600이다.
  통합 색인의 effective_status를 stopped_incomplete로 적었으며 원시 manifest는 변경하지 않았다.
- MR FULL의 completion.json `ready_not_uploaded`는 패키지 준비 시점 기록이다.
  이후 사용자 제공 공식 응답0.185968928을 현재 제출 상태의 근거로 사용한다.
- G/S watcher는 정상 완료했다. Command watcher는 학습·평가·개입 진단 후 Git 공백 검사에서 멈췄다.
  Markdown 줄 끝 공백과 CSV CRLF를 수정하고 재학습 없이 결과 생성/commit/push를 복구했다.
- 현재 두 watcher는 완료 상태이며, 최종 상태 파일도 이번 정리에서 Git에 보존한다.
- 실측 timestamp A2, nominal A2, FULL in-fit, 중간 선택 checkpoint를 서로 바꿔 인용하지 않는다.
- 과거 순위·서버−DEV +0.007 환산·완주가 학습량 최적임을 뜻한다는 표현은 현재 판단에 사용하지 않는다.
- 제출 quota는 9/18 사용자 확인2/5가 마지막 근거이며 계정 현재 상태를 조회하지 않았다.
- 이전 worktree·backup·9/9 미추적 산출물은 임의 삭제하거나 이번 commit에 섞지 않았다.

## 10. 다음 세션에서 바로 찾을 것

1. 배포 확보: A2 FULL 완료 checkpoint와 [FULL 기록](../md_a2_nominal_mh4_20260918/full_terminal_summary.json).
2. DEV 기준: QREFINE terminal, G1 step2,284 선택 기록, Command terminal의 해시와 계보.
3. 입력 경계: nominal producer, 공통 scene query 조건, 영상 motion/state 분리.
4. 결과 해석: [G/S 판정](../a2_next_20260919/DECISION_KO.md), [Command 판정](../a2_command_20260919/DECISION_KO.md).
5. 이전 실행: [24단계 원시 색인](../md_shared_dynamics_20260917/records_20260918/experiment_index.json).
6. 소스 역추적: [49개 작업–미러 commit](commit_index.json).

주력 모델 교체나 추가 FULL/스윕/공식 업로드를 이 문서로 새로 실행하지 않는다.
