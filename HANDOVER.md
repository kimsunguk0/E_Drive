# MotionDrive V2 — 현재 인수인계

**최신 확인: 2026-09-20.** C2F/FRESH는 모두20,554 update를 완료했다. C2F DEV0.164204741은 기존과 거의 같고, 공개 nuImages trunk에서 새로 학습한 FRESH는0.158707259다. 기존 QREFINE+FRESH 저장 예측의 고정1:1 평균은0.150723838이며 배포·서버 평가 전이다. GPU0–3의 이번 학습은 종료됐다.
[Terminal 결과](reports/a2_motion_fresh_20260919/RESULTS_KO.md), [상세 해석·고정 평균](reports/a2_motion_fresh_20260919/INTERPRETATION_20260920_KO.md), [실행 기록](reports/a2_motion_fresh_20260919/EXECUTION_KO.md). 추가 학습·FULL·공식 제출은 새로 시작하지 않았다.

**2026-09-20 추가 진단 완료:** 고정 QREFINE/FRESH의 V0 1,998행 및 무증강 train 1,024행을 GPU0/1에서 비교했다. 예측 state/history를 0으로 해도 점수 변화는 약 0.0002지만 continuous motion 제거는 +0.0710/+0.0441 악화했다. FRESH는 V0 정지 유지 99행을 모두 현재 정지로 분류하면서도 미래 경로를 전진시키며, train에서도 기존보다 정지·출발·횡방향 fitting이 부족하다. 미래 첫2초 속도 변화가 작은 일반 주행도 고정 평균 전체 오차의60.8%다. [잔여 오차 진단과 다음 우선순위](reports/a2_error_diagnosis_20260920/RESULTS_KO.md). 추가 학습·FULL 이전·제출은 시작하지 않았다.

**직전 완료 상태:** State ON/OFF와 RGB 교사 학습·평가·strict student export를 완료했다. A2 FULL raw 추론·패키지도 확보했다.
[이번 최종 결과](reports/a2_visual_teacher_20260919/RESULTS_KO.md), [FULL 배포 안내](reports/a2_full_submission_20260919/README_KO.md).
이 문서가 현재 상태와 결정을 나타낸다. 과거 실행 계획의 ‘진행 중·다음 실행’ 문구는 당시 기록이다.
[최근 작업 통합 보고서](reports/recent_work_20260919/SUMMARY_KO.md)에서 근거와 변경 이력을 확인할 수 있다.

## 1. 지금 보존할 후보

|역할|후보|점수|해석|
|---|---|---:|---|
|확인된 공식 제출 기준|MR-NATIVE-FULL-s1|**0.185968928**|사용자가 전달한 공식 서버 결과|
|기존 A2 고정 terminal 대조|A2-QREFINE-NOM-s1|**0.164251770**|DEV V0; 이전 A2 대비 이득은 작고 불확실|
|새 최선 단일 DEV terminal|A2-FRESH-NUIM-s1|**0.158707259**|일반 주행·첫2초 개선, 정지/출발·횡방향·인지 회귀 동반|
|추가 저장 예측 분석|QREFINE+FRESH 고정1:1 평균|**0.150723838**|같은 DEV, 계수 스윕 없음, raw 배포/실측 비용/서버 미검증|
|선택된 DEV 중간 후보|A2-G1-s1 / step 2,284|**0.163882485**|같은 V0의 3개 예정 평가 중 선택; 일반 주행은 부모보다 약간 악화|
|raw 배포·ZIP 확보|A2-FULL-NOM-s1|공식 점수 없음|8 fixture 정확 일치, 1125 clip 완료; 4090 시간은 아직 미측정|
|최신 command 보조 후보|A2-COMMAND-NOM-s1|**0.164884294**|BASE보다 0.38% 개선 관측, QREFINE보다 높음|

공식 서버·DEV·FULL 학습 내 점수는 서로 다른 평가다. 하나의 순위표로 합치지 않는다.
A3-DIRECT의 DEV 0.149289는 제공 status가 motion/state에 영향을 주는 경로 때문에 현재 제출 방향에서 제외됐다.
A2 또는 command를 운영국이 개별 승인했다고 주장하지 않는다.

## 2. 완료된 A2 비교

아래는 같은 V0 1,998행의 예정 terminal 점수다. G0/G1과 VIS-TEACHER는 QREFINE terminal 이후 3,426 update의 추가 학습이다.

|실험|변경|Update|PREFIX|현재 판단|
|---|---|---:|---:|---|
|BASE-NOM|배포형 nominal status 대조|20,554|0.165510649|완료 control 재사용|
|MH4-NOM|공통 scene의 독립 attention 4개|20,554|0.164495430|작은 이득; 기존 A2 배포형 평가보다 높음|
|SIDE-SCENE-NOM|FL/FR 과거 4장을 공통 scene에 추가|20,554|0.172406018|이번 설정 추가 확대 제외|
|QREFINE-NOM|첫 visual read로 query 갱신 후 재조회|20,554|0.164251770|현재 fixed terminal 기준으로 보존|
|G0|기존 보조 비중으로 matched continuation|3,426|0.165084069|부모보다 악화|
|G1|같은 continuation, 보조 묶음 ×0.25|3,426|0.164740804|terminal 악화; step 2,284만 별도 보존|
|LEARNED-SAMPLE|영상·고정 metadata 기반 sampling offset|20,554|0.164667226|BASE 소폭 개선, QREFINE 미달|
|COMMAND-NOM|제공 6종 command → 공통 scene query|20,554|0.164884294|보조 후보 보존|
|VIS-TEACHER|QREFINE 이후 RGB 공간 특징 증류|3,426|0.166137158|G0·부모 모두 미달, FULL 이전 제외|
|C2F-MOTION|stride16 대응→stride4 세밀한 영상 비교|20,554|0.164204741|일반 주행 개선 미확인, 확대 후순위|
|FRESH-NUIM|공개 trunk만 로드, 나머지 새 초기화|20,554|0.158707259|새 단일 DEV 후보, 회귀·추가 학습 검토 필요|

- 기존 A2-DIRECT의 실측 timestamp 평가 **0.164280979**, 동일 checkpoint의 배포형 nominal 평가 **0.164455314**.
  새 BASE는 nominal status로 다시 학습했으므로 기존 모델과의 비교에는 입력 학습 정책 차이도 포함된다.
- G1−G0, sampling−BASE, command−BASE의 session 신뢰구간은 모두 0을 포함한다.
- 여러 예정 checkpoint 중 고른 G1 값과 fixed terminal을 구분한다.
- FULL 가중치·teacher·feature cache를 DEV로 되돌리지 않는다.

## 3. 최신 command 결과와 남은 병목

Command는 원본 parquet의 LANE_KEEP / TURN_LEFT / TURN_RIGHT / LANE_CHANGE_L / LANE_CHANGE_R / U_TURN 6종이다.
`vad_cmd`나 GT 미래 XY에서 계산한 지시를 이번 입력으로 사용하지 않았다.
좌우 반전 때 회전·차선변경 command도 교환한다.

- 전체: BASE **0.165511 → 0.164884**, 약 **0.38%** 개선.
- 일반 주행: **0.169591 → 0.169269**, 약 **0.19%** 개선.
- 의미 좌회전: **0.210316 → 0.206248**; 우회전: **0.288671 → 0.284718**.
- 같은 완료 모델의 지시를 전부 LANE_KEEP으로 바꾸면 **0.165817**로 악화한다.
  이는 command를 활용한다는 진단이며 독립 학습 대조의 개선폭은 아니다.
- Command를 바꿔도 완료 checkpoint의 motion/state/history 출력 차이는 0이다.
- V0 의미 좌회전은 39행/1session, 유턴은 0행이다. 회전 일반화에 대한 결론은 제한적이다.

이 command 비교에서는 일반 주행과 첫 2초 진행량의 큰 개선을 얻지 못했다. 이후 FRESH의 개선과 회귀는 상단 최신 결과를 따른다.
오차 대부분을 ‘가속·감속 타이밍 하나’로 설명하거나 현재 점수를 아키텍처의 한계로 확정하지 않는다.
자세한 분해는 [command 판정](reports/a2_command_20260919/DECISION_KO.md)에 있다.

## 4. 고정할 입력·평가 계약

- 현재 A2: 제공 status5는 **공통 scene query** 조건. Goal도 기존 공통 scene 조건이다.
- Command arm만 같은 query에 6종 지시를 추가한다. 새 raw planner token/value/gate는 없다.
- 영상 기반 motion/state/history 경로는 이 조건 입력과 분리한다.
- 같은 최종 scene tensor를 occupancy·lane·planner가 읽는다.
- 기존 pose alignment는 유지한다. 새 pose 수치 운동 token을 만들지 않는다.
- Planner의 `state_on`은 예측 state 사용 설정이며 제공 status 직접 입력과 구분한다.
- `provided_status5`는 train/DEV/test에서 동일한 nominal causal producer를 사용한다.
  영상 state/history supervision은 기존 실측 timestamp 정의를 유지한다.
- [OPEN_ISSUE.md](OPEN_ISSUE.md)의 Q1·Q7·Q8·Q10 및 공지2가 보존된 원문이다.
  Q10에는 영상에서 추론한 state/history 사용 허용 답변이 있다.

공식 지표는 **PREFIX**다. L2_1s/2s/3s는 앞 2/4/6점 평균이고 세 값의 평균이 점수다.
시점별 가중치는 **[11,11,5,5,2,2]/36**. L2_3s를 3초 endpoint로 해석하지 않는다.
판정은 plain V0로 한다. 과거 test-matched 재가중이나 단일 서버–DEV 차이를 새 모델의 점수 환산식으로 쓰지 않는다.

DEV: train 310 scenes / 83,700행, V0 37 scenes / 1,998행 / 11sessions.
FULL: **376 unique scenes / 101,520행 / 24,931 update**.
historical_val 9 scenes는 H29에 포함되므로 385 scenes로 세지 않는다.
FULL 사용과 독립 DEV 검증은 가중치 계보를 분리하면 병행할 수 있다.

## 5A. 2026-09-19 저녁 완료 결과

- QREFINE 고정 DEV의 예측 state/history ON/OFF: **0.164251762 / 0.164534473**. OFF의 제거 이득 미확인, CI는 0 포함. 제공 status와 continuous motion은 유지했다. ON을 유지하며 자동 OFF 학습은 하지 않는다.
- DINOv2 ViT-B/14 register RGB teacher, frozen, train-only spatial loss 한 종류를 QREFINE terminal에서 추가 3,426 update 학습했다.
- 최종 **VIS 0.166137158 / G0 0.165084065 / QREFINE 0.164251767**. VIS는 control과 부모에 모두 미달해 FULL로 이전하지 않는다.
- VIS−G0 **+0.001053093**, 11-session paired95%CI **[+0.000131418, +0.002577468]**. 일반 주행은 부모 **0.168445113 → 0.170013739**로 악화했다.
- 입력/목표 정규화/공간 정합/gradient 경로와 teacher 불변을 검사했다. Teacher/projector 없는 student strict export의 B1 XY 차이는 0이다. Visual loss 감소가 planning 개선으로 이어지지 않았다.
- 같은 초기값·recipe·행 순서의 G0를 재사용했다. 기록된 rolling SHA는 step3,400까지 모두 일치하며, 마지막 26 update는 같은 epoch0 결정적 sampler 정책이지만 별도 terminal rolling hash가 기록되지 않았다는 한계를 남겼다.
- A2 FULL은 재학습하지 않았다. 8 raw fixture bitwise parity, 1,125개 실제 test clip, 729.815616192G counter, ZIP, Docker raw smoke를 완료했다.
- 로컬 ~/Downloads/A2-FULL-NOM-s1_submission_20260919/submission.zip. **공식 미업로드**, 이번 A2의 **RTX4090 시간은 장비 확인 대기**다.
- 이번 결과는 짧은 RGB 교사 continuation 한 종류에 대한 것이며 A2 상한이나 모든 사전학습 방식의 실패를 뜻하지 않는다. 후속 스윕/FULL은 시작하지 않았다.

## 5B. 2026-09-20 motion/fresh 결과

- C2F−QREFINE −0.000047026, session CI95 [−0.000882082,+0.000931003]. 일반 주행은 소폭 악화했다.
- FRESH−QREFINE −0.005544508(3.38%); 일반 주행0.168445113→0.155274043(7.82%). 첫2초 기여는0.122951391→0.114141415.
- FRESH의 마지막 두 포인트·정지/출발·횡방향 및 occupancy/lane 지표는 악화했다. 단독 delta의 session CI는0을 포함한다.
- 추가 학습 없이 고정1:1 예측 평균0.150723838. QREFINE 대비 delta−0.013527929, CI95 [−0.017196053,−0.007854766]. 반복사용 DEV11sessions의 조건부 분석이며 서버 성능 보장이 아니다.
- FRESH 후반0.184436→0.162706→0.158707. 추가 수렴 비교를 검토할 근거이며 자동 예산 확대나 local minimum 증명은 아니다.
- 구현/terminal 결과는 미러에 반영됐다. 상세 분해와 평균 계산 코드는 위 해석 문서에 연결된다.

## 5. 재개할 때의 작업

|항목|확인된 상태|남은 일|
|---|---|---|
|MR FULL 제출|패키지·raw B1 parity·공식 결과 확보|기존 후보 보존|
|A2 FULL|24,931 update 유지, raw parity/1125 추론/729.816G/ZIP/Docker smoke 완료|이번 A2의 RTX4090 시간 측정, 사용자 판단에 따른 공식 제출|
|DEV 기준·후보|FRESH terminal / QREFINE / G1 중간 / command 보존|FRESH 수렴 및 회귀, 고정 평균의 raw/비용 검토|
|OOF-MR-T203|54,810행 producer 학습 완료|new107 예측/진단은 미완료, 재우선순위 시 진행|
|G/S/command 결과 기록|완료 및 GitHub 반영|이번 통합본에서 연결|
|실험 자동 watcher|두 개 모두 완료|재시작할 이유 없음|

지금 새 FULL, 추가 λ/head/radius 스윕 또는 공식 업로드를 예약하지 않았다.
준비되지 않은 recurrent BEV, A3 gate 복구, residual 추가를 자동 후속으로 취급하지 않는다.
과거 SIDE residual의 일부 단계는 미완주·미평가였으므로 전체 계열을 완전학습 후 실패로 묶지 않는다.
중단된 A3-FP-VA는 마지막 학습 로그 600이며 프로세스가 없다. 오래된 manifest의 `running`을 현재 실행으로 읽지 않는다.

제출 횟수는 **2026-09-18 사용자 확인 기준 5회 중 2회 사용**이다. 계정의 현재 값을 재조회한 것은 아니다.
저장소의 다른 계정 제출 기록을 우리 quota에 더하지 않는다.

## 6. 실행·배포 메모

작업 루트는 `/NHNHOME/data/sukim/adcl`. 현재 할당은 **GPU 0,1,2,3**이며 재사용 전 실제 점유를 확인한다.
Python은 `~/cv2env/bin/python`. 공식 test 추출 위치 `/tmp/etri_test`는 재확인이 필요한 임시 저장소다.
새 데이터 분할이나 checkpoint 계보 변경 없이 각 run의 `manifest.json`·`experiment.json`을 기준으로 재현한다.

제출 출력은 이미 누적 absolute XY `6×2`이므로 `cumsum`을 다시 적용하지 않는다.
ZIP 내부 `submission.json` 하나, 1,125 clip와 정수 `__flops__`를 검증한다.
비용은 공식과 같은 `FlopCounterMode`의 전체 B1 forward를 측정한다.
서버 응답 `elapsed_ms`를 모델 latency로 쓰지 않는다. 새 graph의 B200 측정은 RTX4090 시간 인증을 대신하지 않는다.

## 7. 파일과 Git 연결

|용도|문서|
|---|---|
|최근 작업 통합 보고서·판정 근거|[SUMMARY_KO.md](reports/recent_work_20260919/SUMMARY_KO.md)|
|14개 주요 완료 학습·중간 후보·24개 이전 단계·SHA|[run_registry.json](reports/recent_work_20260919/run_registry.json)|
|기계적으로 읽을 완료 결과표|[results.csv](reports/recent_work_20260919/results.csv)|
|9/17–19 작업–미러 커밋 49개 대응|[commit_index.json](reports/recent_work_20260919/commit_index.json)|
|읽기 쉬운 커밋 대응표|[COMMITS_KO.md](reports/recent_work_20260919/COMMITS_KO.md)|
|공식 MR FULL 결과|[서버 결과](reports/md_full_submission_20260918/SERVER_RESULT_20260918_KO.md)|
|G/S 최종 판정|[G/S decision](reports/a2_next_20260919/DECISION_KO.md)|
|Command 최종 판정|[Command decision](reports/a2_command_20260919/DECISION_KO.md)|
|정리 전 전체 시간 기록|[이전 HANDOVER, 고정 GitHub commit](https://github.com/kimsunguk0/E_Drive/blob/a8a6ee9ff6fee2597fce991f4300670d8d3f41ae/HANDOVER.md)|

GitHub: `kimsunguk0/E_Drive`, branch `motiondrive-v2-20260910`.
작업 저장소 이력을 직접 push하지 않고, 변경 파일을 명시해 commit한 뒤 별도 미러로 cherry-pick한다.
작업 commit과 미러 commit은 SHA가 다르다. 비밀·접속 정보와 대형 가중치는 새 commit에 넣지 않는다.
checkpoint·대형 예측은 서버에 보존하고 경로·크기·SHA를 색인에 남긴다.
과거 미추적 worktree/backup/9월9일 산출물은 이번 정리 범위에서 보존했다.
