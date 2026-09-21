# PROGRESS 네 실험 최종 결과와 기록 안내

2026-09-21 갱신. 네 DEV 실험은 모두 3,426 update를 완료했다.
주 비교는 동일한 terminal 3,426이며, 중간 checkpoint에서 고른 최저값은 별도로 표시한다.

DEV 부모 PREFIX는 **0.151178860**이다. 공식 제출 FULL의 **0.133684828**과는 데이터·평가 범위가 다르며, DEV 개선량을 서버 점수로 환산하지 않는다.

## 네 실험에서 바꾼 것과 최종 결과

|실험|변경|terminal DEV PREFIX|P-CTRL 대비|판정|
|---|---|---:|---:|---|
|P-CTRL|기존 PROGRESS + 길이 loss 유지, 작은 LR로 추가 학습|0.150285502|—|네 terminal 중 최저. 부모 대비 −0.000893358|
|P-VECTOR|길이 L1 보조 항을 구간 변위 벡터 L2로 교체, 계수 0.25 유지|0.150329892|+0.000044390|방향·벡터 오차 일부 감소, 전체 PREFIX 추가 개선 미확인|
|P-FINE|기존 correlation map을 24×32로 추가 조회, 기존 12×16 경로 유지|0.150370043|+0.000084541|길이·방향의 변화가 작고, 전체 PREFIX 추가 개선 미확인|
|P-SHARED768|scene history가 별도 384 특징 대신 이미 계산한 768 motion용 영상 특징을 공유|0.242503976|+0.092218474|이번 짧은 적응에서는 크게 악화. 고해상도 방식 전체의 실패로 일반화하지 않음|

P-FINE은 backbone 재실행이나 새 RGB 관측을 추가하지 않는다.
P-SHARED768은 시작 함수가 달라 step 0부터 성능 변화가 크다. 기존 384 경로와 초기 출력이 같다고 주장하지 않는다.

## 학습 중 평가 기록

|Stage step|P-CTRL|P-VECTOR|P-FINE|P-SHARED768|
|---|---:|---:|---:|---:|
|0|0.151178860|0.151178860|0.151178860|13.144899210|
|1,142|0.152279242|0.151520283|0.158229915|0.274815725|
|2,284|0.151344204|0.149929678|0.150315328|0.255129187|
|3,426|0.150285502|0.150329892|0.150370043|0.242503976|

P-VECTOR의 0.149929678은 재사용한 V0에서 선택 가능한 **중간값**이다. 고정 terminal 결과와 혼용하지 않는다.
동일 sample stream 3,426 update를 확인했다. VECTOR/FINE 대 CTRL의 세션 paired bootstrap 구간은 모두 0을 포함하며, 확정된 추가 이득으로 설명하지 않는다.

## 무엇을 어디서 확인할 수 있는가

|내용|파일|
|---|---|
|실행 조건, 입력 경계, 네 arm의 구현 차이|[EXECUTION_KO.md](EXECUTION_KO.md)|
|사용자 실행 명세 원문|[USER_EXECUTION_ORDER.md](USER_EXECUTION_ORDER.md)|
|모든 평가 시점의 점수 표|[results.csv](results.csv), [results.json](results.json)|
|최종 시점별·주행군별·종횡 오차, 인지 지표, 세션 비교|[result_step3426.json](result_step3426.json)|
|길이와 방향이 실제로 함께 줄었는지에 대한 해설|[TERMINAL_LENGTH_HEADING_REVIEW.md](TERMINAL_LENGTH_HEADING_REVIEW.md)|
|위 분석의 구간별 수치·예측 경로·hash|[TERMINAL_LENGTH_HEADING_REVIEW.json](TERMINAL_LENGTH_HEADING_REVIEW.json)|
|각 실험의 부모·데이터·LR·loss·학습 예산|[CTRL protocol](protocol_P-CTRL.json), [VECTOR protocol](protocol_P-VECTOR.json), [FINE protocol](protocol_P-FINE.json), [SHARED768 protocol](protocol_P-SHARED768.json)|
|초기 함수·loss·gradient·reload 검사|[smoke_and_reload.json](smoke_and_reload.json), 각 P-*_initial_checks.json / P-*_reload.json|
|FINE의 실제 4090 smoke 검사|[PRODUCTION_FINE_RTX4090_SMOKE.json](PRODUCTION_FINE_RTX4090_SMOKE.json)|
|당시 terminal 선택 근거와 후보 색인|[decision.json](decision.json), [candidate_registry.json](candidate_registry.json)|
|선택한 CTRL의 별도 FULL 추가 학습·배포 완료|[FULL RESULTS_KO.md](../a2_progress_fourarm_full_20260921/RESULTS_KO.md)|

길이·방향 분석의 핵심은 다음과 같다. VECTOR는 matched CTRL보다 방향 오차는 줄었지만 길이 오차는 늘었다.
FINE은 같은 이동 구간 mask에서 길이·방향이 모두 아주 조금 줄었지만, 전체 행 길이 오차와 최종 PREFIX는 조금 늘었다.
부모 대비 개선에는 공통 추가 학습 효과가 포함되므로, 새 변경의 효과는 CTRL과 비교한다.
구간 길이·방향 평균을 공식 PREFIX의 가산 분해로 해석하지 않는다.

## FULL 후속 상태와 원본 산출물

선택한 **P-CTRL-FULL-STAGE2-s1**은 별도 FULL 부모에서 전체 train 101,520행을 사용해 **4,156 update**를 완료했다.
Raw parity, 전체 1,125 test clip 추론, portable export, 실제 최종 가중치의 RTX4090 Docker 검사, 로컬 제출 패키지 전달을 완료했다.
**새 FULL의 공식 업로드·서버 점수는 아직 없다.** FULL에 포함된 V0 평가는 in-fit이며 DEV 일반화 지표가 아니다.

decision.json 등 선택 시점에 생성된 파일의 FULL 상태는 당시 snapshot이다.
최신 배포 상태는 [LOCAL_DELIVERY.json](../a2_progress_fourarm_full_20260921/LOCAL_DELIVERY.json)과 [RTX4090_validation.json](../a2_progress_fourarm_full_20260921/RTX4090_validation.json)을 기준으로 한다.

대형 checkpoint·저장 예측·학습 로그는 B200의 아래 경로에 보존하며, Git에는 구현·프로토콜·수치·경로·hash를 기록한다.

- DEV: /NHNHOME/data/sukim/adcl/work_dirs/a2_progress_fourarm_20260921/
- FULL: /NHNHOME/data/sukim/adcl/work_dirs/a2_progress_fourarm_full_20260921/
- 로컬 제출 패키지: /home/a/Downloads/P-CTRL-FULL-STAGE2-s1_submission_20260921/

원래 제출한 A2-H4-PROGRESS-FULL-s1은 보존했다.
