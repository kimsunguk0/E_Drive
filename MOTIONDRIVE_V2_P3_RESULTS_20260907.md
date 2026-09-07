# P3 결과 — 상태 query 연결은 소폭 이득, 사전 진전 기준 미달

2026-09-07 22:31 KST. source `e4a1c24ca6046f824e5c5b3bdb2a683c7b4cce19`.
GPU4 control / GPU5 state, 두 팔 모두 실제 OS rc0로1000steps 완주했다.
정상 종료 시각은22:26:52/53 KST, 약8분30초다. 이전 r1의 step0 실패는 보존한다.

## 사전 고정 주결과

같은 tune1998/37scene/11session, 공식 시간가중 D3의 동일 프레임 평균이다.
수정 기하/nominal, 같은 frozen trunk, 동일60.3만 planner parameter와 일정이며,
차이는 query adapter에 영상 추정 상태22차원을 주는지0을 주는지뿐이다.

| step | control | state |
|---|---:|---:|
| 초기 |0.4454619773599255|0.4454619773599255|
|250|0.49842590495134825|0.5074006238685654|
|500|0.4483847929266383|0.46502508536038045|
|750|0.4475107641589527|0.4442932558790394|
|**LAST1000**|**0.4459774155454995**|**0.4379407064933274**|

state−control = **−0.0080367091 (−1.802%)**.
11세션 paired cluster bootstrap10000회(seed20260907)95%구간은
**[−0.013903543, −0.002724411]**이다. 8/11세션에서 개선됐다.
세션을 동일 확률로 복원추출하되 각 draw에서는 모든 포함 프레임을 같은 가중치로
합산했다. frame-IID 또는 세션 평균의 단순 평균으로 계산하지 않았다.

사전 기준은 `delta≤−0.01 AND CI.upper<0`이었다. **구간 조건은 만족하지만
개선량 조건은 미달하므로 screening 진전 gate는 FAIL**이다. 이후 기준을−0.008로
바꾸지 않는다. 양팔 BEST도 자연스럽게 LAST1000이지만 주결과는 처음부터 LAST다.

관측된 이득을0이라고 하지 않는다. 다만 단일seed·반복 tune의 조건부 비교이며,
seed 불확실성이나 독립 최종 holdout 일반화를 검증한 구간은 아니다.
이는 상태 query 연결만으로 크게 성능이 올라간다는 가설을 지지하지 않는다.

## 실제 실행과 무결성

- 동일 초기 전체 tensor SHA `29b1d73c…`, frozen 시작/종료 SHA `430e2525…`.
- 마지막 row-order SHA `c58d4116…` 동일.101개 학습 로그 시점의 순서 SHA 및
  동결 auxiliary6항(occ/lane/history/state/stop/motion)이 모두 exact 동일.
- 시작/끝 checkpoint를 CPU로 읽어 frozen tensor SHA, 모든 model/optimizer tensor
  유한성, 실제 optimizer step1000, 공통 초기 optimizer empty를 검증했다.
- control parent/child2065871/2065978, state2065881/2065979. 모두 actualrc0,
  supervisorrc0, PID부재. signal/pressure/nonfinite0. 이전 실패 기록 SHA도 보존됐다.
- peak allocated1092.69MiB/reserved1758MiB(두 팔 동일), 최소 관측 free179984MiB.
  끝난 뒤 GPU4/5 모두0MiB/compute PID없음. GPU0–3의 기존 네 작업은 그대로이며6·7은 미사용.
- 이 메모리와 학습시간은3090/4090 추론 지연시간 측정이 아니다.

근거:
[paired 분석](reports/p3_query_r2_last1000_pair.json),
[실제 실행 검증](reports/p3_query_r2_execution.json),
[r1 수치 조건 정정](MOTIONDRIVE_V2_P3_INITIAL_CONTEXT_20260907.md).
paired 분석 SHA `05fc105d248e0e48fb4e1b85b9a528f0656f7d30201e0b7936ca11410637f8ee`.

## 다음 우선순위와 미달 조건

P3 query adapter를 새 우승 구조로 승격하거나 seed 대회를 확대하지 않는다.
본질적으로 아직0.44 수준의 반복 tune 모델이며, 순위권 성능을 입증하지 못했다.
다음 실용적 대조의 우선순위는 **수정된 기하와 nominal 시간 조건을 처음부터
사용하는 공개 nuImages R50 기반 baseline 재학습**이다. 잘못된 기하로 학습한
가중치를 계속 보정한 이력부터 제거한다.

이는 성능 탐색 방향이지 기하 이력이 유일한 원인이라는 결론은 아니다.
새로운 공통 공개 초기값·전체 backward canary·분리된 GPU4/5 감독·동일 데이터/지표를
먼저 고정해야 한다. 기존 C0 선행 모델과 기하 이력의 순효과를 주장하려면 별도의
동일 예산 대조가 필요하다. 이 재학습은 이 문서 작성 시점에는 아직 발사하지 않았다.

원 P2의4팔 clean-exit gate FAIL, 최종 val/test 격리, 공식 제출 미실행을 유지한다.
새 P3 체크포인트의 실제3090 전체 forward·seed 복제·최종 배포 검증 역시 미완료다.
