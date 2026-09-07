# 19:05 KST — P2 자원 대기 중 추가 근거

이 절 작성 시점(19:05)에는 아래 신규 파일이 **로컬 보존·B200 커밋 대기** 상태였다.
B200는 `cab53b12524ae5236bb7913fb07a70c1f57f6059`의 tracked-clean 상태로
자동 실행 에이전트가 소스를 고정하고 있으므로 지금 원격 파일을 바꾸지 않는다.
기존 사용자 변경, 다른 프로젝트 프로세스, GPU4–7은 건드리지 않았다.

## 실제 3090 CPU 입력 정합

기존 cu128 런타임을 읽기 전용으로 사용했다. raw train8 클립에서 만든 6종 입력
48개가 모두 기존 system-Python tensor export와 bitwise 일치했고 최대 절대오차는0이다.
재생성 Q95 JPEG80개도 기존 export에 기록된 캐시 SHA와 일치했다.
reference의 rear calibration/time만 검증된 C1/nominal 계약으로 교체한 비교이며,
영상·과거 영상·과거 변환·goal 원본 tensor는 그대로 보존했다.

- `reports/motiondrive_v2_3090_deploy_input_parity_cab53b1.json`
  SHA `3924ff256c0b695db3ba167aac757f023937b76ac329200c1469bcba0fdb8fca`
- `reports/motiondrive_v2_3090_deploy_input_parity_cab53b1.execution.json`
  SHA `3dd09bad49cfe5ea5d5142c534022ae25cad61456209f9649cc46c42252f6130`
- 새 원격 보존 경로:
  `/home/intern/adcl_motiondrive_v2/measurements/deploy_inputs_cpu_cab53b1.lEi72Zr2`

실제 CPU Docker 종료0, source10/raw96/reference/calibration 전후 SHA 불변이다.
루트도 보고서의48개 aggregate와 기존 B200 replay 일치, 로컬 source10 SHA,
전후 raw SHA 및 환경 불변을 독립 확인했다. CUDA·model forward·최종 속도·공식
제출은 실행하지 않았으며 그 성능이나 규정 승인을 증명하는 결과가 아니다.

## P1 횡방향 시간 형태 진단

기존 nominal tune1998 결과만 사용했다. 각 표본의6점에 intercept 없이
`q(t)=a*t+b*t²`를 비가중 OLS로 적합했다. y의 b 표준편차는 예측0.00634,
GT0.14156으로 비율4.476%, 분산 비율0.2004%이며, b MAE0.06843,
Pearson−0.08473이다. 11개 세션 모두 예측 b의 분산이 작았다.

이는 시간에 따른 횡방향 변화의 과소 표현이지 공간 곡률·물리 가속도·다중모드
평균화·손실/구조 원인의 증명이 아니다. GT `[t,t²]` 적합 잔차 RMSE는
x0.02736m/y0.01464m이며 높은 에너지 설명률에는 큰 위치값 분모의 영향이 있다.
원본 GT는 exact polynomial이 아니다. GT3은 미래 pose의 frame+5..+30을 현재
좌표로 변환한 캐시를 dataset/evaluator가 그대로 전달한다. 우리 코드에서
상태 적분·미래 시간 보간·다항 fitting으로 GT3을 만드는 경로는 확인되지 않았다.
주최측 원 pose 생성 전의 센서융합/평활 여부는 미확정이다.

- `scripts/analyze_motiondrive_v2_temporal_shape.py`
  SHA `ae2924c8293c28f8f3e266075db48c47889221a80859c6114ab53638332fad49`
- `tests/test_motiondrive_v2_temporal_shape.py`
  SHA `8f2c306796027bf763eadd613315ae1ef967dc261470ed2e662563b6d5625551`
- `reports/p1_g1s1_last6000_nominal_temporal_shape.json`
  SHA `292f2b63d58d407768455a8ba0004557fc7523625ee68253536f491a0a54611f`

실제 분석 CLI 종료0. 새14개 테스트 및 기존 planning-error11개를 함께 실행한
루트 독립 CPU 검증25개 통과. 표본별 보정 좌표는 출력하지 않는다.
P2에서도 같은 현상이 남는지 먼저 확인하며 새 손실/구조는 아직 변경하지 않았다.

## 다음 실행과 미확정 사항

19:02:28 KST 자연 주기 검사에서 별도 alpamayo coordinator1963557 및
workers1963560–1963563이 GPU0–3에서 실행 중이었다. **P2 발사0개**다.
기존55초 주기 감시는19:38:08 KST까지 유효하다. 네 GPU 모두 메모리1000MiB 미만,
선택 GPU compute PID 없음, 기존 coordinator 없음이2회 연속 관측되고 HEAD/plan
SHA/작업트리 조건이 유지될 때만 원4판 supervised launcher가 실행된다.
관찰 기한 종료는 학습 완료나 영구 장애가 아니다.

P2가 실제로 시작되면 초기 tensor/데이터 순서/실제 PID·로그를 확인한다.
신규 근거를 B200에 커밋할 때는 대기 에이전트의 소스 고정을 먼저 해제하거나
실제 시작 완료를 확인해 자동 실행과 충돌하지 않도록 한다.
최신 공개 리더보드는 정상 접근에서 순위 본문을 반환하지 않아 현재1위/3위 점수와
정확한 마감은 미확인이다. 기존 tune 결과를 현재 리더보드 순위로 환산하지 않는다.

## 19:26 KST 반영

자동 실행 권한을 회수한 뒤19:24:36에 전용 감시 PID1982225와 SSH 세션43887의
종료를 확인했다. P2 발사/아티팩트는0이며 다른 작업에는 signal을 보내지 않았다.
이 문서의 신규 근거와 이후 flow 비용 검증, opt-in neural state 기록 평가기를
이번 B200 커밋에 함께 반영한다. 학습기·모델·P2계획·초기 가중치는 바꾸지 않는다.
새 HEAD에서의 대기 재개는 별도 gate 확인 후에만 허용한다.
