# H4 status + 구간 진행량/방향 비교

학습 중. 예정 중간 평가를 고정 terminal 결과로 해석하지 않는다.
동일 TemporalRead·공개 초기값·5시점 status·loss·sample stream·budget에서 출력 구조를 비교한다.
기존 11시점 status 모델은 이 구조의 matched control이 아니다. 초기 trainable tensor는 같지만 초기 궤적은 다르다.

|Step|DIRECT PREFIX|PROGRESS PREFIX|차이|DIRECT 일반주행|PROGRESS 일반주행|
|---|---:|---:|---:|---:|---:|

Sample stream: 7개 로그 비교, 불일치 [].
인지·state/history·구간 진행량·첫2초·종횡 지표는 result_step*.json에 기록한다.
서버 점수 예측이나 0.12 달성을 주장하지 않는다. 자동 추가 학습·FULL·공식 제출은 없다.
