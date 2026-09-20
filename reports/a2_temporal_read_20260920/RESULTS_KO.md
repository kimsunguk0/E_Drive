# A2 시점별 motion read 결과

실행 중. 아래 예정 중간점은 최종 결과가 아니다.
동일 공개 FRESH 초기값·학습 조건에서 planner의 unpooled visual motion read만 추가했다.
주 비교는 20,554-update terminal끼리다. 더 오래 학습한 FRESH-CONT와 같은 예산이라고 설명하지 않는다.

|Step|Temporal read PREFIX|동일 step FRESH|차이|일반 주행|출발|정지 유지|
|---|---:|---:|---:|---:|---:|---:|

Sample stream 확인: 6개 로그, 마지막 step 250, 불일치 [].
시점별 L2·첫2초·종/횡·인지/state/history·checkpoint SHA는 result_step*.json에 있다.
Attention이나 state MAE를 성능으로 대체하지 않는다. 새 학습/FULL/제출/가중치 평균은 자동 실행하지 않는다.
