# A2 motion / fresh initializer 결과

주 비교는 완료된 QREFINE과 같은 20,554-update terminal이다. 공식 서버 점수가 아니다.

|Run|PREFIX|QREFINE 대비|Session paired 95% CI|
|---|---:|---:|---|
|A2-C2F-MOTION|0.164204741|-0.000047026|[-0.000882082, +0.000931003]|
|A2-FRESH-NUIM|0.158707259|-0.005544508|[-0.013048052, +0.008364741]|

Fresh 초기값은 과거 ETRI 학습 노출을 제거했다. 동일 새 update 예산의 결과가
scratch 수렴이나 구조 성능의 상한을 입증하지 않는다. 반복 사용한 V0 11개 session의
조건부 bootstrap이며, 중간점 선택과 독립 검증을 혼동하지 않는다.

세부 그룹·첫 2초·종/횡 오차·인지 지표·checkpoint SHA는 각 result JSON에 있다.
새 FULL이나 공식 제출은 자동으로 시작하지 않았다.
