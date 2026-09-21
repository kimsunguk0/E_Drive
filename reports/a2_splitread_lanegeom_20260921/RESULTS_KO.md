# CTRL 이후 SplitRead / LaneGeometry 동일 조건 continuation

실행 중. 예정 중간 결과를 terminal로 해석하지 않는다.
DEV 부모0.150285502, 공식 FULL0.133684828은 서로 다른 평가다. 서버 점수 환산 없음.

|Stage step|CTRL-NEXT|SPLITREAD|LANE-GEOM|
|---|---:|---:|---:|
|0|0.150285502|0.150285548|0.150285502|

동일 sample stream 755개 update 확인.
시점·주행군·signed 구간 길이·횡/종 오차·인지 지표·세션 집중도는 result_step*.json에 보존한다.
TemporalRead/readout 공유 범위와 연속 lane 감독은 독립 비교이며 효과를 더해서 예측하지 않는다.
