# CTRL 이후 SplitRead / LaneGeometry 동일 조건 continuation

완료: terminal6,852이 주 비교다.
DEV 부모0.150285502, 공식 FULL0.133684828은 서로 다른 평가다. 서버 점수 환산 없음.

|Stage step|CTRL-NEXT|SPLITREAD|LANE-GEOM|
|---|---:|---:|---:|
|0|0.150285502|0.150285548|0.150285502|
|3426|0.156572405|0.150132025|0.154208853|
|6852|0.147582070|0.147393734|0.147662130|

동일 sample stream 6852개 update 확인.
시점·주행군·signed 구간 길이·횡/종 오차·인지 지표·세션 집중도는 result_step*.json에 보존한다.
TemporalRead/readout 공유 범위와 연속 lane 감독은 독립 비교이며 효과를 더해서 예측하지 않는다.

FULL 검토 후보: P-SPLITREAD. FULL은 별도 실행이며 아직 시작 상태가 아니다.

[최종 해석·종횡·추가 학습 결정](TERMINAL_REVIEW_KO.md): SplitRead의 CTRL 대비 추가 이득은 0.000188337이며 CI는 0 포함. 추가 학습 효과와 구조 효과를 구분한다. 새 FULL/연장/업로드는 시작하지 않았다.
