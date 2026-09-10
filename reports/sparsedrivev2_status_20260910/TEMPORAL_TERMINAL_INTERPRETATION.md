# 고정 2,000-step temporal 대조: 실측 해석

세 run의 terminal 자료를 CPU로 검증했다. 공개 초기화·은행·source·parameter 수·train/tune 행은 같고, 초기 planning 배열도 bitwise 같았다. 기록된 201개 step의 batch row SHA·epoch·LR·occ/lane 유효 label 수가 모두 일치했다. 저장하지 않은 나머지 batch의 순서를 사후에 직접 증명한 것은 아니다. Root가 세 프로세스의 정상 종료를 확인했으며, 이 분석기는 모델 또는 GPU를 로드하지 않았다.

| Arm | 개입 | D3 | Shortlist oracle | Regret | 3초 L2 |
|---|---|---:|---:|---:|---:|
| A | 현재 front를 과거 슬롯에도 반복 | 1.037030 | 0.919662 | 0.117367 | 2.444337 |
| B | 실제 과거 front .1/.5초 | 0.996320 | 0.889414 | 0.106906 | 2.337275 |
| C | B + 공통 perception attention의 원시 상태 | 0.313323 | 0.164612 | 0.148711 | 0.912121 |

**Temporal 단독 이득은 이번 실험에서 확정되지 않았다.** B−A D3는 −0.040709이며 paired 95% CI는 [−0.088223, +0.012939]로 0을 포함한다. 11개 session 중 7개가 개선됐다. Oracle 차이 −0.030248의 CI [−0.084565, +0.025133], regret 차이 −0.010461의 CI [−0.021291, +0.002782]도 모두 0을 포함한다. 이는 이 architecture·간격·해상도·손실·2,000-step 예산에서의 결과이며 시간 영상 전반의 가능성을 부정하지 않는다.

**C의 큰 개선은 더 좋은 후보가 shortlist에 남은 변화가 중심이다.** C−B D3는 −0.682997, CI [−0.876525, −0.574046]이며 11개 session 모두 개선됐다. Oracle 차이는 −0.724802, CI [−0.944678, −0.600695]다. 동시에 regret은 +0.041805, CI [+0.012986, +0.076293]로 증가했다. 즉 후보 포함과 최종 선택을 구분해야 한다. D3 = oracle + regret이라는 정의에 따른 분해이며, 두 좌표축의 Euclidean 기여를 더한 주장이 아니다.

**현재 C shortlist를 고정한 최종 점수 보정만으로 이 tune 평균 .15를 달성할 수는 없다.** 저장된 C의 1,998행별 shortlist 최소 오차의 평균이 이미 0.164612다. 이 하한은 현재 평가에서 만들어진 후보 집합을 고정했을 때만 적용한다. Base/temporal/coarse를 다시 학습하거나 후보 집합을 바꾸는 실험의 하한 또는 전체 bank의 하한은 아니다.

**큰 계획 개선을 큰 perception 개선으로 확인한 것은 아니다.** C−B occupancy IoU는 +0.002011, lane IoU는 +0.000543이고, 두 지표의 집계값만 저장되어 CI는 없다. 원시 상태를 직접 읽지 않는 영상 state auxiliary의 vx MAE는 B 1.365826 → C 2.073799m/s로 오히려 증가했다. 그 차이는 +0.707973m/s, CI [+0.510456, +0.846369]다. ax MAE는 0.304910 → 0.301479m/s²이며 차이 CI [−0.007547, +0.003075]가 0을 포함한다. 따라서 이번 자료는 공통 경로의 계획상 유용성을 보여 주지만 영상 속도 추정의 향상, 의미 있는 perception 개선, 규정 승인을 함께 입증하지 않는다. B1 영상 개입 결과는 이 문서 작성 시 포함하지 않았다.

**보조 축 오차도 C의 변화가 종방향 중심임을 보여 준다.** 공식 시간 가중치로 계산한 평균 종방향 절대오차는 B 0.982728 → C 0.285675m, 횡방향 절대오차는 0.073287 → 0.072287m다. 전체 6개 시점과 행을 시간 가중치 없이 합산한 종방향 제곱오차 에너지 비율은 A 98.210%, B 98.270%, C 85.374%다. 이 비율은 D3의 축 기여율이나 확률이 아니다. 3초에서 B의 종/횡 평균 절대오차는 2.265265/0.267251m, C는 0.802388/0.261163m다. Oracle/regret 해석을 우선하고 이 통계는 오차 방향의 보조 근거로 사용한다.

모든 CI는 같은 11-session bootstrap draws 20,000회(seed 0)에서 각 복원추출 session의 전체 행을 포함한 frame-weighted 평균의 percentile 구간이다. 반복 사용한 tune의 단일 학습 seed 결과이며, 학습 seed 불확실성·다중 비교·독립 confirmation을 검증하지 않았다. 이전 raw-status 모델의 D3 0.130749는 입력 경로와 base/head 학습 절차가 달라 이번 matched architecture control로 취급할 수 없다.

근거 파일:

- [검증·수치·paired CI JSON](temporal_analysis_v1/aggregate.json)
- [Terminal 결과 표](temporal_analysis_v1/RESULTS_KO.md)
- [Horizon별 축 오차 JSON](axis_errors_temporal_v1/axis_errors.json)
- [축 오차 보고서](axis_errors_temporal_v1/AXIS_ERRORS_KO.md)
