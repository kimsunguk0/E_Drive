# SplitRead / LaneGeometry terminal 판단 — 2026-09-21

세 DEV arm 모두 6,852 update를 완료했다. 본 학습 시작 19:57:32 KST, 전체 완료 20:59:53 KST다. 아래는 같은 V0 1,998행 / 37 scene / 11 session의 저장 예측을 float64로 재계산한 PREFIX다. Trainer 집계와 약 3e-9 차이는 정밀도 차이다. 공식 서버 점수 또는 FULL 학습 내 점수가 아니다.

|후보|PREFIX|부모 대비|동일 CTRL 대비|일반 주행 PREFIX|
|---|---:|---:|---:|---:|
|부모 P-CTRL /3426|0.150285502|—|—|0.153568988|
|P-CTRL-NEXT /6852|0.147582070|-0.002703431|+0.000000000|0.150947358|
|P-SPLITREAD /6852|0.147393734|-0.002891768|-0.000188337|0.150758669|
|P-LANE-GEOM /6852|0.147662130|-0.002623371|+0.000080060|0.151034600|

## 판단

- 기존 부모보다 CTRL은 1.799%, SplitRead는 1.924% 개선했다. SplitRead의 같은 예산 CTRL 대비 추가 감소는 0.000188337 (0.128%)다.
- SplitRead는 길이/방향의 TemporalRead와 비선형 readout을 분리했다. Common decoder와 영상 관측은 유지했다. 개선 대부분을 추가 학습 control이 재현하므로, 길이/방향 분리로 큰 도약을 만들었다고 판단하지 않는다.
- SplitRead–CTRL session paired bootstrap 95% CI는 [-0.000702089, +0.000084342]이며 11개 중 5개 session에서 개선했다. 가장 큰 기여 session을 제외하면 차이는 -0.000069269다. 반복 사용한 DEV 11세션의 조건부 재표집이며, CI가 0을 포함한다고 효과가 정확히 0이라고 결론 내리지 않는다.
- 부모 대비 SplitRead는 11/11 session 개선, CI [-0.005382168, -0.001668854]다. CTRL도 부모 대비 10/11 session 개선, CI [-0.004946469, -0.001517974]다.
- LaneGeometry는 CTRL보다 +0.000080060 높다. 이 설정의 새 기하 감독에 따른 planning 개선은 확인하지 못했다. 해당 감독 종류 전체가 무효라는 결론은 아니다.

## 실제 길이·방향·종횡 변화

|지표|부모|CTRL-NEXT|SPLITREAD|LANE-GEOM|
|---|---:|---:|---:|---:|
|종방향 위치 절대오차 m|0.120757840|0.119483914|0.119439252|0.119643814|
|횡방향 위치 절대오차 m|0.068332437|0.066054583|0.065835447|0.065989351|
|PREFIX 1초 누적 m|0.085258167|0.082925719|0.082757857|0.083047408|
|PREFIX 2초 누적 m|0.141825031|0.139607450|0.139514089|0.139746755|
|PREFIX 3초 누적 m|0.223773307|0.220213042|0.219909255|0.220192229|
|GT 곡선 50행 PREFIX m|0.414470848|0.394035766|0.388783102|0.394865782|
|6구간 길이 MAE 평균 m|0.074353090|0.072053246|0.071918022|0.072073298|
|6구간 heading MAE 평균 도|0.703427077|0.675057621|0.670986497|0.673714052|

종횡 수치는 GT 구간 길이 >0.05m인 같은 mask에서 공식 시점 가중치를 유효 질량으로 재정규화했다. 두 값을 더해 PREFIX로 해석할 수 없다. Heading은 GT 유효 mask에서 predicted zero를 따로 계수하며 제외하지 않는다. 곡선은 GT 모든 구간 유효, 전체 길이>=3m, 최대 구간 heading 절댓값>=15도인 50행의 진단 subset이다.

SplitRead의 부모 대비 종오차는 1.09%, 횡오차는 3.65% 감소했다. 그러나 matched CTRL 대비는 각각 0.037%, 0.332% 감소다. 길이와 방향 개선의 대부분 역시 공통 continuation에서 관측된다. SplitRead는 6개 구간 heading 모두 CTRL보다 작지만 구간 길이 MAE는 다섯 구간만 작고 세 번째는 약간 크다.

SplitRead의 일반 주행 평균은 0.150758669, 전체 기여는 0.141477730 (95.99%)다. 전체 점수의 첫 0.5/1초 기여 0.050574246, 1.5/2초 0.054519533, 2.5/3초 0.042299954다. 첫 2초 비중은 71.30%다. 일반 주행의 종방향 정밀도가 여전히 큰 잔여 오차이며, 이번 결과만으로 원인을 가감속 타이밍 또는 시각 해상도로 확정하지 않는다.

## Train·gradient와 구현 정합

|고정 train256 probe|step0|step3426|step6852|
|---|---:|---:|---:|
|P-CTRL-NEXT|0.123167718|0.130477936|0.118561710|
|P-SPLITREAD|0.123167742|0.126502380|0.118020579|
|P-LANE-GEOM|0.123167718|0.129157718|0.118077403|

Train만 개선된 결과는 아니다. 다만 고정 train256과 V0는 표본·분포가 다르므로 두 수치 차이를 곧바로 과적합의 인과 증거로 쓰지 않는다. 학습 전 과정의 sample stream 6,852 update와 마지막 augmentation fingerprint가 세 arm 모두 일치했다. 새 모듈의 gradient와 실제 optimizer update가 확인됐고 nonfinite는 0이다.

LaneGeometry weighted gradient / PREFIX+LEN gradient norm은 terminal 첫 microbatch의 backbone layer1 conv1에서 약 0.160%, scene key projection에서 약 0.0245%였다. 새 head의 gradient는 존재한다. 이는 이 가중치에서 관측된 공유 gradient 기여가 작다는 근거이며, 전체 backbone의 영향 또는 실패 원인을 확정하는 진단은 아니다. λ를 자동 스윕하거나 geometry를 이 설정보다 폭넓게 반증한 것으로 기록하지 않는다. 전체 확인값은 TERMINAL_SUPPORT.json에 있다.

## 비용·후속 결정

동일 student graph의 RTX4090 smoke 가중치 측정은 CTRL median25.38–25.47ms, SplitRead25.66–25.67ms, Lane student25.53–25.54ms다. 전체 B1 BF16 forward이며 I/O·전처리·전송은 제외했다. SplitRead FLOPs는 730.098142G, 다른 두 student는 730.044861G다. Terminal 가중치의 raw 배포 인증을 완료했다는 뜻은 아니다.

최저 terminal SplitRead와 기존 graph CONTROL을 모두 보존한다. LaneGeometry 추가 확대는 하지 않는다. 현재 horizon에서 학습/평가를 종료하고 자동 추가 update는 시작하지 않는다. 중간 step3426보다 terminal이 좋아졌다는 사실은 cosine 감쇠 구간의 변화이기도 하므로 계속 연장하면 더 좋아진다는 근거가 충분하지 않다. 수렴 완료나 구조적 성능 상한도 주장하지 않는다.

Collector의 candidate_for_FULL은 최저 terminal의 후보 지명이다. 새 FULL/공식 업로드는 시작하지 않았다. 이후 FULL을 이전한다면 대응 부모는 이미 완료한 CTRL-FULL-STAGE2이고 동일 추가 노출량은 8,311 update다. DEV 가중치를 FULL에 복사하는 실험과 구분한다.

0.133684828은 기존 제출의 실제 서버 결과다. 이번 DEV를 서버 0.12로 환산할 근거는 없다. 더 큰 구조 변경을 검토할 경우에는 출력 읽기/loss의 추가 스윕보다 초기 함수 변화를 관리한 원본 해상도 temporal 관측의 공동 학습이 아직 남은 독립 가설이다. 해당 새 실험은 이번 기록으로 시작하거나 예약하지 않았다.

## 재현 기록

- 실행 계약: EXECUTION_KO.md / protocol_*.json
- 초기·중간·terminal: result_step0.json / result_step3426.json / result_step6852.json / results.csv
- Train probe·gradient·optimizer·완료 상태: TERMINAL_SUPPORT.json
- 대형 예측·terminal 가중치: raw_prediction_index.json의 B200 경로·크기·SHA256
- 최저 SplitRead checkpoint SHA256: 8252ae12063b0e684899ae22cfac2f3a29fedc0ea06df41849bab3d47c2767f6
- 결과 생성 source 기준: 891588889c34292d07ed05bc21eaf4e0c561774a
