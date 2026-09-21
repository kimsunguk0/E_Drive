# SplitRead 이후 남은 방법 — 2026-09-21

최신 DEV 저장 예측과 실제 코드, 이전 원본 해상도 비용 측정을 다시 확인했다. 새 학습·GPU 평가·FULL·제출은 시작하지 않았다. 본 문서는 실행 protocol 확정이 아닌 다음 투자 우선순위다.

## 최신 오차: 일반 주행의 작은 진행량·방향 차이가 계속 남음

P-SPLITREAD terminal PREFIX 0.147393734의 같은 1,998행이다. 과거 FULL in-fit 통계를 새 DEV 결과로 옮기지 않고 재계산했다.

|집단|행|PREFIX|전체 점수 기여|전체 점수 비중|
|---|---:|---:|---:|---:|
|일반 주행|1,875|0.150759|0.141478|95.99%|
|일반 주행 중 첫 2초 GT 구간 진행속도 범위 <=0.5m/s|1,306|0.137443|0.089840|60.95%|
|GT 굽음 <5도, 충분한 진행량|1,648|0.138580|0.114304|77.55%|
|위 속도 안정과 직진 조건 교집합|1,179|0.128695|0.075942|51.52%|

집단은 중첩되므로 비중을 더하지 않는다. 속도 안정은 미래 구간 GT의 진행속도 범위이며 실제 현재 vx나 가감속 없는 상태라는 뜻이 아니다. 직진 조건은 모든 GT 구간 길이>0.05m, 총 길이>=3m, 최대 구간 heading 절댓값<5도다.

직진에 가까운 행이 전체 유효 횡오차 질량의76.94%를 차지한다. 종 절대오차11.94cm와 횡6.58cm를 공식 PREFIX의 합으로 쓰지 않는다. 이 결과는 회전/급감속만 겨냥해서 전체 개선을 기대하기 어렵다는 근거다. 해상도가 현재 오차의 원인이라는 인과 증거는 아니다.

## 우선 권고: 원본 세부 정보를 실제 motion 관측에 추가

현재 motion은 전방 현재와 과거4장(-.1/-.2/-.5/-1.0초),768x432다. 원본 JPEG1920x1536을 같은 undistort/1920x1080 crop으로 처리해1152x648을 만들면, 현재768 캐시에 없는 세부 정보를 읽을 수 있다. 기존768을 단순 확대하는 것은 새 정보 비교가 아니다.

FINE은 같은768 영상으로 얻은 correlation_fuse map을 더 세밀하게 읽는 실험이었다. SHARED768은 이미 계산한 과거768 FPN을 scene에 연결한 변경이었다. 현재 H4-PROGRESS에서 원본1152 motion의 동일 예산 학습 결과는 아직 없다. C2F는 같은768 관측의 matching refinement였으며 새 원본 세부 정보를 추가하는 실험을 대신하지 않는다.

가장 먼저 검토할 구체적 구현은 기존768 경로를 유지하고1152 temporal 특징을 zero-initialized projection으로 추가 통합하는 것이다. 새 RGB 관측을 사용하는 시각 특징 분기이며 raw status/pose/goal의 새 motion 입력이나 수식 XY 보정을 만들지 않는다. 기존 pair memory와 continuous motion/state/history 소비 지점을 일관되게 연결해야 한다. 상세 graph와 학습 예산은 구현 전 단일 protocol로 고정한다.

- 초기 correction=0에서 실제 checkpoint의 FP32/BF16 plan과 auxiliary 출력 parity를 먼저 확인한다.
- 공통 backbone의 BN 실행 정책과 RNG를 유지한다. 해상도 변경 branch의 존재만으로 baseline이 바뀌지 않게 한다.
- 새 motion matching은 기존 radius4가 표현하던 물리 범위를 고려한다. 입력을1.5배 키우고 feature-cell radius를 그대로 두면 같은 시야에서 상대 탐색 범위가 작아진다. radius6 또는 동등한 좌표 처리를 명시한다. 채널/초기화 변화도 실험 일부다.
- 직접 주 비교는 같은1152 canvas/graph/초기화에서 low-detail control(같은 원본을768로 축소 후1152 확대)과 native1152다. 단순 기존768 모델 비교는 matching/graph 변경까지 포함한 실용 비교로 따로 표시한다.
- 두 arm 모두 같은 DEV 부모, data stream, 추가 학습량을 적용하고 train/V0 PREFIX·첫2초·일반주행·종횡을 평가한다. 수백 update의 손실만으로 정보의 효용을 판단하지 않는다.

이렇게 기존 경로를 보존하면 SHARED768의 step0 붕괴를 피할 수 있는 설계를 만들 수 있다. 실제 parity 전에는 보존 완료라고 주장하지 않는다. 또한 영구적인 성능 유지 또는 새 정보 활용 성공을 보장하지 않는다.

## 독립 다음 후보: 현재6-camera scene 원본1152

Motion 관측 확대가 새 움직임 정보를 읽는 가설이라면, scene 확대는 차선·원거리 객체·도로 방향의 세부 근거를 읽는 가설이다. 6-camera는 이미 모두 사용하고 있으므로 추가 카메라라고 부르지 않는다. 같은 최종 공통 scene을 occupancy/lane/planner가 함께 사용하고 A2 query-only status 경계를 유지한다.

두 변경을 처음부터 합치지 않는다. Motion 고해상도의 성능과 비용을 본 뒤 또는 독립 matched arm으로만 비교한다. 새 backbone 계열·새 agent task·과거시점 추가까지 동시에 넣지 않는다.

## 비용과 투자의 한계

이전4090 비용 시제품에서 motion 교체1152/radius6는 약41.29ms, 현재6-camera scene만1152는37.87ms였다. 정확도 결과가 아니며, 위에서 권고한 기존768+새1152 추가 경로는 이전 교체 graph보다 연산이 많다. 41.29ms를 새 추가 경로의 측정값으로 승계하지 않는다. 같은 전체 forward/FLOPs 측정과 원본 cache 생성시간을 먼저 확인한다.

LaneGeometry의 공유층 gradient가 작다는 관측은 있다. 하지만 두 parameter의 일부 microbatch gradient norm만으로 실패 원인을 확정할 수 없고, AdamW의 최종 update 영향과도 같지 않다. lambda를 크게 늘리면 해결된다고 가정하거나 자동 스윕하지 않는다. SplitRead를 더 깊게 분리하는 후보도 완전히 반증된 것은 아니지만, 지금은 동일 영상 표현의 작은 변경을 반복하기보다 새로운 관측 비교를 우선한다.

원본 해상도 강화는 아직 검증하지 않은 정보 가설이다. 큰 개선을 확인한 해결책이나 서버0.12 예측이 아니다. 기존 서버0.133684828 및 완료된 CTRL FULL은 보존한다.

## 근거

- ../a2_splitread_lanegeom_20260921/TERMINAL_REVIEW_KO.md
- LATEST_ERROR_GROUPS.json: 이번 최신 DEV 그룹 재계산
- ../a2_full_error_audit_20260921/RESULTS_KO.md: 실제 입력 크기/원본 crop/4090 비용
- ../../models/motiondrive_v2/motion_encoder.py: local correlation와 pooling
- ../../experiments/a2_progress_fourarm_20260921/arm_model.py: FINE/SHARED의 실제 변경
- ../../experiments/a2_motion_fresh_20260919/motion_model.py: 기존 C2F
- [RAFT 원문](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf): 여러 해상도의 대응 정보와 고해상도 flow 유지의 설계 참고. RAFT 이식 지시가 아니며 MotionDrive 성능 예상치의 근거로 사용하지 않는다.
