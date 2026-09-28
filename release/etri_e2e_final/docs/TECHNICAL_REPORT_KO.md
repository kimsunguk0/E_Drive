# 기술 및 전략 설명서 — MotionDrive V2 + EXT (goal 종점 기반 후보 선택)

## 1. 결과 요약
- 리더보드 L2_avg **0.117067** (L2_1s 0.0654 / 2s 0.1153 / 3s 0.1705)
- 1회 forward, `__flops__` 743.8 G (cutoff 7,053 G), RTX 4090 추론 31.2 ms (시간 벌점 없음)
- 제출 이력: 0.198 → 0.186 → 0.134 → 0.120 → **0.117**

## 2. 입력과 사용 경로
| 입력 | 사용 여부 | 경로 |
|---|---|---|
| 6개 카메라 현재 영상 | 사용 | 백본 → scene(BEV) raster, 전역 영상 문맥 |
| 전방 카메라 과거 영상 (−0.1/−0.2/−0.5/−1.0 s) | 사용 | motion encoder, scene 과거 시점 샘플링 |
| 과거 ego pose | 사용 | (a) 위 4개 과거 프레임의 기하 정렬, (b) 현재 ego status 5값(vx, vy, ax, ay, yaw rate) 계산 |
| 현재 ego status 5값 | 사용 | **공통 scene attention query에만** 더한다. planner·motion encoder에는 들어가지 않는다 |
| 5초 목표점(+50 frame) | 사용 | (a) 공통 scene query 조건, (b) **후보 궤적 선택**(5초 끝점과의 거리 argmin) |
| command / vad_cmd | 미사용 | — |

과거 궤적과 status를 planner에 직접 넣거나 단순 임베딩으로 넣지 않는다. 최종 궤적 좌표는 모두 영상 feature를 읽는 planner가 만든다. 영상을 단색으로 바꾸면 held-out L2가 0.129에서 3.877로 무너진다.

## 3. 모델 구조
1. **백본:** ResNet-50 + 경량 FPN(128 ch). 6 카메라 768×432, 과거 전방 영상 384×216.
2. **Scene encoder (공유 BEV raster):**
   - BEV 격자 셀마다 query를 둔다. query = 셀 위치 + goal 상대 위치 + status 조건.
   - 셀 3D 점을 카메라에 투영해 현재·과거 영상 feature를 샘플링하고, attention으로 읽는다(2회 refinement). query는 어느 영상 증거를 읽을지만 정하고, 값은 영상 feature다.
   - 전역 영상 문맥을 더한 뒤 SpatialMix 2블록으로 정제한다.
   - 결과 raster는 occupancy head, lane head, planner가 공유한다.
3. **Motion encoder:** 현재·과거 전방 feature 쌍의 correlation(반경 4)으로 motion token을 만든다. 영상만으로 ego 상태·과거 궤적을 예측하는 head가 있다(보조 감독). pose·goal·status 입력은 없다.
4. **Planner:**
   - transformer decoder가 waypoint query 6개로 scene token, motion token, 영상 예측 상태 token을 읽는다.
   - waypoint마다 시간별 motion feature를 다시 읽는다(temporal read).
   - 출력은 PROGRESS 형식이다: 구간마다 길이(softplus)와 진행 방향을 내고 누적해서 위치를 만든다.
5. **EXT 후보 확장 (3단계):**
   - planner가 후보 **15개**(종방향 5 × 횡방향 3)를 만든다. 후보마다 mode 임베딩과 mode별 길이·방향 bias가 있다.
   - 각 후보는 **5초까지 10점**이다. 3.5~5.0 s는 6번째 구간에서 이어지는 학습된 잔차로 만든다.
   - 추론: `argmin_k ‖후보_k(5.0 s) − goal‖`로 하나를 고르고, 앞 6점(0.5~3.0 s)을 좌표 변경 없이 제출한다.
   - 선택 단계에는 학습 모듈이 없다. status·과거 정보도 쓰지 않는다. 영상부터 출력까지 하나의 네트워크다.

## 4. 학습
공통: 제공 train 376 scene, 프레임 30 이상의 모든 프레임(101,520행). AdamW, BF16, BN 통계 고정, gradient clip 5.

| | 1단계 | 2단계 | 3단계 (EXT) |
|---|---|---|---|
| 시작 | 공개 초기값 | 1단계 | 2단계 |
| 학습 파라미터 | 전체 | 전체 | planner + EXT 신규 파라미터 (몸통 동결) |
| update / batch | 24,931 / 16 | 38,070 / 16 | 6,344 / 16 (2프레임 간격 50,760행 × 2 epoch) |
| 학습률 (backbone / head) | 5e-6 / 5e-5, warmup 200, cosine | 1e-6 / 1e-5, warmup 100, cosine | planner 2e-5, 신규 1e-3, warmup 100, cosine |
| 손실 | 채점식 가중 L2 + 구간 길이 보조(λ 0.25) + occupancy·lane·motion 보조(각 0.2, 불확실성 가중) | 1단계와 같음 | 배정 후보의 채점식 가중 L2 + 0.3 × 5초 연장 L2 |
| 증강 | 좌우 반전 p=0.5 | 1단계와 같음 | 없음 |

**3단계 후보 배정:** 학습 샘플마다 5.0 s 끝점이 정답 5.0 s 위치에 가장 가까운 후보를 골라 그 후보만 학습한다. 학습 데이터에서 정답 5.0 s 위치는 목표점(+50 frame)과 같다. 그래서 학습과 추론이 같은 선택 규칙을 쓴다.

## 5. 검증 전략
- **held-out 쌍둥이:**
  - 레시피 판단은 37 scene(11 session)을 뺀 310 scene으로 학습한 쌍둥이 모델로 했다.
  - 제출 모델은 같은 레시피를 376 scene 전체에 적용했다.
  - 판정은 session 단위 bootstrap(2만 회)과 개선된 session 수로 했다.
- **서버 대응:** 서버 점수 ≈ 0.90 × 쌍둥이 held-out 점수였다(3회 관측).
- **후보 수 선택:**
  - held-out에서 후보 6·9·15·27·35·63·77개를 비교했다.
  - 후보가 많을수록 주행 구간은 좋아졌지만 정지 근처 구간은 나빠졌다.
  - 두 구간에서 모두 안정적인 15개를 채택했다.

## 6. 주요 실험에서 얻은 것
| 효과 있음 | held-out 효과 |
|---|---|
| 학습 데이터 확대 (203 → 310 → 339 → 376 scene) | 가장 일관된 개선 |
| 공통 scene query에 status 조건 | 큰 개선 |
| PROGRESS 출력, temporal read, 낮은 학습률 장기 추가학습 | 각각 개선 |
| **EXT: 영상 후보 + goal 종점 선택** | **−7.7% (서버 −10.5%)** |

| 효과 없음 | 비고 |
|---|---|
| 과거 프레임 추가(6시점), 측방 과거 영상, 주변 차량 미래궤적 보조 감독, optical flow 증류 | 가감속·회전 부분집합에서도 ±2% 안 |
| 입력 해상도 1152, ResNet-101 | — |
| checkpoint 평균, 가중치 soup, 학습 연장, 몸통까지 end-to-end 재학습 | — |
| command 필터, goal yaw 가중 선택 | goal 종점 선택과 정보가 겹친다 |

## 7. 재현성
- 같은 코드·seed로 B200에서 2단계와 3단계를 다시 학습했다. 종료 값이 1e-4 이내로 재현됐고, 테스트 예측은 제출본과 평균 2.3 cm 차이다.
- 이 제출물의 Docker 이미지로 RTX 4090에서 추론하면 제출본과의 차이 중앙값이 1.5 mm다(`VALIDATION_KO.md`).
- 결과가 비트 단위로 같지 않은 원인은 BF16과 비결정적 CUDA 누적 연산이다.
