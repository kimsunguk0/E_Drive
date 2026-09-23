# 사후 분석: VAD 방식 can_bus를 EXT에 더한 결과 (2026-09-23 밤, 제출과 무관)

## 구조 (`experiments/a2_final_push_20260923/ext_canbus/ext_model.py`)
- 제공 status5를 공통 scene raster에 **더한다**: `raster += canbus_mlp(status/scale)`.
  - 위치는 `scene_encoder.refine` 직전이며, 영상 전역 문맥 `context_global`을 더하는 자리와 같다.
  - 공식 VAD의 `bev_queries + can_bus_mlp(can_bus)`(`VAD_transformer.py:267-270`, 베이스라인 설정 `use_can_bus=True`)와 같은 역할이다.
- refine 뒤의 raster는 occupancy·lane·planner가 공유한다. motion encoder는 status를 받지 않는다.
- goal은 5 s 종점 argmin 선택에만 쓴다(v7과 같다).
- canbus MLP는 0으로 초기화했다. 시작 V0는 0.16124027로 v7과 비트 단위로 같다.
- 학습: 몸통 동결 상태에서 canbus MLP + `scene_encoder.refine` + planner만 학습했다. 설정은 v7 레시피(K=15, 5230 step).

## 결과 (쌍둥이, held-out V0)
| | v7 | cb | 차이 |
|---|---:|---:|---:|
| V0 | 0.13024 | **0.12695** | −0.00328, 세션 bootstrap CI [−0.00447, −0.00180], 9/11 |
| 정지 / 주행 / 29% 혼합 | 0.05755 / 0.13542 / 0.11284 | 0.05710 / 0.13193 / 0.11023 | |

- seed 잡음 기준(v7 재현본 − v7): +0.00009, CI [−0.00116, +0.00101].

## 규정 특성
- status를 다른 행 값으로 바꿔도 motion_features·motion_pair_features·state_hat·history_hat 변화는 0.0이다(질문10 경계 유지).
- 영상을 회색으로 바꾸면 0.1259 → 3.7867이다.
- status를 다른 행 값으로 바꾸면 0.1259 → 0.9249다(의존도가 크다).

## 해석
- 9/18 A3-DIRECT(FPN gate, motion에도 status가 들어감, −9%)를 뺀 이유는 질문10이었다. motion encoder를 분리한 이 형태는 그 문제를 피한다.
- 서버 환산은 약 0.114다. 몸통까지 처음부터 학습하는 경우는 검증하지 않았다.
