# Status 선택기의 실제 추론 확인 (2026-09-10)

캐시 특징이 아니라 **원 C strict loader + 최종 선택기 조합으로 원 TUNE37
1,998행을 실제로 추론**했다. head는 `pv_status_real_s0_v1/step_004000.pth`
(SHA `3b1d9827…`), 원 C는 `temporal_c_common_s0_v1` (SHA `b9dcc56a…`),
path/velocity filter (128,20)/(64,64).

## 결과

| 지표 | B8 | **B1 (배포 조건)** |
|---|---:|---:|
| official_d3 | 0.13827430 | **0.13802299** |
| shortlist oracle | 0.07145978 | 0.07152224 |
| selection regret | 0.06681452 | 0.06650075 |
| C base scorer D3 | 0.33592426 | 0.33908774 |
| 3초 지점 L2 | 0.47735912 | 0.47492556 |

같은 조건의 zero-status head는 B8 0.2336235836, B1 0.2324676942였다.
따라서 실제 추론 기준 **B1 0.2324677 → 0.1380230, −0.0944447**이다.

## 검증

- **B8는 offline 캐시 평가와 사실상 동일하다.** `xy_bitwise_equal: true`,
  `ids_bitwise_equal: true`, 선택 ID 변경 0건, 최대 좌표차 0.0,
  D3 차이 평균 −3.14e−09.
- **B8 cache parity 통과.** 원 frozen tune 캐시의 `rows`, `candidate_ids`,
  `candidate_valid`, `scores`, `token`, `goal_xy`를 live 값과 bit 단위로 대조했다.
- **B1은 B8과 다르다.** 1,998행 중 732행에서 선택 ID가 달라졌고 D3 평균 차이는
  −0.00025131이다. BF16 batch 산술 차이이므로 두 조건을 같은 추론으로 보지 않는다.
  배포 조건은 B1이다.
- **goal 경로 감사 통과.** `goal_only_after_complete_candidates: true`,
  `candidate_token_goal_invariant: true`. head 서명은
  `output, goal_xy, provided causal status for completed-candidate selection only`.
- status는 offline 오버레이 조인이 아니라 **live batch의 `perception_status`**에서
  읽는다. 이미 공통 인지 query를 조건짓는 바로 그 텐서다.
- `GT_passed_to_model: false`. route counts는 B1에서 public/temporal/relative 각 1,998.

## 시간

B200, batch1, GPU 상주 입력, warmup 10 + 50회 측정, 모델 forward만.

| | mean | p50 | p95 |
|---|---:|---:|---:|
| wall | 22.037 ms | 22.038 ms | 22.076 ms |
| CUDA event | 22.016 ms | 22.017 ms | 22.055 ms |

zero-status head가 22.8434 / 22.8765 ms였으므로 status 추가의 시간 비용은
측정 오차 안이다. raw decode·I/O·H2D·GT·route hook은 제외했고 RTX4090
측정이 아니다.

공식 가이드의 `D3 × (1 + max(0, T_ms − 100)/200)`에 D3 0.13802299를 넣으면,
Error Score ≤ 0.15를 만족하는 RTX4090 시간은 **약 117.4 ms 이하**다. 이는
시간 예측이 아니라 필요조건의 역산이다.

## 범위

TUNE은 프로젝트 전반에서 반복 사용된 탐색 집합이라 독립 최종 확인이 아니다.
seed 하나다. 그룹 제외 재학습 확인은 하지 않았다. 제출 wrapper 연결과 RTX4090
실측도 없다.
