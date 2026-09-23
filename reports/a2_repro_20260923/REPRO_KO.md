# EXT-FULL-v7 재현 검증 (2026-09-23)

## 1. EXT 2단계 재학습 (부모 checkpoint 고정)
같은 코드·env·seed로 다시 학습했다. 설정: `EXT_K=5 EXT_LAT=3 EXT_SPREAD=0.06 EXT_LAT_SPREAD=0.04 EXT_EXT_W=0.3 EXT_ALL_W=0`, seed 20260923.

| 항목 | 원래 | 재현 |
|---|---:|---:|
| 쌍둥이 step 0 V0 | 0.16124027 | 0.16124027 (일치) |
| **쌍둥이 최종 V0 (held-out)** | **0.130237** | **0.130324** (+0.00009) |
| 쌍둥이 정지 / 주행 / 29% 혼합 | 0.05756 / 0.13542 / 0.11284 | 0.05890 / 0.13542 / 0.11323 |
| FULL step 0 in-fit | 0.11480739 | 0.11480739 (일치) |
| FULL 최종 in-fit | 0.087320 | 0.087342 |
| 테스트 1,125 clip, 제출본과의 가중 거리 | — | 평균 0.0206 m / 중앙값 0.0161 / p90 0.0447 |

- 재현 제출 zip sha256: `573d5251…`. 패키지 검사 실패는 없다.
- 비트 단위 불일치의 원인은 bf16과 비결정적 CUDA 누적 연산이다. 입력·초기 상태·데이터 순서는 step 0 비트 일치로 확인했다.
- 학습률이 높은 중반(1308 step)에는 곡선이 벌어졌다가(0.1598 대 0.1894) 종료 시점에는 수렴한다.

## 2. L-FULL6 몸통 재현 (FULL 부모 0.133685에서 +38,070 update, 진행 중)
- 래퍼: `experiments/a2_final_push_20260923/repro_lfull6.py`. 출력은 `work_dirs/a2_repro_20260923/`와 `reports/a2_repro_20260923/lfull6/`로 원본과 분리했다.
- 원래 실행(commit ac62e26)과 소스 차이는 pinned 24개 파일 중 `temporal_model.py` 하나다. 바뀐 것은 입력 형태 검사(`==4` → `>=4` time grid)뿐이고, 4-grid H4 graph의 계산은 동일하다.
- smoke(5 update): step 0 in-fit 0.10660578이 원래와 일치했다.
- verify_long 결과:
  - 새 프로세스 재로딩 예측 차이 0.0
  - BN·buffer 불변
  - **status를 +0.1 바꿔도 motion_features·motion_pair_features·state_hat·history_hat 변화 0.0.** A2 정보 경계가 유지된다는 뜻이고, 규정 검증 근거로 쓸 수 있다.
