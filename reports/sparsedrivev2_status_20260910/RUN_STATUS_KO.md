# 영상 status 대체 검증과 공통 인지 모델 — 진행 중

활성 goal은 아직 완료되지 않았다. 상태 오차 진단은 끝났고, 실제 status를
planner에 직접 넣지 않는 세 모델의 대조 학습이 진행 중이다.

## 완료된 진단

- 기존 원본 재현: TUNE 1,998행, batch8 D3 **0.13074861**.
- 실제 P7 영상 status를 양쪽 입력에 그대로 치환: D3 **1.170503**,
  shortlist oracle **1.112857**. 원후보 평균 **17.58/200** 유지.
- 최종 CE head에만 P7 값을 치환: D3 **0.243276**, oracle **0.090494**.
- vx bias ±0.1m/s 양쪽 적용: **0.157206 / 0.160285**.
- 24개 조건 × 3개 입력 위치, 동일 1,998행. 3 GPU의 baseline 출력/ID
  bitwise 동일, 원본 캐시 재현, GT 분리·고정 bank 좌표 검사 통과.
- P7 양쪽 치환 ΔD3=+1.03975, paired 11-session bootstrap 95% CI
  [+0.83495,+1.20652], 20,000회 seed0.

이는 고정 모델의 입력 교란 결과다. 새로운 영상 추정기의 불가능성이나
재학습 모델의 성능 상한으로 해석하지 않는다. 기존 0.13은 실제 pose에서
계산한 status를 사용하는 결과이며 규정 승인 또는 영상 status 성능이 아니다.

전체 진단: [RESULTS_KO.md](analysis_v3/RESULTS_KO.md),
[aggregate.json](analysis_v3/aggregate.json),
[그래프](analysis_v3/status_sensitivity.png).

## 현재 실행

원격 worktree:
`/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910`

브랜치: `codex/sparsedrivev2-status-20260910`.
진단 코드 commit `63c220a`, 새 대조 모델 코드 commit `2319076`.

| GPU | 실행 | 차이 |
|---|---|---|
| 0 | temporal_a_repeat_s0_v1 | 과거 입력에 현재 정면 영상을 반복 |
| 1 | temporal_b_history_s0_v1 | 실제 과거 정면 -0.1/-0.5초 |
| 4 | temporal_c_common_s0_v1 | 실제 과거 + 공통 영상 인지 query에만 실제 status 조건 |

모든 모델은 원 planner와 최종 CE head에 constant zero status를 전달한다.
영상 상태 예측은 보조 loss/진단 전용이다. Goal은 원모델이 완료한 후보에
최종 점수를 매길 때만 쓴다. 현재 3카메라와 과거 정면 2장을 처리하며,
공통 영상 특징은 실제 current occupancy/lane과 planning에 함께 쓰인다.
같은 public initialization, bank, train/tune, 파라미터, 증강, seed,
2,000 step 예산이다. 공개 학습 파라미터 41,808,827개를 재사용하고,
새 파라미터는 562,055개다. 새 모델 CPU 테스트 19개 및 실제 GPU의
초기 공개 모델 동일성·직접 status=0 관측·forward/backward canary가 통과했다.

실행 상태의 authoritative source는 원격
`reports/sparsedrivev2_status_20260910/launches/*.json`와 실제 PID,
`work_dirs/sparsedrivev2_status_20260910/*/train.jsonl`이다.
이 파일의 설명만으로 실행 생존이나 완료를 판단하지 않는다.

## 남은 일

1. 세 모델 terminal 결과와 같은 행/초기화/소스/학습 순서를 검증한다.
2. A→B, B→C의 D3·후보군 최선·선택 오차·인지 성능을 비교한다.
3. 학습된 모델의 batch1 입력 경로·실제 입력 전처리·추론 비용을 확인한다.
4. 0.15 달성 여부와 미달 원인 및 후속 결정을 재현 가능한 결과로 남긴다.

규정의 공통 특징 간접 활용을 바탕으로 C를 설계했지만, 이름만 인지 모듈인
상태 전달을 정당화하지 않는다. 실제 데이터 흐름과 인지 효과를 함께 검토한다.
최종 주최 측 code-review 승인을 받은 것으로 표현하지 않는다.
