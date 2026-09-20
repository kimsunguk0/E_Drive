# H4-PROGRESS FULL — 2026-09-21

사용자 요청: “일단 지금까지 가장 잘 나온거 해서 Full 돌리자 1회 제출하게”. 현재 최고 단일 DEV 후보 H4-PROGRESS0.151178860의 학습 레시피를 공식 train 전체에 적용하고 한 번 제출할 파일을 준비한다. 기존3모델 저장 예측 평균0.146197790은 단일 모델 점수·배포 검증과 구분한다.

- 모델: QREFINE + TemporalRead +6구간 길이/방향 → absolute XY. DEV와 같은 graph/입력 경계/loss.
- 초기값: 공개 nuImages R50 trunk + 동일 random nontrunk. 초기 model SHA `75eec301bdc2abb533e0f0fa4ffa73c5b2e4340b7119a84f11b2d1b2212ed200`.
- 데이터: 공식 train376 unique scenes/101,520행(train+tune+val). 중복 scene/row 없음.
- 예산: `ceil(20554*101520/83700)=24931` update. Batch16/micro8, seed1, backbone5e-6/head5e-5, warmup200/cosine, fixedBN/BF16, flip0.5, LEN0.25, 기존aux0.2.
- 선택: 고정 terminal24931. 기존 V0는 FULL 학습에 들어가므로 평가 수치를 일반화 성능이나 후보 선택 기준으로 사용하지 않는다.
- 입력: 별도 FULL 캐시에 동일5시점 [-10,-5,-2,-1,0] nominal causal status. 기존 train/tune 캐시와 값이 정확히 같고 val만 추가. Raw status는 공통 scene query에만 사용한다.
- DEV 격리: FULL 가중치·특징·통계를 DEV에 되돌리지 않는다. DEV terminal 가중치를 이어서 학습하지 않고 성공한 초기값부터 전체 레시피를 이전한다.

전체 데이터의2-step smoke로 실제 initial SHA/graph/loss/행 수/BN/정규화 경로를 확인한다. 같은 graph의 기존 preflight를 재사용하고 raw adapter와 full terminal 가중치 검사를 추가한다.

학습 후에는 terminal load 검증 → raw fixture8개의 cache/adapter 출력 정합 → 공식 방식 전체 forward FLOPs → 실제 test1,125clip 추론/shape/유한값/clip 격리 → `submission.zip` 생성으로 이어진다. ZIP은1,125clip과 정수 `__flops__`가 든 `submission.json` 하나만 포함한다. 출력은 이미absolute6×2이므로 serving에서 cumsum하지 않는다.

최종 제출 후보에는 checkpoint·portable code·Dockerfile·raw 입력 예제·hash/검사 기록을 함께 보존한다. 업로드가 실제 수행되기 전에는 제출 완료나 quota 소모로 기록하지 않는다. 테스트 정답은 사용하지 않으며 공식 서버 점수를 미리 예측하지 않는다.

진행: launch_main.json / runtime/watcher_status.json. 결과: completion.json / PACKAGE_KO.md. 학습 경로: work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1. 제출 패키지: work_dirs/a2_progress_full_20260921/package.
