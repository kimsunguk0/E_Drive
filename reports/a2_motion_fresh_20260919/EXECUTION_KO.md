# A2 motion / fresh 초기값: 2026-09-19 실행 기록

사용자가 승인한 1·2번을 23:43:33 KST에 독립 본 학습으로 시작했다.
구현 commit: `83a292f63245aea83516723a9fa222f2d3453421`.
이 파일은 착수 기록이다. 최신 상태는 `watcher_status.json`, terminal은 `RESULTS_KO.md` 및
`result_A2-*.json`을 우선한다. 아래 첫 batch 수치를 DEV 개선으로 해석하지 않는다.

|GPU|실험|주 변경|신규 학습 예산|
|---|---|---|---:|
|0|A2-C2F-MOTION-s1|stride16 대응 후 stride4 주변 재비교, motion feature에 추가|20,554|
|1|A2-FRESH-NUIM-s1|공개 nuImages trunk + FPN/scene/motion/planner 새 초기화|20,554|

GPU2에서 사전 검사를 완료했고 GPU3은 평가 여유 슬롯이다. 다른 사용자의 GPU4–6 작업은
이 실험과 무관하다. 두 실험의 변경을 섞지 않았다.

## 비교와 입력

완료된 A2-QREFINE-NOM-s1 terminal(DEV 약0.164252)을 대조로 재사용한다.
1번은 기존 QREFINE과 공유 초기 tensor SHA, 첫 batch의 모든 기존 loss, 행 순서가
정확히 같음을 확인했다. 새 분기의 생성은 원래 초기화 RNG를 바꾸지 않는다.
2번은 같은 graph·split·신규 update 예산·LR·sample stream을 사용하지만 기존 ETRI 학습
가중치를 상속하지 않는다. 과거 학습 노출까지 같은 실험이거나 scratch 수렴을 보장하는
예산이라고 설명하지 않는다.

기존 train310 83,700행 / V0 1,998행, nominal causal provided status, supervision 정의,
PREFIX 가중치, LEN 및 공동 인지 loss를 유지한다. 1번도 정확한 v0 복원을 통과 조건으로
삼지 않는다. 실제 terminal PREFIX와 일반 주행·첫 2초의 변화가 판정 기준이다.

Provided status는 공통 scene query에만 들어간다. 새 fine correspondence는 정렬 전
RGB 특징에서 계산하며 raw pose/status/goal을 입력하지 않는다. 같은 scene을
occupancy/lane/planner가 읽는다. 규정 개별 승인 주장은 하지 않는다.

## 실제로 완료한 검사

- Zero-init 시 scene·인지·motion·state/history·plan 전 출력이 FP32/BF16에서 정확히 일치.
- 영상 대응의 방향, 좌우 반전, 좌표 ramp, 공간적으로 변하는 flow, 영상 밖 mask 검사.
- 기존 최종 PREFIX에서 새 출력층으로 gradient가 흐르고, 출력층을 임시 1회 업데이트한
  검사에서는 fine descriptor·중간층·backbone까지 유한한 gradient가 연결됨.
- Status/goal만 바꿔도 motion/state/history 차이0. Shared scene 소비 tensor도 동일.
- Public trunk318개 tensor 완전 로드, 나머지161개 초기 tensor는 public load 전후 불변.
- 각 arm의 실제 batch16/micro8 2-update 학습과 V0 전체 평가 완료, NaN0.
- 사전 검사 가중치는 본 학습에 사용하지 않음. 본 학습은 새 프로세스·optimizer·초기값에서 시작.

전체 B1 FlopCounterMode: QREFINE729.992G, C2F744.130G.
B200 B1 BF16 median 약19.18/21.88ms. 이 값은 RTX4090 시간이 아니다.
변경 graph는 새 비용을 측정했으며 기존 single-head A2의 FLOPs를 승계하지 않았다.

## 진행 확인과 결과 보존

23:46 KST 확인: C2F200 step, FRESH250 step 모두 running. 최근 step 시간 약0.628/0.528초.
이는 초기 처리량 추정이다. 평가·I/O·자원 경합에 따라 완료 시각은 바뀔 수 있다.

모든 예정 평가의 예측과 진단을 `predictions_step*.json`, `diagnostics_step*.json`으로
각 run에 남긴다. 주 비교는 terminal이고 중간 V0 best는 자동 채택하지 않는다.
CPU collector는 각 terminal 완료 후 동일 row/GT 및 logged row/LR stream을 확인하고,
11-session paired bootstrap, 그룹·종/횡·첫2초 및 auxiliary를 기록한다.
양쪽이 완료되면 이번 결과 파일만 Git commit하고 미러에 push한다. 동시 작업·Git 충돌이
있으면 결과 파일을 보존하고 publication 오류를 별도 기록한다.

새 FULL·예산 확대·공식 제출은 collector가 시작하지 않는다.
