# A2 nominal status + 4-head shared scene 실행 기록 (2026-09-18)

사용자가 승인한 통합안의 첫 실행이다. 현 단계는 A2 FULL 확보와 동일 조건의 BASE/MH4 DEV 비교다.
후속 SIDE-SCENE/QREFINE은 이번 세 run에 포함하지 않는다.

| GPU | Run | 학습 행/scene | Updates | 평가 역할 |
|---|---|---:|---:|---|
| 0 | A2-FULL-NOM-s1 | 101,520 / 376 | 24,931 | V0는 in-fit 진단 |
| 1 | A2-BASE-NOM-s1 | 83,700 / 310 | 20,554 | 미학습 V0 1,998행 |
| 2 | A2-MH4-NOM-s1 | 83,700 / 310 | 20,554 | 같은 V0, 같은 예산 |
| 3 | 검증 및 후속 SIDE-SCENE 준비 | — | — | 독립 보류 슬롯 |

## 입력과 구조

- 입력 provided_status5만 causal pose + frame 차이 × 0.1초의 배포형 producer로 통일했다.
  실제 timestamps는 pose와 frame을 연결하는 데이터 준비용 join에만 사용하고 derivative 시간은 nominal이다.
  미래 pose는 fit에서 제외한다. state/history supervision은 기존 실측 시간 정의를 보존한다.
- status는 공통 scene query를 조건화한다. image-derived value를 선택하는 가중치에 영향을 준다.
  제공 status를 scene value에 더하거나 motion/state/history 및 planner에 직접 주는 경로는 추가하지 않는다.
- MH4: head당 query/key 32, image value 32, concat 128.
  각 head의 독립적인 Q/K 선형변환을 identity로, 출력 128×128 선형변환도 identity로 초기화한다.
  기존 Q/K 생성과 camera/time/height/scale metadata, sampling/grid, backbone, planner, loss는 유지한다.
  기존 single-head와 파라미터 수가 같은 실험은 아니다: 26,557,208 → 26,581,784.
- 배포 입력 정책을 맞춘 기존 A2의 V0 PREFIX는 0.164455314083.
  실측 시간 status를 쓴 기존 0.164280978527와 구분한다.
  이 세 새 run의 성능은 아직 측정되지 않았다.

## 공통 학습 조건

기존 성공한 r0_init_tplus.pth upstream에서 seed1로 새 optimizer와 전체 공동 학습을 시작한다.
A2 terminal에 이어 붙이는 continuation이 아니다.
Shared initial tensor SHA256: 116f67a476b5ab519ec4384d6e15d5f5e36e83d1c89812f733b056ba5874593f.
BASE와 MH4는 같은 초기 공통 tensor, sample order, loss, LR, budget를 사용한다.

유효 batch16, microbatch8, BF16 / planning FP32, fixed BN, flip0.5,
backbone LR 5e-6 / 기타 5e-5, warmup200, AdamW weight decay0.01, clip5, LEN0.25.
DEV 평가 주기3426, FULL 평가 주기6345. 고정 terminal이 기본 후보이며 FULL의 V0로 선택하지 않는다.

## 실행 전 검증 결과

preflight.json:
- 실제 영상 8개에서 BASE→MH4 초기 scene/occupancy/lane/plan 및 motion/state/history 차이: FP32/BF16 모두 0.
- 기존 등록 A2 예측 재현 차이 0; 학습된 A2를 MH4에 옮긴 초기 plan 차이도 0.
- 실제 planning loss backward에서 head별 Q/K gradient가 유한하고 서로 다르다.
- status 교체 시 motion/state/history 변화 0, invisible source의 evidence/gradient 0.
- 1,998개 nominal status가 기존 P0 producer 출력과 bitwise 일치.
- 원본 state/history/GT 및 좌우반전 정합성 보존.

세 arm 모두 2 optimizer updates + V0 전체 평가 + checkpoint 저장 smoke 완료.
smoke_summary.json과 protocol_*_smoke_mb8.json에 결과와 원본 위치를 보존한다.
2-step 점수는 학습 배관 검사이며 모델 성능 비교나 후보 선택에 사용하지 않는다.

throughput_benchmark.json:
고정 유효 batch16, lr0, 입력 decode 제외/전송 포함 기준 median update 시간은
BASE mb8 0.5078s / mb16 0.4675s, MH4 mb8 0.5424s / mb16 0.4911s였다.
사전에 정한 '두 arm 모두 10% 이상 단축' 조건을 충족하지 않아 기존 microbatch8을 유지했다.
학습 weights는 보존/수정하지 않았다. 이 수치를 실제 전체 학습 속도로 단정하지 않는다.

## 산출물과 재현

- 코드: experiments/md_a2_nominal_mh4_20260918/
- 공통 encoder 변경: models/motiondrive_v2/scene_encoder.py
- 원본 입력 cache: data/etri/motiondrive_v2/a2_nominal_status_20260918/
- cache manifest 사본: nominal_input_manifest.json. 대형 입력 배열은 서버에 보존한다.
- main run: work_dirs/md_a2_nominal_mh4_20260918/<ARM>-s1/
- main 실행: experiments/md_a2_nominal_mh4_20260918/launch.py
- 실행 후 PID/명령/코드 commit: launch_receipt.json
- stdout: runtime/<ARM>-s1.log
- metrics.jsonl, manifest.json, eval_step*.json, diagnostics_step*.json은 각 run 폴더에 저장된다.
- diagnostics는 첫 2초 기여, 일반 주행, 진행량 공통/시간변화 오차, 동일 GT mask 종·횡 절대오차를 기록한다.
  종·횡 오차를 더해 D3라고 부르지 않고 진행속도 오차를 v0/가감속 타이밍의 인과로 해석하지 않는다.
