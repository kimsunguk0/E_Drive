# SplitRead / LaneGeometry 실제 설계 검증

2026-09-21. 사용자 요청: 작업에 들어가기 전에 설계를 한 번 더 검증.

**판정: 수정한 SplitRead를 첫 구조 대조로 유지한다. LaneGeometry는 별도 보조 감독 대조로 유지하되, 약한 초기 gradient와 지도 정합의 한계를 기록한다. 새 학습·예약·DEV 전체 평가·공식 제출은 실행하지 않았다.**

이번 검사는 논문 또는 합성 tensor만의 검토가 아니다. B200 GPU0에서 실제 DEV P-CTRL checkpoint와 사전에 고른 train 3행으로 초기 함수·gradient·입력 의존성·재구성을 검사했다. Lane target은 실제 map.parquet와 pose로 생성했다. Optimizer를 만들지 않았고 parameter/buffer hash는 전후 같았다.

## 기준과 범위

- 기준 소스: `b3e967a0c692db8319ac29d7821f0bf3308ca542`.
- 부모: `work_dirs/a2_progress_fourarm_20260921/P-CTRL-s1/ckpt_step3426.pth`.
- 실제 파일 SHA: `5021300b0d876a6b59b2137d78fd96ab21b80bfed7141ebc1248e5c8bc05a37a`.
- 고정 train index: 0 / 30,000 / 60,000. 해당 row는 30 / 37,560 / 77,790이다. 오류나 V0 성능으로 선택하지 않았다.
- 실험 구현은 `experiments/a2_design_preflight_20260921/`에 격리했다. 기존 학습·배포 모델 소스와 가중치는 변경하지 않았다.
- 이전 검토: `A2_Next_012_Combined_Review_20260921.md`. 이번 기록은 그 문서의 미실행 항목 중 제한된 실제 검사를 추가한 것이다.

## 1. SplitRead: 초기 함수와 의도한 분리 확인

공통 decoder의 `decoded`를 유지하고 각 branch가 `decoded + TemporalRead(decoded, pair_memory)`를 읽는다. TemporalRead가 반환하는 잔차만 출력 head에 보내는 구조로 바꾸지 않았다.

|검사|실제 결과|
|---|---:|
|추가 파라미터|83,328|
|FP32 초기 plan 최대 절대차|0.0000114441 m|
|BF16 초기 plan 최대 절대차|0.0000038147 m|
|공통 scene / occ / lane / motion / state / history 출력 차이|0|
|길이 loss → 길이 branch gradient norm|7.092799|
|길이 loss → 방향 branch gradient norm|6.24e-8|
|위 두 norm의 비율|8.79e-9|
|PREFIX → 길이 / 방향 branch norm|6.069252 / 29.981380|
|새 프로세스 strict 재구성 후 출력 차이|0|

두 branch는 Parameter storage를 공유하지 않는다. 복사 전후 CPU RNG도 같다. 초기 두 branch의 gradient를 합하면 원래 공유 TemporalRead/hidden의 gradient와 상대오차 3.35e-7 / 9.14e-8로 일치했다.

이는 분리 구현이 의도대로 연결됐다는 근거다. **공유 backbone/decoder의 경합이 제거됐다거나, 기존 길이·방향 오류의 원인이 경합으로 입증됐다는 뜻은 아니다.** 새 정보는 추가되지 않고, 실제 이득은 동일 예산 CONTROL 대비 PREFIX로 판단해야 한다.

### 발견한 구현 문제

전체 A2 모델을 `deepcopy`하면 status query의 Python hook closure가 원래 모델을 계속 참조했다. 첫 시도는 `shared causal status context is missing` 오류로 중단됐다. 이는 기존 제출 모델의 오류가 아니라 새 시제품을 구성하는 방법의 문제였다.

수정: 정상 model factory로 새 공통 graph를 만들고 기존 state를 strict load한 다음, 분리할 planner 모듈만 복사한다. 실제 forward와 별도 프로세스 재구성을 통과했다. 첫 실패 기록과 당시 소스도 보존했다.

## 2. LaneGeometry: 실제 지도에서 성립하지만 mask를 보완

기존 producer와 같은 current ego 변환으로 원본 polyline의 연속 segment를 사용했다. 기존 binary lane target과 valid mask도 3행 모두 정확히 재현했다. 표본 schema는 `lane_id`, `points`뿐이며 centerline/edge를 임의로 분류하지 않았다.

|고정 train 표본|4m 이내 기존 support|최종 offset valid|최종 axis valid|
|---|---:|---:|---:|
|20260112-105434 / frame30|737|729|730|
|20260209-154800 / frame60|1,110|1,098|1,110|
|20260213-135258 / frame90|1,090|1,076|1,090|

좌우 flip의 grid 반전·offset_y·sin(2θ) 부호와 polyline 순서/방향 반전은 실제 표본에서 차이 0이었다. 최근접 vertex 대신 segment상의 최근접점을 사용했다. 평행선 동거리에서는 offset만, 교차점 방향 모호성에서는 axis만 제외하는 합성 경계조건도 확인했다.

수정한 두 문제:

1. 같은 polyline의 인접 segment를 서로 다른 경쟁 선으로 취급하면 offset valid가 477/799/664개로 과도하게 줄었다. **먼저 polyline별 최근접 후보를 구하고, 선 사이의 모호성을 판단**하도록 고쳤다. 같은 선 내부의 실제 동거리 분기도 별도로 처리한다.
2. 첫 후보와만 방향을 비교하면 0°/10°/20° 세 선이 같은 점에서 만나는 경우 저장 순서에 따라 axis mask가 달라졌다. **모든 후보 쌍이 허용 범위 안에서 일치해야 한다**는 규칙으로 고쳤다. 6가지 순열을 모두 통과했다.

마지막 순서 의존성 수정은 CPU로만 재검사했다. 고정 3행의 최종 offset/axis/mask는 GPU gradient 검사 때의 target과 byte 단위로 같으며, SplitRead 및 geometry head 구조도 그대로다. 근거는 `verification_receipt.json`이다.

### 지도 overlay 해석

`check3/train_geometry_overlay.png`의 3개 표본에서는 축 반전이나 큰 투영 오류를 보지 못했다. 차선 표식 및 도로 굽음과 대체로 정합하지만, 차량에 가려진 선도 투영된다. 기존 camera visibility는 가림 검사가 아니다. 이 관찰은 **전체 데이터 정합이나 cm급 지도 정확도를 인증하지 않는다.**

Offset은 위치, `[cos(2θ), sin(2θ)]`는 무방향 축이다. 실제 진행방향·차선 중심·통행 가능 corridor의 정답으로 확대 해석하지 않는다. GT map은 loss에만 쓰고 planner 입력이나 추론 후 궤적 snapping에는 쓰지 않는다.

## 3. Loss와 입력 경계

- Full effective batch의 offset/axis 유효 성분 분모를 각각 사용한다. Microbatch 합과 전체 batch의 loss 차이는 1.19e-7, gradient 차이는 0이었다.
- 일부 invalid target을 NaN으로 바꿔도 유한했다. 전체 invalid는 graph와 연결된 0 loss와 0 gradient였다.
- Geometry head가 읽는 raster는 기존 occupancy/lane head와 planner가 읽는 동일 shared scene임을 확인했다.
- 새 head의 모든 파라미터와 공유 backbone/scene 표본 파라미터에 유한한 nonzero gradient가 발생했다.
- 고정 영상에서 제공 status 또는 goal만 바꾸어도 motion / pair memory / state / history 출력은 차이 0이었다. 새 외부 status 경로는 추가하지 않았다. 기존 scene의 간접 조건화는 유지한다.

λ_geo=0.05를 적용한 초기 gradient의 크기는 검사한 backbone conv에서 PREFIX+LEN의 **0.1494%**, scene key projection에서 **0.0536%**였다. 새 head가 무작위 초기화된 train 3행의 두 파라미터 표본 결과이며, 전체 gradient나 학습 이후의 비율이 아니다. 첫 500-step ramp 중에는 더 작다.

**결정:** 이 숫자만으로 λ를 늘리지 않는다. 제안값 0.05와 ramp500을 유지하고, 실제 학습 시 초기·ramp 종료·후반에 같은 종류의 gradient/update와 planning 결과를 기록한다. Head loss만 줄고 공유 표현의 유효한 변화가 없으면 그 제한을 보고한다. V0를 보며 λ를 자동 스윕하지 않는다.

## 4. 다음 실행 명세와 아직 미검증인 부분

설계 수준에서는 **CTRL-NEXT / SPLITREAD / LANE-GEOM**의 독립 비교를 유지한다. 동일 DEV CTRL 부모, 동일 sample/flip stream, fresh AdamW, 6,852 update, batch16, 기존 loss와 LEN0.25를 사용한다. SplitRead는 기존 LR를, 새 geometry head만 제안한 별도 LR를 적용한다. 현재 완료 FULL이나 제출물은 다시 만들지 않는다.

실행 구현에서 남은 것은 기존 trainer의 하드코딩된 부모·3,426 scheduler를 새 명세로 정확히 교체하고, optimizer group·sample stream·새 graph export 및 실제 비용을 확인하는 일이다. 지금 만든 시제품 파일은 생산용 checkpoint 형식이나 launch 승인을 대신하지 않는다.

이번에 하지 않은 일:

- 새 모델 DEV PREFIX와 서버 점수 측정.
- 새 SplitRead의 RTX4090 전체 forward/FLOPs, 학습 처리량 측정.
- Geometry head를 optimizer에 넣은 실제 update 및 학습 후 head 제거 export 검사.
- 전체 train geometry cache 생성, 새 학습 시작/예약, FULL 재학습/제출.

**0.12를 보장하는 설계는 아니다.** 확인한 것은 기존 함수를 보존하며 공유 범위를 바꾸는 구현과, 실제 지도에 연결되는 새 감독의 계산적 정합성이다. 다음 성능 판단은 실제 matched CONTROL 대비 결과가 필요하다.

## 재현 자료

- `check1/checks.json`: 첫 hook 소유권 실패와 보존 소스.
- `check2/checks.json`: 실제 checkpoint GPU 검사 및 새 프로세스 재구성.
- `check3/checks.json`: 마지막 geometry 순서 불변성 CPU 검사.
- `verification_receipt.json`: 마지막 target 동일성, source graph 동일성, 대형 runtime artifact 색인.
- `run_artifact_index.json`: 파일 hash.
- `experiments/a2_design_preflight_20260921/`: 격리한 시제품과 재현 스크립트.

대형 시제품 weights와 3행 입력 tensor는 `work_dirs/a2_design_preflight_20260921/check2/`에만 보존하며 Git에 넣지 않는다. Overlay PNG도 저장소의 이미지 제외 정책을 유지해 Git에 넣지 않고, 서버 및 로컬 검증 자료에 보존한다. Git에는 재현 코드·검사 JSON·검토문·hash 색인을 기록한다.
