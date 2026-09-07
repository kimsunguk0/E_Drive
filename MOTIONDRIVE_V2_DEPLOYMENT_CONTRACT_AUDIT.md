# MotionDrive V2 — 배포 입력 정합 감사와 수정 절차

2026-09-07. 기존 P1 학습은 원 조건으로 보존한다. 이 문서는 발견한 오류와
후속 수정 범위를 기록하며, 수정 후 성능이나 운영국 승인을 주장하지 않는다.

## 1. 테스트에서 실제 제공되는 입력

아래 근거는 B200에 보관된 공식 배포 소스/설명서다. 실제 test/final-val 자료는
열지 않고 인터페이스만 검사했다. `OPEN_ISSUE.md`의 정보 흐름 제약도 계속 적용한다.

공식 코드 루트:
`src/etri_vad/ETRI_E2E_Driving_Challenge` (A),
`src/etri_vad/ETRI_E2E_Driving_Challenge_Guideline/site` (G).

| 항목 | 확인한 계약 | V2 적용 |
|---|---|---|
| 시간 | G/test.html:817–818: 원 timestamp/date/scenario ID 미제공. A/tools/data_converter/etri_test_converter.py:25,160–164: DT=.1로 합성 | 모델에는 nominal [.1,.2,.5,1.]초. 실제 timestamp가 필요한 추론을 가정하지 않음 |
| 과거 pose | G/test.html:755–764, converter:82–89: xyz + roll/pitch/yaw | full SE(3) scene 정합에만 사용. 숫자를 motion/state/planner token으로 보내지 않음 |
| 각도 | G/dataset.html:768,810: ego rpy rad, camera extrinsic Euler deg | 단위를 분리하여 변환 |
| calibration | G/test.html:735: clip별 제공 | 실제 undistort/crop/resize와 같은 K 사용. 학습 공통 행렬을 테스트에 무조건 고정하지 않음 |
| 영상 | 과거 3초 31frame 제공, 기존 converter는 7frame 선택 | V2는 현재 6대 + front 과거 −1/−2/−5/−10frame을 직접 구성 |
| 궤적 | V2 `plan_abs`는 현재 ego의 6개 누적 XY | A/tools/etri_test_submit.py:64의 VAD 전용 cumsum을 V2에 재적용하지 않음 |

입력 정규화는 RGB/255 후 ImageNet mean/std이며 기존 0–255 RGB 정규화와 수학적으로
동일하다. BGR 이중 변환·이중 정규화·VAD padding448을 붙이지 않는다.
현재 6대는 768×432, 과거 front는 동일 FOV에서 PIL bilinear로 384×216이다.
저해상도 current front의 추가 resize/encoder는 모델 안에 있으며 시간 측정에도 포함한다.

goal은 full SE(3) 현재 ego 좌표 XY(m)다. world XY 차이 또는 yaw-only 변환으로
대체하지 않는다. clip 간 feature/state 재사용은 없다. 학습용 `MotionDriveDataset`은
GT/supervision을 요구하므로 제출 adapter로 그대로 사용할 수 없다.
현재 inference bundle은 **가중치 export이며 완성된 제출 wrapper가 아니다.**

## 2. 원 timestamp와 nominal 입력은 분리한다

현재 P1은 실제 train/tune timestamp 차이를 입력으로 사용했다. 이 조건을 뒤늦게
바꾸지 않는다. `evaluate_motiondrive_v2_planning.py --time-input raw`로 원 결과를
재현하며, LAST6000 G×S 주 분석은 raw만 받아 nominal 결과 혼입을 거절한다.

별도 `--time-input nominal`은 model-input dict의 `time_offsets`만 FP32 고정값으로
교체한다. 영상·pose·goal·GT·valid·행 순서·rawtime session split은 보존한다.
따라서 nominal 평가가 원 P1보다 좋거나 나쁘더라도 별도 배포 조건 결과로 표시한다.

실제 기존 supervision의 모델 입력 차이(train54810 / tune1998):

- train의 최대 절대 차이는 .063556초. 20260205의 5개 scene에서 각각270frame이
  5ms를 넘고, 20260219-105056의 10frame도 5ms를 넘는다.
- tune의 최대 차이는 .001021초이며 5ms 초과 scene은 없다.
- 이는 입력 차이의 크기이지 모델 오차 변화량이 아니다. rawtime grouping은
  시나리오 독립성 검증에 필요하므로 nominal 입력으로 바꿔도 그대로 유지한다.

## 3. rear_wide 캐시/투영행렬 오류 확정

원본 train `20260112-105434/frame30`을 메모리에서 재구성했다.
rear 원본은1920×1536이며 캐시는 bottom crop `[0,456,1920,1080]` → .4 resize → JPEG95다.
bottom 재구성은 기존 캐시와 JPEG bytes 및 pixels가 100% 일치했다.
top crop 재구성은 MAE29.5934/255로 일치하지 않았다.

반면 `scripts/motiondrive_v2_data.py:calibration_from_info`는 rear를 top crop으로
취급했다. 저장 행렬은 그 잘못된 top 기준과 float32 오차 수준으로 일치한다.
실제 영상에 대한 투영 위치 오차는 세로182.4px다. 다른5개 카메라는 정상 범위였다.
train203/tune37 모두 동일 bottom crop 메타지만, 전체 JPEG 픽셀을 검사한 것은 아니다.

근거: `reports/motiondrive_v2_rear_crop_audit_train.json`,
SHA256 `5a7a93185165dc5bd82342360699ccdb1c3b15d98c7791523f149a8ab46ef097`.
원본/캐시/calibration 데이터는 변경하지 않았다.

독립 CPU 확인에서 supervision의 높이0/1m 합집합 visibility는 수정 전후3064셀,
XOR=0이었다. `occ/lane` target은 crop과 무관하고 valid는 이 visibility에 의존하므로,
**정확한 최종 행렬로 동일성을 다시 검사한 뒤** 기존 scene 감독 배열을 보존할 수 있다.
그러나 모델 높이2m의 rear visibility는127→123셀이고 rear 픽셀 위치는 이동한다.
감독 mask 불변을 모델 무영향으로 해석하면 안 된다.

## 4. 변경 범위와 fail-closed gate

1. 기존 `train_tune_rawtime` 및 hardlink 원본은 불변으로 둔다.
2. 실제 cache_meta K와 calibration extrinsic을 사용한 별도
   `train_tune_geometry_v2` edition을 만든다. rear만 교체하고 나머지5행렬은 bitwise 보존한다.
3. train/tune의 공통 geometry와 `camera_visible(old)==camera_visible(new)`를 검사한다.
   같으면 scene NPZ/JSON을 독립 복사하고 SHA·배열 불변을 기록한다.
   다르면 중단하고 원 객체/지도에서 valid를 다시 생성한다. 기존 valid에 단순 AND는 금지한다.
4. 원 PKL SHA, 원 canonical NPZ SHA, 새 canonical NPZ SHA, cache_meta SHA를 구분한다.
   기존 manifest의 `calibration_sha256`은 원 PKL SHA였으며 derived 행렬 SHA가 아니었다.
5. 새 fixture는 새 이름으로 생성하고 `lidar2img` 외 입력/GT 불변을 검사한다.
6. 수정 입력을 기존 checkpoint에 적용한 진단과 수정 조건으로 재학습한 결과를 구분한다.
   P1은 잘못된 rear geometry를 포함한 통제 실험이지 구조 성능 상한이 아니다.

후속 제출 adapter는 위 crop/nominal/full-SE3/absolute-output 조건으로 train clip을
재구성하여 학습 입력과 대조하고, 최종 wrapper 전체 forward를 3090에서 다시 측정한다.
현재 측정된32.23ms는 학습된 P1 step1000 모델 자체이며 최종 wrapper/4090 수치가 아니다.

## 5. GT-free raw adapter의 실제 train8 입력 검증 완료

`models/motiondrive_v2_inputs.py`는 clip별 calibration, 과거31행과 제공+50goal행 pose,
현재6대/과거front4개 JPEG만 사용한다. timestamp/status/정답 중간 궤적은 모델 입력에
넣지 않는다. full-SE3 과거 정합은 scene 입력이며 raw motion/planner 수치 토큰이 아니다.
네트워크 호출과 제출 파일 작성은 이 adapter의 역할이 아니다.

`build_motiondrive_v2_deploy_fixture.py`는 지정 train4scene×frame30/180으로
test-shaped fixture8개를 만들었고, 새 `audit_motiondrive_v2_deploy_inputs.py`가
학습 systemPython의 기존 tensor export와 직접 비교했다. 기준은 export 이후
영상 재디코딩 없이 사용했고 calibration/time만 P2 C1/T1 계약으로 교체했다.

- 현재영상/과거영상/투영행렬/과거정렬/nominal시간/goal 모두8/8 bitwise, maxabs0.
- 재구성 JPEG80/80 SHA가 저장된 학습 캐시 출처 SHA와 동일.
- 원본 파일·메모리 GT·코드 SHA 불변, CUDA/model forward 없음.
- adapter/fixture/runner B200 CPU tests55개 통과.

보고서 `reports/motiondrive_v2_deploy_input_parity_train8.json`의 SHA는
`5f60c0f6dbaaadb0774a2596f1c1f2e2be920b25f17bdd9e166ff323c7c00fa7`이다.
8개 train clip에 대한 입력 재현이며 모든 clip·다른 라이브러리 빌드·최종 forward까지
보장하지 않는다. 실제 제출 환경에도 이 계약과 검사를 묶어야 한다.
