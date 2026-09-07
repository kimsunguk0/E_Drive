# E2E Driving 공식 지표 정합 감사 — 2026-09-08

## 결론

현재 MotionDrive V2의 `weighted_d3`, 즉 6개 누적 waypoint의 시점별 Euclidean 오차에
`[11,11,5,5,2,2]/36`을 적용한 값은 운영국이 후속으로 명시한 **ADE@1s, ADE@2s,
ADE@3s의 평균(`L2_avg`)**과 정확히 같다. 따라서 P7의 고정 loss·metric·protocol은 이
감사로 변경하지 않는다.

다만 이 수식 정합은 내부 tune 점수와 비공개 test/public leaderboard 점수의 직접 비교를
허용하지 않는다. 공식 제공 test 경로는 과거 7개 입력 프레임을 처리한 뒤 **clip당 마지막
frame의 궤적 하나**를 제출하지만, 내부 tune은 `frame >= 30`인 반복 window 1,998행을
동일 가중한다. 서버 scorer와 최종 clip 가중 구현은 제공되지 않았다.

이 보고서는 앞선 검토에서 PDF 13쪽의 일반식만 보고 공식 지표를 6점 단순평균으로 설명한
부분을 명시적으로 정정한다.

## 문서 간 충돌과 해소 순서

1. [공식 대회 API](https://dxchallenge.ai.kr/dxchallenge/backend/api/competitions/5)가
   2026-09-08 07:47 KST에 제공한 참여 가이드 PDF는 SHA-256
   `4a18cf76cca2b20171c6f41d6d95e321085f85c9cf6433a4f14cb65d9cade9c1`, PDF metadata
   생성일 2026-06-23이다. PDF physical page 14, 인쇄면 **13쪽**은
   `L2=(1/T) sum_t ||pred(t)-GT(t)||_2`라고만 적는다. 6개 점 전체에 적용하면 이것은
   ADE@3s 단독으로 읽히므로 후속 자료와 충돌하는 초기·불완전 정의다.
2. 운영국 제공 HTML 가이드
   `/NHNHOME/data/sukim/adcl/src/etri_vad/ETRI_E2E_Driving_Challenge_Guideline/site/test.html`
   (SHA-256 `07ce97fd969a1fda17574ee53e5349b5d29f1c02cb57ae030498b87184ca5b7c`,
   timestamp 2026-07-28) lines 871–877은 각 1/2/3초 L2를 해당 누적구간 ADE로 계산하고,
   최종 `L2_avg`를 세 값의 평균으로 계산한다고 명시한다.
3. 운영국 사전설명회 화면 보존본 `adcl_pre.zip`
   (`/home/a/Pictures/Screenshots/adcl_pre.zip`, SHA-256
   `19dc56594cfc8c35d2c3ff927847a75ed58114a0565f56b305cf2df1feeabf31`)의 slide 44
   (`Screenshot from 2026-08-07 10-42-56.png`, SHA-256
   `3bd0f76add29bc5632fc417c34476b78a2361475d72a4c1997e5a12668d54088`)도 같은 정의와
   VAD tiny 예 `(.299 + .541 + .842)/3 = .561m`를 제시한다.
4. 최신 [`OPEN_ISSUE.md`](/home/a/adcl_motiondrive_v2/OPEN_ISSUE.md) SHA-256
   `ba8b50097450612b985a250fc7f75027467d1da1140555fdc155926efc079006` lines 87–102는
   바로 위 PDF/slide 충돌을 운영국에 질의한 기록이다. 답변은 slide가 맞으며
   1초=첫 2점 평균, 2초=첫 4점 평균, 3초=첫 6점 평균을 다시 평균한다고 명시한다.
   이 파일은 commit `c92ceb7676367ef750bf31afe1d991b2c6067a1c`에 2026-09-07 11:43 KST
   기록됐지만, 답변 원문의 별도 공개 URL과 응답 시각은 파일에 없다.

따라서 더 구체적인 후속 HTML·설명회·운영 답변을 적용하면 PDF의 모호성은 지표식에
대해서는 해소된다.

## 네 계산의 정확한 구분

각 frame/clip에서 0.5초 간격 6개 누적 위치의 pointwise Euclidean 오차를
`d1,...,d6`라 두면 다음은 서로 다른 값이다.

- **6점 전체 ADE 또는 ADE@3s:** `(d1+d2+d3+d4+d5+d6)/6`.
- **운영국 최종 `L2_avg`:** `mean((d1+d2)/2, (d1+...+d4)/4,
  (d1+...+d6)/6)`, 즉 `sum(d * [11,11,5,5,2,2]/36)`.
- **1/2/3초 endpoint-only 평균:** `(d2+d4+d6)/3`; 운영 답변이 배제한 해석이다.
- **여섯 endpoint의 단순 평균:** 첫 항 ADE@3s와 같고, 최종 `L2_avg`와는 다르다.

## 제공 코드와 현재 코드의 정합

운영국 제공 VAD baseline에서 확인되는 계산층은 다음과 같다.

- `.../projects/mmdet3d_plugin/VAD/VAD.py:447-456,650-661` (SHA-256
  `3760466baabe43e6c596c158df44442fff1d14f4cc91b5cb410b446e49d170c9`)은 증분
  예측·GT를 누적 위치로 바꾼 뒤 first 2/4/6 waypoint를 각각 `compute_L2`에 전달한다.
- `.../projects/mmdet3d_plugin/VAD/planner/metric_stp3.py:290-308` (SHA-256
  `d492b2bee64b84e4980f772b06d293851118ac05e0b7b7bc904ab4e5cb0aad57`)은 전달받은
  구간의 waypoint Euclidean 거리를 산술평균한다.
- `.../projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py:1823-1838` (SHA-256
  `6c2cc70ada95e5a1c7fab40d40959e73b2df8ae17f986064f19eeb704e68e384`)은
  `fut_valid_flag`가 참인 sample만 포함해 세 누적 ADE를 각각 동일 sample 가중으로
  평균한다. 세 값의 최종 평균은 위 HTML·slide·운영 답변에 규정돼 있으며 baseline
  파일 안에 `L2_avg` scorer 자체는 없다.

현재 구현은
[`scripts/motiondrive_v2_training.py`](/home/a/adcl_motiondrive_v2/scripts/motiondrive_v2_training.py)
SHA-256 `3929538dcc58c786b242680e84ca768ecb3674ec7bfbbd784a8143fd7d1aed70`
lines 17,94–99에서 같은 가중치를 사용한다. 이미 누적인 `plan_abs`와 `gt_plan` 사이의
FP32 Euclidean 거리를 계산하므로 추가 `cumsum`을 하지 않는 것이 맞다.
[`scripts/train_motiondrive_v2.py`](/home/a/adcl_motiondrive_v2/scripts/train_motiondrive_v2.py)
SHA-256 `52eac2a28ba12f4724b7f18a3c2d6889241a58ec5b665ef44bd0e2f5966268d2`
lines 202–218,264–285는 6점 GT가 모두 유효한 행만 허용하고 per-row D3의 산술평균을
`official_d3`로 기록한다. 세션 동일가중 평균과 proxy 가중값은 별도 보조값이다.

가중치 유도 helper `/NHNHOME/data/sukim/adcl/src/challenge_metrics.py` (SHA-256
`243b9c20c3c402869bea047bc7111e75955acbc8868b992d344a5f39f3c7e845`, mtime
2026-08-10)는 `.gitignore`의 `/src/` 규칙 아래 있는 우리 측 보조 코드이며 운영국 scorer가
아니다. MotionDrive V2의 위 구현은 commit
`bd38aa84a7b98da80a41fbdcf61e111550d19cd6`(2026-09-07 14:41 KST)에서 처음 들어왔다.
따라서 정합의 1차 근거는 helper의 `official` 주석이 아니라 운영국 HTML·slide·답변과 제공
VAD `first 2/4/6` 계산이다.

## 평가 모집단과 mask의 남은 비동치

운영국 제공 `tools/data_converter/etri_test_converter.py:19-24,152-208` (SHA-256
`77a700dff251c4a1a8f45734ee50bf5e3e3c08a46ab28c110069ff2c9def2de2`)은 각 test
clip에서 `[-30,-25,-20,-15,-10,-5,0]`의 7개 history 입력을 만든다. 제공
`tools/etri_test_submit.py:48-65` (SHA-256
`39520f5437f78bee37cab3b0009a1b9a0c05869e53a9123178b4a468c74e7323`)은 clip마다
stream state를 초기화하고 7개를 순서대로 forward한 뒤 frame 0의 6-point trajectory
하나만 제출한다. 이는 organizer가 임의의 cold-start training frame을 점수에서 제외한다는
일반 규칙이 아니라, 제공 test 제출경로에서 실제 확인되는 범위다.

내부 `MotionDriveDataset`은
[`scripts/motiondrive_v2_data.py`](/home/a/adcl_motiondrive_v2/scripts/motiondrive_v2_data.py)
SHA-256 `83b6c74177c131116f48feb968aa8f96821b5e820823843e8b2e97ba7af171ca`
lines 140–174에서 `frame >= 30`인 모든 선택 행을 만들고 lines 250–267에서 각 행에
6점 `plan_valid=True`를 붙인다. tune 보고값은 37 scenes/11 sessions의 1,998 반복-window
행평균이다. 공식 private test는 clip당 제출 하나이므로 모집단과 가중 단위가 다르다.

실제 서버 scorer 코드, invalid/missing clip 처리, 중복 제출 행 처리, clip/scenario의 추가
가중은 제공되지 않았다. 따라서 현재 수식은 운영국 정의와 맞지만 내부 `.3x` D3를
리더보드 `.0x` score 또는 예상 순위로 변환해서는 안 된다.

## 2026-09-08 공개 리더보드 문맥

2026-09-08 08:02 KST에 [공식 E2E leaderboard API](https://dxchallenge.ai.kr/dxchallenge/backend/api/competitions/8/leaderboard?public=true&page=1&limit=50&mode=all)를
확인했다. 응답 SHA-256은
`7b575723a733cfe4dc6cb98fb14a2cc3bf37cd503afd1a32d3f92d2d47f680b2`,
`visibility=public`, 기본 mode `all`, 총 33 entries였다. 표시 상위 3개 entry는
Neural Drive `0.09241055500217414`, E2Ego `0.13053658320753567`, spilab
`0.13758930028156613`이었다. UI의 정확한 열 이름은 `점수`, API 필드는 `score`이며,
공개 응답은 이것이 1차 L2인지 최종 latency-adjusted Error Score인지 구분하지 않는다.

[공식 E2E track metadata](https://dxchallenge.ai.kr/dxchallenge/backend/api/competitions/8)는
ACTIVE, 공개 종료 2026-09-24 00:00 KST를 나타낸다. 대회 진행 중이고 이후 코드·설명·재현성
심사가 남으므로 현재 표가 수상 결과 관점에서 잠정적이라는 것은 공식 상태와 절차에 근거한
해석이지, API가 `provisional` 필드를 제공한다는 뜻은 아니다.

우리 RTX 3090 timing은 공식 RTX 4090 `Tinfer`가 아니며, 모델·장치·평가 모집단이 다른
내부 D3/timing을 공개 `score`와 직접 결합하지 않는다.
