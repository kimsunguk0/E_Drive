# A2 다음 실험안 통합 검토

2026-09-21. 검토 대상: `A2_Next_012_SplitRead_LaneGeometry_20260921.md`와 직전 제안인 길이·방향 decoder 분리 및 원본 1152 motion.

기준 작업 소스: `b3e967a0c692db8319ac29d7821f0bf3308ca542`.
대응 공개 미러: `7245848c547984100093a97bfde8b17fed1d139a`.
이번에는 원문·현재 소스·기존 결과를 읽고 checkpoint 파일 SHA와 산술을 확인했다. 새 학습, 실제 checkpoint forward, DEV 재평가, 공식 제출은 하지 않았다.

## 1. 결정

즉시 실행 후보는 리뷰의 **CTRL-NEXT / SPLITREAD / LANE-GEOM**을 채택한다. 이번 요청은 검토이며 아직 실행을 시작한 것은 아니다.

- 첫 구조 비교는 전체 query/decoder 분리보다 좁은 **SplitRead**로 한다.
- **LaneGeometry**는 다른 학습 신호를 주는 별도 후보로 병행할 가치가 있다. 아래 최근접점 모호성 처리를 보완한다.
- 원본 **전방 현재+H4 motion 1152**는 준비 후보로 유지한다. 리뷰의 ‘현재 6-camera scene 1152’와 다른 실험이다.
- 세 변경을 처음부터 합치지 않는다. 완료된 CTRL FULL은 보존한다.

새 관측·새 감독·공유 범위 변경은 각각 다른 질문이다. 이 계획도 0.11~0.12를 보장하지 않는다.

## 2. SplitRead: 첫 버전은 리뷰 쪽이 더 구체적이다

|구분|직전 제안|리뷰의 SplitRead|
|---|---|---|
|공유 유지|backbone / scene / motion|왼쪽 + 공통 decoder와 기본 query|
|분리|query / decoder / TemporalRead / 출력부|TemporalRead / LN / 비선형 hidden / scalar 출력부|
|질문|scene·motion 조회 전반을 분리하면 좋은가|공통 문맥 뒤 temporal 조회와 readout 공유를 풀면 좋은가|
|첫 시행|더 큰 범위와 비용|기존 함수를 복사해 시작하는 작은 비교|

SplitRead에서 분리되는 직접 조회 대상은 기존 temporal pair memory다. 두 branch가 scene을 각각 독립적으로 다시 읽는 구조는 아니다. 공통 decoder의 scene 문맥은 두 branch 모두 유지한다.

현재 소스의 `TemporalRead`는 최종 특징이 아니라 **잔차**를 반환한다. 실제 명세에는 아래 덧셈을 반드시 포함한다.

```text
d = shared_decoder(queries, scene + motion + predicted_state_memory)
h_length  = d + read_length(d, pair_memory)
h_heading = d + read_heading(d, pair_memory)
z_length  = head_length(h_length)
z_heading = head_heading(h_heading)
```

`read_*`의 출력만 head에 보내면 기존 함수가 보존되지 않는다. 두 read를 현재 학습된 read에서 복사하고, 두 head의 LN/hidden을 복사하며 마지막 row 0/1을 각각 옮긴다. 무작위/zero 초기화로 바꾸지 않는다.

현재 채널 128 기준 추가 파라미터 계산은 리뷰와 일치한다.

- TemporalRead 한 개 추가: 66,560개.
- LN + 128→128 hidden 복사에 따른 증가: 16,768개.
- 합계: **83,328개**.

복사 후 Parameter storage가 독립적인지와 실제 parent 출력 parity를 구현 시 확인한다. 길이 loss만으로는 heading 전용 branch에 이론상 직접 gradient가 없지만, FP32 삼각함수·누적 차분에는 작은 수치 잔차가 있을 수 있다. 이 성질과 AdamW weight decay에 의한 parameter 변화는 구분한다. 공유 decoder/backbone은 계속 양쪽 gradient를 받는다.

리뷰가 말하는 CPU 합성 검사 파일 `check_proposals.py`, `synthetic_checks.json`은 현재 전달 경로에서 찾지 못했다. 따라서 제시된 parity 수치를 독립 재현했다고 주장하지 않는다. 위 parameter 계산과 실제 기존 구조는 확인했다.

## 3. LaneGeometry: 연속 기하 감독은 새 축이지만 mask를 보완한다

실제 producer는 `map.parquet`의 연속 segment를 width=1의 binary raster로 만든다. scene grid 중심은 cell-center 방식이고 크기는 64×48, 범위는 x[-10,70], y[-32,32]m다. 따라서 리뷰의 1.25×1.333…m cell 크기는 맞다. 이것은 궤적 정확도의 하한이 아니다.

새 offset/axis target은 기존 raster에서 복원하면 안 된다. 원래 map polyline을 현재 ego 좌표로 변환해 연속 segment 위 최근접점을 계산한다. 미래 ego 경로로 선을 고르지 않는다.

**가장 중요한 추가 수정: offset과 axis에 독립적인 유효 mask를 둔다.**

- 평행한 두 선에서 거리가 거의 같으면 방향축은 같아도 offset이 왼쪽/오른쪽으로 모호할 수 있다. axis만 끄는 규칙으로 해결되지 않는다.
- 교차점에서는 최근접점이 같아 offset은 명확하지만 axis가 모호할 수 있다.
- 거의 같은 거리의 유효 후보들이 만드는 최근접점과 무방향 axis를 각각 비교한다. 모호한 target만 제외한다.
- 같은 line의 중복·인접 segment를 다른 의미의 경쟁 선처럼 취급하지 않는다. 같은 최근접점·같은 axis는 유지할 수 있다.
- 길이 0, 비정상 좌표, 지원하지 않는 line geometry는 제외한다. 유효성·거리·방향 차이 문턱은 train 자료만 보고 사전에 고정한다.

`[cos(2θ), sin(2θ)]`는 line 순서에 주행 방향을 강요하지 않는 적절한 표현이다. 좌우 flip에서는 grid y축 반전과 offset_y 및 sin(2θ) 부호 반전을 함께 수행한다.

기존 `lane_valid`는 알려진 선 주변 width=13 support와 camera visibility의 교집합이다. 이것은 영상에서 실제 가림 없이 보인다는 인증은 아니다. 기존 의미를 유지하고, 새 4m support 및 별도 모호성 mask를 추가한다.

새 offset/axis는 서로 valid count가 다를 수 있다. 각 항에 대해 **전체 effective batch의 유효 성분 수**로 정규화한 뒤 합한다. 기존 loss normalizer 계약에 새 키를 무심코 섞지 말고 geometry 분모를 별도로 관리한다. NaN target은 loss 연산 전에 치환하며 all-invalid는 graph와 연결된 0을 반환한다.

λ=0.05, 첫 500 update 선형 증가, 새 head LR=5e-5는 단일 제안값으로 사용할 수 있다. 초기 train batch에서 backbone/scene으로 들어가는 gradient와 이후 update를 기록한다. head만 target을 맞히는지와 planner PREFIX 개선은 구분한다. V0를 보며 λ를 반복 선택하지 않는다.

가능하면 geometry head는 학습 전용으로 두고 배포에서는 제거한다. 그 경우 head를 제외한 student를 명시적으로 export하고 기존 추론 graph에 strict load하여 출력 parity를 확인한다. 이 head의 결과를 planner의 새 숫자 입력으로 쓰는 것은 이번 제안에 포함하지 않는다.

## 4. 학습 예산: 변화의 종류에 따라 판단을 달리한다

직전 답변에서 ‘전체 예산부터 다시 학습’이라고 넓게 말한 부분은 구분한다.

- **SplitRead**: 함수 보존 복제이므로 동일 CTRL에서 6,852-update matched continuation으로 먼저 확인하는 것이 합리적이다.
- **LaneGeometry**: 기존 planning forward를 보존하고 감독을 추가한다. 같은 continuation 비교가 가능하되, 약 1.31회 노출의 결과를 새 감독 방식의 모든 수렴 조건으로 일반화하지 않는다.
- **원본 1152 motion**: 입력 특징과 matching 범위가 바뀐다. 별도 초기화·적응/전체 공동 학습 계획이 필요하며 위 continuation 대조군과 순수 한 변수 비교로 섞지 않는다.

리뷰의 세 run은 같은 부모·sample/augmentation stream·fresh AdamW·기존 loss·LR를 사용한다. 기존 3,426 결과가 새 CONTROL을 대신하지 않는다. 신규 head 생성은 RNG를 격리하고 기존 tensor·샘플 순서를 바꾸지 않는다.

평가는 0/3,426/6,852, primary terminal을 유지한다. 일반 주행 first1s뿐 아니라 **1~2초와 2~3초 기여**, 길이·방향·종횡 손익도 기존 producer로 기록한다. 공식 서버 집계에서는 첫 두 포인트, 다음 두 포인트, 마지막 두 포인트가 각각 약 33.3%, 36.9%, 29.8%를 기여한다.

## 5. FULL 부모는 반드시 명시한다

DEV는 이미 한 차례 continuation을 끝낸 CTRL에서 시작한다. 그 이후의 새 레시피를 FULL로 옮긴다면 대응하는 부모도 **완료된 CTRL FULL stage2**다.

|역할|파일|실제 파일 SHA-256 확인|
|---|---|---|
|새 DEV 부모|`work_dirs/a2_progress_fourarm_20260921/P-CTRL-s1/ckpt_step3426.pth`|`5021300b0d876a6b59b2137d78fd96ab21b80bfed7141ebc1248e5c8bc05a37a`|
|대응 FULL 부모|`work_dirs/a2_progress_fourarm_full_20260921/P-CTRL-FULL-STAGE2-s1/ckpt_step4156.pth`|`fe821be8546cc54db27c3a81e57bc04126461b29d1a74732e3ad308953c3a3a2`|

두 hash 모두 이번 검토에서 B200 실제 파일을 읽어 확인했다. 이를 원래 서버 0.1336848 FULL에 곧바로 +8,311 하는 것과 혼동하지 않는다.

`ceil(6852 × 101520 / 83700) = 8311`은 맞다. 이것은 추가 노출량 이전 기준이며 최적 update라는 뜻은 아니다. FULL의 in-fit V0로 새 설정이나 best checkpoint를 고르지 않는다.

## 6. 1152 및 정밀도 후보의 위치를 정정한다

직전 제안은 **motion의 전방 현재+과거 4장만 원본 1152×648**이고, 리뷰의 대기 후보는 **현재 6-camera scene 1152**다. 서로 다른 영상 경로를 바꾸는 실험이다.

이번 즉시 세 run에 고해상도를 섞지 않는 판단에는 동의한다. 다만 motion 1152는 새 RGB 세부 정보라는 독립 가설이므로 대기 준비 후보로 보존한다. 768 cache 확대가 아닌 원본에서 동일 crop/FOV로 생성해야 한다. 기존 비용 시제품 약 41.3ms는 정확도·최종 배포 인증이 아니다.

리뷰의 ‘scene sampler FP32는 과거 A2-BASE-NOM에서만 확인’ 부분은 최신 기록으로 갱신한다. 현재 CTRL에서도 이미 같은 프로세스의 frozen 비교를 했다.

- CTRL baseline: 0.150285498.
- sampling만 FP32: 0.150304091.
- 차이: +0.000018593.

따라서 이것은 이번 라운드에서 새로 재평가할 후보가 아니다. 근거: `reports/a2_next_direction_20260921/FROZEN_FP32_SAMPLING.json`.

## 7. 근거 및 한계

직접 확인한 소스:

- `models/motiondrive_v2/planner.py`: 현재 LN→Linear→GELU→2-channel 출력부.
- `experiments/a2_temporal_read_20260920/temporal_model.py`: TemporalRead 잔차 및 공통 decoder.
- `models/motiondrive_v2/scene_encoder.py`: 실제 cell-center와 공통 raster/인지/계획 연결.
- `scripts/build_scene_supervision_v2.py`: map 변환, width=1 target, width=13 support, camera visibility.
- `scripts/motiondrive_v2_training.py`: 기존 loss 묶음 및 normalizer.
- `reports/a2_progress_fourarm_20260921/result_step3426.json`: 현재 CTRL 및 후보 기록.

Recon은 gradient 충돌을 측정한 층을 task-specific하게 분리하는 연구다. 이 문서의 수동 SplitRead가 같은 진단을 완료했다는 의미가 아니다. [Recon 원문](https://arxiv.org/abs/2302.11289)

VAD는 vectorized map/agent 표현의 연구 근거다. 이번 nearest-segment offset/axis auxiliary의 성능이나 대회 점수를 검증한 연구는 아니다. [VAD 원문](https://arxiv.org/abs/2303.12077)

모든 후보는 기존 A2 query-only status 경계를 유지한다. 구조 승인을 운영국에서 새로 받았다는 주장도, 예상 서버 점수도 하지 않는다. 이번 결과물은 수정 검토이며 실행·예약 기록이 아니다.
