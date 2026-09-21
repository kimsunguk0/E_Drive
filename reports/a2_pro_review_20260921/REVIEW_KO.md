# PRO 제안 검토: interval VECTOR + FINE-READ — 2026-09-21

**판정: 두 제안 모두 독립된 다음 실험으로 채택할 근거가 있다.** VECTOR는 기존 LENGTH 항을 교체하는 학습 목표 실험, FINE-READ는 이미 계산한 시각 motion map의 공간 정보를 더 보존하는 실험이다. 처음부터 결합하지 않는다. 과거 768 FPN의 shared scene 재사용은 다른 정보 경로의 독립 비교로 유지한다.

이번에는 문서·소스 대조, 저장 예측 재계산, CPU loss 검사, 학습하지 않은 FINE-READ 시제품의 4090 비용/초기 parity만 수행했다. **신규 학습·FULL 재학습·공식 제출은 시작하지 않았다.**

## 1. 실제로 받은 것과 확인한 범위

- `A2_H4_PROGRESS_013368_SingleModel_Evidence_20260921.zip`: 분석 JSON/producer, vector loss 참조 구현, CPU 검사 코드/결과, source provenance의 6개 파일. FINE-READ 구현이나 새 학습 결과는 포함되어 있지 않다.
- ZIP SHA256: `8d758a6c950509c34faba9ee721fa85184046b05d37bec3739e2b279009709b3`.
- `A2_73f9ea4_Review_Bundle_Request.md`: 73f9ea4 기준의 과거 검토 자료 요청서다. 새 모델 제안 또는 최신 소스·결과 묶음 자체는 아니다. 이 요청서 때문에 과거 자료 묶음을 다시 만들거나 새 전체 평가를 실행하지 않았다.
- ZIP이 기록한 `progress_model.py`, `temporal_model.py`, `length_auxiliary.py`, `motiondrive_v2_training.py`의 네 SHA256이 현재 B200 source와 **모두 일치**했다.
- 검토 시작 source: `d6680eaf9af7d79f0ff53200e70b4d7a7d6b1c94`, 공개 미러 `8931739cb3c14dd15ccca1f8cfd87cf2ca52d2b0`.
- DIRECT/PROGRESS 예측 파일 SHA와 1,998개 row/GT를 확인하고 제안의 구간 벡터 오차를 재현했다. 별도로 FULL의 **학습 내 V0**도 같은 계산을 추가했다.

## 2. 서버 산술과 개선 목표는 맞다

PREFIX 가중치 `[11,11,5,5,2,2]/36`에 따라 공식 집계의 기여는 다음과 같다.

|포인트|평균 L2|전체 기여|
|---|---:|---:|
|0.5/1.0초|0.072772575|0.044472129|
|1.5/2.0초|0.177717373|0.049365937|
|2.5/3.0초|0.358620855|0.039846762|

마지막 두 포인트만 20% 줄이는 가정은 **0.125715476**, 마지막 네 포인트를 각각 15% 줄이는 가정은 **0.120302923**다. 앞 두 포인트를 정확히 유지한다면 0.120000까지는 마지막 네 포인트 기여를 약 **15.34%** 줄여야 한다. 산술적 목표 배분이며 성능 예측이나 구간 독립성 가정의 검증이 아니다.

현재 모델의 초기 장점을 지키면서 중후반을 개선하자는 목적에 동의한다. 단 새 loss나 shared backbone 학습이 앞 두 포인트를 실제로 고정하는 것은 아니므로 전체 PREFIX를 주 평가로 유지한다.

## 3. 구간 오차는 재현됐다. 방향 하나로 원인을 좁히지는 않는다

아래는 `||delta(pred)-delta(GT)||`를 두 구간씩 평균한 값이다. 공식 PREFIX의 가산 분해가 아니다.

|모델/범위|0~1초|1~2초|2~3초|
|---|---:|---:|---:|
|DEV DIRECT|0.063260|0.082644|0.141074|
|DEV PROGRESS|0.060327|0.086601|0.161083|
|FULL PROGRESS, **in-fit**|0.040603|0.068354|0.126932|

제안의 DEV 표는 정확하다. 하지만 다음 두 사실을 추가해야 한다.

1. **일반 주행만** 보면 초기 0~1초도 DIRECT 0.059676 → PROGRESS 0.061642로 악화했다. 전체 초기 구간의 개선에는 정지·출발 개선이 섞여 있다. 일반 주행의 초기 진행량까지 이미 해결됐다고 해석하지 않는다.
2. 마지막 2.5~3초 구간의 길이 MAE는 DIRECT **0.111209** → PROGRESS **0.131200m**로 악화했다. 반면 제안서와 같은 moving mask에서 단순 평균 angle MAE는 **1.7241° → 1.5192°**로 작아졌다. 이 angle 평균만으로 위치 횡오차를 대표할 수도 없다. 길이·속도·오차의 분포 및 이전 구간의 누적이 다르기 때문이다.

정확한 관계 `E²=(pred_length-GT_length)² + 2*pred_length*GT_length*(1-cos(delta_theta))`로 확인하면, 마지막 구간의 평균 제곱 오차에서 radial 항은 0.025928→0.034261, angular 항은 0.024581→0.033652로 **둘 다 증가**했다. 이는 구간 벡터 제곱오차의 항이며 공식 PREFIX 기여가 아니다.

따라서 타깃은 **중후반 구간 이동 벡터의 정밀도**다. 길이-only 보조 감독이 원인이라고 입증된 것은 아니며, 방향 감독을 처음 도입하는 것도 아니다.

## 4. VECTOR 교체는 타당하다. 두 가지 효과를 명시하자

제안대로 기존 `common + 0.25*LENGTH`를 `common + 0.25*VECTOR`로 교체한다. LENGTH 위에 VECTOR를 추가하지 않는다. Inference graph·provided status producer·A2 query-only 경계·PREFIX·occupancy/lane/state/history는 유지한다.

### 유지되는 장점

- GT 구간이 완전정지면 기존 길이 벌점과 동일하다.
- 방향이 같으면 기존 길이 차이와 동일하다.
- 길이는 같고 방향이 다르면 기존 LENGTH가 보지 못하던 구간 방향 차이를 직접 감독한다.
- GT heading의 별도 정의, 정지 구간 tangent fallback, 새 학습 label/cache가 필요 없다.

### 효과 A: 같은 0.25라도 같은 비중은 아니다

DEV PROGRESS 저장 출력에서 평균 LENGTH는 **0.075478**, VECTOR는 **0.102670m**, 약 **1.360배**다. FULL in-fit에서는 0.058512→0.078630m, 약 1.344배다. 이는 평가 출력의 loss 규모이며 train gradient 비율이 아니다.

계수 0.25를 유지하는 한 번의 실용 비교에는 동의한다. 다만 이것을 순수한 방향 신호만의 추가나 동일 gradient budget 비교라고 부르지 않는다. V0를 보고 계수를 여러 번 바꾸지 않는다.

PREFIX의 구간 변위 gradient norm 상한은 제안대로 `1, .6944, .3889, .25, .1111, .0556`이다. 균일 VECTOR 항의 계수는 구간당 `0.25/6=.041667`이어서 첫 구간 상한의 4.17%, 마지막의 75%에 해당한다. 실제 network gradient는 각 출력 Jacobian·길이·오차 방향에 따라 달라진다. 이 비율을 실측 gradient로 해석하지 않는다.

### 효과 B: 방향 오차가 남으면 길이를 줄이는 해도 생긴다

방향을 고정하고 길이만 최적화하면 VECTOR 거리의 최적 양의 길이는

`length* = max(0, GT_length * cos(delta_theta))`

다. GT 5m, 방향 오차 30°에서는 **4.33m**, 45°에서는 **3.54m**다. 방향과 길이를 함께 학습하면 두 오차를 모두 줄일 수 있지만, 방향 수정이 어려운 경우 길이 축소로 손실 일부를 줄일 수도 있다. 최종 PREFIX도 유사한 절충을 가질 수 있어 새 loss만의 고유한 실패라고 단정하지 않는다.

이 때문에 VECTOR를 기각할 필요는 없다. 대신 구간별 **signed length error**, 일반 주행의 PREFIX, 출발·정지, 회전에서의 짧아짐을 함께 기록한다. VECTOR loss만 내려갔다고 성공으로 보지 않는다.

### 참조 구현의 실제 검사

첨부 CPU 검사를 다시 실행했고 결과가 첨부 JSON과 모두 같았다. 추가로 실제 `motiondrive_v2_training.compute_loss` 및 uncertainty 설정에 연결해 검사했다.

- Common loss 각 항이 LENGTH/VECTOR 두 wrapper에서 동일.
- total 차이가 정확히 `0.25*(VECTOR-LENGTH)`에 대응.
- lambda0에서 원본 common loss와 bitwise 동일.
- LENGTH 또는 VECTOR wrapper 중첩은 모두 예외 발생.
- batch15 / complete10행 / microbatch1,2,8,15에서 full-effective-batch loss·gradient 일치. 최대 loss 차이4.77e-7, output gradient 차이5.83e-11 미만.
- 미완성 행의 NaN GT, 정지, 반전, 동일방향 검사 통과.

현재 `train_shared_dynamics.patched_runtime()`는 내부에서 LENGTH wrapper를 설치한다. 향후 실행에서는 이미 wrapping된 `trainer.compute_loss`에 VECTOR를 다시 씌우지 않고 원본 common 함수에 연결해야 한다. 이번 검사 코드는 실제 학습 launcher를 변경하지 않았다.

## 5. FINE-READ는 실제로 다른 축이고 비용도 작다

현재 pre-pool correlation map은 **[4,128,54,96]**, 기존 조회 memory는 **4×192** token이다. 제안의 새 24×32 경로는 **4×768=3,072** token을 제공한다.

- 기존 12×16 motion/state/history/coarse temporal read를 유지.
- 같은 correlation_fuse 출력을 한 번 받아 별도로 24×32 pool.
- Fine 위치 embedding과 네 시점의 시간 embedding을 유지. 12×16 token을 보간해서 24×32로 만드는 것이 아니다.
- 기존 coarse temporal read 이후의 6개 **구간 출력 query**가 fine memory를 추가 조회.
- 기존 길이·방향 출력과 cumsum 유지. 추가 DIRECT 모델·teacher·checkpoint 없음.
- 새 attention의 **output projection만 zero-init**. 초기 함수 보존 후 내부 Q/K/V와 fine memory로 gradient가 흐를 수 있어야 함.

이렇게 만든 **검토용 시제품**을 FULL 가중치와 raw train fixture 2개에서 검사했다. Production factory/strict export를 갖춘 학습 모델로 완료한 것은 아니다.

|검사|결과|
|---|---|
|초기 FP32/BF16 parity|두 fixture의 plan/scene/motion/state/history 최대차이 모두0|
|추가 backbone|0회|
|추가 파라미터|66,560|
|전체 FLOPs|730.044861G → **730.256544G** (+0.211683G)|
|동일 세션 baseline median|25.768 / 25.784ms|
|동일 세션 FINE median|**26.161 / 26.144ms**|
|FINE 최대 p95|26.246ms|
|Gradient 합성 검사|첫 backward output projection 학습 가능; projection을 열면 Q/K/V와 fine memory로 gradient 발생|

B1 BF16 + FP32 planner, fixture당 warmup30/repeat100, baseline/fine 실행 순서를 교차했다. 전체 backbone 포함, 전처리 제외다. 이 범위에서는 약 **0.36~0.39ms** 추가 비용이다. 정확도는 측정하지 않았다.

6개 query의 cross attention이므로 3,072-token 전체 self attention과 비용이 다르다. 계산 여유를 이유로 처음부터 모든 54×96 token, MH8, 새 residual까지 함께 추가하지 않는다.

Fine K/V는 조건화 전 시각 correlation map과 기존 위치·frame-time embedding에서 만든다. Raw status/pose/goal을 새로 넣지 않는다. Query는 기존 scene 영향을 받으므로 전체 fine 출력을 goal-independent라고 부르지 않는다. 기존 A2 입력 경계를 유지하는 설계이며 운영국 개별 승인을 주장하지 않는다.

## 6. 이전 제안과 합친 실행 순서

두 제안의 착수 가능성과 낮은 변경 비용이 확인됐으므로 **VECTOR/FINE을 우선 비교**한다. 이전의 과거 scene768 공유를 버릴 필요는 없다. GPU0–3을 활용한다면 다음 4-arm 하나로 정리할 수 있다.

|슬롯 제안|동일 DEV parent에서의 단일 변경|Loss|
|---|---|---|
|GPU0 CONTROL|같은 추가 학습|기존 LENGTH0.25|
|GPU1 VECTOR|구간 보조 loss 교체|VECTOR0.25, LENGTH 없음|
|GPU2 FINE-READ|24×32 fine visual memory 추가 조회|기존 LENGTH0.25|
|GPU3 SHARED-HISTORY768|기존768 과거 FPN을 공통 scene에도 공유|기존 LENGTH0.25|

모든 parent는 **DEV H4-PROGRESS step20,554**로 고정하고 FULL 가중치를 반입하지 않는다. 같은 data/seed/sample order/augmentation/optimizer 시작/LR/추가 update로 맞춘다. CONTROL은 세 변경이 단순 추가 학습보다 나은지 비교하는 데 공동 사용한다.

짧은 1차 stage-2 비교의 제안값은 이전 계획과 같은 **3,426 update, batch16, backbone1e-6/나머지1e-5, fresh AdamW, warmup100, cosine 종료**다. 평가0/1,142/2,284/3,426, 주 판정은 terminal과 parent 대비다. 이는 검증된 최적 레시피가 아니며 이번에 실행하거나 예약하지 않았다. 새 FINE 분기는 같은 head LR로 시작하는 한 설정이고, 짧은 stage-2 정체를 모든 fine memory 설계의 실패로 확대하지 않는다.

주 평가: 전체 plain V0 PREFIX. 함께 볼 값은 첫 두/중간 두/마지막 두 포인트의 기여, nonstop/depart/steady, 구간 벡터·길이 MAE 및 signed 길이, 동일 GT tangent mask의 종·횡 오차다. 이전 인지·state/history 지표도 유지한다.

처음부터 VECTOR+FINE+SHARED를 결합하지 않는다. 서로 보완적인 실제 DEV 이득이 생겼을 때 결합을 검토한다. 원본1152, 길이/방향 decoder 분리는 이 짧은 비교 결과 뒤로 둔다. 무제한 loss/head/해상도 sweep은 시작하지 않는다.

## 7. 판단의 한계와 산출물

0.12로 가는 확정 해법을 찾은 것은 아니다. **현재 병목에 대응하고, 서로 다른 가설을 적은 변경으로 시험할 수 있는 실행 후보**를 확인한 것이다. VEC는 supervision 배분, FINE은 motion 공간 정보, SHARED는 scene에 전달되는 과거 정보라는 차이가 있다.

`INDEPENDENT_INTERVALS.json`, `VECTOR_INTEGRATION.json`, `VECTOR_REFERENCE_REPLAY.json`, `FINE_READ_PAIRED_REVIEW.json`, `ANALYTIC_CHECKS.json`, `PROVENANCE.json`을 함께 보존한다. ZIP 원문 참조 구현은 `experiments/a2_pro_review_20260921/reference/`에 두며 검토용 코드와 구분한다.
