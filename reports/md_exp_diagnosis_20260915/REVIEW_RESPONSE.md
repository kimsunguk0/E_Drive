# 2026-09-16 리뷰 대응

리뷰의 지적을 수용한 것, 계산으로 답한 것, 실행한 것을 구분해 적는다.

## 1. 수용하고 고친 해석

### 1.1 "관측 head는 planner의 병목이 아니다"는 너무 강했다

0.873 m/s는 **과거 pose fit 기반 현재 vx** 오차이고, 0.167 m/s는 **미래 첫 구간 chord
진행속도** 오차다. 목표도 관측 경로도 다르므로 같은 축의 비교가 아니다.
자료가 말하는 것은 **planner가 head의 수치 오차를 그대로 적분하지 않는다**는 것이지,
더 나은 motion 표현이 무가치하다는 것이 아니다. 문서를 그렇게 고쳤다.

유지하는 부분: **state head vx MAE 0.2를 planning 개발의 문턱으로 삼지 않는다.**
그 0.2는 선택기 구조의 요구였고 이 planner에 그대로 옮겨오지 않는다.

### 1.2 평탄한 구간 MAE는 최근 관측 강화의 반증이 아니다

리뷰의 반례가 맞다. 직선 주행에서 sample마다 +0.18 / −0.18 m/s의 **일정한** 속도 오차가
있으면 전역 signed bias는 0이고 구간 MAE는 전 구간 평탄하며 시점별 L2는 선형으로 자란다 —
지금 보고된 집계와 구별되지 않는다.

따라서 "구간 MAE가 평탄하다"에서 도출되는 것은 **"후반 전환만 지배한다는 증거가 약하다"**
까지이고, "최근 운동 정보가 충분하다"는 결론은 도출되지 않는다. 관측 강화 경로를 닫지 않는다.

### 1.3 auxiliary 한 번의 실패는 관측의 정보 한계를 입증하지 않는다

실패 시 **"이 loss 설정에서 개선이 확인되지 않았다"**로 닫는다.
"현 영상의 원리적 한계"라고 쓰지 않는다. 앞선 문서의 실패 조건 문구를 그 표현으로 고쳤다.

### 1.4 "체계적 과주행/저주행이 아니다"는 전역 평균에 한정된다

아래 §2가 이를 실제로 뒤집는 조건부 사례를 찾았다.

## 2. 리뷰 §5.1 계산 결과 — 조건부 편향이 실재한다

`delta_v[n,k] = 2 (ell_pred − ell_gt)`, `b[n] = mean_k delta_v`, `r = delta_v − b`.
항등식 `sum delta_v² = 6 sum b² + sum r²`의 잔차는 **정확히 0**이다.

| 그룹 | 행 | 공통성분 에너지 | mean\|b\| (m/s) | **signed mean b (m/s)** | rms r (m/s) |
|---|---:|---:|---:|---:|---:|
| 전체 | 1,998 | **67.4%** | 0.1469 | +0.0162 | 0.1442 |
| constant | 1,402 | 62.6% | 0.1222 | −0.0083 | 0.1238 |
| accel/decel | 449 | 61.3% | 0.1629 | −0.0062 | 0.1768 |
| **stop / depart** | 147 | **82.2%** | 0.3332 | **+0.3174** | 0.2019 |

**정지·출발 구간에는 전역 평균이 가리고 있던 조건부 편향이 있다.** signed 평균이
+0.317 m/s로, 그 그룹에서는 모델이 **체계적으로 더 나아간다**. 리뷰 §4.4의 지적이 맞고,
"체계적 과주행이 아니다"는 전체 평균에만 해당한다.

전체적으로는 오차 에너지의 **67.4%가 sample 내부 공통 성분**이다. 다만 signed delta_v의
6×6 상관행렬은 인접 구간 0.90에서 최대 lag 0.079로 **띠 구조로 감소**한다 —
완전한 상수 offset(모든 상관 1.0)이 아니라 **느리게 변하는 공통 성분**이다.

이는 오차 신호의 분해이고 causal attribution이 아니다. `b[n]`은 미래 GT가 필요한
진단값이며 모델 입력이나 추론 보정에 쓰지 않는다.

## 3. 리뷰 §6 표기 정정 — 전부 반영

| 지적 | 조치 |
|---|---|
| 2.8e-17과 1.107e-7 혼동 | `diagnosis.json`에 `internal_algebra_max_diff`(2.7755576e-17)와 `vs_stored_row_d3_max_diff`(1.1074879e-07)를 **의미와 함께 분리** 기록. RECOMMENDATION 도입부 문장도 고침 |
| `chord_abs_error_mean_m` 명명 | `mean_sum_abs_interval_chord_error_m`으로 개명(6구간 **합**의 평균 0.5419이지 구간 평균 0.0903이 아님). signed 열도 동일 |
| 상대오차를 MAPE로 오해 | `relative_error_definition` 열 추가: `mean(abs error) / mean(gt magnitude)`, per-row MAPE 아님, 정지·저속이 분모에 가려질 수 있음 |
| head_vs_plan의 정의 혼재 | `first_segment_mean_abs_speed_error_ms = 2·mean(|chord error|)`는 **스칼라 진행속도 크기**이며 state head의 x-velocity와 같은 축 비교가 아님을 명시 |
| 조건 분류 임계값·SHA 미보존 | 여섯 그룹의 **정확한 임계값과 우선순위**, producer SHA, source pred/GT SHA를 `diagnosis.json`에 기록 |

## 4. 리뷰 §5.2 — length-only auxiliary 실행

리뷰의 지적대로 **길이만** 넣는다. sample별 공통 오차 `d[n,k]=c[n]`이면 인접 차분은 0이므로,
차분 loss로는 §2가 찾은 67.4%의 공통 성분을 교정할 수 없다.

```
ell(p)[k] = ||p[k] − p[k−1]||,  p[−1] = ego origin
L_length  = Σ_n complete[n] · Σ_k |ell_pred − ell_gt| / (6 · C_full)
L_total   = L_existing + 0.25 · L_length
```

`C_full`은 D3 loss가 쓰는 것과 **같은** full-effective-batch complete 수
(`normalizers["plan_complete"]`)다. microbatch마다 같은 분모를 쓴다.

**계약검사 5종 전부 통과**(`length_auxiliary_contract.json`):

- 기하: 알려진 polyline의 구간 길이를 정확히 재현, 정지 → 0
- 항등·방향 불감성: pred=GT → 0. **y축 반사 경로는 길이가 같아 이 항으로 0인데 D3는 >1** —
  그래서 D3를 대체할 수 없고 더하기만 한다는 것을 실증
- λ=0이 원본 loss와 **bitwise 동일**
- microbatch 동치: dense B16 대 micro 1/2/8/16이 손실·각 항·gradient에서 일치
  (B15 홀수, 무효 행 포함)
- 전 행 무효 시 길이 항 정확히 0, total 유한

**실행**: `E1-EXP-LEN-s0`. 같은 R0 초기값, 같은 Tplus 310 scene / 83,700행
(row SHA `e20e4541…` 동일), 20,554 update, seed 0, 같은 LR·schedule·BN·flip·time 계약,
command OFF. **step 1과 50의 sample_order_sha256이 control과 bitwise 동일**하다 —
기존 E1-EXP가 정확한 control이다.

판정 시 함께 볼 것: V0 D3, 구간 길이 MAE, §2의 공통성분, 방향 오차,
constant/stop/depart 기여, state/history. **train 길이 오차만 좋아지고 V0가 그대로면
채택하지 않는다.** H는 열지 않는다. 추가 본 학습은 이 1회뿐이고 λ sweep을 하지 않는다.

## 5. 동의하되 이번에 하지 않은 것

- 24개 clip 육안 검토: 목록은 `diagnosis.json`에 있고 아직 보지 않았다.
- H6-NEAR / H6-LONG 동수 대조: length-only 결과를 본 뒤의 다음 단계다.
  correlation 전 pooling 정밀도 실험과 동시에 하지 않는다.
- 0.15는 이 진단에서 도출되는 보장 성능이 아니다. 방향과 정지·출발의 잔여 예산이 남아 있다.
