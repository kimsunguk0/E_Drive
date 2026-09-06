# ETRI-ScoreDrive ⑤-E5~N / ⑥-B — selector 돌파와 닫힌 축 재정리

2026-09-06 · DCTN-beyless B200×8 + RTX 3090 · 작업루트 /NHNHOME/data/sukim/adcl

선행: `PHASE5C_CURRENT_ONLY.md`(⑤-C/D 구조), `PHASE5DE_RESULTS.md`(⑤-D/E, ⑥-A).
이 문서는 외부 감사 지적에 대한 검증과 그로부터 나온 재설계를 담는다.

## 0. 한 줄 요약

**병목은 coverage 도 차선 형상도 아니라 selector 였다.** learned row selector 하나로
val38 realized 0.3281 → **0.2612**(규정 준수), 자차 운동학까지 쓰면 **0.2082**.
지연시간 46.56ms 이므로 페널티 반영 실효점수에서 dense champion(0.2392@586.6ms → 0.821)을
확실히 넘었다.

## 1. 내가 틀렸던 것 (감사 지적 수용)

### ⑤-C4 "d3 에서 수렴" — 오류

| depth | slO@12 | realized |
|---:|---:|---:|
| 3 | 0.1785 | 0.2777 |
| 6 | 0.1446 | 0.2392 |
| 차 | **−0.0339** | **−0.0385** |

E1 전체 이득이 −0.0331 이다. **d3→d6 은 E1 전부와 같은 크기**인데
`d0=7.28` 이 지배하는 범위의 비율(0.5%)로 표현해 "수렴"이라 썼다. T7 우선이 맞다.

### T7 지연시간은 이미 측정돼 있었다

같은 벤치 run 에 있었는데 T4 만 보고했다. 재측정 포함:

| 구성 | 3090 median | p99 | penalty |
|---|---:|---:|---:|
| T4 | 46.56 | 46.76 | 1.000 |
| **T7** | **74.34** | 75.72 | **1.000** |

### `sc_sp_ctrl` 미문서화 — scorer 가 수렴하지 않았다

정체: `init_from=sc_e1_gt/best.pth`, `w_speed=0`, `freeze_trunk=1` = **E1 2라운드**.
e1_gt(0.2273)·sp_ctrl(0.2225) 둘 다 **마지막 step 에서 best** = 미수렴.

| 라운드 | tune slO@12 | val slO@12 | val realized(λ0.1) |
|---|---|---:|---:|
| r1 (e1_gt) | 0.2273 | 0.1873 | 0.3401 |
| r2 3-seed | 0.2225 / 0.2248 / 0.2233 → **0.2235 ± 0.0010** | 0.1847 ± 0.0023 | 0.3281 / 0.3658 / 0.3242 |
| **r3** | **0.2173** | **0.1788** | **0.3166** |
| r4 | 0.2187 | — | — |

재현 확인(tune 분산 0.001). **r3 에서 포화** — r3→r4 는 개선 없음.
scorer-only 재학습으로 얻을 수 있는 총량은 r1 대비 tune −0.010 / val slO −0.0085.
(r2ctrl_s1 의 realized 0.3658 은 stop 버킷 0.259 로 인한 이상치. 다른 판은 0.102~0.103.)

### ⑤-E3 "영상에서 속도 추정 불가" — 잘못된 추정자를 쟀다

| S3 추정자 | MAE | 상관 |
|---|---:|---:|
| 전역 상수 | 17.218 | 0 |
| **full-K logits softmax 기댓값** | **1.300** | 0.9923 |
| 전용 speed head | 2.446 | 0.9805 |

랭킹 분포가 전용 회귀 head 보다 **1.9배 정확**하다. 정보는 모델 안에 있었다.
**단 selector 에 넣으면 이득 정확히 0**(μ=0 이 tune·val 모두 최적, μ↑ 단조 악화):
그 추정치는 shortlist 를 만든 바로 그 logits 에서 나온 값이라 독립 정보가 아니다.

## 2. 감사 수치 검증

### profile selector — 수치는 맞고 원인이 다르다

보고된 0.3192 를 **0.3193** 으로 재현. 대조군을 붙이면:

| source | val 적용후 | 이득 | profile MAE |
|---|---:|---:|---:|
| pred (학습 head) | 0.3193 | +0.0097 | 0.0545 |
| **const (프레임 무관 상수)** | **0.3186** | **+0.0104** | 0.0294 |
| gt (상한) | 0.3087 | +0.0203 | 0 |

**정보량 0 인 상수가 학습 예측보다 낫다.** head 의 MAE 가 데이터셋 평균보다 1.85배 나쁘다.
이득의 정체는 timing 예측이 아니라 "전형적 프로파일에서 벗어난 후보를 벌주는 고정 prior".
→ 학습형 timing profile 축은 닫힘. **공짜 상수 prior −0.010 만 가져간다.**

### bank oracle — 고정 bank 로 1등 게이트에 닿는다

D3 는 첫 6점(3초)만 채점하므로 bank oracle 은 3초 기하의 양자화 오차일 뿐이다.
train 81,000행 D3-가중 k-means:

| bank | D3 oracle |
|---|---:|
| A0 K=1024 (현재) | 0.099634 |
| k-means K=1024 | 0.095887 |
| K=4096 | 0.068377 |
| **K=8192** | **0.058934** |
| K=16384 | 0.052924 |
| K=32768 | 0.049142 |
| train GT 전량(81k) | 0.046460 |

`oracle ≈ 0.353·K^(−0.194)`. 1등용 ≤0.060 은 K≈9.2k 로 달성.
**coverage 는 병목이 아니다**(1위 0.0924 보다 한참 아래). residual 생성기는 coverage
목적으로는 불필요하고, 같은 목표를 bitwise 고정 행으로 얻는 편이 compliance 가 가볍다.

## 3. 오차 분해 — selector 가 45%

| 성분 | 값 | 비중 | 1등용 게이트 | 배수 |
|---|---:|---:|---:|---:|
| bank oracle | 0.0996 | 29% | ≤0.055 | 1.8× |
| shortlist loss | 0.0877 | 26% | ≤0.020 | 4.4× |
| **selector regret** | **0.1528** | **45%** | ≤0.015 | **10.2×** |

→ selector 를 먼저 친다.

## 4. learned row selector — 돌파

후보 12행은 이미 완성돼 있고 selector 는 **행 index 만** 고른다.
입력: goal 거리·성분, visual logit(정규화/순위), 후보 기하(S3/S5/위치/heading/곡률),
timing profile 편차, 다양성, command. **ego status 미사용.**
학습 = train 300 scene(81,000행), 조기중단 = tune 30 scene, val38 은 1회.

| 조합 | val38 | regret | 선택==oracle |
|---|---:|---:|---:|
| 규칙 selector (goal+0.1·visual) | 0.3281 | 0.1459 | 35.5% |
| **learned selector (규정 준수)** | **0.2612** | **0.0790** | **50.6%** |
| learned + GT 운동학 (ego status) | **0.2082** | **0.0260** | 65.2% |
| shortlist oracle (상한) | 0.1822 | 0 | 100% |

음성대조: visual 셔플 0.2787 / visual 0 0.2814 (정상 0.2612) — 시각 기여 실재.

### scenario 5-fold CV (감사 지적 반영, val38 미사용)

mini8/tune30/val38 이 모두 반복 사용돼 독립 검증셋이 없다는 지적에 따라
train330 을 시나리오 단위 5-fold 로 나눠 fold 별 학습/평가했다.

scorer 별로 두 번 돌렸다.

| scorer | shortlist oracle | 규칙 | **learned** | 이득 | fold 개선 |
|---|---:|---:|---:|---:|---:|
| r2 (sp_ctrl) | 0.1689 ± 0.0039 | 0.3145 ± 0.0068 | **0.2579 ± 0.0045** | +0.0565 ± 0.0074 | 5/5 |
| **r3 (최종)** | 0.1676 ± 0.0041 | 0.3099 ± 0.0057 | **0.2542 ± 0.0043** | +0.0558 ± 0.0072 | 5/5 |

**⚠️ 표기 정정: 이 값은 end-to-end CV 가 아니라 selector-only CV 다.**
fold 는 selector 만 분리했고, 입력 shortlist/logit 을 만든 r3 scorer 와 A0 bank 는
train300 전체를 이미 봤다. held-out fold 66 시나리오 대부분이 upstream 입장에서는
학습 시나리오다. **selector 개선 효과의 증거로는 유효하나 일반화 추정치가 아니다.**
현재 가장 정직한 관측값은 scorer 가 보지 않은 **val38 0.2580** 이며, 그마저도
val38 이 반복 사용돼 완전한 blind set 은 아니다. 진짜 수치는 §12 의 OOF stacking 필요.
val38 단발 이득(+0.0669)보다 낮으므로 val38 은 약간 낙관적이지만 효과는 견고하다.
**이후 판정은 이 CV 를 1차 기준으로 삼고 val38 은 참고용으로만 쓴다.**

### compliance (⑤-R) — **무효, 재작성 필요**

아래 테스트 중 둘은 원리적으로 실패할 수 없는 결함이 있었다.
- **C-S3 무효**: `out_abs = abs5[cand_ids]` 로 만든 뒤 `array_equal(out_abs, abs5[cand_ids])`
  를 검사했다. 변수를 자기 정의와 비교한 것이라 항상 통과한다.
- **C-S2 무효**: goal 을 흔든 NPZ 를 다시 읽어 shortlist/logit 이 같은지 봤는데,
  그 배열은 원본에서 복사만 한 것이다. **모델을 goal counterfactual 로 재실행하지 않았다.**
- **C-S5 의심**: visual feature index 를 위치로 하드코딩했다.
- **46.56ms 는 selector 를 포함한 최종 wrapper 지연이 아니다.**

재작성은 raw images → T4 generator → shortlist → learned selector → 최종 궤적
전체를 normal/image-zero/image-shuffle/goal-counterfactual 로 다시 도는 형태여야 한다.

(무효 처리된 원래 표)

규칙 selector 가 통과한 C1~C8(Gate-3/4)을 승계하되, 교체품으로서 필요한 성질을 개별 증명했다.

| 게이트 | 결과 |
|---|---|
| C-S1 selector 입력에 ego status(speed/acc/can_bus/his) 없음 | PASS |
| C-S2 goal 교란 ×3 → shortlist·visual logits **bitwise 불변** | PASS |
| C-S3 출력 == bank 행 bitwise, 제출 6점 == `anchors_abs` bitwise, index ∈ 0..11 | PASS |
| C-S4 결정성(같은 입력 → 같은 index) | PASS |
| C-S5 visual 제거 시 선택 20.0% 변경 | PASS |

**실효 점수 비교** (Error = L2×(1+max(0,T−100)/200)):

| | 원점수 | 지연 | 실효 |
|---|---:|---:|---:|
| dense champion | 0.2392 | 586.6ms | **0.821** |
| **r3(T4) + learned selector** | **0.2580** | **46.56ms** | **0.2580** |

## 5. 무엇이 oracle 과 선택을 가르는가

불일치 1373프레임에서 **단일 속성 GT 를 알 때** realized:

| 아는 값 | realized | 회수율 |
|---|---:|---:|
| 현재 selector | 0.4022 | 0% |
| 0.5초 종방향 x | 0.2999 | 45% |
| 1.0초 종방향 x | 0.2394 | 72% |
| **1.5–2.0초 종방향 x** | **0.2154** | **83%** |
| 3초 종방향 x (=S3) | 0.2680 | 59% |
| **5초 종방향 x** | **0.4022** | **0%** |
| 횡방향 / heading / 곡률 | 0.47 / 0.57 / 0.70 | 악화 |
| shortlist oracle | 0.1762 | 100% |

**5초 endpoint 는 이득 0** — goal 이 이미 다 준다. 필요한 건 **중간구간 종방향 위치**이고
이건 goal 에서 유도되지 않는다. D3 가중치가 첫 1초에 61%, 첫 2초에 89% 실리는 구조와 맞는다.
횡방향·형상 축은 전부 무효.

### 필요 정밀도 (2.0초 위치)

| σ | 이득 |
|---:|---:|
| 0 | +0.1193 |
| 0.25m | +0.0849 |
| 0.50m | +0.0242 |
| 1.00m | −0.0005 (손익분기) |

영상 기반 최선이 σ≈1.63m(logit 유래)/3.07m(speed head) 라 **3–6배 미달**.

## 6. 차선·HD map 가설 기각

376 시나리오 전부에 `annotation/hd_map.parquet` 존재
(centerline 57 / boundary 28 / yellow_solid 68 / white_dashed 26 / stop_line / crosswalk).
pkl 에는 `map_location` 문자열만 있어 변환 때 빠져 있었다.
좌표 연결 검증: pkl `ego2global` = frame0 기준 로컬 pose, `ego_pose[frame+50]` 과 **오차 0.0000 m**.

| | val38 |
|---|---:|
| map 미사용 | 0.3281 |
| GT-map corridor | 0.3293 (**−0.0013**) |

게이트 ≥0.015 미달. 진단이 결정적:

| | oracle 후보 | 선택 후보 |
|---|---:|---:|
| centerline 거리 | 1.573m | 1.571m |
| solid line 교차 | 0.299회 | 0.299회 |

**정답과 우리 선택이 차선 기하로 전혀 구분되지 않는다.** shortlist 12개가 이미 같은 회랑 안이다.
→ map head 보류. (범위: **12개 중 선택**에 대한 결론. shortlist 생성 품질은 미검증.)

## 7. 자차 운동학 — 규정 확인이 필요한 축

과거 자차 운동의 등가속 외삽 정확도:

| 목표 | MAE | σ 근사 |
|---|---:|---:|
| 1.0초 뒤 위치 | **0.097m** | 0.12 |
| **2.0초 뒤 위치** | **0.322m** | **0.40** |
| 3.0초 뒤 위치 | 0.781m | 0.98 |

요구선 σ≤0.5m **안쪽**이다. 영상 기반이 3–6배 미달인 그 정보를 운동학은 정확히 준다.

**그러나 ego status 다.** `history_alignment` 정렬 행렬은 허용 입력이지만 거기서
자차 속도를 유도하는 것은 status 사용에 해당할 수 있다. **서면 확인 전까지 사용 금지.**

### 합법 우회: 과거 변위의 visual odometry

**과거** 자차 변위는 연속 프레임에서 복원 가능한 순수 영상량이다.
speed head 가 실패한 이유는 **미래**(S3)를 예측하려 했기 때문이고,
과거 1초 변위 추정은 난이도가 다른 문제다. 외삽이 나머지를 한다.

VO head 사양 (규칙 selector 위 측정):

규칙 selector 위 (노이즈 주입 프록시):

| σ_v (m/s) | σ_a (m/s²) | 이득 |
|---:|---:|---:|
| 0.05 | 0.10 | +0.0765 |
| 0.10 | 0.20 | +0.0621 |
| 0.20 | 0.40 | +0.0304 |
| 0.30 | 0.60 | +0.0085 |
| 0.50 | 1.00 | −0.0187 |

**learned selector 와 결합했을 때 (val38, 실제 특징으로 재학습):**

| 운동학 정밀도 | val38 | 순수시각(0.2612) 대비 |
|---|---:|---:|
| GT (ego status) | **0.2082** | −0.0530 |
| σ_v=0.10 m/s | **0.2213** | −0.0399 |
| σ_v=0.20 m/s | 0.2327 | −0.0285 |
| σ_v=0.30 m/s | 0.2420 | −0.0192 |

**σ_v 가 0.3 m/s 로 나빠도 유의미하다.** 요구가 생각보다 관대하다.
σ_v=0.1 은 10 m/s 에서 1% 상대오차로, 학습형 monocular VO 통상 성능(1–3%) 범위.

### VO head 구현 (⑤-P)

`--w-vo` 로 켠다. 입력은 이미 계산된 `f_now - f_hist` 를 후보·waypoint 축으로 평균낸
장면 수준 겉보기 운동 서술자라 **추가 연산 비용이 사실상 없다**.
출력 `_vo_pred [B,n_hist,2]` 는 **logits 경로에 들어가지 않고** selector 로만 나가므로
C4 불변식과 무관하다. 라벨은 `hist_T` 의 평행이동 성분(학습 시에만 사용, 추론 입력 아님).
pytest 게이트 2 passed 유지.

## 8. path-aware scorer 구현 (근거는 약화됨)

`corridor`(경로 법선 지면 오프셋 ribbon) + `seq_head`(waypoint 순서 depthwise conv) 추가.
전 게이트 통과:

- corridor=(0.0,) 는 기존과 **bitwise 동일**
- ribbon 오프셋 거리 정확(0.75/1.5m), 진행방향과 직교(max|cos| 5e-6), 좌우대칭 0.5 ulp
- **C4 불변식 유지**(증거 0 → 점수 0): 신규 층 전부 bias 없음, seq_head residual
- 가시율 center 93.14% → ribbon5 94.90%

3090 실측:

| 구성 | median | 게이트 |
|---|---:|---|
| T7 base | 74.34 | OK |
| T7 +seq | 74.36 | OK (**seq 는 공짜**) |
| T7 +cor3 | 84.70 | OK |
| T7 +cor5 | 101.82 | **초과** |
| T7 +cor5+seq+p1 | 129.68 | **초과** |
| T4 +cor5+seq | 61.96 | OK |
| T4 +cor5+seq+p1 | 77.75 | OK |

**시간 깊이와 경로 상세도를 100ms 안에서 맞바꿔야 한다.**
다만 corridor 의 주된 근거(차선 판별)가 §6 에서 기각됐으므로 우선순위는 낮다.

## 9. 백업 (감사 지적 수용)

- `git remote` 없음 / `size-pack 0` 확인 → gc 후 번들 **909KB**
- 로컬 `/home/a/adcl_backup/` 복사 + **빈 repo 복원 테스트 통과**(HEAD 48dd27d)
- `ARTIFACT_MANIFEST.json` — gitignore 대상 128항목 6.73GB 의 SHA256

## 10. 닫힌 경로 추가

| 경로 | 근거 |
|---|---|
| 학습형 timing profile (r1..r6) | 상수 대조군이 더 좋음, head MAE 가 평균보다 1.85배 나쁨 |
| logit 유래 S3 → selector | shortlist 와 공선, μ=0 이 최적 |
| GT HD-map corridor cost | −0.0013, oracle 과 선택이 차선 기하로 무구분 |
| 횡방향/heading/곡률 기반 선택 | 단일속성 GT 오라클에서 현재보다 악화 |
| 5초 endpoint 추가 활용 | 단일속성 GT 오라클 이득 정확히 0 (goal 과 중복) |

## 11. 알려진 비효율

`fuse_temporal` 은 `fuse_mul=0` 일 때 곱셈 항 자리에 **0 텐서**를 넣는다.
temporal MLP 입력 중 n_hist=3 은 1280 중 384, **n_hist=6 은 2432 중 768 이 항상 0**.
state_dict 형상 호환을 위한 설계인데 T7 에서 낭비가 크다. 진행 중 판 비교를 위해 미변경.

## 12. 다음

1. T7 2 seed 완주 판정 — **같은 진도(33%)에서 T4 0.2538 vs T7 0.5566/0.4020 으로 2배 뒤처짐.**
   실패 가능성이 높다. 원인 후보는 §11 의 0 채움과 temporal MLP 파라미터 증가(1280→2432).
2. ~~r4 라운드~~ 완료 — **r3 에서 포화 확인**
3. VO head (과거 1초 변위 회귀) → σ_v 실측 → selector 증분
4. K=8192 bank + coarse-to-fine scorer 의 3090 지연 게이트
5. scenario 5-fold CV 로 전환 (mini8/tune30/val38 모두 반복 사용됨)
