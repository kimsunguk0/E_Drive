# ETRI-ScoreDrive ⑤-D / ⑤-E / ⑥-A — temporal sparse 결과

2026-09-05 ~ 09-06 · DCTN-beyless B200×8 · 작업루트 /NHNHOME/data/sukim/adcl

⑤-C(current-only 실패)와 ⑤-D 구조·지연시간 기록은 `PHASE5C_CURRENT_ONLY.md` 참조.
이 문서는 그 이후의 학습·평가 결과를 담는다.

## 0. 한 줄 요약

temporal(과거 3장) 도입이 처음으로 실질 전진을 만들었고(realized 0.7226 → 0.3382),
E1(trunk 고정 + ranker 재학습)이 slO@12 0.1873로 1차 목표를 통과했다.
그러나 KD·속도·보조 supervision·trunk 개방·backbone 이식은 **전부 무효**로 확인됐다.
남은 병목은 속도가 아니라 **경로 형상 판별**이다.

## 1. 현재 위치

| 모델 | top1 | o@12 | slO@12 | realized |
|---|---:|---:|---:|---:|
| current-only | 0.9250 | 0.5001 | 0.4569 | 0.7226 |
| T4 temporal (⑤-D) | 0.5937 | 0.2528 | 0.2204 | 0.3923 |
| **+ E1 scorer 재학습** | 0.4943 | 0.2162 | **0.1873** | **0.3382** |
| dense champion (참고) | 0.3516 | 0.1615 | 0.1446 | 0.2392 |
| shortlist oracle 상한 | — | — | — | 0.1826 |
| bank full-K oracle | — | — | — | 0.0996 |

전부 val38 공식가중(D3 [11,11,5,5,2,2]/36 + proxy weight), frame≥30, n=2052.
평가 harness 는 dense champion 을 1e-4 이내로 재현한다(회귀 검증 통과).

## 2. ⑤-D — 4-frame temporal tournament

### tune (probe B slO@12)

| run | 구성 | best |
|---|---|---:|
| z_T3_gt_s0 | T3, GT | 0.2451 |
| z_T4_kd_s1 | T4, depth3 KD, seed1 | 0.2506 |
| z_T4_gt_s1 | T4, GT, seed1 | 0.2538 |
| z_T4_gt_s0 | T4, GT, seed0 | 0.2618 |
| z_T4_setkd_s0 | T4, set+KD | 0.2650 |
| z_T4_kd_a1 | T4, KD α=1.0 | 0.2717 |
| w_T2_gt_s0 | T2, GT | 0.2719 |
| z_T4_kd_s0 | T4, depth3 KD, seed0 | 0.3502 |
| d_T1_ctrl / z_T1_ctrl_s1 | current-only | 0.5891 / 0.6410 |

### 판정

- **temporal 성공.** current-only 대비 slO@12 0.4569 → 0.2204(2.1배), realized 0.7226 → 0.3923(1.8배).
- **T4 채택, T3 기각.** T3 의 단일-seed 우위(0.2451)는 seed 운이었다. 3-seed 로 T3 0.2611±0.031 vs T4 GT 0.2578±0.004 — 평균은 T4 우세이고 분산이 8배 작다. latency 46.56ms 로 여유 2.1배라 T3 의 10ms 절감을 좇을 이유가 없다.
- **KD 무효 (1차).** GT 2-seed 0.2578±0.004 vs KD 2-seed 0.3004±0.050. 이득 없고 분산만 12배. α=1.0, set+KD 도 GT 를 못 넘었다.
- **nuImages 사전학습 무효.** w_T3_nuim_s0 0.3588 vs 같은 조건 ImageNet 0.2611. 오히려 나쁘다. VAD backbone 이식 무효(⑤-C3)에 이은 두 번째 음성이며, 이번엔 temporal 체제라 앞선 교란(저 LR, current-only)이 없다. **nuScenes 다운로드 근거는 더 약해졌다.**

### 방법론 정정

val38 에서 최적 λ 를 골라 보고했다. tune 고정 λ 로 재계산한 차이는 s1 0.0000 / s0 0.0042 로 결론은 불변이나, **이후 λ·selector 규칙은 tune/CV 에서 고정한 뒤 val38 에 적용한다.**

seed 를 하나만 val 평가해 "게이트 통과"로 보고한 것도 오류였다. T4 GT 2-seed val slO@12 는 0.2204 / 0.2586 = 0.2395±0.019 로, s0 는 게이트 0.25 를 넘는다. tune↔val 순위 격차가 5배로 벌어지므로(tune 0.008 → val 0.038) **단일 seed 비교로 설정을 고르면 안 된다.**

## 3. ⑤-E1 — frozen-trunk ranker tournament

`z_T4_gt_s1/best.pth` 에서 시작, backbone+FPN 동결(scorer 230,273 파라미터만 학습), 동일 seed·데이터·스케줄, 3 epoch. 차이는 ranker 목적함수뿐. 로드는 missing=0 unexpected=0.

| 판 | 목적함수 | tune slO@12 |
|---|---|---:|
| **e1_gt** | GT loss (대조군) | **0.2273** |
| e1_setsl | set loss + teacher shortlist 집합 | 0.2301 |
| e1_top64 | teacher 상위64 listwise KL | 0.2306 |
| e1_kd025 | full-K KL α=0.25 | 0.2324 |
| (기준) | z_T4_gt_s1 base | 0.2538 |

**4판이 0.005 안에 몰렸고 대조군이 1위.** trunk 고정으로 교란이 제거된 조건이라 결론이 확정적이다. logit KD(⑤-C2), 정보량 맞춘 depth3 KD(⑤-D)에 이어 **KD 3차 확인 — 경로 종료.**

val38: top1 0.4943 / o@12 0.2162 / **slO@12 0.1873** / realized 0.3382(λ=0.2, tune·val 일치).

게이트: slO@12 ≤0.25 및 realized ≤0.35 **통과**, E1 1차 목표 ≤0.20 **통과**, dense 근접 ≤0.18 미달(0.007), 성공선 realized ≤0.30 미달.

**scorer 재학습만으로 slO −0.033, realized −0.054.** dense champion 까지 slO 0.043 남음.

버킷(λ0.1): stop 0.117 / accel 0.497 / left 0.685 / right 0.415 (champion 0.080 / 0.398 / 0.657 / 0.330).

## 4. ⑤-E2 — selector regret 분해

### λ 고정 (tune scenario 2-fold CV)

fold0 최적 λ=0.05(0.3983), fold1 λ=0.2(0.3828) — **불일치**. 평균 0.125 적용 시 val 0.3401 로, val 에서 고른 λ=0.2(0.3382)와 0.002 차이. **λ 는 평탄하고 튜닝 여지가 없다.**

### 핵심

| 항목 | 값 |
|---|---:|
| shortlist oracle | 0.1873 |
| realized | 0.3401 |
| selector regret | **0.1529** |
| 선택 == oracle | **38.5%** |

### shortlist 내 oracle 후보의 순위

| 기준 | 중앙값 | 1위 | top3 | 평균 |
|---|---:|---:|---:|---:|
| goal 거리 순위 | **2** | 35.5% | 71.4% | 2.88 |
| visual 순위 | **6** | 8.9% | 29.4% | 6.26 |

goal rank ↔ visual rank 상관 **−0.088**. **goal 순서는 이미 정확하고 visual 은 거의 무작위다.** 둘이 무상관이라 섞어도 이득이 없고, 이것이 λ 평탄의 원인이다.

### endpoint 오차와 버킷

선택 후보의 5초 endpoint: |err| p50 0.82 / p90 4.19m. 종방향 |x| p50 0.54 / p90 **3.24**, 횡방향 |y| p50 0.32 / p90 1.73.

**endpoint 가 GT 2m 이내인 프레임이 76.9%인데 그 안에서 regret 0.1471 — 전체 regret 의 77%.** 도착지는 맞는데 타이밍이 틀린다.

| 버킷 | n | 가중 | regret | 기여 |
|---|---:|---:|---:|---:|
| **accel** | 520 | 22.3% | **0.2486** | **36.3%** |
| stop | 147 | 25.1% | 0.1088 | 17.9% |
| right | 232 | 9.3% | 0.1694 | 10.3% |
| left | 71 | 2.6% | 0.2469 | 4.1% |

E2 목표 regret ≤0.10, 실측 0.1529 — 미달.

## 5. ⑤-E3 / E4 — 속도 축은 닫혔다

E2 는 "속도/타이밍이 병목"으로 읽혔다. 세 갈래로 검증했고 전부 닫혔다.

### E3 속도 회귀 head

설계 §4.5 권장 라벨(progress_3s/5s, 스텝별 arc length)로 학습. compliance 안전: 속도는 **출력**(입력 아님), 예측은 **모델 내부에서만** logits 보정에 사용, 외부 노출은 후보 12개+logits 그대로, goal counterfactual 자동 불변.

**예측 정확도는 우수**: MAE 2.446m / GT평균 30.41m = 8.0%, 상관 **0.9805**.
버킷별 MAE: stop 1.68 / accel 3.11 / left 4.42 / right 2.63 m.

**그런데 selector 에 넣으면 크게 악화한다.**

| | realized |
|---|---:|
| shortlist oracle | 0.1826 |
| 현재 selector | 0.3302 |
| goal + 2.0·**GT**진행량 | 0.2514 |
| goal + 1.0·**예측**진행량 | 0.5410 |
| goal + 2.0·**예측**진행량 | 0.6237 |

원인: shortlist 12개의 S3 가 **범위 4.15m, 인접 간격 0.32m** 로 촘촘하다. MAE 2.45m 는 간격의 8배.

요구 정밀도(GT+노이즈, μ=2.0): σ=0.5m → 0.3178(현재 수준), σ=1.0m → 0.3847(이미 손해), σ=2.45m → 0.5585. **σ ≤ 0.5m 필요.**

구현 함정: gamma 를 처음부터 켜면 무작위 head 의 난수 예측이 path_score([-2,2])를 덮어 slO@12 가 10.77 로 붕괴한다. **2단계 학습 필수** — (A) gamma=0 으로 head 만 학습, (B) 그 체크포인트에서 gamma 켜기.

### E4 factorized bank velocity 인덱싱

"선택이 아니라 bank 축 좁히기면 정밀도 요구가 낮을 것" 이라는 가설 — **틀렸다.**

| k | 후보수 | GT S3 | 예측 S3 |
|---:|---:|---:|---:|
| 4 | 2,048 | 0.1019 | 0.4888 |
| 8 | 4,096 | **0.0933** | 0.3353 |
| 16 | 8,192 | 0.0929 | 0.1801 |
| 32 | 16,384 | 0.0929 | **0.1058** |

기준: legacy K=1024 = 0.0996, factorized 전체 65,025 = 0.0929.

GT 면 k=8(16배 압축)로 전체 coverage 를 얻지만, **예측으로는 k=32 까지 늘려도 legacy 1024개보다 나쁘다.** velocity 축 128개의 3초 진행량 간격이 **0.46m** 이라 축을 좁히는 것도 결국 0.5m 정밀도를 요구한다(σ=0.5m → 0.0958, σ=1.0m → 0.1093).

### command 필터 (§3.4/§7.1 허용)

현재 selector 0.3382 → command 필터 0.3384. **효과 없음.** val 의 85%가 straight 라 필터가 거의 거르지 않는다.

### 결정적 발견 — goal 이 이미 속도를 해결하고 있다

| | \|S3 − GT\| 중앙값 |
|---|---:|
| shortlist 내 이론 최적 | 0.359m |
| **goal 로만 고른 후보** | **0.483m** |
| speed head 예측 | 1.771m |

**goal 매칭만으로 S3 를 0.48m 오차로 맞히고 있고 이론 최적과 0.12m 차이뿐이다.** speed head 는 3.7배 나쁘다. shortlist 내 S3 퍼짐도 goal 2m 이내로 좁히면 4.15m → 1.40m 로 3배 줄어든다.

속도 예측이 실패한 이유는 "없는 정보를 만들려 해서"가 아니라 **goal 이 이미 더 정확하게 주고 있는 정보를 더 나쁘게 재생산하려 했기 때문**이다.

**→ "속도가 병목" 이라는 E2 1차 해석은 수정한다. 속도 축에 남은 여지는 0.12m 뿐이다.**

## 6. ⑥-A — dense 최고성능 트랙: 실패

champion 은 trunk 를 얼린 채 goal_decoder 만 2 epoch 학습한 것이다. 설계 §9.3 Stage 2(open-trunk)는 한 번도 실행되지 않았고, `aux_loss_scale=0.0` 으로 det/map/traj 보조 loss 가 전부 꺼져 있었다.

`aux_loss_scale` 은 Hungarian matching 과 loss 계산이 끝난 **뒤** 반환값에 0 을 곱하는 구조다(matcher 를 건드리지 않으려는 의도된 설계). loss_single 은 정상적으로 cls≈32 를 반환한다.

### val38 (depth-6 배포충실, n=2052)

| run | aux | top1_w | o@6 | o@20 |
|---|---:|---:|---:|---:|
| **champion (기준)** | — | **0.3516** | **0.2032** | **0.1408** |
| s2_open_lo (bb ×0.02) | 0.0 | 0.3651 | 0.1985 | 0.1348 |
| s2_aux002 | 0.02 | 0.3648 | 0.1942 | 0.1323 |
| s2_open (bb ×0.05) | 0.0 | 0.3692 | 0.1963 | 0.1346 |
| s2_aux005 | 0.05 | 0.3724 | 0.1947 | 0.1354 |
| s2_aux015 | 0.15 | 0.3802 | 0.1946 | 0.1353 |
| s2_open_hi (bb ×0.10) | 0.0 | 0.3849 | 0.1994 | 0.1353 |
| s2_aux030 | 0.30 | 0.3924 | 0.2007 | 0.1356 |

**7판 전부 champion 보다 나쁘고 두 축 모두 단조 악화한다** — trunk 를 세게 열수록(0.3651→0.3692→0.3849), aux 를 세게 줄수록(0.3648→0.3924) 나빠진다. grad_norm 도 aux 0.30 에서 44.9 까지 오른다.

**`aux_loss_scale=0.0` 은 원 설계자가 옳게 고른 값이었다.** champion 은 이미 국소 최적이고 랭킹 loss 로 시각 표현을 건드리면 일반화가 깨진다 — ⑤-C 에서 sparse 가 겪은 것과 같은 구조.

이는 이전에 "dense 는 보조 perception supervision 덕에 일반화된다"고 한 설명을 부정한다. **dense 도 그 supervision 을 쓴 적이 없고, 켜면 나빠진다.**

## 7. 닫힌 경로 목록 (재시도 금지)

| 경로 | 근거 |
|---|---|
| logit KD (full-K) | ⑤-C2 무효, ⑤-D 무효, ⑤-E1 무효 — 3차 확인 |
| teacher top-64 listwise KD | ⑤-E1 무효 |
| teacher shortlist set KD | ⑤-E1 무효 |
| 속도 예측 → selector | ⑤-E3, σ≤0.5m 요구 vs 실측 2.45m |
| 속도 예측 → bank velocity 인덱싱 | ⑤-E4, 동일 정밀도 벽 |
| command 필터 | 무효(0.3382→0.3384), Phase B 에서도 무효 |
| λ 튜닝 | 평탄, CV fold 불일치, 이득 −0.007 |
| dense trunk 개방 (§9.3 Stage 2) | ⑥-A 3판 전부 악화, 강도와 단조 |
| dense 보조 perception supervision | ⑥-A 4판 전부 악화, 강도와 단조 |
| nuImages / VAD backbone 이식 | ⑤-C3, ⑤-D 두 번 무효 |
| T3(과거 2장) | 3-seed 로 T4 우세, 분산 8배 |

## 8. 남은 축

1. **shortlist 품질** — slO@12 0.1873 vs dense 0.1446. E1 이 −0.033 을 냈으므로 여지가 확인된 축.
2. **경로 형상 선택** — 선택==oracle 이 38.5% 뿐이고, 속도가 아니라면 나머지는 같은 진행량·같은 목적지 후보 중 **형상**을 잘못 고르는 것. visual 이 여기서 무정보(oracle 의 visual 순위 중앙값 6위). 차선·경계 판별이 필요하며 SparseDrive 식 map instance 소비가 후보다.

## 9. 목표 도달 가능성

| 기준 | 값 |
|---|---:|
| 리더보드 1위 / 3위 | 0.1305 / 0.1424 |
| 이상적 GT 랭킹 + goal 선택 M=3 (설계 §2.4) | 0.1220 |
| bank full-K oracle (K=1024) | 0.0996 |
| factorized P512×V128 oracle | 0.0929 |

**0.12 는 시각 ranker 가 완벽할 때의 천장이고 0.09 는 현재 bank 바닥 아래다.** 즉 0.09~0.12 는 점진 개선의 목표가 아니라 bank 교체와 거의 완벽한 ranker 를 동시에 요구한다. 이 구조에서 현실적 목표는 0.15 전후로 본다.

## 10. 산출물

- 스크립트: `scripts/{e2_selector_analysis,e3_speed_oracle,e3_speed_realize,e4_bank_index}.py`
- 모델 옵션 추가: speed head(`--w-speed`, `--speed-gamma`), frozen trunk(`--freeze-trunk`), `--init-from`, KD 모드(`--kd-mode full|top64|shortlist`)
- dense config: `VAD_etri_s2_open{,_lo,_hi}.py`, `VAD_etri_s2_aux{002,005,015,030}.py`
- 체크포인트: `work_dirs/sc_{z_T4_gt_s0,z_T4_gt_s1,e1_gt,sp_a_w10,...}/best.pth`, `work_dirs/s2_*/epoch_3.pth`
- dump: `logs/dump_sparse_c/*.npz` (logits/D3gt/weight), `logs/dump/dense_s2_*.npz`
- 평가 harness 회귀 검증: dense champion 재현 top1 0.3516 / slO 0.1446 / λ0.1 0.2392 (1e-4 이내)
