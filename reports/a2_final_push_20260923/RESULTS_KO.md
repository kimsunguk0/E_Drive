# Final push 2026-09-23 — held-out 판정과 제출 계획

마감 2026-09-23 23:59 KST, 남은 공식 제출 **2회**(우리 계정 3회 사용: 0.197988 / 0.185969 / 0.133685).
모든 수치는 held-out V0(37 scene / 1,998행 / 11 session), paired session bootstrap 20,000회.
FULL 계열(V0 in-fit) 수치는 선택에 쓰지 않는다. `views/*.npz`에 행별 upright·mirrored 예측을 저장해
조합은 재추론 없이 `combine.py`로 채점한다(npz/예측/제출 JSON은 데이터 파생물이라 git 제외).

## 1. 단일 모델 upright / Flip TTA (fwd=2)

| tag | 모델 | upright | TTA |
|---|---|---:|---:|
| L28/L31/L34 | L-TRAIN3106 step 28605/31466/34326 (L-FULL6의 held-out 쌍둥이) | 0.142835 / 0.144164 / 0.143554 | 0.138617 / 0.139888 / **0.139214** |
| W34 | L-TRAIN3106-LENW step34326 | 0.142764 | **0.138376** |
| F17/F22 | F-CTRL step17163/22884 (train339) | 0.138406 / 0.138117 | 0.135125 / 0.134559 |
| G22 | F-FLOW step22884 (train339) | 0.138001 | 0.134544 |
| SWA_L28_34 / SWA_L28_31_34 / SWA_F17_22 | 같은 궤적 checkpoint 가중치 평균 | 0.142901 / 0.143153 / 0.137927 | 0.138725 / 0.138960 / 0.134681 |
| SOUP_L34_W34 | L34+W34 가중치 평균 | 0.143385 | 0.138971 |
| SOUP_F22_G22 | F-CTRL+F-FLOW 가중치 평균 | 0.137832 | **0.133805** |

## 2. 조합 판정

* **TTA**: L34 up→both −0.004340 CI[−0.006619,−0.002639] 9/11. 네 번째 독립 재현. 채택.
* **LENW+TTA**: W34:both vs L34:both −0.000839 CI[−0.001341,−0.000363] 9/11. TTA와 누적된다. 채택 후보.
* SWA(같은 궤적): 전부 0 또는 악화. 폐기.
* checkpoint 출력 앙상블(fwd=4): L28+L34 −0.000526(CI 0 포함), F17+F22 +0.000071. 폐기.
* **형제 soup**: SOUP_F22_G22 vs F22:both −0.000754 CI[−0.001479,−0.000206] 8/11. 품질이 같은
  형제(같은 부모·같은 sample order, aux만 다름)에서만 이득. SOUP_L34_W34는 W34보다 +0.000595(악화).
* 계보 간 출력 앙상블(W34 + F soup, fwd=4): F soup 단독보다 +0.001484 악화. 데이터가 많은 모델이 지배. 폐기.
* **사후 보정**(`calib.py`, leave-one-session-out): 횡방향 y×1.02~1.03, x는 1.00. L34 TTA −0.000488 8/11,
  W34 −0.000474, F22 −0.000399, soup −0.000224. 모든 fold에서 y 배율 1.0175~1.0325로 안정적이지만 이득이 작다.
  TTA 평균이 횡방향을 약간 수축시키는 것을 되돌리는 효과. 2번 슬롯에서만 선택적으로 쓴다.

## 3. 제출 슬롯

* **슬롯 1 (빌드 완료, 사용자 업로드 대기)**: L-FULL6-s1 step38070 + Flip TTA.
  `sub_L-FULL6-TTA/package/submission.zip`, sha256 `eedd7560145030e94617070f35f6bdf412e6338d2c36d1aad5808036c9791313`,
  1,125 clip + `__flops__`=1,460,089,722,240(2 forward, cutoff 7,053G의 1/4.83). RTX4090 1 forward 26.1ms라 2 forward도
  100ms 이하(시간 벌점 ×1.0). held-out 쌍둥이 기준 기대: 부모 대비 연속학습 −0.0076, TTA −0.0043.
  0.133685의 held-out→server 비율 0.884를 적용하면 대략 0.123(추정일 뿐).
* **슬롯 2 (학습 중)**: L-FULL6-LENW + TTA, 또는 seed soup이 쌍둥이에서 확인되면 soup(LENW s1, s2) + TTA.
  * GPU6 L-FULL6-LENW-s1 (38,070 update, ETA ~15:45)
  * GPU4 L-FULL6-LENW-s2 (seed 2, ETA ~16:00)
  * GPU3 L-TRAIN3106-LENW-s2 (쌍둥이, seed 2, ETA ~15:30) → SOUP(W34, W34s2)가 W34:both보다 좋으면 FULL soup 사용
  * 판정 규칙: 쌍둥이 soup이 W34:both 대비 CI가 0을 넘지 않고 음수면 soup, 아니면 LENW-s1 단독.
    y×1.03 보정은 슬롯 1 서버 결과를 보고 결정한다.
