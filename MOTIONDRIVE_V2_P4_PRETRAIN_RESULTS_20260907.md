# MotionDrive V2 P4 Fresh Pretrain Results — 2026-09-07

## 판정

공개 nuImages R50 기반 공통 I0에서 시작한 두 P0 auxiliary pretrain은 각각 정확히
2,000 updates를 수행하고 actual child/supervisor/SSH exit 0으로 종료했다. strict CPU
LAST load, optimizer tensor finite/step2000, fixed BN running state, 네 primary auxiliary
head의 동일 parameter 변화와 finite nonzero Adam `exp_avg`가 모두 확인됐다. 따라서
P0 learning/safety gate는 두 seed 모두 PASS다.

이는 planning 성능 판정이나 경쟁력 PASS가 아니다. P0는 G0S0이고 planning loss가
0이므로 기록된 D3를 모델 선택, P3/P2 비교 또는 경로 성능 주장에 사용하지 않는다.

## 계보와 고정 조건

- I0 producer commit: `1947cb33134dba9559c14f1e8c47301c6ef52c7e`
- P0 trainer commit: `806f0fa8c7447a7f66f59462977d87e00b492d0c`
- 공통 I0 file SHA: `06d2e68e15ca00d2c0f9ed3c965e198db603fe7e80007c39319d76eba9832e5a`
- 공통 I0 model tensor SHA: `7ac28a8f26dc796be70ddb78edff1e1c128c64f43b5982747ba419d5d5200683`
- I0는 ETRI optimizer update 0, optimizer state empty다.
- 두 실행 모두 train 54,810 / tune 1,998, C1 geometry, nominal time, G0S0,
  batch16/microbatch2/eval4, BF16, fixed BN, 12 GiB allocator cap/8 GiB reserve다.
- 두 seed는 독립 full initialization이 아니다. 같은 I0에서 data/optimizer RNG만
  seed 0/1로 나눈 파이프라인 반복이다.

## 실제 종료와 LAST

| seed | GPU | parent/child PID | actual rc | LAST SHA-256 | LAST model tensor SHA-256 |
|---|---:|---|---:|---|---|
| 0 | 4 | `2087099` / `2087334` | 0 / 0 | `8e91c30947a040f267f01f0642db3f2725900545deeb6d4087c795b5eb398a0c` | `f69e1c52ccd4b9908130ffe02170c0f08bff21712c634fee064517009a6c2c7d` |
| 1 | 5 | `2087109` / `2087332` | 0 / 0 | `6067ba9cc543e6e1be837d85883696bc7ad12c3da9b970edba3008e30eff83c5` | `d595f0788fbb5ae31ea2eaa2f3344ac300028756dcfe8f313ec1dbb761673e0d` |

종료 뒤 네 PID는 모두 부재했고 GPU4/5는 각각 used 0 MiB, free 182632 MiB로
복귀했다. 양쪽 minimum observed free는 176060 MiB였고 pressure/OOM/signal/nonfinite는
없었다. 두 LAST의 fixed BN state SHA는 동일하게
`15f7b2ea977aedbe202fc42669e687692d779de4fe501cf342a30454b86dfc99`다.

## Auxiliary tune 지표

`history`는 네 offset의 XY Euclidean MAE 평균이다. 아래 D3는 의도적으로 싣지 않는다.
전체 offset/state component와 매 250-step 기록은
`reports/p4_fresh_pretrain_metrics.json`에 보존한다.

| seed | step | history mean (m) | occ IoU | lane IoU |
|---|---:|---:|---:|---:|
| 0 | 0 | 4.891818 | 0.037140 | 0.207914 |
| 0 | 250 | 1.420498 | 0.338614 | 0.324879 |
| 0 | 750 | 0.574829 | 0.415213 | 0.451050 |
| 0 | 1500 | 0.499405 | 0.440505 | 0.486570 |
| 0 | 2000 LAST | 0.492844 | 0.435134 | 0.489078 |
| 1 | 0 | 4.891818 | 0.037140 | 0.207914 |
| 1 | 250 | 1.230307 | 0.365428 | 0.317635 |
| 1 | 750 | 0.571388 | 0.384270 | 0.444380 |
| 1 | 1500 | 0.530558 | 0.449647 | 0.489771 |
| 1 | 2000 LAST | 0.498015 | 0.432998 | 0.487612 |

LAST history mean의 두-seed 평균은 `0.4954296744`, range는 `0.0051706297`이다.
seed0은 step1750 `0.4880293470`에서 LAST로 `+0.0048150126` 반등했고 seed1은
`0.5087980599`에서 `0.4980149893`으로 계속 개선됐다. LAST가 primary이며 seed0의
중간 BEST만 골라 대표 결과로 사용하지 않는다.

LAST state `[vx, vy, ax, ay, yaw_rate]` MAE는 다음과 같다.

- seed0: `[1.044759, 0.038175, 0.289362, 0.091639, 0.010236]`
- seed1: `[1.059006, 0.037837, 0.287445, 0.091396, 0.010100]`

CPU label-only train-component-median reference와 비교하면 양 LAST의 history 4개와
state 5개 component가 모두 낮다. 다만 ax 차이는 작다. 이 reference는 이미지나 모델
forward가 없는 진단값이며 모델 baseline이나 planning baseline이 아니다.

## Auxiliary update 감사와 실패 보존

최종 CPU-only auditor는 producer/trainer commit을 구분하면서 실제 소비 I0의 file/tensor
SHA를 고정했다. 원격 23 tests가 통과했고 두 P0 audit은 actual rc0이었다.

- seed0 audit SHA: `1727b946c01dafa5f36b07b5d073242f85f0767884e796acffb60cec48dd3d93`
- seed1 audit SHA: `552f1f105a774cc942632f917a060675052f4fb7c8c1be2781a947305c5cf59c`
- occupancy, lane, history, state가 각 seed에서 모두 4/4 changed parameters와
  4/4 finite nonzero `exp_avg`를 가졌다.
- audit은 CPU-only, no CUDA, no model forward, no dataset/label read였고 입력 SHA는
  전후 동일했다.

앞선 두 audit helper 범위 오류는 삭제하거나 성공으로 정정하지 않는다.
`reports/p4_pretrain_aux_scope_failure.json`과
`reports/p4_pretrain_aux_lineage_failure.json`에 actual rc1을 별도 보존한다. 두 실패는
P0 모델 학습 실패가 아니다.

## 해석 경계와 다음 단계

P0는 auxiliary 학습이 실제로 일어나고 안전하게 종료된 증거다. P2 mature run과 같은
초기화 조건이 아니며 geometry 단독 causal ablation도 아니다. P3 C1의 `0.4379407`과
과거 clean C0T1의 `0.366175`는 후속 joint의 서로 다른 참고점일 뿐이고, 어느 것도
P0 planning 성능과 비교하지 않는다. 역대 최고, 1위 경쟁력 또는 독립 holdout 개선을
주장하지 않는다.

다음 승인 단계는 각 seed의 own LAST2000을 weights-only initializer로 쓰는 G1S1 joint
6,000 updates다. optimizer는 reset되고 warmup 200부터 시작한다. joint 결과는 두 seed의
own LAST6000을 primary로 평가하며 중간 BEST만 선택하지 않는다. final validation/test는
계속 격리한다.
