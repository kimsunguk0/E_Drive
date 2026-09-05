# ETRI-ScoreDrive ⑤-C — current-only sparse trunk 학습 결과

2026-09-04 · DCTN-beyless B200×8 · 작업루트 /NHNHOME/data/sukim/adcl

## 1. 판정: FAIL

3초-only / 5초-aux × seed 0,1 네 판을 완주했다. val38 공식가중 realized(λ0.1)가 0.662–0.751로, KEEP 게이트 0.25와 부족판정선 0.27을 2.4–2.8배 초과한다. dense champion 0.2392 대비 2.8–3.1배다.

다만 이것은 "current-only 구조의 상한"이 아니다. 원인은 과적합이며 trunk를 일반화시키는 문제가 아직 해결되지 않았다. 2-frame과 factorized bank는 용량과 후보공간을 늘리는 방향이므로 이 상태에서 붙이면 판정이 오염된다.

## 2. val38 결과 (공식 D3 가중 + proxy weight, frame≥30, n=2052)

| 모델 | top1 | o@3 | o@6 | o@12 | slO@12 | λ0 | λ0.1 | λ0.25 | λ0.5 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense champion (기준) | 0.3516 | 0.2500 | 0.2032 | 0.1615 | 0.1446 | 0.2476 | **0.2392** | 0.2415 | 0.2585 | — |
| d3_s0 | 0.9367 | 0.7503 | 0.6414 | 0.5373 | 0.4814 | 0.7120 | **0.7065** | 0.7085 | 0.7508 | FAIL |
| d3_s1 | 1.1329 | 0.8312 | 0.6742 | 0.5744 | 0.5210 | 0.7174 | **0.7173** | 0.7246 | 0.7555 | FAIL |
| d3aux5_s0 | 1.0101 | 0.7276 | 0.5914 | 0.4860 | 0.4311 | 0.6810 | **0.6620** | 0.6697 | 0.7061 | FAIL |
| d3aux5_s1 | 0.9836 | 0.8053 | 0.7077 | 0.6073 | 0.5439 | 0.7564 | **0.7505** | 0.7623 | 0.8085 | FAIL |

tuneval(30 scene, plain weight, n=1620) λ0.1은 d3_s0 0.8510, d3_s1 0.8968, aux5_s0 0.8773, aux5_s1 0.9566이다.

핵심은 모델 oracle@12(0.486–0.607)가 champion의 top-1(0.3516)보다도 나쁘다는 점이다. bank는 그대로이므로(full-K oracle 0.0996) 문제는 전적으로 ranker에 있다.

## 3. 5초 tail auxiliary: 판정 불가

| | seed 0 | seed 1 | 평균 | 표준편차 |
|---|---:|---:|---:|---:|
| 3초 only | 0.7065 | 0.7173 | 0.7119 | 0.0054 |
| +5초 aux (β=0.1) | 0.6620 | 0.7505 | 0.7062 | 0.0443 |

seed마다 부호가 뒤집히고(−0.045 / +0.033), aux5의 seed 분산이 3초-only의 8배다. 평균차 −0.006은 분산에 완전히 묻힌다. 이득도 손해도 입증하지 못했고 분산만 키웠으므로 채택과 기각을 모두 보류한다. 설계 §8.3.E의 "3초를 0.003 이상 해치면 weight 하향" 기준으로는 명확한 해악은 없다. 채택하려면 최소 3 seed가 필요하다(§9.2).

## 4. 버킷별 realized (λ0.1, val38)

| run | stop (147) | accel (520) | left (71) | right (232) |
|---|---:|---:|---:|---:|
| d3_s0 | 0.486 | 0.875 | 0.815 | 0.675 |
| d3_s1 | 0.276 | 0.838 | 0.917 | 1.005 |
| d3aux5_s0 | 0.342 | 0.870 | 0.880 | 0.652 |
| d3aux5_s1 | 0.312 | 0.878 | 1.029 | 0.857 |

accel이 최악(0.84–0.88), stop이 최선(0.28–0.49)으로, Phase B에서 밝힌 종방향 병목과 같은 방향이다.

## 5. λ sweep — selector로 메울 격차가 아니다

d3_s0은 λ0에서 0.7120, λ0.1에서 0.7065로 이득이 −0.0055에 그친다. champion은 0.2476에서 0.2392로 −0.0084다. 시각 점수가 약하면 selector의 visual fusion 기여가 사라진다. λ 재보정은 답이 아니다.

## 6. 원인: 과적합

| 지표 | 추이 |
|---|---|
| 학습셋 expected-D3 | 1.275 → 0.752 → 0.254 → 0.172 → 0.161 → **0.099** |
| tuneval realized | 2.59 → 0.895 → **0.851(best)** → 0.96 → 1.25 → **1.39** |

학습셋 expected-D3가 0.099로 bank oracle 0.0996에 도달했다. 즉 학습 데이터를 완전히 암기했다. 반면 tuneval은 best 이후 단조 악화된다. best 지점은 d3_s0 step 2248, d3_s1 4496, aux5_s0 1405, aux5_s1 3091로 모두 epoch 0 안이며, 전체 5 epoch 중 0.4 epoch 부근이다.

유효 표본은 90,000 프레임이 아니라 300 시나리오다. 한 시나리오는 10Hz × 300프레임 = 30초 주행이라 프레임끼리 거의 독립이 아니다. 데이터를 18k에서 90k로 5배 늘린 것은 iteration당 개선만 줬고 시나리오 다양성은 그대로였다.

## 7. 음성 대조 (C4/C5 동형, tuneval)

| 조건 | top1 | o@12 | slO@12 | realized |
|---|---:|---:|---:|---:|
| 정적 빈도 사전분포 (이미지 무관) | 13.15 | 2.318 | 0.989 | 1.266 |
| 무작위 logits | 7.33 | 1.248 | 1.231 | 1.629 |
| 모델 normal | 1.216 | 0.691 | 0.607 | **0.842** |
| 모델 image-SHUFFLE | 2.195 | 1.571 | 1.465 | 1.734 |
| 모델 image-ZERO | 13.15 | — | 0.774 | 1.104 |

이미지를 섞으면 0.842에서 1.734로 무너지고 top-1이 77% 바뀐다. 시각 신호는 진짜로 작동한다. image-ZERO에서 top1이 13.15, 즉 정지 후보 index 0을 고르므로 C4를 준수한다. 모델이 사전분포만 학습했다는 가설은 기각됐다.

## 8. 학습 안정화 필수조건: logit 유계화 + bf16

원래 스켈레톤은 logit scale이 무계였다. path_score에 global_score를 더한 값이 그대로 logit이 되는데, 정규화되지 않은 내적의 평균이라 크기 제한이 없다. 실측 프레임간 std가 11–18로, 1024-way softmax가 완전히 포화된다. 그 결과 플래토에 빠지고 이어서 fp16 오버플로우로 NaN이 발생했다(softce 406 관측).

2×2 요인설계 결과(1690 iter, tuneval realized λ0.1):

| | fp16 | bf16 |
|---|---:|---:|
| logit 유계화 없음 | 1.0030 | 1.3925 (grad norm 1592, softce 13.8) |
| logit 유계화 있음 | 3.1564 (실패, grad norm 1.46 기울기소멸) | **0.8631 (단조 수렴, grad norm 9.63)** |

두 요인은 상호작용한다. 따로 켜면 둘 다 손해이고 같이 켜야 이득이다. 유계화는 코사인 유사도에 학습가능 온도를 곱하는 CLIP 방식이며 local 항은 tanh로 묶었다. 학습 파라미터는 1개만 늘어난다. fp16은 코사인화로 작아진 기울기에서 언더플로우가 나므로 bf16이 필수다.

## 9. 기각된 처방 (반복 금지)

**frozen BN(§6.2) + backbone stage1–3 freeze(§9.3)를 적용하면 학습이 실패한다.** §9.3의 LR 표는 이미 학습된 ranker를 Stage-2에서 fine-tuning하는 기준이며, ImageNet에서 새 trunk를 처음 학습하는 ⑤-C에는 적용되지 않는다. backbone이 실제로 주행 영상에 적응해야 한다.

**sampled/scene feature에 LayerNorm을 넣는 것은 정규화 위치가 틀렸다.** feature가 아니라 logit을 묶어야 한다.

두 옵션 모두 플래그로 남아 있으나 기본값은 off다. 기존 구조 게이트는 2 passed를 유지한다.

## 10. 확정 학습 설정

| 항목 | 값 |
|---|---|
| 데이터 | 학습 300 scene × 300 frame = 90,000장 / tuneval 30 scene 1,620장 |
| 스케줄 | 5 epoch × 5,625 = 28,125 iter, batch 16, warmup 125 후 cosine |
| LR | head 2e-4, FPN 2e-4, backbone 1e-4 (freeze 없음, BN 학습형) |
| 수치 | bf16 autocast, logit 유계화 on, grad clip 5.0 |
| loss | softCE(D3, τ=0.1) + 0.25·L_exp(D3), 5초판은 + 0.1·L_exp(D_tail) |
| loss 하이퍼 | Phase A 확정값 그대로 (τ_exp=0.1, neg_topk=32, gt_near=16, ε_abs=0.05) |
| 체크포인트 | 281 iter마다 tuneval realized(λ0.1) 평가 후 best 저장 |
| backbone | ImageNet ResNet-34 (180 tensor 주입, FPN/head는 random) |

## 11. 데이터 관련 확인 사실

비-main frame(frame%5≠0)의 GT가 유효함을 확인했다. nearest-anchor D3가 0.1018로 main frame의 0.1007과 사실상 같고, mask5는 전부 1.0, zero-fut은 0%이며 frame 270–300도 유효하다. 덕분에 학습 데이터를 18,000장에서 90,000장으로 늘릴 수 있었다.

카메라 캘리브레이션은 전 scene·전 frame에서 완전히 동일하다(차이 0.0). 따라서 lidar2img를 한 번만 만들면 된다.

waypoint 가시성은 t0 78.8%, t1 90.9%, t2 93.3%, t5 95.3%이고 공식 D3 가중 기준으로는 88.5%다. 근거리는 전방 카메라만 기여하며(t0·t1에서 front 84.9%, 나머지 5대 0.0%), 가시성 자체는 병목이 아니다.

## 12. split 정책과 그 대가

train330을 학습 300 scene과 tuneval 30 scene으로 결정적으로 분할했다. dev38은 최종보고 전용이고 mini-val8은 blind로 남긴다. 설계 §4.1은 mini-val8을 hyperparameter 선택용으로 지정하지만, 더 보수적인 지시를 따라 train330에서 별도 holdout을 뗐다.

대가는 두 가지다. 학습 데이터가 9% 줄었고, A0 bank가 tuneval 30 scene을 포함한 train330 전체로 만들어졌으므로 tuneval 절대값이 약간 낙관적이다. 판정은 val38 기준으로만 하면 된다.

## 13. 평가 harness 신뢰성

평가 harness가 dense champion을 정확히 재현한다. λ0.1에서 0.2392, λ0에서 0.2476, top1 0.3516, oracle@12 0.1615, shortlist oracle@12 0.1446으로 Phase B FINAL canonical과 일치한다. 따라서 위 sparse 숫자는 신뢰할 수 있다.

## 14. 다음 단계

과적합이 병목이므로 용량과 입력을 늘리는 방향보다 일반화가 먼저다.

1. **조기중단과 정규화 강화.** best가 0.4 epoch에 나오므로 스케줄을 1 epoch 이하로 줄이고 weight decay와 드롭아웃을 올린다. 가장 저렴하고 과적합에 직접 작용한다.
2. **광도 augmentation.** 색·노출·블러. 후보 기하와 무관하므로 compliance에 안전하다. 1번과 묶어 네 판을 재실행할 수 있다.
3. **설계 §9.5 distillation.** dense champion(top1 0.3516)을 teacher로 후보 랭킹 분포를 증류한다. 시나리오 부족을 라벨 품질로 보완하는 정공법이지만 구현 규모가 크다.
4. **train330 전량 사용.** tuneval 30을 학습으로 되돌리고 mini-val8을 선택용으로 쓴다. dev38 보존 원칙과의 절충 판단이 필요하다.

권장 순서는 1+2를 먼저 돌리고, 그래도 0.4 이상이면 3번으로 가는 것이다. 1+2로도 게이트를 넘지 못하면 "current-only 학습으로는 dense 사전학습 trunk를 따라갈 수 없다"가 근거 있는 결론이 되고 distillation이 정공법이 된다.

## 15. 미해결 사항

logit 유계화는 구조 변경이므로 채택 시 3090 latency를 재측정해야 한다. ⑤-B의 13.003ms(AMP FP16)는 유계화가 없는 형상 기준이다. 다만 정확도가 게이트를 넘기 전에는 측정할 필요가 없다.

배포 forward는 변경하지 않았다. 여전히 완성 후보 12개와 visual logits만 반환하고 full-K는 노출하지 않는다. 학습용 full-K logits는 별도 경로로만 뽑는다.

---

# ⑤-C2 / ⑤-C3 — 일반화 시도 전부 실패 (2026-09-05)

## 판정 요약

⑤-C 의 과적합을 겨냥해 loss 교체, distillation, backbone 이식, 보조 perception
supervision 을 총 16판 시도했다. **기준선을 유의미하게 넘은 것이 없다.**

| 라운드 | 최고 | probe B slO@12 | 비고 |
|---|---|---:|---|
| ⑤-C2 (loss/KD) | c2_kd | **0.5190** | 전체 최고 |
| ⑤-C2 | c2_ctrl | 0.5378 | 대조군 |
| ⑤-C3 (VAD 이식) | v_vad_kd | 0.6533 | 전부 악화 |
| ⑤-C3 (보조 supervision) | a_occ025 | 0.5253 | 편차 내 |

val38 공식가중 기준 최고는 c2_kd: top1 0.8872 / slO@12 0.3763 / realized 0.6310
(champion 0.3516 / 0.1446 / 0.2392). 게이트 전 항목 미달.

## probe A/B 가 확정한 것

학습 시나리오의 미학습 frame offset(A) 과 새 시나리오(B) 를 매 checkpoint 동시 측정.

| | A (학습 시나리오) | B (새 시나리오) |
|---|---:|---:|
| c2_ctrl 최종 | 0.1179 | 1.0244 |
| c2_set 최종 | 0.1377 | 0.7343 |
| 모든 판 범위 | 0.117 ~ 0.157 | 0.519 ~ 0.774 |

A 는 어떤 설정에서도 0.12 부근까지 내려가고 B 는 0.52 아래로 내려간 적이 없다.
**scenario-level 일반화 실패**가 확정이다. 단, A 의 frame%5==4 프레임은 학습된
%5==3 과 0.1초 차이라 이미지가 거의 같으므로 A 는 사실상 train fitting 이다.
"A 가 champion val 0.1446 보다 좋다" 는 주장은 성립하지 않는다.

## 개별 결론

- **set loss(GT-근접 16 집합)**: best 기준 무효(0.5447 vs 0.5378). 다만 후반 발산은
  뚜렷이 억제(최종 B 0.734 vs 1.024). 목표 불일치 교정은 과적합 속도를 늦추지만
  일반화를 만들지 못한다. accel 버킷은 오히려 악화(1.345).
- **logit KD(α=0.5, τ=1, frame>=30 마스킹)**: 이득 −0.019 로 4판 편차 0.034 안. 무효.
- **VAD backbone 이식**: v_vad 0.7258 vs v_in50 0.7004 — 같은 arch·같은 LR 에서
  **도메인 사전학습이 ImageNet 보다 나을 게 없었다.** 접근 자체가 부정됨.
  단 R50 판은 backbone lr 2e-5, R34 판은 1e-4 로 걸어 **arch 간 비교는 교란**됐다
  (설계자 오류). "R50 이 나쁘다" 는 성립하지 않는다.
- **frozen BN**: 세 번째로 학습 붕괴(v_vad_fbn A=0.774≈B=0.774). dense 자체 레시피지만
  이 구조에서는 재현되지 않는다.
- **occupancy 보조 supervision**: w=0.25 에서 0.5253(대조군 0.5378 대비 −0.013, 편차 내),
  w=1.0 에서 0.5887 로 악화. **더 중요한 것은 보조 과제 자체가 학습되지 않았다는 점**
  (occ BCE 0.82~0.97, 무작위 수준 약 1.09). 즉 의도한 supervision 이 주입되지 않았다.

## 후보 판별력(aliasing) 측정

sparse scorer 가 읽는 feature 격자에서 1024 후보가 실제로 구분되는지 측정했다.

| 샘플링 | 구분 가능 후보 | 후보당 셀(중앙값) |
|---|---:|---:|
| 현재 stride 8 | 710/1024 (69.3%) | 10 |
| 현재 stride 16 | 431/1024 (42.1%) | 7 |
| stride 4 추가 | 916/1024 (89.5%) | 14 |
| stride 8 + 4 offset (설계 §6.3) | 873/1024 (85.3%) | 23 |
| 높이 4단 | 747/1024 (72.9%) | 16 |

aliasing 은 실재하고, **설계 §6.3 의 "projection 주변 4 offset point" 를 구현하지
않은 것은 명백한 누락**이다(69%→85%). 그러나 aliasing 은 A 와 B 에 동일하게 작용하는데
A 는 0.118 로 멀쩡하므로 **A/B 격차의 원인일 수 없다.** 보조 과제 학습 실패의 원인일
가능성은 남아 있어 별도 검증 중.

## 투영 기하 검증 (육안 + 수치)

캐시 이미지에 GT 궤적과 에이전트 박스를 투영해 렌더링 확인:
전방 카메라의 GT waypoint 가 노면 위에 정상 배치되고 에이전트 박스도 차량에 대체로 맞는다.
`cam_intrinsic`(undistorted new_K) 이 맞고 `cam_intrinsic_raw` 는 우측으로 밀린다
(두 행렬 최대차 120.9). 캐시 생성기의 crop 규칙(`(w-1920)//2`, CROP_KEEP_TOP, scale 0.4)과
`build_cached_lidar2img` 가 일치. **기하 가설은 제거됨.**

## 아직 구분되지 않은 두 원인 (중요)

지금까지의 데이터는 다음을 구분하지 못한다.

- A. t0 이미지에 필요한 정보는 있으나 R34 가 표현하지 못한다
- B. 속도·정지·운전자 의도 정보가 t0 한 장에는 애초에 없다

특히 **KD teacher 는 7-frame temporal BEV 를 쓴다.** 학생이 teacher 를 못 따라간 것이
representation 부족인지, 관측 불가능한 정보를 distill 하려 한 것인지 미결이다.
teacher 분해도 temporal 중요성을 시사한다: 전체 slO 0.3279 / frame>=30 0.1832 /
offset0&frame>=30 0.1632. 초기 frame 이 나쁜 이유가 prev_bev warm-up 부족이라면
dense 일반화의 상당 부분이 temporal state 에서 나온다는 뜻이다.

→ **다음 작업은 학습이 아니라 dense teacher 의 controlled history-depth curve** 다.
val38 frame>=30 동일 프레임·체크포인트·selector 로 depth 0/1/2/3/6 을 재고
top-1/slO@12/realized 와 버킷을 비교한다.

- depth0 slO@12 <= 0.20~0.25 -> t0 에 정보 있음 = representation 문제. 다음은 VAD
  feature distillation, trunk freeze + adapter, lane/drivable pseudo supervision.
- depth0 >= 0.40 이고 depth2~6 에서 크게 개선 -> t0 에 정보 없음 = **current-only 가설 실패**.
  보조 perception 보다 temporal sparse 모델이 먼저다.
- depth1~2 에서 대부분 회복 -> 7-frame 불필요. 2~3 frame sparse 로 충분하고 3090
  100ms 안에 들어올 가능성이 높다.

## 현재 확정 사항

- K1024 sparse scorer 의 fitting capacity: 충분
- global branch 가 주범: 아님 (local ≈ both, global 단독은 무작위보다 나쁨)
- exact top-1 loss 만 바꿔서는 해결 안 됨
- frame 단위 split 은 의미 없음 (scenario 단위만 유의미)
- current-only 가 새 시나리오에 일반화: 실패
- factorized bank: 아직 금지
- 5초 auxiliary: 계속 보류
- bounded logits + bf16: 수치 안정성 기본값으로 유지
- 투영 기하: 정상
