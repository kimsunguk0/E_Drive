# Phase A 결과 — ranking-loss 토너먼트

작성일: 2026-09-03 KST · 노드: DCTN B200 · `A=/NHNHOME/data/sukim/adcl`
기준 문서: `ETRI_SCOREDRIVE_DESIGN.md` §8 / §9.2 / §15(Phase A)

## 1. 무엇을 했나

기존 규정준수 K=1024 anchor 모델의 **top-1 순위 능력**이 병목(§2.2)이라는 진단을
검증하기 위해, `AnchorGroundedVocabularyDecoder.vocabulary_loss` 를 새 loss 모듈로
분리하고 4가지 목적함수를 **단일 변수 토너먼트**로 비교했다.

- **Stage-1 frozen trunk**: `epoch_4.pth` 로드 후 `freeze_except_goal_decoder=True`
  (backbone/neck/BEV 인코더 등 시각 trunk 동결, goal_decoder 스코어러 1.12M param 만 학습).
  → loss 효과를 순수 분리, ~1.6h/판.
- 4판 단일-GPU(0~3), **동일 seed 0 / sample order / update 수 / lr 2e-4 / batch 4**,
  2 epoch. 유일 변수 = `loss_mode`.
- 진단지표(top-1/oracle@M/gap/hit)는 모든 mode에서 동일 정의로 로깅 → 사과-대-사과.
- 평가: `scripts/eval_vocab_ranking.py` — 우리 val 38시나리오, **배포충실**(depth 6,
  시나리오별 reset), goal_decoder logits 를 forward hook 으로 캡처해 challenge 가중
  D3([11,11,5,5,2,2]/36) 로 순위 지표 산출. baseline top-1 가중 **0.5394 = 문서 0.5393**
  재현으로 하네스 검증 완료.

loss 항(§8.3): **L_pos**(positive-set 확률질량), **L_rank**(hard-negative pairwise),
**L_exp**(expected official L2). mode: `softce`(기존, 대조군) / `pos` / `pos_rank` /
`pos_rank_exp`.

> 문서 §8.5의 GPU3(velocity/stop auxiliary)은 새 head가 필요(Phase C 구조)라 Phase A
> "구조 고정"에 어긋나므로, GPU3을 **기존 soft-CE 대조군**으로 사용. velocity-aux는 이월.

## 2. 결과 (val, 배포충실 frame≥30, n=2052)

| 판 | top-1 L2 (가중) | top-1 (plain) | oracle@3 | oracle@6 | gap3 | top1hit | rank med |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline ep4 (기존) | 0.5394 | 0.4837 | 0.3065 | 0.2441 | 0.1772 | 8.5% | 11 |
| pa_softce (대조군) | 0.4186 | 0.4238 | 0.2888 | 0.2345 | 0.1350 | 9.2% | 10 |
| pa_pos | 0.4033 | 0.4243 | 0.2939 | 0.2379 | 0.1304 | 8.9% | 10 |
| pa_posrank | 0.4495 | 0.4363 | 0.2922 | 0.2363 | 0.1441 | 8.8% | 10 |
| **pa_posrankexp** 🏆 | **0.3865** | **0.4161** | 0.2901 | 0.2351 | **0.1260** | 8.4% | 10 |

전체 val(n=2280) 에서도 순위 동일: posrankexp top-1 가중 0.6244(최저), gap3 0.1433.

## 3. 판정

1. **승자 = `pos+rank+exp`.** top-1 가중 **0.5394 → 0.3865 (−28.3%)**, gap3
   **0.177 → 0.126 (−29%)**. 세 핵심지표 전부 1위. 2등(pos 0.4033) 대비 0.017,
   대조군 대비 0.032 앞서 — 채택 노이즈 임계(±0.005~0.007)보다 확실히 큼.
2. **문서 진단 정확.** oracle@3/6/20·coverK 가 4판 전부 사실상 동일
   (0.29 / 0.235 / 0.168, coverK 0.1213 고정) → **coverage는 불변, 오직 top-1 순위만
   개선**. 병목이 "커버리지가 아니라 상단 순위"라는 §2.2 진단대로다.
3. **L_exp 가 핵심.** pos+rank 만은 대조군보다도 나빴다(0.4495). 공식 가중 L2를 직접
   최적화하는 L_exp를 더해야 최고가 된다.
4. **이득의 상당부분은 "frozen 디코더 추가학습" 자체.** 대조군(옛 loss)도
   0.539→0.419. 새 loss의 **순수 기여분은 0.419→0.387 (−0.032, ~7.6% 상대)**.
   작지만 명확하고 gap3에서도 일관.
5. NanSkip 0×4, 크래시 없음, 0.62s/iter.

**Phase A 성공 기준(top-1↓ & gap↓) 충족.** val coverK 0.121 이 천장이므로
top-1 0.3865 는 아직 순위 개선 여지가 크다 → 다음은 Stage-2(승자 loss로 trunk 열어
E2E) 또는 Phase B(5초 extension + M=3 selector).

## 4. 자산 / 재현

- 코드: `dense_vocab_v1/projects/mmdet3d_plugin/VAD/anchor_vocab_decoder.py`
  (원본 `.orig` 백업). loss_mode + §8.3 항 구현.
- config: `dense_vocab_v1/projects/configs/VAD/VAD_etri_pa_{pos,posrank,posrankexp,softce}.py`
- 체크포인트: `$A/work_dirs/pa_<name>/epoch_{1,2}.pth`, run.log
- 평가 스크립트: `$A/scripts/eval_vocab_ranking.py` (baseline 0.5394 로 검증됨)
- 평가 결과 JSON: `$A/logs/rank_{baseline_ep4,pa_*}_*.json`
- 재현 학습: `bash /tmp/launch_pa.sh` (GPU0-3), 평가: `bash /tmp/eval_all_tournament.sh 2`

---

# Phase A r2 — rank/pos 성분 분리 + 재현성 + 가중 평가

작성일: 2026-09-03 KST. r1의 개선폭을 성분별로 나누고, 의심되는 L_rank를 분리하며,
평가를 대회-정합 **가중(official)** 기준으로 재산출했다.

## 평가 하네스 수정 (중요)
`eval_vocab_ranking.py` 를 OFFICIAL 블록 출력으로 수정: 모든 지표를 challenge D3
[11,11,5,5,2,2]/36(per-timestep) + val proxy weight(per-anchor)로 계산. 이 하네스로
baseline ep4 를 재평가하니 **문서 표와 완전 일치**(검증 완결):
top-1 W 0.5394 / oracle@3 0.2677 / @6 0.2071 / @20 0.1408 / coverK 0.0996.
→ 문서의 oracle 테이블은 **가중(proxy)** 기준이었다. r1에서 쓴 gap3 0.177 은 plain
진단값이고, **공식 가중 gap(top1−o3)=0.2717** 이 맞다. plain 지표는 진단용으로만 유지.

## r2 결과 (val, OFFICIAL weighted, frame≥30, n=2052)

| loss (seed) | top-1 W | oracle@3 W | gap top1−o3 | grad_norm |
|---|---:|---:|---:|---:|
| baseline ep4 (학습X) | 0.5394 | 0.2677 | 0.2717 | — |
| **softCE + L_exp** (s0) | **0.3761** | 0.2497 | 0.1264 | 2.6 |
| pos + 0.25·rank + exp (s0) | 0.3987 | 0.2520 | 0.1467 | 8.3 |
| pos + L_exp, rank=0 (s0) | 0.4588 | 0.2479 | 0.2109 | 8.0 |
| pos+rank(0.5)+exp **seed 1** | 0.5393 | 0.3571 | 0.1822 | 7.8 |

참고 r1 weighted top-1: softce 0.4186 / pos 0.4033 / posrank 0.4495 / **posrankexp(s0) 0.3865**.

## 판정 (r1 결론을 뒤집음)

1. **재현 실패.** posrankexp seed0=0.3865 → **seed1=0.5393**(≈baseline), oracle@3 도
   baseline보다 악화(0.357>0.268). r1 "승자"는 seed 운이 섞인 값이었다.
2. **seed 분산이 크다** (posrankexp 0.39~0.54, 스윙 ~0.15). 변형 간 차이(~0.02~0.08)는
   이 노이즈 안 → 단일 seed 순위는 신뢰 불가.
3. **L_exp = 유일하게 견고한 이득 성분.** softCE에 exp 추가 시 0.4186→0.3761 일관 개선.
4. **L_rank(0.5) 불안정** — grad_norm ~8로 분산 주입, seed 스윙 최대. 최종 loss 제외.
   rank=0.25는 완화(0.399)되나 rank 없는 편이 안정.
5. **L_pos ≤ softCE(base).** softCE+exp(0.3761) < pos+exp(0.4588). positive-set 미기여.
6. **잠정 최유력 = softCE + L_exp** (최고 top-1 W 0.3761, 최저 grad_norm 2.6 = 최안정).
   단 다중 seed 확정 전엔 잠정.

## 결론
- 확실: (a) 병목은 순위(coverK 0.0996 vs oracle@3 0.27), (b) L_exp가 그 순위를
  끌어올림, (c) L_rank·L_pos는 정당화 안 됨, (d) frozen-2ep 세팅 분산이 커 단일 seed
  주의.
- 다음(사용자 결정 대기): 3-seed로 softce_exp vs pos_exp 확정, 또는 lr 낮춰 분산 축소,
  또는 softce_exp로 확정하고 Stage2/Phase B.

## 자산 (r2)
- configs: `VAD_etri_pa2_{posexp,rank025,softceexp,posrankexp_s1}.py`
- ckpt: `$A/work_dirs/pa2_*/epoch_{1,2}.pth`
- eval JSON: `$A/logs/rank_pa2_*_epoch_2.json`, baseline `$A/logs/rank_baseline_ep4w_epoch_4.json`
- loss 코드: `anchor_vocab_decoder.py` (modes: softce/pos/pos_rank/pos_rank_exp/pos_exp/softce_exp)

---

# Phase A r3 — 재현성 감사 + seed1 발산 규명 (softCE+L_exp)

작성일: 2026-09-03 KST. r2의 seed 분산·seed1 이상을 감사. 계측 추가(entropy/maxprob/
loss 성분값), start-identity probe, softce_exp 3-seed + softce paired.

## start-identity 증명
`probe_start.py`: softceexp/softce config 모두 ep4 로드 시 **decoder_sha=386fa220…,
logits32_sha=295400ce…, load missing/unexpected=0** 로 동일 → 모든 run 이 bit-identical
ep4 에서 출발. seed 는 로드 이후 궤적만 바꾼다.

## seed1 발산 규명 (pa2_posrankexp_s1)
epoch_1 top-1 W **0.4118**(정상) → epoch_2 **0.5393**(붕괴, oracle@3 0.357). train sel 도
ep1 0.4395→ep2 0.4450 악화. 즉 로드/평가 버그 아닌 **epoch-2 발산**. rank(0.5)가 증폭.

## 재현성 결과 (val OFFICIAL weighted, frame≥30, top-1 W)
| run | ep1 | ep2 | best | Δ(ep2−ep1) |
|---|---:|---:|---:|---:|
| softce_exp s1 | 0.3908 | 0.5257 | 0.3908 | +0.135 |
| softce_exp s2 | 0.4471 | 0.4009 | 0.4009 | −0.046 |
| softce_exp s3 | 0.4053 | 0.4898 | 0.4053 | +0.085 |
| softce s1 | 0.4107 | 0.5897 | 0.4107 | +0.179 |
| (posrankexp s1) | 0.4118 | 0.5393 | 0.4118 | +0.128 |

## 판정
1. **epoch-2 과적합이 "seed 분산"의 실체.** lr 2e-4 × 2ep frozen 디코더는 과적합 →
   ep2 가 거의 항상 퇴화. 마지막-epoch 평가가 오해를 낳았다. checkpoint selection 필수.
2. **softce_exp 재현성 높음 (best-epoch): 0.3908/0.4009/0.4053 = mean 0.399 ± 0.006**
   (seed0 포함 0.393 ± 0.011). baseline 0.5394 대비 −26%. seed 운 아님.
3. **L_exp 순효과 확정 (seed1 paired, best-epoch): softce 0.4107 → softce_exp 0.3908 = −0.020.**
4. **oracle@3 ~0.267 (≈baseline 0.2677) 전 seed 불변** → L_exp 는 top-3 내 top-1 선택만
   개선, top-3 후보 품질 불변. 다음 병목 = candidate 압축·다양성(Phase B/C).

## 최종 loss 판정
**softCE + official weighted expected-L2 (softce_exp)** 로 확정. L_rank(발산), L_pos(무효)
제외. 학습은 ~1epoch/early-stop/lr 인하 + best-epoch 선택 필요.

## 자산 (r3)
- 계측 디코더(entropy/maxprob/성분값), `scripts/probe_start.py`
- ckpt `$A/work_dirs/pa3_*/epoch_{1,2}.pth`, eval `$A/logs/eval_pa3_*_ep{1,2}.log`
- seed1 ep1 eval `$A/logs/eval_pa2_posrankexp_s1_ep1.log`

---

# Phase B 예비 — shortlist diversity + visual-aware goal selection (무학습, 저장 logits)

작성일: 2026-09-03 KST. champion=pa2_softceexp/epoch_2. val frame≥30 n=2052.

## 기준점 통일
결정성 플래그(cudnn.deterministic + use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG)로
재평가: detA=detB=**0.3761** bit-identical. 이전 dump의 0.3516은 추론 nondeterminism 지터
(~0.025, 근접 logit tie flip)였다. canonical: champion top-1 W **0.3761**, baseline **0.5393**.
이후 표는 결정적 덤프 champ_det.npz 기준.

## shortlist oracle (weighted, min Dgt over shortlist)
| M | score | nms/diversity | bucket |
|---|---:|---:|---:|
| 3 | 0.2497 | 0.2395 | 0.2661 |
| 6 | 0.2003 | 0.1884 | 0.2470 |
| 12 | 0.1623 | 0.1475 | 0.2454 |
| 20 | 0.1397 | 0.1305 | 0.1305 |
- diversity/NMS는 oracle 소폭 개선(M=3 −4%, M=12 −9%). bucket-balanced는 악화.
- top-3 set-overlap(champ vs baseline) Jaccard=0.537 (절반가량 집합이 바뀜, 순서만 아님).

## goal-selection (endpoint-only, λ=0)
oracle는 M↑에 급개선(0.25→0.14)인데 goal-selection은 정체(0.294→0.253) → selector regret 이
M↑에 커짐(@20 oracle 0.14 vs 실제 0.253, regret 0.11). **selector 가 binding constraint.**
NMS shortlist는 goal-selection 을 악화(다양 후보 중 endpoint만 그럴듯한 false-positive).

## visual-aware selector: J = norm_goal + λ·norm_visual (2-fold 시나리오 CV)
| shortlist | CV L2 | λ* | λ=0 | visual-top1(λ=∞) |
|---|---:|---:|---:|---:|
| score12 | 0.2532 | 0.1/0.25 | 0.2538 | 0.3599 |
| **score6+nms6** | **0.2452** | 0.1/0.25 | 0.2465 | 0.3603 |
| score3+nms9 | 0.2476 | 0.1/0.25 | 0.2528 | 0.3657 |
| softnms12 | 0.3086 | 0.25/0.5 | 0.3733 | 0.3599 |

## 판정
1. visual-aware fusion 이 endpoint-only 를 이긴다(소폭). 최적 λ≈0.1(양 fold 안정).
2. 순수 soft-NMS 나쁨 → hybrid(score6 보존 + 잔여 NMS)가 최고. "상위 score 보존 + 잔여만
   diversity" 검증.
3. **regret 여전 ~0.09 (≥0.07)** → 다음 실질 레버 = **5초 tail/selector 구조 재설계**
   (crude train-median 5s-endpoint 가 한계). command 항 미적용(다음 단계).
- 즉시 배포 최선: score6+nms6 + fusion(λ=0.1) → CV val **0.2452** (기존 0.2538, 무학습).

## 자산
- `scripts/shortlist_experiment.py`, `scripts/selector_experiment.py`,
  `scripts/eval_vocab_ranking.py`(+결정성/--dump), 덤프 `$A/logs/dump/{champ_det,base}.npz`

---

# Phase B 확정 — tie-break 버그 수정 + 불변식 + selector 확정

작성일: 2026-09-03 KST. `scripts/phaseB_final.py` (오프라인, champ_det.npz+base.npz).

## tie-break 버그 (이전 "nondeterminism"의 정체)
champ_det 187/2052 앵커가 max logit 동점(BEV밖/zero-evidence). eval은 argsort(-logits)[0]
(불안정)로 top-1=0.3761, deployment 모델은 torch.argmax(첫 인덱스=stop우선)=0.3516.
같은 logits, 다른 tie-break. detA==detB bit-identical → forward는 이미 결정적(CUDA 아님).
**수정:** 모든 정렬 argsort(kind='stable')→동점 시 최소 인덱스(=argmax), λ=inf는 순수
visual(S[0]) 특수처리.

## 불변식 (PASS)
- 모든 shortlist S[0]==argmax : PASS
- λ=inf 선택==argmax==canonical 0.3516 : PASS

## 정정 기준점 (argmax = deployment)
baseline 0.5389 · visual champion top-1 0.3516 · 최종 selector 0.2451
(baseline 대비 −54.5%, champion 대비 −30.3%). 버그 수정 후에도 0.245 유지.

## selector 2-fold CV
| shortlist | CV L2 | λ* | λ=0 |
|---|---:|---:|---:|
| score3+nms9 | 0.2451 | 0.1/0.25 | 0.2492 |
| score6+nms6 | 0.2482 | 0.1/0.25 | 0.2493 |
| score12 | 0.2531 | 0.1/0.25 | 0.2542 |
visual fusion 순기여 소폭(−0.008), λ=0.1 권장. soft-NMS 폐기.

## hybrid oracle / regret / goal-rank (score6+nms6)
| M | oracle | realized(λ0.1) | regret | oracle후보 goal순위 median |
|---:|---:|---:|---:|---:|
| 3 | 0.2500 | 0.2843 | 0.034 | 1 |
| 6 | 0.2032 | 0.2534 | 0.050 | 2 |
| 12 | 0.1501 | 0.2434 | 0.093 | 2 |

**재해석:** oracle 후보의 goal-distance 순위 median=1~2 → 5초 endpoint 순서가 이미 꽤
informative. regret 은 "5s tail 붕괴"가 아니라 last-mile 선택(오라클이 goal 2위인 케이스를
1위만 집어 놓침 + 꼬리). 5s tail 전면 재설계 upside 제한적, selector(goal top-2~3 내 visual
선택)/꼬리 케이스 공략이 더 효율적일 수 있음.

## 남은 순서
step4 vad_cmd soft penalty, step5 oracle goal-rank 꼬리 분석(부분완료: median 1-2),
step6 5s tail endpoint/heading 정합, 이후 temporal velocity/stop head.
학습 champion 은 ~1ep/early-stop + mini-val checkpoint 선택으로 안정화.

---

# Phase B selector v2 + regret 분해 (`scripts/phaseB_selector2.py`)

작성일: 2026-09-03 KST. winner=score3+nms9. vad_cmd: 0=우(y<0)/1=좌(y>0)/2=직(1942/2280).

## B. selector 규칙 (M=12, CV) — linear 최고
| rule | full | CV |
|---|---:|---:|
| linear λ=0.1 | 0.2411 | 0.2398 |
| goal-top2→visual | 0.2472 | 0.2465 |
| goal-top3→visual | 0.2575 | 0.2569 |
| visual-top3→goal | 0.2839 | 0.2841 |
| rank α=1/0.5 | 0.282/0.258 | 0.282/0.257 |
| goal2+visual+cmd | 0.2510 | 0.2503 |
- 순위규칙이 linear 를 못 이김(연속 goal 거리를 hard cutoff 로 버림). command 필터는 악화.
- λ 는 fold별 재선택 말고 **0.1 고정**(CV 0.2398 < fold별 0.2451). 대표값 **~0.240**.
- **selector 확정 = score3+nms9 + linear λ0.1 고정. selector 튜닝 수확 종료.**

## C. regret 분해 (mean 0.10, endpoint-close 지표)
- endpoint 근접(<2m) 87.6% 인데 그 안 regret 0.086 → **끝점 맞고 첫 3초 timing/속도 틀림**.
- endpoint 오차 종(x) p90 1.50 >> 횡(y) p90 0.91 → 종방향 지배.
- 가속 regret 0.134(최고), 정지 8.5%프레임/21%regret, 좌회전 per-anchor 0.180.

## 결론 → Phase C
selector plateau ~0.240. 잔여 regret 은 순위 아닌 **종방향 velocity/timing**.
→ 다음 = **image-only temporal velocity/stop head** (같은 path geometry 의 정지/감속/진행을
영상으로 구별, goal 미사용). oracle 0.145 → 목표 realized 0.16~0.18. selector만으론 ≤0.20 불가.

## 제출코드 고정 사항 (tie-break)
top-1=argmax first-index; top-K=candidate ID 2차키 stable descending; synthetic
equal-logit unit test; offline dump vs deployment top-12 ID 완전일치 검사.

---

# Phase C — image-only velocity/stop residual (학습, 4판)

작성일: 2026-09-04 KST. champion base+trunk freeze, residual(100k)만 학습, zero-init.
`anchor_vocab_decoder.py`(residual/kinematics/L_timing/stop), `VAD.py`(train_only_goal_residual).

## realized selector L2 (score3+nms9 + λ0.1), champion=0.2411
| ckpt | top-1 | oracle@12 | realized |
|---|---:|---:|---:|
| champion | 0.3516 | 0.1446 | 0.2411 |
| pc_res_base ep2 | 0.3370 | 0.1459 | 0.2362 |
| pc_res_stop ep2 | 0.3382 | 0.1454 | 0.2365 |
| pc_res_timing ep2 | 0.3392 | 0.1460 | 0.2358 |
| pc_res_ts ep2 (best) | 0.3380 | 0.1451 | 0.2357 |

## best residual λ-sweep (pc_res_ts_ep2, score3+nms9)
λ0.1 0.2357 / **λ0.25 0.2345(best full)** / CV 0.2372. velocity가 visual 신뢰도↑ -> 최적 λ 0.1->0.25.

## 판정
- residual 작동: top-1 0.3516->0.338 (-3.5%). timing+stop 최고, ep2>ep1(과적합 없음).
- realized 이득 작음: champion 0.2411 -> **0.2345 (λ0.25) / CV 0.2372** = **-0.007 (-3%)**.
- 원인: (a) goal-지배 selector 가 개선된 visual 을 덜 씀, (b) mean-pooled 시간융합 BEV 의
  명시적 motion 신호 부족. 목표 <=0.20 미달.
- 다음: **explicit t=0 vs t=-0.5s history visual motion branch** (history frame forward 추가).
  융합 BEV pooled feature 만으론 velocity 신호 한계 (사용자 예측대로).

## 누적 체인
baseline 0.5389 -> visual champion 0.3516 -> selector 0.2411 -> +residual 0.2345 (baseline 대비 -56.5%).
