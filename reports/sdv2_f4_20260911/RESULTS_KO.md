# SDV2 F4: velocity coverage 보존 + 영상 조건 최종 선택 — 결과

작성일: 2026-09-11. 지시서 `SDV2_F4_work_order_20260911.md`에 대한 실행 결과.

## 결과 표

frozen B8k(제공 status 없음, checkpoint `09a6dab7…`) 위에서 새 head만 학습.
4,000 step, effective batch 128, seed 0, AdamW lr 1e-3, warmup 100 + cosine,
D3 soft-CE temperature 0.1. 네 팔의 head 구조·파라미터 수·초기화·배치 순서가
동일하다. 평가는 원 TUNE 1,998행.

| arm | 후보 | 시각 특징 | **D3** | oracle | regret |
|---|---|---|---:|---:|---:|
| `n64` | P20 × 행별 stage1 V64 | 있음 | 0.792945 | 0.578403 | 0.214542 |
| `n64_novisual` | 같음 | 없음 | **0.787714** | 0.578403 | 0.209311 |
| `w1024` | P20 × 전체 V1024 | 있음 | **1.448812** | 0.068105 | 1.380706 |
| `w1024_novisual` | 같음 | 없음 | **0.614326** | 0.068105 | 0.546221 |

참고: frozen B8k 자체 0.9~1.02, MotionDrive Q10+flip **0.258544**.

## 두 축의 분리

**coverage 확장은 실제로 이득이지만 작다.** 시각 특징 없는 두 팔에서
0.787714 → 0.614326, **−0.173**이다. 내역은 oracle이 0.578403 → 0.068105로
0.510 좋아지고 regret이 0.209311 → 0.546221로 0.337 나빠진 결과다. 후보를
넓히면 좋은 행이 들어오지만 고르기는 그만큼 어려워진다.

**추가 시각 특징은 가치가 없거나 해롭다.**

- N64에서 0.792945 대 0.787714. 차이 0.005로 없는 것과 같다.
- W1024에서 0.614326 → 1.448812. **+0.834로 크게 악화된다.**

후보가 20,480개가 되면 path token은 velocity 축에, temporal readout은 path
축에 상수로 펼쳐지므로 강한 저계수 구조가 생긴다. head가 그쪽에 적합하면서
후보별 기하를 덜 쓰게 된 것으로 보인다. 이는 관측이고, 원인은 가설이다.

M13의 shared latent(0.2754 대 0.2712)·candidate BEV(0.2751 대 0.2723),
C의 scene token(−0.012)에 이어 **네 번째 같은 방향의 결과**다.

## 판정

지시서 §7의 배분 결정선에 따른다.

> 약 0.25 이상이거나 oracle만 개선: 이번 frozen B + 이 head 조합을 주력으로
> 확대할 근거 부족. MotionDrive 기준점을 유지한다.

최선 팔이 **0.614326**으로 0.25의 2.5배다. **근거 부족이 확정된다.**
MotionDrive Q10+flip 0.258544를 기준점으로 유지한다.

이는 frozen B와 이 head 조합에 대한 판정이다. SDV2나 Q10 경로 전체가
불가능하다는 결론이 아니다.

## Phase 0 기록

**3.1 기준 재현.** B8k를 원본 source로 다시 불러 재현했다. 학습 시 기록된
eval D3 1.0159149, 이번 진단의 B8 D3 1.0115981. 차이 0.0043은 forward 경로가
같고 평가 배치 구성이 다른 데서 온다. parity 검사는 C 기준 예측에 고정돼 있어
B에서는 실패로 표시된다.

**3.2 coverage 재측정.** 같은 P20 고정, 속도 축만 교체.

| 후보 집합 | B (0.584 epoch) | **B8k (2.3 epoch)** |
|---|---:|---:|
| 최종 V10 | 0.884802 | 0.905633 |
| stage1 V64 | 0.606489 | 0.572821 |
| 전체 V1024 | 0.069498 | **0.068050** |

넓은 집합 oracle 0.068050 ≤ 0.10이므로 지시서의 투자 기준선을 통과했고,
그래서 head 실험까지 진행했다.

**3.3 비용·목적함수·적합 검사.**

- 속도 descriptor의 경로 독립성: 전체 arclength의 경로 간 상대 산포
  중앙값 0.0001. bank가 형상×속도로 분해된다는 전제가 성립한다.
- W1024 microbatch 8: step 13 ms, peak 770 MiB. 비용은 병목이 아니다.
- **dense와 4-way accumulation의 loss 차이 0.000e+00, gradient 최대 차이
  3.31e-08.** 지시서가 요구한 전체 후보 단일 softmax가 유지된다.
- 64행 고정 적합: 13.957971 → 0.195383 (step 700, 해당 집합 oracle 0.076401).
  구현·최적화 실패가 아니다.

## 입력 경계

- 새 head의 입력은 frozen B의 image-conditioned path token, 공간 8×16 bin을
  보존한 image-only temporal readout, 모든 V에 정의된 stage1 velocity logit,
  고정 bank 기하다.
- 제공 status는 그래프 어디에도 없다(`provided_status_in_graph: false`).
- 제공 goal은 terminal per-candidate score에만 들어간다.
- GT는 forward 이후 cost/target 계산에만 쓰이며 별도 배열에 보관한다.
- 출력은 고정 bank 행 그대로다. 회귀·보간·속도 스케일링이 없다.
- pinned source를 수정하지 않기 위해 중간 특징은 런타임 훅으로 받았다.
  path token은 `traj[i,j] = path[i] + velocity[j]`의 `traj[:,:,0]`이므로
  행별 상수만큼 이동한 값이며, 상대 점수에는 영향이 없다.

## 하지 않은 것

- B1 live RGB 평가와 누적 forward 시간. 판정선에 크게 미달해 진행하지 않았다.
- seed 복제, 그룹 제외 확인, goal perturbation·다른 장면 영상 교란 검사.
- TUNE은 반복 사용된 개발 집합이며 독립 검증이 아니다.
