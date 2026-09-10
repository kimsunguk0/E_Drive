# 500-step 고정 snapshot: fine entropy·선택 진단

같은 tune 8행에서 V1024 fine loss≈5.3은 **높은 정답 entropy만으로 설명되지 않았다**. 모델의 확률은 거의 균등했지만, 정답 비용으로 정의한 분포는 그보다 더 집중돼 있었다. 차이의 대부분은 경로에 대해 합친 속도별 확률에 있었다. 다만 이 결과만으로 입력 정보 부족, 최적화, 모델 구조 중 어느 원인이 주도하는지 확정할 수 없다.

세 checkpoint의 embedded step은 모두 **500**이다. 완료된 기존 V256 실험의 immutable `last.pth`와 root가 원자적 publication 이후 복사해 SHA를 고정한 V1024 두 snapshot만 읽었다. 현재 실행 중인 `last.pth`는 읽지 않았다. GPU·held·reserve 평가는 하지 않았으며 live runtime 파일을 수정하지 않았다.

## fine loss의 분해

후보 비용을 c, 온도를 0.1로 두면 정답 분포 q=softmax(−c/0.1), 모델 분포 p=softmax(logits)이다. 실제 trainer와 대조해 `CE(q,p)=H(q)+KL(q||p)`의 계산 일치를 검증했다. 후보 200개의 균등 모델이면 CE=log(200)=5.298317이다.

| 고정 snapshot | CE | 정답 H(q) | KL(q‖p) | 모델 H(p) | 균등 CE 대비 |
|---|---:|---:|---:|---:|---:|
| 기존 V256, causal status | 5.1484 | 4.5999 | 0.5485 | 5.0484 | −0.1499 |
| V1024, causal status | 5.2785 | 4.7871 | 0.4914 | 5.2834 | −0.0199 |
| V1024, causal status+goal | 5.3323 | 4.7007 | 0.6316 | 5.2844 | +0.0340 |

V1024 두 모델의 exp(H(p))는 각각 평균 약 197.1, 197.3개로 200개 균등 분포에 가깝다. 정답의 exp(H(q))는 약 127.9, 115.3개이다. 따라서 `5.3≈정답의 불가피한 entropy`라고 단정할 수 없다. 반대로 각 모델이 다른 shortlist를 만들기 때문에 원시 CE나 KL만으로 전체 모델 우열을 판정하는 것도 적절하지 않다.

| 고정 snapshot | 경로별 확률 KL | 속도별 확률 KL | fine score 대 −cost Spearman | fine 선택 regret |
|---|---:|---:|---:|---:|
| V256 | 0.06482 | 0.47664 | 0.5063 | 0.25617m |
| V1024 status | 0.02421 | 0.46768 | 0.2327 | 0.17740m |
| V1024 status+goal | 0.01958 | 0.60898 | 0.0013 | 0.20420m |

V1024 status+goal의 속도별 확률 KL은 전체 joint KL의 약 96.4%이다. 모델의 속도별 entropy는 2.2916으로 log(10)=2.3026에 가깝지만, 정답 속도별 entropy는 1.7279이다. 이 8행에서는 fine velocity 선택을 따로 검사할 이유가 있다.

동일 shortlist에서 모델이 고른 경로를 고정하고 속도만 정답으로 다시 고를 때, V1024 status+goal의 oracle 대비 잔여 오차는 평균 0.02019m이다. 고른 속도를 고정하고 경로만 다시 고르면 0.19016m이다. 이는 서로 다른 조건부 oracle이며 가산적인 인과적 오차 분해가 아니다. 선택 궤적의 시간 가중 절대 XY 오차는 x 0.3764m, y 0.0772m로 종방향 오차도 더 컸다.

## 후보의 기하학적 유사성

V1024 status+goal에서 한 후보와 D3 거리 0.01m 이내인 후보 수는 자기 자신을 포함해 평균 2.50개, 0.025m 이내는 6.91개, 0.05m 이내는 12.00개였다. V1024 status도 각각 2.45, 6.62, 12.50개로 비슷하다. 가까운 경로가 여럿 존재하지만, 현재 모든 후보가 같은 궤적이어서 loss가 log(200)이 된다는 설명은 맞지 않는다.

이 진단은 τ=0.1로 만든 비용 기반 soft label에 대한 것이다. 저장한 확률 calibration bin도 모델 p와 soft target q를 비교하며, 실제 주행 성공 확률의 calibration으로 해석할 수 없다. 후보 비용·좌표·logits·p/q·IDs·각 coarse stage의 logits와 GT 비용을 모두 `candidate_dump.npz`에 저장해 다른 의사결정 규칙을 추가 forward 없이 검사할 수 있다.

## BF16 최종 점수 양자화 가설

CPU FP32의 후보와 feature 경로를 고정한 채 **최종 logits만 BF16으로 cast**했다. 실제 CUDA BF16의 upstream·coarse 변화까지 재현하는 실험은 아니다.

| snapshot | cast 뒤 최댓값 동점 후보 평균 | cast로 top1 변경 | cast D3 평균 변화 |
|---|---:|---:|---:|
| V256 | 2.000 | 3/8 | +0.00736m |
| V1024 status | 1.125 | 0/8 | 0 |
| V1024 status+goal | 1.375 | 1/8 | −0.00979m |

V1024 status의 top2 margin 중앙값은 0.01122, 최댓값 부근 BF16 ULP 중앙값은 0.0078125였다. status+goal은 각각 0.00577과 0.0078125였다. margin이 ULP보다 작은 행도 존재하지만 실제 cast가 잘못된 최종 선택을 광범위하게 만들지는 않았다. **이번 8행에서 BF16 최종 점수 동점이 주병목이라는 근거는 약하다.** FP32 최종 head 개입은 아직 구현·실행하지 않았다.

실제 저장된 CUDA BF16 평가와 동일 8행을 비교하면 다음과 같다.

| snapshot | CPU FP32 D3 | 저장 CUDA BF16 D3 | 선택 ID 일치 |
|---|---:|---:|---:|
| V256 | 0.38454 | 0.37320 | 5/8 |
| V1024 status | 0.28243 | 0.30380 | 7/8 |
| V1024 status+goal | 0.39572 | 0.38574 | 6/8 |

V1024 status는 한 행에서 D3 차이가 0.17091m였다. 따라서 CPU FP32 수치를 저장된 CUDA BF16 결과와 정확히 같은 추론 결과로 취급할 수 없다. 이 8행의 평균을 전체 tune 성능이나 두 학습 arm의 최종 우열로 일반화해서도 안 된다.

## 재현 및 provenance

- 도구: `experiments/sparsedrivev2_20260910/selection_diagnostics.py`.
- 코드 SHA: `bbf27216b4cbbe26b78c83a0fc6de325620863a30bb6ab96ade945664f2be576`.
- 동일 8행 row SHA: `6960e074988ed35e240780f2a0a835cc0ac9a5bfe04d0d950f920bd600ac180b`.
- 기존 V256 checkpoint SHA: `0e92346d2f3b063f11326a28305b5513fe1f348d5d3250484cd4969330cf2b2e`.
- V1024 status checkpoint SHA: `b7c14eff4e704803d112365c6912767a87f62fe69bff2ae42eb974ace0071fc7`.
- V1024 status+goal checkpoint SHA: `d8ff3b99f502dcc14f6ca5908670d8191c0d14db528fd114fc78d6ac6757bc8f`.

Checkpoint의 저장 source/data/model/losses를 우선 로딩하고, manifest SHA와 비교했다. checkpoint는 로딩 전후 파일 identity·크기·mtime·SHA를 확인했다. 자기 검증에서는 균등 분포의 CE/H/KL, p=q의 KL=0, factorized 분포의 marginal entropy를 검증했다. 실제 행에서는 trainer의 soft CE와 자체 계산이 2e-5 이내인지, 선택 출력이 기존 후보 행과 동일한지도 검사했다.

전체 산출물은 원격 `reports/sparsedrivev2_20260910/selection_diagnostics/`에 있다. 세 하위 디렉터리의 `report.json`과 `candidate_dump.npz`, `fine_precision_comparison.json`을 함께 읽으면 된다. 다음 개입 후보인 candidate-relative score head는 별도 prototype으로만 준비하며, 현재 학습 결과와 full-tune 진단 전에 원인이나 효과를 단정하지 않는다.
