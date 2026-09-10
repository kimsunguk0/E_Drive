# SparseDriveV2식 ETRI 고정 경로×속도 사전: 2026-09-10 실측

고정 경로 1,024개와 대회 3초 누적 진행량으로 학습한 속도 1,024개를 조합하면, tune 1,998행에서 **전체 사전 GT oracle D3=0.053674**이다. 같은 사전에서 대회 지표에 맞춘 정답 기반 경로·속도 상위 20×10개의 oracle도 **0.053675**로 거의 같다. 사전의 표현 범위는 0.15를 목표로 모델을 학습할 여유가 있다. **이 수치는 정답으로 후보를 고르는 진단값이며 실제 모델 성능이 아니다.** 현재 모델의 shortlist oracle과 최종 선택 오차를 이 결과와 함께 측정해야 한다.

추천 사전은 원격 worktree의 `cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz`이다. SHA256은 `4aff9b40696f91383bfbec377f388e6209d619fa509e51a2019b19af977b5e04`, 파일 크기는 250,799,510 bytes이다. 이미 모델 학습에 사용 중인 `p1024_v256_native100m_v8.npz`는 수정하지 않았다. 더 큰 속도 사전이 실제 분류를 더 어렵게 만들 가능성은 있으므로, 실제 선택 오차와 처리 시간을 함께 비교해 채택해야 한다.

## 같은 데이터에서의 결과

metric은 0.5, 1, 1.5, 2, 2.5, 3초의 유클리드 오차에 `[11,11,5,5,2,2]/36`을 곱한 평균이다. 이는 1초·2초·3초 prefix ADE의 동일 가중 평균과 같다. 아래 수치는 모두 같은 tune 1,998행이다.

| 공간 경로 | P | V | 속도 군집 기준 | 전체 사전 GT oracle | GT D3 coarse 20×10 oracle |
|---|---:|---:|---|---:|---:|
| 50m, 1m 간격 | 256 | 128 | 4초 absolute speed | 0.505095 | 전체행 평가 불가: 일부 shortlist가 모두 invalid |
| 100m, 2m 간격 | 256 | 128 | 4초 absolute speed | 0.149044 | 이 행의 D3 coarse 별도 평가 안 함 |
| 100m, 2m 간격 | 1024 | 256 | 4초 absolute speed | 0.103896 | 0.103896 |
| 위 P1024 경로 고정 | 1024 | 512 | 4초 absolute speed | 0.077919 | 0.077919 |
| 위 P1024 경로 고정 | 1024 | 512 | 3초 weighted progress | 0.069254 | 0.069254 |
| 위 P1024 경로 고정 | 1024 | 1024 | 4초 absolute speed | 0.059436 | 0.059437 |
| 위 P1024 경로 고정 | 1024 | 1024 | 3초 weighted progress | **0.053674** | **0.053675** |

P1024 이후 경로 배열은 bitwise 동일하며 SHA256은 `626e9ea479ca90783432185f479f7dbfb748f8edd321e7cc960ab212f4f1a03d`이다. 따라서 이 구간의 비교에서는 경로 재학습 효과를 제거했다. 모든 100m 사전은 첫 6개 점뿐 아니라 8개 점도 전체 조합에서 유효하다. 50m 사전에서 3초 이동거리 50m 이상인 159행의 oracle 평균은 4.617009로, 경로 길이를 늘려야 하는 문제가 실측으로 확인됐다.

최종 사전의 0.5–1초, 1.5–2초, 2.5–3초 두 점씩 묶은 평균 오차는 각각 **0.035606, 0.055920, 0.147435m**이다. P1024×V256에서는 0.066394, 0.114520, 0.283601m였다.

11개 tune 세션을 통째로 20,000번 복원추출한 bootstrap에서 최종 oracle의 95% 구간은 **[0.048368, 0.059026]**, P1024×V256 대비 paired 차이 -0.050222의 구간은 **[-0.052848, -0.047300]**이다. 동일 tune에서 후보 설계를 비교한 결과이며, 이 구간이 최종 평가 데이터나 학습된 모델의 0.15 달성을 보장하지 않는다.

## 선택 목표와 대회 지표의 일치

P1024×V256 사전에서 공개 구현식 진단 비용은 `native spatial path masked XY MSE`와 `8구간 속도 MAE`이다. 이 비용으로 각각 한 개씩 정답 기반 선택하면 D3가 0.187180이다. 실제 trainer의 `losses.coarse_costs`는 경로를 GT 6점 누적 진행량에서 보간한 D3, 속도를 GT 진행량과의 시간 가중 절대 오차로 평가한다. 이를 직접 호출하면 한 개씩 선택한 D3는 **0.104139**로 줄어든다.

| P1024×V256 고정 | GT coarse 1×1 | GT coarse 20×10 | 전체 사전 oracle |
|---|---:|---:|---:|
| 공간 경로 MSE + 4초 속도 MAE | 0.187180 | 0.108033 | 0.103896 |
| 실제 trainer D3 coarse 비용 | 0.104139 | 0.103896 | 0.103896 |

`--coarse-mode d3`는 비용을 복제해 작성하지 않고 실제 `experiments/sparsedrivev2_20260910/losses.py::coarse_costs`를 호출한다. 결과 sidecar에 losses/data 소스 SHA를 기록했다. 정답 기반 shortlist 결과는 학습된 모델이 이 shortlist를 찾았다는 뜻이 아니다. exact-stop 중복이나 동점 때문에 기록된 특정 oracle winner의 recall이 낮아도, 동일 최적 오차의 다른 후보를 남겼을 수 있다.

## 무엇을 개선한 사전인가

기존 normalized-progress 경로를 속도 총이동거리로 확대·축소하는 방식과 달리, 이 사전은 **실제 미터 단위 경로의 곡률을 유지**하고 절대 속도를 적분한 진행량에서 보간한다. 경로에 status·goal을 넣어 좌표를 바꾸는 연산은 없다. 모델은 이미 완성된 조합 중 행을 선택한다.

원시 pose의 같은 장면 내 현재 frame부터 마지막 frame 349까지 실제 남아 있는 기록을 사용했다. train에서 3초 이동거리 최댓값은 69.542m, 4초 최댓값은 92.249m여서 공개 NAVSIM의 50m 경로는 부족했다. 공개 임베딩의 50-point 형상은 유지하면서 2m 간격 100m로 확장했다. 곡선·정지·후진 방향의 좌표를 임의로 직선 padding하지 않는다. 정지와 중복 arclength는 제거해 보간하고, 지원 길이 밖 좌표에는 invalid mask를 둔다. 후보를 선택할 때 첫 6개 점이 모두 valid여야 한다.

경로는 100m 전체가 관측된 train **40,730/54,810행(74.31%)**에서 MiniBatchKMeans로 학습했다. 포함 행의 첫 구간 속도 평균은 12.704m/s, 제외 행은 6.281m/s이므로 이 조건에는 고속 편향이 있다. 1m 이상 움직이는 행에서 3초 endpoint 방향각이 10도보다 큰 비율도 포함 3.80%, 제외 6.92%로 차이가 있다. 이 문제를 숨기지 않고 별도 audit으로 저장했다. 다만 현재 사전의 경로를 GT 진행량으로 보간한 최선 D3는 **0.009797**이고 속도 진행량의 최선 오차는 **0.101862**여서, 이번 P1024×V256에서는 속도 양자화가 더 큰 병목이라는 근거가 있다. 두 수치는 가산적인 인과적 오차 하한이 아니다.

GT native 경로를 2m 간격으로 샘플링하고 GT 속도를 함께 쓰는 재구성 오차는 지원되는 **1,918행에서 0.001835**였다. 짧은 native 경로 때문에 80행은 이 재구성 진단에서 제외되므로 전행의 보편적인 오차 하한으로 해석하면 안 된다.

속도는 모든 train 54,810행을 사용했다. `absolute8`은 8개 구간 절대 속도의 군집 중심이다. `progress6`은 첫 6개 누적 진행량에 D3 시간 가중치의 제곱근을 곱해 군집화한 뒤 차분으로 첫 6개 속도를 복원하고, 뒤 2개는 같은 cluster에 배정된 실제 train 속도의 평균을 붙인다. 이로써 공개 8차원 속도 임베딩을 유지한다. 군집화의 weighted 제곱오차는 trainer의 weighted 절대오차와 동일한 목적함수가 아니며, 실제 개선은 별도 전체 사전 D3 평가로 확인했다.

## 데이터 경계와 출력 계약

- 원 split SHA: `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`.
- train203, frame≥30, stride1: 54,810행, SHA `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`.
- tune37, frame≥30, stride5: 1,998행, SHA `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`.
- 각각 독립 extract → train-only fit → bank SHA 고정 → tune oracle 순서로 실행했다. 136개 reserve 장면은 추출·평가하지 않았다. 원시 pose는 선택된 장면의 파일만 읽었다. 전체 native 범위 −50..349에 대한 별도 split 감사에서도 교차 집합 timestamp 중복은 0이다.
- 전체 SE3 역변환으로 x-forward/y-left 좌표를 만들었으며 기존 fut6와의 최대 차이는 train 3.82e-6m, tune 3.31e-6m 미만이다. fut5의 첫 6점과 fut6는 bitwise 동일하다.
- `path_xy[P,50,2]`, `path_xyz[P,50,3]`, `path_mask[P,50]`, `path_stations[50]`, `velocity[V,6]`, `velocity8[V,8]`, `traj_xy[P,V,6,2]`, `traj_xy8[P,V,8,2]`, `traj_xyz8[P,V,8,3]`, `traj_mask[P,V,6]`, `mask8[P,V,8]`, `candidate_valid[P,V]`를 제공한다. V=0은 exact stop이다. Cartesian 형상을 유지하므로 stop은 P개 중복 저장된다.
- confirmation12 또는 train 내부 crossfit에서는 그 fold의 train만으로 새 사전을 적합해야 한다. 현재 train203 사전을 train171 확인 실험에 재사용할 수 없다.

## 검증 및 재현 위치

코드는 원격 worktree의 `experiments/sparsedrivev2_20260910/bank.py`, 은행 계약 검증은 같은 디렉터리의 `test_bank.py`에 있다. exact stop, 중복점/역방향, 물리 곡률 유지, 외삽 mask, 공식 지표, split/hash 경계, chunked oracle, 고정 경로 재사용과 progress6/8차원 보존을 포함한 **7개 테스트가 통과**했다.

CPU NumPy float64에서 11개 tune 세션의 대표 프레임마다 **전체 1,048,576개 후보를 독립 완전탐색**했다. 실제 유리수 D3 가중치로 계산한 값과 CUDA float32 oracle의 최대 차이는 **6.68e-9**였다. 별도 검토 agent는 P1024×V512 progress6 사전의 8개 경로×전체 512개 속도를 재합성해 저장 궤적·mask와 bitwise 일치도 확인했다.

모든 생성·평가 파일은 새 worktree 안의 아래 위치에 저장했다. GPU 작업은 허가된 물리 GPU4의 UUID `GPU-4b804d68-fd61-af14-393a-573c533d5006`만 사용했으며 각 시작 전에 idle을 확인하고 고유 PID/log receipt를 남겼다. GPU 작업은 모두 끝났다.

- `cache/sparsedrivev2_20260910/bank/`: train/tune native 추출, 각 사전 NPZ, SHA 포함 sidecar.
- `reports/sparsedrivev2_20260910/bank/oracle_comparison_summary.json`: 전체 비교와 paired session bootstrap.
- `reports/sparsedrivev2_20260910/bank/*_oracle*.npz[.json]`: 각 행의 oracle, 선택 index, 시간별 오차, GT coarse 결과, 거리 bin.
- `reports/sparsedrivev2_20260910/bank/p1024_v256_native100m_v8_decomposition.npz.json`: 경로/속도/2m 샘플링 진단과 train eligibility 편향.
- `reports/sparsedrivev2_20260910/bank/independent_numerical_validation.json`: CPU 독립 oracle, 고정 경로 bitwise 확인, 각 사전 hash/크기.

최종 사전 재현 시 아래 명령의 출력 경로를 **아직 존재하지 않는 새 경로**로 지정한다. 기존 결과를 덮어쓰지 않도록 코드가 거부한다.

```bash
/usr/bin/python experiments/sparsedrivev2_20260910/bank.py fit \
  --train cache/sparsedrivev2_20260910/bank/train_native100m_v8.npz \
  --path-bank cache/sparsedrivev2_20260910/bank/p1024_v256_native100m_v8.npz \
  --paths 1024 --velocities 1024 --velocity-mode progress6 --threads 4 \
  --output NEW_BANK.npz

CUDA_VISIBLE_DEVICES=GPU-4b804d68-fd61-af14-393a-573c533d5006 \
/usr/bin/python experiments/sparsedrivev2_20260910/bank.py oracle \
  --bank NEW_BANK.npz \
  --tune cache/sparsedrivev2_20260910/bank/tune_native100m_v8.npz \
  --device cuda:0 --coarse-mode d3 --output NEW_ORACLE.npz
```
