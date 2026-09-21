# Native1152 이후 남은 방법 — 2026-09-22

이번 작업은 완료 모델의 고정 가중치 개입과 저장 예측 재계산이다. Optimizer update 0, checkpoint 파일 변경 0, 신규 FULL/제출 0. 기준 작업 HEAD는 `5d9960200e110490f58e3750839cb7abc713f353`이다.

## 1. 이번에 실제 확인한 것

같은 V0 1,998행/11세션에서 native_projection을 메모리에서만 0으로 바꿔 추가 분기의 출력을 제거했다. 같은 batch/같은 process의 ON/OFF를 비교했다. 끝난 뒤 원래 값을 복원하고 모든 parameter/buffer의 hash 일치를 확인했다. 이것은 재학습 대조군이 아니라 학습된 모델의 의존성 개입이다.

|모델|추가 분기 ON PREFIX|추가 분기 OFF PREFIX|OFF−ON|두 예측 사이 PREFIX 거리|
|---|---:|---:|---:|---:|
|M-NATIVE terminal|0.144597723|0.144655365|+0.000057642|0.000467688m|
|S-NATIVE terminal|0.144711520|0.154468267|+0.009756747|0.059109222m|

- **M의 추가 고해상도 motion 경로에 대한 현재 출력 의존은 매우 작다.** 모든 고해상도 정보가 무용하거나 학습 내내 무관했다는 뜻은 아니다. 공유 backbone은 추가 분기의 학습 중 변했을 수 있다.
- S는 추가 scene 경로를 실제로 사용한다. 그러나 독립 학습 S-LOW도 0.144906334이고 S-NATIVE는 0.144711511이다. OFF 악화량 0.00976을 원본1152의 독립 학습 이득으로 해석하면 안 된다.
- 따라서 네 모델이 비슷한 점수로 끝난 것을 모두 '분기를 무시한다'로 설명할 수 없다. M/S가 다르다. 더 큰 이미지 입력의 우수성 역시 아직 확인되지 않았다.
- ON 재현의 저장 예측 대비 최대 좌표 차이는 M 0.000170946m, S 0.000057220m였다. 기존 1mm 재현 계약 이내이며 정확한 bit parity라고 하지 않는다. ON/OFF는 동일 process 기준이다.
- GPU0/1 진단은 각각 약77초의 forward 평가였으며 종료 후 GPU0–3 모두 유휴다.

파일: `M-NATIVE_FROZEN_BRANCH.json`, `S-NATIVE_FROZEN_BRANCH.json`, 각 `*_protocol.json`. 행별 예측은 각 JSON에 기록한 work_dirs 경로/SHA로 보존한다.

## 2. 최신 모델에서도 길이/방향 상보성이 남는가

같은 행·GT·session을 확인하고 기존과 동일한 구간 분해 함수를 적용했다. 모든 예측 구간 길이가 양수임을 확인했다. GT 방향 치환은 GT 구간 길이 >0.05m에서만 적용하고 나머지는 모델 방향을 유지한다.

|M-NATIVE 저장 예측에 대한 진단|PREFIX|해석|
|---|---:|---|
|원래 단일 모델|0.144597662|실제 DEV terminal|
|현재 예측 길이 + 예전 H4-DIRECT 예측 방향|0.142351560|두 모델의 저장 출력 재조합; 새 학습/배포 후보 아님|
|GT 길이 + 현재 예측 방향|0.059796801|정답을 사용한 성분 치환|
|현재 예측 길이 + 유효점 GT 방향|0.115140106|정답을 사용한 성분 치환|

DIRECT 방향 재조합의 차이는 -0.002246102이며 session CI95는 [-0.005999093, +0.000261720]으로 0을 포함한다. 이전 CTRL에서의 약0.0060 이득을 현재 모델에도 그대로 적용하면 안 된다. GT 성분 치환 수치는 학습 가능한 하한이나 회수 가능한 점수의 예측이 아니고 서로 가산할 수도 없다.

최신 M-NATIVE의 nonstop 1,875행은 전체 점수의95.806%다. 동일 GT tangent mask에서 종/횡 절대오차는 약0.11846/0.06291m다. SplitRead 부모에서 조사한 정속에 가까운 집단과 직진 집단의 기여도도 높았으므로, '남은 원인은 가감속 타이밍뿐'으로 좁힐 수 없다. 이 통계는 DEV이며 서버 test의 종횡 분해가 아니다.

## 3. 아직 구분해서 시험할 가치가 있는 세 축

### A. 주력 제안: 영상 backbone의 표현 용량 변경

현재까지 주력 계열은 같은 ResNet50 + 128D FPN을 공유한다. MH4/QREFINE/FINE/SplitRead/native1152는 이 표현을 읽거나 추가 관측으로 보강하는 실험이었다. DINO 특징 교사 continuation의 실패도 실제 backbone 교체/확장과 동일한 실험은 아니다.

구체적인 첫 후보는 **R50의 학습된 trunk를 보존하며 R101 깊이로 확장하는 한 종류**다. 입력 해상도·시점·A2 status 경계·FPN 출력 차원·planner·loss를 고정한다. 기존 block은 그대로 옮기고 추가 residual block은 identity가 되도록 초기화하는 설계를 검토한다. 이렇게 시작하면 기존 인지/계획 함수의 큰 붕괴를 피하면서 시각 특징 자체를 더 깊게 학습할 수 있는지를 시험할 수 있다.

**아직 구현·초기 parity·gradient·4090 비용을 확인한 모델은 아니다.** Trunk 용량이 원인으로 확정된 것도 아니다. 비교는 같은 parent에서 기존 R50 control과 새 trunk를 동일 추가 예산으로 학습해야 한다. 기존 layer까지 학습시키되, 새 block과 기존 layer의 LR을 사전 고정하고 실측으로 완료 가능한 budget을 잡는다. 모든 제안 변경을 한 번에 섞지 않는다.

대체 가능한 공개 nuImages R101 가중치도 존재하지만 무조건 더 우수한 초기값으로 취급하지 않는다. 현재 R50은 COCO20e+nuImages20e 계보이며 공개 R101의 다른 사전학습 schedule과 차이가 있다. 이 경우는 backbone 용량만의 비교가 아니라 initializer 패키지 비교다. 단순히 '더 큰 공개 checkpoint니까 좋아진다'고 주장하지 않는다.

기존 함수를 보존해 네트워크를 확대한다는 원리의 참고: [Net2Net](https://arxiv.org/abs/1511.05641). 공개 R50/R101의 출처·서로 다른 레시피: [MMDetection3D nuImages model zoo](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages/README.md). 논문/검출 AP는 MotionDrive PREFIX 개선폭의 근거가 아니다.

### B. 작은 독립 비교: 길이/방향 query와 decoder까지 분리

`experiments/a2_design_preflight_20260921/design.py::SplitReadPlanner`는 **length_read/heading_read와 출력 head**만 분리했다. `waypoint_queries`와 `decoder`는 공유하고 두 readout에 같은 decoded feature를 준다. 따라서 query/decoder까지 분리하는 실험은 아직 완료하지 않았다.

기존 query/decoder를 두 경로에 복사해 초기 함수를 유지하고, 같은 image-derived scene/motion memory에서 각각 진행량과 방향을 읽는 한 종류의 비교가 남아 있다. 기존 SplitRead 전체가 미실행이었다고 표현하지 않는다. 두 checkpoint를 배포하거나 GT로 방향을 선택하는 제안도 아니다.

다만 최신 DIRECT 방향 진단의 이득은0.00225이고 CI도0을 포함한다. 이것이 전체 분리 학습의 상한은 아니지만, 이 축 하나로 서버0.12를 자신 있게 예측할 근거도 아니다. 주력 시각 표현 변경과 독립된 작은 비교로 둔다.

### C. 더 큰 감독 변경: 주변 agent의 미래 움직임을 공통 scene에 학습

현 A2는 객체의 현재 footprint occupancy와 lane 등을 학습하지만 주변 agent의 미래 trajectory를 현재 모델의 정식 task로 학습하지 않는다. 따라서 '현재 어디에 있는가'와 별도로 '어느 방향으로 얼마나 이동하는가'를 공통 scene feature에 감독하는 변경이 남아 있다. 미래 GT는 loss 전용이고 inference는 이미지에서 만든 feature만 사용한다. A2의 raw status query-only 경계와 motion/state 분리를 유지한다.

전체 UniAD나 새 detector/tracker를 이식하기보다, 기존 shared scene에서 agent 중심의 미래 변위를 보조 예측하는 한 가지로 시작하는 쪽이 작다. 이 방향은 [UniAD](https://arxiv.org/abs/2212.10156)의 perception/prediction/planning 연결과 개념적 관련이 있지만, 그 벤치마크 성과를 우리 점수로 옮길 수 없다.

과거 `scripts/build_agent_labels.py`의 cache는 train330, 최대40agent 절단, FP16 좌표를 사용한 다른 모델용이다. 최신 DEV310/V0 계약으로 그대로 재사용해서는 안 된다. 원본 라벨의 좌표·증분/절대 의미·유효 mask·scene split을 확인해 train310용 target을 만들어야 한다. 정속 구간 오차도 크므로, agent 감독만으로 남은 오차 전부를 설명하지 않는다.

## 4. 우선 반복하지 않을 것

- 원본 해상도를1536/1920으로 계속 확대: native1152의 추가 정보 이득이 먼저 입증되지 않았다.
- MH8/16, VECTOR lambda, FINE token 수, 작은 ego residual의 자동 sweep.
- 공개 RAFT를 그대로 달아 metric v0를 얻는 안: 과거 zero-shot 노면 flow 진단에서 큰 scale/상관 문제가 있었고, 별도 control-flow 실험도 이득을 내지 못했다. C2F full-budget 실험도 이미 있다. 도메인별 flow/depth 학습이 모두 불가능하다는 뜻은 아니지만 새로운 간단한 정답처럼 다시 제안하지 않는다.
- 같은 DINO 특징 증류 재실행: 완료된 설정은 control/parent보다 악화했다.
- 준비되지 않은 recurrent BEV 전면 교체: 지금도 구현·학습·배포 비용이 큰 별도 프로젝트다.

## 5. 판단과 남은 작업

**큰 개선을 노리는 다음 주력은 backbone 표현 변경, 작은 별도 비교는 완전한 길이/방향 decoder 분리로 권고한다.** Agent 미래 감독은 실제로 남은 새 감독 축이지만 라벨 정합 작업과 불확실성이 더 크다. 현재 자료로 어느 안도 서버0.12를 보장하지 않는다.

기존 최선 DEV와 완료된 FULL/제출물을 보존한다. 최신 native/SplitRead 개선 레시피의 새 FULL은 아직 실행하지 않았다. 이 문서는 후보와 실험 이유를 정리한 것이며 새 학습이나 예약을 뜻하지 않는다. Official0.133684828과 DEV0.144597662를 환산식으로 연결하지 않는다.

재현:

```bash
python experiments/a2_post_native_review_20260922/review.py --mode components
CUDA_VISIBLE_DEVICES=0 python experiments/a2_post_native_review_20260922/review.py --mode ablate --arm M-NATIVE
CUDA_VISIBLE_DEVICES=1 python experiments/a2_post_native_review_20260922/review.py --mode ablate --arm S-NATIVE
```

원본 산출물 덮어쓰기를 거부하므로 재실행은 별도 출력 경로를 사용한다. 새 학습은 이 스크립트에 없다.
