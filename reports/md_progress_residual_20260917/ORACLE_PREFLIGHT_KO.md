# 배포형 progress geometry oracle preflight

작성일: 2026-09-17

등록된 `MR-NATIVE-s1 step20554`의 저장 예측에 실제 학습·추론 코드와 같은 함수를 적용했다.

- 예측 segment tangent만 사용
- 길이 `<1e-3m`인 segment는 같은 예측 경로에서 가장 가까운 유효 tangent 사용
- 전 segment가 0이면 ego `+x` 사용
- 수정 interval 길이는 0 이상으로 clamp
- `|delta_v| <= 1.0m/s`, `|delta_a| <= 1.0m/s^2`
- 목적함수는 `[11,11,5,5,2,2]/36` PREFIX
- GT는 oracle 계수 최적화에만 사용하며 모델 입력·후처리에 사용하지 않음

## 결과

| 자료/출력 | L2_1s | L2_2s | L2_3s | PREFIX |
|---|---:|---:|---:|---:|
| train probe base | 0.049711 | 0.085716 | 0.134499 | 0.089975 |
| train probe scalar oracle | 0.033667 | 0.053084 | 0.102007 | 0.062920 |
| train probe dv+da oracle | 0.024027 | 0.046004 | 0.084360 | 0.051464 |
| V0 base | 0.110316 | 0.191025 | 0.271663 | 0.191002 |
| V0 scalar oracle | 0.040707 | 0.068466 | 0.133545 | **0.080906** |
| V0 dv+da oracle | 0.027957 | 0.058809 | 0.105112 | **0.063959** |

V0에서 scalar만으로도 충분한 표현 여지가 있고, 두 계수는 첫 2초를 포함해 PREFIX를 약
`0.01695` 더 낮춘다. 따라서 scalar를 버리지 않고 SIDE-S와 SIDE-VA를 같은 예산으로
동시에 검증한다. 낮은 oracle이 실제 학습 용이성을 보장하지는 않는다.

train probe와 V0의 차이가 크다. Scalar 최적계수 절대값 평균은 train `0.04745m/s`,
V0 `0.14160m/s`다. 이는 in-sample residual 분포 차이가 실제로 존재함을 보여준다. 이번
실험은 OOF 모델을 새로 만들지 않고, 영상 증강을 거친 새 temporal 표현과 최종 PREFIX를
joint 학습한다. 최종 판정은 oracle이나 coefficient MAE가 아니라 V0 final PREFIX로 한다.

V0에는 all-zero base row가 없고 짧은 base segment가 4개뿐이었다. Scalar cap 포화는
0.4004%, 두 계수는 각각 0.5005%/0.0501%였다. `cap=1.0`은 표현 병목을 거의 만들지 않는다.

원시 결과:

- `oracle_v0_cap1.json`
- `oracle_train_probe_cap1.json`
- producer: `experiments/md_progress_residual_20260917/analyze_deploy_oracle.py`
