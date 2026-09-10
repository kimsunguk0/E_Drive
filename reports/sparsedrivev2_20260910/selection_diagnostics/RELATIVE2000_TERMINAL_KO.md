# 상대 후보 선택 head: 고정 2,000-step tune 결과

2026-09-10. 원 tune37의 1,998행, 11개 세션을 동일 순서로 비교했다. 두 head의 고정 terminal `eval_002000.npz`만 사용했으며, head soft CE의 tune-best 1,500-step 결과는 선택하지 않았다. CPU 배열 분석만 실행했고 새 모델 forward·GPU·held 평가를 하지 않았다.

**결과:** soft CE 0.130749, centered D3 0.131394로 원 고정 base 0.193903보다 개선됐다. 완료된 200개 후보와 shortlist oracle은 동일하므로 이번 D3 감소는 fine 후보 선택 오차 감소와 정확히 같다. 원 MotionDrive 0.269906보다도 낮다. 이 수치는 개발에 반복 사용한 tune 결과이며 독립 confirmation 결과는 아니다.

| 방법 | D3 | 11-session bootstrap 95% CI | Fine regret | 3초 끝점 L2 |
|---|---:|---|---:|---:|
| MotionDrive long s1 | 0.269906 | [0.215854, 0.327942] | — | 0.589208 |
| SparseDriveV2 고정 base | 0.193903 | [0.172477, 0.220611] | 0.103409 | 0.785202 |
| 상대 head: soft CE | 0.130749 | [0.111638, 0.153115] | 0.040255 | 0.442827 |
| 상대 head: centered D3 | 0.131394 | [0.110168, 0.155189] | 0.040900 | 0.454825 |

공통 shortlist oracle D3는 0.09049408이다. 원 fine regret 0.10340884가 soft CE에서 0.04025454, centered D3에서 0.04089999로 줄었다. 세션을 동일 확률로 20,000번 복원 추출하고 각 복원 표본에서 행 수로 가중한 평균을 사용했다(seed 0). 행을 독립 표본으로 취급하지 않았다.

| 비교: 새 방법 − 기준 | D3 차이 | paired 95% CI | 개선 세션 |
|---|---:|---|---:|
| relative_soft_ce_2000 minus dense_none_2000 | -0.063154 | [-0.070657, -0.059074] | 11/11 |
| relative_centered_d3_2000 minus dense_none_2000 | -0.062509 | [-0.069514, -0.058734] | 11/11 |
| relative_soft_ce_2000 minus motiondrive_long_s1 | -0.139157 | [-0.180258, -0.094773] | 11/11 |
| relative_centered_d3_2000 minus relative_soft_ce_2000 | 0.000645 | [-0.001887, 0.002639] | 2/11 |

두 head 모두 11개 세션 전체에서 원 base 및 MotionDrive보다 D3가 개선됐다. CE와 centered D3의 차이 CI는 0을 포함하므로 손실 함수 사이의 우열을 확증하지 않는다. 두 방법의 개별 D3 CI 상단이 0.15를 약간 넘고, 반복된 tune 개발의 적응적 선택까지 반영한 CI도 아니므로 새로운 분포에서 ≤0.15를 보장하지 않는다.

| 방법 | 0.5초 | 1.0초 | 1.5초 | 2.0초 | 2.5초 | 3.0초 |
|---|---:|---:|---:|---:|---:|---:|
| MotionDrive long s1 | 0.116042 | 0.228277 | 0.331350 | 0.418169 | 0.501542 | 0.589208 |
| SparseDriveV2 고정 base | 0.040284 | 0.102880 | 0.203447 | 0.347579 | 0.540087 | 0.785202 |
| 상대 head: soft CE | 0.038257 | 0.082636 | 0.144199 | 0.224706 | 0.323471 | 0.442827 |
| 상대 head: centered D3 | 0.039798 | 0.082883 | 0.142025 | 0.222428 | 0.324388 | 0.454825 |

soft CE의 3초 L2 감소는 base 대비 −0.342375m, paired CI [−0.379466, −0.312159]이며 MotionDrive 대비 −0.146381m [−0.210443, −0.072076]이다. 따라서 평균 3초 오차 악화는 이번 tune 비교에서 해소됐다. 다만 아래 tail과 일부 집단까지 모두 해결된 것은 아니다.

| 방법 | 시간가중 평균 절대 x 오차 | 시간가중 평균 절대 y 오차 | D3 p95 / p99 | 3초 L2 p95 / p99 |
|---|---:|---:|---|---|
| MotionDrive long s1 | 0.238492 | 0.072180 | 0.738457 / 1.106390 | 1.553982 / 2.455419 |
| SparseDriveV2 고정 base | 0.167934 | 0.061098 | 0.519294 / 0.971841 | 2.164610 / 4.112349 |
| 상대 head: soft CE | 0.109314 | 0.046292 | 0.350177 / 0.645190 | 1.358086 / 2.576428 |
| 상대 head: centered D3 | 0.109290 | 0.047382 | 0.338623 / 0.660034 | 1.421403 / 2.576428 |

x/y 절대 오차는 진단용이며 두 값을 합쳐 D3로 해석하면 안 된다. CE의 3초 p99 2.576428m은 MotionDrive 2.455419m보다 높다. 3초 최대도 CE 5.836469m, MotionDrive 4.917325m이다. 원 base의 p99 4.112349m·최대 8.442844m보다는 개선됐지만 극단 사례의 3초 꼬리 오차는 남았다.

| 진단 집단 | 행 수 | MotionDrive D3 | Base D3 | CE D3 | Centered D3 |
|---|---:|---:|---:|---:|---:|
| 현재 속력 [0,0.5)m/s | 132 | 0.720379 | 0.110872 | 0.078730 | 0.080273 |
| 현재 속력 [0.5,5)m/s | 137 | 0.343090 | 0.210124 | 0.164422 | 0.166015 |
| 현재 속력 [5,10)m/s | 498 | 0.255759 | 0.230692 | 0.158037 | 0.158222 |
| 현재 속력 [10,15)m/s | 958 | 0.197306 | 0.184234 | 0.119491 | 0.118916 |
| 현재 속력 [15,20)m/s | 135 | 0.268029 | 0.207915 | 0.145114 | 0.150156 |
| 현재 속력 ≥20m/s | 138 | 0.323243 | 0.177874 | 0.112696 | 0.117379 |
| 현재 ax<−0.5m/s² | 189 | 0.281188 | 0.249300 | 0.178560 | 0.184153 |
| 현재 ax>0.5m/s² | 205 | 0.274049 | 0.225205 | 0.159619 | 0.155686 |
| GT 구간 속도 감소 >0.5m/s | 420 | 0.242188 | 0.220586 | 0.156677 | 0.158779 |
| GT 구간 속도 변화 ≤0.5m/s | 1177 | 0.268349 | 0.156210 | 0.101836 | 0.102240 |
| GT 구간 속도 증가 >0.5m/s | 401 | 0.303507 | 0.276591 | 0.188455 | 0.188284 |
| GT 3초 |y|>1m | 716 | 0.267391 | 0.229407 | 0.155940 | 0.157442 |
| GT 3초 |y|≤1m | 1282 | 0.271310 | 0.174074 | 0.116679 | 0.116846 |

속력은 causal sqrt(vx²+vy²)이며 모든 1,998행을 포함한다. 원 helper의 vx≥0 집단은 음수 vx 63행을 누락하므로 별도로 보완했다. GT 속도 변화는 0~0.5초와 2.5~3초 구간 이동 거리/0.5의 차이다. 이 GT 집단은 사후 진단에만 쓰며 실제 브레이크 조작 정답을 뜻하지 않는다. |y| 기준도 횡방향 이동 집단의 대용 지표이지 정확한 회전 라벨이 아니다.

모든 위 집단에서 두 head의 D3는 base보다 낮다. 그러나 CE의 3초 L2는 GT 감속 집단에서 0.534782m로 MotionDrive 0.525917m보다 약간 높고, 현재 속력 15~20m/s 집단에서도 0.459232m 대 0.437585m이다. 남은 문제는 평균 D3 실패가 아니라 일부 동역학 집단과 큰 오차 사례의 후반 예측이다.

원인 해석에는 한계가 있다. 원 base의 goal_mode는 none이지만 상대 head 특징에는 제공 목표점이 들어간다. 따라서 상대 운동 특징, 추가 MLP, 제공 목표점의 효과를 이 두 실험만으로 분리할 수 없다. 두 손실이 비슷하게 개선됐다는 사실만으로 CE가 병목이 아니었다는 일반 결론도 내리지 않는다.

감사 결과: 세 방법의 selected candidate ID가 캐시의 완료된 후보에 존재하며 저장 pred가 해당 후보 좌표와 bitwise 동일하다. shortlist oracle 배열도 세 방법 사이에서 bitwise 동일하다. 독립 FP64 D3 재계산과 저장값의 최대 차이는 1e-5 미만이었다.

근거 파일:

- `reports/sparsedrivev2_20260910/selection_diagnostics/relative2000_terminal_analysis.json`: 전체 시간별 x/y, session bootstrap, 세션별 차이, tail, 집단별 결과, 입력 SHA.
- `relative2000_error_groups.json`: 원 `analyze_selection_errors.py` 실행 결과.
- `analyze_relative_terminal.py`: CPU 확장 분석 코드.

고정 terminal 원본과 SHA:

- MotionDrive long s1: `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/reports/sparsedrivev2_20260910/baseline_long_s1_tune/predictions.npz`; SHA `c299f7462c51d9a5cd835ffdc4ee262a680a87614b5cc760d33cae6a535ff410`.
- SparseDriveV2 고정 base: `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/work_dirs/sparsedrivev2_20260910/dense2000_none_s0_v1/eval_002000.npz`; SHA `2fc771b0aea4acd08d9a6f527c0270a42a0c372f161dc7ca2a09dbe3aa741d22`.
- 상대 head: soft CE: `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/work_dirs/sparsedrivev2_20260910/relative2000_soft_ce_s0_v1/eval_002000.npz`; SHA `1063d81b459161851919926b1fc8a9e5ae10f26e2e95cc2d3254b23532c738b1`.
- 상대 head: centered D3: `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/work_dirs/sparsedrivev2_20260910/relative2000_centered_d3_s0_v1/eval_002000.npz`; SHA `d76597f94021d1777d27d747479364cf332254fa82db23ac55df19bf2d140508`.
