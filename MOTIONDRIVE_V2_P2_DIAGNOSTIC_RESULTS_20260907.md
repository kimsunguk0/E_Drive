# P2 정상 종료 3팔 분리 진단 결과

2026-09-07 · B200 · 작업 루트 `/NHNHOME/data/sukim/adcl`

## 판정 범위

**전체 P2 4팔 clean-exit gate는 FAIL을 유지한다.** C1T0는 LAST3000 저장·평가 뒤 SIGSEGV로 종료했다(actual return code −11, supervisor 139). 해당 팔을 정상 완료로 재분류하거나 빈 결과를 대입하지 않았다. 이 문서는 정상 종료한 C0T0/C0T1/C1T1의 사전 지정 LAST3000을 각각 재평가한 진단 기록이다. strict P2 analyzer를 호출하거나 gate를 우회하지 않았으며, 완전한 2×2 효과·상호작용 판정 또는 모델 선택은 하지 않는다.

집계와 실행 증거 참조: [p2_three_clean_arm_diagnostic_execution.json](reports/p2_three_clean_arm_diagnostic_execution.json). 원 frame별 대형 JSON을 이 문서에 복제하지 않는다.

## 실행·무결성 검증

- 학습 source: `c6845fb33e548462f0fecead8baa1ec259d10d4a`.
- 재평가 및 집계 시 B200 HEAD: `59ee5f1a58ea9b92a3574811a743684d58140342`.
- 집계 무결성 검사는 2026-09-07 21:22:48 KST까지 수행했다. 기록된 실행 source 17개를 실제 재해시했으며, 체크포인트·실행 기록 등을 포함한 53개 파일을 이번 집계의 시작/끝에서 다시 해시했다. 기록된 입력/source SHA 불일치와 관찰 구간 내 변경은 모두 0개다. 이것을 전체 학습·평가 시간 동안 모든 파일이 연속 감시되었다는 주장으로 확대하지 않는다.
- 공통 초기 checkpoint SHA `e2a0e2dc…`, 초기 tensor SHA `e3823058…`, 301개 학습 로그 시점 전체 sample-order SHA 일치. 최종 sample-order SHA `cf4a27a4…`.
- 세 팔 모두 원 config를 strict load했으며 override는 없다. step=3000, bf16, B4, workers4, tune stride5, 1,998프레임·37 scene·11 session이다. C0T0는 raw time, C0T1/C1T1은 nominal time이다.
- `normal/image_shuffle/repeat_current/reverse_history`의 수신 row 순서·GT가 동일하고, plan/state/history 예측은 모두 유한하다. 수신 row SHA `1a65ade6…`, 고정 donor SHA `c579365e…`; donor seed=20260907. 추가 GT 조회와 final-val 접근은 없다.
- 정상 D3는 세 팔 모두 원 학습 LAST3000 로그의 float 값과 정확히 일치한다.

| 팔 | 실제 평가 장치 | child PID | 실제 OS 종료 | 평가 종료 KST |
|---|---|---:|---|---|
| C1T1 | 물리 GPU4 → 단일 UUID → cuda:0 | 2043268 | 0 / completed_cleanly | 21:13:07 |
| C0T1 | 물리 GPU5 → 단일 UUID → cuda:0 | 2043285 | 0 / completed_cleanly | 21:13:08 |
| C0T0 | 물리 GPU4 → 단일 UUID → cuda:0 | 2044854 | 0 / completed_cleanly | 21:16:24 |

관측 UUID와 `CUDA_VISIBLE_DEVICES`가 요청 물리 장치와 일치했다. GPU4 UUID는 `GPU-4b804d68-fd61-af14-393a-573c533d5006`, GPU5는 `GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8`이다. PyTorch의 접두사 없는 raw UUID도 실행 receipt에 따로 보존돼 있다. 최초 두 UUID 표기 검증 실패(rc=1) 기록은 삭제하지 않고 r2 성공과 분리했다.

모든 평가에 allocator cap=12000MiB/reserve=8192MiB가 적용됐고 pressure event와 수신 signal은 없다. 최대 allocated 약 1078.55MiB/reserved 1764MiB, 최소 관측 free=180052MiB였다. 평가 완료 후 parent/child PID 부재 및 GPU4/5 compute PID 부재를 확인했다. 이 수치는 추론 지연시간 벤치마크가 아니며 OOM 방지 보장도 아니다.

## 공식 D3와 교란 진단

지표는 6개 cumulative absolute waypoint의 Euclidean L2에 `[11,11,5,5,2,2]/36`을 적용한 프레임 평균이다. 세션 평균은 11개 세션에 같은 가중치를 준 별도 요약이다.

| 조건 | C0T0 | C0T1 | C1T1 |
|---|---:|---:|---:|
| normal | 0.3671331447833502 | 0.36617518485979633 | 0.4454620049779748 |
| image shuffle | 2.1262699022051974 | 2.116021299059759 | 2.6040949631702377 |
| repeat current | 0.3881417257963932 | 0.38545121907539015 | 0.49038521708637195 |
| reverse history | 0.36825016775150765 | 0.36714039460659864 | 0.45071614623205875 |
| normal 세션 평균 | 0.4187268837451988 | 0.4191721309524125 | 0.4885254611937917 |

전체 이미지를 바꾸면 크게 악화하므로 영상 의존성은 확인된다. 과거를 현재 이미지로 대체하면 D3가 +0.021009/+0.019276/+0.044923 악화하지만, 역사 순서를 뒤집은 변화는 +0.001117/+0.000965/+0.005254로 작다. 교란 입력의 분포 변화가 포함되므로 이 숫자를 과거 정보의 이론적 가치, 순서 무시의 확정 증명, 또는 개선 가능한 D3로 해석하지 않는다.

## 정상 조건의 GT state 버킷

버킷은 현재 GT state로 고정했다. stop은 `hypot(vx,vy)<0.2m/s`가 우선하고, 나머지는 ego-x `ax≥0.5`를 accel, `ax≤−0.5`를 decel, 중간을 cruise로 나눈다. 미래 경로로 버킷을 튜닝하지 않았다. x/y는 현재 ego 고정 축이지 각 경로의 접선/법선이 아니다.

| 버킷 | 프레임 / 세션 수 | C0T0 D3 | C0T1 D3 | C1T1 D3 |
|---|---|---:|---:|---:|
| stop | 123 / 6 | 0.544003 | 0.543136 | 0.517663 |
| accel | 203 / 10 | 0.430651 | 0.432324 | 0.531782 |
| decel | 186 / 9 | 0.411836 | 0.408895 | 0.451922 |
| cruise | 1486 / 11 | 0.338221 | 0.337144 | 0.426885 |

버킷별 세션 동일가중 평균은 집계 JSON에 함께 보존했다. 특히 stop 123프레임은 전체의 약 6.16%이며, 그 안의 세션 수가 6개뿐이다. 버킷 최악값이나 상관 하나로 다음 모델을 선택하지 않는다.

## 신경 상태와 계획의 간극: 사후 관찰

근거는 [C1T1 motion 분석](reports/p2_c1t1_last3000_motion_analysis.json)과 [C0T1 motion 분석](reports/p2_c0t1_last3000_motion_analysis.json)이다. 이 분석기는 기존 동일 forward가 기록한 neural state/history를 읽었으며 추가 forward·GT 조회·예측 보정을 하지 않았다.

| normal 사후 지표 | C0T1 | C1T1 |
|---|---:|---:|
| neural yaw-rate 대 GT Pearson r | 0.946174 | 0.946664 |
| neural yaw-rate 대 미래 GT y의 b 계수 r | 0.818613 | 0.816171 |
| 예측 y의 b 대 GT y의 b 계수 r | −0.265140 | −0.314587 |
| b의 **표준편차 비율** pred/GT | 0.072543 | 0.075044 |
| b의 **분산 비율** pred/GT | 0.005262 | 0.005632 |
| ax MAE (m/s²) | 0.282506 | 0.284481 |
| 동일 GT의 고정 zero ax MAE (m/s²) | 0.297278 | 0.297278 |

중요 정정: `0.075`는 분산 비율이 아니라 표준편차 비율이다. C1T1의 실제 분산 비율은 약 **0.563%**다. `b`는 각 미래 경로 6점에 비가중·무절편 OLS `y(t)=a·t+b·t²`를 적합한 사후 시간 계수이며, 현재 가속도나 공간 곡률 그 자체가 아니다(적합 곡선의 시간 2차 미분은 `2b`).

현재 yaw 신호와 미래 GT의 측면 변화 사이에는 강한 관측 상관이 있는데, 계획의 해당 시간 2차 성분은 변동이 작고 GT와 약한 음의 상관을 보인다. **상태/장면 정보가 계획의 시간 변화로 충분히 이어지는지 점검할 근거**지만, 특정 planner 모듈의 결함이나 연결 변경의 예상 이득을 증명하지 않는다. 골, 속도, 장면, 세션 등 공변량과 출력/GT의 좌표·시간 정의를 구분해야 한다. 상관의 부호만 보고 배포 좌표를 뒤집거나 수식을 덧붙여 보정하면 안 된다.

종방향 가속도도 해결됐다고 볼 수 없다. C1T1 ax MAE 0.284481은 zero 기준 0.297278보다 약 4.3% 낮을 뿐이고 r=0.467864다. C0T1도 약 5.0% 개선에 그친다. 높은 yaw 상관이나 낮은 학습 loss가 정밀한 가속도 추정·계획 활용을 보장하지 않는다. 이 diagnostic records에는 GT history/mask가 없으므로 history 정확도는 여기서 새로 계산하지 않았다.

## 결론을 제한하는 조건

1. C1T1은 잘못된 rear geometry로 학습된 P1 초기 모델에 보정 geometry를 적용했을 때 0.989626에서 시작해 3000-step 후 0.445462로 적응했다. 동일 nominal 시간의 C0T1 0.366175보다 +0.079287 나쁘지만, 이것은 **기존 checkpoint의 단기 보정 적응** 결과이지 보정 geometry로 처음부터 학습하는 모델의 한계가 아니다.
2. C0에서 nominal time의 프레임 D3 차이는 −0.000958이고, 세션 동일가중 차이는 +0.000445로 부호가 바뀐다. 확정 우위나 실질적인 성능 도약으로 보고하지 않는다.
3. 주판정은 LAST3000을 유지한다. C1T1의 더 낮은 중간 BEST2250=0.439644를 사후 주결과로 교체하지 않는다.
4. 단일 seed, 반복 관찰한 tune 37 scene/11 session의 탐색 결과다. 독립 holdout 성과나 정식 신뢰구간을 주장하지 않는다. 정상 3팔만으로 빠진 C1T0를 복원하거나 전체 P2 gate를 통과시킬 수 없다.
5. 공개 리더보드와 이 tune 점수를 직접 비교해 예상 순위를 산출하지 않는다. 이번 문서는 결과 증거 보존이며 모델 선택·재학습·배포 승인이 아니다.
