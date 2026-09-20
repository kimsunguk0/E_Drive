# FRESH-CONT planner 관측과 다음 구조 제안

작성: 2026-09-20. 새 학습 없이 선택된 FRESH-CONT step5710에서 V0 전체를 관측했다. 아래 구조 변경은 **제안**이며 구현·학습·예약하지 않았다.

## 1. 이번에 실제 실행한 것

- 부모 source commit: `9ed139a5e3ec3730ab86c06e1db1c4802d2cb9ef`.
- Checkpoint: `work_dirs/a2_fresh_continue_20260920/A2-FRESH-CONT-s1/ckpt_step5710.pth`.
- Checkpoint SHA256: `073dc47df742b49a93a53ff7cb4bb3f3af7ea7a174a998dc6126597e184696cd`.
- 동일 nominal 입력, V0 1,998행, GPU0, B8. Optimizer/update 없음.
- Planner의 두 cross-attention에 수동 관측 hook을 걸어 원래 출력은 그대로 반환했다. Motion의 시간 가중치도 관측만 했다.
- Parameters/buffers SHA 전후 동일. 저장 예측과 최대 좌표 차이 `0.000026703 m`, 재현 PREFIX `0.153684068`.
- 원자료: [probe.json](probe.json). 재현 코드: [probe.py](../../experiments/a2_planner_readout_20260920/probe.py).

## 2. 확인된 정보 전달 구조

`motion_encoder.py:73–80`은 현재↔과거 네 시점의 특징을 `[B,4,192,128]`로 만든 후 시간축의 학습 가중합으로 `[B,192,128]`로 합친다. 시간/영상 위치 embedding은 합치기 전에 포함된다. 이 가중합은 미래 waypoint query를 받지 않는다.

`planner.py:45–47`은 scene 3,072개, 합쳐진 motion 192개, 예측 state/history token 1개를 하나의 memory로 읽는다. 미래 여섯 waypoint는 이미 합쳐진 motion에서 서로 다른 위치를 선택할 수 있으나, 원래 네 과거 시점의 feature를 따로 선택할 수는 없다.

|관측|1층|2층|
|---|---:|---:|
|Scene attention 총질량|91.8649%|94.0406%|
|Motion attention 총질량|8.0813%|5.9126%|
|State/history attention 총질량|0.0541%|0.0470%|
|Scene 성분의 out-projection 후 평균 L2 norm|2.38813|2.39112|
|Motion 성분의 out-projection 후 평균 L2 norm|0.21242|0.12943|
|State/history 성분의 out-projection 후 평균 L2 norm|0.001785|0.001579|

시간 가중치 평균은 과거 `[0.1,0.2,0.5,1.0]`초 순으로 `[0.15260,0.20080,0.27741,0.36919]`. 1초 과거가 가장 높은 가중치를 받는 sample×grid 비율은 87.02%지만, 이는 hard selection이 아니다. 다른 세 시점도 계속 가중합에 기여한다.

**이 수치로 motion이 무시되거나 가속도 정보가 사라졌다고 결론 내리지 않는다.** Attention 질량은 인과적 중요도가 아니며 성분 norm은 서로 더해지는 기여도가 아니다. Motion은 token 수 비율 자체가 5.88%이고, 이전 부모 모델의 motion 제거는 실제 PREFIX를 크게 악화시켰다. 시간 가중합도 시간 변화 정보를 encoding할 수 있다. 확인한 것은 정보 전달 위치와 접근 제한이며, 이것이 성능 병목인지는 학습 대조가 필요하다.

## 3. 우선 제안: 미래 waypoint가 합치기 전 motion을 직접 읽게 한다

기존 경로를 보존하고, planner의 decoded waypoint feature가 네 시점의 영상 motion memory를 한 번 더 읽도록 한다.

```text
기존 scene + pooled motion + 예측 state/history
                     ↓
                기존 planner → decoded waypoint feature
                                      ↓ query
합치기 전 motion [4×192,128] ─────→ 별도 cross-attention
                                      ↓ zero-init output projection
                         decoded feature에 잔차 추가
                                      ↓
                               기존 XY head
```

- 바뀌는 것은 미래 waypoint가 접근할 수 있는 **영상 시간 정보**다. 새 카메라/프레임/GT/current status 입력을 늘리지 않는다.
- 네 시점의 feature에는 기존 time/position embedding을 유지한다. XY를 외부 status로 적분하거나 scalar residual로 보정하지 않는다.
- Zero-init은 새 read의 마지막 출력 projection에만 적용해 초기 계획을 보존한다. 내부 Q/K/V까지 동시에 0으로 만들어 학습을 막지 않는다.
- 기존 pooled motion, state/history supervision, 공통 scene의 occupancy/lane/planner 연결은 유지한다. 미래 query를 motion/state head로 되먹임하지 않는다.
- Raw provided status/goal/pose는 새 memory/query 입력에 추가하지 않는다. 기존 planner query는 scene의 영향을 이미 받으므로 새 read 전체를 goal-independent라고 설명하지 않는다.
- 이 제안은 A2의 기존 정보 경계를 유지하는 설계다. 운영국의 개별 구조 승인을 대신하지 않는다.

일반적인 출력별 query 설계 참고는 [Perceiver IO](https://arxiv.org/abs/2107.14795)다. 해당 논문이 이번 MotionDrive의 PREFIX 개선을 검증했다는 뜻은 아니다.

## 4. 이전 실험과 실제 차이

9/9의 `OrderedTemporalResidual`도 네 시점을 활용했다. `4×128 → 64 → 128` MLP 결과를 pooled motion에 더했으며, planner는 여전히 시간축이 합쳐진 192개 token만 받았다.

|과거 2,000-update continuation|PREFIX|
|---|---:|
|Control|0.298734819|
|Ordered motion residual|0.298891119|
|Early delta auxiliary|0.298962121|

위 값은 `work_dirs/motiondrive_v2/early_precision_b0_*_last2000/final_eval.json`에서 확인했다. 당시 train54,810행/옛 부모의 결과이며 현재 FRESH와 직접 점수를 비교하지 않는다. 시간 정보를 다루는 아이디어를 한 번도 시도하지 않았다고 주장해서는 안 된다.

새 제안은 **미래 출력 query가 시간축을 유지한 memory를 읽는 위치**가 다르다. 이전 음성 결과를 무효화하는 설명은 아니며 새 효과 역시 미측정이다. C2F는 영상 대응을 바꿨고, MH4/QREFINE은 주로 scene read를 바꿨다. 이번에는 motion readout을 바꾼다.

## 5. 실행한다면 고정할 비교

주 구조 비교는 공개 nuImages trunk에서 시작한 기존 FRESH의 전체 20,554-update 레시피와 맞춘다. 기존 tensor 초기화/RNG/row stream/loss/nominal 입력이 같다는 확인을 거쳐 완료 FRESH control을 재사용한다. 새 arm은 같은 초기값에 새 read만 더한다. Terminal 0.158707259와 비교하며, 27,406-update continuation 또는 V0 선택 중간점과 학습량이 같은 것처럼 비교하지 않는다.

구조가 유효하면 그때 동일한 continuation을 별도로 적용한다. 실행 시간상 terminal에서 짧게 적응시키기로 바꾸면 같은 parent와 같은 예산의 control을 새로 두고 별도 stage-2로 기록한다. 수백 update 결과를 전체 구조의 한계로 일반화하지 않는다.

성공 지표는 실제 PREFIX 및 일반 주행/첫2초 기여다. Attention이 motion 쪽으로 이동하거나 state MAE가 줄어드는 것은 성공 기준이 아니다. 초기 parity, 추가 branch gradient, 입력 경계, 전체 forward 비용을 확인하고 V0/FULL 계보를 유지한다. 새로운 구조의 raw 배포/공식 counter/4090 시간은 따로 측정해야 한다.

## 6. 비용이 작은 별도 후보

같은 FRESH continuation의 마지막 두 예정 checkpoint(step5710,6852)의 **파라미터를 1:1 평균**한 단일 모델을 한 번 평가하는 방법도 있다. 이것은 기존 QREFINE+FRESH의 출력 평균과 다르며 inference에는 한 모델만 사용한다. Fixed-BN buffers가 실제 같은지 먼저 확인하고, 다르면 임의 평균/검증 데이터 재보정으로 처리하지 않는다. Integer buffers는 평균하지 않는다.

이 평균은 아직 계산·평가하지 않았다. 두 checkpoint가 같은 학습 궤적에 있다는 이유만으로 좋은 결과를 보장하지 않는다. 같은 V0 결과를 이미 본 상태에서 정한 후속이므로 독립 검증으로 표현하지 않는다. 비율/조합 스윕으로 확대하지 않는 저비용 비교이지 큰 개선을 기대하는 주 구조안은 아니다.

## 7. 현재 결론

다음 주 구조 후보는 planner의 시점별 visual motion read다. 큰 성능 개선을 확약할 근거는 없다. 현재 보존된 단일 DEV 최저는 선택된 step5710의 0.153684060, QREFINE과 고정1:1 출력 평균은 0.148555967이다. 서버 점수/새 FULL 결과로 환산하지 않는다. 이번 턴은 관측 및 제안 기록만 완료했으며 새 학습·FULL·제출을 시작하지 않았다.
