현재 SDV2 tune 1,998행에 **P7 영상 상태 예측을 그대로 넣는 진단**이 가능합니다. 새 모델 추론·학습 없이 기존 두 seed의 저장 예측을 추출했고, 원예측 및 SDV2 평가 행과 전수 대응시켰습니다.

- b0 입력: `empirical_p7_b0_status.npz`; SHA256 `e2e5d55790c0a8efd4637939e1f896c95620ae9671e845439083cfc35fad273e`
- b1 입력: `empirical_p7_b1_status.npz`; SHA256 `ca38cd5f605906e1f457856cb824867ec4b80224481b5e88480a66ca1ce525cd`
- 두 파일의 key는 `rows: int64[1998]`, `status8: float32[1998,8]`뿐입니다. 앞 네 채널은 정확히 0, 뒤 네 채널은 저장된 `pred_state[:,:4]`와 bitwise 동일한 `vx,vy,ax,ay`입니다. 행 SHA는 `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`입니다.

원본은 원격 `/NHNHOME/data/sukim/adcl/reports/p7_c_state_sufficiency_20260908_ops/pred_b{0,1}_tune.pt`입니다. b0 checkpoint SHA는 `6145255ab2b1efd3deb169b873896e9b1b623ca49f11e5a46a71b138e3b0355e`, b1은 `8ffb429b17820b63477987f5c7de151dfa9e0e2b12378e247aed0b0e41de2563`입니다. 원 저장 추론 정밀도는 `bf16_encoder_fp32_heads`이고 전체 forward의 결과입니다. 원본 manifest의 artifact SHA와 로컬 원본 파일을 검증했습니다. 현재는 checkpoint를 열거나 실행하지 않았습니다.

P7의 `model.py:142`에서 motion 분기는 pose 정렬 및 goal 사용 전에 실행됩니다. `motion_encoder.py`에는 pose·goal·provided-status 인자가 없으며, 현재/과거 front image 특징과 nominal time gap만 받습니다. 출력 `state_hat`에는 이미 물리 단위 scale이 적용되어 있어 추가 정규화·역정규화가 필요하지 않습니다. 보존한 두 소스의 SHA는 P7 원 추론 manifest와 일치합니다. 다만 이 예측은 기존 MotionDrive의 보조 상태 head이며, 새 SDV2 영상 상태 모델의 결과가 아닙니다.

**시간 정의 차이를 보정하지 않았습니다.** 좌표는 양쪽 모두 현재 ego의 full SE(3), x-forward/y-left이며 단위는 m/s와 m/s²입니다. P7 감독 GT는 실제 timestamp에서 과거 ≤1.001초의 pose를 선택한 자유 절편 quadratic fit입니다. SDV2 기준은 정확히 −10..0의 11개 pose를 nominal 10Hz로 fitting합니다. P7 모델의 영상 time 입력도 nominal이었지만, 감독 label은 raw-time이었습니다. 아래 오차는 float64로 다시 계산했습니다.

| 비교 | vx MAE (m/s) | vy MAE (m/s) | ax MAE (m/s²) | ay MAE (m/s²) |
|---|---:|---:|---:|---:|
| P7 raw-time label − SDV2 nominal 기준 | 0.008850 | 0.000239 | 0.016267 | 0.000449 |
| P7 b0 예측 − SDV2 기준 | 0.983984 | 0.036834 | 0.280207 | 0.084016 |
| P7 b1 예측 − SDV2 기준 | 0.973687 | 0.036482 | 0.271641 | 0.083352 |

따라서 canonical 입력은 **원예측의 literal swap**입니다. `SDV2 기준 + (예측 − P7 GT)`는 행별 label 차이를 이용한 가상 보정이므로 literal swap과 다릅니다. 그런 보정 입력은 생성하지 않았습니다. 별도 `empirical_status_residuals.npz`에는 `예측−SDV2 기준`, `예측−P7 GT`, `P7 GT−SDV2 기준`을 구분해 저장했습니다. 정확한 분해 항등식도 전수 확인했습니다.

오차는 독립 백색 노이즈와 다릅니다. 같은 scene의 연속 0.5초 간격 1,961쌍에서 b0 잔차 상관은 `[0.9394, 0.6489, 0.8614, 0.6874]`, b1은 `[0.9381, 0.6411, 0.8624, 0.6933]`입니다. 이는 단순 관측 상관으로, OU 과정이나 특정 인과적 오류 모델의 적합을 증명하지는 않습니다. b0의 vx 절대오차 p95는 2.6507m/s이며, 기준 속도 20m/s 이상 138행의 vx MAE는 1.9175m/s, 정지 0.2m/s 미만 123행에서는 1.1344m/s입니다. 전체 MAE만 맞춘 노이즈는 이러한 조건부 오류와 긴 상관을 놓칠 수 있습니다.

`empirical_status_inputs.npz`에는 세부 상태 및 `scene/session/frame/raw_timestamp_ms/session_raw_time_s/nominal_time_s`를 함께 보존했습니다. 37개 scene·11개 session은 frozen grouped split과 정확히 같습니다. timestamp parquet 37개를 실제 원격에서 읽고 split 및 status overlay의 source SHA와 모두 비교했습니다. raw timestamp는 원래 float64 millisecond 값입니다. `session_raw_time_s`는 해당 session의 최소 scene frame0을 원점으로 뺀 실제 시간입니다. 여기의 `nominal_time_s`는 scene frame0 차이를 0.1초 격자에 반올림한 뒤 `frame/10`을 더한 별도 명시적 레시피이며, 파일명 시각을 파싱한 시간이 아닙니다. raw 상대시간과 최대 차이는 0.001164초입니다. runner가 이미 정한 nominal 시각과 혼용하지 말고, 동일 진단 안에서는 한 레시피를 유지해야 합니다.

행 순서·dtype·유효값·앞4zero·뒤4 원예측 bitwise 대응, 동일 scene/frame/session, 원본 SHA, overlay SHA, raw timestamp source SHA를 확인했습니다. 모델 입력용 canonical 파일에는 목표 경로·goal·GT 상태·잔차가 없습니다. 별도 진단용 원본 GT pack에는 future 배열이 포함되지만, 이번 상태 통계·모델 입력에는 사용하지 않았고 출력하지도 않았습니다. held/reserve 자료는 열지 않았습니다.

이 자료로 얻을 결과는 **고정된 SDV2에 기존 P7 상태 추정기를 이식했을 때의 민감도**입니다. 성능이 악화되더라도 predicted-status로 다시 학습한 SDV2나 더 나은 영상 추정기의 한계로 일반화하면 안 됩니다. 성능이 유지되더라도 P7 추론의 추가 지연과 전체 입력 규정 검증을 대신하지 않습니다. 자세한 파일·배열 SHA, seed별/속도별/session별 통계는 `EMPIRICAL_STATUS_MANIFEST.json`에 있습니다.
