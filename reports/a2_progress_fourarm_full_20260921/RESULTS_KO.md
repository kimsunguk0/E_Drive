# P-CTRL FULL stage2 완료와 제출 패키지

2026-09-21 갱신. DEV의 [네 실험 terminal 비교](../a2_progress_fourarm_20260921/RESULTS_KO.md)에서 선택한 CTRL 레시피를 기존 FULL 부모에 적용했다.

- 모델: P-CTRL-FULL-STAGE2-s1
- 학습: 전체 공식 train 376 unique scenes / 101,520행, 추가 4,156 update 완료
- 원래 FULL 부모: A2-H4-PROGRESS-FULL-s1, 공식 서버 PREFIX 0.1336848279459137
- 최종 checkpoint SHA-256: fe821be8546cc54db27c3a81e57bc04126461b29d1a74732e3ad308953c3a3a2
- 로컬 제출 패키지: /home/a/Downloads/P-CTRL-FULL-STAGE2-s1_submission_20260921/
- 공식 업로드: 미실행. 새 모델의 서버 성능은 미측정

Raw 8-clip parity, 전체 test 1,125 clip 추론, strict portable export 및 실제 최종 가중치의 RTX4090 Docker 검사를 완료했다.
원래 FULL 가중치와 제출 패키지는 보존했다. V0는 FULL 학습에 포함되므로 이 모델의 로컬 V0 값은 in-fit 진단이며, 미관측 성능이나 서버 성능 예측으로 사용하지 않는다.

|검사|결과|원본 기록|
|---|---|---|
|B200 패키징 완료|1,125 clips, portable exact|[completion_receipt.json](completion_receipt.json)|
|Raw adapter parity|완료|[raw_parity.json](raw_parity.json)|
|전체 clip 추론 검수|완료|[inference_validation_summary.json](inference_validation_summary.json)|
|FLOPs|730,044,861,120|[flops.json](flops.json)|
|RTX4090 GPU Docker|실제 최종 가중치, raw train fixture 2개 통과|[RTX4090_validation.json](RTX4090_validation.json)|
|4090 전체 B1 model forward|두 fixture 중 큰 median 26.137ms, p95 26.198ms|동일 RTX4090 기록|
|로컬 전달|완료|[LOCAL_DELIVERY.json](LOCAL_DELIVERY.json)|

시간 측정은 BF16 이미지 경로와 FP32 planner를 포함한 student 전체 forward이며 전처리는 제외했다.
2개 fixture 측정의 범위를 넘어 전체 데이터 latency나 공식 서버 elapsed_ms와 동일하다고 주장하지 않는다.

completion_receipt.json 내부의 4090 pending 문구는 B200 패키징 종료 시점 snapshot이다.
그 이후 완료된 장치 검사와 전달 상태는 RTX4090_validation.json 및 LOCAL_DELIVERY.json을 기준으로 한다.
