# H4-PROGRESS 네 arm 실행 계약

사용자 제공 `A2_PROGRESS_FourArm_Execution_Order_20260921.md`를 실행한다.
GPU0 CONTROL / GPU1 VECTOR / GPU2 FINE / GPU3 SHARED768. 다른 GPU 작업은 건드리지 않는다.

DEV 부모는 A2-H4-PROGRESS-s1 step20554, bytes SHA
`18128b1af6624333a00baa493a0601459804e8dfb660bf4dd7dbf60f381f2958`이다.
모든 기존 tensor를 그대로 읽고, FINE의 신규 module 및 계산 경로를 검증하는 buffer만 추가한다.
FULL0.1336848279459137은 보존하며 DEV에는 반입하지 않는다.

- 3,426 stage update, seed1, batch16/micro8/eval8, fresh AdamW.
- Trunk1e-6, 그 외 및 FINE1e-5, decay0.01, clip5, warmup100, 새로운 cosine horizon.
- 원래 BF16 영상/FP32 planner·loss, fixed BN, flip0.5 및 기존 인지 가중치를 유지한다.
- 0/1142/2284/3426에서 같은 V0 1,998행. Terminal을 주 비교로 고정한다.
- 같은 무증강 train256 probe를 각 평가 때 기록한다. Probe는 성능 선택용 검증이 아니다.
- Parent20554와 stage step을 분리한다. Smoke는 5update 후 폐기하고 본 학습은 부모에서 다시 시작한다.

VECTOR는 원본 common 함수에 vector0.25만 연결한다. LENGTH와 중첩하지 않는다.
FINE은 같은 forward의 원본54×96 correlation map에서24×32를 읽으며 기존12×16 경로는 유지한다.
Production은 명시적 tensor 반환을 사용한다. 새 fine module 파라미터66,560개이며 output projection만0이다.
SHARED768은 motion 전에 계산한 원시 history FPN을 detach 없이 공통 scene에 전달한다.
모든 arm은 하나의 모델이며 제공 status/goal의 입력 경계는 기존 A2와 같다.

명시적 execution config와 persistent digest를 함께 저장한다. 같은 tensor key라도
잘못된 history feature-source 설정은 strict loader가 거부한다. 새 프로세스에서 초기 및
5update checkpoint의 export 출력을 확인한 뒤 본 학습을 시작한다.

초기 검사는 CTRL/VECTOR/FINE이 부모의 FP32·BF16 출력을 정확히 재현했고,
V0 PREFIX0.151178862를 얻었다. SHARED768은 별도768 재인코딩과 정확히 일치했지만
기존384 대비 큰 출력 변화가 생겨 V013.144899였다. 이는 성능 개선 증거가 아니며
새 특징 해상도에 대한 적응 위험으로 기록한다. 좌표는 기존 normalized grid/FOV를 유지한다.

최종 판정은 CONTROL 대비와 부모 대비를 모두 기록한다. 일반 주행 첫1초,
signed 구간 길이·회전 길이 축소·GT 고정 mask의 종횡 오차·인지 지표·세션 집중도를 함께 확인한다.
중간 best와 terminal을 섞지 않고 CI 또는 임의 -0.005 threshold로 작은 이득을 자동 폐기하지 않는다.

DEV 승자만 별도 FULL 복사본 stage2의 후보가 된다. 같은 노출량은4,156update다.
FULL 또는 제출은 시작과 완료를 별도 기록하며 현재 결과 수집기는 새 학습을 기동하지 않는다.
업로드는 자동화하지 않는다. 기존 제출물은 불변 보존한다.

기존 CPU loss 검사와 시제품 비용 근거는 `reports/a2_pro_review_20260921/`를 재사용한다.
본 실행의 실제 상태는 `runtime/orchestrator.json`, 체크는 `smoke_and_reload.json`,
점수는 `results.csv`와 `result_step*.json`, 최종 결정은 `decision.json`에 기록한다.
