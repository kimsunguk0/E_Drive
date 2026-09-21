# H4-PROGRESS FULL 실행 및 제출 준비

사용자의 2026-09-21 요청에 따라 현재 최고 단일 DEV 후보 A2-H4-PROGRESS(0.15117885989032362)를 FULL로 학습한다. 기존 저장 예측 3종 평균 0.146197790은 다른 입력 정책과 배포 미검증 상태이므로 이번 단일 FULL의 레시피가 아니다. DEV에서 일반 주행이 matched DIRECT보다 악화한 한계는 그대로 기록한다.

GPU 0에서 08:44:51 KST 시작. 공개 nuImages trunk 초기값에서 전체 376 scene / 101,520행 / 24,931 update를 학습한다. 기존 DEV terminal에 추가 update를 붙이지 않는다. 같은 graph·loss·LR·seed·effective batch 16·microbatch 8을 유지하고 데이터 행과 전체 schedule만 FULL 노출량으로 옮겼다. 중간 평가는 in-fit 진단이며 최종 24,931 checkpoint를 고정 사용한다.

이미지에서 소비하는 5시점 [-10,-5,-2,-1,0] pose만 nominal status producer에 사용한다. Status는 기존 A2 공통 scene query를 조건화하며 motion/state/history에는 직접 넣지 않는다. 입력의 시간 coverage와 query-only 구조를 확인한 것이며, 운영국의 개별 구조 승인이나 서버 점수 보장을 뜻하지 않는다.

완료된 사전 검사:

- FULL train/tune/val 101,520행, 376 unique scene과 status cache 검증. 기존 DEV train/tune의 동일 producer 결과도 일치.
- 실제 FULL 2-update smoke. Initial state SHA가 선택한 DEV의 초기값과 같고 nonfinite 0.
- DEV 가중치로 raw 8 training fixture 검사. 모든 입력과 FP32/BF16 경로 출력이 cached 입력과 정확히 일치.
- 전체 forward의 torch FLOPs 730,044,861,120. 신규 graph로 측정했으며 이전 MR 값을 승계하지 않음.
- Portable source의 B200 FP32 출력은 원본과 정확히 일치. 격리한 Docker CPU에서는 입력 tensor hash가 모두 같고 출력 최대 차이 0.000009537 m. 첫 비교의 0.004392 m 차이는 cuDNN TF32를 FP32 reference에서 명시적으로 끈 후 해소했다. BF16 학습과 제출 graph는 그대로다.
- PC의 NVML driver/library mismatch로 RTX4090 시간은 미측정. CPU 정합성이나 B200 시간을 4090 시간으로 설명하지 않는다.

학습 종료 후 finish.py가 자동으로 다음을 실행한다.

1. 완료 상태·step·nonfinite·초기값·학습 소스 hash 확인.
2. 실제 FULL terminal로 raw parity와 FLOPs 재검사.
3. 1,125개 raw test clip B1 추론, shape/finite/pose-time coverage/clip isolation 확인.
4. submission.json 하나를 포함한 submission.zip 생성. Absolute XY에 serving cumsum을 추가하지 않음.
5. 실제 FULL 가중치·portable source·Dockerfile·train fixture 2개를 별도 reproduction 폴더에 보존하고 재현 확인.
6. 사용자 PC의 Downloads/A2-H4-PROGRESS-FULL-s1_submission_20260921로 자동 복사. ZIP checksum과 동일 FULL 가중치의 clean-container CPU 출력을 다시 확인.
7. 완료 결과를 HANDOVER와 보고서에 기록하고 공개 미러에 커밋/푸시. PC가 연결되지 않으면 원격 패키지는 보존하고 복사 미완료를 구분.

공식 계정 업로드를 수행하는 코드는 없다. 생성·복사·검증과 실제 공식 제출/채점은 구분하며, 이 파이프라인은 제출 횟수를 소비하지 않는다. 제출할 파일은 완료 후 만들어지는 submission.zip 한 개다. 기존 MR/A2 패키지는 덮어쓰지 않는다.

상태 파일: reports/a2_progress_full_20260921/runtime/finish_status.json, completion.json. PC 상태 파일: /home/a/a2_progress_full_20260921/local_delivery_state.json.

이번 실험은 FULL→DEV 가중치·teacher·feature·통계 재사용을 하지 않는다. GPU 4–7의 다른 작업을 건드리지 않는다. 자동 추가 학습, 후보 ensemble 탐색, 복수 제출은 없다.
