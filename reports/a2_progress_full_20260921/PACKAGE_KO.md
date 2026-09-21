# A2-H4-PROGRESS-FULL-s1 제출 준비 완료

DEV 최고 단일 후보 PREFIX 0.151178860의 레시피를 전체 376 scene / 101,520행에서 24,931 update 학습했다. 공개 초기값에서 학습했으며 DEV terminal에 추가 학습한 결과가 아니다. 완료된 FULL의 로컬 평가는 학습 내 진단이고 공식 서버 점수는 아직 없다.

제출 파일: `/NHNHOME/data/sukim/adcl/work_dirs/a2_progress_full_20260921/package/submission.zip`

ZIP 안에는 submission.json 하나만 있다. 1,125개 clip의 absolute XY 6×2와 정수 __flops__를 검수했다. Raw 8 fixture 입력/출력 일치, 전체 test 추론, clip isolation, 동일 FULL 가중치의 portable source 재현을 통과했다. FLOPs는 730,044,861,120이다.

Checkpoint SHA256: dae99f86f29e92296b1d036db3f7933d41bf9a5c3dd3af5153416a8eaa676b14
Submission ZIP SHA256: 3981ee5541458d62d1b3cb94cfe9f67ab619672e892879eec92f8e08e8f0c1cc

Status는 RGB를 실제 소비하는 5시점 [-10,-5,-2,-1,0]으로 만들고 공통 scene query에만 넣는다. Motion/state/history에는 제공 status를 직접 넣지 않는다. 이는 구현 경계 설명이며 개별 운영국 승인 주장이 아니다.

`reproduction/`에는 가중치, portable code, Dockerfile, train fixture 2개와 B200 FP32 기준 출력이 있다. 공식 ZIP에는 이 자료를 넣지 않았다. PC의 NVIDIA 드라이버/NVML 불일치 때문에 RTX4090 시간은 미측정이며, clean-container CPU 정합 검사는 GPU latency 검증이 아니다.

사용자 PC의 Downloads/A2-H4-PROGRESS-FULL-s1_submission_20260921로 자동 복사를 연결했다. 복사 완료 여부는 PC의 LOCAL_DELIVERY.json을 확인한다. 공식 업로드는 실행하지 않았으며 이 작업으로 제출 횟수를 사용하지 않았다.

추가 검증 완료: chi@192.168.10.102의 RTX4090에서 **동일 FULL terminal**의 전체 B1 BF16 forward 26.103ms, p95 26.231ms. 두 train fixture 각각 warmup30/repeat200, 전처리 제외. [측정·FP32 재현](RTX4090_FULL_validation.json). 앞의 4090 미측정 문구는 패키지 생성 당시 상태이며 이번 기록으로 갱신된다. 공식 업로드는 수행하지 않았다.
