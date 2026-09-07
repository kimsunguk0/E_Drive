# P2 — rear 기하 × 배포 시간 입력 통제 수정

2026-09-07. P2 결과를 보기 전에 실행 조건을 고정한다. P1은 LAST6000 원 조건으로
완주했고 독립 재평가 중이다. 새로운 geometry edition 검사와 배포 입력 정책 테스트가
통과하기 전에는 이 계획을 발사하지 않는다.

## 질문과 네 조건

이번 C/T는 P1의 G/S와 다르다. **네 판 모두 goal ON / image-inferred status ON**이다.
바꾸는 것은 rear 영상과 투영행렬의 정합 C, 그리고 원 timestamp 대신 테스트에서
사용 가능한 nominal 시간 입력 T뿐이다.

| GPU | run | rear geometry | 모델 time_offsets |
|---|---|---|---|
| 0 | p2_c0t0_s0 | 원본 오류 보존 | raw |
| 1 | p2_c1t0_s0 | 실제 bottom-crop 캐시와 정합 | raw |
| 2 | p2_c0t1_s0 | 원본 오류 보존 | nominal [.1,.2,.5,1.] |
| 3 | p2_c1t1_s0 | 수정 | nominal [.1,.2,.5,1.] |

C0/raw는 추가 학습 자체의 효과를 분리하는 대조군이다. 오류/미제공 입력을 유지하는
조건이 좋은 점수를 얻어도 제출 구조로 채택하지 않는다. 올바른 최종 입력 계약은 C1/T1이다.
nominal은 motion과 scene의 시간 입력 양쪽에 영향을 줄 수 있으며, GT/상태 라벨은 바꾸지 않는다.

## 공통 초기값과 일정

`work_dirs/motiondrive_v2/p1_g1s1_s0/last.pth`, step6000,
SHA256 `e2a0e2dc9f3320f5e9744e3e4ef0b77bf9dffffee33aed1274b610bd6faac80e`.
팔별 BEST를 섞거나 seed를 바꾸지 않는다. P1 G1S1은 주력의 goal+영상상태 경로이며,
S의 확정적 우월성을 주장하여 고른 것은 아니다. 원 P0/P1의 잘못된 rear 기하 학습 이력도
공통으로 고정된다. 그러므로 이는 **오류 수정 후 적응 실험**이지 수정된 P0부터의 완전 재학습이 아니다.

- weights-only 초기화, 새 AdamW. 네 판 initial tensor SHA 동일 확인.
- 동일 train203/72세션/54810frame, tune37/11세션/1998frame, 기존 rawtime split.
- 3000step, batch16/eval4/workers4, seed0, bf16 + 기존 FP32 최종 head.
- head/FPN LR5e-5, backbone5e-6, wd.01, clip5, warmup100+cosine3000.
- BN statistics fixed, 모든 가중치는 학습. architecture/low_feature/scale(10,5) 유지.
- loss는 P1과 동일: D3 + .2 footprint + .2 lane + .2 motion, 기존 uncertainty 항.
- 250step마다 full tune D3 평가, 500step마다 LAST 저장. 정확한 공통 sample-order SHA 대조.
- 새 supervision edition의 **모든 scene 배열/JSON은 원본과 bitwise 동일**해야 한다.
  달라지는 것은 rear canonical 행렬과 전역 provenance뿐이다. timestamp/GT 자체를 수정하지 않는다.

생성 완료 edition 고정값:
canonical SHA `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961`,
manifest SHA `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93`.
원본 manifest SHA `073d9d2eb89956cac5c3d78b91622f5e397cfa116b1ed8f9a9f639ec3003a8d0`.

실행 JSON: `configs/motiondrive_v2/p2_geometry_time_r1_s0.json`.
추가 학습량은 약48000표본(작은 마지막 batch 제외 방식 그대로), 약0.88 train epoch다.
판정 중 조기 종료하거나 결과를 보고 일정·loss·선택식을 변경하지 않는다.

## 결과를 읽는 순서

1. 실제 OS 종료0, 비유한 값0, checkpoint 유한성, 공통 initialization 및 sample order부터 검증.
2. **LAST3000 주표**에서 C 효과(T 고정), T 효과(C 고정), 상호작용을 계산한다.
   네 조건은 각각의 명시된 입력 정책으로 평가하며 원 P1 raw 전용 G×S 분석기에 넣지 않는다.
3. 같은11 rawtime 세션을 동시에 재표집하는 paired bootstrap10000회, seed20260907.
   frame가중 평균을 주 지표, 세션 동일가중을 보조 지표로 한다. 단일seed·반복tune 한계 명시.
4. 팔별 동일 tune-D3 BEST는 보조표다. LAST 요인효과와 섞지 않는다.
5. C1/T1의 normal 및 영상 교란 평가, 시간별/종횡/stop·accel 오차를 보고 다음 학습 처방을 정한다.
   개선이 없으면 구현 정합은 유지한 채 표현/운동추정 병목을 다시 분해한다.
6. 3090에는 C1/T1 배포 계약을 보존한 실제 모델을 옮겨 전체 forward와 clip adapter를 검증한다.

P1의 독립 정상 replay를 먼저 마치고 그동안 공유 model/trainer/data 소스는 바꾸지 않는다.
GPU0–3만 사용하며 final-val136/test/기존 타인 작업은 계속 보존한다.
점수 하락을 기대한다는 것과 1등 경쟁력을 입증했다는 것은 별개다.

## 17:50 KST 운영상 분리 — 결과 관측 전

4판 발사 전 사전 검사에서 GPU0의 별도 `alpamayo` 벤치마크(PID1962093,
17:48 시작, 약138GB)를 확인했다. 이 프로세스에는 signal을 보내지 않았고
launcher는 학습/출력 디렉터리 생성 전에 원4판 발사를 거절했다.

비어 있는 GPU1–3은 `p2_geometry_time_r1_s0_gpus123.json`으로 먼저 시작하고,
GPU0 조건은 유휴 확인 후 `p2_geometry_time_r1_s0_gpu0.json`으로 시작한다.
두 JSON은 원 계획 jobs의 **그대로인 부분집합**이며 이름/GPU/학습 인자는 바꾸지 않는다.
초기 가중치·데이터 순서·노출 수·평가 기준은 동일하고 시작 시각만 다르다.
소스가 바뀐 경우 후발 조건을 조용히 섞지 않고 core file SHA를 다시 검토한다.
GPU 간 wall-clock 학습 시간은 이 결과에서 모델 효율 지표로 비교하지 않는다.

17:52 추가: 별도 벤치마크가17:50:55에 GPU0–3 전체로 확장되어 subset123 발사도
사전 검사에서 거절됐다. coordinator1963557, compute children1963560–1963563을
프로세스 정보로 확인했다. **P2 trainer/launch artifact는 아직 생성되지 않았다.**
GPU0 자동 발사도 취소했고, 가용 자원은 읽기 전용으로 관찰한다.
네 조건 모두 미실행이므로 전체 유휴 확인 후 원4판 계획으로 발사할 수 있다.
타 작업 중지나 GPU4 이상 사용을 임의로 결정하지 않는다. 그동안 CPU 배포 입력 검증을 진행한다.
