# P2 공유 GPU 실행 계획 — 2026-09-07

사용자가 GPU 공동 사용을 명시적으로 허용했다. 범위는 기존 GPU 0–3을 유지한다.
GPU 4–7을 사용하거나 다른 프로젝트의 프로세스를 중단하는 권한으로 확대하지 않는다.
기존 idle-only watch cell273은 19:36:56 KST 종료 확인했고, 그 시점 P2 발사는 0회였다.

## 메모리 근거와 변경 범위

19:36 GPU0–3 free는 각각 25088/22332/27844/22332 MiB였다.
기존 P1 학습은 다른 프로세스가 없던 16:47 관찰에서 GPU당 약 43844 MiB였다
(`MOTIONDRIVE_V2_PROGRESS_20260907.md`). 이것은 프로세스 메모리 관찰이며
PyTorch allocated/reserved peak 측정값은 아니다. 배치16을 그대로 공유 실행하지 않는다.

새 계획 `configs/motiondrive_v2/p2_geometry_time_shared_r1_s0.json`은 이전 계획을
보존하고 별도 파일로 등록한다. C/T treatment, P1 LAST 초기 가중치, 모델, split,
logical batch16, 3000 optimizer steps, optimizer/LR, 평가와 checkpoint 선택은 동일하다.
실행 방식만 모든 팔에 microbatch2를 적용한다. 즉 2개씩 8번 backward 후 clip/AdamW
각 1회다. 전체 logical batch의 GT/mask 분모를 먼저 계산하여 microbatch 손실 기여를
합산한다. raster의 positive/negative class balance도 전체 batch 기준으로 보존한다.
BN은 기존 fixed running statistics다. 미세한 연산 순서/kernel 차이 때문에 기존
unsplit 학습과 bitwise 동일한 trajectory는 약속하지 않는다. 네 팔의 microbatch 정책은 같다.

## 안전 장치

- 명시적 shared plan에만 GPU 공동 사용을 허용한다. 생략 시 이전 idle-only 검사 유지.
- 실행 직전 두 차례 메모리를 조회한다. 각 GPU free ≥ allocator12000 + reserve8192 MiB일 때만 발사.
- 모델의 GPU 할당 전 PyTorch caching allocator를 12000 MiB로 제한한다.
- trainer는 각 microbatch/평가 batch 전에 최소 8192 MiB free를 확인한다.
- supervisor는 5초마다 physical GPU free를 검사한다. 부족 또는 조회 실패 시
  자신이 시작한 정확한 child에만 SIGTERM, 30초 후 미종료이면 그 child에만 kill한다.
  다른 PID/process group에 신호를 보내지 않는다. 안전 중단을 실험 성공으로 기록하지 않는다.
- 실제 GPU free와 allocated/reserved/peak를 학습 로그 및 종료 manifest에 기록한다.

이것은 메모리 독점 예약이나 OOM 불가능 보장이 아니다. PyTorch allocator 밖의 CUDA
메모리와 다른 프로세스의 갑작스러운 증가는 별도다. cap 초과는 우리 프로세스에서도
allocation 오류를 낼 수 있으므로 먼저 microbatch2의 실제 backward/optimizer 계측을 한다.
[PyTorch 공식 allocator 제한 문서](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)

## 실제 발사 순서

1. CPU에서 공유 admission/own-child pressure guard 및 전체-batch 손실·gradient 보존 검증.
2. GPU2에서 `p2_shared_memory_probe_s0` 두 logical steps: 전체 네트워크 backward/AdamW,
   train 균등추출32행/tune 균등추출8행, evalbatch4. 결과는 성능 비교용 P2 데이터가 아니다.
3. 실제 clean exit, 정상 loss/gradient, 메모리 사용과 다른 작업의 생존을 확인한다.
4. 통과 시 P2의 GPU2 C0T1 팔을 먼저 올리고 정상 학습 확인 후 GPU0/1/3 나머지 세 팔 발사.
5. P2 분석에는 `--plan configs/motiondrive_v2/p2_geometry_time_shared_r1_s0.json`과
   두 subset launch record를 함께 입력한다. 구 idle-only plan을 실행했다고 보고하지 않는다.

작성 시 GPU 실험은 아직 시작 전이다. 실제 측정/프로세스/exit는 별도 실행 기록으로 확인한다.
최종 val136/test와 공식 제출은 접근하지 않는다.

## 실행 결과 추가 — 19:55 KST

구현 커밋 `c6845fb33e548462f0fecead8baa1ec259d10d4a`, B200 CPU276 tests PASS.
두-step canary는 19:50:42 시작, 19:51:03 supervisor가 실제 종료0을 기록했다.
Peak allocated5076.996 MiB / reserved5668 MiB, 관측 최저 free19884 MiB,
pressure_event 없음. AdamW 상태 생성 이후 두 번째 backward와 eval 전환까지 통과했다.
`--max-*-samples`는 데이터셋 전체에서 linspace 균등 추출이다. 초기 계획의
영문 description에 남아 있는 'First32/8' 표현은 부정확하며 실제 선택은 위와 같다.
실행에 사용한 plan 파일은 기록 보존을 위해 변경하지 않았다.

P2 GPU2는 19:51:38, GPU0/1/3은 19:53:12에 순차 발사했다.

| GPU | 팔 | 실제 trainer PID | 19:55 관측 step | 초기 full-tune D3 |
|---|---|---:|---:|---:|
| 0 | C0T0 | 2008118 | 80 | 0.3720431819 |
| 1 | C1T0 | 2008121 | 80 | 0.9895546994 |
| 2 | C0T1 | 2007362 | 190 | 0.3720592221 |
| 3 | C1T1 | 2008122 | 80 | 0.9896259228 |

네 팔의 초기 tensor SHA는 모두
`e3823058037c5aa4422ff3f7900cbcf7bf40374fd04dec1d4391d1572bf8e0ba`다.
기존 raw/nominal full-tune 기준점을 재현했다. C1의 초기 악화는 같은 가중치에서
rear geometry를 바꾼 직후의 관측이며, 보정 적응이 필요한 정도를 보여준다.
아직 회복·개선·우승 가능성이나 원인 자체를 단정하지 않는다. LAST3000이 판정 기준이다.
네 팔 모두 학습 중, pressure_event 없음, 기존 다른 프로젝트의 다섯 PID도 생존했다.
`reports/p2_shared_gpu_startup_20260907.json`에 원본 아티팩트 SHA와 관측을 기록했다.
