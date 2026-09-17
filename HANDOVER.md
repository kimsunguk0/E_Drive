# MotionDrive V2 — 인수인계 (2026-09-17)

새 세션이 이 문서 하나로 이어받을 수 있게 쓴다. 최상위 색인이다.

## 0A. 2026-09-17 후속 교정 — 아래의 오래된 수치보다 우선한다

- 공개 leaderboard `mode=best` 재확인 기준 현재 제출은 14위이고 3위는 `0.1305365832`다.
  아래의 8위 및 3위 `0.14664` 표기는 당시 불완전한 snapshot이다.
- split의 `historical_val` 9 scene은 `val` 29 scene에 전부 포함된다. 전체 자료는 385가 아니라
  **376 unique scene / 101,520 stride-1 rows**이며 동일 노출량 terminal은 **24,931 update**다.
- `OPEN_ISSUE.md` 질문 10에는 답변이 있다. 영상에서 직접 추론한 ego history/status를
  planner가 사용하는 것은 허용된다. raw provided history/status 입력과 구분한다.
- 사용자는 서버 응답의 `elapsed_ms`는 채점 harness 시간이라며 이번 모델 판단에서 제외하도록
  지시했다. 아래 미해결 문단을 다시 실험 우선순위로 올리지 않는다.
- 최신 공격안은 `reports/md_progress_residual_20260917/EXECUTION_REVIEW_KO.md`를 따른다.
  FULL과 DEV를 분리하고, FRONT/SIDE temporal + detached-base-plan-conditioned neural progress
  residual을 비교한다. C/raw-status selector와 준비되지 않은 recurrent BEV는 제출 계보에서 제외한다.

---

## 0. 가장 먼저 알아야 할 것 — 2차 제출 결과와 교정

**2026-09-17 제출 (MR-NATIVE-s1): 리더보드 L2_avg = 0.19798776670488366, 8위.**

| | L2_1s | L2_2s | L2_3s | **L2_avg** |
|---|---:|---:|---:|---:|
| 서버 실측 | 0.113672 | 0.196344 | 0.283947 | **0.197988** |
| 우리 V0 plain | 0.110316 | 0.191025 | 0.271663 | **0.191002** |
| 차이 | −2.95% | −2.71% | −4.33% | **−3.53%** |

### ⚠️ 가장 중요한 발견 — test-matched 재가중을 쓰지 마라

| 추정기 | 예측 | 서버 대비 |
|---|---:|---:|
| **V0 plain** | 0.191002 | **−3.53%** ✅ |
| V0 test-matched 재가중 | 0.165804 | **−16.26%** ❌ |

직전 세션에서 내가 test-matched 재가중을 "리더보드 예측기"로 권했다. **틀렸다.**
plain V0가 훨씬 정확하다. 재가중은 2026-09-01에 실격된 `v2a_goal` 그래프에서
0.236 수준에 맞춰 만들어진 것이고, 우리 그래프·0.17 수준으로 **이전되지 않았다**.
(effective_n이 888/1998로 낮고 w_max 8.33으로 극단적이었던 것이 징후였다.)

**앞으로 규칙**: 판정도 후보 선택도 **plain V0**로 한다.
`reports/md_exp_diagnosis_20260915/test_matched_weighting.json`은 기록으로 남기되
의사결정에 쓰지 않는다. `analyze_mr_round3.py`가 두 지표를 모두 출력하는데,
**plain 열만 본다.**

### 실용 환산식

서버 ≈ V0_plain + 0.0070 (또는 × 1.0366). 단일 관측 1점에서 나온 값이다.

| 목표 | 필요한 V0_plain |
|---|---:|
| 0.15 (목표) | ≈ **0.1430** |
| 0.17421 (7위) | ≈ 0.1672 |
| 0.14664 (3위) | ≈ 0.1396 |
| 0.12803 (1위) | ≈ 0.1210 |

현재 최고 V0_plain이 0.190515(MR-ADJ1)이므로 **0.15까지 약 −0.048, 25% 추가 감소**가
필요하다. 짧은 거리가 아니다.

### ⚠️ 미해결 — `elapsed_ms: 310`

응답에 `"elapsed_ms": 310`이 있다. 우리 4090 실측은 clip당 24.27–24.52 ms다.
이게 만약 T_infer라면 시간 페널티 `×(1 + (310−100)/200) = ×2.05`가 붙어 실효점수가
0.406이 된다. **다만 표시된 점수 0.19798776670488366은 L2_avg와 정확히 같아서
페널티가 적용돼 있지 않다.** `"cutoff": true`는 FLOPs 컷오프 통과로 보인다.

**다음 세션에서 확인할 것**: elapsed_ms가 무엇의 시간인지(clip당? 전체 배치?
harness 왕복?), 시간 페널티가 최종 순위에 반영되는지. 함께 온 `runtime` 블록
(`pull_mode: true, num_inflight: 1, want: 31, pull_interval_sec: 10`)을 보면
polling harness의 값일 가능성이 있다. **확인 전까지 "시간 페널티 ×1.0"이라고 단정하지 마라.**

### 제출 잔여

**5회 중 2회 사용 → 3회 남음.**
1차 2026-09-01 `v2a_goal` 0.236239 / 2차 2026-09-17 `MR-NATIVE-s1` 0.197988.
개선폭 −16.2%.

---

## 1. 어디에 무엇이 있나

| 위치 | 내용 |
|---|---|
| **B200** (주 작업) | `ssh -i ./<B200-KEY> -p 42101 <B200-USER>@<B200>` → `/NHNHOME/data/sukim/adcl` |
| **H200** (구 프로젝트) | `ssh e2e` → `/home/pm97/workspace/sukim/adcl` |
| **미러** | B200의 `~/edrive_mirror` → GitHub `kimsunguk0/E_Drive`, 브랜치 `motiondrive-v2-20260910` |
| 로컬 제출물 | `~/Downloads/MR-NATIVE-s1_submission_20260917/` |

### 커밋·푸시 절차 (반드시 지킬 것)

작업 저장소를 **직접 push하면 안 된다** — 히스토리에 B200 접속 정보가 있고 저장소는 공개다.

```bash
# 1) 작업 저장소에서 커밋 (메시지는 파일로 전달; 인라인 heredoc은 따옴표 때문에 깨진다)
cd /NHNHOME/data/sukim/adcl && git commit -F /tmp/msg.txt

# 2) 미러로 cherry-pick
cd ~/edrive_mirror && git fetch /NHNHOME/data/sukim/adcl main:refs/remotes/work/main --force
git cherry-pick <sha>

# 3) 가드 grep — 트리와 커밋 메시지 둘 다 0이어야 함
git grep -nIE "<B200-IP>|<B200-USER>|<B200-KEY>" HEAD
git log --format=%B -1 | grep -cIE "<B200-IP>|<B200-USER>|<B200-KEY>"

# 4) push (remote 이름은 origin이 아니라 github)
git push github HEAD:motiondrive-v2-20260910
```

미러 SHA는 설계상 로컬과 다르다. GitHub push가 간헐적으로 실패하니 2–3회 재시도.

---

## 2. 현재 성능 상태 (전부 plain V0)

| run | V0 plain | 비고 |
|---|---:|---|
| R0 (기준선) | 0.258544 | |
| E1-EXP | 0.225592 | 배포 fallback, 산출물 완비 |
| LEN-s0 | 0.223644 | |
| MR-LOWDETAIL-s0 | 0.193763 | |
| MR-NATIVE-s0 | 0.191892 | |
| MR-NATIVE-s1 | 0.191002 | **← 2차 제출, 서버 0.197988** |
| MR-W64-s0 | 0.190892 | |
| **MR-ADJ1-s0** | **0.190515** | 현재 최저 |
| MR-ADJ0-s0 | 0.191743 | ADJ 대조군 |
| MR-NATIVE-LONG-s0 | 0.194655 | 더 긴 일정, 나쁨 |

**다섯 MR arm이 0.0014 안에 몰려 있다. 이 안에서 후보를 고르는 것은 잡음 고르기다.**
seed 산포(s0 대 s1)가 0.000890으로 arm 간 차이와 같은 크기다.

---

## 3. 무엇이 통했고 무엇이 안 통했나

| 변경 | 결과 |
|---|---|
| **데이터 확대 203 → 310 scene** | **−0.033, 복제됨** ✅ |
| **matching graph (radius 2→4, native 768×432 canvas)** | **−0.033, 두 seed 복제** ✅ |
| native detail 대 lowdetail | 미확정 (CI 0 포함). readout은 −28% |
| 길이 auxiliary λ=0.25 | 작음 |
| descriptor 32 → 64 (W64) | 효과 없음 |
| seed weight average | 효과 없음 (두 seed보다 나쁨) |
| 인접 edge (ADJ) | 효과 없음 |
| 더 긴 일정 (8 exposure) | 효과 없음 (오히려 나쁨) |

**통한 건 둘뿐 — 데이터를 늘리는 것, matching 격자를 세밀하게 하는 것.**
모듈 추가·폭 확대·연산 증가 계열은 네 번 연속 0이었다. 이 패턴이 강하다.

### 더 긴 학습이 안 되는 이유 (반복하지 말 것)

같은 3.93 exposure에서 LONG이 0.214189, 짧은 일정이 0.191892다. 차이는 데이터도
구조도 아니고 **cosine LR 감쇠가 예산 안에서 끝나느냐**다. horizon을 2배로 늘리면
같은 지점에서 LR이 훨씬 높고 끝까지 가도 회복하지 못한다. **조기 종료 질문은 닫혔다.**
현재 정지점(3.93 exposure = 20,554 update)이 옳다.

---

## 4. 다음 단계 — 전체 set 재학습

점수를 보기 **전에** 정해 둔 다음 후보다. 리더보드를 보고 바꾼 것이 아니다.

지금 held-out으로 놀리는 것: tune37(V0) + H29 + hist9 = **75 scene**.
train 310 → 385 scene(**+24%**). 통한 두 가지 중 하나가 데이터 확대이므로 근거가 있다.

### 실행 시 주의

1. **정지점은 exposure로 이전한다.** 현재 20,554 update = 3.93 exposure of 83,700행.
   385 scene이면 행이 약 103,900개가 되므로 같은 3.93 exposure는 약 **25,520 update**다.
   cosine horizon을 그 총량으로 **처음부터** 설정해야 한다(나중에 이어 붙이면 안 됨).
2. **V0를 학습에 넣으면 교정용 held-out이 사라진다.** 그러면 서버 점수만이 유일한 신호다.
   위 환산식(V0_plain + 0.0070)은 그때 못 쓴다.
3. **H를 넣으려면** `H_PREREGISTRATION_KO.md`에 revision 3을 먼저 써야 한다.
   지금까지 revision 1(후보 목록 재동결), 2(s0→s1)가 있다. H는 아직 한 번도 열리지 않았다.

### 대안으로 검토할 만한 것 (아직 안 해봄)

- **H6-NEAR / H6-LONG** — 관측 시점 확대. 기존 H4(−0.1/−0.2/−0.5/−1.0)에
  −0.3/−0.4를 더하거나(NEAR) −2.0/−2.5를 더한다(LONG). 기존 근접 관측을 교체하지 말 것.
- matching 격자를 **더** 세밀하게 (radius 4 → 6/8). radius 2→4가 통했으므로
  같은 축의 연장이다. 다만 W64가 실패한 것을 보면 "무조건 더"가 통하지는 않는다.
- 3초 시점 오차가 서버에서 상대적으로 더 나쁘다(−4.33% 대 1s의 −2.95%).
  후반 horizon을 표적으로 하는 실험이 근거가 있다.

---

## 5. 함정 모음 (내가 실제로 밟은 것들)

### 제출물
- **`submission.json`에 `__flops__`가 반드시 있어야 한다.** 최상위 키로 정수. 없으면
  7,053 GFLOPs 컷오프 판정 자체가 불가하고 누락은 `[0,0]` 처리된다.
  앞서 만든 `official_test_MR-NATIVE-s0.json`은 1,125키로 **이 키가 없다 — 업로드 불가**.
- `__flops__`는 주최측과 **같은 counter**로 재야 한다:
  `torch.utils.flop_counter.FlopCounterMode`, Global 합, 1회 forward.
  `torch.profiler`의 with_flops와 값이 다르다(729.8 G 대 1,119.9 G). **판정 기준은 전자.**
  세기 전에 `torch.utils.module_tracker.register_multi_grad_hook`을 무력화해야 한다
  (안 하면 `no_grad`에서 grad_fn assert로 죽는다). 주최측 스크립트도 그렇게 한다.
- zip 안에는 `submission.json` **하나만**. 형식은
  `experiments/md_r0_reset_20260914/package_submission.py`가 강제한다(실패 시 포장 거부).
- 모델 출력은 **이미 누적 절대 좌표**다. 예제 스크립트의 `cumsum`을 다시 적용하면
  끝점이 3.50배로 부푼다.

### 환경
- B200 기본 python3에 **cv2가 없다.** raw 입력 경로(제출물 빌드, raw B1 parity, flops)는
  **`~/cv2env/bin/python`**을 써야 한다. (cv2 5.0.0; 기록된 4.8.1.78과 다르지만
  parity가 bitwise로 나와 문제없음을 확인했다.)
- 내부 trainer의 `--gpu`는 **0–3만** 받는다. 물리 GPU 5,6을 쓰려면
  `CUDA_VISIBLE_DEVICES=5 ... --gpu 0`.
- **GPU 4와 7은 다른 사람이 쓰고 있다.** 건드리지 말 것.
- 공식 test clip 1,125개는 `/tmp/etri_test`에 추출돼 있다. **tmpfs라 사라질 수 있다.**
  다시 필요하면 `ls test/*.tar | xargs -P 16 -I{} tar xf {} -C /tmp/etri_test`.
- `evaluate.py`는 그래프를 알아야 한다: MR은 `--mr-detail native`,
  ADJ는 거기에 `--adj off|on`을 더해야 한다(mask는 tensor가 아니라 forward 인자라
  체크포인트에서 추론 불가). split도 명시해야 한다 —
  `--split-manifest .../grouped_split_r0reset_tplus.json --supervision-root .../r0reset_tplus_geometry_v2`.
  기본값은 다른 split이라 "Initialization/resume split lineage mismatch"가 난다.

### 코드 수정
- **`str.replace`로 패치할 때 반드시 `assert s.count(old) == 1`을 넣어라.**
  앵커가 안 맞으면 조용히 no-op이 되고, 나는 그것 때문에 MR arm이 학습 중에
  `KeyError`로 죽는 것을 겪었다. **정규식으로 중괄호를 잡지 마라** — `mr_restore = {`를
  `{, "encoder_forward": None}`으로 망가뜨린 적이 있다. 수정 후 `ast.parse`로 검사.
- `train.py`의 계획 파일은 `experiment_{arm}-s{seed}.json`이다. 예전에는 seed가 없어서
  seed 1 실행이 seed 0 기록을 덮어썼다(9개 전부 run dir 사본에서 복구했다).
- `flip_item`은 아는 키만 뒤집는다. **새 이미지 텐서를 추가하면 flip 래퍼도 추가해야 한다.**
  안 하면 미러링된 샘플에서 라벨이 조용히 어긋난다.
- pyramid **correlation은 각 level 고유 해상도에서** 하고 결과를 나중에 올려야 한다.
  먼저 올려서 correlate하면 coarse level 탐색 범위가 ±64 → ±32 원본 px로 조용히 반토막 난다.
  (ADJ 구현에서 실제로 밟았고 step-zero 차이 0.198로 드러났다.)

---

## 6. 규정 관련

- **실격 판정 받은 구조**: goal이 planner의 cross-attention **query항**으로 들어가는 것
  (2026-08-31, `v2a_goal`). 현재 계보는 그 판정 이후 재작성한 것이다.
  `DirectTrajectoryPlanner.forward`에 goal 인자가 없고, query는 학습된 `waypoint_queries`,
  goal은 공유 scene feature에만 들어간다. `provided_status_used: false`.
- **`OPEN_ISSUE.md`**(작업 저장소 루트)에 주최측 문의·답변 전문이 있다. Q3(지표=ADE1/2/3 평균,
  서버 결과로 확인됨), Q4(T_infer 정의), Q7·Q8(과거 궤적·goal 사용 제한), Q10(미답변).
- 시간 페널티식: `×(1 + (T_ms − 100)/200)`. T ≤ 100 ms면 ×1.0.
  (§0의 `elapsed_ms: 310` 미해결 항목 참조.)

---

## 7. 주요 산출물 위치

전부 `/NHNHOME/data/sukim/adcl/` 아래.

| 파일 | 내용 |
|---|---|
| `reports/md_exp_diagnosis_20260915/MR_ROUND3_RESULTS_KO.md` | ADJ·LONG 판정 (최신 실험 보고) |
| `reports/md_exp_diagnosis_20260915/MR_ROUND2_RESULTS_KO.md` | seed 복제·W64·soup |
| `reports/md_exp_diagnosis_20260915/MR_RESULTS_KO.md` | matching graph 1차 판정 |
| `reports/md_exp_diagnosis_20260915/mr_round3_judgement.json` | 모든 CI·비교 수치 |
| `reports/md_exp_diagnosis_20260915/test_matched_weighting.json` | **기록용. 의사결정에 쓰지 말 것** |
| `reports/md_r0_reset_20260914/H_PREREGISTRATION_KO.md` | H 사용 규칙 + revision 1, 2 |
| `reports/md_r0_reset_20260914/mr_candidate_registry.json` | 후보 동결 (SHA 포함) |
| `reports/md_r0_reset_20260914/submission/2026-09-17_MR-NATIVE-s1/` | **2차 제출물 + README** |
| `reports/md_r0_reset_20260914/mr_flops_s1.json` | `__flops__` 측정 근거 |
| `reports/md_r0_reset_20260914/rtx4090_mr_forward_cost.json` | 4090 시간·FLOPs |
| `experiments/md_r0_reset_20260914/train.py` | 모든 arm 정의·launcher |
| `experiments/md_r0_reset_20260914/analyze_mr_round3.py` | 판정 스크립트 |
| `experiments/md_r0_reset_20260914/package_submission.py` | 제출물 포장·검증 |
| `experiments/md_r0_reset_20260914/measure_flops.py` | `__flops__` 측정 |

---

## 8. 열려 있는 것 / 닫힌 것

**닫혔다** (다시 하지 말 것): descriptor 폭 확대, seed weight average, 인접 edge(ADJ),
더 긴 일정, test-matched 재가중을 예측기로 쓰는 것, 조기 종료 연구.

**열려 있다**: 전체 set 재학습(다음 후보로 확정), H6-NEAR/H6-LONG,
matching radius 추가 확대, 후반 horizon 표적 실험, `elapsed_ms` 해석.

**H는 아직 한 번도 열리지 않았다.** 후보를 바꾸려면 revision을 먼저 쓴다.

**리더보드 점수를 보고 후보를 바꾸면 그 순간부터 test set이 개발 집합이 된다.**
남은 3회를 그렇게 쓰지 말 것.
