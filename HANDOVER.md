# MotionDrive V2 — 인수인계 (2026-09-18)

새 세션이 이 문서 하나로 이어받을 수 있게 쓴다. 최상위 색인이다.

## 0E. 2026-09-18 MR FULL 공식 점수 — 현재 확인된 서버 결과

사용자가 MR과 FULL 두 제출의 공식 채점 응답을 전달했다. FULL은 **0.18596892793122946**으로,
기존 MR **0.19798776670488366** 대비 **0.0120188388 / 6.0705% 개선**됐다.
세 PREFIX 지표 모두 개선: 1s `0.113672→0.105357`, 2s `0.196344→0.184546`,
3s `0.283947→0.268004`. 두 제출의 FLOPs는 `729815613824`, cutoff=true로 같다.

- 두 모델 모두 제공 status 입력이 없는 MR 계열이다. 모델 config·초기 tensor hash·seed가 같다.
- train310/83,700행/20,554 update → FULL376/101,520행/24,931 update.
  약 3.93회 노출을 유지했고, microbatch는 2→8이다. 데이터량만 바꾼 단독 대조는 아니다.
- FULL의 V0 `0.097245`는 학습 행의 in-fit 진단이며 공식 점수로 대체해서 읽지 않는다.
- 아래 0D의 미업로드·미측정 문구는 파일 준비 당시의 기록이다. 현재 두 MR 제출은 사용자 확인 완료다.
- `elapsed_ms`는 응답 원문에 보존하며, 모델 latency 개선으로 해석하지 않는다.
- 현재 status-free 서버 기준은 **0.185969**. 0.15까지 추가 절대 개선 0.035969가 필요하다.
- A2/A3의 서버 점수는 여전히 없다. 이번 FULL 이득을 그 모델에 그대로 외삽하지 않는다.

상세: `reports/md_full_submission_20260918/SERVER_RESULT_20260918_KO.md`.
원문 응답·정확한 산술·설정 대조는 `server_comparison_20260918.json` 및 각 제출 폴더의
`server_result_20260918.json`에 있다. 결과 출처는 사용자 전달이며 서버 API 재조회는 아니다.

## 0D. 2026-09-18 MR FULL 제출 후보 확보 — 당시 준비 기록

사용자의 요청에 따라 **완료된 status-free MR FULL을 재학습 없이 실제 제출 파일로 보존**했다.
추가 학습은 0 update이고 서버 업로드도 하지 않았다.

- 후보: `MR-NATIVE-FULL-s1`, 376 unique scenes / 101,520 rows / terminal step 24,931.
- 모델 구조: 기존 제출 MR과 같은 native MR + direct XY. 제공 status 입력 및 A2/A3 경로 없음.
  Pose의 scene 정렬과 goal의 공통 scene 조건은 기존대로 사용한다.
- **업로드용 파일:** `reports/md_full_submission_20260918/submission/submission.zip`.
  내부는 `submission.json` 하나이며 1,125 clip + 정수 `__flops__`로 구성된다.
- 전 clip 누락·잉여 0, 6×2/유한값 검사 통과, clip 재실행 예측 차이 0.
- FULL의 raw/cache B1 parity: 8 fixture 입력·출력 모두 bitwise 일치.
  기존에 채점된 MR의 첫 공식 clip도 기존 예측과 정확히 재현됐다.
- 공식 counter FLOPs: **729,815,613,824 = 729.816G**, cutoff 7,053G 통과.
- checkpoint 원본과 별도 보존본 해시가 이전 실험 색인의 terminal 해시와 같다.
  보존본: `work_dirs/md_full_submission_20260918/preserved/ckpt_step24931.pth`.
- 로컬 사본: `~/Downloads/MR-NATIVE-FULL-s1_submission_20260918/`.
- **FULL의 공식 점수는 아직 없다.** V0 0.097245는 학습 행의 in-fit 진단이다.

재현 명령·입력 경로·검사 기록은 `reports/md_full_submission_20260918/README.md`,
완료 상태와 ZIP/checkpoint 해시는 같은 폴더 `completion.json`에 있다.
실행 스크립트는 `experiments/md_full_submission_20260918/prepare_mr_full_submission.py`다.

## 0C. 2026-09-18 완료 결과와 규정 점검 — 가장 먼저 읽을 것

등록 MR V0는 `0.191002`다. 같은 train310/tune37에서 20,554 update를 마친
`A2-DIRECT-s1`은 **0.164281**, `A3-DIRECT-s1`은 **0.149289**였다. A3는 등록 MR 대비
21.84% 개선됐고 검증 11개 session 모두 개선됐다. 아직 하나의 training seed 결과이며
새 서버 제출 결과는 없다. A3의 일반 주행은 `0.196742→0.152271`, steady stop은
`0.035835→0.045817`로 악화됐다.

- `MR-NATIVE-FULL-s1`: 376 unique scenes / 101,520행 / 24,931 update 완료.
  `0.097245`는 **학습에 포함된 V0 행의 in-fit 진단**이다. held-out 성능으로 비교하지 않는다.
- `OOF-MR-T203-s1`: old203 / 54,810행 / 20,554 update와 V0 평가 완료, **V0 0.226444**.
  unseen new107의 예측 생성·residual 분석은 아직 하지 않았다. producer의 학습 완료와
  후속 OOF prediction 완료를 혼동하지 않는다.
- A2/A3-DIRECT는 기존 제출 terminal에 부착해 이어 학습한 모델이 아니다. 기존 MR과 같은
  `r0_init_tplus.pth`에서 status 조건을 추가한 전체 구조를 다시 공동 학습했다.
- 현재 A3-DIRECT에는 progress residual / dv+da factorization / P×V selector가 없다.
  `A3-FP-VA`의 실제 마지막 학습 로그는 **step 600**이다. 이전의 350은 중간 관측값이었다.

**규정 및 입력의 정확한 설명:** A2부터 영상 추정값이 아닌, 제공된 past/current pose로
계산한 `vx,vy,ax,ay,yaw_rate`를 학습·추론 입력으로 쓴다. A2는 공통 scene query만
조건화한다. A3는 scene/motion/global FPN 채널 gate를 추가하므로 state/history 출력도
제공 status의 영향을 받는다. A3를 Q10의 image-only state 추론이라고 설명하지 않는다.
허용 근거는 Q6의 현재 status 계산과 Q7/Q8의 공통 인지 특징 간접 활용이며, 개별 구조의
최종 승인은 코드 심사에 달려 있다. supervision으로만 사용했다는 설명도 틀리다.

완료된 A3 checkpoint의 전체 V0 입력 교체 검사:

| 조건 | PREFIX |
|---|---:|
| 정상 | 0.149289 |
| 영상만 다른 session 영상으로 교체 | 0.894936 |
| 영상 전체 단색 | 2.118935 |
| status만 다른 session 값으로 교체 | 0.718751 |

정상 예측은 저장 terminal과 좌표별 오차 0으로 재현됐다. 원본 ego_pose/timestamps에서
미래 pose를 제외하고 현재까지 31개 pose만으로 status를 다시 계산한 결과, 전체 1,998행의
다섯 status 값이 캐시 입력과 정확히 일치했다. 영상 의존성과 causal source의 근거이며,
이 검사 자체가 운영국 승인이나 단순 임베딩 우회 부재의 증명은 아니다.

새 기록 위치:

- `reports/md_shared_dynamics_20260917/RESULTS_20260918_KO.md`: 완료 점수와 해석.
- `reports/md_shared_dynamics_20260917/COMPLIANCE_AND_CHANGES_20260918_KO.md`: 실제 정보 경로와 Q&A 근거.
- `reports/md_shared_dynamics_20260917/records_20260918/README.md`: **24개 실행 단계** 전체 색인.
- 같은 폴더의 `experiment_index.json`: 설정·평가·paired 비교·원본 artifact 경로/해시 및 20개 로그 색인.
- 같은 폴더 `runs/`: 학습 metrics.jsonl, manifest, experiment, 저장 평가 요약과 종료 기록.
- `A2-DIRECT_terminal_input_audit_20260918.json`, `A3-DIRECT_terminal_input_audit_20260918.json`,
  `causal_status_replay_20260918.json`: 실제 검사 결과.
- 재실행 코드는 `experiments/md_shared_dynamics_20260917/`의 audit/replay/archive 스크립트.

checkpoint, 이미지, 대형 prediction 배열은 서버에 보존한다. GitHub에는 텍스트 기록과
원본 위치·크기·주요 파일 SHA256을 올린다. 다음 실험의 우선순위는 A3 재현 및 별도 FULL,
그리고 새 strong base의 deployable oracle/OOF 분석이며 아직 새 학습은 시작하지 않았다.

## 0B. 2026-09-17 19시대 방향 전환 — 당시 결정 기록

최근 `FRONT/SIDE residual`, auxiliary, denoise, 짧은 A2 warmup은 모두 일반 주행
1,875행을 `0.001`도 고치지 못했다. 마지막 `STATUS-A2-S-SCENE` 두 seed도
base→final `0.191341→0.189591`, `0.190334→0.189498`이었고, nonstop 개선은
각각 `0.000960`, `0.000257`뿐이었다. 이 작은 post-hoc adapter 계열은 종료한다.

그러나 status-conditioned shared perception 자체를 기각한 것은 아니다. 과거 깨끗한
flip50 비교에서 제공-status A2 `0.228957` 대 Q10 `0.258544`로 약 `0.02959`의
차이가 있었다. 현재 warmup은 완성된 MR에 zero-init 경로를 사후 부착한 조건이었다.

새 주력은 `reports/md_progress_residual_20260917/DIRECTION_RECHECK_KO.md`와 커밋
`e101dc3`에 기록돼 있다.

- GPU 0: `MR-NATIVE-FULL-s1` 제출 안전망.
- GPU 1: `A2-DIRECT-s1`. 등록 MR과 같은 초기화·train310·20,554 update에서 shared
  status query를 처음부터 함께 학습하는 control.
- GPU 2: `OOF-MR-T203-s1`. old203만 학습해 unseen new107의 honest residual 분포를 생성.
- GPU 3: `A3-DIRECT-s1`. A2의 shared query에 multiplicative status conditioning을
  공통 image FPN 전체로 확장하되, direct XY planner의 정상적인 길이 gradient를 유지한다.

최초 `A3-FP-VA-s1`은 마지막 학습 로그 step 600에서 종료했다. MR 이전 초기 base가 `0.477651`이고,
실제 cap의 `delta_v+delta_a` oracle도 `0.222499`여서 등록 MR `0.191002`보다 나빴다.
방향은 학습될 수 있지만, 약한 초기값에서 proposal 길이 gradient를 처음부터 막는 것은
status 전달과 factorization을 불필요하게 섞는다. Factorization은 강한 direct checkpoint
또는 GPU 2의 honest error distribution 위에서 stage-2로만 다시 검증한다.

새 코드의 단위검사 6개와 `A3-DIRECT` 실제 2-step smoke가 통과했다. A2/A3-DIRECT의
첫 step planning loss는 모두 `0.6594299078`로 같아 초기 함수 보존을 확인했다. 현재
GPU1과 GPU3은 seed·sample order·budget이 같고, 공통 FPN status gate만 다르다.

판정은 같은 plain V0와 같은 checkpoint budget에서 한다. 등록 MR seed1의
step 3426/6852/10278/13704/17130/20554 값은
`0.254811/0.218241/0.222304/0.206502/0.192196/0.191002`다. 곡선이 비단조이므로
첫 중간점 하나만 보고 중단하지 않는다. 3위에는 단일 서버 관측 환산으로 DEV 약
`0.12355`가 필요하며, 작은 `0.001~0.005` 개선을 모으는 전략으로는 닿지 않는다.

## 0A. 2026-09-17 후속 교정 — 아래의 오래된 수치보다 우선한다

- 공개 leaderboard `mode=best` 재확인 기준 현재 제출은 14위이고 3위는 `0.1305365832`다.
  아래의 8위 및 3위 `0.14664` 표기는 당시 불완전한 snapshot이다.
- split의 `historical_val` 9 scene은 `val` 29 scene에 전부 포함된다. 전체 자료는 385가 아니라
  **376 unique scene / 101,520 stride-1 rows**이며 동일 노출량 terminal은 **24,931 update**다.
- `OPEN_ISSUE.md` 질문 10에는 답변이 있다. 영상에서 직접 추론한 ego history/status를
  planner가 사용하는 것은 허용된다. raw provided history/status 입력과 구분한다.
- 사용자는 서버 응답의 `elapsed_ms`는 채점 harness 시간이라며 이번 모델 판단에서 제외하도록
  지시했다. 아래 미해결 문단을 다시 실험 우선순위로 올리지 않는다.
- `EXECUTION_REVIEW_KO.md`의 FRONT/SIDE residual 순서는 이후 실측으로 종료됐다. 최신 판단은
  위 0C의 완료 결과와 최신 A2/A3-DIRECT / OOF 상태를 따른다.

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
