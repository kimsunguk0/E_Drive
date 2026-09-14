# R0 재검증 및 데이터 확대 — 실행 프로토콜

- 근거 문서: `E_Drive_R0_Execution_Roadmap_20260914.md` (2026-09-14)
- 기준 모델 R0: `q10_q10_flip50_s0` (A2-off, flip p=0.5, 2층 direct planner)
- 기준 코드: git `ef49bad`의 31개 pinned 파일, 실측 재확인 완료
- 원칙: 기존 trainer·dataset·loss를 **호출**한다. 실험마다 복제하지 않는다.

## 파일

| 파일 | 역할 |
|---|---|
| `p0_registry.py` | R0 실체 고정 (checkpoint/모델상태 SHA, config, split, 전처리, A2 부재) |
| `evaluate.py` | trainer `--eval-only` 경로 호출 + 상세 record·ADE1/2/3·시점별 L2·행 부분집합 |
| `contract_tests.py` | P2 계약검사 9종 |
| `p3_split.py` | H(최종 확인 세션) 해시 선정, Tplus 정의, 새 split manifest |
| `p3_supervision.py` / `p3_supervision_finish.py` / `p3_rebind_reports.py` | 확대 split용 supervision 판본 조립·검증 |
| `p3_rebind_init.py` | R0 가중치 그대로, 확대 split을 선언하는 초기화 artifact |
| `train.py` | E1 두 arm 런처 (T203 / EXP) |
| `p1_raw_b1.py` | 원본 JPEG/parquet → 입력·출력 parity (batch 1) |
| `assemble_reports.py` | 결과표·제출 등록부 |

## 고정한 정보 경계 (로드맵 §2.2)

- 제공 수치 ego status: scene/motion/planner/selector 어디에도 입력하지 않음. R0 체크포인트에 A2 텐서가 **없음**을 실측 확인.
- goal: 공통 scene feature 형성 조건으로만.
- command/vad_cmd: E1에서 OFF. E3에서만 별도 비교.
- 과거 상대 pose: 영상 정렬에만. 과거 프레임 영상은 항상 함께 입력.
- 모델 forward가 받는 텐서는 정확히 6개: `images, history_images, lidar2img, history_transforms, time_offsets, goal_xy`.

## E1 예산

```
N0 = 54,810 (기존 train 행)
B  = 16, microbatch 2
UPDATES   = ceil(6 * N0 / B) = 20,554
EVAL_EVERY= ceil(N0 / B)     = 3,426   (기존 train 1회 노출)
warmup 200 → cosine → 20,554
```

두 arm은 **같은 update 수**를 쓴다. 확대 arm은 자기 데이터를 3.93회, 기존 arm은 6.00회 본다.
이것은 고정 계산량에서의 데이터 확대 효과이며 동일 epoch 비교가 아니다.

checkpoint 선택 규칙(사전 고정): 계획된 평가 시점 중 **V0 official_d3 최소**, 동률이면 더 이른 시점.
`best-on-V0`와 `terminal`을 둘 다 기록한다.

## 평가 집합

| 이름 | 정의 |
|---|---|
| `train_probe` | 기존 train 각 session에서 row id 기준 균등 최대 48행 (3,456행), 증강 OFF |
| `train_full` | 기존 train 54,810행 전체, 증강 OFF |
| `tune_legacy` | tune 1,998행, B4/bf16/nominal (R0 원 조건) |
| `tune_B1` | 같은 1,998행, batch 1 |
| `raw_B1_fixture` | 사전등록 8 clip을 원본 JPEG/parquet에서 구성, batch 1 |
| `H` | reserve에서 해시로 고른 10 세션 / 29 scene / 7,830행. **최종 1회만 연다** |

## H에 대한 정확한 표현

H는 R0의 모든 조상 학습에서 제외된 집합이다(조상 계보 전체가 train203만 학습).
그러나 `historical_val38` 중 29 scene이 A로 편입되어 학습에 들어간다. 그 38 scene은
과거 dense-VAD/ScoreDrive 계열에서 반복 평가된 이력이 있다. 따라서 H는
**ancestor-fit-excluded confirmation set**이며, 한 번도 열람되지 않은 blind set이 아니다.
