# MotionDrive V2 P4 Fresh Public-Baseline Protocol

작성일: 2026-09-07 KST

## 1. 목적과 해석 경계

P4는 공개 nuImages ResNet-50 backbone에서 시작하는 새 공통 초기값으로 원 학습기를
다시 실행해, 규정 준수·공식 성능·재현성을 함께 확인하는 실험이다. 기존 P3 query
adapter는 사용하지 않는다. 최종 목표에는 동일한 입력 계약으로 실물 3090에서 전체
11-image forward 지연을 측정하는 배포 검증도 포함한다.

P4 결과는 다음보다 좁게 해석한다.

- 동일 tune 조건의 경쟁력 screening에서 기존 P3 수치(예: D3 `0.4379407`)보다
  `0.01` 이상 개선되는지 참고한다.
- 이 차이는 geometry의 단독 순효과, 독립 holdout의 개선 증거, 대회 1위 증거가 아니다.
- 원거리 기대 점수 `0.1`, leaderboard 점수 또는 서로 다른 평가 조건의 수치와 직접
  비교하지 않는다.
- 공통 초기화, 모델 구성 또는 시간 측정 조건을 바꾸려면 새 실험 이름과 별도
  프로토콜을 먼저 만든다.

## 2. 고정 provenance와 사전점검

실행 기준 저장소는 B200의 `/NHNHOME/data/sukim/adcl`이다. 2026-09-07
22:54:40 KST 읽기 전용 사전점검에서 실제 HEAD는
`ebd9ef366c6ba814218ad9a1158246a84cd9b9be`였고 tracked 파일은 clean이었다.
기존 runtime source 19개는 로컬과 B200에서 byte-identical SHA를 가졌다. 이 19개
소스는 P4를 위해 수정하지 않는다.

고정 입력은 다음과 같다.

| 입력 | SHA-256 |
|---|---|
| `ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth` | `4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e` |
| `data/etri/motiondrive_v2/grouped_split_rawtime.json` | `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936` |
| `data/etri/motiondrive_v2/train_tune_geometry_v2/supervision_manifest.json` | `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93` |
| `data/etri/motiondrive_v2/train_tune_geometry_v2/calibration.npz` | `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961` |

사전점검 당시 GPU4와 GPU5는 모두 used `0 MiB`, free `182632 MiB`였고 compute
PID가 없었다. GPU4 UUID는
`GPU-4b804d68-fd61-af14-393a-573c533d5006`, GPU5 UUID는
`GPU-1e9aea73-4e6b-2cb5-b788-f3d99e6dc6f8`이다. GPU0--3에는 별도 alpamayo
작업과 compute PID `1963560`--`1963563`이 있었으며 반드시 보존한다. 최신 사용자
허가에 따라 P4는 GPU4/5만 사용하고 GPU6/7은 사용하지 않는다. 과거 목표나 설정의
GPU0--3 표기는 이 최신 허가를 대체하지 못한다.

사전점검 당시 다음 P4 경로는 모두 없었다.

- 공통 초기값 `work_dirs/motiondrive_v2/p4_public_init_s0.pth`
- 초기화 보고서 `reports/p4_public_init_s0.json`
- `work_dirs/motiondrive_v2/p4_fresh_{canary_s0,pretrain_s0,pretrain_s1,joint_s0,joint_s1}`
- 위 실행명에 대응하는 `logs/motiondrive_v2/*.log`와 `*.supervisor.json`

## 3. 공통 초기값 I0

I0는 위 공개 nuImages R50 checkpoint에서 backbone 항목 318개만 완전히 주입한다.
compact FPN, scene encoder, motion encoder와 planner는 CPU random seed 0으로
초기화한다. I0 생성 중 ETRI optimizer update는 0회이며 ETRI 이미지나 라벨 tensor를
읽거나 forward하지 않는다. split manifest는 provenance 확인에만 사용한다.

I0의 고정 모델 조건은 low-feature R50, `plan_output_scale=(10, 5)`, goal OFF,
state OFF이다. P3 adapter나 P3 학습 checkpoint를 섞지 않는다. 생성은 CUDA가 보이지
않는 CPU 환경에서 수행하고, 실행 직전에 실제 Git HEAD와 tracked-clean 상태, 공개
checkpoint SHA 및 split SHA를 다시 검사한다. 생성 보고서에는 실제 Git SHA, 입력
SHA, source-file SHA, public backbone 주입 수, seed, ETRI optimizer step 0과 출력 I0
SHA를 기록한다.

기존 I0 checkpoint 또는 보고서 경로가 하나라도 존재하면 중단한다. 덮어쓰기,
이름 재사용, 기존 파일의 성공 처리 또는 삭제는 허용하지 않는다.

## 4. 공통 학습 계약

첫 optimizer update부터 C1 corrected geometry와 nominal time을 사용한다. 공통 설정은
다음과 같다.

| 항목 | 고정값 |
|---|---|
| 모델 | R50 low-feature, output scale `10/5`, P3 adapter 없음 |
| batch | global 16, microbatch 2, eval batch 4 |
| loader | workers 4, train stride 1, eval stride 5 |
| precision | BF16 |
| BN | running statistics fixed; affine와 다른 모든 학습 weight는 learnable |
| learning rate | head `1e-4`, backbone `1e-5` |
| regularization | weight decay `0.01`, gradient clip `5` |
| warmup | 본 단계 200 updates, canary 2 updates |
| auxiliary loss | occupancy/state/history/lane 각각 `0.2`, uncertainty weighting ON |
| data | train 54,810 frames, tune 1,998 frames |
| evaluation/save cadence | 매 250 updates |
| memory gate | CUDA allocator cap 12 GiB, 시작·실행 reserve 8 GiB |

최종 validation 136 scenes/31 sessions와 test는 실험 설계·선택·튜닝에서 격리한다.
별도 승인 전에는 읽거나 평가하지 않는다.

두 seed는 동일한 I0와 동일한 data provenance를 쓴다. seed 0/1은 data order와
optimizer/training RNG의 반복성 범위이며, compact 모듈까지 각각 새로 초기화한 독립
full-initialization seed 실험이 아니다. 따라서 seed range는 그 제한 안에서 보고한다.

## 5. 단계와 장치 배치

### 5.1 GPU5 plumbing canary

공통 I0에서 GPU5로 G1S1 joint 조건 2 updates를 실행한다. train 16, tune 8만
사용한다. 이 canary는 data/model/optimizer/save/supervisor 배관과 finite backward를
검증할 뿐 accuracy 결과가 아니며, 모델 선택이나 성능 주장에 쓰지 않는다.

### 5.2 P0 auxiliary pretrain

- seed 0: GPU4, G0S0, 2,000 updates
- seed 1: GPU5, G0S0, 2,000 updates

두 팔 모두 같은 I0에서 시작한다. planning loss는 정확히 0이고 auxiliary losses만
학습한다. 각 팔의 own `LAST2000`이 그 팔의 joint initializer다. P0의 random-planning
D3로 checkpoint나 seed를 채택하지 않으며, 2,000 updates를 수렴 상한으로 주장하지
않는다.

원 trainer가 initial과 LAST 평가에 이미 기록하는 history/state MAE 및 occupancy/lane
IoU를 추가 forward 없이 비교한다. 정확한 기록 key는
`history_position_mae_by_offset`, `state_mae_vx_vy_ax_ay_yawrate`, `occ_iou`,
`lane_iou`다. 이 평가에는 stop validation metric이 없으므로 이를 만들거나 보고하지
않는다. MAE는 감소 방향, IoU는 증가 방향을 보되, 수치 방향 검토는 안전 gate와
별개로 root가 수행한다.

planning loss가 0이어서 planner의 task gradient도 0이지만 optimizer의 weight decay는
planner parameter에 적용될 수 있다. 따라서 P0 own LAST의 planner가 I0와 동일하다고
주장하거나 그 동일성을 gate로 요구하지 않는다.

### 5.3 Joint training

P0 두 팔의 안전 gate가 통과하고 root가 auxiliary 지표 방향을 검토해 명시적으로
승인한 뒤에만 joint를 시작한다.

- seed 0: own seed-0 `LAST2000`에서 G1S1 6,000 updates
- seed 1: own seed-1 `LAST2000`에서 G1S1 6,000 updates

다른 seed의 LAST, BEST, I0 외 checkpoint 또는 P3 결과로 초기값을 바꾸지 않는다.

## 6. 단계별 안전 gate

canary, 각 pretrain과 각 joint에서 다음 조건을 모두 실제 증거로 확인한다.

1. supervisor 추정이 아니라 native child actual return code와 supervisor OS exit가
   모두 0이다.
2. 기록된 parent PID와 child PID가 종료 후 `/proc`에 존재하지 않고, GPU compute
   process에도 남지 않는다.
3. OOM, pressure event, signal 또는 nonfinite loss/gradient/tensor가 없다.
4. 시작 전과 실행 중 allocator 12 GiB cap 및 최소 free-memory reserve 8 GiB 계약을
   지킨다.
5. 단계의 own LAST가 존재하고 strict CPU load, 전체 tensor finite, 정확한 optimizer
   step 수와 source/data/initializer provenance를 통과한다.
6. fixed BN running statistics가 initializer와 byte/tensor 수준에서 불변이고 BN affine은
   학습 가능 상태다.
7. P0에서는 state, history, occupancy, lane 네 auxiliary head 모두 실제 tensor update가
   있었고 대응 optimizer state의 `exp_avg`가 nonzero임을 initializer와 optimizer
   payload 대비 확인한다.

어느 하나라도 실패하면 해당 단계를 성공으로 처리하거나 후속 단계로 우회하지 않는다.
원 실패 로그, checkpoint, report와 protocol은 삭제·수정·덮어쓰기하지 않는다.

## 7. 평가와 보고

joint의 primary 결과는 seed 0과 seed 1의 own `LAST6000` 모두, 두 값의 mean/range,
그리고 동일 조건 초기값 대비 변화다. `BEST`와 250-step trace는 secondary 진단으로
함께 보존한다. 중간 BEST 하나만 고르거나 한 seed만 골라 대표 결과로 보고하지 않는다.

경쟁력 screening은 기존 P3와 동일한 tune 1,998-frame 공식 시간가중 D3 조건에서
수행한다. `0.01` 개선 여부는 다음 단계 판단의 참고값일 뿐이며 geometry 단독 효과,
독립 holdout, leaderboard 또는 최종 공식 성능으로 확대 해석하지 않는다.

실물 3090의 전체 11-image forward 측정은 학습 결과와 별도의 배포 계약으로 수행한다.
입력 구성, precision, batch, warmup, 동기화, 측정 구간과 통계량을 고정·기록한다.
조건이 달라지면 같은 측정명이나 이전 수치와의 직접 비교를 허용하지 않는다.

## 8. Git·동시 작업·보존 규칙

코드와 프로토콜의 로컬/CPU 검증 후 clean source commit을 만들고, 그 commit에서 I0를
생성한다. 실행 중에는 wrapper가 실제 Git HEAD까지 고정 검사하므로 코드뿐 아니라
문서·보고서도 commit하지 않는다. 모든 live P4 프로세스가 실제 종료하고 source bytes
불변을 확인한 뒤에만 문서와 report commit을 허용한다.

기존 runtime source 19개, B200의 `adcl_wt_phase5a/`, `adcl_wt_phase5b/`, `backup/`,
기존 raw reports/protocols와 앞선 모든 실패 기록을 보존한다. 로컬 stale/dirty worktree의
사용자 `.gitignore` 및 `ETRI_MOTIONDRIVE_V2_EXECUTION_SPEC.md` 변경도 reset, checkout,
pull 또는 덮어쓰기로 정리하지 않는다. 각 업로드·commit·CPU I0 생성·GPU launch는
그 단계의 별도 승인 범위 안에서만 수행한다.
