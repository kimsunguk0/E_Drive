# SparseDriveV2 split audit — 2026-09-10

결론: **기존 train203 / tune37 / reserve136을 그대로 유지한다.** 원본을 재분할하거나 수정하지 않았다. 공개 초기값 또는 해당 fit split만 본 조상 체크포인트에서 시작해야 이 경계가 유효하다.

- 원격 작업트리: `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910`
- 코드: `experiments/sparsedrivev2_20260910/split_audit.py`
- 출력: `reports/sparsedrivev2_20260910/split_audit/`
- 원본 split 및 `primary_manifest.json` SHA: `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`
- 감사 결과 `audit.json` SHA: `6eef5e499bd242d0c978c0f3962a44c65cd8df9e0c9eb9361112dc95b179e90f`

## 검증 결과

원본 376개 시나리오, 114개 raw timestamp session, 112,800개 row를 확인했다. 모든 시나리오의 timestamp parquet을 실제로 읽어 원본 hash와 session 묶음을 재검증했다. GPU·test 자료·trajectory/goal/status 값·예측 성능은 접근하지 않았다. 읽은 NPZ 배열은 `scenarios`, `scen_idx`, `frame`뿐이다.

학습 anchor frame30..299에 과거 최대30frame, 미래 goal +50frame을 붙이면 필요한 timestamp support는 frame0..349다. 이 전체 support를 비교한 교차 split overlap은 0이다.

| 경계 | 가장 가까운 support 간격 |
|---|---:|
| train–tune | 65.100초 |
| train–reserve | 65.000초 |
| tune–reserve | 105.100초 |

과거 여분까지 포함한 raw parquet 전체 범위(frame−50..349)로 더 보수적으로 비교해도 overlap은 0이며 최소간격은 각각 60.100 / 60.000 / 100.100초다. 시간 분리는 확인했지만 공간·도로가 겹치지 않는다고 주장하지 않는다.

## 기본 screen 계약

| split | 시나리오 | session | row | frame stride |
|---|---:|---:|---:|---:|
| train | 203 | 72 | 54,810 | 1 |
| tune | 37 | 11 | 1,998 | 5 |
| reserve (`val`) | 136 | 31 | 7,344 | 5 |

모든 row는 원본 ego cache index 오름차순이다. frame≥30을 적용한다. `rows/primary_train.npy`, `rows/primary_tune.npy`, `rows/primary_val.npy`로 고정했다.

- train row 배열 SHA: `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`
- tune row 배열 SHA: `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`
- reserve row 배열 SHA: `2377ab585036366af2040673cfb3fc19121bf426c3d375e06497e92db1ee7617`

이는 NPY 파일 SHA가 아니라 `<i8` 배열 raw bytes의 SHA다. NPY 파일 SHA와 label을 제외한 `(scene,row,frame)` identity SHA는 `audit.json`의 `row_contracts`에 별도로 기록했다.

`primary_manifest.json`은 원본의 byte copy이므로 기존 supervision의 split SHA와 호환된다. screen에서는 이 manifest 또는 원래 canonical manifest를 그대로 사용한다. reserve136은 최종 확인을 위해 남기며 이번 screen에서 성능을 보지 않는다. 역사적으로 문서와 일부 라벨을 보았기 때문에 pristine blind set이라는 이름은 쓰지 않는다.

## 선택적 3-fold OOF

`crossfit_fold0_manifest.json` / `crossfit_fold1_manifest.json` / `crossfit_fold2_manifest.json`은 원래 train203 내부에서만 session을 분할한다. label을 보지 않고 session별 시나리오 수로 균형을 맞췄다. 원래 tune37과 reserve136은 세 fold에서 모두 fit 제외다.

| fold | fit 시나리오/session | OOF 시나리오/session | fit row | OOF row(2Hz) |
|---|---|---|---:|---:|
| 0 | 135 / 49 | 68 / 23 | 36,450 | 3,672 |
| 1 | 135 / 47 | 68 / 25 | 36,450 | 3,672 |
| 2 | 136 / 48 | 67 / 24 | 36,720 | 3,618 |

각 fold의 `tune`이 그 fold의 OOF 생성 대상이다. 모든 OOF 시나리오는 정확히 한 fold에만 등장한다. 진짜 OOF dump를 만들 때 OOF 정답으로 checkpoint를 고르지 않는다. 고정 update budget 또는 fit 내부의 별도 early-stop 분리를 쓴다. encoder뿐 아니라 bank·normalizer·teacher adaptation까지 같은 fit 경계를 지켜야 한다.

파생 manifest의 SHA는 원본과 다르다. 기존 loader는 이를 원래 supervision과 혼용하면 올바르게 거부한다. 새 provenance envelope 또는 엄격한 fit whitelist를 구현해야 하며 기존 artifact를 덮어쓰거나 SHA 검사를 느슨하게 바꾸면 안 된다.

## 선택적 12-session 확인

`confirmation12_manifest.json`은 기존에 정했던 12session/32scene 목록을 그대로 사용한다. fit은 train171/60session/46,170rows, tune은 기존37 그대로, `confirmation12`는 1,728개 row다. 원래 reserve136은 계속 fit에서 제외한다.

이 실험은 공개 초기값부터 train171 경계를 지키는 별도 run용이다. train203 또는 fulltrain checkpoint·bank를 이어받으면 confirmation12가 다시 오염된다. 단순히 마지막 fine-tuning에서 32scene을 빼는 것으로는 충분하지 않다.

## 시간 입력 주의점

train의 5개 시나리오는 실제 frame 간격이 약0.10624초다. 따라서 frame−30은 약3.187초 전일 수 있다. 공식 미래 label offset(+5,+10,…,+30; goal+50)은 임의 resampling하지 않는다. 새 모델이 과거−30을 추가한다면 nominal frame 계약과 실제 wall-clock 3초 제한을 구분해 명시적으로 처리해야 한다. 현재 분할 사이의 시간 누수와는 별개다.

## 실행 및 확인

`execution.log`는 실제 성공 실행 출력이다. 첫 경로 탐색 실패는 `execution_attempt1_missing_ego_path.log`에 보존했다. 모든 생성 manifest는 기존 `build_grouped_split_v2.validate_manifest`로도 재검증했고, NPY 파일 및 배열 hash, row 정렬, OOF 완전·서로소 partition을 별도 CPU 명령으로 재확인했다.
