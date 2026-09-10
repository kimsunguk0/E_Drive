# Confirmation12 실행 계약

원래 train203 안에서 정했던 12 session / 32 scene을 그대로 제외한다. 새 모델은 **고정된 공개 NAVSIMv1 R34 checkpoint부터 다시 학습**하고, bank도 train171만으로 새로 적합한다. 기존 primary/fulltrain checkpoint 및 그 bank를 이어받으면 이 확인 계약을 만족하지 않는다.

아래 경로는 작업트리 `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910` 기준이다.

| 역할 | 실제 row 파일 | 행 / scene / session | stride |
|---|---|---:|---:|
| 학습 | `reports/sparsedrivev2_20260910/split_audit/rows/confirmation12_train.npy` | 46,170 / 171 / 60 | 1 |
| 기존 tune37 | `reports/sparsedrivev2_20260910/split_audit/rows/confirmation12_tune.npy` | 1,998 / 37 / 11 | 5 |
| 최종 confirmation12 | `reports/sparsedrivev2_20260910/split_audit/rows/confirmation12_held_rows.npy` | 1,728 / 32 / 12 | 5 |

`confirmation12_held_rows.npy`는 기존 `confirmation12_confirmation12.npy`의 byte-identical 별칭이다. 모두 frame≥30, 원본 ego cache row index 오름차순 `<i8` 배열이다. 학습 row 배열 SHA는 `701e7ea7b76acd6b0d99845a0f400c9b65a5c470131d2d3b6324192e09c28a35`, held는 `2809febcd692040870821e2575062722226b4f58152e53b7eed27d9cc5aaa3b7`, tune은 기존 `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`이다. 이 값은 NPY 컨테이너 파일 hash와 구분한다.

## 현재 loader와 연결

모든 데이터셋에 byte copy인 **`primary_manifest.json`을 전달**하고 rows 파일로 해당 집합을 제한한다. 파생 `confirmation12_manifest.json`은 membership 설명용이며 현재 PlanDataset의 고정 primary SHA 검사에 그대로 전달하지 않는다.

```python
from pathlib import Path
from data import PlanDataset

root = Path('reports/sparsedrivev2_20260910/split_audit')
common = dict(base='/NHNHOME/data/sukim/adcl',
              split_manifest=root / 'primary_manifest.json',
              status_mode='zero')  # causal_selection도 같은 row 계약으로 검증됨
train = PlanDataset(**common, split='train',
                    rows_file=root / 'rows/confirmation12_train.npy')
tune = PlanDataset(**common, split='tune', stride=5,
                   rows_file=root / 'rows/confirmation12_tune.npy')
held = PlanDataset(**common, split='train', stride=5,
                   rows_file=root / 'rows/confirmation12_held_rows.npy')
```

held는 원래 primary의 train203 내부이므로 `split='train'`을 쓰는 것이 맞다. 현재 loader의 provenance에 표시되는 `split='train'`은 원본 population을 뜻하며 학습 사용을 뜻하지 않는다. 별도의 역할명, row SHA 및 run 경계를 함께 기록한다.

훈련 CLI에는 `--split-manifest .../primary_manifest.json --train-rows .../rows/confirmation12_train.npy --eval-split tune --eval-rows .../rows/confirmation12_tune.npy`를 사용한다. 모델·bank 크기·학습 예산·checkpoint 선택은 tune37에서 확정한다. held32를 반복 평가하여 모델이나 checkpoint를 선택하지 않는다. 선택을 고정한 뒤 held loader로 별도 최종 확인한다. 원래 reserve136은 이번 학습·선택·확인 범위에서 제외한다.

## 새 bank 경계

`confirmation12_train_scenes.json`은 `{"scenes": [171개 scene]}` 형식이다. 새 bank extraction에는 원 `primary_manifest.json`, `--partition train --scenes-json .../confirmation12_train_scenes.json --min-frame 30 --stride 1`을 전달한다. 추출의 row 배열은 위 46,170행 SHA와 정확히 같아야 한다. 새 bank fit은 이 추출만 사용한다. 기존 train203에서 적합된 path bank도 재사용할 수 없다.

모델 CUDA 생성 전에 실제 `train.verify_initialization(public_checkpoint, new_bank, train, held)`가 통과해야 한다. 공개 checkpoint SHA는 `330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735`로 고정한다. 현재 검사기는 bank sidecar SHA, bank 내부 metadata 및 bank.train_rows와 train.allowed_rows의 정확한 일치를 강제한다.

## 실제 CPU 확인 결과

`confirmation12_runtime_check.py` 실행 결과는 `confirmation12_runtime.json`에 있다. receipt SHA는 `bea2f48bc90e3d91a91848cfd3911a14ef27338fee6f036984f278a07f0e021e`다.

- zero / causal_selection 두 모드에서 세 데이터셋의 row, scene 및 session 경계가 모두 통과했다.
- 기존 primary bank + primary train203 + tune37 조합은 통과했다.
- 기존 primary bank + confirmation train171 조합은 tune37 및 held32 검증 모두에서 실제 거부되었다.
- 비공개 초기화 파일과 잘못된 bank sidecar hash도 실제 거부되었다.
- 모델을 로드하거나 실행하지 않았고 GPU·예측 성능·label 분포·reserve 데이터셋 평가를 사용하지 않았다.

이 추가 검증은 원래 metadata-only 감사와 범위가 다르다. 실제 PlanDataset을 생성하면서 loader 내부의 전체 `fut` 배열 메모리 로드와 선택한 train/held/tune label의 유한성 검사는 실행했다. 따라서 모든 label bytes를 전혀 읽지 않았다고 주장하지 않는다. confirmation12 역시 역사적 노출이 없는 새 데이터라는 의미가 아니라, 새 공개 초기화 run에서 이 집합을 본 learned ancestry와 bank를 배제한다는 의미다.

현재 학습 중인 `data.py`, `train.py`, `losses.py`, `public_model.py`는 변경하지 않았다. 이 문서와 추가 검사 파일은 모두 split_audit 보고서 디렉터리에만 저장했다.
