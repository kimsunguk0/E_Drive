# 데이터셋 준비

## 1. 추론 (테스트 데이터)
대회에서 배포한 테스트 폴더를 그대로 사용합니다. 각 clip에는 `camera_*/frame_{0..-30}.jpg`, `ego_pose.parquet`, `calibration.parquet`가 있어야 합니다.

왜곡 보정, crop, resize(768×432, 384×216), 과거 pose로 ego 상태 계산, +50 frame 목표점 읽기는 모두 추론 코드가 수행합니다.

```sh
bash scripts/infer.sh /path/to/test OUT_DIR
```

## 2. 학습 데이터
### 2.1 원본
대회 제공 train 376 scenario 폴더(`<scenario>/annotation/`, `calibration/`, `camera_*/`, `meta/`)를 그대로 사용합니다. `scripts/train.sh`의 두 번째 인자로 이 폴더를 지정합니다.

### 2.2 파생 캐시 (동봉)
원본에서 생성한 캐시입니다. 동봉 파일을 그대로 사용하는 것을 권장합니다.

| 파일 (`data/`) | 크기 | 내용 | 생성 코드 (`src/data_prep/`) |
|---|---:|---|---|
| `ego_cache.npz` | 45 MB | 376 × 300 프레임의 ego 과거·미래 궤적, 목표점, 속도·가속도 | `etri_ego_cache.py` |
| `ego_cache_5s.npz` | 8 MB | 5초 10점 정답 궤적 (3단계 후보 배정용) | `etri_label_5s.py` |
| `split.json` | 0.2 MB | 시간 연속 session 단위 train/tune/val 분할 | `build_grouped_split_v2.py` |
| `supervision/` | 54 MB | scenario별 occupancy·lane·ego 상태/과거 궤적 감독과 카메라 보정 | `build_scene_supervision_v2.py`, `derive_motiondrive_v2_geometry.py` |
| `status_full/` | 2 MB | 행별 ego 상태 5값 (과거 pose 5개로 계산) | `build_full_cache.py` |
| `status_dev/`, `status_nominal/` | 4 MB | 같은 형식의 부분 분할 캐시 (학습 코드가 참조) | — |
| `metadata/` | 1 MB | 학습 코드가 읽는 이전 단계의 실행 설정과 검증 기록 | — |
| `fixtures_train8/` | 0.5 GB | 검증용 학습 clip 8개 (테스트 폴더와 같은 형식) | — |

재생성 순서는 원본 → ego 캐시 → 5초 정답 → 분할 → scenario 감독 → geometry 파생 → 상태 캐시입니다. 각 코드는 기존 파일을 덮어쓰지 않습니다. 재생성 결과는 동봉 파일의 manifest sha256과 비교해 확인할 수 있습니다.

## 3. Checkpoint (`checkpoints/`)
| 파일 | 용도 |
|---|---|
| `backbone_nuimages_cascade_r50.pth` | 공개 백본 (외부 데이터) |
| `stage1_initializer.pth` | 1단계 초기값 (공개 백본 + 무작위 초기화, ETRI 학습 0회) |
| `stage1_h4_progress_full.pth` | 1단계 산출이자 2단계 시작점 |
| `stage2_l_full6.pth` | 2단계 산출이자 제출 모델의 시작점 |
| `ext_full_v7.pth` | **제출 모델** |

2단계 코드는 시작 checkpoint의 sha256을 확인하므로 동봉한 1단계 산출에서 시작합니다. 3단계는 동봉한 2단계 산출에서 시작합니다.
