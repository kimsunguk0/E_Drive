# 데이터셋 준비

## 1. 추론 (테스트 데이터)
대회에서 배포한 테스트 폴더를 그대로 사용합니다. 각 clip에는 `camera_*/frame_{0..-30}.jpg`, `ego_pose.parquet`, `calibration.parquet`가 있어야 합니다.

왜곡 보정, crop, resize(768×432, 384×216), 과거 pose로 ego 상태 계산, +50 frame 목표점 읽기는 모두 추론 코드가 수행합니다.

```sh
bash scripts/infer.sh /path/to/test OUT_DIR
```

## 2. 학습 데이터
### 2.1 원본 → 학습 입력 (`scripts/prepare_train.sh`)
대회 제공 train 376 scenario를 입력으로 사용합니다. 배포 형식인 `<scenario>.tar`와, 압축을 푼 `<scenario>/` 폴더를 모두 받습니다.

```sh
bash scripts/prepare_train.sh /path/to/train PREP     # 약 70 GB, 1회
```

| 출력 (`PREP/`) | 내용 | 학습 시 컨테이너 경로 |
|---|---|---|
| `etri_768/<scenario>/<camera>/<frame>.jpg` | 왜곡 보정 → 1920×1080 crop → 0.4배 → JPEG 품질 95 (768×432) | `/tmp/pm97/cache/etri_768` |
| `meta_train/<scenario>/{annotation,calibration,meta}/` | 원본 parquet 그대로 | `/tmp/pm97/data/etri/meta_train` |

영상 변환은 학습 캐시를 만들 때 쓴 `src/data_prep/etri_build_cache.py`의 함수를 그대로 사용합니다(`tools/build_train_cache.py`). 원본 tar 2개로 다시 만든 결과를 학습에 사용한 캐시와 비교했을 때, 영상 3,600장과 parquet 16개가 바이트 단위로 모두 일치했습니다. `scripts/train.sh`는 두 번째 인자인 `PREP`의 두 폴더를 위 경로에 마운트합니다.

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

생성 순서는 원본 → ego 캐시 → 5초 정답 → 분할 → scenario 감독 → geometry 파생 → 상태 캐시입니다. 생성 코드는 사용한 그대로 동봉합니다. 일부 코드는 당시의 중간 산출물(카메라 보정 pkl, 과거 검증 분할 등)을 입력으로 읽으므로 동봉 캐시를 그대로 사용하는 것을 권장합니다. 각 폴더의 manifest에 파일별 sha256이 기록되어 있습니다.

## 3. Checkpoint (`checkpoints/`)
| 파일 | 용도 |
|---|---|
| `backbone_nuimages_cascade_r50.pth` | 공개 백본 (외부 데이터) |
| `stage1_initializer.pth` | 1단계 초기값 (공개 백본 + 무작위 초기화, 대회 데이터 학습 전) |
| `stage1_h4_progress_full.pth` | 1단계 산출이자 2단계 시작점 |
| `stage2_l_full6.pth` | 2단계 산출이자 제출 모델의 시작점 |
| `ext_full_v7.pth` | **제출 모델** |

2단계 코드는 시작 checkpoint의 sha256을 확인하므로 동봉한 1단계 산출에서 시작합니다. 3단계는 동봉한 2단계 산출에서 시작합니다.
