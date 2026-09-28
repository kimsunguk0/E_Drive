# 데이터셋 준비

## 1. 추론 (테스트)
대회 배포 테스트 폴더를 그대로 쓴다. 전처리는 추론 코드가 한다.
- clip마다 `camera_*/frame_{0..-30}.jpg`, `ego_pose.parquet`, `calibration.parquet`가 있어야 한다.
- 처리 내용: 왜곡 보정, crop, 768×432·384×216 resize, 과거 pose로 status 계산, +50 frame 목표점 읽기.

```sh
bash scripts/run_inference.sh /path/to/test OUT_DIR
```

## 2. 학습
### 2.1 원본 데이터
대회 제공 train 376 scenario 폴더를 그대로 쓴다(`<scenario>/annotation/`, `calibration/`, `camera_*/`, `meta/`). `scripts/train.sh`가 이 폴더를 컨테이너의 `/tmp/pm97/data/etri/train`으로 마운트한다.

### 2.2 파생 캐시 (동봉, 그대로 쓰는 것을 권장)
원본에서 만든 캐시다. 학습 코드가 읽는 경로 그대로 동봉했다. sha256은 `MANIFEST.json`과 각 폴더의 manifest에 있다.

| 파일 | 크기 | 내용 | 생성 스크립트 |
|---|---:|---|---|
| `data_pm97/data/etri/ego_cache.npz` | 45 MB | 376×300 프레임의 ego 과거/미래 궤적, 목표점, 속도·가속도 등 | `code/scripts/etri_ego_cache.py` |
| `data_pm97/data/etri/ego_cache_5s.npz` | 8 MB | 5초 10점 정답 궤적 (3단계 후보 배정용) | `code/scripts/etri_label_5s.py` |
| `code/data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json` | 0.2 MB | 시간 연속 session 단위 train/tune/val 분할 | `code/scripts/build_grouped_split_v2.py` |
| `code/data/etri/motiondrive_v2/r0reset_tplus_geometry_v2/` | 54 MB | scene별 occupancy·lane·ego state/history 감독과 보정 | `code/scripts/build_scene_supervision_v2.py` → `derive_motiondrive_v2_geometry.py` |
| `code/data/etri/motiondrive_v2/a2_h4_status_full_20260921/` | 2 MB | 행별 status 5값 (과거 5개 pose 적합) | `code/experiments/a2_progress_full_20260921/build_full_cache.py` |
| `code/data/etri/motiondrive_v2/a2_h4_status_20260920/`, `a2_nominal_status_20260918/` | 4 MB | 같은 형식의 부분 분할 캐시 (코드가 참조) | `h4_status.py` / `nominal_status.py` |

재생성 순서: 원본 → ego 캐시 → 5초 정답 → 분할 → scene 감독 → geometry 파생 → status 캐시. 각 스크립트는 기존 파일을 덮어쓰지 않도록 되어 있다. 재생성한 결과는 동봉 캐시의 sha256과 비교해서 검증한다.

### 2.3 checkpoint (동봉)
| 경로 (`code/` 기준) | 용도 |
|---|---|
| `ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_…pth` | 공개 백본 (외부 데이터) |
| `work_dirs/a2_motion_fresh_20260919/initializers/public_nuimages_qrefine_s1.pth` | 1단계 초기값 (공개 백본 + 무작위 나머지, ETRI 학습 0회) |
| `work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth` | 1단계 산출 = 2단계 부모 (sha 고정) |
| `work_dirs/a2_long_motion_20260922/L-FULL6-s1/ckpt_step38070.pth` | 2단계 산출 = 제출 모델의 부모 |
| `work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth` | **제출 모델** |

2단계 스크립트는 부모 checkpoint의 sha256을 확인한다. 그래서 2단계는 동봉한 1단계 산출에서 출발한다. 3단계는 `train.sh 3`이면 새로 학습한 2단계 산출에서, `train.sh 3-shipped-parent`이면 동봉한 2단계 산출에서 출발한다.
