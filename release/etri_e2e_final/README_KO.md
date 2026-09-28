# ETRI 2026 E2E Driving — 최종 제출물 (EXT-FULL-v7)

리더보드 최종 점수 **L2_avg 0.117067** 모델의 평가용 제출물이다.

| L2_1s | L2_2s | L2_3s | L2_avg | `__flops__` | RTX 4090 1회 forward |
|---:|---:|---:|---:|---:|---:|
| 0.065436 | 0.115299 | 0.170465 | **0.117067** | 743,777,233,600 | 중앙값 31.2 ms (시간 벌점 ×1.0) |

## 구성
| 공지 항목 | 위치 |
|---|---|
| Docker 이미지 | `docker/Dockerfile`, 빌드 `scripts/build_image.sh` (이미지명 `etri-e2e-ext:final`) |
| 리더보드 성능 checkpoint | `code/work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth` (sha256 `c3a0bc10…0a2920`) |
| 학습·추론 코드 (모델 포함) | `code/` (저장소 원본과 바이트 단위로 같다. `MANIFEST.json`에 파일별 sha256) |
| 데이터셋 준비 및 실행 방법 | `docs/DATA_PREP_KO.md`, 아래 "실행" 절 |
| 기술 및 전략 설명서 | `docs/TECHNICAL_REPORT_KO.md` |
| 부가 자료 | `docs/VALIDATION_KO.md`(4090 검증·재현), `docs/COMPLIANCE_KO.md`(규정 대응) |

```
etri_e2e_final/
├── code/            학습·추론 코드 + 파생 데이터 캐시 + checkpoint (컨테이너에서 /NHNHOME/data/sukim/adcl 로 마운트)
├── data_pm97/       ego 캐시 2종 (컨테이너에서 /tmp/pm97/data/etri 로 마운트)
├── docker/          Dockerfile
├── scripts/         build_image.sh / run_inference.sh / validate.sh / train.sh
├── tools/           validate_release.py (검증), make_references.py (참조 생성)
├── validation/      검증용 테스트 20 clip, 참조 예측, 4090 결과
├── docs/            설명서
└── MANIFEST.json    코드·가중치·캐시 sha256
```

코드는 이미지에 넣지 않고 **원래 저장소 경로(`/NHNHOME/data/sukim/adcl`)로 마운트**한다. 학습 스크립트는 소스 해시를 고정해 확인하는데, 이렇게 해야 모든 소스가 학습 당시와 비트 단위로 같아진다. `code/`는 git 저장소다. 학습 스크립트가 `git rev-parse`로 커밋을 기록하기 때문이다.

## 실행
```sh
bash scripts/build_image.sh                                  # 1) 이미지 빌드
bash scripts/validate.sh                                     # 2) 동봉 clip으로 정합성 + FLOPs + 4090 시간 확인
bash scripts/run_inference.sh /path/to/test OUT_DIR          # 3) 테스트 1,125 clip → OUT_DIR/package/submission.zip
bash scripts/train.sh 1 /path/to/train                       # 4) 학습 1단계 (선택)
bash scripts/train.sh 2 /path/to/train                       #    2단계
bash scripts/train.sh 3 /path/to/train                       #    3단계 (2단계 산출물을 부모로 사용)
```
- 추론: GPU 1장, 테스트 폴더는 대회 배포 형식(`<clip_hash>/camera_*/frame_*.jpg`, `ego_pose.parquet`, `calibration.parquet`)이다.
- 학습: 40 GB 이상 GPU 1장이 필요하다(NVIDIA B200에서 학습). 원본 train 376 scene 폴더를 인자로 준다.
- `train.sh 3-shipped-parent`는 동봉한 2단계 checkpoint(제출 모델의 실제 부모)에서 3단계만 다시 학습한다.

## 학습 단계 요약
| 단계 | 시작점 | 학습 대상 | update | 산출 checkpoint |
|---|---|---|---:|---|
| 1 | 공개 초기값(nuImages Cascade R-CNN R50 백본 + 무작위 나머지) | 전체 모델 | 24,931 | `A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth` |
| 2 | 1단계 | 전체 모델, 낮은 학습률 | 38,070 | `L-FULL6-s1/ckpt_step38070.pth` |
| 3 | 2단계 | planner만(몸통 동결), 후보 15개 | 6,344 | `EXT-FULL-v7/ckpt_step6344.pth` |

외부 데이터는 1단계 초기값의 공개 백본 가중치 하나뿐이다: mmdetection3d `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth` (COCO → nuImages 학습, sha256 `4096396…d58bb4e`). 학습 데이터는 대회 제공 train 376 scene만 썼다.
