# ETRI 2026 E2E Driving — 최종 제출물 (EXT-FULL-v7)

리더보드 최종 점수 **L2_avg 0.117067** 모델의 평가용 제출물입니다.

| L2_1s | L2_2s | L2_3s | L2_avg | `__flops__` | RTX 4090 1회 forward |
|---:|---:|---:|---:|---:|---:|
| 0.065436 | 0.115299 | 0.170465 | **0.117067** | 743,777,233,600 | 약 31 ms |

## 구성
| 제출 요구 항목 | 위치 |
|---|---|
| Docker 이미지 | `docker/Dockerfile` (빌드: `scripts/build_image.sh`, 이미지명 `etri-e2e-ext:final`) |
| 리더보드 성능 checkpoint | `checkpoints/ext_full_v7.pth` |
| 학습·추론 코드 (모델 포함) | `src/` |
| 데이터셋 준비 및 코드 작동 방법 | 이 문서의 "실행" 절, `docs/DATA_PREP.md` |
| 사용 기술 및 전략 설명서 | `docs/TECHNICAL_REPORT.md` (아키텍처 그림 `docs/architecture.png`) |
| 참고 자료 | `docs/VALIDATION.md` (검증·재현 결과), `docs/COMPLIANCE.md` (규정 대응) |

```
.
├── src/            모델, 학습, 추론, 데이터 준비 코드
├── checkpoints/    제출 모델과 학습 단계별 checkpoint
├── data/           학습용 파생 캐시, 실행 메타데이터, 검증용 학습 clip 8개
├── docker/         Dockerfile
├── scripts/        build_image.sh, infer.sh, validate.sh, train.sh
├── tools/          assemble.py (컨테이너 경로 구성), validate_release.py (검증)
├── validation/     검증용 테스트 clip 20개와 참조 예측
└── layout.json     src/·checkpoints/·data/ 파일과 코드 내부 경로의 대응표
```

## 실행
```sh
bash scripts/build_image.sh                         # 1) 이미지 빌드
bash scripts/validate.sh                            # 2) 동봉 clip 정합성, FLOPs, GPU 추론 시간 확인
bash scripts/infer.sh /path/to/test OUT_DIR         # 3) 테스트 1,125 clip 추론 → OUT_DIR/package/submission.zip
bash scripts/train.sh 1 /path/to/train OUT_DIR      # 4) 학습 1단계 (선택)
bash scripts/train.sh 2 /path/to/train OUT_DIR      #    학습 2단계
bash scripts/train.sh 3 /path/to/train OUT_DIR      #    학습 3단계 (제출 모델 레시피)
```
- **추론:** GPU 1장이 필요합니다. 테스트 폴더는 대회 배포 형식(`<clip_hash>/camera_*/frame_*.jpg`, `ego_pose.parquet`, `calibration.parquet`)을 그대로 사용합니다.
- **학습:** 40 GB 이상 GPU 1장과 대회 제공 train 376 scenario 폴더가 필요합니다. 학습은 NVIDIA B200에서 수행했습니다.
- **컨테이너 경로:** 코드는 학습 당시의 소스 해시와 저장소 기준 상대 경로를 검사합니다. 그래서 컨테이너가 시작될 때 `tools/assemble.py`가 `layout.json`에 따라 원래 경로 구조를 컨테이너 내부에 구성합니다. 소스 파일은 수정하지 않고 그대로 복사합니다.

## 학습 단계
| 단계 | 시작점 | 학습 대상 | update | 산출 checkpoint |
|---|---|---|---:|---|
| 1 | `stage1_initializer.pth` (공개 백본 + 무작위 초기화) | 전체 모델 | 24,931 | `stage1_h4_progress_full.pth` |
| 2 | 1단계 산출 | 전체 모델, 낮은 학습률 | 38,070 | `stage2_l_full6.pth` |
| 3 | 2단계 산출 | planner (몸통 동결), 후보 15개 | 6,344 | `ext_full_v7.pth` (제출 모델) |

외부 데이터는 1단계 초기값에 들어간 공개 백본 가중치 하나입니다. mmdetection3d `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`(COCO, nuImages 학습)이고, 파일은 `checkpoints/backbone_nuimages_cascade_r50.pth`입니다. 학습 데이터는 대회 제공 train 376 scenario만 사용했습니다.
