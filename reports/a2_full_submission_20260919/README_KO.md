# A2-FULL-NOM-s1 제출 후보 — 2026-09-19

`submission.zip`은 **1,125개 clip + 정수 `__flops__`**가 있는 `submission.json` 하나만 담는다. 공식 업로드는 수행하지 않았다.

- 완료돼 있던 A2 FULL step 24,931을 사용했다. 재학습하지 않았다.
- checkpoint SHA-256: `aabdb2dca24491b46fd2e56e66d57fe0742e17a5b4c63c29f3199137cf8ff297`.
- 출력은 0.5초 간격의 **absolute XY 6×2**다. 두 번째 cumsum이나 위치 보정을 적용하지 않았다.
- 공식과 같은 종류의 PyTorch Global FLOPs counter: **729,815,616,192 FLOPs**, 7,053G 한도 이내.
- 8개 train raw fixture에서 입력 9개와 최종 XY가 모두 bitwise 일치했다.
- 실제 test 1,125개 전체 추론, shape/유한값/clip 격리 검사를 완료했다.
- A2의 제공 status는 nominal causal producer로 만들고 공통 scene query에만 사용한다. Motion/state/history로 보내는 A3 gate는 없다.
- FULL의 기존 0.088934는 학습에 포함된 V0의 진단값이다. 이번 후보의 공식 서버 성능은 아직 없다.

## 재현 자료

`reproduction/`에 가중치, 소스, 두 raw clip 입력 예제, B1 tensor, 검사 JSON이 있다. 교사나 visual projector는 이 FULL 모델과 패키지에 들어가지 않는다.

깨끗한 Docker 환경에서 두 raw clip의 추론을 실행했다. 첫 clip의 모든 입력은 B200 입력과 bitwise 일치했다. BF16 최종 XY는 B200과 RTX4060 Laptop/다른 PyTorch 실행 환경 사이에서 최대 0.002829m 차이가 났다. 이는 입력 정합 검사와 분리해 기록했다.

**RTX4090에서 이번 A2를 측정한 시간은 아직 없다.** 과거 MR의 4090 측정값이나 서버 `elapsed_ms`를 대신 쓰지 않는다. B200 측정은 `portable_parity.json`에 별도로 기록했다.

## 재현 실행

저장소 소스 루트는 `reproduction/code`다. 다음 명령은 해당 폴더에서 실행한다. 아래 Docker 실행은 정상적으로 설정된 NVIDIA Container Toolkit 환경을 가정한다.

```bash
docker build -f experiments/a2_visual_teacher_20260919/Dockerfile -t adcl-a2-full:20260919 .
```

측정 장비에 `code`, checkpoint, 입력 예제를 복사하고 다음과 같이 실행한다. 경로는 장비의 실제 절대경로로 바꾼다.

```bash
docker run --rm --gpus device=0 \
  -v /absolute/reproduction/code:/workspace/code:ro \
  -v /absolute/reproduction:/workspace/artifacts:ro \
  -v /absolute/output:/output \
  adcl-a2-full:20260919 \
  --checkpoint /workspace/artifacts/ckpt_step24931.pth \
  --clips-root /workspace/artifacts/raw \
  --output /output/rtx4090_forward.json \
  --expected-device '4090' --timing
```

이 측정은 JPEG 처리 등을 제외하고 현재·과거 영상 인코딩을 모두 포함한 최상위 forward 전체를 잰다. `--expected-device`가 실제 장치를 검사하므로 다른 GPU 결과를 4090으로 잘못 저장하지 않는다.

`infer_a2.py`는 이 single-head A2 FULL용이다. QREFINE DEV/시각 교사 실험 checkpoint의 배포 스크립트로 혼용하지 않는다.
