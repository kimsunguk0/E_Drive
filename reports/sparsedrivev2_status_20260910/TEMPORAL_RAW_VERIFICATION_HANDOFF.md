# Temporal 모델 raw 입력 검증 인계

실제 tune row **14730**, scene `20260113-102709`, frame 30에서 A(repeat), B(real history), C(real history + common perception status)의 raw 전처리 결과가 각각 고정된 cache loader 입력과 **모든 텐서에서 바이트 단위 동일**했다. CPU 테스트 12개도 통과했다. 이 단계에서는 모델 forward와 GPU 실행을 하지 않았다. 학습된 모델의 출력 동일성·내부 입력 경로·모델 지연은 아래 terminal 검증을 실행한 뒤에 판정해야 한다.

근거: [A receipt](raw_verification/raw_preflight_cpu_v1/preprocess_A/receipt.json), [B receipt](raw_verification/raw_preflight_cpu_v1/preprocess_B/receipt.json), [C receipt](raw_verification/raw_preflight_cpu_v1/preprocess_C/receipt.json), [fixture provenance](raw_verification/raw_preflight_cpu_v1/fixture_tune14730/fixture_receipt.json), [최종 소스·테스트 manifest](raw_verification/raw_preflight_cpu_v1/final_source/manifest.json).

## 동결 소스와 측정 버전

최종 실행 디렉터리(원격):

```text
/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/reports/sparsedrivev2_status_20260910/raw_preflight_cpu_v1/final_source
```

| 파일 | 최종 SHA256 |
|---|---|
| verify_temporal_raw_inference.py | `09a9e6e476143d51c4dd4f6f20c77760176c9144022b40d842a095c52e1598bb` |
| temporal_deployment.py | `cc0339425ab2bc43553b4b16ae4c0811f248f96a370417afb0740b4c47425f20` |
| evaluate_temporal_checkpoint.py | `9007a4517adaf663d402535bdd56dd5d8340ad675db0f3a3ef44d8d867529242` |
| models/motiondrive_v2_inputs.py (픽셀 의존성) | `33b64b3fe87a46e6b28587282e5615fbb69bb745d5ee32afd0e5d20ae56e3a29` |

실제 CPU A/B/C receipt는 verifier `3d4fffc6c0b45e4fd1051b6c3e9ade9066808eeb78d8df7496a18a413cdf03d8`로 생성했다. 해당 버전은 `source/` 및 각 receipt의 `verification_source/`에 보존했다. 이후 최종판은 OpenCV 스레드를 명시적으로 1로 고정하고 Python/package/base/environment 정보를 추가했다. 픽셀·필터·비교 로직은 같다. **최종 스레드 설정으로 실제 픽셀 측정은 반복하지 않았다.**

## 입출력 경계와 fixture

Raw adapter는 현재 front-left/front/front-right 3장과, B/C에서 front −1/−5 frame 2장을 읽는다. 과거 이미지 pose warp는 없다. 이미지 처리는 기존 undistort → crop → 768×432 JPEG95 재구성 → 512×256 resize/normalization 계약이다. geometry remap만 calibration 값으로 캐시하며 이미지 feature는 캐시하지 않는다. 공용 bundle의 `models.motiondrive_v2_inputs`, `models.motiondrive_v2_input_contract`, `models.motiondrive_v2_temporal_contract` 의존성을 worktree에서 읽는다.

`prepared.inputs`에는 images, history_images, lidar2img, image_hw, time_offsets만 있다. **C에서만** 과거 −10..0 pose의 nominal 10 Hz quadratic fit으로 계산한 vx/vy/ax/ay가 `perception_status`에 추가된다. A/B는 해당 상태 계산을 호출하지 않는다. Goal은 현재 ego 좌표로 변환한 제공 +50 XYZ를 `prepared.selector_inputs['goal_xy']`로 별도 반환한다. 원 planner status8 입력은 없다. Goal selection이 꺼진 A/B는 pose 파일 자체를 열지 않는다. 실제 A/B/C 비교는 학습 계약에 맞춰 goal selection을 켰다.

Fixture는 원본 scene tar의 지정 JPEG 5개와 calibration, 공식 pose subset만 포함한다. 먼저 timestamp/frame metadata로 −30..0 및 +50의 정확한 32개 timestamp를 얻고, pose parquet 읽기에 `timestamp in [...]` pushdown 필터를 적용한다. timestamp 실제 타입은 float64 밀리초다. +50 orientation 열도 이 필터된 표에서 읽힌 뒤 fixture에서 NaN으로 바뀐다. 따라서 **미래 orientation을 읽지 않았다는 주장은 하지 않으며, adapter가 수치적으로 사용하지 않는다는 것만 검증했다.** 중간 미래 trajectory point는 fixture에 없다. Ego cache에서는 row identity 배열만 읽으며 future trajectory 배열은 열지 않는다. Tar 전체 SHA는 없고 선택한 원본 JPEG 및 meta 파일 SHA를 보존했다.

Fixture receipt SHA: `9a141770df94f83ef461069430322bf1d86ba2deb12d647b9c88e236e59f98b0`.

## 재현 명령

CPU 테스트(로컬 실행 결과 12 passed; GPU 불사용):

```bash
python3 -m pytest -q -p no:cacheprovider \
  /home/a/adcl_status_20260910/test_temporal_deployment.py \
  /home/a/adcl_status_20260910/test_verify_temporal_raw_inference.py
```

원격에서 공통 변수:

```bash
TASK_BASE=/NHNHOME/data/sukim/adcl
TASK_WT=$TASK_BASE/experiment_worktrees/sparsedrivev2_20260910
TASK_RAW=$TASK_WT/reports/sparsedrivev2_status_20260910/raw_preflight_cpu_v1
TASK_PY=$TASK_BASE/env/venv/bin/python
TASK_VERIFY=$TASK_RAW/final_source/verify_temporal_raw_inference.py
```

기존 fixture는 `$TASK_RAW/fixture_tune14730`이다. 원본에서 새로 만들 경우 출력 경로가 존재하면 안 된다:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$TASK_PY" "$TASK_VERIFY" prepare-fixture \
  --base "$TASK_BASE" --row 14730 --output /tmp/temporal_fixture_14730_new
```

CPU 입력 비교 재현 예시(B); A는 `--history-mode repeat`, C는 `--history-mode real --common-status`로 변경하고 각자 새 출력 경로를 사용한다:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  "$TASK_PY" "$TASK_VERIFY" preprocess \
  --fixture "$TASK_RAW/fixture_tune14730" --output /tmp/temporal_cpu_B_new \
  --base "$TASK_BASE" --worktree "$TASK_WT" \
  --source-root "$TASK_WT/work_dirs/sparsedrivev2_status_20260910/temporal_b_history_s0_v1/source" \
  --history-mode real --camera-workers 3 --opencv-threads 1 \
  --preprocess-warmup 2 --preprocess-iterations 10
```

Terminal GPU 검증 예시(B, GPU1). 담당자가 같은 GPU의 전체 B1 평가가 종료된 뒤 실행한다. A/C는 해당 terminal checkpoint·할당 GPU·새 출력 경로만 바꾸며 arm 옵션을 별도로 넘기지 않는다:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  "$TASK_PY" "$TASK_VERIFY" verify \
  --checkpoint "$TASK_WT/work_dirs/sparsedrivev2_status_20260910/temporal_b_history_s0_v1/last.pth" \
  --fixture "$TASK_RAW/fixture_tune14730" \
  --output "$TASK_WT/reports/sparsedrivev2_status_20260910/temporal_b_terminal_raw_b1_v1" \
  --base "$TASK_BASE" --worktree "$TASK_WT" --gpu 1 \
  --camera-workers 3 --opencv-threads 1 \
  --preprocess-warmup 2 --preprocess-iterations 10 \
  --model-warmup 10 --model-iterations 50
```

GPU checker는 지정 GPU0/1/4 하나만 노출되고 사용 중인 프로세스가 없는지 검사한다. Strict loader는 terminal checkpoint 및 source/public weights/bank/split을 검증하고 manifest에서 history/common-status 설정을 읽는다. 별도 느슨한 checkpoint loader는 없다. 원 학습 snapshot의 final wrapper에 `model(**prepared.inputs, **prepared.selector_inputs)` 형태로 입력한다.

## 결과 판정과 시간 범위

`receipt.json`에서 `all_inputs_bitwise_equal`, `all_outputs_bitwise_equal`, `selected_trajectory_bitwise_equal`, `selected_id_equal`, `route_audit`를 확인한다. **`status: completed`는 실행 완료일 뿐 출력 parity 통과와 같지 않다.** 불일치도 `outputs.npz`와 텐서별 차이에 보존하므로 임의 재실행으로 성공 결과를 선택하지 않는다. selected row가 고정 bank와 일치하는지도 검사한다. 실제 float byte를 uint8로 비교하며 signed zero 차이도 탐지한다.

Raw adapter에는 GT/aux/status cache가 없다. 비교 기준 cache Dataset은 외부 label을 materialize하지만, 입력 whitelist에 해당 label을 넣지 않고 label 값으로 비교·모델 forward·D3 계산을 하지 않는다. 이 구분을 receipt에 명시한다. GPU route audit에서는 원 base/relative status가 0인지, goal이 최종 선택 경로에만 들어가는지, C의 image-only state auxiliary가 perception status와 독립인지 추가 검사한다.

지연은 두 범위로 분리한다. `preprocessing_latency`는 fixture disk/parquet/image 읽기와 geometry cache lookup, raw pixel 처리 및 허용 state/goal 조립이다(geometry/OS 파일 캐시가 warm일 수 있음). `model_only_latency`는 GPU에 이미 올린 입력의 전체 모델 forward이며 이미지/history encoding을 포함하고 checkpoint 로딩, H2D, 전처리, metric, route hook은 제외한다. 두 값의 합을 실제 전체 파이프라인 지연으로 단정하지 않는다. B200 측정은 RTX4090 충족 증거가 아니다.

이전 CPU 참고 평균 A/B/C는 74.85/112.57/123.57 ms였다. OpenCV 스레드 미고정 및 학습 3개 동시 실행 상황의 10회 측정이므로 성능 판정에 쓰지 않는다. 최종판은 OpenCV threads=1, torch threads=4 및 package/base/environment를 기록한다. 최종 GPU 출력 및 지연 결과는 담당자 실행 후 별도로 인용해야 한다.
