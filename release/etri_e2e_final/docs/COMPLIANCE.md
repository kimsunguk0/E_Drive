# 규정 대응

| 규정 | 구현 | 근거 |
|---|---|---|
| 과거 궤적과 ego 상태를 planner에 직접 또는 단순 임베딩으로 입력하지 않습니다. 공통 특징 형성에 간접 활용하는 것은 허용됩니다 | ego 상태는 공통 scene attention의 query에만 사용합니다. planner 입력에는 목표점, ego 상태, pose가 없습니다 | `src/models/motiondrive_v2/shared_status_query.py`, `src/ext_model.py` (`ModeExtPlanner.forward`) |
| 운동학·규칙 기반 궤적을 사용하지 않고, 영상이 결과에 실질적으로 기여해야 합니다 | 모든 좌표는 영상 특징을 읽는 planner가 생성합니다 | 영상을 단색으로 바꾸면 held-out L2가 0.129 → 3.877 |
| 영상에서 추론한 상태는 planner에 사용할 수 있습니다 | motion encoder는 영상만 입력받습니다 | `src/models/motiondrive_v2/motion_encoder.py`. ego 상태를 바꿔도 motion·상태 예측 변화 0.0 |
| 목표점은 scene 형성 조건과 후보 선택에만 사용하고, 학습된 선택기는 사용하지 않습니다 | scene query 조건과, 영상이 생성한 후보 15개 중 `argmin ‖후보(5 s) − 목표점‖` 선택 | 출력이 선택된 후보의 앞 6점과 비트 단위로 일치합니다(128/128). 선택에 학습 모듈과 ego 상태가 없습니다 |
| 영상 입력부터 궤적까지 gradient가 이어지는 하나의 네트워크여야 합니다. 2-stage 학습은 허용됩니다 | 단일 모듈입니다. 3단계는 몸통을 동결하고 planner를 학습합니다 | 모든 영상 입력에 대해 ∂출력/∂입력 ≠ 0 |
| 후처리는 제출 규격 변환만 허용됩니다 | 앞 6점을 자르는 것 외에 보정이나 TTA가 없습니다 | — |
| 과거 정보를 쓰는 프레임은 영상도 함께 입력해야 합니다. 과거 pose로 현재 ego 상태를 계산하는 것은 허용됩니다 | 기하 정렬 pose는 영상을 입력하는 과거 4 프레임에만 사용합니다 | — |
| FLOPs cutoff와 RTX 4090 추론 시간 | 743.8 G / 7,053 G, 4090 약 31 ms | `docs/VALIDATION.md` |
| 외부 데이터 명시 | 공개 백본 `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth` (mmdetection3d, COCO·nuImages) | sha256 `4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e` |
| 테스트 데이터로 학습하거나 라벨을 생성하지 않습니다 | 학습에는 제공 train 라벨만 사용했습니다 | — |
