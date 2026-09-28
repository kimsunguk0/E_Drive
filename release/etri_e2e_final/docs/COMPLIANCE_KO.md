# 규정 대응 (운영국 공지·질의응답 기준)

| 규정 | 구현 | 근거 |
|---|---|---|
| 과거 궤적·status를 planner에 직접 또는 단순 임베딩으로 넣지 않는다. 공통 특징의 간접 활용은 허용 | status는 공통 scene attention query에만 들어간다. planner의 입력 인자에 goal·status·pose가 없다 | `code/models/motiondrive_v2/shared_status_query.py`, `code/experiments/a2_final_push_20260923/ext_v6/ext_model.py` (`ModeExtPlanner.forward`) |
| 운동학·규칙 기반 궤적 금지. 영상이 결과에 실질적으로 기여해야 한다 | 모든 좌표는 영상 feature를 읽는 planner가 만든다 | 영상을 단색으로 바꾸면 held-out L2 0.129 → 3.877 |
| 영상에서 추론한 상태를 planner에 쓰는 것은 허용 | motion encoder는 영상만 받는다 | `code/models/motiondrive_v2/motion_encoder.py`. status를 바꿔도 motion·state 출력 변화 0.0 |
| goal은 scene 형성 조건과 후보 선택에만 쓴다. 학습된 선택기는 쓰지 않는다 | scene query 조건 + 영상이 생성한 후보 15개 중 `argmin ‖후보(5 s) − goal‖` | 출력 = 선택 후보의 앞 6점과 비트 단위 일치(128/128). 선택 단계에 학습 모듈과 status가 없다 |
| 영상 입력부터 궤적까지 gradient가 이어지는 하나의 네트워크. 2-stage 학습은 허용 | 단일 모듈. 3단계는 몸통 동결 후 planner 학습 | 모든 영상 입력에 대해 ∂출력/∂입력 ≠ 0 |
| 후처리는 제출 규격 변환만 | 앞 6점을 잘라낼 뿐이다. 보정·TTA 없음 | — |
| 과거 정보를 쓰는 프레임은 영상도 입력한다. 과거 pose로 현재 status를 계산하는 것은 허용 | 기하 정렬 pose는 영상을 입력하는 과거 4 프레임에만 쓴다 | — |
| FLOPs cutoff, 4090 추론 시간 | 743.8 G / 7,053 G, 4090 31.2 ms | `docs/VALIDATION_KO.md` |
| 외부 데이터 명시 | 1단계 초기값의 공개 백본: mmdetection3d `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth` (COCO, nuImages). 그 밖의 학습 데이터는 대회 제공 train 376 scene뿐이다 | sha256 `4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e` |
| 테스트 데이터로 학습·라벨 생성 금지 | 학습은 제공 train 라벨만 썼다 | — |
