# 최종 제출 모델 규정 검증 — EXT-FULL-v7 (서버 0.117067)

- 대상: `work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth`, 제출 zip sha256 `768786ec…`.
- 근거 규정: `OPEN_ISSUE.md`(질문1~10, 공지1·2), 2026-09-23 오픈채팅 답변(사용자 전달).
- 실측 스크립트: `experiments/a2_final_push_20260923/ext_v6/compliance_audit.py`.
  - 제출 모델로 테스트 clip 64개를 추론했다.
  - held-out 쌍둥이 모델로 V0 500행을 평가했다.

## 데이터 흐름 (코드)
- **영상:** 6카메라 현재 영상 + 전방 과거 4장 → ResNet-50 FPN → (a) scene raster, (b) motion encoder.
- **(a) scene raster:** 각 셀 query = 셀 위치 + goal + (셀−goal) + **status5 delta**. 값은 영상 feature만 쓴다.
  - 코드: `models/motiondrive_v2/scene_encoder.py:170-176`, `models/motiondrive_v2/shared_status_query.py:36-49`, hook은 `experiments/md_shared_dynamics_20260917/factorized_model.py:192-197`.
  - 이 raster는 occupancy·lane head와 planner가 **공유**한다(`scene_encoder.py:94-98`).
- **(b) motion encoder:** pose·goal·status 인자가 없다(`models/motiondrive_v2/motion_encoder.py:3`). 영상만으로 state_hat·history_hat을 예측한다.
- **planner (`ext_v6/ext_model.py` `ModeExtPlanner.forward`):** 입력은 scene·motion·state_hat·history_hat·pair feature다. **goal·status·pose 인자가 없다.** K=15 후보 × 10점(5 s)을 낸다.
- **선택 (`ext_model.py` `select_by_goal`):** argmin_k ‖후보_k(5.0 s) − goal‖이다. 학습 모듈은 없다. 선택 후보의 앞 6점을 좌표 변경 없이 출력한다.

## 규정별 판정
| # | 규정 | 우리 구현 · 근거 | 실측 | 판정 |
|---|---|---|---|---|
| 1 | 과거궤적·status를 planner에 직접/단순 임베딩 금지, 공통 특징 간접 활용 허용 (공지1 3항, 질문7) | status는 공유 scene query에만 들어간다(값 경로 없음). planner 인자에 없다 | status를 0으로 → 출력 평균 4.25 m 변화. 크게 의존하지만 간접 경로다 | 준수 |
| 2 | 운동학·규칙 기반·동등 NN 금지 (3-2·3-3) | 외삽식 없음. 모든 좌표가 학습된 planner 출력이다 | 영상을 회색으로 → V0 0.129 → **3.877**. status+goal만으로는 궤적이 성립하지 않는다 | 준수 |
| 3 | 영상에서 추론한 status를 planner에 사용 허용 (질문10) | state_hat·history_hat는 영상만 받는 motion encoder 출력이다 | — | 준수 |
| 4 | goal: scene 형성 조건 허용(질문8 재확인), planner query 금지('중요' 답변), 후보를 goal만으로 수학 선택 허용, 학습형 선택기 금지(9/23) | goal은 scene query와 argmin 선택에만 쓴다. 선택에 학습 모듈·status 없음 | 출력 = 선택 후보의 앞 6점, 비트 단위 일치하고 argmin과도 일치 **128/128**. goal +5 m → 후보 0.51 m 이동(scene query 효과), 선택 57/64 변경 | 준수 |
| 5 | 영상→궤적 gradient가 이어지는 하나의 네트워크, 2-stage 학습 허용 (9/23) | `ExtModel` 단일 모듈. 몸통 동결 후 planner 학습 | ‖∂출력/∂입력‖: 현재영상 0.99, 과거영상 1.74, motion 입력 1.88·1.77 (모두 0 아님) | 준수 |
| 6 | 후처리는 규격 변환만, 위치 보정 금지 (질문1·5) | 앞 6점 slicing만 한다. TTA·보정 없음 | 1번 항목과 같은 비트 단위 일치 | 준수 |
| 7 | 과거를 쓰는 모든 프레임은 영상 필수, 과거 pose로 현재 status 계산은 허용 (공지 4항, 질문4·6) | 기하 정렬 pose는 영상을 입력하는 4개 과거 프레임에만 쓴다. status5는 과거 pose로 계산한다(질문6 허용) | — | 준수 |
| 8 | 추론시간(4090, forward 누적)·FLOPs cutoff | forward 1회. `__flops__` 743.8 G (FlopCounterMode, cutoff 7,053 G) | B200 BF16 **22.0 ms**. 4090은 미측정(같은 몸통 1회 forward가 4090에서 26.1 ms) | 준수 (4090 실측 권장) |
| 9 | 테스트 데이터로 학습·라벨·역추적·최적화 금지 (FAQ) | 학습은 train 376 scene 라벨만 썼다 | 아래 "공개 필요 사항" 참고 | 준수 (공개 권장 1건) |
| 10 | 외부 데이터 사용 시 제출 명시 (FAQ, 안내서) | 백본 초기값 = 공개 mmdetection3d **nuImages Cascade Mask R-CNN R50** (COCO → nuImages), sha256 `40963960…`, ETRI 학습 전에 적용 | — | **명시 필요** |
| 11 | command 사용 허용 | 사용하지 않는다 | — | 해당 없음 |

## 조치·공개 필요 사항
1. **외부 사전학습 명시:** 제출 서류에 백본 초기값을 적어야 한다.
   - 적을 내용: `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth` (mmdetection3d 공개, COCO·nuImages)
   - 근거 기록: `reports/md_r0_reset_20260914/reproduction_manifest.json` `pretrained_origin`
2. **4090 실측:** 코드 심사 전에 최종 산출물(K=15)의 1회 forward 시간을 RTX 4090에서 재는 것을 권장한다. B200 22.0 ms, 부모 몸통 4090 26.1 ms라 100 ms에 크게 못 미칠 것으로 예상하지만 측정값은 아니다.
3. **테스트 입력 통계 사용 공개 여부:**
   - v7과 v10 중 최종 선택에 테스트 입력의 goal<5 m 비율을 민감도 분석으로 참고했다(test 29.4%, train 8.1%, V0 6.7%).
   - 라벨·학습·가중치 조정에는 쓰지 않았고, 이미 학습된 두 모델 중 하나를 고르는 데만 썼다.
   - FAQ의 "테스트 데이터를 학습·최적화에 활용하는 모든 행위 금지"를 엄격히 읽으면 모델 선택도 포함될 여지가 있다. 질문받으면 사실대로 설명하는 것을 권장한다.
4. status 의존도가 크다(0으로 두면 4.25 m 변화). 그래도 경로는 공유 scene query 전용이라 질문7의 "간접 활용" 기준에 맞는다. 심사에서 물으면 `shared_status_query.py`의 query 전용 구조와 영상 제거 시 성능 붕괴(0.129 → 3.877)를 근거로 든다.
