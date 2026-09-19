# 시각 교사 실행 상태

2026-09-19 18시대 KST. A2-VIS-TEACHER-s1-r2가 GPU1에서 3,426-update continuation을 진행한다. 고정 부모는 QREFINE terminal이며 state/history는 ON이다.

- 고정 teacher: DINOv2 ViT-B/14 registers, RGB only, frozen, online.
- 현재 6카메라 상위 FPN 한 level, teacher patch grid와 대응, train-only 1×1 projection.
- λ_vis=0.25: train 4batch gradient만으로 사전 규칙에 따라 선택. 최초 ratio의 약 1.16%에 해당하는 보조 backbone gradient이며 목표 10% 규칙의 상한에 걸린 값이다. 충분한 최적 강도라는 주장은 하지 않는다.
- 같은 초기 tensor/recipe/nominal 입력과 sample stream을 확인한 기존 G0를 대조로 재사용. CUDA backward의 bitwise 재현을 주장하지 않는다.
- 첫 시도는 추가 normalizer key 때문에 optimizer update 전에 종료. r2가 기존 normalizer와 visual count를 분리했다.
- 1,142 update: VIS 0.169695487, G0 0.168488675. 중간 평가에서 개선은 없으며 예정 terminal까지 조건을 유지한다.
- Teacher/층/λ 추가 탐색, OFF 학습, 신규 FULL 또는 공식 업로드를 예약하지 않았다.

주요 산출물: CURRENT_BASE.json, INPUT_POLICY.json, state_on_off.json, visual_preflight.json, teacher_manifest.json, control_reuse.json, COST_CHECK.json, protocol_A2-VIS-TEACHER-s1-r2.json.
